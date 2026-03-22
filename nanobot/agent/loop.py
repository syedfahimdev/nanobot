"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|(?<=\n)')
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.restart import RestartTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import ListSubagentsTool, SpawnTool, UpdateSubagentTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ProfileConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        profiles: "dict[str, ProfileConfig] | None" = None,
        profile_factory: "Callable[[str], LLMProvider] | None" = None,
        profile_save_callback: "Callable[[str], None] | None" = None,
        routing_model: str | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.routing_model = routing_model  # Fast model for first iteration routing
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._profiles: dict[str, "ProfileConfig"] = profiles or {}
        self._profile_factory = profile_factory
        self._profile_save_callback = profile_save_callback
        self._active_profile: str | None = None

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)

        # Hook engine — deterministic code-level event handlers
        from nanobot.hooks import HookEngine
        from nanobot.hooks.builtin import register_builtin_hooks
        self.hooks = HookEngine()
        register_builtin_hooks(self.hooks, workspace, bus)

        self.tools = ToolRegistry(hooks=self.hooks)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            session_manager=self.sessions,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}  # per-session locks
        self._pending_messages: dict[str, list[InboundMessage]] = {}  # queued conversational msgs
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )

        # Reflection engine — learns from user corrections
        from nanobot.hooks.builtin.reflection import ReflectionEngine
        self.reflection = ReflectionEngine(workspace, provider, self.model)

        # Morning briefing — proactive context-aware summaries
        from nanobot.hooks.builtin.briefing_hook import make_briefing_hook
        self.hooks.on("turn_completed", make_briefing_hook(workspace, provider, self.model, bus))

        # Inbox indexer — auto-indexes uploaded files for RAG search
        from nanobot.hooks.builtin.inbox_indexer import make_inbox_indexer_hook
        self.hooks.on("turn_completed", make_inbox_indexer_hook(workspace, provider, self.model))

        # Workflow recorder — detects repeated tool sequences
        from nanobot.hooks.builtin.workflow_recorder import make_workflow_recorder_tool_hook, make_workflow_recorder_turn_hook
        self.hooks.on("tool_after", make_workflow_recorder_tool_hook(workspace))
        self.hooks.on("turn_completed", make_workflow_recorder_turn_hook(workspace, bus))

        # Prompt optimizer — tracks instruction effectiveness
        from nanobot.hooks.builtin.prompt_optimizer import make_prompt_optimizer_tool_hook, make_prompt_optimizer_turn_hook
        self.hooks.on("tool_after", make_prompt_optimizer_tool_hook(workspace))
        self.hooks.on("turn_completed", make_prompt_optimizer_turn_hook(workspace))

        # Proactive notifications — checks goals, bills, calendar
        from nanobot.hooks.builtin.proactive import make_proactive_hook
        self.hooks.on("turn_completed", make_proactive_hook(workspace, bus))

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(RestartTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(UpdateSubagentTool(manager=self.subagents))
        self.tools.register(ListSubagentsTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        from nanobot.agent.tools.memory_save import MemorySaveTool
        self.tools.register(MemorySaveTool(workspace=self.workspace))
        from nanobot.agent.tools.memory_search import MemorySearchTool
        self.tools.register(MemorySearchTool(workspace=self.workspace))
        from nanobot.agent.tools.goals import GoalsTool
        self.tools.register(GoalsTool(workspace=self.workspace))
        from nanobot.agent.tools.media_memory import MediaMemoryTool
        self.tools.register(MediaMemoryTool(workspace=self.workspace))
        from nanobot.agent.tools.inbox import InboxTool
        self.tools.register(InboxTool(workspace=self.workspace))
        # Skill acquisition — learn new skills from descriptions/URLs
        from nanobot.agent.tools.skill_creator import SkillCreatorTool
        self.tools.register(SkillCreatorTool(self.workspace, self.provider, self.model))
        # Knowledge ingestion — learn from web pages and blogs
        from nanobot.agent.tools.knowledge_ingest import KnowledgeIngestTool
        self.tools.register(KnowledgeIngestTool(self.workspace, self.provider, self.model))
        # Credentials — securely store and use passwords
        from nanobot.agent.tools.credentials import CredentialsTool
        self.tools.register(CredentialsTool())
        # Native Playwright browser
        try:
            from nanobot.agent.tools.browser import BrowserTool
            self.tools.register(BrowserTool())
        except Exception:
            pass  # Playwright not installed — skip silently

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))
        # Set hook context so approval guard knows the channel
        session_key = f"{channel}:{chat_id}"
        self.tools.set_context(channel, chat_id, session_key)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    # Conversational message detection patterns
    _CONVERSATIONAL_SKIP_PATTERNS = re.compile(
        r"^(/\w|hi\b|hello\b|hey\b|thanks|ok\b|yes\b|no\b|good|how are|what's up"
        r"|why\b|what do you|who are you|tell me about yourself|how do you|are you"
        r"|do you|can you help|what can you|nice|cool|great|sure|alright|bye|see you"
        r"|sorry|excuse me|never ?mind|forget it|stop|wait|hold on|go ahead)",
        re.IGNORECASE,
    )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict], dict]:
        """Run the agent iteration loop. Returns (content, tools, messages, usage)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _last_call_sig: str | None = None  # detect repeated identical tool calls
        _repeat_count = 0

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            # Use fast routing model for first iteration if configured
            current_model = self.routing_model if (iteration == 1 and self.routing_model) else self.model

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=current_model,
            )

            # Accumulate token usage across iterations
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] += response.usage.get(k, 0)

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    tool_hint = self._tool_hint(response.tool_calls)
                    tool_hint = self._strip_think(tool_hint)
                    await on_progress(tool_hint, tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                # Execute tool calls — parallel when multiple, sequential when single
                if len(response.tool_calls) > 1:
                    # Parallel execution for multiple independent tool calls
                    logger.info("Parallel execution: {} tool calls", len(response.tool_calls))

                    async def _run_tool(tc):
                        args_str = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
                        logger.info("Tool call (parallel): {}({})", tc.name, args_str[:200])
                        return await self.tools.execute(tc.name, tc.arguments)

                    results = await asyncio.gather(
                        *[_run_tool(tc) for tc in response.tool_calls],
                        return_exceptions=True,
                    )
                    for tc, result in zip(response.tool_calls, results):
                        tools_used.append(tc.name)
                        if isinstance(result, Exception):
                            logger.error("Parallel tool call failed: {}: {}", tc.name, result)
                            result = f"(Tool call failed: {type(result).__name__}: {result})"
                        if on_progress:
                            preview = str(result)[:300]
                            await on_progress(f"[{tc.name}] → {preview}", tool_hint=True)
                        messages = self.context.add_tool_result(
                            messages, tc.id, tc.name, result
                        )
                else:
                    # Single tool call — sequential with retry detection
                    tool_call = response.tool_calls[0]
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)
                    call_sig = f"{tool_call.name}|{args_str}"
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])

                    # Detect repeated identical calls (infinite loop protection)
                    if call_sig == _last_call_sig:
                        _repeat_count += 1
                        if _repeat_count >= 2:
                            logger.warning("Breaking repeated tool call loop: {} ({}x)", tool_call.name, _repeat_count + 1)
                            result = (
                                f"ERROR: You have called {tool_call.name} with the same arguments "
                                f"{_repeat_count + 1} times. STOP retrying. "
                                f"Either provide different arguments or tell the user you cannot "
                                f"complete this action."
                            )
                            messages = self.context.add_tool_result(
                                messages, tool_call.id, tool_call.name, result
                            )
                            continue
                    else:
                        _last_call_sig = call_sig
                        _repeat_count = 0

                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    if on_progress:
                        preview = str(result)[:300]
                        await on_progress(f"[{tool_call.name}] → {preview}", tool_hint=True)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages, total_usage

    async def _run_agent_loop_streaming(
        self,
        initial_messages: list[dict],
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict], dict]:
        """Streaming variant of agent loop — fires on_sentence as sentences complete.

        Used by voice channels for low-latency TTS: each sentence is sent to TTS
        as soon as it's accumulated from the token stream, rather than waiting for
        the full response.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _last_call_sig: str | None = None
        _repeat_count = 0

        while iteration < self.max_iterations:
            iteration += 1
            tool_defs = self.tools.get_definitions()
            current_model = self.routing_model if (iteration == 1 and self.routing_model) else self.model

            # Stream tokens from the LLM
            accumulated_content = ""
            sentence_buffer = ""
            all_tool_calls: list = []
            finish_reason = None
            usage = {}
            reasoning_content = None

            try:
                async for chunk in self.provider.stream_chat(
                    messages=messages,
                    tools=tool_defs,
                    model=current_model,
                ):
                    if chunk.delta_content:
                        accumulated_content += chunk.delta_content
                        sentence_buffer += chunk.delta_content

                        # Check for sentence boundaries and fire TTS
                        if on_sentence:
                            while True:
                                match = _SENTENCE_RE.search(sentence_buffer)
                                if not match:
                                    break
                                sentence = sentence_buffer[:match.end()].strip()
                                sentence_buffer = sentence_buffer[match.end():]
                                if len(sentence) >= 3:
                                    await on_sentence(sentence)

                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason
                        all_tool_calls = chunk.tool_calls
                        usage = chunk.usage
                        reasoning_content = chunk.reasoning_content

            except Exception as e:
                logger.error("Streaming error: {}", e)
                # Fall back to non-streaming
                return await self._run_agent_loop(initial_messages, on_progress)

            # Accumulate token usage across iterations
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] += (usage or {}).get(k, 0)

            # Flush remaining sentence buffer
            if on_sentence and sentence_buffer.strip() and len(sentence_buffer.strip()) >= 3:
                await on_sentence(sentence_buffer.strip())

            if finish_reason == "error":
                logger.error("LLM stream error: {}", accumulated_content[:200])
                final_content = accumulated_content or "Sorry, I encountered an error calling the AI model."
                break

            if all_tool_calls:
                # Tool calls detected — process them
                if on_progress:
                    thought = self._strip_think(accumulated_content)
                    if thought:
                        await on_progress(thought)
                    tool_hint = self._tool_hint(all_tool_calls)
                    tool_hint = self._strip_think(tool_hint)
                    await on_progress(tool_hint, tool_hint=True)

                tool_call_dicts = [tc.to_openai_tool_call() for tc in all_tool_calls]
                messages = self.context.add_assistant_message(
                    messages, accumulated_content or None, tool_call_dicts,
                    reasoning_content=reasoning_content,
                )

                # Execute tools (same logic as non-streaming)
                if len(all_tool_calls) > 1:
                    async def _run_tool(tc):
                        args_str = json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)
                        logger.info("Tool call (parallel): {}({})", tc.name, args_str[:200])
                        return await self.tools.execute(tc.name, tc.arguments)

                    results = await asyncio.gather(
                        *[_run_tool(tc) for tc in all_tool_calls],
                        return_exceptions=True,
                    )
                    for tc, result in zip(all_tool_calls, results):
                        tools_used.append(tc.name)
                        if isinstance(result, Exception):
                            result = f"(Tool call failed: {type(result).__name__}: {result})"
                        if on_progress:
                            preview = str(result)[:300]
                            await on_progress(f"[{tc.name}] → {preview}", tool_hint=True)
                        messages = self.context.add_tool_result(messages, tc.id, tc.name, result)
                else:
                    tool_call = all_tool_calls[0]
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)
                    call_sig = f"{tool_call.name}|{args_str}"
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])

                    if call_sig == _last_call_sig:
                        _repeat_count += 1
                        if _repeat_count >= 2:
                            logger.warning("Breaking repeated tool call loop: {}", tool_call.name)
                            result = (
                                f"ERROR: You have called {tool_call.name} with the same arguments "
                                f"{_repeat_count + 1} times. STOP retrying."
                            )
                            messages = self.context.add_tool_result(messages, tool_call.id, tool_call.name, result)
                            continue
                    else:
                        _last_call_sig = call_sig
                        _repeat_count = 0

                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    if on_progress:
                        preview = str(result)[:300]
                        await on_progress(f"[{tool_call.name}] → {preview}", tool_hint=True)
                    messages = self.context.add_tool_result(messages, tool_call.id, tool_call.name, result)
            else:
                # Final text response
                clean = self._strip_think(accumulated_content)
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=reasoning_content,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task."
            )

        return final_content, tools_used, messages, total_usage

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        from nanobot.hooks.builtin.backup import start_backup_loop
        self._schedule_background(start_backup_loop(self.workspace))
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process — use systemd if available, else os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            try:
                # Try systemd restart first (if running as a service)
                import subprocess
                result = subprocess.run(
                    ["systemctl", "restart", "nanobot"],
                    capture_output=True, timeout=10,
                )
                if result.returncode == 0:
                    return  # systemd will handle the restart
                logger.warning("systemctl restart failed (rc={}), falling back to execv", result.returncode)
            except Exception:
                pass
            # Fallback: in-place restart
            try:
                os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])
            except Exception as e:
                logger.error("Restart failed (os.execv): {}", e)

        asyncio.create_task(_do_restart())

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Get or create a per-session lock."""
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]

    # Common conversational words that should NOT spawn a subagent
    _CONVERSATIONAL_WORDS = frozenset({
        "ok", "okay", "thanks", "thank you", "yes", "no", "sure",
        "cool", "great", "nice", "got it", "alright", "fine",
        "good", "perfect", "awesome", "bye", "hi", "hello",
        "hey", "yep", "nope", "right", "yeah", "nah", "hmm",
    })

    def _is_conversational(self, text: str) -> bool:
        """Check if message is conversational (not a task worth spawning)."""
        stripped = text.strip()
        # Short messages matching common conversational patterns
        if len(stripped) < 15 and self._CONVERSATIONAL_SKIP_PATTERNS.match(stripped):
            return True
        # Single/two word acknowledgments
        normalized = stripped.lower().rstrip(".!?,")
        if len(normalized.split()) <= 2 and normalized in self._CONVERSATIONAL_WORDS:
            return True
        return False

    # Context-dependent message detection — pronouns/references that need conversation context
    _CONTEXT_MARKERS = re.compile(
        r"\b(that|this|those|these|it|its|them|him|her|his|hers|theirs"
        r"|also|too|same|again|like before|the one|the last"
        r"|previous|earlier|just now|what you|you just|my last"
        r"|did (it|they|he|she)|was (it|that|this)"
        r"|how about|what about|and also|one more)\b",
        re.IGNORECASE,
    )

    _CLEAR_TASK_RE = re.compile(
        r"\b(check|fetch|search|create|send|schedule|book|open|list|show|set|get|find)\b"
        r".*\b(email|calendar|slack|github|weather|stock|task|file|report|meeting|news|reddit)\b",
        re.IGNORECASE,
    )

    def _is_context_dependent(self, text: str) -> bool:
        """Detect messages needing conversation context (pronouns, references)."""
        stripped = text.strip()
        if len(stripped) > 120:
            return False  # Long messages are likely self-contained
        if self._CLEAR_TASK_RE.search(stripped):
            return False  # Clear verb+noun = self-contained task
        return bool(self._CONTEXT_MARKERS.search(stripped))

    # Dynamic skill pattern cache — built from skill names/trigger phrases
    _skill_pattern: re.Pattern | None = None
    _skill_cache_ts: float = 0

    def _needs_main_agent(self, text: str) -> bool:
        """Check if message matches a known skill (requires interactive main agent).

        Dynamically loads skill names and trigger phrases from the workspace
        Refreshes every 5 minutes. Matches on skill names as
        phrases (e.g. "weekly report", "cea report") not individual words,
        to avoid false positives.
        """
        import time as _time

        now = _time.monotonic()
        if self._skill_pattern is None or (now - self._skill_cache_ts) > 300:
            self._refresh_skill_patterns()
            self._skill_cache_ts = now

        if self._skill_pattern is None:
            return False
        return bool(self._skill_pattern.search(text))

    @staticmethod
    def _extract_skill_phrases(name: str, description: str) -> set[str]:
        """Extract match phrases from a skill name and description."""
        phrases: set[str] = set()
        # Skill name as phrase (e.g. "cea-weekly-report" → "cea weekly report")
        name_phrase = name.replace("-", " ").replace("_", " ").strip()
        if len(name_phrase) >= 3:
            phrases.add(name_phrase.lower())
        # Consecutive word pairs from name
        parts = name_phrase.lower().split()
        if len(parts) >= 2:
            for i in range(len(parts) - 1):
                phrases.add(f"{parts[i]} {parts[i+1]}")
        # Extract quoted trigger phrases from description ("Use when: user says 'X', 'Y'")
        if description:
            for match in re.findall(r"['\"]([^'\"]{3,30})['\"]", description):
                phrases.add(match.lower())
        return phrases

    def _refresh_skill_patterns(self) -> None:
        """Build skill match patterns from workspace skills."""
        phrases: set[str] = {"skill"}  # "use the skill" always routes to main agent

        # Workspace skills directory
        try:
            for skills_dir in [self.workspace / "skills"]:
                if not skills_dir.is_dir():
                    continue
                for skill_dir in skills_dir.iterdir():
                    if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                        continue
                    desc = ""
                    try:
                        content = (skill_dir / "SKILL.md").read_text()
                        for line in content.split("\n")[:15]:
                            if line.startswith("description:"):
                                desc = line.split(":", 1)[1].strip().strip('"').strip("'")
                    except Exception:
                        pass
                    phrases |= self._extract_skill_phrases(skill_dir.name, desc)
        except Exception:
            pass

        if phrases:
            # Sort longest first so "weekly report" matches before "report"
            escaped = [re.escape(p) for p in sorted(phrases, key=len, reverse=True)]
            AgentLoop._skill_pattern = re.compile(
                r"\b(" + "|".join(escaped) + r")\b", re.I,
            )
            logger.info("Loaded {} skill phrases for routing: {}",
                         len(phrases), ", ".join(sorted(phrases)[:8]))
        else:
            AgentLoop._skill_pattern = None

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under a per-session lock.

        If the session is busy:
        - Conversational messages ("ok", "thanks") are queued and appended to
          session history after the current task completes.
        - Task messages are auto-spawned as subagents for parallel execution.
        """
        # Intercept approval responses (yes/no) — consumed before normal dispatch
        from nanobot.hooks.builtin.approval import resolve_approval
        if resolve_approval(msg.session_key, msg.content):
            logger.info("Approval response consumed for {}: '{}'", msg.session_key, msg.content[:20])
            return

        lock = self._get_session_lock(msg.session_key)

        if lock.locked():
            text = msg.content.strip()

            if self._is_conversational(text):
                logger.info("Session {} busy, queuing conversational: '{}'",
                            msg.session_key, text[:40])
                self._pending_messages.setdefault(msg.session_key, []).append(msg)
                return

            # Context-dependent messages (pronouns, references) need main agent context
            if self._is_context_dependent(text):
                logger.info("Session {} busy, queuing context-dependent: '{}'",
                            msg.session_key, text[:60])
                self._pending_messages.setdefault(msg.session_key, []).append(msg)
                return

            # Skills/interactive tasks need the main agent — queue instead of spawning
            if self._needs_main_agent(text):
                logger.info("Session {} busy, queuing skill/interactive request: '{}'",
                            msg.session_key, text[:60])
                self._pending_messages.setdefault(msg.session_key, []).append(msg)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="I'll handle that as soon as I'm done with the current task.",
                    metadata={"_tts_sentence": True},
                ))
                return

            logger.info("Session {} busy, auto-spawning for: '{}'",
                        msg.session_key, msg.content[:60])
            # Notify the channel
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=msg.content,
                metadata={"_parallel": True},
            ))
            # Spawn as a subagent — runs in parallel with its own context
            await self.subagents.spawn(
                task=msg.content,
                label=msg.content[:30],
                origin_channel=msg.channel,
                origin_chat_id=msg.chat_id,
                session_key=msg.session_key,
            )
            return

        async with lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))
            finally:
                # Process queued messages
                pending = self._pending_messages.pop(msg.session_key, [])
                if pending:
                    key = msg.session_key
                    session = self.sessions.get_or_create(key)
                    # Separate: conversational → add to history, tasks → re-dispatch
                    for pending_msg in pending:
                        if self._needs_main_agent(pending_msg.content):
                            # Re-dispatch skill/interactive requests to main agent
                            logger.info("Re-dispatching queued skill request: '{}'",
                                        pending_msg.content[:60])
                            asyncio.create_task(self._dispatch(pending_msg))
                        else:
                            session.add_message("user", pending_msg.content)
                            logger.info("Appended queued conversational msg to session: '{}'",
                                        pending_msg.content[:40])
                    self.sessions.save(session)

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _handle_profile_cmd(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /profile [list|<name>] command."""
        logger.debug("_handle_profile_cmd: content={!r} profiles={}", msg.content, list(self._profiles))
        parts = msg.content.strip().split(maxsplit=1)
        sub = parts[1].strip() if len(parts) > 1 else "list"

        if sub == "list" or not sub:
            if not self._profiles:
                content = "No profiles configured. Add a 'profiles' section to config.json or config.yaml."
            else:
                lines = ["Available profiles:"]
                for name, p in self._profiles.items():
                    marker = " ✓ (active)" if name == self._active_profile else ""
                    model = p.model or "—"
                    provider = p.provider or "—"
                    lines.append(f"  {name}{marker}  [{provider} / {model}]")
                if self._active_profile is None:
                    lines.append("\nNo profile active (using config defaults).")
                content = "\n".join(lines)
            logger.debug("_handle_profile_cmd: returning list response")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        name = sub
        if name not in self._profiles:
            available = ", ".join(self._profiles) or "(none)"
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Profile '{name}' not found. Available: {available}",
            )

        if not self._profile_factory:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="Profile switching not available in this mode.",
            )

        try:
            logger.debug("_handle_profile_cmd: switching to profile {!r}", name)
            new_provider = self._profile_factory(name)
            self.provider = new_provider
            profile = self._profiles[name]
            if profile.model:
                self.model = profile.model
            self._active_profile = name
            parts_info = []
            if profile.provider:
                parts_info.append(f"provider: {profile.provider}")
            if profile.model:
                parts_info.append(f"model: {profile.model}")
            detail = f" ({', '.join(parts_info)})" if parts_info else ""
            logger.debug("_handle_profile_cmd: switched OK")
            if self._profile_save_callback:
                try:
                    self._profile_save_callback(name)
                    saved = " (saved to config)"
                except Exception as save_err:
                    logger.warning("Failed to save profile to config: {}", save_err)
                    saved = " (not saved)"
            else:
                saved = ""
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Switched to profile: **{name}**{detail}{saved}",
            )
        except BaseException as e:
            logger.exception("_handle_profile_cmd: error switching to profile {!r}", name)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"Failed to switch to profile '{name}': {e}",
            )

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )

            # Voice channels: use streaming TTS for subagent summaries too
            _sys_is_voice = channel in ("discord_voice", "web_voice")
            if _sys_is_voice:
                async def _sys_on_sentence(sentence: str) -> None:
                    from nanobot.channels.web_voice import _strip_markdown
                    clean = _strip_markdown(sentence)
                    if clean and len(clean) >= 3:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=channel, chat_id=chat_id,
                            content=clean,
                            metadata={"_tts_sentence": True},
                        ))

                final_content, _, all_msgs, _sys_usage = await self._run_agent_loop_streaming(
                    messages, on_sentence=_sys_on_sentence,
                )
            else:
                final_content, _, all_msgs, _sys_usage = await self._run_agent_loop(messages)

            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

            final_content = final_content or "Background task completed."
            if _sys_is_voice:
                # Already TTS'd via streaming — just send text for display (no _subagent_result to avoid double TTS)
                logger.info("Voice response (system) to {}:{}: {}", channel, chat_id, final_content[:120])
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel, chat_id=chat_id, content=final_content,
                ))
                return None

            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            snapshot = session.messages[session.last_consolidated:]
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            if snapshot:
                self._schedule_background(self.memory_consolidator.archive_messages(snapshot))

            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 nanobot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/profile [name|list] — Switch provider profile",
                "/doctor — Run health checks",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        if cmd.startswith("/profile"):
            return self._handle_profile_cmd(msg)
        if cmd == "/doctor":
            from nanobot.hooks.builtin.health import run_doctor
            report = await run_doctor(
                workspace=self.workspace,
                provider=self.provider,
                model=self.model,
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=report)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        # Reflection: check if this message is correcting the previous turn
        self._schedule_background(self.reflection.check_for_correction(key, msg.content))

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # Lifecycle tracking
        from nanobot.hooks.builtin.lifecycle import mark_turn_started
        mark_turn_started(key, msg.channel)
        _turn_t0 = __import__("time").monotonic()

        _is_voice = msg.channel in ("discord_voice", "web_voice")

        history = session.get_history(max_messages=0)
        enriched_content = msg.content

        # Mask secrets in user message before LLM sees them
        from nanobot.hooks.builtin.secret_mask import detect_and_mask_secrets, save_detected_secrets
        enriched_content, detected_secrets = detect_and_mask_secrets(enriched_content)
        if detected_secrets:
            self._schedule_background(save_detected_secrets(detected_secrets))

        # Inject running subagent context so the main agent can route updates
        running = self.subagents.get_running_tasks()
        if running:
            lines = ["[ACTIVE SUBAGENTS]"]
            for t in running:
                lines.append(f"- id={t['id']} label=\"{t['label']}\" task=\"{t['task'][:80]}\"")
            enriched_content = f"{enriched_content}\n\n" + "\n".join(lines)

        initial_messages = self.context.build_messages(
            history=history,
            current_message=enriched_content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # Voice channels use streaming for low-latency TTS
        _is_voice_channel = msg.channel in ("discord_voice", "web_voice")

        if _is_voice_channel:
            # Streaming: fire TTS per sentence as tokens arrive
            async def _on_sentence(sentence: str) -> None:
                from nanobot.channels.web_voice import _strip_markdown
                clean = _strip_markdown(sentence)
                if clean and len(clean) >= 3:
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content=clean,
                        metadata={"_tts_sentence": True},
                    ))

            final_content, _, all_msgs, turn_usage = await self._run_agent_loop_streaming(
                initial_messages, on_sentence=_on_sentence,
                on_progress=on_progress or _bus_progress,
            )
        else:
            final_content, _, all_msgs, turn_usage = await self._run_agent_loop(
                initial_messages, on_progress=on_progress or _bus_progress,
            )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        # Broadcast token usage as activity entry
        if turn_usage and turn_usage.get("total_tokens", 0) > 0:
            from nanobot.hooks.builtin.usage_tracker import record_usage, estimate_cost
            cost = estimate_cost(turn_usage, self.model)
            cost_str = f" · ${cost:.4f}" if cost > 0 else ""
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"⬡ {turn_usage['prompt_tokens']:,} in · {turn_usage['completion_tokens']:,} out · {turn_usage['total_tokens']:,} total{cost_str}",
                metadata={"_token_usage": True, "_usage_data": {**turn_usage, "cost": cost}},
            ))
            # Persist to daily usage file
            self._schedule_background(record_usage(self.workspace, turn_usage, msg.channel, self.model))

        # Emit turn_completed for lifecycle tracking
        _turn_elapsed = (__import__("time").monotonic() - _turn_t0) * 1000
        from nanobot.hooks.events import TurnCompleted
        await self.hooks.emit("turn_completed", TurnCompleted(
            session_key=key, final_content=final_content,
            tools_used=[], iterations=0, duration_ms=_turn_elapsed,
            channel=msg.channel, chat_id=msg.chat_id,
        ))

        # Reflection: record this turn for future correction detection
        self.reflection.on_turn_completed(key, msg.content, final_content or "")

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        # Voice channel: route response back through TTS
        if _is_voice_channel:
            preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
            logger.info("Voice response to {}:{}: {}", msg.channel, msg.sender_id, preview)
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            ))
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""

