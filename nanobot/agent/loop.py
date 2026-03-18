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
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.restart import RestartTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ProfileConfig, ToolsDNSConfig, WebSearchConfig
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
        toolsdns_config: ToolsDNSConfig | None = None,
        profiles: "dict[str, ProfileConfig] | None" = None,
        profile_factory: "Callable[[str], LLMProvider] | None" = None,
        profile_save_callback: "Callable[[str], None] | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._toolsdns_config = toolsdns_config
        self._profiles: dict[str, "ProfileConfig"] = profiles or {}
        self._profile_factory = profile_factory
        self._profile_save_callback = profile_save_callback
        self._active_profile: str | None = None

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
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
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        if self._toolsdns_config and self._toolsdns_config.enabled:
            from nanobot.agent.tools.toolsdns import ToolsDNSTool
            self.tools.register(ToolsDNSTool(
                base_url=self._toolsdns_config.url,
                api_key=self._toolsdns_config.api_key,
                timeout=self._toolsdns_config.timeout,
            ))

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

    # ------------------------------------------------------------------
    # ToolsDNS pre-flight search (multi-strategy)
    # ------------------------------------------------------------------

    _TOOLSDNS_SKIP_PATTERNS = re.compile(
        r"^(/\w|hi\b|hello\b|hey\b|thanks|ok\b|yes\b|no\b|good|how are|what's up)",
        re.IGNORECASE,
    )

    _QUERY_CLEAN_RE = re.compile(
        r"\S+@\S+\.\S+|https?://\S+|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b",
    )

    # Intent patterns: natural language → tool-friendly search queries
    # Each entry: (regex, [list of search queries])
    # Multiple queries per pattern: descriptive (matches description) + name-style (matches tool name)
    # ToolsDNS searches both name and description fields, so we hit both angles.
    _INTENT_MAP = [
        # Email — send
        (re.compile(r"\b(send|write|draft|compose|shoot|fire off)\b.*\b(email|mail|message|note)\b", re.I),
         ["GMAIL_SEND_EMAIL", "gmail send email", "send an email message"]),
        (re.compile(r"\b(check|read|fetch|get|see|look at)\b.*\b(email|mail|inbox)\b", re.I),
         ["GMAIL_FETCH_EMAILS", "gmail get inbox emails", "fetch email messages"]),
        (re.compile(r"\b(reply|respond)\b.*\b(email|mail|thread)\b", re.I),
         ["GMAIL_REPLY_TO_THREAD", "gmail reply email thread"]),
        # Slack
        (re.compile(r"\b(send|post|write|notify|tell|message)\b.*\b(slack|channel|team)\b", re.I),
         ["SLACK_SEND_MESSAGE", "SLACK_SENDS_A_MESSAGE_TO_A_SLACK", "slack send message channel"]),
        (re.compile(r"\b(slack)\b", re.I),
         ["SLACK_SEND_MESSAGE", "slack channel message"]),
        # Calendar
        (re.compile(r"\b(schedule|create|book|set up|add)\b.*\b(meeting|event|appointment|call|calendar)\b", re.I),
         ["GOOGLECALENDAR_CREATE_EVENT", "google calendar create event meeting"]),
        (re.compile(r"\b(check|show|list|what's on|see)\b.*\b(calendar|schedule|agenda|meetings)\b", re.I),
         ["GOOGLECALENDAR_FIND_EVENT", "google calendar list events schedule"]),
        # GitHub
        (re.compile(r"\b(create|open|file|submit)\b.*\b(issue|bug|ticket)\b", re.I),
         ["GITHUB_CREATE_AN_ISSUE", "github create issue bug"]),
        (re.compile(r"\b(create|open|submit)\b.*\b(pr|pull request)\b", re.I),
         ["GITHUB_CREATE_A_PULL_REQUEST", "github create pull request"]),
        (re.compile(r"\b(merge|review)\b.*\b(pr|pull request)\b", re.I),
         ["GITHUB_MERGE_A_PULL_REQUEST", "github merge pull request review"]),
        (re.compile(r"\b(star|fork|clone)\b.*\b(repo|repository)\b", re.I),
         ["GITHUB_STAR_A_REPOSITORY", "github star fork repository"]),
        (re.compile(r"\b(github|repo)\b", re.I),
         ["GITHUB_LIST_REPOS", "github repository actions"]),
        # Browser — try name patterns for playwright tools
        (re.compile(r"\b(open|go to|navigate|visit|browse|check)\b.*\b(website|page|site|url|link)\b", re.I),
         ["browser_navigate", "playwright navigate open webpage url"]),
        (re.compile(r"\b(click|press|tap)\b.*\b(button|link|element)\b", re.I),
         ["browser_click", "playwright click button element"]),
        (re.compile(r"\b(fill|type|enter|input)\b.*\b(form|field|text|box)\b", re.I),
         ["browser_fill", "playwright fill form input type"]),
        (re.compile(r"\b(screenshot|capture|snap)\b", re.I),
         ["browser_screenshot", "playwright screenshot capture page"]),
        (re.compile(r"\b(browse|search the web|look up|google)\b", re.I),
         ["browser_navigate", "web browser search navigate"]),
        # Salesforce
        (re.compile(r"\b(salesforce|sfdc|sf)\b.*\b(task|create|check)\b", re.I),
         ["SALESFORCE_CREATE_TASK", "salesforce create task opportunity"]),
        (re.compile(r"\b(salesforce|sfdc|sf)\b.*\b(contact|lead|account)\b", re.I),
         ["SALESFORCE_FETCH_CONTACT", "salesforce contact lead account"]),
        (re.compile(r"\b(salesforce|sfdc|sf)\b", re.I),
         ["SALESFORCE", "salesforce CRM"]),
        # Files & docs
        (re.compile(r"\b(create|generate|make)\b.*\b(spreadsheet|excel|csv|sheet)\b", re.I),
         ["GOOGLESHEETS_CREATE_GOOGLE_SHEET", "google sheets create spreadsheet"]),
        (re.compile(r"\b(create|write|generate|make)\b.*\b(doc|document|report)\b", re.I),
         ["GOOGLEDOCS_CREATE_DOCUMENT", "google docs create document write"]),
        (re.compile(r"\b(upload|download|share)\b.*\b(file|document|pdf)\b", re.I),
         ["GOOGLEDRIVE_UPLOAD_FILE", "google drive upload download file"]),
        # Tasks / todo
        (re.compile(r"\b(create|add|make)\b.*\b(task|todo|reminder)\b", re.I),
         ["create task todo", "TODOIST_CREATE_TASK"]),
        (re.compile(r"\b(list|show|check)\b.*\b(tasks?|todos?)\b", re.I),
         ["list tasks", "TODOIST_GET_TASKS"]),
        # Twitter / social
        (re.compile(r"\b(tweet|post|publish)\b.*\b(twitter|x\.com|social)\b", re.I),
         ["TWITTER_CREATION_OF_A_TWEET", "twitter post tweet publish"]),
        (re.compile(r"\b(tweet|post on twitter)\b", re.I),
         ["TWITTER_CREATION_OF_A_TWEET", "twitter create tweet"]),
        # Notion
        (re.compile(r"\b(notion)\b.*\b(page|create|add|write)\b", re.I),
         ["NOTION_CREATE_A_PAGE", "notion create page database"]),
        (re.compile(r"\b(notion)\b", re.I),
         ["NOTION", "notion page workspace"]),
        # Linear
        (re.compile(r"\b(linear)\b.*\b(issue|ticket|bug)\b", re.I),
         ["LINEAR_CREATE_LINEAR_ISSUE", "linear create issue ticket"]),
        (re.compile(r"\b(linear)\b", re.I),
         ["LINEAR", "linear project issue"]),
        # Report / weekly
        (re.compile(r"\b(weekly|report|fill)\b.*\b(report|cea|weekly)\b", re.I),
         ["weekly report", "CEA weekly report fill"]),
        # Jira
        (re.compile(r"\b(jira)\b.*\b(issue|ticket|create)\b", re.I),
         ["JIRA_CREATE_ISSUE", "jira create issue ticket"]),
        (re.compile(r"\b(jira)\b", re.I),
         ["JIRA", "jira project issue"]),
        # Discord
        (re.compile(r"\b(discord)\b.*\b(send|message|post)\b", re.I),
         ["DISCORD_SEND_MESSAGE", "discord send message channel"]),
        (re.compile(r"\b(discord)\b", re.I),
         ["DISCORD", "discord server channel"]),
        # Telegram
        (re.compile(r"\b(telegram)\b.*\b(send|message)\b", re.I),
         ["TELEGRAM_SEND_MESSAGE", "telegram send message chat"]),
        (re.compile(r"\b(telegram)\b", re.I),
         ["TELEGRAM", "telegram bot message"]),
        # WhatsApp
        (re.compile(r"\b(whatsapp|whats app)\b", re.I),
         ["WHATSAPP_SEND_MESSAGE", "whatsapp send message"]),
        # Generic send/notify — broad fallback
        (re.compile(r"\b(notify|alert|inform|tell)\b.*\b(someone|team|user|them)\b", re.I),
         ["send notification", "SLACK_SEND_MESSAGE", "GMAIL_SEND_EMAIL"]),
    ]

    @staticmethod
    def _clean_query(text: str) -> str:
        """Strip user data (emails, URLs, numbers) to get a cleaner tool search query."""
        cleaned = AgentLoop._QUERY_CLEAN_RE.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if len(cleaned) >= 8 else text

    @staticmethod
    def _extract_intent_queries(text: str) -> list[str]:
        """Extract tool-friendly search queries from natural language using intent patterns.

        Each intent map entry returns multiple queries (name-style + descriptive).
        We collect from the first matching pattern (most specific) and deduplicate.
        """
        queries: list[str] = []
        seen = set()
        matched = 0
        for pattern, query_list in AgentLoop._INTENT_MAP:
            if pattern.search(text):
                if isinstance(query_list, str):
                    query_list = [query_list]
                for q in query_list:
                    q_lower = q.lower()
                    if q_lower not in seen:
                        seen.add(q_lower)
                        queries.append(q)
                matched += 1
                if matched >= 2 or len(queries) >= 4:
                    break
        return queries

    @staticmethod
    def _compact_schema(schema: dict, max_params: int = 10) -> str:
        """Build a compact one-line-per-param schema summary."""
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        if not props:
            return "    (no parameters)"
        lines = []
        for name, info in list(props.items())[:max_params]:
            ptype = info.get("type", "any")
            desc = info.get("description", "").split(".")[0].split("\n")[0][:60]
            req = " [REQUIRED]" if name in required else ""
            default = f" (default: {info['default']})" if "default" in info else ""
            lines.append(f"      {name}: {ptype}{req}{default} — {desc}")
        if len(props) > max_params:
            lines.append(f"      ... and {len(props) - max_params} more params")
        return "\n".join(lines)

    async def _toolsdns_preflight(self, user_message: str) -> str | None:
        """
        Call ToolsDNS server-side preflight endpoint.

        The /v1/preflight endpoint handles all intelligence:
        - Query cleaning (strip emails, URLs, dates)
        - Intent extraction (regex → tool-name + descriptive queries)
        - Multi-strategy parallel search
        - Merge, dedup, rank
        - Context block generation with schemas + call templates
        """
        if not self._toolsdns_config or not self._toolsdns_config.enabled:
            return None

        text = user_message.strip()
        if len(text) < 10 or self._TOOLSDNS_SKIP_PATTERNS.match(text):
            return None

        try:
            import httpx
            url = self._toolsdns_config.url.rstrip("/")
            headers = {
                "Authorization": f"Bearer {self._toolsdns_config.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{url}/v1/preflight",
                    headers=headers,
                    json={
                        "message": text,
                        "top_k": 5,
                        "threshold": 0.1,
                        "max_results": 5,
                        "include_schemas": True,
                        "include_call_templates": True,
                        "include_macros": True,
                        "agent_id": "mawa",
                        "format": "context_block",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            if not data.get("found"):
                return None

            context_block = data.get("context_block")
            if not context_block:
                return None

            queries = data.get("queries_used", [])
            tools_count = len(data.get("tools", []))
            query_info = " + ".join(queries[:3])
            logger.info("ToolsDNS preflight: {} tools found via {} queries ('{}')",
                         tools_count, len(queries), query_info[:80])
            return context_block

        except Exception as e:
            logger.debug("ToolsDNS preflight failed (non-blocking): {}", e)
            return None

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        _last_call_sig: str | None = None  # detect repeated identical tool calls
        _repeat_count = 0

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

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

                for tool_call in response.tool_calls:
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

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
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
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            try:
                os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])
            except Exception as e:
                logger.error("Restart failed (os.execv): {}", e)

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
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
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

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
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        if cmd.startswith("/profile"):
            return self._handle_profile_cmd(msg)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # ToolsDNS pre-flight: search for relevant tools before LLM starts
        toolsdns_context = await self._toolsdns_preflight(msg.content)

        history = session.get_history(max_messages=0)
        enriched_content = msg.content
        if toolsdns_context:
            enriched_content = f"{msg.content}\n\n{toolsdns_context}"

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

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        # Voice channel: route response back through TTS instead of text
        if msg.channel == "discord_voice":
            await self.bus.publish_outbound(OutboundMessage(
                channel="discord_voice", chat_id=msg.chat_id, content=final_content,
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

