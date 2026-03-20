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
from nanobot.agent.subagent import PreflightCache, SubagentManager
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
        self._toolsdns_config = toolsdns_config
        self._profiles: dict[str, "ProfileConfig"] = profiles or {}
        self._profile_factory = profile_factory
        self._profile_save_callback = profile_save_callback
        self._active_profile: str | None = None

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self._preflight_cache = PreflightCache()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            toolsdns_config=toolsdns_config,
            session_manager=self.sessions,
            preflight_cache=self._preflight_cache,
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
        r"^(/\w|hi\b|hello\b|hey\b|thanks|ok\b|yes\b|no\b|good|how are|what's up"
        r"|why\b|what do you|who are you|tell me about yourself|how do you|are you"
        r"|do you|can you help|what can you|nice|cool|great|sure|alright|bye|see you"
        r"|sorry|excuse me|never ?mind|forget it|stop|wait|hold on|go ahead)",
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
        # Weather
        (re.compile(r"\b(weather|temperature|forecast|rain|snow|humid|wind)\b", re.I),
         ["WEATHERMAP_WEATHER", "WEATHERMAP_GEOCODE_LOCATION", "weather temperature forecast"]),
        # Generic send/notify — broad fallback
        (re.compile(r"\b(notify|alert|inform|tell)\b.*\b(someone|team|user|them)\b", re.I),
         ["send notification", "SLACK_SEND_MESSAGE", "GMAIL_SEND_EMAIL"]),
    ]

    # ── Meta-tools to filter from preflight results ──
    _META_TOOL_PREFIXES = frozenset({
        "COMPOSIO_SEARCH_TOOLS",
        "COMPOSIO_GET_TOOL_SCHEMAS",
        "COMPOSIO_MANAGE_CONNECTIONS",
        "COMPOSIO_MULTI_EXECUTE_TOOL",
        "COMPOSIO_CHECK_CONNECTION",
    })

    # ── App prefix pattern: Composio tools follow APPNAME_TOOLACTION ──
    _APP_PREFIX_RE = re.compile(r"^([A-Z][A-Z0-9]+)_")

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

    @staticmethod
    def _extract_app_prefix(tool_name: str) -> str | None:
        """Extract app prefix from tool name (e.g., GMAIL_SEND_EMAIL → GMAIL)."""
        m = AgentLoop._APP_PREFIX_RE.match(tool_name)
        return m.group(1) if m else None

    @staticmethod
    def _detect_dominant_app(tools: list[dict]) -> str | None:
        """Find the most common app prefix among non-meta tools."""
        from collections import Counter
        prefixes = []
        for t in tools:
            name = t.get("name", "")
            if name in AgentLoop._META_TOOL_PREFIXES:
                continue
            prefix = AgentLoop._extract_app_prefix(name)
            if prefix and prefix != "COMPOSIO":
                prefixes.append(prefix)
        if not prefixes:
            return None
        # Return the most common prefix
        counts = Counter(prefixes)
        return counts.most_common(1)[0][0]

    async def _toolsdns_preflight(self, user_message: str, timeout: float = 8.0) -> str | None:
        """
        Two-tier ToolsDNS preflight with app-aware fallback.

        Tier 1: Standard preflight search across all tools.
        Tier 2: If results only contain meta-tools (COMPOSIO_SEARCH_TOOLS etc.),
                 detect the app from user intent and do a targeted search
                 within that app's tools.
        """
        if os.environ.get("NANOBOT_TOOLSDNS_PREFLIGHT", "1").lower() in ("0", "false", "off", "no"):
            return None
        if not self._toolsdns_config or not self._toolsdns_config.enabled:
            return None

        text = user_message.strip()
        if len(text) < 10 or self._TOOLSDNS_SKIP_PATTERNS.match(text):
            return None

        # Check preflight cache first
        cached = self._preflight_cache.get(text)
        if cached is not PreflightCache._MISS:
            if cached:
                logger.info("ToolsDNS preflight cache hit for: '{}'", text[:60])
            return cached

        try:
            import httpx
            url = self._toolsdns_config.url.rstrip("/")
            headers = {
                "Authorization": f"Bearer {self._toolsdns_config.api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=timeout) as client:
                # ── Tier 1: Preflight with context_block format ──
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

            # Check tool names in context_block for meta-tools
            context_block = data.get("context_block", "")
            has_meta = any(meta in context_block for meta in self._META_TOOL_PREFIXES)

            # Check if intent map detects a specific app for this query
            app_prefix = None
            intent_queries = self._extract_intent_queries(text)
            for iq in intent_queries:
                prefix = self._extract_app_prefix(iq)
                if prefix and prefix != "COMPOSIO":
                    app_prefix = prefix
                    break

            # If intent map detected an app, check if tier 1 results contain it
            app_missing = False
            if app_prefix and context_block:
                app_missing = (app_prefix + "_") not in context_block

            if data.get("found") and context_block and not has_meta and not app_missing:
                # Tier 1 found relevant tools — use as-is
                queries = data.get("queries_used", [])
                logger.info("ToolsDNS preflight: tools found via '{}'",
                             " + ".join(queries[:3])[:80])
                self._preflight_cache.set(text, context_block)
                return context_block

            # ── Tier 2: Targeted search to find real tools ──
            # Triggered when: meta-tools detected, or expected app missing from results

            logger.info("ToolsDNS tier 2: {} for '{}'",
                         f"app={app_prefix}" if app_prefix else "broad re-search",
                         text[:60])
            try:
                all_results: list[dict] = []
                seen_ids: set[str] = set()

                async with httpx.AsyncClient(timeout=timeout) as client:
                    # Strategy A: Fetch exact tools by ID from intent map
                    # (bypasses semantic search — guaranteed to find the right tools)
                    exact_tool_names = [
                        iq for iq in intent_queries
                        if self._extract_app_prefix(iq) is not None
                    ]
                    for tool_name in exact_tool_names:
                        tool_id = f"tooldns__{tool_name}"
                        if tool_id in seen_ids:
                            continue
                        try:
                            resp_t = await client.get(
                                f"{url}/v1/tool/{tool_id}",
                                headers=headers,
                            )
                            if resp_t.status_code == 200:
                                tool_data = resp_t.json()
                                tool_data["confidence"] = 0.95  # high confidence — exact match
                                seen_ids.add(tool_id)
                                all_results.append(tool_data)
                        except Exception:
                            pass

                    # Strategy B: Semantic search as supplement
                    cleaned = self._clean_query(text)
                    search_queries = []
                    if app_prefix:
                        search_queries.append(f"{app_prefix} {cleaned}")
                    # Add descriptive intent queries (not tool names)
                    for iq in intent_queries:
                        if self._extract_app_prefix(iq) is None and iq not in search_queries:
                            search_queries.append(iq)
                    if cleaned not in search_queries:
                        search_queries.append(cleaned)

                    for sq in search_queries[:3]:
                        resp3 = await client.post(
                            f"{url}/v1/search",
                            headers=headers,
                            json={"query": sq, "top_k": 8, "threshold": 0.3},
                        )
                        resp3.raise_for_status()
                        for t in resp3.json().get("results", []):
                            tid = t.get("id", t.get("name", ""))
                            if tid not in seen_ids and t.get("name") not in self._META_TOOL_PREFIXES:
                                seen_ids.add(tid)
                                all_results.append(t)

                # If we know the app, prefer tools from that app
                if app_prefix:
                    app_tools = [t for t in all_results if t.get("name", "").startswith(app_prefix + "_")]
                    if app_tools:
                        all_results = app_tools

                # Sort by confidence and take top 5
                all_results.sort(key=lambda t: t.get("confidence", 0), reverse=True)
                if all_results:
                    result_block = self._build_context_block(all_results[:5])
                    names = [t["name"] for t in all_results[:5]]
                    logger.info("ToolsDNS tier 2: found {} tools: {}", len(names), ", ".join(names))
                    self._preflight_cache.set(text, result_block)
                    return result_block
            except Exception as e:
                logger.debug("ToolsDNS tier 2 search failed: {}", e)

            # Fallback: return original context_block if available
            if data.get("found") and context_block:
                self._preflight_cache.set(text, context_block)
                return context_block

            self._preflight_cache.set(text, None)
            return None

        except Exception as e:
            logger.debug("ToolsDNS preflight failed (non-blocking): {}", e)
            return None

    @staticmethod
    def _build_context_block(tools: list[dict]) -> str:
        """Build a context block matching the ToolsDNS preflight format."""
        lines = ["[ToolsDNS Auto-Discovery] — tools already found, DO NOT search again.\n"]

        # Build call template for the best match
        if tools:
            best = tools[0]
            best_name = best.get("name", "unknown")
            best_id = f"tooldns__{best_name}"
            # Build example arguments from required params + key optional ones
            schema = best.get("input_schema", {})
            required = schema.get("required", [])
            props = schema.get("properties", {})
            example_args = {}
            # Always include required params
            for param in required:
                ptype = props.get(param, {}).get("type", "string")
                if ptype == "string":
                    example_args[param] = f"<{param}>"
                elif ptype == "number":
                    example_args[param] = 0
                elif ptype == "boolean":
                    example_args[param] = True
                else:
                    example_args[param] = f"<{param}>"
            # If no required params, include up to 3 useful optional params
            # so the LLM knows to pass useful arguments
            if not required:
                for param, info in list(props.items())[:3]:
                    ptype = info.get("type", "string")
                    if ptype == "string":
                        example_args[param] = f"<{param}>"
                    elif ptype == "integer" or ptype == "number":
                        example_args[param] = 10
                    elif ptype == "boolean":
                        example_args[param] = True
                    else:
                        example_args[param] = f"<{param}>"
            import json as _json
            args_str = _json.dumps(example_args) if example_args else "{}"
            best_desc = best.get("description", "").split("\n")[0][:80]
            lines.append(f'>>> CALL THIS NOW: toolsdns(action="call", tool_id="{best_id}", arguments={args_str})')
            lines.append(f"    (tool: {best_name} — {best_desc})\n")

        lines.append("RULES: Your FIRST tool call must be toolsdns action=call. Do NOT action=search. Do NOT action=get. Do NOT use mcp_tooldns_* tools.\n")

        for i, t in enumerate(tools):
            name = t.get("name", "unknown")
            tool_id = f"tooldns__{name}"
            desc = t.get("description", "").split("\n")[0][:200]
            confidence = t.get("confidence", 0)
            label = "BEST MATCH" if i == 0 else f"Match {i + 1}"
            lines.append(f"{label}  [{confidence:.0%}]  TOOL_ID: {tool_id}")
            lines.append(f"  {name} — {desc}")

            schema = t.get("input_schema", {})
            if schema and schema.get("properties"):
                req_set = set(schema.get("required", []))
                for pname, info in list(schema["properties"].items())[:10]:
                    ptype = info.get("type", "any")
                    pdesc = info.get("description", "").split(".")[0].split("\n")[0][:60]
                    req = " [REQUIRED]" if pname in req_set else ""
                    lines.append(f"    {pname}: {ptype}{req} — {pdesc}")
                # Add a CALL example for non-best-match tools
                if i > 0:
                    call_args = {}
                    for pname in req_set:
                        call_args[pname] = f"<{pname}>"
                    lines.append(f'  CALL: toolsdns(action="call", tool_id="{tool_id}", arguments={_json.dumps(call_args)})')
                if not req_set:
                    lines.append("  NOTE: All params optional — fill in what's relevant to the user's request")
            lines.append("")
        return "\n".join(lines)

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

            # Use fast routing model for first iteration if configured
            current_model = self.routing_model if (iteration == 1 and self.routing_model) else self.model

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=current_model,
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

        return final_content, tools_used, messages

    async def _run_agent_loop_streaming(
        self,
        initial_messages: list[dict],
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Streaming variant of agent loop — fires on_sentence as sentences complete.

        Used by voice channels for low-latency TTS: each sentence is sent to TTS
        as soon as it's accumulated from the token stream, rather than waiting for
        the full response.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
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
        if len(stripped) < 15 and self._TOOLSDNS_SKIP_PATTERNS.match(stripped):
            return True
        # Single/two word acknowledgments
        normalized = stripped.lower().rstrip(".!?,")
        if len(normalized.split()) <= 2 and normalized in self._CONVERSATIONAL_WORDS:
            return True
        return False

    # Dynamic skill pattern cache — built from skill names/trigger phrases
    _skill_pattern: re.Pattern | None = None
    _skill_cache_ts: float = 0

    def _needs_main_agent(self, text: str) -> bool:
        """Check if message matches a known skill (requires interactive main agent).

        Dynamically loads skill names and trigger phrases from the workspace
        and ToolsDNS. Refreshes every 5 minutes. Matches on skill names as
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
        """Build skill match patterns from ToolsDNS skills and workspace skills."""
        phrases: set[str] = {"skill"}  # "use the skill" always routes to main agent

        # Primary source: ToolsDNS skills API (has all skills with descriptions)
        if self._toolsdns_config and self._toolsdns_config.enabled:
            try:
                import httpx
                url = self._toolsdns_config.url.rstrip("/")
                resp = httpx.get(
                    f"{url}/v1/skills",
                    headers={"Authorization": f"Bearer {self._toolsdns_config.api_key}"},
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    for skill_data in resp.json().get("skills", []):
                        phrases |= self._extract_skill_phrases(
                            skill_data.get("name", ""),
                            skill_data.get("description", ""),
                        )
            except Exception:
                pass

        # Fallback: workspace skills directory
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
        lock = self._get_session_lock(msg.session_key)

        if lock.locked():
            text = msg.content.strip()

            if self._is_conversational(text):
                logger.info("Session {} busy, queuing conversational: '{}'",
                            msg.session_key, text[:40])
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
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
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

                final_content, _, all_msgs = await self._run_agent_loop_streaming(
                    messages, on_sentence=_sys_on_sentence,
                )
            else:
                final_content, _, all_msgs = await self._run_agent_loop(messages)

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

        # ToolsDNS pre-flight + speech intel run in parallel for voice channels
        _is_voice = msg.channel in ("discord_voice", "web_voice")
        toolsdns_context = await self._toolsdns_preflight(
            msg.content, timeout=3.0 if _is_voice else 8.0,
        )

        history = session.get_history(max_messages=0)
        enriched_content = msg.content
        if voice_hint := (msg.metadata or {}).get("voice_instruction"):
            enriched_content = f"{voice_hint}\n\n{enriched_content}"

        # Inject running subagent context so the main agent can route updates
        running = self.subagents.get_running_tasks()
        if running:
            lines = ["[ACTIVE SUBAGENTS]"]
            for t in running:
                lines.append(f"- id={t['id']} label=\"{t['label']}\" task=\"{t['task'][:80]}\"")
            enriched_content = f"{enriched_content}\n\n" + "\n".join(lines)

        if toolsdns_context:
            enriched_content = f"{enriched_content}\n\n{toolsdns_context}"

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

            final_content, _, all_msgs = await self._run_agent_loop_streaming(
                initial_messages, on_sentence=_on_sentence,
                on_progress=on_progress or _bus_progress,
            )
        else:
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

