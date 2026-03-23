"""4-layer memory system for persistent agent memory.

Layers:
  SHORT_TERM.md  — Today's context (auto-cleared daily, injected every turn)
  LONG_TERM.md   — Permanent facts (searched on demand via memory_search)
  OBSERVATIONS.md — Patterns detected from tool usage behavior
  EPISODES.md    — Key moments worth remembering
  HISTORY.md     — Grep-searchable timestamped log (unchanged)
"""

from __future__ import annotations

import asyncio
import json
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save the memory consolidation result to the 4-layer memory system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. "
                        "Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search.",
                    },
                    "long_term": {
                        "type": "string",
                        "description": "Full updated permanent memory as markdown. Include all existing "
                        "facts plus new ones (name, preferences, relationships, active projects). "
                        "Return unchanged if nothing new to add.",
                    },
                    "short_term": {
                        "type": "string",
                        "description": "What happened in this conversation chunk: tasks done, "
                        "topics discussed, current context. Brief, 2-3 sentences max.",
                    },
                    "episode": {
                        "type": "string",
                        "description": "A significant moment worth remembering long-term: key decisions, "
                        "milestones, emotional events, breakthroughs. Only set if something notable happened. "
                        "Start with [YYYY-MM-DD HH:MM].",
                    },
                    "behavioral_insight": {
                        "type": "string",
                        "description": "If you noticed a user preference, communication style, or behavioral "
                        "pattern (e.g., 'user prefers concise summaries', 'user dislikes verbose responses', "
                        "'user always checks email first'), save it here. Only set if genuinely new insight.",
                    },
                },
                "required": ["history_entry", "long_term"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


class MemoryStore:
    """4-layer memory with daily lifecycle.

    Files:
      LONG_TERM.md   — permanent facts (full rewrite on consolidation)
      SHORT_TERM.md  — today's context (append, auto-cleared daily)
      OBSERVATIONS.md — behavioral patterns (managed by PatternObserver)
      EPISODES.md    — significant moments (append)
      HISTORY.md     — grep-searchable log (append)
    """

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.long_term_file = self.memory_dir / "LONG_TERM.md"
        self.short_term_file = self.memory_dir / "SHORT_TERM.md"
        self.observations_file = self.memory_dir / "OBSERVATIONS.md"
        self.episodes_file = self.memory_dir / "EPISODES.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._consecutive_failures = 0

        # Backward compat: migrate MEMORY.md → LONG_TERM.md on first access
        old_memory = self.memory_dir / "MEMORY.md"
        if old_memory.exists() and not self.long_term_file.exists():
            old_memory.rename(self.long_term_file)
            logger.info("Migrated MEMORY.md → LONG_TERM.md")

    # -- Backward-compatible aliases for tests and external callers --

    @property
    def memory_file(self) -> Path:
        """Alias for long_term_file (backward compat)."""
        return self.long_term_file

    # -- Read/write for each layer --

    def read_long_term(self) -> str:
        if self.long_term_file.exists():
            return self.long_term_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.long_term_file.write_text(content, encoding="utf-8")

    def read_short_term(self) -> str:
        if self.short_term_file.exists():
            return self.short_term_file.read_text(encoding="utf-8")
        return ""

    def append_short_term(self, entry: str) -> None:
        with open(self.short_term_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n")

    def clear_short_term(self) -> str:
        """Clear SHORT_TERM.md and return the old content (for archival)."""
        content = self.read_short_term()
        if content:
            self.short_term_file.write_text("", encoding="utf-8")
        return content

    def read_observations(self) -> str:
        if self.observations_file.exists():
            return self.observations_file.read_text(encoding="utf-8")
        return ""

    def write_observations(self, content: str) -> None:
        self.observations_file.write_text(content, encoding="utf-8")

    def read_episodes(self) -> str:
        if self.episodes_file.exists():
            return self.episodes_file.read_text(encoding="utf-8")
        return ""

    def append_episode(self, entry: str) -> None:
        with open(self.episodes_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        """Build minimal context for the system prompt.

        Injects SHORT_TERM (today's context) + top observations only.
        LONG_TERM is searched on demand via memory_search tool.
        """
        parts = []

        short_term = self.read_short_term().strip()
        if short_term:
            parts.append(f"## Today\n{short_term}")

        observations = self.read_observations().strip()
        if observations:
            # Only include top 5 observations to keep tokens low
            lines = [l for l in observations.split("\n") if l.strip()]
            top = "\n".join(lines[:5])
            parts.append(f"## Patterns\n{top}")

        # Inject learned rules from user corrections (high priority)
        learnings_file = self.memory_dir / "LEARNINGS.md"
        if learnings_file.exists():
            content = learnings_file.read_text(encoding="utf-8")
            rules = [l for l in content.split("\n") if l.strip().startswith("- ")]
            if rules:
                top_rules = "\n".join(rules[-10:])
                parts.append(f"## User Preferences (MUST follow)\n{top_rules}")

        # Inject tool reliability warnings (lower priority, just awareness)
        tool_learnings_file = self.memory_dir / "TOOL_LEARNINGS.md"
        if tool_learnings_file.exists():
            content = tool_learnings_file.read_text(encoding="utf-8")
            rules = [l for l in content.split("\n") if l.strip().startswith("- ")]
            if rules:
                # Only inject top 5 tool warnings to save tokens
                top_rules = "\n".join(rules[-5:])
                parts.append(f"## Tool Warnings\n{top_rules}")

        # Inject tool reliability warnings from tool_scores.json
        scores_file = self.memory_dir / "tool_scores.json"
        if scores_file.exists():
            try:
                scores = json.loads(scores_file.read_text(encoding="utf-8"))
                warnings = []
                for tool, data in scores.items():
                    total = data.get("success", 0) + data.get("fail", 0)
                    if total >= 5:
                        rate = data["success"] / total
                        if rate < 0.7:
                            warnings.append(f"- {tool}: {int(rate * 100)}% success ({data['fail']} failures)")
                if warnings:
                    parts.append("## Tool Reliability\n" + "\n".join(warnings))
            except (json.JSONDecodeError, OSError, TypeError):
                pass

        # Inject pending goals so agent is always aware of them
        goals_file = self.memory_dir / "GOALS.md"
        if goals_file.exists():
            try:
                content = goals_file.read_text(encoding="utf-8")
                pending = [l.strip()[6:] for l in content.split("\n") if l.strip().startswith("- [ ] ")]
                if pending:
                    overdue = []
                    from datetime import datetime, date
                    import re
                    today = date.today().isoformat()
                    for p in pending:
                        m = re.search(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)", p)
                        if m and m.group(1) < today:
                            overdue.append(p)
                    goals_text = "\n".join(f"- [ ] {p}" for p in pending[:8])
                    if overdue:
                        goals_text = "OVERDUE: " + ", ".join(overdue) + "\n" + goals_text
                    parts.append(f"## Pending Goals ({len(pending)})\n{goals_text}\nUse the goals tool to manage these.")
            except (OSError, TypeError):
                pass

        parts.append("Use memory_search to recall long-term facts, past episodes, or user preferences.")

        return "\n\n".join(parts)

    def daily_cleanup(self) -> None:
        """Archive SHORT_TERM.md to HISTORY.md and clear it.

        Called on date change (first interaction of the day or heartbeat).
        """
        content = self.clear_short_term()
        if content.strip():
            ts = datetime.now().strftime("%Y-%m-%d")
            self.append_history(f"[{ts}] [SHORT_TERM archived]\n{content.strip()}")
            logger.info("Daily cleanup: archived SHORT_TERM.md to HISTORY.md")

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        """Consolidate messages into the 4-layer memory system."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

Categorize information into layers:
- long_term: permanent facts about the user (name, preferences, relationships, projects, skills)
- short_term: what happened in this conversation (tasks, topics, current context) — brief
- episode: only if something truly notable happened (decisions, milestones, emotional moments)
- history_entry: timestamped summary for the searchable log

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": "You are a memory consolidation agent. Categorize conversation content into memory layers by calling the save_memory tool."},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            # Validate required fields (long_term replaces memory_update)
            history_entry = args.get("history_entry")
            long_term = args.get("long_term") or args.get("memory_update")  # backward compat

            if history_entry is None or long_term is None:
                logger.warning("Memory consolidation: save_memory payload missing required fields")
                return self._fail_or_raw_archive(messages)

            entry = _ensure_text(history_entry).strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty after normalization")
                return self._fail_or_raw_archive(messages)

            # Write to all layers
            self.append_history(entry)

            lt_text = _ensure_text(long_term)
            if lt_text != current_memory:
                self.write_long_term(lt_text)

            short_term = args.get("short_term")
            if short_term:
                self.append_short_term(_ensure_text(short_term))

            episode = args.get("episode")
            if episode:
                ep_text = _ensure_text(episode).strip()
                if ep_text:
                    self.append_episode(ep_text)

            # Save behavioral insights to LEARNINGS.md
            insight = args.get("behavioral_insight")
            if insight:
                insight_text = _ensure_text(insight).strip()
                if insight_text and len(insight_text) > 10:
                    learnings_file = self.memory_dir / "LEARNINGS.md"
                    existing = ""
                    if learnings_file.exists():
                        existing = learnings_file.read_text(encoding="utf-8")
                    # Dedup
                    if insight_text.lower() not in existing.lower():
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
                        lines = [l for l in existing.split("\n") if l.strip().startswith("- ")]
                        lines.append(f"- {insight_text} [{ts}]")
                        if len(lines) > 30:
                            lines = lines[-30:]
                        learnings_file.write_text("# Learned Rules\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
                        logger.info("Consolidation: saved behavioral insight — {}", insight_text[:60])

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(messages, self.provider, self.model)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages):
                return True
        return True

    _MSG_COUNT_THRESHOLD = 20  # Force consolidation after this many unconsolidated messages

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within half the context window.

        Also triggers if unconsolidated message count exceeds threshold,
        even if token count is within limits (prevents LONG_TERM.md from
        staying empty on large context windows).
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            unconsolidated_count = len(session.messages) - session.last_consolidated
            target = self.context_window_tokens // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return

            force_by_count = unconsolidated_count >= self._MSG_COUNT_THRESHOLD
            if estimated < self.context_window_tokens and not force_by_count:
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {} ({} unconsolidated msgs)",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                return

            if force_by_count:
                logger.info(
                    "Message-count consolidation triggered for {}: {} unconsolidated msgs",
                    session.key,
                    unconsolidated_count,
                )
                # Force-consolidate the first batch of messages (bypass token check)
                end_idx = session.last_consolidated + self._MSG_COUNT_THRESHOLD
                end_idx = min(end_idx, len(session.messages))
                chunk = session.messages[session.last_consolidated:end_idx]
                if chunk:
                    logger.info("Force consolidating {} messages for {}", len(chunk), session.key)
                    if await self.consolidate_messages(chunk):
                        session.last_consolidated = end_idx
                        self.sessions.save(session)
                        logger.info("Force consolidation done for {}, last_consolidated={}", session.key, end_idx)
                    else:
                        logger.warning("Force consolidation failed for {}", session.key)
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return
