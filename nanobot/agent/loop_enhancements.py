"""Agent loop enhancements — modular improvements to the core loop.

Contains:
1. Tool Result Feedback Loop — structured error classification with recovery hints
2. Conversation Intent Tracking — track active intent across turns
3. Smart Context Window Management — dynamic tool result budgeting
4. Tool Call Validation — pre-execution argument validation
5. Parallel Tool Execution Safety — detect conflicting tools & serialize
6. Response Quality Gate — check if final response addresses user's question
7. (Streaming fix is inline in loop.py)
8. (Pending message fix is inline in loop.py)
9. MCP Reconnection — auto-reconnect dead MCP servers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Settings loader — reads intelligence.json toggles
# ---------------------------------------------------------------------------

_settings_cache: dict[str, Any] = {}
_settings_mtime: float = 0.0


def load_intelligence_settings(workspace: Path) -> dict[str, bool]:
    """Load intelligence toggles from workspace/intelligence.json with caching."""
    global _settings_cache, _settings_mtime
    path = workspace / "intelligence.json"
    defaults = {
        "smartErrorRecovery": True,
        "intentTracking": True,
        "dynamicContextBudget": True,
        "responseQualityGate": True,
        "mcpAutoReconnect": True,
    }
    try:
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime != _settings_mtime:
                _settings_cache = {**defaults, **json.loads(path.read_text())}
                _settings_mtime = mtime
            return _settings_cache
    except Exception:
        pass
    return defaults


# ---------------------------------------------------------------------------
# 1. Tool Result Feedback Loop — classify errors and suggest recovery
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"timeout|timed out", re.I),
     "timeout",
     "The operation timed out. Try a simpler query, reduce scope, or check if the service is reachable."),
    (re.compile(r"401|403|unauthorized|forbidden|authentication", re.I),
     "auth",
     "Authentication/authorization failed. Check credentials or API keys. Use the credentials tool if available."),
    (re.compile(r"404|not found|no such file|does not exist", re.I),
     "not_found",
     "Resource not found. Verify the path/URL/ID is correct. Try listing available resources first."),
    (re.compile(r"429|rate.?limit|too many requests", re.I),
     "rate_limit",
     "Rate limited. Wait a moment before retrying, or try a different approach that requires fewer API calls."),
    (re.compile(r"500|502|503|504|internal.?server|service.?unavailable", re.I),
     "server_error",
     "Remote server error. This is not your fault — try again in a moment or use an alternative tool/service."),
    (re.compile(r"connection.?refused|connection.?reset|network|dns|resolve", re.I),
     "network",
     "Network error. The service may be down or unreachable. Try an alternative approach."),
    (re.compile(r"permission.?denied|access.?denied|read.?only", re.I),
     "permission",
     "Permission denied. Check file/resource permissions or try a different path."),
    (re.compile(r"invalid|malformed|bad.?request|parse.?error|syntax", re.I),
     "validation",
     "Invalid input. Review the parameters — check for typos, wrong format, or missing required fields."),
    (re.compile(r"no.?results|empty|nothing.?found|0 results", re.I),
     "empty",
     "No results found. Try broader search terms, different keywords, or check spelling."),
]


def classify_tool_error(tool_name: str, result: str) -> str:
    """Classify a tool error and append a structured recovery hint.

    Returns the original result with an appended recovery section if an
    error pattern is detected. Returns the result unchanged if no error.
    """
    if not isinstance(result, str):
        return result

    # Only process error results
    is_error = (
        result.startswith("Error")
        or "failed" in result.lower()[:80]
        or "exception" in result.lower()[:80]
    )
    if not is_error:
        return result

    for pattern, category, hint in _ERROR_PATTERNS:
        if pattern.search(result[:500]):
            return (
                f"{result}\n\n"
                f"[Error type: {category}] {hint}\n"
                f"Do NOT retry with the same arguments. Change your approach."
            )

    # Generic error — no specific pattern matched
    return (
        f"{result}\n\n"
        "[Error type: unknown] Analyze the error message carefully. "
        "Try a completely different approach rather than retrying the same call."
    )


# ---------------------------------------------------------------------------
# 2. Conversation Intent Tracking — maintain active intent across turns
# ---------------------------------------------------------------------------

class IntentTracker:
    """Tracks the active conversational intent across turns.

    Extracts a short intent label from user messages and carries it forward
    so the LLM knows the ongoing topic even when pronouns are used.
    """

    def __init__(self) -> None:
        self._current_intent: str = ""
        self._intent_context: str = ""  # Last assistant response snippet
        self._turn_count: int = 0

    def update(self, user_msg: str, assistant_msg: str | None = None) -> None:
        """Update intent from the latest user message and optional assistant response."""
        self._turn_count += 1
        stripped = user_msg.strip()

        # Short pronoun-heavy messages ("send this", "share that") carry forward intent
        if len(stripped) < 50 and self._has_pronoun_reference(stripped):
            # Keep existing intent — the user is referencing previous context
            if assistant_msg:
                self._intent_context = assistant_msg[:200]
            return

        # Longer or self-contained messages set a new intent
        self._current_intent = self._extract_intent(stripped)
        if assistant_msg:
            self._intent_context = assistant_msg[:200]

    def get_intent_block(self) -> str:
        """Get intent context block for injection into the prompt."""
        if not self._current_intent:
            return ""

        lines = [f"[Active intent: {self._current_intent}]"]
        if self._intent_context:
            lines.append(f"[Last response preview: {self._intent_context}]")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset intent tracking (e.g., on /new)."""
        self._current_intent = ""
        self._intent_context = ""
        self._turn_count = 0

    @staticmethod
    def _has_pronoun_reference(text: str) -> bool:
        """Check if text heavily relies on pronouns/references."""
        pronoun_re = re.compile(
            r"\b(this|that|those|these|it|them|the same|send|share|forward)\b", re.I
        )
        matches = pronoun_re.findall(text)
        words = text.split()
        if not words:
            return False
        return len(matches) / len(words) > 0.15

    @staticmethod
    def _extract_intent(text: str) -> str:
        """Extract a short intent label from user text."""
        # Truncate to first sentence or 80 chars
        for sep in (".  ", "? ", "! ", "\n"):
            idx = text.find(sep)
            if 0 < idx < 80:
                text = text[:idx + 1]
                break
        return text[:80].strip()


# ---------------------------------------------------------------------------
# 3. Smart Context Window Management — dynamic tool result budgeting
# ---------------------------------------------------------------------------

def compute_tool_result_budget(
    context_window: int,
    current_prompt_tokens: int,
    tool_count: int,
) -> int:
    """Compute the max chars allowed per tool result based on remaining context.

    Reserves 30% of context for the LLM response. Divides remaining budget
    among pending tool results. Returns chars (not tokens) using ~4 chars/token.
    """
    CHARS_PER_TOKEN = 4
    RESPONSE_RESERVE = 0.30  # Reserve 30% for response
    MIN_BUDGET = 2_000  # Never go below 2K chars per tool
    MAX_BUDGET = 24_000  # Never exceed 24K chars per tool

    available_tokens = int(context_window * (1 - RESPONSE_RESERVE)) - current_prompt_tokens
    if available_tokens <= 0:
        return MIN_BUDGET

    per_tool_tokens = available_tokens // max(tool_count, 1)
    per_tool_chars = per_tool_tokens * CHARS_PER_TOKEN

    return max(MIN_BUDGET, min(per_tool_chars, MAX_BUDGET))


def truncate_tool_result(result: str, budget: int) -> str:
    """Truncate a tool result to fit within the budget."""
    if not isinstance(result, str) or len(result) <= budget:
        return result

    # Try to cut at a natural boundary (newline, sentence)
    cut = result[:budget]
    last_newline = cut.rfind("\n", budget - 500)
    if last_newline > budget // 2:
        cut = cut[:last_newline]

    return cut + f"\n... (truncated from {len(result):,} chars to fit context window)"


# ---------------------------------------------------------------------------
# 4. Tool Call Validation — pre-execution validation
# ---------------------------------------------------------------------------

def validate_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    available_tools: list[str],
) -> str | None:
    """Validate a tool call before execution. Returns error string or None if valid."""
    # Check tool exists
    if tool_name not in available_tools:
        # Find closest match
        close = [t for t in available_tools if tool_name.lower() in t.lower() or t.lower() in tool_name.lower()]
        hint = f" Did you mean: {', '.join(close[:3])}?" if close else ""
        return f"Tool '{tool_name}' does not exist.{hint} Available: {', '.join(sorted(available_tools)[:15])}"

    # Check arguments is a dict
    if not isinstance(arguments, dict):
        return f"Tool arguments must be an object/dict, got {type(arguments).__name__}. Pass named parameters."

    return None  # Valid


# ---------------------------------------------------------------------------
# 5. Parallel Tool Execution Safety — detect conflicting tools
# ---------------------------------------------------------------------------

# Tools that modify shared state and should not run in parallel with each other
_WRITE_TOOLS = frozenset({
    "write_file", "edit_file", "exec", "goals", "memory_save",
    "browser", "credentials", "message", "cron",
})

# Tools that are always safe to run in parallel (read-only)
_SAFE_PARALLEL = frozenset({
    "read_file", "list_dir", "web_search", "web_fetch",
    "memory_search", "inbox", "list_subagents",
})


def partition_tool_calls(tool_calls: list) -> tuple[list, list]:
    """Partition tool calls into parallel-safe and must-serialize groups.

    Returns (parallel_batch, serial_batch). All read-only tools go in parallel.
    Write tools that might conflict go in serial.
    """
    parallel = []
    serial = []

    write_targets: set[str] = set()

    for tc in tool_calls:
        name = tc.name

        if name in _SAFE_PARALLEL:
            parallel.append(tc)
            continue

        if name in _WRITE_TOOLS:
            # Check for file path conflicts
            path = _extract_path(tc.arguments)
            if path and path in write_targets:
                serial.append(tc)  # Conflict — serialize
            else:
                if path:
                    write_targets.add(path)
                parallel.append(tc)  # First write to this path is fine
            continue

        # Unknown tools — assume safe for parallel
        parallel.append(tc)

    return parallel, serial


def _extract_path(args: dict | Any) -> str | None:
    """Extract file path from tool arguments."""
    if not isinstance(args, dict):
        return None
    for key in ("path", "file_path", "filename", "file"):
        if key in args and isinstance(args[key], str):
            return args[key]
    return None


# ---------------------------------------------------------------------------
# 6. Response Quality Gate — check if response addresses the question
# ---------------------------------------------------------------------------

_NON_ANSWER_PATTERNS = re.compile(
    r"^(I('m| am) (sorry|unable|not able)|"
    r"I (cannot|can't|don't have|do not have)|"
    r"Unfortunately|"
    r"I apologize|"
    r"I don't (know|understand)|"
    r"Sorry, I)",
    re.I,
)

_GENERIC_FILLER = re.compile(
    r"^(Sure|Of course|Absolutely|Happy to help|I'd be happy)[!.,]?\s*$",
    re.I | re.MULTILINE,
)


def check_response_quality(
    user_message: str,
    response: str | None,
    tools_used: list[str],
) -> tuple[bool, str]:
    """Check if the response adequately addresses the user's message.

    Returns (is_ok, issue). If is_ok is False, `issue` describes the problem
    and can be used to request a retry.
    """
    if not response:
        if tools_used:
            return True, ""  # Tools were used — response may be intentionally empty
        return False, "Empty response with no tool usage."

    response_stripped = response.strip()

    # Very short responses to non-trivial questions
    if len(response_stripped) < 20 and len(user_message.strip()) > 30:
        if not tools_used:
            return False, "Response too brief for the question asked."

    # Non-answer detection — the LLM deflected instead of trying
    if _NON_ANSWER_PATTERNS.match(response_stripped) and not tools_used:
        return False, "Deflection detected — the model refused without attempting tool use."

    return True, ""


# ---------------------------------------------------------------------------
# 9. MCP Reconnection Helper
# ---------------------------------------------------------------------------

class MCPReconnector:
    """Tracks MCP connection health and triggers reconnection when needed."""

    def __init__(self, max_failures: int = 3, cooldown_seconds: float = 30.0):
        self._failure_count: int = 0
        self._max_failures = max_failures
        self._cooldown = cooldown_seconds
        self._last_reconnect: float = 0.0

    def record_failure(self, tool_name: str, error: str) -> bool:
        """Record a tool failure. Returns True if reconnection should be attempted."""
        # Only count MCP-related failures
        if not tool_name.startswith("mcp_"):
            return False

        mcp_errors = ("connection", "transport", "closed", "reset", "refused", "eof", "broken pipe")
        if not any(e in error.lower() for e in mcp_errors):
            return False

        self._failure_count += 1
        logger.warning("MCP failure #{} for {}: {}", self._failure_count, tool_name, error[:100])

        if self._failure_count >= self._max_failures:
            import time
            now = time.monotonic()
            if now - self._last_reconnect > self._cooldown:
                self._last_reconnect = now
                self._failure_count = 0
                return True  # Trigger reconnection

        return False

    def record_success(self, tool_name: str) -> None:
        """Record a successful tool call — resets failure counter."""
        if tool_name.startswith("mcp_"):
            self._failure_count = 0
