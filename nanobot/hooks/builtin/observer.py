"""Pattern observer — detects recurring behavior from tool usage.

Tracks tool calls and writes detected patterns to OBSERVATIONS.md
after 3+ occurrences. No LLM calls — pure statistics.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.hooks.events import ToolAfter

# Tools too noisy to observe (internal file ops)
_SKIP = frozenset({"read_file", "write_file", "edit_file", "list_dir", "exec"})

_MIN_OCCURRENCES = 3  # Minimum count before writing a pattern

_TIME_BUCKETS = {
    "early morning": range(5, 8),
    "morning": range(8, 12),
    "afternoon": range(12, 17),
    "evening": range(17, 21),
    "night": range(21, 24),
    "late night": range(0, 5),
}

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _get_time_bucket(hour: int) -> str:
    for name, hours in _TIME_BUCKETS.items():
        if hour in hours:
            return name
    return "unknown"


def _normalize_tool_name(event: ToolAfter) -> str:
    """Get a human-readable tool name."""
    return event.name


class PatternObserver:
    """Accumulates tool usage statistics and writes patterns to OBSERVATIONS.md."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._obs_file = workspace / "memory" / "OBSERVATIONS.md"

        # Counters
        self._tool_counts: dict[str, int] = defaultdict(int)
        self._time_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._day_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._sequences: dict[str, int] = defaultdict(int)
        self._last_tool: str | None = None
        self._total_calls = 0

    def _maybe_write(self) -> None:
        """Write observations if we have enough data."""
        if self._total_calls < _MIN_OCCURRENCES * 2:
            return  # Wait for enough data

        lines = []

        # Top tools by frequency
        sorted_tools = sorted(self._tool_counts.items(), key=lambda x: -x[1])
        if len(sorted_tools) >= 2:
            top = sorted_tools[:5]
            tools_line = ", ".join(f"{name} ({count}x)" for name, count in top)
            lines.append(f"- Most used tools: {tools_line}")

        # Time-of-day habits
        for tool, buckets in self._time_buckets.items():
            top_bucket = max(buckets, key=buckets.get)  # type: ignore[arg-type]
            count = buckets[top_bucket]
            if count >= _MIN_OCCURRENCES:
                lines.append(f"- Uses {tool} mostly in the {top_bucket} ({count}x)")

        # Day-of-week patterns
        for tool, days in self._day_buckets.items():
            top_day = max(days, key=days.get)  # type: ignore[arg-type]
            count = days[top_day]
            if count >= _MIN_OCCURRENCES:
                lines.append(f"- Uses {tool} often on {top_day}s ({count}x)")

        # Sequential patterns (A → B)
        for pair, count in sorted(self._sequences.items(), key=lambda x: -x[1]):
            if count >= _MIN_OCCURRENCES:
                lines.append(f"- Often does {pair} ({count}x)")

        if not lines:
            return

        content = "# Observed Patterns\n\n" + "\n".join(lines[:10]) + "\n"

        try:
            self._obs_file.parent.mkdir(parents=True, exist_ok=True)
            self._obs_file.write_text(content, encoding="utf-8")
            logger.debug("PatternObserver: wrote {} observations", len(lines))
        except Exception as e:
            logger.debug("PatternObserver write failed: {}", e)

    async def on_tool_after(self, event: ToolAfter) -> None:
        """Track a tool call and update patterns."""
        if event.name in _SKIP or event.error:
            return

        tool = _normalize_tool_name(event)
        now = datetime.now()
        self._total_calls += 1

        # Frequency
        self._tool_counts[tool] += 1

        # Time-of-day
        bucket = _get_time_bucket(now.hour)
        self._time_buckets[tool][bucket] += 1

        # Day-of-week
        day = _WEEKDAYS[now.weekday()]
        self._day_buckets[tool][day] += 1

        # Sequential patterns
        if self._last_tool and self._last_tool != tool:
            pair = f"{self._last_tool} → {tool}"
            self._sequences[pair] += 1
        self._last_tool = tool

        # Write every 10 calls to avoid excessive I/O
        if self._total_calls % 10 == 0:
            self._maybe_write()


def make_observer_hook(workspace: Path):
    """Create a pattern observer hook bound to the workspace."""
    observer = PatternObserver(workspace)
    return observer.on_tool_after
