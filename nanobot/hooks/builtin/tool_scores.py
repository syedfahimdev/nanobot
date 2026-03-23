"""Tool success scoring — tracks success rates, latency, and error patterns.

Maintains tool_scores.json with per-tool statistics. Surfaces insights
in the system prompt when tool reliability drops below threshold.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.hooks.events import ToolAfter

# Tools too noisy to score (internal file ops)
_SKIP = frozenset({"read_file", "write_file", "edit_file", "list_dir", "exec"})

_RELIABILITY_WARNING_THRESHOLD = 0.7  # Warn when success rate drops below 70%
_MIN_CALLS_FOR_INSIGHT = 5  # Need at least N calls to generate insights
_MAX_ERROR_SAMPLES = 5  # Keep last N error messages per tool
_FLUSH_INTERVAL = 10  # Write to disk every N events


def _normalize_tool_name(event: ToolAfter) -> str:
    """Get a human-readable tool name."""
    return event.name


class ToolScorer:
    """Tracks tool success/failure rates and generates reliability insights."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._scores_file = workspace / "memory" / "tool_scores.json"
        self._scores: dict[str, dict[str, Any]] = {}
        self._event_count = 0
        self._load()

    def _load(self) -> None:
        """Load existing scores from disk."""
        try:
            if self._scores_file.exists():
                raw = self._scores_file.read_text(encoding="utf-8")
                if isinstance(raw, str) and raw.strip():
                    self._scores = json.loads(raw)
        except (json.JSONDecodeError, OSError, TypeError):
            self._scores = {}

    def _save(self) -> None:
        """Persist scores to disk."""
        try:
            self._scores_file.parent.mkdir(parents=True, exist_ok=True)
            self._scores_file.write_text(
                json.dumps(self._scores, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug("ToolScorer save failed: {}", e)

    async def on_tool_after(self, event: ToolAfter) -> None:
        """Record a tool call result."""
        if event.name in _SKIP:
            return

        tool = _normalize_tool_name(event)
        self._event_count += 1

        if tool not in self._scores:
            self._scores[tool] = {
                "success": 0,
                "fail": 0,
                "total_duration_ms": 0,
                "recent_errors": [],
                "last_used": "",
            }

        entry = self._scores[tool]
        if event.error:
            entry["fail"] += 1
            # Store error sample (truncated)
            error_preview = (event.result or "")[:200].strip()
            if error_preview:
                errors = entry["recent_errors"]
                errors.append(error_preview)
                if len(errors) > _MAX_ERROR_SAMPLES:
                    entry["recent_errors"] = errors[-_MAX_ERROR_SAMPLES:]
        else:
            entry["success"] += 1

        entry["total_duration_ms"] += event.duration_ms
        entry["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Flush every event — CLI sessions are short-lived
        self._save()

    def get_insights(self) -> str:
        """Generate insights for the system prompt.

        Only returns warnings for tools with low reliability.
        Returns empty string if everything is healthy.
        """
        warnings: list[str] = []

        for tool, data in self._scores.items():
            total = data.get("success", 0) + data.get("fail", 0)
            if total < _MIN_CALLS_FOR_INSIGHT:
                continue

            success_rate = data["success"] / total
            avg_ms = data.get("total_duration_ms", 0) / total

            if success_rate < _RELIABILITY_WARNING_THRESHOLD:
                pct = int(success_rate * 100)
                warning = f"- {tool}: {pct}% success rate ({data['fail']} failures out of {total} calls)"

                # Add common error hint
                errors = data.get("recent_errors", [])
                if errors:
                    # Find most common error substring
                    common = _find_common_error_pattern(errors)
                    if common:
                        warning += f" — common error: {common}"

                warnings.append(warning)

            elif avg_ms > 10000:
                # Warn about slow tools (>10s average)
                warnings.append(f"- {tool}: averaging {avg_ms / 1000:.1f}s per call — consider timeout handling")

        if not warnings:
            return ""

        return "## Tool Reliability\n" + "\n".join(warnings)

    def get_full_report(self) -> dict[str, Any]:
        """Return full scores for the settings API."""
        report = {}
        for tool, data in self._scores.items():
            total = data.get("success", 0) + data.get("fail", 0)
            if total == 0:
                continue
            report[tool] = {
                "total": total,
                "success": data["success"],
                "fail": data["fail"],
                "successRate": round(data["success"] / total * 100, 1),
                "avgDurationMs": round(data.get("total_duration_ms", 0) / total),
                "lastUsed": data.get("last_used", ""),
            }
        return report


def _find_common_error_pattern(errors: list[str]) -> str:
    """Find the most common meaningful substring across error messages."""
    if not errors:
        return ""
    if len(errors) == 1:
        return errors[0][:80]

    # Find common words across errors
    word_counts: dict[str, int] = defaultdict(int)
    for err in errors:
        words = set(err.lower().split())
        for w in words:
            if len(w) > 3:  # Skip short words
                word_counts[w] += 1

    # Words appearing in majority of errors
    threshold = len(errors) * 0.5
    common_words = [w for w, c in word_counts.items() if c >= threshold]
    if common_words:
        return " ".join(sorted(common_words)[:5])

    return errors[-1][:80]


def make_tool_scorer_hook(workspace: Path) -> tuple[Any, ToolScorer]:
    """Create a tool scorer hook and return both the callback and the scorer instance."""
    scorer = ToolScorer(workspace)
    return scorer.on_tool_after, scorer
