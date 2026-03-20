"""Self-correction hook — verifies tool call results and learns from failures.

Runs after each tool call to detect failures that the agent might miss.
Tracks error patterns and injects correction hints into the next prompt.
Extends the reflection system to learn from tool failures, not just user corrections.

Patterns detected:
  - Silent failures (tool returns "ok" but result is empty/useless)
  - Repeated errors on same tool (same error 3x = suggest alternative)
  - Timeout patterns (slow tools that might need retry)
  - Partial results (truncated output, missing fields)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.hooks.events import ToolAfter

# Tools exempt from self-correction (internal ops)
_SKIP = frozenset({"read_file", "write_file", "edit_file", "list_dir"})

# Error patterns to detect in tool results
_ERROR_PATTERNS = [
    (re.compile(r"error|failed|exception|traceback|timeout", re.I), "error_keyword"),
    (re.compile(r"401|403|unauthorized|forbidden", re.I), "auth_failure"),
    (re.compile(r"404|not found|does not exist", re.I), "not_found"),
    (re.compile(r"429|rate.?limit|too many requests", re.I), "rate_limit"),
    (re.compile(r"500|502|503|504|internal server|service unavailable", re.I), "server_error"),
    (re.compile(r"timeout|timed out|deadline exceeded", re.I), "timeout"),
    (re.compile(r"empty|no results|nothing found|0 items", re.I), "empty_result"),
]

_MAX_CORRECTIONS_FILE = 15  # Max entries in CORRECTIONS.md
_REPEATED_ERROR_THRESHOLD = 3  # Suggest alternative after N same errors


class SelfCorrector:
    """Detects tool failures and generates correction hints."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._corrections_file = workspace / "memory" / "CORRECTIONS.md"
        # Track errors per tool per session
        self._error_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Track repeated errors
        self._error_counts: dict[str, int] = defaultdict(int)

    def _classify_result(self, event: ToolAfter) -> list[str]:
        """Classify a tool result into error categories."""
        if event.error:
            return ["explicit_error"]

        categories = []
        result = event.result or ""

        for pattern, category in _ERROR_PATTERNS:
            if pattern.search(result):
                categories.append(category)

        # Detect silent failures — tool "succeeded" but result is suspiciously short
        if not event.error and len(result.strip()) < 5 and event.name not in ("exec", "message"):
            categories.append("silent_failure")

        # Detect slow tools
        if event.duration_ms > 15000:
            categories.append("slow_tool")

        return categories

    def _should_correct(self, categories: list[str]) -> bool:
        """Decide if this result warrants a correction entry."""
        # Skip if just empty result or slow (low confidence)
        high_confidence = {"explicit_error", "auth_failure", "rate_limit", "server_error", "timeout"}
        return bool(set(categories) & high_confidence)

    def _format_correction(self, event: ToolAfter, categories: list[str]) -> str:
        """Format a correction entry for CORRECTIONS.md."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        tool_name = event.name
        if event.name == "toolsdns" and event.params.get("action") == "call":
            tool_name = event.params.get("tool_id", "toolsdns").replace("tooldns__", "")

        cats = ", ".join(categories)
        result_preview = (event.result or "")[:150].replace("\n", " ").strip()
        return f"- [{ts}] {tool_name}: {cats} — {result_preview}"

    def _check_repeated_errors(self, tool_name: str, categories: list[str]) -> str | None:
        """Check if a tool has repeated the same error and suggest alternatives."""
        key = f"{tool_name}:{','.join(sorted(categories))}"
        self._error_counts[key] += 1

        if self._error_counts[key] >= _REPEATED_ERROR_THRESHOLD:
            self._error_counts[key] = 0  # Reset counter
            if "auth_failure" in categories:
                return f"{tool_name} has failed auth {_REPEATED_ERROR_THRESHOLD}+ times — check API credentials"
            if "rate_limit" in categories:
                return f"{tool_name} is rate-limited — wait before retrying or use a different approach"
            if "timeout" in categories:
                return f"{tool_name} keeps timing out — try with simpler parameters or a smaller scope"
            return f"{tool_name} keeps failing ({','.join(categories)}) — consider an alternative approach"

        return None

    def _save_correction(self, entry: str) -> None:
        """Append a correction to CORRECTIONS.md."""
        self._corrections_file.parent.mkdir(parents=True, exist_ok=True)

        existing: list[str] = []
        if self._corrections_file.exists():
            content = self._corrections_file.read_text(encoding="utf-8")
            existing = [l for l in content.split("\n") if l.strip().startswith("- ")]

        existing.append(entry)
        if len(existing) > _MAX_CORRECTIONS_FILE:
            existing = existing[-_MAX_CORRECTIONS_FILE:]

        content = "# Tool Corrections\n\n" + "\n".join(existing) + "\n"
        self._corrections_file.write_text(content, encoding="utf-8")

    async def on_tool_after(self, event: ToolAfter) -> None:
        """Analyze tool result and record corrections if needed."""
        if event.name in _SKIP:
            return

        categories = self._classify_result(event)
        if not categories:
            return

        tool_name = event.name
        if event.name == "toolsdns" and event.params.get("action") == "call":
            tool_name = event.params.get("tool_id", "toolsdns").replace("tooldns__", "")

        # Track error history
        self._error_history[tool_name].append({
            "categories": categories,
            "timestamp": datetime.now().isoformat(),
            "result_preview": (event.result or "")[:100],
        })

        # Only persist high-confidence corrections
        if self._should_correct(categories):
            entry = self._format_correction(event, categories)
            self._save_correction(entry)
            logger.debug("Self-correction: recorded {} for {}", categories, tool_name)

        # Check for repeated errors
        hint = self._check_repeated_errors(tool_name, categories)
        if hint:
            # Write hint to LEARNINGS.md so it gets injected into context
            learnings_file = self._workspace / "memory" / "LEARNINGS.md"
            learnings_file.parent.mkdir(parents=True, exist_ok=True)

            existing = ""
            if learnings_file.exists():
                existing = learnings_file.read_text(encoding="utf-8")

            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"- {hint} [{ts}]"

            if entry not in existing:
                lines = [l for l in existing.split("\n") if l.strip().startswith("- ")]
                lines.append(entry)
                if len(lines) > 20:
                    lines = lines[-20:]
                content = "# Learned Rules\n\n" + "\n".join(lines) + "\n"
                learnings_file.write_text(content, encoding="utf-8")
                logger.info("Self-correction: promoted repeated error to learning — {}", hint)

    def get_recent_corrections(self) -> str:
        """Return recent corrections for context injection."""
        if not self._corrections_file.exists():
            return ""
        content = self._corrections_file.read_text(encoding="utf-8")
        lines = [l for l in content.split("\n") if l.strip().startswith("- ")]
        if not lines:
            return ""
        return "\n".join(lines[-5:])


def make_self_correct_hook(workspace: Path):
    """Create a self-correction hook bound to the workspace."""
    corrector = SelfCorrector(workspace)
    return corrector.on_tool_after
