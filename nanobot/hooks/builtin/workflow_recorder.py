"""Workflow Recorder — detects repeated multi-step patterns and offers automation.

Watches tool call sequences across sessions. When the same sequence appears
3+ times, suggests creating a macro or workflow.

Example detection:
  Session 1: GMAIL_FETCH_EMAILS → read specific email → draft reply → send
  Session 2: GMAIL_FETCH_EMAILS → read specific email → draft reply → send
  Session 3: GMAIL_FETCH_EMAILS → ...
  → "I notice you check and reply to emails every morning. Want me to automate this?"

Stores patterns in memory/WORKFLOWS.md for transparency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.hooks.events import TurnCompleted

_MIN_SEQUENCE_LENGTH = 2  # Minimum tool calls to form a pattern
_MIN_OCCURRENCES = 3  # Times a pattern must repeat before suggesting
_MAX_PATTERNS = 20  # Keep top N patterns


class WorkflowRecorder:
    """Records tool sequences and detects repeating workflows."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._patterns_file = workspace / "memory" / "WORKFLOWS.md"
        self._state_file = workspace / "memory" / ".workflow_state.json"
        self._current_turn_tools: list[str] = []
        self._session_sequences: dict[str, list[list[str]]] = defaultdict(list)
        self._pattern_counts: dict[str, int] = defaultdict(int)
        self._suggested: set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        try:
            if self._state_file.exists():
                raw = self._state_file.read_text(encoding="utf-8")
                if isinstance(raw, str) and raw.strip():
                    data = json.loads(raw)
                    self._pattern_counts = defaultdict(int, data.get("patterns", {}))
                    self._suggested = set(data.get("suggested", []))
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "patterns": dict(self._pattern_counts),
                "suggested": list(self._suggested),
            }
            self._state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def record_tool(self, tool_name: str) -> None:
        """Record a tool call in the current turn."""
        # Skip noisy internal tools
        if tool_name in ("read_file", "write_file", "edit_file", "list_dir"):
            return
        self._current_turn_tools.append(tool_name)

    def on_turn_completed(self, session_key: str) -> str | None:
        """Called when a turn completes. Returns suggestion if pattern detected."""
        tools = self._current_turn_tools
        self._current_turn_tools = []

        if len(tools) < _MIN_SEQUENCE_LENGTH:
            return None

        # Store this sequence
        self._session_sequences[session_key].append(tools)

        # Create a pattern key from the tool sequence
        pattern_key = " → ".join(tools)
        self._pattern_counts[pattern_key] += 1
        self._save_state()

        # Check if this pattern should be suggested
        count = self._pattern_counts[pattern_key]
        if count >= _MIN_OCCURRENCES and pattern_key not in self._suggested:
            self._suggested.add(pattern_key)
            self._save_state()
            self._write_pattern(pattern_key, count, tools)
            logger.info("Workflow detected: {} ({} times)", pattern_key, count)
            return (
                f"I notice you frequently use this tool sequence ({count} times): "
                f"{pattern_key}. "
                f"Want me to create an automated workflow for this?"
            )

        return None

    def _write_pattern(self, pattern: str, count: int, tools: list[str]) -> None:
        """Write detected pattern to WORKFLOWS.md."""
        self._patterns_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{ts}] Detected Pattern ({count}x)\n- Sequence: {pattern}\n- Tools: {', '.join(tools)}\n- Status: Suggested\n"
        with open(self._patterns_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def get_top_patterns(self, n: int = 5) -> list[dict[str, Any]]:
        """Return top N patterns for the settings page."""
        sorted_patterns = sorted(self._pattern_counts.items(), key=lambda x: -x[1])
        return [
            {"pattern": k, "count": v, "suggested": k in self._suggested}
            for k, v in sorted_patterns[:n]
        ]


# Global recorder instance
_recorder: WorkflowRecorder | None = None


def get_recorder(workspace: Path) -> WorkflowRecorder:
    global _recorder
    if _recorder is None:
        _recorder = WorkflowRecorder(workspace)
    return _recorder


def make_workflow_recorder_tool_hook(workspace: Path):
    """Hook for tool_after: records tool calls."""
    recorder = get_recorder(workspace)

    async def on_tool_after(event) -> None:
        tool_name = event.name
        if event.name == "toolsdns" and event.params.get("action") == "call":
            tool_name = event.params.get("tool_id", "toolsdns").replace("tooldns__", "")
        recorder.record_tool(tool_name)

    return on_tool_after


def make_workflow_recorder_turn_hook(workspace: Path, bus=None):
    """Hook for turn_completed: checks for patterns and suggests automation."""
    recorder = get_recorder(workspace)

    async def on_turn_completed(event) -> None:
        suggestion = recorder.on_turn_completed(event.session_key)
        if suggestion and bus:
            from nanobot.bus.events import OutboundMessage
            channel = event.channel or "cli"
            chat_id = event.chat_id or "direct"
            await bus.publish_outbound(OutboundMessage(
                channel=channel, chat_id=chat_id,
                content=suggestion,
                metadata={"_workflow_suggestion": True},
            ))

    return on_turn_completed
