"""Session lifecycle tracking — track turn states for debugging and voice UI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.hooks.events import ToolAfter, TurnCompleted

# In-memory state for active turns
_active: dict[str, dict[str, Any]] = {}


def get_active_tasks() -> list[dict[str, Any]]:
    """Get currently active task states (for UI activity feed)."""
    return [v for v in _active.values() if v.get("status") == "running"]


def make_tool_tracker():
    """Track tool calls within a turn for the lifecycle log."""

    async def track_tool(event: ToolAfter) -> None:
        key = event.session_key
        if not key or key not in _active:
            return
        entry = _active[key]
        entry.setdefault("tools", []).append({
            "name": event.name,
            "duration_ms": round(event.duration_ms),
            "error": event.error,
        })

    return track_tool


def make_turn_tracker(workspace: Path):
    """Track turn completion for the lifecycle log."""
    activity_path = workspace / "activity.jsonl"

    async def track_turn(event: TurnCompleted) -> None:
        info = _active.pop(event.session_key, {})
        info.update({
            "status": "done",
            "session_key": event.session_key,
            "channel": event.channel,
            "tools_used": event.tools_used,
            "iterations": event.iterations,
            "duration_ms": round(event.duration_ms),
            "completed_at": time.time(),
        })

        try:
            with open(activity_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(info) + "\n")
        except Exception as e:
            logger.debug("Activity log write failed: {}", e)

    return track_turn


def mark_turn_started(session_key: str, channel: str | None = None) -> None:
    """Called from the agent loop to mark a turn as started."""
    _active[session_key] = {
        "status": "running",
        "session_key": session_key,
        "channel": channel,
        "started_at": time.time(),
        "tools": [],
    }
