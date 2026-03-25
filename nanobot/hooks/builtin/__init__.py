"""Register all built-in hooks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.hooks.engine import HookEngine, HookMode

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus


def register_builtin_hooks(
    hooks: HookEngine,
    workspace: Path,
    bus: "MessageBus",
) -> None:
    """Wire up all built-in hooks to the engine."""
    from nanobot.hooks.builtin.auto_log import make_auto_log_hook
    from nanobot.hooks.builtin.approval import make_approval_hook
    from nanobot.hooks.builtin.lifecycle import make_tool_tracker, make_turn_tracker

    # Auto-log tool calls to HISTORY.md (fire-and-forget)
    hooks.on("tool_after", make_auto_log_hook(workspace))

    # Approval guard for dangerous tool calls (blocking, highest priority)
    hooks.on("tool_before", make_approval_hook(bus), mode=HookMode.BLOCKING, priority=10)

    # Lifecycle tracking (fire-and-forget)
    hooks.on("tool_after", make_tool_tracker())
    hooks.on("turn_completed", make_turn_tracker(workspace))

    # Pattern observer — detects recurring behavior from tool usage (fire-and-forget)
    from nanobot.hooks.builtin.observer import make_observer_hook
    hooks.on("tool_after", make_observer_hook(workspace))

    # Daily cleanup — archives SHORT_TERM.md on date change (fire-and-forget)
    from nanobot.hooks.builtin.daily_cleanup import make_daily_cleanup_hook
    hooks.on("turn_completed", make_daily_cleanup_hook(workspace))

    # Tool success scoring — tracks reliability and latency (fire-and-forget)
    from nanobot.hooks.builtin.tool_scores import make_tool_scorer_hook
    scorer_callback, scorer_instance = make_tool_scorer_hook(workspace)
    hooks.on("tool_after", scorer_callback)
    # Store scorer on hooks engine so context builder can access insights
    hooks._tool_scorer = scorer_instance  # type: ignore[attr-defined]

    # Self-correction — detects tool failures and learns from repeated errors
    from nanobot.hooks.builtin.self_correct import make_self_correct_hook
    hooks.on("tool_after", make_self_correct_hook(workspace))

    # Generative UI — send structured tool results to frontend (fire-and-forget)
    from nanobot.hooks.builtin.tool_ui import make_tool_ui_hook
    hooks.on("tool_after", make_tool_ui_hook(bus))

    # Calendar cache — when calendar tools return, cache events for proactive meeting alerts
    async def _cache_calendar(event):
        if "CALENDAR" not in (event.name or "").upper():
            return
        try:
            import json as _json
            result = event.result or ""
            # Try to parse calendar events from the tool result
            if "successfull" in result or "events" in result.lower():
                data = _json.loads(result) if result.startswith("{") else {}
                events = []
                # Composio format: data.data.events or data.data.items
                raw = data.get("data", {})
                items = raw.get("events", raw.get("items", []))
                if isinstance(items, list):
                    for item in items:
                        events.append({
                            "title": item.get("summary", item.get("title", "")),
                            "start": item.get("start", {}).get("dateTime", item.get("start", {}).get("date", "")),
                            "end": item.get("end", {}).get("dateTime", ""),
                        })
                if events:
                    from nanobot.hooks.builtin.jarvis import cache_calendar_events
                    cache_calendar_events(workspace, events)
        except Exception:
            pass
    hooks.on("tool_after", _cache_calendar)

