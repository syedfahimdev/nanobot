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
    toolsdns_url: str = "",
    toolsdns_api_key: str = "",
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

    # Generative UI — send structured tool results to frontend (fire-and-forget)
    from nanobot.hooks.builtin.tool_ui import make_tool_ui_hook
    hooks.on("tool_after", make_tool_ui_hook(bus))

    # Memory re-index on file writes (fire-and-forget)
    if toolsdns_url and toolsdns_api_key:
        from nanobot.hooks.builtin.memory_reindex import make_memory_reindex_hook
        hooks.on("tool_after", make_memory_reindex_hook(workspace, toolsdns_url, toolsdns_api_key))
