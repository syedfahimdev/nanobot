"""Briefing delivery hook — checks if a briefing is due on each turn.

Runs as fire-and-forget on turn_completed. If a briefing is due,
generates and publishes it to the user's channel via the message bus.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.hooks.events import TurnCompleted

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import LLMProvider


def make_briefing_hook(
    workspace: Path,
    provider: "LLMProvider",
    model: str,
    bus: "MessageBus",
):
    """Create a briefing hook that checks for due briefings on each turn."""
    from nanobot.hooks.builtin.briefing import BriefingEngine

    engine = BriefingEngine(workspace, provider, model)

    async def on_turn_completed(event: TurnCompleted) -> None:
        result = await engine.maybe_generate()
        if result is None:
            return

        briefing, priority = result
        channel = event.channel or "cli"
        chat_id = event.chat_id or "direct"

        # Deliver briefing via message bus
        from nanobot.bus.events import OutboundMessage
        await bus.publish_outbound(OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=briefing,
            metadata={"_briefing": True, "_priority": priority},
        ))

    return on_turn_completed
