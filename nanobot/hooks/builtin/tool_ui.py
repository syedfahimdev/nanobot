"""Generative UI hook — sends structured tool results to the frontend.

When a ToolsDNS tool call completes, this hook parses the JSON result
and broadcasts it to all connected web voice clients as a `tool_result`
message. The frontend maps tool names to React components for rich
inline rendering (email cards, calendar events, weather widgets, etc.).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.hooks.events import ToolAfter

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus

# Tools whose results are worth rendering as UI components
_UI_TOOLS = frozenset({
    "GMAIL_FETCH_EMAILS", "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
    "GOOGLECALENDAR_FIND_EVENT", "GOOGLECALENDAR_EVENTS_LIST",
    "GOOGLECALENDAR_EVENTS_LIST_ALL_CALENDARS",
    "WEATHERMAP_WEATHER", "WEATHERMAP_GEOCODE_LOCATION",
    "HACKERNEWS_SEARCH_POSTS",
    "REDDIT_SEARCH_ACROSS_SUBREDDITS",
})


def _parse_tool_result(result: str) -> dict | None:
    """Try to extract structured data from a tool result string."""
    try:
        parsed = json.loads(result)
        # Handle ToolsDNS wrapper format
        if isinstance(parsed, dict):
            # Composio tools wrap in content[].text
            content = parsed.get("content", [])
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                if text:
                    inner = json.loads(text)
                    return inner.get("data", inner)
            # Direct result
            return parsed.get("data", parsed.get("result", parsed))
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return None


def make_tool_ui_hook(bus: "MessageBus"):
    """Create a hook that sends structured tool results for generative UI."""

    async def tool_ui(event: ToolAfter) -> None:
        if event.error or not event.channel:
            return

        # Only process toolsdns calls
        tool_name = ""
        if event.name == "toolsdns" and event.params.get("action") == "call":
            tool_id = event.params.get("tool_id", "")
            tool_name = tool_id.replace("tooldns__", "")
        else:
            return

        # Check if this tool has a UI component
        if tool_name not in _UI_TOOLS:
            logger.debug("Tool UI: {} not in UI tools list", tool_name)
            return

        # Parse the result
        data = _parse_tool_result(event.result)
        if not data:
            logger.debug("Tool UI: failed to parse result for {} (len={})", tool_name, len(event.result))
            return

        # Send to the channel via bus
        from nanobot.bus.events import OutboundMessage
        await bus.publish_outbound(OutboundMessage(
            channel=event.channel,
            chat_id=event.chat_id or "voice",
            content=f"Tool result: {tool_name}",
            metadata={
                "_tool_result": True,
                "_tool_name": tool_name,
                "_tool_data": data,
            },
        ))
        logger.debug("Tool UI: sent {} result for generative UI", tool_name)

    return tool_ui
