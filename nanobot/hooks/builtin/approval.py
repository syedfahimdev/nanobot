"""Code-level approval guard for dangerous tool calls.

LLM cannot bypass this — it runs before the tool executes.
On voice channels, the approval question is spoken via TTS and
the user's voice response (yes/no) is detected.
"""

from __future__ import annotations

import asyncio
import re
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.hooks.events import ToolBefore

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus

# Tool patterns that require user confirmation before execution.
CONFIRM_PATTERNS = [
    "GMAIL_SEND_*",
    "GMAIL_CREATE_*",
    "GMAIL_REPLY_*",
    "*_DELETE_*",
    "*_REMOVE_*",
    "DISCORDBOT_CREATE_MESSAGE",
    "DISCORDBOT_DELETE_*",
]

# Pending approval futures keyed by session_key
_pending: dict[str, asyncio.Future[bool]] = {}

# Regex for yes/no detection in voice responses
_YES_RE = re.compile(r"^(yes|yeah|yep|yup|sure|go ahead|do it|confirm|approved|send it)\b", re.I)
_NO_RE = re.compile(r"^(no|nope|nah|don'?t|cancel|stop|deny|denied|abort)\b", re.I)


def _get_tool_name(event: ToolBefore) -> str:
    """Extract the actual tool name for pattern matching."""
    return event.name


def _needs_approval(tool_name: str) -> bool:
    for pattern in CONFIRM_PATTERNS:
        if fnmatch(tool_name, pattern):
            return True
    return False


def _describe_action(tool_name: str, params: dict[str, Any]) -> str:
    """Build a short human-readable description of the action."""
    args = params.get("arguments", params)
    if isinstance(args, str):
        return f"{tool_name}"

    # Try to extract key info for common tools
    if "GMAIL_SEND" in tool_name or "GMAIL_REPLY" in tool_name:
        to = args.get("recipient_email", args.get("to", "someone"))
        subj = args.get("subject", "")
        if subj:
            return f"send email to {to} about {subj[:40]}"
        return f"send email to {to}"
    if "DELETE" in tool_name or "REMOVE" in tool_name:
        return f"delete via {tool_name}"
    if "CREATE_MESSAGE" in tool_name:
        channel = args.get("channel_id", "a channel")
        return f"post message to Discord channel {channel}"

    return tool_name.lower().replace("_", " ")


def make_approval_hook(bus: "MessageBus"):
    """Create an approval hook bound to the message bus."""

    async def approval_guard(event: ToolBefore) -> ToolBefore:
        tool_name = _get_tool_name(event)
        if not _needs_approval(tool_name):
            return event

        if not event.channel or not event.chat_id:
            return event  # Can't ask for approval without a channel

        description = _describe_action(tool_name, event.params)
        logger.info("Approval required for {} on session {}", tool_name, event.session_key)

        # Ask the user via the channel (TTS on voice)
        from nanobot.bus.events import OutboundMessage
        is_voice = event.channel in ("discord_voice", "web_voice")
        if is_voice:
            question = f"Should I {description}? Say yes or no."
        else:
            question = f"Should I **{description}**? Reply `yes` or `no`."

        await bus.publish_outbound(OutboundMessage(
            channel=event.channel,
            chat_id=event.chat_id,
            content=question,
            metadata={"_tts_sentence": True, "_approval_request": True},
        ))

        # Wait for user response
        future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        session_key = event.session_key or f"{event.channel}:{event.chat_id}"
        _pending[session_key] = future

        try:
            approved = await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError:
            approved = False
            logger.info("Approval timed out for {}", tool_name)
        finally:
            _pending.pop(session_key, None)

        if not approved:
            event.denied = True
            event.deny_reason = f"Action not approved: {description}."
            # Tell the user
            await bus.publish_outbound(OutboundMessage(
                channel=event.channel,
                chat_id=event.chat_id,
                content="Got it, cancelled." if is_voice else "Action cancelled.",
                metadata={"_tts_sentence": True},
            ))
        else:
            await bus.publish_outbound(OutboundMessage(
                channel=event.channel,
                chat_id=event.chat_id,
                content="Alright, doing it now." if is_voice else "Approved.",
                metadata={"_tts_sentence": True},
            ))

        return event

    return approval_guard


def resolve_approval(session_key: str, text: str) -> bool:
    """Check if text is an approval response and resolve the pending future.

    Called from the message dispatch path to intercept yes/no answers.
    Returns True if the message was consumed as an approval response.
    """
    future = _pending.get(session_key)
    if future is None or future.done():
        return False

    stripped = text.strip()
    if _YES_RE.match(stripped):
        future.set_result(True)
        return True
    if _NO_RE.match(stripped):
        future.set_result(False)
        return True

    return False


def has_pending_approval(session_key: str) -> bool:
    """Check if there's a pending approval for this session."""
    future = _pending.get(session_key)
    return future is not None and not future.done()
