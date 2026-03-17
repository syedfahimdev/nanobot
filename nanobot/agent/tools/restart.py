"""Restart tool — safely restarts the nanobot process via os.execv."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

RESTART_CONTEXT_PATH = Path("/tmp/nanobot_restart_context.json")


class RestartTool(Tool):
    """Restart the nanobot process in-place, preserving the restart reason for a post-boot message."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str, *_args: Any) -> None:
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "restart"

    @property
    def description(self) -> str:
        return (
            "Restart the nanobot process. "
            "Always use this tool when asked to restart — never use pkill or exec-based workarounds. "
            "Provide a human-readable reason; it will be sent as a confirmation message after the bot comes back online."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the bot is restarting (shown to user after restart)",
                }
            },
            "required": ["reason"],
        }

    async def execute(self, reason: str = "manual restart", **kwargs: Any) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        # Persist restart context so the new process can announce itself
        RESTART_CONTEXT_PATH.write_text(
            json.dumps({"reason": reason, "channel": channel, "chat_id": chat_id})
        )

        # Acknowledge before dying
        if self._send_callback and channel and chat_id:
            await self._send_callback(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=f"Restarting... (reason: {reason})",
                )
            )

        async def _do_restart() -> None:
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())
        return "Restart initiated."
