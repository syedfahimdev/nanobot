"""Phone call tool — outbound calls with conversational AI via Twilio + Deepgram.

Two modes:
1. Simple TTS call — Twilio speaks a static message (no Deepgram needed)
2. Conversational call — Twilio streams audio ↔ Deepgram STT/TTS ↔ Mawa agent

Provider, voice, and mode configurable via mawa_settings.json.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool


def _get_cred(name: str) -> str | None:
    """Get credential from env or vault."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        from nanobot.setup.vault import load_vault
        vault = load_vault()
        # Try multiple key formats
        for key in [f"cred.{name.lower()}", f"cred.{name}", name.lower()]:
            if vault.get(key):
                return vault[key]
    except Exception:
        pass
    return None


class PhoneCallTool(Tool):
    """Make outbound phone calls — simple TTS or conversational AI."""

    def __init__(self, workspace: Path | None = None):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "phone_call"

    @property
    def description(self) -> str:
        return (
            "Make an outbound phone call. Two modes:\n"
            "1. 'tts' — speak a one-way message (Twilio TTS)\n"
            "2. 'conversation' — two-way AI call (Twilio + Deepgram STT/TTS + Mawa agent)\n"
            "Requires Twilio credentials. Use when the user asks to call someone."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Phone number in E.164 format (e.g., +12035551234)",
                },
                "message": {
                    "type": "string",
                    "description": "For TTS mode: the message to speak. For conversation mode: the opening greeting.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["tts", "conversation"],
                    "description": "tts = one-way message. conversation = two-way AI call. Default from settings.",
                },
                "context": {
                    "type": "string",
                    "description": "For conversation mode: context about why you're calling (helps Mawa respond appropriately).",
                },
            },
            "required": ["to", "message"],
        }

    def _get_settings(self) -> dict:
        if not self._workspace:
            return {}
        try:
            from nanobot.hooks.builtin.feature_registry import get_setting
            return {
                "voice": get_setting(self._workspace, "phoneCallDefaultVoice", "alice"),
                "mode": get_setting(self._workspace, "phoneCallMode", "tts"),
                "enabled": get_setting(self._workspace, "phoneCallEnabled", True),
            }
        except Exception:
            return {"voice": "alice", "mode": "tts", "enabled": True}

    async def execute(self, to: str, message: str, mode: str = "", context: str = "", **kwargs) -> str:
        settings = self._get_settings()

        if not settings.get("enabled", True):
            return "Phone calls are disabled. Enable in settings: settings(set, 'phoneCallEnabled', true)"

        # Get Twilio credentials
        account_sid = _get_cred("TWILIO_ACCOUNT_SID") or _get_cred("twilio_sid")
        auth_token = _get_cred("TWILIO_AUTH_TOKEN") or _get_cred("twilio_token")
        from_number = _get_cred("TWILIO_PHONE_NUMBER") or _get_cred("twilio_phone")

        if not all([account_sid, auth_token, from_number]):
            return (
                "Error: Twilio credentials not configured. Save them:\n"
                "- credentials(save, 'twilio_sid', 'your_account_sid')\n"
                "- credentials(save, 'twilio_token', 'your_auth_token')\n"
                "- credentials(save, 'twilio_phone', '+1your_number')\n\n"
                "Get free at https://www.twilio.com/try-twilio"
            )

        # Normalize phone number
        if not to.startswith("+"):
            to = f"+1{to}"

        call_mode = mode or settings.get("mode", "tts")
        voice = settings.get("voice", "alice")

        if call_mode == "conversation":
            return await self._make_conversation_call(account_sid, auth_token, from_number, to, message, context, voice)
        else:
            return await self._make_tts_call(account_sid, auth_token, from_number, to, message, voice)

    async def _make_tts_call(self, sid: str, token: str, from_num: str, to: str, message: str, voice: str) -> str:
        """Simple one-way TTS call via Twilio."""
        twiml = f'<Response><Say voice="{voice}">{_escape_xml(message)}</Say></Response>'

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
                    auth=(sid, token),
                    data={"To": to, "From": from_num, "Twiml": twiml},
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    logger.info("TTS call initiated: {} → {} (SID: {})", from_num, to, data.get("sid"))
                    return f"Call initiated to {to}. Mawa will speak: \"{message[:60]}...\"\nCall SID: {data.get('sid')}"
                else:
                    error = resp.json().get("message", resp.text[:200])
                    return f"Error: Twilio call failed — {error}"
        except Exception as e:
            return f"Error: Failed to make call — {e}"

    async def _make_conversation_call(self, sid: str, token: str, from_num: str, to: str, greeting: str, context: str, voice: str) -> str:
        """Two-way conversational call using Twilio Media Streams + Deepgram.

        Flow:
        1. Twilio calls the number
        2. On connect, TwiML starts a media stream to our WebSocket
        3. Our server receives audio → Deepgram STT → agent → Deepgram TTS → back to Twilio
        """
        # Check if Deepgram is configured
        dg_key = _get_cred("DEEPGRAM_API_KEY") or _get_cred("deepgram")
        if not dg_key:
            # Fall back to TTS mode
            logger.warning("Deepgram not configured, falling back to TTS call")
            return await self._make_tts_call(sid, token, from_num, to, greeting, voice)

        # For conversation mode, we need a publicly accessible WebSocket URL
        # The TwiML connects Twilio's media stream to our server
        # For now, use TTS with a note about conversation setup
        twiml = (
            f'<Response>'
            f'<Say voice="{voice}">{_escape_xml(greeting)}</Say>'
            f'<Pause length="1"/>'
            f'<Say voice="{voice}">If you need anything, just let me know through the app.</Say>'
            f'</Response>'
        )

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls.json",
                    auth=(sid, token),
                    data={"To": to, "From": from_num, "Twiml": twiml},
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    logger.info("Conversation call initiated: {} → {} (SID: {})", from_num, to, data.get("sid"))
                    return (
                        f"Call initiated to {to}.\n"
                        f"Greeting: \"{greeting[:60]}...\"\n"
                        f"Context: {context[:60] if context else 'none'}\n"
                        f"Call SID: {data.get('sid')}\n"
                        f"Note: Full two-way conversation requires a public WebSocket endpoint. "
                        f"Currently using TTS mode with greeting."
                    )
                else:
                    error = resp.json().get("message", resp.text[:200])
                    return f"Error: Twilio call failed — {error}"
        except Exception as e:
            return f"Error: Failed to make call — {e}"


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
