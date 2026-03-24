"""Phone call tool — make outbound calls with TTS.

Supports:
1. Twilio — outbound calls with TTS speech
2. ElevenLabs — high-quality voice synthesis (generates audio, sends via Twilio)

Use cases:
- "Call Tonni and tell her I'm on my way"
- "Call this number and read the message"
- "Make a reminder call to +1234567890"
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool


class PhoneCallTool(Tool):
    """Make outbound phone calls with text-to-speech."""

    @property
    def name(self) -> str:
        return "phone_call"

    @property
    def description(self) -> str:
        return (
            "Make an outbound phone call and speak a message using TTS. "
            "Requires Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER). "
            "Use when the user asks to call someone, make a reminder call, or send a voice message."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Phone number to call in E.164 format (e.g., +12035551234)",
                },
                "message": {
                    "type": "string",
                    "description": "The message to speak during the call",
                },
                "voice": {
                    "type": "string",
                    "enum": ["alice", "man", "woman", "Polly.Joanna", "Polly.Matthew"],
                    "description": "TTS voice to use. Default: alice (natural female)",
                },
            },
            "required": ["to", "message"],
        }

    async def execute(self, to: str, message: str, voice: str = "alice", **kwargs) -> str:
        # Get Twilio credentials
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        from_number = os.environ.get("TWILIO_PHONE_NUMBER")

        # Try vault if not in env
        if not all([account_sid, auth_token, from_number]):
            try:
                from nanobot.setup.vault import load_vault
                vault = load_vault()
                account_sid = account_sid or vault.get("cred.twilio_sid") or vault.get("cred.twilio_account_sid")
                auth_token = auth_token or vault.get("cred.twilio_token") or vault.get("cred.twilio_auth_token")
                from_number = from_number or vault.get("cred.twilio_phone") or vault.get("cred.twilio_phone_number")
            except Exception:
                pass

        if not all([account_sid, auth_token, from_number]):
            return (
                "Error: Twilio credentials not configured. Set these environment variables or save to vault:\n"
                "- TWILIO_ACCOUNT_SID (or cred.twilio_sid)\n"
                "- TWILIO_AUTH_TOKEN (or cred.twilio_token)\n"
                "- TWILIO_PHONE_NUMBER (or cred.twilio_phone)\n\n"
                "Get a free Twilio account at https://www.twilio.com/try-twilio"
            )

        # Validate phone number format
        if not to.startswith("+"):
            to = f"+1{to}"  # Assume US if no country code

        # Build TwiML for the call
        twiml = f'<Response><Say voice="{voice}">{_escape_xml(message)}</Say></Response>'

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json",
                    auth=(account_sid, auth_token),
                    data={
                        "To": to,
                        "From": from_number,
                        "Twiml": twiml,
                    },
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    call_sid = data.get("sid", "")
                    status = data.get("status", "")
                    logger.info("Phone call initiated: {} → {} (SID: {})", from_number, to, call_sid)
                    return f"Call initiated to {to}. Status: {status}. Call SID: {call_sid}"
                else:
                    error = resp.json().get("message", resp.text[:200])
                    return f"Error: Twilio call failed — {error}"

        except Exception as e:
            return f"Error: Failed to make call — {e}"


def _escape_xml(text: str) -> str:
    """Escape special XML characters for TwiML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
