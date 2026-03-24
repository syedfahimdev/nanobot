"""Phone call tool — outbound calls via ElevenLabs Conversational AI.

Two modes:
1. TTS call — Twilio speaks a static message (one-way, cheap)
2. Conversation — ElevenLabs agent handles the full call (STT + LLM + TTS)
   Mawa's context (memory, goals, user profile) is injected into the agent.

For conversation mode, ElevenLabs handles everything — no serverless GPUs needed.
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
        for key in [f"cred.{name.lower()}", f"cred.{name}", name.lower()]:
            if vault.get(key):
                return vault[key]
    except Exception:
        pass
    return None


def _build_mawa_context(workspace: Path) -> str:
    """Build Mawa's context from memory layers for the ElevenLabs agent."""
    parts = []
    mem_dir = workspace / "memory"

    # User profile from LONG_TERM
    lt = mem_dir / "LONG_TERM.md"
    if lt.exists():
        content = lt.read_text(encoding="utf-8")[:2000]
        parts.append("User profile and facts:\n" + content)

    # Today's context from SHORT_TERM
    st = mem_dir / "SHORT_TERM.md"
    if st.exists():
        content = st.read_text(encoding="utf-8")[:1000]
        parts.append("Today's context:\n" + content)

    # Goals
    goals = mem_dir / "GOALS.md"
    if goals.exists():
        content = goals.read_text(encoding="utf-8")[:500]
        parts.append("Goals:\n" + content)

    # Learnings (how user likes to be helped)
    learnings = mem_dir / "LEARNINGS.md"
    if learnings.exists():
        content = learnings.read_text(encoding="utf-8")[:500]
        parts.append("User preferences:\n" + content)

    return "\n\n".join(parts) if parts else "No context available."


AGENT_SYSTEM_PROMPT = """You are Mawa, a warm and intelligent personal AI assistant.
You are currently on a phone call with your user.

RULES:
- Be concise — this is a phone call, not a chat
- Be warm, natural, and conversational
- Use the context below to personalize your responses
- If asked about something not in your context, say you'll check and follow up in the app
- Never say "I'm an AI" — you are Mawa, the user's personal assistant

USER CONTEXT:
{context}

CALL CONTEXT:
{call_context}"""


class PhoneCallTool(Tool):
    """Make outbound phone calls — TTS message or full AI conversation via ElevenLabs."""

    def __init__(self, workspace: Path | None = None):
        self._workspace = workspace
        self._agent_id: str | None = None

    @property
    def name(self) -> str:
        return "phone_call"

    @property
    def description(self) -> str:
        return (
            "Make an outbound phone call. Two modes:\n"
            "1. 'tts' — speak a one-way message (Twilio TTS, cheap)\n"
            "2. 'conversation' — full two-way AI call (ElevenLabs agent with Mawa's memory)\n"
            "Requires Twilio + ElevenLabs credentials for conversation mode."
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
                    "description": "For TTS: the message. For conversation: the greeting + why you're calling.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["tts", "conversation"],
                    "description": "tts = one-way message. conversation = two-way AI call via ElevenLabs.",
                },
                "context": {
                    "type": "string",
                    "description": "For conversation: context about why you're calling.",
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
            return "Phone calls are disabled. Enable in settings."

        # Normalize phone number
        if not to.startswith("+"):
            to = f"+1{to}"

        call_mode = mode or settings.get("mode", "tts")

        if call_mode == "conversation":
            return await self._make_elevenlabs_call(to, message, context)
        else:
            return await self._make_tts_call(to, message, settings.get("voice", "alice"))

    async def _make_tts_call(self, to: str, message: str, voice: str) -> str:
        """Simple one-way TTS call via Twilio."""
        sid = _get_cred("TWILIO_ACCOUNT_SID") or _get_cred("twilio_sid")
        token = _get_cred("TWILIO_AUTH_TOKEN") or _get_cred("twilio_token")
        from_num = _get_cred("TWILIO_PHONE_NUMBER") or _get_cred("twilio_phone")

        if not all([sid, token, from_num]):
            return (
                "Error: Twilio credentials not configured. Save them:\n"
                "- credentials(save, 'twilio_sid', 'your_account_sid')\n"
                "- credentials(save, 'twilio_token', 'your_auth_token')\n"
                "- credentials(save, 'twilio_phone', '+1your_number')"
            )

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
                    return f"Call initiated to {to}.\nMessage: \"{message[:60]}...\"\nCall SID: {data.get('sid')}"
                else:
                    return f"Error: Twilio call failed — {resp.json().get('message', resp.text[:200])}"
        except Exception as e:
            return f"Error: Failed to make call — {e}"

    async def _make_elevenlabs_call(self, to: str, greeting: str, call_context: str) -> str:
        """Two-way conversational call via ElevenLabs Conversational AI.

        Flow:
        1. Build Mawa's context from memory
        2. Create/update ElevenLabs agent with context
        3. Initiate outbound call via ElevenLabs phone API
        """
        el_key = _get_cred("ELEVENLABS_API_KEY") or _get_cred("elevenlabs_api_key")
        if not el_key:
            return (
                "Error: ElevenLabs API key not configured for conversation calls.\n"
                "Save it: credentials(save, 'elevenlabs_api_key', 'your_key')\n"
                "Get one at: https://elevenlabs.io"
            )

        # Build context from Mawa's memory
        mawa_context = _build_mawa_context(self._workspace) if self._workspace else "No workspace context."

        system_prompt = AGENT_SYSTEM_PROMPT.format(
            context=mawa_context[:3000],
            call_context=call_context or f"Greeting: {greeting}",
        )

        try:
            headers = {"xi-api-key": el_key, "Content-Type": "application/json"}

            # Step 1: Create or update the agent
            agent_id = await self._ensure_agent(headers, system_prompt, greeting)
            if not agent_id:
                return "Error: Failed to create ElevenLabs conversation agent."

            # Step 2: Check if Twilio is configured for outbound
            twilio_sid = _get_cred("TWILIO_ACCOUNT_SID") or _get_cred("twilio_sid")
            twilio_token = _get_cred("TWILIO_AUTH_TOKEN") or _get_cred("twilio_token")
            twilio_phone = _get_cred("TWILIO_PHONE_NUMBER") or _get_cred("twilio_phone")

            if all([twilio_sid, twilio_token, twilio_phone]):
                # Outbound call via ElevenLabs + Twilio
                result = await self._initiate_outbound_call(headers, agent_id, to, twilio_sid, twilio_token, twilio_phone)
                return result
            else:
                # No Twilio — return agent info for manual connection
                return (
                    f"ElevenLabs conversation agent ready!\n"
                    f"Agent ID: {agent_id}\n"
                    f"Context: {call_context or 'general'}\n"
                    f"Connect via: wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}\n\n"
                    f"To make outbound calls, save Twilio credentials."
                )

        except Exception as e:
            logger.error("ElevenLabs call error: {}", e)
            return f"Error: ElevenLabs call failed — {e}"

    async def _ensure_agent(self, headers: dict, system_prompt: str, greeting: str) -> str | None:
        """Create or update the ElevenLabs conversation agent."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Try to update existing agent first
                if self._agent_id:
                    resp = await client.patch(
                        f"https://api.elevenlabs.io/v1/convai/agents/{self._agent_id}",
                        headers=headers,
                        json={
                            "conversation_config": {
                                "agent": {
                                    "prompt": {"prompt": system_prompt},
                                    "first_message": greeting,
                                }
                            }
                        },
                    )
                    if resp.status_code == 200:
                        logger.info("Updated ElevenLabs agent: {}", self._agent_id)
                        return self._agent_id

                # Create new agent
                resp = await client.post(
                    "https://api.elevenlabs.io/v1/convai/agents/create",
                    headers=headers,
                    json={
                        "name": "Mawa Phone Agent",
                        "conversation_config": {
                            "agent": {
                                "first_message": greeting,
                                "language": "en",
                                "prompt": {
                                    "prompt": system_prompt,
                                    "temperature": 0.7,
                                },
                            },
                            "tts": {
                                "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Rachel
                            },
                        },
                    },
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    self._agent_id = data.get("agent_id")
                    logger.info("Created ElevenLabs agent: {}", self._agent_id)
                    return self._agent_id
                else:
                    logger.error("Agent create failed: {} {}", resp.status_code, resp.text[:200])
                    return None
        except Exception as e:
            logger.error("Agent ensure error: {}", e)
            return None

    async def _initiate_outbound_call(
        self, headers: dict, agent_id: str, to: str,
        twilio_sid: str, twilio_token: str, twilio_phone: str,
    ) -> str:
        """Initiate an outbound call through ElevenLabs + Twilio."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"https://api.elevenlabs.io/v1/convai/twilio/outbound-call",
                    headers=headers,
                    json={
                        "agent_id": agent_id,
                        "agent_phone_number_id": twilio_phone,
                        "to_number": to,
                        "twilio_account_sid": twilio_sid,
                        "twilio_auth_token": twilio_token,
                    },
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    call_sid = data.get("call_sid", data.get("sid", ""))
                    return (
                        f"Conversation call initiated to {to}!\n"
                        f"Agent: Mawa (ElevenLabs AI)\n"
                        f"Call SID: {call_sid}\n"
                        f"The AI agent has Mawa's full context — memory, goals, user profile.\n"
                        f"Agent ID: {agent_id}"
                    )
                else:
                    error = resp.text[:200]
                    logger.error("Outbound call failed: {}", error)
                    # Fallback: return agent ID for manual connection
                    return (
                        f"Outbound call API returned {resp.status_code}.\n"
                        f"Agent is ready — connect manually or via Twilio webhook:\n"
                        f"Agent ID: {agent_id}\n"
                        f"WebSocket: wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}\n"
                        f"Error: {error}"
                    )
        except Exception as e:
            return f"Error initiating call: {e}\nAgent ID: {agent_id}"


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
