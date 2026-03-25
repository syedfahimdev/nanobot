"""Voice provider registry — validation, voice listing, cloning.

All providers registered with:
- Required credentials (vault key names)
- Available voices
- Voice cloning support
- Validation endpoint
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
from pathlib import Path
from typing import Any

import httpx
import websockets
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# Provider Registry
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDERS = {
    "deepgram": {
        "label": "Deepgram",
        "type": "cloud",
        "credentials": [
            {"key": "DEEPGRAM_API_KEY", "vault": "deepgram", "label": "Deepgram API Key", "url": "https://console.deepgram.com/signup"},
        ],
        "supports_stt": True,
        "supports_tts": True,
        "supports_clone": False,
        "emotions": False,
        "languages": ["en"],
        "voices": [
            {"id": "aura-2-luna-en", "name": "Luna", "gender": "female", "preview": None},
            {"id": "aura-2-asteria-en", "name": "Asteria", "gender": "female", "preview": None},
            {"id": "aura-2-stella-en", "name": "Stella", "gender": "female", "preview": None},
            {"id": "aura-2-athena-en", "name": "Athena", "gender": "female", "preview": None},
            {"id": "aura-2-orion-en", "name": "Orion", "gender": "male", "preview": None},
            {"id": "aura-2-arcas-en", "name": "Arcas", "gender": "male", "preview": None},
        ],
    },
    "elevenlabs": {
        "label": "ElevenLabs",
        "type": "cloud",
        "credentials": [
            {"key": "ELEVENLABS_API_KEY", "vault": "elevenlabs", "label": "ElevenLabs API Key", "url": "https://elevenlabs.io"},
        ],
        "supports_stt": False,
        "supports_tts": True,
        "supports_clone": True,
        "emotions": False,
        "languages": ["en", "bn", "hi", "zh", "es", "fr", "de", "ar", "ja", "ko", "pt", "ru", "ur", "pl", "it", "tr", "nl"],
        "voices": [
            {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "gender": "female"},
            {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "gender": "female"},
            {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "gender": "female"},
            {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "gender": "male"},
            {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold", "gender": "male"},
            {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "gender": "male"},
            {"id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam", "gender": "male"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Credential Validation
# ═══════════════════════════════════════════════════════════════════════════════

def _get_credential(cred_info: dict) -> str | None:
    """Get a credential from env or vault."""
    val = os.environ.get(cred_info["key"])
    if val:
        return val
    try:
        from nanobot.setup.vault import load_vault
        vault = load_vault()
        for vkey in [f"cred.{cred_info['vault']}_api_key", f"cred.{cred_info['vault']}", cred_info["vault"]]:
            if vault.get(vkey):
                return vault[vkey]
    except Exception:
        pass
    return None


async def validate_provider(provider_name: str, workspace: Path) -> dict:
    """Validate a provider's credentials and connectivity.

    Returns: {ok, provider, message, missing_credentials}
    """
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return {"ok": False, "provider": provider_name, "message": f"Unknown provider: {provider_name}"}

    # Check credentials
    missing = []
    for cred in provider.get("credentials", []):
        if not _get_credential(cred):
            missing.append(cred)

    if missing:
        instructions = []
        for m in missing:
            instructions.append(
                f"• Save `{m['label']}` to vault: credentials(save, '{m['vault']}_api_key', 'your_key')\n"
                f"  Get one at: {m.get('url', 'provider website')}"
            )
        return {
            "ok": False,
            "provider": provider_name,
            "message": f"Missing credentials for {provider['label']}",
            "missing_credentials": missing,
            "instructions": "\n".join(instructions),
        }

    # Check Modal endpoint connectivity
    if provider["type"] == "modal":
        setting_key = provider.get("endpoint_setting", "")
        if setting_key:
            from nanobot.hooks.builtin.feature_registry import get_setting
            endpoint = get_setting(workspace, setting_key, "")
            if not endpoint:
                return {
                    "ok": False, "provider": provider_name,
                    "message": f"No endpoint URL configured. Set '{setting_key}' in settings.",
                }
            # Ping the endpoint
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get(endpoint.replace("/api", "").rstrip("/"))
                    # Modal returns 405 for GET on POST endpoint — that's fine, it means it's alive
                    if r.status_code in (200, 405, 422):
                        return {"ok": True, "provider": provider_name, "message": f"{provider['label']} is reachable (Modal)"}
                    return {"ok": False, "provider": provider_name, "message": f"Endpoint returned {r.status_code}"}
            except Exception as e:
                return {"ok": False, "provider": provider_name, "message": f"Cannot reach endpoint: {e}"}

    # Cloud providers — check with a minimal API call
    if provider_name == "deepgram":
        key = _get_credential(provider["credentials"][0])
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("https://api.deepgram.com/v1/projects",
                    headers={"Authorization": f"Token {key}"})
                if r.status_code == 200:
                    return {"ok": True, "provider": provider_name, "message": "Deepgram connected"}
                return {"ok": False, "provider": provider_name, "message": f"Deepgram auth failed: {r.status_code}"}
        except Exception as e:
            return {"ok": False, "provider": provider_name, "message": f"Cannot reach Deepgram: {e}"}

    return {"ok": True, "provider": provider_name, "message": f"{provider['label']} configured"}


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Listing
# ═══════════════════════════════════════════════════════════════════════════════

def get_provider_voices(provider_name: str) -> dict:
    """Get available voices for a provider."""
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return {"provider": provider_name, "voices": [], "error": "Unknown provider"}

    return {
        "provider": provider_name,
        "label": provider["label"],
        "voices": provider.get("voices", []),
        "languages": provider.get("languages", []),
        "supports_clone": provider.get("supports_clone", False),
        "emotions": provider.get("emotions", False),
    }


def get_all_providers() -> list[dict]:
    """Get summary of all providers."""
    return [
        {
            "id": name,
            "label": p["label"],
            "type": p["type"],
            "stt": p.get("supports_stt", False),
            "tts": p.get("supports_tts", False),
            "clone": p.get("supports_clone", False),
            "emotions": p.get("emotions", False),
            "languages": p.get("languages", []),
            "needs_credentials": len(p.get("credentials", [])) > 0,
        }
        for name, p in PROVIDERS.items()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Cloning
# ═══════════════════════════════════════════════════════════════════════════════

def save_voice_sample(workspace: Path, audio_b64: str, name: str = "my_voice") -> str:
    """Save a voice sample for cloning. Returns the file path."""
    voice_dir = workspace / "voice_samples"
    voice_dir.mkdir(parents=True, exist_ok=True)

    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    path = voice_dir / f"{safe_name}.wav"
    path.write_bytes(base64.b64decode(audio_b64))
    logger.info("Voice sample saved: {} ({} bytes)", path, path.stat().st_size)
    return str(path)


def get_voice_samples(workspace: Path) -> list[dict]:
    """List saved voice samples."""
    voice_dir = workspace / "voice_samples"
    if not voice_dir.exists():
        return []

    samples = []
    for f in sorted(voice_dir.glob("*.wav")):
        samples.append({
            "name": f.stem,
            "path": str(f),
            "size": f.stat().st_size,
        })
    return samples


def delete_voice_sample(workspace: Path, name: str) -> bool:
    """Delete a voice sample."""
    path = workspace / "voice_samples" / f"{name}.wav"
    if path.exists():
        path.unlink()
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# TTS Dispatcher — routes to the correct provider
# ═══════════════════════════════════════════════════════════════════════════════

async def generate_tts(
    text: str,
    workspace: Path,
    provider_name: str | None = None,
    voice_id: str | None = None,
    emotion: str = "neutral",
    deepgram_api_key: str | None = None,
    deepgram_model: str = "aura-2-luna-en",
) -> bytes | None:
    """Generate TTS audio using the configured provider.

    Returns raw PCM/WAV audio bytes, or None on failure.
    Falls back to Deepgram if the selected provider fails.
    """
    from nanobot.hooks.builtin.feature_registry import get_setting

    if not provider_name:
        provider_name = get_setting(workspace, "voiceTtsProvider", "deepgram")

    # Strip markdown from text
    clean = _strip_md(text)
    if not clean or len(clean) < 2:
        return None
    if len(clean) > 2000:
        clean = clean[:2000]

    try:
        if provider_name == "deepgram":
            return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
        elif provider_name == "elevenlabs":
            el_key = _get_elevenlabs_key()
            el_voice = voice_id or get_setting(workspace, "elevenlabsVoiceId", "21m00Tcm4TlvDq8ikWAM")
            el_model = get_setting(workspace, "elevenlabsModel", "eleven_flash_v2_5")
            use_flash = "flash" in el_model
            if not el_key:
                logger.warning("ElevenLabs: no API key, falling back to Deepgram")
                return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
            result = await _tts_elevenlabs(clean, el_key, el_voice, flash=use_flash, model_id=el_model, workspace=workspace)
            if result:
                return result
            logger.warning("ElevenLabs TTS failed, falling back to Deepgram")
            return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
        else:
            logger.warning("Unknown TTS provider '{}', using Deepgram", provider_name)
            return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
    except Exception as e:
        logger.error("TTS dispatch error ({}): {}", provider_name, e)
        if provider_name != "deepgram":
            return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
        return None


def _strip_md(text: str) -> str:
    """Strip markdown formatting for TTS."""
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'[-*]\s+', '', text)
    return text.strip()


def _get_clone_sample(workspace: Path) -> str | None:
    """Get the default voice clone sample path."""
    samples = get_voice_samples(workspace)
    if samples:
        return samples[0]["path"]
    return None


def _get_hf_token() -> str | None:
    val = os.environ.get("HF_TOKEN")
    if val:
        return val
    try:
        from nanobot.setup.vault import load_vault
        vault = load_vault()
        for k in ["cred.huggingface_api_key", "cred.huggingface", "HF_TOKEN"]:
            if vault.get(k):
                return vault[k]
    except Exception:
        pass
    return None


async def _tts_deepgram(text: str, api_key: str | None, model: str) -> bytes | None:
    """Deepgram TTS — fast cloud API."""
    if not api_key:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            try:
                from nanobot.setup.vault import load_vault
                vault = load_vault()
                api_key = vault.get("cred.deepgram_api_key") or vault.get("cred.deepgram")
            except Exception:
                pass
    if not api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.deepgram.com/v1/speak",
                headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
                params={"model": model, "encoding": "linear16", "sample_rate": "24000", "container": "none"},
                json={"text": text},
            )
            resp.raise_for_status()
            audio = resp.content
            return audio if len(audio) > 100 else None
    except Exception as e:
        logger.error("Deepgram TTS failed: {}", e)
        return None










def _get_elevenlabs_key() -> str | None:
    """Get ElevenLabs API key from env or vault."""
    val = os.environ.get("ELEVENLABS_API_KEY")
    if val:
        return val
    try:
        from nanobot.setup.vault import load_vault
        vault = load_vault()
        for k in ["cred.elevenlabs_api_key", "cred.elevenlabs"]:
            if vault.get(k):
                return vault[k]
    except Exception:
        pass
    return None


def _build_elevenlabs_voice_settings(workspace: Path | None = None, model_id: str = "") -> dict:
    """Build voice_settings dict from user settings. Reads dynamically — never hardcoded.

    v3 doesn't support style/speaker_boost — emotion comes from the model's
    contextual understanding of the text. v2 models use explicit style params.
    """
    from nanobot.hooks.builtin.feature_registry import get_setting
    ws = workspace or Path("/root/.nanobot/workspace")
    settings: dict = {
        "stability": get_setting(ws, "elevenlabsStability", 0.3),
        "similarity_boost": get_setting(ws, "elevenlabsSimilarity", 0.7),
    }
    # Only v2 models support style and speaker_boost
    if "v2" in model_id and "v3" not in model_id:
        settings["style"] = get_setting(ws, "elevenlabsStyle", 0.4)
        settings["use_speaker_boost"] = get_setting(ws, "elevenlabsSpeakerBoost", True)
    return settings


async def _tts_elevenlabs(text: str, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", flash: bool = True, model_id: str = "", workspace: Path | None = None) -> bytes | None:
    """ElevenLabs TTS — buffered. Returns full MP3."""
    model = model_id or ("eleven_flash_v2_5" if flash else "eleven_multilingual_v2")
    voice_settings = _build_elevenlabs_voice_settings(workspace, model_id=model)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text,
                    "model_id": model,
                    "voice_settings": voice_settings,
                },
            )
            resp.raise_for_status()
            audio = resp.content
            return audio if len(audio) > 100 else None
    except Exception as e:
        logger.error("ElevenLabs TTS failed: {}", e)
    return None


async def stream_tts_elevenlabs(
    text: str,
    api_key: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    model_id: str = "eleven_flash_v2_5",
    chunk_callback=None,
    workspace: Path | None = None,
) -> bytes | None:
    """ElevenLabs streaming TTS — yields audio chunks via callback as they arrive.

    Uses the /v1/text-to-speech/{voice_id}/stream endpoint.

    Args:
        chunk_callback: async fn(chunk_bytes) called for each audio chunk.
            If None, buffers and returns the full audio.
    Returns:
        Full audio bytes if no callback, else None (chunks sent via callback).
    """
    voice_settings = _build_elevenlabs_voice_settings(workspace, model_id=model_id)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": model_id,
                    "voice_settings": voice_settings,
                    "output_format": "mp3_44100_128",
                },
            ) as resp:
                resp.raise_for_status()
                if chunk_callback:
                    # Stream chunks to callback as they arrive
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes(4096):
                        buf.extend(chunk)
                        # Send chunks of ~8KB+ for reliable decoding
                        if len(buf) >= 8192:
                            await chunk_callback(bytes(buf))
                            buf.clear()
                    # Flush remaining
                    if buf:
                        await chunk_callback(bytes(buf))
                    return None
                else:
                    # Buffer full response
                    audio = await resp.aread()
                    return audio if len(audio) > 100 else None
    except Exception as e:
        logger.error("ElevenLabs streaming TTS failed: {}", e)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Deepgram TTS WebSocket Streaming
# ═══════════════════════════════════════════════════════════════════════════════

class DeepgramTTSStream:
    """Persistent WebSocket connection to Deepgram's TTS streaming API.

    Keeps a single connection open per voice session. Text is sent as chunks,
    audio arrives as raw PCM (linear16) and is forwarded to the client immediately.

    Key Deepgram features used:
    - WebSocket streaming for ultra-low latency (~200ms TTFB)
    - linear16 encoding at 24000Hz (direct PCM — no MP3 decode needed)
    - container=none (prevents clicking from header misinterpretation)
    - Flush command to finalize audio after each sentence
    - Aura-2 context-aware voices with natural emotion from text formatting
    """

    def __init__(self, api_key: str, model: str = "aura-2-thalia-en", sample_rate: int = 24000):
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self._ws = None
        self._audio_callback = None
        self._recv_task = None
        self._connected = False

    async def connect(self, audio_callback):
        """Connect to Deepgram TTS WebSocket. audio_callback(bytes) receives PCM chunks."""
        self._audio_callback = audio_callback
        url = (
            f"wss://api.deepgram.com/v1/speak"
            f"?model={self.model}"
            f"&encoding=linear16"
            f"&sample_rate={self.sample_rate}"
            f"&container=none"
        )
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Token {self.api_key}"},
                ping_interval=20,
            )
            self._connected = True
            self._recv_task = asyncio.create_task(self._receive_loop())
            logger.debug("Deepgram TTS WebSocket connected (model={}, sr={})", self.model, self.sample_rate)
        except Exception as e:
            logger.error("Deepgram TTS WebSocket connect failed: {}", e)
            self._connected = False

    async def _receive_loop(self):
        """Receive audio data from Deepgram and forward to callback."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes) and len(message) > 0:
                    if self._audio_callback:
                        await self._audio_callback(message)
                elif isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        if msg_type == "Flushed":
                            logger.debug("Deepgram TTS: flush acknowledged")
                        elif msg_type == "Warning":
                            logger.warning("Deepgram TTS warning: {}", data.get("warn_msg", ""))
                        elif msg_type == "Error":
                            logger.error("Deepgram TTS error: {}", data.get("err_msg", ""))
                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed as e:
            logger.debug("Deepgram TTS WebSocket closed: {}", e)
        except Exception as e:
            logger.error("Deepgram TTS receive error: {}", e)
        finally:
            self._connected = False

    async def speak(self, text: str):
        """Send text to be spoken. Audio arrives via the callback."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": "Speak", "text": text}))
        except Exception as e:
            logger.error("Deepgram TTS send error: {}", e)
            self._connected = False

    async def flush(self):
        """Flush remaining audio. Call after sending the last text chunk."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": "Flush"}))
        except Exception as e:
            logger.error("Deepgram TTS flush error: {}", e)

    async def clear(self):
        """Clear the audio buffer (interrupt current speech)."""
        if not self._connected or not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": "Clear"}))
        except Exception:
            pass

    async def close(self):
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "Close"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None

    @property
    def is_connected(self) -> bool:
        return self._connected


def format_text_for_aura2(text: str) -> str:
    """Format text for natural Aura-2 speech output.

    Aura-2 is context-aware — proper punctuation directly affects
    pacing, intonation, and expressiveness.
    """
    import re

    text = text.strip()
    if text and text[-1] not in '.!?':
        text += '.'

    # Add natural pauses for lists (comma before last item)
    text = re.sub(r'(\w+)\s+(\w+)\s+and\s+(\w+)', r'\1, \2, and \3', text)

    # Add comma after direct address ("Hello Maria" → "Hello, Maria")
    text = re.sub(r'\b(Hello|Hey|Hi|Okay|Sure|Well|Look|See|Right)\s+([A-Z])', r'\1, \2', text)

    # Numbers: add periods between groups for phone numbers
    text = re.sub(r'(\d{3})(\d{3})(\d{4})', r'\1.\2.\3', text)

    return text
