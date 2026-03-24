"""Voice provider registry — validation, voice listing, cloning.

All providers registered with:
- Required credentials (vault key names)
- Available voices
- Voice cloning support
- Validation endpoint
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx
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
    "mimo-audio": {
        "label": "MiMo-Audio",
        "type": "modal",
        "credentials": [],  # No API key — uses Modal endpoint
        "endpoint_setting": "mimoAudioEndpoint",
        "supports_stt": True,
        "supports_tts": True,
        "supports_clone": True,
        "emotions": True,
        "languages": ["en", "zh"],
        "voices": [
            {"id": "default", "name": "MiMo Default", "gender": "neutral", "preview": None},
            {"id": "happy", "name": "Happy", "gender": "neutral", "emotion": "happy", "preview": None},
            {"id": "sad", "name": "Sad", "gender": "neutral", "emotion": "sad", "preview": None},
            {"id": "excited", "name": "Excited", "gender": "neutral", "emotion": "excited", "preview": None},
            {"id": "whisper", "name": "Whisper", "gender": "neutral", "emotion": "whisper", "preview": None},
            {"id": "laughing", "name": "Laughing", "gender": "neutral", "emotion": "laughing", "preview": None},
            {"id": "custom", "name": "Your Voice (clone)", "gender": "custom", "clone": True, "preview": None},
        ],
    },
    "coqui-xtts": {
        "label": "Coqui XTTS v2",
        "type": "modal",
        "credentials": [],
        "endpoint_setting": "coquiXttsEndpoint",
        "supports_stt": False,
        "supports_tts": True,
        "supports_clone": True,
        "emotions": False,
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu", "hi"],
        "voices": [
            {"id": "default", "name": "XTTS Default", "gender": "female", "preview": None},
            {"id": "custom", "name": "Your Voice (clone)", "gender": "custom", "clone": True, "preview": None},
        ],
    },
    "mms-tts": {
        "label": "Meta MMS-TTS",
        "type": "huggingface",
        "credentials": [
            {"key": "HF_TOKEN", "vault": "huggingface", "label": "HuggingFace Token (free)", "url": "https://huggingface.co/settings/tokens"},
        ],
        "supports_stt": False,
        "supports_tts": True,
        "supports_clone": False,
        "emotions": False,
        "languages": ["en", "bn", "hi", "ur", "ar", "es", "fr", "de", "zh", "ja", "ko", "ta", "te", "ml", "gu", "pa"],
        "voices": [
            {"id": "facebook/mms-tts-eng", "name": "English", "language": "en"},
            {"id": "facebook/mms-tts-ben", "name": "Bengali", "language": "bn"},
            {"id": "facebook/mms-tts-hin", "name": "Hindi", "language": "hi"},
            {"id": "facebook/mms-tts-urd", "name": "Urdu", "language": "ur"},
            {"id": "facebook/mms-tts-ara", "name": "Arabic", "language": "ar"},
            {"id": "facebook/mms-tts-spa", "name": "Spanish", "language": "es"},
            {"id": "facebook/mms-tts-fra", "name": "French", "language": "fr"},
            {"id": "facebook/mms-tts-tam", "name": "Tamil", "language": "ta"},
            {"id": "facebook/mms-tts-guj", "name": "Gujarati", "language": "gu"},
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
        elif provider_name == "mimo-audio":
            endpoint = get_setting(workspace, "mimoAudioEndpoint", "")
            clone_path = _get_clone_sample(workspace) if voice_id == "custom" else None
            result = await _tts_mimo_audio(clean, endpoint, emotion, clone_path)
            if result:
                return result
            # Fallback to Deepgram
            logger.warning("MiMo-Audio TTS failed, falling back to Deepgram")
            return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
        elif provider_name == "coqui-xtts":
            endpoint = get_setting(workspace, "coquiXttsEndpoint", "")
            lang = get_setting(workspace, "voiceTtsLanguage", "en")
            clone_path = _get_clone_sample(workspace) if voice_id == "custom" else None
            result = await _tts_coqui(clean, endpoint, lang, clone_path)
            if result:
                return result
            logger.warning("Coqui XTTS TTS failed, falling back to Deepgram")
            return await _tts_deepgram(clean, deepgram_api_key, deepgram_model)
        elif provider_name == "mms-tts":
            model = get_setting(workspace, "mmsTtsModel", "facebook/mms-tts-eng")
            hf_token = _get_hf_token()
            result = await _tts_mms(clean, model, hf_token)
            if result:
                return result
            logger.warning("MMS-TTS failed, falling back to Deepgram")
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


async def _tts_mimo_audio(text: str, endpoint: str, emotion: str = "neutral", clone_path: str | None = None) -> bytes | None:
    """MiMo-Audio TTS via Modal endpoint."""
    if not endpoint:
        return None
    try:
        payload: dict[str, Any] = {"mode": "tts", "text": text, "emotion": emotion}
        if clone_path and os.path.exists(clone_path):
            with open(clone_path, "rb") as f:
                payload["mode"] = "tts"
                payload["voice_sample_b64"] = base64.b64encode(f.read()).decode()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            audio_b64 = data.get("audio_b64")
            if audio_b64:
                return base64.b64decode(audio_b64)
    except Exception as e:
        logger.error("MiMo-Audio TTS failed: {}", e)
    return None


async def _tts_coqui(text: str, endpoint: str, language: str = "en", clone_path: str | None = None) -> bytes | None:
    """Coqui XTTS v2 via Modal endpoint."""
    if not endpoint:
        return None
    try:
        payload: dict[str, Any] = {"mode": "xtts", "text": text, "language": language}
        if clone_path and os.path.exists(clone_path):
            with open(clone_path, "rb") as f:
                payload["voice_sample_b64"] = base64.b64encode(f.read()).decode()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            audio_b64 = data.get("audio_b64")
            if audio_b64:
                return base64.b64decode(audio_b64)
    except Exception as e:
        logger.error("Coqui XTTS TTS failed: {}", e)
    return None


async def _tts_mms(text: str, model: str = "facebook/mms-tts-eng", hf_token: str | None = None) -> bytes | None:
    """Meta MMS-TTS via HuggingFace Inference API."""
    try:
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://router.huggingface.co/hf-inference/models/{model}",
                headers=headers,
                json={"inputs": text},
            )
            resp.raise_for_status()
            audio = resp.content
            return audio if len(audio) > 100 else None
    except Exception as e:
        logger.error("MMS-TTS failed: {}", e)
    return None
