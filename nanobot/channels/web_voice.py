"""Web Voice channel — browser-based voice interface with Deepgram STT/TTS.

Serves a web dashboard where users speak via their browser mic.
Audio flow:
  Browser mic → WebSocket → Deepgram live STT → InboundMessage → Agent Loop
  → OutboundMessage → Deepgram REST TTS → WebSocket → Browser playback

Features:
  - Streaming TTS: sentences are spoken as they arrive, not after the full response
  - Activity feed: shows tool calls, tool results, and reasoning in real time
  - Deepgram text intelligence: sentiment, intents, topics on each utterance
  - Message queuing: handles multiple messages gracefully
  - HTTPS via Tailscale certs for mic access
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import os
import re
import ssl
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import aiohttp
from aiohttp import web
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base

# Tailscale CGNAT range
_TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
_LOCALHOST = {ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")}

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|(?<=\n)')

# Phrases that should interrupt/cancel current work
_INTERRUPT_PATTERNS = re.compile(
    r"^(no[,.]?\s|stop[,.]?\s|wait[,.]?\s|cancel|nevermind|never mind|hold on|don'?t|not that)"
    r"|"
    r"(^no\.?$|^stop\.?$|^wait\.?$|^cancel\.?$)",
    re.IGNORECASE,
)


def _is_tailscale_or_local(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip in _LOCALHOST or ip in _TAILSCALE_NET
    except ValueError:
        return False


def _get_tailscale_dns() -> str | None:
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            import json as _json
            data = _json.loads(out.stdout)
            dns = data.get("Self", {}).get("DNSName", "").rstrip(".")
            return dns if dns else None
    except Exception:
        pass
    return None


def _ensure_tailscale_cert(dns_name: str) -> tuple[Path, Path] | None:
    cert_dir = Path.home() / ".nanobot" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / f"{dns_name}.crt"
    key_path = cert_dir / f"{dns_name}.key"

    if cert_path.exists() and key_path.exists():
        age = time.time() - cert_path.stat().st_mtime
        if age < 30 * 86400:
            return cert_path, key_path

    try:
        subprocess.run(
            ["tailscale", "cert",
             "--cert-file", str(cert_path),
             "--key-file", str(key_path),
             dns_name],
            capture_output=True, text=True, timeout=15, check=True,
        )
        logger.info("Web Voice: generated Tailscale HTTPS cert for {}", dns_name)
        return cert_path, key_path
    except Exception as e:
        logger.warning("Web Voice: failed to generate Tailscale cert: {}", e)
        return None


# ── Smart endpointing: semantic sentence completion check ──

# Words that signal an incomplete sentence when they appear at the end
_INCOMPLETE_ENDINGS = re.compile(
    r"\b(?:is|are|was|were|the|a|an|and|or|but|if|to|for|in|on|at|of|with|"
    r"my|your|his|her|its|our|their|this|that|these|those|"
    r"can|could|would|should|will|shall|may|might|must|"
    r"do|does|did|have|has|had|"
    r"not|no|any|some|every|"
    r"about|from|into|by|than|"
    r"there|here|where|when|what|which|who|whom|how|why|"
    r"also|just|even|still|already|"
    r"I|you|he|she|it|we|they)\s*$",
    re.I,
)

# Sentence-ending punctuation
_COMPLETE_ENDINGS = re.compile(r"[.!?;:)\"]\s*$")

# Very short utterances are likely incomplete
_MIN_COMPLETE_LENGTH = 15


def _is_sentence_complete(text: str) -> bool:
    """Check if text looks like a complete sentence.

    Returns True if the sentence appears complete (should submit immediately).
    Returns False if it appears incomplete (should wait for more speech).
    """
    text = text.strip()
    if not text:
        return False

    # Very short → probably incomplete
    if len(text) < _MIN_COMPLETE_LENGTH:
        return False

    # Ends with punctuation → likely complete
    if _COMPLETE_ENDINGS.search(text):
        return True

    # Ends with an incomplete word → definitely incomplete
    if _INCOMPLETE_ENDINGS.search(text):
        return False

    # If it got past the incomplete check and is reasonably long → complete
    if len(text) > 25:
        return True

    # Short, no punctuation, no obvious incomplete ending → wait
    return False


def _strip_markdown(text: str, keep_reasoning: bool = False) -> str:
    # Strip thinking/reasoning blocks unless user wants them spoken
    if not keep_reasoning:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text)
        text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-|: ]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~`#>\[\]()]", "", text)
    text = re.sub(r"\n{2,}", ". ", text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class WebVoiceConfig(Base):
    enabled: bool = False
    deepgram_api_key: str = ""
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    host: str = "0.0.0.0"
    port: int = 8765
    tailscale_only: bool = False
    tts_model: str = "aura-2-luna-en"
    stt_model: str = "nova-3"
    app_name: str = "Mawa"  # Display name for the assistant


_INTELLIGENCE_DEFAULTS = {
    "smartErrorRecovery": True,
    "intentTracking": True,
    "dynamicContextBudget": True,
    "responseQualityGate": True,
    "mcpAutoReconnect": True,
}


def _load_intelligence_settings(workspace: Path) -> dict:
    """Load intelligence settings from workspace/intelligence.json."""
    path = workspace / "intelligence.json"
    if path.exists():
        try:
            return {**_INTELLIGENCE_DEFAULTS, **json.loads(path.read_text())}
        except Exception:
            pass
    return dict(_INTELLIGENCE_DEFAULTS)


def _save_intelligence_settings(workspace: Path, settings: dict) -> None:
    """Save intelligence settings to workspace/intelligence.json."""
    path = workspace / "intelligence.json"
    # Only save known keys
    clean = {k: bool(v) for k, v in settings.items() if k in _INTELLIGENCE_DEFAULTS}
    path.write_text(json.dumps(clean, indent=2))


class WebVoiceChannel(BaseChannel):
    """Web-based voice channel using browser mic + Deepgram STT/TTS."""

    name = "web_voice"
    display_name = "Web Voice"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WebVoiceConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WebVoiceConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WebVoiceConfig = config
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._clients: dict[str, web.WebSocketResponse] = {}
        self._utterance_buffer: dict[str, list[str]] = {}
        self._streamed_text: dict[str, str] = {}
        # Per-session TTS queue to preserve sentence order
        self._tts_queues: dict[str, asyncio.Queue] = {}
        self._tts_workers: dict[str, asyncio.Task] = {}
        # Track pending count per session for UI
        self._pending_count: dict[str, int] = {}
        # Latency tracking per session
        self._activity_ts: dict[str, float] = {}  # last activity timestamp
        # Detected language from multi-language STT (auto-switches TTS)
        self._detected_language: str | None = None

    @staticmethod
    def _get_workspace():
        from nanobot.config.paths import get_workspace_path
        return get_workspace_path()

    async def start(self) -> None:
        if not self.config.deepgram_api_key:
            logger.error("Web Voice: deepgram_api_key not configured")
            return

        self._running = True
        self._app = web.Application()
        if self.config.tailscale_only:
            self._app.middlewares.append(self._tailscale_middleware)

        self._app.router.add_get("/ws", self._ws_handler)
        self._app.router.add_get("/health", self._health_handler)
        self._app.router.add_get("/api/profiles", self._profiles_handler)
        self._app.router.add_get("/api/config", self._config_handler)
        self._app.router.add_post("/api/config", self._config_update_handler)
        self._app.router.add_get("/api/memory", self._memory_handler)
        self._app.router.add_post("/api/memory/clear-short-term", self._memory_clear_short_term_handler)
        self._app.router.add_post("/api/memory/consolidate", self._memory_consolidate_handler)
        self._app.router.add_get("/api/memory/timeline", self._memory_timeline_handler)
        self._app.router.add_post("/api/events", self._events_handler)
        self._app.router.add_get("/api/goals", self._goals_handler)
        self._app.router.add_post("/api/goals", self._goals_update_handler)
        self._app.router.add_get("/api/cron", self._cron_handler)
        self._app.router.add_get("/api/activity", self._activity_handler)
        self._app.router.add_post("/api/memory/search", self._memory_search_handler)
        self._app.router.add_get("/api/memory/export", self._memory_export_handler)
        self._app.router.add_get("/api/tools", self._tools_handler)
        self._app.router.add_get("/api/files/{path:.*}", self._file_download_handler)
        self._app.router.add_post("/api/inbox/upload", self._inbox_upload_handler)
        self._app.router.add_get("/api/generated", self._generated_list_handler)
        self._app.router.add_post("/api/generated/cleanup", self._generated_cleanup_handler)
        self._app.router.add_get("/api/inbox", self._inbox_list_handler)
        self._app.router.add_post("/api/reaction", self._reaction_handler)
        self._app.router.add_get("/api/autonomy", self._autonomy_handler)
        self._app.router.add_get("/api/intelligence", self._intelligence_get_handler)
        self._app.router.add_get("/api/learnings", self._learnings_handler)
        self._app.router.add_get("/api/notifications", self._notifications_handler)
        self._app.router.add_post("/api/notifications/read", self._notifications_read_handler)
        self._app.router.add_post("/api/intelligence", self._intelligence_set_handler)
        self._app.router.add_get("/api/skills/search", self._skills_search_handler)
        self._app.router.add_post("/api/skills/install", self._skills_install_handler)
        self._app.router.add_get("/api/skills/installed", self._skills_installed_handler)
        self._app.router.add_post("/api/skills/remove", self._skills_remove_handler)
        self._app.router.add_get("/api/search", self._search_handler)
        self._app.router.add_get("/api/sessions", self._sessions_list_handler)
        self._app.router.add_post("/api/sessions/switch", self._sessions_switch_handler)
        self._app.router.add_get("/api/suggestions", self._suggestions_handler)
        self._app.router.add_get("/api/shortcut/{action}", self._shortcut_download_handler)
        self._app.router.add_get("/api/mcp-servers", self._mcp_servers_list_handler)
        self._app.router.add_post("/api/mcp-servers", self._mcp_servers_save_handler)
        self._app.router.add_delete("/api/mcp-servers/{name}", self._mcp_servers_delete_handler)
        self._app.router.add_get("/api/credentials", self._credentials_list_handler)
        self._app.router.add_post("/api/credentials", self._credentials_update_handler)
        self._app.router.add_get("/api/usage", self._usage_handler)
        # Code features (all 15 — zero LLM)
        self._app.router.add_get("/api/health/dashboard", self._health_dashboard_handler)
        self._app.router.add_get("/api/suggestions/predictive", self._predictive_suggestions_handler)
        self._app.router.add_post("/api/cleanup", self._cleanup_handler)
        self._app.router.add_get("/api/sessions/search", self._session_search_handler)
        self._app.router.add_get("/api/cron/dashboard", self._cron_dashboard_handler)
        self._app.router.add_get("/api/anomalies", self._anomaly_handler)
        self._app.router.add_get("/api/tools/favorites", self._tool_favorites_handler)
        self._app.router.add_get("/api/schedule/templates", self._schedule_templates_handler)
        self._app.router.add_get("/api/rules", self._rules_list_handler)
        self._app.router.add_post("/api/rules", self._rules_save_handler)
        self._app.router.add_get("/api/sessions/tags", self._session_tags_handler)
        self._app.router.add_post("/api/sessions/tags", self._session_tags_set_handler)
        self._app.router.add_get("/api/inbox/batch", self._inbox_batch_handler)
        self._app.router.add_get("/api/tools/alternatives", self._tool_alternatives_handler)
        self._app.router.add_post("/api/snapshot", self._snapshot_handler)
        self._app.router.add_get("/api/snapshot/diff", self._snapshot_diff_handler)
        self._app.router.add_get("/api/sessions/health", self._session_health_handler)
        self._app.router.add_get("/api/contacts", self._contacts_handler)
        self._app.router.add_post("/api/contacts", self._contacts_save_handler)
        self._app.router.add_get("/api/habits", self._habits_handler)
        self._app.router.add_post("/api/habits", self._habits_save_handler)
        self._app.router.add_delete("/api/habits/{name}", self._habits_delete_handler)
        self._app.router.add_get("/api/quiet-hours", self._quiet_hours_handler)
        self._app.router.add_post("/api/quiet-hours", self._quiet_hours_save_handler)
        self._app.router.add_get("/api/export/conversation", self._export_conversation_handler)
        self._app.router.add_get("/api/undo", self._undo_handler)
        self._app.router.add_post("/api/maintenance", self._maintenance_handler)
        # Jarvis intelligence
        self._app.router.add_get("/api/jarvis/settings", self._jarvis_settings_get_handler)
        self._app.router.add_post("/api/jarvis/settings", self._jarvis_settings_save_handler)
        self._app.router.add_get("/api/jarvis/morning-prep", self._jarvis_morning_prep_handler)
        self._app.router.add_get("/api/jarvis/digest", self._jarvis_digest_handler)
        self._app.router.add_get("/api/jarvis/dashboard", self._jarvis_dashboard_handler)
        self._app.router.add_get("/api/jarvis/correlations", self._jarvis_correlations_handler)
        self._app.router.add_get("/api/jarvis/relationships", self._jarvis_relationships_handler)
        self._app.router.add_get("/api/jarvis/financial", self._jarvis_financial_handler)
        self._app.router.add_get("/api/jarvis/routines", self._jarvis_routines_handler)
        self._app.router.add_get("/api/projects", self._projects_handler)
        self._app.router.add_post("/api/projects", self._projects_save_handler)
        self._app.router.add_get("/api/delegations", self._delegations_handler)
        self._app.router.add_post("/api/delegations", self._delegations_save_handler)
        self._app.router.add_get("/api/decisions", self._decisions_handler)
        self._app.router.add_get("/api/people-prep", self._people_prep_handler)
        # Voice providers
        self._app.router.add_get("/api/voice/providers", self._voice_providers_handler)
        self._app.router.add_get("/api/voice/voices", self._voice_voices_handler)
        self._app.router.add_post("/api/voice/validate", self._voice_validate_handler)
        self._app.router.add_get("/api/voice/samples", self._voice_samples_handler)
        self._app.router.add_post("/api/voice/samples", self._voice_sample_upload_handler)
        self._app.router.add_delete("/api/voice/samples/{name}", self._voice_sample_delete_handler)
        self._app.router.add_get("/api/features", self._features_manifest_handler)
        self._app.router.add_post("/api/features", self._features_save_handler)
        self._app.router.add_post("/api/budget", self._budget_save_handler)
        self._app.router.add_post("/api/behavior", self._behavior_save_handler)
        self._app.router.add_post("/api/maintenance/settings", self._maintenance_settings_handler)
        self._app.router.add_get("/api/search", self._search_handler)
        self._app.router.add_get("/api/sessions", self._sessions_list_handler)
        self._app.router.add_get("/api/sessions/{key:.+}/export", self._session_export_handler)
        self._app.router.add_get("/api/sessions/{key:.+}", self._session_detail_handler)
        self._app.router.add_delete("/api/sessions/{key:.+}", self._session_delete_handler)
        self._app.router.add_get("/api/budget", self._budget_handler)
        self._app.router.add_post("/api/budget", self._budget_update_handler)
        self._app.router.add_get("/api/favorites", self._favorites_handler)
        # Serve React build from /root/mawabot/dist (if exists), fallback to inline HTML
        _react_dist = Path("/root/mawabot/dist")
        if _react_dist.exists():
            self._app.router.add_static("/assets", _react_dist / "assets")
            self._app.router.add_get("/manifest.json", lambda r: web.FileResponse(_react_dist / "manifest.json") if (_react_dist / "manifest.json").exists() else self._pwa_asset_handler(r))
            # SPA fallback: all non-API/non-WS routes serve index.html
            self._app.router.add_get("/{path:.*}", self._spa_handler)
        else:
            self._app.router.add_get("/", self._index_handler)
            self._app.router.add_get("/manifest.json", self._pwa_asset_handler)
            self._app.router.add_get("/sw.js", self._pwa_asset_handler)
            self._app.router.add_get("/icon-192.png", self._pwa_asset_handler)
            self._app.router.add_get("/icon-512.png", self._pwa_asset_handler)

        ssl_ctx = None
        ts_dns = _get_tailscale_dns() if self.config.tailscale_only else None
        if ts_dns:
            certs = _ensure_tailscale_cert(ts_dns)
            if certs:
                ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_ctx.load_cert_chain(str(certs[0]), str(certs[1]))
                self._tailscale_dns = ts_dns

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port, ssl_context=ssl_ctx)
        await site.start()

        proto = "https" if ssl_ctx else "http"
        host_display = ts_dns or self.config.host
        logger.info("Web Voice dashboard running on {}://{}:{}", proto, host_display, self.config.port)

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        for ws in list(self._clients.values()):
            if not ws.closed:
                await ws.close()
        self._clients.clear()
        for task in self._tts_workers.values():
            task.cancel()
        self._tts_workers.clear()
        if self._runner:
            await self._runner.cleanup()

    async def send(self, msg: OutboundMessage) -> None:
        """Receive agent output -> stream activity + chunked TTS to browser.

        Broadcasts to ALL connected web voice clients so every device
        (PC + phone) sees responses and gets TTS audio.
        """
        if not msg.content:
            return

        # Collect all live client WebSockets
        live_clients: list[tuple[str, web.WebSocketResponse]] = [
            (cid, cws) for cid, cws in self._clients.items() if not cws.closed
        ]
        if not live_clients:
            return

        # Pick first live client as reference for session_id (for TTS queue etc.)
        session_id = live_clients[0][0]

        meta = msg.metadata or {}
        is_progress = meta.get("_progress", False)
        is_tool_hint = meta.get("_tool_hint", False)

        # Broadcast helper — send JSON to all connected clients
        async def broadcast(data: dict) -> None:
            for _, cws in live_clients:
                if not cws.closed:
                    try:
                        await cws.send_json(data)
                    except Exception:
                        pass

        # Parallel task spawned — resolve pending bubble, show in activity
        if meta.get("_parallel"):
            await broadcast({"type": "parallel", "text": msg.content})
            return

        # Streaming TTS sentence — enqueue for TTS + send text chunk for live display
        if meta.get("_tts_sentence"):
            await broadcast({"type": "response_chunk", "text": msg.content})
            for cid, _ in live_clients:
                self._enqueue_tts(cid, msg.content)
            # Mark that this session had streaming TTS so final response skips TTS
            self._streamed_text[session_id] = True
            return

        # Proactive notification — send with notification type for browser push
        if meta.get("_notification") or meta.get("_proactive") or meta.get("_briefing"):
            payload = {
                "type": "notification",
                "text": msg.content,
                "priority": meta.get("_priority", "normal"),
                "proactive": bool(meta.get("_proactive")),
                "briefing": bool(meta.get("_briefing")),
                "cron": bool(meta.get("_cron")),
                "job_id": meta.get("_job_id", ""),
            }
            # Always persist notification so it's not lost if offline
            from nanobot.hooks.builtin.notification_store import save_notification
            save_notification(self._get_workspace(), msg.content, meta)

            # Respect quiet hours — defer non-urgent notifications
            from nanobot.hooks.builtin.maintenance import should_send_notification
            priority = meta.get("_priority", "normal")
            if not should_send_notification(self._get_workspace(), priority):
                logger.info("Notification deferred (quiet hours): {}", msg.content[:60])
                return  # Saved but not sent — will be delivered on next connect

            if live_clients:
                await broadcast(payload)
            else:
                logger.info("Notification saved (no clients connected): {}", msg.content[:80])
            return

        # Structured tool result — send data for generative UI rendering
        if meta.get("_tool_result"):
            await broadcast({
                "type": "tool_result",
                "tool": meta.get("_tool_name", ""),
                "data": meta.get("_tool_data", {}),
                "summary": msg.content,
            })
            return

        # Token usage — send as activity entry
        if meta.get("_token_usage"):
            usage_data = meta.get("_usage_data", {})
            await broadcast({
                "type": "activity",
                "kind": "tokens",
                "text": msg.content,
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            })
            return

        if is_progress:
            now = time.time()
            prev = self._activity_ts.get(session_id, now)
            delta_ms = round((now - prev) * 1000)
            self._activity_ts[session_id] = now

            if is_tool_hint:
                text = msg.content
                if text.startswith("[") and "] \u2192 " in text:
                    await broadcast({"type": "activity", "kind": "result", "text": text, "latency_ms": delta_ms})
                else:
                    await broadcast({"type": "activity", "kind": "tool", "text": text, "latency_ms": delta_ms})
            else:
                await broadcast({"type": "activity", "kind": "thinking", "text": msg.content, "latency_ms": delta_ms})
            return

        # Final response
        is_subagent = meta.get("_subagent_result", False)
        _was_streamed = session_id in self._streamed_text  # Check BEFORE popping
        self._streamed_text.pop(session_id, None)
        await broadcast({
            "type": "response_text",
            "text": msg.content,
            "subagent_result": is_subagent,
        })

        # Detect profile switch and notify frontend
        if msg.content.startswith("Switched to profile:"):
            profile_name = self._get_active_profile_name()
            await broadcast({"type": "profile_switched", "name": profile_name})

        # Update pending count
        count = self._pending_count.get(session_id, 0)
        if count > 0:
            self._pending_count[session_id] = count - 1
            remaining = count - 1
            if remaining > 0:
                await broadcast({"type": "queue_status", "pending": remaining})

        # TTS: only for responses that were NOT already streamed and NOT flagged as voice_final.
        # Streaming path handles TTS per-sentence via _tts_sentence metadata.
        # voice_final is the display-only message sent after streaming completes.
        if not _was_streamed and not meta.get("_voice_final"):
            _keep_reasoning = getattr(self, '_speak_reasoning', {}).get(session_id, False)
            clean = _strip_markdown(msg.content, keep_reasoning=_keep_reasoning)
            if clean:
                sentences = _split_sentences(clean)
                for sentence in sentences:
                    if len(sentence) >= 3:
                        self._enqueue_tts(session_id, sentence)

    def _enqueue_tts(self, session_id: str, text: str) -> None:
        """Add a sentence to the ordered TTS queue for this session."""
        if session_id not in self._tts_queues:
            self._tts_queues[session_id] = asyncio.Queue()
        self._tts_queues[session_id].put_nowait(text)
        if session_id not in self._tts_workers or self._tts_workers[session_id].done():
            self._tts_workers[session_id] = asyncio.create_task(self._tts_worker(session_id))

    async def _tts_worker(self, session_id: str) -> None:
        """Process TTS queue sequentially to preserve sentence order."""
        q = self._tts_queues.get(session_id)
        if not q:
            return
        while True:
            try:
                text = await asyncio.wait_for(q.get(), timeout=30.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break
            ws = self._clients.get(session_id)
            if not ws or ws.closed:
                break
            audio = await self._generate_tts(text)
            if audio and not ws.closed:
                await ws.send_json({
                    "type": "tts_audio",
                    "audio_b64": base64.b64encode(audio).decode(),
                    "sample_rate": 24000,
                })

    # ── Middleware ──────────────────────────────────────────────────

    @web.middleware
    async def _tailscale_middleware(self, request: web.Request, handler):
        peer = request.remote or ""
        if not _is_tailscale_or_local(peer):
            return web.Response(status=403, text="Access denied. Tailscale devices only.")
        return await handler(request)

    # ── HTTP handlers ──────────────────────────────────────────────

    async def _index_handler(self, request: web.Request) -> web.Response:
        return web.Response(text=_load_dashboard_html(), content_type="text/html")

    async def _spa_handler(self, request: web.Request) -> web.Response:
        """Serve React SPA — all routes get index.html (client-side routing)."""
        dist = Path("/root/mawabot/dist")
        return web.FileResponse(
            dist / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    async def _pwa_asset_handler(self, request: web.Request) -> web.Response:
        """Serve PWA assets (manifest, service worker, icons)."""
        filename = request.path.lstrip("/")
        filepath = _UI_DIR / filename
        if not filepath.exists():
            return web.Response(status=404, text="Not found")
        content_types = {
            ".json": "application/json",
            ".js": "application/javascript",
            ".png": "image/png",
        }
        ct = content_types.get(filepath.suffix, "application/octet-stream")
        if filepath.suffix == ".png":
            return web.Response(body=filepath.read_bytes(), content_type=ct)
        return web.Response(text=filepath.read_text(), content_type=ct)

    async def _health_handler(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "active_sessions": len(self._clients),
        })

    async def _config_handler(self, request: web.Request) -> web.Response:
        """Return full safe config for the settings UI."""
        try:
            import json as _json
            from nanobot.config.loader import load_config
            config = load_config()
            profiles = {}
            for name, p in (config.profiles or {}).items():
                profiles[name] = {"provider": p.provider or "—", "model": p.model or "—"}

            # Memory layer stats
            from nanobot.config.paths import get_workspace_path
            mem_dir = get_workspace_path() / "memory"
            memory_info = {}
            for fname in ("SHORT_TERM.md", "LONG_TERM.md", "OBSERVATIONS.md", "EPISODES.md", "HISTORY.md"):
                fpath = mem_dir / fname
                memory_info[fname] = {
                    "exists": fpath.exists(),
                    "sizeKB": round(fpath.stat().st_size / 1024, 1) if fpath.exists() else 0,
                }

            return web.json_response({
                "profiles": profiles,
                "activeProfile": self._get_active_profile_name(),
                "voice": {
                    "ttsModel": self.config.tts_model,
                    "sttModel": self.config.stt_model,
                    "deepgramConfigured": bool(self.config.deepgram_api_key),
                },
                "connection": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "tailscaleOnly": self.config.tailscale_only,
                },
                "memory": memory_info,
            })
        except Exception as e:
            logger.warning("API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    # Allowlisted voice models for input validation
    _VALID_TTS = frozenset({
        "aura-2-luna-en", "aura-2-asteria-en", "aura-2-zeus-en",
        "aura-2-orion-en", "aura-2-stella-en", "aura-2-hera-en",
        "aura-asteria-en", "aura-luna-en", "aura-zeus-en", "aura-orion-en",
    })
    _VALID_STT = frozenset({"nova-3", "nova-2", "nova", "enhanced", "base"})

    async def _config_update_handler(self, request: web.Request) -> web.Response:
        """Update config values from the settings UI. Only whitelisted fields accepted."""
        try:
            import json as _json
            body = await request.json()
            config_path = Path("/root/.nanobot/config.json")
            cfg = _json.loads(config_path.read_text())

            changed = []
            # Update voice settings — validated against allowlist
            if "ttsModel" in body and body["ttsModel"] in self._VALID_TTS:
                cfg.setdefault("channels", {}).setdefault("web_voice", {})["ttsModel"] = body["ttsModel"]
                self.config.tts_model = body["ttsModel"]
                changed.append("ttsModel")
            if "sttModel" in body and body["sttModel"] in self._VALID_STT:
                cfg.setdefault("channels", {}).setdefault("web_voice", {})["sttModel"] = body["sttModel"]
                self.config.stt_model = body["sttModel"]
                changed.append("sttModel")

            if changed:
                config_path.write_text(_json.dumps(cfg, indent=2))

            return web.json_response({"ok": True, "changed": changed})
        except Exception as e:
            logger.warning("API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _memory_handler(self, request: web.Request) -> web.Response:
        """Return memory layer stats for the settings UI."""
        try:
            from nanobot.config.paths import get_workspace_path
            mem_dir = get_workspace_path() / "memory"

            def _file_stats(name: str) -> dict:
                path = mem_dir / name
                if not path.exists():
                    return {"exists": False, "sizeKB": 0, "lines": 0, "preview": ""}
                content = path.read_text(encoding="utf-8")
                lines = [l for l in content.split("\n") if l.strip()]
                return {
                    "exists": True,
                    "sizeKB": round(path.stat().st_size / 1024, 1),
                    "lines": len(lines),
                    "preview": content.strip()[:200],
                }

            # Tool scores
            tool_scores = {}
            scores_path = mem_dir / "tool_scores.json"
            if scores_path.exists():
                try:
                    import json as _j
                    raw = _j.loads(scores_path.read_text(encoding="utf-8"))
                    for tool, data in raw.items():
                        total = data.get("success", 0) + data.get("fail", 0)
                        if total > 0:
                            tool_scores[tool] = {
                                "total": total,
                                "success": data["success"],
                                "fail": data["fail"],
                                "successRate": round(data["success"] / total * 100, 1),
                                "avgDurationMs": round(data.get("total_duration_ms", 0) / total),
                                "lastUsed": data.get("last_used", ""),
                            }
                except Exception:
                    pass

            return web.json_response({
                "shortTerm": _file_stats("SHORT_TERM.md"),
                "longTerm": _file_stats("LONG_TERM.md"),
                "observations": _file_stats("OBSERVATIONS.md"),
                "episodes": _file_stats("EPISODES.md"),
                "history": _file_stats("HISTORY.md"),
                "learnings": _file_stats("LEARNINGS.md"),
                "goals": _file_stats("GOALS.md"),
                "media": _file_stats("MEDIA.md"),
                "corrections": _file_stats("CORRECTIONS.md"),
                "toolScores": tool_scores,
            })
        except Exception as e:
            logger.warning("Memory API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _memory_clear_short_term_handler(self, request: web.Request) -> web.Response:
        """Clear SHORT_TERM.md — consolidates via LLM first, then archives to HISTORY.md."""
        try:
            workspace = self._get_workspace()
            from nanobot.agent.memory import MemoryStore
            store = MemoryStore(workspace)

            # Read current short-term content before clearing
            short_term = store.read_short_term().strip()
            if short_term:
                # Consolidate short-term content through LLM before archiving
                try:
                    from nanobot.providers.litellm_provider import LiteLLMProvider
                    provider = LiteLLMProvider()
                    model = provider.get_default_model()
                    messages = [{"role": "user", "content": short_term, "timestamp": "today"}]
                    await store.consolidate(messages, provider, model)
                    logger.info("Memory clear: consolidated short-term via LLM before clearing")
                except Exception as e:
                    logger.warning("Memory clear: LLM consolidation failed ({}), falling back to raw archive", e)

            store.daily_cleanup()
            return web.json_response({"ok": True, "consolidated": bool(short_term)})
        except Exception as e:
            logger.warning("Memory clear error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _memory_consolidate_handler(self, request: web.Request) -> web.Response:
        """POST /api/memory/consolidate — manually trigger LLM consolidation of session history."""
        try:
            workspace = self._get_workspace()

            # Find active session and consolidate unconsolidated messages
            from nanobot.session.manager import SessionManager
            sessions = SessionManager(workspace)
            # Use the web_voice:voice session (the active one)
            session = sessions.get_or_create("web_voice:voice")
            unconsolidated = session.messages[session.last_consolidated:]

            if not unconsolidated:
                return web.json_response({"ok": True, "consolidated": 0, "message": "Nothing to consolidate"})

            from nanobot.agent.memory import MemoryStore
            from nanobot.providers.litellm_provider import LiteLLMProvider
            store = MemoryStore(workspace)
            provider = LiteLLMProvider()
            model = provider.get_default_model()

            result = await store.consolidate(unconsolidated, provider, model)
            if result:
                session.last_consolidated = len(session.messages)
                sessions.save(session)
                return web.json_response({
                    "ok": True,
                    "consolidated": len(unconsolidated),
                    "message": f"Consolidated {len(unconsolidated)} messages into memory layers",
                })

            return web.json_response({"ok": False, "message": "Consolidation failed"}, status=500)
        except Exception as e:
            logger.warning("Memory consolidate error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _memory_timeline_handler(self, request: web.Request) -> web.Response:
        """GET /api/memory/timeline — return memory activity events for timeline visualization."""
        try:
            workspace = self._get_workspace()
            mem_dir = workspace / "memory"
            events = []

            # Parse HISTORY.md for timestamped events
            history = mem_dir / "HISTORY.md"
            if history.exists():
                import re
                for line in history.read_text().split("\n"):
                    m = re.match(r"\[(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\](.+)", line.strip())
                    if m:
                        events.append({
                            "ts": m.group(1).strip(),
                            "type": "history",
                            "text": m.group(2).strip()[:150],
                        })

            # Parse LEARNINGS.md
            learnings = mem_dir / "LEARNINGS.md"
            if learnings.exists():
                import re
                for line in learnings.read_text().split("\n"):
                    m = re.search(r"\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]", line)
                    if m:
                        text = line.strip().lstrip("- ").split("[")[0].strip()
                        events.append({"ts": m.group(1), "type": "learning", "text": text[:150]})

            # Parse EPISODES.md
            episodes = mem_dir / "EPISODES.md"
            if episodes.exists():
                import re
                for line in episodes.read_text().split("\n"):
                    m = re.search(r"\[(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\]", line)
                    if m:
                        text = line.strip().lstrip("- ").split("[")[0].strip()
                        events.append({"ts": m.group(1), "type": "episode", "text": text[:150]})

            # Parse FEEDBACK.md
            feedback = mem_dir / "FEEDBACK.md"
            if feedback.exists():
                import re
                for line in feedback.read_text().split("\n"):
                    m = re.search(r"\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]", line)
                    if m:
                        text = line.strip().lstrip("- ").split("[")[0].strip()
                        events.append({"ts": m.group(1), "type": "feedback", "text": text[:100]})

            # Sort by timestamp descending, limit to 50
            events.sort(key=lambda e: e["ts"], reverse=True)
            return web.json_response({"events": events[:50]})
        except Exception as e:
            logger.warning("Memory timeline error: {}", e)
            return web.json_response({"events": []})

    async def _events_handler(self, request: web.Request) -> web.Response:
        """Ingest external events via webhook. POST /api/events."""
        try:
            from nanobot.hooks.builtin.events import parse_event, format_event_as_message, validate_signature

            body = await request.json()

            # Optional HMAC signature verification
            sig = request.headers.get("X-Nanobot-Signature", "")
            if sig:
                import os
                secret = os.environ.get("NANOBOT_WEBHOOK_SECRET", "")
                if secret and not validate_signature(await request.read(), sig, secret):
                    return web.json_response({"error": "Invalid signature"}, status=401)

            event = parse_event(body)
            if isinstance(event, str):
                return web.json_response({"error": event}, status=400)

            # Route event to agent via message bus as a system message
            from nanobot.bus.events import InboundMessage
            msg = InboundMessage(
                channel="system",
                sender_id="webhook",
                chat_id=body.get("deliver_to", "web_voice:direct"),
                content=format_event_as_message(event),
                metadata={"_event": event},
            )
            await self.bus.publish_inbound(msg)

            # Log to HISTORY.md
            from nanobot.config.paths import get_workspace_path
            mem_dir = get_workspace_path() / "memory"
            history = mem_dir / "HISTORY.md"
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(history, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] EVENT {event['type']} from {event['source']}: {event['title']}\n\n")

            return web.json_response({"ok": True, "event_id": event["received_at"]})
        except Exception as e:
            logger.warning("Events API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _goals_handler(self, request: web.Request) -> web.Response:
        """Return goals/tasks for the settings UI."""
        try:
            from nanobot.config.paths import get_workspace_path
            goals_path = get_workspace_path() / "memory" / "GOALS.md"
            if not goals_path.exists():
                return web.json_response({"goals": [], "raw": ""})

            content = goals_path.read_text(encoding="utf-8")
            # Parse goals from markdown checkboxes
            goals = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("- [ ] "):
                    goals.append({"text": line[6:], "done": False})
                elif line.startswith("- [x] ") or line.startswith("- [X] "):
                    goals.append({"text": line[6:], "done": True})
            return web.json_response({"goals": goals, "raw": content})
        except Exception as e:
            logger.warning("Goals API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _goals_update_handler(self, request: web.Request) -> web.Response:
        """Create, complete, or remove goals. POST /api/goals."""
        try:
            from nanobot.config.paths import get_workspace_path
            from nanobot.agent.tools.goals import GoalsTool
            import asyncio

            body = await request.json()
            action = body.get("action", "")
            tool = GoalsTool(workspace=get_workspace_path())

            if action == "add":
                goal = body.get("goal", "")
                subtask = body.get("subtask", "")
                due = body.get("due", "")
                result = await tool.execute(action="add", goal=goal, subtask=subtask, due=due)
                return web.json_response({"ok": True, "result": result})

            elif action == "complete":
                index = body.get("index")
                if index is None:
                    return web.json_response({"error": "index required"}, status=400)
                result = await tool.execute(action="complete", index=int(index))
                return web.json_response({"ok": True, "result": result})

            elif action == "remove":
                index = body.get("index")
                if index is None:
                    return web.json_response({"error": "index required"}, status=400)
                result = await tool.execute(action="remove", index=int(index))
                return web.json_response({"ok": True, "result": result})

            else:
                return web.json_response({"error": f"Unknown action: {action}"}, status=400)

        except Exception as e:
            logger.warning("Goals update error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _cron_handler(self, request: web.Request) -> web.Response:
        """Return scheduled cron jobs."""
        try:
            from nanobot.config.paths import get_workspace_path
            import json as _j
            config_dir = Path("/root/.nanobot")
            jobs_file = config_dir / "cron" / "jobs.json"
            if not jobs_file.exists():
                return web.json_response({"jobs": []})
            data = _j.loads(jobs_file.read_text(encoding="utf-8"))
            jobs = []
            for job in data.get("jobs", []):
                jobs.append({
                    "id": job.get("id", ""),
                    "name": job.get("name", ""),
                    "message": job.get("message", ""),
                    "schedule": job.get("schedule", {}),
                    "enabled": job.get("enabled", True),
                    "channel": job.get("channel", ""),
                    "lastError": job.get("last_error", ""),
                })
            return web.json_response({"jobs": jobs})
        except Exception as e:
            logger.warning("Cron API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _activity_handler(self, request: web.Request) -> web.Response:
        """Return recent activity from HISTORY.md (rich) + activity.jsonl (turns)."""
        try:
            from nanobot.config.paths import get_workspace_path
            import re
            workspace = get_workspace_path()

            # Parse HISTORY.md for detailed tool calls and summaries
            history_file = workspace / "memory" / "HISTORY.md"
            tool_calls = []
            summaries = []
            if history_file.exists():
                content = history_file.read_text(encoding="utf-8")
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Tool call lines: [2026-03-20 13:05] TOOL goals OK (0ms)
                    m = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\] TOOL (\S+) (OK|ERR) \((\d+)ms\)", line)
                    if m:
                        tool_calls.append({
                            "type": "tool",
                            "timestamp": m.group(1),
                            "tool": m.group(2),
                            "status": m.group(3),
                            "duration_ms": int(m.group(4)),
                        })
                    # Summary lines: [2026-03-20 01:44-04:52] User asked...
                    elif line.startswith("[2") and "] " in line and "TOOL" not in line and "EVENT" not in line and "[RAW]" not in line:
                        bracket_end = line.index("] ")
                        ts = line[1:bracket_end]
                        text = line[bracket_end + 2:]
                        # Skip internal/noisy lines
                        if len(text) > 30:
                            summaries.append({
                                "type": "summary",
                                "timestamp": ts,
                                "text": text[:500],
                            })

            # Merge and return last 80 entries, newest first
            all_entries = tool_calls + summaries
            # Sort by timestamp string (works for ISO-like format)
            all_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

            return web.json_response({
                "entries": all_entries[:80],
                "total_tools": len(tool_calls),
                "total_summaries": len(summaries),
            })
        except Exception as e:
            logger.warning("Activity API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _memory_search_handler(self, request: web.Request) -> web.Response:
        """Search memory via local grep-based search."""
        try:
            body = await request.json()
            query = body.get("query", "").strip()
            if not query:
                return web.json_response({"error": "query required"}, status=400)

            from nanobot.config.paths import get_workspace_path
            mem_dir = get_workspace_path() / "memory"
            results = []

            if mem_dir.exists():
                import re
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                for md_file in sorted(mem_dir.rglob("*.md")):
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        for i, line in enumerate(content.split("\n")):
                            if pattern.search(line):
                                # Include surrounding context (2 lines before/after)
                                lines = content.split("\n")
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                snippet = "\n".join(lines[start:end])
                                results.append({
                                    "title": md_file.name,
                                    "content": snippet[:300],
                                    "score": 100.0,
                                    "source": str(md_file),
                                })
                                if len(results) >= 10:
                                    break
                    except Exception:
                        continue
                    if len(results) >= 10:
                        break

            return web.json_response({"results": results})
        except Exception as e:
            logger.warning("Memory search API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _memory_export_handler(self, request: web.Request) -> web.Response:
        """Export all memory files as JSON."""
        try:
            from nanobot.config.paths import get_workspace_path
            mem_dir = get_workspace_path() / "memory"
            export = {}
            if mem_dir.exists():
                for f in sorted(mem_dir.rglob("*.md")):
                    key = str(f.relative_to(mem_dir))
                    export[key] = f.read_text(encoding="utf-8")
                # Include tool_scores.json
                scores = mem_dir / "tool_scores.json"
                if scores.exists():
                    export["tool_scores.json"] = scores.read_text(encoding="utf-8")
            return web.json_response({"files": export, "count": len(export)})
        except Exception as e:
            logger.warning("Memory export error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _tools_handler(self, request: web.Request) -> web.Response:
        """Return list of available tools with metadata."""
        try:
            from nanobot.config.paths import get_workspace_path
            import json as _j
            # Read tool scores for reliability data
            mem_dir = get_workspace_path() / "memory"
            scores = {}
            scores_path = mem_dir / "tool_scores.json"
            if scores_path.exists():
                try:
                    scores = _j.loads(scores_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            tools = []

            # Dynamically list ALL registered tools from the agent loop — never hardcoded
            from nanobot.agent.tools.registry import ToolRegistry
            from nanobot.agent.tools.base import Tool

            # Get the active agent loop's tool registry if available
            registered: dict[str, Tool] = {}
            try:
                # Build a fresh registry to list all tools that WOULD be registered
                from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
                from nanobot.agent.tools.shell import ExecTool
                from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
                from nanobot.agent.tools.message import MessageTool
                from nanobot.agent.tools.memory_search import MemorySearchTool
                from nanobot.agent.tools.memory_save import MemorySaveTool
                from nanobot.agent.tools.goals import GoalsTool
                from nanobot.agent.tools.media_memory import MediaMemoryTool
                from nanobot.agent.tools.cron import CronTool
                from nanobot.agent.tools.spawn import SpawnTool
                from nanobot.agent.tools.background_shell import BackgroundShellTool
                from nanobot.agent.tools.credentials import CredentialsTool
                from nanobot.agent.tools.settings_tool import SettingsTool
                from nanobot.agent.tools.image_gen import ImageGenTool
                from nanobot.agent.tools.phone_call import PhoneCallTool
                from nanobot.agent.tools.skill_creator import SkillCreatorTool
                from nanobot.agent.tools.skills_marketplace import SkillsMarketplaceTool
                from nanobot.agent.tools.knowledge_ingest import KnowledgeIngestTool
                from nanobot.agent.tools.inbox import InboxTool
                ws = get_workspace_path()
                for cls in [ReadFileTool, WriteFileTool, EditFileTool, ListDirTool]:
                    t = cls(None, None)
                    registered[t.name] = t
                for cls_ws in [ExecTool, MemorySearchTool, MemorySaveTool, GoalsTool,
                               MediaMemoryTool, CronTool, BackgroundShellTool,
                               CredentialsTool, SettingsTool, ImageGenTool, PhoneCallTool,
                               SkillCreatorTool, SkillsMarketplaceTool, KnowledgeIngestTool, InboxTool]:
                    try:
                        t = cls_ws(ws)
                        registered[t.name] = t
                    except Exception:
                        try:
                            t = cls_ws()
                            registered[t.name] = t
                        except Exception:
                            pass
                for cls_simple in [WebSearchTool, WebFetchTool, MessageTool, SpawnTool]:
                    try:
                        t = cls_simple()
                        registered[t.name] = t
                    except Exception:
                        pass
                # GenUI tools
                try:
                    from nanobot.agent.tools.genui_tools import get_genui_tools
                    for gt in get_genui_tools(ws):
                        registered[gt.name] = gt
                except Exception:
                    pass
            except Exception as e:
                logger.debug("Tool listing fallback: {}", e)

            # Build response from registered tools
            for name, tool in registered.items():
                score_data = scores.get(name, {})
                total = score_data.get("success", 0) + score_data.get("fail", 0)
                tools.append({
                    "name": name,
                    "type": "genui" if name in ("weather","stock_chart","qr_code","run_code","compare","timeline","show_map","system_monitor") else "builtin",
                    "desc": tool.description[:80] if hasattr(tool, 'description') else "",
                    "calls": total,
                    "successRate": round(score_data["success"] / total * 100, 1) if total > 0 else None,
                    "lastUsed": score_data.get("last_used", ""),
                })

            return web.json_response({"tools": tools})
        except Exception as e:
            logger.warning("Tools API error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _file_download_handler(self, request: web.Request) -> web.Response:
        """Serve workspace + temp files for download/preview. GET /api/files/{path}."""
        try:
            from nanobot.config.paths import get_workspace_path
            import mimetypes

            rel_path = request.match_info.get("path", "")
            if not rel_path:
                return web.json_response({"error": "Path required"}, status=400)

            workspace = get_workspace_path()

            # Allow serving from workspace OR /tmp/ (generated files like work orders)
            _ALLOWED_ROOTS = [
                str(workspace.resolve()),
                "/tmp/work-orders",
                "/tmp/nanobot",
                str(Path.home() / ".nanobot"),
            ]

            # Try workspace-relative first, then absolute path
            if rel_path.startswith("/"):
                file_path = Path(rel_path).resolve()
            else:
                file_path = (workspace / rel_path).resolve()

            # Security: ensure file is within an allowed root
            file_str = str(file_path)
            if not any(file_str.startswith(root) for root in _ALLOWED_ROOTS):
                return web.json_response({"error": "Access denied"}, status=403)

            if not file_path.exists() or not file_path.is_file():
                return web.json_response({"error": "File not found"}, status=404)

            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or "application/octet-stream"

            # For viewable types, serve inline; for others, force download
            viewable = {"text/", "image/", "application/pdf", "application/json"}
            disposition = "inline" if any(content_type.startswith(v) for v in viewable) else "attachment"

            return web.FileResponse(
                file_path,
                headers={
                    "Content-Disposition": f'{disposition}; filename="{file_path.name}"',
                    "Content-Type": content_type,
                },
            )
        except Exception as e:
            logger.warning("File download error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _inbox_upload_handler(self, request: web.Request) -> web.Response:
        """Upload a file to the inbox. POST /api/inbox/upload (multipart form)."""
        try:
            from nanobot.config.paths import get_workspace_path
            from nanobot.agent.tools.inbox import VALID_FOLDERS

            reader = await request.multipart()
            folder = "general"
            saved_files = []

            async for part in reader:
                if part.name == "folder":
                    val = (await part.text()).strip()
                    if val in VALID_FOLDERS:
                        folder = val
                elif part.name == "file" and part.filename:
                    inbox_dir = get_workspace_path() / "inbox" / folder
                    inbox_dir.mkdir(parents=True, exist_ok=True)

                    # Sanitize filename
                    safe_name = part.filename.replace("/", "_").replace("\\", "_").replace("..", "_")
                    dest = inbox_dir / safe_name

                    # Write file
                    size = 0
                    with open(dest, "wb") as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > 50 * 1024 * 1024:  # 50MB limit
                                dest.unlink(missing_ok=True)
                                return web.json_response({"error": "File too large (max 50MB)"}, status=413)
                            f.write(chunk)

                    saved_files.append({"name": safe_name, "folder": folder, "size": size})
                    logger.info("Inbox upload: {}/{} ({}KB)", folder, safe_name, size // 1024)

            return web.json_response({"ok": True, "files": saved_files})
        except Exception as e:
            logger.warning("Inbox upload error: {}", e)
            return web.json_response({"error": "Upload failed"}, status=500)

    async def _inbox_list_handler(self, request: web.Request) -> web.Response:
        """List all inbox files. GET /api/inbox."""
        try:
            from nanobot.config.paths import get_workspace_path
            from nanobot.agent.tools.inbox import VALID_FOLDERS

            inbox_dir = get_workspace_path() / "inbox"
            folders: dict[str, list] = {}

            for f in sorted(VALID_FOLDERS):
                folder_path = inbox_dir / f
                files = []
                if folder_path.exists():
                    for p in sorted(folder_path.iterdir()):
                        if p.is_file() and not p.name.startswith("."):
                            files.append({
                                "name": p.name,
                                "size": p.stat().st_size,
                                "ext": p.suffix.lower(),
                                "modified": p.stat().st_mtime,
                            })
                folders[f] = files

            total = sum(len(v) for v in folders.values())
            return web.json_response({"folders": folders, "total": total})
        except Exception as e:
            logger.warning("Inbox list error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _generated_list_handler(self, request: web.Request) -> web.Response:
        """List generated files (work orders, reports, etc.)."""
        try:
            from nanobot.config.paths import get_workspace_path
            gen_dir = get_workspace_path() / "generated"
            files = []
            if gen_dir.exists():
                for sub in sorted(gen_dir.iterdir()):
                    if sub.is_dir():
                        for f in sorted(sub.iterdir()):
                            if f.is_file() and not f.name.startswith("."):
                                files.append({
                                    "name": f.name,
                                    "folder": sub.name,
                                    "size": f.stat().st_size,
                                    "modified": f.stat().st_mtime,
                                    "path": str(f),
                                })
            return web.json_response({"files": files, "total": len(files)})
        except Exception as e:
            logger.warning("Generated list error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _generated_cleanup_handler(self, request: web.Request) -> web.Response:
        """Delete old generated files. POST with optional {older_than_days: N}."""
        try:
            from nanobot.config.paths import get_workspace_path
            import time

            body = await request.json() if request.content_length else {}
            days = body.get("older_than_days", 7)
            cutoff = time.time() - (days * 86400)

            gen_dir = get_workspace_path() / "generated"
            deleted = 0
            if gen_dir.exists():
                for sub in gen_dir.iterdir():
                    if sub.is_dir():
                        for f in sub.iterdir():
                            if f.is_file() and f.stat().st_mtime < cutoff:
                                f.unlink()
                                deleted += 1

            return web.json_response({"ok": True, "deleted": deleted, "older_than_days": days})
        except Exception as e:
            logger.warning("Generated cleanup error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _reaction_handler(self, request: web.Request) -> web.Response:
        """Process emoji reaction feedback. POST /api/reaction."""
        try:
            from nanobot.config.paths import get_workspace_path
            from nanobot.config.loader import load_config

            body = await request.json()
            reaction = body.get("reaction", "")
            message_content = body.get("messageContent", "")
            message_role = body.get("messageRole", "assistant")

            if not reaction or not message_content:
                return web.json_response({"error": "reaction and messageContent required"}, status=400)

            workspace = get_workspace_path()
            config = load_config()
            provider_name = config.agents.defaults.provider
            model = config.agents.defaults.model

            # Get provider
            from nanobot.providers.litellm_provider import LiteLLMProvider
            provider = LiteLLMProvider()

            from nanobot.hooks.builtin.reaction_feedback import ReactionFeedback, REACTION_MEANINGS
            feedback = ReactionFeedback(workspace, provider, model)

            # Save the reaction
            feedback.save_reaction(message_content, reaction, message_role)

            meaning = REACTION_MEANINGS.get(reaction, {"signal": "neutral"})
            lesson = None

            if meaning["signal"] == "negative":
                # Extract lesson from negative reaction (async LLM call)
                lesson = await feedback.process_negative_reaction(message_content, reaction)
            elif meaning["signal"] == "positive":
                # Reinforce positive pattern
                feedback.process_positive_reaction(message_content, reaction)

            return web.json_response({
                "ok": True,
                "signal": meaning["signal"],
                "lesson": lesson,
            })
        except Exception as e:
            logger.warning("Reaction handler error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _skills_search_handler(self, request: web.Request) -> web.Response:
        """Search skills.sh via npx skills find. GET /api/skills/search?q=..."""
        try:
            import subprocess, re
            query = request.query.get("q", "").strip()
            if not query or len(query) < 2:
                return web.json_response({"results": []})

            result = subprocess.run(
                ["npx", "skills", "find", query],
                capture_output=True, text=True, timeout=15,
                env={**__import__("os").environ, "NO_COLOR": "1", "FORCE_COLOR": "0"},
            )
            # Strip ANSI escape codes
            ansi_re = re.compile(r'\x1b\[[0-9;]*m')
            output = ansi_re.sub("", result.stdout)

            # Parse results: "owner/repo@skill-name" and "N installs"
            results = []
            lines = output.split("\n")
            for i, line in enumerate(lines):
                # Match skill lines like "anthropics/skills@frontend-design"
                m = re.search(r"(\S+/\S+@\S+)", line)
                if m:
                    skill_id = m.group(1)
                    # Skip template/example lines
                    if "<" in skill_id or ">" in skill_id:
                        continue
                    # Extract install count from same line
                    installs_m = re.search(r"([\d.]+[KM]?)\s*installs", line)
                    installs = installs_m.group(1) if installs_m else "0"
                    # Extract URL from next line
                    url = ""
                    if i + 1 < len(lines):
                        url_m = re.search(r"(https://skills\.sh/\S+)", lines[i + 1])
                        if url_m:
                            url = url_m.group(1)
                    results.append({
                        "id": skill_id,
                        "installs": installs,
                        "url": url,
                    })

            return web.json_response({"results": results, "query": query})
        except subprocess.TimeoutExpired:
            return web.json_response({"results": [], "error": "Search timed out"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _skills_install_handler(self, request: web.Request) -> web.Response:
        """Install a skill via npx skills add. POST /api/skills/install."""
        try:
            import subprocess
            body = await request.json()
            skill_id = body.get("id", "").strip()
            if not skill_id:
                return web.json_response({"error": "id required"}, status=400)

            result = subprocess.run(
                ["npx", "skills", "add", skill_id],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return web.json_response({"ok": True, "output": result.stdout[:500]})
            else:
                return web.json_response({"error": result.stderr[:300] or result.stdout[:300]}, status=400)
        except subprocess.TimeoutExpired:
            return web.json_response({"error": "Install timed out"}, status=408)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _skills_installed_handler(self, request: web.Request) -> web.Response:
        """List installed skills via npx skills list. GET /api/skills/installed."""
        try:
            import subprocess, re
            result = subprocess.run(
                ["npx", "skills", "list"],
                capture_output=True, text=True, timeout=10,
            )
            # Parse installed skills
            skills = []
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line and not line.startswith("No ") and "@" in line:
                    skills.append({"id": line.strip()})
            return web.json_response({"skills": skills})
        except Exception as e:
            return web.json_response({"skills": [], "error": str(e)})

    async def _skills_remove_handler(self, request: web.Request) -> web.Response:
        """Remove a skill via npx skills remove. POST /api/skills/remove."""
        try:
            import subprocess
            body = await request.json()
            skill_id = body.get("id", "").strip()
            if not skill_id:
                return web.json_response({"error": "id required"}, status=400)

            result = subprocess.run(
                ["npx", "skills", "remove", skill_id],
                capture_output=True, text=True, timeout=15,
            )
            return web.json_response({"ok": True, "output": result.stdout[:500]})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _autonomy_handler(self, request: web.Request) -> web.Response:
        """Return autonomy system stats for the settings page."""
        try:
            from nanobot.config.paths import get_workspace_path
            workspace = get_workspace_path()

            # Workflow patterns
            workflow_patterns = []
            try:
                from nanobot.hooks.builtin.workflow_recorder import get_recorder
                recorder = get_recorder(workspace)
                workflow_patterns = recorder.get_top_patterns(5)
            except Exception:
                pass

            # Prompt optimization stats
            prompt_stats = {}
            try:
                from nanobot.hooks.builtin.prompt_optimizer import get_optimizer
                optimizer = get_optimizer(workspace)
                prompt_stats = optimizer.get_stats_summary()
            except Exception:
                pass

            # Skills count
            skills_dir = workspace / "skills"
            skills_count = 0
            if skills_dir.exists():
                skills_count = sum(1 for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists())

            # Knowledge digest size
            digest = workspace / "memory" / "KNOWLEDGE_DIGEST.md"
            knowledge_entries = 0
            if digest.exists():
                knowledge_entries = digest.read_text().count("## [")

            return web.json_response({
                "skills": skills_count,
                "knowledgeEntries": knowledge_entries,
                "workflowPatterns": workflow_patterns,
                "promptStats": prompt_stats,
            })
        except Exception as e:
            logger.warning("Autonomy API error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    async def _learnings_handler(self, request: web.Request) -> web.Response:
        """GET /api/learnings — return what Mawa has learned."""
        ws = self._get_workspace()
        user_file = ws / "memory" / "LEARNINGS.md"
        tool_file = ws / "memory" / "TOOL_LEARNINGS.md"

        def _parse(path):
            if not path.exists():
                return []
            return [l.strip().lstrip("- ") for l in path.read_text().split("\n") if l.strip().startswith("- ")]

        return web.json_response({
            "userLearnings": _parse(user_file),
            "toolLearnings": _parse(tool_file),
        })

    async def _notifications_handler(self, request: web.Request) -> web.Response:
        """GET /api/notifications — return all stored notifications."""
        from nanobot.hooks.builtin.notification_store import get_all
        notifications = get_all(self._get_workspace())
        return web.json_response({"notifications": notifications})

    async def _notifications_read_handler(self, request: web.Request) -> web.Response:
        """POST /api/notifications/read — mark all as read."""
        from nanobot.hooks.builtin.notification_store import mark_all_read
        count = mark_all_read(self._get_workspace())
        return web.json_response({"ok": True, "marked": count})

    # ── Code Features API (all 15 — zero LLM tokens) ──────────────────

    async def _health_dashboard_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import get_health_dashboard
        return web.json_response(get_health_dashboard(self._get_workspace()))

    async def _predictive_suggestions_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import get_predictive_suggestions
        return web.json_response({"suggestions": get_predictive_suggestions(self._get_workspace())})

    async def _cleanup_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import auto_cleanup
        result = auto_cleanup(self._get_workspace())
        return web.json_response(result)

    async def _session_search_handler(self, request: web.Request) -> web.Response:
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"results": []})
        from nanobot.hooks.builtin.code_features import search_sessions
        return web.json_response({"results": search_sessions(self._get_workspace(), q)})

    async def _cron_dashboard_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import get_cron_dashboard
        return web.json_response(get_cron_dashboard(self._get_workspace()))

    async def _anomaly_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import detect_anomalies
        return web.json_response({"anomalies": detect_anomalies(self._get_workspace())})

    async def _tool_favorites_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import get_tool_favorites
        return web.json_response({"favorites": get_tool_favorites(self._get_workspace())})

    async def _schedule_templates_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import get_schedule_templates
        return web.json_response({"templates": get_schedule_templates()})

    async def _rules_list_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import load_rules
        return web.json_response({"rules": load_rules(self._get_workspace())})

    async def _rules_save_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import save_rules
        body = await request.json()
        save_rules(self._get_workspace(), body.get("rules", []))
        return web.json_response({"ok": True})

    async def _session_tags_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import get_session_tags
        return web.json_response({"tags": get_session_tags(self._get_workspace())})

    async def _session_tags_set_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.code_features import set_session_tags
        body = await request.json()
        set_session_tags(self._get_workspace(), body.get("session_key", ""), body.get("tags", []))
        return web.json_response({"ok": True})

    async def _inbox_batch_handler(self, request: web.Request) -> web.Response:
        action = request.query.get("action", "list")
        from nanobot.hooks.builtin.code_features import batch_process_inbox
        result = await batch_process_inbox(self._get_workspace(), action)
        return web.json_response(result)

    async def _tool_alternatives_handler(self, request: web.Request) -> web.Response:
        tool = request.query.get("tool", "")
        from nanobot.hooks.builtin.claude_capabilities import get_alternative_tool
        alt = get_alternative_tool(tool) if tool else None
        return web.json_response({"tool": tool, "alternative": alt})

    async def _snapshot_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        name = body.get("name", "auto")
        from nanobot.hooks.builtin.claude_capabilities import take_snapshot
        result = take_snapshot(self._get_workspace(), name)
        return web.json_response(result)

    async def _snapshot_diff_handler(self, request: web.Request) -> web.Response:
        name = request.query.get("name", "auto")
        from nanobot.hooks.builtin.claude_capabilities import compare_snapshot
        result = compare_snapshot(self._get_workspace(), name)
        return web.json_response(result)

    async def _session_health_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.smart_responses import get_all_session_health
        return web.json_response(get_all_session_health())

    async def _contacts_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import get_contacts
        q = request.query.get("q", "")
        if q:
            from nanobot.hooks.builtin.maintenance import find_contact
            return web.json_response({"contacts": find_contact(self._get_workspace(), q)})
        return web.json_response({"contacts": get_contacts(self._get_workspace())})

    async def _contacts_save_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import save_contact
        body = await request.json()
        save_contact(self._get_workspace(), body)
        return web.json_response({"ok": True})

    async def _habits_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import get_habits, get_due_habits
        return web.json_response({"habits": get_habits(self._get_workspace()), "due": get_due_habits(self._get_workspace())})

    async def _habits_save_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import save_habit
        body = await request.json()
        save_habit(self._get_workspace(), body)
        return web.json_response({"ok": True})

    async def _habits_delete_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import delete_habit
        name = request.match_info.get("name", "")
        ok = delete_habit(self._get_workspace(), name)
        return web.json_response({"ok": ok})

    async def _quiet_hours_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import load_quiet_hours
        return web.json_response(load_quiet_hours(self._get_workspace()))

    async def _quiet_hours_save_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import save_quiet_hours
        body = await request.json()
        save_quiet_hours(self._get_workspace(), body)
        return web.json_response({"ok": True})

    async def _export_conversation_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import export_conversation
        fmt = request.query.get("format", "markdown")
        session_key = request.query.get("session", "web_voice:voice")
        content = export_conversation(self._get_workspace(), session_key, fmt)
        if fmt == "json":
            return web.json_response(json.loads(content) if content else {})
        return web.Response(text=content, content_type="text/markdown")

    async def _undo_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import get_undo_history
        return web.json_response({"actions": get_undo_history()})

    async def _maintenance_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.maintenance import run_maintenance
        results = run_maintenance(self._get_workspace())
        return web.json_response(results)

    # ── Jarvis Intelligence API ──

    async def _jarvis_settings_get_handler(self, request: web.Request) -> web.Response:
        path = self._get_workspace() / "jarvis_settings.json"
        if path.exists():
            return web.json_response(json.loads(path.read_text()))
        return web.json_response({})

    async def _jarvis_settings_save_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        path = self._get_workspace() / "jarvis_settings.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data.update(body)
        path.write_text(json.dumps(data, indent=2))
        return web.json_response({"ok": True})

    async def _jarvis_morning_prep_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import build_morning_prep, format_morning_prep
        prep = build_morning_prep(self._get_workspace())
        return web.json_response({"prep": prep, "formatted": format_morning_prep(prep)})

    async def _jarvis_digest_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import build_daily_digest, format_digest
        digest = build_daily_digest(self._get_workspace())
        return web.json_response({"digest": digest, "formatted": format_digest(digest)})

    async def _jarvis_dashboard_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import get_life_dashboard
        return web.json_response(get_life_dashboard(self._get_workspace()))

    async def _jarvis_correlations_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import detect_correlations
        return web.json_response({"correlations": detect_correlations(self._get_workspace())})

    async def _jarvis_relationships_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import get_relationship_reminders
        return web.json_response({"reminders": get_relationship_reminders(self._get_workspace())})

    async def _jarvis_financial_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import get_financial_pulse
        return web.json_response(get_financial_pulse(self._get_workspace()))

    async def _jarvis_routines_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import detect_routines
        return web.json_response({"routines": detect_routines(self._get_workspace())})

    async def _projects_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import get_projects
        return web.json_response({"projects": get_projects(self._get_workspace())})

    async def _projects_save_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import save_project
        body = await request.json()
        save_project(self._get_workspace(), body)
        return web.json_response({"ok": True})

    async def _delegations_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import get_delegations
        return web.json_response({"delegations": get_delegations(self._get_workspace())})

    async def _delegations_save_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import add_delegation
        body = await request.json()
        add_delegation(self._get_workspace(), body.get("task", ""), body.get("deadline", ""), body.get("check_interval_hours", 24))
        return web.json_response({"ok": True})

    async def _decisions_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import find_related_decisions
        q = request.query.get("q", "")
        if q:
            return web.json_response({"decisions": find_related_decisions(self._get_workspace(), q)})
        path = self._get_workspace() / "decisions.json"
        decisions = json.loads(path.read_text()) if path.exists() else []
        return web.json_response({"decisions": decisions[-20:]})

    async def _people_prep_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.jarvis import get_people_prep
        name = request.query.get("name", "")
        if not name:
            return web.json_response({"error": "name required"}, status=400)
        return web.json_response(get_people_prep(self._get_workspace(), name))

    # ── Voice Provider API ──

    async def _voice_providers_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.voice_providers import get_all_providers
        return web.json_response({"providers": get_all_providers()})

    async def _voice_voices_handler(self, request: web.Request) -> web.Response:
        provider = request.query.get("provider", "deepgram")
        from nanobot.hooks.builtin.voice_providers import get_provider_voices
        return web.json_response(get_provider_voices(provider))

    async def _voice_validate_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        provider = body.get("provider", "")
        from nanobot.hooks.builtin.voice_providers import validate_provider
        result = await validate_provider(provider, self._get_workspace())
        return web.json_response(result)

    async def _voice_samples_handler(self, request: web.Request) -> web.Response:
        from nanobot.hooks.builtin.voice_providers import get_voice_samples
        return web.json_response({"samples": get_voice_samples(self._get_workspace())})

    async def _voice_sample_upload_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        from nanobot.hooks.builtin.voice_providers import save_voice_sample
        path = save_voice_sample(self._get_workspace(), body.get("audio_b64", ""), body.get("name", "my_voice"))
        return web.json_response({"ok": True, "path": path})

    async def _voice_sample_delete_handler(self, request: web.Request) -> web.Response:
        name = request.match_info.get("name", "")
        from nanobot.hooks.builtin.voice_providers import delete_voice_sample
        ok = delete_voice_sample(self._get_workspace(), name)
        return web.json_response({"ok": ok})

    async def _features_manifest_handler(self, request: web.Request) -> web.Response:
        """GET /api/features — full manifest of all configurable features."""
        from nanobot.hooks.builtin.feature_registry import get_feature_manifest, get_feature_categories
        ws = self._get_workspace()
        return web.json_response({
            "features": get_feature_manifest(ws),
            "categories": get_feature_categories(),
        })

    async def _features_save_handler(self, request: web.Request) -> web.Response:
        """POST /api/features — save a feature value to unified mawa_settings.json.

        Body: {key, value} — no category routing needed.
        """
        body = await request.json()
        key = body.get("key", "")
        value = body.get("value")

        from nanobot.hooks.builtin.feature_registry import save_setting
        save_setting(self._get_workspace(), key, value)

        return web.json_response({"ok": True, "key": key, "value": value})

    async def _budget_save_handler(self, request: web.Request) -> web.Response:
        """Legacy — redirects to unified settings."""
        body = await request.json()
        from nanobot.hooks.builtin.feature_registry import save_settings_bulk
        prefixed = {f"budget_{k}": v for k, v in body.items()}
        save_settings_bulk(self._get_workspace(), prefixed)
        return web.json_response({"ok": True})

    async def _behavior_save_handler(self, request: web.Request) -> web.Response:
        """Legacy — redirects to unified settings."""
        body = await request.json()
        from nanobot.hooks.builtin.feature_registry import save_settings_bulk
        save_settings_bulk(self._get_workspace(), body)
        return web.json_response({"ok": True})

    async def _maintenance_settings_handler(self, request: web.Request) -> web.Response:
        body = await request.json()
        path = self._get_workspace() / "maintenance_settings.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data.update(body)
        path.write_text(json.dumps(data, indent=2))
        return web.json_response({"ok": True})

    async def _intelligence_get_handler(self, request: web.Request) -> web.Response:
        """GET /api/intelligence — return current intelligence toggle states."""
        settings = _load_intelligence_settings(self._get_workspace())
        return web.json_response(settings)

    async def _intelligence_set_handler(self, request: web.Request) -> web.Response:
        """POST /api/intelligence — save intelligence toggle states."""
        try:
            body = await request.json()
            _save_intelligence_settings(self._get_workspace(), body)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _usage_handler(self, request: web.Request) -> web.Response:
        """Return token usage summary for the dashboard."""
        from nanobot.hooks.builtin.usage_tracker import get_usage_summary
        return web.json_response(get_usage_summary(self._get_workspace()))

    async def _search_handler(self, request: web.Request) -> web.Response:
        """Search conversations and memory."""
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"sessions": [], "memory": []})
        from nanobot.hooks.builtin.conversation_search import search_all
        return web.json_response(search_all(self._get_workspace(), q))

    async def _sessions_list_handler(self, request: web.Request) -> web.Response:
        """List past sessions."""
        from nanobot.hooks.builtin.session_manager import list_sessions
        limit = int(request.query.get("limit", "50"))
        return web.json_response({"sessions": list_sessions(self._get_workspace(), limit)})

    async def _session_detail_handler(self, request: web.Request) -> web.Response:
        """Get a single session's messages."""
        key = request.match_info["key"]
        from nanobot.hooks.builtin.session_manager import get_session
        session = get_session(self._get_workspace(), key)
        if not session:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(session)

    async def _session_export_handler(self, request: web.Request) -> web.Response:
        """Export a session as markdown."""
        key = request.match_info["key"]
        fmt = request.query.get("format", "markdown")
        from nanobot.hooks.builtin.session_manager import export_session
        content = export_session(self._get_workspace(), key, fmt)
        if not content:
            return web.json_response({"error": "not found"}, status=404)
        ct = "text/markdown" if fmt == "markdown" else "application/json"
        return web.Response(text=content, content_type=ct)

    async def _session_delete_handler(self, request: web.Request) -> web.Response:
        """Delete a session."""
        key = request.match_info["key"]
        from nanobot.hooks.builtin.session_manager import delete_session
        ok = delete_session(self._get_workspace(), key)
        return web.json_response({"deleted": ok})

    async def _budget_handler(self, request: web.Request) -> web.Response:
        """Get budget config and current status."""
        from nanobot.hooks.builtin.cost_budget import load_budget, check_budget
        return web.json_response({
            "config": load_budget(self._get_workspace()),
            "status": check_budget(self._get_workspace()),
        })

    async def _budget_update_handler(self, request: web.Request) -> web.Response:
        """Update budget config."""
        data = await request.json()
        from nanobot.hooks.builtin.cost_budget import save_budget
        save_budget(self._get_workspace(), data)
        return web.json_response({"ok": True})

    async def _favorites_handler(self, request: web.Request) -> web.Response:
        """Get favorite tools based on usage."""
        from nanobot.hooks.builtin.tool_favorites import get_favorites
        return web.json_response({"favorites": get_favorites(self._get_workspace())})

    async def _search_handler(self, request: web.Request) -> web.Response:
        """Search across chat history and memory. GET /api/search?q=..."""
        try:
            from nanobot.config.paths import get_workspace_path
            import re

            query = request.query.get("q", "").strip()
            if not query or len(query) < 2:
                return web.json_response({"results": []})

            workspace = get_workspace_path()
            results = []
            words = [w for w in query.lower().split() if len(w) >= 2]
            pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)

            # Search HISTORY.md
            history = workspace / "memory" / "HISTORY.md"
            if history.exists():
                for line in history.read_text(encoding="utf-8").split("\n"):
                    if pattern.search(line) and len(line.strip()) > 10:
                        results.append({"source": "history", "text": line.strip()[:200]})
                        if len(results) >= 20:
                            break

            # Search LONG_TERM.md
            lt = workspace / "memory" / "LONG_TERM.md"
            if lt.exists():
                content = lt.read_text(encoding="utf-8")
                for para in content.split("\n\n"):
                    if pattern.search(para):
                        results.append({"source": "memory", "text": para.strip()[:200]})
                        if len(results) >= 20:
                            break

            return web.json_response({"results": results[:20], "query": query})
        except Exception as e:
            logger.warning("Search error: {}", e)
            return web.json_response({"error": str(e)}, status=500)

    # Duplicate _sessions_list_handler removed — using session_manager.list_sessions above

    async def _sessions_switch_handler(self, request: web.Request) -> web.Response:
        """Switch to a different session. POST /api/sessions/switch."""
        try:
            body = await request.json()
            session_key = body.get("key", "")
            if not session_key:
                return web.json_response({"error": "key required"}, status=400)
            # The actual session switch happens on the WebSocket side
            # Just return the session info for the frontend
            return web.json_response({"ok": True, "key": session_key})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _suggestions_handler(self, request: web.Request) -> web.Response:
        """Generate context-aware quick reply suggestions. GET /api/suggestions."""
        try:
            from datetime import datetime

            now = datetime.now()
            hour = now.hour
            weekday = now.strftime("%A")
            suggestions = []

            # Time-based suggestions
            if 6 <= hour <= 10:
                suggestions.append({"text": "Check my email", "icon": "mail"})
                suggestions.append({"text": "What's on my calendar today?", "icon": "calendar"})
            elif 11 <= hour <= 14:
                suggestions.append({"text": "Any urgent emails?", "icon": "mail"})
            elif 17 <= hour <= 21:
                suggestions.append({"text": "Summarize what happened today", "icon": "sparkles"})

            # Always available
            suggestions.append({"text": "Check my goals", "icon": "target"})

            # [#2] Merge predictive suggestions from usage patterns
            from nanobot.hooks.builtin.code_features import get_predictive_suggestions
            for ps in get_predictive_suggestions(self._get_workspace()):
                if not any(s["text"] == ps for s in suggestions):
                    suggestions.append({"text": ps, "icon": "sparkles"})

            return web.json_response({"suggestions": suggestions[:5]})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _shortcut_download_handler(self, request: web.Request) -> web.Response:
        """Serve a helper page that guides the user to create an Apple Shortcut."""
        action = request.match_info.get("action", "")
        if not action:
            return web.json_response({"error": "action required"}, status=400)

        host = request.headers.get("Host", "localhost:3000")
        scheme = "https" if "taile" in host or request.secure else "http"
        base_url = f"{scheme}://{host}"

        shortcuts = {
            "check-email": ("Check Email", f"{base_url}/?action=check-email"),
            "check-calendar": ("Today's Schedule", f"{base_url}/?action=check-calendar"),
            "check-goals": ("My Goals", f"{base_url}/?view=goals"),
            "voice": ("Talk to Mawa", f"{base_url}/?view=chat&mode=voice"),
            "briefing": ("Morning Briefing", f"{base_url}/?action=briefing"),
            "search": ("Search Mawa", f"{base_url}/?view=chat&search=1"),
            "browser": ("Open Browser", f"{base_url}/?action=open-browser"),
        }

        if action not in shortcuts:
            return web.json_response({"error": f"Unknown shortcut: {action}"}, status=404)

        name, url = shortcuts[action]
        import urllib.parse
        encoded_url = urllib.parse.quote(url, safe="")

        # Serve a page that opens the Shortcuts app with a pre-built shortcut
        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Add Siri Shortcut: {name}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0a0a0a; color: #e0e0e0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 100vh; margin: 0; padding: 20px; text-align: center; }}
h1 {{ font-size: 1.5em; margin-bottom: 8px; }}
p {{ color: #888; font-size: 0.9em; max-width: 320px; line-height: 1.5; }}
.url {{ background: #1a1a1a; padding: 12px; border-radius: 12px; font-family: monospace;
  font-size: 0.75em; word-break: break-all; margin: 16px 0; color: #14b8a6; max-width: 320px; }}
.btn {{ display: inline-block; background: #14b8a6; color: #000; padding: 14px 28px;
  border-radius: 12px; font-weight: 600; text-decoration: none; margin: 8px; font-size: 1em; }}
.btn-secondary {{ background: #222; color: #e0e0e0; }}
.steps {{ text-align: left; max-width: 320px; margin: 20px 0; }}
.steps li {{ margin: 8px 0; font-size: 0.85em; color: #aaa; }}
.steps strong {{ color: #e0e0e0; }}
</style>
</head><body>
<h1>Mawa: {name}</h1>
<p>Add this as a Siri Shortcut to trigger with your voice.</p>
<div class="url">{url}</div>

<a class="btn" href="shortcuts://create-shortcut" id="openBtn">Open Shortcuts App</a>
<button class="btn btn-secondary" onclick="copy()">Copy URL</button>

<ol class="steps">
  <li>Tap <strong>"Open Shortcuts App"</strong> above</li>
  <li>Tap <strong>+</strong> to create a new shortcut</li>
  <li>Search <strong>"Open URL"</strong> and add it</li>
  <li>Tap the URL field → <strong>Paste</strong> (already copied!)</li>
  <li>Tap the name → rename to <strong>"Mawa: {name}"</strong></li>
  <li>Tap <strong>"Add to Siri"</strong> and record your phrase</li>
</ol>

<a href="{base_url}/" class="btn btn-secondary" style="margin-top: 20px;">← Back to Mawa</a>

<script>
// Auto-copy the URL to clipboard when page loads
async function copy() {{
  try {{
    await navigator.clipboard.writeText("{url}");
    document.querySelector('.btn-secondary').textContent = 'Copied!';
    setTimeout(() => document.querySelector('.btn-secondary').textContent = 'Copy URL', 2000);
  }} catch(e) {{
    // Fallback for iOS
    const ta = document.createElement('textarea');
    ta.value = "{url}";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    document.querySelector('.btn-secondary').textContent = 'Copied!';
  }}
}}
// Auto-copy on page load
copy();
</script>
</body></html>"""

        return web.Response(text=html, content_type="text/html")

    async def _credentials_list_handler(self, request: web.Request) -> web.Response:
        """List stored credentials — names only, NEVER actual values."""
        try:
            from nanobot.setup.vault import load_vault

            vault = load_vault()
            creds = []

            # Show ALL vault entries grouped by category
            for k in sorted(vault.keys()):
                value = str(vault[k])
                # Categorize
                if k.startswith("cred."):
                    if k.endswith(".username"):
                        continue  # Skip username entries, shown with their credential
                    display_name = k[5:]  # Remove "cred." prefix
                    username = vault.get(f"{k}.username", "")
                    category = "credential"
                elif k.startswith("providers."):
                    parts = k.split(".")
                    display_name = f"{parts[1]} ({parts[2] if len(parts) > 2 else 'key'})"
                    username = ""
                    category = "provider"
                elif k.startswith("channels."):
                    parts = k.split(".")
                    display_name = f"{parts[1]} ({'.'.join(parts[2:]) if len(parts) > 2 else 'config'})"
                    username = ""
                    category = "channel"
                elif k.startswith("tools."):
                    parts = k.split(".")
                    display_name = f"{parts[1]} ({'.'.join(parts[2:]) if len(parts) > 2 else 'config'})"
                    username = ""
                    category = "tool"
                else:
                    display_name = k
                    username = ""
                    category = "other"

                creds.append({
                    "name": display_name,
                    "key": k,
                    "username": username,
                    "masked": "*" * min(len(value), 12) if value else "",
                    "length": len(value),
                    "category": category,
                })

            return web.json_response({
                "credentials": creds,
                "apiKeyCount": sum(1 for c in creds if c["category"] in ("provider", "tool")),
                "totalSecrets": len(vault),
            })
        except Exception as e:
            logger.warning("Credentials list error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _credentials_update_handler(self, request: web.Request) -> web.Response:
        """Add, update, or delete credentials. POST /api/credentials."""
        try:
            from nanobot.setup.vault import load_vault, save_to_vault

            body = await request.json()
            action = body.get("action", "")

            if action == "save":
                name = body.get("name", "").strip()
                value = body.get("value", "")
                username = body.get("username", "")
                if not name or not value:
                    return web.json_response({"error": "name and value required"}, status=400)

                secrets = {f"cred.{name}": value}
                if username:
                    secrets[f"cred.{name}.username"] = username
                save_to_vault(secrets)
                return web.json_response({"ok": True, "result": f"Credential '{name}' saved"})

            elif action == "delete":
                name = body.get("name", "").strip()
                if not name:
                    return web.json_response({"error": "name required"}, status=400)

                vault = load_vault()
                key = f"cred.{name}"
                if key not in vault:
                    return web.json_response({"error": f"'{name}' not found"}, status=404)

                del vault[key]
                vault.pop(f"cred.{name}.username", None)
                # Use replace_vault (not save_to_vault which merges and re-adds deleted keys)
                from nanobot.setup.vault import replace_vault
                replace_vault(vault)
                return web.json_response({"ok": True, "result": f"Credential '{name}' deleted"})

            else:
                return web.json_response({"error": f"Unknown action: {action}"}, status=400)

        except Exception as e:
            logger.warning("Credentials update error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _mcp_servers_list_handler(self, request: web.Request) -> web.Response:
        """List configured MCP servers. GET /api/mcp-servers."""
        try:
            import json as _json
            config_path = Path("/root/.nanobot/config.json")
            cfg = _json.loads(config_path.read_text())
            servers = cfg.get("tools", {}).get("mcpServers", {})

            result = []
            for name, srv in servers.items():
                result.append({
                    "name": name,
                    "type": srv.get("type", "streamableHttp"),
                    "url": srv.get("url", ""),
                    "hasHeaders": bool(srv.get("headers")),
                    "enabledTools": srv.get("enabledTools", ["*"]),
                    "toolTimeout": srv.get("toolTimeout", 30),
                })
            return web.json_response({"servers": result})
        except Exception as e:
            logger.warning("MCP servers list error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _mcp_servers_save_handler(self, request: web.Request) -> web.Response:
        """Add or update an MCP server. POST /api/mcp-servers."""
        try:
            import json as _json
            from nanobot.setup.vault import save_to_vault

            body = await request.json()
            name = body.get("name", "").strip()
            if not name:
                return web.json_response({"error": "name required"}, status=400)
            url = body.get("url", "").strip()
            if not url:
                return web.json_response({"error": "url required"}, status=400)

            transport = body.get("type", "streamableHttp")
            headers = body.get("headers", {})
            enabled_tools = body.get("enabledTools", ["*"])
            tool_timeout = body.get("toolTimeout", 30)

            config_path = Path("/root/.nanobot/config.json")
            cfg = _json.loads(config_path.read_text())
            mcp = cfg.setdefault("tools", {}).setdefault("mcpServers", {})

            # Store secrets in vault, put references in config
            vault_secrets = {}
            config_headers = {}
            for hk, hv in headers.items():
                if hv and len(hv) > 5:
                    vault_key = f"tools.mcpServers.{name}.headers.{hk}"
                    vault_secrets[vault_key] = hv
                    config_headers[hk] = f"${{vault:{vault_key}}}"
                else:
                    config_headers[hk] = hv

            # Store URL in vault if it contains secrets (API keys in query params)
            if "api_key" in url.lower() or "token" in url.lower() or len(url) > 100:
                vault_key = f"tools.mcpServers.{name}.url"
                vault_secrets[vault_key] = url
                config_url = f"${{vault:{vault_key}}}"
            else:
                config_url = url

            if vault_secrets:
                save_to_vault(vault_secrets)

            mcp[name] = {
                "type": transport,
                "url": config_url,
            }
            if config_headers:
                mcp[name]["headers"] = config_headers
            if enabled_tools != ["*"]:
                mcp[name]["enabledTools"] = enabled_tools
            if tool_timeout != 30:
                mcp[name]["toolTimeout"] = tool_timeout

            config_path.write_text(_json.dumps(cfg, indent=2))
            return web.json_response({"ok": True, "result": f"MCP server '{name}' saved. Restart nanobot to connect."})
        except Exception as e:
            logger.warning("MCP servers save error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    async def _mcp_servers_delete_handler(self, request: web.Request) -> web.Response:
        """Delete an MCP server. DELETE /api/mcp-servers/{name}."""
        try:
            import json as _json
            name = request.match_info.get("name", "")
            if not name:
                return web.json_response({"error": "name required"}, status=400)

            config_path = Path("/root/.nanobot/config.json")
            cfg = _json.loads(config_path.read_text())
            mcp = cfg.get("tools", {}).get("mcpServers", {})

            if name not in mcp:
                return web.json_response({"error": f"Server '{name}' not found"}, status=404)

            del mcp[name]
            config_path.write_text(_json.dumps(cfg, indent=2))
            return web.json_response({"ok": True, "result": f"MCP server '{name}' removed. Restart nanobot to apply."})
        except Exception as e:
            logger.warning("MCP servers delete error: {}", e)
            return web.json_response({"error": "Internal error"}, status=500)

    def _get_active_profile_name(self) -> str:
        """Get the name of the currently active LLM profile.

        Matches the current config's model+provider against defined profiles.
        """
        try:
            from nanobot.config.loader import load_config
            config = load_config()
            current_model = config.agents.defaults.model
            current_provider = config.agents.defaults.provider
            for name, p in (config.profiles or {}).items():
                if p.model == current_model and p.provider == current_provider:
                    return name
            # If no profile matches, check just model
            for name, p in (config.profiles or {}).items():
                if p.model == current_model:
                    return name
            return "default"
        except Exception:
            return "default"

    async def _profiles_handler(self, request: web.Request) -> web.Response:
        """Return available LLM profiles for the UI switcher."""
        try:
            from nanobot.config.loader import load_config
            config = load_config()
            profiles = {}
            for name, p in (config.profiles or {}).items():
                profiles[name] = {
                    "provider": p.provider or "—",
                    "model": p.model or "—",
                }
            return web.json_response({"profiles": profiles})
        except Exception as e:
            logger.warning("Profiles API error: {}", e)
            return web.json_response({"profiles": {}}, status=500)

    # ── WebSocket handler ──────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=20 * 1024 * 1024)  # 20MB for image attachments
        await ws.prepare(request)

        session_id = f"webvoice_{id(ws)}"  # Default; overridden by identify action
        self._clients[session_id] = ws
        self._utterance_buffer[session_id] = []
        self._pending_count[session_id] = 0

        dg_ws = None
        recv_task = None

        logger.info("Web Voice client connected: {}", session_id)

        # Send initial config to frontend
        await ws.send_json({
            "type": "config",
            "app_name": self.config.app_name,
        })

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    action = data.get("action")

                    # Client identification — reuse session across reconnects
                    if action == "identify":
                        client_id = data.get("client_id", "").strip()
                        if client_id:
                            old_id = session_id
                            session_id = client_id
                            # Close only the previous connection with the SAME client ID (same device reconnect)
                            old_ws = self._clients.get(session_id)
                            if old_ws and old_ws is not ws and not old_ws.closed:
                                logger.info("Web Voice: closing old connection for reconnecting client {}", session_id)
                                await self._send_stop(session_id)
                                try:
                                    await old_ws.close()
                                except Exception:
                                    pass
                            # Migrate from temp ID to persistent ID
                            self._clients.pop(old_id, None)
                            self._utterance_buffer.pop(old_id, None)
                            self._pending_count.pop(old_id, None)
                            self._clients[session_id] = ws
                            self._utterance_buffer.setdefault(session_id, [])
                            self._pending_count.setdefault(session_id, 0)
                            logger.info("Web Voice client identified: {} (was {})", session_id, old_id)

                            # Deliver any pending notifications that were stored while offline
                            try:
                                from nanobot.hooks.builtin.notification_store import get_pending, mark_all_read
                                pending = get_pending(self._get_workspace())
                                if pending:
                                    for notif in pending:
                                        await ws.send_json({
                                            "type": "notification",
                                            "text": notif["content"],
                                            "priority": notif.get("metadata", {}).get("_priority", "normal"),
                                            "proactive": bool(notif.get("metadata", {}).get("_proactive")),
                                            "cron": bool(notif.get("metadata", {}).get("_cron")),
                                        })
                                    mark_all_read(self._get_workspace())
                                    logger.info("Delivered {} pending notifications to {}", len(pending), session_id)
                            except Exception as e:
                                logger.debug("Pending notification delivery error: {}", e)
                        continue

                    if action == "start":
                        # Close any existing Deepgram connection first
                        if dg_ws:
                            logger.info("Web Voice: closing existing STT for {}", session_id)
                            try:
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                                await dg_ws.close()
                            except Exception:
                                pass
                            dg_ws = None
                        if recv_task:
                            recv_task.cancel()
                            recv_task = None

                        # Store voice preferences (per-session + workspace file for agent loop access)
                        if not hasattr(self, '_speak_reasoning'):
                            self._speak_reasoning = {}
                        self._speak_reasoning[session_id] = bool(data.get("speak_reasoning", False))
                        # Write to workspace so the agent loop can read it
                        try:
                            vp = self._get_workspace() / "voice_prefs.json"
                            vp.write_text(json.dumps({"speak_reasoning": self._speak_reasoning.get(session_id, False)}))
                        except Exception:
                            pass

                        # Support client-specified encoding (for Opus codec)
                        client_encoding = data.get("encoding", "linear16")
                        client_sample_rate = data.get("sample_rate", "16000")

                        params = {
                            "model": self.config.stt_model,
                            "language": data.get("language", "multi"),
                            "smart_format": "true",
                            "punctuate": "true",
                            "interim_results": "true",
                            "utterance_end_ms": "1000",
                            "vad_events": "true",
                            "encoding": client_encoding,
                            "sample_rate": client_sample_rate,
                            "channels": "1",
                            "endpointing": "400",
                            # Audio intelligence
                            "sentiment": "true",
                        }

                        try:
                            dg_ws = await self._connect_deepgram(params, ws)
                            stt_params = params  # Store for reconnection
                            self._utterance_buffer[session_id] = []

                            async def forward_transcripts():
                                nonlocal dg_ws
                                while True:
                                    try:
                                        async for dg_msg in dg_ws:
                                            result = json.loads(dg_msg)
                                            msg_type = result.get("type", "")

                                            if msg_type == "Results":
                                                channel_data = result.get("channel", {})
                                                alts = channel_data.get("alternatives", [])
                                                is_final = result.get("is_final", False)
                                                if alts:
                                                    text = alts[0].get("transcript", "").strip()
                                                    confidence = alts[0].get("confidence", 0)
                                                    if text:
                                                        # Extract sentiment from audio intelligence
                                                        sentiment_data = None
                                                        if is_final:
                                                            sentiments = result.get("channel", {}).get("alternatives", [{}])[0].get("sentiment", {})
                                                            if sentiments:
                                                                sentiment_data = sentiments
                                                        # Extract detected language from multi-language STT
                                                        detected_lang = result.get("channel", {}).get("detected_language") or result.get("metadata", {}).get("detected_language")
                                                        if detected_lang and is_final:
                                                            self._detected_language = detected_lang

                                                        msg_out = {
                                                            "type": "transcript",
                                                            "is_final": is_final,
                                                            "text": text,
                                                            "confidence": round(confidence, 3),
                                                        }
                                                        if detected_lang:
                                                            msg_out["detected_language"] = detected_lang
                                                        if sentiment_data:
                                                            msg_out["sentiment"] = sentiment_data
                                                        # Broadcast to all clients
                                                        for cid, cws in list(self._clients.items()):
                                                            if not cws.closed:
                                                                try:
                                                                    await cws.send_json(msg_out)
                                                                except Exception:
                                                                    pass
                                                        if is_final and confidence > 0.5:
                                                            self._utterance_buffer[session_id].append(text)

                                            elif msg_type == "SpeechStarted":
                                                await ws.send_json({"type": "speech_started"})

                                            elif msg_type == "UtteranceEnd":
                                                await ws.send_json({"type": "utterance_end"})
                                                buf = self._utterance_buffer.get(session_id, [])
                                                if buf:
                                                    full_text = " ".join(buf)
                                                    if _is_sentence_complete(full_text):
                                                        # Sentence is complete — submit immediately
                                                        self._utterance_buffer[session_id] = []
                                                        self._enqueue_message(session_id, full_text)
                                                    else:
                                                        # Sentence incomplete — wait 1.5s for more speech
                                                        logger.debug("Smart endpointing: incomplete '{}', waiting...", full_text[:50])
                                                        _pending_key = f"_pending_submit_{session_id}"
                                                        # Cancel any existing pending submit
                                                        prev = getattr(self, _pending_key, None)
                                                        if prev and not prev.done():
                                                            prev.cancel()
                                                        async def _delayed_submit(sid=session_id, txt=full_text):
                                                            await asyncio.sleep(1.5)
                                                            # Check if buffer was already submitted or got more text
                                                            cur_buf = self._utterance_buffer.get(sid, [])
                                                            if cur_buf:
                                                                merged = " ".join(cur_buf)
                                                                self._utterance_buffer[sid] = []
                                                                self._enqueue_message(sid, merged)
                                                                logger.debug("Smart endpointing: submitted after wait '{}'", merged[:50])
                                                        setattr(self, _pending_key, asyncio.create_task(_delayed_submit()))

                                            elif msg_type == "Error":
                                                await ws.send_json({"type": "error", "message": str(result)})
                                        # Normal end of stream — no reconnect needed
                                        break
                                    except asyncio.CancelledError:
                                        return
                                    except Exception as e:
                                        if ws.closed:
                                            return
                                        logger.warning("Deepgram connection lost: {}, reconnecting...", e)
                                        await ws.send_json({"type": "status", "message": "Speech service disconnected, reconnecting..."})
                                        try:
                                            dg_ws = await self._connect_deepgram(stt_params, ws)
                                        except ConnectionError:
                                            await ws.send_json({"type": "error", "message": "Speech service unavailable"})
                                            return

                            recv_task = asyncio.create_task(forward_transcripts())
                            await ws.send_json({"type": "started", "model": self.config.stt_model})
                            logger.info("Web Voice streaming started for {}", session_id)

                        except Exception as e:
                            await ws.send_json({"type": "error", "message": f"Failed to connect: {e}"})
                            dg_ws = None

                    elif action == "stop":
                        if dg_ws:
                            try:
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                                await dg_ws.close()
                            except Exception:
                                pass
                            dg_ws = None
                        if recv_task:
                            recv_task.cancel()
                            recv_task = None
                        buf = self._utterance_buffer.get(session_id, [])
                        if buf:
                            full_text = " ".join(buf)
                            self._utterance_buffer[session_id] = []
                            self._enqueue_message(session_id, full_text)
                        await ws.send_json({"type": "stopped"})

                    elif action == "profile":
                        profile_name = data.get("name", "").strip()
                        if profile_name:
                            self._enqueue_message(session_id, f"/profile {profile_name}")
                            await ws.send_json({"type": "profile_switched", "name": profile_name})

                    elif action == "interrupt":
                        # User tapped interrupt — stop TTS, cancel pending, go to listening
                        logger.info("Web Voice interrupt from {}", session_id)
                        # Clear TTS queue for this session
                        if session_id in self._tts_queues:
                            while not self._tts_queues[session_id].empty():
                                try:
                                    self._tts_queues[session_id].get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                        # Cancel any pending agent work
                        await self._send_stop(session_id)
                        self._pending_count[session_id] = 0
                        await ws.send_json({"type": "queue_status", "pending": 0})
                        # Clear utterance buffer so old speech doesn't get submitted
                        self._utterance_buffer[session_id] = []
                        # Tell frontend to go back to listening mode
                        await ws.send_json({"type": "status", "message": "listening"})

                    elif action == "submit_now":
                        # Push-to-talk release: immediately submit buffered utterance
                        buf = self._utterance_buffer.get(session_id, [])
                        if buf:
                            full_text = " ".join(buf)
                            self._utterance_buffer[session_id] = []
                            self._enqueue_message(session_id, full_text)

                    elif action == "text":
                        text = data.get("text", "").strip()
                        media = data.get("media", [])  # List of base64 data URIs or file paths
                        if text or media:
                            self._enqueue_message(session_id, text or "(attachment)", media=media if media else None)

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if dg_ws:
                        try:
                            await dg_ws.send(msg.data)
                        except Exception:
                            pass

        except Exception as e:
            logger.error("Web Voice WebSocket error: {}", e)
        finally:
            if dg_ws:
                try:
                    await dg_ws.close()
                except Exception:
                    pass
            if recv_task:
                recv_task.cancel()
            self._clients.pop(session_id, None)
            self._utterance_buffer.pop(session_id, None)
            self._streamed_text.pop(session_id, None)
            self._pending_count.pop(session_id, None)
            self._activity_ts.pop(session_id, None)
            tts_worker = self._tts_workers.pop(session_id, None)
            if tts_worker:
                tts_worker.cancel()
            self._tts_queues.pop(session_id, None)

            # Auto-consolidate on disconnect so no conversation is lost
            try:
                workspace = self._get_workspace()
                from nanobot.session.manager import SessionManager
                sessions = SessionManager(workspace)
                session = sessions.get_or_create("web_voice:voice")
                unconsolidated = session.messages[session.last_consolidated:]
                if len(unconsolidated) >= 6:  # Only if there's meaningful content
                    from nanobot.agent.memory import MemoryStore
                    from nanobot.providers.litellm_provider import LiteLLMProvider
                    store = MemoryStore(workspace)
                    provider = LiteLLMProvider()
                    model = provider.get_default_model()
                    result = await store.consolidate(unconsolidated, provider, model)
                    if result:
                        session.last_consolidated = len(session.messages)
                        sessions.save(session)
                        logger.info("Auto-consolidated {} messages on disconnect", len(unconsolidated))
            except Exception as e:
                logger.debug("Auto-consolidation on disconnect failed: {}", e)

            logger.info("Web Voice client disconnected: {}", session_id)

        return ws

    # ── Message queue (serialize per session) ──────────────────────

    def _enqueue_message(self, session_id: str, text: str, media: list[str] | None = None) -> None:
        """Submit a message to the agent immediately (no queuing).

        The agent loop handles concurrency: if it's already busy on this session,
        the message gets live-injected into the running conversation so the LLM
        sees it on its next iteration.

        Interrupt patterns ('no', 'stop', 'wait') still cancel via /stop.
        media: list of base64 data URIs (images) or file paths.
        """
        is_interrupt = bool(_INTERRUPT_PATTERNS.search(text.strip()))

        if is_interrupt:
            asyncio.create_task(self._send_stop(session_id))
            self._pending_count[session_id] = 0
            ws = self._clients.get(session_id)
            if ws and not ws.closed:
                asyncio.create_task(ws.send_json({"type": "queue_status", "pending": 0}))

        # Fire directly to the agent — no channel-level queue
        asyncio.create_task(self._submit_to_agent(session_id, text, media=media))

    async def _send_stop(self, session_id: str) -> None:
        """Send /stop command to cancel any running agent task for this session."""
        try:
            from nanobot.bus.events import InboundMessage
            stop_msg = InboundMessage(
                channel=self.name,
                sender_id=session_id,
                chat_id="voice",  # shared session
                content="/stop",
                media=[],
                metadata={},
            )
            await self.bus.publish_inbound(stop_msg)
            logger.info("Web Voice: sent /stop for session {}", session_id)
        except Exception as e:
            logger.warning("Web Voice: failed to send /stop: {}", e)

    # ── Agent integration ──────────────────────────────────────────

    async def _submit_to_agent(self, session_id: str, text: str, media: list[str] | None = None) -> None:
        logger.info("Web Voice heard: '{}' (media: {})", text, len(media) if media else 0)

        ws = self._clients.get(session_id)
        if ws and not ws.closed:
            await ws.send_json({"type": "processing", "text": text})

        # Reset latency tracking for this request
        self._activity_ts[session_id] = time.time()

        # Fire-and-forget: intel goes to activity feed + stashed for LLM enrichment
        if len(text.split()) >= 4:
            asyncio.create_task(self._get_intel_for_llm(session_id, text))

        # Use fixed chat_id so all devices share the same nanobot session
        shared_chat_id = "voice"

        # Save non-image attachments to inbox and convert media to paths
        processed_media = []
        if media:
            for item in media:
                # item can be: string (data URI or path) or dict {dataUri, name}
                data_uri = item.get("dataUri", item) if isinstance(item, dict) else item
                original_name = item.get("name", "") if isinstance(item, dict) else ""

                if isinstance(data_uri, str) and data_uri.startswith("data:image/"):
                    processed_media.append(data_uri)
                elif isinstance(data_uri, str) and data_uri.startswith("data:"):
                    # Route to folder based on message context
                    folder = self._detect_inbox_folder(text)
                    saved_name, saved_path = await self._save_data_uri_to_inbox(
                        data_uri, original_name=original_name, folder=folder,
                    )
                    if saved_path:
                        text = (
                            f"{text}\n\n[Attached file: {saved_path} (original: {saved_name})]\n"
                            "You can read, copy, or process this file using your tools. "
                            "If the user asks to use it with a skill, copy it to the skill's workspace."
                        )
                elif isinstance(data_uri, str):
                    processed_media.append(data_uri)

        await self._handle_message(
            sender_id=session_id,
            chat_id=shared_chat_id,
            content=text,
            media=processed_media if processed_media else None,
            metadata={
                "source": "voice",
                "_ws_session_id": session_id,
            },
        )

    @staticmethod
    def _detect_inbox_folder(text: str) -> str:
        """Detect which inbox folder to use based on message context."""
        text_lower = text.lower()
        # Work-related keywords
        if any(w in text_lower for w in [
            "work order", "work", "everi", "salesforce", "report",
            "invoice", "order", "shipping", "business", "client",
            "cea", "casino", "property", "task",
        ]):
            return "work"
        # Personal keywords
        if any(w in text_lower for w in [
            "personal", "family", "wedding", "tonni", "receipt",
            "photo", "vacation", "trip",
        ]):
            return "personal"
        return "general"

    async def _save_data_uri_to_inbox(
        self, data_uri: str, original_name: str = "", folder: str = "general",
    ) -> tuple[str, str] | tuple[None, None]:
        """Save a data URI to the inbox. Returns (filename, full_path) or (None, None)."""
        import base64 as b64
        try:
            header, encoded = data_uri.split(",", 1)
            mime = header.split(":")[1].split(";")[0]
            ext_map = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "text/csv": ".csv", "text/plain": ".txt", "application/json": ".json",
                "application/zip": ".zip",
            }
            ext = ext_map.get(mime, ".bin")

            # Use original filename if provided, sanitize it
            if original_name:
                # Sanitize: remove path separators, double dots
                safe = original_name.replace("/", "_").replace("\\", "_").replace("..", "_")
                # If no extension, append from MIME
                if "." not in safe:
                    safe = f"{safe}{ext}"
                name = safe
            else:
                name = f"upload_{int(time.time())}{ext}"

            inbox_dir = self._get_workspace() / "inbox" / folder
            inbox_dir.mkdir(parents=True, exist_ok=True)
            full_path = inbox_dir / name

            # Avoid overwriting — add suffix if exists
            if full_path.exists():
                stem = full_path.stem
                suffix = full_path.suffix
                name = f"{stem}_{int(time.time())}{suffix}"
                full_path = inbox_dir / name

            full_path.write_bytes(b64.b64decode(encoded))
            logger.info("Saved attachment: {}/{} ({})", folder, name, mime)
            return name, str(full_path)
        except Exception as e:
            logger.warning("Failed to save data URI: {}", e)
            return None, None

    async def _get_intel_for_llm(self, session_id: str, text: str) -> str | None:
        """Analyze speech, send to activity feed, and return summary for LLM context."""
        try:
            intel = await self._deepgram_analyze(text)
            logger.debug("Web Voice intel for '{}': {}", text[:40], intel)
            if intel:
                # Broadcast to ALL connected clients
                for cid, cws in list(self._clients.items()):
                    if not cws.closed:
                        try:
                            await cws.send_json({"type": "intel", "data": intel})
                        except Exception:
                            pass
                # Build a concise summary for the LLM
                parts = []
                if intel.get("sentiment"):
                    parts.append(f"Sentiment: {intel['sentiment']} ({intel.get('sentiment_score', '')})")
                if intel.get("intents"):
                    parts.append(f"Intents: {', '.join(intel['intents'])}")
                if intel.get("topics"):
                    parts.append(f"Topics: {', '.join(intel['topics'])}")
                if intel.get("summary"):
                    parts.append(f"Summary: {intel['summary']}")
                return " | ".join(parts) if parts else None
            return None
        except Exception as e:
            logger.warning("Web Voice intel failed: {}", e)
            return None

    # ── Deepgram connection ─────────────────────────────────────────

    async def _connect_deepgram(
        self,
        params: dict[str, str],
        ws: web.WebSocketResponse,
    ):
        """Connect to Deepgram STT with exponential backoff retry."""
        import websockets

        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"wss://api.deepgram.com/v1/listen?{qs}"
        delays = [1, 2, 4, 8, 16]
        max_retries = 5

        for attempt in range(max_retries):
            try:
                dg_ws = await websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Token {self.config.deepgram_api_key}"},
                    ping_interval=20,
                )
                if attempt > 0:
                    logger.info("Deepgram reconnected after {} attempts", attempt + 1)
                    if not ws.closed:
                        await ws.send_json({"type": "status", "message": "Speech service reconnected"})
                return dg_ws
            except Exception as e:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.warning("Deepgram connect attempt {}/{} failed: {}, retrying in {}s",
                              attempt + 1, max_retries, e, delay)
                if not ws.closed:
                    await ws.send_json({
                        "type": "status",
                        "message": f"Reconnecting to speech service... (attempt {attempt + 1})",
                    })
                await asyncio.sleep(delay)

        raise ConnectionError("Failed to connect to Deepgram after max retries")

    # ── Deepgram APIs ──────────────────────────────────────────────

    async def _generate_tts(self, text: str) -> bytes | None:
        """Generate TTS using the configured provider (dispatches to Deepgram/MiMo/Coqui/MMS).

        If multi-language STT detected a non-English language and the provider
        supports it, automatically adjust TTS language.
        """
        from nanobot.hooks.builtin.voice_providers import generate_tts
        from nanobot.hooks.builtin.feature_registry import get_setting

        ws = self._get_workspace()
        provider = get_setting(ws, "voiceTtsProvider", "deepgram")
        detected = getattr(self, "_detected_language", None)

        # Auto-switch TTS for non-English detected speech
        if detected and detected != "en" and provider == "deepgram":
            # Deepgram TTS is English-only — auto-switch to a multilingual provider
            # Map Deepgram language codes to MMS-TTS models
            lang_to_mms = {
                "bn": "facebook/mms-tts-ben",  # Bengali
                "hi": "facebook/mms-tts-hin",  # Hindi
                "ur": "facebook/mms-tts-urd",  # Urdu
                "ar": "facebook/mms-tts-ara",  # Arabic
                "es": "facebook/mms-tts-spa",  # Spanish
                "fr": "facebook/mms-tts-fra",  # French
                "de": "facebook/mms-tts-deu",  # German
                "zh": "facebook/mms-tts-cmn",  # Chinese Mandarin
                "ja": "facebook/mms-tts-jpn",  # Japanese
                "ko": "facebook/mms-tts-kor",  # Korean
                "ta": "facebook/mms-tts-tam",  # Tamil
                "gu": "facebook/mms-tts-guj",  # Gujarati
            }
            mms_model = lang_to_mms.get(detected)
            if mms_model:
                logger.info("Auto-switching TTS to MMS-TTS for detected language: {}", detected)
                from nanobot.hooks.builtin.voice_providers import _tts_mms, _get_hf_token, _strip_md
                clean = _strip_md(text)
                if clean and len(clean) >= 2:
                    result = await _tts_mms(clean[:2000], mms_model, _get_hf_token())
                    if result:
                        return result
                # Fallback to Deepgram English if MMS fails

        return await generate_tts(
            text=text,
            workspace=ws,
            deepgram_api_key=self.config.deepgram_api_key,
            deepgram_model=self.config.tts_model,
        )

    async def _deepgram_tts(self, text: str) -> bytes | None:
        text = _strip_markdown(text)
        if not text or len(text) < 2:
            return None
        if len(text) > 2000:
            text = text[:2000]

        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.deepgram.com/v1/speak",
                    headers={
                        "Authorization": f"Token {self.config.deepgram_api_key}",
                        "Content-Type": "application/json",
                    },
                    params={
                        "model": self.config.tts_model,
                        "encoding": "linear16",
                        "sample_rate": "24000",
                        "container": "none",
                    },
                    json={"text": text},
                )
                resp.raise_for_status()
                audio = resp.content
                return audio if len(audio) > 100 else None
        except Exception as e:
            logger.error("Web Voice TTS failed: {}", e)
            return None

    async def _deepgram_analyze(self, text: str) -> dict | None:
        """Run Deepgram Text Intelligence — sentiment, intents, topics, summary."""
        if not text or len(text) < 5:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    "https://api.deepgram.com/v1/read",
                    headers={
                        "Authorization": f"Token {self.config.deepgram_api_key}",
                        "Content-Type": "application/json",
                    },
                    params={
                        "language": "en",
                        "sentiment": "true",
                        "intents": "true",
                        "topics": "true",
                        "summarize": "v2",
                    },
                    json={"text": text},
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", {})

                output = {}
                avg = results.get("sentiments", {}).get("average", {})
                if avg:
                    output["sentiment"] = avg.get("sentiment", "")
                    output["sentiment_score"] = round(avg.get("sentiment_score", 0), 2)

                intents = results.get("intents", {}).get("segments", [])
                if intents:
                    output["intents"] = [
                        s.get("intents", [{}])[0].get("intent", "")
                        for s in intents[:5] if s.get("intents")
                    ]

                topics = results.get("topics", {}).get("segments", [])
                if topics:
                    output["topics"] = [
                        s.get("topics", [{}])[0].get("topic", "")
                        for s in topics[:5] if s.get("topics")
                    ]

                summary = results.get("summary", {}).get("text", "")
                if summary:
                    output["summary"] = summary

                return output if output else None
        except Exception as e:
            logger.debug("Web Voice text intelligence failed: {}", e)
            return None


# ── Dashboard HTML ─────────────────────────────────────────────────

_UI_DIR = Path(__file__).parent / "web_voice_ui"
_DASHBOARD_HTML_CACHE: str | None = None


def _load_dashboard_html() -> str:
    """Load dashboard HTML from web_voice_ui/index.html, with hot-reload in dev."""
    global _DASHBOARD_HTML_CACHE
    html_path = _UI_DIR / "index.html"
    if html_path.exists():
        # Always reload from disk so you can edit without restarting
        return html_path.read_text()
    # Fallback: shouldn't happen if the file exists
    if _DASHBOARD_HTML_CACHE:
        return _DASHBOARD_HTML_CACHE
    return "<html><body><h1>Error: web_voice_ui/index.html not found</h1></body></html>"


# Keep an inline fallback just in case (but the file takes priority)
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jarvis Voice</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0a0a0f; color: #e0e0e0; min-height: 100vh;
  display: flex; flex-direction: column;
}

.header {
  background: linear-gradient(135deg, #1a1a2e, #16213e);
  padding: 14px 24px; border-bottom: 1px solid #2a2a4a;
  display: flex; align-items: center; justify-content: space-between;
}
.header h1 {
  font-size: 1.4rem;
  background: linear-gradient(90deg, #00d4ff, #7b2ff7);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-right { display: flex; align-items: center; gap: 16px; }
.status { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; }
.status-dot {
  width: 10px; height: 10px; border-radius: 50%; background: #555;
  transition: background 0.3s;
}
.status-dot.active { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
.status-dot.listening { background: #ff4444; box-shadow: 0 0 10px #ff4444; animation: pulse 1.5s infinite; }
.status-dot.processing { background: #ffaa00; box-shadow: 0 0 10px #ffaa00; animation: pulse 0.8s infinite; }
.queue-badge {
  background: #7b2ff7; color: #fff; padding: 2px 8px; border-radius: 10px;
  font-size: 0.7rem; font-weight: 700; display: none;
}

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.layout { flex: 1; display: flex; gap: 0; overflow: hidden; }

.conv-panel {
  flex: 1; display: flex; flex-direction: column; padding: 14px; gap: 10px;
  min-width: 0;
}

.controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button {
  padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer;
  font-size: 0.9rem; font-weight: 600; transition: all 0.2s;
}
button:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-listen {
  background: linear-gradient(135deg, #00d4ff, #0099cc); color: #000;
  font-size: 1rem; padding: 12px 28px;
}
.btn-listen:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,212,255,0.3); }
.btn-listen.recording { background: linear-gradient(135deg, #ff4757, #cc0022); color: #fff; }

.visualizer { height: 36px; background: #12121f; border-radius: 8px; overflow: hidden; }
canvas { width: 100%; height: 100%; }

.conversation {
  flex: 1; overflow-y: auto; background: #12121f; border: 1px solid #2a2a4a;
  border-radius: 10px; padding: 14px; min-height: 200px;
  display: flex; flex-direction: column; gap: 10px;
  scroll-behavior: smooth;
}

.msg {
  max-width: 88%; padding: 10px 14px; border-radius: 10px;
  font-size: 0.9rem; line-height: 1.5; word-wrap: break-word;
}
.msg.user { align-self: flex-end; background: #1a3a5c; border: 1px solid #2a5a8c; color: #cde; }
.msg.assistant { align-self: flex-start; background: #1a1a2e; border: 1px solid #3a3a5a; color: #e0e0e0; }
.msg.interim { align-self: flex-end; background: transparent; border: 1px dashed #333; color: #666; font-style: italic; }
.msg .label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; opacity: 0.5; }

.text-input-row { display: flex; gap: 8px; }
.text-input-row input {
  flex: 1; padding: 10px 14px; border: 1px solid #2a2a4a; border-radius: 8px;
  background: #12121f; color: #e0e0e0; font-size: 0.9rem; outline: none;
}
.text-input-row input:focus { border-color: #00d4ff; }
.text-input-row button { padding: 10px 16px; background: #7b2ff7; color: #fff; border-radius: 8px; }

/* Activity panel */
.activity-panel {
  width: 380px; border-left: 1px solid #1a1a2e; background: #08080d;
  display: flex; flex-direction: column; overflow: hidden;
}
.activity-header {
  padding: 10px 14px; border-bottom: 1px solid #1a1a2e;
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: #555;
  display: flex; justify-content: space-between; align-items: center;
}
.activity-header button {
  padding: 3px 8px; font-size: 0.65rem; background: #1a1a2e; color: #555;
  border-radius: 4px;
}
.activity-feed {
  flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 5px;
  scroll-behavior: smooth;
}

.activity-item {
  padding: 7px 9px; border-radius: 5px; font-size: 0.75rem; line-height: 1.4;
  border-left: 3px solid transparent; word-wrap: break-word;
  max-height: 120px; overflow-y: auto;
}
.activity-item.tool {
  background: #0d1117; border-left-color: #f0883e; color: #d2a8ff;
  font-family: 'SF Mono', 'Fira Code', monospace;
}
.activity-item.result {
  background: #0d1117; border-left-color: #3fb950; color: #8b949e;
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.7rem;
}
.activity-item.thinking {
  background: #0d1117; border-left-color: #58a6ff; color: #8b949e;
  font-style: italic;
}
.activity-item.intel {
  background: #0d1420; border-left-color: #a371f7; color: #c9d1d9;
}
.activity-item .activity-label {
  font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.5px;
  margin-bottom: 2px; opacity: 0.6;
}
.activity-item.tool .activity-label { color: #f0883e; }
.activity-item.result .activity-label { color: #3fb950; }
.activity-item.thinking .activity-label { color: #58a6ff; }
.activity-item.intel .activity-label { color: #a371f7; }
.activity-item .activity-time {
  font-size: 0.55rem; color: #484f58; float: right;
}

.intel-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.intel-tag {
  padding: 1px 6px; border-radius: 3px; font-size: 0.65rem; font-weight: 500;
}
.intel-tag.sentiment-positive { background: #0d2818; color: #3fb950; }
.intel-tag.sentiment-negative { background: #280d0d; color: #f85149; }
.intel-tag.sentiment-neutral { background: #161b22; color: #8b949e; }
.intel-tag.intent { background: #1a1533; color: #a371f7; }
.intel-tag.topic { background: #0d1b2a; color: #58a6ff; }

@media (max-width: 900px) {
  .layout { flex-direction: column; }
  .activity-panel { width: 100%; max-height: 220px; border-left: none; border-top: 1px solid #1a1a2e; }
}
</style>
</head>
<body>

<div class="header">
  <h1>JARVIS</h1>
  <div class="header-right">
    <span class="queue-badge" id="queueBadge"></span>
    <div class="status">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Ready</span>
    </div>
  </div>
</div>

<div class="layout">
  <div class="conv-panel">
    <div class="controls">
      <button class="btn-listen" id="btnListen" onclick="toggleVoice()">Start Listening</button>
    </div>
    <div class="visualizer"><canvas id="visualizer"></canvas></div>
    <div class="conversation" id="conversation"></div>
    <div class="text-input-row">
      <input type="text" id="textInput" placeholder="Or type a message..." onkeydown="if(event.key==='Enter')sendText()">
      <button onclick="sendText()">Send</button>
    </div>
  </div>

  <div class="activity-panel">
    <div class="activity-header">
      <span>Agent Activity</span>
      <button onclick="clearActivity()">Clear</button>
    </div>
    <div class="activity-feed" id="activityFeed"></div>
  </div>
</div>

<script>
let ws = null;
let mediaStream = null;
let audioContext = null;
let processor = null;
let analyser = null;
let animFrame = null;
let isListening = false;
let ttsQueue = [];
let isPlayingTTS = false;

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`;

function setStatus(text, state) {
  document.getElementById('statusText').textContent = text;
  document.getElementById('statusDot').className = 'status-dot' + (state ? ' ' + state : '');
}

function scrollToBottom(el) {
  requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
}

function addMessage(role, text, cls) {
  const conv = document.getElementById('conversation');
  conv.querySelectorAll('.interim').forEach(el => el.remove());

  // For assistant messages, update existing "pending" assistant bubble if present
  if (role === 'assistant') {
    const pending = conv.querySelector('.msg.assistant.pending');
    if (pending) {
      pending.classList.remove('pending');
      pending.querySelector('.msg-body').textContent = text;
      scrollToBottom(conv);
      return pending;
    }
  }

  const div = document.createElement('div');
  div.className = 'msg ' + (cls || role);
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = role === 'user' ? 'You' : 'Jarvis';
  div.appendChild(label);
  const body = document.createElement('div');
  body.className = 'msg-body';
  body.textContent = text;
  div.appendChild(body);
  conv.appendChild(div);
  scrollToBottom(conv);
  return div;
}

function updateInterim(text) {
  const conv = document.getElementById('conversation');
  let interim = conv.querySelector('.interim');
  if (!interim) {
    interim = document.createElement('div');
    interim.className = 'msg interim';
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = 'You';
    interim.appendChild(label);
    const body = document.createElement('div');
    body.className = 'interim-body';
    interim.appendChild(body);
    conv.appendChild(interim);
  }
  interim.querySelector('.interim-body').textContent = text;
  scrollToBottom(conv);
}

function addActivity(kind, text, data) {
  const feed = document.getElementById('activityFeed');
  const item = document.createElement('div');
  item.className = 'activity-item ' + kind;

  const now = new Date();
  const ts = now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});

  const labels = {tool: 'Tool Call', result: 'Tool Result', thinking: 'Thinking', intel: 'Speech Intelligence'};

  if (kind === 'intel' && data) {
    let html = `<span class="activity-time">${ts}</span><div class="activity-label">${labels[kind]}</div>`;
    html += '<div class="intel-tags">';
    if (data.sentiment) {
      const cls = data.sentiment === 'positive' ? 'sentiment-positive' :
                  data.sentiment === 'negative' ? 'sentiment-negative' : 'sentiment-neutral';
      html += `<span class="intel-tag ${cls}">${data.sentiment} ${data.sentiment_score || ''}</span>`;
    }
    if (data.intents) data.intents.forEach(i => { html += `<span class="intel-tag intent">${esc(i)}</span>`; });
    if (data.topics) data.topics.forEach(t => { html += `<span class="intel-tag topic">${esc(t)}</span>`; });
    html += '</div>';
    if (data.summary) html += `<div style="margin-top:4px;font-size:0.7rem;color:#8b949e;">${esc(data.summary)}</div>`;
    item.innerHTML = html;
  } else {
    item.innerHTML = `<span class="activity-time">${ts}</span><div class="activity-label">${labels[kind] || kind}</div>${esc(text)}`;
  }

  feed.appendChild(item);
  scrollToBottom(feed);

  // Cap at 300 items
  while (feed.children.length > 300) feed.removeChild(feed.firstChild);
}

function clearActivity() {
  document.getElementById('activityFeed').innerHTML = '';
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function updateQueueBadge(count) {
  const badge = document.getElementById('queueBadge');
  if (count > 0) {
    badge.textContent = count + ' queued';
    badge.style.display = 'inline';
  } else {
    badge.style.display = 'none';
  }
}

function connectWS() {
  if (ws && ws.readyState <= 1) return;
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => setStatus('Connected', 'active');
  ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  ws.onerror = () => setStatus('Error', '');
  ws.onclose = () => { setStatus('Disconnected', ''); setTimeout(connectWS, 2000); };
}

async function toggleVoice() {
  if (isListening) stopListening();
  else await startListening();
}

async function startListening() {
  try {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connectWS();
      await new Promise(r => setTimeout(r, 500));
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Mic requires HTTPS. Access via https://<tailscale-hostname>:' + location.port);
      return;
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
    });
    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(mediaStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    drawVisualizer();
    processor = audioContext.createScriptProcessor(2048, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);
    processor.onaudioprocess = (e) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        const f = e.inputBuffer.getChannelData(0);
        const i16 = new Int16Array(f.length);
        for (let i = 0; i < f.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(f[i] * 32767)));
        ws.send(i16.buffer);
      }
    };
    ws.send(JSON.stringify({ action: 'start' }));
    isListening = true;
    const btn = document.getElementById('btnListen');
    btn.textContent = 'Listening...';
    btn.classList.add('recording');
    setStatus('Listening', 'listening');
  } catch (err) {
    alert('Mic access error: ' + err.message);
  }
}

function stopListening() {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'stop' }));
  cleanupAudio();
  isListening = false;
  const btn = document.getElementById('btnListen');
  btn.textContent = 'Start Listening';
  btn.classList.remove('recording');
  setStatus('Connected', 'active');
}

function cleanupAudio() {
  if (processor) { processor.disconnect(); processor = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
}

function handleMessage(data) {
  switch (data.type) {
    case 'started':
      setStatus('Listening (' + data.model + ')', 'listening');
      break;
    case 'stopped':
      setStatus('Connected', 'active');
      break;
    case 'transcript':
      if (!data.is_final) updateInterim(data.text);
      break;
    case 'speech_started':
      setStatus('Hearing you...', 'listening');
      break;
    case 'utterance_end':
      document.querySelectorAll('.interim').forEach(el => el.remove());
      break;
    case 'processing':
      addMessage('user', data.text);
      // Add a pending assistant bubble immediately
      const pending = addMessage('assistant', 'Thinking...', 'assistant pending');
      pending.querySelector('.msg-body').style.opacity = '0.4';
      setStatus('Thinking...', 'processing');
      break;
    case 'activity':
      addActivity(data.kind, data.text, data.data);
      // Update pending bubble with latest thinking
      if (data.kind === 'thinking') {
        const p = document.querySelector('.msg.assistant.pending .msg-body');
        if (p) { p.textContent = data.text.substring(0, 80) + '...'; }
      }
      break;
    case 'response_text':
      addMessage('assistant', data.text);
      setStatus(isListening ? 'Listening' : 'Connected', isListening ? 'listening' : 'active');
      break;
    case 'tts_audio':
      queueTTS(data.audio_b64, data.sample_rate);
      break;
    case 'queue_status':
      updateQueueBadge(data.pending);
      break;
    case 'error':
      console.error('Server error:', data.message);
      addActivity('tool', 'Error: ' + data.message);
      break;
  }
}

function queueTTS(b64, sampleRate) {
  ttsQueue.push({ b64, sampleRate });
  if (!isPlayingTTS) playNextTTS();
}

function playNextTTS() {
  if (ttsQueue.length === 0) { isPlayingTTS = false; return; }
  isPlayingTTS = true;
  const { b64, sampleRate } = ttsQueue.shift();
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  const ctx = new AudioContext({ sampleRate: sampleRate || 24000 });
  const buffer = ctx.createBuffer(1, float32.length, sampleRate || 24000);
  buffer.getChannelData(0).set(float32);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
  source.onended = () => { ctx.close(); playNextTTS(); };
}

function sendText() {
  const input = document.getElementById('textInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  if (!ws || ws.readyState !== WebSocket.OPEN) connectWS();
  addMessage('user', text);
  ws.send(JSON.stringify({ action: 'text', text }));
  setStatus('Thinking...', 'processing');
}

function drawVisualizer() {
  const canvas = document.getElementById('visualizer');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  function draw() {
    animFrame = requestAnimationFrame(draw);
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    ctx.fillStyle = '#12121f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const bw = canvas.width / data.length;
    for (let i = 0; i < data.length; i++) {
      const h = (data[i] / 255) * canvas.height;
      ctx.fillStyle = `hsla(${(i / data.length) * 120 + 180}, 80%, 60%, 0.8)`;
      ctx.fillRect(i * bw, canvas.height - h, bw - 1, h);
    }
  }
  draw();
}

connectWS();
</script>
</body>
</html>"""
