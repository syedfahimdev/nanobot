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


def _strip_markdown(text: str) -> str:
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
        self._streamed_text.pop(session_id, None)
        await broadcast({
            "type": "response_text",
            "text": msg.content,
            "subagent_result": is_subagent,
        })

        # Update pending count
        count = self._pending_count.get(session_id, 0)
        if count > 0:
            self._pending_count[session_id] = count - 1
            remaining = count - 1
            if remaining > 0:
                await broadcast({"type": "queue_status", "pending": remaining})

        # TTS: subagent results and non-streamed responses need sentence-based TTS here.
        # When streaming was active, sentences were already sent via _tts_sentence messages,
        # so we only TTS if this is a subagent result (skip-summarization path) or
        # the response came from a non-streaming path.
        if is_subagent:
            clean = _strip_markdown(msg.content)
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
            audio = await self._deepgram_tts(text)
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
        return web.FileResponse(dist / "index.html")

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
            return web.json_response({"profiles": {}, "error": str(e)})

    # ── WebSocket handler ──────────────────────────────────────────

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
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

                        # Support client-specified encoding (for Opus codec)
                        client_encoding = data.get("encoding", "linear16")
                        client_sample_rate = data.get("sample_rate", "16000")

                        params = {
                            "model": self.config.stt_model,
                            "language": data.get("language", "en"),
                            "smart_format": "true",
                            "punctuate": "true",
                            "interim_results": "true",
                            "utterance_end_ms": "1000",
                            "vad_events": "true",
                            "encoding": client_encoding,
                            "sample_rate": client_sample_rate,
                            "channels": "1",
                            "endpointing": "400",
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
                                                        await ws.send_json({
                                                            "type": "transcript",
                                                            "is_final": is_final,
                                                            "text": text,
                                                            "confidence": round(confidence, 3),
                                                        })
                                                        if is_final and confidence > 0.5:
                                                            self._utterance_buffer[session_id].append(text)

                                            elif msg_type == "SpeechStarted":
                                                await ws.send_json({"type": "speech_started"})

                                            elif msg_type == "UtteranceEnd":
                                                await ws.send_json({"type": "utterance_end"})
                                                buf = self._utterance_buffer.get(session_id, [])
                                                if buf:
                                                    full_text = " ".join(buf)
                                                    self._utterance_buffer[session_id] = []
                                                    self._enqueue_message(session_id, full_text)

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

                    elif action == "submit_now":
                        # Push-to-talk release: immediately submit buffered utterance
                        buf = self._utterance_buffer.get(session_id, [])
                        if buf:
                            full_text = " ".join(buf)
                            self._utterance_buffer[session_id] = []
                            self._enqueue_message(session_id, full_text)

                    elif action == "text":
                        text = data.get("text", "").strip()
                        if text:
                            self._enqueue_message(session_id, text)

                    # Forward Deepgram keepalive to STT WebSocket
                    elif data.get("type") == "KeepAlive" and dg_ws:
                        try:
                            await dg_ws.send('{"type": "KeepAlive"}')
                        except Exception:
                            pass

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
            logger.info("Web Voice client disconnected: {}", session_id)

        return ws

    # ── Message queue (serialize per session) ──────────────────────

    def _enqueue_message(self, session_id: str, text: str) -> None:
        """Submit a message to the agent immediately (no queuing).

        The agent loop handles concurrency: if it's already busy on this session,
        the message gets live-injected into the running conversation so the LLM
        sees it on its next iteration.

        Interrupt patterns ('no', 'stop', 'wait') still cancel via /stop.
        """
        is_interrupt = bool(_INTERRUPT_PATTERNS.search(text.strip()))

        if is_interrupt:
            asyncio.create_task(self._send_stop(session_id))
            self._pending_count[session_id] = 0
            ws = self._clients.get(session_id)
            if ws and not ws.closed:
                asyncio.create_task(ws.send_json({"type": "queue_status", "pending": 0}))

        # Fire directly to the agent — no channel-level queue
        asyncio.create_task(self._submit_to_agent(session_id, text))

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

    async def _submit_to_agent(self, session_id: str, text: str) -> None:
        logger.info("Web Voice heard: '{}'", text)

        ws = self._clients.get(session_id)
        if ws and not ws.closed:
            await ws.send_json({"type": "processing", "text": text})

        # Reset latency tracking for this request
        self._activity_ts[session_id] = time.time()

        # Fire-and-forget: intel goes to activity feed + stashed for LLM enrichment
        # Runs concurrently with the agent loop's preflight — zero added latency
        if len(text.split()) >= 4:
            asyncio.create_task(self._get_intel_for_llm(session_id, text))

        # Use fixed chat_id so all devices share the same nanobot session
        shared_chat_id = "voice"

        await self._handle_message(
            sender_id=session_id,
            chat_id=shared_chat_id,
            content=text,
            metadata={
                "source": "voice",
                "_ws_session_id": session_id,
            },
        )

    async def _get_intel_for_llm(self, session_id: str, text: str) -> str | None:
        """Analyze speech, send to activity feed, and return summary for LLM context."""
        try:
            intel = await self._deepgram_analyze(text)
            logger.debug("Web Voice intel for '{}': {}", text[:40], intel)
            if intel:
                # Send to browser activity feed
                ws = self._clients.get(session_id)
                if ws and not ws.closed:
                    await ws.send_json({"type": "activity", "kind": "intel", "data": intel})
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
