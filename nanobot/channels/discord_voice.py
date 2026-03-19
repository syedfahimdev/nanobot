"""Discord Voice channel — auto-joins voice, STT via Deepgram, TTS playback.

Uses discord.py + davey (DAVE protocol) for voice gateway.
Uses discord-ext-voice-recv for audio capture.
Audio flow:
  User speaks → VoiceRecvClient/BasicSink → silence detection → Deepgram REST STT
  → InboundMessage → Agent Loop → OutboundMessage → Deepgram REST TTS → FFmpeg → Discord
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import struct
import time as _time
import wave
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base

_dave_log = logging.getLogger("discord_voice.dave")


def _patch_voice_recv_for_dave():
    """Monkey-patch voice_recv's opus decoder to DAVE-decrypt before opus decode.

    discord-ext-voice-recv does transport decryption (nacl) in the AudioReader callback,
    then feeds packets to PacketRouter → SsrcAudioDecoder which does opus decode.
    With DAVE, the opus payload is still E2EE after transport decryption.
    We patch _decode_packet in SsrcAudioDecoder to DAVE-decrypt first.
    """
    try:
        from discord.ext.voice_recv.opus import PacketDecoder
        import davey

        _orig_decode = PacketDecoder._decode_packet

        _stats = {"ok": 0, "fail": 0, "no_dave": 0, "no_uid": 0, "logged": False}

        def _patched_decode(self, packet):
            """DAVE-decrypt before opus decode."""
            did_decrypt = False
            if packet and hasattr(packet, 'decrypted_data') and packet.decrypted_data:
                vc = getattr(self.sink, '_voice_client', None) or getattr(self.sink, 'voice_client', None)
                if vc:
                    dave_session = getattr(getattr(vc, '_connection', None), 'dave_session', None)
                    if dave_session:
                        user_id = None
                        try:
                            if hasattr(vc, '_ssrc_to_id') and self.ssrc in vc._ssrc_to_id:
                                user_id = vc._get_id_from_ssrc(self.ssrc)
                            elif hasattr(self, '_cached_id') and self._cached_id:
                                user_id = self._cached_id
                        except Exception:
                            pass

                        if user_id:
                            try:
                                decrypted = dave_session.decrypt(
                                    user_id, davey.MediaType.audio, packet.decrypted_data
                                )
                                if decrypted:
                                    packet.decrypted_data = decrypted
                                    did_decrypt = True
                            except Exception:
                                pass
                        else:
                            _stats["no_uid"] += 1
                    else:
                        _stats["no_dave"] += 1

            try:
                result = _orig_decode(self, packet)
                _stats["ok"] += 1
                return result
            except Exception:
                _stats["fail"] += 1
                total = _stats["ok"] + _stats["fail"]
                if total % 50 == 0 or (total <= 10):
                    _dave_log.warning(
                        "DAVE stats: ok=%d fail=%d no_dave=%d no_uid=%d (this: decrypt=%s)",
                        _stats["ok"], _stats["fail"], _stats["no_dave"], _stats["no_uid"], did_decrypt,
                    )
                return packet, b'\x00' * 3840

        PacketDecoder._decode_packet = _patched_decode
        _dave_log.info("voice_recv opus decoder patched for DAVE decryption")
    except ImportError:
        pass
    except Exception as e:
        _dave_log.error("Failed to patch voice_recv for DAVE: %s", e)


_patch_voice_recv_for_dave()


# Audio constants
SAMPLE_RATE = 48000
BYTES_PER_SAMPLE = 2          # 16-bit PCM
CHANNELS = 2                  # Discord stereo
SILENCE_DURATION_S = 1.5      # seconds of silence before processing
MIN_SPEECH_S = 0.3            # minimum speech duration to process
VAD_AGGRESSIVENESS = 2        # 0-3, higher = more aggressive filtering

# WebRTC VAD setup (per-thread instances via _get_vad)
_vad_instance = None

def _get_vad():
    global _vad_instance
    if _vad_instance is None:
        try:
            import webrtcvad
            _vad_instance = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        except ImportError:
            pass
    return _vad_instance

def _stereo_to_mono(pcm: bytes) -> bytes:
    """Convert stereo 16-bit PCM to mono by averaging channels."""
    count = len(pcm) // 4  # 4 bytes per stereo sample (2 channels * 2 bytes)
    if count == 0:
        return b''
    samples = struct.unpack(f"<{count * 2}h", pcm[:count * 4])
    mono = struct.pack(f"<{count}h", *((samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)))
    return mono

def _is_speech(pcm_stereo: bytes) -> bool:
    """Check if audio frame contains speech using WebRTC VAD."""
    vad = _get_vad()
    if vad is None:
        # Fallback to RMS if webrtcvad not available
        return _compute_rms(pcm_stereo) > 300
    try:
        mono = _stereo_to_mono(pcm_stereo)
        # webrtcvad needs 10/20/30ms frames. Discord sends 20ms at 48kHz = 960 mono samples = 1920 bytes
        if len(mono) == 1920:
            return vad.is_speech(mono, SAMPLE_RATE)
        # If frame size doesn't match, check in 20ms chunks
        frame_bytes = 1920  # 20ms at 48kHz mono
        speech_frames = 0
        total_frames = 0
        for i in range(0, len(mono) - frame_bytes + 1, frame_bytes):
            total_frames += 1
            if vad.is_speech(mono[i:i + frame_bytes], SAMPLE_RATE):
                speech_frames += 1
        return speech_frames > total_frames // 2 if total_frames > 0 else False
    except Exception:
        return _compute_rms(pcm_stereo) > 300


def _generate_chime() -> bytes:
    """Generate a short two-tone chime as 48kHz stereo 16-bit PCM.

    Returns raw PCM bytes suitable for discord.FFmpegPCMAudio with
    before_options="-f s16le -ar 48000 -ac 2".
    """
    import math
    sr = 48000
    duration = 0.15  # seconds per tone
    freqs = [880, 1108]  # A5 → C#6 (pleasant rising interval)
    samples = []
    for freq in freqs:
        n = int(sr * duration)
        for i in range(n):
            # Apply fade envelope to avoid clicks
            env = min(1.0, i / (sr * 0.01)) * min(1.0, (n - i) / (sr * 0.01))
            val = int(12000 * env * math.sin(2 * math.pi * freq * i / sr))
            val = max(-32768, min(32767, val))
            samples.append(val)  # L
            samples.append(val)  # R
    return struct.pack(f"<{len(samples)}h", *samples)

# Cache the chime so we don't regenerate it every time
_CHIME_PCM: bytes | None = None

def _get_chime() -> bytes:
    global _CHIME_PCM
    if _CHIME_PCM is None:
        _CHIME_PCM = _generate_chime()
    return _CHIME_PCM


class DiscordVoiceConfig(Base):
    """Discord voice channel configuration."""

    enabled: bool = False
    token: str = ""
    deepgram_api_key: str = ""
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    auto_join_mode: Literal["anyone", "tracked", "off"] = "anyone"
    tracked_users: list[str] = Field(default_factory=list)
    tts_model: str = "aura-2-luna-en"
    stt_model: str = "nova-3"


class DiscordVoiceChannel(BaseChannel):
    """Discord voice channel using discord.py + voice_recv + Deepgram STT/TTS."""

    name = "discord_voice"
    display_name = "Discord Voice"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return DiscordVoiceConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = DiscordVoiceConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: DiscordVoiceConfig = config

        self._bot = None
        self._bot_task: asyncio.Task | None = None
        self._voice_clients: dict[int, Any] = {}       # guild_id -> VoiceRecvClient
        self._listening: dict[int, bool] = {}
        self._tracked: set[int] = {int(u) for u in config.tracked_users if u.isdigit()}
        self._guild_text_channels: dict[int, str] = {}  # guild_id -> text channel id
        self._tts_queues: dict[int, asyncio.Queue] = {}
        self._paused: dict[int, bool] = {}              # guild_id -> TTS playing

        # Per-user audio buffers for silence detection
        self._user_buffers: dict[int, bytearray] = {}
        self._user_silence_start: dict[int, float] = {}
        self._user_guild: dict[int, int] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        if not self.config.token:
            logger.error("Discord Voice: bot token not configured")
            return
        if not self.config.deepgram_api_key:
            logger.error("Discord Voice: deepgram_api_key not configured")
            return

        self._running = True
        self._loop = asyncio.get_event_loop()
        self._bot_task = asyncio.create_task(self._run_bot())
        await self._bot_task

    async def stop(self) -> None:
        self._running = False
        for guild_id in list(self._voice_clients):
            await self._leave_voice(guild_id)
        if self._bot and not self._bot.is_closed():
            await self._bot.close()
        if self._bot_task:
            self._bot_task.cancel()

    async def send(self, msg: OutboundMessage) -> None:
        """Receive outbound from agent → TTS → play in voice channel."""
        if not msg.content:
            return

        meta = msg.metadata or {}
        if meta.get("_progress"):
            # Skip tool hints (technical), but allow thought progress (natural language)
            if meta.get("_tool_hint"):
                return

        guild_id = None
        for gid, ch_id in self._guild_text_channels.items():
            if ch_id == msg.chat_id:
                guild_id = gid
                break

        if guild_id is None:
            try:
                guild_id = int(msg.chat_id)
            except (ValueError, TypeError):
                return

        if guild_id not in self._voice_clients:
            return

        q = self._tts_queues.get(guild_id)
        if q is None:
            q = asyncio.Queue()
            self._tts_queues[guild_id] = q
            asyncio.create_task(self._tts_worker(guild_id))
        await q.put(msg.content)

    # ── Bot lifecycle ───────────────────────────────────────────────

    async def _run_bot(self) -> None:
        try:
            import discord
        except ImportError:
            logger.error("Discord Voice: discord.py not installed")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True

        bot = discord.Client(intents=intents)
        self._bot = bot
        parent = self

        @bot.event
        async def on_ready():
            logger.info("Discord Voice bot connected as {}", bot.user)

        @bot.event
        async def on_voice_state_update(member, before, after):
            if member.bot:
                return
            guild_id = member.guild.id

            if before.channel is None and after.channel is not None:
                if parent._should_auto_join(member):
                    await parent._join_voice(after.channel, guild_id)
            elif before.channel is not None and after.channel is None:
                await parent._check_and_leave(guild_id)
            elif before.channel != after.channel and after.channel is not None:
                vc = parent._voice_clients.get(guild_id)
                if vc and vc.is_connected():
                    await vc.move_to(after.channel)

        @bot.event
        async def on_message(message):
            pass

        try:
            await bot.start(self.config.token)
        except Exception as e:
            logger.error("Discord Voice bot error: {}", e)

    # ── Voice join/leave ────────────────────────────────────────────

    def _should_auto_join(self, member) -> bool:
        mode = self.config.auto_join_mode
        if mode == "off":
            return False
        if mode == "tracked":
            return member.id in self._tracked
        return True

    async def _join_voice(self, channel, guild_id: int) -> None:
        if guild_id in self._voice_clients:
            vc = self._voice_clients[guild_id]
            if vc.is_connected() and vc.channel == channel:
                return
            if vc.is_connected():
                await vc.move_to(channel)
                return

        try:
            from discord.ext.voice_recv import VoiceRecvClient

            vc = await channel.connect(cls=VoiceRecvClient, timeout=15.0)
            self._voice_clients[guild_id] = vc
            self._listening[guild_id] = True

            guild = channel.guild
            for ch in guild.text_channels:
                try:
                    if ch.permissions_for(guild.me).send_messages:
                        self._guild_text_channels[guild_id] = str(ch.id)
                        break
                except Exception:
                    continue

            if guild_id not in self._guild_text_channels:
                if guild.text_channels:
                    self._guild_text_channels[guild_id] = str(guild.text_channels[0].id)

            logger.info("Discord Voice: joined #{} (text_ch={})", channel.name, self._guild_text_channels.get(guild_id, "none"))
            self._start_listening(guild_id)
        except Exception as e:
            logger.error("Discord Voice: failed to join #{}: {}: {}", channel.name, type(e).__name__, e)
            self._voice_clients.pop(guild_id, None)
            self._listening.pop(guild_id, None)

    async def _leave_voice(self, guild_id: int) -> None:
        self._listening[guild_id] = False
        vc = self._voice_clients.pop(guild_id, None)
        self._guild_text_channels.pop(guild_id, None)
        self._tts_queues.pop(guild_id, None)
        self._paused.pop(guild_id, None)
        if vc and vc.is_connected():
            try:
                if vc.is_listening():
                    vc.stop_listening()
            except Exception:
                pass
            await vc.disconnect()

    async def _check_and_leave(self, guild_id: int) -> None:
        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return
        humans = [m for m in vc.channel.members if not m.bot]
        if not humans:
            logger.info("Discord Voice: no humans left, disconnecting")
            await self._leave_voice(guild_id)

    # ── Audio capture via voice_recv ────────────────────────────────

    def _start_listening(self, guild_id: int) -> None:
        """Start listening for audio using VoiceRecvClient + BasicSink."""
        vc = self._voice_clients.get(guild_id)
        if not vc:
            return

        try:
            from discord.ext.voice_recv import BasicSink

            parent = self

            _audio_dbg = {"calls": 0, "speech": 0, "logged": 0}

            def on_audio(user, voice_data):
                """Called from voice thread for each decoded audio packet."""
                if user is None or not parent._listening.get(guild_id, False):
                    return
                if parent._paused.get(guild_id, False):
                    return

                pcm = voice_data.pcm
                if not pcm:
                    return

                _audio_dbg["calls"] += 1

                uid = user.id
                parent._user_guild[uid] = guild_id

                if uid not in parent._user_buffers:
                    parent._user_buffers[uid] = bytearray()
                    parent._user_silence_start[uid] = 0.0

                speech = _is_speech(pcm)
                if speech:
                    _audio_dbg["speech"] += 1

                # Debug log every 500 packets
                if _audio_dbg["calls"] % 500 == 0 and _audio_dbg["logged"] < 20:
                    _audio_dbg["logged"] += 1
                    rms = _compute_rms(pcm)
                    buf_kb = len(parent._user_buffers.get(uid, b'')) // 1024
                    _dave_log.info(
                        "on_audio: calls=%d speech=%d rms=%.0f buf=%dKB pcm_len=%d",
                        _audio_dbg["calls"], _audio_dbg["speech"], rms, buf_kb, len(pcm),
                    )

                if not speech:
                    if len(parent._user_buffers[uid]) > 0:
                        now = _time.monotonic()
                        if parent._user_silence_start[uid] == 0.0:
                            parent._user_silence_start[uid] = now
                        elif now - parent._user_silence_start[uid] >= SILENCE_DURATION_S:
                            audio = bytes(parent._user_buffers[uid])
                            parent._user_buffers[uid] = bytearray()
                            parent._user_silence_start[uid] = 0.0
                            min_bytes = int(SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * MIN_SPEECH_S)
                            if len(audio) > min_bytes and parent._loop:
                                asyncio.run_coroutine_threadsafe(
                                    parent._process_audio(guild_id, uid, audio),
                                    parent._loop,
                                )
                else:
                    parent._user_silence_start[uid] = 0.0
                    parent._user_buffers[uid].extend(pcm)

                    max_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * 30
                    if len(parent._user_buffers[uid]) > max_bytes:
                        audio = bytes(parent._user_buffers[uid])
                        parent._user_buffers[uid] = bytearray()
                        if parent._loop:
                            asyncio.run_coroutine_threadsafe(
                                parent._process_audio(guild_id, uid, audio),
                                parent._loop,
                            )

            sink = BasicSink(on_audio)
            vc.listen(sink)
            logger.info("Discord Voice: listening for audio in guild {}", guild_id)

        except Exception as e:
            logger.error("Discord Voice: failed to start listening: {}: {}", type(e).__name__, e)

    # ── STT via Deepgram REST ───────────────────────────────────────

    async def _process_audio(self, guild_id: int, user_id: int, pcm_data: bytes) -> None:
        logger.debug("Discord Voice: processing {}KB audio from user {}", len(pcm_data) // 1024, user_id)

        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(BYTES_PER_SAMPLE)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm_data)
            wav_bytes = wav_buffer.getvalue()

            text = await self._deepgram_stt(wav_bytes)
            if text:
                logger.info("Discord Voice heard from {}: '{}'", user_id, text)
                # Run text intelligence in parallel with speech handling
                intel_task = asyncio.create_task(self._deepgram_analyze(text))
                await self._handle_speech(guild_id, user_id, text, intel_task)
        except Exception as e:
            logger.error("Discord Voice: audio processing error: {}", e)

    async def _deepgram_stt(self, wav_bytes: bytes) -> str | None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api.deepgram.com/v1/listen",
                    headers={
                        "Authorization": f"Token {self.config.deepgram_api_key}",
                        "Content-Type": "audio/wav",
                    },
                    params={
                        "model": self.config.stt_model,
                        "language": "en",
                        "smart_format": "true",
                        "punctuate": "true",
                    },
                    content=wav_bytes,
                )
                resp.raise_for_status()
                data = resp.json()

                channels = data.get("results", {}).get("channels", [])
                if channels:
                    alts = channels[0].get("alternatives", [])
                    if alts:
                        text = alts[0].get("transcript", "").strip()
                        confidence = alts[0].get("confidence", 0)
                        if text and confidence > 0.5:
                            return text
                return None
        except Exception as e:
            logger.error("Discord Voice STT failed: {}", e)
            return None

    async def _deepgram_analyze(self, text: str) -> str | None:
        """Analyze text using Deepgram Text Intelligence for intent/sentiment."""
        if not text or len(text) < 3:
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
                        "intents": "true",
                        "sentiment": "true",
                        "summarize": "v2",
                        "topics": "true",
                    },
                    json={"text": text},
                )
                resp.raise_for_status()
                data = resp.json()

                parts = []
                results = data.get("results", {})

                # Extract intents
                intents = results.get("intents", {}).get("segments", [])
                if intents:
                    top_intents = [s["intent"] for s in intents[:3] if s.get("intent")]
                    if top_intents:
                        parts.append(f"Intents: {', '.join(top_intents)}")

                # Extract sentiment
                sentiments = results.get("sentiments", {}).get("average", {})
                if sentiments:
                    sentiment = sentiments.get("sentiment", "")
                    confidence = sentiments.get("sentiment_score", 0)
                    if sentiment:
                        parts.append(f"Sentiment: {sentiment} ({confidence:.1f})")

                # Extract topics
                topics = results.get("topics", {}).get("segments", [])
                if topics:
                    top_topics = [s["topic"] for s in topics[:3] if s.get("topic")]
                    if top_topics:
                        parts.append(f"Topics: {', '.join(top_topics)}")

                result = " | ".join(parts) if parts else None
                if result:
                    logger.debug("Discord Voice intel: {}", result)
                return result

        except Exception as e:
            logger.debug("Discord Voice text intel failed: {}", e)
            return None

    async def _handle_speech(
        self, guild_id: int, user_id: int, text: str,
        intel_task: asyncio.Task | None = None,
    ) -> None:
        chat_id = self._guild_text_channels.get(guild_id, str(guild_id))

        # Play a short chime so user knows they were heard
        await self._play_chime(guild_id)

        # Gather text intelligence if available
        intel_context = ""
        if intel_task:
            try:
                intel = await asyncio.wait_for(intel_task, timeout=3.0)
                if intel:
                    intel_context = f"\n[Speech Analysis] {intel}"
            except (asyncio.TimeoutError, Exception):
                pass

        await self._handle_message(
            sender_id=str(user_id),
            chat_id=chat_id,
            content=text,
            metadata={
                "source": "voice",
                "guild_id": str(guild_id),
                "voice_instruction": (
                    "[VOICE MODE] This message came from voice input and will be spoken aloud via TTS. "
                    "IMPORTANT RULES:\n"
                    "1. ALWAYS say a brief acknowledgment BEFORE making any tool calls "
                    "(e.g. 'Sure, let me check your email.' or 'One moment, pulling up your calendar.'). "
                    "This is critical — the user is waiting in a voice channel and needs to know you heard them.\n"
                    "2. Respond in plain conversational English — no markdown, no tables, "
                    "no bullet points, no emojis, no code blocks, no special formatting.\n"
                    "3. Keep your final answer brief and natural, as if speaking to someone. "
                    "Max 2-3 sentences unless more detail is explicitly requested.\n"
                    "4. For lists, speak them naturally (e.g. 'You have three emails: first from John about..., "
                    "second from...' instead of bullet points)."
                    + intel_context
                ),
            },
        )

    async def _play_chime(self, guild_id: int) -> None:
        """Play a short notification chime to acknowledge speech was heard."""
        import discord

        vc = self._voice_clients.get(guild_id)
        if not vc or not vc.is_connected():
            return
        try:
            chime = _get_chime()
            source = discord.FFmpegPCMAudio(
                io.BytesIO(chime),
                pipe=True,
                before_options="-f s16le -ar 48000 -ac 2",
            )

            if vc.is_playing():
                vc.stop()
                await asyncio.sleep(0.05)

            loop = asyncio.get_running_loop()
            done = asyncio.Event()
            vc.play(source, after=lambda e: loop.call_soon_threadsafe(done.set))
            await asyncio.wait_for(done.wait(), timeout=5.0)
        except Exception as e:
            logger.debug("Discord Voice: chime failed: {}", e)

    async def _queue_tts(self, guild_id: int, text: str) -> None:
        """Queue text for TTS playback in a guild's voice channel."""
        if guild_id not in self._voice_clients:
            return
        q = self._tts_queues.get(guild_id)
        if q is None:
            q = asyncio.Queue()
            self._tts_queues[guild_id] = q
            asyncio.create_task(self._tts_worker(guild_id))
        await q.put(text)

    # ── TTS via Deepgram REST ───────────────────────────────────────

    async def _tts_worker(self, guild_id: int) -> None:
        q = self._tts_queues[guild_id]
        while self._listening.get(guild_id, False):
            try:
                text = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            vc = self._voice_clients.get(guild_id)
            if not vc or not vc.is_connected():
                continue

            self._paused[guild_id] = True
            try:
                audio_bytes = await self._deepgram_tts(text)
                if audio_bytes:
                    await self._play_audio(vc, audio_bytes)
                else:
                    logger.warning("Discord Voice: TTS returned no audio for: {}...", text[:50])
            except Exception as e:
                logger.error("Discord Voice TTS error: {}", e)
            finally:
                self._paused[guild_id] = False

    async def _deepgram_tts(self, text: str) -> bytes | None:
        if len(text) > 2000:
            text = text[:2000] + "... truncated for voice."

        # Strip markdown formatting for natural TTS
        text = re.sub(r"```[\s\S]*?```", " ", text)       # code blocks
        text = re.sub(r"`[^`]+`", "", text)                # inline code
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)   # images
        text = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)    # links
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)  # table rows
        text = re.sub(r"^[-|: ]+$", "", text, flags=re.MULTILINE)  # table separators
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # headings
        text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)   # list markers
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)   # numbered lists
        text = re.sub(r"[*_~`#>\[\]()]", "", text)         # remaining markdown chars
        text = re.sub(r"\n{2,}", ". ", text)
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\s{2,}", " ", text)                # collapse whitespace
        if not text:
            return None

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
                if len(audio) < 100:
                    logger.warning("Discord Voice: TTS response too small ({} bytes)", len(audio))
                    return None
                return audio
        except Exception as e:
            logger.error("Discord Voice TTS failed: {}", e)
            return None

    async def _play_audio(self, vc, audio_bytes: bytes) -> None:
        import discord

        try:
            source = discord.FFmpegPCMAudio(
                io.BytesIO(audio_bytes),
                pipe=True,
                before_options="-f s16le -ar 24000 -ac 1",
            )

            if vc.is_playing():
                vc.stop()
                await asyncio.sleep(0.1)

            loop = asyncio.get_running_loop()
            done = asyncio.Event()

            def after_play(error):
                if error:
                    logger.error("Discord Voice playback error: {}", error)
                loop.call_soon_threadsafe(done.set)

            vc.play(source, after=after_play)
            logger.info("Discord Voice: playing TTS audio ({} bytes)", len(audio_bytes))
            try:
                await asyncio.wait_for(done.wait(), timeout=120.0)
                logger.debug("Discord Voice: playback finished")
            except asyncio.TimeoutError:
                logger.warning("Discord Voice: playback timed out after 120s")
                if vc.is_playing():
                    vc.stop()
        except Exception as e:
            logger.error("Discord Voice playback failed: {}", e)


def _compute_rms(data: bytes) -> float:
    if len(data) < 2:
        return 0.0
    count = len(data) // 2
    try:
        samples = struct.unpack(f"<{count}h", data[:count * 2])
        if not samples:
            return 0.0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5
    except struct.error:
        return 0.0
