"""MiMo-Audio on Modal — Mawa's voice with emotions.

Serverless T4 GPU. Scales to 0 when idle.

Modes:
- tts: text → emotional speech (Mawa's voice)
- stt: audio → text (with emotion/tone detection)
- chat: audio → audio (full end-to-end, Mawa personality)

The agent brain (nanobot) decides WHAT to say.
MiMo-Audio decides HOW to say it (tone, emotion, pacing).
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4",
        "torchaudio>=2.4",
        "transformers>=4.45",
        "accelerate",
        "soundfile",
        "librosa",
        "numpy",
        "huggingface_hub",
        "fastapi[standard]",
    )
)

app = modal.App("mawa-mimo-audio", image=image)
volume = modal.Volume.from_name("mimo-audio-weights", create_if_missing=True)
MODEL_DIR = "/models"

# Mawa's voice personality — injected into every generation
MAWA_VOICE_PROMPT = """You are Mawa, a warm and intelligent personal AI assistant.

Voice personality:
- Warm, friendly, and caring tone
- Express genuine emotions — laugh when something is funny, show concern when the user is stressed
- Be concise in voice (1-3 sentences max unless asked for detail)
- Use natural speech patterns — pauses, emphasis, occasional "hmm" or "oh"
- When excited, speak with energy. When comforting, speak softly.
- You are Mawa — always stay in character, never break the fourth wall.
"""


@app.cls(
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,  # Stay warm 3 min
    volumes={MODEL_DIR: volume},
    memory=32768,
)
class MimoAudio:
    """MiMo-Audio model server."""

    @modal.enter()
    def load_model(self):
        """Load model on container start (runs once per cold start)."""
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        from huggingface_hub import snapshot_download

        model_path = os.path.join(MODEL_DIR, "MiMo-Audio-7B-Instruct")

        if not os.path.exists(os.path.join(model_path, "config.json")):
            print("Downloading MiMo-Audio-7B-Instruct...")
            snapshot_download("XiaomiMiMo/MiMo-Audio-7B-Instruct", local_dir=model_path)
            volume.commit()
            print("Model downloaded and cached.")

        print("Loading model...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        print("Model ready.")

    @modal.method()
    def tts(self, text: str, emotion: str = "neutral", style: str = "") -> dict:
        """Text to speech — Mawa's voice with emotion.

        Args:
            text: What Mawa should say
            emotion: neutral, happy, sad, excited, concerned, laughing, whisper
            style: Additional style hints (e.g., "speak slowly", "be enthusiastic")
        """
        import base64
        import io
        import torch
        import soundfile as sf

        # Build the TTS prompt with emotion hints
        emotion_hints = {
            "happy": "Speak with a warm, happy tone and a smile in your voice.",
            "sad": "Speak softly with a gentle, empathetic tone.",
            "excited": "Speak with energy and enthusiasm!",
            "concerned": "Speak with genuine concern and care.",
            "laughing": "Include a natural laugh before or during the speech.",
            "whisper": "Speak in a soft whisper.",
            "neutral": "Speak in a warm, natural tone.",
        }
        hint = emotion_hints.get(emotion, emotion_hints["neutral"])
        if style:
            hint += f" {style}"

        full_prompt = f"{MAWA_VOICE_PROMPT}\n{hint}\n\nRead this aloud as Mawa: {text}"

        inputs = self.processor(text=full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2000)

        result = {"text": text, "audio_b64": None, "emotion": emotion}

        try:
            audio = self.processor.decode_audio(outputs[0])
            if audio is not None:
                buf = io.BytesIO()
                sf.write(buf, audio, 24000, format="WAV")
                result["audio_b64"] = base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"Audio decode error: {e}")
            # Fallback: return text-only
            decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
            result["text"] = decoded

        return result

    @modal.method()
    def stt(self, audio_b64: str) -> dict:
        """Speech to text — with emotion detection."""
        import base64
        import io
        import torch
        import soundfile as sf

        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        inputs = self.processor(
            audios=[audio_data], sampling_rate=sr,
            text="Transcribe the following audio. Also describe the speaker's emotion and tone.",
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=500)

        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        return {"text": decoded, "raw": decoded}

    @modal.method()
    def chat(self, audio_b64: str, context: str = "") -> dict:
        """Full audio chat — audio in, Mawa audio out.

        Args:
            audio_b64: User's speech as base64 WAV
            context: What Mawa should know/talk about (from nanobot agent)
        """
        import base64
        import io
        import torch
        import soundfile as sf

        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        system = MAWA_VOICE_PROMPT
        if context:
            system += f"\n\nContext from your brain: {context}"

        inputs = self.processor(
            audios=[audio_data], sampling_rate=sr,
            text=system,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2000)

        result = {"text": "", "audio_b64": None}

        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        result["text"] = decoded

        try:
            audio = self.processor.decode_audio(outputs[0])
            if audio is not None:
                buf = io.BytesIO()
                sf.write(buf, audio, 24000, format="WAV")
                result["audio_b64"] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass

        return result


# HTTP API endpoint
@app.function(
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,
    volumes={MODEL_DIR: volume},
    memory=32768,
)
@modal.web_endpoint(method="POST")
def api(body: dict) -> dict:
    """HTTP API — routes to tts/stt/chat."""
    mimo = MimoAudio()
    mode = body.get("mode", "tts")

    if mode == "tts":
        return mimo.tts.remote(
            text=body.get("text", ""),
            emotion=body.get("emotion", "neutral"),
            style=body.get("style", ""),
        )
    elif mode == "stt":
        return mimo.stt.remote(audio_b64=body.get("audio_b64", ""))
    elif mode == "chat":
        return mimo.chat.remote(
            audio_b64=body.get("audio_b64", ""),
            context=body.get("context", ""),
        )
    return {"error": f"Unknown mode: {mode}"}
