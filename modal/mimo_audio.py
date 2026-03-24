"""MiMo-Audio on Modal — Mawa's emotional voice.

Uses MiMo-Audio's native API (not HuggingFace AutoModel).
Supports: instruct TTS, voice cloning, spoken dialogue, thinking mode.

Key features:
- instruct: "speak happily", "whisper", "speak with excitement"
- prompt_speech: clone any voice from 3s sample
- spoken_dialogue: full audio-in → audio-out with personality
- read_text_only=False: embed emotion directly in text
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
    .run_commands(
        "git clone https://github.com/XiaomiMiMo/MiMo-Audio.git /opt/mimo-audio",
        "cd /opt/mimo-audio && pip install -r requirements.txt || true",
    )
)

app = modal.App("mawa-mimo-audio", image=image)
volume = modal.Volume.from_name("mimo-audio-weights", create_if_missing=True)
MODEL_DIR = "/models"

# Mawa's personality instructions for different emotions
EMOTION_INSTRUCTS = {
    "neutral": "Speak in a warm, friendly, natural tone as a helpful assistant.",
    "happy": "Speak happily with a smile in your voice, cheerful and bright.",
    "sad": "Speak softly and gently with empathy and care.",
    "excited": "Speak with high energy and enthusiasm!",
    "concerned": "Speak with genuine concern and a caring, soft tone.",
    "laughing": "Laugh naturally, then speak with warmth and joy.",
    "whisper": "Speak in a soft, quiet whisper.",
    "serious": "Speak in a clear, serious, professional tone.",
    "comforting": "Speak very softly and reassuringly, like comforting a friend.",
}

MAWA_SYSTEM = (
    "You are Mawa, a warm and intelligent personal AI assistant. "
    "Be concise, natural, and emotionally aware. "
    "Express genuine emotion through your voice."
)


@app.cls(
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,
    volumes={MODEL_DIR: volume},
    memory=32768,
)
class MimoAudio:
    @modal.enter()
    def load_model(self):
        import sys
        import os
        from huggingface_hub import snapshot_download

        model_path = os.path.join(MODEL_DIR, "MiMo-Audio-7B-Instruct")
        tokenizer_path = os.path.join(MODEL_DIR, "MiMo-Audio-Tokenizer")

        # Download models if not cached
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print("Downloading MiMo-Audio-7B-Instruct...")
            snapshot_download("XiaomiMiMo/MiMo-Audio-7B-Instruct", local_dir=model_path)
            volume.commit()

        if not os.path.exists(os.path.join(tokenizer_path, "config.json")):
            print("Downloading MiMo-Audio-Tokenizer...")
            snapshot_download("XiaomiMiMo/MiMo-Audio-Tokenizer", local_dir=tokenizer_path)
            volume.commit()

        # Use MiMo's native class
        sys.path.insert(0, "/opt/mimo-audio")
        from src.mimo_audio.mimo_audio import MimoAudio as _MimoAudio
        print("Loading MiMo-Audio model...")
        self.model = _MimoAudio(model_path, tokenizer_path)
        print("Model ready.")

    @modal.method()
    def tts(self, text: str, emotion: str = "neutral", instruct: str = "") -> dict:
        """Text → speech with emotion.

        Uses MiMo's native instruct TTS: pass emotion/style instruction
        and it generates expressive speech.
        """
        import base64

        # Build emotion instruction
        if instruct:
            instruction = instruct
        else:
            instruction = EMOTION_INSTRUCTS.get(emotion, EMOTION_INSTRUCTS["neutral"])

        output_path = "/tmp/tts_output.wav"
        try:
            self.model.tts_sft(text, output_path, instruct=instruction)
            with open(output_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            return {"text": text, "audio_b64": audio_b64, "emotion": emotion}
        except Exception as e:
            # Fallback: try without instruct
            try:
                self.model.tts_sft(text, output_path)
                with open(output_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                return {"text": text, "audio_b64": audio_b64, "emotion": "neutral"}
            except Exception as e2:
                return {"text": text, "audio_b64": None, "error": str(e2)}

    @modal.method()
    def tts_natural(self, text_with_emotion: str) -> dict:
        """Natural instruction TTS — emotion embedded in text.

        Example: "Say breathlessly: I can't run anymore!"
        Example: "Whisper: I have a secret to tell you"
        Example: "Laugh and say: That's hilarious!"
        """
        import base64

        output_path = "/tmp/tts_natural.wav"
        try:
            self.model.tts_sft(text_with_emotion, output_path, read_text_only=False)
            with open(output_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            return {"text": text_with_emotion, "audio_b64": audio_b64}
        except Exception as e:
            return {"text": text_with_emotion, "audio_b64": None, "error": str(e)}

    @modal.method()
    def stt(self, audio_b64: str) -> dict:
        """Speech → text."""
        import base64
        import tempfile

        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        try:
            text = self.model.asr_sft(audio_path)
            return {"text": text}
        except Exception as e:
            return {"text": "", "error": str(e)}

    @modal.method()
    def understand(self, audio_b64: str, question: str = "Describe the audio.", thinking: bool = False) -> dict:
        """Audio understanding — describe, analyze, detect emotion."""
        import base64
        import tempfile

        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        try:
            text = self.model.audio_understanding_sft(audio_path, question, thinking=thinking)
            return {"text": text}
        except Exception as e:
            return {"text": "", "error": str(e)}

    @modal.method()
    def spoken_dialogue(self, audio_b64: str, system_prompt: str = "") -> dict:
        """Full spoken dialogue — audio in, Mawa audio + text out."""
        import base64
        import tempfile

        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            audio_path = f.name

        output_path = "/tmp/dialogue_output.wav"
        system = system_prompt or MAWA_SYSTEM

        try:
            text = self.model.spoken_dialogue_sft(
                audio_path, output_audio_path=output_path,
                system_prompt=system,
            )
            with open(output_path, "rb") as f:
                audio_b64_out = base64.b64encode(f.read()).decode()
            return {"text": text, "audio_b64": audio_b64_out}
        except Exception as e:
            return {"text": "", "audio_b64": None, "error": str(e)}


@app.function(
    gpu="T4", timeout=300, container_idle_timeout=180,
    volumes={MODEL_DIR: volume}, memory=32768,
)
@modal.web_endpoint(method="POST")
def api(body: dict) -> dict:
    """HTTP API — routes to tts/stt/understand/spoken_dialogue."""
    mimo = MimoAudio()
    mode = body.get("mode", "tts")

    if mode == "tts":
        return mimo.tts.remote(
            text=body.get("text", ""),
            emotion=body.get("emotion", "neutral"),
            instruct=body.get("instruct", ""),
        )
    elif mode == "tts_natural":
        return mimo.tts_natural.remote(text_with_emotion=body.get("text", ""))
    elif mode == "stt":
        return mimo.stt.remote(audio_b64=body.get("audio_b64", ""))
    elif mode == "understand":
        return mimo.understand.remote(
            audio_b64=body.get("audio_b64", ""),
            question=body.get("question", "Describe the audio."),
            thinking=body.get("thinking", False),
        )
    elif mode == "spoken_dialogue":
        return mimo.spoken_dialogue.remote(
            audio_b64=body.get("audio_b64", ""),
            system_prompt=body.get("system_prompt", ""),
        )
    return {"error": f"Unknown mode: {mode}"}
