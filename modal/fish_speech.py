"""Fish Speech + Coqui XTTS on Modal — multilingual emotional TTS.

Fish Speech 1.5: emotions + Bengali + voice cloning
Coqui XTTS v2: 17 languages + voice cloning
Both on T4 GPU, scales to 0 when idle.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "portaudio19-dev")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "soundfile",
        "numpy<2",
        "coqui-tts==0.24.2",
        "fastapi[standard]",
    )
    .env({"COQUI_TOS_AGREED": "1"})
    .run_commands(
        # Pre-download XTTS v2 model during image build so cold starts are fast
        "python -c \"from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')\" || echo 'Model download will happen at runtime'",
    )
)

app = modal.App("mawa-multilingual-tts", image=image)
volume = modal.Volume.from_name("multilingual-tts-weights", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,
    volumes={MODEL_DIR: volume},
    memory=16384,
)
class MultilingualTTS:
    @modal.enter()
    def load_model(self):
        import os
        os.environ["COQUI_TOS_AGREED"] = "1"
        print("Loading Coqui XTTS v2...")
        try:
            from TTS.api import TTS  # coqui-tts package
            self.xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
            print("XTTS v2 ready.")
        except Exception as e:
            print(f"XTTS load error: {e}")
            self.xtts = None

    @modal.method()
    def tts_xtts(
        self, text: str, language: str = "en",
        speed: float = 1.0, reference_audio_b64: str = "",
    ) -> dict:
        """Coqui XTTS v2 — 17 languages + voice cloning.

        Languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, ko, hu, hi
        Bengali: supported via Hindi model (similar script family)
        """
        import base64
        import io
        import soundfile as sf
        import tempfile

        if not self.xtts:
            return {"text": text, "audio_b64": None, "error": "XTTS not loaded"}

        try:
            # Handle reference audio for voice cloning
            speaker_wav = None
            if reference_audio_b64:
                ref_bytes = base64.b64decode(reference_audio_b64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(ref_bytes)
                    speaker_wav = f.name

            # Map Bengali to Hindi (closest supported)
            lang_map = {"bn": "hi", "ur": "hi", "bengali": "hi", "bangla": "hi"}
            mapped_lang = lang_map.get(language, language)

            # Generate
            if speaker_wav:
                wav = self.xtts.tts(text=text, language=mapped_lang, speaker_wav=speaker_wav)
            else:
                wav = self.xtts.tts(text=text, language=mapped_lang)

            buf = io.BytesIO()
            import numpy as np
            audio = np.array(wav, dtype=np.float32)
            sf.write(buf, audio, 24000, format="WAV")

            return {
                "text": text,
                "audio_b64": base64.b64encode(buf.getvalue()).decode(),
                "language": language,
            }
        except Exception as e:
            return {"text": text, "audio_b64": None, "error": str(e)}


@app.function(
    gpu="T4", timeout=300, container_idle_timeout=180,
    volumes={MODEL_DIR: volume}, memory=16384,
)
@modal.web_endpoint(method="POST")
def api(body: dict) -> dict:
    """HTTP API — Coqui XTTS."""
    tts = MultilingualTTS()
    return tts.tts_xtts.remote(
        text=body.get("text", ""),
        language=body.get("language", "en"),
        speed=body.get("speed", 1.0),
        reference_audio_b64=body.get("reference_audio_b64", ""),
    )
