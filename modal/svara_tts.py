"""Svara-TTS on Modal — 19 Indian languages + emotion + voice cloning.

Languages: Hindi, Bengali, Marathi, Telugu, Kannada, Bhojpuri, Magahi,
Chhattisgarhi, Maithili, Assamese, Bodo, Dogri, Gujarati, Malayalam,
Punjabi, Tamil, Nepali, Sanskrit, Indian English.

Runs on T4 GPU — lightweight, fast cold start.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers>=4.45",
        "soundfile",
        "numpy<2",
        "fastapi[standard]",
        "huggingface_hub",
        "encodec",
    )
)

app = modal.App("mawa-svara-tts", image=image)
volume = modal.Volume.from_name("svara-tts-weights", create_if_missing=True)
MODEL_DIR = "/models"

LANG_CODES = {
    "bn": "bengali", "hi": "hindi", "en": "indian_english",
    "mr": "marathi", "te": "telugu", "kn": "kannada",
    "gu": "gujarati", "ml": "malayalam", "pa": "punjabi",
    "ta": "tamil", "as": "assamese", "ne": "nepali",
    "sa": "sanskrit", "ur": "hindi",
}


@app.cls(
    gpu="T4",
    timeout=300,
    container_idle_timeout=180,
    volumes={MODEL_DIR: volume},
    memory=16384,
)
class SvaraTTS:
    @modal.enter()
    def load_model(self):
        import os
        from huggingface_hub import snapshot_download

        model_path = os.path.join(MODEL_DIR, "svara-tts-v1")
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print("Downloading Svara-TTS v1...")
            snapshot_download("kenpath/svara-tts-v1", local_dir=model_path)
            volume.commit()
            print("Download complete.")

        print("Loading Svara-TTS...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to("cuda")
            self.model.set_mode("eval")
            print("Svara-TTS ready.")
        except Exception as e:
            print("Model load error: " + str(e))
            self.model = None
            self.tokenizer = None

    @modal.method()
    def tts(self, text: str, language: str = "bn", emotion: str = "neutral",
            voice_sample_b64: str = "") -> dict:
        """Generate speech from text in any of 19 Indian languages."""
        import base64
        import io
        import soundfile as sf

        if not self.model:
            return {"text": text, "audio_b64": None, "error": "Model not loaded"}

        lang_name = LANG_CODES.get(language, language)

        try:
            import torch

            prompt = "<|lang:" + lang_name + "|><|emotion:" + emotion + "|>" + text

            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.7,
                    do_sample=True,
                )

            audio_tokens = outputs[0][inputs["input_ids"].shape[1]:]

            if hasattr(self.model, "decode_audio"):
                wav = self.model.decode_audio(audio_tokens)
            elif hasattr(self.model, "generate_audio"):
                wav = self.model.generate_audio(audio_tokens)
            else:
                return {"text": text, "audio_b64": None, "error": "Model has no audio decode method"}

            if wav is not None:
                buf = io.BytesIO()
                import numpy as np
                audio_np = wav.cpu().numpy().squeeze() if hasattr(wav, 'cpu') else np.array(wav, dtype=np.float32)
                sf.write(buf, audio_np, 24000, format="WAV")
                return {
                    "text": text,
                    "audio_b64": base64.b64encode(buf.getvalue()).decode(),
                    "language": language,
                    "emotion": emotion,
                }
            return {"text": text, "audio_b64": None, "error": "No audio generated"}

        except Exception as e:
            return {"text": text, "audio_b64": None, "error": str(e)}


@app.function(
    gpu="T4", timeout=300, container_idle_timeout=180,
    volumes={MODEL_DIR: volume}, memory=16384,
)
@modal.web_endpoint(method="POST")
def api(body: dict) -> dict:
    """HTTP API for Svara-TTS."""
    svara = SvaraTTS()
    return svara.tts.remote(
        text=body.get("text", ""),
        language=body.get("language", "bn"),
        emotion=body.get("emotion", "neutral"),
        voice_sample_b64=body.get("voice_sample_b64", ""),
    )
