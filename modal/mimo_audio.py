"""MiMo-Audio on Modal — serverless GPU for voice processing.

Deploys MiMo-Audio-7B-Instruct on a serverless A10G GPU.
Spins up on demand, scales to 0 when idle (zero cost when not in use).

Endpoints:
- /stt — speech to text (audio bytes → text)
- /tts — text to speech (text → audio bytes)
- /chat — full audio chat (audio in → audio out)
"""

import modal

# Build the container image with all dependencies
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

# Persistent volume for model weights (so they don't re-download every cold start)
volume = modal.Volume.from_name("mimo-audio-weights", create_if_missing=True)
MODEL_DIR = "/models"


@app.function(
    gpu="T4",
    timeout=300,
    container_idle_timeout=120,  # Keep warm for 2 min after last request
    volumes={MODEL_DIR: volume},
    memory=32768,
)
def process_audio(
    audio_b64: str | None = None,
    text: str | None = None,
    mode: str = "chat",
    system_prompt: str = "You are Mawa, a helpful personal AI assistant. Be concise and natural.",
) -> dict:
    """Process audio or text through MiMo-Audio.

    Args:
        audio_b64: Base64-encoded audio (for stt or chat mode)
        text: Text input (for tts mode)
        mode: 'stt' (audio→text), 'tts' (text→audio), 'chat' (audio→audio)
        system_prompt: System prompt for chat mode

    Returns:
        {text: str, audio_b64: str | None}
    """
    import base64
    import io
    import os
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from huggingface_hub import snapshot_download

    model_path = os.path.join(MODEL_DIR, "MiMo-Audio-7B-Instruct")

    # Download model if not cached
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print("Downloading MiMo-Audio-7B-Instruct...")
        snapshot_download(
            "XiaomiMiMo/MiMo-Audio-7B-Instruct",
            local_dir=model_path,
        )
        volume.commit()
        print("Model downloaded and cached.")

    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded.")

    result = {"text": "", "audio_b64": None}

    if mode == "stt" and audio_b64:
        # Speech to text
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        inputs = processor(
            audios=[audio_data],
            sampling_rate=sr,
            text="Transcribe the following audio:",
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500)
        result["text"] = processor.decode(outputs[0], skip_special_tokens=True)

    elif mode == "tts" and text:
        # Text to speech
        inputs = processor(
            text=f"Read the following text aloud: {text}",
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2000)

        # Extract audio from output
        audio = processor.decode_audio(outputs[0])
        if audio is not None:
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format="WAV")
            result["audio_b64"] = base64.b64encode(buf.getvalue()).decode()
            result["text"] = text

    elif mode == "chat" and audio_b64:
        # Full audio chat: audio in → process → audio out
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        inputs = processor(
            audios=[audio_data],
            sampling_rate=sr,
            text=system_prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2000)

        # Decode both text and audio
        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        result["text"] = decoded

        audio = processor.decode_audio(outputs[0])
        if audio is not None:
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format="WAV")
            result["audio_b64"] = base64.b64encode(buf.getvalue()).decode()

    return result


# Web endpoint for HTTP access
@app.function(
    gpu="T4",
    timeout=300,
    container_idle_timeout=120,
    volumes={MODEL_DIR: volume},
    memory=32768,
)
@modal.web_endpoint(method="POST")
def api(body: dict) -> dict:
    """HTTP API endpoint for MiMo-Audio."""
    return process_audio.local(
        audio_b64=body.get("audio_b64"),
        text=body.get("text"),
        mode=body.get("mode", "chat"),
        system_prompt=body.get("system_prompt", "You are Mawa, a helpful personal AI assistant."),
    )
