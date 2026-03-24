"""Svara-TTS on Modal — 19 Indian languages + emotion + voice cloning.

Uses vLLM + SNAC codec. Orpheus-style discrete audio tokens.
7 codes per audio frame, decoded via SNAC 24kHz.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers>=4.45",
        "vllm>=0.6.0",
        "snac",
        "soundfile",
        "numpy<2",
        "fastapi[standard]",
        "huggingface_hub",
    )
)

app = modal.App("mawa-svara-tts", image=image)
volume = modal.Volume.from_name("svara-tts-weights", create_if_missing=True)
MODEL_DIR = "/models"

# Prompt format constants
BOS_TOKEN = 128000
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
END_OF_TURN = 128009
AUDIO_TOKEN = 156939

SPEAKERS = {
    "bn": "Bengali", "hi": "Hindi", "en": "English",
    "mr": "Marathi", "te": "Telugu", "kn": "Kannada",
    "gu": "Gujarati", "ml": "Malayalam", "pa": "Punjabi",
    "ta": "Tamil", "as": "Assamese", "ne": "Nepali",
    "sa": "Sanskrit", "ur": "Hindi",
}


def raw_to_code(raw_num, good_idx):
    """Convert raw token number to SNAC code. 7 tokens per frame."""
    return raw_num - 10 - ((good_idx % 7) * 4096)


def decode_snac_window(snac_model, codes, device="cuda"):
    """Decode a list of SNAC codes (multiple of 7) to audio."""
    import torch
    import numpy as np

    F = len(codes) // 7
    if F == 0:
        return None

    codes = codes[:F * 7]
    t = torch.tensor(codes, dtype=torch.int32, device=device).view(F, 7)

    codes_0 = t[:, 0].reshape(1, -1).long()
    codes_1 = t[:, [1, 4]].reshape(1, -1).long()
    codes_2 = t[:, [2, 3, 5, 6]].reshape(1, -1).long()

    with torch.no_grad():
        audio = snac_model.decode([codes_0, codes_1, codes_2])

    return audio.squeeze().cpu().numpy()


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
        import torch
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        model_name = "kenpath/svara-tts-v1"

        # Load SNAC
        print("Loading SNAC 24kHz codec...")
        try:
            from snac import SNAC
            self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
            print("SNAC ready.")
        except Exception as e:
            print("SNAC error: " + str(e))
            self.snac = None

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load vLLM
        print("Loading vLLM...")
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, dtype="half", max_model_len=2048, gpu_memory_utilization=0.80)
            self.SamplingParams = SamplingParams
            print("vLLM ready.")
        except Exception as e:
            print("vLLM error: " + str(e) + " — trying transformers fallback")
            self.llm = None
            try:
                from transformers import AutoModelForCausalLM
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16
                ).to("cuda").eval()
                print("Transformers fallback ready.")
            except Exception as e2:
                print("Fallback error: " + str(e2))
                self.hf_model = None

    def _build_prompt_ids(self, text, speaker_id):
        """Build token IDs in Svara prompt format."""
        import torch
        text_ids = self.tokenizer.encode(speaker_id + ": " + text, add_special_tokens=False)
        # Format: BOS, START_HUMAN, AUDIO_TOKEN, text_ids, END_HUMAN, EOT, START_AI, START_SPEECH
        ids = [BOS_TOKEN, START_OF_HUMAN, AUDIO_TOKEN] + text_ids + [END_OF_HUMAN, END_OF_TURN, START_OF_AI, START_OF_SPEECH]
        return ids

    @modal.method()
    def tts(self, text, language="bn", gender="female", emotion="") -> dict:
        """Generate speech."""
        import base64, io, soundfile as sf

        if not self.snac:
            return {"text": text, "audio_b64": None, "error": "SNAC not loaded"}

        lang_name = SPEAKERS.get(language, "Bengali")
        speaker_id = lang_name + " (" + gender.capitalize() + ")"

        if emotion and emotion != "neutral":
            text = text + " <" + emotion + ">"

        prompt_ids = self._build_prompt_ids(text, speaker_id)

        try:
            if self.llm:
                params = self.SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048, repetition_penalty=1.1)
                outputs = self.llm.generate(prompt_token_ids=[prompt_ids], sampling_params=params)
                gen_ids = list(outputs[0].outputs[0].token_ids)
            elif hasattr(self, "hf_model") and self.hf_model:
                import torch
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
                with torch.no_grad():
                    out = self.hf_model.generate(input_ids, max_new_tokens=2048, temperature=0.6, top_p=0.95, do_sample=True)
                gen_ids = out[0][len(prompt_ids):].tolist()
            else:
                return {"text": text, "audio_b64": None, "error": "No engine"}

            # Debug: show what the model generated
            total_gen = len(gen_ids)
            sample_ids = gen_ids[:20]
            audio_range_count = sum(1 for t in gen_ids if t >= 128266)

            # Convert token IDs to SNAC codes using mapper logic
            codes = []
            good_idx = 0
            for tid in gen_ids:
                if tid == END_OF_SPEECH or tid == 128001:
                    break
                if tid >= 128266:
                    raw_num = tid - 128256
                    code = raw_to_code(raw_num, good_idx)
                    if 0 <= code < 4096:
                        codes.append(code)
                        good_idx += 1

            if len(codes) < 7:
                return {
                    "text": text, "audio_b64": None,
                    "error": "Too few codes: " + str(len(codes)),
                    "debug": {
                        "total_tokens": total_gen,
                        "audio_range_tokens": audio_range_count,
                        "first_20_ids": sample_ids,
                        "codes_found": len(codes),
                    },
                }

            # Decode with SNAC
            audio_np = decode_snac_window(self.snac, codes, "cuda")
            if audio_np is None or len(audio_np) < 100:
                return {"text": text, "audio_b64": None, "error": "SNAC decode empty"}

            buf = io.BytesIO()
            sf.write(buf, audio_np, 24000, format="WAV")
            return {
                "text": text,
                "audio_b64": base64.b64encode(buf.getvalue()).decode(),
                "language": language,
                "speaker": speaker_id,
            }
        except Exception as e:
            return {"text": text, "audio_b64": None, "error": str(e)}


@app.function(gpu="T4", timeout=300, container_idle_timeout=180, volumes={MODEL_DIR: volume}, memory=16384)
@modal.web_endpoint(method="POST")
def api(body: dict) -> dict:
    svara = SvaraTTS()
    return svara.tts.remote(
        text=body.get("text", ""),
        language=body.get("language", "bn"),
        gender=body.get("gender", "female"),
        emotion=body.get("emotion", ""),
    )
