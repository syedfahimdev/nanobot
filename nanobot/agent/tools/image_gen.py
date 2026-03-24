"""Image/video generation tool — configurable providers.

Supported providers (configurable via settings):
- together: Together AI (free tier, Flux models)
- fal: Fal.ai (Flux, SDXL, video)
- replicate: Replicate (any model)
- openai: OpenAI DALL-E
- stability: Stability AI

Provider + model + API key configured via mawa_settings.json.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool


# Provider configurations
_PROVIDERS = {
    "pollinations": {
        "url": "https://gen.pollinations.ai/prompt/{prompt}",
        "default_model": "flux",
        "env_key": "",  # No API key needed — 100% free
        "vault_key": "",
        "free": True,
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/models/{model}",
        "default_model": "black-forest-labs/FLUX.1-schnell",
        "env_key": "HF_TOKEN",
        "vault_key": "huggingface",
    },
    "together": {
        "url": "https://api.together.xyz/v1/images/generations",
        "default_model": "black-forest-labs/FLUX.1-schnell-Free",
        "env_key": "TOGETHER_API_KEY",
        "vault_key": "together",
    },
    "fal": {
        "url": "https://queue.fal.run/{model}",
        "default_model": "fal-ai/flux/schnell",
        "env_key": "FAL_KEY",
        "vault_key": "fal",
    },
    "replicate": {
        "url": "https://api.replicate.com/v1/predictions",
        "default_model": "black-forest-labs/flux-schnell",
        "env_key": "REPLICATE_API_TOKEN",
        "vault_key": "replicate",
    },
    "openai": {
        "url": "https://api.openai.com/v1/images/generations",
        "default_model": "dall-e-3",
        "env_key": "OPENAI_API_KEY",
        "vault_key": "openai",
    },
    "stability": {
        "url": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        "default_model": "sd3-large",
        "env_key": "STABILITY_API_KEY",
        "vault_key": "stability",
    },
}


def _get_api_key(provider_config: dict) -> str | None:
    """Get API key from env or vault."""
    key = os.environ.get(provider_config["env_key"])
    if key:
        return key
    try:
        from nanobot.setup.vault import load_vault
        vault = load_vault()
        vault_key = provider_config["vault_key"]
        return vault.get(f"cred.{vault_key}_api_key") or vault.get(f"cred.{vault_key}")
    except Exception:
        return None


class ImageGenTool(Tool):
    """Generate images from text descriptions. Provider configurable via settings."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._output_dir = workspace / "generated" / "images"

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "Generate an image from a text description. "
            "Use when the user asks to create, draw, generate, or visualize an image. "
            "Also use proactively when a visual would help explain something. "
            "Supports styles: photo, art, illustration, sketch, 3d."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate.",
                },
                "style": {
                    "type": "string",
                    "enum": ["photo", "art", "illustration", "sketch", "3d"],
                    "description": "Visual style. Default: photo.",
                },
                "size": {
                    "type": "string",
                    "enum": ["square", "landscape", "portrait"],
                    "description": "Image dimensions. Default: square.",
                },
            },
            "required": ["prompt"],
        }

    def _get_settings(self) -> dict:
        """Read image gen settings from unified config."""
        try:
            from nanobot.hooks.builtin.feature_registry import get_setting
            return {
                "provider": get_setting(self._workspace, "imageGenProvider", "pollinations"),
                "model": get_setting(self._workspace, "imageGenModel", ""),
            }
        except Exception:
            return {"provider": "together", "model": ""}

    async def execute(self, prompt: str, style: str = "photo", size: str = "square", **kwargs) -> str:
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Enhance prompt with style
        style_prefix = {
            "art": "artistic painting, ",
            "illustration": "digital illustration, ",
            "sketch": "pencil sketch, ",
            "3d": "3D rendered, ",
        }
        enhanced = style_prefix.get(style, "") + prompt

        # Size mapping
        size_map = {
            "square": (1024, 1024),
            "landscape": (1280, 768),
            "portrait": (768, 1280),
        }
        width, height = size_map.get(size, (1024, 1024))

        # Get configured provider
        settings = self._get_settings()
        provider_name = settings["provider"]
        provider = _PROVIDERS.get(provider_name)

        if not provider:
            return f"Error: Unknown image provider '{provider_name}'. Available: {', '.join(_PROVIDERS.keys())}"

        # Free providers don't need API keys
        if provider.get("free"):
            api_key = "free"
        else:
            api_key = _get_api_key(provider)
            if not api_key:
                # Try free providers first, then others
                for name, prov in _PROVIDERS.items():
                    if prov.get("free"):
                        provider_name = name
                        provider = prov
                        api_key = "free"
                        break
                    key = _get_api_key(prov)
                    if key:
                        provider_name = name
                        provider = prov
                        api_key = key
                        break

        if not api_key:
            return (
                "Error: No image generation API key found. Configure one in settings:\n"
                "1. `settings(action='set', key='imageGenProvider', value='pollinations')` (FREE, no key needed)\n"
                "2. Save your API key: `credentials(action='save', name='together_api_key', value='...')`\n\n"
                "Supported: together (free tier), fal, replicate, openai, stability"
            )

        model = settings.get("model") or provider["default_model"]

        # Route to the right provider
        result = None
        if provider_name == "pollinations":
            result = await self._gen_pollinations(enhanced, model, width, height)
        elif provider_name == "huggingface":
            result = await self._gen_huggingface(api_key, enhanced, model)
        elif provider_name == "together":
            result = await self._gen_together(api_key, enhanced, model, width, height)
        elif provider_name == "fal":
            result = await self._gen_fal(api_key, enhanced, model, width, height)
        elif provider_name == "replicate":
            result = await self._gen_replicate(api_key, enhanced, model, width, height)
        elif provider_name == "openai":
            result = await self._gen_openai(api_key, enhanced, model, size)
        elif provider_name == "stability":
            result = await self._gen_stability(api_key, enhanced, model, width, height)

        if not result:
            return f"Error: Image generation failed with {provider_name}. Check API key and try again."

        filename = f"image_{int(time.time())}.png"
        filepath = self._output_dir / filename
        filepath.write_bytes(result)
        logger.info("Generated image via {}: {} ({} bytes)", provider_name, filepath, len(result))
        return f"Image generated: {filepath}"

    async def _gen_pollinations(self, prompt: str, model: str, w: int, h: int) -> bytes | None:
        """Generate via Pollinations.ai — 100% free, no API key."""
        try:
            from urllib.parse import quote
            url = f"https://gen.pollinations.ai/prompt/{quote(prompt)}?width={w}&height={h}&model={model}&nologo=true"
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(url)
                if r.status_code == 200 and len(r.content) > 1000:
                    return r.content
                logger.warning("Pollinations: {} ({}b)", r.status_code, len(r.content))
        except Exception as e:
            logger.warning("Pollinations error: {}", e)
        return None

    async def _gen_huggingface(self, key: str, prompt: str, model: str) -> bytes | None:
        """Generate via Hugging Face Inference API (free with token)."""
        try:
            url = f"https://router.huggingface.co/hf-inference/models/{model}"
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(url,
                    headers={"Authorization": f"Bearer {key}"},
                    json={"inputs": prompt})
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    return r.content
                logger.warning("HuggingFace: {} {}", r.status_code, r.text[:100])
        except Exception as e:
            logger.warning("HuggingFace error: {}", e)
        return None

    async def _gen_together(self, key: str, prompt: str, model: str, w: int, h: int) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post("https://api.together.xyz/v1/images/generations",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "prompt": prompt, "width": w, "height": h, "steps": 4, "n": 1, "response_format": "b64_json"})
                if r.status_code == 200:
                    return base64.b64decode(r.json()["data"][0]["b64_json"])
                logger.warning("Together: {} {}", r.status_code, r.text[:100])
        except Exception as e:
            logger.warning("Together error: {}", e)
        return None

    async def _gen_fal(self, key: str, prompt: str, model: str, w: int, h: int) -> bytes | None:
        try:
            url = f"https://fal.run/{model}"
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(url,
                    headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
                    json={"prompt": prompt, "image_size": {"width": w, "height": h}, "num_images": 1})
                if r.status_code == 200:
                    data = r.json()
                    img_url = data.get("images", [{}])[0].get("url")
                    if img_url:
                        img_resp = await c.get(img_url)
                        if img_resp.status_code == 200:
                            return img_resp.content
                logger.warning("Fal: {} {}", r.status_code, r.text[:100])
        except Exception as e:
            logger.warning("Fal error: {}", e)
        return None

    async def _gen_replicate(self, key: str, prompt: str, model: str, w: int, h: int) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                # Start prediction
                r = await c.post("https://api.replicate.com/v1/predictions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "input": {"prompt": prompt, "width": w, "height": h}})
                if r.status_code != 201:
                    return None
                pred = r.json()
                get_url = pred.get("urls", {}).get("get")
                if not get_url:
                    return None
                # Poll for result
                import asyncio
                for _ in range(30):
                    await asyncio.sleep(2)
                    poll = await c.get(get_url, headers={"Authorization": f"Bearer {key}"})
                    data = poll.json()
                    if data.get("status") == "succeeded":
                        output = data.get("output")
                        if isinstance(output, list) and output:
                            img = await c.get(output[0])
                            return img.content if img.status_code == 200 else None
                    elif data.get("status") == "failed":
                        return None
        except Exception as e:
            logger.warning("Replicate error: {}", e)
        return None

    async def _gen_openai(self, key: str, prompt: str, model: str, size: str) -> bytes | None:
        size_map = {"square": "1024x1024", "landscape": "1792x1024", "portrait": "1024x1792"}
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post("https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": model, "prompt": prompt, "size": size_map.get(size, "1024x1024"), "n": 1, "response_format": "b64_json"})
                if r.status_code == 200:
                    return base64.b64decode(r.json()["data"][0]["b64_json"])
        except Exception as e:
            logger.warning("OpenAI image error: {}", e)
        return None

    async def _gen_stability(self, key: str, prompt: str, model: str, w: int, h: int) -> bytes | None:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"https://api.stability.ai/v2beta/stable-image/generate/{model}",
                    headers={"Authorization": f"Bearer {key}", "Accept": "image/*"},
                    data={"prompt": prompt, "width": w, "height": h, "output_format": "png"},
                )
                if r.status_code == 200:
                    return r.content
        except Exception as e:
            logger.warning("Stability error: {}", e)
        return None
