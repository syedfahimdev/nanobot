"""Anthropic OAuth provider — uses Claude Code subscription tokens directly."""

from __future__ import annotations

import json
import secrets
import string
from typing import Any

import anthropic
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALNUM = string.ascii_letters + string.digits

# Models that support adaptive thinking (no budget needed).
_ADAPTIVE_THINKING_MODELS = ("claude-sonnet-4-6", "claude-opus-4-6")

CLAUDE_CODE_VERSION = "2.1.75"


def _short_id() -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _supports_adaptive(model_id: str) -> bool:
    return any(model_id.startswith(m) for m in _ADAPTIVE_THINKING_MODELS)


class AnthropicOAuthProvider(LLMProvider):
    """Provider for Claude Code OAuth tokens (sk-ant-oat01-...)."""

    def __init__(
        self,
        auth_token: str,
        default_model: str = "claude-sonnet-4-6",
    ):
        super().__init__(api_key=auth_token, api_base=None)
        self.default_model = default_model
        self._client = anthropic.AsyncAnthropic(
            api_key=None,
            auth_token=auth_token,
            default_headers={
                "accept": "application/json",
                "anthropic-beta": (
                    "claude-code-20250219,"
                    "oauth-2025-04-20,"
                    "fine-grained-tool-streaming-2025-05-14,"
                    "interleaved-thinking-2025-05-14"
                ),
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                "x-app": "cli",
            },
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        max_tokens = max(1, max_tokens)

        # Build system prompt — OAuth requires Claude Code identity prefix.
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
        ]

        # Extract system messages from the conversation and move to system param.
        filtered_messages = []
        for msg in self._sanitize_empty_content(messages):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    system_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    system_blocks.extend(content)
            else:
                filtered_messages.append(msg)

        # Convert messages to Anthropic format.
        api_messages = self._convert_messages(filtered_messages)

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
            "system": system_blocks,
        }

        # Thinking config.
        if _supports_adaptive(model):
            params["thinking"] = {"type": "adaptive"}
        else:
            # Budget-based thinking for older models.
            budget = {"low": 2048, "medium": 8192, "high": 32768}.get(reasoning_effort or "medium", 8192)
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            params["max_tokens"] = max(max_tokens, budget + 1024)

        # Tools.
        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice == "required":
                params["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                params["tool_choice"] = tool_choice
            else:
                params["tool_choice"] = {"type": "auto"}

        try:
            response = await self._client.messages.create(**params)
            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format."""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue  # Already handled above.

            if role == "tool":
                # Tool result → user message with tool_result content.
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": str(msg.get("content", "")),
                    }],
                })
                continue

            if role == "assistant":
                content_blocks = []
                # Add thinking blocks if present (only if they have required signature).
                for tb in (msg.get("thinking_blocks") or []):
                    if isinstance(tb, dict) and tb.get("signature"):
                        content_blocks.append(tb)
                    # Skip thinking blocks without signature — API rejects them.
                # Text content.
                text = msg.get("content")
                if isinstance(text, str) and text:
                    content_blocks.append({"type": "text", "text": text})
                elif isinstance(text, list):
                    content_blocks.extend(text)
                # Tool calls → tool_use blocks.
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", tc) if isinstance(tc, dict) else tc
                    name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    tc_id = tc.get("id", _short_id()) if isinstance(tc, dict) else getattr(tc, "id", _short_id())
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": name,
                        "input": args,
                    })
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": ""})
                result.append({"role": "assistant", "content": content_blocks})
                continue

            # User message.
            content = msg.get("content", "")
            if isinstance(content, str):
                result.append({"role": "user", "content": content or "(empty)"})
            elif isinstance(content, list):
                result.append({"role": "user", "content": content})
            else:
                result.append({"role": "user", "content": str(content)})

        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format."""
        result = []
        for tool in tools:
            fn = tool.get("function", tool)
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response to LLMResponse."""
        text_parts = []
        tool_calls = []
        thinking_blocks = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
            elif block.type == "thinking":
                tb: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": block.thinking,
                }
                if getattr(block, "signature", None):
                    tb["signature"] = block.signature
                thinking_blocks.append(tb)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        finish_reason = "stop"
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"

        return LLMResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            thinking_blocks=thinking_blocks or None,
        )

    def get_default_model(self) -> str:
        return self.default_model
