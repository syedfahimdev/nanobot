"""Tool registry for dynamic tool management."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.hooks.engine import HookEngine


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self, hooks: HookEngine | None = None):
        self._tools: dict[str, Tool] = {}
        self._hooks = hooks
        # Current execution context (set by agent loop before each turn)
        self._channel: str | None = None
        self._chat_id: str | None = None
        self._session_key: str | None = None

    def set_context(self, channel: str | None, chat_id: str | None, session_key: str | None) -> None:
        """Set execution context for hook payloads."""
        self._channel = channel
        self._chat_id = chat_id
        self._session_key = session_key

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT

            # Hook: tool_before (blocking — approval guard can deny)
            if self._hooks:
                from nanobot.hooks.events import ToolBefore
                before = ToolBefore(
                    name=name, params=params,
                    channel=self._channel, chat_id=self._chat_id,
                    session_key=self._session_key,
                )
                before = await self._hooks.emit("tool_before", before)
                if before.denied:
                    return before.deny_reason or f"Action denied: {name}"

            t0 = time.monotonic()
            result = await tool.execute(**params)
            elapsed_ms = (time.monotonic() - t0) * 1000
            is_error = isinstance(result, str) and result.startswith("Error")

            # Hook: tool_after (fire-and-forget — logging, tracking)
            if self._hooks:
                from nanobot.hooks.events import ToolAfter
                await self._hooks.emit("tool_after", ToolAfter(
                    name=name, params=params, result=result[:500] if isinstance(result, str) else str(result)[:500],
                    duration_ms=elapsed_ms, error=is_error,
                    channel=self._channel, chat_id=self._chat_id,
                    session_key=self._session_key,
                ))

            if is_error:
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
