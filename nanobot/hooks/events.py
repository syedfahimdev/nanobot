"""Event payloads for the hook system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolBefore:
    """Emitted before a tool executes. Blocking hooks can set denied=True."""

    name: str
    params: dict[str, Any]
    channel: str | None = None
    chat_id: str | None = None
    session_key: str | None = None
    denied: bool = False
    deny_reason: str | None = None


@dataclass
class ToolAfter:
    """Emitted after a tool executes."""

    name: str
    params: dict[str, Any]
    result: str
    duration_ms: float = 0.0
    channel: str | None = None
    chat_id: str | None = None
    session_key: str | None = None
    error: bool = False


@dataclass
class TurnCompleted:
    """Emitted after the agent finishes a full turn."""

    session_key: str
    final_content: str | None = None
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    duration_ms: float = 0.0
    channel: str | None = None
    chat_id: str | None = None
