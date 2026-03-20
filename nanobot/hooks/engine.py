"""Lightweight hook engine — emit events, hooks subscribe."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from loguru import logger


class HookMode(Enum):
    FIRE_AND_FORGET = "fire_and_forget"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class _Registration:
    event: str
    callback: Callable[..., Awaitable[Any]]
    mode: HookMode
    priority: int


class HookEngine:
    """Subscribe to events with async callbacks. Blocking hooks can mutate payloads."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[_Registration]] = {}

    def on(
        self,
        event: str,
        callback: Callable[..., Awaitable[Any]],
        *,
        mode: HookMode = HookMode.FIRE_AND_FORGET,
        priority: int = 100,
    ) -> None:
        regs = self._hooks.setdefault(event, [])
        regs.append(_Registration(event, callback, mode, priority))
        regs.sort(key=lambda r: r.priority)

    async def emit(self, event: str, payload: Any) -> Any:
        """Emit an event. Blocking hooks run in order; fire-and-forget are spawned."""
        regs = self._hooks.get(event)
        if not regs:
            return payload

        for reg in regs:
            try:
                if reg.mode == HookMode.BLOCKING:
                    result = await reg.callback(payload)
                    if result is not None:
                        payload = result
                else:
                    asyncio.create_task(reg.callback(payload))
            except Exception:
                logger.opt(exception=True).warning("Hook {} failed on {}", reg.callback.__name__, event)

        return payload
