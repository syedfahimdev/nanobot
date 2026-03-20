"""Daily cleanup hook — archives SHORT_TERM.md on date change.

Triggers on turn_completed events. Checks a date marker file to detect
day transitions and archive yesterday's short-term memory to HISTORY.md.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.hooks.events import TurnCompleted


def make_daily_cleanup_hook(workspace: Path):
    """Create a daily cleanup hook bound to the workspace."""
    marker_file = workspace / "memory" / ".last_cleanup_date"
    store = MemoryStore(workspace)

    def _read_marker() -> str:
        if marker_file.exists():
            return marker_file.read_text(encoding="utf-8").strip()
        return ""

    def _write_marker(d: str) -> None:
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text(d, encoding="utf-8")

    async def daily_cleanup(event: TurnCompleted) -> None:
        today = date.today().isoformat()
        last = _read_marker()

        if last == today:
            return  # Already cleaned up today

        if last:
            # Date changed — archive yesterday's short-term memory
            store.daily_cleanup()

        _write_marker(today)

    return daily_cleanup
