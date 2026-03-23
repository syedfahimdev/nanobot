"""Auto-log tool calls to HISTORY.md — pure code, no LLM."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from nanobot.hooks.events import ToolAfter

# Tools that are too noisy to log (internal file/exec ops)
_SKIP = frozenset({"read_file", "write_file", "edit_file", "list_dir", "exec"})


def make_auto_log_hook(workspace: Path):
    """Create an auto-log hook bound to the workspace."""
    history_path = workspace / "memory" / "HISTORY.md"

    async def auto_log(event: ToolAfter) -> None:
        if event.name in _SKIP:
            return
        tool_label = event.name

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        status = "ERR" if event.error else "OK"
        line = f"[{ts}] TOOL {tool_label} {status} ({event.duration_ms:.0f}ms)\n"

        try:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.debug("Auto-log write failed: {}", e)

    return auto_log
