"""Auto-backup workspace memory to git on a schedule.

Commits any changed knowledge, learnings, rules, and memory files.
Runs as a background task every 30 minutes.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from loguru import logger


async def start_backup_loop(workspace: Path, interval: int = 1800) -> None:
    """Background task: auto-commit workspace changes every `interval` seconds."""
    git_dir = workspace / ".git"
    if not git_dir.exists():
        logger.info("Workspace backup: no git repo at {}, skipping", workspace)
        return

    logger.info("Workspace backup: started (every {}s)", interval)

    while True:
        await asyncio.sleep(interval)
        try:
            await _backup_once(workspace)
        except Exception as e:
            logger.warning("Workspace backup failed: {}", e)


async def _backup_once(workspace: Path) -> None:
    """Stage and commit any changes in the workspace."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _git_backup, workspace)


def _git_backup(workspace: Path) -> None:
    """Synchronous git add + commit (runs in executor)."""
    def _run(cmd: list[str]) -> str:
        result = subprocess.run(
            cmd, cwd=str(workspace),
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip()

    # Check for changes
    status = _run(["git", "status", "--porcelain"])
    if not status:
        return  # Nothing to commit

    # Stage tracked dirs only (not sessions or temp files)
    for d in ["knowledge", "learnings", "rules", "memory", "SOUL.md", "USER.md", "AGENTS.md", "TOOLS.md"]:
        _run(["git", "add", d])

    # Commit
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    _run(["git", "commit", "-m", f"auto-backup {ts}", "--allow-empty"])
    logger.info("Workspace backup: committed at {}", ts)
