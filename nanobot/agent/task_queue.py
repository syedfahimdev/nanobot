"""Persistent task queue inspired by the ralph-loop pattern.

Tasks are stored in workspace/tasks.json and survive restarts.
Each task has status, priority, timestamps, and optional block/result info.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger


class TaskQueue:
    """Manage a persistent task queue stored as JSON."""

    def __init__(self, workspace: Path):
        self._file = workspace / "tasks.json"

    # -- persistence --

    def _load(self) -> list[dict]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt tasks.json, resetting")
        return []

    def _save(self, tasks: list[dict]) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(tasks, indent=2, default=str), encoding="utf-8")

    # -- public API --

    def add_task(self, description: str, priority: int = 3) -> dict:
        """Add a new pending task and return it."""
        tasks = self._load()
        task_id = os.urandom(3).hex()
        task = {
            "id": task_id,
            "description": description,
            "status": "pending",
            "priority": max(1, min(5, priority)),
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "block_reason": None,
            "result_preview": None,
        }
        tasks.append(task)
        self._save(tasks)
        return task

    def get_next_task(self) -> dict | None:
        """Return the highest-priority pending task, or None."""
        tasks = self._load()
        pending = [t for t in tasks if t["status"] == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda t: t["priority"])
        return pending[0]

    def update_task(
        self,
        task_id: str,
        status: str,
        result_preview: str | None = None,
        block_reason: str | None = None,
    ) -> dict | None:
        """Update a task's status and optional fields. Returns updated task or None."""
        tasks = self._load()
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = status
                if result_preview is not None:
                    task["result_preview"] = result_preview
                if block_reason is not None:
                    task["block_reason"] = block_reason
                if status == "done":
                    task["completed_at"] = datetime.now().isoformat()
                self._save(tasks)
                return task
        return None

    def get_all_tasks(self) -> list[dict]:
        """Return all tasks."""
        return self._load()

    def clear_done(self) -> int:
        """Remove completed tasks older than 24h. Returns count removed."""
        tasks = self._load()
        cutoff = datetime.now() - timedelta(hours=24)
        original = len(tasks)
        kept = []
        for t in tasks:
            if t["status"] == "done" and t.get("completed_at"):
                try:
                    completed = datetime.fromisoformat(t["completed_at"])
                    if completed < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            kept.append(t)
        removed = original - len(kept)
        if removed:
            self._save(kept)
        return removed
