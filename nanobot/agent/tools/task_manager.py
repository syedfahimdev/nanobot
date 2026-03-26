"""Task manager tool — persistent task queue for the ralph-loop pattern.

Allows the agent to manage a structured task queue with priorities,
status tracking, and blocking reasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.task_queue import TaskQueue
from nanobot.agent.tools.base import Tool


class TaskManagerTool(Tool):
    """Manage a persistent task queue (add, next, update, list, clear)."""

    def __init__(self, workspace: Path):
        self._queue = TaskQueue(workspace)

    @property
    def name(self) -> str:
        return "task_manager"

    @property
    def description(self) -> str:
        return (
            "Manage a persistent task queue. "
            "Actions: add (new task), next (get highest-priority pending), "
            "update (change status), list (show all), clear (remove old done tasks)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "next", "update", "list", "clear", "breakdown"],
                    "description": "Action to perform on the task queue",
                },
                "description": {
                    "type": "string",
                    "description": "Task description (for 'add' action)",
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority 1-5, 1=highest (for 'add', default 3)",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (for 'update' action)",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "done", "blocked", "needs_input"],
                    "description": "New status (for 'update' action)",
                },
                "result": {
                    "type": "string",
                    "description": "Result preview (for 'update' when done)",
                },
                "block_reason": {
                    "type": "string",
                    "description": "Why the task is blocked (for 'update' when blocked)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")

        if action == "add":
            desc = kwargs.get("description", "")
            if not desc:
                return "Error: 'description' is required for add action."
            priority = kwargs.get("priority", 3)
            task = self._queue.add_task(desc, priority)
            return json.dumps(task, default=str)

        if action == "next":
            task = self._queue.get_next_task()
            if task is None:
                return "No pending tasks."
            return json.dumps(task, default=str)

        if action == "update":
            task_id = kwargs.get("task_id", "")
            status = kwargs.get("status", "")
            if not task_id or not status:
                return "Error: 'task_id' and 'status' are required for update."
            task = self._queue.update_task(
                task_id,
                status,
                result_preview=kwargs.get("result"),
                block_reason=kwargs.get("block_reason"),
            )
            if task is None:
                return f"Error: task '{task_id}' not found."
            return json.dumps(task, default=str)

        if action == "list":
            tasks = self._queue.get_all_tasks()
            if not tasks:
                return "No tasks in queue."
            return json.dumps(tasks, default=str)

        if action == "clear":
            removed = self._queue.clear_done()
            return f"Cleared {removed} completed task(s)."

        if action == "breakdown":
            desc = kwargs.get("description", "")
            if not desc:
                return "Error: 'description' is required for breakdown action."
            from nanobot.hooks.builtin.claude_capabilities import generate_task_breakdown
            steps = generate_task_breakdown(desc)
            added = []
            for step in steps:
                task = self._queue.add_task(step["description"], step["priority"])
                added.append(task)
            return json.dumps({"added": len(added), "tasks": added}, default=str)

        return f"Unknown action: {action}"
