"""Spawn, update, and list tools for background subagents."""

import json
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use for multi-step or time-consuming tasks (3+ tool calls, research, "
            "complex workflows). The subagent runs independently and reports back when done. "
            "For quick 1-2 tool call tasks, handle directly instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )


class UpdateSubagentTool(Tool):
    """Tool to send a live update to a running subagent."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "update_subagent"

    @property
    def description(self) -> str:
        return (
            "Send a live update or additional instruction to a running subagent. "
            "Use this when the user provides new information or changes related to "
            "a task that a subagent is already working on."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The ID of the running subagent task",
                },
                "message": {
                    "type": "string",
                    "description": "The update or new instruction to send",
                },
            },
            "required": ["task_id", "message"],
        }

    async def execute(self, task_id: str, message: str, **kwargs: Any) -> str:
        return self._manager.send_update(task_id, message)


class ListSubagentsTool(Tool):
    """Tool to list currently running subagents."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager

    @property
    def name(self) -> str:
        return "list_subagents"

    @property
    def description(self) -> str:
        return (
            "List all currently running subagent tasks. "
            "Use this to check what tasks are in progress before deciding "
            "whether to spawn a new subagent or update an existing one."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        tasks = self._manager.get_running_tasks()
        if not tasks:
            return "No subagents currently running."
        lines = ["Running subagents:"]
        for t in tasks:
            lines.append(f"  - [{t['id']}] {t['label']}: {t['task'][:80]}")
        return "\n".join(lines)


class CancelSubagentTool(Tool):
    """Tool to cancel a running subagent."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "cancel_subagent"

    @property
    def description(self) -> str:
        return (
            "Cancel a running subagent task. Use when:\n"
            "- User says 'cancel that task' or 'stop that'\n"
            "- A task is taking too long and you want to take over\n"
            "- User changed their mind about a background task\n"
            "Pass task_id to cancel a specific task, or 'all' to cancel all."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to cancel (from list_subagents), or 'all' to cancel all",
                },
            },
            "required": ["task_id"],
        }

    async def execute(self, task_id: str, **kwargs: Any) -> str:
        if task_id == "all":
            count = await self._manager.cancel_by_session(self._session_key)
            return f"Cancelled {count} subagent(s)." if count else "No running subagents to cancel."

        # Cancel specific task
        tasks = self._manager.get_running_tasks()
        for t in tasks:
            if t["id"] == task_id:
                count = await self._manager.cancel_by_session(self._session_key)
                return f"Cancelled task [{task_id}]: {t['label']}"

        return f"Task '{task_id}' not found. Use list_subagents to see running tasks."
