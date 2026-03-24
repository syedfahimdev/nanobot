"""Spawn, update, and list tools for background subagents."""

import json
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


# Specialist profiles — each has a focused system prompt addition
SPECIALIST_PROFILES = {
    "researcher": {
        "label": "Deep Researcher",
        "prompt": (
            "You are a thorough researcher. Your job is to find comprehensive, accurate information.\n"
            "- Search multiple sources, not just one\n"
            "- Compare and cross-reference findings\n"
            "- Note conflicting information\n"
            "- Cite sources (URLs) for key facts\n"
            "- Organize findings with headings and bullet points\n"
            "- Include prices, dates, ratings where relevant"
        ),
    },
    "analyst": {
        "label": "Data Analyst",
        "prompt": (
            "You are a data analyst. Your job is to analyze information and provide insights.\n"
            "- Extract key metrics and numbers\n"
            "- Compare options with pros/cons tables\n"
            "- Identify trends and patterns\n"
            "- Provide clear recommendations with reasoning\n"
            "- Use tables and structured formats"
        ),
    },
    "writer": {
        "label": "Content Writer",
        "prompt": (
            "You are a skilled writer. Your job is to create well-written content.\n"
            "- Match the requested tone and style\n"
            "- Be concise but complete\n"
            "- Use proper formatting (headings, paragraphs, lists)\n"
            "- Proofread for grammar and clarity\n"
            "- Adapt to the audience (professional, casual, technical)"
        ),
    },
    "coder": {
        "label": "Code Specialist",
        "prompt": (
            "You are a coding specialist. Your job is to write, debug, or analyze code.\n"
            "- Write clean, well-commented code\n"
            "- Follow the project's existing patterns\n"
            "- Test your work with exec when possible\n"
            "- Handle errors gracefully\n"
            "- Explain key decisions"
        ),
    },
    "planner": {
        "label": "Project Planner",
        "prompt": (
            "You are a project planner. Your job is to break down complex tasks.\n"
            "- Create step-by-step plans with clear milestones\n"
            "- Identify dependencies between steps\n"
            "- Estimate effort for each step\n"
            "- Flag risks and blockers\n"
            "- Prioritize by impact"
        ),
    },
}

# Auto-detect specialist from task keywords
_SPECIALIST_KEYWORDS = {
    "researcher": ["research", "find", "look up", "compare options", "best", "review", "investigate"],
    "analyst": ["analyze", "compare", "pros and cons", "which is better", "evaluate", "metrics"],
    "writer": ["write", "draft", "compose", "email", "letter", "report", "document", "blog"],
    "coder": ["code", "script", "debug", "fix the bug", "implement", "refactor", "program"],
    "planner": ["plan", "break down", "steps to", "roadmap", "timeline", "organize", "schedule"],
}


def detect_specialist(task: str) -> str | None:
    """Auto-detect the best specialist for a task. Returns profile name or None."""
    task_lower = task.lower()
    best = None
    best_score = 0
    for profile, keywords in _SPECIALIST_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > best_score:
            best_score = score
            best = profile
    return best if best_score >= 2 else None


class SpawnTool(Tool):
    """Tool to spawn a specialist subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a specialist subagent for background task execution.\n"
            "Specialists: researcher (deep search), analyst (compare/evaluate), "
            "writer (draft content), coder (write/debug code), planner (break down tasks).\n"
            "Auto-detects the best specialist from the task, or specify one.\n"
            "Use for 3+ tool call tasks. For quick tasks, handle directly."
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
                    "description": "Short label for the task (for display)",
                },
                "specialist": {
                    "type": "string",
                    "enum": ["auto", "researcher", "analyst", "writer", "coder", "planner"],
                    "description": "Specialist type. 'auto' detects from task. Default: auto.",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, specialist: str = "auto", **kwargs: Any) -> str:
        # Auto-detect or use specified specialist
        if specialist == "auto" or not specialist:
            specialist = detect_specialist(task)

        # Prepend specialist prompt to task
        enhanced_task = task
        if specialist and specialist in SPECIALIST_PROFILES:
            profile = SPECIALIST_PROFILES[specialist]
            enhanced_task = f"[Specialist: {profile['label']}]\n{profile['prompt']}\n\n## Your Task\n{task}"
            label = label or f"{profile['label']}: {task[:30]}"

        result = await self._manager.spawn(
            task=enhanced_task,
            label=label,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )

        specialist_note = f" (specialist: {specialist})" if specialist else ""
        return f"{result}{specialist_note}"


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
