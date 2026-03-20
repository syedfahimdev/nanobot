"""Goal tracking tool — persistent agent-managed task queue.

The agent uses this tool to track multi-session goals, break them into
subtasks, update progress, and get reminded about pending work.

Goals are stored in memory/GOALS.md as markdown checkboxes for readability
and portability. The format is human-editable.

Format:
  ## Goal: Plan wedding
  - [ ] Book photographer
  - [x] Choose venue
  - [ ] Send invitations (due: 2026-05-01)
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class GoalsTool(Tool):
    """Track persistent goals and subtasks across sessions."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._goals_file = workspace / "memory" / "GOALS.md"

    @property
    def name(self) -> str:
        return "goals"

    @property
    def description(self) -> str:
        return (
            "Track persistent goals and tasks across sessions. "
            "Actions: add (new goal/subtask), complete (mark done), list (show all), "
            "remove (delete goal), update (edit goal text)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "complete", "list", "remove", "update"],
                    "description": "Action to perform on goals",
                },
                "goal": {
                    "type": "string",
                    "description": "Goal title (for add: creates new goal section)",
                },
                "subtask": {
                    "type": "string",
                    "description": "Subtask text (for add: adds under the goal)",
                },
                "due": {
                    "type": "string",
                    "description": "Optional due date in YYYY-MM-DD format",
                },
                "index": {
                    "type": "integer",
                    "description": "Task index to complete/remove/update (1-based, from list output)",
                },
                "new_text": {
                    "type": "string",
                    "description": "New text for update action",
                },
            },
            "required": ["action"],
        }

    def _read(self) -> str:
        if self._goals_file.exists():
            return self._goals_file.read_text(encoding="utf-8")
        return ""

    def _write(self, content: str) -> None:
        self._goals_file.parent.mkdir(parents=True, exist_ok=True)
        self._goals_file.write_text(content, encoding="utf-8")

    def _parse_tasks(self) -> list[dict[str, Any]]:
        """Parse all tasks from GOALS.md into structured format."""
        content = self._read()
        tasks: list[dict[str, Any]] = []
        current_goal = ""

        for line in content.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("## Goal:"):
                current_goal = line_stripped[8:].strip()
            elif line_stripped.startswith("- [ ] "):
                text = line_stripped[6:]
                due = self._extract_due(text)
                tasks.append({"goal": current_goal, "text": text, "done": False, "due": due})
            elif line_stripped.startswith("- [x] ") or line_stripped.startswith("- [X] "):
                text = line_stripped[6:]
                due = self._extract_due(text)
                tasks.append({"goal": current_goal, "text": text, "done": True, "due": due})

        return tasks

    @staticmethod
    def _extract_due(text: str) -> str | None:
        match = re.search(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)", text)
        return match.group(1) if match else None

    async def execute(
        self,
        action: str,
        goal: str = "",
        subtask: str = "",
        due: str = "",
        index: int | None = None,
        new_text: str = "",
        **kwargs: Any,
    ) -> str:
        if action == "add":
            return self._add(goal, subtask, due)
        elif action == "complete":
            return self._complete(index)
        elif action == "list":
            return self._list()
        elif action == "remove":
            return self._remove(index)
        elif action == "update":
            return self._update(index, new_text)
        return f"Unknown action: {action}"

    def _add(self, goal: str, subtask: str, due: str) -> str:
        content = self._read()

        if goal and not subtask:
            # Add new goal section
            if f"## Goal: {goal}" in content:
                return f"Goal '{goal}' already exists. Add subtasks with subtask parameter."
            new_section = f"\n\n## Goal: {goal}\n"
            content = content.rstrip() + new_section
            self._write(content)
            return f"Created goal: {goal}"

        if subtask:
            due_str = f" (due: {due})" if due else ""
            task_line = f"- [ ] {subtask}{due_str}"

            if goal:
                # Add under specific goal
                marker = f"## Goal: {goal}"
                if marker not in content:
                    content = content.rstrip() + f"\n\n{marker}\n{task_line}\n"
                else:
                    # Find the goal section and append after last task
                    lines = content.split("\n")
                    insert_idx = None
                    in_goal = False
                    for i, line in enumerate(lines):
                        if line.strip() == marker:
                            in_goal = True
                            insert_idx = i + 1
                        elif in_goal and (line.startswith("- [") or line.strip() == ""):
                            insert_idx = i + 1 if line.startswith("- [") else i
                            if not line.startswith("- ["):
                                break
                        elif in_goal and line.startswith("##"):
                            break
                    if insert_idx:
                        lines.insert(insert_idx, task_line)
                        content = "\n".join(lines)
                    else:
                        content = content.rstrip() + f"\n{task_line}\n"
            else:
                # Add to a general section
                if "## Goal: General" not in content:
                    content = content.rstrip() + f"\n\n## Goal: General\n{task_line}\n"
                else:
                    content = content.replace("## Goal: General\n", f"## Goal: General\n{task_line}\n")

            self._write(content)
            return f"Added: {subtask}" + (f" (due: {due})" if due else "")

        return "Error: provide either a goal name or a subtask"

    def _complete(self, index: int | None) -> str:
        if index is None:
            return "Error: index required (1-based, from list output)"

        tasks = self._parse_tasks()
        pending = [(i, t) for i, t in enumerate(tasks) if not t["done"]]

        if index < 1 or index > len(pending):
            return f"Error: invalid index {index}. Use 'list' to see pending tasks."

        target = pending[index - 1][1]
        content = self._read()
        old_line = f"- [ ] {target['text']}"
        new_line = f"- [x] {target['text']}"
        content = content.replace(old_line, new_line, 1)
        self._write(content)
        return f"Completed: {target['text']}"

    def _list(self) -> str:
        tasks = self._parse_tasks()
        if not tasks:
            return "No goals or tasks tracked. Use goals(action='add', goal='...') to create one."

        lines = []
        pending_idx = 0
        current_goal = ""

        for t in tasks:
            if t["goal"] != current_goal:
                current_goal = t["goal"]
                lines.append(f"\n## {current_goal}")

            if t["done"]:
                lines.append(f"  ✓ {t['text']}")
            else:
                pending_idx += 1
                due_warn = ""
                if t["due"]:
                    try:
                        due_date = datetime.strptime(t["due"], "%Y-%m-%d").date()
                        days_left = (due_date - datetime.now().date()).days
                        if days_left < 0:
                            due_warn = " ⚠️ OVERDUE"
                        elif days_left <= 3:
                            due_warn = f" ⏰ {days_left}d left"
                    except ValueError:
                        pass
                lines.append(f"  {pending_idx}. [ ] {t['text']}{due_warn}")

        pending_count = sum(1 for t in tasks if not t["done"])
        done_count = sum(1 for t in tasks if t["done"])
        lines.insert(0, f"Goals: {pending_count} pending, {done_count} done")

        return "\n".join(lines)

    def _remove(self, index: int | None) -> str:
        if index is None:
            return "Error: index required"

        tasks = self._parse_tasks()
        pending = [(i, t) for i, t in enumerate(tasks) if not t["done"]]

        if index < 1 or index > len(pending):
            return f"Error: invalid index {index}"

        target = pending[index - 1][1]
        content = self._read()
        line = f"- [ ] {target['text']}"
        content = content.replace(line + "\n", "", 1)
        self._write(content)
        return f"Removed: {target['text']}"

    def _update(self, index: int | None, new_text: str) -> str:
        if index is None or not new_text:
            return "Error: index and new_text required"

        tasks = self._parse_tasks()
        pending = [(i, t) for i, t in enumerate(tasks) if not t["done"]]

        if index < 1 or index > len(pending):
            return f"Error: invalid index {index}"

        target = pending[index - 1][1]
        content = self._read()
        old_line = f"- [ ] {target['text']}"
        new_line = f"- [ ] {new_text}"
        content = content.replace(old_line, new_line, 1)
        self._write(content)
        return f"Updated: {target['text']} → {new_text}"
