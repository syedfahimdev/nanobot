"""Spawn, update, and list tools for background subagents."""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


# ── Dynamic specialist prompt builder ────────────────────────────────────────
# Generates a focused specialist prompt on-the-fly based on task analysis.
# No hardcoded profiles — builds from task characteristics.

import re

# Skill atoms — composable behaviors detected from task keywords
_SKILL_ATOMS: list[tuple[list[str], str]] = [
    # Research skills
    (["research", "find", "look up", "search for", "investigate"], "Search multiple sources, not just one. Cross-reference findings. Cite URLs."),
    (["best", "top", "recommended", "review"], "Include ratings, reviews, and rankings where available. Compare at least 3 options."),
    (["price", "cost", "cheap", "expensive", "budget", "deal"], "Include prices, costs, and value comparisons. Note any deals or discounts."),

    # Analysis skills
    (["compare", "vs", "versus", "difference", "pros and cons"], "Create a comparison table with pros/cons for each option. Give a clear recommendation."),
    (["analyze", "evaluate", "assess", "metrics"], "Extract key metrics and numbers. Identify trends and patterns. Provide data-driven insights."),

    # Writing skills
    (["write", "draft", "compose"], "Write in a clear, professional style. Proofread for grammar."),
    (["email", "letter", "message to"], "Use appropriate greeting and sign-off. Match the formality to the recipient."),
    (["report", "document", "summary"], "Use headings, sections, and bullet points. Include an executive summary."),
    (["blog", "article", "post"], "Write engagingly with a clear hook. Use subheadings for scanability."),

    # Coding skills
    (["code", "script", "program", "implement"], "Write clean, well-commented code. Follow existing project patterns."),
    (["debug", "fix", "error", "bug"], "Read the error carefully. Trace the root cause. Test the fix."),
    (["refactor", "improve", "optimize"], "Keep functionality identical. Improve readability and performance. Explain what changed."),

    # Planning skills
    (["plan", "roadmap", "timeline"], "Create numbered steps with clear milestones. Estimate effort per step."),
    (["break down", "steps to", "how to"], "Break into small actionable steps. Identify dependencies between steps."),
    (["organize", "schedule", "coordinate"], "Prioritize by impact and urgency. Flag risks and blockers."),

    # Thoroughness skills
    (["comprehensive", "thorough", "detailed", "everything"], "Be exhaustive — cover every angle. Don't skip edge cases. Length is fine."),
    (["quick", "brief", "fast", "just"], "Be concise — get to the point fast. Skip unnecessary detail."),

    # Domain skills
    (["legal", "law", "court", "ticket"], "Be precise with legal terminology. Note jurisdiction differences. Suggest professional help for complex matters."),
    (["medical", "health", "symptom"], "Provide general info only. Always recommend consulting a doctor for medical decisions."),
    (["financial", "invest", "stock", "crypto"], "Include risk disclaimers. Use current data. Note that past performance doesn't predict future."),
    (["travel", "flight", "hotel", "trip"], "Include booking links. Note cancellation policies. Check seasonal pricing."),
    (["food", "restaurant", "recipe", "cook"], "Include ratings and price ranges. Note dietary options. Include addresses for restaurants."),
]


def _find_relevant_skills(task: str, workspace_path: str = "") -> list[dict]:
    """Find installed skills relevant to the task. Zero LLM cost."""
    from pathlib import Path

    workspace = Path(workspace_path) if workspace_path else Path("/root/.nanobot/workspace")
    skills_dir = workspace / "skills"
    if not skills_dir.exists():
        return []

    task_lower = task.lower()
    relevant = []

    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
            continue

        skill_name = skill_dir.name.replace("-", " ").replace("_", " ").lower()
        skill_md = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

        # Check if skill name or description matches task keywords
        # Extract description from frontmatter
        desc = ""
        for line in skill_md.split("\n")[:15]:
            if line.startswith("description:"):
                desc = line.split(":", 1)[1].strip().strip('"').strip("'").lower()
                break

        # Match by meaningful words (>2 chars, not stopwords)
        stopwords = {"the", "and", "for", "with", "from", "this", "that", "your", "when", "what", "how", "use", "can"}
        name_words = {w for w in skill_name.split() if len(w) > 2 and w not in stopwords}
        task_words = {w for w in task_lower.split() if len(w) > 2 and w not in stopwords}
        overlap = name_words & task_words

        # Check description for keyword matches (substring to handle plurals)
        desc_match = sum(1 for w in task_words if w in desc)

        if len(overlap) >= 1 or desc_match >= 2:
            relevant.append({
                "name": skill_dir.name,
                "path": str(skill_dir / "SKILL.md"),
                "description": desc[:100],
            })

    return relevant[:3]  # Max 3 relevant skills


def build_specialist_prompt(task: str, workspace: str = "") -> tuple[str, str]:
    """Dynamically build a specialist prompt from task analysis.

    Includes:
    - Composable skill atoms from task keywords
    - Relevant installed skills (auto-detected)
    - Marketplace hint if no matching skill installed

    Returns (prompt, label). Prompt is empty if task is too simple.
    Zero LLM cost — pure keyword matching and composition.
    """
    task_lower = task.lower()
    matched_skills = []

    for keywords, skill in _SKILL_ATOMS:
        if any(kw in task_lower for kw in keywords):
            matched_skills.append(skill)

    # Check for relevant installed skills even if no atoms matched
    relevant_skills = _find_relevant_skills(task, workspace)

    if not matched_skills and not relevant_skills:
        return "", ""

    # Build a focused prompt from matched skills
    prompt_parts = [
        "You are a specialist subagent. Focus ONLY on the assigned task.",
        "Do a better job than a generic assistant would — go deeper, be more thorough.",
        "",
        "## Your specialized approach for this task:",
    ]
    # Deduplicate while preserving order
    seen = set()
    for skill in matched_skills:
        if skill not in seen:
            seen.add(skill)
            prompt_parts.append(f"- {skill}")

    # Attach relevant installed skills (already found above)
    if relevant_skills:
        prompt_parts.append("")
        prompt_parts.append("## Available skills for this task:")
        prompt_parts.append("Read the SKILL.md file BEFORE using any skill tools.")
        for s in relevant_skills:
            prompt_parts.append(f"- **{s['name']}**: `read_file(\"{s['path']}\")`")
            if s["description"]:
                prompt_parts.append(f"  {s['description']}")
    else:
        # No relevant skill installed — hint to search marketplace
        # Check if auto-install is enabled
        auto_install = False
        try:
            from nanobot.hooks.builtin.feature_registry import get_setting
            from pathlib import Path
            ws = Path(workspace) if workspace else Path("/root/.nanobot/workspace")
            auto_install = get_setting(ws, "subagentAutoInstallSkills", False)
        except Exception:
            pass

        prompt_parts.append("")
        prompt_parts.append("## Skills:")
        prompt_parts.append("If you need a specialized tool, search the marketplace: `skills_marketplace(action=\"search\", query=\"...\")`")
        if auto_install:
            prompt_parts.append("You MAY install skills automatically without asking: `skills_marketplace(action=\"install\", skill_id=\"...\")`")
        else:
            prompt_parts.append("Do NOT install skills without user approval. Report what you found and recommend which to install.")

    prompt_parts.extend([
        "",
        "## Output quality:",
        "- Structure your response with clear headings",
        "- Include specific details, not vague generalizations",
        "- If you searched for info, cite your sources",
        "- End with a clear conclusion or recommendation",
    ])

    # Generate a short label
    label_keywords = []
    label_map = {
        "research": "Research", "compare": "Analysis", "write": "Writing",
        "code": "Code", "debug": "Debug", "plan": "Planning",
        "analyze": "Analysis", "find": "Search", "draft": "Draft",
    }
    for kw, lbl in label_map.items():
        if kw in task_lower and lbl not in label_keywords:
            label_keywords.append(lbl)
    label = " + ".join(label_keywords[:2]) if label_keywords else "Specialist"

    return "\n".join(prompt_parts), label


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
            "Spawn a specialist subagent for background task execution. "
            "Automatically generates a focused specialist prompt based on the task — "
            "no need to specify a type. The subagent gets specialized instructions "
            "for research, analysis, writing, coding, planning, or domain-specific work. "
            "Use for 3+ tool call tasks. For quick tasks, handle directly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Detailed task description. Be specific about what you want — "
                    "the specialist prompt is generated from the task content.",
                },
                "label": {
                    "type": "string",
                    "description": "Short label for display (auto-generated if omitted)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        # Dynamically build specialist prompt from task
        workspace = str(self._manager.workspace) if hasattr(self._manager, 'workspace') else ""
        specialist_prompt, auto_label = build_specialist_prompt(task, workspace)

        enhanced_task = task
        if specialist_prompt:
            enhanced_task = f"{specialist_prompt}\n\n## Your Task\n{task}"
            label = label or f"{auto_label}: {task[:30]}"

        result = await self._manager.spawn(
            task=enhanced_task,
            label=label or task[:30],
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )

        return result


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
                bg_task = self._manager._running_tasks.get(task_id)
                if bg_task and not bg_task.done():
                    bg_task.cancel()
                    await asyncio.gather(bg_task, return_exceptions=True)
                return f"Cancelled task [{task_id}]: {t['label']}"

        return f"Task '{task_id}' not found. Use list_subagents to see running tasks."
