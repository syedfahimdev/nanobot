"""Skill Creator — Mawa learns new skills from descriptions, URLs, or observation.

Following Anthropic's skill-creator patterns:
- Skills are SKILL.md files with YAML frontmatter (name, description)
- Progressive disclosure: metadata → instructions → bundled resources
- Description optimized for triggering accuracy
- Skills registered in ToolsDNS for discovery

Actions:
  create: Generate a new skill from a description + optional docs URL
  improve: Improve an existing skill based on feedback
  list: List all skills
  test: Test a skill with a sample prompt
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_SKILL_GENERATOR_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "generate_skill",
            "description": "Generate a skill definition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name in kebab-case (e.g., 'jira-ticket-creator')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Triggering description (100-200 words). Imperative voice. Focus on user intent. Be pushy — include explicit contexts when this skill should activate.",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Full skill instructions as markdown. Include: workflow phases, tool usage, rules, examples.",
                    },
                    "tools": {
                        "type": "string",
                        "description": "JSON array of tool definitions if the skill needs new tools. Each: {name, description, parameters}. Empty array if using existing tools.",
                    },
                },
                "required": ["name", "description", "instructions"],
            },
        },
    }
]


class SkillCreatorTool(Tool):
    """Create, improve, and manage skills for Mawa."""

    def __init__(self, workspace: Path, provider: "LLMProvider", model: str):
        self._workspace = workspace
        self._provider = provider
        self._model = model
        self._skills_dir = workspace / "skills"

    @property
    def name(self) -> str:
        return "learn_skill"

    @property
    def description(self) -> str:
        return (
            "Teach Mawa a new skill or improve an existing one. "
            "Create skills from descriptions, documentation URLs, or observed patterns. "
            "Actions: create (new skill), improve (refine existing), list (show all), delete (remove)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "improve", "list", "delete"],
                    "description": "Action to perform.",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill name in kebab-case (for create/improve/delete).",
                },
                "skill_description": {
                    "type": "string",
                    "description": "What the skill should do (for create). Be detailed about when to trigger and what to accomplish.",
                },
                "docs_url": {
                    "type": "string",
                    "description": "URL to documentation or reference material (optional, for create).",
                },
                "feedback": {
                    "type": "string",
                    "description": "What to improve (for improve action).",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        skill_name: str = "",
        skill_description: str = "",
        docs_url: str = "",
        feedback: str = "",
        **kwargs: Any,
    ) -> str:
        if action == "create":
            return await self._create(skill_name, skill_description, docs_url)
        elif action == "improve":
            return await self._improve(skill_name, feedback)
        elif action == "list":
            return self._list()
        elif action == "delete":
            return self._delete(skill_name)
        return f"Unknown action: {action}"

    async def _create(self, name: str, description: str, docs_url: str) -> str:
        if not name or not description:
            return "Error: skill_name and skill_description required."

        # Sanitize name
        import re
        name = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
        if not name:
            return "Error: invalid skill name."

        skill_dir = self._skills_dir / name
        if skill_dir.exists():
            return f"Skill '{name}' already exists. Use action='improve' to update it."

        # Fetch docs if URL provided
        docs_context = ""
        if docs_url:
            try:
                import httpx
                resp = httpx.get(docs_url, timeout=15, follow_redirects=True)
                if resp.status_code == 200:
                    # Extract text content (strip HTML if needed)
                    content = resp.text[:8000]
                    docs_context = f"\n\n## Reference Documentation (from {docs_url})\n{content}"
            except Exception as e:
                docs_context = f"\n\n(Failed to fetch docs: {e})"

        # Generate skill using LLM
        prompt = f"""Create a skill definition for nanobot (personal AI assistant).

Skill request: {description}
{docs_context}

Requirements:
- name: {name}
- description: 100-200 words, imperative voice, focus on user intent, be "pushy" about when to trigger
- instructions: detailed workflow with phases, tool usage, rules
- Follow Anthropic's skill format: YAML frontmatter (name, description) + markdown body

Call generate_skill with your skill definition."""

        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a skill designer for a personal AI assistant. Create clear, actionable skill definitions."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SKILL_GENERATOR_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "generate_skill"}},
            )

            if not response.has_tool_calls:
                return "Error: skill generation failed."

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            skill_name = args.get("name", name)
            skill_desc = args.get("description", description)
            instructions = args.get("instructions", "")

            # Create skill directory and SKILL.md
            skill_dir.mkdir(parents=True, exist_ok=True)

            skill_md = f"""---
name: {skill_name}
description: "{skill_desc}"
---

{instructions}
"""
            (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

            # Register with ToolsDNS if available
            registered = await self._register_skill(skill_name, skill_desc, instructions)

            logger.info("Skill created: {} at {}", skill_name, skill_dir)
            result = f"Skill '{skill_name}' created at {skill_dir}/SKILL.md"
            if registered:
                result += "\nRegistered with ToolsDNS for discovery."
            result += f"\n\nDescription: {skill_desc[:200]}"
            return result

        except Exception as e:
            return f"Error creating skill: {e}"

    async def _register_skill(self, name: str, description: str, content: str) -> bool:
        """Register the skill with ToolsDNS MCP."""
        try:
            from nanobot.config.loader import load_config
            config = load_config()
            td = getattr(getattr(config, "tools", None), "toolsdns", None)
            if not td or not td.url:
                return False

            import httpx
            resp = httpx.post(
                f"{td.url}/v1/skills",
                json={"name": name, "description": description, "content": content},
                headers={"Authorization": f"Bearer {td.api_key}"},
                timeout=10,
            )
            return resp.status_code in (200, 201)
        except Exception:
            return False

    async def _improve(self, name: str, feedback: str) -> str:
        if not name:
            return "Error: skill_name required."

        skill_dir = self._skills_dir / name
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return f"Skill '{name}' not found."

        current = skill_file.read_text(encoding="utf-8")

        prompt = f"""Improve this skill based on the feedback.

## Current SKILL.md
{current}

## Feedback
{feedback}

Call generate_skill with the improved version. Keep the same name."""

        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "Improve the skill definition based on feedback. Maintain the format."},
                    {"role": "user", "content": prompt},
                ],
                tools=_SKILL_GENERATOR_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "generate_skill"}},
            )

            if not response.has_tool_calls:
                return "Error: improvement failed."

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            skill_desc = args.get("description", "")
            instructions = args.get("instructions", "")

            skill_md = f"""---
name: {name}
description: "{skill_desc}"
---

{instructions}
"""
            skill_file.write_text(skill_md, encoding="utf-8")
            await self._register_skill(name, skill_desc, instructions)

            return f"Skill '{name}' improved and saved."
        except Exception as e:
            return f"Error improving skill: {e}"

    def _list(self) -> str:
        if not self._skills_dir.exists():
            return "No skills directory. Create your first skill!"

        skills = []
        for d in sorted(self._skills_dir.iterdir()):
            if d.is_dir() and (d / "SKILL.md").exists():
                content = (d / "SKILL.md").read_text(encoding="utf-8")
                # Extract description from frontmatter
                desc = ""
                if "description:" in content:
                    for line in content.split("\n"):
                        if line.strip().startswith("description:"):
                            desc = line.split(":", 1)[1].strip().strip('"\'')[:100]
                            break
                skills.append(f"- **{d.name}**: {desc}")

        if not skills:
            return "No skills created yet. Use learn_skill(action='create') to create one."

        return f"Skills ({len(skills)}):\n" + "\n".join(skills)

    def _delete(self, name: str) -> str:
        if not name:
            return "Error: skill_name required."
        skill_dir = self._skills_dir / name
        if not skill_dir.exists():
            return f"Skill '{name}' not found."

        import shutil
        shutil.rmtree(skill_dir)
        return f"Skill '{name}' deleted."
