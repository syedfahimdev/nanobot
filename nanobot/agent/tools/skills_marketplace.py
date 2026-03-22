"""Skills Marketplace tool — search and install skills from skills.sh.

Mawa can search for skills when she encounters tasks she can't handle,
discover community skills, and install them with user approval.
"""

from __future__ import annotations

import subprocess
import re
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class SkillsMarketplaceTool(Tool):
    """Search and install skills from skills.sh marketplace."""

    @property
    def name(self) -> str:
        return "skills_marketplace"

    @property
    def description(self) -> str:
        return (
            "Search the skills.sh marketplace to find and install community skills. "
            "Use when: you can't complete a task and need a new skill, "
            "the user asks to find/install a skill, "
            "or you want to suggest a skill that could help. "
            "Actions: search (find skills by keyword), install (add a skill), "
            "list (show installed skills)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "install", "list"],
                    "description": "Action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action). E.g., 'pdf', 'frontend', 'slack'.",
                },
                "skill_id": {
                    "type": "string",
                    "description": "Skill ID to install (for install action). Format: owner/repo@skill-name.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, query: str = "", skill_id: str = "", **kwargs: Any) -> str:
        if action == "search":
            return self._search(query)
        elif action == "install":
            return self._install(skill_id)
        elif action == "list":
            return self._list()
        return f"Unknown action: {action}"

    def _search(self, query: str) -> str:
        if not query or len(query) < 2:
            return "Error: query required (at least 2 characters)."

        try:
            result = subprocess.run(
                ["npx", "skills", "find", query],
                capture_output=True, text=True, timeout=15,
            )

            # Parse results
            results = []
            lines = result.stdout.split("\n")
            for i, line in enumerate(lines):
                m = re.search(r"(\S+/\S+@\S+)", line)
                if m:
                    skill_id = m.group(1)
                    installs_m = re.search(r"([\d.]+[KM]?)\s*installs", line)
                    installs = installs_m.group(1) if installs_m else "0"
                    url = ""
                    if i + 1 < len(lines):
                        url_m = re.search(r"(https://skills\.sh/\S+)", lines[i + 1])
                        if url_m:
                            url = url_m.group(1)
                    results.append(f"- **{skill_id}** ({installs} installs)" + (f"\n  {url}" if url else ""))

            if not results:
                return f"No skills found for '{query}' on skills.sh."

            return f"Found {len(results)} skills for '{query}':\n\n" + "\n".join(results) + \
                f"\n\nTo install: skills_marketplace(action='install', skill_id='owner/repo@skill-name')"

        except subprocess.TimeoutExpired:
            return "Search timed out."
        except Exception as e:
            return f"Error searching: {e}"

    def _install(self, skill_id: str) -> str:
        if not skill_id:
            return "Error: skill_id required (format: owner/repo@skill-name)."

        try:
            result = subprocess.run(
                ["npx", "skills", "add", skill_id],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("Skills marketplace: installed {}", skill_id)
                return f"Skill '{skill_id}' installed successfully from skills.sh."
            else:
                error = result.stderr[:200] or result.stdout[:200]
                return f"Install failed: {error}"
        except subprocess.TimeoutExpired:
            return "Install timed out."
        except Exception as e:
            return f"Error installing: {e}"

    def _list(self) -> str:
        try:
            result = subprocess.run(
                ["npx", "skills", "list"],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
            if "No " in output and "found" in output:
                return "No skills installed from skills.sh."
            return f"Installed skills:\n{output}"
        except Exception as e:
            return f"Error listing: {e}"
