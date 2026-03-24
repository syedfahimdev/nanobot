"""Skills Marketplace tool — search and install skills from skills.sh + clawhub.

Two sources:
1. skills.sh (npx skills) — original marketplace
2. clawhub (npx clawhub@latest) — newer, more skills, companion files

Mawa can search for skills when she encounters tasks she can't handle,
discover community skills, and install them with user approval.
"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
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
            "Search and install skills from two marketplaces:\n"
            "1. skills.sh — format: owner/repo@skill-name\n"
            "2. clawhub — format: skill-slug (downloads with companion files)\n"
            "Actions: search, install, list. Source: 'skills.sh' or 'clawhub'."
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
                    "description": "Search query. E.g., 'pdf', 'weather', 'excel'.",
                },
                "skill_id": {
                    "type": "string",
                    "description": "Skill ID. skills.sh: owner/repo@skill. clawhub: skill-slug.",
                },
                "source": {
                    "type": "string",
                    "enum": ["skills.sh", "clawhub", "both"],
                    "description": "Which marketplace. Default: both.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, query: str = "", skill_id: str = "", source: str = "both", **kwargs: Any) -> str:
        if action == "search":
            results = []
            if source in ("skills.sh", "both"):
                results.append(self._search_skillssh(query))
            if source in ("clawhub", "both"):
                results.append(self._search_clawhub(query))
            return "\n\n".join(r for r in results if r)
        elif action == "install":
            # Auto-detect source from skill_id format
            if "@" in skill_id:
                return self._install_skillssh(skill_id)
            else:
                return self._install_clawhub(skill_id)
        elif action == "list":
            return self._list()
        return f"Unknown action: {action}"

    def _search_skillssh(self, query: str) -> str:
        if not query or len(query) < 2:
            return "Error: query required (at least 2 characters)."

        try:
            import os
            result = subprocess.run(
                ["npx", "skills", "find", query],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"},
            )

            # Strip ANSI escape codes and parse
            ansi_re = re.compile(r'\x1b\[[0-9;]*m')
            output = ansi_re.sub("", result.stdout)
            results = []
            lines = output.split("\n")
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

    def _install_skillssh(self, skill_id: str) -> str:
        if not skill_id:
            return "Error: skill_id required (format: owner/repo@skill-name)."

        try:
            # Download skill directly from GitHub to workspace/skills/
            # Format: owner/repo@skill-name → https://raw.githubusercontent.com/owner/repo/main/skills/skill-name/SKILL.md
            parts = skill_id.split("@")
            if len(parts) != 2:
                return f"Error: Invalid skill_id format. Expected: owner/repo@skill-name, got: {skill_id}"

            owner_repo = parts[0]  # e.g., "steipete/clawdis"
            skill_name = parts[1]  # e.g., "weather"

            # Fetch SKILL.md from GitHub
            import httpx
            base_url = f"https://raw.githubusercontent.com/{owner_repo}/main/skills/{skill_name}/SKILL.md"
            resp = httpx.get(base_url, timeout=15, follow_redirects=True)

            if resp.status_code != 200:
                # Try alternate paths
                for path in [f"skills/{skill_name}/SKILL.md", f"{skill_name}/SKILL.md", f".skills/{skill_name}/SKILL.md"]:
                    alt_url = f"https://raw.githubusercontent.com/{owner_repo}/main/{path}"
                    resp = httpx.get(alt_url, timeout=10, follow_redirects=True)
                    if resp.status_code == 200:
                        break

            if resp.status_code != 200:
                return f"Error: Could not fetch skill from GitHub ({resp.status_code}). URL tried: {base_url}"

            # Save to workspace/skills/
            from nanobot.config.paths import get_workspace_path
            skill_dir = get_workspace_path() / "skills" / skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(resp.text, encoding="utf-8")

            # Try to download companion files (scripts, templates, configs)
            try:
                tree_url = f"https://api.github.com/repos/{owner_repo}/git/trees/main?recursive=1"
                tree_resp = httpx.get(tree_url, timeout=10)
                if tree_resp.status_code == 200:
                    tree = tree_resp.json().get("tree", [])
                    skill_prefix = f"skills/{skill_name}/"
                    companion_files = [f for f in tree if f["path"].startswith(skill_prefix) and f["type"] == "blob" and not f["path"].endswith("SKILL.md")]
                    for cf in companion_files[:10]:  # Max 10 companion files
                        file_url = f"https://raw.githubusercontent.com/{owner_repo}/main/{cf['path']}"
                        file_resp = httpx.get(file_url, timeout=10)
                        if file_resp.status_code == 200:
                            rel_path = cf["path"].replace(skill_prefix, "")
                            dest = skill_dir / rel_path
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(file_resp.content)
                            logger.debug("Skill companion file: {}", rel_path)
            except Exception as e:
                logger.debug("Could not fetch companion files: {}", e)

            logger.info("Skills marketplace: installed {} to {}", skill_id, skill_dir)
            return f"Skill '{skill_name}' installed to {skill_dir}. Read SKILL.md before using."

        except Exception as e:
            return f"Error installing: {e}"

    def _search_clawhub(self, query: str) -> str:
        """Search clawhub marketplace."""
        if not query or len(query) < 2:
            return ""
        try:
            result = subprocess.run(
                ["npx", "clawhub@latest", "search", query],
                capture_output=True, text=True, timeout=20,
                env={**__import__("os").environ, "NO_COLOR": "1"},
            )
            output = result.stdout.strip()
            # Parse clawhub output — format: "slug  Name  (score)"
            lines = [l.strip() for l in output.split("\n") if l.strip() and not l.startswith("-") and not l.startswith("[")]
            if not lines:
                return ""
            results = []
            for line in lines[:7]:
                parts = line.split("  ")
                slug = parts[0].strip() if parts else line.strip()
                name = parts[1].strip() if len(parts) > 1 else slug
                results.append(f"- **{slug}** — {name}")
            return f"**Clawhub** ({len(results)} results for '{query}'):\n" + "\n".join(results) + \
                f"\n\nTo install from clawhub: skills_marketplace(action='install', skill_id='{results[0].split('**')[1]}')" if results else ""
        except Exception as e:
            return f"Clawhub search error: {e}"

    def _install_clawhub(self, slug: str) -> str:
        """Install skill from clawhub to workspace/skills/."""
        if not slug:
            return "Error: skill slug required."
        try:
            from nanobot.config.paths import get_workspace_path
            skill_dir = get_workspace_path() / "skills"
            skill_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["npx", "clawhub@latest", "install", slug, "--force"],
                capture_output=True, text=True, timeout=30,
                cwd=str(skill_dir),
            )

            # clawhub creates skills/slug/ inside cwd — check for double nesting
            nested = skill_dir / "skills" / slug
            target = skill_dir / slug
            if nested.exists() and not target.exists():
                import shutil
                shutil.move(str(nested), str(target))
                # Clean up empty skills/skills/ dir
                try:
                    (skill_dir / "skills").rmdir()
                except OSError:
                    pass

            if target.exists() and (target / "SKILL.md").exists():
                files = list(target.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                logger.info("Clawhub: installed {} ({} files)", slug, file_count)
                return f"Skill '{slug}' installed from clawhub ({file_count} files). Read SKILL.md before using."
            else:
                error = result.stderr[:200] or result.stdout[:200]
                return f"Install may have failed: {error}"

        except Exception as e:
            return f"Error installing from clawhub: {e}"

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
