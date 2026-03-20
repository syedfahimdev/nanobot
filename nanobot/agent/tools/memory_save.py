"""Memory save tool — routes new knowledge to the right file automatically.

The agent calls this when it learns something worth remembering.
Code logic routes to the correct file based on category — no prompt bloat needed.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:60]


class MemorySaveTool(Tool):
    """Save knowledge, learnings, or corrections to persistent memory files."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "memory_save"

    @property
    def description(self) -> str:
        return (
            "Save something worth remembering long-term. "
            "Use when you learn a new fact about the user, receive a correction, "
            "discover a preference, or encounter something the user will want to recall later. "
            "Categories: knowledge (facts about people/projects/things), "
            "learning (corrections, best practices), "
            "rule (task-specific rules for future use)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["knowledge", "learning", "rule"],
                    "description": "Type of memory to save.",
                },
                "topic": {
                    "type": "string",
                    "description": "Short topic name (e.g. 'tonni', 'car-buying', 'email-drafting').",
                },
                "content": {
                    "type": "string",
                    "description": "The information to save. Markdown formatted.",
                },
                "subcategory": {
                    "type": "string",
                    "description": "Optional subfolder for knowledge (e.g. 'people', 'projects', 'references').",
                },
            },
            "required": ["category", "topic", "content"],
        }

    async def execute(
        self,
        category: str,
        topic: str,
        content: str,
        subcategory: str = "",
        **kwargs: Any,
    ) -> str:
        topic_slug = _slugify(topic)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        if category == "knowledge":
            return self._save_knowledge(topic, topic_slug, content, subcategory)
        elif category == "learning":
            return self._save_learning(topic, content, now)
        elif category == "rule":
            return self._save_rule(topic, topic_slug, content)
        else:
            return f"Error: unknown category '{category}'. Use: knowledge, learning, rule."

    def _save_knowledge(self, topic: str, slug: str, content: str, subcategory: str) -> str:
        """Save or update a knowledge file."""
        sub = _slugify(subcategory) if subcategory else "general"
        base = self._workspace / "knowledge" / sub
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{slug}.md"

        if path.exists():
            # Append to existing file
            existing = path.read_text(encoding="utf-8")
            updated = f"{existing.rstrip()}\n\n{content}\n"
            path.write_text(updated, encoding="utf-8")
            logger.info("Memory save: updated knowledge/{}/{}", sub, slug)
            return f"Updated knowledge file: {path}"
        else:
            # Create new file
            header = f"# {topic}\n\n{content}\n"
            path.write_text(header, encoding="utf-8")
            logger.info("Memory save: created knowledge/{}/{}", sub, slug)
            return f"Created knowledge file: {path}"

    def _save_learning(self, topic: str, content: str, timestamp: str) -> str:
        """Append a learning entry."""
        path = self._workspace / "learnings" / "LEARNINGS.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        entry = f"\n### {topic} ({timestamp})\n{content}\n"

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            # Insert before the trailing ---
            if existing.rstrip().endswith("---"):
                updated = existing.rstrip()[:-3] + entry + "\n---\n"
            else:
                updated = existing.rstrip() + "\n" + entry + "\n---\n"
            path.write_text(updated, encoding="utf-8")
        else:
            path.write_text(f"# Learnings Log\n\n---\n{entry}\n---\n", encoding="utf-8")

        logger.info("Memory save: added learning '{}'", topic)
        return f"Saved learning: {topic}"

    def _save_rule(self, topic: str, slug: str, content: str) -> str:
        """Save a task-specific rule."""
        base = self._workspace / "rules"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{slug}.md"

        header = f"# {topic}\n\n{content}\n"
        path.write_text(header, encoding="utf-8")
        logger.info("Memory save: created rule '{}'", slug)
        return f"Saved rule: {path}"
