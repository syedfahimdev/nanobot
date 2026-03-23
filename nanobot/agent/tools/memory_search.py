"""Local grep-based memory search tool.

Searches memory files (LONG_TERM.md, SHORT_TERM.md, EPISODES.md,
OBSERVATIONS.md, HISTORY.md) and inbox files using keyword matching.
No external dependencies — works entirely on local filesystem.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class MemorySearchTool(Tool):
    """Search memory and inbox files locally with keyword matching."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._memory_dir = workspace / "memory"
        self._inbox_dir = workspace / "inbox"

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search memory files and inbox documents for information. "
            "Uses keyword matching across LONG_TERM.md, SHORT_TERM.md, "
            "EPISODES.md, OBSERVATIONS.md, HISTORY.md, and uploaded inbox files. "
            "Returns matching snippets with surrounding context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — keywords to find in memory files.",
                },
                "folder": {
                    "type": "string",
                    "description": "Restrict search to a specific source: 'memory', 'inbox', or 'all'.",
                    "enum": ["memory", "inbox", "all"],
                    "default": "all",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 10,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        if not query:
            return "Error: query is required."

        folder = kwargs.get("folder", "all")
        top_k = min(kwargs.get("top_k", 10), 20)

        results: list[dict[str, str]] = []

        # Build search pattern — match any word in the query
        words = [w for w in query.split() if len(w) >= 2]
        if not words:
            return "Error: query too short."
        pattern = re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)

        # Search memory files
        if folder in ("memory", "all") and self._memory_dir.exists():
            for md_file in sorted(self._memory_dir.glob("*.md")):
                if len(results) >= top_k:
                    break
                self._search_file(md_file, pattern, results, top_k)

        # Search inbox files
        if folder in ("inbox", "all") and self._inbox_dir.exists():
            for subfolder in ("work", "personal", "general"):
                inbox_sub = self._inbox_dir / subfolder
                if not inbox_sub.exists():
                    continue
                for f in sorted(inbox_sub.iterdir()):
                    if len(results) >= top_k:
                        break
                    if f.is_file() and f.suffix in (".md", ".txt", ".csv"):
                        self._search_file(f, pattern, results, top_k)

        if not results:
            return f"No results found for '{query}'."

        # Format output
        lines = [f"Found {len(results)} result(s) for '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"--- Result {i} ({r['source']}) ---")
            lines.append(r["snippet"])
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _search_file(
        path: Path,
        pattern: re.Pattern,
        results: list[dict[str, str]],
        top_k: int,
    ) -> None:
        """Search a single file and append matching snippets."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return

        lines = content.split("\n")
        matched_ranges: set[int] = set()

        for i, line in enumerate(lines):
            if pattern.search(line) and i not in matched_ranges:
                # Context: 2 lines before, 2 lines after
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet = "\n".join(lines[start:end]).strip()
                if snippet and len(snippet) > 5:
                    results.append({
                        "source": path.name,
                        "snippet": snippet[:500],
                    })
                    # Mark lines as matched to avoid duplicate snippets
                    for j in range(start, end):
                        matched_ranges.add(j)
                    if len(results) >= top_k:
                        return
