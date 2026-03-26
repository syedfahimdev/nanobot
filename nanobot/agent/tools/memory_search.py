"""Local grep-based memory search tool.

Searches memory files (LONG_TERM.md, SHORT_TERM.md, EPISODES.md,
OBSERVATIONS.md, HISTORY.md) and inbox files using keyword matching.
No external dependencies — works entirely on local filesystem.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool

# Zero-cost synonym expansion for lightweight semantic search (#7)
_SYNONYM_MAP: dict[str, list[str]] = {
    "weather": ["temperature", "forecast", "rain", "sunny", "cloud", "wind", "storm", "humid"],
    "email": ["inbox", "mail", "message", "gmail", "outlook", "send", "received"],
    "money": ["budget", "cost", "payment", "price", "expense", "salary", "income", "financial"],
    "work": ["job", "office", "meeting", "project", "task", "deadline", "colleague"],
    "health": ["doctor", "medical", "exercise", "fitness", "symptom", "medicine", "hospital"],
    "food": ["restaurant", "recipe", "cook", "meal", "dinner", "lunch", "breakfast", "eat"],
    "travel": ["flight", "hotel", "trip", "booking", "vacation", "airport", "destination"],
    "code": ["programming", "bug", "feature", "deploy", "commit", "debug", "error", "function"],
    "music": ["song", "playlist", "album", "artist", "spotify", "listen"],
    "shopping": ["buy", "order", "purchase", "cart", "delivery", "amazon", "store"],
    "schedule": ["calendar", "appointment", "reminder", "event", "meeting", "plan"],
    "family": ["parent", "child", "wife", "husband", "brother", "sister", "mom", "dad"],
    "car": ["drive", "vehicle", "parking", "fuel", "gas", "repair", "mechanic"],
    "home": ["house", "apartment", "rent", "mortgage", "room", "clean", "furniture"],
    "study": ["learn", "course", "exam", "class", "lecture", "homework", "school", "university"],
    "photo": ["image", "picture", "camera", "screenshot", "gallery"],
    "news": ["article", "headline", "update", "report", "media"],
    "call": ["phone", "dial", "ring", "voicemail", "contact"],
    "sleep": ["nap", "rest", "bedtime", "alarm", "wake", "insomnia"],
    "movie": ["film", "watch", "cinema", "show", "series", "netflix", "stream"],
}

# Build reverse map for bidirectional lookup
_REVERSE_SYNONYMS: dict[str, list[str]] = {}
for _key, _vals in _SYNONYM_MAP.items():
    for _v in _vals:
        _REVERSE_SYNONYMS.setdefault(_v.lower(), []).append(_key)


def _expand_query_words(words: list[str]) -> list[str]:
    """Expand query words with synonyms for better recall."""
    expanded = set(words)
    for w in words:
        wl = w.lower()
        if wl in _SYNONYM_MAP:
            expanded.update(_SYNONYM_MAP[wl])
        if wl in _REVERSE_SYNONYMS:
            expanded.update(_REVERSE_SYNONYMS[wl])
    return list(expanded)


def _parse_entry_date(line: str) -> date | None:
    """Extract date from a [YYYY-MM-DD ...] timestamp line."""
    m = re.match(r"^\[(\d{4}-\d{2}-\d{2})", line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    return None


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
            "Returns matching snippets with surrounding context. "
            "Supports date-range filtering and semantic keyword expansion."
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
                "date_from": {
                    "type": "string",
                    "description": "Filter results from this date (YYYY-MM-DD). Only applies to HISTORY.md entries.",
                },
                "date_to": {
                    "type": "string",
                    "description": "Filter results up to this date (YYYY-MM-DD). Only applies to HISTORY.md entries.",
                },
                "semantic": {
                    "type": "boolean",
                    "description": "Enable semantic keyword expansion for broader matching.",
                    "default": False,
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
        semantic = kwargs.get("semantic", False)

        # Parse date range filters
        date_from: date | None = None
        date_to: date | None = None
        if kwargs.get("date_from"):
            try:
                date_from = datetime.strptime(kwargs["date_from"], "%Y-%m-%d").date()
            except ValueError:
                pass
        if kwargs.get("date_to"):
            try:
                date_to = datetime.strptime(kwargs["date_to"], "%Y-%m-%d").date()
            except ValueError:
                pass

        results: list[dict[str, str]] = []

        # Build search pattern — match any word in the query
        words = [w for w in query.split() if len(w) >= 2]
        if not words:
            return "Error: query too short."

        # Semantic keyword expansion (#7)
        search_words = _expand_query_words(words) if semantic else words
        pattern = re.compile("|".join(re.escape(w) for w in search_words), re.IGNORECASE)

        # Search memory files (exclude internal system files)
        _EXCLUDE = {"CORRECTIONS.md", "TOOL_LEARNINGS.md", "PROMPT_STATS.json", ".workflow_state.json"}
        if folder in ("memory", "all") and self._memory_dir.exists():
            for md_file in sorted(self._memory_dir.glob("*.md")):
                if md_file.name in _EXCLUDE:
                    continue
                if len(results) >= top_k:
                    break
                # Date-range filtering for HISTORY.md (#2)
                if md_file.name == "HISTORY.md" and (date_from or date_to):
                    self._search_file_with_dates(md_file, pattern, results, top_k, date_from, date_to)
                else:
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

        # If keyword search returns few results and semantic is enabled,
        # retry with expanded keywords (#7 fallback)
        if semantic and len(results) < 3 and search_words == words:
            expanded = _expand_query_words(words)
            if len(expanded) > len(words):
                exp_pattern = re.compile("|".join(re.escape(w) for w in expanded), re.IGNORECASE)
                if folder in ("memory", "all") and self._memory_dir.exists():
                    for md_file in sorted(self._memory_dir.glob("*.md")):
                        if md_file.name in _EXCLUDE or len(results) >= top_k:
                            break
                        self._search_file(md_file, exp_pattern, results, top_k)

        if not results:
            return f"No results found for '{query}'."

        # Format output
        suffix = " (semantic expanded)" if semantic else ""
        lines = [f"Found {len(results)} result(s) for '{query}'{suffix}:\n"]
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

    @staticmethod
    def _search_file_with_dates(
        path: Path,
        pattern: re.Pattern,
        results: list[dict[str, str]],
        top_k: int,
        date_from: date | None,
        date_to: date | None,
    ) -> None:
        """Search HISTORY.md with date-range filtering on [YYYY-MM-DD] timestamps."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return

        # Split into entries by double newline (each entry starts with [YYYY-MM-DD HH:MM])
        entries = content.split("\n\n")
        for entry in entries:
            if len(results) >= top_k:
                return
            entry = entry.strip()
            if not entry:
                continue
            # Check date filter
            entry_date = _parse_entry_date(entry)
            if entry_date:
                if date_from and entry_date < date_from:
                    continue
                if date_to and entry_date > date_to:
                    continue
            elif date_from or date_to:
                # Entry has no parseable date — skip when date filter is active
                continue
            # Check keyword match
            if pattern.search(entry):
                results.append({
                    "source": path.name,
                    "snippet": entry[:500],
                })
