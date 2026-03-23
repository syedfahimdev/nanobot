"""Search past conversations and memory files.

Greps through session JSONL files and memory markdown files
to find matching content.
"""

from __future__ import annotations

import json
from pathlib import Path


def search_sessions(workspace: Path, query: str, max_results: int = 20) -> list[dict]:
    """Search through session JSONL files for matching messages."""
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        return []

    query_lower = query.lower()
    results: list[dict] = []

    for path in sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("_type") == "metadata":
                    continue
                content = data.get("content", "")
                if not content or query_lower not in content.lower():
                    continue

                # Extract snippet around match
                idx = content.lower().index(query_lower)
                start = max(0, idx - 60)
                end = min(len(content), idx + len(query) + 60)
                snippet = content[start:end].strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."

                results.append({
                    "session_key": path.stem,
                    "message": content[:200],
                    "role": data.get("role", ""),
                    "timestamp": data.get("timestamp", ""),
                    "snippet": snippet,
                })
                if len(results) >= max_results:
                    return results
        except (json.JSONDecodeError, OSError):
            continue

    return results


def search_memory(workspace: Path, query: str) -> list[dict]:
    """Search through memory/*.md files for matching lines."""
    mem_dir = workspace / "memory"
    if not mem_dir.exists():
        return []

    query_lower = query.lower()
    results: list[dict] = []

    for path in mem_dir.glob("*.md"):
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query_lower in line.lower():
                    results.append({
                        "file": path.name,
                        "line": i,
                        "snippet": line.strip()[:200],
                    })
        except OSError:
            continue

    return results


def search_all(workspace: Path, query: str) -> dict:
    """Search both sessions and memory, return combined results."""
    return {
        "sessions": search_sessions(workspace, query),
        "memory": search_memory(workspace, query),
    }
