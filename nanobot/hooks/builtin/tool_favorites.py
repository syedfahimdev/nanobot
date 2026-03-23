"""Auto-detect favorite tools from usage patterns.

Reads tool_scores.json (maintained by tool_scores.py) and ranks
tools by total call count to surface quick-action suggestions.
"""

from __future__ import annotations

import json
from pathlib import Path

# Maps common MCP tool names to natural language prompts
_PROMPT_MAP: dict[str, str] = {
    "GMAIL_FETCH_EMAILS": "Check my email",
    "GMAIL_SEND_EMAIL": "Send an email",
    "GMAIL_SEARCH_EMAILS": "Search my email",
    "WEATHERMAP_WEATHER": "What's the weather?",
    "GOOGLE_CALENDAR_LIST_EVENTS": "What's on my calendar?",
    "GOOGLE_CALENDAR_CREATE_EVENT": "Schedule an event",
    "web_search": "Search the web",
    "exec": "Run a command",
    "read_file": "Read a file",
    "SPOTIFY_GET_CURRENT_TRACK": "What's playing?",
    "SPOTIFY_PLAY": "Play music",
    "HOME_ASSISTANT_CALL_SERVICE": "Control smart home",
    "TELEGRAM_SEND_MESSAGE": "Send a Telegram message",
    "NOTION_QUERY_DATABASE": "Search my notes",
}


def get_favorites(workspace: Path, limit: int = 6) -> list[dict]:
    """Return top tools ranked by usage count.

    Reads the same tool_scores.json that ToolScorer maintains.
    """
    scores_file = workspace / "memory" / "tool_scores.json"
    if not scores_file.exists():
        return []

    try:
        scores = json.loads(scores_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    # Build ranked list
    ranked: list[tuple[str, int, str]] = []
    for name, data in scores.items():
        total = data.get("success", 0) + data.get("fail", 0)
        if total == 0:
            continue
        last_used = data.get("last_used", "")
        ranked.append((name, total, last_used))

    ranked.sort(key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for name, count, last_used in ranked[:limit]:
        results.append({
            "name": name,
            "count": count,
            "last_used": last_used,
            "suggested_prompt": _PROMPT_MAP.get(name, ""),
        })

    return results
