"""Configurable intent map for ToolsDNS preflight.

Loads intent→tool mappings from workspace/INTENT_MAP.md or falls back
to built-in defaults. Users can customize which queries trigger which
tool searches without editing code.

Format of INTENT_MAP.md:
  ## email
  - GMAIL_FETCH_EMAILS
  - GMAIL_SEND_EMAIL
  queries: check email, read inbox, send email, new messages

  ## calendar
  - GOOGLECALENDAR_FIND_EVENT
  queries: schedule, calendar, meeting, appointment
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

# Built-in defaults (used when no INTENT_MAP.md exists)
_DEFAULT_INTENT_MAP: dict[str, dict[str, Any]] = {
    "email": {
        "tools": ["GMAIL_FETCH_EMAILS", "GMAIL_SEND_EMAIL", "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID"],
        "queries": ["check email", "read inbox", "send email", "new messages", "unread mail", "email from"],
        "patterns": [r"\bemail\b", r"\binbox\b", r"\bmail\b", r"\bgmail\b"],
    },
    "calendar": {
        "tools": ["GOOGLECALENDAR_FIND_EVENT", "GOOGLECALENDAR_EVENTS_LIST", "GOOGLECALENDAR_CREATE_EVENT"],
        "queries": ["check calendar", "today's schedule", "upcoming meetings", "create event", "book meeting"],
        "patterns": [r"\bcalendar\b", r"\bschedule\b", r"\bmeeting\b", r"\bappointment\b"],
    },
    "weather": {
        "tools": ["WEATHERMAP_WEATHER"],
        "queries": ["check weather", "weather forecast", "temperature", "is it raining"],
        "patterns": [r"\bweather\b", r"\bforecast\b", r"\btemperature\b"],
    },
    "slack": {
        "tools": ["SLACK_SEND_MESSAGE", "SLACK_LIST_CHANNELS"],
        "queries": ["send slack message", "slack channel", "post to slack"],
        "patterns": [r"\bslack\b"],
    },
    "github": {
        "tools": ["GITHUB_CREATE_ISSUE", "GITHUB_LIST_REPOS"],
        "queries": ["create github issue", "list repos", "pull request"],
        "patterns": [r"\bgithub\b", r"\brepo\b", r"\bPR\b"],
    },
    "salesforce": {
        "tools": ["SALESFORCE_QUERY", "SALESFORCE_CREATE_RECORD"],
        "queries": ["salesforce query", "check salesforce", "CRM lookup"],
        "patterns": [r"\bsalesforce\b", r"\bCRM\b", r"\bopportunity\b"],
    },
    "news": {
        "tools": ["HACKERNEWS_SEARCH_POSTS", "REDDIT_SEARCH_ACROSS_SUBREDDITS"],
        "queries": ["latest news", "hacker news", "search reddit", "trending"],
        "patterns": [r"\bnews\b", r"\bhacker ?news\b", r"\breddit\b"],
    },
    "browser": {
        "tools": ["BROWSER_NAVIGATE", "BROWSER_SCREENSHOT"],
        "queries": ["open website", "browse URL", "take screenshot"],
        "patterns": [r"\bbrowse\b", r"\bnavigate\b", r"\bwebsite\b", r"\bopen\s+http"],
    },
}


def load_intent_map(workspace: Path) -> dict[str, dict[str, Any]]:
    """Load intent map from workspace or use defaults."""
    intent_file = workspace / "INTENT_MAP.md"
    if not intent_file.exists():
        return _DEFAULT_INTENT_MAP.copy()

    try:
        content = intent_file.read_text(encoding="utf-8")
        custom_map = _parse_intent_map(content)
        if custom_map:
            # Merge: custom overrides defaults
            merged = _DEFAULT_INTENT_MAP.copy()
            merged.update(custom_map)
            logger.info("Loaded custom intent map: {} intents ({} custom)", len(merged), len(custom_map))
            return merged
    except Exception as e:
        logger.warning("Failed to load INTENT_MAP.md: {}", e)

    return _DEFAULT_INTENT_MAP.copy()


def _parse_intent_map(content: str) -> dict[str, dict[str, Any]]:
    """Parse INTENT_MAP.md format."""
    result = {}
    current_section = ""
    current_tools: list[str] = []
    current_queries: list[str] = []

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            # Save previous section
            if current_section and (current_tools or current_queries):
                result[current_section] = {
                    "tools": current_tools,
                    "queries": current_queries,
                    "patterns": [rf"\b{re.escape(current_section)}\b"],
                }
            current_section = line[3:].strip().lower()
            current_tools = []
            current_queries = []
        elif line.startswith("- ") and current_section:
            tool = line[2:].strip()
            if tool.isupper() or "_" in tool:
                current_tools.append(tool)
        elif line.startswith("queries:") and current_section:
            qs = line[8:].strip()
            current_queries = [q.strip() for q in qs.split(",") if q.strip()]

    # Save last section
    if current_section and (current_tools or current_queries):
        result[current_section] = {
            "tools": current_tools,
            "queries": current_queries,
            "patterns": [rf"\b{re.escape(current_section)}\b"],
        }

    return result


def match_intent(text: str, intent_map: dict[str, dict[str, Any]]) -> str | None:
    """Match user text against intent map. Returns intent name or None."""
    text_lower = text.lower()
    for intent_name, config in intent_map.items():
        # Check patterns
        for pattern in config.get("patterns", []):
            if re.search(pattern, text_lower):
                return intent_name
        # Check query phrases
        for query in config.get("queries", []):
            if query.lower() in text_lower:
                return intent_name
    return None
