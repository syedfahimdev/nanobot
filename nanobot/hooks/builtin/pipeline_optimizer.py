"""Pipeline optimizer — intent routing, response control, history compression.

All pure code, zero LLM tokens:
1. Intent-based tool filtering — only send relevant tools
2. Response length control — voice=short, text=detailed
3. History compression — old turns to one-line summaries
4. Follow-up chaining — detect multi-step and auto-chain
5. Parallel prefetch detection — identify independent tools
6. Output format hints — list→bullets, compare→table
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Intent-Based Tool Filtering
# ═══════════════════════════════════════════════════════════════════════════════

# Map intent keywords → required tool names
_INTENT_TOOL_MAP: dict[str, list[str]] = {
    # Communication
    "email": ["search_tools", "message", "memory_search", "inbox"],
    "calendar": ["search_tools", "memory_search"],
    "send": ["message", "search_tools", "memory_search"],
    "telegram": ["message"],
    "discord": ["message"],
    "slack": ["message"],

    # Web
    "search": ["web_search", "web_fetch"],
    "browse": ["browser", "web_fetch"],
    "website": ["browser", "web_fetch"],
    "open": ["browser", "web_fetch"],
    "fetch": ["web_fetch"],
    "download": ["web_fetch", "exec", "background_exec"],

    # Files & System
    "file": ["read_file", "write_file", "edit_file", "list_dir", "exec"],
    "run": ["exec", "background_exec"],
    "command": ["exec", "background_exec"],
    "deploy": ["exec", "background_exec"],
    "build": ["exec", "background_exec"],
    "install": ["exec", "skills_marketplace"],

    # Memory & Knowledge
    "remember": ["memory_save", "memory_search"],
    "forget": ["memory_save"],
    "memory": ["memory_search", "memory_save"],
    "learn": ["skill_creator", "learn_from_url"],
    "know": ["memory_search"],

    # Goals & Tasks
    "goal": ["goals"],
    "task": ["goals"],
    "todo": ["goals"],
    "remind": ["goals", "cron"],

    # Schedule
    "schedule": ["cron"],
    "timer": ["cron"],
    "cron": ["cron"],
    "every": ["cron"],

    # Skills
    "skill": ["skills_marketplace", "skill_creator", "read_file"],
    "work order": ["search_tools", "read_file"],
    "report": ["search_tools", "read_file"],

    # Settings & Meta
    "setting": ["settings"],
    "feature": ["settings"],
    "toggle": ["settings"],
    "enable": ["settings"],
    "disable": ["settings"],
    "turn on": ["settings"],
    "turn off": ["settings"],

    # Credentials
    "password": ["credentials"],
    "credential": ["credentials"],
    "login": ["credentials", "browser"],

    # Media
    "image": ["media_memory"],
    "photo": ["media_memory"],
    "picture": ["media_memory"],
}

# Tools that should ALWAYS be available (core functionality)
_ALWAYS_TOOLS = frozenset({
    "read_file", "write_file", "edit_file", "list_dir",
    "exec", "web_search", "message", "memory_search",
    "settings",
})

# Tools that are cheap to include (small definitions)
_CHEAP_TOOLS = frozenset({
    "goals", "memory_save", "credentials",
})


def filter_tools_by_intent(message: str, all_tool_names: list[str]) -> list[str]:
    """Select relevant tools based on message intent. Pure keyword matching.

    Returns a filtered list of tool names. Falls back to all tools if no
    intent is detected (safety — never block functionality).
    """
    msg_lower = message.lower()
    matched_tools: set[str] = set(_ALWAYS_TOOLS)
    matched_tools.update(_CHEAP_TOOLS)

    intent_found = False
    for keyword, tools in _INTENT_TOOL_MAP.items():
        if keyword in msg_lower:
            matched_tools.update(tools)
            intent_found = True

    # MCP tools: include if message mentions the service name
    for tool in all_tool_names:
        if tool.startswith("mcp_"):
            # Extract service hint from tool name
            parts = tool.split("_")
            if len(parts) >= 3:
                service = parts[1].lower()  # e.g., "composio", "playwright"
                if service in msg_lower or any(kw in msg_lower for kw in _get_mcp_keywords(tool)):
                    matched_tools.add(tool)

    if not intent_found:
        # No clear intent — send core tools + search_tools for discovery
        # Don't send all 30+ tools — the LLM can search_tools() on demand
        matched_tools.add("search_tools")
        matched_tools.add("spawn")
        matched_tools.add("web_fetch")

    # Filter to only tools that exist in the registry
    filtered = [t for t in all_tool_names if t in matched_tools]

    # If we filtered too aggressively (< 3 tools), fall back to all
    if len(filtered) < 3:
        return all_tool_names

    savings = len(all_tool_names) - len(filtered)
    if savings > 3:
        logger.debug("Tool filter: {} → {} tools (saved {})", len(all_tool_names), len(filtered), savings)

    return filtered


def _get_mcp_keywords(tool_name: str) -> list[str]:
    """Extract search keywords from MCP tool names."""
    keywords = []
    # mcp_composio_GMAIL_FETCH_EMAILS → ["gmail", "email", "fetch"]
    parts = tool_name.lower().replace("mcp_", "").split("_")
    for p in parts:
        if len(p) > 2 and p not in ("composio", "playwright", "browseruse"):
            keywords.append(p)
    return keywords


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Response Length Control
# ═══════════════════════════════════════════════════════════════════════════════

def get_response_hint(message: str, channel: str) -> str:
    """Generate a response length/format hint based on channel and question type.

    Returns a hint string to inject into the prompt, or empty string.
    """
    msg_lower = message.lower().strip()
    hints = []

    # Channel-based length
    if channel in ("discord_voice", "web_voice"):
        hints.append(
            "Keep response SHORT (1-3 sentences) — this is voice mode, the user is listening. "
            "Do NOT chain multiple tool calls — use at most 1-2 simple ones. "
            "Prefer answering from memory over calling tools. Never loop on the same tool."
        )
    elif channel == "telegram":
        hints.append("Keep response concise — this is a chat message.")

    # Question type detection
    if _is_yes_no_question(msg_lower):
        hints.append("This is a yes/no question. Answer directly, then brief explanation.")
    elif _is_count_question(msg_lower):
        hints.append("User wants a number/count. Lead with the number.")
    elif _is_list_request(msg_lower):
        hints.append("User wants a list. Use bullet points, keep each item to one line.")
    elif _is_comparison(msg_lower):
        hints.append("User wants a comparison. Use a table or side-by-side format.")
    elif _is_how_to(msg_lower):
        hints.append("User wants step-by-step instructions. Number the steps.")
    elif _is_explain(msg_lower):
        hints.append("User wants an explanation. Start simple, add detail if asked.")

    if not hints:
        return ""
    return "[Response guidance: " + " ".join(hints) + "]"


def _is_yes_no_question(text: str) -> bool:
    return bool(re.match(r"^(is |are |do |does |did |has |have |can |will |should |was |were )", text))


def _is_count_question(text: str) -> bool:
    return bool(re.search(r"\b(how many|how much|count|total|number of)\b", text))


def _is_list_request(text: str) -> bool:
    return bool(re.search(r"\b(list|show me|what are|give me all|enumerate)\b", text))


def _is_comparison(text: str) -> bool:
    return bool(re.search(r"\b(compare|vs|versus|difference|better|worse|pros and cons)\b", text))


def _is_how_to(text: str) -> bool:
    return bool(re.search(r"\b(how (do|to|can)|steps to|guide|tutorial|walk me through)\b", text))


def _is_explain(text: str) -> bool:
    return bool(re.search(r"\b(explain|what is|what does|what's|tell me about|describe)\b", text))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. History Compression
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_FULL_TURNS = 6  # Keep last 6 turns in full
_COMPRESSED_PREFIX = "[earlier] "


def compress_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compress old history turns to save tokens.

    Keeps the last _MAX_FULL_TURNS messages in full.
    Earlier messages get compressed to one-line summaries.
    Tool messages are dropped from old turns.
    """
    if len(history) <= _MAX_FULL_TURNS * 2:
        return history  # Small enough — no compression needed

    # Split into old and recent
    cutoff = len(history) - (_MAX_FULL_TURNS * 2)
    old = history[:cutoff]
    recent = history[cutoff:]

    # Compress old messages
    compressed = []
    for msg in old:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip tool messages entirely in old history
        if role == "tool":
            continue

        # Skip assistant messages with tool_calls (just the call, result is gone)
        if role == "assistant" and msg.get("tool_calls"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        # Compress to first sentence or 80 chars
        short = content.strip()[:80]
        if ". " in short:
            short = short[:short.index(". ") + 1]
        short = short.replace("\n", " ").strip()

        if short:
            compressed.append({
                "role": role,
                "content": _COMPRESSED_PREFIX + short,
            })

    return compressed + recent


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Follow-Up Chaining Detection
# ═══════════════════════════════════════════════════════════════════════════════

_CHAIN_PATTERNS = [
    re.compile(r"\b(?:and then|then|after that|next|also|and also)\b", re.I),
    re.compile(r",\s*(?:then|and)\s+\w+", re.I),
]

_CHAIN_VERBS = re.compile(
    r"\b(check|send|create|search|find|open|list|get|set|update|delete|add|run|reply|forward)\b",
    re.I,
)


def detect_follow_up_chain(message: str) -> list[str]:
    """Detect if message contains chained requests.

    Returns list of sub-requests, or empty if single request.
    """
    # Must have a chain word AND multiple verbs
    has_chain = any(p.search(message) for p in _CHAIN_PATTERNS)
    verb_count = len(_CHAIN_VERBS.findall(message))

    if not has_chain or verb_count < 2:
        return []

    # Split on chain words
    parts = re.split(r"\b(?:and then|then|after that|next)\b", message, flags=re.I)
    parts = [p.strip().strip(",").strip() for p in parts if p.strip() and len(p.strip()) > 5]

    if len(parts) >= 2:
        return parts

    # Try splitting on "and also" / ", and"
    parts = re.split(r",\s*(?:and|also)\s+", message, flags=re.I)
    parts = [p.strip() for p in parts if p.strip() and _CHAIN_VERBS.search(p)]

    return parts if len(parts) >= 2 else []


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Parallel Prefetch Detection
# ═══════════════════════════════════════════════════════════════════════════════

# Keywords that map to independent data sources (can be fetched in parallel)
_PARALLEL_SOURCES = {
    "email": "email",
    "calendar": "calendar",
    "weather": "weather",
    "news": "news",
    "goals": "goals",
    "inbox": "inbox",
    "stocks": "stocks",
    "bitcoin": "crypto",
    "crypto": "crypto",
}


def detect_parallel_fetches(message: str) -> list[str]:
    """Detect if message asks for multiple independent data sources.

    Returns list of source names that can be fetched in parallel.
    """
    msg_lower = message.lower()
    sources = []

    for keyword, source in _PARALLEL_SOURCES.items():
        if keyword in msg_lower and source not in sources:
            sources.append(source)

    # Only suggest parallel if 2+ independent sources
    return sources if len(sources) >= 2 else []


def build_parallel_hint(sources: list[str]) -> str:
    """Build a hint for the LLM to fetch these sources in parallel."""
    if len(sources) < 2:
        return ""
    return f"[Optimization: the user asked about {', '.join(sources)}. These are independent — call the tools in PARALLEL (multiple tool_calls in one response) for faster results.]"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Output Format Hints
# ═══════════════════════════════════════════════════════════════════════════════

def get_format_hint(message: str) -> str:
    """Detect desired output format and return a hint.

    Returns empty string if no specific format detected.
    """
    msg_lower = message.lower()

    if re.search(r"\b(table|tabulate|spreadsheet|grid)\b", msg_lower):
        return "[Format: Use a markdown table.]"

    if re.search(r"\b(bullet|bullets|list|itemize)\b", msg_lower):
        return "[Format: Use bullet points.]"

    if re.search(r"\b(step|steps|guide|how to|tutorial|instructions)\b", msg_lower):
        return "[Format: Use numbered steps.]"

    if re.search(r"\b(brief|short|quick|tldr|summary|one.?line)\b", msg_lower):
        return "[Format: Keep it very brief — 1-2 sentences max.]"

    if re.search(r"\b(detail|detailed|thorough|comprehensive|full|everything)\b", msg_lower):
        return "[Format: Provide a detailed, thorough response.]"

    if re.search(r"\b(json|code|snippet)\b", msg_lower):
        return "[Format: Use a code block.]"

    if re.search(r"\b(compare|vs|versus|difference)\b", msg_lower):
        return "[Format: Use a comparison table with columns for each option.]"

    return ""
