"""Dynamic capabilities manifest — Mawa knows what she can do.

Auto-generates a capabilities document from:
- Registered tools (names + descriptions)
- Settings toggles (from intelligence.json)
- Installed skills (from workspace/skills/)
- Available channels
- Memory layers

Injected into context when user asks about features/capabilities.
Security: strips vault paths, credential values, API keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


# Patterns that should NEVER appear in capability descriptions
_SECURITY_STRIP = [
    re.compile(r"\$\{vault:[^}]+\}", re.I),
    re.compile(r"\{cred:[^}]+\}", re.I),
    re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S+", re.I),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub tokens
    re.compile(r"xox[bps]-[a-zA-Z0-9-]+"),  # Slack tokens
]


def _sanitize(text: str) -> str:
    """Remove any credential/secret patterns from text."""
    for pattern in _SECURITY_STRIP:
        text = pattern.sub("[REDACTED]", text)
    return text


def generate_capabilities(
    workspace: Path,
    tool_names: list[str] | None = None,
    tool_descriptions: dict[str, str] | None = None,
) -> str:
    """Generate the full capabilities manifest.

    Returns a markdown string describing everything Mawa can do,
    safe to inject into the system prompt.
    """
    parts = []

    # ── Core Tools ──
    if tool_names:
        tool_section = ["## My Tools\n"]
        # Group by category
        categories: dict[str, list[str]] = {
            "Communication": [], "Memory & Knowledge": [],
            "Productivity": [], "Browser & Web": [],
            "Files & System": [], "AI & Automation": [],
        }
        tool_cat_map = {
            "message": "Communication", "memory_save": "Memory & Knowledge",
            "memory_search": "Memory & Knowledge", "goals": "Productivity",
            "inbox": "Productivity", "cron": "Productivity",
            "web_search": "Browser & Web", "web_fetch": "Browser & Web",
            "browser": "Browser & Web", "exec": "Files & System",
            "background_exec": "Files & System", "read_file": "Files & System",
            "write_file": "Files & System", "edit_file": "Files & System",
            "list_dir": "Files & System", "credentials": "Memory & Knowledge",
            "skill_creator": "AI & Automation", "learn_from_url": "AI & Automation",
            "skills_marketplace": "AI & Automation", "media_memory": "Memory & Knowledge",
            "spawn": "AI & Automation", "restart": "Files & System",
        }
        for t in sorted(tool_names):
            cat = tool_cat_map.get(t, "AI & Automation" if t.startswith("mcp_") else "Files & System")
            desc = tool_descriptions.get(t, "") if tool_descriptions else ""
            categories.setdefault(cat, []).append(f"- **{t}**: {_sanitize(desc[:100])}" if desc else f"- **{t}**")

        for cat, tools in categories.items():
            if tools:
                tool_section.append(f"### {cat}")
                tool_section.extend(tools)
                tool_section.append("")

        parts.append("\n".join(tool_section))

    # ── Settings & Toggles ──
    settings_path = workspace / "intelligence.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            toggles = []
            toggle_labels = {
                "smartErrorRecovery": "Smart Error Recovery — I classify errors and suggest recovery strategies",
                "intentTracking": "Intent Tracking — I remember what we're talking about across messages",
                "dynamicContextBudget": "Dynamic Context Budget — I adjust how much tool output I keep based on context size",
                "responseQualityGate": "Response Quality Gate — I detect when I'm deflecting and retry harder",
                "mcpAutoReconnect": "MCP Auto-Reconnect — I automatically reconnect to broken MCP servers",
            }
            for key, label in toggle_labels.items():
                status = "ON" if settings.get(key, True) else "OFF"
                toggles.append(f"- [{status}] {label}")
            parts.append("## Intelligence Features\n" + "\n".join(toggles))
        except Exception:
            pass

    # ── Skills ──
    skills_dir = workspace / "skills"
    if skills_dir.exists():
        skill_list = []
        for sd in sorted(skills_dir.iterdir()):
            if sd.is_dir() and (sd / "SKILL.md").exists():
                name = sd.name.replace("-", " ").replace("_", " ").title()
                skill_list.append(f"- **{name}** ({sd.name})")
        if skill_list:
            parts.append("## Installed Skills\n" + "\n".join(skill_list))

    # ── Memory System ──
    mem_dir = workspace / "memory"
    if mem_dir.exists():
        layers = []
        for name, desc in [
            ("SHORT_TERM.md", "Today's context"),
            ("LONG_TERM.md", "Permanent facts about you"),
            ("LEARNINGS.md", "Rules I learned from your feedback"),
            ("GOALS.md", "Your tracked goals"),
            ("OBSERVATIONS.md", "Behavioral patterns I detected"),
            ("EPISODES.md", "Key moments and decisions"),
        ]:
            f = mem_dir / name
            if f.exists() and f.stat().st_size > 10:
                layers.append(f"- {desc} ({name})")
        if layers:
            parts.append("## My Memory\n" + "\n".join(layers))

    # ── Security Policy ──
    parts.append("""## Security Policy
- I NEVER share credentials, API keys, vault contents, or passwords in chat
- Credentials are stored encrypted in the vault — I use {cred:name} references
- I mask passwords before they reach the LLM
- I cannot read .env files or print environment variables""")

    # ── What I Can Help With ──
    parts.append("""## What I Can Help With
- Check and summarize your email, calendar, and messages
- Search the web, fetch pages, and browse websites live
- Create and track goals with due dates
- Run commands in the background and notify you when done
- Save and recall information from memory
- Learn from your feedback — tell me "next time do X" and I'll remember
- Manage credentials securely (store, use, never expose)
- Generate files (Excel, reports) and send them via Telegram/Discord
- Automate workflows — I detect patterns and offer to automate them
- Install new skills from the marketplace""")

    return "\n\n".join(parts)


def should_inject_capabilities(user_message: str) -> bool:
    """Check if the user is asking about Mawa's capabilities."""
    patterns = [
        r"\bwhat can you\b",
        r"\bwhat do you\b.*\b(do|know|have)\b",
        r"\bwhat are your\b.*\b(features?|capabilities?|tools?|skills?)\b",
        r"\bwhat\b.*\bfeatures?\b.*\b(do|have|are)\b",
        r"\bcan you\b.*\b(do|help|handle)\b",
        r"\bhow do(?:es)?\b.*\bwork\b",
        r"\bturn (?:on|off)\b.*\bfeature\b",
        r"\benable\b.*\b(feature|setting|toggle)\b",
        r"\bdisable\b.*\b(feature|setting|toggle)\b",
        r"\bhelp me\b.*\bunderstand\b",
        r"\bwhat('s| is)\b.*\b(available|possible)\b",
        r"\btell me about\b.*\b(yourself|your|mawa)\b",
        r"\bhow does?\b.*\b(memory|learning|notification|goal|skill|feature)\b",
        r"\bwhat\b.*\b(ability|abilities|power)\b",
        r"\bshow me\b.*\b(features?|tools?|capabilities?)\b",
    ]
    msg_lower = user_message.lower()
    return any(re.search(p, msg_lower) for p in patterns)
