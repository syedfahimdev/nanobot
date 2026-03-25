"""Capability tips — proactively surface Mawa's features when relevant.

Detects patterns in user messages and tool results, then surfaces contextual
tips as inline annotations under the response. Each tip shown once per user.

All detection is pure code — no LLM tokens.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ── Tip definitions ──────────────────────────────────────────────────────────
# Each tip has:
#   id: unique key (shown once)
#   trigger: regex on user message OR tool result
#   trigger_on: "message" | "tool_result" | "response"
#   condition: optional callable(workspace) → bool (check if feature is off)
#   text: the tip text
#   feature_key: optional settings key to suggest toggling
#   action: "enable" | "disable" | None

_TIPS: list[dict[str, Any]] = [
    # ── Research delegation ──
    {
        "id": "auto_delegate_research",
        "trigger": re.compile(r"\b(research|look up|find me|look into|investigate)\b", re.I),
        "trigger_on": "message",
        "text": "I can run research tasks in the background while you keep chatting. Just ask naturally — I'll auto-detect heavy tasks and delegate them.",
        "feature_key": None,
    },
    # ── Morning briefing ──
    {
        "id": "morning_briefing",
        "trigger": re.compile(r"\b(morning|good morning|wake up|start.*(my |the )?day)\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "morningPrep"),
        "text": "I can generate a daily morning briefing with your goals, calendar, bills, and relationship reminders. Want me to enable it?",
        "feature_key": "morningPrep",
        "action": "enable",
    },
    # ── Weather shortcut ──
    {
        "id": "weather_cron",
        "trigger": re.compile(r"\b(weather|forecast|temperature|rain)\b", re.I),
        "trigger_on": "message",
        "text": "I can send you a daily weather briefing automatically. Say 'schedule daily weather at 7am' to set it up.",
        "feature_key": None,
    },
    # ── Email checking ──
    {
        "id": "email_priority",
        "trigger": re.compile(r"\b(email|inbox|mail|gmail)\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "priorityInbox"),
        "text": "I can score your messages by urgency so important ones stand out. Want me to enable Priority Inbox?",
        "feature_key": "priorityInbox",
        "action": "enable",
    },
    # ── Repeated questions ──
    {
        "id": "response_caching",
        "trigger_on": "cache_miss_repeat",  # Special: triggered by smart_responses detecting repeat
        "trigger": re.compile(r".*"),  # Unused for this trigger type
        "condition": lambda ws: not _is_enabled(ws, "responseCaching"),
        "text": "I noticed you ask similar questions. I can cache responses to answer instantly next time. Want me to enable Response Caching?",
        "feature_key": "responseCaching",
        "action": "enable",
    },
    # ── Frustration ──
    {
        "id": "frustration_help",
        "trigger": re.compile(r"\b(doesn't work|broken|wrong|not working|bug|error|fail|stuck)\b", re.I),
        "trigger_on": "message",
        "text": "I can detect when you're frustrated and adjust my tone to be more helpful. This is controlled by the Frustration Detection setting.",
        "feature_key": "frustrationDetection",
    },
    # ── Voice mode ──
    {
        "id": "voice_mode",
        "trigger": re.compile(r"\b(call|phone|speak|voice|talk to)\b", re.I),
        "trigger_on": "message",
        "text": "I support voice mode — tap the microphone to talk to me. I'll respond with speech too.",
        "feature_key": None,
    },
    # ── Delegation/background tasks ──
    {
        "id": "delegation_queue",
        "trigger": re.compile(r"\b(follow up|check (on|back)|remind me to check|later|tomorrow)\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "delegationQueue"),
        "text": "I can track tasks you want me to check on periodically — like following up on something tomorrow. Want me to enable Delegation Queue?",
        "feature_key": "delegationQueue",
        "action": "enable",
    },
    # ── Decision memory ──
    {
        "id": "decision_memory",
        "trigger": re.compile(r"\b(decided|decision|chose|picked|went with|why did (I|we))\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "decisionMemory"),
        "text": "I can remember WHY you made decisions so you can recall your reasoning later. Want me to enable Decision Memory?",
        "feature_key": "decisionMemory",
        "action": "enable",
    },
    # ── Subagent agents panel ──
    {
        "id": "agents_panel",
        "trigger_on": "subagent_spawned",
        "trigger": re.compile(r".*"),
        "text": "A background agent is working on your task. Press G or tap the Agents button to see live progress, cancel, or send updates.",
        "feature_key": None,
    },
    # ── Daily digest ──
    {
        "id": "daily_digest",
        "trigger": re.compile(r"\b(what did (I|we) do|summary|recap|today.*(done|accomplish|finish))\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "dailyDigest"),
        "text": "I can send you an automatic end-of-day summary with usage, goals completed, and learnings. Want me to enable Daily Digest?",
        "feature_key": "dailyDigest",
        "action": "enable",
    },
    # ── Relationship tracker ──
    {
        "id": "relationship_tracker",
        "trigger": re.compile(r"\b(haven't (talked|spoken|called|texted)|been a while|miss|catch up|reconnect)\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "relationshipTracker"),
        "text": "I can track when you last contacted people and remind you to reconnect. Want me to enable Relationship Tracker?",
        "feature_key": "relationshipTracker",
        "action": "enable",
    },
    # ── Financial pulse ──
    {
        "id": "financial_alert",
        "trigger": re.compile(r"\b(spending|budget|cost|expensive|bill|payment|subscription)\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "financialPulse"),
        "text": "I can track your AI spending and alert you about unusual charges or upcoming bills. Want me to enable Financial Pulse?",
        "feature_key": "financialPulse",
        "action": "enable",
    },
    # ── Quiet hours ──
    {
        "id": "quiet_hours",
        "trigger": re.compile(r"\b(late|night|sleep|quiet|disturb|notification|stop.*(notif|alert|buzz))\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "quietHoursEnabled"),
        "text": "I can mute notifications during sleeping hours. Want me to enable Quiet Hours?",
        "feature_key": "quietHoursEnabled",
        "action": "enable",
    },
    # ── Image generation ──
    {
        "id": "image_gen",
        "trigger": re.compile(r"\b(generate|create|make|draw).*(image|picture|photo|illustration|art|logo)\b", re.I),
        "trigger_on": "message",
        "condition": lambda ws: not _is_enabled(ws, "imageGenEnabled"),
        "text": "I can generate images! This feature is currently disabled. Want me to enable Image Generation?",
        "feature_key": "imageGenEnabled",
        "action": "enable",
    },
]


def _is_enabled(workspace: Path, key: str) -> bool:
    from nanobot.hooks.builtin.feature_registry import get_setting
    return get_setting(workspace, key, True)


# ── Tip state (shown tips) ──────────────────────────────────────────────────

def _tips_file(workspace: Path) -> Path:
    return workspace / "shown_tips.json"


def _load_shown(workspace: Path) -> set[str]:
    f = _tips_file(workspace)
    try:
        return set(json.loads(f.read_text())) if f.exists() else set()
    except Exception:
        return set()


def _mark_shown(workspace: Path, tip_id: str) -> None:
    shown = _load_shown(workspace)
    shown.add(tip_id)
    try:
        _tips_file(workspace).write_text(json.dumps(list(shown)))
    except Exception:
        pass


# ── Main detection function ──────────────────────────────────────────────────

def detect_tips(
    workspace: Path,
    message: str,
    trigger_type: str = "message",
) -> list[dict[str, Any]]:
    """Detect relevant capability tips for the current message.

    Returns list of tip dicts: {id, text, feature_key, action}
    Max 1 tip per message to avoid spamming.
    """
    shown = _load_shown(workspace)
    tips = []

    for tip in _TIPS:
        if tip["id"] in shown:
            continue
        if tip["trigger_on"] != trigger_type:
            continue
        if not tip["trigger"].search(message):
            continue
        # Check condition (e.g., only show if feature is OFF)
        if "condition" in tip and callable(tip["condition"]):
            if not tip["condition"](workspace):
                continue

        tips.append({
            "id": tip["id"],
            "text": tip["text"],
            "feature_key": tip.get("feature_key"),
            "action": tip.get("action"),
        })

    if not tips:
        return []

    # Return max 1 tip per message
    best = tips[0]
    _mark_shown(workspace, best["id"])
    logger.debug("Capability tip triggered: {} for '{}'", best["id"], message[:40])
    return [best]


def reset_tips(workspace: Path) -> None:
    """Reset all shown tips (for testing or after major updates)."""
    f = _tips_file(workspace)
    if f.exists():
        f.unlink()
