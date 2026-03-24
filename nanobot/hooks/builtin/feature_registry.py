"""Feature registry — single source of truth for all configurable features.

Exposes a manifest of every toggleable/configurable feature with:
- Current value
- Type (boolean, number, string, select)
- Description
- Category
- API endpoint to change it

The frontend fetches this manifest and renders settings dynamically.
No hardcoded feature lists in the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def get_feature_manifest(workspace: Path) -> list[dict[str, Any]]:
    """Build the complete feature manifest from all config sources."""
    features = []

    # ── Intelligence toggles (from intelligence.json) ──
    intel_path = workspace / "intelligence.json"
    intel = {}
    if intel_path.exists():
        try:
            intel = json.loads(intel_path.read_text())
        except Exception:
            pass

    intel_features = [
        ("smartErrorRecovery", "Smart Error Recovery",
         "Classifies tool errors (timeout, auth, 404) with recovery hints",
         "intelligence"),
        ("intentTracking", "Intent Tracking",
         "Tracks active topic across turns — resolves 'this', 'that', 'send it'",
         "intelligence"),
        ("dynamicContextBudget", "Dynamic Context Budget",
         "Sizes tool results based on remaining context window",
         "intelligence"),
        ("responseQualityGate", "Response Quality Gate",
         "Catches LLM deflections and retries with tools",
         "intelligence"),
        ("mcpAutoReconnect", "MCP Auto-Reconnect",
         "Reconnects to MCP servers after 3 consecutive failures",
         "intelligence"),
    ]
    for key, label, desc, cat in intel_features:
        features.append({
            "key": key, "label": label, "desc": desc, "category": cat,
            "type": "boolean", "value": intel.get(key, True),
            "endpoint": "/api/intelligence", "method": "POST",
        })

    # ── Quiet hours ──
    qh_path = workspace / "quiet_hours.json"
    qh = {"enabled": True, "start": 22, "end": 7}
    if qh_path.exists():
        try:
            qh = {**qh, **json.loads(qh_path.read_text())}
        except Exception:
            pass

    features.append({
        "key": "quietHoursEnabled", "label": "Quiet Hours",
        "desc": "Defer non-urgent notifications during quiet hours. High priority always goes through.",
        "category": "notifications", "type": "boolean", "value": qh.get("enabled", True),
        "endpoint": "/api/quiet-hours", "method": "POST",
    })
    features.append({
        "key": "quietHoursStart", "label": "Quiet Hours Start",
        "desc": "Hour to start quiet mode (24h format)",
        "category": "notifications", "type": "number", "value": qh.get("start", 22),
        "min": 0, "max": 23,
        "endpoint": "/api/quiet-hours", "method": "POST",
    })
    features.append({
        "key": "quietHoursEnd", "label": "Quiet Hours End",
        "desc": "Hour to end quiet mode (24h format)",
        "category": "notifications", "type": "number", "value": qh.get("end", 7),
        "min": 0, "max": 23,
        "endpoint": "/api/quiet-hours", "method": "POST",
    })

    # ── Budget ──
    from nanobot.hooks.builtin.cost_budget import load_budget
    budget = load_budget(workspace)
    features.append({
        "key": "daily_limit", "label": "Daily Budget ($)",
        "desc": "Maximum daily spending on LLM calls. Set to 0 to disable.",
        "category": "budget", "type": "number", "value": budget.get("daily_limit", 5.0),
        "min": 0, "max": 100, "step": 0.5,
        "endpoint": "/api/budget", "method": "POST",
    })
    features.append({
        "key": "weekly_limit", "label": "Weekly Budget ($)",
        "desc": "Maximum weekly spending on LLM calls.",
        "category": "budget", "type": "number", "value": budget.get("weekly_limit", 25.0),
        "min": 0, "max": 500, "step": 1,
        "endpoint": "/api/budget", "method": "POST",
    })
    features.append({
        "key": "enforce", "label": "Hard Budget Enforcement",
        "desc": "Block LLM calls when budget is exceeded (not just warn).",
        "category": "budget", "type": "boolean", "value": budget.get("enforce", False),
        "endpoint": "/api/budget", "method": "POST",
    })
    features.append({
        "key": "auto_switch_model", "label": "Budget Fallback Model",
        "desc": "Auto-switch to this cheaper model when budget is >80%. Leave empty to disable.",
        "category": "budget", "type": "string", "value": budget.get("auto_switch_model", ""),
        "placeholder": "e.g., claude-haiku-3-5",
        "endpoint": "/api/budget", "method": "POST",
    })

    # ── Maintenance ──
    features.extend([
        {
            "key": "historyAutoArchive", "label": "History Auto-Archive",
            "desc": "Split HISTORY.md into monthly chunks when it exceeds 100KB. Runs on heartbeat.",
            "category": "maintenance", "type": "boolean", "value": True,
            "endpoint": "/api/maintenance/settings", "method": "POST",
        },
        {
            "key": "sessionAutoCleanup", "label": "Session Auto-Cleanup",
            "desc": "Delete inactive sessions older than 7 days (keeps 5 most recent).",
            "category": "maintenance", "type": "boolean", "value": True,
            "endpoint": "/api/maintenance/settings", "method": "POST",
        },
        {
            "key": "contactAutoExtract", "label": "Contact Auto-Extract",
            "desc": "Automatically extract contacts (names, emails) from memory on heartbeat.",
            "category": "maintenance", "type": "boolean", "value": True,
            "endpoint": "/api/maintenance/settings", "method": "POST",
        },
    ])

    # ── Agent behavior ──
    features.extend([
        {
            "key": "languageDetection", "label": "Multi-Language Detection",
            "desc": "Detect Bangla, Hindi, Urdu, Spanish, French, Arabic and respond in user's language.",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
        {
            "key": "destructiveConfirmation", "label": "Destructive Action Confirmation",
            "desc": "Ask for confirmation before delete all, clear, reset, rm -rf.",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
        {
            "key": "responseCaching", "label": "Response Caching",
            "desc": "Cache identical questions for 5 minutes — skip LLM on repeat.",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
        {
            "key": "frustrationDetection", "label": "Frustration Detection",
            "desc": "Detect user frustration (caps, !!!, angry words) and respond empathetically.",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
        {
            "key": "messageDedup", "label": "Message Dedup",
            "desc": "Drop duplicate messages sent within 30 seconds (network glitch protection).",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
        {
            "key": "greetingInterceptor", "label": "Smart Greetings",
            "desc": "Answer 'hello' with a time-aware greeting + goals context (zero LLM tokens).",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
        {
            "key": "mathInterceptor", "label": "Math Interceptor",
            "desc": "Answer math questions (15% of $347) with pure code — zero LLM tokens.",
            "category": "behavior", "type": "boolean", "value": True,
            "endpoint": "/api/behavior", "method": "POST",
        },
    ])

    # ── Jarvis proactive features ──
    jarvis_path = workspace / "jarvis_settings.json"
    jarvis = {}
    if jarvis_path.exists():
        try:
            jarvis = json.loads(jarvis_path.read_text())
        except Exception:
            pass

    jarvis_features = [
        ("morningPrep", "Morning Prep", "Auto-generate briefing from goals, bills, relationships, habits before you wake up.", True),
        ("crossSignalCorrelation", "Cross-Signal Correlation", "Connect dots: travel+weather, bills+dates, goals+deadlines → proactive alerts.", True),
        ("meetingIntelligence", "Meeting Intelligence", "Prep before meetings — pull relevant emails, notes about attendees.", True),
        ("relationshipTracker", "Relationship Tracker", "Track contact frequency. Alert when you haven't talked to someone in 2+ weeks.", True),
        ("financialPulse", "Financial Pulse", "Track AI spending trends, bill countdown, unusual charges.", True),
        ("projectTracker", "Project Tracker", "Multi-day tasks with progress tracking across sessions.", True),
        ("dailyDigest", "Daily Digest", "End-of-day summary: usage, goals completed, new learnings.", True),
        ("priorityInbox", "Priority Inbox", "Score messages by urgency (keywords, caps, sender).", True),
        ("delegationQueue", "Delegation Queue", "Async tasks Mawa checks on periodically — 'handle this over the week'.", True),
        ("routineDetection", "Routine Detection", "Detect daily patterns and offer to automate them.", True),
        ("decisionMemory", "Decision Memory", "Remember WHY you made decisions for future reference.", True),
        ("peoplePrepBeforeCalls", "People Prep", "Before a call, surface your last interactions with the person.", True),
    ]
    for key, label, desc, default in jarvis_features:
        features.append({
            "key": key, "label": label, "desc": desc,
            "category": "jarvis", "type": "boolean",
            "value": jarvis.get(key, default),
            "endpoint": "/api/jarvis/settings", "method": "POST",
        })

    # Jarvis configurable numbers
    features.append({
        "key": "relationshipReminderDays", "label": "Relationship Reminder (days)",
        "desc": "Alert when you haven't contacted someone for this many days.",
        "category": "jarvis", "type": "number", "value": jarvis.get("relationshipReminderDays", 14),
        "min": 3, "max": 90,
        "endpoint": "/api/jarvis/settings", "method": "POST",
    })
    features.append({
        "key": "delegationCheckHours", "label": "Delegation Check Interval (hours)",
        "desc": "How often to check on delegated tasks.",
        "category": "jarvis", "type": "number", "value": jarvis.get("delegationCheckHours", 24),
        "min": 1, "max": 168,
        "endpoint": "/api/jarvis/settings", "method": "POST",
    })
    features.append({
        "key": "correlationLookaheadDays", "label": "Correlation Lookahead (days)",
        "desc": "How far ahead to scan for correlated events (bills, travel, deadlines).",
        "category": "jarvis", "type": "number", "value": jarvis.get("correlationLookaheadDays", 3),
        "min": 1, "max": 14,
        "endpoint": "/api/jarvis/settings", "method": "POST",
    })

    return features


def get_feature_categories() -> list[dict[str, str]]:
    """Return the ordered list of feature categories."""
    return [
        {"id": "intelligence", "label": "Intelligence", "icon": "brain"},
        {"id": "behavior", "label": "Agent Behavior", "icon": "sparkles"},
        {"id": "jarvis", "label": "Jarvis Intelligence", "icon": "zap"},
        {"id": "notifications", "label": "Notifications", "icon": "bell"},
        {"id": "budget", "label": "Cost & Budget", "icon": "dollar-sign"},
        {"id": "maintenance", "label": "Maintenance", "icon": "wrench"},
    ]
