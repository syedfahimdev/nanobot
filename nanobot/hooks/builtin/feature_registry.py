"""Feature registry — single unified settings file for all configurable features.

All settings stored in ONE file: workspace/mawa_settings.json
No more scattered intelligence.json, jarvis_settings.json, behavior_settings.json, etc.

The frontend fetches the manifest via GET /api/features
and saves any value via POST /api/features {key, value}
— no category routing needed, everything goes to the same file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


# ── Unified settings file ───────────────────────────────────────────────────

_SETTINGS_FILE = "mawa_settings.json"
_cache: dict[str, Any] = {}
_cache_mtime: float = 0.0


def _settings_path(workspace: Path) -> Path:
    return workspace / _SETTINGS_FILE


def load_settings(workspace: Path) -> dict[str, Any]:
    """Load all settings from the unified file. Cached by mtime."""
    global _cache, _cache_mtime
    path = _settings_path(workspace)
    try:
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime != _cache_mtime:
                _cache = json.loads(path.read_text())
                _cache_mtime = mtime
            return _cache
    except Exception:
        pass
    return {}


def save_setting(workspace: Path, key: str, value: Any) -> None:
    """Save a single setting to the unified file."""
    global _cache, _cache_mtime
    path = _settings_path(workspace)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    data[key] = value
    path.write_text(json.dumps(data, indent=2))
    _cache = data
    _cache_mtime = path.stat().st_mtime


def save_settings_bulk(workspace: Path, updates: dict[str, Any]) -> None:
    """Save multiple settings at once."""
    global _cache, _cache_mtime
    path = _settings_path(workspace)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    data.update(updates)
    path.write_text(json.dumps(data, indent=2))
    _cache = data
    _cache_mtime = path.stat().st_mtime


def get_setting(workspace: Path, key: str, default: Any = None) -> Any:
    """Get a single setting value. Falls back to _FEATURE_DEFS default if not in file."""
    val = load_settings(workspace).get(key)
    if val is not None:
        return val
    # Check feature definitions for default
    for fdef in _FEATURE_DEFS:
        if fdef["key"] == key:
            return fdef.get("default", default)
    return default


# ── Migrate old scattered files into unified ────────────────────────────────

def migrate_old_settings(workspace: Path) -> int:
    """Merge old scattered settings files into the unified file. Run once."""
    migrated = 0
    data = {}

    old_files = {
        "intelligence.json": None,  # Direct key merge
        "jarvis_settings.json": None,
        "behavior_settings.json": None,
        "maintenance_settings.json": None,
        "voice_prefs.json": None,
    }

    # Quiet hours has different key names
    qh_path = workspace / "quiet_hours.json"
    if qh_path.exists():
        try:
            qh = json.loads(qh_path.read_text())
            data["quietHoursEnabled"] = qh.get("enabled", True)
            data["quietHoursStart"] = qh.get("start", 22)
            data["quietHoursEnd"] = qh.get("end", 7)
            migrated += 1
        except Exception:
            pass

    # Budget has its own path
    budget_path = workspace / "usage" / "budget.json"
    if budget_path.exists():
        try:
            budget = json.loads(budget_path.read_text())
            for k, v in budget.items():
                data[f"budget_{k}"] = v
            migrated += 1
        except Exception:
            pass

    # Direct merge files
    for fname in old_files:
        path = workspace / fname
        if path.exists():
            try:
                old = json.loads(path.read_text())
                data.update(old)
                migrated += 1
            except Exception:
                pass

    if data:
        # Load existing unified settings first
        path = _settings_path(workspace)
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                pass
        # Old values don't overwrite existing unified values
        merged = {**data, **existing}
        path.write_text(json.dumps(merged, indent=2))
        logger.info("Migrated {} old settings files into {}", migrated, _SETTINGS_FILE)

    return migrated


# ── Feature manifest ────────────────────────────────────────────────────────

# All features defined in ONE place with their defaults
_FEATURE_DEFS: list[dict[str, Any]] = [
    # Intelligence
    {"key": "smartErrorRecovery", "label": "Smart Error Recovery", "desc": "Classifies tool errors (timeout, auth, 404) with recovery hints.", "category": "intelligence", "type": "boolean", "default": True},
    {"key": "intentTracking", "label": "Intent Tracking", "desc": "Tracks active topic across turns — resolves 'this', 'that', 'send it'.", "category": "intelligence", "type": "boolean", "default": True},
    {"key": "dynamicContextBudget", "label": "Dynamic Context Budget", "desc": "Sizes tool results based on remaining context window.", "category": "intelligence", "type": "boolean", "default": True},
    {"key": "responseQualityGate", "label": "Response Quality Gate", "desc": "Catches LLM deflections and retries with tools.", "category": "intelligence", "type": "boolean", "default": True},
    {"key": "mcpAutoReconnect", "label": "MCP Auto-Reconnect", "desc": "Reconnects to MCP servers after 3 consecutive failures.", "category": "intelligence", "type": "boolean", "default": True},

    # Behavior
    {"key": "languageDetection", "label": "Multi-Language Detection", "desc": "Detect Bangla, Hindi, Urdu, Spanish, French, Arabic and respond in user's language.", "category": "behavior", "type": "boolean", "default": True},
    {"key": "destructiveConfirmation", "label": "Destructive Action Confirmation", "desc": "Ask for confirmation before delete all, clear, reset, rm -rf.", "category": "behavior", "type": "boolean", "default": True},
    {"key": "responseCaching", "label": "Response Caching", "desc": "Cache identical questions for 5 minutes — skip LLM on repeat.", "category": "behavior", "type": "boolean", "default": True},
    {"key": "frustrationDetection", "label": "Frustration Detection", "desc": "Detect user frustration (caps, !!!, angry words) and respond empathetically.", "category": "behavior", "type": "boolean", "default": True},
    {"key": "messageDedup", "label": "Message Dedup", "desc": "Drop duplicate messages sent within 30 seconds.", "category": "behavior", "type": "boolean", "default": True},
    {"key": "greetingInterceptor", "label": "Smart Greetings", "desc": "Answer 'hello' with a time-aware greeting + goals context (zero tokens).", "category": "behavior", "type": "boolean", "default": True},
    {"key": "mathInterceptor", "label": "Math Interceptor", "desc": "Answer math questions (15% of $347) with pure code — zero tokens.", "category": "behavior", "type": "boolean", "default": True},
    {"key": "llmFollowUps", "label": "AI Follow-Up Suggestions", "desc": "Use the LLM to generate smart follow-up questions (costs tokens). OFF = free pattern-based suggestions.", "category": "behavior", "type": "boolean", "default": False},

    # Jarvis Intelligence
    {"key": "morningPrep", "label": "Morning Prep", "desc": "Auto-generate briefing from goals, bills, relationships, habits.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "crossSignalCorrelation", "label": "Cross-Signal Correlation", "desc": "Connect dots: travel+weather, bills+dates, goals+deadlines → alerts.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "meetingIntelligence", "label": "Meeting Intelligence", "desc": "Prep before meetings — pull relevant emails, attendee notes.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "relationshipTracker", "label": "Relationship Tracker", "desc": "Track contact frequency. Alert when you haven't talked to someone.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "financialPulse", "label": "Financial Pulse", "desc": "Track AI spending trends, bill countdown, unusual charges.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "projectTracker", "label": "Project Tracker", "desc": "Multi-day tasks with progress tracking across sessions.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "dailyDigest", "label": "Daily Digest", "desc": "End-of-day summary: usage, goals, learnings.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "priorityInbox", "label": "Priority Inbox", "desc": "Score messages by urgency (keywords, caps, sender).", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "delegationQueue", "label": "Delegation Queue", "desc": "Async tasks Mawa checks on periodically.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "routineDetection", "label": "Routine Detection", "desc": "Detect daily patterns and offer to automate them.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "decisionMemory", "label": "Decision Memory", "desc": "Remember WHY you made decisions for future reference.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "peoplePrepBeforeCalls", "label": "People Prep", "desc": "Before a call, surface your last interactions with the person.", "category": "jarvis", "type": "boolean", "default": True},
    {"key": "relationshipReminderDays", "label": "Relationship Reminder (days)", "desc": "Alert when you haven't contacted someone for this many days.", "category": "jarvis", "type": "number", "default": 14, "min": 3, "max": 90},
    {"key": "delegationCheckHours", "label": "Delegation Check (hours)", "desc": "How often to check on delegated tasks.", "category": "jarvis", "type": "number", "default": 24, "min": 1, "max": 168},
    {"key": "correlationLookaheadDays", "label": "Correlation Lookahead (days)", "desc": "How far ahead to scan for correlated events.", "category": "jarvis", "type": "number", "default": 3, "min": 1, "max": 14},

    # Notifications
    {"key": "quietHoursEnabled", "label": "Quiet Hours", "desc": "Defer non-urgent notifications during quiet hours.", "category": "notifications", "type": "boolean", "default": True},
    {"key": "quietHoursStart", "label": "Quiet Start (hour)", "desc": "Hour to start quiet mode (24h).", "category": "notifications", "type": "number", "default": 22, "min": 0, "max": 23},
    {"key": "quietHoursEnd", "label": "Quiet End (hour)", "desc": "Hour to end quiet mode (24h).", "category": "notifications", "type": "number", "default": 7, "min": 0, "max": 23},

    # Budget
    {"key": "budget_daily_limit", "label": "Daily Budget ($)", "desc": "Maximum daily spending on LLM calls.", "category": "budget", "type": "number", "default": 5.0, "min": 0, "max": 100, "step": 0.5},
    {"key": "budget_weekly_limit", "label": "Weekly Budget ($)", "desc": "Maximum weekly spending.", "category": "budget", "type": "number", "default": 25.0, "min": 0, "max": 500, "step": 1},
    {"key": "budget_enforce", "label": "Hard Budget Enforcement", "desc": "Block LLM calls when budget exceeded (not just warn).", "category": "budget", "type": "boolean", "default": False},
    {"key": "budget_auto_switch_model", "label": "Budget Fallback Model", "desc": "Auto-switch to this cheaper model when budget >80%.", "category": "budget", "type": "string", "default": "", "placeholder": "e.g. claude-haiku-3-5"},

    # Maintenance
    {"key": "historyAutoArchive", "label": "History Auto-Archive", "desc": "Split HISTORY.md into monthly chunks when it exceeds 100KB.", "category": "maintenance", "type": "boolean", "default": True},
    {"key": "sessionAutoCleanup", "label": "Session Auto-Cleanup", "desc": "Delete inactive sessions older than 7 days.", "category": "maintenance", "type": "boolean", "default": True},
    {"key": "contactAutoExtract", "label": "Contact Auto-Extract", "desc": "Extract contacts from memory on heartbeat.", "category": "maintenance", "type": "boolean", "default": True},

    # Subagent behavior
    {"key": "subagentAutoInstallSkills", "label": "Subagent Auto-Install Skills", "desc": "OFF = subagents ask before installing skills from marketplace. ON = install automatically without asking.", "category": "behavior", "type": "boolean", "default": False},
    {"key": "subagentMarketplaceSource", "label": "Skill Marketplace Source", "desc": "Where subagents search for skills. both = skills.sh + clawhub.", "category": "behavior", "type": "string", "default": "both", "placeholder": "skills.sh, clawhub, both"},

    # Media generation
    {"key": "imageGenProvider", "label": "Image Provider", "desc": "Provider for image generation. Free: pollinations (no key), together, huggingface. Paid: fal, replicate, openai, stability.", "category": "media", "type": "string", "default": "pollinations", "placeholder": "pollinations, huggingface, together, fal, replicate, openai, stability"},
    {"key": "imageGenModel", "label": "Image Model", "desc": "Model to use. Leave empty for provider default.", "category": "media", "type": "string", "default": "", "placeholder": "provider default"},
    {"key": "imageGenEnabled", "label": "Image Generation", "desc": "Allow Mawa to generate images when asked or when a visual would help.", "category": "media", "type": "boolean", "default": True},

    # Voice providers
    {"key": "voiceSttProvider", "label": "Voice STT Provider", "desc": "Speech-to-text. deepgram (fast cloud) (GPU, emotion detection).", "category": "media", "type": "string", "default": "deepgram", "placeholder": "deepgram"},
    {"key": "voiceTtsProvider", "label": "Voice TTS Provider", "desc": "TTS engine. deepgram (fast EN), fish-speech (80+ langs), elevenlabs (premium quality, 29 langs).", "category": "media", "type": "string", "default": "deepgram", "placeholder": "deepgram, fish-speech, elevenlabs"},
    {"key": "voiceSttLanguage", "label": "STT Language (listen)", "desc": "What language you speak. multi=auto-detect, en=English, bn=Bengali, hi=Hindi. Can switch during voice call.", "category": "media", "type": "string", "default": "multi", "placeholder": "multi, en, bn, hi, zh, ur, ar, es, fr"},
    {"key": "voiceTtsLanguage", "label": "TTS Language (respond)", "desc": "What language Mawa speaks back. en=English, bn=Bengali, hi=Hindi. Independent from STT language.", "category": "media", "type": "string", "default": "en", "placeholder": "en, bn, hi, zh, ur, ar, es, fr"},
    {"key": "elevenlabsVoiceId", "label": "ElevenLabs Voice", "desc": "Voice ID for ElevenLabs TTS. Use built-in or custom cloned voice.", "category": "media", "type": "string", "default": "21m00Tcm4TlvDq8ikWAM", "placeholder": "voice ID"},
    {"key": "elevenlabsModel", "label": "ElevenLabs Model", "desc": "Flash (~75ms, fast) or Multilingual v2 (29 langs, best quality).", "category": "media", "type": "string", "default": "eleven_flash_v2_5", "placeholder": "eleven_flash_v2_5, eleven_multilingual_v2, eleven_turbo_v2_5"},

    # Phone calls
    {"key": "phoneCallEnabled", "label": "Phone Calls", "desc": "Allow Mawa to make outbound phone calls via Twilio.", "category": "media", "type": "boolean", "default": True},
    {"key": "phoneCallVoiceProvider", "label": "Call Voice Provider", "desc": "Voice provider for phone calls. deepgram = Twilio TTS.", "category": "media", "type": "string", "default": "deepgram", "placeholder": "deepgram, fish-speech"},
    {"key": "phoneCallMode", "label": "Call Mode", "desc": "tts = one-way message. conversation = two-way AI call (requires Deepgram).", "category": "media", "type": "string", "default": "tts", "placeholder": "tts, conversation"},
    {"key": "phoneCallDefaultVoice", "label": "Call Voice", "desc": "TTS voice. Options: alice, man, woman, Polly.Joanna, Polly.Matthew.", "category": "media", "type": "string", "default": "alice", "placeholder": "alice, man, woman, Polly.Joanna, Polly.Matthew"},
]


def get_feature_manifest(workspace: Path) -> list[dict[str, Any]]:
    """Build the complete feature manifest from unified settings."""
    settings = load_settings(workspace)
    features = []

    for fdef in _FEATURE_DEFS:
        feature = dict(fdef)
        feature["value"] = settings.get(fdef["key"], fdef["default"])
        feature.pop("default", None)
        features.append(feature)

    return features


def get_feature_categories() -> list[dict[str, str]]:
    """Return the ordered list of feature categories."""
    return [
        {"id": "intelligence", "label": "Intelligence", "icon": "brain"},
        {"id": "behavior", "label": "Agent Behavior", "icon": "sparkles"},
        {"id": "jarvis", "label": "Jarvis Intelligence", "icon": "zap"},
        {"id": "media", "label": "Media & Calls", "icon": "image"},
        {"id": "notifications", "label": "Notifications", "icon": "bell"},
        {"id": "budget", "label": "Cost & Budget", "icon": "dollar-sign"},
        {"id": "maintenance", "label": "Maintenance", "icon": "wrench"},
    ]
