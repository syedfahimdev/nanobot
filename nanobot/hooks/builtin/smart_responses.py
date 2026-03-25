"""Smart response features — all pure code, zero LLM tokens.

1. Response caching
2. Priority/urgency detection
3. Entity extraction (regex NER)
4. Conversation loop detector
5. Smart defaults (auto-fill from memory)
6. Progressive disclosure
7. Error translation
8. Link enrichment
9. Time-aware greeting
10. Frustration detection
11. Message dedup
12. Tool result merging
13. Semantic truncation
14. Auto-retry with context
15. Session health metrics
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Response Caching
# ═══════════════════════════════════════════════════════════════════════════════

_response_cache: dict[str, tuple[str, float]] = {}  # hash → (response, timestamp)
_CACHE_TTL = 300  # 5 minutes
_CACHE_MAX = 50


def cache_key(message: str) -> str:
    """Generate cache key from message (normalized)."""
    normalized = message.strip().lower()
    # Remove time-varying words
    normalized = re.sub(r"\b(today|now|current|latest)\b", "", normalized)
    return hashlib.md5(normalized.encode()).hexdigest()


_TIME_SENSITIVE_WORDS = re.compile(r"\b(today|now|current|latest|tonight|yesterday|tomorrow)\b", re.I)


def get_cached_response(message: str) -> str | None:
    """Get cached response for identical question. Returns None if miss."""
    if _TIME_SENSITIVE_WORDS.search(message):
        return None
    key = cache_key(message)
    if key in _response_cache:
        response, ts = _response_cache[key]
        if time.time() - ts < _CACHE_TTL:
            logger.info("Cache hit: {} (saved LLM call)", message[:40])
            return response
        del _response_cache[key]
    return None


def cache_response(message: str, response: str) -> None:
    """Cache a response for future identical questions."""
    # Don't cache very short or error responses
    if not response or len(response) < 10 or response.startswith("Error"):
        return
    # Don't cache time-sensitive queries
    if any(w in message.lower() for w in ["today", "now", "current", "latest", "weather"]):
        return

    key = cache_key(message)
    _response_cache[key] = (response, time.time())

    # Evict oldest if over limit
    if len(_response_cache) > _CACHE_MAX:
        oldest = min(_response_cache, key=lambda k: _response_cache[k][1])
        del _response_cache[oldest]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Priority / Urgency Detection
# ═══════════════════════════════════════════════════════════════════════════════

_URGENT_PATTERNS = [
    (re.compile(r"\b(urgent|emergency|asap|immediately|critical|broken|down|crashed)\b", re.I), "high"),
    (re.compile(r"\b(important|priority|deadline|soon|quickly|hurry)\b", re.I), "medium"),
    (re.compile(r"!!+"), "high"),  # Multiple exclamation marks
]


def detect_priority(text: str) -> str:
    """Detect message urgency. Returns 'high', 'medium', or 'normal'."""
    # Check for all-caps (frustration/urgency)
    words = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
    if caps_ratio > 0.5 and len(words) > 3:
        return "high"

    for pattern, level in _URGENT_PATTERNS:
        if pattern.search(text):
            return level

    return "normal"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Entity Extraction (Regex NER)
# ═══════════════════════════════════════════════════════════════════════════════

_ENTITY_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "money": re.compile(r"\$[\d,]+\.?\d{0,2}"),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b", re.I),
    "time": re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b"),
    "url": re.compile(r"https?://[^\s<>\"']+"),
    "percentage": re.compile(r"\b\d+\.?\d*\s*%"),
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text. Zero LLM cost."""
    entities: dict[str, list[str]] = {}
    for name, pattern in _ENTITY_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Flatten tuples from groups
            flat = [m if isinstance(m, str) else m[0] if m else "" for m in matches]
            entities[name] = [m for m in flat if m]
    return entities


def build_entity_context(text: str) -> str:
    """Build entity context block for injection. Only if entities found."""
    entities = extract_entities(text)
    if not entities:
        return ""

    lines = ["[Extracted entities]"]
    for etype, values in entities.items():
        lines.append(f"- {etype}: {', '.join(values[:3])}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Conversation Loop Detector
# ═══════════════════════════════════════════════════════════════════════════════

_recent_responses: dict[str, list[str]] = {}  # session → last N response hashes
_LOOP_WINDOW = 5


def detect_loop(session_key: str, response: str) -> bool:
    """Detect if the agent is producing repetitive responses (stuck in a loop)."""
    if not response or len(response) < 20:
        return False

    resp_hash = hashlib.md5(response[:200].encode()).hexdigest()
    history = _recent_responses.setdefault(session_key, [])
    history.append(resp_hash)

    # Keep only last N
    if len(history) > _LOOP_WINDOW:
        history.pop(0)

    # Check for repeated response (same hash appearing 2+ times in window)
    if history.count(resp_hash) >= 2:
        logger.warning("Loop detected in session {}: same response repeated", session_key)
        return True

    return False


def get_loop_breaker() -> str:
    """Get a message to inject when a loop is detected."""
    return (
        "[System: You are repeating yourself. STOP and try a completely different approach. "
        "If you cannot complete the task, say so clearly instead of retrying the same thing.]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Smart Defaults (auto-fill from memory)
# ═══════════════════════════════════════════════════════════════════════════════

def get_smart_defaults(text: str, workspace: Path) -> dict[str, str]:
    """Infer defaults from memory for incomplete requests.

    E.g., "send email" → auto-fill recipient if only one person in memory.
    """
    defaults = {}

    # Load long-term memory for known contacts/preferences
    lt_file = workspace / "memory" / "LONG_TERM.md"
    if not lt_file.exists():
        return defaults

    memory = lt_file.read_text(encoding="utf-8").lower()
    text_lower = text.lower()

    # Email default recipient
    if "email" in text_lower and "to" not in text_lower:
        # Check if there's a known primary contact
        email_match = re.search(r"email[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+)", memory)
        if email_match:
            defaults["suggested_recipient"] = email_match.group(1)

    # Calendar default — today's date
    if any(w in text_lower for w in ["schedule", "calendar", "meeting", "event"]):
        defaults["default_date"] = date.today().isoformat()

    # Timezone default
    tz_match = re.search(r"timezone?[:\s]+(\w+/\w+|\w{2,4})", memory)
    if tz_match:
        defaults["timezone"] = tz_match.group(1)

    return defaults


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Progressive Disclosure
# ═══════════════════════════════════════════════════════════════════════════════

def should_be_brief(text: str) -> bool:
    """Detect if the user wants a brief answer (yes/no, quick check, status)."""
    brief_patterns = [
        re.compile(r"^(is |are |do |does |did |has |have |can |will |should )", re.I),
        re.compile(r"\b(quick|brief|short|status|check)\b", re.I),
        re.compile(r"^(yes|no|ok|done|sure)\b", re.I),
    ]
    return any(p.search(text.strip()) for p in brief_patterns) or len(text.strip()) < 20


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Error Translation
# ═══════════════════════════════════════════════════════════════════════════════

_ERROR_TRANSLATIONS = {
    r"ConnectionRefusedError": "The service isn't running or isn't accepting connections.",
    r"TimeoutError": "The request took too long. The service might be overloaded.",
    r"JSONDecodeError": "The response wasn't in the expected format.",
    r"FileNotFoundError": "The file doesn't exist at that path.",
    r"PermissionError": "You don't have permission to access this.",
    r"401.*Unauthorized": "Authentication failed — check your credentials.",
    r"403.*Forbidden": "Access denied — you don't have permission.",
    r"404.*Not Found": "That resource doesn't exist.",
    r"429.*Too Many": "Rate limited — too many requests. Wait a moment.",
    r"500.*Internal": "The server had an error. Not your fault — try again.",
    r"502.*Bad Gateway": "The server is temporarily unreachable.",
    r"503.*Unavailable": "The service is down for maintenance.",
    r"ECONNREFUSED": "Can't connect to the service. Is it running?",
    r"ETIMEDOUT": "Connection timed out. Check your network.",
    r"SSL.*certificate": "SSL certificate issue — the connection isn't trusted.",
    r"ENOMEM|MemoryError": "Ran out of memory. Try a smaller operation.",
}


def translate_error(error_text: str) -> str:
    """Convert technical error messages to human-friendly language."""
    for pattern, friendly in _ERROR_TRANSLATIONS.items():
        if re.search(pattern, error_text, re.I):
            return friendly
    return ""


def humanize_error(result: str) -> str:
    """If result is an error, append a human-friendly translation."""
    if not isinstance(result, str) or not result.startswith("Error"):
        return result

    translation = translate_error(result)
    if translation:
        return f"{result}\n\n**In plain English:** {translation}"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Link Enrichment
# ═══════════════════════════════════════════════════════════════════════════════

_url_cache: dict[str, dict] = {}  # url → {title, domain, ts}


def enrich_urls(text: str) -> str:
    """Detect URLs in text and add domain context. No HTTP fetch — just parsing."""
    urls = re.findall(r"https?://[^\s<>\"']+", text)
    if not urls:
        return ""

    enrichments = []
    for url in urls[:3]:  # Max 3
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        path = parsed.path.strip("/")

        # Infer content type from domain
        domain_hints = {
            "github.com": "GitHub repository",
            "stackoverflow.com": "Stack Overflow question",
            "reddit.com": "Reddit post",
            "youtube.com": "YouTube video",
            "docs.google.com": "Google Doc",
            "drive.google.com": "Google Drive file",
            "linkedin.com": "LinkedIn profile",
            "twitter.com": "Twitter/X post",
            "x.com": "Twitter/X post",
            "medium.com": "Medium article",
            "arxiv.org": "Research paper",
        }
        hint = domain_hints.get(domain, f"{domain} page")
        enrichments.append(f"- {hint}: {url}")

    return "[Links detected]\n" + "\n".join(enrichments) if enrichments else ""


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Time-Aware Greeting
# ═══════════════════════════════════════════════════════════════════════════════

def get_time_greeting(name: str = "") -> str:
    """Generate a time-appropriate greeting."""
    hour = datetime.now().hour
    name_part = f" {name}" if name else ""

    if 5 <= hour < 12:
        return f"Good morning{name_part}"
    elif 12 <= hour < 17:
        return f"Good afternoon{name_part}"
    elif 17 <= hour < 21:
        return f"Good evening{name_part}"
    else:
        return f"Hey{name_part}, you're up late"


def is_greeting(text: str) -> bool:
    """Detect if the message is ONLY a greeting — not a greeting followed by a question.

    'Hey' → True, 'Hey Mawa' → True, 'Hey, what's the bitcoin price?' → False
    """
    stripped = text.strip()
    # If it has a question mark, it's a question not just a greeting
    if "?" in stripped:
        return False
    # If it's longer than 5 words, it probably has a real request after the greeting
    if len(stripped.split()) > 5:
        return False
    return bool(re.match(r"^(hi|hello|hey|yo|sup|good morning|good evening|good afternoon|howdy|greetings)\b", stripped, re.I))


def get_greeting_response(text: str, workspace: Path) -> str | None:
    """Generate a greeting response with context. Returns None if not a greeting."""
    if not is_greeting(text):
        return None

    # Get user's name from memory
    name = ""
    lt = workspace / "memory" / "LONG_TERM.md"
    if lt.exists():
        m = re.search(r"\*?\*?Name\*?\*?[:\s]+(\w+)", lt.read_text())
        if m:
            name = m.group(1)

    greeting = get_time_greeting(name)

    # Add contextual awareness
    extras = []
    goals_file = workspace / "memory" / "GOALS.md"
    if goals_file.exists():
        pending = sum(1 for l in goals_file.read_text().split("\n") if "- [ ]" in l)
        if pending > 0:
            extras.append(f"You have {pending} pending goal{'s' if pending > 1 else ''}")

    if extras:
        return f"{greeting}! {'. '.join(extras)}. How can I help?"
    return f"{greeting}! How can I help?"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Frustration Detection
# ═══════════════════════════════════════════════════════════════════════════════

_frustration_history: dict[str, list[float]] = {}  # session → timestamps of frustrated msgs

_FRUSTRATION_PATTERNS = [
    (re.compile(r"!!+"), 0.6),
    (re.compile(r"\b(ugh|wtf|ffs|omg|damn|shit|fuck)\b", re.I), 0.8),
    (re.compile(r"\b(this (doesn't|doesn't|isnt|isn't) work)", re.I), 0.7),
    (re.compile(r"\b(still|again|already told|keep telling)\b", re.I), 0.5),
    (re.compile(r"\b(useless|terrible|awful|worst|garbage|broken)\b", re.I), 0.7),
    (re.compile(r"\bwhy (can't|cant|won't|wont|doesn't|doesnt) (it|you|this)\b", re.I), 0.6),
]


def detect_frustration(text: str, session_key: str = "") -> tuple[bool, float]:
    """Detect user frustration. Returns (is_frustrated, score 0-1)."""
    score = 0.0

    # Pattern matching
    for pattern, weight in _FRUSTRATION_PATTERNS:
        if pattern.search(text):
            score = max(score, weight)

    # All caps check
    words = text.split()
    if len(words) > 3:
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / len(words)
        if caps_ratio > 0.5:
            score = max(score, 0.6)

    # Repeated frustration in session (escalation)
    if session_key and score > 0.3:
        history = _frustration_history.setdefault(session_key, [])
        history.append(time.time())
        # Keep last 5 minutes
        history[:] = [t for t in history if time.time() - t < 300]
        if len(history) >= 3:
            score = min(1.0, score + 0.2)  # Escalate

    return score >= 0.5, score


def get_frustration_preamble(score: float) -> str:
    """Get an empathetic preamble when frustration is detected."""
    if score >= 0.8:
        return "I understand this is really frustrating. Let me try a completely different approach. "
    elif score >= 0.6:
        return "Sorry about the trouble. Let me fix this. "
    return "Let me try that again. "


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Message Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

_recent_messages: dict[str, list[tuple[str, float]]] = {}  # session → [(hash, ts)]
_DEDUP_WINDOW = 30  # seconds


def is_duplicate(session_key: str, message: str) -> bool:
    """Check if this message was already sent within the dedup window."""
    msg_hash = hashlib.md5(message.strip().lower().encode()).hexdigest()
    now = time.time()

    history = _recent_messages.setdefault(session_key, [])
    # Clean old entries
    history[:] = [(h, t) for h, t in history if now - t < _DEDUP_WINDOW]

    # Check for duplicate
    if any(h == msg_hash for h, _ in history):
        logger.info("Dedup: dropping duplicate message in session {}", session_key)
        return True

    history.append((msg_hash, now))
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Tool Result Merging
# ═══════════════════════════════════════════════════════════════════════════════

def merge_tool_results(results: list[tuple[str, str]]) -> str:
    """Merge overlapping tool results to save context tokens.

    Input: [(tool_name, result_text), ...]
    Returns merged text with duplicates removed.
    """
    if len(results) <= 1:
        return results[0][1] if results else ""

    # Extract unique lines across all results
    seen_lines: set[str] = set()
    merged_sections: list[str] = []

    for tool_name, result in results:
        unique_lines = []
        for line in result.split("\n"):
            normalized = line.strip().lower()
            if len(normalized) < 5:
                unique_lines.append(line)  # Keep short lines (headers, separators)
                continue
            if normalized not in seen_lines:
                seen_lines.add(normalized)
                unique_lines.append(line)

        if unique_lines:
            section = "\n".join(unique_lines)
            if section.strip():
                merged_sections.append(f"[{tool_name}]\n{section}")

    merged = "\n\n".join(merged_sections)
    savings = sum(len(r) for _, r in results) - len(merged)
    if savings > 100:
        logger.debug("Tool merge: saved {} chars by deduplicating", savings)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Semantic Truncation
# ═══════════════════════════════════════════════════════════════════════════════

def semantic_truncate(text: str, max_chars: int) -> str:
    """Truncate at semantic boundaries (paragraph, sentence, line) not mid-word."""
    if len(text) <= max_chars:
        return text

    # Try paragraph boundary
    cut = text[:max_chars]
    last_para = cut.rfind("\n\n")
    if last_para > max_chars * 0.5:
        return cut[:last_para] + f"\n\n... ({len(text) - last_para:,} chars omitted)"

    # Try sentence boundary
    last_sentence = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if last_sentence > max_chars * 0.4:
        return cut[:last_sentence + 1] + f"\n... ({len(text) - last_sentence:,} chars omitted)"

    # Try line boundary
    last_line = cut.rfind("\n")
    if last_line > max_chars * 0.3:
        return cut[:last_line] + f"\n... ({len(text) - last_line:,} chars omitted)"

    # Last resort: word boundary
    last_space = cut.rfind(" ")
    if last_space > 0:
        return cut[:last_space] + f"... ({len(text) - last_space:,} chars omitted)"

    return cut + "..."


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Auto-Retry with Context
# ═══════════════════════════════════════════════════════════════════════════════

_retry_context: dict[str, list[str]] = {}  # session → list of failure reasons


def record_failure_context(session_key: str, tool_name: str, error: str) -> None:
    """Record a failure for retry context."""
    reasons = _retry_context.setdefault(session_key, [])
    reasons.append(f"{tool_name}: {error[:100]}")
    # Keep last 3
    if len(reasons) > 3:
        reasons.pop(0)


def get_retry_context(session_key: str) -> str:
    """Get accumulated failure context for smarter retries."""
    reasons = _retry_context.get(session_key, [])
    if not reasons:
        return ""

    lines = ["[Previous attempts that failed — try a DIFFERENT approach]"]
    for r in reasons:
        lines.append(f"- {r}")
    return "\n".join(lines)


def clear_retry_context(session_key: str) -> None:
    """Clear retry context (e.g., on success)."""
    _retry_context.pop(session_key, None)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Session Health Metrics
# ═══════════════════════════════════════════════════════════════════════════════

_session_metrics: dict[str, dict[str, Any]] = {}


def record_turn_metric(
    session_key: str,
    tokens: int = 0,
    tools_used: int = 0,
    duration_ms: float = 0,
    error: bool = False,
) -> None:
    """Record per-turn metrics for session health tracking."""
    metrics = _session_metrics.setdefault(session_key, {
        "turns": 0, "total_tokens": 0, "total_tools": 0,
        "total_duration_ms": 0, "errors": 0, "started_at": time.time(),
    })
    metrics["turns"] += 1
    metrics["total_tokens"] += tokens
    metrics["total_tools"] += tools_used
    metrics["total_duration_ms"] += duration_ms
    if error:
        metrics["errors"] += 1


def get_session_health(session_key: str) -> dict:
    """Get health metrics for a session."""
    metrics = _session_metrics.get(session_key)
    if not metrics:
        return {"status": "no_data"}

    turns = metrics["turns"]
    elapsed = time.time() - metrics["started_at"]

    return {
        "turns": turns,
        "tokens": metrics["total_tokens"],
        "avg_tokens_per_turn": metrics["total_tokens"] // max(turns, 1),
        "tools_used": metrics["total_tools"],
        "errors": metrics["errors"],
        "error_rate": round(metrics["errors"] / max(turns, 1) * 100, 1),
        "avg_response_ms": round(metrics["total_duration_ms"] / max(turns, 1)),
        "session_duration_min": round(elapsed / 60, 1),
        "status": "healthy" if metrics["errors"] / max(turns, 1) < 0.3 else "degraded",
    }


def get_all_session_health() -> dict[str, dict]:
    """Get health for all active sessions."""
    return {k: get_session_health(k) for k in _session_metrics}
