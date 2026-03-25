"""Maintenance features — auto-archive, cleanup, safety, contacts, habits, quiet hours.

All pure code, zero LLM tokens:
1. HISTORY.md auto-archive (monthly chunks)
2. Session auto-cleanup (>7d inactive)
3. Confirmation before destructive actions (approval guard)
4. Undo/rollback journal
5. Conversation export
6. Contact book (structured)
7. Quiet hours (smart notification timing)
8. Habit tracking (recurring micro-reminders)
9. Multi-language detection
"""

from __future__ import annotations

import json
import re
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HISTORY.md Auto-Archive
# ═══════════════════════════════════════════════════════════════════════════════

def archive_history(workspace: Path, max_size_kb: int = 100) -> dict:
    """Archive HISTORY.md into monthly chunks when it exceeds max_size_kb.

    Splits by [YYYY-MM] date prefixes and moves old months to history/YYYY-MM.md.
    """
    history_file = workspace / "memory" / "HISTORY.md"
    if not history_file.exists():
        return {"archived": 0}

    size_kb = history_file.stat().st_size / 1024
    if size_kb <= max_size_kb:
        return {"archived": 0, "size_kb": round(size_kb, 1)}

    content = history_file.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Group lines by month
    current_month = date.today().strftime("%Y-%m")
    months: dict[str, list[str]] = {}
    current_key = current_month

    for line in lines:
        m = re.match(r"\[(\d{4}-\d{2})", line)
        if m:
            current_key = m.group(1)
        months.setdefault(current_key, []).append(line)

    # Archive all months except current
    archive_dir = workspace / "memory" / "history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = 0

    for month, month_lines in months.items():
        if month == current_month:
            continue
        archive_path = archive_dir / f"{month}.md"
        # Append if file exists
        existing = archive_path.read_text(encoding="utf-8") if archive_path.exists() else ""
        new_content = "\n".join(month_lines)
        if new_content.strip() and new_content.strip() not in existing:
            archive_path.write_text(existing + "\n" + new_content if existing else new_content, encoding="utf-8")
            archived += 1

    # Keep only current month in HISTORY.md
    current_lines = months.get(current_month, [])
    history_file.write_text("\n".join(current_lines), encoding="utf-8")

    new_size = history_file.stat().st_size / 1024
    logger.info("History archive: {} months archived, {:.0f}KB → {:.0f}KB", archived, size_kb, new_size)
    return {"archived": archived, "old_size_kb": round(size_kb, 1), "new_size_kb": round(new_size, 1)}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Session Auto-Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_old_sessions(workspace: Path, max_age_days: int = 7, keep_min: int = 5) -> dict:
    """Delete session files older than max_age_days, keeping at least keep_min."""
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        return {"deleted": 0}

    cutoff = time.time() - (max_age_days * 86400)
    files = sorted(sessions_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)

    # Never delete the most recent keep_min sessions
    protected = set(f.name for f in files[:keep_min])
    deleted = 0

    for f in files:
        if f.name in protected:
            continue
        if f.stat().st_mtime < cutoff:
            try:
                f.unlink()
                deleted += 1
            except OSError:
                pass

    if deleted:
        logger.info("Session cleanup: deleted {} old sessions", deleted)
    return {"deleted": deleted, "remaining": len(files) - deleted}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Destructive Action Confirmation
# ═══════════════════════════════════════════════════════════════════════════════

_DESTRUCTIVE_PATTERNS = [
    re.compile(r"\bdelete\s+(?:all|every)", re.I),
    re.compile(r"\bremove\s+(?:all|every)", re.I),
    re.compile(r"\bclear\s+(?:all|every|my)", re.I),
    re.compile(r"\breset\b", re.I),
    re.compile(r"\bdrop\s+(?:table|database|collection)", re.I),
    re.compile(r"\brm\s+-rf\b", re.I),
]

_DESTRUCTIVE_TOOLS = frozenset({
    "exec",  # Shell commands can be destructive
})

_DESTRUCTIVE_TOOL_ARGS = [
    (re.compile(r"\brm\s+-[rf]"), "exec"),
    (re.compile(r"\bdelete|DROP|TRUNCATE"), "exec"),
]


def needs_confirmation(tool_name: str, args: dict) -> str | None:
    """Check if a tool call needs user confirmation before execution.

    Returns a confirmation message or None if safe.
    """
    if tool_name == "exec":
        command = args.get("command", "")
        for pattern, _ in _DESTRUCTIVE_TOOL_ARGS:
            if pattern.search(command):
                return f"This will run a destructive command: `{command[:80]}`. Proceed?"

    if tool_name == "goals" and args.get("action") == "clear":
        return "This will clear all your goals. Are you sure?"

    return None


def is_destructive_message(text: str) -> bool:
    """Check if a user message requests a destructive action."""
    return any(p.search(text) for p in _DESTRUCTIVE_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Undo/Rollback Journal
# ═══════════════════════════════════════════════════════════════════════════════

_undo_journal: list[dict] = []
_MAX_UNDO = 20


def record_action(action_type: str, details: dict, rollback_cmd: str | None = None) -> None:
    """Record an action for potential undo."""
    _undo_journal.append({
        "type": action_type,
        "details": details,
        "rollback": rollback_cmd,
        "ts": time.time(),
    })
    if len(_undo_journal) > _MAX_UNDO:
        _undo_journal.pop(0)


def get_last_action() -> dict | None:
    """Get the most recent undoable action."""
    return _undo_journal[-1] if _undo_journal else None


def get_undo_history() -> list[dict]:
    """Get undo history for the UI."""
    return list(reversed(_undo_journal[-10:]))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Conversation Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_conversation(workspace: Path, session_key: str = "web_voice:voice", fmt: str = "markdown") -> str:
    """Export a conversation as markdown or JSON."""
    from nanobot.hooks.builtin.session_manager import get_session
    session = get_session(workspace, session_key.replace(":", "_"))
    if not session:
        return ""

    if fmt == "json":
        return json.dumps(session, indent=2, ensure_ascii=False, default=str)

    # Markdown format
    lines = [f"# Conversation Export\n", f"*Exported: {datetime.now().isoformat()}*\n"]
    for msg in session.get("messages", []):
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        if role and content:
            lines.append(f"### {role}" + (f" ({ts[:16]})" if ts else ""))
            lines.append(content + "\n")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Contact Book
# ═══════════════════════════════════════════════════════════════════════════════

def _contacts_path(workspace: Path) -> Path:
    return workspace / "contacts.json"


def get_contacts(workspace: Path) -> list[dict]:
    """Get all contacts."""
    path = _contacts_path(workspace)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_contact(workspace: Path, contact: dict) -> None:
    """Add or update a contact. Matches by name."""
    contacts = get_contacts(workspace)
    name = contact.get("name", "").strip()
    if not name:
        return

    # Update existing or add new
    found = False
    for i, c in enumerate(contacts):
        if c.get("name", "").lower() == name.lower():
            contacts[i] = {**c, **contact, "updated": datetime.now().isoformat()}
            found = True
            break
    if not found:
        contact["created"] = datetime.now().isoformat()
        contacts.append(contact)

    _contacts_path(workspace).write_text(json.dumps(contacts, indent=2, ensure_ascii=False))


def find_contact(workspace: Path, query: str) -> list[dict]:
    """Search contacts by name, email, or phone."""
    contacts = get_contacts(workspace)
    query_lower = query.lower()
    return [c for c in contacts if query_lower in json.dumps(c).lower()]


# Auto-extract contacts from memory
def extract_contacts_from_memory(workspace: Path) -> int:
    """Extract known contacts from LONG_TERM.md. Zero LLM cost."""
    lt = workspace / "memory" / "LONG_TERM.md"
    if not lt.exists():
        return 0

    content = lt.read_text(encoding="utf-8")
    contacts = get_contacts(workspace)
    existing_names = {c.get("name", "").lower() for c in contacts}
    added = 0

    # Extract name + email patterns
    email_pattern = re.compile(r"(\w[\w\s]+?)\s*[-—:]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
    for m in email_pattern.finditer(content):
        name = m.group(1).strip()
        email = m.group(2).strip()
        if name.lower() not in existing_names and len(name) > 2 and len(name) < 30:
            save_contact(workspace, {"name": name, "email": email, "source": "memory"})
            existing_names.add(name.lower())
            added += 1

    return added


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Quiet Hours (Smart Notification Timing)
# ═══════════════════════════════════════════════════════════════════════════════

_QUIET_HOURS_DEFAULT = {"start": 22, "end": 7, "enabled": True}


def load_quiet_hours(workspace: Path) -> dict:
    from nanobot.hooks.builtin.feature_registry import get_setting
    return {
        "enabled": get_setting(workspace, "quietHoursEnabled", _QUIET_HOURS_DEFAULT["enabled"]),
        "start": get_setting(workspace, "quietHoursStart", _QUIET_HOURS_DEFAULT["start"]),
        "end": get_setting(workspace, "quietHoursEnd", _QUIET_HOURS_DEFAULT["end"]),
    }


def save_quiet_hours(workspace: Path, settings: dict) -> None:
    from nanobot.hooks.builtin.feature_registry import save_setting
    save_setting(workspace, "quietHoursEnabled", settings.get("enabled", _QUIET_HOURS_DEFAULT["enabled"]))
    save_setting(workspace, "quietHoursStart", settings.get("start", _QUIET_HOURS_DEFAULT["start"]))
    save_setting(workspace, "quietHoursEnd", settings.get("end", _QUIET_HOURS_DEFAULT["end"]))


def is_quiet_time(workspace: Path) -> bool:
    """Check if current time is within quiet hours."""
    settings = load_quiet_hours(workspace)
    if not settings.get("enabled", True):
        return False

    hour = datetime.now().hour
    start = settings.get("start", 22)
    end = settings.get("end", 7)

    if start > end:  # Crosses midnight (e.g., 22-7)
        return hour >= start or hour < end
    return start <= hour < end


def should_send_notification(workspace: Path, priority: str = "normal") -> bool:
    """Check if a notification should be sent now (respects quiet hours).

    High priority notifications always go through.
    """
    if priority == "high":
        return True
    return not is_quiet_time(workspace)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Habit Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def _habits_path(workspace: Path) -> Path:
    return workspace / "habits.json"


def get_habits(workspace: Path) -> list[dict]:
    path = _habits_path(workspace)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_habit(workspace: Path, habit: dict) -> None:
    """Add a habit. habit = {name, interval_hours, last_reminded, enabled}."""
    habits = get_habits(workspace)
    habit.setdefault("enabled", True)
    habit.setdefault("last_reminded", 0)
    habit.setdefault("created", datetime.now().isoformat())

    # Update if exists
    for i, h in enumerate(habits):
        if h.get("name", "").lower() == habit.get("name", "").lower():
            habits[i] = {**h, **habit}
            _habits_path(workspace).write_text(json.dumps(habits, indent=2))
            return

    habits.append(habit)
    _habits_path(workspace).write_text(json.dumps(habits, indent=2))


def delete_habit(workspace: Path, name: str) -> bool:
    habits = get_habits(workspace)
    new = [h for h in habits if h.get("name", "").lower() != name.lower()]
    if len(new) < len(habits):
        _habits_path(workspace).write_text(json.dumps(new, indent=2))
        return True
    return False


def get_due_habits(workspace: Path) -> list[dict]:
    """Get habits that are due for a reminder. Zero LLM cost."""
    habits = get_habits(workspace)
    now = time.time()
    due = []

    for h in habits:
        if not h.get("enabled", True):
            continue
        interval = h.get("interval_hours", 24) * 3600
        last = h.get("last_reminded", 0)
        if now - last >= interval:
            due.append(h)

    return due


def mark_habit_reminded(workspace: Path, name: str) -> None:
    """Mark a habit as reminded (reset timer)."""
    habits = get_habits(workspace)
    for h in habits:
        if h.get("name", "").lower() == name.lower():
            h["last_reminded"] = time.time()
    _habits_path(workspace).write_text(json.dumps(habits, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Multi-Language Detection
# ═══════════════════════════════════════════════════════════════════════════════

# Common words for language detection (top 10 per language)
_LANG_MARKERS: dict[str, set[str]] = {
    "bangla": {"আমি", "তুমি", "আপনি", "করা", "হয়", "এটা", "কি", "না", "আছে", "হবে", "ভালো", "ধন্যবাদ"},
    "spanish": {"hola", "gracias", "por", "favor", "como", "está", "tengo", "quiero", "puedo", "necesito"},
    "french": {"bonjour", "merci", "comment", "pourquoi", "aussi", "mais", "avec", "dans", "pour", "très"},
    "arabic": {"مرحبا", "شكرا", "كيف", "أنا", "هل", "ما", "من", "في", "على", "هذا"},
    "hindi": {"नमस्ते", "कैसे", "हैं", "क्या", "मैं", "यह", "है", "को", "में", "धन्यवाद"},
    "urdu": {"آپ", "ہے", "کیا", "میں", "نہیں", "شکریہ", "کر", "ہوں", "سے", "یہ"},
}


def detect_language(text: str) -> str:
    """Detect message language. Returns 'english' or detected language code."""
    words = set(text.lower().split())

    best_lang = "english"
    best_score = 0

    for lang, markers in _LANG_MARKERS.items():
        score = len(words & markers)
        if score > best_score:
            best_score = score
            best_lang = lang

    # Also check for non-ASCII scripts
    if best_score == 0:
        # Bengali script
        if re.search(r'[\u0980-\u09FF]', text):
            return "bangla"
        # Arabic script
        if re.search(r'[\u0600-\u06FF]', text):
            return "arabic"
        # Devanagari
        if re.search(r'[\u0900-\u097F]', text):
            return "hindi"

    return best_lang if best_score >= 2 else "english"


# ═══════════════════════════════════════════════════════════════════════════════
# Heartbeat maintenance hook — runs archive + cleanup periodically
# ═══════════════════════════════════════════════════════════════════════════════

def run_maintenance(workspace: Path) -> dict:
    """Run all maintenance tasks. Called by heartbeat every 30 min."""
    from nanobot.hooks.builtin.feature_registry import get_setting

    results = {}

    # Archive bloated history
    if get_setting(workspace, "historyAutoArchive", True):
        results["history"] = archive_history(workspace)

    # Clean old sessions
    if get_setting(workspace, "sessionAutoCleanup", True):
        results["sessions"] = cleanup_old_sessions(workspace)

    # Extract contacts from memory
    if get_setting(workspace, "contactAutoExtract", True):
        results["contacts_extracted"] = extract_contacts_from_memory(workspace)

    return results
