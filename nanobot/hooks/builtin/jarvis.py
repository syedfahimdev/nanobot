"""Jarvis-level proactive intelligence — makes Mawa feel alive.

Tier 1: Proactive awareness
1. Morning Prep Engine — auto-briefing ready before wake up
2. Meeting Intelligence — prep before meetings, debrief after
3. Cross-signal Correlation — connect dots across signals
4. Relationship Tracker — contact frequency, birthdays, follow-ups
5. Financial Pulse — spending patterns, bill countdown

Tier 2: Smart behavior
6. Project Tracker — multi-day tasks across sessions
7. Context Handoff — continue across devices
8. Smart Digest — end-of-day summary
9. Priority Inbox — score messages by urgency
10. Delegation Queue — async multi-day tasks

Tier 3: Irreplaceable
11. Routine Detection — automate detected patterns
12. Decision Memory — remember WHY decisions were made
13. People Prep — surface history before calls
14. Life Dashboard data — unified view

All pure code — zero LLM tokens.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: Proactive Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Morning Prep Engine ──────────────────────────────────────────────────

def build_morning_prep(workspace: Path) -> dict:
    """Build a morning prep package from all available data. No LLM.

    Checks: goals, calendar hints, weather hints, financial alerts,
    relationship reminders, habit reminders.
    Returns structured data the briefing system can format.
    """
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "morningPrep", True):
        return {"generated_at": "", "sections": []}

    prep = {
        "generated_at": datetime.now().isoformat(),
        "sections": [],
    }

    # Goals due today/tomorrow
    goals = _get_upcoming_goals(workspace)
    if goals:
        prep["sections"].append({
            "title": "Goals",
            "icon": "target",
            "items": goals,
            "priority": "high" if any("overdue" in g.lower() for g in goals) else "normal",
        })

    # Financial alerts
    finance = _get_financial_alerts(workspace)
    if finance:
        prep["sections"].append({
            "title": "Financial",
            "icon": "dollar-sign",
            "items": finance,
            "priority": "high" if any("overdue" in f.lower() for f in finance) else "normal",
        })

    # Relationship reminders
    relationships = get_relationship_reminders(workspace)
    if relationships:
        prep["sections"].append({
            "title": "People",
            "icon": "users",
            "items": relationships,
            "priority": "normal",
        })

    # Habits due
    habits = _get_habit_reminders(workspace)
    if habits:
        prep["sections"].append({
            "title": "Habits",
            "icon": "repeat",
            "items": habits,
            "priority": "low",
        })

    # Cross-signal correlations
    correlations = detect_correlations(workspace)
    if correlations:
        prep["sections"].append({
            "title": "Heads Up",
            "icon": "alert-triangle",
            "items": correlations,
            "priority": "high",
        })

    # Usage/spending summary
    spending = _get_spending_summary(workspace)
    if spending:
        prep["sections"].append({
            "title": "AI Usage",
            "icon": "cpu",
            "items": [spending],
            "priority": "low",
        })

    return prep


def format_morning_prep(prep: dict) -> str:
    """Format prep data into a readable briefing message."""
    if not prep.get("sections"):
        return ""

    # Sort by priority
    priority_order = {"high": 0, "normal": 1, "low": 2}
    sections = sorted(prep["sections"], key=lambda s: priority_order.get(s.get("priority", "normal"), 1))

    lines = ["**Morning Prep**\n"]
    for section in sections:
        emoji = {"target": "🎯", "dollar-sign": "💰", "users": "👥",
                 "repeat": "🔄", "alert-triangle": "⚠️", "cpu": "⬡"}.get(section["icon"], "•")
        lines.append(f"{emoji} **{section['title']}**")
        for item in section["items"][:5]:
            lines.append(f"  - {item}")
        lines.append("")

    return "\n".join(lines)


# ── 2. Meeting Intelligence ─────────────────────────────────────────────────

def get_meeting_prep(workspace: Path, event_title: str = "", attendees: list[str] | None = None) -> dict:
    """Build meeting prep from memory. No LLM.

    Searches HISTORY.md and LONG_TERM.md for mentions of the meeting
    topic and attendees.
    """
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "meetingIntelligence", True):
        return {"title": event_title, "context": [], "attendee_notes": {}}

    prep = {"title": event_title, "context": [], "attendee_notes": {}}

    # Search history for relevant context
    history = workspace / "memory" / "HISTORY.md"
    if history.exists() and event_title:
        content = history.read_text(encoding="utf-8")
        keywords = event_title.lower().split()
        for line in content.split("\n"):
            if any(kw in line.lower() for kw in keywords if len(kw) > 3):
                prep["context"].append(line.strip()[:150])
                if len(prep["context"]) >= 5:
                    break

    # Search for attendee info
    if attendees:
        lt = workspace / "memory" / "LONG_TERM.md"
        lt_content = lt.read_text(encoding="utf-8") if lt.exists() else ""
        contacts_path = workspace / "contacts.json"
        contacts = json.loads(contacts_path.read_text()) if contacts_path.exists() else []

        for name in attendees:
            notes = []
            # Check contacts
            for c in contacts:
                if name.lower() in json.dumps(c).lower():
                    notes.append(f"Contact: {c.get('name', '')} — {c.get('email', '')}")
            # Check memory
            for line in lt_content.split("\n"):
                if name.lower() in line.lower():
                    notes.append(line.strip()[:100])
            if notes:
                prep["attendee_notes"][name] = notes[:3]

    return prep


def cache_calendar_events(workspace: Path, events: list[dict]) -> None:
    """Cache calendar events for proactive meeting notifications.

    Called by the agent loop after calendar tool calls return results.
    Events format: [{"title": "...", "start": "2026-03-25T15:00:00", "end": "..."}]
    """
    cache_file = workspace / "calendar_cache.json"
    try:
        cache_file.write_text(json.dumps({"events": events, "fetched": datetime.now().isoformat()}, indent=2))
    except Exception:
        pass


def get_upcoming_meetings_from_memory(workspace: Path) -> list[dict]:
    """Get upcoming events from calendar cache + memory files.

    1. Calendar cache (written by agent after Google Calendar API calls) — most reliable
    2. Memory files (SHORT_TERM.md, HISTORY.md) — fallback for text-mentioned events
    """
    events = []
    today = date.today()

    # Source 1: Calendar cache (from actual Google Calendar API calls)
    cache_file = workspace / "calendar_cache.json"
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            # Only use cache if fetched today
            fetched = cache.get("fetched", "")
            if fetched.startswith(today.isoformat()):
                for ev in cache.get("events", []):
                    start = ev.get("start", "")
                    if not start:
                        continue
                    # Parse ISO datetime
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", start)
                    time_match = re.search(r"T(\d{2}:\d{2})", start)
                    if date_match and date_match.group(1) >= today.isoformat():
                        events.append({
                            "date": date_match.group(1),
                            "time": time_match.group(1) if time_match else "",
                            "title": ev.get("title", ev.get("summary", "Event"))[:100],
                        })
        except Exception:
            pass

    # Source 2: Memory files (text-mentioned events)
    for fname in ["SHORT_TERM.md", "HISTORY.md"]:
        f = workspace / "memory" / fname
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")

        for line in content.split("\n"):
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["meeting", "call", "appointment", "interview", "standup", "sync"]):
                date_match = re.search(r"(\d{4}-\d{2}-\d{2})", line)
                time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:am|pm)?)", line, re.I)
                if date_match:
                    event_date = date_match.group(1)
                    if event_date >= today.isoformat():
                        events.append({
                            "date": event_date,
                            "time": time_match.group(1) if time_match else "",
                            "title": line.strip()[:100],
                        })

    return sorted(events, key=lambda e: e["date"])[:10]


# ── 3. Cross-Signal Correlation ─────────────────────────────────────────────

def detect_correlations(workspace: Path) -> list[str]:
    """Detect cross-signal correlations. No LLM — pure pattern matching.

    Connects: goals + calendar, finances + dates, relationships + events.
    """
    alerts = []
    today = date.today()
    tomorrow = today + timedelta(days=1)

    lt = workspace / "memory" / "LONG_TERM.md"
    lt_content = lt.read_text(encoding="utf-8") if lt.exists() else ""
    st = workspace / "memory" / "SHORT_TERM.md"
    st_content = st.read_text(encoding="utf-8") if st.exists() else ""
    combined = lt_content + "\n" + st_content

    goals = workspace / "memory" / "GOALS.md"
    goals_content = goals.read_text(encoding="utf-8") if goals.exists() else ""

    # Pattern: Travel + weather
    travel_keywords = ["flight", "trip", "travel", "drive", "road trip", "airport", "hotel"]
    weather_keywords = ["snow", "storm", "rain", "freeze", "ice", "delay", "cancel"]
    has_travel = any(kw in combined.lower() for kw in travel_keywords)
    has_weather_risk = any(kw in combined.lower() for kw in weather_keywords)
    if has_travel and has_weather_risk:
        alerts.append("Travel planned + weather risk mentioned — check conditions before heading out")

    # Pattern: Bill due + low spending alert
    bill_keywords = ["overdue", "payment due", "suspend", "bill", "invoice"]
    for line in combined.split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in bill_keywords):
            # Extract date
            date_match = re.search(r"(\w+ \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})", line)
            if date_match:
                try:
                    for fmt in ("%B %d, %Y", "%B %d %Y", "%Y-%m-%d"):
                        try:
                            d = datetime.strptime(date_match.group(1).replace(",", ""), fmt).date()
                            from nanobot.hooks.builtin.feature_registry import get_setting as _gs
                            _lookahead = _gs(workspace, "correlationLookaheadDays", 3)
                            if 0 <= (d - today).days <= _lookahead:
                                alerts.append(f"Bill/payment due soon: {line.strip()[:80]}")
                                break
                        except ValueError:
                            continue
                except Exception:
                    pass

    # Pattern: Goal deadline + no progress
    for line in goals_content.split("\n"):
        if "- [ ]" in line:
            due_match = re.search(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)", line)
            if due_match:
                due = due_match.group(1)
                task = line.strip()[6:].split("(due:")[0].strip()
                days_left = (date.fromisoformat(due) - today).days
                if days_left == 0:
                    alerts.append(f"Goal due TODAY: {task}")
                elif days_left == 1:
                    alerts.append(f"Goal due tomorrow: {task}")
                elif days_left < 0:
                    alerts.append(f"OVERDUE goal ({-days_left}d): {task}")

    # Pattern: Wedding date approaching (from memory) — check ALL mentions
    for wedding_match in re.finditer(r"wedding.*?(\d{4}-\d{2}-\d{2}|\w+ \d{1,2},? \d{4})", combined, re.I):
        try:
            for fmt in ("%Y-%m-%d", "%B %d, %Y", "%B %d %Y"):
                try:
                    wd = datetime.strptime(wedding_match.group(1).replace(",", ""), fmt).date()
                    days = (wd - today).days
                    if 0 < days <= 30:
                        alerts.append(f"Wedding in {days} days — review wedding checklist")
                    break
                except ValueError:
                    continue
        except Exception:
            pass

    return alerts[:5]


# ── 4. Relationship Tracker ─────────────────────────────────────────────────

def _load_relationship_data(workspace: Path) -> dict:
    """Load relationship tracking data."""
    path = workspace / "relationships.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"contacts": {}, "interactions": []}


def _save_relationship_data(workspace: Path, data: dict) -> None:
    path = workspace / "relationships.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def record_interaction(workspace: Path, name: str, interaction_type: str = "message") -> None:
    """Record an interaction with a person. Called when messages mention people."""
    data = _load_relationship_data(workspace)
    data["contacts"].setdefault(name.lower(), {"name": name, "interactions": 0, "last_contact": ""})
    data["contacts"][name.lower()]["interactions"] += 1
    data["contacts"][name.lower()]["last_contact"] = datetime.now().isoformat()
    data["interactions"].append({
        "name": name, "type": interaction_type, "ts": datetime.now().isoformat(),
    })
    # Keep last 100 interactions
    data["interactions"] = data["interactions"][-100:]
    _save_relationship_data(workspace, data)


def get_relationship_reminders(workspace: Path) -> list[str]:
    """Get relationship reminders — people you haven't contacted recently."""
    reminders = []
    data = _load_relationship_data(workspace)
    now = datetime.now()

    from nanobot.hooks.builtin.feature_registry import get_setting
    threshold_days = get_setting(workspace, "relationshipReminderDays", 14)

    for key, contact in data.get("contacts", {}).items():
        last = contact.get("last_contact", "")
        if not last:
            continue
        try:
            last_dt = datetime.fromisoformat(last)
            days_ago = (now - last_dt).days
            name = contact.get("name", key)

            if days_ago >= threshold_days:
                reminders.append(f"Haven't contacted {name} in {days_ago} days")
            elif days_ago >= threshold_days // 2:
                reminders.append(f"It's been {days_ago} days since you talked to {name}")
        except Exception:
            continue

    # Check for birthdays from contacts.json
    contacts_path = workspace / "contacts.json"
    if contacts_path.exists():
        try:
            contacts = json.loads(contacts_path.read_text())
            today = date.today()
            for c in contacts:
                bday = c.get("birthday", "")
                if bday:
                    try:
                        bd = datetime.strptime(bday, "%Y-%m-%d").date()
                        bd_this_year = bd.replace(year=today.year)
                        days_until = (bd_this_year - today).days
                        if 0 <= days_until <= 7:
                            name = c.get("name", "Someone")
                            if days_until == 0:
                                reminders.append(f"🎂 {name}'s birthday is TODAY!")
                            else:
                                reminders.append(f"🎂 {name}'s birthday in {days_until} days")
                    except Exception:
                        continue
        except Exception:
            pass

    return reminders[:5]


def extract_people_from_text(text: str, workspace: Path) -> list[str]:
    """Extract mentioned people names from a message using contacts list."""
    contacts_path = workspace / "contacts.json"
    if not contacts_path.exists():
        return []

    try:
        contacts = json.loads(contacts_path.read_text())
    except Exception:
        return []

    mentioned = []
    text_lower = text.lower()
    for c in contacts:
        name = c.get("name", "")
        if name and name.lower() in text_lower:
            mentioned.append(name)

    return mentioned


# ── 5. Financial Pulse ──────────────────────────────────────────────────────

def get_financial_pulse(workspace: Path) -> dict:
    """Build a financial pulse from available data. No LLM.

    Checks: AI spending, mentioned bills, Zelle transactions, subscriptions.
    """
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "financialPulse", True):
        return {"ai_spending": "", "bills": [], "patterns": []}

    pulse = {
        "ai_spending": _get_spending_summary(workspace),
        "bills": _get_financial_alerts(workspace),
        "patterns": [],
        "alerts": [],
    }

    # Analyze AI spending trends
    from nanobot.hooks.builtin.usage_tracker import get_daily_totals
    today = date.today()
    week_costs = []
    for i in range(7):
        day = today - timedelta(days=i)
        totals = get_daily_totals(workspace, day)
        week_costs.append(totals.get("cost", 0.0))

    avg = sum(week_costs) / 7 if week_costs else 0
    today_cost = week_costs[0] if week_costs else 0

    if today_cost > avg * 2 and avg > 0:
        pulse["patterns"].append(f"AI spending today (${today_cost:.2f}) is {today_cost/avg:.1f}x your average")
        pulse["alerts"].append(f"Spending spike: ${today_cost:.2f} today vs ${avg:.2f} daily avg ({today_cost/avg:.1f}x)")
    if sum(week_costs) > 0:
        pulse["patterns"].append(f"This week: ${sum(week_costs):.2f} on AI ({sum(week_costs)/7:.2f}/day avg)")

    # Bill alerts
    for bill in pulse["bills"]:
        if "OVERDUE" in bill or "TODAY" in bill:
            pulse["alerts"].append(bill)

    return pulse


# ── Helper functions ────────────────────────────────────────────────────────

def _get_upcoming_goals(workspace: Path) -> list[str]:
    goals_file = workspace / "memory" / "GOALS.md"
    if not goals_file.exists():
        return []

    items = []
    today = date.today()
    for line in goals_file.read_text().split("\n"):
        if "- [ ]" in line:
            task = line.strip()[6:].split("(")[0].strip()
            due_match = re.search(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)", line)
            if due_match:
                due = date.fromisoformat(due_match.group(1))
                days = (due - today).days
                if days < 0:
                    items.append(f"OVERDUE ({-days}d): {task}")
                elif days == 0:
                    items.append(f"Due TODAY: {task}")
                elif days <= 3:
                    items.append(f"Due in {days}d: {task}")
            else:
                items.append(f"Pending: {task}")

    return items[:5]


def _get_financial_alerts(workspace: Path) -> list[str]:
    lt = workspace / "memory" / "LONG_TERM.md"
    if not lt.exists():
        return []

    alerts = []
    today = date.today()
    for line in lt.read_text().split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["overdue", "due", "payment", "suspend", "bill"]):
            dates = re.findall(r"(\w+ \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})", line)
            for d in dates:
                try:
                    for fmt in ("%B %d, %Y", "%B %d %Y", "%Y-%m-%d"):
                        try:
                            parsed = datetime.strptime(d.replace(",", ""), fmt).date()
                            days = (parsed - today).days
                            if -7 <= days <= 7:
                                status = "OVERDUE" if days < 0 else f"in {days}d" if days > 0 else "TODAY"
                                alerts.append(f"{status}: {line.strip()[:80]}")
                                break
                        except ValueError:
                            continue
                except Exception:
                    pass

    return alerts[:5]


def _get_habit_reminders(workspace: Path) -> list[str]:
    try:
        from nanobot.hooks.builtin.maintenance import get_due_habits
        return [f"Due: {h['name']}" for h in get_due_habits(workspace)]
    except Exception:
        return []


def _get_spending_summary(workspace: Path) -> str:
    try:
        from nanobot.hooks.builtin.usage_tracker import get_daily_totals
        today = date.today()
        totals = get_daily_totals(workspace, today)
        if totals.get("cost", 0) > 0:
            return f"Today: {totals['turns']} turns, ${totals['cost']:.4f}"

        # Check yesterday
        yesterday = today - timedelta(days=1)
        yt = get_daily_totals(workspace, yesterday)
        if yt.get("cost", 0) > 0:
            return f"Yesterday: {yt['turns']} turns, ${yt['cost']:.4f}"
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: Smart Behavior
# ═══════════════════════════════════════════════════════════════════════════════

# ── 6. Project Tracker ──────────────────────────────────────────────────────

def _projects_path(workspace: Path) -> Path:
    return workspace / "projects.json"


def get_projects(workspace: Path) -> list[dict]:
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "projectTracker", True):
        return []

    path = _projects_path(workspace)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def save_project(workspace: Path, project: dict) -> None:
    """Add or update a project."""
    projects = get_projects(workspace)
    name = project.get("name", "").strip()
    if not name:
        return

    # Update existing or add new
    found = False
    for i, p in enumerate(projects):
        if p.get("name", "").lower() == name.lower():
            projects[i] = {**p, **project, "updated": datetime.now().isoformat()}
            found = True
            break
    if not found:
        project["created"] = datetime.now().isoformat()
        project.setdefault("status", "active")
        project.setdefault("tasks", [])
        project.setdefault("progress", 0)
        projects.append(project)

    _projects_path(workspace).write_text(json.dumps(projects, indent=2, ensure_ascii=False))


def update_project_progress(workspace: Path, name: str) -> dict | None:
    """Recalculate project progress from tasks."""
    projects = get_projects(workspace)
    for p in projects:
        if p.get("name", "").lower() == name.lower():
            tasks = p.get("tasks", [])
            if tasks:
                done = sum(1 for t in tasks if t.get("done"))
                p["progress"] = round(done / len(tasks) * 100)
            _projects_path(workspace).write_text(json.dumps(projects, indent=2, ensure_ascii=False))
            return p
    return None


# ── 8. Smart Digest ─────────────────────────────────────────────────────────

def build_daily_digest(workspace: Path) -> dict:
    """Build end-of-day digest from all available data. No LLM."""
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "dailyDigest", True):
        return {"date": date.today().isoformat(), "sections": []}

    today = date.today()
    digest = {"date": today.isoformat(), "sections": []}

    # AI usage
    from nanobot.hooks.builtin.usage_tracker import get_daily_totals
    totals = get_daily_totals(workspace, today)
    if totals.get("turns", 0) > 0:
        digest["sections"].append({
            "title": "AI Usage",
            "items": [
                f"{totals['turns']} conversations",
                f"{totals['total_tokens']:,} tokens used",
                f"${totals['cost']:.4f} spent",
            ],
        })

    # Goals completed today
    goals = workspace / "memory" / "GOALS.md"
    if goals.exists():
        content = goals.read_text()
        done = sum(1 for l in content.split("\n") if "- [x]" in l)
        pending = sum(1 for l in content.split("\n") if "- [ ]" in l)
        if done > 0 or pending > 0:
            digest["sections"].append({
                "title": "Goals",
                "items": [f"{done} completed, {pending} remaining"],
            })

    # Learnings today
    learnings = workspace / "memory" / "LEARNINGS.md"
    if learnings.exists():
        today_str = today.strftime("%Y-%m-%d")
        today_rules = [l for l in learnings.read_text().split("\n")
                      if l.strip().startswith("- ") and today_str in l]
        if today_rules:
            digest["sections"].append({
                "title": "New Learnings",
                "items": [r.strip().lstrip("- ").split("[")[0].strip()[:80] for r in today_rules],
            })

    # Session count
    sessions_dir = workspace / "sessions"
    if sessions_dir.exists():
        today_sessions = [f for f in sessions_dir.glob("*.jsonl")
                         if datetime.fromtimestamp(f.stat().st_mtime).date() == today]
        if today_sessions:
            digest["sections"].append({
                "title": "Sessions",
                "items": [f"{len(today_sessions)} active sessions"],
            })

    return digest


def format_digest(digest: dict) -> str:
    """Format digest into readable message."""
    if not digest.get("sections"):
        return ""

    lines = [f"**Daily Digest — {digest['date']}**\n"]
    for section in digest["sections"]:
        lines.append(f"**{section['title']}**")
        for item in section["items"]:
            lines.append(f"  - {item}")
        lines.append("")

    return "\n".join(lines)


# ── 9. Priority Inbox ───────────────────────────────────────────────────────

_PRIORITY_KEYWORDS = {
    "high": ["urgent", "asap", "emergency", "critical", "deadline", "overdue", "final notice",
             "action required", "immediate", "important"],
    "medium": ["follow up", "reminder", "update", "review", "approval", "pending",
               "question", "help", "request"],
    "low": ["newsletter", "digest", "weekly", "promotion", "unsubscribe",
            "no-reply", "noreply", "automated"],
}


def score_message_priority(text: str, sender: str = "", workspace: Path | None = None) -> tuple[str, float]:
    """Score a message's priority. Returns (level, score 0-1). No LLM."""
    if workspace is not None:
        from nanobot.hooks.builtin.feature_registry import get_setting
        if not get_setting(workspace, "priorityInbox", True):
            return ("normal", 0.5)

    text_lower = text.lower()
    score = 0.5  # Default medium

    for kw in _PRIORITY_KEYWORDS["high"]:
        if kw in text_lower:
            score = max(score, 0.9)
    for kw in _PRIORITY_KEYWORDS["medium"]:
        if kw in text_lower:
            score = max(score, 0.6)
    for kw in _PRIORITY_KEYWORDS["low"]:
        if kw in text_lower:
            score = min(score, 0.3)

    # Known sender boost (from contacts)
    if sender:
        score = min(1.0, score + 0.1)

    level = "high" if score >= 0.7 else "low" if score <= 0.3 else "normal"
    return level, round(score, 2)


# ── 10. Delegation Queue ────────────────────────────────────────────────────

def _delegation_path(workspace: Path) -> Path:
    return workspace / "delegations.json"


def get_delegations(workspace: Path) -> list[dict]:
    path = _delegation_path(workspace)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def add_delegation(workspace: Path, task: str, deadline: str = "", check_interval_hours: int | None = None) -> None:
    """Add a delegated task that Mawa checks on periodically."""
    if check_interval_hours is None:
        from nanobot.hooks.builtin.feature_registry import get_setting
        check_interval_hours = get_setting(workspace, "delegationCheckHours", 24)
    delegations = get_delegations(workspace)
    delegations.append({
        "task": task,
        "deadline": deadline,
        "check_interval_hours": check_interval_hours,
        "status": "active",
        "created": datetime.now().isoformat(),
        "last_checked": "",
        "updates": [],
    })
    _delegation_path(workspace).write_text(json.dumps(delegations, indent=2))


def get_delegations_due_for_check(workspace: Path) -> list[dict]:
    """Get delegated tasks that need a check-in."""
    delegations = get_delegations(workspace)
    now = time.time()
    due = []

    for d in delegations:
        if d.get("status") != "active":
            continue
        last = d.get("last_checked", "")
        interval = d.get("check_interval_hours", 24) * 3600

        if not last:
            due.append(d)
        else:
            try:
                last_ts = datetime.fromisoformat(last).timestamp()
                if now - last_ts >= interval:
                    due.append(d)
            except Exception:
                due.append(d)

    return due


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: Irreplaceable
# ═══════════════════════════════════════════════════════════════════════════════

# ── 11. Routine Detection ───────────────────────────────────────────────────

def detect_routines(workspace: Path) -> list[dict]:
    """Detect daily routines from observation data. No LLM.

    Looks for tool sequences that happen at the same time of day regularly.
    """
    obs_file = workspace / "memory" / "OBSERVATIONS.md"
    if not obs_file.exists():
        return []

    routines = []
    content = obs_file.read_text()

    # Extract time-of-day patterns
    for line in content.split("\n"):
        if "mostly in the" in line and "(" in line:
            match = re.search(r"Uses (\S+) mostly in the (\w+(?:\s\w+)?)\s+\((\d+)x\)", line)
            if match:
                tool, time_bucket, count = match.group(1), match.group(2), int(match.group(3))
                if count >= 5:
                    routines.append({
                        "tool": tool,
                        "time": time_bucket,
                        "frequency": count,
                        "suggestion": f"You check {tool} every {time_bucket} — want me to do it automatically?",
                    })

    # Extract sequential patterns
    for line in content.split("\n"):
        if "Often does" in line:
            match = re.search(r"Often does (.+?) \((\d+)x\)", line)
            if match:
                sequence, count = match.group(1), int(match.group(2))
                if count >= 5:
                    routines.append({
                        "sequence": sequence,
                        "frequency": count,
                        "suggestion": f"You often do {sequence} — want me to automate this?",
                    })

    return routines


# ── 12. Decision Memory ─────────────────────────────────────────────────────

def _decisions_path(workspace: Path) -> Path:
    return workspace / "decisions.json"


def record_decision(workspace: Path, decision: str, reason: str, context: str = "") -> None:
    """Record a decision with its reasoning for future reference."""
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "decisionMemory", True):
        return

    path = _decisions_path(workspace)
    decisions = json.loads(path.read_text()) if path.exists() else []
    decisions.append({
        "decision": decision,
        "reason": reason,
        "context": context,
        "date": datetime.now().isoformat(),
    })
    # Keep last 50
    decisions = decisions[-50:]
    path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False))


def find_related_decisions(workspace: Path, query: str) -> list[dict]:
    """Find past decisions related to a query. Simple keyword match."""
    path = _decisions_path(workspace)
    if not path.exists():
        return []

    decisions = json.loads(path.read_text())
    query_lower = query.lower()
    matches = []
    for d in decisions:
        combined = f"{d.get('decision', '')} {d.get('reason', '')} {d.get('context', '')}".lower()
        if any(word in combined for word in query_lower.split() if len(word) > 3):
            matches.append(d)

    return matches[:5]


# ── 13. People Prep ─────────────────────────────────────────────────────────

def get_people_prep(workspace: Path, name: str) -> dict:
    """Get everything Mawa knows about a person for call/meeting prep."""
    from nanobot.hooks.builtin.feature_registry import get_setting
    if not get_setting(workspace, "peoplePrepBeforeCalls", True):
        return {"name": name, "interactions": [], "from_memory": [], "from_contacts": None}

    prep = {"name": name, "interactions": [], "from_memory": [], "from_contacts": None}

    # From relationship data
    rel_data = _load_relationship_data(workspace)
    contact = rel_data.get("contacts", {}).get(name.lower(), {})
    if contact:
        prep["last_contact"] = contact.get("last_contact", "")
        prep["total_interactions"] = contact.get("interactions", 0)

    # Recent interactions
    for interaction in reversed(rel_data.get("interactions", [])):
        if interaction.get("name", "").lower() == name.lower():
            prep["interactions"].append(interaction)
            if len(prep["interactions"]) >= 5:
                break

    # From contacts.json
    contacts_path = workspace / "contacts.json"
    if contacts_path.exists():
        for c in json.loads(contacts_path.read_text()):
            if name.lower() in c.get("name", "").lower():
                prep["from_contacts"] = c
                break

    # From memory
    for fname in ["LONG_TERM.md", "HISTORY.md"]:
        f = workspace / "memory" / fname
        if not f.exists():
            continue
        for line in f.read_text().split("\n"):
            if name.lower() in line.lower() and len(line.strip()) > 10:
                prep["from_memory"].append(line.strip()[:120])
                if len(prep["from_memory"]) >= 5:
                    break

    return prep


# ── 14. Life Dashboard Data ─────────────────────────────────────────────────

def get_life_dashboard(workspace: Path) -> dict:
    """Build unified life dashboard data. No LLM."""
    today = date.today()

    dashboard = {
        "health": {
            "goals_pending": 0,
            "goals_done": 0,
            "goals_overdue": 0,
            "habits_due": len(_get_habit_reminders(workspace)),
        },
        "wealth": get_financial_pulse(workspace),
        "relationships": {
            "reminders": get_relationship_reminders(workspace),
            "total_contacts": 0,
        },
        "work": {
            "projects": [{"name": p["name"], "progress": p.get("progress", 0), "status": p.get("status")}
                        for p in get_projects(workspace) if p.get("status") == "active"],
            "delegations_active": len([d for d in get_delegations(workspace) if d.get("status") == "active"]),
        },
        "upcoming": get_upcoming_meetings_from_memory(workspace)[:5],
        "correlations": detect_correlations(workspace),
    }

    # Goals stats
    goals = workspace / "memory" / "GOALS.md"
    if goals.exists():
        for line in goals.read_text().split("\n"):
            if "- [x]" in line:
                dashboard["health"]["goals_done"] += 1
            elif "- [ ]" in line:
                dashboard["health"]["goals_pending"] += 1
                due_match = re.search(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)", line)
                if due_match and due_match.group(1) < today.isoformat():
                    dashboard["health"]["goals_overdue"] += 1

    # Contact count
    contacts_path = workspace / "contacts.json"
    if contacts_path.exists():
        try:
            dashboard["relationships"]["total_contacts"] = len(json.loads(contacts_path.read_text()))
        except Exception:
            pass

    return dashboard


# ═══════════════════════════════════════════════════════════════════════════════
# Proactive Hook — runs on heartbeat to check for actionable items
# ═══════════════════════════════════════════════════════════════════════════════

_LAST_NOTIFIED: dict[str, datetime] = {}


def check_proactive_jarvis(workspace: Path) -> list[dict]:
    """Run all Jarvis proactive checks. Called by heartbeat.

    Returns list of notifications to send.
    """
    from nanobot.hooks.builtin.feature_registry import get_setting

    notifications = []

    # Cross-signal correlations
    if get_setting(workspace, "crossSignalCorrelation", True):
        correlations = detect_correlations(workspace)
        for c in correlations:
            notifications.append({"content": f"⚠️ {c}", "priority": "high"})

    # Relationship reminders (once per day)
    if get_setting(workspace, "relationshipTracker", True):
        reminders = get_relationship_reminders(workspace)
        for r in reminders[:2]:
            notifications.append({"content": f"👥 {r}", "priority": "normal"})

    # Delegation check-ins
    if get_setting(workspace, "delegationQueue", True):
        due_delegations = get_delegations_due_for_check(workspace)
        for d in due_delegations:
            notifications.append({
                "content": f"📋 Delegation check: {d['task'][:60]}",
                "priority": "normal",
            })

    # Routine suggestions
    if get_setting(workspace, "routineDetection", True):
        routines = detect_routines(workspace)
        for r in routines[:1]:
            notifications.append({
                "content": f"🔄 {r.get('suggestion', '')}",
                "priority": "low",
            })

    # Meeting intelligence — alert for events starting in the next 15 min
    if get_setting(workspace, "meetingIntelligence", True):
        upcoming = get_upcoming_meetings_from_memory(workspace)
        now = datetime.now()
        for event in upcoming:
            try:
                event_date = event.get("date", "")
                event_time = event.get("time", "")
                if event_date and event_time:
                    # Parse event datetime
                    dt_str = f"{event_date} {event_time}"
                    for fmt in ["%Y-%m-%d %I:%M %p", "%Y-%m-%d %I:%M%p", "%Y-%m-%d %H:%M"]:
                        try:
                            event_dt = datetime.strptime(dt_str.strip(), fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        continue
                    # Check if within 15 minutes
                    delta = (event_dt - now).total_seconds()
                    if 0 < delta <= 900:  # 0-15 minutes from now
                        meeting_key = f"meeting_{event_date}_{event_time}"
                        if meeting_key not in _LAST_NOTIFIED:
                            _LAST_NOTIFIED[meeting_key] = now
                            title = event.get("title", "Meeting")[:60]
                            mins = int(delta / 60)
                            notifications.append({
                                "content": f"📅 Meeting in {mins} min: {title}",
                                "priority": "high",
                                "metadata": {"_notification": True, "_proactive": True, "_priority": "high"},
                            })
            except Exception:
                continue

    # Daily digest — send once per day after 8pm
    if get_setting(workspace, "dailyDigest", True):
        now = datetime.now()
        if now.hour >= 20:
            digest_key = f"digest_{now.strftime('%Y-%m-%d')}"
            if digest_key not in _LAST_NOTIFIED:
                digest = build_daily_digest(workspace)
                if digest:
                    formatted = format_digest(digest)
                    if formatted:
                        _LAST_NOTIFIED[digest_key] = now
                        notifications.append({"content": formatted, "metadata": {"_notification": True, "_proactive": True}})

    # Financial spending spike alerts
    if get_setting(workspace, "financialPulse", True):
        now = datetime.now()
        pulse = get_financial_pulse(workspace)
        if pulse.get("alerts"):
            for alert in pulse["alerts"][:2]:
                alert_key = f"fin_{alert[:30]}_{now.strftime('%Y-%m-%d')}"
                if alert_key not in _LAST_NOTIFIED:
                    _LAST_NOTIFIED[alert_key] = now
                    notifications.append({"content": f"💰 {alert}", "metadata": {"_notification": True, "_proactive": True, "_priority": "medium"}})

    return notifications
