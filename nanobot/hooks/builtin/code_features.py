"""Pure code-level features — zero LLM tokens.

All 15 features implemented as standalone functions/classes:
1. Hard budget enforcement
2. Predictive suggestions
3. File auto-cleanup
4. Session search
5. File watcher
6. Cron dashboard
7. Smart retry queue
8. Health dashboard (expanded)
9. Auto-model downgrade
10. Event→Action rules
11. Session tags
12. Tool favorites
13. Anomaly detection
14. Batch file processing
15. Schedule templates
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Hard Budget Enforcement
# ═══════════════════════════════════════════════════════════════════════════════

def enforce_budget(workspace: Path) -> tuple[bool, str]:
    """Check if execution should be blocked due to budget.

    Returns (allowed, reason). If not allowed, caller should refuse the LLM call.
    """
    from nanobot.hooks.builtin.cost_budget import check_budget, load_budget

    budget = load_budget(workspace)
    if not budget.get("enforce", False):
        return True, ""

    status = check_budget(workspace)
    if status["exceeded"]:
        return False, (
            f"Daily budget exceeded (${status['daily_used']:.2f} / ${status['daily_limit']:.2f}). "
            f"Use /budget to adjust or wait until tomorrow."
        )
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Predictive Suggestions
# ═══════════════════════════════════════════════════════════════════════════════

def get_predictive_suggestions(workspace: Path) -> list[str]:
    """Suggest actions based on observed usage patterns.

    Uses time-of-day and day-of-week patterns from OBSERVATIONS.md
    and usage history. Zero LLM cost — pure stats.
    """
    now = datetime.now()
    hour = now.hour
    weekday = now.strftime("%A")
    suggestions = []

    # Check observations for time-of-day patterns
    obs_file = workspace / "memory" / "OBSERVATIONS.md"
    if obs_file.exists():
        content = obs_file.read_text(encoding="utf-8")
        time_bucket = _get_time_bucket(hour)

        for line in content.split("\n"):
            if f"in the {time_bucket}" in line.lower():
                # Extract tool name
                m = re.search(r"Uses (\S+) mostly", line)
                if m:
                    tool = m.group(1)
                    suggestions.append(_tool_to_suggestion(tool))

            if f"on {weekday}s" in line.lower():
                m = re.search(r"Uses (\S+) often", line)
                if m:
                    tool = m.group(1)
                    suggestion = _tool_to_suggestion(tool)
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)

    # Check goals due today/tomorrow
    goals_file = workspace / "memory" / "GOALS.md"
    if goals_file.exists():
        today = date.today().isoformat()
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        for line in goals_file.read_text().split("\n"):
            if "- [ ]" in line and (today in line or tomorrow in line):
                task = line.strip()[6:].split("(")[0].strip()
                suggestions.append(f"You have a task due: {task}")

    return suggestions[:5]


def _get_time_bucket(hour: int) -> str:
    buckets = {
        "early morning": range(5, 8), "morning": range(8, 12),
        "afternoon": range(12, 17), "evening": range(17, 21),
        "night": range(21, 24), "late night": range(0, 5),
    }
    for name, hours in buckets.items():
        if hour in hours:
            return name
    return "unknown"


_TOOL_SUGGESTIONS = {
    "mcp_composio_GMAIL_FETCH_EMAILS": "Check your email?",
    "mcp_composio_GOOGLECALENDAR_EVENTS_LIST": "Check your calendar?",
    "web_search": "Search the web for something?",
    "goals": "Review your goals?",
    "browser": "Open a website?",
}


def _tool_to_suggestion(tool: str) -> str:
    return _TOOL_SUGGESTIONS.get(tool, f"You often use {tool} around this time")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. File Auto-Cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def auto_cleanup(workspace: Path, max_age_days: int = 30, max_size_mb: int = 500) -> dict:
    """Clean up old generated files and enforce disk quotas.

    Returns summary of what was cleaned.
    """
    cleaned = {"files_deleted": 0, "bytes_freed": 0, "dirs_cleaned": []}
    cutoff = time.time() - (max_age_days * 86400)

    # Clean generated files
    for gen_dir in [workspace / "generated", workspace / "inbox"]:
        if not gen_dir.exists():
            continue
        for f in gen_dir.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                size = f.stat().st_size
                try:
                    f.unlink()
                    cleaned["files_deleted"] += 1
                    cleaned["bytes_freed"] += size
                except OSError:
                    pass

    # Check total workspace size and warn
    total = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
    cleaned["total_size_mb"] = round(total / (1024 * 1024), 1)
    cleaned["over_quota"] = cleaned["total_size_mb"] > max_size_mb

    # Clean empty directories
    for d in sorted(workspace.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            try:
                d.rmdir()
                cleaned["dirs_cleaned"].append(str(d.relative_to(workspace)))
            except OSError:
                pass

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Session Search
# ═══════════════════════════════════════════════════════════════════════════════

def search_sessions(workspace: Path, query: str, max_results: int = 20) -> list[dict]:
    """Search across ALL sessions for a keyword. Pure grep — no embeddings."""
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists() or not query.strip():
        return []

    results = []
    query_lower = query.lower()

    for path in sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = msg.get("content", "")
                if not isinstance(content, str):
                    continue

                if query_lower in content.lower():
                    # Find the match position for context snippet
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(query) + 80)
                    snippet = content[start:end].strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet += "..."

                    results.append({
                        "session": path.stem,
                        "role": msg.get("role", ""),
                        "snippet": snippet,
                        "timestamp": msg.get("timestamp", ""),
                        "line": i,
                    })

                    if len(results) >= max_results:
                        return results
        except (OSError, UnicodeDecodeError):
            continue

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. File Watcher
# ═══════════════════════════════════════════════════════════════════════════════

_file_watcher_task: asyncio.Task | None = None


async def start_file_watcher(
    workspace: Path,
    watch_dirs: list[str] | None = None,
    callback=None,
    poll_interval: float = 5.0,
) -> None:
    """Watch directories for file changes. Uses polling (no external deps).

    Triggers callback(event_type, path) when files are created/modified/deleted.
    """
    global _file_watcher_task

    dirs = [Path(d) for d in (watch_dirs or [str(workspace / "inbox")])]
    snapshots: dict[str, float] = {}  # path → mtime

    # Initial snapshot
    for d in dirs:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    snapshots[str(f)] = f.stat().st_mtime

    async def _poll():
        nonlocal snapshots
        while True:
            await asyncio.sleep(poll_interval)
            current: dict[str, float] = {}
            for d in dirs:
                if not d.exists():
                    continue
                for f in d.rglob("*"):
                    if f.is_file():
                        current[str(f)] = f.stat().st_mtime

            # New files
            for path, mtime in current.items():
                if path not in snapshots:
                    logger.info("FileWatcher: new file {}", path)
                    if callback:
                        await callback("created", path)
                elif mtime > snapshots[path]:
                    logger.info("FileWatcher: modified {}", path)
                    if callback:
                        await callback("modified", path)

            # Deleted files
            for path in set(snapshots) - set(current):
                logger.info("FileWatcher: deleted {}", path)
                if callback:
                    await callback("deleted", path)

            snapshots = current

    _file_watcher_task = asyncio.create_task(_poll())
    logger.info("FileWatcher started: watching {}", [str(d) for d in dirs])


def stop_file_watcher() -> None:
    """Stop the file watcher."""
    global _file_watcher_task
    if _file_watcher_task:
        _file_watcher_task.cancel()
        _file_watcher_task = None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cron Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def get_cron_dashboard(workspace: Path) -> dict:
    """Get cron job status for frontend dashboard. No LLM."""
    cron_path = workspace.parent / "cron" / "jobs.json"
    if not cron_path.exists():
        return {"jobs": [], "stats": {"total": 0, "enabled": 0, "failed": 0}}

    try:
        data = json.loads(cron_path.read_text())
        jobs = data.get("jobs", [])
    except (json.JSONDecodeError, OSError):
        return {"jobs": [], "stats": {"total": 0, "enabled": 0, "failed": 0}}

    enriched = []
    now_ms = int(time.time() * 1000)

    for j in jobs:
        state = j.get("state", {})
        next_run = state.get("nextRunAtMs")
        last_run = state.get("lastRunAtMs")

        enriched.append({
            "id": j.get("id"),
            "name": j.get("name"),
            "enabled": j.get("enabled", False),
            "lastStatus": state.get("lastStatus"),
            "lastError": state.get("lastError"),
            "lastRun": datetime.fromtimestamp(last_run / 1000).isoformat() if last_run else None,
            "nextRun": datetime.fromtimestamp(next_run / 1000).isoformat() if next_run else None,
            "overdue": next_run is not None and next_run < now_ms,
            "schedule": j.get("schedule", {}).get("kind"),
        })

    stats = {
        "total": len(jobs),
        "enabled": sum(1 for j in jobs if j.get("enabled")),
        "failed": sum(1 for j in enriched if j["lastStatus"] == "error"),
        "overdue": sum(1 for j in enriched if j.get("overdue")),
    }

    return {"jobs": enriched, "stats": stats}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Smart Retry Queue (disk-backed outbound)
# ═══════════════════════════════════════════════════════════════════════════════

_RETRY_DIR = "retry_queue"
_MAX_RETRIES = 3
_RETRY_DELAY_S = 30


def queue_for_retry(workspace: Path, channel: str, chat_id: str, content: str, metadata: dict | None = None) -> None:
    """Save a failed outbound message for retry."""
    retry_dir = workspace / _RETRY_DIR
    retry_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "channel": channel,
        "chat_id": chat_id,
        "content": content,
        "metadata": metadata or {},
        "queued_at": time.time(),
        "retries": 0,
    }

    path = retry_dir / f"{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(entry))
    logger.info("RetryQueue: saved message for {}/{}", channel, chat_id)


def get_pending_retries(workspace: Path) -> list[dict]:
    """Get all pending retry messages."""
    retry_dir = workspace / _RETRY_DIR
    if not retry_dir.exists():
        return []

    pending = []
    for f in sorted(retry_dir.glob("*.json")):
        try:
            entry = json.loads(f.read_text())
            entry["_path"] = str(f)
            if entry.get("retries", 0) < _MAX_RETRIES:
                pending.append(entry)
            else:
                f.unlink()  # Exhausted retries — drop
        except (json.JSONDecodeError, OSError):
            pass

    return pending


def mark_retry_done(path: str) -> None:
    """Remove a retry entry after successful delivery."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def increment_retry(path: str) -> None:
    """Increment retry count for a failed delivery."""
    try:
        p = Path(path)
        entry = json.loads(p.read_text())
        entry["retries"] = entry.get("retries", 0) + 1
        p.write_text(json.dumps(entry))
    except (json.JSONDecodeError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Health Dashboard (expanded)
# ═══════════════════════════════════════════════════════════════════════════════

def get_health_dashboard(workspace: Path) -> dict:
    """Comprehensive health check — disk, memory, cron, tools, budget."""
    checks = {}

    # Disk
    try:
        usage = shutil.disk_usage(str(workspace))
        checks["disk"] = {
            "ok": usage.free > 1024**3,
            "free_gb": round(usage.free / (1024**3), 1),
            "used_pct": round((usage.used / usage.total) * 100, 1),
        }
    except Exception:
        checks["disk"] = {"ok": False, "error": "Cannot read disk"}

    # Workspace size
    try:
        total = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
        checks["workspace_mb"] = round(total / (1024 * 1024), 1)
    except Exception:
        checks["workspace_mb"] = -1

    # Sessions health
    sessions_dir = workspace / "sessions"
    if sessions_dir.exists():
        files = list(sessions_dir.glob("*.jsonl"))
        total_size = sum(f.stat().st_size for f in files)
        checks["sessions"] = {
            "count": len(files),
            "size_mb": round(total_size / (1024 * 1024), 1),
            "ok": total_size < 500 * 1024 * 1024,
        }
    else:
        checks["sessions"] = {"count": 0, "size_mb": 0, "ok": True}

    # Memory files
    mem_dir = workspace / "memory"
    if mem_dir.exists():
        mem_files = {}
        for f in mem_dir.glob("*.md"):
            mem_files[f.name] = round(f.stat().st_size / 1024, 1)
        checks["memory_files_kb"] = mem_files

    # Cron health
    checks["cron"] = get_cron_dashboard(workspace).get("stats", {})

    # Tool scores summary
    scores_file = workspace / "memory" / "tool_scores.json"
    if scores_file.exists():
        try:
            scores = json.loads(scores_file.read_text())
            failing_tools = [
                name for name, data in scores.items()
                if isinstance(data, dict) and data.get("total", 0) > 3
                and data.get("success", 0) / max(data.get("total", 1), 1) < 0.5
            ]
            checks["failing_tools"] = failing_tools
        except Exception:
            pass

    # Budget status
    try:
        from nanobot.hooks.builtin.cost_budget import check_budget
        checks["budget"] = check_budget(workspace)
    except Exception:
        pass

    # Retry queue
    pending = get_pending_retries(workspace)
    checks["retry_queue"] = len(pending)

    return checks


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Auto-Model Downgrade
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered from expensive → cheap. When budget hits threshold, step down.
_MODEL_TIERS = [
    ("claude-opus-4-5", "claude-sonnet-4-6"),
    ("claude-sonnet-4-6", "claude-haiku-3-5"),
    ("claude-sonnet-4-5", "claude-haiku-3-5"),
    ("gpt-4o", "gpt-4o-mini"),
    ("gpt-4-turbo", "gpt-4o-mini"),
    ("gemini-1.5-pro", "gemini-2.0-flash"),
]


def get_downgrade_model(current_model: str, workspace: Path) -> str | None:
    """If budget is >80%, return a cheaper model. Otherwise None."""
    from nanobot.hooks.builtin.cost_budget import check_budget

    status = check_budget(workspace)
    if not status["alert"]:
        return None  # Under budget — keep current model

    bare = current_model.split("/", 1)[-1] if "/" in current_model else current_model
    for expensive, cheap in _MODEL_TIERS:
        if expensive in bare:
            logger.info("Auto-downgrade: {} → {} (budget at {:.0f}%)", bare, cheap, status["daily_pct"] * 100)
            return cheap

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Event → Action Rules
# ═══════════════════════════════════════════════════════════════════════════════

def load_rules(workspace: Path) -> list[dict]:
    """Load event→action rules from rules.json."""
    path = workspace / "rules.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_rules(workspace: Path, rules: list[dict]) -> None:
    """Save rules to disk."""
    path = workspace / "rules.json"
    path.write_text(json.dumps(rules, indent=2))


def match_rules(workspace: Path, event_type: str, event_content: str) -> list[dict]:
    """Match incoming events against rules. Returns matching actions."""
    rules = load_rules(workspace)
    matched = []

    for rule in rules:
        if not rule.get("enabled", True):
            continue
        # Match by event type
        if rule.get("event_type") and rule["event_type"] != event_type:
            continue
        # Match by keyword in content
        keywords = rule.get("keywords", [])
        if keywords:
            content_lower = event_content.lower()
            if not any(kw.lower() in content_lower for kw in keywords):
                continue
        # Match by sender
        sender = rule.get("from_sender")
        if sender and sender.lower() not in event_content.lower():
            continue

        matched.append({
            "rule_id": rule.get("id", ""),
            "name": rule.get("name", "Unnamed rule"),
            "action": rule.get("action", "notify"),  # notify, run_command, send_message
            "action_data": rule.get("action_data", ""),
            "priority": rule.get("priority", "normal"),
        })

    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Session Tags
# ═══════════════════════════════════════════════════════════════════════════════

def _tags_path(workspace: Path) -> Path:
    return workspace / "session_tags.json"


def get_session_tags(workspace: Path) -> dict[str, list[str]]:
    """Get all session → tags mapping."""
    path = _tags_path(workspace)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def set_session_tags(workspace: Path, session_key: str, tags: list[str]) -> None:
    """Set tags for a session."""
    all_tags = get_session_tags(workspace)
    all_tags[session_key] = tags
    _tags_path(workspace).write_text(json.dumps(all_tags, indent=2))


def get_sessions_by_tag(workspace: Path, tag: str) -> list[str]:
    """Find sessions with a specific tag."""
    all_tags = get_session_tags(workspace)
    return [key for key, tags in all_tags.items() if tag in tags]


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Tool Favorites
# ═══════════════════════════════════════════════════════════════════════════════

def get_tool_favorites(workspace: Path, top_n: int = 10) -> list[dict]:
    """Get most-used tools from tool_scores.json. No LLM."""
    scores_file = workspace / "memory" / "tool_scores.json"
    if not scores_file.exists():
        return []

    try:
        scores = json.loads(scores_file.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    tools = []
    for name, data in scores.items():
        if not isinstance(data, dict):
            continue
        total = data.get("total", 0)
        success = data.get("success", 0)
        if total >= 2:
            tools.append({
                "name": name,
                "total": total,
                "success_rate": round((success / total) * 100, 0) if total > 0 else 0,
                "avg_ms": round(data.get("total_ms", 0) / total, 0) if total > 0 else 0,
            })

    tools.sort(key=lambda t: t["total"], reverse=True)
    return tools[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(workspace: Path) -> list[dict]:
    """Detect usage anomalies by comparing today vs. 7-day average. No LLM."""
    from nanobot.hooks.builtin.usage_tracker import get_daily_totals

    today = date.today()
    today_totals = get_daily_totals(workspace, today)

    # Compute 7-day averages (excluding today)
    week_totals = {"tokens": [], "cost": [], "turns": []}
    for i in range(1, 8):
        day = today - timedelta(days=i)
        totals = get_daily_totals(workspace, day)
        week_totals["tokens"].append(totals.get("total_tokens", 0))
        week_totals["cost"].append(totals.get("cost", 0.0))
        week_totals["turns"].append(totals.get("turns", 0))

    anomalies = []
    for metric, values, today_val in [
        ("tokens", week_totals["tokens"], today_totals.get("total_tokens", 0)),
        ("cost", week_totals["cost"], today_totals.get("cost", 0.0)),
        ("turns", week_totals["turns"], today_totals.get("turns", 0)),
    ]:
        avg = sum(values) / len(values) if values else 0
        if avg > 0 and today_val > avg * 3:
            anomalies.append({
                "metric": metric,
                "today": round(today_val, 4) if isinstance(today_val, float) else today_val,
                "avg_7d": round(avg, 4) if isinstance(avg, float) else int(avg),
                "ratio": round(today_val / avg, 1),
                "severity": "high" if today_val > avg * 5 else "medium",
            })

    return anomalies


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Batch File Processing
# ═══════════════════════════════════════════════════════════════════════════════

async def batch_process_inbox(workspace: Path, action: str = "list") -> dict:
    """Process multiple files in the inbox at once. No LLM.

    Actions: list, count, delete_old (>30d), categorize
    """
    inbox_dir = workspace / "inbox"
    if not inbox_dir.exists():
        return {"files": [], "count": 0}

    files = []
    for f in sorted(inbox_dir.rglob("*")):
        if not f.is_file():
            continue
        stat = f.stat()
        files.append({
            "path": str(f.relative_to(workspace)),
            "name": f.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "age_days": (time.time() - stat.st_mtime) / 86400,
            "ext": f.suffix.lower(),
        })

    if action == "count":
        by_ext: dict[str, int] = defaultdict(int)
        for f in files:
            by_ext[f["ext"]] += 1
        return {"count": len(files), "by_extension": dict(by_ext)}

    if action == "delete_old":
        deleted = 0
        for f in files:
            if f["age_days"] > 30:
                try:
                    (inbox_dir / f["path"].replace("inbox/", "")).unlink()
                    deleted += 1
                except OSError:
                    pass
        return {"deleted": deleted, "remaining": len(files) - deleted}

    if action == "categorize":
        categories: dict[str, list[str]] = {
            "documents": [], "images": [], "data": [], "code": [], "other": [],
        }
        ext_map = {
            ".pdf": "documents", ".docx": "documents", ".txt": "documents", ".md": "documents",
            ".png": "images", ".jpg": "images", ".jpeg": "images", ".gif": "images",
            ".csv": "data", ".xlsx": "data", ".json": "data", ".xml": "data",
            ".py": "code", ".js": "code", ".ts": "code", ".sh": "code",
        }
        for f in files:
            cat = ext_map.get(f["ext"], "other")
            categories[cat].append(f["name"])
        return {"categories": categories}

    return {"files": files[:50], "count": len(files)}


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Schedule Templates
# ═══════════════════════════════════════════════════════════════════════════════

SCHEDULE_TEMPLATES = [
    {
        "id": "daily_email",
        "name": "Daily Email Check",
        "description": "Check email inbox every morning at 9am",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "message": "Check my email inbox and summarize what's new",
    },
    {
        "id": "weekly_report",
        "name": "Weekly Status Report",
        "description": "Generate a weekly summary every Friday at 5pm",
        "schedule": {"kind": "cron", "expr": "0 17 * * 5"},
        "message": "Generate a weekly status report: summarize goals progress, emails, and calendar for this week",
    },
    {
        "id": "daily_standup",
        "name": "Daily Standup Prep",
        "description": "Prepare standup notes every weekday at 8:30am",
        "schedule": {"kind": "cron", "expr": "30 8 * * 1-5"},
        "message": "Prepare my standup: what did I do yesterday, what's on my calendar today, any blocked goals?",
    },
    {
        "id": "goal_review",
        "name": "Goal Review",
        "description": "Review and update goals every Sunday evening",
        "schedule": {"kind": "cron", "expr": "0 19 * * 0"},
        "message": "Review all my goals: what's overdue, what's progressing, any new goals I should add?",
    },
    {
        "id": "morning_briefing",
        "name": "Morning Briefing",
        "description": "Full morning briefing at 7:30am — weather, calendar, emails, news",
        "schedule": {"kind": "cron", "expr": "30 7 * * *"},
        "message": "Morning briefing: weather, today's calendar, important emails, and any tech news",
    },
    {
        "id": "bill_reminder",
        "name": "Bill Payment Reminder",
        "description": "Check for upcoming bills every 1st and 15th",
        "schedule": {"kind": "cron", "expr": "0 10 1,15 * *"},
        "message": "Check for any upcoming bills or payments due in the next 7 days",
    },
    {
        "id": "cleanup",
        "name": "Workspace Cleanup",
        "description": "Clean up old files and sessions monthly",
        "schedule": {"kind": "cron", "expr": "0 3 1 * *"},
        "message": "Clean up: delete generated files older than 30 days, archive old sessions",
    },
]


def get_schedule_templates() -> list[dict]:
    """Return available schedule templates."""
    return SCHEDULE_TEMPLATES
