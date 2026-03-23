"""Token cost budgets with alerts.

Budget config stored at <workspace>/usage/budget.json.
Uses usage_tracker.get_daily_totals for actual spending data.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

_DEFAULTS = {
    "daily_limit": 5.0,
    "weekly_limit": 25.0,
    "auto_switch_model": None,
    "alert_threshold": 0.8,
}


def load_budget(workspace: Path) -> dict:
    """Read budget config, filling in defaults for missing keys."""
    path = workspace / "usage" / "budget.json"
    budget = dict(_DEFAULTS)
    if path.exists():
        try:
            stored = json.loads(path.read_text())
            budget.update(stored)
        except (json.JSONDecodeError, OSError):
            pass
    return budget


def save_budget(workspace: Path, budget: dict) -> None:
    """Write budget config to disk."""
    usage_dir = workspace / "usage"
    usage_dir.mkdir(parents=True, exist_ok=True)
    path = usage_dir / "budget.json"
    path.write_text(json.dumps(budget, indent=2))


def check_budget(workspace: Path) -> dict:
    """Check current spending against budget limits."""
    from nanobot.hooks.builtin.usage_tracker import get_daily_totals

    budget = load_budget(workspace)
    today = date.today()

    # Daily
    daily = get_daily_totals(workspace, today)
    daily_used = daily.get("cost", 0.0)
    daily_limit = budget["daily_limit"]
    daily_pct = (daily_used / daily_limit) if daily_limit > 0 else 0.0

    # Weekly (last 7 days)
    weekly_used = 0.0
    for i in range(7):
        day = today - timedelta(days=i)
        weekly_used += get_daily_totals(workspace, day).get("cost", 0.0)
    weekly_limit = budget["weekly_limit"]
    weekly_pct = (weekly_used / weekly_limit) if weekly_limit > 0 else 0.0

    threshold = budget["alert_threshold"]
    alert = daily_pct >= threshold or weekly_pct >= threshold
    exceeded = daily_pct >= 1.0 or weekly_pct >= 1.0

    return {
        "daily_used": round(daily_used, 4),
        "daily_limit": daily_limit,
        "daily_pct": round(daily_pct, 3),
        "weekly_used": round(weekly_used, 4),
        "weekly_limit": weekly_limit,
        "weekly_pct": round(weekly_pct, 3),
        "alert": alert,
        "exceeded": exceeded,
        "suggested_model": budget["auto_switch_model"] if exceeded else None,
    }
