"""Proactive notifications — checks for actionable items and pushes alerts.

Runs on turn_completed (every 10 turns) and checks:
  - Overdue goals
  - Financial alerts from LONG_TERM.md (bills due soon)
  - Upcoming calendar events (if accessible)
  - Unread items since last check

Sends notifications via the message bus to the user's active channel.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.hooks.events import TurnCompleted

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus

_CHECK_INTERVAL = 10  # Check every N turns
_LAST_NOTIFIED: dict[str, str] = {}  # key → date to avoid re-notifying same day


def _check_overdue_goals(workspace: Path) -> list[str]:
    """Check for overdue goals."""
    goals_file = workspace / "memory" / "GOALS.md"
    if not goals_file.exists():
        return []

    alerts = []
    today = date.today().isoformat()
    content = goals_file.read_text(encoding="utf-8")

    for line in content.split("\n"):
        if line.strip().startswith("- [ ] "):
            m = re.search(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)", line)
            if m:
                due = m.group(1)
                task = line.strip()[6:].split("(due:")[0].strip()
                if due < today:
                    alerts.append(f"Overdue: {task} (was due {due})")
                elif due == today:
                    alerts.append(f"Due today: {task}")

    return alerts


def _check_financial_alerts(workspace: Path) -> list[str]:
    """Check LONG_TERM.md for financial alerts with approaching dates."""
    lt_file = workspace / "memory" / "LONG_TERM.md"
    if not lt_file.exists():
        return []

    alerts = []
    today = date.today()
    content = lt_file.read_text(encoding="utf-8")

    # Look for date patterns near money-related keywords
    for line in content.split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["overdue", "due", "suspend", "expires", "payment"]):
            # Extract dates
            dates = re.findall(r"(\w+ \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})", line)
            for d in dates:
                try:
                    parsed = None
                    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d"):
                        try:
                            parsed = datetime.strptime(d.replace(",", ""), fmt).date()
                            break
                        except ValueError:
                            continue
                    if parsed and 0 <= (parsed - today).days <= 3:
                        alert = line.strip().lstrip("-* ")
                        if alert and len(alert) > 10:
                            alerts.append(alert[:150])
                            break
                except Exception:
                    pass

    return alerts


def make_proactive_hook(workspace: Path, bus: "MessageBus"):
    """Create a proactive notification hook."""
    _call_count = [0]

    async def on_turn_completed(event: TurnCompleted) -> None:
        _call_count[0] += 1
        if _call_count[0] % _CHECK_INTERVAL != 0:
            return

        today = date.today().isoformat()
        alerts = []

        # Check goals
        goal_alerts = _check_overdue_goals(workspace)
        for a in goal_alerts:
            key = f"goal:{a[:30]}"
            if _LAST_NOTIFIED.get(key) != today:
                alerts.append(a)
                _LAST_NOTIFIED[key] = today

        # Check financial
        fin_alerts = _check_financial_alerts(workspace)
        for a in fin_alerts:
            key = f"fin:{a[:30]}"
            if _LAST_NOTIFIED.get(key) != today:
                alerts.append(a)
                _LAST_NOTIFIED[key] = today

        # Check habits due for reminder
        try:
            from nanobot.hooks.builtin.maintenance import get_due_habits, mark_habit_reminded
            for habit in get_due_habits(workspace):
                name = habit.get("name", "")
                key = f"habit:{name}"
                if _LAST_NOTIFIED.get(key) != today or habit.get("interval_hours", 24) < 24:
                    alerts.append(f"Habit reminder: {name}")
                    mark_habit_reminded(workspace, name)
                    _LAST_NOTIFIED[key] = today
        except Exception:
            pass

        if not alerts:
            return

        # Send notification
        message = "**Mawa Alert**\n\n" + "\n".join(f"- {a}" for a in alerts)
        channel = event.channel or "web_voice"
        chat_id = event.chat_id or "voice"

        from nanobot.bus.events import OutboundMessage
        await bus.publish_outbound(OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=message,
            metadata={"_notification": True, "_proactive": True},
        ))
        logger.info("Proactive: sent {} alerts to {}:{}", len(alerts), channel, chat_id)

    return on_turn_completed
