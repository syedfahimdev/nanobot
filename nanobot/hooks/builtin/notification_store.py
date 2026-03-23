"""Notification persistence — stores notifications so they're not lost when offline.

When a notification is sent but no WebSocket client is connected,
it's saved to a file. On next connect, pending notifications are delivered.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

_MAX_STORED = 20
_MAX_AGE_HOURS = 48


def _store_path(workspace: Path) -> Path:
    return workspace / "notifications.json"


def save_notification(workspace: Path, content: str, metadata: dict[str, Any] | None = None) -> None:
    """Save a notification to the persistent store."""
    path = _store_path(workspace)
    notifications = _load(path)

    notifications.append({
        "content": content,
        "metadata": metadata or {},
        "ts": time.time(),
        "read": False,
    })

    # Cap and age-filter
    cutoff = time.time() - (_MAX_AGE_HOURS * 3600)
    notifications = [n for n in notifications if n["ts"] > cutoff][-_MAX_STORED:]

    path.write_text(json.dumps(notifications, indent=2), encoding="utf-8")


def get_pending(workspace: Path) -> list[dict]:
    """Get unread notifications."""
    path = _store_path(workspace)
    notifications = _load(path)
    return [n for n in notifications if not n.get("read")]


def get_all(workspace: Path) -> list[dict]:
    """Get all notifications (for the UI)."""
    path = _store_path(workspace)
    return _load(path)


def mark_all_read(workspace: Path) -> int:
    """Mark all notifications as read. Returns count marked."""
    path = _store_path(workspace)
    notifications = _load(path)
    count = 0
    for n in notifications:
        if not n.get("read"):
            n["read"] = True
            count += 1
    path.write_text(json.dumps(notifications, indent=2), encoding="utf-8")
    return count


def _load(path: Path) -> list[dict]:
    """Load notifications from disk."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
