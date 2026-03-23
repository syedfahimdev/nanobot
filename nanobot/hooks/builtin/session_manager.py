"""Browse and manage past sessions.

Provides listing, loading, exporting, and deleting of session JSONL files
stored at <workspace>/sessions/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def list_sessions(workspace: Path, limit: int = 50) -> list[dict]:
    """List session files sorted by mtime desc."""
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        return []

    results: list[dict] = []
    paths = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)

    for path in paths[:limit]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue

            # Parse metadata from first line
            first = json.loads(lines[0].strip())
            key = first.get("key", path.stem) if first.get("_type") == "metadata" else path.stem

            # Count actual messages (skip metadata line)
            messages = [l for l in lines[1:] if l.strip()]
            msg_count = len(messages)

            # Preview: first user message
            preview = ""
            for raw in messages:
                try:
                    msg = json.loads(raw)
                    if msg.get("role") == "user" and msg.get("content"):
                        preview = msg["content"][:120]
                        break
                except json.JSONDecodeError:
                    continue

            results.append({
                "key": key,
                "message_count": msg_count,
                "last_active": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "preview": preview,
            })
        except (json.JSONDecodeError, OSError):
            continue

    return results


def get_session(workspace: Path, key: str) -> dict | None:
    """Load a session by key, return {key, messages}."""
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        return None

    # Try exact match, then with : → _ replacement, then scan stems
    from nanobot.utils.helpers import safe_filename
    candidates = [
        sessions_dir / f"{key}.jsonl",
        sessions_dir / f"{key.replace(':', '_')}.jsonl",
        sessions_dir / f"{safe_filename(key.replace(':', '_'))}.jsonl",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    # Fallback: scan for partial match
    if not path:
        for p in sessions_dir.glob("*.jsonl"):
            if key.replace(":", "_") in p.stem or p.stem in key.replace(":", "_"):
                path = p
                break
    if not path:
        return None

    try:
        messages: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("_type") == "metadata":
                continue
            messages.append({
                "role": data.get("role") or "",
                "content": data.get("content") or "",
                "timestamp": data.get("timestamp") or "",
            })
        return {"key": key, "messages": messages}
    except (json.JSONDecodeError, OSError):
        return None


def export_session(workspace: Path, key: str, fmt: str = "markdown") -> str:
    """Export a session as markdown or JSON string."""
    session = get_session(workspace, key)
    if not session:
        return ""

    if fmt == "json":
        return json.dumps(session, indent=2, ensure_ascii=False)

    # Markdown format
    lines = [f"# Session: {key}\n"]
    for msg in session["messages"]:
        role = msg["role"].capitalize()
        ts = msg.get("timestamp", "")
        lines.append(f"### {role}" + (f" ({ts})" if ts else ""))
        lines.append((msg.get("content") or "") + "\n")

    return "\n".join(lines)


def delete_session(workspace: Path, key: str) -> bool:
    """Delete a session file. Returns True if deleted."""
    sessions_dir = workspace / "sessions"
    candidates = [
        sessions_dir / f"{key}.jsonl",
        sessions_dir / f"{key.replace(':', '_')}.jsonl",
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if not path:
        for p in sessions_dir.glob("*.jsonl"):
            if key.replace(":", "_") in p.stem:
                path = p
                break

    if path and path.exists():
        try:
            path.unlink()
            return True
        except OSError:
            return False
    return False
