"""External event ingestion — processes webhooks from external services.

Supports event types: email, calendar, file, custom webhook.
Events are routed to the agent via the message bus as system messages.

Security: HMAC-SHA256 signature verification via X-Nanobot-Signature header.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime
from typing import Any

from loguru import logger

# Event type definitions
VALID_EVENT_TYPES = frozenset({
    "email",       # New email received
    "calendar",    # Calendar event (created, updated, starting soon)
    "file",        # File created/modified in watched directory
    "webhook",     # Generic webhook from IFTTT, Zapier, etc.
    "reminder",    # Scheduled reminder triggered
    "alert",       # System alert (billing, service status)
})

VALID_PRIORITIES = frozenset({"urgent", "high", "normal", "low", "background"})


def validate_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature."""
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def parse_event(body: dict[str, Any]) -> dict[str, Any] | str:
    """Parse and validate an incoming event. Returns parsed event or error string."""
    event_type = body.get("type")
    if not event_type or event_type not in VALID_EVENT_TYPES:
        return f"Invalid event type: {event_type}. Valid: {', '.join(sorted(VALID_EVENT_TYPES))}"

    priority = body.get("priority", "normal")
    if priority not in VALID_PRIORITIES:
        priority = "normal"

    title = body.get("title", "").strip()
    if not title:
        return "Missing required field: title"

    content = body.get("content", "").strip()
    source = body.get("source", "external").strip()
    metadata = body.get("metadata", {})

    return {
        "type": event_type,
        "priority": priority,
        "title": title,
        "content": content,
        "source": source,
        "metadata": metadata,
        "received_at": datetime.now().isoformat(),
    }


def format_event_as_message(event: dict[str, Any]) -> str:
    """Format a parsed event as a natural-language message for the agent."""
    priority_prefix = ""
    if event["priority"] in ("urgent", "high"):
        priority_prefix = f"[{event['priority'].upper()}] "

    parts = [f"{priority_prefix}External event ({event['type']}): {event['title']}"]

    if event["content"]:
        parts.append(event["content"])

    if event["source"] != "external":
        parts.append(f"Source: {event['source']}")

    if event["metadata"]:
        meta_str = ", ".join(f"{k}={v}" for k, v in event["metadata"].items() if v)
        if meta_str:
            parts.append(f"Details: {meta_str}")

    return "\n\n".join(parts)
