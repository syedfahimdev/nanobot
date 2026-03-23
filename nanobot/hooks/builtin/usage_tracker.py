"""Token usage tracker — persists per-turn token counts to daily JSON files.

Usage data is stored at <workspace>/usage/YYYY-MM-DD.json as a list of turn
entries. The /api/usage endpoint reads these files for dashboard display.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

# Per-million-token pricing: (input_cost, output_cost)
# Update these when pricing changes or new models are added.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-5":        (15.0, 75.0),
    "claude-sonnet-4-5":      (3.0, 15.0),
    "claude-sonnet-4-6":      (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-3-5":       (0.80, 4.0),
    # Kimi
    "kimi-for-coding":        (0.0, 0.0),  # Free during beta
    # OpenAI
    "gpt-4o":                 (2.50, 10.0),
    "gpt-4o-mini":            (0.15, 0.60),
    "gpt-4-turbo":            (10.0, 30.0),
    "o1":                     (15.0, 60.0),
    "o1-mini":                (3.0, 12.0),
    "o3-mini":                (1.10, 4.40),
    # Google
    "gemini-2.0-flash":       (0.10, 0.40),
    "gemini-1.5-pro":         (1.25, 5.0),
    "gemini-1.5-flash":       (0.075, 0.30),
    # Groq (hosted)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant":    (0.05, 0.08),
    # DeepSeek
    "deepseek-chat":          (0.27, 1.10),
    "deepseek-reasoner":      (0.55, 2.19),
}


def estimate_cost(usage: dict, model: str = "") -> float:
    """Estimate cost in USD from token usage and model name.

    Strips provider prefixes (e.g. "anthropic/claude-opus-4-5" → "claude-opus-4-5")
    and does substring matching against known models.
    """
    # Strip provider prefix
    bare = model.split("/", 1)[-1] if "/" in model else model

    # Exact match first, then substring
    pricing = MODEL_PRICING.get(bare)
    if not pricing:
        for key, val in MODEL_PRICING.items():
            if key in bare or bare in key:
                pricing = val
                break

    if not pricing:
        return 0.0

    input_cost, output_cost = pricing
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    return (prompt * input_cost + completion * output_cost) / 1_000_000


async def record_usage(
    workspace: Path, usage: dict, channel: str = "", model: str = "",
) -> None:
    """Append a usage record to today's file."""
    usage_dir = workspace / "usage"
    usage_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    path = usage_dir / f"{today}.json"

    entries: list[dict] = []
    if path.exists():
        try:
            entries = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            entries = []

    cost = estimate_cost(usage, model)
    entries.append({
        "ts": time.time(),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost": round(cost, 6),
        "model": model,
        "channel": channel,
    })

    try:
        path.write_text(json.dumps(entries))
    except OSError as e:
        logger.warning("Failed to write usage: {}", e)


def get_daily_totals(workspace: Path, day: date) -> dict:
    """Get aggregated totals for a single day."""
    path = workspace / "usage" / f"{day.isoformat()}.json"
    if not path.exists():
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "turns": 0, "cost": 0.0}

    try:
        entries = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "turns": 0, "cost": 0.0}

    totals: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "turns": len(entries), "cost": 0.0}
    for e in entries:
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[k] += e.get(k, 0)
        totals["cost"] += e.get("cost", 0.0)
    totals["cost"] = round(totals["cost"], 4)
    return totals


def get_usage_summary(workspace: Path) -> dict:
    """Build a usage summary for the dashboard API."""
    today = date.today()
    today_totals = get_daily_totals(workspace, today)

    # 7-day history
    history = []
    week_total = 0
    week_cost = 0.0
    for i in range(7):
        day = today - timedelta(days=i)
        totals = get_daily_totals(workspace, day)
        history.append({"date": day.isoformat(), **totals})
        week_total += totals["total_tokens"]
        week_cost += totals["cost"]

    # Today's per-turn entries (for the detail view)
    turns_today: list[dict] = []
    path = workspace / "usage" / f"{today.isoformat()}.json"
    if path.exists():
        try:
            turns_today = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "today": today_totals,
        "week_total": week_total,
        "week_cost": round(week_cost, 4),
        "history": history,
        "turns_today": turns_today[-20:],  # Last 20 turns
    }
