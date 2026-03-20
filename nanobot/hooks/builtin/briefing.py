"""Morning Briefing — proactive context-aware summaries.

Generates time-aware briefings by checking the user's tools (email, calendar,
weather) and memory state. Delivers to the user's preferred channel.

Briefing types:
  Morning (7-10 AM):  Full briefing — email, calendar, weather, pending goals, financial alerts
  Pre-meeting (15m):  Meeting prep — agenda, relevant context from memory
  Evening (6-8 PM):   Day summary — what happened, what's pending
  Weekly (Sunday):    Week review — patterns, achievements, upcoming deadlines
"""

from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_BRIEFING_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "compose_briefing",
            "description": "Compose a briefing based on available context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "briefing": {
                        "type": "string",
                        "description": "The formatted briefing text. Conversational tone, 2-4 paragraphs max.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "normal", "low"],
                        "description": "Urgency level. High = financial alerts, meeting in <1h. Normal = routine. Low = FYI.",
                    },
                    "skip": {
                        "type": "boolean",
                        "description": "True if there's nothing worth reporting right now.",
                    },
                },
                "required": ["briefing", "skip"],
            },
        },
    }
]


def _get_briefing_type(hour: int, weekday: int) -> str | None:
    """Determine which briefing type to generate based on time."""
    if 7 <= hour <= 9:
        return "morning"
    if 18 <= hour <= 20:
        return "evening"
    if weekday == 6 and 10 <= hour <= 14:  # Sunday
        return "weekly"
    return None


_BRIEFING_PROMPTS = {
    "morning": """Generate a morning briefing for the user. You have access to their memory context below.

Include (if relevant data exists):
- Urgent items: overdue bills, expiring subscriptions, approaching deadlines
- Today's schedule: meetings, appointments, tasks
- Recent context: what happened yesterday, ongoing projects
- Weather/travel: if there's a trip coming up in the next 3 days

Keep it conversational, warm, and concise. 2-3 paragraphs max.
Start with "Good morning" and the user's name if known.""",

    "evening": """Generate an evening summary for the user.

Include:
- What was accomplished today (from SHORT_TERM.md)
- Anything still pending
- Tomorrow's key items if known

Keep it brief and encouraging. 1-2 paragraphs.""",

    "weekly": """Generate a weekly review for the user.

Include:
- Key accomplishments this week
- Patterns observed (from OBSERVATIONS.md)
- Upcoming week's important dates
- Any recurring issues or themes

Keep it reflective and forward-looking. 2-3 paragraphs.""",
}


class BriefingEngine:
    """Generates and delivers context-aware briefings."""

    def __init__(self, workspace: Path, provider: LLMProvider, model: str):
        self._workspace = workspace
        self._provider = provider
        self._model = model
        self._state_file = workspace / "memory" / ".briefing_state.json"
        self._state = self._load_state()

    def _load_state(self) -> dict:
        try:
            if self._state_file.exists():
                raw = self._state_file.read_text(encoding="utf-8")
                if isinstance(raw, str) and raw.strip():
                    return json.loads(raw)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return {"last_morning": "", "last_evening": "", "last_weekly": ""}

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(self._state), encoding="utf-8")
        except OSError:
            pass

    def _already_delivered(self, briefing_type: str) -> bool:
        """Check if this briefing type was already delivered today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self._state.get(f"last_{briefing_type}", "") == today

    def _mark_delivered(self, briefing_type: str) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        self._state[f"last_{briefing_type}"] = today
        self._save_state()

    def _gather_context(self) -> str:
        """Gather all available memory context for the briefing."""
        mem_dir = self._workspace / "memory"
        parts = []

        for fname, label in [
            ("SHORT_TERM.md", "Today's Context"),
            ("LONG_TERM.md", "User Profile & Facts"),
            ("OBSERVATIONS.md", "Behavior Patterns"),
            ("GOALS.md", "Active Goals"),
            ("LEARNINGS.md", "Learned Rules"),
        ]:
            path = mem_dir / fname
            if path.exists():
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    parts.append(f"## {label}\n{content}")

        return "\n\n".join(parts) if parts else "(no memory context available)"

    async def maybe_generate(self) -> tuple[str, str] | None:
        """Check if a briefing is due and generate it.

        Returns (briefing_text, priority) or None if nothing to report.
        """
        now = datetime.now()
        briefing_type = _get_briefing_type(now.hour, now.weekday())

        if briefing_type is None:
            return None

        if self._already_delivered(briefing_type):
            return None

        logger.info("Briefing: generating {} briefing", briefing_type)

        context = self._gather_context()
        prompt_template = _BRIEFING_PROMPTS[briefing_type]

        prompt = f"""{prompt_template}

## Current Time
{now.strftime("%A, %B %d, %Y at %I:%M %p")}

## Memory Context
{context}

Call compose_briefing with your briefing. Set skip=true if there's genuinely nothing to report."""

        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a personal briefing agent. Compose concise, warm, actionable briefings."},
                    {"role": "user", "content": prompt},
                ],
                tools=_BRIEFING_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "compose_briefing"}},
            )

            if not response.has_tool_calls:
                return None

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            if args.get("skip"):
                logger.info("Briefing: {} skipped (nothing to report)", briefing_type)
                self._mark_delivered(briefing_type)
                return None

            briefing = args.get("briefing", "").strip()
            if not briefing:
                return None

            priority = args.get("priority", "normal")
            self._mark_delivered(briefing_type)

            # Also save to SHORT_TERM for context
            from nanobot.agent.memory import MemoryStore
            store = MemoryStore(self._workspace)
            store.append_short_term(f"[Briefing delivered: {briefing_type}]")

            logger.info("Briefing: {} delivered (priority={})", briefing_type, priority)
            return briefing, priority

        except Exception:
            logger.opt(exception=True).warning("Briefing generation failed")
            return None
