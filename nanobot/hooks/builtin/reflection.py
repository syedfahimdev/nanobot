"""Reflection hook — learns from user corrections to improve behavior.

Detects when a user corrects the agent (retries, "no not that", "wrong")
and extracts behavioral rules into memory/LEARNINGS.md. These learnings
are injected into the system prompt to prevent repeated mistakes.

Two-phase approach:
  1. Detection: pure string matching on user messages (every turn, ~0 cost)
  2. Extraction: LLM call to extract the lesson (only when correction detected)
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.hooks.events import TurnCompleted

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

# Correction signal patterns — weighted by strength
_CORRECTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    # Strong corrections
    (re.compile(r"\bno[,.]?\s+(not|don'?t|never|stop)\b", re.I), 0.9),
    (re.compile(r"\bthat'?s (wrong|incorrect|not right|not what)\b", re.I), 0.9),
    (re.compile(r"\bnot what I\b", re.I), 0.9),
    (re.compile(r"\bI already (told|said|mentioned)\b", re.I), 0.8),
    (re.compile(r"\btry again\b", re.I), 0.8),
    (re.compile(r"\bstop (doing|adding|using)\b", re.I), 0.8),
    (re.compile(r"\bI (said|meant|asked|wanted)\b", re.I), 0.7),
    (re.compile(r"\bdon'?t (do|use|add|include|put)\b", re.I), 0.7),
    (re.compile(r"\bwrong\b", re.I), 0.6),
    # Softer hints — "do this instead", "remember", "next time", "use X not Y"
    (re.compile(r"\binstead\s+(?:of|do|use|try)\b", re.I), 0.6),
    (re.compile(r"\bdo this\b", re.I), 0.5),
    (re.compile(r"\bnext time\b", re.I), 0.6),
    (re.compile(r"\bremember\s+(?:to|that|this|for)\b", re.I), 0.6),
    (re.compile(r"\buse\s+\w+\s+(?:not|instead|rather)\b", re.I), 0.6),
    (re.compile(r"\blike this\b", re.I), 0.4),
    (re.compile(r"\bnot like\s+(?:this|that)\b", re.I), 0.6),
    (re.compile(r"\bshould(?:n'?t| not)\b", re.I), 0.5),
    (re.compile(r"\bprefer\s+\w+\s+over\b", re.I), 0.5),
    (re.compile(r"\balways\s+(?:do|use|try)\b", re.I), 0.5),
    (re.compile(r"\bnever\s+(?:do|use|add)\b", re.I), 0.7),
    (re.compile(r"\bplease (just|actually|instead)\b", re.I), 0.5),
    (re.compile(r"\bno,\s", re.I), 0.5),
    # Implicit hints — sharing preferences
    (re.compile(r"\bI (?:like|prefer|want) (?:it |you to )", re.I), 0.4),
    (re.compile(r"\bfor (?:future|next|later)\b", re.I), 0.5),
    (re.compile(r"\bkeep (?:in mind|that)\b", re.I), 0.5),
]

_CORRECTION_THRESHOLD = 0.4  # Lowered — catch softer hints too
_MAX_LEARNINGS = 20  # Keep file from growing too large

_EXTRACT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_learning",
            "description": "Save a behavioral rule learned from the user's correction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule": {
                        "type": "string",
                        "description": "A concise behavioral rule in imperative form. "
                        "Example: 'Don't add type annotations to code unless asked.' "
                        "Start with what to do or not do.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Brief context — what triggered this learning. One sentence.",
                    },
                    "skip": {
                        "type": "boolean",
                        "description": "Set true if this is NOT actually a correction (just a normal follow-up).",
                    },
                },
                "required": ["rule", "skip"],
            },
        },
    }
]


def _score_correction(text: str) -> float:
    """Score how likely a user message is a correction (0.0 - 1.0)."""
    score = 0.0
    for pattern, weight in _CORRECTION_PATTERNS:
        if pattern.search(text):
            score = max(score, weight)
    return score


class ReflectionEngine:
    """Detects user corrections and extracts behavioral rules."""

    def __init__(self, workspace: Path, provider: LLMProvider, model: str):
        self._workspace = workspace
        self._provider = provider
        self._model = model
        self._learnings_file = workspace / "memory" / "LEARNINGS.md"
        # Track last turn per session for comparison
        self._last_assistant_response: dict[str, str] = {}
        self._last_user_message: dict[str, str] = {}
        self._pending_check: dict[str, tuple[str, str]] = {}  # session → (user_msg, assistant_resp)

    def on_turn_completed(self, session_key: str, user_message: str, assistant_response: str) -> None:
        """Record a completed turn for reflection on the next user message."""
        self._last_assistant_response[session_key] = assistant_response
        self._last_user_message[session_key] = user_message

    async def check_for_correction(self, session_key: str, new_user_message: str) -> None:
        """Check if new_user_message is correcting the previous turn."""
        prev_response = self._last_assistant_response.get(session_key)
        if not prev_response:
            return

        score = _score_correction(new_user_message)
        if score < _CORRECTION_THRESHOLD:
            return

        logger.info(
            "Reflection: correction detected (score={:.1f}) in session {}",
            score, session_key,
        )

        await self._extract_learning(
            user_correction=new_user_message,
            assistant_response=prev_response[:500],  # Truncate to save tokens
            prev_user_message=self._last_user_message.get(session_key, "")[:200],
        )

    async def _extract_learning(
        self,
        user_correction: str,
        assistant_response: str,
        prev_user_message: str,
    ) -> None:
        """Use LLM to extract a behavioral rule from the correction."""
        prompt = f"""The user corrected the assistant. Extract a behavioral rule to prevent this mistake in the future.

## Previous user message
{prev_user_message}

## Assistant's response (that was corrected)
{assistant_response}

## User's correction
{user_correction}

Call save_learning with the rule. If this isn't actually a correction (just a normal follow-up), set skip=true."""

        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "Extract a concise behavioral rule from user corrections. Be specific and actionable."},
                    {"role": "user", "content": prompt},
                ],
                tools=_EXTRACT_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "save_learning"}},
            )

            if not response.has_tool_calls:
                return

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                return

            if args.get("skip"):
                logger.debug("Reflection: LLM determined this is not a correction")
                return

            rule = args.get("rule", "").strip()
            if not rule:
                return

            context = args.get("context", "").strip()
            self._save_learning(rule, context)

        except Exception:
            logger.opt(exception=True).debug("Reflection extraction failed")

    def _save_learning(self, rule: str, context: str) -> None:
        """Append a learning to LEARNINGS.md, keeping max size."""
        self._learnings_file.parent.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"- {rule}"
        if context:
            entry += f" _{context}_"
        entry += f" [{ts}]"

        # Read existing
        existing: list[str] = []
        if self._learnings_file.exists():
            content = self._learnings_file.read_text(encoding="utf-8")
            existing = [l for l in content.split("\n") if l.strip().startswith("- ")]

        # Dedup: skip if a very similar rule exists
        rule_lower = rule.lower()
        for ex in existing:
            # Simple overlap check — if 60%+ of words match, skip
            ex_words = set(ex.lower().split())
            rule_words = set(rule_lower.split())
            if rule_words and len(rule_words & ex_words) / len(rule_words) > 0.6:
                logger.debug("Reflection: skipping duplicate learning")
                return

        existing.append(entry)

        # Keep only the most recent learnings
        if len(existing) > _MAX_LEARNINGS:
            existing = existing[-_MAX_LEARNINGS:]

        content = "# Learned Rules\n\n" + "\n".join(existing) + "\n"
        self._learnings_file.write_text(content, encoding="utf-8")
        logger.info("Reflection: saved learning — {}", rule[:80])

    def get_learnings_context(self) -> str:
        """Return learnings for injection into system prompt."""
        if not self._learnings_file.exists():
            return ""
        content = self._learnings_file.read_text(encoding="utf-8")
        lines = [l for l in content.split("\n") if l.strip().startswith("- ")]
        if not lines:
            return ""
        # Only inject last 10 learnings to keep tokens low
        top = "\n".join(lines[-10:])
        return f"## Learned Rules\n{top}"
