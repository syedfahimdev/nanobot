"""Reaction Feedback — emoji reactions drive self-improvement.

When users react to messages:
  👍 = "this worked well" → save positive pattern to SOUL reinforcement
  👎 = "this was wrong" → extract correction, save to LEARNINGS.md
  ❤️ = "loved this" → reinforce this style/approach strongly
  💡 = "interesting" → save as notable insight

Reactions are saved to memory/FEEDBACK.md and processed:
- Positive reactions reinforce behaviors in the agent file
- Negative reactions trigger reflection to extract a lesson
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_EXTRACT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_feedback_lesson",
            "description": "Save a lesson learned from the user's reaction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lesson": {
                        "type": "string",
                        "description": "A concise behavioral rule. For positive: 'Keep doing X when Y.' For negative: 'Don't do X, instead do Y.'",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["style", "accuracy", "tool_usage", "tone", "format"],
                        "description": "What aspect of behavior this feedback is about.",
                    },
                },
                "required": ["lesson", "category"],
            },
        },
    }
]

# Reaction meanings
REACTION_MEANINGS = {
    "👍": {"signal": "positive", "weight": 1, "label": "approved"},
    "👎": {"signal": "negative", "weight": 2, "label": "disapproved"},
    "❤️": {"signal": "positive", "weight": 3, "label": "loved"},
    "💡": {"signal": "neutral", "weight": 1, "label": "insightful"},
}


class ReactionFeedback:
    """Process emoji reactions into behavioral improvements."""

    def __init__(self, workspace: Path, provider: "LLMProvider", model: str):
        self._workspace = workspace
        self._provider = provider
        self._model = model
        self._feedback_file = workspace / "memory" / "FEEDBACK.md"
        self._learnings_file = workspace / "memory" / "LEARNINGS.md"

    def save_reaction(self, message_content: str, reaction: str, message_role: str) -> None:
        """Save a reaction to the feedback file."""
        self._feedback_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        meaning = REACTION_MEANINGS.get(reaction, {"label": "unknown", "signal": "neutral"})

        entry = (
            f"[{ts}] {reaction} ({meaning['label']}) on {message_role} message:\n"
            f"  {message_content[:200]}\n\n"
        )
        with open(self._feedback_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def _count_similar_negative(self, content_preview: str) -> int:
        """Count how many times similar content has been 👎'd."""
        if not self._feedback_file.exists():
            return 0
        existing = self._feedback_file.read_text(encoding="utf-8")
        # Count 👎 entries with overlapping keywords
        words = set(content_preview.lower().split()[:5])
        count = 0
        for line in existing.split("\n"):
            if "👎" in line or "disapproved" in line:
                line_words = set(line.lower().split())
                if len(words & line_words) >= 2:
                    count += 1
        return count

    async def process_negative_reaction(self, message_content: str, reaction: str) -> str | None:
        """For 👎 reactions, extract a lesson via LLM. Escalates on repeated issues."""
        # Check if this is a repeated complaint
        repeat_count = self._count_similar_negative(message_content[:100])
        escalation = ""
        if repeat_count >= 3:
            escalation = " This is a CRITICAL rule — the user has flagged this same issue multiple times. NEVER do this again."
        elif repeat_count >= 2:
            escalation = " The user has flagged this issue before. Make this rule stronger."
        prompt = f"""The user reacted with {reaction} (disapproval) to this assistant message:

"{message_content[:500]}"{escalation}

What was likely wrong? Extract a concise behavioral rule to prevent this in the future.
Call save_feedback_lesson with the lesson."""

        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "Extract a behavioral correction from negative user feedback. Be specific and actionable."},
                    {"role": "user", "content": prompt},
                ],
                tools=_EXTRACT_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "save_feedback_lesson"}},
            )

            if not response.has_tool_calls:
                return None

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            lesson = args.get("lesson", "").strip()
            if not lesson:
                return None

            # Save to LEARNINGS.md
            self._save_learning(lesson, "negative_reaction")
            logger.info("Reaction feedback: saved negative lesson — {}", lesson[:80])
            return lesson

        except Exception:
            logger.opt(exception=True).debug("Reaction feedback extraction failed")
            return None

    def process_positive_reaction(self, message_content: str, reaction: str) -> None:
        """For 👍/❤️ reactions, reinforce the positive pattern."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        meaning = REACTION_MEANINGS.get(reaction, {"label": "positive"})

        # Save as positive reinforcement
        # Take the first sentence of the message as the pattern to reinforce
        first_sentence = message_content.split(".")[0][:100] if message_content else ""
        entry = f"- Keep this approach: {first_sentence}... (user {meaning['label']}) [{ts}]"

        self._save_learning(entry, "positive_reaction")
        logger.info("Reaction feedback: reinforced positive pattern")

    def _save_learning(self, lesson: str, source: str) -> None:
        """Append to LEARNINGS.md with deduplication."""
        self._learnings_file.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if self._learnings_file.exists():
            existing = self._learnings_file.read_text(encoding="utf-8")

        # Simple dedup — skip if very similar lesson already exists
        lesson_lower = lesson.lower()
        for line in existing.split("\n"):
            if line.strip().startswith("- "):
                existing_words = set(line.lower().split())
                new_words = set(lesson_lower.split())
                if new_words and len(new_words & existing_words) / len(new_words) > 0.6:
                    return  # Too similar

        lines = [l for l in existing.split("\n") if l.strip().startswith("- ")]
        lines.append(f"- {lesson}" if not lesson.startswith("- ") else lesson)

        # Keep max 30 learnings
        if len(lines) > 30:
            lines = lines[-30:]

        content = "# Learned Rules\n\n" + "\n".join(lines) + "\n"
        self._learnings_file.write_text(content, encoding="utf-8")
