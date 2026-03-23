"""Auto-pick the cheapest model based on message complexity.

Uses simple heuristics (no LLM call) to classify messages and
suggest the best-fit profile.
"""

from __future__ import annotations

import re

MODEL_TIERS: dict[str, list[str]] = {
    "fast": ["groq", "gemini", "modelrelay"],
    "balanced": ["kimi", "default"],
    "powerful": ["claude", "openrouter"],
}

# Indicators of complexity
_CODE_BLOCK = re.compile(r"```")
_MULTI_QUESTION = re.compile(r"\?\s*\n|\?\s+\w")
_REASONING_WORDS = re.compile(
    r"\b(explain|analyze|compare|refactor|architect|design|debug|optimize|why|how does)\b",
    re.IGNORECASE,
)


def classify_complexity(message: str) -> str:
    """Classify a message as simple, moderate, or complex."""
    length = len(message)
    code_blocks = len(_CODE_BLOCK.findall(message))
    reasoning_hits = len(_REASONING_WORDS.findall(message))
    multi_q = len(_MULTI_QUESTION.findall(message))

    score = 0
    # Length scoring
    if length > 2000:
        score += 2
    elif length > 500:
        score += 1

    # Code blocks
    score += min(code_blocks, 3)

    # Reasoning keywords
    score += min(reasoning_hits, 3)

    # Multiple questions
    score += min(multi_q, 2)

    if score >= 4:
        return "complex"
    elif score >= 2:
        return "moderate"
    return "simple"


def suggest_model(message: str, profiles: dict) -> str | None:
    """Suggest the best profile name for the given message.

    Args:
        message: The user message text.
        profiles: Dict of available profile names (from config.profiles).

    Returns:
        Profile name string or None if no match found.
    """
    complexity = classify_complexity(message)
    available = set(profiles.keys()) if profiles else set()
    if not available:
        return None

    if complexity == "simple":
        tier_order = ["fast", "balanced", "powerful"]
    elif complexity == "moderate":
        tier_order = ["balanced", "powerful", "fast"]
    else:
        tier_order = ["powerful", "balanced"]

    for tier in tier_order:
        for profile in MODEL_TIERS.get(tier, []):
            if profile in available:
                return profile

    # Fallback: return first available profile
    return next(iter(available), None)
