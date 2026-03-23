"""Prompt Optimizer — tracks instruction effectiveness and auto-adjusts.

Monitors which tool routing instructions succeed vs fail, tracks correction
frequency per instruction, and periodically suggests prompt improvements.

Stores optimization data in memory/PROMPT_STATS.json.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.hooks.events import ToolAfter, TurnCompleted


class PromptOptimizer:
    """Tracks prompt instruction effectiveness and suggests improvements."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._stats_file = workspace / "memory" / "PROMPT_STATS.json"
        self._suggestions_file = workspace / "memory" / "PROMPT_IMPROVEMENTS.md"
        self._stats = self._load_stats()
        # Track per-session: was the correct tool used?
        self._session_tool_routing: dict[str, list[str]] = defaultdict(list)
        self._session_corrections: dict[str, int] = defaultdict(int)

    def _load_stats(self) -> dict[str, Any]:
        try:
            if self._stats_file.exists():
                raw = self._stats_file.read_text(encoding="utf-8")
                if isinstance(raw, str) and raw.strip():
                    return json.loads(raw)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return {
            "routing_success": {},  # tool_name → {correct: N, wrong: N}
            "correction_triggers": {},  # instruction_keyword → correction_count
            "total_turns": 0,
            "last_optimized": "",
        }

    def _save_stats(self) -> None:
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)
            self._stats_file.write_text(json.dumps(self._stats, indent=2), encoding="utf-8")
        except OSError:
            pass

    def on_tool_call(self, tool_name: str, session_key: str) -> None:
        """Track which tools are being called."""
        self._session_tool_routing[session_key].append(tool_name)

        # Track routing success
        if tool_name not in self._stats["routing_success"]:
            self._stats["routing_success"][tool_name] = {"correct": 0, "wrong": 0}
        self._stats["routing_success"][tool_name]["correct"] += 1

    def on_correction_detected(self, session_key: str) -> None:
        """Track when the user corrects the agent."""
        self._session_corrections[session_key] += 1

        # Mark last tool as potentially wrong routing
        tools = self._session_tool_routing.get(session_key, [])
        if tools:
            last_tool = tools[-1]
            if last_tool in self._stats["routing_success"]:
                self._stats["routing_success"][last_tool]["wrong"] += 1
                self._stats["routing_success"][last_tool]["correct"] = max(
                    0, self._stats["routing_success"][last_tool]["correct"] - 1
                )

    def on_turn_completed(self, session_key: str) -> str | None:
        """Analyze turn and optionally suggest improvements."""
        self._stats["total_turns"] += 1
        self._save_stats()

        # Every 50 turns, check for optimization opportunities
        if self._stats["total_turns"] % 50 == 0:
            return self._analyze_and_suggest()
        return None

    def _analyze_and_suggest(self) -> str | None:
        """Analyze routing stats and suggest prompt improvements."""
        suggestions = []

        for tool, stats in self._stats.get("routing_success", {}).items():
            total = stats.get("correct", 0) + stats.get("wrong", 0)
            if total < 5:
                continue
            wrong_rate = stats.get("wrong", 0) / total
            if wrong_rate > 0.3:
                suggestions.append(
                    f"- **{tool}**: {int(wrong_rate * 100)}% wrong routing "
                    f"({stats['wrong']} corrections out of {total} calls). "
                    f"Consider adding more explicit routing rules."
                )

        if not suggestions:
            return None

        # Write suggestions
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._suggestions_file.parent.mkdir(parents=True, exist_ok=True)
        entry = f"\n## [{ts}] Prompt Optimization Suggestions\n" + "\n".join(suggestions) + "\n"
        with open(self._suggestions_file, "a", encoding="utf-8") as f:
            f.write(entry)

        self._stats["last_optimized"] = ts
        self._save_stats()

        logger.info("Prompt optimizer: {} suggestions generated", len(suggestions))
        return f"Prompt optimization: {len(suggestions)} routing issues detected. See PROMPT_IMPROVEMENTS.md."

    def get_stats_summary(self) -> dict[str, Any]:
        """Return stats for the settings page."""
        routing = self._stats.get("routing_success", {})
        problem_tools = []
        for tool, stats in routing.items():
            total = stats.get("correct", 0) + stats.get("wrong", 0)
            if total >= 3:
                rate = stats.get("correct", 0) / total * 100
                problem_tools.append({"tool": tool, "accuracy": round(rate, 1), "total": total})

        return {
            "totalTurns": self._stats.get("total_turns", 0),
            "lastOptimized": self._stats.get("last_optimized", "never"),
            "routingAccuracy": sorted(problem_tools, key=lambda x: x["accuracy"]),
        }


# Global instance
_optimizer: PromptOptimizer | None = None


def get_optimizer(workspace: Path) -> PromptOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer(workspace)
    return _optimizer


def make_prompt_optimizer_tool_hook(workspace: Path):
    """Hook for tool_after: tracks tool routing."""
    optimizer = get_optimizer(workspace)

    async def on_tool_after(event: ToolAfter) -> None:
        tool_name = event.name
        session = event.session_key or "unknown"
        optimizer.on_tool_call(tool_name, session)

    return on_tool_after


def make_prompt_optimizer_turn_hook(workspace: Path):
    """Hook for turn_completed: periodic analysis."""
    optimizer = get_optimizer(workspace)

    async def on_turn_completed(event: TurnCompleted) -> None:
        optimizer.on_turn_completed(event.session_key)

    return on_turn_completed
