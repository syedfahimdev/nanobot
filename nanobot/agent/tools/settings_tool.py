"""Settings tool — lets Mawa read and change her own configuration.

Actions:
- list: show all settings grouped by category
- get: read a specific setting value
- set: change a setting value
- search: find settings by keyword
- categories: list all categories

This gives Mawa self-awareness about her capabilities and lets users
say "turn off frustration detection" or "set quiet hours to 11pm"
and Mawa can do it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class SettingsTool(Tool):
    """Tool for reading and modifying Mawa's settings."""

    def __init__(self, workspace: Path):
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "settings"

    @property
    def description(self) -> str:
        return (
            "Read or change Mawa's settings and feature toggles. "
            "Actions: 'list' (all settings by category), 'get' (read one), "
            "'set' (change one), 'search' (find by keyword). "
            "Use when user asks to enable/disable features, change preferences, "
            "or asks what Mawa can do."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "get", "set", "search"],
                    "description": "Action to perform",
                },
                "key": {
                    "type": "string",
                    "description": "Setting key (for get/set). e.g., 'quietHoursEnabled', 'morningPrep'",
                },
                "value": {
                    "description": "New value to set (for 'set' action). Boolean, number, or string.",
                },
                "query": {
                    "type": "string",
                    "description": "Search keyword (for 'search' action)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, key: str = "", value: Any = None, query: str = "", **kwargs) -> str:
        from nanobot.hooks.builtin.feature_registry import (
            get_feature_manifest, get_feature_categories,
            save_setting, get_setting, load_settings,
        )

        if action == "list":
            manifest = get_feature_manifest(self._workspace)
            categories = get_feature_categories()
            cat_map = {c["id"]: c["label"] for c in categories}

            grouped: dict[str, list] = {}
            for f in manifest:
                cat = cat_map.get(f["category"], f["category"])
                grouped.setdefault(cat, []).append(f)

            lines = []
            for cat, features in grouped.items():
                lines.append(f"\n## {cat}")
                for f in features:
                    val = f["value"]
                    if f["type"] == "boolean":
                        status = "ON" if val else "OFF"
                        lines.append(f"- [{status}] **{f['label']}** (`{f['key']}`): {f['desc']}")
                    elif f["type"] == "number":
                        lines.append(f"- **{f['label']}** (`{f['key']}`): {val} — {f['desc']}")
                    else:
                        lines.append(f"- **{f['label']}** (`{f['key']}`): \"{val}\" — {f['desc']}")

            return f"Mawa has {len(manifest)} configurable settings:\n" + "\n".join(lines)

        elif action == "get":
            if not key:
                return "Error: 'key' required. Use action='search' to find settings by keyword."
            manifest = get_feature_manifest(self._workspace)
            for f in manifest:
                if f["key"] == key:
                    return f"**{f['label']}** (`{key}`): {f['value']} ({f['type']})\n{f['desc']}"
            return f"Setting '{key}' not found. Use action='search' or action='list' to find available settings."

        elif action == "set":
            if not key:
                return "Error: 'key' required."
            if value is None:
                return "Error: 'value' required."

            # Validate the key exists
            manifest = get_feature_manifest(self._workspace)
            feature = None
            for f in manifest:
                if f["key"] == key:
                    feature = f
                    break
            if not feature:
                return f"Setting '{key}' not found. Use action='search' to find the right key."

            # Type coercion
            if feature["type"] == "boolean":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                value = bool(value)
            elif feature["type"] == "number":
                try:
                    value = float(value)
                    if value == int(value):
                        value = int(value)
                except (ValueError, TypeError):
                    return f"Error: '{key}' expects a number, got '{value}'."

            save_setting(self._workspace, key, value)
            return f"Setting updated: **{feature['label']}** = {value}"

        elif action == "search":
            if not query:
                return "Error: 'query' required for search."
            manifest = get_feature_manifest(self._workspace)
            query_lower = query.lower()
            matches = [
                f for f in manifest
                if query_lower in f["label"].lower() or query_lower in f["desc"].lower() or query_lower in f["key"].lower()
            ]
            if not matches:
                return f"No settings found matching '{query}'."
            lines = [f"Found {len(matches)} settings matching '{query}':"]
            for f in matches:
                val = f["value"]
                if f["type"] == "boolean":
                    status = "ON" if val else "OFF"
                    lines.append(f"- [{status}] **{f['label']}** (`{f['key']}`)")
                else:
                    lines.append(f"- **{f['label']}** (`{f['key']}`): {val}")
            return "\n".join(lines)

        return f"Unknown action '{action}'. Use: list, get, set, search"
