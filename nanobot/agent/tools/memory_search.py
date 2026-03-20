"""Memory search tool — semantic search over knowledge, learnings, rules, and history."""

from __future__ import annotations

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class MemorySearchTool(Tool):
    """Search your knowledge base, learnings, rules, and history."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search your knowledge base, learnings, rules, and conversation history. "
            "Use this when you need to recall past decisions, people, preferences, or events."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results (default 5).",
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self._url}/v1/search",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": query,
                        "top_k": top_k,
                        "threshold": 0.15,
                        "id_prefix": "memory__",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return f"Error searching memory: {e}"

        results = data.get("results", [])
        if not results:
            return "No relevant memories found."

        lines = []
        for r in results:
            title = r.get("name", "")
            desc = r.get("description", "")
            source = r.get("source_info", {})
            file_path = source.get("file_path", "")
            confidence = r.get("confidence", 0)

            lines.append(f"## {title} [{confidence:.0%}]")
            if file_path:
                lines.append(f"Source: {file_path}")
            lines.append(desc[:1000])
            lines.append("")

        return "\n".join(lines)
