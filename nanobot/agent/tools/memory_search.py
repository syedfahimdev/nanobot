"""Memory search tool — semantic search with folder/type filtering."""

from __future__ import annotations

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class MemorySearchTool(Tool):
    """Search your knowledge base, learnings, rules, inbox, and history."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self._url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search your knowledge base, learnings, rules, inbox files, and conversation history. "
            "Use this to recall past decisions, people, preferences, events, or uploaded documents. "
            "Filter by folder (work, personal) or source type."
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
                    "description": "Max results (default 5, max 20).",
                },
                "folder": {
                    "type": "string",
                    "enum": ["all", "memory", "inbox", "inbox_work", "inbox_personal", "inbox_general"],
                    "description": "Filter by source: 'memory' (facts, history), 'inbox_work' (work docs), 'inbox_personal', or 'all' (default).",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence threshold 0.0-1.0 (default 0.15). Higher = fewer but more relevant results.",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        folder: str = "all",
        min_confidence: float = 0.15,
        **kwargs: Any,
    ) -> str:
        top_k = min(max(top_k, 1), 20)
        min_confidence = max(0.0, min(min_confidence, 1.0))

        # Build id_prefix filter based on folder
        id_prefix = ""
        if folder == "memory":
            id_prefix = "memory__"
        elif folder.startswith("inbox"):
            parts = folder.split("_", 1)
            if len(parts) > 1 and parts[1] in ("work", "personal", "general"):
                id_prefix = f"inbox__{parts[1]}__"
            else:
                id_prefix = "inbox__"
        # "all" = no prefix filter

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                body: dict[str, Any] = {
                    "query": query,
                    "top_k": top_k,
                    "threshold": min_confidence,
                }
                if id_prefix:
                    body["id_prefix"] = id_prefix

                resp = await client.post(
                    f"{self._url}/v1/search",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return f"Error searching memory: {e}"

        results = data.get("results", [])
        if not results:
            return f"No relevant memories found for '{query}'" + (f" in {folder}" if folder != "all" else "") + "."

        lines = [f"Found {len(results)} result(s):\n"]
        for r in results:
            title = r.get("name", r.get("title", ""))
            desc = r.get("description", r.get("content", ""))
            source = r.get("source_info", {})
            file_path = source.get("file_path", "")
            confidence = r.get("confidence", r.get("score", 0))

            lines.append(f"## {title} [{confidence:.0%}]")
            if file_path:
                lines.append(f"Source: {file_path}")
            lines.append(desc[:1000])
            lines.append("")

        return "\n".join(lines)
