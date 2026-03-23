"""Tool catalog — lazy discovery across MCP servers.

On MCP connect, fetches all tool names + descriptions (lightweight).
Provides a search_tools built-in tool that the agent calls to discover
and dynamically load tools on demand — no hardcoded enabledTools needed.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class ToolCatalog:
    """In-memory catalog of all available MCP tools (name + description only).

    Full schemas are NOT stored here — they're fetched on demand when the
    agent calls search_tools and the matching tools get registered.
    """

    def __init__(self):
        # server_name -> list of {name, description, tool_def}
        self._entries: dict[str, list[dict]] = {}

    def add_server(self, server_name: str, tools: list) -> None:
        """Add all tools from an MCP server to the catalog."""
        entries = []
        for tool_def in tools:
            entries.append({
                "name": tool_def.name,
                "description": (tool_def.description or "")[:200],
                "tool_def": tool_def,  # Keep full def for lazy registration
            })
        self._entries[server_name] = entries
        logger.info("ToolCatalog: indexed {} tools from '{}'", len(entries), server_name)

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Search tools by keyword matching on name + description."""
        words = [w.lower() for w in query.split() if len(w) >= 2]
        if not words:
            return []

        results: list[tuple[int, str, dict]] = []

        for server_name, entries in self._entries.items():
            for entry in entries:
                name_lower = entry["name"].lower()
                desc_lower = entry["description"].lower()
                searchable = f"{name_lower} {desc_lower}"

                # Score: exact name match > word in name > word in description
                score = 0
                for word in words:
                    if word == name_lower:
                        score += 10
                    elif word in name_lower:
                        score += 5
                    elif word in searchable:
                        score += 2

                if score > 0:
                    results.append((score, server_name, entry))

        # Sort by score descending, take top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "server": server,
                "name": entry["name"],
                "description": entry["description"],
                "tool_def": entry["tool_def"],
            }
            for _, server, entry in results[:top_k]
        ]

    def total_tools(self) -> int:
        return sum(len(entries) for entries in self._entries.values())

    def server_summary(self) -> str:
        """One-line summary per server for system prompt."""
        parts = []
        for name, entries in self._entries.items():
            parts.append(f"{name}: {len(entries)} tools")
        return ", ".join(parts) if parts else "no MCP servers"


class SearchToolsTool(Tool):
    """Built-in tool that searches the MCP tool catalog and dynamically loads matches."""

    def __init__(self, catalog: ToolCatalog, registry: "ToolRegistry", sessions: dict[str, Any]):
        self._catalog = catalog
        self._registry = registry
        self._sessions = sessions  # server_name -> ClientSession for lazy registration

    @property
    def name(self) -> str:
        return "search_tools"

    @property
    def description(self) -> str:
        total = self._catalog.total_tools()
        summary = self._catalog.server_summary()
        return (
            f"Search {total} available external tools across MCP servers ({summary}). "
            "Call this BEFORE using any external tool (email, calendar, weather, GitHub, Slack, etc.) "
            "to discover the right tool name and have it loaded. "
            "Returns matching tools with descriptions. After searching, call the tool directly by its full name."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — keywords like 'gmail send email', 'weather', 'calendar create event', 'github issue'.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        if not query:
            return "Error: query is required."

        matches = self._catalog.search(query, top_k=8)
        if not matches:
            return f"No tools found matching '{query}'. Try broader keywords."

        # Dynamically register matched tools so the agent can call them
        from nanobot.agent.tools.mcp import MCPToolWrapper
        newly_registered = []
        for match in matches:
            server_name = match["server"]
            tool_def = match["tool_def"]
            wrapped_name = f"mcp_{server_name}_{tool_def.name}"

            # Skip if already registered
            if self._registry.get(wrapped_name):
                continue

            session = self._sessions.get(server_name)
            if session:
                wrapper = MCPToolWrapper(session, server_name, tool_def, tool_timeout=30)
                self._registry.register(wrapper)
                newly_registered.append(wrapped_name)
                logger.info("SearchTools: dynamically registered '{}'", wrapped_name)

        # Format results
        lines = [f"Found {len(matches)} tool(s) for '{query}':\n"]
        for m in matches:
            full_name = f"mcp_{m['server']}_{m['name']}"
            lines.append(f"  {full_name}")
            if m["description"]:
                lines.append(f"    {m['description']}")
            lines.append("")

        if newly_registered:
            lines.append(f"Loaded {len(newly_registered)} new tool(s). You can now call them directly.")
        else:
            lines.append("All matching tools already loaded — call them directly.")

        return "\n".join(lines)
