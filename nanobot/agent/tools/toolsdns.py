"""ToolsDNS built-in tool — semantic search and execution of indexed tools."""

import json
import urllib.parse
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


class ToolsDNSTool(Tool):
    """Search, inspect, and call tools indexed in ToolsDNS."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "toolsdns"

    @property
    def description(self) -> str:
        return (
            "Search, inspect, and call tools indexed in ToolsDNS. "
            "Actions: search (find tools by intent), list (all tools), "
            "get (full schema by id), call (execute a tool), skills (list skills)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "list", "get", "call", "skills"],
                    "description": "search: semantic search. list: all tools. get: full schema by id. call: execute tool. skills: list skills.",
                },
                "query": {
                    "type": "string",
                    "description": "Natural language query (required for search).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return for search (default 3).",
                    "minimum": 1,
                    "maximum": 20,
                },
                "threshold": {
                    "type": "number",
                    "description": "Minimum confidence 0-1 for search (default 0.1).",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "tool_id": {
                    "type": "string",
                    "description": "Tool identifier (required for get and call).",
                },
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments as key-value dict (required for call).",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        query: str = "",
        top_k: int = 3,
        threshold: float = 0.1,
        tool_id: str = "",
        arguments: dict | None = None,
        **kwargs: Any,
    ) -> str:
        if action == "search":
            return await self._search(query, top_k, threshold)
        if action == "list":
            return await self._list_tools()
        if action == "get":
            return await self._get_tool(tool_id)
        if action == "call":
            return await self._call_tool(tool_id, arguments or {})
        if action == "skills":
            return await self._list_skills()
        return f"Error: unknown action '{action}'. Choose from: search, list, get, call, skills."

    # -- HTTP helpers --

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def _get(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(f"{self._base_url}{path}", headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base_url}{path}", headers=self._headers(), json=body)
            r.raise_for_status()
            return r.json()

    # -- Action handlers --

    async def _search(self, query: str, top_k: int, threshold: float) -> str:
        if not query.strip():
            return "Error: query is required for search."
        try:
            data = await self._post("/v1/search", {"query": query, "top_k": top_k, "threshold": threshold})
        except httpx.HTTPStatusError as e:
            return f"Error: ToolsDNS returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        results = data.get("results", [])
        if not results:
            hint = data.get("hint", "")
            msg = f"No tools found for '{query}'."
            if hint:
                msg += f" Hint: {hint}"
            return msg

        lines = [f"Found {len(results)} tool(s) for '{query}' (searched {data.get('total_tools_indexed', '?')} tools):"]
        lines.append("")
        for i, r in enumerate(results, 1):
            desc = r.get("description", "")[:100]
            lines.append(f"{i}. [{r.get('confidence', 0):.2f}] {r.get('name', '')}  (id: {r.get('id', '')})")
            lines.append(f"   {desc}")
            lines.append(f"   Source: {r.get('source', '')} | Category: {r.get('category', '')} | Match: {r.get('match_reason', '')}")
            lines.append("")
        lines.append(f"Tokens saved: {data.get('tokens_saved', '?')} | Search time: {data.get('search_time_ms', '?'):.1f}ms")
        return "\n".join(lines)

    async def _list_tools(self) -> str:
        try:
            data = await self._get("/v1/tools")
        except httpx.HTTPStatusError as e:
            return f"Error: ToolsDNS returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        tools = data if isinstance(data, list) else data.get("tools", [])
        if not tools:
            return "No tools indexed in ToolsDNS."
        lines = [f"{len(tools)} tools indexed in ToolsDNS:", ""]
        for i, t in enumerate(tools, 1):
            desc = t.get("description", "")[:80]
            lines.append(f"{i}. [{t.get('source_info', {}).get('source_name', '?')}] {t.get('name', '')} — {desc}")
        out = "\n".join(lines)
        return out[:8000] + "\n...(truncated)" if len(out) > 8000 else out

    async def _get_tool(self, tool_id: str) -> str:
        if not tool_id.strip():
            return "Error: tool_id is required for get."
        safe_id = urllib.parse.quote(tool_id, safe="")
        try:
            t = await self._get(f"/v1/tool/{safe_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Error: tool '{tool_id}' not found in ToolsDNS."
            return f"Error: ToolsDNS returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        lines = [
            f"Tool: {t.get('name', '')}",
            f"ID: {t.get('id', '')}",
            f"Description: {t.get('description', '')}",
            f"Source: {t.get('source_info', {}).get('source_name', '')} ({t.get('source_info', {}).get('source_type', '')})",
            f"Tags: {', '.join(t.get('tags', []))}",
            "",
            "Input Schema:",
            json.dumps(t.get("input_schema", {}), indent=2),
        ]
        how = t.get("how_to_call")
        if how:
            lines += ["", "How to call:", json.dumps(how, indent=2)]
        skill = t.get("skill_content")
        if skill:
            lines += ["", "Skill instructions:", skill]
        return "\n".join(lines)

    async def _call_tool(self, tool_id: str, arguments: dict) -> str:
        if not tool_id.strip():
            return "Error: tool_id is required for call."
        try:
            data = await self._post("/v1/call", {"tool_id": tool_id, "arguments": arguments})
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Error: tool '{tool_id}' not found in ToolsDNS."
            if e.response.status_code == 502:
                detail = e.response.json().get("detail", e.response.text[:200])
                return f"Error: ToolsDNS could not reach the backing server: {detail}"
            return f"Error: ToolsDNS returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        result_type = data.get("type", "")
        if result_type == "skill":
            return f"Skill '{data.get('name', '')}' content:\n\n{data.get('content', '')}\n\nInstruction: {data.get('instruction', '')}"
        result = data.get("result", data)
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        return str(result)

    async def _list_skills(self) -> str:
        try:
            data = await self._get("/v1/skills")
        except httpx.HTTPStatusError as e:
            return f"Error: ToolsDNS returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        skills = data if isinstance(data, list) else data.get("skills", [])
        if not skills:
            return "No skills found in ToolsDNS."
        lines = [f"{len(skills)} skill(s) available:", ""]
        for i, s in enumerate(skills, 1):
            desc = s.get("description", "")
            lines.append(f"{i}. {s.get('name', '')} — {desc}")
        return "\n".join(lines)
