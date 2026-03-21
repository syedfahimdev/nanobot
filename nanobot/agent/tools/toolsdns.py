"""ToolsDNS built-in tool — semantic search and execution of indexed tools."""

import json
import urllib.parse
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

_ALL_ACTIONS = [
    "search", "list", "get", "call", "skills",
    "analytics", "macros", "create_macro", "delete_macro",
    "workflows", "suggest_workflow", "create_workflow", "execute_workflow",
    "my_usage", "smart_suggest",
]


class ToolsDNSTool(Tool):
    """Search, inspect, and call tools indexed in ToolsDNS."""

    # Parameters that belong to the toolsdns action itself, NOT to the target tool
    _OWN_PARAMS = frozenset({
        "action", "query", "top_k", "threshold", "tool_id", "arguments",
        "macro_name", "macro_description", "steps", "workflow_id", "analytics_type",
    })

    # Read-only tools that can be cached (fetch = no side effects)
    _CACHEABLE_TOOLS = frozenset({
        "GMAIL_FETCH_EMAILS", "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
        "GOOGLECALENDAR_FIND_EVENT", "GOOGLECALENDAR_EVENTS_LIST",
        "WEATHERMAP_WEATHER", "WEATHERMAP_GEOCODE_LOCATION",
        "HACKERNEWS_SEARCH_POSTS", "REDDIT_SEARCH_ACROSS_SUBREDDITS",
    })
    _RESULT_CACHE_TTL = 300  # 5 minutes

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._turn_calls: list[dict] = []
        self._schema_returned_for: set[str] = set()
        self._result_cache: dict[str, tuple[str, float]] = {}  # cache_key → (result, expires_at)
        from nanobot.agent.tools.toolsdns_cache import get_cache
        self._cache = get_cache(base_url, api_key)

    @property
    def name(self) -> str:
        return "toolsdns"

    @property
    def description(self) -> str:
        return (
            "Search, inspect, and call tools indexed in ToolsDNS. "
            "Actions: search (find tools by intent), list (all tools), "
            "get (full schema by id), call (execute a tool), skills (list skills), "
            "analytics (tool usage stats), macros (list saved macros), "
            "create_macro (save a reusable multi-tool workflow), "
            "delete_macro (remove a macro), workflows (list workflows), "
            "suggest_workflow (get workflow suggestion for a task), "
            "create_workflow (save a new workflow), "
            "execute_workflow (run a saved workflow), "
            "my_usage (your recent tool call history), "
            "smart_suggest (analyze your usage and suggest macros/workflows to create)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _ALL_ACTIONS,
                    "description": (
                        "search: semantic search. list: all tools. get: full schema by id. "
                        "call: execute tool. skills: list skills. analytics: usage stats. "
                        "macros: list macros. create_macro: save multi-step workflow as macro. "
                        "delete_macro: remove macro. workflows: list workflows. "
                        "suggest_workflow: suggest workflow for a query. "
                        "create_workflow: save a workflow. execute_workflow: run a workflow. "
                        "my_usage: your recent call history. "
                        "smart_suggest: analyze usage patterns and suggest macros to save time."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Natural language query (for search, suggest_workflow, smart_suggest).",
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
                    "description": "Tool identifier (for get, call, delete_macro).",
                },
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments (for call, execute_workflow).",
                },
                "macro_name": {
                    "type": "string",
                    "description": "Name for the macro (for create_macro).",
                },
                "macro_description": {
                    "type": "string",
                    "description": "Description of what the macro does (for create_macro).",
                },
                "steps": {
                    "type": "array",
                    "description": "List of steps for create_macro/create_workflow. Each step: {tool_id, arg_template}.",
                    "items": {"type": "object"},
                },
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID (for execute_workflow).",
                },
                "analytics_type": {
                    "type": "string",
                    "enum": ["popular", "unused", "agents", "conversion"],
                    "description": "Type of analytics report (default: popular).",
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
        macro_name: str = "",
        macro_description: str = "",
        steps: list | None = None,
        workflow_id: str = "",
        analytics_type: str = "popular",
        **kwargs: Any,
    ) -> str:
        # Reset turn tracking when a new task starts (search = new intent)
        if action == "search":
            self._turn_calls.clear()
            self._schema_returned_for.clear()
            return await self._search(query, top_k, threshold)
        # Reset on macro creation (task completed, macro saved)
        if action == "create_macro":
            self._turn_calls.clear()
            return await self._create_macro(macro_name, macro_description, steps or [])
        if action == "list":
            return await self._list_tools()
        if action == "get":
            return await self._get_tool(tool_id)
        if action == "call":
            return await self._call_tool(tool_id, arguments or {})
        if action == "skills":
            return await self._list_skills()
        if action == "analytics":
            return await self._analytics(analytics_type)
        if action == "macros":
            return await self._list_macros()
        if action == "delete_macro":
            return await self._delete_macro(tool_id)
        if action == "workflows":
            return await self._list_workflows()
        if action == "suggest_workflow":
            return await self._suggest_workflow(query)
        if action == "create_workflow":
            return await self._create_workflow(query, steps or [])
        if action == "execute_workflow":
            return await self._execute_workflow(workflow_id, arguments or {})
        if action == "my_usage":
            return await self._my_usage()
        if action == "smart_suggest":
            return await self._smart_suggest(query)
        return f"Error: unknown action '{action}'. Choose from: {', '.join(_ALL_ACTIONS)}."

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

    async def _delete(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.delete(f"{self._base_url}{path}", headers=self._headers())
            r.raise_for_status()
            return r.json() if r.text else {}

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

    # Regex for detecting placeholder strings the LLM copied from the context block
    _PLACEHOLDER_RE = __import__("re").compile(r"^<\w+>$")

    async def _sanitize_args(self, tool_id: str, arguments: dict) -> dict:
        """Code-level argument sanitizer — fix LLM mistakes before calling the tool.

        1. Strip toolsdns-own params leaked into arguments
        2. Remove placeholder strings like <query>, <user_id>
        3. Strip params not in the tool's schema (unknown junk)
        4. Auto-fill from tool memory if args are empty/minimal
        """
        # Step 1: strip own params
        arguments = {k: v for k, v in arguments.items() if k not in self._OWN_PARAMS}

        # Step 2: remove placeholder strings
        arguments = {
            k: v for k, v in arguments.items()
            if not (isinstance(v, str) and self._PLACEHOLDER_RE.match(v))
        }

        # Step 3: fetch schema (cached) and strip unknown params
        try:
            schema = await self._cache.get_schema(tool_id)
            valid_params = set(schema.get("properties", {}).keys())
            if valid_params:
                stripped = {k: v for k, v in arguments.items() if k in valid_params}
                if stripped != arguments:
                    from loguru import logger
                    removed = set(arguments) - set(stripped)
                    if removed:
                        logger.info("ToolsDNS sanitizer: stripped unknown params {} from {}", removed, tool_id)
                arguments = stripped
        except Exception:
            pass  # Schema fetch failed — proceed with what we have

        # Step 4: auto-fill from tool memory if args are empty or very sparse
        if len(arguments) <= 1:
            try:
                hints_data = await self._post("/v1/tool-hints", {
                    "agent_id": "mawa",
                    "tool_ids": [tool_id],
                })
                hints = hints_data.get("hints", {}).get(tool_id, [])
                if hints:
                    remembered = hints[0].get("arguments", {})
                    # Merge: remembered args fill in gaps, but don't overwrite what the LLM set
                    for k, v in remembered.items():
                        if k not in arguments and v:
                            arguments[k] = v
                    if remembered:
                        from loguru import logger
                        logger.info("ToolsDNS sanitizer: auto-filled args from memory for {}", tool_id)
            except Exception:
                pass

        return arguments

    async def _call_tool(self, tool_id: str, arguments: dict, agent_id: str = "mawa") -> str:
        if not tool_id.strip():
            return "Error: tool_id is required for call."

        # Code-level argument sanitization — fixes LLM mistakes
        arguments = await self._sanitize_args(tool_id, arguments)

        # Check result cache for read-only tools
        import time as _time
        tool_name = tool_id.replace("tooldns__", "").replace("tooldns-skills__", "")
        if tool_name in self._CACHEABLE_TOOLS:
            import hashlib
            cache_key = hashlib.md5(f"{tool_id}:{json.dumps(arguments, sort_keys=True)}".encode()).hexdigest()
            cached = self._result_cache.get(cache_key)
            if cached and cached[1] > _time.time():
                from loguru import logger
                logger.info("ToolsDNS cache hit for {} (saved {}ms)", tool_name, "~27000")
                return cached[0]

        try:
            data = await self._post("/v1/call", {
                "tool_id": tool_id,
                "arguments": arguments,
                "agent_id": agent_id,
            })
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Error: tool '{tool_id}' not found in ToolsDNS."
            if e.response.status_code == 502:
                detail = e.response.json().get("detail", e.response.text[:200])
                return f"Error: ToolsDNS could not reach the backing server: {detail}"
            return f"Error: ToolsDNS returned HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        # Track this call for auto-macro detection
        self._turn_calls.append({"tool_id": tool_id, "arguments": arguments})

        result_type = data.get("type", "")
        if result_type == "skill":
            return f"Skill '{data.get('name', '')}' content:\n\n{data.get('content', '')}\n\nInstruction: {data.get('instruction', '')}"
        if result_type == "macro_result":
            lines = [f"Macro '{data.get('macro_id', '')}' executed:"]
            for step in data.get("steps", []):
                status = step.get("status", "?")
                tid = step.get("tool_id", "?")
                lines.append(f"  - {tid}: {status}")
                if status == "failed":
                    lines.append(f"    Error: {step.get('error', '')}")
            return "\n".join(lines)

        result = data.get("result", data)
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)

        # Auto-macro hint: if 2+ tool calls in this turn, nudge the LLM
        if len(self._turn_calls) >= 2 and not tool_id.startswith("macro__"):
            tool_ids = [c["tool_id"] for c in self._turn_calls]
            arg_templates = []
            for c in self._turn_calls:
                tmpl = {}
                for k, v in c["arguments"].items():
                    tmpl[k] = f"{{{k}}}" if isinstance(v, str) else v
                arg_templates.append({"tool_id": c["tool_id"], "arg_template": tmpl})
            hint = (
                f"\n\n[AUTO-MACRO HINT] You called {len(self._turn_calls)} tools this turn: "
                f"{' → '.join(tool_ids)}. "
                f"Consider saving this as a macro so it's one call next time. "
                f"Use: action=create_macro, macro_name=<descriptive-name>, "
                f"macro_description=<what it does>, steps={json.dumps(arg_templates)}"
            )
            result_str += hint

        # Cache result for read-only tools
        if tool_name in self._CACHEABLE_TOOLS:
            self._result_cache[cache_key] = (result_str, _time.time() + self._RESULT_CACHE_TTL)

        return result_str

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

    # -- Analytics --

    async def _analytics(self, report_type: str) -> str:
        try:
            data = await self._get(f"/v1/analytics/{report_type}")
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        if report_type == "popular":
            tools = data.get("popular_tools", [])
            if not tools:
                return "No tool call data yet."
            lines = [f"Top {len(tools)} most-used tools:", ""]
            for i, t in enumerate(tools, 1):
                lines.append(f"{i}. {t.get('tool_name', '?')} — {t.get('call_count', 0)} calls (last: {t.get('last_called', '?')[:10]})")
            return "\n".join(lines)

        if report_type == "unused":
            tools = data.get("unused_tools", [])
            if not tools:
                return "All indexed tools have been called at least once."
            lines = [f"{len(tools)} tools never called:", ""]
            for t in tools[:20]:
                lines.append(f"  - {t.get('tool_name', '?')}: {t.get('description', '')[:60]}")
            if len(tools) > 20:
                lines.append(f"  ...and {len(tools) - 20} more")
            return "\n".join(lines)

        if report_type == "agents":
            agents = data.get("agents", data) if isinstance(data, dict) else data
            if not agents:
                return "No agent activity recorded yet."
            return json.dumps(agents, indent=2)[:4000]

        if report_type == "conversion":
            return json.dumps(data, indent=2)[:4000]

        return json.dumps(data, indent=2)[:4000]

    # -- Macros --

    async def _list_macros(self) -> str:
        try:
            data = await self._get("/v1/macros")
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        macros = data if isinstance(data, list) else data.get("macros", [])
        if not macros:
            return "No macros saved yet. Use create_macro to save a reusable multi-tool workflow."
        lines = [f"{len(macros)} macro(s):", ""]
        for i, m in enumerate(macros, 1):
            steps_desc = ", ".join(s.get("tool_id", "?") for s in m.get("steps", []))
            lines.append(f"{i}. macro__{m.get('name', '?')} — {m.get('description', '')}")
            lines.append(f"   Steps: {steps_desc}")
            lines.append(f"   Used {m.get('usage_count', 0)} times")
            lines.append("")
        return "\n".join(lines)

    async def _create_macro(self, name: str, description: str, steps: list) -> str:
        if not name.strip():
            return "Error: macro_name is required."
        if not steps:
            return "Error: steps is required (list of {tool_id, arg_template} objects)."
        try:
            data = await self._post("/v1/macros", {
                "name": name,
                "description": description,
                "steps": steps,
            })
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"
        macro_id = data.get("id", data.get("name", "?"))
        return f"Macro created: macro__{macro_id}\nCall it with: action=call, tool_id=macro__{name}"

    async def _delete_macro(self, macro_id: str) -> str:
        if not macro_id.strip():
            return "Error: tool_id is required (the macro ID to delete)."
        safe_id = urllib.parse.quote(macro_id, safe="")
        try:
            data = await self._delete(f"/v1/macros/{safe_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Macro '{macro_id}' not found."
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"
        return f"Macro '{macro_id}' deleted."

    # -- Workflows --

    async def _list_workflows(self) -> str:
        try:
            data = await self._get("/v1/workflows")
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        workflows = data if isinstance(data, list) else data.get("workflows", [])
        if not workflows:
            return "No workflows saved. Use suggest_workflow to get suggestions or create_workflow to make one."
        lines = [f"{len(workflows)} workflow(s):", ""]
        for i, w in enumerate(workflows, 1):
            steps = w.get("steps", [])
            step_names = ", ".join(s.get("tool_id", "?") for s in steps[:5])
            lines.append(f"{i}. {w.get('id', '?')} — {w.get('description', w.get('trigger_phrase', ''))[:80]}")
            lines.append(f"   Steps: {step_names}")
            lines.append("")
        out = "\n".join(lines)
        return out[:6000] + "\n...(truncated)" if len(out) > 6000 else out

    async def _suggest_workflow(self, query: str) -> str:
        if not query.strip():
            return "Error: query is required for suggest_workflow."
        try:
            data = await self._post("/v1/suggest-workflow", {"query": query, "agent_id": "mawa"})
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        workflows = data.get("workflows", [])
        if not workflows:
            return f"No workflow suggestions for '{query}'. You can create one with create_workflow."
        lines = [f"Suggested workflow(s) for '{query}':", ""]
        for w in workflows:
            lines.append(f"  Workflow: {w.get('id', '?')}")
            lines.append(f"  Description: {w.get('description', w.get('trigger_phrase', ''))}")
            for j, s in enumerate(w.get("steps", []), 1):
                lines.append(f"    Step {j}: {s.get('tool_id', '?')} — args: {json.dumps(s.get('arg_template', {}))}")
            lines.append(f"  To run: action=execute_workflow, workflow_id={w.get('id', '?')}")
            lines.append("")
        return "\n".join(lines)

    async def _create_workflow(self, description: str, steps: list) -> str:
        if not steps:
            return "Error: steps is required."
        try:
            data = await self._post("/v1/workflows", {
                "trigger_phrase": description,
                "steps": steps,
                "agent_id": "mawa",
            })
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"
        wf_id = data.get("id", data.get("workflow_id", "?"))
        return f"Workflow created: {wf_id}\nRun with: action=execute_workflow, workflow_id={wf_id}"

    async def _execute_workflow(self, workflow_id: str, context: dict) -> str:
        if not workflow_id.strip():
            return "Error: workflow_id is required."
        try:
            data = await self._post("/v1/execute-workflow", {
                "workflow_id": workflow_id,
                "context": context,
                "agent_id": "mawa",
            })
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        lines = [f"Workflow {workflow_id} — status: {data.get('overall_status', '?')}", ""]
        for step in data.get("steps", []):
            status = step.get("status", "?")
            tid = step.get("tool_id", "?")
            lines.append(f"  {tid}: {status}")
            if status == "failed":
                lines.append(f"    Error: {step.get('error', '')}")
            elif step.get("result"):
                result_str = json.dumps(step["result"]) if isinstance(step["result"], dict) else str(step["result"])
                lines.append(f"    Result: {result_str[:200]}")
        return "\n".join(lines)

    # -- Agent usage & smart suggestions --

    async def _my_usage(self) -> str:
        try:
            data = await self._get("/v1/analytics/agents")
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code}: {e.response.text[:200]}"
        except Exception as e:
            return f"Error: {e}"

        agents = data.get("agents", data) if isinstance(data, dict) else data
        mawa = None
        if isinstance(agents, list):
            for a in agents:
                if a.get("agent_id") == "mawa":
                    mawa = a
                    break
        if not mawa:
            return "No usage data for agent 'mawa' yet. Start calling tools to build history."

        lines = [
            f"Agent: mawa",
            f"Total calls: {mawa.get('total_calls', 0)}",
            "",
            "Top tools:",
        ]
        for t in mawa.get("top_tools", mawa.get("favorite_tools", []))[:10]:
            if isinstance(t, dict):
                lines.append(f"  - {t.get('tool_name', t.get('tool_id', '?'))}: {t.get('call_count', '?')} calls")
            else:
                lines.append(f"  - {t}")
        return "\n".join(lines)

    async def _smart_suggest(self, query: str = "") -> str:
        """Analyze mawa's usage patterns and suggest macros/workflows to create.

        Returns actionable data for the LLM to reason about and auto-create macros.
        """
        try:
            popular = await self._get("/v1/analytics/popular")
            agent_data = await self._get("/v1/analytics/agents")
            existing_macros = await self._get("/v1/macros")
        except Exception as e:
            return f"Error fetching usage data: {e}"

        popular_tools = popular.get("popular_tools", [])
        if not popular_tools:
            return "Not enough usage data yet. Keep using tools and try again later."

        agents = agent_data.get("agents", agent_data) if isinstance(agent_data, dict) else agent_data
        mawa_data = None
        if isinstance(agents, list):
            for a in agents:
                if a.get("agent_id") == "mawa":
                    mawa_data = a
                    break

        macros = existing_macros if isinstance(existing_macros, list) else existing_macros.get("macros", [])
        existing_macro_names = {m.get("name", "") for m in macros}

        lines = [
            "=== USAGE INTELLIGENCE REPORT ===",
            "",
            "MOST-USED TOOLS:",
        ]
        for t in popular_tools[:10]:
            tid = t.get("tool_id", t.get("tool_name", "?"))
            lines.append(f"  - {t.get('tool_name', '?')} (id: {tid}) — {t.get('call_count', 0)} calls")

        if mawa_data:
            lines.append("")
            lines.append(f"MAWA TOTAL CALLS: {mawa_data.get('total_calls', 0)}")
            top = mawa_data.get("top_tools", mawa_data.get("favorite_tools", []))
            if top:
                lines.append("MAWA'S TOP TOOLS:")
                for t in top[:8]:
                    if isinstance(t, dict):
                        lines.append(f"  - {t.get('tool_name', t.get('tool_id', '?'))}: {t.get('call_count', '?')} calls")

        if existing_macro_names:
            lines.append("")
            lines.append(f"ALREADY SAVED MACROS: {', '.join(existing_macro_names)}")
            lines.append("(Do NOT recreate these — they already exist)")

        # Detect repeated patterns from recent usage
        detected_patterns = []
        try:
            learn_data = await self._post("/v1/learn", {"time_window_hours": 24, "min_occurrences": 2})
            detected_patterns = learn_data.get("workflows", [])
        except Exception:
            pass

        if detected_patterns:
            lines.append("")
            lines.append("DETECTED REPEATED SEQUENCES (auto-learned from your usage):")
            for i, p in enumerate(detected_patterns[:5], 1):
                steps = p.get("steps", [])
                step_ids = [s.get("tool_id", "?") for s in steps]
                lines.append(f"  Pattern {i}: {' → '.join(step_ids)}")
                # Generate a ready-to-use macro creation command
                suggested_name = "-then-".join(
                    s.get("tool_id", "").split("__")[-1].lower().replace("_", "-")[:20]
                    for s in steps[:3]
                )
                macro_steps = []
                for s in steps:
                    tmpl = {}
                    for k in s.get("arg_template", {}):
                        tmpl[k] = f"{{{k}}}"
                    macro_steps.append({"tool_id": s["tool_id"], "arg_template": tmpl})

                if suggested_name not in existing_macro_names:
                    lines.append(f"    → SUGGESTED MACRO: name={suggested_name}")
                    lines.append(f"      steps={json.dumps(macro_steps)}")
                else:
                    lines.append(f"    → Already saved as macro: {suggested_name}")

        lines.append("")
        lines.append("=== ACTION REQUIRED ===")
        lines.append(
            "Based on the data above, you SHOULD create macros for any repeated patterns "
            "that don't already exist. Use action=create_macro for each one. "
            "This will save tool calls and tokens in future conversations."
        )

        if query:
            lines.append("")
            lines.append(f"USER CONTEXT: {query}")
            lines.append(
                "Also consider whether the user's hint suggests a new macro or workflow "
                "that should be created proactively."
            )

        return "\n".join(lines)
