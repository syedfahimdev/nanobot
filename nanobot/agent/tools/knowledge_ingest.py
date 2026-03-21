"""Knowledge Ingestion — learn from URLs, blogs, and documentation.

Scrapes web pages, extracts actionable knowledge, and saves it to memory
or creates new skills. Supports one-time reads and recurring scrapes via cron.

Examples:
  "Read unwindai.com and tell me what's new in AI"
  "Follow this blog weekly and summarize new posts"
  "Read this API doc and learn how to use it"
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_EXTRACT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_knowledge",
            "description": "Save extracted knowledge from the web content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise summary of what was learned (2-3 sentences).",
                    },
                    "key_insights": {
                        "type": "string",
                        "description": "Bullet-pointed list of actionable insights or facts.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["ai_news", "tool_discovery", "technique", "reference", "personal", "work"],
                        "description": "Category of the knowledge.",
                    },
                    "suggest_skill": {
                        "type": "boolean",
                        "description": "True if this knowledge suggests creating a new skill.",
                    },
                    "skill_idea": {
                        "type": "string",
                        "description": "If suggest_skill is true, describe the skill idea.",
                    },
                },
                "required": ["summary", "key_insights", "category"],
            },
        },
    }
]


class KnowledgeIngestTool(Tool):
    """Read URLs, extract knowledge, save to memory or suggest skills."""

    def __init__(self, workspace: Path, provider: "LLMProvider", model: str):
        self._workspace = workspace
        self._provider = provider
        self._model = model
        self._knowledge_dir = workspace / "knowledge"
        self._digest_file = workspace / "memory" / "KNOWLEDGE_DIGEST.md"

    @property
    def name(self) -> str:
        return "learn_from_url"

    @property
    def description(self) -> str:
        return (
            "Read a web page, blog, or documentation and extract actionable knowledge. "
            "Saves insights to memory. Can suggest new skills based on what's learned. "
            "Use when: user says 'read this', 'learn from', 'check this blog', 'what's new on'. "
            "Actions: read (single URL), digest (summarize + save)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "digest"],
                    "description": "read = fetch and return content. digest = fetch, analyze, save knowledge.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to read.",
                },
                "focus": {
                    "type": "string",
                    "description": "What to focus on (e.g., 'AI tools', 'new techniques', 'API changes').",
                },
            },
            "required": ["action", "url"],
        }

    async def execute(
        self,
        action: str,
        url: str = "",
        focus: str = "",
        **kwargs: Any,
    ) -> str:
        if not url:
            return "Error: url required."

        if action == "read":
            return await self._read(url)
        elif action == "digest":
            return await self._digest(url, focus)
        return f"Unknown action: {action}"

    async def _read(self, url: str) -> str:
        """Fetch and return raw content."""
        try:
            import httpx
            resp = httpx.get(url, timeout=15, follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Mawa/1.0)"})
            if resp.status_code != 200:
                return f"Error: HTTP {resp.status_code}"

            # Try Jina reader for clean text extraction
            try:
                jina_resp = httpx.get(f"https://r.jina.ai/{url}", timeout=15)
                if jina_resp.status_code == 200:
                    return jina_resp.text[:10000]
            except Exception:
                pass

            return resp.text[:10000]
        except Exception as e:
            return f"Error fetching URL: {e}"

    async def _digest(self, url: str, focus: str) -> str:
        """Fetch, analyze, and save knowledge from a URL."""
        content = await self._read(url)
        if content.startswith("Error:"):
            return content

        focus_instruction = f"\nFocus specifically on: {focus}" if focus else ""

        prompt = f"""Analyze this web content and extract actionable knowledge.
{focus_instruction}

URL: {url}
Content:
{content[:6000]}

Call save_knowledge with your analysis. If the content suggests a tool or automation
that could be turned into a skill, set suggest_skill=true."""

        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "You are a knowledge extraction agent. Analyze web content and extract actionable insights."},
                    {"role": "user", "content": prompt},
                ],
                tools=_EXTRACT_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "save_knowledge"}},
            )

            if not response.has_tool_calls:
                return "Could not extract knowledge from this URL."

            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                args = json.loads(args)

            summary = args.get("summary", "")
            insights = args.get("key_insights", "")
            category = args.get("category", "reference")
            suggest_skill = args.get("suggest_skill", False)
            skill_idea = args.get("skill_idea", "")

            # Save to knowledge digest
            self._save_digest(url, summary, insights, category)

            # Save to category-specific knowledge file
            self._save_knowledge(url, summary, insights, category)

            result = f"**Learned from {url}**\n\n{summary}\n\n{insights}"
            if suggest_skill:
                result += f"\n\n**Skill Suggestion:** {skill_idea}\nUse `learn_skill(action='create')` to build it."

            return result

        except Exception as e:
            return f"Error analyzing content: {e}"

    def _save_digest(self, url: str, summary: str, insights: str, category: str) -> None:
        """Append to the knowledge digest file."""
        self._digest_file.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{ts}] {category.upper()} — {url}\n{summary}\n{insights}\n"
        with open(self._digest_file, "a", encoding="utf-8") as f:
            f.write(entry)

    def _save_knowledge(self, url: str, summary: str, insights: str, category: str) -> None:
        """Save to category-specific knowledge file."""
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        path = self._knowledge_dir / f"{category}.md"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n### [{ts}] {url}\n{summary}\n{insights}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
