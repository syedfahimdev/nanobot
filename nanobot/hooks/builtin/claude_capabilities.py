"""Claude-level capabilities — task detection, delegation, and pre-LLM interceptors.

1. Plan→Execute task decomposer (regex + LLM hybrid)
2. Parallel tool dispatch
3. Source citation tracker
4. Smart output formatter
5. State diff snapshots
6. Built-in calculator
7. Timezone resolver
8. Clipboard/paste pipeline
9. Strategy rotator (alternative tools on failure)
10. Research pipeline (search→fetch→extract)
11. Regex builder
"""

from __future__ import annotations

import ast
import math
import operator
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Plan→Execute Task Decomposer (Advanced)
# ═══════════════════════════════════════════════════════════════════════════════

# Task verbs — categorized by weight (heavy = likely needs delegation)
_HEAVY_VERBS = frozenset({
    "research", "investigate", "analyze", "compare", "review", "summarize",
    "find", "look", "search", "browse", "explore", "compile", "gather",
    "write", "draft", "compose", "generate", "create", "build", "design",
    "download", "fetch", "scrape", "crawl", "monitor", "book", "order",
})

_LIGHT_VERBS = frozenset({
    "tell", "show", "check", "get", "list", "set", "update", "delete",
    "add", "remove", "open", "send", "reply", "forward", "schedule",
    "remind", "toggle", "enable", "disable", "run", "deploy",
})

_ALL_TASK_VERBS = _HEAVY_VERBS | _LIGHT_VERBS

_TASK_VERBS = re.compile(
    r"\b(" + "|".join(_ALL_TASK_VERBS) + r")\b", re.I,
)

# Connectors between tasks — ordered by strength
_SPLITTERS = [
    # Explicit sequence
    re.compile(r"\b(?:and\s+then|then|after\s+that|next|afterwards|once\s+done)\b", re.I),
    # Parallel/additive
    re.compile(r"\b(?:also|and\s+also|plus|as\s+well\s+as|on\s+top\s+of\s+that)\b", re.I),
    # Soft conjunction — "X and verb Y" (only split if both sides have verbs)
    re.compile(r"\band\s+(?=" + "|".join(_ALL_TASK_VERBS) + r")", re.I),
    # Comma + verb
    re.compile(r",\s*(?=" + "|".join(_ALL_TASK_VERBS) + r")", re.I),
]

# Numbered list patterns — matches both "1) X 2) Y" inline and newline-separated
_NUMBERED_RE = re.compile(r"(?:^|\n|\s)\d+[.)]\s+", re.M)

# Heuristic: messages this short are almost never multi-step
_MIN_MULTI_STEP_LEN = 20


def is_multi_step(text: str) -> bool:
    """Detect if user message requires multiple independent steps.

    Uses verb counting + connector detection. Avoids false positives on
    single-task messages like "search and compare hotels" (one task, two verbs).
    """
    if len(text) < _MIN_MULTI_STEP_LEN:
        return False
    verbs = _TASK_VERBS.findall(text.lower())
    if len(verbs) < 2:
        return False
    # Need at least one connector between verbs, or numbered items
    if _NUMBERED_RE.search(text):
        return True
    return any(s.search(text) for s in _SPLITTERS)


def decompose_task(text: str) -> list[str]:
    """Break a complex request into ordered steps. Pure regex — no LLM.

    Tries progressively weaker split strategies. Returns list of step
    descriptions, or empty list if single-step.
    """
    if not is_multi_step(text):
        return []

    # Strategy 1: Numbered items "1) do X  2) do Y"
    numbered = _NUMBERED_RE.split(text)
    numbered = [p.strip() for p in numbered if p.strip() and len(p.strip()) > 5]
    if len(numbered) >= 2:
        return numbered

    # Strategy 2: Try each splitter pattern in order of strength
    for splitter in _SPLITTERS:
        parts = splitter.split(text)
        steps = []
        for part in parts:
            part = part.strip().strip(",").strip()
            # Each step needs a verb AND enough context (not just "search" alone)
            if len(part) > 10 and _TASK_VERBS.search(part):
                steps.append(part)
        if len(steps) >= 2:
            # Reject splits where all steps share the same core object
            # e.g., "search and compare hotels" → same object "hotels"
            nouns = []
            for s in steps:
                # Extract non-verb content words as a rough "object" fingerprint
                words = set(re.findall(r"\b[a-z]{4,}\b", s.lower())) - _ALL_TASK_VERBS
                nouns.append(words)
            if len(nouns) >= 2 and nouns[0] == nouns[1] and len(nouns[0]) <= 2:
                continue  # Same object — likely one task, try next splitter
            return steps

    return []


def classify_step(step: str) -> str:
    """Classify a decomposed step as 'heavy' (delegate) or 'light' (inline).

    Heavy steps involve research, generation, or multi-tool operations.
    Light steps are quick lookups, status checks, or simple actions.
    """
    text_lower = step.lower()
    heavy_count = sum(1 for v in _HEAVY_VERBS if re.search(r"\b" + v + r"\b", text_lower))
    light_count = sum(1 for v in _LIGHT_VERBS if re.search(r"\b" + v + r"\b", text_lower))

    # Explicit time/effort indicators boost weight
    if re.search(r"\b(best|top|detailed|thorough|comprehensive|all|every|complete)\b", text_lower):
        heavy_count += 1

    # Short steps with simple verbs are light
    if len(step) < 30 and light_count > 0 and heavy_count == 0:
        return "light"

    return "heavy" if heavy_count > 0 else "light"


# Patterns that signal an explicit research/background request
_STANDALONE_HEAVY_SIGNALS = re.compile(
    r"^\s*("
    r"research\b|look\s+up\b|look\s+into\b|find\s+(me\s+)?(the\s+)?(best|top|good|cheap)"
    r"|compare\b.*\b(vs|versus|or|and)\b"
    r"|investigate\b|analyze\b|compile\b|gather\b|survey\b"
    r"|write\s+(me\s+)?(a|an|the)\b|draft\s+(a|an|the)\b"
    r"|summarize\b.*\b(article|page|doc|report|thread)"
    r")",
    re.I,
)

# Short queries that should NEVER be delegated even if they match heavy verbs
_NEVER_DELEGATE = re.compile(
    r"^\s*(what|who|when|where|why|how|is|are|do|does|did|can|will|should|tell me|show me|check)\b",
    re.I,
)


def is_standalone_heavy(text: str) -> bool:
    """Regex fallback for standalone heavy detection. Used when LLM classifier is unavailable."""
    text = text.strip()
    if len(text) < 25:
        return False
    if _NEVER_DELEGATE.match(text):
        return False
    if _STANDALONE_HEAVY_SIGNALS.search(text):
        return True
    return classify_step(text) == "heavy" and len(text) > 50


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Intent Classifier — fast model classifies task intent
# ═══════════════════════════════════════════════════════════════════════════════

_INTENT_CLASSIFY_PROMPT = """Classify this user message into ONE category. Reply with ONLY the category name, nothing else.

Categories:
- RESEARCH: needs web search, comparison, investigation, finding options (3+ tool calls likely)
- GENERATE: needs writing, drafting, composing, creating content
- QUICK: simple lookup, status check, time, weather, email check, setting change
- CHAT: greeting, follow-up, conversation, thank you, clarification
- MULTI: contains 2+ independent tasks that could run in parallel

Message: {message}

Category:"""


async def classify_intent_llm(text: str, provider, model: str) -> dict:
    """Classify user intent using a fast LLM call (~50 tokens, <500ms).

    Returns dict with:
        category: RESEARCH | GENERATE | QUICK | CHAT | MULTI
        delegate: bool — should this be auto-delegated to a subagent
        steps: list[str] — decomposed steps if MULTI
    """
    from loguru import logger

    # Regex pre-filter: skip LLM for obvious cases
    text_stripped = text.strip()
    if len(text_stripped) < 15:
        return {"category": "CHAT", "delegate": False, "steps": []}
    if _NEVER_DELEGATE.match(text_stripped) and len(text_stripped) < 60:
        return {"category": "QUICK", "delegate": False, "steps": []}

    try:
        prompt = _INTENT_CLASSIFY_PROMPT.format(message=text_stripped[:200])
        response = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            model=model,
        )

        raw = (response.content or "").strip().upper()
        # Extract category from response (handle verbose models)
        category = "QUICK"
        for cat in ("RESEARCH", "GENERATE", "MULTI", "CHAT", "QUICK"):
            if cat in raw:
                category = cat
                break

        delegate = category in ("RESEARCH", "GENERATE")

        # For MULTI, decompose into steps
        steps = []
        if category == "MULTI":
            steps = decompose_task(text_stripped)
            # If decomposition fails, fall back to single task classification
            if not steps:
                delegate = is_standalone_heavy(text_stripped)
                category = "RESEARCH" if delegate else "QUICK"

        logger.debug("LLM intent: '{}' → {} (delegate={})", text_stripped[:50], category, delegate)
        return {"category": category, "delegate": delegate, "steps": steps}

    except Exception as e:
        logger.debug("LLM intent classification failed ({}), falling back to regex", e)
        # Fallback to regex
        if is_multi_step(text_stripped):
            steps = decompose_task(text_stripped)
            if steps:
                return {"category": "MULTI", "delegate": True, "steps": steps}
        if is_standalone_heavy(text_stripped):
            return {"category": "RESEARCH", "delegate": True, "steps": []}
        return {"category": "QUICK", "delegate": False, "steps": []}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Parallel Tool Dispatch Helper
# ═══════════════════════════════════════════════════════════════════════════════

# Tools that are safe to run in parallel (read-only, independent)
_PARALLEL_SAFE = frozenset({
    "web_search", "web_fetch", "read_file", "list_dir",
    "memory_search", "inbox", "goals",
})

# Tools that MUST be sequential (write, state-changing)
_SEQUENTIAL = frozenset({
    "write_file", "edit_file", "exec", "message", "credentials",
    "browser", "background_exec",
})


def classify_for_parallel(tool_calls: list[dict]) -> tuple[list[dict], list[dict]]:
    """Classify tool calls into parallel-safe and sequential groups.

    Returns (parallel_batch, sequential_batch).
    """
    parallel = []
    sequential = []

    for tc in tool_calls:
        name = tc.get("name", "")
        if name in _PARALLEL_SAFE:
            parallel.append(tc)
        elif name in _SEQUENTIAL:
            sequential.append(tc)
        elif name.startswith("mcp_composio_") and "FETCH" in name:
            parallel.append(tc)  # Read-only Composio tools are safe
        else:
            sequential.append(tc)  # Unknown → be safe

    return parallel, sequential


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Source Citation Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class SourceTracker:
    """Track where information came from — tag facts with sources."""

    def __init__(self):
        self._sources: list[dict] = []

    def add(self, content: str, source_type: str, source_id: str) -> None:
        """Record a source. source_type: 'web', 'email', 'memory', 'tool', 'file'."""
        self._sources.append({
            "content_preview": content[:100],
            "type": source_type,
            "id": source_id,
            "ts": time.time(),
        })

    def get_citations(self) -> str:
        """Format citations for the response."""
        if not self._sources:
            return ""

        seen = set()
        lines = ["**Sources:**"]
        for s in self._sources:
            key = f"{s['type']}:{s['id']}"
            if key in seen:
                continue
            seen.add(key)

            if s["type"] == "web":
                lines.append(f"- [{s['id']}]({s['id']})")
            elif s["type"] == "email":
                lines.append(f"- Email: {s['id']}")
            elif s["type"] == "memory":
                lines.append(f"- Memory: {s['id']}")
            elif s["type"] == "tool":
                lines.append(f"- Tool: {s['id']}")
            else:
                lines.append(f"- {s['type']}: {s['id']}")

        return "\n".join(lines[:8]) if len(lines) > 1 else ""

    def clear(self) -> None:
        self._sources.clear()


# Global instance per session
_source_trackers: dict[str, SourceTracker] = {}


def get_source_tracker(session_key: str) -> SourceTracker:
    """Get or create source tracker for a session."""
    if session_key not in _source_trackers:
        _source_trackers[session_key] = SourceTracker()
    return _source_trackers[session_key]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Smart Output Formatter
# ═══════════════════════════════════════════════════════════════════════════════

def smart_format(data: Any) -> str:
    """Auto-detect data shape and format as table/list/code/json.

    Pure code — detects lists-of-dicts → markdown table, etc.
    """
    if isinstance(data, str):
        # Check if it's JSON
        try:
            import json
            parsed = json.loads(data)
            return smart_format(parsed)
        except (json.JSONDecodeError, ValueError):
            return data

    if isinstance(data, list):
        if not data:
            return "(empty list)"

        # List of dicts → markdown table
        if all(isinstance(item, dict) for item in data):
            return _format_table(data)

        # List of strings/numbers → bullet list
        lines = []
        for i, item in enumerate(data[:20], 1):
            lines.append(f"{i}. {item}")
        if len(data) > 20:
            lines.append(f"... and {len(data) - 20} more")
        return "\n".join(lines)

    if isinstance(data, dict):
        # Dict → key: value pairs
        lines = []
        for k, v in data.items():
            if isinstance(v, (list, dict)):
                lines.append(f"**{k}:** {len(v)} items")
            else:
                lines.append(f"**{k}:** {v}")
        return "\n".join(lines)

    return str(data)


def _format_table(rows: list[dict]) -> str:
    """Format list of dicts as markdown table."""
    if not rows:
        return ""

    # Get all keys
    keys = list(rows[0].keys())[:8]  # Max 8 columns

    # Header
    header = "| " + " | ".join(str(k) for k in keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"

    # Rows
    lines = [header, sep]
    for row in rows[:30]:  # Max 30 rows
        cells = [str(row.get(k, ""))[:40] for k in keys]
        lines.append("| " + " | ".join(cells) + " |")

    if len(rows) > 30:
        lines.append(f"\n*({len(rows) - 30} more rows)*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. State Diff Snapshots
# ═══════════════════════════════════════════════════════════════════════════════

_snapshots: dict[str, dict[str, str]] = {}  # key → {file_path: content_hash}


def take_snapshot(workspace: Path, name: str = "auto") -> dict:
    """Capture current state of key files for later comparison."""
    import hashlib

    snapshot: dict[str, str] = {}
    for pattern in ["memory/*.md", "memory/*.json", "*.json"]:
        for f in workspace.glob(pattern):
            if f.is_file() and f.stat().st_size < 100_000:
                h = hashlib.md5(f.read_bytes()).hexdigest()
                snapshot[str(f.relative_to(workspace))] = h

    _snapshots[name] = snapshot
    return {"name": name, "files": len(snapshot), "ts": datetime.now().isoformat()}


def compare_snapshot(workspace: Path, name: str = "auto") -> dict:
    """Compare current state against a named snapshot."""
    import hashlib

    old = _snapshots.get(name)
    if not old:
        return {"error": f"No snapshot named '{name}'"}

    current: dict[str, str] = {}
    for pattern in ["memory/*.md", "memory/*.json", "*.json"]:
        for f in workspace.glob(pattern):
            if f.is_file() and f.stat().st_size < 100_000:
                h = hashlib.md5(f.read_bytes()).hexdigest()
                current[str(f.relative_to(workspace))] = h

    changed = [f for f in current if f in old and current[f] != old[f]]
    added = [f for f in current if f not in old]
    removed = [f for f in old if f not in current]

    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "total_changes": len(changed) + len(added) + len(removed),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Built-in Calculator
# ═══════════════════════════════════════════════════════════════════════════════

# Safe operators for eval
_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

_MATH_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round,
    "ceil": math.ceil, "floor": math.floor,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "pi": math.pi, "e": math.e,
}

# Patterns that indicate a math question
_MATH_PATTERNS = [
    re.compile(r"(\d+\.?\d*)\s*%\s*(?:of)\s*\$?([\d,.]+)", re.I),  # "15% of $347" — must be before generic
    re.compile(r"\$?([\d,.]+)\s*[+\-*/]\s*\$?([\d,.]+)"),  # "$100 + $50"
    re.compile(r"(?:what(?:'s| is)|calculate|compute|how much is)\s+([\d\s+\-*/().^]+)", re.I),  # generic math (no % to avoid stealing percentage)
]


def detect_math(text: str) -> str | None:
    """Detect if the user message contains a math expression. Returns the expression or None."""
    for pattern in _MATH_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(0)
    # Also detect standalone math expressions
    stripped = text.strip()
    if re.match(r"^[\d\s+\-*/().%,$]+$", stripped) and any(c in stripped for c in "+-*/%"):
        return stripped
    return None


def safe_eval_math(expr: str) -> str | None:
    """Safely evaluate a math expression. Returns result string or None on failure.

    Uses AST parsing — NO exec/eval. Cannot execute arbitrary code.
    """
    # Normalize
    expr = expr.replace("$", "").replace(",", "").replace("^", "**")

    # Handle "X% of Y"
    pct_match = re.match(r"([\d.]+)\s*%\s*(?:of)\s*([\d.]+)", expr)
    if pct_match:
        pct, total = float(pct_match.group(1)), float(pct_match.group(2))
        result = (pct / 100) * total
        return f"{pct}% of {total} = **{result:,.2f}**"

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        if isinstance(result, float):
            # Clean float display
            if result == int(result) and abs(result) < 1e15:
                return f"= **{int(result):,}**"
            return f"= **{result:,.4f}**".rstrip("0").rstrip(".")
        return f"= **{result:,}**"
    except Exception:
        return None


def _eval_node(node: ast.expr) -> int | float:
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _MATH_FUNCS.get(node.func.id)
        if func and not callable(func):
            return func  # pi, e
        if func:
            args = [_eval_node(a) for a in node.args]
            return func(*args)
    if isinstance(node, ast.Name) and node.id in _MATH_FUNCS:
        val = _MATH_FUNCS[node.id]
        if not callable(val):
            return val
    raise ValueError(f"Unsafe expression: {ast.dump(node)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Timezone Resolver
# ═══════════════════════════════════════════════════════════════════════════════

_TZ_ALIASES: dict[str, str] = {
    "EST": "America/New_York", "EDT": "America/New_York",
    "CST": "America/Chicago", "CDT": "America/Chicago",
    "MST": "America/Denver", "MDT": "America/Denver",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
    "UTC": "UTC", "GMT": "UTC",
    "IST": "Asia/Kolkata", "BST": "Europe/London",
    "CET": "Europe/Paris", "CEST": "Europe/Paris",
    "JST": "Asia/Tokyo", "KST": "Asia/Seoul",
    "AEST": "Australia/Sydney", "BDT": "Asia/Dhaka",
    "SGT": "Asia/Singapore", "HKT": "Asia/Hong_Kong",
    "PKT": "Asia/Karachi", "AST": "Asia/Riyadh",
    # City names
    "TOKYO": "Asia/Tokyo", "LONDON": "Europe/London",
    "PARIS": "Europe/Paris", "BERLIN": "Europe/Berlin",
    "SYDNEY": "Australia/Sydney", "DHAKA": "Asia/Dhaka",
    "SEOUL": "Asia/Seoul", "SINGAPORE": "Asia/Singapore",
    "DUBAI": "Asia/Dubai", "MUMBAI": "Asia/Kolkata",
    "DELHI": "Asia/Kolkata", "KARACHI": "Asia/Karachi",
    "SHANGHAI": "Asia/Shanghai", "BEIJING": "Asia/Shanghai",
    "HONG KONG": "Asia/Hong_Kong", "BANGKOK": "Asia/Bangkok",
    "CHICAGO": "America/Chicago", "DENVER": "America/Denver",
    "LA": "America/Los_Angeles", "NYC": "America/New_York",
    "NEW YORK": "America/New_York", "SAN FRANCISCO": "America/Los_Angeles",
}

_TZ_PATTERN = re.compile(
    r"(?:what(?:'s| is)|convert)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+"
    r"([\w/]+)\s+(?:in|to)\s+([\w/]+)",
    re.I,
)


def detect_timezone_query(text: str) -> dict | None:
    """Detect timezone conversion requests. Returns parsed query or None."""
    m = _TZ_PATTERN.search(text)
    if not m:
        return None
    return {
        "time": m.group(1).strip(),
        "from_tz": m.group(2).strip(),
        "to_tz": m.group(3).strip(),
    }


def convert_timezone(time_str: str, from_tz: str, to_tz: str) -> str | None:
    """Convert a time between timezones. Returns formatted result or None."""
    from zoneinfo import ZoneInfo

    # Resolve aliases
    from_zone = _TZ_ALIASES.get(from_tz.upper(), from_tz)
    to_zone = _TZ_ALIASES.get(to_tz.upper(), to_tz)

    try:
        from_zi = ZoneInfo(from_zone)
        to_zi = ZoneInfo(to_zone)
    except Exception:
        return None

    # Parse time
    time_str = time_str.strip().upper()
    now = datetime.now()

    for fmt in ("%I:%M%p", "%I:%M %p", "%I%p", "%I %p", "%H:%M"):
        try:
            t = datetime.strptime(time_str, fmt)
            dt = now.replace(hour=t.hour, minute=t.minute, second=0, tzinfo=from_zi)
            converted = dt.astimezone(to_zi)
            return f"**{time_str}** {from_tz} = **{converted.strftime('%I:%M %p')}** {to_tz}"
        except ValueError:
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Clipboard/Paste Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def detect_paste_type(text: str) -> dict | None:
    """Detect what the user pasted and suggest processing.

    Returns {type, data, suggestion} or None if not a paste.
    """
    stripped = text.strip()

    # URL detection
    if re.match(r"https?://\S+$", stripped):
        domain = urlparse(stripped).netloc
        return {"type": "url", "data": stripped, "suggestion": f"fetch and summarize {domain}"}

    # JSON detection
    if (stripped.startswith("{") and stripped.endswith("}")) or \
       (stripped.startswith("[") and stripped.endswith("]")):
        try:
            import json
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return {"type": "json_array", "data": parsed, "suggestion": f"format {len(parsed)} items as table"}
            return {"type": "json_object", "data": parsed, "suggestion": f"format {len(parsed)} fields"}
        except json.JSONDecodeError:
            pass

    # Email detection
    if re.search(r"^(From|To|Subject|Date):", stripped, re.M) and "@" in stripped:
        return {"type": "email", "data": stripped, "suggestion": "parse and summarize this email"}

    # Code detection
    code_indicators = [
        (r"^(import |from .+ import |#include|using )", "code"),
        (r"^(def |class |function |const |let |var )", "code"),
        (r"^(if |for |while |switch |try )", "code"),
        (r"\{[\s\S]*\}", "code"),
    ]
    for pattern, ptype in code_indicators:
        if re.search(pattern, stripped, re.M):
            # Detect language
            lang = "unknown"
            if "import " in stripped and "def " in stripped:
                lang = "python"
            elif "const " in stripped or "let " in stripped or "=>" in stripped:
                lang = "javascript"
            elif "#include" in stripped:
                lang = "c/c++"
            return {"type": "code", "data": stripped, "language": lang, "suggestion": f"analyze this {lang} code"}

    # CSV detection
    lines = stripped.split("\n")
    if len(lines) >= 2:
        first_commas = lines[0].count(",")
        if first_commas >= 2 and all(abs(line.count(",") - first_commas) <= 1 for line in lines[:5]):
            return {"type": "csv", "data": stripped, "suggestion": f"parse {len(lines)} rows of CSV data"}

    # Long text (>200 chars) = article/content
    if len(stripped) > 200 and "\n" in stripped:
        return {"type": "text_block", "data": stripped, "suggestion": "summarize this text"}

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Strategy Rotator (alternative tools on failure)
# ═══════════════════════════════════════════════════════════════════════════════

# When tool A fails, try tool B
_ALTERNATIVES: dict[str, list[str]] = {
    "web_search": ["web_fetch"],  # If search fails, try direct fetch
    "web_fetch": ["web_search"],  # If fetch fails, try searching
    "browser": ["web_fetch", "web_search"],  # Browser down → fallback to fetch
    "mcp_composio_GMAIL_FETCH_EMAILS": ["web_fetch"],  # Gmail API down → try web
    "mcp_playwright_browser_navigate": ["browser", "web_fetch"],
    "mcp_playwright_browser_click": ["mcp_playwright_browser_run_code"],
    "mcp_playwright_browser_snapshot": ["mcp_playwright_browser_take_screenshot"],
    "exec": ["background_exec"],  # If exec times out → background
}

# Track failures for rotation
_failure_counts: dict[str, int] = defaultdict(int)


def get_alternative_tool(failed_tool: str) -> str | None:
    """Get an alternative tool to try after failure."""
    _failure_counts[failed_tool] += 1

    alts = _ALTERNATIVES.get(failed_tool, [])
    if not alts:
        return None

    # Rotate through alternatives based on failure count
    idx = (_failure_counts[failed_tool] - 1) % len(alts)
    alt = alts[idx]
    logger.info("Strategy rotator: {} failed ({}x), suggesting {}", failed_tool, _failure_counts[failed_tool], alt)
    return alt


def get_failure_hint(tool_name: str, error: str) -> str:
    """Generate a hint suggesting an alternative approach."""
    alt = get_alternative_tool(tool_name)
    if alt:
        return f"Try using {alt} instead — it might work where {tool_name} failed."
    return ""


def reset_failures(tool_name: str | None = None) -> None:
    """Reset failure tracking (e.g., on successful call)."""
    if tool_name:
        _failure_counts.pop(tool_name, None)
    else:
        _failure_counts.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Research Pipeline (search→fetch→extract)
# ═══════════════════════════════════════════════════════════════════════════════

_RESEARCH_PATTERNS = [
    re.compile(r"\b(research|look up|find out|investigate|deep dive)\b", re.I),
    re.compile(r"\bwhat is\b.*\b(latest|current|recent|new)\b", re.I),
    re.compile(r"\bcompare\b.*\b(vs|versus|against|or)\b", re.I),
]


def is_research_query(text: str) -> bool:
    """Detect if the user wants a multi-hop research answer."""
    return any(p.search(text) for p in _RESEARCH_PATTERNS)


def build_research_plan(query: str) -> list[dict]:
    """Build a research pipeline: search → fetch top results → extract.

    Returns a list of tool calls to execute in sequence.
    """
    steps = []

    # Step 1: Search
    steps.append({
        "step": 1,
        "tool": "web_search",
        "args": {"query": query},
        "purpose": "Find relevant sources",
    })

    # Step 2-4: Fetch top 3 results (will be populated after search)
    for i in range(3):
        steps.append({
            "step": i + 2,
            "tool": "web_fetch",
            "args": {"url": f"{{result_{i + 1}_url}}"},  # Placeholder
            "purpose": f"Fetch source #{i + 1}",
            "depends_on": 1,
        })

    return steps


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Regex Builder
# ═══════════════════════════════════════════════════════════════════════════════

# Common pattern templates
_REGEX_TEMPLATES = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
    "url": r"https?://[^\s<>\"']+",
    "ip": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "date": r"\d{4}-\d{2}-\d{2}",
    "time": r"\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?",
    "money": r"\$[\d,]+\.?\d{0,2}",
    "hex_color": r"#[0-9a-fA-F]{6}\b",
    "zip_code": r"\b\d{5}(?:-\d{4})?\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}

_REGEX_QUERY = re.compile(
    r"(?:regex|pattern)\s+(?:for|to match|to find|to extract|that matches)\s+(.+)",
    re.I,
)


def detect_regex_request(text: str) -> str | None:
    """Detect if user is asking for a regex pattern."""
    m = _REGEX_QUERY.search(text)
    return m.group(1).strip() if m else None


def build_regex(description: str) -> dict:
    """Build a regex pattern from a natural language description.

    Uses template matching first, then basic construction rules.
    """
    desc_lower = description.lower()

    # Check templates
    for name, pattern in _REGEX_TEMPLATES.items():
        if name in desc_lower or name.replace("_", " ") in desc_lower:
            return {
                "pattern": pattern,
                "template": name,
                "example": _get_regex_example(name),
                "flags": "",
            }

    # Basic construction from description
    result = _construct_regex(description)
    return result


def _get_regex_example(template: str) -> str:
    examples = {
        "email": "user@example.com",
        "phone": "+1-555-123-4567",
        "url": "https://example.com",
        "ip": "192.168.1.1",
        "date": "2026-03-23",
        "time": "3:30 PM",
        "money": "$1,234.56",
        "hex_color": "#FF5733",
        "zip_code": "06510",
    }
    return examples.get(template, "")


def _construct_regex(description: str) -> dict:
    """Construct a basic regex from description keywords."""
    desc = description.lower()
    parts = []
    flags = ""

    # Word boundaries
    if "word" in desc or "words" in desc:
        if "starts with" in desc:
            word = re.search(r"starts? with\s+(\w+)", desc)
            if word:
                parts.append(rf"\b{word.group(1)}\w*\b")
        elif "ends with" in desc:
            word = re.search(r"ends? with\s+(\w+)", desc)
            if word:
                parts.append(rf"\b\w*{word.group(1)}\b")
        elif "contains" in desc:
            word = re.search(r"contains?\s+(\w+)", desc)
            if word:
                parts.append(rf"\b\w*{word.group(1)}\w*\b")

    # Numbers
    if "number" in desc or "digit" in desc:
        if "between" in desc:
            parts.append(r"\d+")
        else:
            parts.append(r"\d+")

    # Lines
    if "line" in desc:
        if "starting with" in desc or "begins with" in desc:
            word = re.search(r"(?:starting|begins?) with\s+(.+?)(?:\s|$)", desc)
            if word:
                parts.append(f"^{re.escape(word.group(1))}.*")
                flags = "re.MULTILINE"

    pattern = "|".join(parts) if parts else f".*{re.escape(description[:20])}.*"

    return {
        "pattern": pattern,
        "template": None,
        "example": "",
        "flags": flags,
        "note": "Auto-constructed — may need refinement",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-LLM Interceptor — check if we can answer without LLM
# ═══════════════════════════════════════════════════════════════════════════════

def try_code_answer(text: str, workspace: Path) -> str | None:
    """Try to answer the user's message with pure code. Returns answer or None.

    If this returns a value, we can skip the LLM call entirely — zero tokens.
    """
    # Skip code answers for long messages — they're likely pasted content, not questions
    if len(text) > 300:
        return None

    # Math
    math_expr = detect_math(text)
    if math_expr:
        result = safe_eval_math(math_expr)
        if result:
            return result

    # Timezone
    tz_query = detect_timezone_query(text)
    if tz_query:
        result = convert_timezone(tz_query["time"], tz_query["from_tz"], tz_query["to_tz"])
        if result:
            return result

    # Regex builder
    regex_desc = detect_regex_request(text)
    if regex_desc:
        result = build_regex(regex_desc)
        lines = [f"**Pattern:** `{result['pattern']}`"]
        if result.get("template"):
            lines.append(f"Template: {result['template']}")
        if result.get("example"):
            lines.append(f"Example match: `{result['example']}`")
        if result.get("flags"):
            lines.append(f"Flags: `{result['flags']}`")
        if result.get("note"):
            lines.append(f"*{result['note']}*")
        return "\n".join(lines)

    return None
