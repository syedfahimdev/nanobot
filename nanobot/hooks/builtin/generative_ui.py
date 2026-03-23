"""Generative UI — system prompt injection for LLM-generated HTML widgets.

When the LLM wants to present data visually (charts, tables, dashboards),
it outputs a self-contained HTML block wrapped in ```html-widget markers.
The frontend extracts these blocks and renders them in sandboxed iframes.
"""

from __future__ import annotations

import re

GENERATIVE_UI_INSTRUCTION = (
    "## Generative UI\n\n"
    "IMPORTANT: When showing ANY visual data (charts, tables, comparisons, dashboards, diagrams), "
    "you MUST output a self-contained HTML block using this exact format:\n\n"
    "```html-widget\n"
    "<div style=\"background:#0a0a0a;color:#e5e5e5;padding:16px;font-family:system-ui\">\n"
    "  <!-- your HTML here -->\n"
    "</div>\n"
    "```\n\n"
    "Rules:\n"
    "- Use ```html-widget as the fence marker (NOT ```html)\n"
    "- HTML must be self-contained with inline styles\n"
    "- Dark theme: bg #0a0a0a, text #e5e5e5, accent #14b8a6\n"
    "- For charts: use <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n"
    "- For tables: use inline CSS, no external stylesheets\n"
    "- The widget renders in a sandboxed iframe with allow-scripts\n"
    "- Always output the widget, never say 'I can\\'t create charts'"
)

# Accept both ```html-widget and ```html blocks
_WIDGET_PATTERN = re.compile(
    r"```(?:html-widget|html)\s*\n(.*?)```",
    re.DOTALL,
)


def extract_widget_blocks(text: str) -> list[tuple[str, str]]:
    """Extract widget blocks from LLM output.

    Returns list of (html_content, remaining_text).
    """
    results: list[tuple[str, str]] = []
    remaining = text
    for match in _WIDGET_PATTERN.finditer(text):
        html = match.group(1).strip()
        # Skip if it's just code explanation (< 50 chars or no tags)
        if len(html) < 50 or "<" not in html:
            continue
        remaining = remaining.replace(match.group(0), "", 1).strip()
        results.append((html, remaining))
    return results


def inject_generative_ui_context(system_parts: list[str]) -> list[str]:
    """Append generative UI instructions to the system prompt parts list."""
    system_parts.append(GENERATIVE_UI_INSTRUCTION)
    return system_parts
