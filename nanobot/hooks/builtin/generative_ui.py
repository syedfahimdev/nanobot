"""Generative UI — system prompt injection for LLM-generated HTML widgets.

When the LLM wants to present data visually (charts, tables, dashboards),
it outputs a self-contained HTML block wrapped in ```html-widget markers.
The frontend extracts these blocks and renders them in sandboxed iframes.
"""

from __future__ import annotations

import re

GENERATIVE_UI_INSTRUCTION = (
    "## Generative UI\n\n"
    "When showing visual data (charts, tables, comparisons, dashboards, diagrams, maps, timelines), "
    "output a self-contained HTML block:\n\n"
    "```html-widget\n"
    "<div style=\"background:#0a0a0a;color:#e5e5e5;padding:16px;border-radius:12px;font-family:system-ui\">\n"
    "  <!-- your HTML here -->\n"
    "</div>\n"
    "```\n\n"
    "Libraries available (CDN):\n"
    "- Charts: `<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>` (bar, line, pie, doughnut, radar)\n"
    "- Rich charts: `<script src=\"https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js\"></script>` (heatmaps, gauges, treemaps, geo)\n"
    "- Diagrams: `<script src=\"https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js\"></script>` (flowcharts, sequence, gantt, mindmap)\n"
    "- Interactive charts: `<script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>` (3D, scatter, heatmap, candlestick)\n"
    "- Math: `<script src=\"https://cdn.jsdelivr.net/npm/katex@0.16/dist/katex.min.js\"></script>` + CSS\n\n"
    "Rules:\n"
    "- Use ```html-widget (NOT ```html)\n"
    "- Self-contained: inline styles, CDN scripts only\n"
    "- Dark theme: bg #0a0a0a, text #e5e5e5, accent #14b8a6, warning #f59e0b\n"
    "- Mobile-friendly: use % widths, responsive layout\n"
    "- Always render the widget, never say 'I can't create charts'\n"
    "- For tool results with structured data, ALWAYS visualize with a widget\n"
    "- The weather, stock_chart, compare, timeline, show_map, system_monitor tools return pre-built widgets — pass them through as-is"
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
