"""Multi-modal memory — remember images, receipts, and documents.

When the agent sees an image or document, it can save a structured summary
to memory/MEDIA.md with a reference to the original file. These summaries
are searchable via memory_search (indexed by ToolsDNS).

Stored format:
  ## [2026-03-20 14:30] Receipt — Amazon order
  - **File:** /path/to/receipt.png
  - **Type:** receipt
  - **Amount:** $42.99
  - **Vendor:** Amazon
  - **Key details:** Order #123-456, shipped to CT address
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


class MediaMemoryTool(Tool):
    """Save structured summaries of images, receipts, and documents to memory."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._media_file = workspace / "memory" / "MEDIA.md"
        self._media_dir = workspace / "memory" / "media"

    @property
    def name(self) -> str:
        return "remember_media"

    @property
    def description(self) -> str:
        return (
            "Save a structured summary of an image, receipt, document, or screenshot to memory. "
            "Use after viewing media to remember key information (amounts, dates, names, content). "
            "The summary becomes searchable via memory_search."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the original media file",
                },
                "media_type": {
                    "type": "string",
                    "enum": ["receipt", "screenshot", "document", "photo", "diagram", "other"],
                    "description": "Type of media",
                },
                "title": {
                    "type": "string",
                    "description": "Short descriptive title (e.g., 'Amazon receipt', 'Architecture diagram')",
                },
                "summary": {
                    "type": "string",
                    "description": "Structured summary of key information extracted from the media",
                },
                "extracted_data": {
                    "type": "object",
                    "description": "Key-value pairs extracted from the media (e.g., amount, vendor, date, names)",
                },
            },
            "required": ["media_type", "title", "summary"],
        }

    async def execute(
        self,
        media_type: str,
        title: str,
        summary: str,
        file_path: str = "",
        extracted_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._media_file.parent.mkdir(parents=True, exist_ok=True)

        # Optionally copy media to persistent storage
        stored_path = ""
        if file_path:
            src = Path(file_path)
            if src.exists():
                self._media_dir.mkdir(parents=True, exist_ok=True)
                dest = self._media_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{src.name}"
                try:
                    shutil.copy2(src, dest)
                    stored_path = str(dest)
                except OSError as e:
                    logger.debug("Failed to copy media: {}", e)
                    stored_path = file_path
            else:
                stored_path = file_path

        # Build the memory entry
        lines = [f"\n## [{ts}] {media_type.title()} — {title}"]
        if stored_path:
            lines.append(f"- **File:** {stored_path}")
        lines.append(f"- **Type:** {media_type}")

        if extracted_data:
            for key, value in extracted_data.items():
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

        lines.append(f"- **Summary:** {summary}")

        entry = "\n".join(lines) + "\n"

        # Append to MEDIA.md
        with open(self._media_file, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.info("Media memory saved: {} — {}", media_type, title)
        return f"Remembered {media_type}: {title}"
