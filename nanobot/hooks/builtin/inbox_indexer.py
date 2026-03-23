"""Inbox indexer hook — auto-indexes uploaded files for local search.

When files are added to the inbox folders, this hook extracts text
and optionally generates an LLM summary stored alongside the file.

Runs on turn_completed to batch-process any new files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.inbox import extract_file_text, VALID_FOLDERS
from nanobot.hooks.events import TurnCompleted

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_MAX_CHUNK_CHARS = 2000
_STATE_FILE_NAME = ".inbox_index_state.json"

_SUMMARIZE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize the document for indexing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A concise summary of the document's key content, suitable for search indexing. 2-3 sentences.",
                    },
                    "keywords": {
                        "type": "string",
                        "description": "Comma-separated keywords for search relevance.",
                    },
                },
                "required": ["summary"],
            },
        },
    }
]

# Binary formats that benefit from LLM summarization
_SUMMARIZE_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".xls"})


class InboxIndexer:
    """Watches inbox folders and indexes new/changed files."""

    def __init__(self, workspace: Path, provider: LLMProvider, model: str):
        self._workspace = workspace
        self._provider = provider
        self._model = model
        self._inbox_dir = workspace / "inbox"
        self._state_file = self._inbox_dir / _STATE_FILE_NAME
        self._state = self._load_state()

    def _load_state(self) -> dict[str, str]:
        """Load file hash state to detect changes."""
        try:
            if self._state_file.exists():
                raw = self._state_file.read_text(encoding="utf-8")
                if isinstance(raw, str) and raw.strip():
                    return json.loads(raw)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return {}

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _scan_new_files(self) -> list[tuple[str, Path]]:
        """Find files that are new or changed since last index."""
        new_files = []
        for folder in VALID_FOLDERS:
            folder_path = self._inbox_dir / folder
            if not folder_path.exists():
                continue
            for f in folder_path.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    key = f"{folder}/{f.name}"
                    current_hash = self._file_hash(f)
                    if self._state.get(key) != current_hash:
                        new_files.append((key, f))
                        self._state[key] = current_hash
        return new_files

    async def _summarize_file(self, text: str, file_name: str) -> tuple[str, str]:
        """Use LLM to generate a summary and keywords for a binary file."""
        truncated = text[:3000]  # Keep token usage low
        try:
            response = await self._provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": "Summarize this document for search indexing. Be concise."},
                    {"role": "user", "content": f"File: {file_name}\n\nContent:\n{truncated}"},
                ],
                tools=_SUMMARIZE_TOOL,
                model=self._model,
                tool_choice={"type": "function", "function": {"name": "summarize"}},
            )
            if response.has_tool_calls:
                args = response.tool_calls[0].arguments
                if isinstance(args, str):
                    args = json.loads(args)
                return args.get("summary", ""), args.get("keywords", "")
        except Exception:
            logger.opt(exception=True).debug("Inbox summarization failed for {}", file_name)
        return "", ""

    def _chunk_text(self, text: str, key: str) -> list[dict[str, Any]]:
        """Split text into chunks for indexing."""
        chunks = []
        # Split by paragraphs, then merge up to max size
        paragraphs = text.split("\n\n")
        current = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current) + len(para) > _MAX_CHUNK_CHARS and current:
                chunks.append({
                    "chunk_id": f"inbox__{key.replace('/', '__')}__{chunk_idx}",
                    "title": key,
                    "content": current.strip(),
                    "file_path": str(self._inbox_dir / key),
                    "section": f"chunk {chunk_idx}",
                })
                chunk_idx += 1
                current = para + "\n\n"
            else:
                current += para + "\n\n"

        if current.strip():
            chunks.append({
                "chunk_id": f"inbox__{key.replace('/', '__')}__{chunk_idx}",
                "title": key,
                "content": current.strip(),
                "file_path": str(self._inbox_dir / key),
                "section": f"chunk {chunk_idx}",
            })

        return chunks

    async def index_new_files(self) -> int:
        """Scan for new files, extract, optionally summarize, and index."""
        new_files = self._scan_new_files()
        if not new_files:
            return 0

        all_chunks: list[dict[str, Any]] = []

        for key, path in new_files:
            text = extract_file_text(path)
            if not text or text.startswith("["):
                continue

            # For binary formats, generate LLM summary
            if path.suffix.lower() in _SUMMARIZE_EXTENSIONS:
                summary, keywords = await self._summarize_file(text, key)
                if summary:
                    # Prepend summary as a high-quality chunk
                    all_chunks.append({
                        "chunk_id": f"inbox__{key.replace('/', '__')}__summary",
                        "title": f"{key} (summary)",
                        "content": f"{summary}\n\nKeywords: {keywords}" if keywords else summary,
                        "file_path": str(path),
                        "section": "summary",
                    })

            # Chunk the raw text
            chunks = self._chunk_text(text, key)
            all_chunks.extend(chunks)
            logger.info("Inbox indexed: {} ({} chunks)", key, len(chunks))

        self._save_state()
        return len(new_files)


def make_inbox_indexer_hook(
    workspace: Path,
    provider: "LLMProvider",
    model: str,
):
    """Create an inbox indexer hook that runs on turn_completed."""
    indexer = InboxIndexer(workspace, provider, model)
    _call_count = [0]

    async def on_turn_completed(event: TurnCompleted) -> None:
        # Only check every 5 turns to avoid excessive scanning
        _call_count[0] += 1
        if _call_count[0] % 5 != 0:
            return
        count = await indexer.index_new_files()
        if count > 0:
            logger.info("Inbox indexer: indexed {} new files", count)

    return on_turn_completed
