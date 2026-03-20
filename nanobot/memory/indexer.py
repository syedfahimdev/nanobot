"""Memory indexer — chunks markdown files and pushes to ToolsDNS for semantic search.

Walks knowledge/, learnings/, rules/, memory/ directories,
splits by headings, and upserts as searchable entries in ToolsDNS.
Incremental: skips unchanged files via content hashing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_MEMORY_DIRS = ["knowledge", "learnings", "rules", "memory"]
_MAX_CHUNK_CHARS = 2000


class MemoryIndexer:
    """Chunks workspace markdown files and pushes to ToolsDNS for vector search."""

    def __init__(self, workspace: Path, toolsdns_url: str, api_key: str) -> None:
        self._workspace = workspace
        self._url = toolsdns_url.rstrip("/")
        self._api_key = api_key
        self._state_file = workspace / "memory" / ".memory_index_state.json"
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self._state_file.exists():
            try:
                return json.loads(self._state_file.read_text())
            except Exception:
                pass
        return {"file_hashes": {}, "history_watermark": 0}

    def _save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(self._state, indent=2))

    def _scan_files(self) -> list[Path]:
        """Find all .md files in memory directories."""
        files = []
        for dirname in _MEMORY_DIRS:
            d = self._workspace / dirname
            if d.exists():
                files.extend(d.rglob("*.md"))
        return sorted(files)

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50]

    @staticmethod
    def _file_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _chunk_file(self, path: Path) -> list[dict[str, Any]]:
        """Split a markdown file into chunks by headings."""
        content = path.read_text(encoding="utf-8")
        rel = path.relative_to(self._workspace)
        file_slug = self._slugify(str(rel).replace("/", "_").replace(".md", ""))

        # Find all headings
        headings = list(_HEADING_RE.finditer(content))

        if not headings:
            # No headings — treat entire file as one chunk
            chunk_id = f"memory__{file_slug}"
            return [{
                "chunk_id": chunk_id,
                "title": path.stem.replace("-", " ").replace("_", " ").title(),
                "content": content[:_MAX_CHUNK_CHARS],
                "file_path": str(path),
                "section": "",
            }]

        chunks = []
        for i, match in enumerate(headings):
            section_title = match.group(2).strip()
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            section_text = content[start:end].strip()

            if not section_text or len(section_text) < 10:
                continue

            section_slug = self._slugify(section_title)
            chunk_id = f"memory__{file_slug}__{section_slug}"
            title = f"{path.stem.replace('-', ' ').replace('_', ' ').title()} > {section_title}"

            # Split large sections
            if len(section_text) > _MAX_CHUNK_CHARS:
                paragraphs = section_text.split("\n\n")
                buffer = ""
                part = 0
                for para in paragraphs:
                    if len(buffer) + len(para) > _MAX_CHUNK_CHARS and buffer:
                        chunks.append({
                            "chunk_id": f"{chunk_id}_p{part}",
                            "title": f"{title} (part {part + 1})",
                            "content": buffer.strip(),
                            "file_path": str(path),
                            "section": section_title,
                        })
                        buffer = para
                        part += 1
                    else:
                        buffer = f"{buffer}\n\n{para}" if buffer else para
                if buffer.strip():
                    chunks.append({
                        "chunk_id": f"{chunk_id}_p{part}" if part > 0 else chunk_id,
                        "title": f"{title} (part {part + 1})" if part > 0 else title,
                        "content": buffer.strip(),
                        "file_path": str(path),
                        "section": section_title,
                    })
            else:
                chunks.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "content": section_text,
                    "file_path": str(path),
                    "section": section_title,
                })

        return chunks

    def _chunk_history(self, path: Path) -> list[dict[str, Any]]:
        """Chunk HISTORY.md incrementally from watermark."""
        lines = path.read_text(encoding="utf-8").split("\n")
        watermark = self._state.get("history_watermark", 0)

        if watermark >= len(lines):
            return []

        new_lines = lines[watermark:]
        self._state["history_watermark"] = len(lines)

        # Group by date entries
        chunks = []
        current_date = ""
        buffer = []

        for line in new_lines:
            date_match = re.match(r"\[(\d{4}-\d{2}-\d{2})", line)
            if date_match:
                new_date = date_match.group(1)
                if new_date != current_date and buffer:
                    text = "\n".join(buffer)
                    if len(text) > 20:
                        slug = self._slugify(current_date)
                        chunks.append({
                            "chunk_id": f"memory__history__{slug}_{len(chunks)}",
                            "title": f"History {current_date}",
                            "content": text[:_MAX_CHUNK_CHARS],
                            "file_path": str(path),
                            "section": current_date,
                        })
                    buffer = []
                current_date = new_date
            buffer.append(line)

        if buffer:
            text = "\n".join(buffer)
            if len(text) > 20:
                slug = self._slugify(current_date or "recent")
                chunks.append({
                    "chunk_id": f"memory__history__{slug}_{len(chunks)}",
                    "title": f"History {current_date or 'recent'}",
                    "content": text[:_MAX_CHUNK_CHARS],
                    "file_path": str(path),
                    "section": current_date,
                })

        return chunks

    async def index_file(self, path: Path) -> int:
        """Index a single file. Returns number of chunks indexed."""
        if not path.exists() or not path.suffix == ".md":
            return 0

        content = path.read_text(encoding="utf-8")
        content_hash = self._file_hash(content)
        path_str = str(path)

        # Skip if unchanged (except HISTORY.md which uses watermark)
        if "HISTORY.md" not in path.name:
            if self._state["file_hashes"].get(path_str) == content_hash:
                return 0

        if "HISTORY.md" in path.name:
            chunks = self._chunk_history(path)
        else:
            chunks = self._chunk_file(path)

        if not chunks:
            return 0

        count = await self._push_chunks(chunks)
        self._state["file_hashes"][path_str] = content_hash
        self._save_state()
        return count

    async def index_all(self) -> int:
        """Full scan — index all memory files. Skips unchanged ones. Detects deletes."""
        files = self._scan_files()
        current_paths = {str(f) for f in files}
        total = 0
        all_chunks: list[dict] = []

        # Detect deleted files — remove from state
        deleted = [p for p in list(self._state["file_hashes"]) if p not in current_paths]
        if deleted:
            for p in deleted:
                del self._state["file_hashes"][p]
            logger.info("Memory indexer: {} files deleted, removed from index state", len(deleted))

        for f in files:
            content = f.read_text(encoding="utf-8")
            content_hash = self._file_hash(content)
            path_str = str(f)

            if "HISTORY.md" in f.name:
                chunks = self._chunk_history(f)
            else:
                if self._state["file_hashes"].get(path_str) == content_hash:
                    continue
                chunks = self._chunk_file(f)
                self._state["file_hashes"][path_str] = content_hash

            all_chunks.extend(chunks)

        if all_chunks:
            total = await self._push_chunks(all_chunks)
            self._save_state()
            logger.info("Memory indexer: indexed {} chunks from {} files", total, len(set(c["file_path"] for c in all_chunks)))
        elif deleted:
            self._save_state()

        return total

    async def _push_chunks(self, chunks: list[dict]) -> int:
        """Push chunks to ToolsDNS /v1/memory/ingest in batches."""
        total = 0
        batch_size = 20  # Embed 20 chunks at a time to avoid timeouts
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    resp = await client.post(
                        f"{self._url}/v1/memory/ingest",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={"chunks": batch},
                    )
                    resp.raise_for_status()
                    total += resp.json().get("indexed", 0)
        except Exception as e:
            logger.error("Memory indexer push failed: {}", e)
        return total
