"""Re-index memory files when they're written/edited by the agent."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from nanobot.hooks.events import ToolAfter

_MEMORY_DIRS = ("knowledge/", "learnings/", "rules/", "memory/")


def make_memory_reindex_hook(workspace: Path, toolsdns_url: str, api_key: str):
    """Create a hook that re-indexes a file when the agent writes to memory dirs."""
    _indexer = None  # lazy init

    async def memory_reindex(event: ToolAfter) -> None:
        if event.name not in ("write_file", "edit_file"):
            return
        file_path = event.params.get("path", event.params.get("file_path", ""))
        if not any(d in file_path for d in _MEMORY_DIRS):
            return
        if event.error:
            return

        nonlocal _indexer
        if _indexer is None:
            from nanobot.memory.indexer import MemoryIndexer
            _indexer = MemoryIndexer(workspace, toolsdns_url, api_key)

        p = Path(file_path)
        if p.exists():
            count = await _indexer.index_file(p)
            if count:
                logger.info("Memory reindex: {} chunks from {}", count, p.name)

    return memory_reindex
