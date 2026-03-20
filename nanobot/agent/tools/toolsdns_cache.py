"""ToolsDNS schema cache + health monitor.

Features:
  - Schema caching with configurable TTL (avoids re-fetching every call)
  - Health monitoring with periodic checks
  - Batch tool hints fetching
  - Confidence filtering on search results
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from loguru import logger

_SCHEMA_TTL_S = 300  # 5 minutes
_HEALTH_CHECK_INTERVAL_S = 120  # 2 minutes
_HINTS_BATCH_SIZE = 10


class ToolsDNSCache:
    """Shared cache for ToolsDNS schemas, health status, and hints."""

    def __init__(self, base_url: str, api_key: str):
        self._url = base_url.rstrip("/")
        self._api_key = api_key
        self._schemas: dict[str, tuple[dict, float]] = {}  # tool_id → (schema, expires_at)
        self._hints: dict[str, tuple[list, float]] = {}  # tool_id → (hints, expires_at)
        self._health: dict[str, Any] = {"status": "unknown", "last_check": 0, "total_tools": 0}
        self._health_task: asyncio.Task | None = None

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    # ── Schema Cache ──

    async def get_schema(self, tool_id: str) -> dict:
        """Get tool schema, using cache if available."""
        now = time.time()
        cached = self._schemas.get(tool_id)
        if cached and cached[1] > now:
            return cached[0]

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._url}/v1/tool/{tool_id}", headers=self.headers)
                resp.raise_for_status()
                data = resp.json()
                schema = data.get("input_schema", data.get("inputSchema", {}))
                self._schemas[tool_id] = (schema, now + _SCHEMA_TTL_S)
                return schema
        except Exception as e:
            logger.debug("Schema cache miss for {}: {}", tool_id, e)
            return {}

    def get_valid_params(self, tool_id: str) -> set[str] | None:
        """Get cached valid params for a tool (sync, from cache only)."""
        cached = self._schemas.get(tool_id)
        if cached and cached[1] > time.time():
            return set(cached[0].get("properties", {}).keys())
        return None

    # ── Health Monitor ──

    async def check_health(self) -> dict[str, Any]:
        """Check ToolsDNS health and return status."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._url}/v1/health", headers=self.headers)
                if resp.status_code == 200:
                    data = resp.json()
                    self._health = {
                        "status": "ok",
                        "last_check": time.time(),
                        "total_tools": data.get("total_tools", 0),
                        "latency_ms": resp.elapsed.total_seconds() * 1000 if resp.elapsed else 0,
                    }
                else:
                    self._health = {"status": "degraded", "last_check": time.time(), "http_code": resp.status_code}
        except Exception as e:
            self._health = {"status": "down", "last_check": time.time(), "error": str(e)}

        return self._health

    def get_health(self) -> dict[str, Any]:
        """Return cached health status."""
        return self._health

    async def start_health_monitor(self) -> None:
        """Start periodic health checks."""
        if self._health_task:
            return

        async def _loop():
            while True:
                try:
                    await self.check_health()
                except Exception:
                    pass
                await asyncio.sleep(_HEALTH_CHECK_INTERVAL_S)

        self._health_task = asyncio.create_task(_loop())
        # Initial check
        await self.check_health()

    def stop_health_monitor(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            self._health_task = None

    # ── Batch Hints ──

    async def get_hints_batch(self, tool_ids: list[str]) -> dict[str, list]:
        """Fetch hints for multiple tools at once."""
        now = time.time()
        # Return cached where available
        result = {}
        uncached = []
        for tid in tool_ids:
            cached = self._hints.get(tid)
            if cached and cached[1] > now:
                result[tid] = cached[0]
            else:
                uncached.append(tid)

        if not uncached:
            return result

        # Fetch uncached in batches
        for i in range(0, len(uncached), _HINTS_BATCH_SIZE):
            batch = uncached[i:i + _HINTS_BATCH_SIZE]
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{self._url}/v1/tool-hints",
                        headers=self.headers,
                        json={"agent_id": "mawa", "tool_ids": batch},
                    )
                    resp.raise_for_status()
                    hints = resp.json().get("hints", {})
                    for tid in batch:
                        h = hints.get(tid, [])
                        self._hints[tid] = (h, now + _SCHEMA_TTL_S)
                        result[tid] = h
            except Exception as e:
                logger.debug("Batch hints fetch failed: {}", e)
                for tid in batch:
                    result[tid] = []

        return result

    # ── Confidence Filtering ──

    @staticmethod
    def filter_by_confidence(results: list[dict], min_confidence: float = 0.3) -> list[dict]:
        """Filter search results below a confidence threshold."""
        return [r for r in results if r.get("score", r.get("confidence", 0)) >= min_confidence]

    # ── Stats ──

    def get_stats(self) -> dict:
        return {
            "cached_schemas": len(self._schemas),
            "cached_hints": len(self._hints),
            "health": self._health,
        }


# Global singleton per base_url
_instances: dict[str, ToolsDNSCache] = {}


def get_cache(base_url: str, api_key: str) -> ToolsDNSCache:
    """Get or create a ToolsDNS cache instance."""
    key = base_url.rstrip("/")
    if key not in _instances:
        _instances[key] = ToolsDNSCache(key, api_key)
    return _instances[key]
