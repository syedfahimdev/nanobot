"""Health check system — /doctor command. Pure code, no LLM."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from loguru import logger


async def run_doctor(
    workspace: Path,
    toolsdns_config: Any = None,
    provider: Any = None,
    model: str | None = None,
) -> str:
    """Run all health checks and return a formatted report."""
    checks: list[tuple[str, bool, str]] = []

    checks.append(_check_disk(workspace))
    checks.append(_check_sessions(workspace))
    checks.append(_check_memory(workspace))
    checks.append(await _check_toolsdns(toolsdns_config))
    checks.append(await _check_llm(provider, model))

    lines = ["Health Check Results:", ""]
    all_ok = True
    for name, ok, detail in checks:
        icon = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        lines.append(f"  [{icon}] {name}: {detail}")

    if all_ok:
        lines.append("\nAll systems operational.")
    else:
        lines.append("\nSome checks failed — review above.")

    return "\n".join(lines)


def _check_disk(workspace: Path) -> tuple[str, bool, str]:
    try:
        usage = shutil.disk_usage(str(workspace))
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        pct = (usage.used / usage.total) * 100
        ok = free_gb > 1.0
        return ("Disk Space", ok, f"{free_gb:.1f}GB free / {total_gb:.1f}GB total ({pct:.0f}% used)")
    except Exception as e:
        return ("Disk Space", False, str(e))


def _check_sessions(workspace: Path) -> tuple[str, bool, str]:
    sessions_dir = workspace / "sessions"
    if not sessions_dir.exists():
        return ("Sessions", True, "No sessions directory")
    files = list(sessions_dir.glob("*.jsonl"))
    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)
    ok = size_mb < 500  # warn if sessions exceed 500MB
    return ("Sessions", ok, f"{len(files)} sessions, {size_mb:.1f}MB total")


def _check_memory(workspace: Path) -> tuple[str, bool, str]:
    history = workspace / "memory" / "HISTORY.md"
    memory = workspace / "memory" / "MEMORY.md"
    parts = []
    if history.exists():
        size_kb = history.stat().st_size / 1024
        parts.append(f"HISTORY.md {size_kb:.0f}KB")
    if memory.exists():
        size_kb = memory.stat().st_size / 1024
        parts.append(f"MEMORY.md {size_kb:.0f}KB")
    detail = ", ".join(parts) if parts else "No memory files"
    # Warn if HISTORY.md > 500KB
    ok = not history.exists() or history.stat().st_size < 512_000
    return ("Memory", ok, detail)


async def _check_toolsdns(config: Any) -> tuple[str, bool, str]:
    if not config or not getattr(config, "enabled", False):
        return ("ToolsDNS", False, "Not configured")
    try:
        import httpx
        url = config.url.rstrip("/")
        headers = {"Authorization": f"Bearer {config.api_key}"}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/v1/health", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                tools = data.get("total_tools", "?")
                return ("ToolsDNS", True, f"Online — {tools} tools indexed")
            return ("ToolsDNS", False, f"HTTP {resp.status_code}")
    except Exception as e:
        return ("ToolsDNS", False, f"Unreachable: {e}")


async def _check_llm(provider: Any, model: str | None) -> tuple[str, bool, str]:
    if not provider:
        return ("LLM Provider", False, "No provider configured")
    try:
        # Tiny completion to verify connectivity
        resp = await provider.chat_with_retry(
            messages=[{"role": "user", "content": "Reply with OK"}],
            tools=[],
            model=model,
        )
        if resp and resp.content:
            return ("LLM Provider", True, f"Responding ({model or 'default'})")
        return ("LLM Provider", False, "Empty response")
    except Exception as e:
        return ("LLM Provider", False, str(e)[:80])
