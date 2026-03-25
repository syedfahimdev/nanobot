"""Background shell tool — run long commands detached, check status, get output.

Allows Mawa to:
1. Launch a command in the background (returns immediately with a job ID)
2. Check if a job is still running
3. Get the output of a completed job
4. List all running/completed jobs
5. Kill a running job

This enables workflows like:
  - "run this build in the background and let me know when it's done"
  - "deploy this and check back in a minute"
  - "run these 3 commands in parallel"
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

_DENY_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\b",
    r"\bdel\s+/[fq]\b",
    r"\brmdir\s+/s\b",
    r"(?:^|[;&|]\s*)format\b",
    r"\b(mkfs|diskpart)\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r"\b(shutdown|reboot|poweroff)\b",
    r"\bpkill\b",
    r"\bkillall\b",
    r"\bkill\s+-(?:9|SIGKILL)\b",
    r":\(\)\s*\{.*\};\s*:",
    r"\.env\b",
    r"\bprintenv\b",
    r"\b(env|set)\s*$",
    r"/etc/(shadow|passwd)",
]


@dataclass
class BackgroundJob:
    id: str
    command: str
    pid: int
    started_at: float
    working_dir: str
    process: asyncio.subprocess.Process
    stdout_buf: bytearray = field(default_factory=bytearray)
    stderr_buf: bytearray = field(default_factory=bytearray)
    returncode: int | None = None
    finished_at: float | None = None
    _reader_task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def is_running(self) -> bool:
        return self.returncode is None

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def status(self) -> str:
        if self.is_running:
            return f"running ({self.elapsed:.0f}s)"
        return f"exited {self.returncode} ({self.elapsed:.0f}s)"


_jobs: dict[str, BackgroundJob] = {}
_job_counter = 0
_MAX_OUTPUT = 15_000
_MAX_JOBS = 10


async def _read_stream(stream: asyncio.StreamReader | None, buf: bytearray) -> None:
    """Read a stream into a buffer until EOF."""
    if not stream:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > _MAX_OUTPUT * 2:
            excess = len(buf) - _MAX_OUTPUT
            del buf[:excess]


async def _monitor_job(job: BackgroundJob) -> None:
    """Monitor a background job — read output and track completion."""
    try:
        await asyncio.gather(
            _read_stream(job.process.stdout, job.stdout_buf),
            _read_stream(job.process.stderr, job.stderr_buf),
        )
        job.returncode = await job.process.wait()
        job.finished_at = time.time()
        logger.info("Background job {} finished (rc={})", job.id, job.returncode)
    except asyncio.CancelledError:
        job.process.kill()
        job.returncode = -9
        job.finished_at = time.time()
    except Exception as e:
        logger.error("Background job {} monitor error: {}", job.id, e)
        job.returncode = -1
        job.finished_at = time.time()


class BackgroundShellTool(Tool):
    """Run shell commands in the background with job management."""

    def __init__(self, working_dir: str | None = None, path_append: str = ""):
        self._working_dir = working_dir
        self._path_append = path_append

    @property
    def name(self) -> str:
        return "background_exec"

    @property
    def description(self) -> str:
        return (
            "Run shell commands in the background (detached). "
            "Actions: 'run' (start background job, returns job ID), "
            "'status' (check if job is running), "
            "'output' (get stdout/stderr of a job), "
            "'list' (show all jobs), "
            "'kill' (stop a running job). "
            "Use for long-running commands like builds, deployments, downloads."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "status", "output", "list", "kill"],
                    "description": "Action to perform",
                },
                "command": {
                    "type": "string",
                    "description": "Shell command to run (required for 'run' action)",
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (required for 'status', 'output', 'kill')",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory (optional, for 'run' action)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, action: str, command: str = "", job_id: str = "",
        working_dir: str | None = None, **kwargs: Any,
    ) -> str:
        global _job_counter

        if action == "run":
            if not command:
                return "Error: 'command' is required for 'run' action"

            lower = command.strip().lower()
            for pattern in _DENY_PATTERNS:
                if re.search(pattern, lower):
                    return "Error: Command blocked by safety guard (dangerous pattern detected)"

            running = [j for j in _jobs.values() if j.is_running]
            if len(running) >= _MAX_JOBS:
                return f"Error: Max {_MAX_JOBS} concurrent background jobs. Kill one first."

            cwd = working_dir or self._working_dir or os.getcwd()
            env = os.environ.copy()
            if self._path_append:
                env["PATH"] = env.get("PATH", "") + os.pathsep + self._path_append

            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            except Exception as e:
                return f"Error: Failed to start command: {e}"

            _job_counter += 1
            jid = f"bg_{_job_counter}"
            job = BackgroundJob(
                id=jid, command=command, pid=process.pid,
                started_at=time.time(), working_dir=cwd, process=process,
            )
            job._reader_task = asyncio.create_task(_monitor_job(job))
            _jobs[jid] = job

            logger.info("Background job {} started: {} (pid={})", jid, command[:80], process.pid)
            return f"Job {jid} started (pid={process.pid}). Use background_exec(action='status', job_id='{jid}') to check progress."

        elif action == "status":
            if not job_id:
                return "Error: 'job_id' is required for 'status' action"
            job = _jobs.get(job_id)
            if not job:
                return f"Error: Job '{job_id}' not found. Use action='list' to see all jobs."

            lines = [
                f"Job: {job.id}",
                f"Command: {job.command[:100]}",
                f"Status: {job.status}",
                f"Working dir: {job.working_dir}",
            ]
            if not job.is_running:
                out = job.stdout_buf.decode("utf-8", errors="replace")[-500:]
                if out:
                    lines.append(f"Output (tail): {out}")
                err = job.stderr_buf.decode("utf-8", errors="replace")[-200:]
                if err.strip():
                    lines.append(f"Stderr (tail): {err}")
            else:
                out = job.stdout_buf.decode("utf-8", errors="replace")[-300:]
                if out:
                    lines.append(f"Output so far (tail): {out}")

            return "\n".join(lines)

        elif action == "output":
            if not job_id:
                return "Error: 'job_id' is required for 'output' action"
            job = _jobs.get(job_id)
            if not job:
                return f"Error: Job '{job_id}' not found."

            parts = []
            stdout = job.stdout_buf.decode("utf-8", errors="replace")
            if stdout:
                if len(stdout) > _MAX_OUTPUT:
                    half = _MAX_OUTPUT // 2
                    stdout = stdout[:half] + f"\n... ({len(stdout) - _MAX_OUTPUT} chars truncated) ...\n" + stdout[-half:]
                parts.append(stdout)

            stderr = job.stderr_buf.decode("utf-8", errors="replace")
            if stderr.strip():
                parts.append(f"STDERR:\n{stderr[-2000:]}")

            parts.append(f"\nStatus: {job.status}")
            return "\n".join(parts) if parts else "(no output yet)"

        elif action == "list":
            if not _jobs:
                return "No background jobs."

            lines = []
            for j in sorted(_jobs.values(), key=lambda x: x.started_at, reverse=True):
                icon = "..." if j.is_running else ("ok" if j.returncode == 0 else "err")
                lines.append(f"[{icon}] {j.id}: {j.command[:60]} — {j.status}")
            return "\n".join(lines)

        elif action == "kill":
            if not job_id:
                return "Error: 'job_id' is required for 'kill' action"
            job = _jobs.get(job_id)
            if not job:
                return f"Error: Job '{job_id}' not found."
            if not job.is_running:
                return f"Job {job_id} already finished ({job.status})"

            job.process.kill()
            try:
                await asyncio.wait_for(job.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            job.returncode = -9
            job.finished_at = time.time()
            return f"Job {job_id} killed."

        else:
            return f"Error: Unknown action '{action}'. Use: run, status, output, list, kill"
