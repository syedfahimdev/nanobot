"""Background job completion notifier.

Checks for completed background jobs after each turn and sends
a notification to the user if any jobs finished while they were talking.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.hooks.events import TurnCompleted


_notified_jobs: set[str] = set()


def make_background_job_hook(bus: MessageBus):
    """Create a turn_completed hook that notifies on background job completion."""

    async def _check_jobs(event: TurnCompleted) -> TurnCompleted:
        try:
            from nanobot.agent.tools.background_shell import _jobs

            for jid, job in _jobs.items():
                if job.is_running or jid in _notified_jobs:
                    continue
                _notified_jobs.add(jid)

                # Build notification
                status = "completed successfully" if job.returncode == 0 else f"failed (exit code {job.returncode})"
                tail = job.stdout_buf.decode("utf-8", errors="replace")[-300:].strip()
                err = job.stderr_buf.decode("utf-8", errors="replace")[-200:].strip()

                lines = [f"Background job **{jid}** {status}"]
                lines.append(f"Command: `{job.command[:80]}`")
                lines.append(f"Duration: {job.elapsed:.0f}s")
                if tail:
                    lines.append(f"Output: {tail[:200]}")
                if err and job.returncode != 0:
                    lines.append(f"Error: {err[:150]}")

                content = "\n".join(lines)

                # Send notification to the channel where the turn happened
                if event.channel and event.chat_id:
                    await bus.publish_outbound(OutboundMessage(
                        channel=event.channel,
                        chat_id=event.chat_id,
                        content=content,
                        metadata={"_notification": True, "_background_job": True},
                    ))
                    logger.info("Notified user about background job {} completion", jid)

        except Exception as e:
            logger.debug("Background job check error: {}", e)

        return event

    return _check_jobs
