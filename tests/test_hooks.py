"""Tests for the hook system."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nanobot.hooks.engine import HookEngine, HookMode
from nanobot.hooks.events import ToolBefore, ToolAfter


class TestHookEngine:
    @pytest.mark.asyncio
    async def test_fire_and_forget_doesnt_block(self):
        engine = HookEngine()
        called = []

        async def hook(event):
            called.append(event.name)

        engine.on("tool_after", hook)
        event = ToolAfter(name="test", params={}, result="ok")
        await engine.emit("tool_after", event)
        await asyncio.sleep(0.05)  # let fire-and-forget run
        assert called == ["test"]

    @pytest.mark.asyncio
    async def test_blocking_hook_can_deny(self):
        engine = HookEngine()

        async def deny_all(event: ToolBefore) -> ToolBefore:
            event.denied = True
            event.deny_reason = "Blocked by test"
            return event

        engine.on("tool_before", deny_all, mode=HookMode.BLOCKING)
        event = ToolBefore(name="dangerous", params={})
        result = await engine.emit("tool_before", event)
        assert result.denied is True
        assert result.deny_reason == "Blocked by test"

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        engine = HookEngine()
        order = []

        async def first(e):
            order.append("first")
            return e

        async def second(e):
            order.append("second")
            return e

        engine.on("tool_before", second, mode=HookMode.BLOCKING, priority=200)
        engine.on("tool_before", first, mode=HookMode.BLOCKING, priority=10)
        await engine.emit("tool_before", ToolBefore(name="x", params={}))
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_no_hooks_passthrough(self):
        engine = HookEngine()
        event = ToolBefore(name="x", params={})
        result = await engine.emit("tool_before", event)
        assert result is event

    @pytest.mark.asyncio
    async def test_failing_hook_doesnt_crash(self):
        engine = HookEngine()

        async def bad_hook(event):
            raise ValueError("boom")

        engine.on("tool_after", bad_hook)
        # Should not raise
        await engine.emit("tool_after", ToolAfter(name="x", params={}, result="ok"))


class TestApprovalGuard:
    def test_resolve_approval_yes(self):
        from nanobot.hooks.builtin.approval import resolve_approval, _pending
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        _pending["test:123"] = future
        assert resolve_approval("test:123", "yes") is True
        assert future.result() is True
        _pending.pop("test:123", None)
        loop.close()

    def test_resolve_approval_no(self):
        from nanobot.hooks.builtin.approval import resolve_approval, _pending
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        _pending["test:456"] = future
        assert resolve_approval("test:456", "no") is True
        assert future.result() is False
        _pending.pop("test:456", None)
        loop.close()

    def test_resolve_approval_irrelevant(self):
        from nanobot.hooks.builtin.approval import resolve_approval
        assert resolve_approval("no_pending", "yes") is False

    def test_needs_approval(self):
        from nanobot.hooks.builtin.approval import _needs_approval
        assert _needs_approval("GMAIL_SEND_EMAIL") is True
        assert _needs_approval("GMAIL_SEND_DRAFT") is True
        assert _needs_approval("DISCORDBOT_DELETE_CHANNEL") is True
        assert _needs_approval("GMAIL_FETCH_EMAILS") is False
        assert _needs_approval("WEATHERMAP_WEATHER") is False
        assert _needs_approval("GOOGLECALENDAR_FIND_EVENT") is False


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_doctor_runs(self, tmp_path):
        from nanobot.hooks.builtin.health import run_doctor
        result = await run_doctor(workspace=tmp_path)
        assert "Health Check Results" in result
        assert "[OK] Disk Space" in result
