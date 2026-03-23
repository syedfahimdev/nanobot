"""Tests for the 10 voice-first AI improvements."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Improvement #2: Skip summarization for simple subagent results ──

class TestSkipSummarization:
    @pytest.mark.asyncio
    async def test_simple_result_publishes_directly(self):
        """Short single-line results bypass LLM summarization."""
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        await mgr._announce_result(
            task_id="t1", label="Send email", task="send email to John",
            result="Email sent successfully.", origin={"channel": "web_voice", "chat_id": "s1"},
            status="ok",
        )

        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "sent successfully" in out.content
        assert out.metadata.get("_subagent_result") is True

    @pytest.mark.asyncio
    async def test_complex_result_uses_summarization(self):
        """Multi-line or structured results go through LLM summarization."""
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        complex_result = "Here are your emails:\n1. From Alice: Meeting tomorrow\n2. From Bob: Report attached"
        await mgr._announce_result(
            task_id="t2", label="Check email", task="check my email",
            result=complex_result, origin={"channel": "web_voice", "chat_id": "s1"},
            status="ok",
        )

        # Complex result should go to inbound (for LLM summarization)
        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert "Subagent" in msg.content
        assert "check my email" in msg.content

    @pytest.mark.asyncio
    async def test_error_result_always_summarized(self):
        """Error results always go through summarization."""
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        await mgr._announce_result(
            task_id="t3", label="Task", task="do something",
            result="Error: connection failed", origin={"channel": "test", "chat_id": "c1"},
            status="error",
        )

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert "failed" in msg.content


# ── Improvement #7: Smarter auto-spawn vs direct ──

class TestSmartAutoSpawn:
    def _make_loop(self):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        workspace = MagicMock()
        workspace.__truediv__ = MagicMock(return_value=MagicMock())

        with patch("nanobot.agent.loop.ContextBuilder"), \
             patch("nanobot.agent.loop.SessionManager"), \
             patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
            MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
            MockSubMgr.return_value.spawn = AsyncMock(return_value="spawned")
            MockSubMgr.return_value.get_running_tasks.return_value = []
            loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
        return loop

    def test_conversational_detection(self):
        loop = self._make_loop()
        assert loop._is_conversational("ok") is True
        assert loop._is_conversational("thanks") is True
        assert loop._is_conversational("yes") is True
        assert loop._is_conversational("cool!") is True
        assert loop._is_conversational("hey") is True
        assert loop._is_conversational("hi") is True

    def test_task_not_conversational(self):
        loop = self._make_loop()
        assert loop._is_conversational("check my email") is False
        assert loop._is_conversational("send a message to John") is False
        assert loop._is_conversational("what's on my calendar today") is False


# ── Improvement #4: Routing model ──

class TestRoutingModel:
    def test_routing_model_config(self):
        from nanobot.config.schema import AgentDefaults

        defaults = AgentDefaults(routing_model="groq/llama-3.3-70b-versatile")
        assert defaults.routing_model == "groq/llama-3.3-70b-versatile"

    def test_routing_model_none_by_default(self):
        from nanobot.config.schema import AgentDefaults

        defaults = AgentDefaults()
        assert defaults.routing_model is None


# ── Improvement #10: Conversation memory for subagents ──

class TestSubagentConversationMemory:
    def test_subagent_prompt_includes_history(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.session.manager import Session, SessionManager

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        session = Session(key="test:c1")
        session.add_message("user", "Send an email to John about the meeting")
        session.add_message("assistant", "I'll send that email to John now.")
        session.add_message("user", "Also remind him about Friday")

        sessions = MagicMock(spec=SessionManager)
        sessions.get_or_create.return_value = session

        mgr = SubagentManager(
            provider=provider, workspace=MagicMock(), bus=bus,
            session_manager=sessions,
        )

        with patch("nanobot.agent.context.ContextBuilder._build_runtime_context", return_value="test time"), \
             patch("nanobot.agent.skills.SkillsLoader.build_skills_summary", return_value=""):

            prompt = mgr._build_subagent_prompt(session_key="test:c1")

        assert "Recent Conversation Context" in prompt
        assert "John" in prompt
        assert "Friday" in prompt

    def test_subagent_prompt_no_history_without_session(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        with patch("nanobot.agent.context.ContextBuilder._build_runtime_context", return_value="test time"), \
             patch("nanobot.agent.skills.SkillsLoader.build_skills_summary", return_value=""):

            prompt = mgr._build_subagent_prompt()

        assert "Recent Conversation Context" not in prompt


# ── Improvement #1: Streaming LLM ──

class TestStreamChat:
    @pytest.mark.asyncio
    async def test_base_provider_fallback_stream(self):
        """Default stream_chat falls back to non-streaming chat."""
        from nanobot.providers.base import LLMResponse, LLMStreamChunk

        class TestProvider:
            async def chat(self, **kwargs):
                return LLMResponse(content="Hello world", finish_reason="stop")

        from nanobot.providers.base import LLMProvider

        class ConcreteProvider(LLMProvider):
            async def chat(self, messages, **kwargs):
                return LLMResponse(content="Hello world", finish_reason="stop")

            def get_default_model(self):
                return "test"

        provider = ConcreteProvider()
        chunks = []
        async for chunk in provider.stream_chat(messages=[]):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta_content == "Hello world"
        assert chunks[0].finish_reason == "stop"


# ── Two-tier preflight: meta-tool filtering + app detection ──

# ── Improvement #5: App name config ──

class TestAppNameConfig:
    def test_app_name_configurable(self):
        from nanobot.channels.web_voice import WebVoiceConfig

        config = WebVoiceConfig(app_name="Friday")
        assert config.app_name == "Friday"

    def test_app_name_default(self):
        from nanobot.channels.web_voice import WebVoiceConfig

        config = WebVoiceConfig()
        assert config.app_name == "Mawa"


# ── Context-Dependent Message Detection ──

class TestContextDependent:
    def _make_loop(self):
        from nanobot.agent.loop import AgentLoop
        loop = AgentLoop.__new__(AgentLoop)
        return loop

    def test_pronoun_references_detected(self):
        loop = self._make_loop()
        assert loop._is_context_dependent("did it all come?")
        assert loop._is_context_dependent("send that to John")
        assert loop._is_context_dependent("also remind him about Friday")
        assert loop._is_context_dependent("what about those?")
        assert loop._is_context_dependent("was it successful?")

    def test_clear_tasks_not_blocked(self):
        loop = self._make_loop()
        assert not loop._is_context_dependent("check my email")
        assert not loop._is_context_dependent("search for AI news on reddit")
        assert not loop._is_context_dependent("send email to john@example.com")
        assert not loop._is_context_dependent("find weather in new york")

    def test_long_messages_pass_through(self):
        loop = self._make_loop()
        long_msg = "I want you to also check " + "x" * 100
        assert not loop._is_context_dependent(long_msg)

    def test_conversational_not_context_dependent(self):
        loop = self._make_loop()
        # "it" is a context marker but "ok" is conversational — caught earlier
        # These should not match because they're too short and simple
        assert not loop._is_context_dependent("ok")
        assert not loop._is_context_dependent("yes")


# ── Subagent Context Injection ──

class TestSubagentContext:
    def test_build_subagent_context(self, tmp_path):
        from nanobot.agent.context import ContextBuilder

        user_md = tmp_path / "USER.md"
        user_md.write_text("# User\n- **Name:** TestUser\n- **Timezone:** ET\n" + "\n".join(f"Line {i}" for i in range(50)))

        tools_md = tmp_path / "TOOLS.md"
        tools_md.write_text("# Tools\n- Use MCP tools first\n" + "\n".join(f"Rule {i}" for i in range(30)))

        ctx = ContextBuilder(tmp_path)
        result = ctx.build_subagent_context()

        assert "TestUser" in result
        assert "Timezone" in result
        assert "MCP tools" in result
        assert "... (truncated)" in result  # Both files exceed limit

    def test_empty_workspace(self, tmp_path):
        from nanobot.agent.context import ContextBuilder

        ctx = ContextBuilder(tmp_path)
        result = ctx.build_subagent_context()
        assert result == ""
