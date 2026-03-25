"""Unit tests for task detection, decomposition, classification, and auto-delegation."""

import pytest
from nanobot.hooks.builtin.claude_capabilities import (
    is_multi_step,
    decompose_task,
    classify_step,
    is_standalone_heavy,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Multi-step Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsMultiStep:
    """Test multi-step request detection."""

    # --- Should detect as multi-step ---

    @pytest.mark.parametrize("msg", [
        "check my email and then tell me the weather",
        "look up hotels in bali, also show me the weather",
        "research restaurants and tell me the time",
        "1) check email 2) send reply 3) update goals",
        "first search for flights then book the cheapest",
        "check the weather after that send it to fahim",
        "search hotels, compare prices, tell me the best",
        "draft an email plus check my calendar",
        "analyze spending and then schedule a reminder",
    ])
    def test_multi_step_detected(self, msg):
        assert is_multi_step(msg), f"Should be multi-step: {msg}"

    # --- Should NOT detect as multi-step ---

    @pytest.mark.parametrize("msg", [
        "check my email",
        "what is the weather",
        "hello",
        "hey how are you doing today",
        "tell me the time",
        "research best restaurants in dhaka",
        "set a reminder",
        "",
    ])
    def test_single_step_not_detected(self, msg):
        assert not is_multi_step(msg), f"Should NOT be multi-step: {msg}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Task Decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecomposeTask:
    """Test task splitting into independent steps."""

    def test_and_verb_split(self):
        steps = decompose_task("research restaurants and tell me the time")
        assert len(steps) == 2
        assert "research" in steps[0].lower()
        assert "tell" in steps[1].lower()

    def test_then_split(self):
        steps = decompose_task("check email and then send a summary to fahim")
        assert len(steps) == 2

    def test_also_split(self):
        steps = decompose_task("look up hotels, also check my calendar")
        assert len(steps) == 2

    def test_numbered_split(self):
        steps = decompose_task("1) find flights 2) book the cheapest 3) send confirmation")
        assert len(steps) >= 2

    def test_same_object_no_split(self):
        """'search and compare hotels' is one task, not two."""
        steps = decompose_task("search and compare hotels")
        assert len(steps) == 0

    def test_single_task_no_split(self):
        steps = decompose_task("check my email")
        assert len(steps) == 0

    def test_short_message_no_split(self):
        steps = decompose_task("hi")
        assert len(steps) == 0

    def test_three_way_split(self):
        steps = decompose_task("search hotels, compare prices, tell me the best")
        assert len(steps) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Step Classification
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassifyStep:
    """Test heavy/light step classification."""

    @pytest.mark.parametrize("step", [
        "research best restaurants in dhaka",
        "find me the best hotels under 200",
        "analyze my spending patterns this month",
        "draft a thank you email to the team",
        "compare iPhone vs Samsung features",
        "investigate why the API is slow",
        "compile a report on AI trends",
        "gather information about visa requirements",
        "summarize the quarterly earnings report",
        "look up flight prices for next month",
    ])
    def test_heavy_classification(self, step):
        assert classify_step(step) == "heavy", f"Should be heavy: {step}"

    @pytest.mark.parametrize("step", [
        "tell me the time",
        "check my email",
        "show me the weather",
        "set a reminder for tomorrow",
        "list my goals",
        "send this to fahim",
        "open the browser",
        "delete that file",
        "toggle dark mode",
        "update my status",
    ])
    def test_light_classification(self, step):
        assert classify_step(step) == "light", f"Should be light: {step}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Standalone Heavy Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsStandaloneHeavy:
    """Test detection of single messages that should auto-delegate to subagent."""

    @pytest.mark.parametrize("msg", [
        "Research on tonni bridal saree where can we get it in toronto canada",
        "Research best restaurants in Dhaka with reviews and prices",
        "Find me the best hotels in Tokyo under 200 dollars per night",
        "Look up flight prices from Toronto to Dhaka for next month",
        "Compare iPhone 16 vs Samsung S25 vs Pixel 9",
        "Draft a thank you email to the hiring manager at Google",
        "Investigate why our API response times increased last week",
        "Compile a report on AI trends in healthcare for 2026",
        "Gather information about visa requirements for Bangladesh",
        "Find me cheap flights from Toronto to London in April",
        "Look into the best credit cards for travel rewards in Canada",
        "Research what features our competitors launched this quarter",
    ])
    def test_should_delegate(self, msg):
        assert is_standalone_heavy(msg), f"Should delegate: {msg}"

    @pytest.mark.parametrize("msg", [
        "Tell me the time",
        "Check my email",
        "What is the weather",
        "Show me my goals",
        "How are you doing",
        "Tell me more about this",
        "Hello",
        "Set a reminder for tomorrow",
        "What is 2 + 2",
        "Who is the president of USA",
        "Send this to Fahim on telegram",
        "Turn off the language detection feature",
        "Open google.com in the browser",
        "Run npm install in my project",
        "Schedule a meeting for tomorrow at 3pm",
        "Research AI",  # Too short
        "",
    ])
    def test_should_not_delegate(self, msg):
        assert not is_standalone_heavy(msg), f"Should NOT delegate: {msg}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. End-to-End Delegation Scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestDelegationScenarios:
    """Test complete delegation decision flow."""

    def test_multi_part_heavy_plus_light(self):
        """Heavy + light → delegate heavy, inline light."""
        msg = "Research best restaurants in Dhaka and tell me the time"
        steps = decompose_task(msg)
        assert len(steps) == 2
        heavy = [s for s in steps if classify_step(s) == "heavy"]
        light = [s for s in steps if classify_step(s) == "light"]
        assert len(heavy) >= 1
        assert len(light) >= 1

    def test_multi_part_all_heavy_no_split(self):
        """All heavy → stays as one task, no delegation split."""
        msg = "Research hotels in Tokyo and also find the best flights"
        steps = decompose_task(msg)
        if steps:
            heavy = [s for s in steps if classify_step(s) == "heavy"]
            light = [s for s in steps if classify_step(s) == "light"]
            # If decomposed, all steps should be heavy → no auto-delegation
            assert len(light) == 0, "All-heavy should not produce light steps"

    def test_standalone_heavy_not_multi(self):
        """Standalone heavy → delegate as single task, not multi-step."""
        msg = "Research on tonni bridal saree where can we get it in toronto canada"
        assert is_standalone_heavy(msg)
        assert not is_multi_step(msg)  # Single task, not multi-step

    def test_voice_quick_queries_never_delegate(self):
        """Quick voice queries must NEVER be delegated."""
        for q in ["whats the time", "check my calendar", "any new emails",
                   "how is the weather", "what are my goals", "play some music"]:
            assert not is_standalone_heavy(q), f"Voice query should not delegate: {q}"
            steps = decompose_task(q)
            assert len(steps) == 0, f"Voice query should not decompose: {q}"

    def test_pronoun_messages_never_delegate(self):
        """Pronoun/follow-up messages must stay inline."""
        for q in ["tell me more about this", "what do you mean", "explain that",
                   "can you elaborate", "show me the details", "yes do it",
                   "no cancel that", "okay go ahead"]:
            assert not is_standalone_heavy(q), f"Follow-up should not delegate: {q}"

    def test_conversation_not_delegated(self):
        """Conversational messages must stay inline."""
        for q in ["thanks", "good job", "that's interesting", "I see",
                   "what else can you do", "never mind", "stop"]:
            assert not is_standalone_heavy(q), f"Conversation should not delegate: {q}"
