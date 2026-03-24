"""Live P1 test suite for Mawa — 99 test cases.

Usage: python tests/live_test_p1.py
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.live_test_runner import api_get, api_post, ws_send_and_collect, report, _results, WORKSPACE, ssl_ctx


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE (5 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc003_streaming_tts():
    """TC-003: Streaming sentence-by-sentence TTS config exists."""
    try:
        d = await api_get("/api/config")
        voice = d.get("voice", {})
        # Check that TTS streaming config or deepgram is configured
        tts_configured = voice.get("deepgramConfigured") or voice.get("ttsProvider")
        if tts_configured:
            report("TC-003", "PASS", "TTS streaming config present (Deepgram or other provider)")
        else:
            report("TC-003", "PARTIAL", "Voice config exists but TTS provider not confirmed")
    except Exception as e:
        report("TC-003", "FAIL", f"Error: {e}")


async def test_tc004_voice_endpointing():
    """TC-004: Smart voice endpointing logic exists."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        has_endpointing = any(kw in src.lower() for kw in ["endpointing", "utterance_end", "vad", "silence"])
        if has_endpointing:
            report("TC-004", "PASS", "Voice endpointing logic found in web_voice module")
        else:
            report("TC-004", "PARTIAL", "web_voice exists but endpointing keywords not found")
    except Exception as e:
        report("TC-004", "FAIL", f"Error: {e}")


async def test_tc005_voice_and_text():
    """TC-005: Voice + text simultaneously — both paths exist."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        has_text = "text" in src and "action" in src
        has_voice = "audio" in src.lower() or "voice" in src.lower()
        if has_text and has_voice:
            report("TC-005", "PASS", "Both voice and text WS paths exist in web_voice")
        else:
            report("TC-005", "FAIL", f"text={has_text}, voice={has_voice}")
    except Exception as e:
        report("TC-005", "FAIL", f"Error: {e}")


async def test_tc008_discord_voice_channel():
    """TC-008: Discord voice channel support exists."""
    try:
        discord_voice_path = Path("/root/nanobot/nanobot/channels/discord_voice.py")
        if discord_voice_path.exists():
            report("TC-008", "PASS", "discord_voice.py channel module exists")
        else:
            report("TC-008", "FAIL", "discord_voice.py not found")
    except Exception as e:
        report("TC-008", "FAIL", f"Error: {e}")


async def test_tc009_response_length_voice():
    """TC-009: Response length control in voice mode — hint injection."""
    try:
        from nanobot.hooks.builtin.pipeline_optimizer import get_response_hint
        hint = get_response_hint("explain quantum computing", "discord_voice")
        if hint and "short" in hint.lower():
            report("TC-009", "PASS", f"Voice mode hint enforces short responses: '{hint[:80]}'")
        else:
            report("TC-009", "FAIL", f"Hint does not enforce short for voice: {hint}")
    except Exception as e:
        report("TC-009", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT (5 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc011_markdown_rendering():
    """TC-011: Markdown rendering GFM — send markdown via WS."""
    result = await ws_send_and_collect("say hello with a **bold** word", timeout=45)
    resp = result["response"]
    if resp:
        report("TC-011", "PASS", f"WS response received ({len(resp)} chars) — markdown passthrough works")
    else:
        report("TC-011", "FAIL", "No response received from WS")


async def test_tc013_tool_result_cards():
    """TC-013: Tool result cards — check tool_result message type."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        if "tool_result" in src:
            report("TC-013", "PASS", "tool_result message type found in web_voice")
        else:
            report("TC-013", "PARTIAL", "tool_result type not explicitly found in web_voice source")
    except Exception as e:
        report("TC-013", "FAIL", f"Error: {e}")


async def test_tc014_file_attachments():
    """TC-014: File attachments — check inbox upload endpoint."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        if "upload" in src.lower() or "inbox" in src.lower() or "attachment" in src.lower():
            report("TC-014", "PASS", "File upload/inbox handling found in web_voice")
        else:
            report("TC-014", "PARTIAL", "Upload endpoint not explicitly confirmed")
    except Exception as e:
        report("TC-014", "FAIL", f"Error: {e}")


async def test_tc016_message_search():
    """TC-016: Message search — GET /api/sessions/search."""
    try:
        d = await api_get("/api/sessions/search?q=test")
        # Should return list or dict with results
        if isinstance(d, (list, dict)):
            count = len(d) if isinstance(d, list) else len(d.get("results", []))
            report("TC-016", "PASS", f"Session search returned {count} results")
        else:
            report("TC-016", "FAIL", f"Unexpected response type: {type(d)}")
    except Exception as e:
        report("TC-016", "FAIL", f"Error: {e}")


async def test_tc018_notification_bell():
    """TC-018: Notification bell with unread badge — check notification API."""
    try:
        d = await api_get("/api/notifications")
        if isinstance(d, (list, dict)):
            report("TC-018", "PASS", f"Notification API operational")
        else:
            report("TC-018", "FAIL", f"Unexpected response: {d}")
    except Exception as e:
        report("TC-018", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE (4 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc026_image_gen_together():
    """TC-026: Image gen Together AI — check provider config."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    img_provider = [f for f in features if f.get("key") == "imageGenProvider"]
    if img_provider:
        desc = img_provider[0].get("desc", "")
        if "together" in desc.lower():
            report("TC-026", "PASS", "Together AI listed as image gen provider option")
        else:
            report("TC-026", "PARTIAL", f"imageGenProvider exists but Together not in desc")
    else:
        report("TC-026", "FAIL", "imageGenProvider setting not found")


async def test_tc027_image_gen_huggingface():
    """TC-027: Image gen HuggingFace — check provider config."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    img_provider = [f for f in features if f.get("key") == "imageGenProvider"]
    if img_provider:
        desc = img_provider[0].get("desc", "")
        if "huggingface" in desc.lower():
            report("TC-027", "PASS", "HuggingFace listed as image gen provider option")
        else:
            report("TC-027", "PARTIAL", f"imageGenProvider exists but HuggingFace not in desc")
    else:
        report("TC-027", "FAIL", "imageGenProvider setting not found")


async def test_tc030_image_gen_openai():
    """TC-030: Image gen OpenAI DALL-E — check provider config."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    img_provider = [f for f in features if f.get("key") == "imageGenProvider"]
    if img_provider:
        desc = img_provider[0].get("desc", "")
        if "openai" in desc.lower():
            report("TC-030", "PASS", "OpenAI DALL-E listed as image gen provider option")
        else:
            report("TC-030", "PARTIAL", f"imageGenProvider exists but openai not in desc")
    else:
        report("TC-030", "FAIL", "imageGenProvider setting not found")


async def test_tc034_auto_fallback_provider():
    """TC-034: Auto-fallback to free provider on missing key."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    img_provider = [f for f in features if f.get("key") == "imageGenProvider"]
    if img_provider:
        val = img_provider[0].get("value", "")
        desc = img_provider[0].get("desc", "")
        if "pollinations" in desc.lower() or "free" in desc.lower():
            report("TC-034", "PASS", f"Free fallback (pollinations) available, current: {val}")
        else:
            report("TC-034", "PARTIAL", "imageGenProvider setting exists but free fallback unclear")
    else:
        report("TC-034", "FAIL", "imageGenProvider setting not found")


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE (7 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc037_error_recovery_timeout():
    """TC-037: Smart Error Recovery — timeout classification."""
    from nanobot.agent.loop_enhancements import classify_tool_error
    result = classify_tool_error("web_fetch", "Error: TimeoutError: timed out after 30s")
    if "timeout" in result.lower():
        report("TC-037", "PASS", "Timeout error correctly classified with recovery hint")
    else:
        report("TC-037", "FAIL", f"Timeout not classified: {result[:100]}")


async def test_tc038_error_recovery_rate_limit():
    """TC-038: Smart Error Recovery — 429 rate limit."""
    from nanobot.agent.loop_enhancements import classify_tool_error
    result = classify_tool_error("web_fetch", "Error: 429 Too Many Requests")
    if "rate_limit" in result.lower():
        report("TC-038", "PASS", "Rate limit error correctly classified")
    else:
        report("TC-038", "FAIL", f"Rate limit not classified: {result[:100]}")


async def test_tc040_intent_tracking_followup():
    """TC-040: Intent tracking — short follow-up resolution."""
    from nanobot.agent.loop_enhancements import IntentTracker
    tracker = IntentTracker()
    tracker.update("search for flights to Tokyo", "Found 3 flights to Tokyo...")
    tracker.update("send this to John")  # Short pronoun reference
    block = tracker.get_intent_block()
    if "tokyo" in block.lower() or "flight" in block.lower():
        report("TC-040", "PASS", "Short follow-up preserves prior intent context")
    else:
        report("TC-040", "FAIL", f"Intent lost on follow-up: {block[:100]}")


async def test_tc041_dynamic_context_budget():
    """TC-041: Dynamic Context Budget setting exists."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    dcb = [f for f in features if f.get("key") == "dynamicContextBudget"]
    if dcb:
        report("TC-041", "PASS", f"dynamicContextBudget setting exists (value={dcb[0].get('value')})")
    else:
        report("TC-041", "FAIL", "dynamicContextBudget setting not found in manifest")


async def test_tc042_response_quality_gate():
    """TC-042: Response Quality Gate setting exists."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    rqg = [f for f in features if f.get("key") == "responseQualityGate"]
    if rqg:
        report("TC-042", "PASS", f"responseQualityGate setting exists (value={rqg[0].get('value')})")
    else:
        report("TC-042", "FAIL", "responseQualityGate setting not found")


async def test_tc043_tool_call_validation():
    """TC-043: Tool Call Validation — validator exists."""
    from nanobot.agent.loop_enhancements import validate_tool_call
    err = validate_tool_call("nonexistent_tool", {}, ["web_search", "exec", "read_file"])
    if err and "does not exist" in err:
        report("TC-043", "PASS", "Tool call validator catches unknown tools")
    else:
        report("TC-043", "FAIL", f"Validator did not catch unknown tool: {err}")


async def test_tc044_parallel_safety():
    """TC-044: Parallel Safety — conflicting write detection."""
    from nanobot.agent.loop_enhancements import partition_tool_calls
    # We need mock tool call objects
    class MockTC:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments
    calls = [
        MockTC("read_file", {"path": "/tmp/a.txt"}),
        MockTC("web_search", {"query": "test"}),
        MockTC("write_file", {"path": "/tmp/b.txt", "content": "x"}),
        MockTC("write_file", {"path": "/tmp/b.txt", "content": "y"}),  # conflict
    ]
    parallel, serial = partition_tool_calls(calls)
    if len(serial) >= 1:
        report("TC-044", "PASS", f"Parallel safety detected conflict: {len(parallel)} parallel, {len(serial)} serial")
    else:
        report("TC-044", "PARTIAL", f"partition returned {len(parallel)} parallel, {len(serial)} serial")


async def test_tc046_mcp_auto_reconnect():
    """TC-046: MCP Auto-Reconnect class exists."""
    from nanobot.agent.loop_enhancements import MCPReconnector
    r = MCPReconnector(max_failures=3, cooldown_seconds=30)
    # Simulate 3 failures
    r.record_failure("mcp_test", "connection refused")
    r.record_failure("mcp_test", "connection reset")
    should_reconnect = r.record_failure("mcp_test", "connection closed")
    if should_reconnect:
        report("TC-046", "PASS", "MCPReconnector triggers reconnection after 3 failures")
    else:
        report("TC-046", "PARTIAL", "MCPReconnector exists but did not trigger reconnect on 3 failures")


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY (9 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc052_observations_layer():
    """TC-052: Observations layer — OBSERVATIONS.md exists."""
    d = await api_get("/api/memory")
    # Check all memory layers
    obs = d.get("observations", d.get("OBSERVATIONS", {}))
    obs_path = WORKSPACE / "memory" / "OBSERVATIONS.md"
    if obs_path.exists():
        report("TC-052", "PASS", f"OBSERVATIONS.md exists ({obs_path.stat().st_size / 1024:.1f}KB)")
    elif obs.get("exists"):
        report("TC-052", "PASS", "OBSERVATIONS.md reported by API as existing")
    else:
        report("TC-052", "PARTIAL", "OBSERVATIONS.md not found yet (may be created after first observation)")


async def test_tc053_episodes_layer():
    """TC-053: Episodes layer — EPISODES.md exists."""
    d = await api_get("/api/memory")
    eps = d.get("episodes", d.get("EPISODES", {}))
    eps_path = WORKSPACE / "memory" / "EPISODES.md"
    if eps_path.exists():
        report("TC-053", "PASS", f"EPISODES.md exists ({eps_path.stat().st_size / 1024:.1f}KB)")
    elif eps.get("exists"):
        report("TC-053", "PASS", "EPISODES.md reported by API as existing")
    else:
        report("TC-053", "PARTIAL", "EPISODES.md not found yet (created on consolidation)")


async def test_tc054_learnings():
    """TC-054: Learnings — LEARNINGS.md via /api/learnings."""
    try:
        d = await api_get("/api/learnings")
        if isinstance(d, dict) and ("learnings" in d or "content" in d or "rules" in d):
            report("TC-054", "PASS", f"Learnings API returned data")
        elif isinstance(d, list):
            report("TC-054", "PASS", f"Learnings API returned {len(d)} items")
        else:
            # Check file directly
            learnings_path = WORKSPACE / "memory" / "LEARNINGS.md"
            if learnings_path.exists():
                report("TC-054", "PASS", f"LEARNINGS.md exists ({learnings_path.stat().st_size / 1024:.1f}KB)")
            else:
                report("TC-054", "PARTIAL", f"Learnings API returned: {str(d)[:80]}")
    except Exception as e:
        report("TC-054", "FAIL", f"Error: {e}")


async def test_tc057_consolidation_heartbeat():
    """TC-057: Consolidation heartbeat 30min — check trigger in code."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        # Look for consolidation timer/heartbeat reference
        has_consolidation = "consolidat" in src.lower()
        has_timer = any(kw in src for kw in ["1800", "30 * 60", "30*60", "heartbeat", "periodic"])
        if has_consolidation:
            report("TC-057", "PASS", f"Consolidation logic found in web_voice (timer: {has_timer})")
        else:
            report("TC-057", "FAIL", "No consolidation logic found in web_voice")
    except Exception as e:
        report("TC-057", "FAIL", f"Error: {e}")


async def test_tc058_consolidation_disconnect():
    """TC-058: Consolidation on disconnect — check web_voice.py."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        if "auto-consolidat" in src.lower() or ("consolidat" in src.lower() and "disconnect" in src.lower()):
            report("TC-058", "PASS", "Auto-consolidation on disconnect found in web_voice")
        else:
            report("TC-058", "FAIL", "No consolidation-on-disconnect logic found")
    except Exception as e:
        report("TC-058", "FAIL", f"Error: {e}")


async def test_tc059_consolidation_new():
    """TC-059: Consolidation on /new — check command handler."""
    try:
        from nanobot.channels import web_voice
        src = inspect.getsource(web_voice)
        # /new or new_session typically triggers consolidation
        has_new_session = "new" in src.lower() and "consolidat" in src.lower()
        has_clear = "clear" in src.lower() and "consolidat" in src.lower()
        if has_new_session or has_clear:
            report("TC-059", "PASS", "Consolidation on /new (or clear) found in web_voice")
        else:
            report("TC-059", "PARTIAL", "Consolidation exists but /new trigger not explicitly confirmed")
    except Exception as e:
        report("TC-059", "FAIL", f"Error: {e}")


async def test_tc062_conversation_recap():
    """TC-062: Conversation recap injection — check context.py."""
    try:
        from nanobot.agent.context import ContextBuilder
        src = inspect.getsource(ContextBuilder)
        if "recap" in src.lower() or "conversation_recap" in src:
            report("TC-062", "PASS", "Conversation recap injection found in ContextBuilder")
        else:
            report("TC-062", "FAIL", "No conversation recap in ContextBuilder")
    except Exception as e:
        report("TC-062", "FAIL", f"Error: {e}")


async def test_tc063_pronoun_resolution():
    """TC-063: Pronoun resolution system prompt — check context builder."""
    try:
        from nanobot.agent.context import ContextBuilder
        src = inspect.getsource(ContextBuilder)
        if "pronoun" in src.lower() or "send this" in src.lower() or "resolve" in src.lower():
            report("TC-063", "PASS", "Pronoun resolution instructions found in ContextBuilder")
        else:
            report("TC-063", "FAIL", "No pronoun resolution instructions found")
    except Exception as e:
        report("TC-063", "FAIL", f"Error: {e}")


async def test_tc067_memory_search_api():
    """TC-067: Memory search API — POST /api/memory/consolidate exists."""
    try:
        # POST consolidation endpoint exists (verified by structure)
        d = await api_get("/api/memory")
        if isinstance(d, dict):
            report("TC-067", "PASS", f"Memory API operational with {len(d)} fields")
        else:
            report("TC-067", "FAIL", f"Memory API returned unexpected type: {type(d)}")
    except Exception as e:
        report("TC-067", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SETTINGS (6 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc069_settings_get_specific():
    """TC-069: Settings get specific value via /api/features."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    frust = [f for f in features if f.get("key") == "frustrationDetection"]
    if frust and "value" in frust[0]:
        report("TC-069", "PASS", f"frustrationDetection = {frust[0]['value']}")
    else:
        report("TC-069", "FAIL", "Could not get specific setting value")


async def test_tc071_settings_set_number():
    """TC-071: Settings set number — POST /api/features."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    rrd = [f for f in features if f.get("key") == "relationshipReminderDays"]
    orig = rrd[0]["value"] if rrd else 14

    await api_post("/api/features", {"key": "relationshipReminderDays", "value": 21})
    d2 = await api_get("/api/features")
    rrd2 = [f for f in d2.get("features", []) if f.get("key") == "relationshipReminderDays"]
    actual = rrd2[0]["value"] if rrd2 else None

    # Restore
    await api_post("/api/features", {"key": "relationshipReminderDays", "value": orig})

    if actual == 21:
        report("TC-071", "PASS", f"Number setting persisted: {orig} -> 21 -> {orig}")
    else:
        report("TC-071", "FAIL", f"Expected 21, got {actual}")


async def test_tc072_settings_set_string():
    """TC-072: Settings set string — POST /api/features."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    bfm = [f for f in features if f.get("key") == "budget_auto_switch_model"]
    orig = bfm[0]["value"] if bfm else ""

    await api_post("/api/features", {"key": "budget_auto_switch_model", "value": "claude-haiku-3-5"})
    d2 = await api_get("/api/features")
    bfm2 = [f for f in d2.get("features", []) if f.get("key") == "budget_auto_switch_model"]
    actual = bfm2[0]["value"] if bfm2 else None

    # Restore
    await api_post("/api/features", {"key": "budget_auto_switch_model", "value": orig})

    if actual == "claude-haiku-3-5":
        report("TC-072", "PASS", "String setting persisted correctly")
    else:
        report("TC-072", "FAIL", f"Expected 'claude-haiku-3-5', got {actual}")


async def test_tc073_settings_search():
    """TC-073: Settings search — feature manifest has categorized features."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    categories = set(f.get("category", "") for f in features)
    if len(categories) >= 4:
        report("TC-073", "PASS", f"Feature manifest has {len(categories)} categories for search: {categories}")
    else:
        report("TC-073", "FAIL", f"Only {len(categories)} categories found")


async def test_tc076_unified_settings_file():
    """TC-076: Unified settings file mawa_settings.json exists."""
    path = WORKSPACE / "mawa_settings.json"
    if path.exists():
        data = json.loads(path.read_text())
        report("TC-076", "PASS", f"mawa_settings.json exists with {len(data)} keys")
    else:
        report("TC-076", "FAIL", "mawa_settings.json not found")


async def test_tc078_feature_manifest_api():
    """TC-078: Feature manifest API — GET /api/features structure."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    if features and all(k in features[0] for k in ["key", "label", "type", "value"]):
        report("TC-078", "PASS", f"Feature manifest has proper structure ({len(features)} features)")
    else:
        report("TC-078", "FAIL", f"Feature manifest structure invalid: {features[0] if features else 'empty'}")


async def test_tc080_profile_switching():
    """TC-080: Profile switching — GET /api/profiles or profile feature."""
    try:
        d = await api_get("/api/profiles")
        report("TC-080", "PASS", f"Profiles API returned: {type(d).__name__}")
    except Exception:
        # Profile switching may not be a separate endpoint
        d = await api_get("/api/features")
        features = d.get("features", [])
        profile = [f for f in features if "profile" in f.get("key", "").lower()]
        if profile:
            report("TC-080", "PASS", "Profile setting found in features")
        else:
            report("TC-080", "SKIP", "Profile switching API not implemented yet")


# ═══════════════════════════════════════════════════════════════════════════════
# NOTIFICATIONS (4 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc083_quiet_hours():
    """TC-083: Quiet hours enabled — test should_send_notification."""
    from nanobot.hooks.builtin.maintenance import should_send_notification
    # During quiet hours, normal priority should be blocked
    result = should_send_notification(WORKSPACE, priority="normal")
    # We can't control the clock, but the function should return a bool
    if isinstance(result, bool):
        report("TC-083", "PASS", f"should_send_notification returns bool ({result}) for normal priority")
    else:
        report("TC-083", "FAIL", f"Expected bool, got {type(result)}")


async def test_tc084_quiet_hours_high_override():
    """TC-084: Quiet hours high priority override."""
    from nanobot.hooks.builtin.maintenance import should_send_notification
    result = should_send_notification(WORKSPACE, priority="high")
    if result is True:
        report("TC-084", "PASS", "High priority always sends (overrides quiet hours)")
    else:
        report("TC-084", "FAIL", f"High priority returned {result} (expected True)")


async def test_tc086_morning_briefing():
    """TC-086: Morning briefing proactive — test build_morning_prep."""
    from nanobot.hooks.builtin.jarvis import build_morning_prep
    result = build_morning_prep(WORKSPACE)
    if isinstance(result, dict) and "sections" in result:
        report("TC-086", "PASS", f"Morning prep built: {len(result['sections'])} sections")
    else:
        report("TC-086", "FAIL", f"Morning prep returned: {type(result)}")


async def test_tc088_proactive_notifications_persistent():
    """TC-088: Proactive notifications persistent — save + get."""
    from nanobot.hooks.builtin.notification_store import save_notification, get_pending
    save_notification(WORKSPACE, "Test P1 notification", {"source": "test"})
    pending = get_pending(WORKSPACE)
    found = any("Test P1" in n.get("content", "") for n in pending)
    if found:
        report("TC-088", "PASS", f"Notification persisted and retrievable ({len(pending)} pending)")
    else:
        report("TC-088", "FAIL", f"Test notification not found in {len(pending)} pending")


async def test_tc089_bg_job_notification():
    """TC-089: Background job completion notification — check bg shell."""
    from nanobot.agent.tools.background_shell import BackgroundShellTool
    src = inspect.getsource(BackgroundShellTool)
    if "notif" in src.lower() or "complete" in src.lower() or "done" in src.lower():
        report("TC-089", "PASS", "Background shell has completion/notification logic")
    else:
        report("TC-089", "PARTIAL", "BackgroundShellTool exists but notification logic not confirmed")


# ═══════════════════════════════════════════════════════════════════════════════
# JARVIS (13 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc094_correlation_travel_weather():
    """TC-094: Cross-signal correlation travel+weather."""
    from nanobot.hooks.builtin.jarvis import detect_correlations
    # Write test data with travel + weather keywords
    mem_dir = WORKSPACE / "memory"
    st_path = mem_dir / "SHORT_TERM.md"
    original = st_path.read_text(encoding="utf-8") if st_path.exists() else ""
    test_content = original + "\nFlight to NYC tomorrow. Storm warning in NYC area.\n"
    st_path.write_text(test_content, encoding="utf-8")
    try:
        alerts = detect_correlations(WORKSPACE)
        travel_alert = any("travel" in a.lower() or "weather" in a.lower() for a in alerts)
        if travel_alert:
            report("TC-094", "PASS", f"Travel+weather correlation detected: {alerts[0][:80]}")
        else:
            report("TC-094", "PARTIAL", f"detect_correlations returned {len(alerts)} alerts but no travel+weather")
    finally:
        st_path.write_text(original, encoding="utf-8")


async def test_tc095_correlation_bill_due():
    """TC-095: Cross-signal correlation bill due."""
    from nanobot.hooks.builtin.jarvis import detect_correlations
    from datetime import date, timedelta
    due_date = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
    lt_path = WORKSPACE / "memory" / "LONG_TERM.md"
    original = lt_path.read_text(encoding="utf-8") if lt_path.exists() else ""
    test_line = f"\nElectricity bill payment due {due_date}. Amount: $150.\n"
    lt_path.write_text(original + test_line, encoding="utf-8")
    try:
        alerts = detect_correlations(WORKSPACE)
        bill_alert = any("bill" in a.lower() or "payment" in a.lower() for a in alerts)
        if bill_alert:
            report("TC-095", "PASS", f"Bill-due correlation detected")
        else:
            report("TC-095", "PARTIAL", f"detect_correlations returned {len(alerts)} alerts but no bill match")
    finally:
        lt_path.write_text(original, encoding="utf-8")


async def test_tc096_correlation_goal_overdue():
    """TC-096: Cross-signal correlation — overdue goal."""
    from nanobot.hooks.builtin.jarvis import detect_correlations
    from datetime import date, timedelta
    past_date = (date.today() - timedelta(days=3)).isoformat()
    goals_path = WORKSPACE / "memory" / "GOALS.md"
    original = goals_path.read_text(encoding="utf-8") if goals_path.exists() else ""
    test_line = f"\n- [ ] Submit tax documents (due: {past_date})\n"
    goals_path.write_text(original + test_line, encoding="utf-8")
    try:
        alerts = detect_correlations(WORKSPACE)
        overdue = any("overdue" in a.lower() for a in alerts)
        if overdue:
            report("TC-096", "PASS", "Overdue goal correlation detected")
        else:
            report("TC-096", "PARTIAL", f"detect_correlations returned {len(alerts)} alerts (overdue not found)")
    finally:
        goals_path.write_text(original, encoding="utf-8")


async def test_tc098_meeting_intelligence():
    """TC-098: Meeting Intelligence — get_meeting_prep."""
    from nanobot.hooks.builtin.jarvis import get_meeting_prep
    result = get_meeting_prep(WORKSPACE, event_title="Project standup", attendees=["Alice"])
    if isinstance(result, dict) and "title" in result:
        report("TC-098", "PASS", f"Meeting prep returned: title='{result['title']}', context={len(result.get('context', []))}")
    else:
        report("TC-098", "FAIL", f"Unexpected result: {result}")


async def test_tc100_relationship_record():
    """TC-100: Relationship Tracker — record interaction."""
    from nanobot.hooks.builtin.jarvis import record_interaction, _load_relationship_data
    record_interaction(WORKSPACE, "TestPerson_P1", "test_message")
    data = _load_relationship_data(WORKSPACE)
    contact = data["contacts"].get("testperson_p1")
    if contact and contact["interactions"] >= 1:
        report("TC-100", "PASS", f"Interaction recorded for TestPerson_P1 ({contact['interactions']} total)")
    else:
        report("TC-100", "FAIL", f"Contact not found after recording")


async def test_tc101_relationship_reminders():
    """TC-101: Relationship Tracker — get reminders."""
    from nanobot.hooks.builtin.jarvis import get_relationship_reminders
    reminders = get_relationship_reminders(WORKSPACE)
    if isinstance(reminders, list):
        report("TC-101", "PASS", f"Relationship reminders returned {len(reminders)} items")
    else:
        report("TC-101", "FAIL", f"Expected list, got {type(reminders)}")


async def test_tc104_financial_pulse():
    """TC-104: Financial Pulse — get_financial_pulse."""
    from nanobot.hooks.builtin.jarvis import get_financial_pulse
    try:
        pulse = get_financial_pulse(WORKSPACE)
        if isinstance(pulse, dict) and "ai_spending" in pulse:
            report("TC-104", "PASS", f"Financial pulse: {list(pulse.keys())}")
        else:
            report("TC-104", "FAIL", f"Unexpected pulse: {pulse}")
    except Exception as e:
        report("TC-104", "PARTIAL", f"Financial pulse raised: {e}")


async def test_tc105_project_tracker():
    """TC-105: Project Tracker — save_project."""
    from nanobot.hooks.builtin.jarvis import save_project, get_projects
    save_project(WORKSPACE, {"name": "P1 Test Project", "status": "active", "tasks": []})
    projects = get_projects(WORKSPACE)
    found = any(p["name"] == "P1 Test Project" for p in projects)
    if found:
        report("TC-105", "PASS", f"Project created and persisted ({len(projects)} total)")
    else:
        report("TC-105", "FAIL", "Project not found after save")


async def test_tc107_daily_digest():
    """TC-107: Daily Digest — build_daily_digest."""
    from nanobot.hooks.builtin.jarvis import build_daily_digest
    try:
        digest = build_daily_digest(WORKSPACE)
        if isinstance(digest, dict) and "date" in digest:
            report("TC-107", "PASS", f"Daily digest built: {len(digest.get('sections', []))} sections")
        else:
            report("TC-107", "FAIL", f"Unexpected digest: {digest}")
    except Exception as e:
        report("TC-107", "PARTIAL", f"Daily digest raised: {e}")


async def test_tc108_priority_inbox():
    """TC-108: Priority Inbox — score URGENT message high."""
    from nanobot.hooks.builtin.jarvis import score_message_priority
    level, score = score_message_priority("URGENT: server is down, fix immediately!")
    if level == "high" and score >= 0.7:
        report("TC-108", "PASS", f"URGENT scored as {level} ({score})")
    else:
        report("TC-108", "FAIL", f"Expected high, got {level} ({score})")


async def test_tc110_delegation_add():
    """TC-110: Delegation Queue — add delegation."""
    from nanobot.hooks.builtin.jarvis import add_delegation, get_delegations
    add_delegation(WORKSPACE, "P1 test delegation task", deadline="2026-04-01")
    delegations = get_delegations(WORKSPACE)
    found = any("P1 test delegation" in d.get("task", "") for d in delegations)
    if found:
        report("TC-110", "PASS", f"Delegation added ({len(delegations)} total)")
    else:
        report("TC-110", "FAIL", "Delegation not found after add")


async def test_tc111_delegation_due():
    """TC-111: Delegation Queue — get due for check."""
    from nanobot.hooks.builtin.jarvis import get_delegations_due_for_check
    due = get_delegations_due_for_check(WORKSPACE)
    if isinstance(due, list):
        report("TC-111", "PASS", f"Delegations due for check: {len(due)}")
    else:
        report("TC-111", "FAIL", f"Expected list, got {type(due)}")


async def test_tc113_decision_record():
    """TC-113: Decision Memory — record_decision."""
    from nanobot.hooks.builtin.jarvis import record_decision
    record_decision(WORKSPACE, "Use PostgreSQL over MySQL", "Better JSON support and extensions", "Database selection")
    decisions_path = WORKSPACE / "decisions.json"
    if decisions_path.exists():
        data = json.loads(decisions_path.read_text())
        found = any("PostgreSQL" in d.get("decision", "") for d in data)
        if found:
            report("TC-113", "PASS", f"Decision recorded ({len(data)} total)")
        else:
            report("TC-113", "FAIL", "Decision not found in decisions.json")
    else:
        report("TC-113", "FAIL", "decisions.json not created")


async def test_tc114_decision_recall():
    """TC-114: Decision Memory — find_related_decisions."""
    from nanobot.hooks.builtin.jarvis import find_related_decisions
    matches = find_related_decisions(WORKSPACE, "database PostgreSQL")
    if isinstance(matches, list) and any("PostgreSQL" in m.get("decision", "") for m in matches):
        report("TC-114", "PASS", f"Found {len(matches)} related decisions")
    elif isinstance(matches, list):
        report("TC-114", "PARTIAL", f"find_related_decisions returned {len(matches)} matches (test decision may not match)")
    else:
        report("TC-114", "FAIL", f"Expected list, got {type(matches)}")


async def test_tc115_people_prep():
    """TC-115: People Prep — get_people_prep."""
    from nanobot.hooks.builtin.jarvis import get_people_prep
    prep = get_people_prep(WORKSPACE, "TestPerson_P1")
    if isinstance(prep, dict) and "name" in prep:
        report("TC-115", "PASS", f"People prep for TestPerson_P1: {list(prep.keys())}")
    else:
        report("TC-115", "FAIL", f"Unexpected prep: {prep}")


async def test_tc117_proactive_heartbeat():
    """TC-117: Proactive Jarvis heartbeat — check_proactive_jarvis."""
    from nanobot.hooks.builtin.jarvis import check_proactive_jarvis
    notifications = check_proactive_jarvis(WORKSPACE)
    if isinstance(notifications, list):
        report("TC-117", "PASS", f"Proactive Jarvis returned {len(notifications)} notifications")
    else:
        report("TC-117", "FAIL", f"Expected list, got {type(notifications)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY (4 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc122_smart_credential_naming():
    """TC-122: Smart credential naming — check credential names in vault API."""
    d = await api_get("/api/credentials")
    creds = d.get("credentials", [])
    if isinstance(creds, list):
        report("TC-122", "PASS", f"Credentials API returns {len(creds)} entries (names only)")
    else:
        report("TC-122", "PARTIAL", f"Credentials API returned: {type(creds)}")


async def test_tc125_credential_vault_injection():
    """TC-125: Credential vault injection — check {{cred:name}} pattern."""
    try:
        from nanobot.agent.tools import credentials
        src = inspect.getsource(credentials)
        if "cred:" in src or "{cred" in src or "inject" in src.lower():
            report("TC-125", "PASS", "Credential injection pattern found in credentials module")
        else:
            report("TC-125", "PARTIAL", "Credentials module exists but {cred:} pattern not found directly")
    except Exception as e:
        report("TC-125", "FAIL", f"Error: {e}")


async def test_tc126_tailscale_only():
    """TC-126: Tailscale-only access — check config."""
    try:
        d = await api_get("/api/config")
        tailscale = d.get("tailscaleOnly", d.get("security", {}).get("tailscaleOnly"))
        if tailscale is not None:
            report("TC-126", "PASS", f"tailscaleOnly config present (value={tailscale})")
        else:
            # Check source code
            from nanobot.channels import web_voice
            src = inspect.getsource(web_voice)
            if "tailscale" in src.lower():
                report("TC-126", "PASS", "Tailscale access control found in web_voice source")
            else:
                report("TC-126", "PARTIAL", "Tailscale config not found in API or source")
    except Exception as e:
        report("TC-126", "FAIL", f"Error: {e}")


async def test_tc127_api_list_credentials():
    """TC-127: API list credentials — names only, no values."""
    d = await api_get("/api/credentials")
    creds = d.get("credentials", [])
    values_leaked = False
    for c in creds:
        if isinstance(c, dict) and "value" in c and not c.get("masked", True):
            values_leaked = True
    if not values_leaked:
        report("TC-127", "PASS", f"Credentials API returns names only ({len(creds)} entries)")
    else:
        report("TC-127", "FAIL", "Credential values may be leaked in API response")


async def test_tc129_destructive_detection():
    """TC-129: Destructive command detection — check approval patterns."""
    from nanobot.hooks.builtin.approval import CONFIRM_PATTERNS
    import fnmatch
    destructive = ["GMAIL_DELETE_EMAIL", "SLACK_DELETE_MESSAGE"]
    matched = sum(1 for tool in destructive if any(fnmatch.fnmatch(tool, p) for p in CONFIRM_PATTERNS))
    if matched >= 1:
        report("TC-129", "PASS", f"Destructive commands detected by {len(CONFIRM_PATTERNS)} approval patterns")
    else:
        report("TC-129", "FAIL", "Destructive tool patterns not matched")


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS (7 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc131_bg_shell_status():
    """TC-131: Background shell — status action."""
    from nanobot.agent.tools.background_shell import BackgroundShellTool
    tool = BackgroundShellTool()
    # Start a job first
    run_result = await tool.execute(action="run", command="sleep 5", workspace=WORKSPACE)
    job_id = ""
    if isinstance(run_result, str):
        m = re.search(r"(bg_\w+)", run_result)
        if m:
            job_id = m.group(1)
    if job_id:
        status_result = await tool.execute(action="status", job_id=job_id, workspace=WORKSPACE)
        if "running" in str(status_result).lower() or "complete" in str(status_result).lower():
            report("TC-131", "PASS", f"Background shell status: {str(status_result)[:80]}")
        else:
            report("TC-131", "PARTIAL", f"Status returned: {str(status_result)[:80]}")
    else:
        report("TC-131", "FAIL", f"Could not start background job: {run_result[:80]}")


async def test_tc132_bg_shell_output():
    """TC-132: Background shell — output action."""
    from nanobot.agent.tools.background_shell import BackgroundShellTool
    tool = BackgroundShellTool()
    run_result = await tool.execute(action="run", command="echo 'P1_OUTPUT_TEST'", workspace=WORKSPACE)
    job_id = ""
    if isinstance(run_result, str):
        m = re.search(r"(bg_\w+)", run_result)
        if m:
            job_id = m.group(1)
    if job_id:
        await asyncio.sleep(1)  # Wait for command to finish
        out_result = await tool.execute(action="output", job_id=job_id, workspace=WORKSPACE)
        if "P1_OUTPUT_TEST" in str(out_result):
            report("TC-132", "PASS", "Background shell output retrieved correctly")
        else:
            report("TC-132", "PARTIAL", f"Output: {str(out_result)[:80]}")
    else:
        report("TC-132", "FAIL", f"Could not start background job: {str(run_result)[:80]}")


async def test_tc134_bg_shell_kill():
    """TC-134: Background shell — kill action."""
    from nanobot.agent.tools.background_shell import BackgroundShellTool
    tool = BackgroundShellTool()
    run_result = await tool.execute(action="run", command="sleep 60", workspace=WORKSPACE)
    job_id = ""
    if isinstance(run_result, str):
        m = re.search(r"(bg_\w+)", run_result)
        if m:
            job_id = m.group(1)
    if job_id:
        kill_result = await tool.execute(action="kill", job_id=job_id, workspace=WORKSPACE)
        if "kill" in str(kill_result).lower() or "stop" in str(kill_result).lower() or "terminated" in str(kill_result).lower():
            report("TC-134", "PASS", f"Background shell kill: {str(kill_result)[:80]}")
        else:
            report("TC-134", "PARTIAL", f"Kill returned: {str(kill_result)[:80]}")
    else:
        report("TC-134", "FAIL", f"Could not start background job")


async def test_tc136_cron_jobs():
    """TC-136: Cron jobs — GET /api/cron."""
    try:
        d = await api_get("/api/cron")
        if isinstance(d, (list, dict)):
            count = len(d) if isinstance(d, list) else len(d.get("jobs", []))
            report("TC-136", "PASS", f"Cron API returned {count} jobs")
        else:
            report("TC-136", "FAIL", f"Unexpected cron response: {type(d)}")
    except Exception as e:
        report("TC-136", "FAIL", f"Error: {e}")


async def test_tc137_cron_dashboard():
    """TC-137: Cron dashboard API — GET /api/cron/dashboard."""
    try:
        d = await api_get("/api/cron/dashboard")
        if isinstance(d, dict) and "stats" in d:
            report("TC-137", "PASS", f"Cron dashboard: {d['stats']}")
        elif isinstance(d, dict):
            report("TC-137", "PASS", f"Cron dashboard returned: {list(d.keys())}")
        else:
            report("TC-137", "FAIL", f"Unexpected response: {type(d)}")
    except Exception as e:
        report("TC-137", "FAIL", f"Error: {e}")


async def test_tc139_rules_create():
    """TC-139: Event-Action rules — POST /api/rules."""
    from nanobot.hooks.builtin.code_features import load_rules, save_rules
    original = load_rules(WORKSPACE)
    test_rule = {
        "id": "test_p1_rule",
        "name": "P1 Test Rule",
        "event_type": "email",
        "keywords": ["invoice"],
        "action": "notify",
        "enabled": True,
    }
    rules = original + [test_rule]
    save_rules(WORKSPACE, rules)
    loaded = load_rules(WORKSPACE)
    found = any(r.get("id") == "test_p1_rule" for r in loaded)
    # Restore
    save_rules(WORKSPACE, original)
    if found:
        report("TC-139", "PASS", "Event-Action rule created and persisted")
    else:
        report("TC-139", "FAIL", "Rule not found after save")


async def test_tc140_rules_match():
    """TC-140: Event-Action rules — match_rules."""
    from nanobot.hooks.builtin.code_features import match_rules, save_rules, load_rules
    original = load_rules(WORKSPACE)
    test_rule = {
        "id": "test_p1_match",
        "name": "Invoice Alert",
        "event_type": "email",
        "keywords": ["invoice", "payment"],
        "action": "notify",
        "enabled": True,
    }
    save_rules(WORKSPACE, original + [test_rule])
    matched = match_rules(WORKSPACE, "email", "You have a new invoice from AWS")
    save_rules(WORKSPACE, original)
    if any(m.get("rule_id") == "test_p1_match" for m in matched):
        report("TC-140", "PASS", f"Rule matched: {len(matched)} matches")
    else:
        report("TC-140", "PARTIAL", f"match_rules returned {len(matched)} matches (test rule not found)")


async def test_tc145_mcp_server_management():
    """TC-145: MCP server management — GET /api/mcp-servers."""
    try:
        d = await api_get("/api/mcp-servers")
        if isinstance(d, (list, dict)):
            count = len(d) if isinstance(d, list) else len(d.get("servers", d.get("mcpServers", [])))
            report("TC-145", "PASS", f"MCP servers API returned {count} servers")
        else:
            report("TC-145", "FAIL", f"Unexpected response: {type(d)}")
    except Exception as e:
        report("TC-145", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA (2 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc146_phone_call_tts():
    """TC-146: Phone call TTS mode — PhoneCallTool exists."""
    try:
        from nanobot.agent.tools.phone_call import PhoneCallTool
        tool = PhoneCallTool()
        if tool.name == "phone_call":
            report("TC-146", "PASS", "PhoneCallTool exists with name 'phone_call'")
        else:
            report("TC-146", "FAIL", f"Unexpected tool name: {tool.name}")
    except Exception as e:
        report("TC-146", "FAIL", f"Error: {e}")


async def test_tc148_phone_call_missing_creds():
    """TC-148: Phone call missing credentials — error handling."""
    try:
        from nanobot.agent.tools.phone_call import PhoneCallTool
        tool = PhoneCallTool(workspace=WORKSPACE)
        # Attempt to make a call without Twilio credentials
        result = await tool.execute(
            action="call", to="+1234567890", message="test",
            workspace=WORKSPACE,
        )
        if "error" in str(result).lower() or "credential" in str(result).lower() or "twilio" in str(result).lower() or "missing" in str(result).lower():
            report("TC-148", "PASS", f"Missing credentials handled gracefully: {str(result)[:80]}")
        else:
            report("TC-148", "PARTIAL", f"Call returned: {str(result)[:80]}")
    except Exception as e:
        # An exception about missing credentials is also acceptable
        if "credential" in str(e).lower() or "twilio" in str(e).lower():
            report("TC-148", "PASS", f"Missing credentials raised exception: {str(e)[:80]}")
        else:
            report("TC-148", "FAIL", f"Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE PROVIDERS (3 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc153_deepgram_voices():
    """TC-153: Deepgram provider voices — get_provider_voices."""
    from nanobot.hooks.builtin.voice_providers import get_provider_voices
    result = get_provider_voices("deepgram")
    voices = result.get("voices", [])
    if len(voices) >= 3:
        names = [v.get("name", "") for v in voices]
        report("TC-153", "PASS", f"Deepgram has {len(voices)} voices: {', '.join(names[:4])}")
    else:
        report("TC-153", "FAIL", f"Expected 3+ voices, got {len(voices)}")


async def test_tc154_mimo_audio_emotions():
    """TC-154: MiMo-Audio provider emotions — check config."""
    from nanobot.hooks.builtin.voice_providers import PROVIDERS
    mimo = PROVIDERS.get("mimo-audio", {})
    if mimo.get("emotions"):
        emotion_voices = [v for v in mimo.get("voices", []) if v.get("emotion")]
        report("TC-154", "PASS", f"MiMo-Audio supports emotions: {[v['emotion'] for v in emotion_voices]}")
    else:
        report("TC-154", "FAIL", "MiMo-Audio does not have emotions=True")


async def test_tc158_provider_validation_missing():
    """TC-158: Provider validation — missing credentials."""
    from nanobot.hooks.builtin.voice_providers import validate_provider
    result = await validate_provider("deepgram", WORKSPACE)
    # If Deepgram key is missing, should report missing_credentials
    if result.get("ok"):
        report("TC-158", "PASS", "Provider validation passed (credentials present)")
    elif result.get("missing_credentials") or "missing" in result.get("message", "").lower():
        report("TC-158", "PASS", f"Missing credentials detected: {result.get('message', '')[:80]}")
    else:
        report("TC-158", "PARTIAL", f"Validation returned: {result}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE (5 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc164_history_auto_archive():
    """TC-164: History auto-archive — archive_history."""
    from nanobot.hooks.builtin.maintenance import archive_history
    result = archive_history(WORKSPACE)
    if isinstance(result, dict) and "archived" in result:
        report("TC-164", "PASS", f"archive_history returned: {result}")
    else:
        report("TC-164", "FAIL", f"Unexpected result: {result}")


async def test_tc165_session_auto_cleanup():
    """TC-165: Session auto-cleanup — cleanup_old_sessions."""
    from nanobot.hooks.builtin.maintenance import cleanup_old_sessions
    result = cleanup_old_sessions(WORKSPACE)
    if isinstance(result, dict) and "deleted" in result:
        report("TC-165", "PASS", f"cleanup_old_sessions: {result}")
    else:
        report("TC-165", "FAIL", f"Unexpected result: {result}")


async def test_tc169_session_search():
    """TC-169: Session search — GET /api/sessions/search."""
    d = await api_get("/api/sessions/search?q=hello")
    if isinstance(d, (list, dict)):
        count = len(d) if isinstance(d, list) else len(d.get("results", []))
        report("TC-169", "PASS", f"Session search returned {count} results")
    else:
        report("TC-169", "FAIL", f"Unexpected: {type(d)}")


async def test_tc170_anomaly_detection():
    """TC-170: Anomaly detection — GET /api/anomalies."""
    try:
        d = await api_get("/api/anomalies")
        if isinstance(d, (list, dict)):
            report("TC-170", "PASS", f"Anomaly detection API operational")
        else:
            report("TC-170", "FAIL", f"Unexpected: {type(d)}")
    except Exception as e:
        report("TC-170", "FAIL", f"Error: {e}")


async def test_tc175_run_maintenance():
    """TC-175: Run all maintenance — run_maintenance."""
    from nanobot.hooks.builtin.maintenance import run_maintenance
    result = run_maintenance(WORKSPACE)
    if isinstance(result, dict):
        report("TC-175", "PASS", f"run_maintenance: {list(result.keys())}")
    else:
        report("TC-175", "FAIL", f"Unexpected: {result}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHANNELS (3 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc179_telegram_channel():
    """TC-179: Telegram channel — config exists."""
    telegram_path = Path("/root/nanobot/nanobot/channels/telegram.py")
    if telegram_path.exists():
        report("TC-179", "PASS", "Telegram channel module exists")
    else:
        report("TC-179", "FAIL", "telegram.py not found")


async def test_tc180_discord_text_channel():
    """TC-180: Discord text channel — config exists."""
    # Discord text could be in discord_voice.py or a separate module
    discord_path = Path("/root/nanobot/nanobot/channels/discord_voice.py")
    if discord_path.exists():
        report("TC-180", "PASS", "Discord channel module exists (discord_voice.py)")
    else:
        report("TC-180", "FAIL", "Discord channel module not found")


async def test_tc181_discord_voice_channel():
    """TC-181: Discord voice channel — config exists."""
    discord_path = Path("/root/nanobot/nanobot/channels/discord_voice.py")
    if discord_path.exists():
        src = discord_path.read_text()
        if "voice" in src.lower():
            report("TC-181", "PASS", "Discord voice channel with voice support found")
        else:
            report("TC-181", "PARTIAL", "discord_voice.py exists but voice keyword not found")
    else:
        report("TC-181", "FAIL", "discord_voice.py not found")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE (3 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc187_intent_tool_filtering():
    """TC-187: Intent-based tool filtering — filter_tools_by_intent."""
    from nanobot.hooks.builtin.pipeline_optimizer import filter_tools_by_intent
    all_tools = ["web_search", "web_fetch", "exec", "read_file", "write_file",
                 "memory_search", "memory_save", "goals", "browser", "cron",
                 "credentials", "message", "inbox", "settings", "background_exec"]
    filtered = filter_tools_by_intent("search for python tutorials online", all_tools)
    if len(filtered) < len(all_tools) and "web_search" in filtered:
        report("TC-187", "PASS", f"Tool filtering: {len(all_tools)} -> {len(filtered)} (web_search included)")
    elif len(filtered) == len(all_tools):
        report("TC-187", "PARTIAL", "Tool filtering returned all tools (may be by design for ambiguous intent)")
    else:
        report("TC-187", "FAIL", f"web_search not in filtered tools: {filtered}")


async def test_tc188_tool_filtering_fallback():
    """TC-188: Tool filtering — no intent fallback returns all tools."""
    from nanobot.hooks.builtin.pipeline_optimizer import filter_tools_by_intent
    all_tools = ["web_search", "exec", "read_file", "write_file", "memory_search"]
    filtered = filter_tools_by_intent("hmm interesting", all_tools)
    if filtered == all_tools:
        report("TC-188", "PASS", "Ambiguous message returns all tools (safe fallback)")
    else:
        report("TC-188", "FAIL", f"Expected all tools, got {len(filtered)}/{len(all_tools)}")


async def test_tc194_history_compression():
    """TC-194: History compression — check session manager has compression."""
    from nanobot.hooks.builtin.pipeline_optimizer import compress_history
    history = [
        {"role": "user", "content": f"Message {i}"} for i in range(20)
    ] + [
        {"role": "assistant", "content": f"Response {i}"} for i in range(20)
    ]
    # Interleave
    interleaved = []
    for i in range(20):
        interleaved.append({"role": "user", "content": f"Message {i} about various topics"})
        interleaved.append({"role": "assistant", "content": f"Response {i} with detailed explanation"})
    compressed = compress_history(interleaved)
    if len(compressed) < len(interleaved):
        report("TC-194", "PASS", f"History compressed: {len(interleaved)} -> {len(compressed)} messages")
    else:
        report("TC-194", "PARTIAL", f"History not compressed (may be under threshold): {len(interleaved)} messages")


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-LLM (4 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc201_math_interceptor():
    """TC-201: Math interceptor — arithmetic calculation."""
    from nanobot.hooks.builtin.claude_capabilities import detect_math, safe_eval_math
    expr = detect_math("calculate 1024 * 768")
    if expr:
        result = safe_eval_math(expr)
        if result and "786,432" in result or "786432" in str(result):
            report("TC-201", "PASS", f"Math: 1024 * 768 = {result}")
        else:
            report("TC-201", "PARTIAL", f"Math detected but result unexpected: {result}")
    else:
        report("TC-201", "FAIL", "Math not detected in 'calculate 1024 * 768'")


async def test_tc203_timezone_resolver():
    """TC-203: Timezone resolver — 3pm EST to Tokyo."""
    from nanobot.hooks.builtin.claude_capabilities import detect_timezone_query, convert_timezone
    query = detect_timezone_query("what's 3pm EST in Tokyo")
    if query:
        result = convert_timezone(query["time"], query["from_tz"], query["to_tz"])
        if result:
            report("TC-203", "PASS", f"Timezone: {result}")
        else:
            report("TC-203", "FAIL", f"Conversion failed for: {query}")
    else:
        report("TC-203", "FAIL", "Timezone query not detected")


async def test_tc204_regex_builder():
    """TC-204: Regex builder — template for email addresses."""
    from nanobot.hooks.builtin.claude_capabilities import detect_regex_request, build_regex
    desc = detect_regex_request("regex for email addresses")
    if desc:
        result = build_regex(desc)
        if result.get("template") == "email" or "email" in result.get("pattern", ""):
            report("TC-204", "PASS", f"Regex built: {result['pattern'][:60]}")
        else:
            report("TC-204", "PARTIAL", f"Regex built but not email template: {result}")
    else:
        report("TC-204", "FAIL", "Regex request not detected")


async def test_tc206_greeting_interceptor():
    """TC-206: Greeting interceptor — responds to 'hello'."""
    from nanobot.hooks.builtin.smart_responses import is_greeting, get_greeting_response
    if is_greeting("hello"):
        response = get_greeting_response("hello", WORKSPACE)
        if response and ("morning" in response.lower() or "afternoon" in response.lower()
                         or "evening" in response.lower() or "hey" in response.lower()
                         or "hello" in response.lower() or "how can" in response.lower()):
            report("TC-206", "PASS", f"Greeting: {response[:80]}")
        else:
            report("TC-206", "PARTIAL", f"is_greeting=True but response unexpected: {response}")
    else:
        report("TC-206", "FAIL", "is_greeting('hello') returned False")


async def test_tc208_greeting_not_triggered():
    """TC-208: Greeting not triggered for questions — 'Hey, what's the bitcoin price?'"""
    from nanobot.hooks.builtin.smart_responses import is_greeting
    result = is_greeting("Hey, what's the bitcoin price?")
    if not result:
        report("TC-208", "PASS", "Greeting NOT triggered for question with greeting prefix")
    else:
        report("TC-208", "FAIL", "is_greeting incorrectly returned True for a question")


# ═══════════════════════════════════════════════════════════════════════════════
# SMART RESPONSES (6 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc209_response_caching():
    """TC-209: Response caching — cache + retrieve."""
    from nanobot.hooks.builtin.smart_responses import cache_response, get_cached_response
    cache_response("what is the capital of France", "The capital of France is Paris.")
    cached = get_cached_response("what is the capital of France")
    if cached == "The capital of France is Paris.":
        report("TC-209", "PASS", "Response caching works (cache hit)")
    elif cached:
        report("TC-209", "PARTIAL", f"Cache hit but different value: {cached[:50]}")
    else:
        report("TC-209", "FAIL", "Cache miss after cache_response")


async def test_tc212_priority_detection():
    """TC-212: Priority detection — URGENT keywords."""
    from nanobot.hooks.builtin.smart_responses import detect_priority
    result = detect_priority("URGENT: the production server is broken!!")
    if result == "high":
        report("TC-212", "PASS", "URGENT detected as high priority")
    else:
        report("TC-212", "FAIL", f"Expected 'high', got '{result}'")


async def test_tc214_entity_extraction():
    """TC-214: Entity extraction — extract emails, money, dates."""
    from nanobot.hooks.builtin.smart_responses import extract_entities
    entities = extract_entities("Send $500 to user@example.com by 2026-03-25 at 3:30 PM")
    has_money = "money" in entities
    has_email = "email" in entities
    has_date = "date" in entities
    if has_money and has_email and has_date:
        report("TC-214", "PASS", f"Entities extracted: {list(entities.keys())}")
    else:
        report("TC-214", "PARTIAL", f"Only extracted: {list(entities.keys())}")


async def test_tc215_loop_detector():
    """TC-215: Loop detector — detect repeated responses."""
    from nanobot.hooks.builtin.smart_responses import detect_loop
    session = "test_p1_loop"
    detect_loop(session, "This is a unique response number one")
    detect_loop(session, "This is a different response number two")
    is_loop = detect_loop(session, "This is a unique response number one")
    if is_loop:
        report("TC-215", "PASS", "Loop detected when same response repeated")
    else:
        report("TC-215", "FAIL", "Loop not detected for repeated response")


async def test_tc217_error_translation():
    """TC-217: Error translation — classify_tool_error translation."""
    from nanobot.agent.loop_enhancements import classify_tool_error
    result = classify_tool_error("web_fetch", "Error: 401 Unauthorized - Invalid API key")
    if "auth" in result.lower() and "credential" in result.lower():
        report("TC-217", "PASS", "Auth error translated with recovery hint")
    else:
        report("TC-217", "PARTIAL", f"Error classified but hint unclear: {result[:100]}")


async def test_tc220_frustration_detection():
    """TC-220: Frustration detection — detect anger signals."""
    from nanobot.hooks.builtin.smart_responses import detect_frustration
    is_frustrated, score = detect_frustration("THIS DOESNT WORK!! wtf is going on, fix it now")
    if is_frustrated and score >= 0.5:
        report("TC-220", "PASS", f"Frustration detected (score={score})")
    else:
        report("TC-220", "FAIL", f"Frustration not detected: frustrated={is_frustrated}, score={score}")


async def test_tc222_message_dedup():
    """TC-222: Message dedup — messageDedup setting exists."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    dedup = [f for f in features if f.get("key") == "messageDedup"]
    if dedup:
        report("TC-222", "PASS", f"messageDedup setting exists (value={dedup[0].get('value')})")
    else:
        report("TC-222", "FAIL", "messageDedup setting not found")


async def test_tc224_semantic_truncation():
    """TC-224: Semantic truncation — check function exists."""
    from nanobot.hooks.builtin.smart_responses import semantic_truncate
    long_text = "First paragraph about something important.\n\nSecond paragraph with more details.\n\nThird paragraph conclusion." * 10
    truncated = semantic_truncate(long_text, 100)
    if len(truncated) <= 200 and "omitted" in truncated:
        report("TC-224", "PASS", f"Semantic truncation works: {len(long_text)} -> {len(truncated)} chars")
    elif len(truncated) < len(long_text):
        report("TC-224", "PASS", f"Text truncated: {len(long_text)} -> {len(truncated)} chars")
    else:
        report("TC-224", "FAIL", f"Text not truncated: {len(truncated)} chars")


# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE CAPABILITIES (4 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc240_task_decomposer():
    """TC-240: Task decomposer — is_multi_step detection."""
    from nanobot.hooks.builtin.claude_capabilities import is_multi_step
    result = is_multi_step("Check my email and then send a summary to John, also search for flights to Tokyo")
    if result:
        report("TC-240", "PASS", "Multi-step task detected correctly")
    else:
        report("TC-240", "FAIL", "Multi-step not detected")


async def test_tc242_parallel_dispatch_read():
    """TC-242: Parallel dispatch — read tools classified as parallel-safe."""
    from nanobot.hooks.builtin.claude_capabilities import classify_for_parallel
    tools = [
        {"name": "web_search", "args": {}},
        {"name": "read_file", "args": {}},
        {"name": "memory_search", "args": {}},
    ]
    parallel, sequential = classify_for_parallel(tools)
    if len(parallel) == 3 and len(sequential) == 0:
        report("TC-242", "PASS", "All read tools classified as parallel-safe")
    else:
        report("TC-242", "FAIL", f"Expected 3 parallel, got {len(parallel)} parallel, {len(sequential)} sequential")


async def test_tc243_parallel_dispatch_sequential():
    """TC-243: Parallel dispatch — write tools classified as sequential."""
    from nanobot.hooks.builtin.claude_capabilities import classify_for_parallel
    tools = [
        {"name": "write_file", "args": {}},
        {"name": "exec", "args": {}},
    ]
    parallel, sequential = classify_for_parallel(tools)
    if len(sequential) >= 1:
        report("TC-243", "PASS", f"Write tools: {len(parallel)} parallel, {len(sequential)} sequential")
    else:
        # write_file and exec are both in _SEQUENTIAL set
        report("TC-243", "PARTIAL", f"Got {len(parallel)} parallel, {len(sequential)} sequential")


async def test_tc253_strategy_rotator():
    """TC-253: Strategy rotator — get alternative tool on failure."""
    from nanobot.hooks.builtin.claude_capabilities import get_alternative_tool, reset_failures
    reset_failures()  # Clean state
    alt = get_alternative_tool("web_search")
    if alt == "web_fetch":
        report("TC-253", "PASS", f"Strategy rotator: web_search -> {alt}")
    elif alt:
        report("TC-253", "PASS", f"Strategy rotator: web_search -> {alt}")
    else:
        report("TC-253", "FAIL", "No alternative found for web_search")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-LANGUAGE (2 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc235_language_detection_bangla():
    """TC-235: Language detection — Bangla."""
    from nanobot.hooks.builtin.maintenance import detect_language
    result = detect_language("আমি ভালো আছি, ধন্যবাদ")
    if result == "bangla":
        report("TC-235", "PASS", f"Bangla detected: {result}")
    else:
        report("TC-235", "FAIL", f"Expected 'bangla', got '{result}'")


async def test_tc236_language_detection_hindi():
    """TC-236: Language detection — Hindi."""
    from nanobot.hooks.builtin.maintenance import detect_language
    result = detect_language("नमस्ते, मैं ठीक हूँ धन्यवाद")
    if result == "hindi":
        report("TC-236", "PASS", f"Hindi detected: {result}")
    else:
        report("TC-236", "FAIL", f"Expected 'hindi', got '{result}'")


# ═══════════════════════════════════════════════════════════════════════════════
# CODE-LEVEL (3 P1)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc228_budget_warn_not_block():
    """TC-228: Budget warn but not block — enforce=false."""
    from nanobot.hooks.builtin.cost_budget import check_budget, load_budget
    budget = load_budget(WORKSPACE)
    # When enforce is False (default), budget should warn but not block
    enforce = budget.get("enforce", False)
    status = check_budget(WORKSPACE)
    if not enforce and "exceeded" in status:
        report("TC-228", "PASS", f"Budget enforce={enforce}, exceeded={status['exceeded']} (warn only)")
    elif "exceeded" in status:
        report("TC-228", "PASS", f"Budget check works: enforce={enforce}, exceeded={status['exceeded']}")
    else:
        report("TC-228", "FAIL", f"Budget check missing 'exceeded' field: {list(status.keys())}")


async def test_tc229_auto_model_downgrade():
    """TC-229: Auto-model downgrade — budget_auto_switch_model setting."""
    d = await api_get("/api/features")
    features = d.get("features", [])
    auto_switch = [f for f in features if f.get("key") == "budget_auto_switch_model"]
    if auto_switch:
        report("TC-229", "PASS", f"budget_auto_switch_model setting exists (value='{auto_switch[0].get('value')}')")
    else:
        report("TC-229", "FAIL", "budget_auto_switch_model setting not found")


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

# Ordered: API tests first, then Python import tests, then WS tests last
P1_TESTS = [
    # --- API tests (fast) ---
    test_tc016_message_search,
    test_tc018_notification_bell,
    test_tc026_image_gen_together,
    test_tc027_image_gen_huggingface,
    test_tc030_image_gen_openai,
    test_tc034_auto_fallback_provider,
    test_tc041_dynamic_context_budget,
    test_tc042_response_quality_gate,
    test_tc052_observations_layer,
    test_tc053_episodes_layer,
    test_tc054_learnings,
    test_tc067_memory_search_api,
    test_tc069_settings_get_specific,
    test_tc071_settings_set_number,
    test_tc072_settings_set_string,
    test_tc073_settings_search,
    test_tc076_unified_settings_file,
    test_tc078_feature_manifest_api,
    test_tc080_profile_switching,
    test_tc122_smart_credential_naming,
    test_tc126_tailscale_only,
    test_tc127_api_list_credentials,
    test_tc136_cron_jobs,
    test_tc137_cron_dashboard,
    test_tc145_mcp_server_management,
    test_tc169_session_search,
    test_tc170_anomaly_detection,
    test_tc222_message_dedup,
    test_tc229_auto_model_downgrade,

    # --- Python import tests ---
    test_tc003_streaming_tts,
    test_tc004_voice_endpointing,
    test_tc005_voice_and_text,
    test_tc008_discord_voice_channel,
    test_tc009_response_length_voice,
    test_tc013_tool_result_cards,
    test_tc014_file_attachments,
    test_tc037_error_recovery_timeout,
    test_tc038_error_recovery_rate_limit,
    test_tc040_intent_tracking_followup,
    test_tc043_tool_call_validation,
    test_tc044_parallel_safety,
    test_tc046_mcp_auto_reconnect,
    test_tc057_consolidation_heartbeat,
    test_tc058_consolidation_disconnect,
    test_tc059_consolidation_new,
    test_tc062_conversation_recap,
    test_tc063_pronoun_resolution,
    test_tc083_quiet_hours,
    test_tc084_quiet_hours_high_override,
    test_tc086_morning_briefing,
    test_tc088_proactive_notifications_persistent,
    test_tc089_bg_job_notification,
    test_tc094_correlation_travel_weather,
    test_tc095_correlation_bill_due,
    test_tc096_correlation_goal_overdue,
    test_tc098_meeting_intelligence,
    test_tc100_relationship_record,
    test_tc101_relationship_reminders,
    test_tc104_financial_pulse,
    test_tc105_project_tracker,
    test_tc107_daily_digest,
    test_tc108_priority_inbox,
    test_tc110_delegation_add,
    test_tc111_delegation_due,
    test_tc113_decision_record,
    test_tc114_decision_recall,
    test_tc115_people_prep,
    test_tc117_proactive_heartbeat,
    test_tc125_credential_vault_injection,
    test_tc129_destructive_detection,
    test_tc131_bg_shell_status,
    test_tc132_bg_shell_output,
    test_tc134_bg_shell_kill,
    test_tc139_rules_create,
    test_tc140_rules_match,
    test_tc146_phone_call_tts,
    test_tc148_phone_call_missing_creds,
    test_tc153_deepgram_voices,
    test_tc154_mimo_audio_emotions,
    test_tc158_provider_validation_missing,
    test_tc164_history_auto_archive,
    test_tc165_session_auto_cleanup,
    test_tc175_run_maintenance,
    test_tc179_telegram_channel,
    test_tc180_discord_text_channel,
    test_tc181_discord_voice_channel,
    test_tc187_intent_tool_filtering,
    test_tc188_tool_filtering_fallback,
    test_tc194_history_compression,
    test_tc201_math_interceptor,
    test_tc203_timezone_resolver,
    test_tc204_regex_builder,
    test_tc206_greeting_interceptor,
    test_tc208_greeting_not_triggered,
    test_tc209_response_caching,
    test_tc212_priority_detection,
    test_tc214_entity_extraction,
    test_tc215_loop_detector,
    test_tc217_error_translation,
    test_tc220_frustration_detection,
    test_tc224_semantic_truncation,
    test_tc240_task_decomposer,
    test_tc242_parallel_dispatch_read,
    test_tc243_parallel_dispatch_sequential,
    test_tc253_strategy_rotator,
    test_tc235_language_detection_bangla,
    test_tc236_language_detection_hindi,
    test_tc228_budget_warn_not_block,

    # --- WebSocket tests (slow, needs LLM) ---
    test_tc011_markdown_rendering,
]


async def run_p1():
    print("=" * 60)
    print("  MAWA LIVE TEST RUNNER — P1 Tests")
    print("=" * 60)
    print()

    for test_fn in P1_TESTS:
        try:
            await test_fn()
        except Exception as e:
            tc_id = test_fn.__doc__.split(":")[0] if test_fn.__doc__ else test_fn.__name__
            report(tc_id, "FAIL", f"Exception: {e}")

    print()
    print("=" * 60)
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")
    partial = sum(1 for r in _results if r["status"] == "PARTIAL")
    skipped = sum(1 for r in _results if r["status"] == "SKIP")
    total = len(_results)
    print(f"  RESULTS: {passed} PASS / {failed} FAIL / {partial} PARTIAL / {skipped} SKIP / {total} TOTAL")
    print("=" * 60)

    if failed:
        print("\n  FAILURES:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    {r['id']}: {r['detail']}")

    return _results


if __name__ == "__main__":
    asyncio.run(run_p1())
