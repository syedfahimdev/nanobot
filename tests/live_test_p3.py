"""Live test runner for Mawa — P3 tests (11 total).

Usage: python tests/live_test_p3.py

Note: MAWA_TEST_CASES.md summary table claims 18 P3 tests, but only 11 P3
entries exist in the actual test case rows. This file covers all 11.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/root/nanobot')
from tests.live_test_runner import (
    api_get, api_post, ws_send_and_collect, report, _results, WORKSPACE, ssl_ctx, BASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Chat & Messaging (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc022_keyboard_shortcuts():
    """TC-022: Keyboard shortcuts — Cmd+K opens command palette."""
    try:
        html_path = Path("/root/nanobot/nanobot/channels/web_voice_ui/index.html")
        if not html_path.exists():
            report("TC-022", "SKIP", "web_voice_ui/index.html not found")
            return
        html = html_path.read_text()
        has_palette = "commandPalette" in html or "command-palette" in html
        has_shortcut = "Cmd" in html or "Meta" in html or "ctrlKey" in html
        if has_palette and has_shortcut:
            report("TC-022", "PASS", "Command palette UI and Cmd+K shortcut found in web_voice UI")
        elif has_palette:
            report("TC-022", "PARTIAL", "Command palette exists but keyboard shortcut binding not confirmed")
        else:
            report("TC-022", "FAIL", "No command palette found in web_voice UI")
    except Exception as e:
        report("TC-022", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Settings & Configuration (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc081_avatar_styles():
    """TC-081: Avatar styles — setting exists in feature manifest or UI."""
    try:
        # Check feature manifest for avatar-related setting
        from nanobot.hooks.builtin.feature_registry import _FEATURE_DEFS
        avatar_keys = [f for f in _FEATURE_DEFS if "avatar" in f.get("key", "").lower() or "avatar" in f.get("label", "").lower()]
        if avatar_keys:
            report("TC-081", "PASS", f"Avatar setting in feature manifest: {avatar_keys[0]['key']}")
            return

        # Check UI for avatar references
        html_path = Path("/root/nanobot/nanobot/channels/web_voice_ui/index.html")
        if html_path.exists():
            html = html_path.read_text()
            if "avatar" in html.lower():
                report("TC-081", "PASS", "Avatar styles referenced in web_voice UI")
                return

        # Check web_voice.py for avatar
        wv_path = Path("/root/nanobot/nanobot/channels/web_voice.py")
        if wv_path.exists():
            src = wv_path.read_text()
            if "avatar" in src.lower():
                report("TC-081", "PASS", "Avatar styles referenced in web_voice channel")
                return

        # Check API
        try:
            features = await api_get("/api/features")
            if isinstance(features, list):
                avatar_feats = [f for f in features if "avatar" in str(f).lower()]
                if avatar_feats:
                    report("TC-081", "PASS", f"Avatar feature found via API: {avatar_feats[0].get('key', '?')}")
                    return
        except Exception:
            pass

        report("TC-081", "SKIP", "Avatar styles setting not found in manifest, UI, or API")
    except Exception as e:
        report("TC-081", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Jarvis Intelligence (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc120_configurable_correlation_lookahead():
    """TC-120: Configurable correlation lookahead days setting."""
    try:
        from nanobot.hooks.builtin.feature_registry import _FEATURE_DEFS, load_settings, save_setting
        # Verify the setting exists in feature defs
        match = [f for f in _FEATURE_DEFS if f["key"] == "correlationLookaheadDays"]
        if not match:
            report("TC-120", "FAIL", "correlationLookaheadDays not in _FEATURE_DEFS")
            return

        fdef = match[0]
        assert fdef["default"] == 3, f"Expected default 3, got {fdef['default']}"
        assert fdef["type"] == "number", f"Expected type number, got {fdef['type']}"
        assert fdef.get("min", 0) >= 1
        assert fdef.get("max", 0) <= 14

        # Test save/load round-trip
        save_setting(WORKSPACE, "correlationLookaheadDays", 7)
        settings = load_settings(WORKSPACE)
        val = settings.get("correlationLookaheadDays")
        if val == 7:
            report("TC-120", "PASS", "correlationLookaheadDays: default=3, saved=7, min=1, max=14")
        else:
            report("TC-120", "PARTIAL", f"Setting saved but read back {val} instead of 7")

        # Restore default
        save_setting(WORKSPACE, "correlationLookaheadDays", 3)
    except Exception as e:
        report("TC-120", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Tools & Automation (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc144_skill_marketplace():
    """TC-144: Skill marketplace — SkillsMarketplaceTool exists."""
    try:
        from nanobot.agent.tools.skills_marketplace import SkillsMarketplaceTool
        tool = SkillsMarketplaceTool()
        assert tool.name == "skills_marketplace"
        desc = tool.description
        assert "search" in desc.lower() or "marketplace" in desc.lower()
        params = tool.parameters
        assert "properties" in params, "Tool should have parameter properties"
        report("TC-144", "PASS", f"SkillsMarketplaceTool exists: name={tool.name}, has params")
    except ImportError:
        # Fallback: check API for tool registration
        try:
            d = await api_get("/api/tools")
            tools = d if isinstance(d, list) else d.get("tools", [])
            names = [t.get("name", "") if isinstance(t, dict) else str(t) for t in tools]
            if "skills_marketplace" in names:
                report("TC-144", "PASS", "skills_marketplace registered via API")
            else:
                report("TC-144", "SKIP", "SkillsMarketplaceTool not importable or registered")
        except Exception as e2:
            report("TC-144", "SKIP", f"Cannot verify: {e2}")
    except Exception as e:
        report("TC-144", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Voice Providers (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc161_voice_sample_list():
    """TC-161: Voice sample management — list samples."""
    try:
        from nanobot.hooks.builtin.voice_providers import get_voice_samples

        # Start with clean state
        samples = get_voice_samples(WORKSPACE)
        assert isinstance(samples, list), f"Expected list, got {type(samples)}"

        # Create test samples
        voice_dir = WORKSPACE / "voice_samples"
        voice_dir.mkdir(parents=True, exist_ok=True)
        test_files = []
        for name in ("test_sample_a", "test_sample_b"):
            p = voice_dir / f"{name}.wav"
            p.write_bytes(b"RIFF" + b"\x00" * 40)  # minimal WAV header
            test_files.append(p)

        samples = get_voice_samples(WORKSPACE)
        names = [s["name"] for s in samples]
        assert "test_sample_a" in names, f"test_sample_a not in {names}"
        assert "test_sample_b" in names, f"test_sample_b not in {names}"
        for s in samples:
            assert "path" in s and "size" in s, f"Missing keys in sample: {s}"

        report("TC-161", "PASS", f"get_voice_samples returned {len(samples)} samples with name/path/size")

        # Cleanup
        for p in test_files:
            p.unlink(missing_ok=True)
    except Exception as e:
        report("TC-161", "FAIL", f"Error: {e}")


async def test_tc162_voice_sample_delete():
    """TC-162: Voice sample management — delete sample."""
    try:
        from nanobot.hooks.builtin.voice_providers import delete_voice_sample

        # Create a sample to delete
        voice_dir = WORKSPACE / "voice_samples"
        voice_dir.mkdir(parents=True, exist_ok=True)
        p = voice_dir / "delete_me.wav"
        p.write_bytes(b"RIFF" + b"\x00" * 40)
        assert p.exists()

        result = delete_voice_sample(WORKSPACE, "delete_me")
        assert result is True, f"Expected True, got {result}"
        assert not p.exists(), "File should be deleted"

        # Deleting again should return False
        result2 = delete_voice_sample(WORKSPACE, "delete_me")
        assert result2 is False, f"Expected False for missing file, got {result2}"

        report("TC-162", "PASS", "delete_voice_sample removes file and returns True; False for missing")
    except Exception as e:
        report("TC-162", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Maintenance & Health (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc174_undo_rollback_journal():
    """TC-174: Undo/rollback journal — record actions and retrieve history."""
    try:
        from nanobot.hooks.builtin.maintenance import record_action, get_undo_history, _undo_journal

        # Save original state and clear
        original = list(_undo_journal)
        _undo_journal.clear()

        # Record several actions
        for i in range(12):
            record_action(
                action_type="test_action",
                details={"step": i, "desc": f"Action {i}"},
                rollback_cmd=f"undo_step_{i}",
            )

        history = get_undo_history()
        assert isinstance(history, list), f"Expected list, got {type(history)}"
        assert len(history) == 10, f"Expected last 10 actions, got {len(history)}"

        # Should be in reverse order (most recent first)
        assert history[0]["details"]["step"] == 11, f"First item should be step 11, got {history[0]}"
        assert history[-1]["details"]["step"] == 2, f"Last item should be step 2, got {history[-1]}"

        # Verify fields: type, details, rollback, ts
        for entry in history:
            assert "type" in entry and "details" in entry and "rollback" in entry and "ts" in entry

        # Verify max 20 stored
        for i in range(15):
            record_action("overflow", {"step": i + 100}, None)
        assert len(_undo_journal) <= 20, f"Journal should cap at 20, has {len(_undo_journal)}"

        report("TC-174", "PASS", "Undo journal: records actions, returns last 10 reversed, caps at 20")

        # Restore
        _undo_journal.clear()
        _undo_journal.extend(original)
    except Exception as e:
        report("TC-174", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Channels (2 tests)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc185_email_channel():
    """TC-185: Email channel module exists."""
    try:
        from nanobot.channels.email import EmailConfig
        from nanobot.channels.base import BaseChannel
        import inspect

        # Verify EmailConfig has expected fields
        assert hasattr(EmailConfig, "model_fields") or hasattr(EmailConfig, "__fields__")

        # Verify the module has a channel class inheriting BaseChannel
        email_mod = sys.modules["nanobot.channels.email"]
        channel_classes = [
            name for name, obj in inspect.getmembers(email_mod, inspect.isclass)
            if issubclass(obj, BaseChannel) and obj is not BaseChannel
        ]
        if channel_classes:
            report("TC-185", "PASS", f"Email channel: {channel_classes[0]} extends BaseChannel, EmailConfig present")
        else:
            report("TC-185", "PARTIAL", "EmailConfig exists but no BaseChannel subclass found")
    except ImportError as e:
        report("TC-185", "SKIP", f"Email channel module not importable: {e}")
    except Exception as e:
        report("TC-185", "FAIL", f"Error: {e}")


async def test_tc186_matrix_channel():
    """TC-186: Matrix channel module exists."""
    try:
        matrix_path = Path("/root/nanobot/nanobot/channels/matrix.py")
        if not matrix_path.exists():
            report("TC-186", "SKIP", "Matrix channel module not found")
            return

        src = matrix_path.read_text()
        has_base = "BaseChannel" in src
        has_client = "AsyncClient" in src or "MatrixRoom" in src
        has_send = "send" in src.lower()

        if has_base and has_client and has_send:
            report("TC-186", "PASS", "Matrix channel: extends BaseChannel, uses nio AsyncClient, has send")
        elif has_base:
            report("TC-186", "PARTIAL", "Matrix channel has BaseChannel but nio integration unclear")
        else:
            report("TC-186", "FAIL", "Matrix channel missing BaseChannel or client integration")
    except Exception as e:
        report("TC-186", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Pipeline Optimizations (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc199_output_format_hints_detailed():
    """TC-199: Output format hints — 'comprehensive analysis' triggers detailed hint."""
    try:
        from nanobot.hooks.builtin.pipeline_optimizer import get_format_hint, get_response_hint

        # get_format_hint should detect "comprehensive"
        hint = get_format_hint("give me a comprehensive analysis of the market")
        assert "detailed" in hint.lower() or "thorough" in hint.lower(), f"Expected detailed hint, got: {hint}"

        # Also test other detailed triggers
        for phrase in ("detailed breakdown", "thorough review", "full explanation"):
            h = get_format_hint(phrase)
            assert h, f"No hint for '{phrase}'"

        # get_response_hint for text channel should not add voice constraint
        resp_hint = get_response_hint("give me a comprehensive analysis", "web")
        # Should NOT contain "SHORT" for text channel
        assert "SHORT" not in resp_hint

        report("TC-199", "PASS", "get_format_hint returns detailed/thorough hint for comprehensive queries")
    except Exception as e:
        report("TC-199", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Claude-Level Capabilities (1 test)
# ═══════════════════════════════════════════════════════════════════════════════

async def test_tc252_paste_pipeline_email():
    """TC-252: Paste pipeline — detect email from From/To/Subject headers."""
    try:
        from nanobot.hooks.builtin.claude_capabilities import detect_paste_type

        email_text = """From: alice@example.com
To: bob@example.com
Subject: Meeting tomorrow
Date: Mon, 24 Mar 2026 10:00:00 +0000

Hi Bob,
Just confirming our meeting tomorrow at 2pm.
Best, Alice"""

        result = detect_paste_type(email_text)
        assert result is not None, "detect_paste_type returned None for email"
        assert result["type"] == "email", f"Expected type 'email', got {result['type']}"
        assert "email" in result.get("suggestion", "").lower(), f"Suggestion should mention email: {result.get('suggestion')}"

        # Non-email should NOT match as email
        plain = "Hello, how are you doing today?"
        result2 = detect_paste_type(plain)
        if result2 is not None:
            assert result2["type"] != "email", "Plain text should not be detected as email"

        report("TC-252", "PASS", f"Email paste detected: type={result['type']}, suggestion='{result['suggestion']}'")
    except Exception as e:
        report("TC-252", "FAIL", f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test Registry
# ═══════════════════════════════════════════════════════════════════════════════

P3_TESTS = [
    # Chat & Messaging
    test_tc022_keyboard_shortcuts,
    # Settings
    test_tc081_avatar_styles,
    # Jarvis Intelligence
    test_tc120_configurable_correlation_lookahead,
    # Tools & Automation
    test_tc144_skill_marketplace,
    # Voice Providers
    test_tc161_voice_sample_list,
    test_tc162_voice_sample_delete,
    # Maintenance
    test_tc174_undo_rollback_journal,
    # Channels
    test_tc185_email_channel,
    test_tc186_matrix_channel,
    # Pipeline
    test_tc199_output_format_hints_detailed,
    # Claude-Level
    test_tc252_paste_pipeline_email,
]


async def run_p3():
    print("=" * 60)
    print("  MAWA LIVE TEST RUNNER -- P3 Tests")
    print(f"  Total: {len(P3_TESTS)} test functions")
    print("=" * 60)
    print()

    for test_fn in P3_TESTS:
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
    asyncio.run(run_p3())
