"""Native Playwright browser tool — visible via noVNC with persistent profile.

Features:
  - Live browser view via noVNC (accessible through Tailscale)
  - Persistent browser profile (cookies, sessions survive restarts)
  - Auto-screenshot on every action (fallback for mobile)
  - Location/geolocation spoofing
  - Full page scraping + link extraction
  - No ToolsDNS, no Composio, no MCP — runs Playwright directly
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool

_playwright = None
_browser = None
_context = None
_page = None
_lock = asyncio.Lock()

# Persistent profile directory
_PROFILE_DIR = Path.home() / ".nanobot" / "browser-profile"

# noVNC display configuration
_DISPLAY = os.environ.get("DISPLAY", ":99")
_NOVNC_PORT = 6080
_SCREENSHOT_DIR = Path.home() / ".nanobot" / "workspace" / "memory" / "media"


def _get_live_url() -> str:
    """Get the noVNC live URL via Tailscale."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            import json as _j
            data = _j.loads(result.stdout)
            dns = data.get("Self", {}).get("DNSName", "").rstrip(".")
            if dns:
                return f"https://{dns}:{_NOVNC_PORT}/vnc.html?autoconnect=true&resize=scale"
    except Exception:
        pass
    # Fallback to IP
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ip = result.stdout.strip().split("\n")[0]
            return f"http://{ip}:{_NOVNC_PORT}/vnc.html?autoconnect=true&resize=scale"
    except Exception:
        pass
    return f"http://localhost:{_NOVNC_PORT}/vnc.html?autoconnect=true"


async def _ensure_browser(
    geolocation: dict | None = None,
):
    """Lazy-init: start Playwright browser with persistent profile on Xvfb display."""
    global _playwright, _browser, _context, _page
    if _page is not None:
        return _page

    async with _lock:
        if _page is not None:
            return _page
        try:
            from playwright.async_api import async_playwright

            _PROFILE_DIR.mkdir(parents=True, exist_ok=True)

            # Set DISPLAY so browser renders on the Xvfb that noVNC is streaming
            os.environ["DISPLAY"] = _DISPLAY

            _playwright = await async_playwright().start()

            # Use persistent context for cookie/session persistence
            # Run HEADED on Xvfb — visible via noVNC
            context_opts: dict[str, Any] = {
                "viewport": {"width": 1280, "height": 720},
                "user_agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "ignore_https_errors": True,
                "accept_downloads": True,
            }

            if geolocation:
                context_opts["geolocation"] = geolocation
                context_opts["permissions"] = ["geolocation"]

            # Headed mode on Xvfb — visible via noVNC
            _context = await _playwright.chromium.launch_persistent_context(
                str(_PROFILE_DIR),
                headless=False,  # HEADED — renders on Xvfb, visible via noVNC
                args=[
                    "--no-sandbox", "--disable-gpu",
                    "--disable-blink-features=AutomationControlled",
                    f"--display={_DISPLAY}",
                    "--window-size=1280,720",
                ],
                **context_opts,
            )

            # Use existing page or create new one
            if _context.pages:
                _page = _context.pages[0]
            else:
                _page = await _context.new_page()

            logger.info("Browser: Playwright started with persistent profile at {}", _PROFILE_DIR)
            return _page
        except ImportError:
            raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install chromium")
        except Exception as e:
            raise RuntimeError(f"Failed to start browser: {e}")


async def _close_browser():
    """Close the browser instance (profile is preserved on disk)."""
    global _playwright, _browser, _context, _page
    if _context:
        await _context.close()
    if _playwright:
        await _playwright.stop()
    _playwright = _browser = _context = _page = None


class BrowserTool(Tool):
    """Browse the web with a persistent profile — cookies and logins are remembered."""

    _screenshot_counter = 0

    async def _save_screenshot(self, page, action_name: str) -> str | None:
        """Auto-save screenshot after actions as fallback for mobile users."""
        try:
            _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            self._screenshot_counter += 1
            path = _SCREENSHOT_DIR / f"browser_{self._screenshot_counter:03d}_{action_name}.png"
            await page.screenshot(path=str(path), type="png")
            return str(path)
        except Exception:
            return None

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Control a real web browser (Playwright Chromium) with a persistent profile. "
            "Cookies and login sessions are preserved across calls and restarts. "
            "Actions: navigate, screenshot, click, fill, text, scrape, links, scroll, "
            "evaluate, wait, set_location, close."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "navigate", "screenshot", "click", "fill", "text",
                        "scrape", "links", "scroll", "evaluate", "wait",
                        "set_location", "close",
                    ],
                    "description": "Action to perform.",
                },
                "url": {
                    "type": "string",
                    "description": "URL to navigate to (for navigate action).",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector or text selector (e.g., 'text=Click me', '#login-btn', '.content').",
                },
                "value": {
                    "type": "string",
                    "description": "Value to fill. Use {cred:name} to inject a saved credential securely (e.g., {cred:outlook}).",
                },
                "script": {
                    "type": "string",
                    "description": "JavaScript to evaluate (for evaluate action).",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction (default: down).",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Take full-page screenshot (default: false, viewport only).",
                },
                "latitude": {
                    "type": "number",
                    "description": "Latitude for set_location (e.g., 40.7128 for NYC).",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude for set_location (e.g., -74.0060 for NYC).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 30000).",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        url: str = "",
        selector: str = "",
        value: str = "",
        script: str = "",
        direction: str = "down",
        full_page: bool = False,
        latitude: float | None = None,
        longitude: float | None = None,
        timeout: int = 30000,
        **kwargs: Any,
    ) -> str:
        try:
            if action == "close":
                await _close_browser()
                return "Browser closed. Profile saved — cookies and sessions preserved for next time."

            if action == "set_location":
                if latitude is None or longitude is None:
                    return "Error: latitude and longitude required for set_location."
                # Close and reopen with new geolocation
                await _close_browser()
                await _ensure_browser(geolocation={"latitude": latitude, "longitude": longitude})
                return f"Location set to ({latitude}, {longitude}). Browser restarted with new location."

            page = await _ensure_browser()

            if action == "navigate":
                if not url:
                    return "Error: url required for navigate action."
                resp = await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                title = await page.title()
                status = resp.status if resp else "unknown"
                current_url = page.url
                live_url = _get_live_url()
                # Auto-screenshot as fallback
                await self._save_screenshot(page, "navigate")
                return (
                    f"Navigated to {current_url}\n"
                    f"Title: {title}\n"
                    f"Status: {status}\n"
                    f"Live view: {live_url}"
                )

            elif action == "screenshot":
                screenshot = await page.screenshot(type="png", full_page=full_page)
                b64 = base64.b64encode(screenshot).decode("utf-8")
                title = await page.title()
                # Save to workspace for file attachment detection
                save_path = Path.home() / ".nanobot" / "workspace" / "memory" / "media" / "screenshot.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(screenshot)
                return f"Screenshot of {page.url} ({title})\nSaved to: {save_path}\nSize: {len(screenshot)} bytes"

            elif action == "click":
                if not selector:
                    return "Error: selector required for click action."
                await page.click(selector, timeout=timeout)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                title = await page.title()
                await self._save_screenshot(page, "click")
                return f"Clicked '{selector}'. Page: {title} ({page.url})"

            elif action == "fill":
                if not selector or value is None:
                    return "Error: selector and value required for fill action."
                # Resolve {cred:name} references to actual credential values
                from nanobot.agent.tools.credentials import resolve_credential_refs
                actual_value = resolve_credential_refs(value)
                is_secret = actual_value != value  # Was a credential reference
                await page.fill(selector, actual_value, timeout=timeout)
                await self._save_screenshot(page, "fill")
                if is_secret:
                    return f"Filled '{selector}' with saved credential (value masked)."
                return f"Filled '{selector}' with value."

            elif action == "text":
                if not selector:
                    text = await page.inner_text("body")
                else:
                    text = await page.inner_text(selector, timeout=timeout)
                if len(text) > 5000:
                    text = text[:5000] + f"\n... (truncated, {len(text)} chars total)"
                return text

            elif action == "scrape":
                """Full page scrape — extracts all visible text, headings, and metadata."""
                result = await page.evaluate("""() => {
                    const data = {
                        title: document.title,
                        url: window.location.href,
                        description: document.querySelector('meta[name="description"]')?.content || '',
                        headings: [],
                        text: '',
                        images: [],
                    };
                    // Headings
                    document.querySelectorAll('h1,h2,h3').forEach(h => {
                        data.headings.push({tag: h.tagName, text: h.innerText.trim()});
                    });
                    // Main text
                    const main = document.querySelector('main, article, [role="main"], .content, #content');
                    data.text = (main || document.body).innerText.substring(0, 10000);
                    // Images with alt text
                    document.querySelectorAll('img[alt]').forEach(img => {
                        if (img.alt.trim()) data.images.push({alt: img.alt, src: img.src?.substring(0, 200)});
                    });
                    return data;
                }""")

                lines = [f"# {result.get('title', '')}", f"URL: {result.get('url', '')}"]
                desc = result.get('description', '')
                if desc:
                    lines.append(f"Description: {desc}")
                for h in result.get('headings', [])[:20]:
                    prefix = "#" * int(h['tag'][1])
                    lines.append(f"{prefix} {h['text']}")
                lines.append("")
                lines.append(result.get('text', '')[:8000])
                return "\n".join(lines)

            elif action == "links":
                """Extract all links from the page."""
                links = await page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('a[href]'))
                        .map(a => ({text: a.innerText.trim(), href: a.href}))
                        .filter(l => l.text && l.href && !l.href.startsWith('javascript:'))
                        .slice(0, 50);
                }""")
                if not links:
                    return "No links found on page."
                lines = [f"Found {len(links)} links:\n"]
                for l in links:
                    text = l.get('text', '')[:60]
                    href = l.get('href', '')
                    lines.append(f"- [{text}]({href})")
                return "\n".join(lines)

            elif action == "scroll":
                delta = -500 if direction == "up" else 500
                await page.mouse.wheel(0, delta)
                await asyncio.sleep(0.5)
                return f"Scrolled {direction}."

            elif action == "evaluate":
                if not script:
                    return "Error: script required for evaluate action."
                result = await page.evaluate(script)
                return str(result)[:3000]

            elif action == "wait":
                if selector:
                    await page.wait_for_selector(selector, timeout=timeout)
                    return f"Element '{selector}' appeared."
                else:
                    await asyncio.sleep(min(timeout / 1000, 10))
                    return f"Waited {min(timeout, 10000)}ms."

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            error_msg = str(e)
            if "Timeout" in error_msg:
                return f"Timeout: {error_msg[:200]}"
            if "Target closed" in error_msg or "Browser" in error_msg:
                await _close_browser()
                return f"Browser session expired: {error_msg[:200]}. It will restart on next call."
            return f"Browser error: {error_msg[:300]}"
