"""Generative UI tools — Chart.js charts, HTML tables, dashboards rendered inline.

Each tool fetches real data from free APIs (no keys needed) and returns
structured results with pre-built html-widget blocks for zero-token rendering.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG = "#0a0a0a"
TEXT = "#e5e5e5"
ACCENT = "#14b8a6"
SECONDARY = "#f59e0b"
MUTED = "#6b7280"
CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js"


def _widget(html: str) -> str:
    """Wrap raw HTML in the html-widget fence."""
    return f"\n```html-widget\n{html}\n```"


def _card(inner: str) -> str:
    """Dark-theme card wrapper."""
    return (
        f'<div style="background:{BG};color:{TEXT};padding:16px;'
        f'border-radius:12px;font-family:system-ui;max-width:600px">'
        f"{inner}</div>"
    )


async def _geocode(city: str) -> tuple[float, float, str]:
    """Resolve city name to (lat, lon, display_name) via Open-Meteo geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params={"name": city, "count": 1, "language": "en"})
        r.raise_for_status()
        data = r.json()
    results = data.get("results")
    if not results:
        raise ValueError(f"Could not geocode '{city}'")
    loc = results[0]
    name = f"{loc.get('name', city)}, {loc.get('country', '')}"
    return loc["latitude"], loc["longitude"], name


# ===== 1. WeatherTool =====================================================

class WeatherTool(Tool):
    """Current weather + 7-day forecast via Open-Meteo (free, no key)."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Get current weather and 7-day forecast for a location. "
            "Returns a visual weather dashboard widget."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {"type": "string", "description": "City name or place"},
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        location: str = kwargs["location"]
        try:
            lat, lon, display = await _geocode(location)
        except Exception as e:
            return f"Could not find location '{location}': {e}"

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto",
            "forecast_days": 7,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            return f"Weather API error: {e}"

        cur = data["current"]
        daily = data["daily"]
        temp = cur["temperature_2m"]
        humidity = cur["relative_humidity_2m"]
        wind = cur["wind_speed_10m"]
        code = cur["weather_code"]

        wmo = {
            0: ("Clear", "☀️"), 1: ("Mostly Clear", "🌤"),
            2: ("Partly Cloudy", "⛅"), 3: ("Overcast", "☁️"),
            45: ("Fog", "🌫"), 48: ("Rime Fog", "🌫"),
            51: ("Light Drizzle", "🌦"), 53: ("Drizzle", "🌧"),
            55: ("Heavy Drizzle", "🌧"), 61: ("Light Rain", "🌦"),
            63: ("Rain", "🌧"), 65: ("Heavy Rain", "🌧"),
            71: ("Light Snow", "🌨"), 73: ("Snow", "❄️"),
            75: ("Heavy Snow", "❄️"), 80: ("Showers", "🌧"),
            95: ("Thunderstorm", "⛈"), 96: ("Hail Storm", "⛈"),
        }
        desc, icon = wmo.get(code, ("Unknown", "🌡"))

        # Build forecast bars
        bars = ""
        for i in range(min(7, len(daily["time"]))):
            day_label = daily["time"][i][5:]  # MM-DD
            hi = daily["temperature_2m_max"][i]
            lo = daily["temperature_2m_min"][i]
            d_code = daily["weather_code"][i]
            d_icon = wmo.get(d_code, ("", "🌡"))[1]
            pct = max(5, min(100, int((hi + 10) / 50 * 100)))
            bars += (
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                f'<span style="width:50px;font-size:12px;color:{MUTED}">{day_label}</span>'
                f'<span>{d_icon}</span>'
                f'<div style="flex:1;background:#1a1a2e;border-radius:4px;height:8px">'
                f'<div style="width:{pct}%;background:linear-gradient(90deg,{ACCENT},{SECONDARY});'
                f'height:100%;border-radius:4px"></div></div>'
                f'<span style="font-size:12px;width:70px;text-align:right">{lo}° / {hi}°</span>'
                f"</div>"
            )

        html = _card(
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div><div style="font-size:12px;color:{MUTED}">{display}</div>'
            f'<div style="font-size:36px;font-weight:700">{temp}°C</div>'
            f'<div style="color:{ACCENT}">{desc}</div></div>'
            f'<div style="font-size:48px">{icon}</div></div>'
            f'<div style="display:flex;gap:16px;margin:12px 0;font-size:13px;color:{MUTED}">'
            f"<span>💧 {humidity}%</span><span>💨 {wind} km/h</span></div>"
            f'<div style="border-top:1px solid #222;padding-top:12px;margin-top:8px">'
            f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px">7-Day Forecast</div>'
            f"{bars}</div>"
        )

        return f"Weather for {display}: {temp}°C, {desc}, humidity {humidity}%, wind {wind} km/h\n{_widget(html)}"


# ===== 2. StockChartTool ==================================================

class StockChartTool(Tool):
    """Stock price chart via Yahoo Finance (free, no key)."""

    @property
    def name(self) -> str:
        return "stock_chart"

    @property
    def description(self) -> str:
        return "Get stock price chart for a ticker symbol. Returns a Chart.js line chart widget."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["symbol"],
            "properties": {
                "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
                "range": {
                    "type": "string",
                    "description": "Time range: 1d, 5d, 1mo, 3mo, 6mo, 1y",
                    "default": "1mo",
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        symbol = kwargs["symbol"].upper()
        rng = kwargs.get("range", "1mo")
        interval_map = {"1d": "5m", "5d": "15m", "1mo": "1d", "3mo": "1d", "6mo": "1wk", "1y": "1wk"}
        interval = interval_map.get(rng, "1d")

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url, params={"range": rng, "interval": interval}, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            return f"Failed to fetch stock data for {symbol}: {e}"

        result = data.get("chart", {}).get("result")
        if not result:
            return f"No data found for symbol '{symbol}'"

        meta = result[0].get("meta", {})
        timestamps = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])

        if not timestamps or not closes:
            return f"No price data for {symbol}"

        # Clean nulls
        labels = []
        prices = []
        for ts, c in zip(timestamps, closes):
            if c is not None:
                labels.append(time.strftime("%m/%d", time.gmtime(ts)))
                prices.append(round(c, 2))

        if not prices:
            return f"No valid price data for {symbol}"

        cur_price = prices[-1]
        prev_price = prices[0]
        change = cur_price - prev_price
        pct = (change / prev_price * 100) if prev_price else 0
        color = ACCENT if change >= 0 else "#ef4444"
        arrow = "▲" if change >= 0 else "▼"
        currency = meta.get("currency", "USD")

        chart_id = f"chart_{symbol}_{int(time.time())}"
        labels_json = json.dumps(labels)
        prices_json = json.dumps(prices)

        html = _card(
            f'<div style="margin-bottom:12px">'
            f'<div style="font-size:12px;color:{MUTED}">{symbol} &middot; {rng}</div>'
            f'<div style="font-size:28px;font-weight:700">{currency} {cur_price:.2f}</div>'
            f'<div style="color:{color};font-size:14px">{arrow} {abs(change):.2f} ({abs(pct):.2f}%)</div>'
            f"</div>"
            f'<canvas id="{chart_id}" height="200"></canvas>'
            f'<script src="{CHARTJS_CDN}"></script>'
            f"<script>"
            f"new Chart(document.getElementById('{chart_id}'),{{"
            f"type:'line',data:{{labels:{labels_json},"
            f"datasets:[{{data:{prices_json},borderColor:'{color}',backgroundColor:'{color}22',"
            f"fill:true,tension:0.3,pointRadius:0,borderWidth:2}}]}},"
            f"options:{{responsive:true,plugins:{{legend:{{display:false}}}},"
            f"scales:{{x:{{display:true,ticks:{{maxTicksLimit:6,color:'{MUTED}'}},grid:{{color:'#1a1a2e'}}}},"
            f"y:{{ticks:{{color:'{MUTED}'}},grid:{{color:'#1a1a2e'}}}}}}}}"
            f"}});</script>"
        )

        return (
            f"{symbol}: {currency} {cur_price:.2f} ({arrow}{abs(pct):.2f}% over {rng})\n"
            f"{_widget(html)}"
        )


# ===== 3. QRCodeTool ======================================================

class QRCodeTool(Tool):
    """Generate QR code as inline SVG (pure Python, no dependencies)."""

    @property
    def name(self) -> str:
        return "qr_code"

    @property
    def description(self) -> str:
        return "Generate a QR code for any text or URL. Returns an SVG widget."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {"type": "string", "description": "Text or URL to encode"},
                "size": {"type": "integer", "description": "Size in pixels", "default": 256},
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        content = kwargs["content"]
        size = kwargs.get("size", 256)

        # Use a free QR code API to get SVG
        try:
            url = "https://api.qrserver.com/v1/create-qr-code/"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, params={
                    "data": content, "size": f"{size}x{size}",
                    "format": "svg", "bgcolor": "0a0a0a", "color": "e5e5e5",
                })
                r.raise_for_status()
                svg = r.text
        except Exception:
            # Fallback: use a simple HTML/canvas approach
            svg = (
                f'<div style="text-align:center;padding:20px">'
                f'<img src="https://api.qrserver.com/v1/create-qr-code/'
                f'?data={content}&size={size}x{size}&bgcolor=0a0a0a&color=e5e5e5" '
                f'alt="QR Code" style="max-width:100%;border-radius:8px"/></div>'
            )

        html = _card(
            f'<div style="text-align:center">'
            f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px">QR Code</div>'
            f'<div style="display:inline-block;background:#fff;padding:12px;border-radius:8px">'
            f"{svg}</div>"
            f'<div style="font-size:11px;color:{MUTED};margin-top:8px;'
            f'word-break:break-all;max-width:280px;margin-left:auto;margin-right:auto">'
            f"{content[:120]}{'...' if len(content) > 120 else ''}</div>"
            f"</div>"
        )

        return f"QR code generated for: {content[:80]}\n{_widget(html)}"


# ===== 4. CodeRunnerTool ==================================================

class CodeRunnerTool(Tool):
    """Run Python code in a sandboxed subprocess with timeout."""

    @property
    def name(self) -> str:
        return "run_code"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a restricted sandbox (10s timeout, no network). "
            "If the output looks like numeric/list data, a Chart.js visualization is included."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "language": {
                    "type": "string",
                    "description": "Language (only python supported)",
                    "default": "python",
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        code = kwargs["code"]
        lang = kwargs.get("language", "python")

        if lang != "python":
            return f"Only Python is supported, got '{lang}'"

        # Security: wrap in restricted environment
        wrapper = textwrap.dedent(f"""\
            import sys, os
            # Block dangerous modules
            _blocked = {{'subprocess', 'shutil', 'socket', 'http', 'urllib',
                        'requests', 'httpx', 'ftplib', 'smtplib', 'ctypes'}}
            _orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
            def _safe_import(name, *args, **kwargs):
                if name.split('.')[0] in _blocked:
                    raise ImportError(f"Module '{{name}}' is blocked in sandbox")
                return _orig_import(name, *args, **kwargs)
            try:
                __builtins__.__import__ = _safe_import
            except AttributeError:
                import builtins
                builtins.__import__ = _safe_import
            os.chdir('/tmp')
        """) + "\n" + code

        try:
            result = subprocess.run(
                ["python3", "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="/tmp",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout = result.stdout[:4000] if result.stdout else ""
            stderr = result.stderr[:2000] if result.stderr else ""
        except subprocess.TimeoutExpired:
            return "Execution timed out after 10 seconds."
        except Exception as e:
            return f"Execution error: {e}"

        output = ""
        if stdout:
            output += f"**stdout:**\n```\n{stdout.strip()}\n```\n"
        if stderr:
            output += f"**stderr:**\n```\n{stderr.strip()}\n```\n"
        if not stdout and not stderr:
            output = "Code executed successfully (no output)."

        # Try to auto-visualize numeric output
        lines = stdout.strip().split("\n") if stdout else []
        nums = []
        for line in lines:
            line = line.strip()
            try:
                nums.append(float(line))
            except ValueError:
                # Try comma-separated
                try:
                    nums.extend([float(x.strip()) for x in line.split(",") if x.strip()])
                except ValueError:
                    pass

        if len(nums) >= 3:
            chart_id = f"code_chart_{int(time.time())}"
            labels = json.dumps(list(range(len(nums))))
            values = json.dumps(nums)
            html = _card(
                f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px">Output Visualization</div>'
                f'<canvas id="{chart_id}" height="180"></canvas>'
                f'<script src="{CHARTJS_CDN}"></script>'
                f"<script>"
                f"new Chart(document.getElementById('{chart_id}'),{{"
                f"type:'bar',data:{{labels:{labels},"
                f"datasets:[{{data:{values},backgroundColor:'{ACCENT}88',borderColor:'{ACCENT}',"
                f"borderWidth:1}}]}},options:{{responsive:true,"
                f"plugins:{{legend:{{display:false}}}},"
                f"scales:{{x:{{ticks:{{color:'{MUTED}'}},grid:{{color:'#1a1a2e'}}}},"
                f"y:{{ticks:{{color:'{MUTED}'}},grid:{{color:'#1a1a2e'}}}}}}}}"
                f"}});</script>"
            )
            output += f"\n{_widget(html)}"

        return output


# ===== 5. ComparisonTableTool =============================================

class ComparisonTableTool(Tool):
    """Render structured comparison data as a styled HTML table."""

    @property
    def name(self) -> str:
        return "compare"

    @property
    def description(self) -> str:
        return (
            "Render a styled comparison table from structured data. "
            "Pass items (array of objects) and columns (array of column names)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["items", "columns"],
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Array of objects to compare",
                    "items": {"type": "object"},
                },
                "columns": {
                    "type": "array",
                    "description": "Column names (keys from items)",
                    "items": {"type": "string"},
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        items: list[dict] = kwargs["items"]
        columns: list[str] = kwargs["columns"]

        if not items or not columns:
            return "No data provided for comparison table."

        # Build header
        header_cells = "".join(
            f'<th style="padding:10px 14px;text-align:left;border-bottom:2px solid {ACCENT};'
            f'color:{ACCENT};font-size:12px;text-transform:uppercase;letter-spacing:0.5px">'
            f"{col}</th>"
            for col in columns
        )
        header = f"<tr>{header_cells}</tr>"

        # Build rows
        rows = ""
        for i, item in enumerate(items):
            bg = "#111" if i % 2 == 0 else BG
            cells = ""
            for j, col in enumerate(columns):
                val = item.get(col, "—")
                weight = "font-weight:600" if j == 0 else ""
                cells += (
                    f'<td style="padding:10px 14px;border-bottom:1px solid #1a1a2e;{weight}">'
                    f"{val}</td>"
                )
            rows += f'<tr style="background:{bg}">{cells}</tr>'

        html = _card(
            f'<div style="overflow-x:auto">'
            f'<table style="width:100%;border-collapse:collapse;font-size:13px">'
            f"<thead>{header}</thead><tbody>{rows}</tbody></table></div>"
        )

        return f"Comparison table ({len(items)} items, {len(columns)} columns):\n{_widget(html)}"


# ===== 6. TimelineTool ====================================================

class TimelineTool(Tool):
    """Render events on a visual vertical timeline."""

    @property
    def name(self) -> str:
        return "timeline"

    @property
    def description(self) -> str:
        return (
            "Render events as a visual vertical timeline widget. "
            "Each event has date, title, description, and optional color."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["events"],
            "properties": {
                "events": {
                    "type": "array",
                    "description": "Array of timeline events",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "color": {"type": "string"},
                        },
                        "required": ["date", "title"],
                    },
                },
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        events: list[dict] = kwargs["events"]

        if not events:
            return "No events provided for timeline."

        items_html = ""
        for i, ev in enumerate(events):
            color = ev.get("color", ACCENT if i % 2 == 0 else SECONDARY)
            date = ev.get("date", "")
            title = ev.get("title", "")
            desc = ev.get("description", "")

            items_html += (
                f'<div style="display:flex;gap:16px;margin-bottom:0">'
                # Left: date
                f'<div style="width:80px;text-align:right;font-size:11px;color:{MUTED};'
                f'padding-top:2px;flex-shrink:0">{date}</div>'
                # Center: dot + line
                f'<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0">'
                f'<div style="width:12px;height:12px;border-radius:50%;background:{color};'
                f'border:2px solid {BG};box-shadow:0 0 0 2px {color}40;flex-shrink:0"></div>'
                f'{"" if i == len(events) - 1 else f"<div style=&quot;width:2px;background:#1a1a2e;flex:1;min-height:40px&quot;></div>"}'
                f"</div>"
                # Right: content
                f'<div style="padding-bottom:24px;flex:1">'
                f'<div style="font-weight:600;font-size:14px">{title}</div>'
                f'{f"<div style=&quot;font-size:12px;color:{MUTED};margin-top:4px&quot;>{desc}</div>" if desc else ""}'
                f"</div></div>"
            )

        html = _card(
            f'<div style="font-size:12px;color:{MUTED};margin-bottom:16px">Timeline</div>'
            f"{items_html}"
        )

        return f"Timeline with {len(events)} events:\n{_widget(html)}"


# ===== 7. MapTool =========================================================

class MapTool(Tool):
    """Show a location on OpenStreetMap."""

    @property
    def name(self) -> str:
        return "show_map"

    @property
    def description(self) -> str:
        return "Show a location on an interactive OpenStreetMap embed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {"type": "string", "description": "City or place name"},
                "zoom": {"type": "integer", "description": "Zoom level (1-18)", "default": 13},
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        location = kwargs["location"]
        zoom = kwargs.get("zoom", 13)

        try:
            lat, lon, display = await _geocode(location)
        except Exception as e:
            return f"Could not find location '{location}': {e}"

        # OpenStreetMap embed URL
        bbox_d = 0.05 * (18 / max(zoom, 1))
        osm_url = (
            f"https://www.openstreetmap.org/export/embed.html"
            f"?bbox={lon - bbox_d},{lat - bbox_d},{lon + bbox_d},{lat + bbox_d}"
            f"&layer=mapnik&marker={lat},{lon}"
        )

        html = _card(
            f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px">'
            f'📍 {display} ({lat:.4f}, {lon:.4f})</div>'
            f'<iframe src="{osm_url}" '
            f'style="width:100%;height:300px;border:1px solid #1a1a2e;border-radius:8px" '
            f'loading="lazy"></iframe>'
            f'<div style="font-size:11px;color:{MUTED};margin-top:6px;text-align:right">'
            f'<a href="https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}" '
            f'target="_blank" style="color:{ACCENT};text-decoration:none">Open full map ↗</a></div>'
        )

        return f"Map of {display} (lat {lat:.4f}, lon {lon:.4f}):\n{_widget(html)}"


# ===== 8. SystemMonitorTool ===============================================

class SystemMonitorTool(Tool):
    """Live system stats: CPU, memory, disk, uptime."""

    @property
    def name(self) -> str:
        return "system_monitor"

    @property
    def description(self) -> str:
        return "Show live system stats (CPU, memory, disk, uptime) as a dashboard widget."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        stats: dict[str, Any] = {}

        # Uptime
        try:
            with open("/proc/uptime") as f:
                up_secs = float(f.read().split()[0])
            days = int(up_secs // 86400)
            hours = int((up_secs % 86400) // 3600)
            mins = int((up_secs % 3600) // 60)
            stats["uptime"] = f"{days}d {hours}h {mins}m"
        except Exception:
            stats["uptime"] = "N/A"

        # CPU usage from /proc/stat (two samples 0.5s apart)
        try:
            def read_cpu():
                with open("/proc/stat") as f:
                    parts = f.readline().split()
                # user, nice, system, idle, iowait, irq, softirq
                vals = [int(x) for x in parts[1:8]]
                idle = vals[3] + vals[4]
                total = sum(vals)
                return idle, total

            idle1, total1 = read_cpu()
            import asyncio
            await asyncio.sleep(0.3)
            idle2, total2 = read_cpu()
            cpu_pct = 100 * (1 - (idle2 - idle1) / max(total2 - total1, 1))
            stats["cpu"] = round(cpu_pct, 1)
        except Exception:
            stats["cpu"] = 0

        # CPU count
        try:
            stats["cpu_count"] = os.cpu_count() or 0
        except Exception:
            stats["cpu_count"] = 0

        # Memory from /proc/meminfo
        try:
            meminfo = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            used_kb = total_kb - avail_kb
            stats["mem_total_gb"] = round(total_kb / 1048576, 1)
            stats["mem_used_gb"] = round(used_kb / 1048576, 1)
            stats["mem_pct"] = round(100 * used_kb / max(total_kb, 1), 1)
        except Exception:
            stats["mem_total_gb"] = 0
            stats["mem_used_gb"] = 0
            stats["mem_pct"] = 0

        # Disk usage via os/shutil
        import shutil
        try:
            usage = shutil.disk_usage("/")
            stats["disk_total_gb"] = round(usage.total / (1024**3), 1)
            stats["disk_used_gb"] = round(usage.used / (1024**3), 1)
            stats["disk_pct"] = round(100 * usage.used / max(usage.total, 1), 1)
        except Exception:
            stats["disk_total_gb"] = 0
            stats["disk_used_gb"] = 0
            stats["disk_pct"] = 0

        # Load average
        try:
            load1, load5, load15 = os.getloadavg()
            stats["load"] = f"{load1:.2f} / {load5:.2f} / {load15:.2f}"
        except Exception:
            stats["load"] = "N/A"

        def gauge(label: str, pct: float, detail: str, color: str = ACCENT) -> str:
            bar_color = color if pct < 80 else "#ef4444"
            return (
                f'<div style="margin-bottom:14px">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">'
                f'<span style="color:{MUTED}">{label}</span>'
                f"<span>{detail}</span></div>"
                f'<div style="background:#1a1a2e;border-radius:4px;height:10px;overflow:hidden">'
                f'<div style="width:{min(pct, 100):.1f}%;background:{bar_color};height:100%;'
                f'border-radius:4px;transition:width 0.3s"></div></div></div>'
            )

        html = _card(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">'
            f'<div style="font-size:14px;font-weight:600">System Monitor</div>'
            f'<div style="font-size:11px;color:{MUTED}">⏱ {stats["uptime"]}</div></div>'
            + gauge("CPU", stats["cpu"], f'{stats["cpu"]}% ({stats["cpu_count"]} cores)', ACCENT)
            + gauge("Memory", stats["mem_pct"],
                     f'{stats["mem_used_gb"]}G / {stats["mem_total_gb"]}G', SECONDARY)
            + gauge("Disk", stats["disk_pct"],
                     f'{stats["disk_used_gb"]}G / {stats["disk_total_gb"]}G', ACCENT)
            + f'<div style="font-size:11px;color:{MUTED};margin-top:8px;border-top:1px solid #1a1a2e;'
            f'padding-top:8px">Load avg: {stats["load"]}</div>'
        )

        return (
            f"System: CPU {stats['cpu']}%, "
            f"Mem {stats['mem_used_gb']}G/{stats['mem_total_gb']}G ({stats['mem_pct']}%), "
            f"Disk {stats['disk_used_gb']}G/{stats['disk_total_gb']}G ({stats['disk_pct']}%), "
            f"Uptime {stats['uptime']}\n{_widget(html)}"
        )


# ===== Registry ============================================================

def get_genui_tools(workspace: Any = None) -> list[Tool]:
    """Return all generative UI tool instances."""
    return [
        WeatherTool(),
        StockChartTool(),
        QRCodeTool(),
        CodeRunnerTool(),
        ComparisonTableTool(),
        TimelineTool(),
        MapTool(),
        SystemMonitorTool(),
    ]
