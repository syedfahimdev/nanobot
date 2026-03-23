# Mawa

Personal AI operating system built on **nanobot** — a lightweight agent framework in Python.

Multi-provider LLM support, persistent memory, voice chat, autonomous scheduling, and a React dashboard.

---

## Features

### AI Core

- **Multi-provider routing** — Anthropic, OpenAI, Gemini, Groq, DeepSeek, Kimi, OpenRouter, Ollama, vLLM, and 15+ more via auto-detection or explicit provider selection
- **Profile switching** — named presets override model, temperature, context window, reasoning effort per task
- **4-layer memory** — short-term, long-term, observations, episodes, plus learnings, goals, corrections, and media
- **Reflection engine** — self-correction loop, tool success scoring, pattern observer, prompt self-optimization
- **Skill acquisition** — `learn_skill` and `learn_from_url` tools for runtime knowledge ingestion

### Intelligence Engine (9 loop enhancements)

All toggleable from Settings > Intelligence:

| Feature | What it does |
|---------|-------------|
| Smart Error Recovery | Classifies tool errors (timeout/auth/404/rate-limit) with targeted recovery hints |
| Intent Tracking | Maintains active conversational topic — resolves "this", "that", "send it" correctly |
| Dynamic Context Budget | Sizes tool results based on remaining context window (not fixed 16K cap) |
| Response Quality Gate | Catches LLM deflections ("I can't help") and retries with tools |
| MCP Auto-Reconnect | Detects dead MCP servers and triggers reconnection after 3 failures |
| Tool Call Validation | Pre-checks tool names and argument types before execution |
| Parallel Safety | Detects conflicting writes to same file and serializes them |
| Streaming Recovery | Retries on stream failure, preserves progress on fallback |
| Pending Message Awareness | Queued messages get re-dispatched (not silently swallowed) |

### Pre-LLM Interceptor (zero-token answers)

These queries are answered by pure code — no LLM call at all:

| You ask | Mawa answers instantly |
|---------|----------------------|
| "what's 15% of $347" | **$52.05** |
| "3pm EST in Tokyo" | **4:00 AM JST** |
| "regex for email addresses" | `[a-zA-Z0-9._%+-]+@...` |
| "hello" / "good morning" | Time-aware greeting with goals context |

### Smart Response Features (15 code-level)

| Feature | How it works |
|---------|-------------|
| Response Caching | Hash-based cache — identical questions skip LLM (5min TTL) |
| Priority Detection | "URGENT", "!!!", ALL CAPS → flagged as high priority |
| Entity Extraction | Regex NER pulls emails, phones, dates, money, URLs pre-LLM |
| Loop Detector | Breaks repetitive response cycles with a different-approach nudge |
| Smart Defaults | Auto-fills recipients, dates, timezone from memory |
| Progressive Disclosure | Detects yes/no questions → prompts brief answers |
| Error Translation | "ConnectionRefusedError" → "The service isn't running" |
| Link Enrichment | URLs get domain + type hints (GitHub repo, SO question, etc.) |
| Time-Aware Greetings | "Good morning Farhan! You have 3 pending goals" |
| Frustration Detection | Detects "!!!", caps, "doesn't work" → empathetic response |
| Message Dedup | Same message within 30s = silently dropped (network glitch protection) |
| Tool Result Merging | Deduplicates overlapping results across multiple tools |
| Semantic Truncation | Cuts at paragraph/sentence boundaries, not mid-word |
| Auto-Retry Context | On failure, injects "Previous attempts failed because..." |
| Session Metrics | Per-session: turns, tokens, errors, avg response time |

### Code-Level Features (15 pure logic, zero LLM)

| Feature | API |
|---------|-----|
| Hard Budget Enforcement | Blocks execution when daily/weekly limit exceeded |
| Predictive Suggestions | "You usually check email at 9am" from usage patterns |
| File Auto-Cleanup | Delete files >30 days, enforce disk quota |
| Session Search | `GET /api/sessions/search?q=keyword` — grep across all sessions |
| File Watcher | Poll-based filesystem monitoring → triggers agent on changes |
| Cron Dashboard | `GET /api/cron/dashboard` — job status, missed runs, overdue |
| Smart Retry Queue | Disk-backed outbound queue for offline resilience |
| Health Dashboard | `GET /api/health/dashboard` — disk, sessions, memory, cron, tools, budget |
| Auto-Model Downgrade | Switches to cheaper model per-turn when budget >80% |
| Event→Action Rules | `GET/POST /api/rules` — "when email from boss → notify immediately" |
| Session Tags | `GET/POST /api/sessions/tags` — organize by work/personal/project |
| Tool Favorites | `GET /api/tools/favorites` — most-used tools by frequency |
| Anomaly Detection | `GET /api/anomalies` — alerts when usage is 3x+ the 7-day average |
| Batch File Processing | `GET /api/inbox/batch?action=count` — bulk inbox operations |
| Schedule Templates | `GET /api/schedule/templates` — 7 pre-built cron patterns |

### Claude-Level Capabilities (11 ported features)

| Feature | How it works |
|---------|-------------|
| Task Decomposer | Breaks "check email and then reply" into ordered steps |
| Parallel Dispatch | Classifies tools as parallel-safe vs sequential |
| Source Citation | Tags facts with web/email/memory sources |
| Smart Formatter | Auto-detects data → markdown table, list, code block |
| State Snapshots | Take before/after snapshots, compare file changes |
| Calculator | Safe AST eval — math expressions without LLM |
| Timezone Resolver | "3pm EST in Tokyo" answered instantly |
| Paste Pipeline | Detects URL/JSON/email/code/CSV pastes, suggests processing |
| Strategy Rotator | On tool failure, suggests alternative tool |
| Research Pipeline | Builds search→fetch→extract multi-hop plan |
| Regex Builder | Common patterns from templates (email, phone, URL) |

### Dashboard (mawabot)

- Token usage tracking with per-model cost estimates
- 7-day usage bar chart, daily prompt/completion breakdown
- Goals preview with tap-to-complete
- Memory health indicators (layer status, sizes)
- Activity feed (recent turns, tools used, durations)
- Editable quick actions grid

### Chat

- **Unified chat + call** — text input always visible, call toggles inline
- Type while on a call — voice panel shows above text input
- Streaming responses over WebSocket with animated typing indicator
- Generative UI — inline HTML widgets rendered in chat
- Tool result cards (email, calendar, weather, news)
- Markdown rendering with GFM support (code blocks, tables, links)
- File attachments with type-specific icons, preview, and download
- Quick reply suggestions — context + pattern aware
- Message search — search across all conversation history and memory
- WhatsApp-style emoji reactions that drive self-improvement
- Notification bell with unread badge + clickable notification tray
- Pinned messages, export (markdown/JSON), keyboard shortcuts (Cmd+K)

### Voice

- Browser-based voice chat via Deepgram STT + TTS
- Streaming sentence-by-sentence TTS playback
- Smart voice endpointing (semantic sentence completion, not just silence timer)
- Discord voice channel support (opus codec)
- 180s stream timeout with partial content recovery
- 45s per-tool timeout via asyncio.wait_for

### Tools

- **MCP servers** — Composio, Playwright, Browser-Use, or any stdio/SSE/streamableHttp server
- **Native browser** — Playwright with Xvfb headed mode, persistent profile, noVNC live view
- **Background shell** — run/status/output/kill long-running commands (detached)
- Email, calendar, weather, web search (Brave/Tavily/DuckDuckGo/Jina/SearXNG)
- Cron jobs, shell exec, goal tracking, inbox with hybrid indexing
- Skills marketplace — search/install from skills.sh

### Memory

- **6-trigger consolidation** — every 20 msgs, heartbeat (30min), disconnect, /new, Clear Today, manual button
- **Separate learnings** — user corrections in LEARNINGS.md, tool errors in TOOL_LEARNINGS.md
- **Conversation recap** — extractive ~40 token summary injected per turn
- **Pronoun resolution** — system prompt rule + intent tracking
- **Dynamic capabilities** — auto-generated manifest when user asks "what can you do?"
- **/learnings command** — see what Mawa learned from your feedback
- **Memory Activity Timeline** — color-coded events (learnings, episodes, feedback)

### Channels

Web dashboard, Telegram, Discord (text + voice), Slack, WeChat, DingTalk, Feishu, Matrix, QQ, Email

### Autonomy

- Morning / evening / weekly briefings (proactive, context-aware)
- Scheduled tasks via cron with 7 pre-built templates
- Webhook event ingestion (POST /api/events, HMAC-secured)
- Workflow recording (detects repeated tool sequences, suggests automation)
- Prompt self-optimization (tracks routing accuracy, suggests fixes every 50 turns)
- Emoji reaction feedback → self-improvement
- Proactive notifications — persistent, delivered on reconnect if offline
- Background job completion notifications

### Security

- Encrypted vault (Fernet) for API keys and secrets
- **Smart credential naming** — `github_token`, `outlook_password` (not `auto_1`)
- Auto-masking of passwords before LLM sees them (3-layer detection)
- Outbound security filter — strips leaked credentials from all responses
- Credential manager with {cred:name} vault injection for browser autofill
- Tailscale-only access enforcement for web dashboard

### Settings

| Tab | What's in it |
|-----|-------------|
| Profile | AI model profiles, switch with one tap |
| Features | 8 chat feature toggles with descriptions |
| Memory | 9 layers with search, export, clear, consolidate, timeline, learnings panel |
| Security | Credential manager, vault stats, security checklist |
| Autonomy | 5 self-evolution toggles with detailed explanations |
| Intelligence | 5 loop enhancement toggles + 4 always-on safety cards |
| Avatar | 4 avatar styles (3D face, emoji, simple, orb) |
| Voice | TTS/STT settings, 6 Deepgram voice models |
| Look | 6 app themes (midnight, whatsapp, telegram, discord, iphone, terminal) |
| MCP | MCP server management, add/remove |
| Status | Service health, recent activity |
| About | System info, tech stack |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node 20+ (for the dashboard)

### Install

```bash
# Backend
git clone <repo-url> nanobot && cd nanobot
pip install -e .

# Frontend
cd ../mawabot
npm install
```

### Configure

```bash
nanobot onboard
```

Or edit `~/.nanobot/config.json` directly. At minimum, set one provider API key.

### Run

```bash
# CLI mode
nanobot

# Web dashboard + voice + API
nanobot web
```

---

## API Endpoints

All served by the `web_voice` channel.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ws` | WebSocket — chat, streaming, voice |
| GET | `/api/config` | Read current config |
| POST | `/api/config` | Update config |
| GET | `/api/memory` | Memory layer status |
| POST | `/api/memory/consolidate` | Manual LLM consolidation |
| GET | `/api/memory/timeline` | Memory activity events |
| GET | `/api/learnings` | What Mawa learned from you |
| GET | `/api/notifications` | Stored notifications |
| POST | `/api/notifications/read` | Mark all as read |
| GET | `/api/intelligence` | Intelligence toggle states |
| POST | `/api/intelligence` | Update intelligence toggles |
| GET | `/api/health/dashboard` | Comprehensive health check |
| GET | `/api/suggestions/predictive` | Pattern-based suggestions |
| GET | `/api/sessions/search?q=X` | Cross-session keyword search |
| GET | `/api/sessions/health` | Per-session health metrics |
| GET | `/api/cron/dashboard` | Cron job status + stats |
| GET | `/api/anomalies` | Usage anomaly alerts |
| GET | `/api/tools/favorites` | Most-used tools ranked |
| GET | `/api/schedule/templates` | Pre-built cron patterns |
| GET/POST | `/api/rules` | Event→action automation rules |
| GET/POST | `/api/sessions/tags` | Session tag management |
| GET | `/api/inbox/batch?action=X` | Bulk inbox operations |
| POST | `/api/snapshot` | Take state snapshot |
| GET | `/api/snapshot/diff?name=X` | Compare against snapshot |
| GET | `/api/goals` | List goals |
| GET | `/api/usage` | Token usage summary |
| GET | `/api/credentials` | List stored credentials |
| GET | `/api/mcp-servers` | List MCP servers |
| POST | `/api/events` | Ingest webhook events |

---

## License

MIT
