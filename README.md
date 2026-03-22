# Mawa

Personal AI assistant built on **nanobot** — a lightweight agent framework in Python.

Multi-provider LLM support, persistent memory, voice chat, autonomous scheduling, and a React dashboard.

---

## Features

### AI Core

- **Multi-provider routing** — Anthropic, OpenAI, Gemini, Groq, DeepSeek, Kimi, OpenRouter, Ollama, vLLM, and 15+ more via auto-detection or explicit provider selection
- **Profile switching** — named presets override model, temperature, context window, reasoning effort per task
- **4-layer memory** — short-term, long-term, observations, episodes, plus learnings, goals, corrections, and media
- **Reflection engine** — self-correction loop, tool success scoring, pattern observer, prompt self-optimization
- **Skill acquisition** — `learn_skill` and `learn_from_url` tools for runtime knowledge ingestion

### Dashboard (mawabot)

- Token usage tracking with per-model cost estimates
- 7-day usage bar chart, daily prompt/completion breakdown
- Goals preview with tap-to-complete
- Memory health indicators (layer status, sizes)
- Activity feed (recent turns, tools used, durations)
- Editable quick actions grid

### Chat

- Streaming responses over WebSocket with animated typing indicator
- Generative UI — inline HTML widgets rendered in chat
- Tool result cards (email, calendar, weather, news)
- Markdown rendering with GFM support (code blocks, tables, links)
- File attachments with type-specific icons, preview, and download
- Quick reply suggestions — context-aware follow-up buttons (time-based)
- Message search — search across all conversation history and memory
- Emoji reactions (👍👎❤️💡) that drive self-improvement via LEARNINGS.md
- Pinned messages, export (markdown/JSON), keyboard shortcuts (Cmd+K)
- New Chat button in header — clears frontend + backend session with consolidation
- Auto-scroll on view switch — chat always shows latest messages

### Voice

- Browser-based voice chat via Deepgram STT + TTS
- Streaming sentence-by-sentence TTS playback
- Discord voice channel support (opus codec)

### Tools

- **MCP servers** — Composio, Playwright, Browser-Use, or any stdio/SSE/streamableHttp server
- **Native browser** — Playwright with Xvfb headed mode, persistent profile, noVNC live view
- Email, calendar, weather, web search (Brave/Tavily/DuckDuckGo/Jina/SearXNG)
- Cron jobs, shell exec, goal tracking, inbox with hybrid indexing

### Channels

Web dashboard, Telegram, Discord (text + voice), Slack, WeChat, DingTalk, Feishu, Matrix, QQ, Email

### Autonomy

- Morning / evening / weekly briefings (proactive, context-aware)
- Scheduled tasks via cron
- Webhook event ingestion (POST /api/events, HMAC-secured)
- Workflow recording (detects repeated tool sequences, suggests automation)
- Prompt self-optimization (tracks routing accuracy, suggests fixes every 50 turns)
- Emoji reaction feedback → self-improvement (👍 reinforces, 👎 extracts correction)
- Session consolidation saves behavioral insights to LEARNINGS.md on /new
- Goal auto-completion for goals that map to available tools

### Security

- Encrypted vault (Fernet) for API keys and secrets
- Auto-masking of passwords before LLM sees them
- Credential manager with {cred:name} vault injection for browser autofill
- Tailscale-only access enforcement for web dashboard

### Mobile

- PWA with bottom tab bar + collapsible sidebar (hamburger menu)
- Keyboard viewport fix for iOS/Android (visualViewport API)
- Push notifications
- Smart voice endpointing (semantic sentence completion, not just silence timer)

### Settings (10 tabs)

| Tab | What's in it |
|-----|-------------|
| Profile | AI model profiles, switch with one tap |
| Memory | 9 layers with search, export, clear today |
| Security | Credential manager, vault stats, security checklist |
| Autonomy | 5 self-evolution toggles + 8 feature toggles with descriptions |
| Avatar | 4 avatar styles (3D face, emoji, simple, orb) |
| Voice | TTS/STT settings, 6 Deepgram voice models |
| Look | Theme (dark/light), accent color (6 options), notifications |
| Status | Service health, ToolsDNS details, recent activity |
| About | System info, tech stack, debug export |

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

The dashboard is served at `https://<hostname>:3000` (or whatever port the frontend dev server uses). The backend API runs on port `18790` by default.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  mawabot (React SPA)                        │
│  Vite + React 18 + Tailwind + Zustand       │
│  Views: Dashboard, Chat, Goals, Schedule,   │
│         Tools, Activity, Inbox, Boardroom,  │
│         Settings (10 tabs)                  │
└──────────────────┬──────────────────────────┘
                   │ WebSocket (/ws) + REST (/api/*)
┌──────────────────▼──────────────────────────┐
│  nanobot (Python)                           │
│                                             │
│  Agent Loop ─► Providers (litellm)          │
│       │            └─ auto-detect or force  │
│       ├─► Memory (4-layer + reflection)     │
│       ├─► Tools (native + MCP servers)      │
│       ├─► Hooks (usage tracker, cron, …)    │
│       └─► Channels                          │
│            ├─ web_voice (aiohttp + WS)      │
│            ├─ telegram, discord, slack, …    │
│            └─ cli (prompt-toolkit)           │
└─────────────────────────────────────────────┘
```

---

## Configuration

Config lives at `~/.nanobot/config.json`. Key sections:

| Section | Purpose |
|---------|---------|
| `agents.defaults` | Default model, provider, max tokens, context window, temperature, reasoning effort |
| `providers` | API keys and base URLs per provider (anthropic, openai, groq, kimi, etc.) |
| `profiles` | Named presets that override `agents.defaults` when activated |
| `channels` | Per-channel config (tokens, settings); extras accepted freely |
| `tools.mcpServers` | MCP server definitions (stdio command+args or HTTP URL) |
| `tools.web.search` | Web search provider and API key |
| `tools.exec` | Shell exec timeout and PATH additions |
| `gateway` | Host, port, heartbeat interval |

Example profile:

```json
{
  "profiles": {
    "fast": {
      "provider": "groq",
      "model": "groq/llama-3.3-70b-versatile",
      "maxTokens": 4096,
      "contextWindowTokens": 32768
    },
    "deep": {
      "provider": "anthropic",
      "model": "anthropic/claude-opus-4-5",
      "reasoningEffort": "high"
    }
  }
}
```

---

## API Endpoints

All served by the `web_voice` channel on the backend port (default `18790`).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ws` | WebSocket — chat, streaming, voice |
| GET | `/health` | Health check |
| GET | `/api/config` | Read current config |
| POST | `/api/config` | Update config |
| GET | `/api/profiles` | List available profiles |
| GET | `/api/memory` | Memory layer status and sizes |
| POST | `/api/memory/search` | Search across memory layers |
| GET | `/api/memory/export` | Export memory as archive |
| POST | `/api/memory/clear-short-term` | Clear short-term memory |
| GET | `/api/goals` | List goals |
| POST | `/api/goals` | Add/update goals |
| GET | `/api/activity` | Recent turn history |
| GET | `/api/cron` | List scheduled cron jobs |
| GET | `/api/tools` | List registered tools |
| POST | `/api/events` | Ingest webhook events |
| GET | `/api/inbox` | List inbox items |
| POST | `/api/inbox/upload` | Upload files to inbox |
| GET | `/api/files/{path}` | Download workspace files |
| GET | `/api/generated` | List generated files |
| POST | `/api/generated/cleanup` | Clean up old generated files |
| GET | `/api/autonomy` | Autonomy settings status |
| GET | `/api/credentials` | List stored credentials |
| POST | `/api/credentials` | Add/update credentials |
| GET | `/api/mcp-servers` | List MCP server configs |
| POST | `/api/mcp-servers` | Add/update MCP server |
| DELETE | `/api/mcp-servers/{name}` | Remove MCP server |
| GET | `/api/usage` | Token usage summary (today + 7-day) |

---

## Profiles

Switch the active model preset at runtime via the Settings page or the API.

Each profile can override: `provider`, `model`, `maxTokens`, `contextWindowTokens`, `temperature`, `reasoningEffort`, `maxToolIterations`, `routingModel`.

The dashboard Settings > Profile tab shows all configured profiles and lets you switch with one tap.

---

## License

MIT
