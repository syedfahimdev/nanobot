# Mawa — Personal AI Assistant

Custom features built on top of [nanobot](https://github.com/HKUDS/nanobot) for Farhan's personal Jarvis AI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CHANNELS                                       │
│                                                                         │
│  ┌──────────┐   ┌───────────────┐   ┌───────────────┐   ┌──────────┐  │
│  │ Discord  │   │ Discord Voice │   │   Web Voice   │   │ Telegram │  │
│  │  (text)  │   │  (STT/TTS)   │   │   (PWA/TTS)   │   │  (text)  │  │
│  └────┬─────┘   └──────┬────────┘   └──────┬────────┘   └────┬─────┘  │
│       │                │                    │                  │        │
│       └────────────────┴────────┬───────────┴──────────────────┘        │
│                                 │                                       │
│                          Message Bus                                    │
│                     (async pub/sub queue)                                │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENT LOOP                                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Message Dispatch                            │    │
│  │                                                                  │    │
│  │  Inbound ──► Approval Check ──► Lock Check ──► Route:            │    │
│  │              (yes/no intercept)                                   │    │
│  │                                    │                             │    │
│  │              ┌─────────────────────┼──────────────────┐          │    │
│  │              ▼                     ▼                  ▼          │    │
│  │        Conversational      Context-Dependent       Task          │    │
│  │        ("ok","thanks")     ("did it all come?")   ("check email")│    │
│  │              │                     │                  │          │    │
│  │           Queue              Queue for             Auto-spawn   │    │
│  │         (append to          main agent             subagent     │    │
│  │          session)           (has pronouns/          (parallel)   │    │
│  │                              references)                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Context Builder                              │    │
│  │                                                                  │    │
│  │  System Prompt = Identity + SOUL.md + USER.md + AGENTS.md        │    │
│  │                + TOOLS.md + MEMORY.md + Active Skills             │    │
│  │                                                                  │    │
│  │  User Message = Runtime Context + Voice Hint + User Text         │    │
│  │               + ToolsDNS Preflight + Active Subagents            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                  LLM Agent Loop                                  │    │
│  │                                                                  │    │
│  │  Messages ──► LLM Call ──► Tool Calls? ──► Execute ──► Repeat    │    │
│  │                  │              │              │                  │    │
│  │            (routing model   (parallel      (via ToolRegistry     │    │
│  │             on iteration 1)  execution)     + Hook Engine)       │    │
│  │                                                                  │    │
│  │  Voice: streams tokens ──► sentence split ──► TTS per sentence   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Subagent Manager                             │    │
│  │                                                                  │    │
│  │  Spawned for parallel tasks when main agent is busy              │    │
│  │  Gets: condensed USER.md + TOOLS.md + last 8 messages            │    │
│  │  Has: own ToolsDNS preflight + tool memory hints                 │    │
│  │  Max: 15 iterations, limited tools (no spawn/message)            │    │
│  │  Results: simple → direct send, complex → main agent summarizes  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         HOOK ENGINE                                     │
│                    (deterministic code, no LLM)                         │
│                                                                         │
│  Event: tool_before ──────► Approval Guard (BLOCKING)                   │
│                              │                                          │
│                              ├─ Matches GMAIL_SEND_*, *_DELETE_*, etc.  │
│                              ├─ Voice: "Should I send email? Yes or no" │
│                              ├─ Waits 60s for response                  │
│                              └─ Denied? Tool returns error, not called  │
│                                                                         │
│  Event: tool_after ───────► Auto-Logger (fire & forget)                 │
│                              └─ Appends to HISTORY.md:                  │
│                                 [2026-03-20 02:39] TOOL Gmail OK (1.2s) │
│                                                                         │
│  Event: tool_after ───────► Lifecycle Tracker (fire & forget)           │
│                              └─ Tracks tool calls per turn              │
│                                                                         │
│  Event: turn_completed ──► Lifecycle Tracker                            │
│                              └─ Writes activity.jsonl with timing       │
│                                                                         │
│  Command: /doctor ────────► Health Checks                               │
│                              ├─ Disk space                              │
│                              ├─ Session integrity                       │
│                              ├─ Memory file sizes                       │
│                              ├─ ToolsDNS connectivity                   │
│                              └─ LLM provider status                     │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOLDNS                                         │
│                   (separate service, port 8787)                         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  /v1/preflight│  │  /v1/search  │  │  /v1/call    │                  │
│  │  (context     │  │  (semantic   │  │  (execute    │                  │
│  │   block for   │  │   + BM25     │  │   tool via   │                  │
│  │   LLM inject) │  │   search)    │  │   MCP/stdio) │                  │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘                  │
│         │                                    │                          │
│         ▼                                    ▼                          │
│  Two-Tier Search                    Tool Execution                      │
│  ├─ Tier 1: Standard preflight      ├─ Composio tools (HTTP)            │
│  │  (semantic + intent matching)     ├─ MCP tools (stdio/HTTP)          │
│  ├─ Tier 2: Triggered when          ├─ Skill tools (python scripts)    │
│  │  meta-tools or wrong app found   └─ Macros (multi-step chains)      │
│  │  ├─ Strategy A: exact tool ID                                       │
│  │  └─ Strategy B: app-scoped search                                   │
│  └─ Tool Memory hints appended                                         │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ /v1/tool-hints│  │ /v1/skills   │  │  Embeddings  │                  │
│  │ (successful   │  │ (skill       │  │  bge-base    │                  │
│  │  arg patterns │  │  registry)   │  │  en-v1.5     │                  │
│  │  per tool)    │  │              │  │  (768d ONNX) │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                         │
│  2600+ tools │ 44 apps │ Composio + MCP + Skills                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Feature List

### Voice-First Experience
- **Web Voice PWA** — Push-to-talk + continuous listening via Deepgram STT
- **Streaming TTS** — Sentences spoken as they arrive, not after full response
- **Deepgram endpointing** — 800ms endpoint, 2s utterance end for natural pauses
- **Cross-device sessions** — Fixed client ID (`voice`) shared across all devices
- **Device takeover** — Opening voice on a new device cancels the old session
- **Mobile TTS fix** — AudioContext resume on visibility change for iOS/Android

### Smart Agent Routing
- **Context-dependent detection** — Messages with pronouns/references ("did it all come?", "send that to John") queued for main agent instead of spawning contextless subagents
- **Conversational detection** — "ok", "thanks", "yes" queued silently
- **Skill routing** — Dynamic skill pattern detection from ToolsDNS API (refreshes every 5 min)
- **Auto-spawn** — Independent tasks spawn as subagents for parallel execution

### Subagent Intelligence
- **User context injection** — Subagents get condensed USER.md (name, timezone, preferences) + TOOLS.md rules
- **Conversation context** — Last 8 messages injected for reference resolution
- **Simple result shortcut** — Short results (<200 chars) skip LLM summarization

### ToolsDNS Integration
- **Two-tier preflight** — Tier 1 semantic search, Tier 2 exact fetch when meta-tools detected
- **Meta-tool filtering** — COMPOSIO_SEARCH_TOOLS and friends filtered from results
- **App-aware search** — Intent map detects app prefix, triggers targeted search
- **Tool memory** — Successful call arguments stored and injected as `[TOOL MEMORY]` hints
- **Agent preference boosts** — ToolsDNS learns which tools you use and boosts them in search
- **Skill execution** — Skill tools with commands (python scripts) executed via stdio, not just returning SKILL.md

### Hook System (Code-Level, No LLM)
- **Approval guard** — Dangerous tools (GMAIL_SEND_*, *_DELETE_*) require voice/text confirmation before execution. LLM cannot bypass.
- **Auto-logging** — Every external tool call logged to HISTORY.md with timing
- **Lifecycle tracking** — Turn states (running/done) with timing written to activity.jsonl
- **Health checks** — `/doctor` command checks disk, sessions, memory, ToolsDNS, LLM provider

### Memory System
- **MEMORY.md** — Long-term facts (LLM-consolidated)
- **HISTORY.md** — Time-indexed log with structured tool call entries
- **Token-based consolidation** — Auto-archives when session exceeds context window / 2
- **Session persistence** — JSONL files per session, survives restarts

### Commands
| Command | Description |
|---------|-------------|
| `/new` | Start fresh session (archives old to memory) |
| `/stop` | Cancel all running tasks |
| `/restart` | Restart the bot |
| `/profile [name]` | Switch LLM provider profile |
| `/doctor` | Run health checks |
| `/help` | Show commands |

---

## File Structure (Custom Code)

```
nanobot/
├── agent/
│   ├── loop.py          # Main agent loop — dispatch, context, LLM iteration
│   ├── subagent.py      # Subagent manager — spawn, execute, announce
│   ├── context.py       # Context builder — system prompt, user content
│   ├── memory.py        # Memory store + consolidator
│   └── tools/
│       ├── registry.py  # Tool registry with hook integration
│       └── toolsdns.py  # ToolsDNS tool (search, call, macros, workflows)
├── hooks/
│   ├── engine.py        # HookEngine — on(), emit(), blocking/fire-and-forget
│   ├── events.py        # Event dataclasses (ToolBefore, ToolAfter, TurnCompleted)
│   └── builtin/
│       ├── approval.py  # Approval guard — voice-first confirmation
│       ├── auto_log.py  # Auto-log tool calls to HISTORY.md
│       ├── health.py    # /doctor health checks
│       └── lifecycle.py # Turn state tracking → activity.jsonl
├── channels/
│   ├── web_voice.py     # Web Voice channel (Deepgram STT + TTS)
│   └── web_voice_ui/
│       └── index.html   # Voice PWA frontend
├── session/
│   └── manager.py       # Session persistence (JSONL)
└── heartbeat/
    └── service.py       # Periodic wake-up (HEARTBEAT.md)

ToolsDNS/
├── tooldns/
│   ├── api.py           # REST API — preflight, search, call, tool-hints
│   ├── caller.py        # Tool execution — MCP proxy, skill execution
│   ├── database.py      # SQLite — tools, preferences, tool_call_args
│   ├── models.py        # Pydantic models
│   └── ingestion.py     # Tool ingestion from Composio, MCP, skills
```

---

## Configuration

All config in `/root/.nanobot/config.json`. Key sections:

- `tools.toolsdns.url` / `api_key` — ToolsDNS connection
- `channels.web_voice` — Deepgram API key, port, host
- `channels.discord` — Bot token
- `agent.model` — LLM model (via LiteLLM)
- `agent.routing_model` — Fast model for iteration 1

## Token Budget

System prompt optimized for minimal token usage (~2.2K voice, ~2.0K text):

| Component | Tokens | Notes |
|-----------|--------|-------|
| Identity | 289 | Core nanobot boilerplate |
| Voice Mode | 189 | Voice channels only |
| SOUL.md | ~200 | Personality + delegation rules (merged with AGENTS.md) |
| AGENTS.md | ~100 | Cron/heartbeat only (delegation merged into SOUL.md) |
| USER.md | ~250 | Core profile only (car/stock rules in separate files) |
| TOOLS.md | ~80 | Minimal — most rules enforced by code |
| Memory | ~25 | MEMORY.md content |
| Skills | ~1000 | Active skills + summary |

Task-specific rules (car buying, stock market) are in `rules/*.md` — only loaded when the agent reads them for a relevant task, not injected every turn.

## Workspace Files

Located at `/root/.nanobot/workspace/`:

| File | Purpose |
|------|---------|
| `SOUL.md` | Personality, delegation model, core rules |
| `USER.md` | Core user profile (name, timezone, email rules) |
| `AGENTS.md` | Cron and heartbeat instructions |
| `TOOLS.md` | Minimal tool routing rules |
| `HEARTBEAT.md` | Periodic tasks |
| `rules/car-buying.md` | Car research rules (loaded on demand) |
| `rules/stock-market.md` | Stock analysis rules (loaded on demand) |
| `memory/MEMORY.md` | Long-term facts |
| `memory/HISTORY.md` | Time-indexed activity log |
| `activity.jsonl` | Turn lifecycle data |
| `sessions/*.jsonl` | Conversation history per session |
