# Mawa Testing Guide

Step-by-step tests for all custom features. Run these on your voice PWA or Discord.

---

## Quick Health Check

Run this first to make sure everything is up:

```bash
echo "=== Services ==="
systemctl is-active nanobot && echo "nanobot: UP" || echo "nanobot: DOWN"
systemctl is-active tooldns && echo "tooldns: UP" || echo "tooldns: DOWN"

echo ""
echo "=== ToolsDNS ==="
curl -s http://127.0.0.1:8787/v1/health -H "Authorization: Bearer $(python3 -c "import json; print(json.load(open('/root/.nanobot/config.json'))['tools']['toolsdns']['apiKey'])")" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"total_tools\",0)} tools indexed')"

echo ""
echo "=== Memory Index ==="
python3 -c "import sqlite3; conn=sqlite3.connect('/root/.tooldns/tooldns.db'); print(conn.execute(\"SELECT COUNT(*) FROM tools WHERE id LIKE 'memory__%'\").fetchone()[0], 'memory chunks indexed')"

echo ""
echo "=== Workspace Backup ==="
cd /root/.nanobot/workspace && git log --oneline | head -3

echo ""
echo "=== Recent Auto-Log ==="
tail -3 /root/.nanobot/workspace/memory/HISTORY.md
```

---

## Test 1: Voice Mode (System Prompt)

**What changed:** Voice instructions moved from per-message (718 chars wasted each turn) to system prompt (once).

**How to test:**
1. Open voice PWA
2. Say: "Hey, how's it going?"
3. Mawa should respond conversationally — plain text, no markdown, no bullets, no emojis

**What to check:**
- Response sounds natural when spoken
- No "Sure, I'd be happy to help!" filler
- If you mumble something unclear, Mawa says "Sorry, I didn't catch that" instead of guessing

---

## Test 2: Context-Dependent Detection

**What changed:** Messages with pronouns/references queue for main agent instead of spawning contextless subagents.

**How to test:**
1. Say: "Check my email" (wait for it to start processing)
2. While it's busy, say: "Did it all come?"
3. The second message should queue, NOT spawn a subagent

**Monitor logs:**
```bash
journalctl -u nanobot -f | grep -E "context-dependent|queuing|auto-spawning"
```

**Expected log:**
```
Session web_voice:voice busy, queuing context-dependent: 'Did it all come?'
```

**More test phrases that should queue (not spawn):**
- "send that to John"
- "also remind him about Friday"
- "was it successful?"
- "what about those?"

**Phrases that SHOULD spawn (independent tasks):**
- "check my calendar for tomorrow"
- "search for AI news on reddit"
- "find weather in new york"

---

## Test 3: Subagent Identity

**What changed:** Subagents now get your name, timezone, email rules, and tool usage rules.

**How to test:**
1. Say two things quickly: "Check my email and also check Reddit news"
2. One task runs on main agent, the other spawns as subagent
3. The subagent should check **Inbox only, today+yesterday only** (from USER.md email rules)

**Monitor logs:**
```bash
journalctl -u nanobot -f | grep -E "Subagent|GMAIL"
```

**Expected:** Subagent calls `GMAIL_FETCH_EMAILS` with `query: "in:inbox"` instead of empty args.

---

## Test 4: Tool Memory + Arg Sanitizer

**What changed:** Successful tool call args are remembered. Next time, auto-filled if LLM passes bad args.

**How to test:**
1. Say: "Check my email"
2. After it succeeds, check what was stored:

```bash
curl -s http://127.0.0.1:8787/v1/tool-hints \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(python3 -c "import json; print(json.load(open('/root/.nanobot/config.json'))['tools']['toolsdns']['apiKey'])")" \
  -d '{"agent_id": "mawa", "tool_ids": ["tooldns__GMAIL_FETCH_EMAILS"]}' | python3 -m json.tool
```

3. Say "Check my email" again — should auto-fill from memory and return proper inbox results

**What the sanitizer fixes (automatic, no action needed):**
- LLM passes `<query>` placeholder → stripped
- LLM passes `workflow_id` junk → stripped
- LLM passes empty `{}` → auto-filled from last successful call

---

## Test 5: Approval Guard

**What changed:** Dangerous tool calls require voice/text confirmation. Code-level — LLM cannot bypass.

**How to test:**
1. Say: "Send an email to syedfahimdev@gmail.com saying hello"
2. Before sending, Mawa should ask: "Should I send email to syedfahimdev@gmail.com? Say yes or no."
3. Say "no" → should cancel
4. Try again, say "yes" → should proceed

**Tools that require approval:**
- `GMAIL_SEND_*` (send/reply/create email)
- `*_DELETE_*` (delete anything)
- `*_REMOVE_*` (remove anything)
- `DISCORDBOT_CREATE_MESSAGE` (post to Discord channel)

**Monitor logs:**
```bash
journalctl -u nanobot -f | grep -E "Approval|approved|denied"
```

**Timeout test:** Say nothing for 60 seconds after the approval question → should auto-deny.

---

## Test 6: Auto-Logging

**What changed:** Every external tool call is logged to HISTORY.md with timing.

**How to test:**
1. Say anything that triggers a tool call (email, calendar, weather, etc.)
2. Check HISTORY.md:

```bash
tail -10 /root/.nanobot/workspace/memory/HISTORY.md
```

**Expected format:**
```
[2026-03-20 04:35] TOOL GMAIL_FETCH_EMAILS OK (1230ms)
[2026-03-20 04:35] TOOL GOOGLECALENDAR_FIND_EVENT OK (890ms)
```

**What's NOT logged (internal tools):** read_file, write_file, edit_file, list_dir, exec

---

## Test 7: /doctor Health Check

**How to test:**
1. Type `/doctor` in Discord or say "slash doctor" on voice
2. Should return:

```
Health Check Results:

  [OK] Disk Space: 45.2GB free / 50.0GB total (10% used)
  [OK] Sessions: 12 sessions, 1.2MB total
  [OK] Memory: HISTORY.md 190KB, MEMORY.md 0KB
  [OK] ToolsDNS: Online — 2618 tools indexed
  [OK] LLM Provider: Responding (auto-fastest)

All systems operational.
```

**What it checks:**
- Disk space (warns if < 1GB free)
- Session file integrity (warns if > 500MB)
- Memory file sizes (warns if HISTORY.md > 500KB)
- ToolsDNS connectivity and tool count
- LLM provider (sends tiny test completion)

---

## Test 8: Memory Search

**What changed:** 304 knowledge chunks indexed in ToolsDNS for semantic search.

**How to test:**
1. Say: "What do you know about Tonni?"
2. Mawa should use `memory_search` tool and return info from `knowledge/people/tonni.md`

**More test queries:**
- "What's my skincare routine?" → finds `knowledge/skincare/farhan-skin-profile.md`
- "How do I write work emails?" → finds `knowledge/people/farhan-email-style.md`
- "What are my goals?" → finds `knowledge/people/farhan-goals.md`
- "Tell me about my car buying rules" → finds `rules/car-buying.md`

**Manual search test:**
```bash
curl -s http://127.0.0.1:8787/v1/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(python3 -c "import json; print(json.load(open('/root/.nanobot/config.json'))['tools']['toolsdns']['apiKey'])")" \
  -d '{"query": "wedding planning Tonni", "top_k": 3, "threshold": 0.0, "id_prefix": "memory__"}' \
  | python3 -c "import sys,json; [print(f'{r[\"confidence\"]:.0%} {r[\"name\"]}') for r in json.load(sys.stdin).get('results',[])]"
```

---

## Test 9: Memory Save

**What changed:** New `memory_save` tool lets the agent save knowledge, learnings, and rules automatically.

**How to test:**
1. Say: "Remember that my favorite restaurant is Olive Garden"
2. Mawa should call `memory_save` with category=knowledge
3. Check the file was created:

```bash
ls /root/.nanobot/workspace/knowledge/general/
cat /root/.nanobot/workspace/knowledge/general/olive-garden.md 2>/dev/null || echo "File not created yet"
```

**More tests:**
- "Remember Tonni's ring size is 6" → should update `knowledge/people/tonni.md`
- "Never use web scraping for stock data" → should save as learning or rule
- "From now on always check my work email first" → should save as learning

**Monitor logs:**
```bash
journalctl -u nanobot -f | grep "Memory save"
```

---

## Test 10: Memory Consolidation

**What changed:** Now triggers after 50 messages (was only at 131K tokens — never triggered).

**How to test:**
1. Have a long conversation (50+ messages in one session)
2. Check if MEMORY.md gets populated:

```bash
cat /root/.nanobot/workspace/memory/MEMORY.md
```

**Or check consolidation status:**
```bash
journalctl -u nanobot | grep -E "consolidation|Token consolidation" | tail -5
```

---

## Test 11: Git Backup

**What changed:** Workspace auto-commits every 30 minutes. Sensitive files encrypted with git-crypt.

**How to test:**
```bash
cd /root/.nanobot/workspace

# Check backup history
git log --oneline

# Check encryption status
git-crypt status | head -10

# Force a manual backup check
git status
```

**After 30 minutes**, a new commit should appear automatically.

**Encryption key location:** `/root/.nanobot/git-crypt-key.bin` — BACK THIS UP!

---

## Test 12: Token Budget

**What changed:** System prompt cut from 5,449 to 2,225 tokens (59% reduction).

**How to verify:**
```bash
python3 -c "
import tiktoken
enc = tiktoken.get_encoding('cl100k_base')
from pathlib import Path
from nanobot.agent.context import ContextBuilder
ctx = ContextBuilder(Path('/root/.nanobot/workspace'))
voice = ctx.build_system_prompt(channel='web_voice')
text = ctx.build_system_prompt(channel='discord')
print(f'Voice: {len(enc.encode(voice))} tokens')
print(f'Text:  {len(enc.encode(text))} tokens')
"
```

**Expected:** ~2200 voice, ~2000 text

---

## Troubleshooting

**Nanobot won't start:**
```bash
journalctl -u nanobot --since "1 min ago" --no-pager | tail -20
```

**ToolsDNS won't start:**
```bash
journalctl -u tooldns --since "1 min ago" --no-pager | tail -20
```

**Memory search returns nothing:**
```bash
# Re-index manually
python3 -c "
import asyncio
from pathlib import Path
from nanobot.memory.indexer import MemoryIndexer
async def main():
    idx = MemoryIndexer(Path('/root/.nanobot/workspace'), 'http://127.0.0.1:8787', '$(python3 -c "import json; print(json.load(open('/root/.nanobot/config.json'))['tools']['toolsdns']['apiKey'])")')
    idx._state = {'file_hashes': {}, 'history_watermark': 0}
    print(f'Indexed {await idx.index_all()} chunks')
asyncio.run(main())
"
```

**Approval guard not triggering:**
- Only triggers on `GMAIL_SEND_*`, `*_DELETE_*`, `*_REMOVE_*`, `DISCORDBOT_CREATE_MESSAGE`
- Regular reads (GMAIL_FETCH_EMAILS, GOOGLECALENDAR_FIND_EVENT) pass through without approval

**Restart both services:**
```bash
sudo systemctl restart tooldns && sleep 15 && sudo systemctl restart nanobot
```
