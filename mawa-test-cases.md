# Mawa AI — Comprehensive Test Cases

**Version:** March 24, 2026
**Total Features:** 100+
**Total Settings:** 50 configurable

---

## 1. Voice Mode & TTS

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-001 | Start voice call | Tap phone icon in chat input | Call panel slides in with avatar + waveform | P0 |
| TC-002 | Speak and get response | Start call → speak a question | Deepgram transcribes → Mawa responds with TTS | P0 |
| TC-003 | Type while on call | Start call → type in text input | Text sends while voice stays active | P0 |
| TC-004 | Interrupt TTS | Tap call panel while Mawa is speaking | Audio stops immediately, status → listening | P0 |
| TC-005 | Mic mute | Tap mic icon during call | Red mic icon, Mawa can't hear you, Deepgram stays connected | P1 |
| TC-006 | Speaker mute | Tap speaker icon during call | Mawa responds text-only, no audio plays | P1 |
| TC-007 | Stop button | Tap stop icon while Mawa speaks | TTS stops + agent cancels + interrupt sent | P1 |
| TC-008 | Hang up | Tap red phone-off button | Call ends, text input returns to normal | P0 |
| TC-009 | Smart endpointing | Speak an incomplete sentence, pause | Waits 1.5s before submitting incomplete sentences | P2 |
| TC-010 | Speak Reasoning toggle | Settings > Voice > Speak Reasoning ON | Mawa reads thinking/reasoning blocks aloud | P3 |
| TC-011 | Mic Sensitivity slider | Settings > Voice > adjust slider | Higher = ignores noise, lower = picks up quiet speech | P2 |
| TC-012 | No double TTS | Ask any question in voice mode | Response plays once, not twice | P0 |

---

## 2. Chat & Messaging

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-020 | Send text message | Type message → press Enter or tap Send | Message appears in chat, Mawa responds | P0 |
| TC-021 | Auto-scroll on view switch | Go to Settings → come back to Chat | Chat scrolled to latest message automatically | P0 |
| TC-022 | Message search | Tap search icon in header | Search bar opens, can search across history | P1 |
| TC-023 | Notification bell | Tap bell icon in header | Shows notification dropdown with unread badge | P1 |
| TC-024 | Click notification | Open bell → click a notification | Content injected into chat | P2 |
| TC-025 | Quick replies | After Mawa responds, check below message | Context-aware suggestion buttons appear | P2 |
| TC-026 | Emoji reactions | Double-tap or long-press assistant message | Reaction picker appears (👍👎❤️💡😂😮) | P2 |
| TC-027 | Thumbs down feedback | React with 👎 to a message | Toast shows lesson learned, saved to LEARNINGS.md | P1 |
| TC-028 | New session | Tap "New" button in header | Chat clears, session consolidated to memory | P1 |
| TC-029 | Keyboard shortcut Cmd+K | Press Cmd+K (or Ctrl+K) | Search toggles | P2 |
| TC-030 | Keyboard shortcut Cmd+N | Press Cmd+N | New chat created | P2 |
| TC-031 | Keyboard shortcut Escape | Press Escape | All overlays close | P2 |
| TC-032 | Auto-focus input | Start typing any letter | Input field auto-focuses | P3 |

---

## 3. Image & File Attachments

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-040 | Paperclip upload | Tap 📎 → select an image | Preview thumbnail appears in strip above input | P0 |
| TC-041 | Send image to LLM | Attach image → type "what's this?" → Send | Image shows in chat bubble + Mawa describes it via vision | P0 |
| TC-042 | Drag-drop image | Drag image file onto chat area | Blue drop zone appears, image attaches | P1 |
| TC-043 | Paste image | Copy image → Ctrl+V in chat | Image attaches from clipboard | P1 |
| TC-044 | Remove attachment | Click X on thumbnail preview | Attachment removed before sending | P2 |
| TC-045 | Document upload (XLSX) | Attach XLSX → "read this file" | Saved to inbox with original name, Mawa reads content | P1 |
| TC-046 | Smart folder routing | Attach file with message "work order form" | Saved to inbox/work/ (not general/) | P1 |
| TC-047 | Original filename kept | Attach "Work_Order_Form.xlsx" | File keeps name (not renamed to upload_123.xlsx) | P0 |
| TC-048 | Image resize | Attach a large phone photo (5MB+) | Resized to max 1024px, compressed to JPEG | P1 |
| TC-049 | Multiple attachments | Attach 3 images at once | All show in preview strip, all sent | P2 |
| TC-050 | File replacement in skill | Attach XLSX + "replace work order template" | Mawa copies file to skill's workspace | P1 |

---

## 4. Intelligence & Context

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-060 | Math interceptor | "What is 15% of $347?" | Instant answer: $52.05 — zero LLM tokens | P0 |
| TC-061 | Timezone interceptor | "3pm EST in Tokyo" | Instant conversion — zero tokens | P1 |
| TC-062 | Regex builder | "Regex for email addresses" | Returns pattern — zero tokens | P2 |
| TC-063 | Greeting interceptor | "Hello" (just greeting, nothing else) | Time-aware greeting with goals context | P1 |
| TC-064 | Greeting with question | "Hey, what's the bitcoin price?" | NOT caught by greeting — goes to LLM | P0 |
| TC-065 | Intent tracking | "Find shops" then "send this to Tonni" | Mawa knows "this" = the shops | P0 |
| TC-066 | Conversation recap | After 4+ turns, check context | Recap of recent exchanges injected (~40 tokens) | P1 |
| TC-067 | Frustration detection | "THIS DOESNT WORK!!!" | Empathetic preamble, concerned tone | P1 |
| TC-068 | Language detection | Write in Bangla: "আমি ভালো আছি" | Mawa responds in Bangla (text) or English (voice) | P2 |
| TC-069 | Destructive confirmation | "Delete all my goals" | Mawa asks for confirmation before proceeding | P0 |
| TC-070 | Response caching | Ask same question twice within 5 min | Second answer from cache (faster, no LLM) | P2 |
| TC-071 | Message dedup | Send same message twice quickly | Second one silently dropped | P2 |
| TC-072 | Error classification | Trigger a tool timeout | Error categorized with recovery hint | P1 |
| TC-073 | Quality gate | If Mawa deflects "I can't help" | Retries once with tools | P1 |
| TC-074 | Loop detection | If Mawa repeats same response | Breaks loop with different-approach nudge | P2 |
| TC-075 | Pronoun resolution | System prompt includes pronoun rules | "Send this" resolves from previous message | P0 |

---

## 5. Memory System

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-080 | Memory graph view | Sidebar → Memory | Animated cards with activity dots, data flow diagram | P1 |
| TC-081 | Tap to expand layer | Tap any memory card | Shows detailed explanation of what it does | P1 |
| TC-082 | Consolidate Now | Memory → "Consolidate Now" button | LLM consolidation runs, toast confirms | P1 |
| TC-083 | Clear Today | Memory → "Clear Today" button | Consolidates first, then archives SHORT_TERM | P1 |
| TC-084 | Learnings panel | Memory → "What Mawa Learned" | Shows user corrections separately from tool errors | P1 |
| TC-085 | Memory timeline | Memory → "Recent Activity" | Color-coded events (learnings, episodes, feedback) | P2 |
| TC-086 | /learnings command | Type "/learnings" in chat | Shows what Mawa learned from feedback | P1 |
| TC-087 | Memory search | Memory → search box → type query | Results from across all memory layers | P2 |
| TC-088 | Heartbeat consolidation | Wait 30 min (or check logs) | Auto-consolidates stale sessions | P2 |
| TC-089 | Disconnect consolidation | Close browser tab | Auto-consolidates on WebSocket disconnect | P2 |
| TC-090 | History auto-archive | HISTORY.md > 100KB | Old months archived to history/YYYY-MM.md | P3 |
| TC-091 | Separate learnings files | Check LEARNINGS.md vs TOOL_LEARNINGS.md | User corrections separate from tool errors | P1 |
| TC-092 | Reflection on soft hints | Say "next time use X instead" | Detected and saved as learning (threshold 0.4) | P1 |

---

## 6. Settings & Configuration

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-100 | Dynamic backend features | Settings > Autonomy > scroll to Backend Features | All 50 features loaded from /api/features | P0 |
| TC-101 | Toggle a boolean | Toggle any switch | Saves immediately to mawa_settings.json | P0 |
| TC-102 | Change a number | Edit Daily Budget ($) field | Value persists on refresh | P0 |
| TC-103 | **Dropdown selector** | Click Image Provider dropdown | Shows: pollinations, huggingface, together, fal, etc. | P0 |
| TC-104 | **Dropdown for TTS** | Click Voice TTS Provider dropdown | Shows: deepgram, mimo-audio, fish-speech, mms-tts | P0 |
| TC-105 | **Dropdown for STT** | Click Voice STT Provider dropdown | Shows: deepgram, mimo-audio | P0 |
| TC-106 | **Dropdown for language** | Click Voice TTS Language dropdown | Shows: en, bn, hi, zh, ur | P0 |
| TC-107 | **Dropdown for call voice** | Click Call Voice dropdown | Shows: alice, man, woman, Polly.Joanna, Polly.Matthew | P0 |
| TC-108 | **Dropdown for call mode** | Click Call Mode dropdown | Shows: tts, conversation | P1 |
| TC-109 | Free text for URLs | Edit MiMo-Audio Endpoint | Text input (not dropdown) for custom URLs | P1 |
| TC-110 | Settings via Mawa chat | "Turn off frustration detection" | Mawa uses settings tool to change it | P0 |
| TC-111 | "What can you do?" | Ask Mawa | Lists all capabilities via settings tool | P0 |
| TC-112 | "Show my settings" | Ask Mawa | Lists all 50 settings with values | P1 |
| TC-113 | Unified settings file | Check workspace/mawa_settings.json | All settings in ONE file | P0 |
| TC-114 | Categories displayed | Check Backend Features section | 7 categories: Intelligence, Behavior, Jarvis, Media, Notifications, Budget, Maintenance | P1 |

---

## 7. Notifications & Proactive

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-120 | Proactive alerts | Have an overdue goal → wait for proactive check | Alert appears in chat + notification bell | P1 |
| TC-121 | Quiet hours | Enable quiet hours (10pm-7am) → trigger notification at night | Non-urgent notifications deferred, high-priority goes through | P1 |
| TC-122 | Notification persistence | Trigger notification while offline → reconnect | Pending notifications delivered on reconnect | P1 |
| TC-123 | Habit reminders | Create habit: "Drink water every 2 hours" | Reminder appears after interval | P2 |
| TC-124 | Background job notification | Run a background job → wait for completion | Mawa notifies you when job finishes | P1 |
| TC-125 | Cron job to web | Create cron job with deliver=true | Result appears in Mawa chat (not just Discord) | P1 |

---

## 8. Jarvis Intelligence

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-130 | Morning prep | `GET /api/jarvis/morning-prep` | Goals, bills, relationships, habits, correlations | P1 |
| TC-131 | Cross-signal correlation | Have travel + weather risk in memory | "Travel planned + weather risk" alert | P1 |
| TC-132 | Relationship tracker | Contact someone → wait 14 days | "Haven't contacted X in 14 days" reminder | P2 |
| TC-133 | Financial pulse | `GET /api/jarvis/financial` | AI spending trends, bill countdown | P2 |
| TC-134 | People prep | `GET /api/people-prep?name=Tonni` | Everything Mawa knows about Tonni | P2 |
| TC-135 | Life dashboard | `GET /api/jarvis/dashboard` | Unified health/wealth/relationships/work view | P2 |
| TC-136 | Daily digest | `GET /api/jarvis/digest` | End-of-day summary | P2 |
| TC-137 | Project tracker | `POST /api/projects` with project data | Project saved with progress tracking | P3 |
| TC-138 | Delegation queue | `POST /api/delegations` with task | Mawa checks on it periodically | P3 |
| TC-139 | Decision memory | Make a decision → later ask "why did I choose X?" | Finds related decision with reasoning | P3 |
| TC-140 | Routine detection | Use same tools at same time for 5+ days | Mawa suggests automation | P3 |

---

## 9. Security & Credentials

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-150 | Smart credential naming | "My GitHub token is ghp_test123" | Saved as `github_token` (not `auto_1`) | P0 |
| TC-151 | URL-based naming | "Password for https://outlook.com is MyP@ss" | Saved as `outlook_password` | P1 |
| TC-152 | Secret masking | Send a password in chat | Masked before LLM sees it: {cred:name} | P0 |
| TC-153 | Outbound security filter | If response contains sk-abc123... | Stripped to [REDACTED] | P0 |
| TC-154 | Credential injection | Use {cred:github_token} in browser | Actual value injected at execution time | P1 |
| TC-155 | Provider credential validation | Select a provider without API key | Clear instructions: "Save X to vault, get key at URL" | P0 |

---

## 10. Tools & Automation

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-160 | Background exec run | "Run sleep 3 && echo done in background" | Returns job ID immediately | P1 |
| TC-161 | Background exec status | "Check job bg_1" | Shows running/completed + output | P1 |
| TC-162 | Background exec kill | Start long job → "Kill job bg_1" | Job terminated | P2 |
| TC-163 | Tool timeout | Tool hangs for >45s | Timeout error, not infinite hang | P0 |
| TC-164 | Stream timeout | LLM stream hangs for >180s | Timeout with partial content recovery | P0 |
| TC-165 | Strategy rotation | Tool fails → retry | Suggests alternative tool | P2 |
| TC-166 | Parallel safety | 2 tools write to same file | Serialized (not corrupted) | P1 |
| TC-167 | Tool call validation | LLM hallucinates tool name | Error with closest match suggestion | P1 |
| TC-168 | MCP auto-reconnect | MCP server crashes | Reconnects after 3 failures | P2 |

---

## 11. Media Generation

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-170 | Image generation (free) | "Generate an image of a sunset" | Image generated via Pollinations (free, no key) | P1 |
| TC-171 | Image style | "Draw a sketch of a cat" | Pencil sketch style applied | P2 |
| TC-172 | **Image provider dropdown** | Settings > Media > Image Provider | Dropdown: pollinations, huggingface, together, etc. | P0 |
| TC-173 | Switch image provider | Change to "huggingface" in dropdown | Next image uses HuggingFace | P1 |
| TC-174 | Phone call (TTS) | "Call +1234567890 and say hello" | Twilio makes call, speaks message | P2 |
| TC-175 | **Call voice dropdown** | Settings > Media > Call Voice | Dropdown: alice, man, woman, Polly voices | P1 |

---

## 12. Voice Providers

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-180 | List providers | `GET /api/voice/providers` | 4 providers with capabilities listed | P1 |
| TC-181 | List voices | `GET /api/voice/voices?provider=deepgram` | 6 Deepgram voices with names | P1 |
| TC-182 | Validate provider | `POST /api/voice/validate {provider:"deepgram"}` | Reports missing credentials with instructions | P0 |
| TC-183 | Upload voice sample | `POST /api/voice/samples {audio_b64, name}` | WAV saved to workspace/voice_samples/ | P2 |
| TC-184 | List voice samples | `GET /api/voice/samples` | Shows saved clone samples | P2 |
| TC-185 | Delete voice sample | `DELETE /api/voice/samples/my_voice` | Sample removed | P3 |
| TC-186 | **TTS provider dropdown** | Settings > Media > Voice TTS Provider | Dropdown: deepgram, mimo-audio, fish-speech, mms-tts | P0 |
| TC-187 | **STT provider dropdown** | Settings > Media > Voice STT Provider | Dropdown: deepgram, mimo-audio | P0 |
| TC-188 | **Language dropdown** | Settings > Media > Voice TTS Language | Dropdown: en, bn, hi, zh, ur | P0 |

---

## 13. Maintenance & Health

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-190 | Health dashboard | `GET /api/health/dashboard` | Disk, sessions, memory, cron, tools, budget | P1 |
| TC-191 | Session cleanup | Wait for heartbeat (30 min) | Old sessions (>7d) deleted, keeps 5 recent | P2 |
| TC-192 | History archive | HISTORY.md > 100KB + heartbeat | Old months moved to history/YYYY-MM.md | P3 |
| TC-193 | Contact auto-extract | Heartbeat runs | Contacts extracted from LONG_TERM.md | P3 |
| TC-194 | File auto-cleanup | `POST /api/cleanup` | Files >30 days deleted, quota enforced | P3 |
| TC-195 | Session search | `GET /api/sessions/search?q=email` | Results from across all past sessions | P2 |
| TC-196 | Anomaly detection | `GET /api/anomalies` | Alerts if usage >3x the 7-day average | P3 |
| TC-197 | Export conversation | `GET /api/export/conversation` | Markdown export of current conversation | P2 |

---

## 14. Discord & Channels

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-200 | Discord DM | Send DM to Mawa bot on Discord | Mawa receives and responds | P0 |
| TC-201 | Discord server message | Send message in allowed server channel | Mawa responds in channel | P1 |
| TC-202 | Telegram | Send message to @MawaOSBot | Mawa receives and responds | P1 |
| TC-203 | Web voice | Open Mawa dashboard → use voice mode | Full voice conversation works | P0 |
| TC-204 | Cron delivery to web | Cron job fires | Result appears in Mawa web chat as notification | P1 |

---

## 15. Pipeline Optimizations

| ID | Feature | Steps | Expected Result | Priority |
|---|---|---|---|---|
| TC-210 | Intent-based tool filtering | "Check my email" | Only ~13 tools sent to LLM (not all 27) | P1 |
| TC-211 | Response length (voice) | Ask question in voice mode | Short response (1-3 sentences) | P1 |
| TC-212 | Response format (table) | "Compare Python vs JavaScript" | Response uses markdown table | P2 |
| TC-213 | Response format (steps) | "How to deploy to Vercel" | Response uses numbered steps | P2 |
| TC-214 | History compression | After 20+ messages | Old turns compressed to one-liners | P2 |
| TC-215 | Parallel prefetch | "Check email and calendar" | Hint to call both tools in parallel | P2 |
| TC-216 | Follow-up chain | "Check email then reply to boss" | Detected as 2 steps, executed in order | P2 |
| TC-217 | Token count | Check logs for token usage | ~5,500 per message (down from 15,164) | P1 |

---

## Quick Smoke Test (5 min)

1. ✅ Open Mawa → say "Hello" → instant greeting
2. ✅ Type "What's 15% of 200?" → instant $30.00
3. ✅ Tap 📎 → attach image → "what's this?" → Mawa sees + describes
4. ✅ Settings > Autonomy > Backend Features > toggle one feature
5. ✅ Ask "What features do you have?" → Mawa lists settings
6. ✅ Settings > Media > Image Provider dropdown → select different provider
7. ✅ Voice mode → speak → Mawa responds with audio
8. ✅ Discord DM → send "Hi" → Mawa responds

---

*Generated: March 24, 2026*
*Total test cases: 120*
*Total features tested: 100+*
