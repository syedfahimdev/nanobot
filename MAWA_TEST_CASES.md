# Mawa AI Assistant — Comprehensive Test Cases

**Version:** 1.0
**Date:** 2026-03-24
**Total Test Cases:** 256
**Categories:** 15+

---

## 1. Voice Mode & TTS

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-001 | Deepgram STT (Speech-to-Text) | P0 | 1. Open web dashboard. 2. Click the call/voice button. 3. Speak a sentence clearly. | Speech is transcribed to text in the chat input; recognized text appears within 2 seconds. |
| TC-002 | Deepgram TTS (Text-to-Speech) | P0 | 1. Enable voice mode. 2. Send a text message. 3. Listen for audio playback. | Mawa reads the response aloud, sentence-by-sentence, with streaming playback. |
| TC-003 | Streaming sentence-by-sentence TTS | P1 | 1. Start voice mode. 2. Ask a multi-paragraph question (e.g., "explain how DNS works"). | Audio starts playing after the first sentence is generated, not after the entire response. |
| TC-004 | Smart voice endpointing | P1 | 1. Start voice mode. 2. Speak a long sentence with a natural mid-sentence pause (e.g., "I want to... check my email"). | The system waits for semantic sentence completion rather than cutting off at the first silence gap. |
| TC-005 | Voice + text simultaneously | P1 | 1. Start a voice call. 2. While call is active, type a message in the text input. | Text message is sent and processed while voice panel remains active above the text input. |
| TC-006 | 180s stream timeout | P2 | 1. Start voice mode. 2. Ask a question that triggers a very long response or long tool chain. | After 180 seconds, stream times out gracefully with partial content recovery, no crash. |
| TC-007 | 45s per-tool timeout | P2 | 1. During voice mode, trigger a tool call that hangs (e.g., fetch an unreachable URL). | Tool times out after 45 seconds via asyncio.wait_for; voice session continues with an error message. |
| TC-008 | Discord voice channel support | P1 | 1. Connect Mawa to a Discord server. 2. Join a voice channel. 3. Speak to Mawa. | Mawa receives audio via opus codec, transcribes, responds with TTS in the voice channel. |
| TC-009 | Response length control, voice mode | P1 | 1. Be in voice mode. 2. Ask "what is machine learning?" | Response is SHORT (1-3 sentences) because channel is voice, not a long essay. |

---

## 2. Chat & Messaging

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-010 | WebSocket streaming responses | P0 | 1. Open dashboard. 2. Send any message. | Response streams token-by-token over WebSocket with an animated typing indicator. |
| TC-011 | Markdown rendering (GFM) | P1 | 1. Ask "show me a code example in Python". | Response renders with proper markdown: code blocks with syntax highlighting, tables, links. |
| TC-012 | Generative UI, inline HTML widgets | P1 | 1. Ask about weather or a topic that returns structured data. | Inline HTML widget renders directly in the chat bubble (not raw HTML text). |
| TC-013 | Tool result cards | P1 | 1. Ask "check my email". | Email results display as formatted cards with sender, subject, date, not raw JSON. |
| TC-014 | File attachments with icons | P1 | 1. Upload a file (PDF, image, etc.) via the chat interface. | File appears with a type-specific icon, a preview thumbnail (for images), and a download button. |
| TC-015 | Quick reply suggestions | P2 | 1. After Mawa asks a follow-up question. | Context-aware quick reply buttons appear below the message. |
| TC-016 | Message search | P1 | 1. Click search icon. 2. Type a keyword from a previous conversation. | Results returned from across all conversation history and memory layers. |
| TC-017 | Emoji reactions (self-improvement) | P2 | 1. Long-press or right-click a Mawa response. 2. React with a thumbs-down emoji. | Reaction is recorded; Mawa's self-improvement system logs it for prompt optimization. |
| TC-018 | Notification bell with unread badge | P1 | 1. Close the dashboard. 2. Wait for a proactive notification. 3. Reopen. | Bell icon shows an unread count badge; clicking opens a notification tray with clickable items. |
| TC-019 | Pinned messages | P2 | 1. Long-press a message. 2. Select "Pin". | Message is pinned and accessible from a pinned messages panel. |
| TC-020 | Export conversation (Markdown) | P2 | 1. Open settings or menu. 2. Click Export. 3. Select Markdown format. | A .md file downloads with proper formatting: headers per role, timestamps, content. |
| TC-021 | Export conversation (JSON) | P2 | 1. Export in JSON format. | A .json file downloads with the full session array including timestamps and roles. |
| TC-022 | Keyboard shortcuts (Cmd+K) | P3 | 1. Press Cmd+K (or Ctrl+K). | Command palette opens for quick actions. |
| TC-023 | Progressive disclosure | P2 | 1. Ask a yes/no question: "Is Python an interpreted language?" | Mawa answers directly (yes/no) then gives a brief explanation, not a lengthy essay. |
| TC-024 | Response length, Telegram channel | P2 | 1. Send a message via Telegram. 2. Ask a detailed question. | Response is concise, formatted for chat, not a long essay. |

---

## 3. Image & File Attachments

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-025 | Image generation, Pollinations (free) | P0 | 1. Set imageGenProvider to "pollinations". 2. Say "generate an image of a sunset over mountains". | Image is generated via Pollinations API (no key needed), saved to workspace/generated/images/, and displayed in chat. |
| TC-026 | Image generation, Together AI | P1 | 1. Save together_api_key in vault. 2. Set provider to "together". 3. Ask for an image. | Image generated via Together AI Flux model, displayed in chat. |
| TC-027 | Image generation, HuggingFace | P1 | 1. Save HF_TOKEN in vault. 2. Set provider to "huggingface". 3. Ask for an image. | Image generated via HuggingFace Inference API. |
| TC-028 | Image generation, Fal.ai | P2 | 1. Save fal_api_key. 2. Set provider to "fal". 3. Request image. | Image generated via Fal Flux/SDXL endpoint. |
| TC-029 | Image generation, Replicate | P2 | 1. Save replicate_api_token. 2. Set provider to "replicate". 3. Request image. | Image generated via Replicate with async polling for completion. |
| TC-030 | Image generation, OpenAI DALL-E | P1 | 1. Set provider to "openai". 2. Request image. | Image generated via DALL-E 3, returned as base64. |
| TC-031 | Image generation, Stability AI | P2 | 1. Set provider to "stability". 2. Request image. | Image generated via Stability AI SD3. |
| TC-032 | Image style parameter | P2 | 1. Say "generate a sketch of a cat". | Style prefix "pencil sketch, " is prepended to the prompt; output matches style. |
| TC-033 | Image size parameter | P2 | 1. Say "generate a landscape image of a beach". | Image dimensions are 1280x768 (landscape), not square. |
| TC-034 | Auto-fallback to free provider | P1 | 1. Set provider to "together" but do NOT save API key. 2. Request image. | System falls back to Pollinations (free) and generates the image successfully. |
| TC-035 | File upload with type detection | P1 | 1. Upload a CSV file. 2. Upload a PDF. 3. Upload a .py file. | Each file gets a type-specific icon; preview is offered where applicable. |

---

## 4. Intelligence & Context

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-036 | Smart Error Recovery | P0 | 1. Trigger a tool that returns a 401 error (e.g., bad API key). | Error is classified as "auth"; recovery hint says "Check credentials or API keys." Agent does NOT blindly retry same call. |
| TC-037 | Smart Error Recovery, timeout | P1 | 1. Trigger a tool timeout (e.g., fetch unreachable URL). | Error classified as "timeout"; hint suggests simpler query or check service reachability. |
| TC-038 | Smart Error Recovery, rate limit | P1 | 1. Trigger a 429 error. | Error classified as "rate_limit"; hint says wait before retrying. |
| TC-039 | Intent Tracking, pronoun resolution | P0 | 1. Say "search for weather in NYC". 2. Then say "send that to my email". | Mawa correctly resolves "that" to the weather results from turn 1, not confused. |
| TC-040 | Intent Tracking, short follow-up | P1 | 1. Ask a complex question. 2. Follow up with "what about Tokyo?" | Intent tracker carries forward the original topic; "Tokyo" is interpreted in context. |
| TC-041 | Dynamic Context Budget | P1 | 1. Trigger multiple tools in a single turn (e.g., "check email, calendar, and weather"). | Each tool result is sized proportionally to remaining context window, not a fixed 16K cap. Min 2K, max 24K per tool. |
| TC-042 | Response Quality Gate | P1 | 1. Ask a question that the LLM might deflect (e.g., something it needs a tool for). | If LLM responds with "I can't help with that" without using any tools, quality gate catches it and retries with tools. |
| TC-043 | Tool Call Validation | P1 | 1. (Internal) LLM hallucinates a non-existent tool name. | Validator catches it, suggests closest matching real tool names. Error message includes available tools. |
| TC-044 | Parallel Safety, conflicting writes | P1 | 1. LLM requests two simultaneous write_file calls to the same path. | Parallel safety detects the conflict; second write is serialized (run after first completes). |
| TC-045 | Parallel Safety, safe reads | P2 | 1. LLM requests web_search + read_file + memory_search simultaneously. | All three run in parallel since they are read-only. |
| TC-046 | MCP Auto-Reconnect | P1 | 1. Disconnect an MCP server mid-session. 2. Try to use an MCP tool 3 times. | After 3 consecutive MCP connection failures, auto-reconnect is triggered (30s cooldown). |
| TC-047 | MCP Reconnect, success resets counter | P2 | 1. Trigger 2 MCP failures. 2. Then a successful MCP call. 3. Then 1 more failure. | Successful call resets the failure counter; reconnect is NOT triggered at failure 3 overall. |
| TC-048 | Streaming Recovery | P2 | 1. Simulate a stream failure mid-response. | System retries on stream failure, preserving any progress already received. |
| TC-049 | Pending Message Awareness | P2 | 1. Send a message while Mawa is already processing another. | Queued message is re-dispatched after the current processing completes, not silently dropped. |

---

## 5. Memory System

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-050 | 4-layer memory, short-term | P0 | 1. Have a conversation. 2. Check workspace/memory/SHORT_TERM.md. | Recent conversation context is stored in SHORT_TERM.md. |
| TC-051 | 4-layer memory, long-term | P0 | 1. Tell Mawa a personal fact ("My birthday is March 15"). 2. Trigger consolidation. 3. Check LONG_TERM.md. | Fact is consolidated into LONG_TERM.md and retrievable in future conversations. |
| TC-052 | 4-layer memory, observations | P1 | 1. Use the same tools repeatedly at the same time of day. 2. Check OBSERVATIONS.md. | Usage patterns are recorded (e.g., "Uses GMAIL_FETCH mostly in the morning (8x)"). |
| TC-053 | 4-layer memory, episodes | P1 | 1. Complete a multi-step task. 2. Check EPISODES.md or equivalent. | The episode is recorded as a sequence of actions and outcomes. |
| TC-054 | Learnings (LEARNINGS.md) | P1 | 1. Correct Mawa: "No, I prefer short responses". 2. Check LEARNINGS.md. | Correction is saved with timestamp in LEARNINGS.md; Mawa follows it in future turns. |
| TC-055 | Tool Learnings (TOOL_LEARNINGS.md) | P2 | 1. A tool fails in a specific way. 2. Mawa learns the workaround. | Tool-specific learning is saved in TOOL_LEARNINGS.md separately from user learnings. |
| TC-056 | 6-trigger consolidation, 20 messages | P0 | 1. Send 20 messages in a session. | Memory consolidation triggers automatically after 20 messages. |
| TC-057 | 6-trigger consolidation, heartbeat | P1 | 1. Keep session open for 30+ minutes with no activity. | Heartbeat triggers consolidation at the 30-minute mark. |
| TC-058 | 6-trigger consolidation, disconnect | P1 | 1. Close the browser tab or disconnect. | Memory consolidation triggers on disconnect event. |
| TC-059 | 6-trigger consolidation, /new command | P1 | 1. Type "/new" to start a new conversation. | Consolidation runs before clearing the session. |
| TC-060 | 6-trigger consolidation, Clear Today | P2 | 1. Click "Clear Today" button in memory settings. | Consolidation runs on the current day's data before clearing. |
| TC-061 | 6-trigger consolidation, manual button | P2 | 1. Go to Settings > Memory. 2. Click "Consolidate Now". | POST /api/memory/consolidate triggers LLM consolidation immediately. |
| TC-062 | Conversation recap injection | P1 | 1. Start a new session. 2. Reference something from yesterday. | ~40 token extractive summary is injected per turn from memory, enabling context continuity. |
| TC-063 | Pronoun resolution, system prompt rule | P1 | 1. Say "send it to him". | System prompt + intent tracking resolves "it" and "him" from recent context. |
| TC-064 | Dynamic capabilities manifest | P2 | 1. Ask "what can you do?" | Mawa auto-generates a capabilities manifest from registered tools and features. |
| TC-065 | /learnings command | P2 | 1. Type "/learnings" or navigate to GET /api/learnings. | Displays all learned corrections and preferences from LEARNINGS.md. |
| TC-066 | Memory Activity Timeline | P2 | 1. Go to Settings > Memory > Timeline. 2. Call GET /api/memory/timeline. | Color-coded events displayed: learnings (one color), episodes (another), feedback (another). |
| TC-067 | Memory search API | P1 | 1. GET /api/memory. | Returns memory layer status with sizes for each layer. |

---

## 6. Settings & Configuration

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-068 | Settings tool, list all | P0 | 1. Say "list all your settings". | Mawa calls settings(action='list') and returns all 50+ settings grouped by category. |
| TC-069 | Settings tool, get specific | P1 | 1. Say "what is your quiet hours setting?" | Mawa calls settings(action='get', key='quietHoursEnabled') and returns current value with description. |
| TC-070 | Settings tool, set boolean | P0 | 1. Say "turn off frustration detection". | Mawa calls settings(action='set', key='frustrationDetection', value=false). Setting updated in mawa_settings.json. |
| TC-071 | Settings tool, set number | P1 | 1. Say "set my daily budget to $10". | Mawa calls settings(set, 'budget_daily_limit', 10). Value stored as number, not string. |
| TC-072 | Settings tool, set string | P1 | 1. Say "change image provider to openai". | Mawa calls settings(set, 'imageGenProvider', 'openai'). Value persisted correctly. |
| TC-073 | Settings tool, search | P1 | 1. Say "do you have any voice settings?" | Mawa calls settings(action='search', query='voice'). Returns all voice-related settings. |
| TC-074 | Settings tool, type coercion | P2 | 1. LLM passes "true" as string for boolean setting. | Tool coerces "true" string to boolean True before saving. |
| TC-075 | Settings tool, invalid key | P2 | 1. Try to set a non-existent key: settings(set, 'fakeKey', true). | Returns "Setting 'fakeKey' not found" with suggestion to search. |
| TC-076 | Unified settings file | P1 | 1. Change multiple settings via different methods. 2. Check workspace/mawa_settings.json. | ALL settings stored in ONE file: mawa_settings.json. No scattered intelligence.json, jarvis_settings.json, etc. |
| TC-077 | Settings migration | P2 | 1. Place old scattered files (intelligence.json, jarvis_settings.json) in workspace. 2. Start Mawa. | migrate_old_settings() merges old files into mawa_settings.json; old values don't overwrite existing unified values. |
| TC-078 | Feature manifest API | P1 | 1. GET /api/features. | Returns full manifest with key, label, desc, category, type, current value for all features. |
| TC-079 | Feature categories API | P2 | 1. Inspect get_feature_categories(). | Returns 7 ordered categories: intelligence, behavior, jarvis, media, notifications, budget, maintenance. |
| TC-080 | Profile switching | P1 | 1. Go to Settings > Profile. 2. Switch to a different named preset. | Model, temperature, context window, and reasoning effort change per the preset. |
| TC-081 | Avatar styles | P3 | 1. Go to Settings > Avatar. 2. Switch between 3D face, emoji, simple, orb. | Avatar changes in the chat interface immediately. |
| TC-082 | Theme switching | P2 | 1. Go to Settings > Look. 2. Switch between midnight, whatsapp, telegram, discord, iphone, terminal. | UI theme changes immediately across the entire dashboard. |

---

## 7. Notifications & Proactive

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-083 | Quiet hours, enabled | P1 | 1. Set quietHoursStart=22, quietHoursEnd=7, quietHoursEnabled=true. 2. At 23:00, trigger a normal-priority notification. | Notification is deferred, not delivered during quiet hours. |
| TC-084 | Quiet hours, high priority override | P1 | 1. During quiet hours, trigger a high-priority notification (e.g., bill overdue). | High priority notifications bypass quiet hours and are delivered immediately. |
| TC-085 | Quiet hours, disabled | P2 | 1. Set quietHoursEnabled=false. 2. Trigger notification at any hour. | Notification is always delivered regardless of time. |
| TC-086 | Morning briefing (proactive) | P1 | 1. Set morningPrep=true. 2. Have goals, bills, relationships data in memory. 3. Wait for morning trigger. | Mawa generates a structured morning prep with sections: Goals, Financial, People, Habits, Heads Up, AI Usage. |
| TC-087 | Evening digest (proactive) | P2 | 1. Set dailyDigest=true. 2. Use Mawa throughout the day. 3. Wait for end-of-day trigger. | Daily digest generated with: AI usage stats, goals progress, new learnings, session count. |
| TC-088 | Proactive notifications, persistent | P1 | 1. Go offline. 2. A proactive notification is generated. 3. Come back online. | Notification is stored and delivered on reconnect, not lost. |
| TC-089 | Background job completion notification | P1 | 1. Run a background job (background_exec). 2. Job completes while user is idle. | Notification appears when job finishes with status and result summary. |
| TC-090 | Notification API, list | P2 | 1. GET /api/notifications. | Returns all stored notifications with timestamps, content, read status. |
| TC-091 | Notification API, mark read | P2 | 1. POST /api/notifications/read. | All notifications marked as read; badge count resets to 0. |
| TC-092 | Predictive suggestions | P2 | 1. Use email tool every morning at 9am for 5+ days. 2. GET /api/suggestions/predictive at 9am. | System suggests "Check your email?" based on observed time-of-day pattern. |

---

## 8. Jarvis Intelligence

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-093 | Morning Prep Engine | P0 | 1. Populate GOALS.md with items due today. 2. Add bills in LONG_TERM.md. 3. Call build_morning_prep(). | Returns structured prep with sections sorted by priority (high first): goals, financial alerts, people reminders, habits, correlations, AI usage. |
| TC-094 | Cross-Signal Correlation, travel+weather | P1 | 1. Mention "flight next Tuesday" in memory. 2. Also have "storm warning" in memory. 3. Call detect_correlations(). | Returns alert: "Travel planned + weather risk mentioned, check conditions before heading out". |
| TC-095 | Cross-Signal Correlation, bill due soon | P1 | 1. Add "payment due 2026-03-26" in LONG_TERM.md. 2. Call detect_correlations(). | Returns alert about bill/payment due within 3 days. |
| TC-096 | Cross-Signal Correlation, goal overdue | P1 | 1. Add a goal with past due date in GOALS.md: "- [ ] Submit report (due: 2026-03-20)". | Returns "OVERDUE goal (4d): Submit report". |
| TC-097 | Cross-Signal Correlation, wedding countdown | P2 | 1. Mention wedding date within 30 days in memory. | Returns "Wedding in N days, review wedding checklist". |
| TC-098 | Meeting Intelligence | P1 | 1. Have past meeting notes in HISTORY.md mentioning "Project Alpha". 2. Call get_meeting_prep(workspace, "Project Alpha", ["John"]). | Returns context snippets from history + attendee notes from contacts/memory for "John". |
| TC-099 | Upcoming meetings from memory | P2 | 1. Mention "meeting with Bob on 2026-03-25 at 3pm" in SHORT_TERM.md. 2. Call get_upcoming_meetings_from_memory(). | Returns event with date, time, and title extracted from memory text. |
| TC-100 | Relationship Tracker, record interaction | P1 | 1. Call record_interaction(workspace, "Alice", "message"). 2. Check relationships.json. | Alice's interaction count incremented; last_contact timestamp updated. |
| TC-101 | Relationship Tracker, reminders | P1 | 1. Record an interaction with "Bob" 15 days ago. 2. Call get_relationship_reminders(). | Returns "Haven't contacted Bob in 15 days". |
| TC-102 | Relationship Tracker, birthday | P2 | 1. Add contact with birthday 3 days from now in contacts.json. 2. Call get_relationship_reminders(). | Returns "Name's birthday in 3 days". |
| TC-103 | Relationship Tracker, birthday today | P2 | 1. Set birthday to today's date. | Returns "Name's birthday is TODAY!" |
| TC-104 | Financial Pulse | P1 | 1. Use Mawa for a week with varying costs. 2. Call get_financial_pulse(). | Returns AI spending summary, bill alerts, spending patterns (e.g., "2.5x your average today"). |
| TC-105 | Project Tracker, create | P1 | 1. Call save_project(workspace, {"name": "Website Redesign", "tasks": [{"name": "wireframe", "done": false}]}). | Project saved to projects.json with created timestamp, status "active", progress 0. |
| TC-106 | Project Tracker, progress | P2 | 1. Mark one of two tasks as done. 2. Call update_project_progress(). | Progress recalculated to 50%. |
| TC-107 | Daily Digest | P1 | 1. Use Mawa throughout the day (multiple turns, tools, goals). 2. Call build_daily_digest(). | Returns digest with: AI usage (turns, tokens, cost), goals (done/pending), new learnings, active sessions. |
| TC-108 | Priority Inbox, high priority | P1 | 1. Call score_message_priority("URGENT: server down, fix ASAP"). | Returns ("high", 0.9+). |
| TC-109 | Priority Inbox, low priority | P2 | 1. Call score_message_priority("Weekly newsletter digest, unsubscribe"). | Returns ("low", 0.3 or below). |
| TC-110 | Delegation Queue, add | P1 | 1. Say "delegate: check if the package arrived" or call add_delegation(). | Task added to delegations.json with status "active", check_interval_hours, created timestamp. |
| TC-111 | Delegation Queue, due for check | P1 | 1. Add a delegation with check_interval_hours=1. 2. Wait 1+ hour. 3. Call get_delegations_due_for_check(). | Returns the delegation as due for check-in. |
| TC-112 | Routine Detection | P2 | 1. Build OBSERVATIONS.md with "Uses GMAIL_FETCH_EMAILS mostly in the morning (8x)". 2. Call detect_routines(). | Returns routine with suggestion: "You check GMAIL_FETCH_EMAILS every morning, want me to do it automatically?" |
| TC-113 | Decision Memory, record | P1 | 1. Call record_decision(workspace, "Chose Claude over GPT-4", "Better at coding", "Model selection"). | Decision saved to decisions.json with reason, context, date. Max 50 kept. |
| TC-114 | Decision Memory, recall | P1 | 1. Record a decision about "hosting". 2. Call find_related_decisions(workspace, "hosting provider"). | Returns the related decision with its reasoning. |
| TC-115 | People Prep, before call | P1 | 1. Have interaction history with "Alice" and mentions in LONG_TERM.md. 2. Call get_people_prep(workspace, "Alice"). | Returns: total interactions, last contact date, recent interactions, contact info, memory mentions. |
| TC-116 | Life Dashboard | P2 | 1. Populate goals, relationships, projects, delegations. 2. Call get_life_dashboard(). | Returns unified dashboard with health (goals), wealth (financial), relationships, work (projects/delegations), upcoming events, correlations. |
| TC-117 | Proactive Jarvis, heartbeat | P1 | 1. Call check_proactive_jarvis(workspace). | Returns notifications from: correlations (high), relationship reminders (normal), delegation check-ins (normal), routine suggestions (low). |
| TC-118 | Configurable relationship reminder days | P2 | 1. Set relationshipReminderDays=30. 2. Contact was 15 days ago. | No reminder generated (below 30-day threshold). |
| TC-119 | Configurable delegation check hours | P2 | 1. Set delegationCheckHours=48. 2. Add delegation checked 24 hours ago. | Not yet due (below 48-hour interval). |
| TC-120 | Configurable correlation lookahead | P3 | 1. Set correlationLookaheadDays=7. | Correlation scanner looks 7 days ahead instead of default 3. |

---

## 9. Security & Credentials

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-121 | Encrypted vault (Fernet) | P0 | 1. Save a credential: credentials(save, 'test_key', 'secret_value'). 2. Check vault file on disk. | Value is encrypted with Fernet, not stored in plaintext. |
| TC-122 | Smart credential naming | P1 | 1. Save credentials for GitHub. | Stored as "github_token" or similar meaningful name, not "auto_1". |
| TC-123 | Auto-masking before LLM | P0 | 1. Paste a message containing a password or API key. | 3-layer detection masks the credential BEFORE it reaches the LLM context. |
| TC-124 | Outbound security filter | P0 | 1. Somehow a credential leaks into the LLM response text. | Outbound filter strips leaked credentials from all responses before they reach the user. |
| TC-125 | Credential manager, vault injection | P1 | 1. Store browser password. 2. Ask Mawa to log into a site. | {cred:name} vault injection fills credentials for browser autofill. |
| TC-126 | Tailscale-only access | P1 | 1. Try to access the web dashboard from a non-Tailscale IP. | Access is denied; only Tailscale IPs are allowed. |
| TC-127 | API, list credentials | P1 | 1. GET /api/credentials. | Returns list of stored credential names (NOT values). |
| TC-128 | Destructive action confirmation | P0 | 1. Say "delete all my goals" or "rm -rf /tmp/test". | Mawa asks for confirmation before executing the destructive action. |
| TC-129 | Destructive command detection | P1 | 1. LLM tries to run a destructive shell command. | needs_confirmation() returns a warning message; blocked until user confirms. |

---

## 10. Tools & Automation

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-130 | Background shell, run | P0 | 1. Say "run npm install in the background". | background_exec(action='run', command='npm install') starts a detached job; returns job ID immediately (e.g., bg_1). |
| TC-131 | Background shell, status | P1 | 1. Run a background job. 2. Say "check status of bg_1". | background_exec(action='status', job_id='bg_1') returns: command, status (running/exited), elapsed time, output tail. |
| TC-132 | Background shell, output | P1 | 1. Wait for job to complete. 2. Say "show output of bg_1". | Full stdout + stderr returned (truncated at 15K chars with head+tail preserved). |
| TC-133 | Background shell, list | P2 | 1. Run multiple background jobs. 2. Say "list background jobs". | Lists all jobs with status icons: [...] running, [ok] exited 0, [err] exited non-zero. |
| TC-134 | Background shell, kill | P1 | 1. Run a long job. 2. Say "kill bg_1". | Job process is killed; returncode set to -9; confirmation returned. |
| TC-135 | Background shell, max 10 jobs | P2 | 1. Try to start an 11th concurrent background job. | Returns error: "Max 10 concurrent background jobs. Kill one first." |
| TC-136 | Cron jobs | P1 | 1. Say "check my email every morning at 9am". | Cron job created with expression "0 9 * * *" and persisted. |
| TC-137 | Cron dashboard API | P1 | 1. GET /api/cron/dashboard. | Returns all jobs with: id, name, enabled, lastStatus, lastError, lastRun, nextRun, overdue flag, stats. |
| TC-138 | Schedule templates | P2 | 1. GET /api/schedule/templates. | Returns 7 pre-built templates: daily email, weekly report, standup prep, goal review, morning briefing, bill reminder, cleanup. |
| TC-139 | Event-Action rules, create | P1 | 1. POST /api/rules with rule: "when email from boss, notify immediately". | Rule saved to rules.json with event_type, keywords, action, priority. |
| TC-140 | Event-Action rules, match | P1 | 1. Create rule matching "boss" keyword. 2. Call match_rules(workspace, "email", "Message from boss about deadline"). | Returns matching action with rule_id, name, action type, priority. |
| TC-141 | Webhook event ingestion | P2 | 1. POST /api/events with HMAC-secured webhook payload. | Event ingested and processed through rules engine. |
| TC-142 | File Watcher | P2 | 1. Start file watcher on inbox directory. 2. Create a new file in inbox. | Callback triggered with event_type="created" and the new file path. |
| TC-143 | Skill acquisition, learn_skill | P2 | 1. Say "learn how to deploy to Vercel". | learn_skill tool ingests runtime knowledge for future use. |
| TC-144 | Skill marketplace | P3 | 1. Say "search for a Kubernetes skill". | Searches skills.sh marketplace and shows available skills. |
| TC-145 | MCP server management | P1 | 1. Go to Settings > MCP. 2. Add a new MCP server. | Server registered; GET /api/mcp-servers shows it; tools from the server become available. |

---

## 11. Media Generation

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-146 | Phone call, TTS mode | P1 | 1. Save Twilio credentials. 2. Say "call +12035551234 and say 'Your appointment is confirmed'". | phone_call(to='+12035551234', message='...', mode='tts') initiates call via Twilio. Returns call SID. |
| TC-147 | Phone call, conversation mode | P2 | 1. Save Twilio + Deepgram credentials. 2. Say "call +12035551234 and have a conversation about the project". | Call initiated with greeting; note about WebSocket requirement for full two-way support. |
| TC-148 | Phone call, missing credentials | P1 | 1. Do NOT save Twilio credentials. 2. Try to make a call. | Returns clear error with instructions on how to save twilio_sid, twilio_token, twilio_phone. |
| TC-149 | Phone call, disabled | P2 | 1. Set phoneCallEnabled=false. 2. Try to make a call. | Returns "Phone calls are disabled. Enable in settings." |
| TC-150 | Phone call, number normalization | P2 | 1. Call with number "2035551234" (no + prefix). | Auto-prepends "+1" to make E.164 format. |
| TC-151 | Phone call, configurable voice | P2 | 1. Set phoneCallDefaultVoice to "Polly.Joanna". 2. Make a call. | TwiML uses the configured voice, not default "alice". |

---

## 12. Voice Providers

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-152 | Deepgram provider, validation | P0 | 1. Save DEEPGRAM_API_KEY. 2. Call validate_provider("deepgram"). | Returns {ok: true, message: "Deepgram connected"} after checking /v1/projects endpoint. |
| TC-153 | Deepgram provider, voices | P1 | 1. Call get_provider_voices("deepgram"). | Returns 6 voices: Luna, Asteria, Stella, Athena (female), Orion, Arcas (male). All aura-2 models. |
| TC-154 | MiMo-Audio provider, emotions | P1 | 1. Set voiceTtsProvider to "mimo-audio". 2. Configure mimoAudioEndpoint. | Provider supports emotional voices: happy, sad, excited, whisper, laughing. |
| TC-155 | MiMo-Audio provider, voice cloning | P2 | 1. Save a voice sample WAV file. 2. Select "custom" voice. | Voice clone used for TTS output. |
| TC-156 | Coqui XTTS, 17 languages | P2 | 1. Set voiceTtsProvider to "coqui-xtts". 2. Set language to "es". | Spanish TTS generated via Coqui XTTS v2 on Modal. |
| TC-157 | Meta MMS-TTS, Bengali | P2 | 1. Set voiceTtsProvider to "mms-tts". 2. Set mmsTtsModel to "facebook/mms-tts-ben". | Bengali TTS generated via HuggingFace MMS-TTS. |
| TC-158 | Provider validation, missing credentials | P1 | 1. Remove DEEPGRAM_API_KEY. 2. Call validate_provider("deepgram"). | Returns {ok: false, missing_credentials: [...], instructions: "Save API key to vault..."}. |
| TC-159 | Provider validation, Modal endpoint | P2 | 1. Set mimoAudioEndpoint to an unreachable URL. 2. Call validate_provider("mimo-audio"). | Returns {ok: false, message: "Cannot reach endpoint: ..."}. |
| TC-160 | Voice sample management, save | P2 | 1. Call save_voice_sample(workspace, audio_b64, "my_voice"). | WAV file saved to workspace/voice_samples/my_voice.wav. |
| TC-161 | Voice sample management, list | P3 | 1. Save multiple voice samples. 2. Call get_voice_samples(). | Returns list of samples with name, path, size. |
| TC-162 | Voice sample management, delete | P3 | 1. Call delete_voice_sample(workspace, "my_voice"). | File removed; returns True. |
| TC-163 | Get all providers summary | P2 | 1. Call get_all_providers(). | Returns 4 providers (deepgram, mimo-audio, coqui-xtts, mms-tts) with stt/tts/clone/emotions/languages flags. |

---

## 13. Maintenance & Health

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-164 | History auto-archive | P1 | 1. Let HISTORY.md grow beyond 100KB. 2. Trigger heartbeat maintenance. | Old months split into memory/history/YYYY-MM.md files; current month retained in HISTORY.md. |
| TC-165 | Session auto-cleanup | P1 | 1. Have sessions older than 7 days (with at least 5 newer ones). 2. Trigger maintenance. | Old sessions deleted; minimum 5 most recent always protected. |
| TC-166 | Contact auto-extract | P2 | 1. Have "Alice, alice@example.com" in LONG_TERM.md. 2. Trigger maintenance. | Contact extracted and saved to contacts.json with name, email, source="memory". |
| TC-167 | File auto-cleanup | P2 | 1. Have files >30 days old in workspace/generated/. 2. Call auto_cleanup(). | Old files deleted; empty directories removed; returns bytes_freed and total_size_mb. |
| TC-168 | Health dashboard API | P0 | 1. GET /api/health/dashboard. | Returns comprehensive health: disk (free_gb, used_pct), workspace_mb, sessions (count, size), memory files (KB each), cron stats, failing tools, budget status, retry queue length. |
| TC-169 | Session search | P1 | 1. GET /api/sessions/search?q=deployment. | Returns matching snippets across all session .jsonl files with session name, role, snippet, timestamp, line number. |
| TC-170 | Anomaly detection | P1 | 1. Use 5x more tokens today than 7-day average. 2. GET /api/anomalies. | Returns anomaly with metric="tokens", today value, avg_7d, ratio, severity="high" (>5x) or "medium" (>3x). |
| TC-171 | Tool favorites | P2 | 1. Use various tools over many sessions. 2. GET /api/tools/favorites. | Returns top 10 tools ranked by usage with: name, total calls, success_rate%, avg_ms. |
| TC-172 | Habit tracking, create | P2 | 1. Call save_habit(workspace, {"name": "Drink water", "interval_hours": 2}). | Habit saved to habits.json with enabled=true, last_reminded=0. |
| TC-173 | Habit tracking, due | P2 | 1. Create habit with interval_hours=2. 2. Wait 2+ hours. 3. Call get_due_habits(). | Returns the habit as due for reminder. |
| TC-174 | Undo/rollback journal | P3 | 1. Perform several actions. 2. Call get_undo_history(). | Returns last 10 actions in reverse order with type, details, rollback command. Max 20 stored. |
| TC-175 | Run all maintenance (heartbeat) | P1 | 1. Call run_maintenance(workspace). | Runs: history archive, session cleanup, contact extraction. Returns results for each. |
| TC-176 | Batch inbox, count | P2 | 1. Have files in inbox. 2. GET /api/inbox/batch?action=count. | Returns total file count and count by file extension. |
| TC-177 | Batch inbox, categorize | P2 | 1. GET /api/inbox/batch?action=categorize. | Returns files grouped into: documents, images, data, code, other. |
| TC-178 | Batch inbox, delete old | P2 | 1. GET /api/inbox/batch?action=delete_old. | Files >30 days deleted; returns deleted count and remaining count. |

---

## 14. Discord & Channels

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-179 | Telegram channel | P1 | 1. Configure Telegram bot token. 2. Send message to bot. | Mawa receives, processes, and replies in Telegram. |
| TC-180 | Discord text channel | P1 | 1. Configure Discord bot. 2. Mention @Mawa in a text channel. | Mawa receives and responds in the Discord channel. |
| TC-181 | Discord voice channel | P1 | 1. Join a Discord voice channel where Mawa is connected. 2. Speak. | Audio transcribed, processed, TTS response played back via opus codec. |
| TC-182 | Slack channel | P2 | 1. Configure Slack integration. 2. Message Mawa in Slack. | Mawa receives and responds in Slack. |
| TC-183 | Web dashboard channel | P0 | 1. Open dashboard URL. 2. Send a message. | Full chat+voice experience over WebSocket. |
| TC-184 | Multi-channel session continuity | P2 | 1. Start conversation on web. 2. Continue on Telegram. | Context carries over via shared memory, Mawa remembers the earlier conversation. |
| TC-185 | Email channel | P3 | 1. Configure email ingestion. 2. Send email to Mawa's address. | Email processed and response sent back. |
| TC-186 | Matrix channel | P3 | 1. Configure Matrix bridge. 2. Send message. | Mawa responds in Matrix room. |

---

## 15. Pipeline Optimizations

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-187 | Intent-based tool filtering | P1 | 1. Say "check my email". | Only email-related tools + always-on tools sent to LLM (not all 50+ tools). Savings logged if >3 tools filtered. |
| TC-188 | Tool filtering, no intent fallback | P1 | 1. Send an ambiguous message with no clear intent keywords. | ALL tools are sent (safety fallback), never blocks functionality. |
| TC-189 | Tool filtering, MCP tool inclusion | P2 | 1. Say "check gmail". | MCP tools with "gmail" in the name are included via keyword extraction from tool names. |
| TC-190 | Response length, voice mode short | P0 | 1. Be in voice mode (discord_voice or web_voice). 2. Ask any question. | Hint injected: "Keep response SHORT (1-3 sentences), voice mode." |
| TC-191 | Response length, yes/no question | P2 | 1. Ask "Is Python faster than JavaScript?" | Hint injected: "This is a yes/no question. Answer directly, then brief explanation." |
| TC-192 | Response length, list request | P2 | 1. Ask "list all the countries in Europe". | Hint injected: "User wants a list. Use bullet points." |
| TC-193 | Response length, comparison | P2 | 1. Ask "compare React vs Vue". | Hint injected: "Use a comparison table." |
| TC-194 | History compression | P1 | 1. Have a conversation with 20+ turns. | Old turns (beyond last 6) are compressed to one-line summaries; tool messages from old turns are dropped entirely. |
| TC-195 | Follow-up chaining | P2 | 1. Say "check my email and then reply to the important ones". | Detected as chained request; split into ["check my email", "reply to the important ones"]. |
| TC-196 | Parallel prefetch detection | P2 | 1. Say "show me my email, calendar, and weather". | Detects 3 independent sources (email, calendar, weather); injects hint to call tools in PARALLEL. |
| TC-197 | Output format hints, table | P2 | 1. Say "show me a table of pricing plans". | Hint injected: "Use a markdown table." |
| TC-198 | Output format hints, brief | P2 | 1. Say "give me a quick summary". | Hint injected: "Keep it very brief, 1-2 sentences max." |
| TC-199 | Output format hints, detailed | P3 | 1. Say "give me a comprehensive analysis". | Hint injected: "Provide a detailed, thorough response." |

---

## Pre-LLM Interceptors (Zero-Token Answers)

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-200 | Math interceptor, percentage | P0 | 1. Say "what's 15% of $347". | Returns "$52.05" instantly, zero LLM tokens used. |
| TC-201 | Math interceptor, arithmetic | P1 | 1. Say "calculate 1024 * 768". | Returns "= 786,432" via safe AST eval, no LLM call. |
| TC-202 | Math interceptor, complex | P2 | 1. Say "what is sqrt(144) + 10". | Returns "= 22" using safe math functions (sqrt, ceil, floor, sin, cos, log, etc.). |
| TC-203 | Timezone resolver | P1 | 1. Say "what's 3pm EST in Tokyo". | Returns "3:00PM EST = 05:00 AM JST", zero tokens. Supports 20+ timezone aliases. |
| TC-204 | Regex builder, template | P1 | 1. Say "regex for email addresses". | Returns pattern from template, zero tokens. |
| TC-205 | Regex builder, custom | P2 | 1. Say "regex for words starting with 'test'". | Constructs pattern auto-built from description keywords. |
| TC-206 | Greeting interceptor | P1 | 1. Say "hello" or "good morning". | Returns time-aware greeting (e.g., "Good morning Farhan! You have 3 pending goals."), zero tokens. |
| TC-207 | Greeting with goals context | P2 | 1. Have pending goals in GOALS.md. 2. Say "hey". | Greeting includes goal count: "You have 5 pending goals. How can I help?" |
| TC-208 | Greeting, not triggered for questions | P1 | 1. Say "Hey, what's the bitcoin price?" | NOT intercepted as greeting because it contains a question mark and >5 words. Falls through to LLM. |

---

## Smart Response Features

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-209 | Response caching | P1 | 1. Ask "what is the capital of France?" 2. Ask the exact same question within 5 minutes. | Second question returns cached response instantly, zero LLM tokens. Cache hit logged. |
| TC-210 | Response caching, TTL expiry | P2 | 1. Ask a question. 2. Wait >5 minutes. 3. Ask again. | Cache expired; new LLM call made. |
| TC-211 | Response caching, no time-sensitive | P2 | 1. Ask "what's the weather today?" twice. | NOT cached because "today" is in the query (time-sensitive words excluded). |
| TC-212 | Priority detection, URGENT keywords | P1 | 1. Send "URGENT: the server is down!". | Priority detected as "high" based on keyword "urgent". |
| TC-213 | Priority detection, ALL CAPS | P2 | 1. Send "WHY IS THIS STILL NOT WORKING". | Priority detected as "high" due to >50% caps ratio. |
| TC-214 | Entity extraction | P1 | 1. Send "email john@example.com about the $5,000 invoice due 2026-04-01 at 3:30pm". | Entities extracted: email (john@example.com), money ($5,000), date (2026-04-01), time (3:30pm). |
| TC-215 | Loop detector | P1 | 1. Agent produces the same response 2+ times within 5 turns. | Loop detected; system message injected telling agent to try a completely different approach. |
| TC-216 | Smart defaults, email recipient | P2 | 1. Have a primary email in LONG_TERM.md. 2. Say "send email" with no recipient. | suggested_recipient auto-filled from memory. |
| TC-217 | Error translation | P1 | 1. A tool returns "ConnectionRefusedError: ...". | Appended: "In plain English: The service isn't running or isn't accepting connections." |
| TC-218 | Error translation, 429 | P2 | 1. Tool returns "429 Too Many Requests". | Translated: "Rate limited, too many requests. Wait a moment." |
| TC-219 | Link enrichment | P2 | 1. Send a message containing "https://github.com/user/repo". | Link detected and enriched: "GitHub repository" with domain hint. |
| TC-220 | Frustration detection | P1 | 1. Send "THIS DOESN'T WORK! FIX IT". | Frustration detected (score >= 0.5). Empathetic preamble injected: "I understand this is really frustrating..." |
| TC-221 | Frustration escalation | P2 | 1. Send 3 frustrated messages within 5 minutes. | Score escalated by +0.2 due to repeated frustration in session. |
| TC-222 | Message dedup | P1 | 1. Send the exact same message twice within 30 seconds (e.g., network glitch). | Second message silently dropped; only one LLM call made. |
| TC-223 | Tool result merging | P2 | 1. Two tools return overlapping content (same lines). | Duplicate lines removed across tool results; merged output with [tool_name] headers. |
| TC-224 | Semantic truncation | P1 | 1. Tool returns a 50K char result with a 10K char budget. | Truncated at paragraph boundary (preferred), sentence boundary, or line boundary, never mid-word. Omission count shown. |
| TC-225 | Auto-retry with context | P2 | 1. Tool fails. 2. Agent retries. | Retry includes injected context: "[Previous attempts failed because: tool_name: error_message]". |
| TC-226 | Session metrics | P2 | 1. Have a multi-turn conversation. 2. Call get_session_health(session_key). | Returns: turns, tokens, avg_tokens_per_turn, tools_used, errors, error_rate%, avg_response_ms, session_duration_min, status (healthy/degraded). |

---

## Code-Level Features (Zero LLM)

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-227 | Hard budget enforcement | P0 | 1. Set budget_daily_limit=$1, budget_enforce=true. 2. Use $1.01 worth of tokens. 3. Send another message. | LLM call BLOCKED. Returns "Daily budget exceeded ($1.01 / $1.00). Use /budget to adjust." |
| TC-228 | Budget, warn but not block | P1 | 1. Set budget_enforce=false. 2. Exceed daily limit. | Warning shown but LLM call proceeds. |
| TC-229 | Auto-model downgrade | P1 | 1. Set budget_auto_switch_model="claude-haiku-3-5". 2. Spend >80% of daily budget. | Model auto-switches from claude-sonnet to claude-haiku for subsequent turns. |
| TC-230 | Auto-model downgrade, tiers | P2 | 1. Use gpt-4o with budget at 85%. | Downgrades to gpt-4o-mini. Mapping: claude-opus->sonnet, sonnet->haiku, gpt-4o->gpt-4o-mini, gemini-pro->flash. |
| TC-231 | Smart retry queue, enqueue | P2 | 1. Outbound message fails (network down). | Message saved to workspace/retry_queue/ as JSON with channel, chat_id, content, retries=0. |
| TC-232 | Smart retry queue, max retries | P2 | 1. Message fails 3 times (MAX_RETRIES). | Entry deleted from queue after exhausting retries. |
| TC-233 | Session tags | P2 | 1. POST /api/sessions/tags with session and tags. | Tags saved; GET returns all sessions with tags; filter by tag returns matching sessions. |
| TC-234 | Batch inbox processing | P2 | 1. GET /api/inbox/batch?action=list. | Returns list of inbox files (max 50) with path, name, size, modified, age_days, extension. |

---

## Multi-Language Support

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-235 | Language detection, Bangla | P1 | 1. Send a message in Bangla script. | Language detected as "bangla"; response in Bangla. |
| TC-236 | Language detection, Hindi | P1 | 1. Send a message in Hindi/Devanagari. | Language detected as "hindi"; response in Hindi. |
| TC-237 | Language detection, Spanish | P2 | 1. Send "Hola, como esta?" | Language detected as "spanish"; response in Spanish. |
| TC-238 | Language detection, Arabic script | P2 | 1. Send text in Arabic script. | Detected via Unicode range even if no marker words match. |
| TC-239 | Language detection, English default | P2 | 1. Send "Hello, how are you?" | Detected as "english" (default when no other language markers found). |

---

## Claude-Level Capabilities

| ID | Feature | Priority | Steps | Expected Result |
|----|---------|----------|-------|-----------------|
| TC-240 | Task decomposer, detection | P1 | 1. Say "check my email and then reply to the important ones". | is_multi_step() returns True; decompose_task() splits into ordered steps. |
| TC-241 | Task decomposer, numbered | P2 | 1. Say "1) check email 2) summarize news 3) update goals". | Split into 3 steps from numbered format. |
| TC-242 | Parallel dispatch, classification | P1 | 1. LLM requests web_search + memory_search + inbox. | All classified as parallel-safe; none go to sequential batch. |
| TC-243 | Parallel dispatch, sequential | P1 | 1. LLM requests write_file + browser. | Both classified as sequential (write/state-changing tools). |
| TC-244 | Source citation tracker | P2 | 1. Use web_search + memory_search to answer a question. | Response includes "Sources:" section with web URLs and memory references. |
| TC-245 | Smart formatter, list of dicts | P2 | 1. Pass a list of dicts to smart_format(). | Auto-formatted as a markdown table with headers. |
| TC-246 | Smart formatter, dict | P2 | 1. Pass a dict to smart_format(). | Formatted as key: value pairs with bold keys. |
| TC-247 | State snapshots, take | P2 | 1. POST /api/snapshot with name="before_migration". | MD5 hashes captured for all memory and json files under 100KB. |
| TC-248 | State snapshots, diff | P2 | 1. Make some changes. 2. GET /api/snapshot/diff?name=before_migration. | Returns changed, added, removed file lists with total_changes count. |
| TC-249 | Paste pipeline, URL | P2 | 1. Paste a URL "https://github.com/user/repo". | Detected as URL paste; suggests "fetch and summarize github.com". |
| TC-250 | Paste pipeline, JSON | P2 | 1. Paste a JSON array. | Detected as json_array; suggests "format N items as table". |
| TC-251 | Paste pipeline, CSV | P2 | 1. Paste multi-line comma-separated data. | Detected as CSV; suggests "parse N rows of CSV data". |
| TC-252 | Paste pipeline, email | P3 | 1. Paste text with From/To/Subject headers. | Detected as email paste; suggests "parse and summarize this email". |
| TC-253 | Strategy rotator | P1 | 1. web_search fails. | Suggests web_fetch as alternative. Failure count tracked; alternatives rotate on repeated failures. |
| TC-254 | Strategy rotator, fallback chain | P2 | 1. Long-running command times out in exec. | Suggests background_exec as alternative. |
| TC-255 | Strategy rotator, reset on success | P2 | 1. Tool fails once. 2. Succeeds next time. | Failure counter reset for that tool. |
| TC-256 | Research pipeline | P2 | 1. Say "research the latest AI regulations". | Detected as research query; plan built: search then fetch top 3 results then extract. |

---

## Summary Statistics

| Category | Test Cases | P0 | P1 | P2 | P3 |
|----------|-----------|----|----|----|----|
| Voice Mode and TTS | 9 | 2 | 5 | 2 | 0 |
| Chat and Messaging | 15 | 1 | 5 | 7 | 2 |
| Image and File Attachments | 11 | 1 | 4 | 5 | 1 |
| Intelligence and Context | 14 | 2 | 7 | 5 | 0 |
| Memory System | 18 | 2 | 9 | 7 | 0 |
| Settings and Configuration | 15 | 2 | 6 | 5 | 2 |
| Notifications and Proactive | 10 | 0 | 4 | 6 | 0 |
| Jarvis Intelligence | 28 | 1 | 13 | 11 | 3 |
| Security and Credentials | 9 | 3 | 4 | 2 | 0 |
| Tools and Automation | 16 | 1 | 7 | 5 | 3 |
| Media Generation | 6 | 0 | 2 | 4 | 0 |
| Voice Providers | 12 | 1 | 3 | 5 | 3 |
| Maintenance and Health | 15 | 1 | 5 | 8 | 1 |
| Discord and Channels | 8 | 1 | 3 | 2 | 2 |
| Pipeline Optimizations | 13 | 1 | 3 | 8 | 1 |
| Pre-LLM Interceptors | 9 | 1 | 4 | 4 | 0 |
| Smart Response Features | 18 | 0 | 6 | 12 | 0 |
| Code-Level Features | 8 | 1 | 3 | 4 | 0 |
| Multi-Language Support | 5 | 0 | 2 | 3 | 0 |
| Claude-Level Capabilities | 17 | 0 | 4 | 11 | 2 |
| **TOTAL** | **256** | **21** | **99** | **118** | **18** |
