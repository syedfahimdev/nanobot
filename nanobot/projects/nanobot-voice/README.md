# nanobot Voice Channel Extension

## Overview
Discord voice channel support with Deepgram streaming STT/TTS. Shares session context with text chat.

## Implementation

### Files Added
- `nanobot/channels/discord_voice.py` - Voice client with Deepgram integration

### Files Modified
- `nanobot/channels/discord.py` - Added voice gateway handlers
- `nanobot/agent/loop.py` - Voice response routing

### Configuration
Add to `~/.nanobot/config.json`:
```json
{
  "channels": {
    "discord": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "deepgram_key": "YOUR_DEEPGRAM_KEY",
      "allowFrom": ["YOUR_USER_ID"]
    }
  }
}
```

### Usage
1. Invite bot to your Discord server
2. Join a voice channel
3. Type `!join VOICE_CHANNEL_ID` in text channel
4. Bot joins voice and listens/speaks

## Session Context Behavior

| Aspect | Text Chat | Voice Channel |
|--------|-----------|---------------|
| **Session key** | `discord:123456` | `discord:123456` (same) |
| **MEMORY.md** | ✅ Shared | ✅ Shared |
| **HISTORY.md** | ✅ Shared | ✅ Shared |
| **Message format** | Full text | `[Voice] transcribed...` |
| **Response** | Text → Discord | Text → Deepgram TTS → Voice |

Voice and text share the **same session**, so:
- Facts learned in voice persist to text
- Context from text chat is available in voice
- Memory consolidation works across both

## Architecture
```
User Voice → Discord Voice Gateway → Deepgram STT → Agent Loop
                                                      ↓
User Ear ← Discord Voice Gateway ← Deepgram TTS ← Response
                ↑
         Same session as text chat
```

## Token Optimization
- Streaming STT (no polling)
- Direct UDP audio (no intermediate storage)
- Shared session context (no duplication)

## Lines of Code
- discord_voice.py: ~250 lines
- discord.py additions: ~50 lines
- loop.py additions: ~20 lines
- **Total: ~320 lines**
