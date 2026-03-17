# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, pkill, killall, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## restart — Process Restart

**Always use the `restart` tool** when asked to restart the bot — NEVER use `pkill`, `kill`, `killall`, or any shell command to terminate the process. The `restart` tool performs a safe in-place restart (os.execv) and sends a confirmation message after the bot comes back online.

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
