"""Secret masking hook — scrubs passwords from messages before LLM sees them.

Detects common password patterns in user messages and replaces them with
vault references. The credential tool stores the actual value securely.

Patterns detected:
  - "my password is X" / "password: X" / "pass: X"
  - "my token is X" / "api key: X"
  - Explicit: "save password outlook MyP@ss123"

When a password is detected:
  1. Extract and save to vault via credentials tool
  2. Replace in the message with {cred:auto_N} reference
  3. LLM never sees the actual password
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

# Patterns that capture credentials in user messages (order matters — first match wins)
_PASSWORD_PATTERNS = [
    # "here is the password X" / "here's the pass X"
    re.compile(r"here(?:'s| is) (?:the |my )?(?:password|pass|pw)\s+(\S+)", re.I),
    # "my password is X" / "password is X" / "password: X"
    re.compile(r"(?:my |the )?(?:password|pass|pw)\s*(?:is|:|=)\s*(\S+)", re.I),
    # "password X" (bare)
    re.compile(r"\bpassword\s+(\S+)", re.I),
    # "pass X" (bare, but not "pass through" / "pass by")
    re.compile(r"\bpass\s+(?!through|by|on|up|away|over)(\S+)", re.I),
    # "token is X" / "api key: X" / "secret: X" / "api key X"
    re.compile(r"(?:token|api[- ]?key|secret|key)\s*(?:is|:|=)?\s+(\S+)", re.I),
    # "credentials user/pass"
    re.compile(r"(?:credentials?|creds?)\s+\S+[/: ]+(\S+)", re.I),
]

# Context patterns to extract the service name
_SERVICE_PATTERNS = [
    re.compile(r"(?:for|to|on|into)\s+(\w+)", re.I),
    re.compile(r"(\w+)\s+(?:password|login|account)", re.I),
]

_auto_counter = [0]


def detect_and_mask_secrets(text: str) -> tuple[str, list[dict[str, str]]]:
    """Detect passwords in text, mask them, return (masked_text, extracted_secrets).

    Returns:
        (masked_text, [{name, value, service}])
    """
    secrets = []
    masked = text
    already_masked = set()  # Avoid double-masking same value

    for pattern in _PASSWORD_PATTERNS:
        match = pattern.search(masked)
        if match:
            password = match.group(1).strip().rstrip(".,;!?\"')")
            if len(password) < 4 or password.lower() in ("is", "the", "my", "to", "for", "it"):
                continue
            if password.startswith("{cred:"):
                continue  # Already masked
            if password in already_masked:
                continue

            # Try to detect service name
            service = "auto"
            for sp in _SERVICE_PATTERNS:
                sm = sp.search(text)
                if sm:
                    svc = sm.group(1).lower()
                    if svc not in ("my", "the", "is", "with", "and", "password", "login"):
                        service = svc
                        break

            _auto_counter[0] += 1
            cred_name = f"{service}_{_auto_counter[0]}" if service == "auto" else service

            secrets.append({"name": cred_name, "value": password, "service": service})
            already_masked.add(password)
            # Replace the password in the text
            masked = masked.replace(password, f"{{cred:{cred_name}}}")
            logger.info("Secret mask: detected credential for '{}', masked in message", cred_name)

    return masked, secrets


async def save_detected_secrets(secrets: list[dict[str, str]]) -> None:
    """Save auto-detected secrets to the vault."""
    if not secrets:
        return
    from nanobot.setup.vault import save_to_vault
    vault_entries = {}
    for s in secrets:
        vault_entries[f"cred.{s['name']}"] = s["value"]
    save_to_vault(vault_entries)
    logger.info("Secret mask: saved {} auto-detected credentials to vault", len(secrets))
