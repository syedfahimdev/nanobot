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
    # "password for X is Y" / "password to X is Y" — skip service, capture the value
    re.compile(r"(?:password|pass|pw|token|key|secret)\s+(?:for|to|on|of)\s+\S+\s+(?:is|:|=)\s*(\S+)", re.I),
    # "my password is X" / "password is X" / "password: X"
    re.compile(r"(?:my |the )?(?:password|pass|pw)\s*(?:is|:|=)\s*(\S+)", re.I),
    # "token is X" / "api key: X" / "secret key X" / "api key X"
    re.compile(r"(?:token|api[- ]?key|secret[- ]?key|secret|key)\s*(?:is|:|=)?\s+(\S+)", re.I),
    # "password X" (bare) — only if X doesn't look like a preposition
    re.compile(r"\bpassword\s+(?!for|to|on|of|is|the|my)(\S+)", re.I),
    # "pass X" (bare, but not "pass through" / "pass by")
    re.compile(r"\bpass\s+(?!through|by|on|up|away|over|for|to|is|the)(\S+)", re.I),
    # "credentials user/pass"
    re.compile(r"(?:credentials?|creds?)\s+\S+[/: ]+(\S+)", re.I),
]

# Context patterns to extract the service name — ordered by specificity
_SERVICE_PATTERNS = [
    # "my outlook password" / "outlook login" / "outlook account"
    re.compile(r"(\w+)\s+(?:password|pass|login|account|token|key|secret|cred)", re.I),
    # "password for outlook" / "token for github" / "login to azure"
    re.compile(r"(?:for|to|on|into|of)\s+(\w+)", re.I),
    # "save this in outlook" / "store as github"
    re.compile(r"(?:save|store|put|add)\s+(?:\w+\s+)*(?:in|as|to|for)\s+(\w+)", re.I),
]

# URL/domain patterns for context extraction
_URL_PATTERN = re.compile(r"https?://(?:www\.)?([a-zA-Z0-9-]+)\.", re.I)
_DOMAIN_WORDS = re.compile(r"\b(gmail|outlook|github|azure|aws|openai|slack|telegram|whatsapp|stripe|discord|twitter|instagram|facebook|salesforce|notion|linear|jira|confluence|firebase|supabase|vercel|netlify|docker|npm|pypi|huggingface)\b", re.I)

# Credential type detection
_CRED_TYPE_PATTERNS = [
    (re.compile(r"\b(?:api[- ]?key|apikey)\b", re.I), "api_key"),
    (re.compile(r"\b(?:access[- ]?token|bearer)\b", re.I), "access_token"),
    (re.compile(r"\b(?:secret[- ]?key|client[- ]?secret)\b", re.I), "secret_key"),
    (re.compile(r"\btoken\b", re.I), "token"),
    (re.compile(r"\bpassword|pass|pw\b", re.I), "password"),
    (re.compile(r"\b(?:ssh|private)[- ]?key\b", re.I), "ssh_key"),
]

_auto_counter = [0]


def _extract_service_name(text: str) -> str:
    """Extract a meaningful service name from the message context.

    Uses multiple strategies in priority order:
    1. Known service names (gmail, github, etc.)
    2. URLs/domains mentioned in the text
    3. Regex context patterns (for X, X password, etc.)
    4. Falls back to 'unknown'
    """
    stop_words = {"my", "the", "is", "with", "and", "password", "pass", "login",
                  "here", "this", "that", "save", "store", "please", "just",
                  "it", "for", "to", "on", "in", "a", "an", "of", "pw",
                  "api", "secret", "key", "token", "cred", "credentials"}

    # Strategy 1: Known service names in text
    dm = _DOMAIN_WORDS.search(text)
    if dm:
        return dm.group(1).lower()

    # Strategy 2: URLs in text
    um = _URL_PATTERN.search(text)
    if um:
        domain = um.group(1).lower()
        if domain not in stop_words and len(domain) > 2:
            return domain

    # Strategy 3: Regex context patterns
    for sp in _SERVICE_PATTERNS:
        sm = sp.search(text)
        if sm:
            svc = sm.group(1).lower()
            if svc not in stop_words and len(svc) > 2:
                return svc

    return ""


def _extract_cred_type(text: str) -> str:
    """Detect what type of credential this is (password, api_key, token, etc.)."""
    for pattern, label in _CRED_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "credential"


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

            # Extract meaningful service name and credential type
            service = _extract_service_name(text)
            cred_type = _extract_cred_type(text)

            # Build a meaningful credential name
            if service:
                # e.g., "github_api_key", "outlook_password", "stripe_token"
                # Avoid duplication: if service already contains the type, skip appending
                if cred_type != "credential" and cred_type not in service:
                    cred_name = f"{service}_{cred_type}"
                else:
                    cred_name = service
            else:
                # Fallback — still more descriptive than auto_N
                _auto_counter[0] += 1
                cred_name = f"{cred_type}_{_auto_counter[0]}"

            secrets.append({"name": cred_name, "value": password, "service": service or cred_name})
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
