"""Load secrets from ~/.nanobot/.secrets into environment variables.

Called on gateway startup to inject HMAC webhook secrets and other
sensitive values without exposing them in config.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger


def load_secrets() -> int:
    """Load secrets from ~/.nanobot/.secrets into os.environ.

    Returns the number of secrets loaded.
    """
    secrets_file = Path.home() / ".nanobot" / ".secrets"
    if not secrets_file.exists():
        return 0

    try:
        data = json.loads(secrets_file.read_text(encoding="utf-8"))
        count = 0
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, str) and key.startswith("NANOBOT_"):
                os.environ.setdefault(key, value)
                count += 1
        if count:
            logger.debug("Loaded {} secrets from {}", count, secrets_file)
        return count
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load secrets: {}", e)
        return 0
