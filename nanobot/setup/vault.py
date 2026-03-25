"""Secrets vault — encrypted storage for API keys.

Moves sensitive values out of config.json into an encrypted vault file.
Config.json stores references like "${vault:providers.kimi.apiKey}" that
are resolved at runtime from the vault.

The vault is encrypted with a machine-specific key derived from the hostname
and a user-provided passphrase (optional). Without a passphrase, it uses
Fernet symmetric encryption with a machine-derived key.

Files:
  ~/.nanobot/.vault       — encrypted secrets (Fernet)
  ~/.nanobot/.vault.key   — machine-derived key (chmod 600)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
from pathlib import Path

from loguru import logger

_VAULT_FILE = Path.home() / ".nanobot" / ".vault"
_KEY_FILE = Path.home() / ".nanobot" / ".vault.key"

# Fields in config that contain secrets
_SECRET_FIELDS = frozenset({
    "apiKey", "apikey", "api_key",
    "token", "secret", "password",
    "deepgramApiKey", "appSecret",
    "clientSecret", "botToken", "appToken",
    "claw_token", "accessToken",
})


def _derive_key(passphrase: str = "") -> bytes:
    """Derive a Fernet key from machine identity + optional passphrase."""
    machine_id = f"{platform.node()}-{os.getuid()}-nanobot"
    combined = f"{machine_id}:{passphrase}".encode()
    key_bytes = hashlib.pbkdf2_hmac("sha256", combined, b"nanobot-vault-salt", 100000)
    return base64.urlsafe_b64encode(key_bytes)


def _get_fernet():
    """Get Fernet cipher with cached key."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        return None

    if _KEY_FILE.exists():
        key = _KEY_FILE.read_bytes()
    else:
        key = _derive_key()
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_bytes(key)
        _KEY_FILE.chmod(0o600)

    return Fernet(key)


def save_to_vault(secrets: dict[str, str]) -> bool:
    """Save secrets to the encrypted vault."""
    f = _get_fernet()
    if f is None:
        # Fallback: save as chmod 600 JSON (no encryption, but protected)
        existing = load_vault()
        existing.update(secrets)
        _VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VAULT_FILE.write_text(json.dumps(existing, indent=2))
        _VAULT_FILE.chmod(0o600)
        logger.info("Vault: saved {} secrets (unencrypted, chmod 600 — install cryptography for encryption)", len(existing))
        return True

    # Merge with existing
    existing = load_vault()
    existing.update(secrets)

    encrypted = f.encrypt(json.dumps(existing).encode())
    _VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _VAULT_FILE.write_bytes(encrypted)
    _VAULT_FILE.chmod(0o600)
    logger.info("Vault: saved {} secrets (encrypted)", len(existing))
    return True


def replace_vault(secrets: dict[str, str]) -> bool:
    """Replace the entire vault contents (for delete operations)."""
    f = _get_fernet()
    if f is None:
        _VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VAULT_FILE.write_text(json.dumps(secrets, indent=2))
        _VAULT_FILE.chmod(0o600)
        return True

    encrypted = f.encrypt(json.dumps(secrets).encode())
    _VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _VAULT_FILE.write_bytes(encrypted)
    _VAULT_FILE.chmod(0o600)
    logger.info("Vault: replaced with {} secrets", len(secrets))
    return True


def load_vault() -> dict[str, str]:
    """Load secrets from the vault."""
    if not _VAULT_FILE.exists():
        return {}

    f = _get_fernet()
    if f is None:
        # Fallback: read unencrypted
        try:
            return json.loads(_VAULT_FILE.read_text())
        except Exception:
            return {}

    try:
        encrypted = _VAULT_FILE.read_bytes()
        decrypted = f.decrypt(encrypted)
        return json.loads(decrypted)
    except Exception as e:
        logger.warning("Vault: decryption failed: {}", e)
        return {}


def resolve_vault_refs(config: dict) -> dict:
    """Replace ${vault:key} references in config with actual values."""
    vault = load_vault()
    if not vault:
        return config

    def _resolve(obj, path=""):
        if isinstance(obj, str) and obj.startswith("${vault:") and obj.endswith("}"):
            key = obj[8:-1]
            return vault.get(key, obj)  # Return original ref if not found
        elif isinstance(obj, dict):
            return {k: _resolve(v, f"{path}.{k}") for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        return obj

    return _resolve(config)


def extract_secrets_from_config(config_path: Path) -> tuple[dict[str, str], dict]:
    """Extract secret fields from config, return (secrets_dict, sanitized_config).

    Secrets are extracted and replaced with ${vault:path} references.
    """
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    secrets: dict[str, str] = {}

    def _extract(obj, path=""):
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                full_path = f"{path}.{k}" if path else k
                if k in _SECRET_FIELDS and isinstance(v, str) and len(v) > 5:
                    secrets[full_path] = v
                    result[k] = f"${{vault:{full_path}}}"
                else:
                    result[k] = _extract(v, full_path)
            return result
        elif isinstance(obj, list):
            return [_extract(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        return obj

    sanitized = _extract(data)
    return secrets, sanitized


def migrate_config_to_vault(config_path: Path) -> int:
    """Move all secrets from config.json into the vault.

    Returns the number of secrets migrated.
    """
    secrets, sanitized = extract_secrets_from_config(config_path)
    if not secrets:
        return 0

    # Save secrets to vault
    save_to_vault(secrets)

    # Write sanitized config (with ${vault:...} references)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    logger.info("Vault: migrated {} secrets from config.json", len(secrets))
    return len(secrets)
