"""Credentials tool — securely store and use passwords/secrets.

Stores credentials in the encrypted vault. When the agent needs a credential,
it retrieves the reference — the actual value is injected at the point of use
(e.g., browser fill) without passing through the LLM.

Usage:
  User: "Save my outlook password: MyP@ss123"
  Agent: credentials(action="save", name="outlook", value="MyP@ss123")
  → Saved to vault, value masked in all logs/history

  User: "Log into outlook"
  Agent: credentials(action="get", name="outlook")
  → Returns "***MASKED***" to LLM, but browser(action="fill") gets the real value
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.setup.vault import load_vault, replace_vault, save_to_vault


class CredentialsTool(Tool):
    """Securely store and retrieve credentials (passwords, tokens, secrets)."""

    @property
    def name(self) -> str:
        return "credentials"

    @property
    def description(self) -> str:
        return (
            "Store and retrieve passwords, tokens, and secrets securely. "
            "Values are encrypted in the vault — never stored in chat history or sent to the LLM. "
            "Actions: save (store a credential), get (retrieve masked reference), list (show stored names), delete (remove)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "get", "list", "delete"],
                    "description": "Action to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Credential name (e.g., 'outlook', 'salesforce', 'github_token').",
                },
                "value": {
                    "type": "string",
                    "description": "The secret value to store (for save action). This will be encrypted immediately.",
                },
                "username": {
                    "type": "string",
                    "description": "Optional username associated with this credential.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        name: str = "",
        value: str = "",
        username: str = "",
        **kwargs: Any,
    ) -> str:
        if action == "save":
            return self._save(name, value, username)
        elif action == "get":
            return self._get(name)
        elif action == "list":
            return self._list()
        elif action == "delete":
            return self._delete(name)
        return f"Unknown action: {action}"

    def _save(self, name: str, value: str, username: str) -> str:
        if not name or not value:
            return "Error: name and value required."

        key = f"cred.{name}"
        secrets = {key: value}
        if username:
            secrets[f"cred.{name}.username"] = username

        save_to_vault(secrets)
        logger.info("Credentials: saved '{}'", name)

        result = f"Credential '{name}' saved securely to vault."
        if username:
            result += f" Username: {username}"
        result += " The password is encrypted and will never appear in chat."
        return result

    def _get(self, name: str) -> str:
        if not name:
            return "Error: name required."

        vault = load_vault()
        key = f"cred.{name}"

        if key not in vault:
            return f"Credential '{name}' not found. Use credentials(action='list') to see stored names."

        username = vault.get(f"cred.{name}.username", "")
        # Return masked — the real value is available via get_credential_value()
        result = f"Credential '{name}': ***MASKED*** (encrypted in vault)"
        if username:
            result += f"\nUsername: {username}"
        result += f"\nTo use in browser: browser(action='fill', selector='input[type=password]', value='{{{{cred:{name}}}}}')"
        return result

    def _list(self) -> str:
        vault = load_vault()
        creds = set()
        for k in vault:
            if k.startswith("cred.") and not k.endswith(".username"):
                creds.add(k[5:])  # Strip "cred." prefix

        if not creds:
            return "No credentials stored. Use credentials(action='save', name='outlook', value='...')."

        lines = ["Stored credentials:"]
        for name in sorted(creds):
            username = vault.get(f"cred.{name}.username", "")
            lines.append(f"  - {name}" + (f" (user: {username})" if username else ""))
        return "\n".join(lines)

    def _delete(self, name: str) -> str:
        if not name:
            return "Error: name required."

        vault = load_vault()
        key = f"cred.{name}"
        if key not in vault:
            return f"Credential '{name}' not found."

        del vault[key]
        vault.pop(f"cred.{name}.username", None)
        replace_vault(vault)
        return f"Credential '{name}' deleted from vault."


def get_credential_value(name: str) -> str | None:
    """Get the actual credential value for use in tools (not LLM).

    Called by the browser tool when it encounters {cred:name} patterns.
    """
    vault = load_vault()
    return vault.get(f"cred.{name}")


def resolve_credential_refs(text: str) -> str:
    """Replace {cred:name} references with actual values.

    Used by browser fill to inject real passwords without LLM seeing them.
    """
    def _replace(match):
        name = match.group(1)
        value = get_credential_value(name)
        return value if value else match.group(0)

    return re.sub(r"\{cred:(\w+)\}", _replace, text)
