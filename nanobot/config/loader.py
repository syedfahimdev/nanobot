"""Configuration loading utilities. Supports both JSON and YAML config files."""

import json
from pathlib import Path

from nanobot.config.schema import Config


# Global variable to store current config path (for multi-instance support)
_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = Path(path) if path else None


def get_config_path() -> Path:
    """Get the configuration file path, preferring YAML over JSON when both exist."""
    if _current_config_path:
        return _current_config_path
    base = Path.home() / ".nanobot"
    for name in ("config.yaml", "config.yml", "config.json"):
        p = base / name
        if p.exists():
            return p
    return base / "config.json"  # default for new installs


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in (".yaml", ".yml")


def _load_raw(path: Path) -> dict:
    """Load raw config dict from a JSON or YAML file."""
    if _is_yaml(path):
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data: dict, path: Path) -> None:
    """Save config dict to a JSON or YAML file, preserving the file format."""
    if _is_yaml(path):
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file (JSON or YAML). Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            data = _load_raw(path)
            data = _migrate_config(data)
            # Resolve vault references (${vault:key} → actual value)
            try:
                from nanobot.setup.vault import resolve_vault_refs
                data = resolve_vault_refs(data)
            except Exception:
                pass  # Vault not set up yet — use raw values
            return Config.model_validate(data)
        except Exception as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file, preserving the file format (JSON or YAML).

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(by_alias=True)
    _save_raw(data, path)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data
