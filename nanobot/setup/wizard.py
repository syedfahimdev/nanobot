"""Unified setup wizard — configures nanobot + mawabot + security.

Called from `nanobot onboard` after basic config is created.
Each step is optional and can be skipped.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()


# ── Mawabot Frontend Setup ──

def setup_mawabot(config_path: Path, workspace: Path) -> None:
    """Step 3: Build and configure the mawabot web dashboard."""
    console.print("\n")
    console.print(Panel(
        "[bold]Mawabot — Web Dashboard[/bold]\n\n"
        "A full control center for Mawa:\n"
        "  • Chat with markdown rendering + generative UI cards\n"
        "  • Voice mode (Deepgram STT/TTS)\n"
        "  • Dashboard with goals, memory, activity\n"
        "  • File inbox for RAG (drag-and-drop upload)\n"
        "  • Mobile-friendly with collapsible sidebar\n\n"
        "Mawabot is bundled with nanobot and served automatically\n"
        "when the web_voice channel is enabled.",
        title="Step 3: Web Dashboard",
        border_style="cyan",
    ))

    if not Confirm.ask("Set up the web dashboard?", default=True):
        console.print("[dim]Skipped mawabot setup[/dim]")
        return

    # Find mawabot directory
    mawabot_dir = _find_mawabot()
    if not mawabot_dir:
        console.print("[yellow]⚠[/yellow] Mawabot not found. It should be at /root/mawabot or alongside the nanobot directory.")
        console.print("[dim]You can set it up manually later.[/dim]")
        return

    # Check if Node.js is available
    if not shutil.which("node"):
        console.print("[red]✗[/red] Node.js not found. Install Node.js 18+ first: https://nodejs.org/")
        return

    # Build
    dist_dir = mawabot_dir / "dist"
    needs_build = not dist_dir.exists() or not (dist_dir / "index.html").exists()

    if needs_build:
        console.print(f"[dim]Building mawabot from {mawabot_dir}...[/dim]")
        try:
            # Install dependencies if needed
            if not (mawabot_dir / "node_modules").exists():
                console.print("[dim]Installing dependencies...[/dim]")
                subprocess.run(
                    ["npm", "install"],
                    cwd=str(mawabot_dir),
                    capture_output=True, timeout=120,
                )

            # Build
            result = subprocess.run(
                ["npx", "vite", "build"],
                cwd=str(mawabot_dir),
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                console.print("[green]✓[/green] Mawabot built successfully")
            else:
                console.print(f"[red]✗[/red] Build failed: {result.stderr[:200]}")
                return
        except Exception as e:
            console.print(f"[red]✗[/red] Build failed: {e}")
            return
    else:
        console.print(f"[green]✓[/green] Mawabot already built at {dist_dir}")

    # Configure web_voice channel
    port = int(Prompt.ask("Dashboard port", default="3000"))

    _update_config(config_path, {
        "channels": {
            "web_voice": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": port,
            }
        }
    })
    console.print(f"[green]✓[/green] Web dashboard configured on port {port}")


def _find_mawabot() -> Path | None:
    """Find the mawabot directory."""
    candidates = [
        Path("/root/mawabot"),
        Path.home() / "mawabot",
        Path(__file__).parent.parent.parent / "mawabot",  # sibling of nanobot
        Path.cwd() / "mawabot",
    ]
    for p in candidates:
        if p.exists() and (p / "package.json").exists():
            return p
    return None


# ── Security Setup ──

def setup_security(config_path: Path) -> None:
    """Step 4: Security hardening."""
    console.print("\n")
    console.print(Panel(
        "[bold]Security Configuration[/bold]\n\n"
        "Recommended security defaults:\n"
        "  • Tailscale for remote access (VPN, no port forwarding)\n"
        "  • HMAC webhook secret for event verification\n"
        "  • Channel allowlist (restrict who can message)\n"
        "  • Workspace restriction (sandbox file access)\n\n"
        "These protect your personal AI from unauthorized access.",
        title="Step 4: Security",
        border_style="red",
    ))

    if not Confirm.ask("Configure security?", default=True):
        console.print("[dim]Skipped security setup[/dim]")
        return

    updates: dict = {}

    # Webhook secret
    webhook_secret = secrets.token_hex(32)
    os.environ["NANOBOT_WEBHOOK_SECRET"] = webhook_secret
    console.print(f"[green]✓[/green] Generated webhook secret: [dim]{webhook_secret[:16]}...[/dim]")
    console.print(f"[dim]Set in environment: NANOBOT_WEBHOOK_SECRET[/dim]")

    # Save to a secure file
    secrets_file = Path.home() / ".nanobot" / ".secrets"
    secrets_file.parent.mkdir(parents=True, exist_ok=True)
    existing_secrets = {}
    if secrets_file.exists():
        try:
            existing_secrets = json.loads(secrets_file.read_text())
        except Exception:
            pass
    existing_secrets["NANOBOT_WEBHOOK_SECRET"] = webhook_secret
    secrets_file.write_text(json.dumps(existing_secrets, indent=2))
    secrets_file.chmod(0o600)
    console.print(f"[green]✓[/green] Saved to {secrets_file} (chmod 600)")

    # Tailscale
    has_tailscale = shutil.which("tailscale") is not None
    if has_tailscale:
        if Confirm.ask("Enable Tailscale-only access for web dashboard?", default=True):
            updates.setdefault("channels", {}).setdefault("web_voice", {})["tailscaleOnly"] = True
            console.print("[green]✓[/green] Tailscale-only access enabled")
        else:
            console.print("[dim]Tailscale access not enforced[/dim]")
    else:
        console.print("[dim]Tailscale not installed — dashboard will be accessible on local network[/dim]")
        console.print("[dim]Install Tailscale for secure remote access: https://tailscale.com/download[/dim]")

    # Workspace restriction
    if Confirm.ask("Restrict file access to workspace directory?", default=False):
        updates["tools"] = updates.get("tools", {})
        updates["tools"]["restrictToWorkspace"] = True
        console.print("[green]✓[/green] Workspace restriction enabled")

    if updates:
        _update_config(config_path, updates)

    console.print("[green]✓[/green] Security configured")


# ── Helper ──

def _update_config(config_path: Path, updates: dict) -> None:
    """Deep-merge updates into the existing config file."""
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    def _deep_merge(base: dict, overlay: dict) -> dict:
        for k, v in overlay.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    _deep_merge(data, updates)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Main Entry Point ──

def run_setup_wizard(config_path: Path, workspace: Path) -> None:
    """Run the full setup wizard after basic onboarding."""
    console.print("\n")
    console.print(Panel(
        "[bold cyan]Mawa Setup Wizard[/bold cyan]\n\n"
        "Let's set up the full Mawa stack:\n"
        "  Step 2: Web Dashboard (mawabot frontend)\n"
        "  Step 3: Security (Tailscale, webhooks, access control)\n\n"
        "Each step is optional — press [bold]n[/bold] to skip.",
        border_style="cyan",
    ))

    if not Confirm.ask("Continue with setup wizard?", default=True):
        console.print("[dim]You can run this later with: nanobot onboard[/dim]")
        return

    setup_mawabot(config_path, workspace)
    setup_security(config_path)

    # Final summary
    console.print("\n")
    console.print(Panel(
        "[bold green]Setup Complete![/bold green]\n\n"
        "Start the gateway:\n"
        "  [cyan]nanobot gateway[/cyan]\n\n"
        "Chat via CLI:\n"
        "  [cyan]nanobot agent -m \"Hello Mawa!\"[/cyan]\n\n"
        "Web dashboard (if configured):\n"
        "  Open the URL shown when gateway starts\n\n"
        "Upload files for RAG:\n"
        "  Drop files in [cyan]~/.nanobot/workspace/inbox/work/[/cyan]\n"
        "  Or use the Inbox tab in the web dashboard",
        border_style="green",
    ))
