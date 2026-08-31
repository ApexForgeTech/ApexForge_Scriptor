"""
src/tools/permission.py — Human-in-the-Loop permission gate.

Every agent write action MUST pass through request_permission() before executing.
Supports two modes:
  - CLI mode (terminal): prints the action and asks [y/N]
  - UI mode (Gradio):   returns the question; the UI handles the response

The permission mode is controlled by settings.write_permission:
  "ask"   → always ask (default, recommended)
  "allow" → silently allow all writes (dangerous, for testing only)
  "deny"  → silently deny all writes (fully read-only)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

console = Console()
log = logging.getLogger(__name__)

# ── Permission log file ───────────────────────────────────────────────────────

_LOG_FILE = settings.log_dir / "permissions.log"


def _log_decision(action: str, target: str, details: str, approved: bool) -> None:
    """Append a permission decision to the log file."""
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "APPROVED" if approved else "DENIED"
        line = f"[{timestamp}] {status} | action={action!r} target={target!r} | {details}\n"
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # Never let logging crash the agent


# ── Main gate ─────────────────────────────────────────────────────────────────

def request_permission(
    action: str,
    target: str,
    details: str = "",
    ui_mode: bool = False,
) -> bool:
    """
    Request user permission before a write operation.

    Args:
        action:   Short description of the action (e.g. "Create note", "Add link").
        target:   The file or note being affected (e.g. "Daily/2025-01-17.md").
        details:  Additional context (e.g. what content will be added).
        ui_mode:  If True, skip the CLI prompt and return False (UI handles it separately).

    Returns:
        True if the action is approved, False otherwise.
    """
    mode = settings.write_permission

    # Silent allow
    if mode == "allow":
        log.warning("Write permission mode is 'allow' — auto-approving: %s → %s", action, target)
        _log_decision(action, target, details, approved=True)
        return True

    # Silent deny
    if mode == "deny":
        console.print(f"[yellow]⚠ Write denied (mode=deny): {action} → {target}[/yellow]")
        _log_decision(action, target, details, approved=False)
        return False

    # UI mode — caller handles the prompt
    if ui_mode:
        return False  # UI will call confirm_permission() separately

    # CLI interactive mode
    console.print(
        Panel(
            f"[bold]Action:[/bold]  {action}\n"
            f"[bold]Target:[/bold]  {target}\n"
            f"[bold]Details:[/bold] {details or '—'}\n\n"
            f"[dim]This change can be undone: python scripts/vault_backup.py revert[/dim]",
            title="🔐 Write Permission Required",
            border_style="yellow",
        )
    )

    approved = Confirm.ask("Approve this action?", default=False)
    _log_decision(action, target, details, approved=approved)

    if approved:
        console.print("[green]✓ Approved[/green]")
    else:
        console.print("[red]✗ Denied — no changes made[/red]")

    return approved


def confirm_permission_ui(action: str, target: str, details: str, user_response: str) -> bool:
    """
    Called from the UI layer after the user submits their response to a permission dialog.
    Returns True if the user said yes/hə/bəli/y.
    """
    positive = {"yes", "y", "hə", "bəli", "ok", "sure", "approve", "təsdiq"}
    approved = user_response.strip().lower() in positive
    _log_decision(action, target, details, approved=approved)
    return approved
