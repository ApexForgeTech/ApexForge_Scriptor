"""
scripts/vault_backup.py — Git-based vault backup & revert utility.

Usage:
  # Back up the vault (called automatically before agent writes)
  python scripts/vault_backup.py backup

  # List agent-made commits
  python scripts/vault_backup.py log

  # Revert to a previous commit interactively
  python scripts/vault_backup.py revert
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt

try:
    import git
    from git import Repo, InvalidGitRepositoryError, GitCommandError
except ImportError:
    print("gitpython is not installed. Run: pip install gitpython")
    sys.exit(1)

from config import settings

console = Console()
app = typer.Typer(help="Vault Git backup and revert utility.")

AGENT_COMMIT_PREFIX = "[AI Agent]"


def _get_repo() -> Repo:
    """Get the Git repo for the vault. Initialise if needed."""
    try:
        return Repo(settings.vault_path)
    except InvalidGitRepositoryError:
        console.print(f"[yellow]No Git repo found in vault. Initialising...[/yellow]")
        repo = Repo.init(settings.vault_path)
        # Create initial commit so we have a baseline
        repo.index.add(["*"])
        repo.index.commit("Initial commit — vault baseline before AI Agent")
        console.print(f"[green]✓ Git repo initialised at {settings.vault_path}[/green]")
        return repo


@app.command()
def backup(message: str = f"Pre-write backup") -> bool:
    """Create a Git commit to back up the current vault state."""
    try:
        repo = _get_repo()
        # Stage all changes
        repo.git.add(A=True)

        # Check if there's anything to commit
        if not repo.is_dirty(untracked_files=True):
            console.print("[dim]No vault changes detected — skipping backup commit.[/dim]")
            return True

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"{AGENT_COMMIT_PREFIX} {message} @ {timestamp}"
        repo.index.commit(commit_msg)
        console.print(f"[green]✓ Vault backed up:[/green] {commit_msg}")
        return True

    except GitCommandError as exc:
        console.print(f"[red]✗ Git backup failed: {exc}[/red]")
        return False


@app.command()
def log(n: int = 20) -> None:
    """List recent Git commits in the vault."""
    repo = _get_repo()
    commits = list(repo.iter_commits(max_count=n))

    table = Table(title=f"Last {n} Vault Commits", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Hash", style="yellow", width=8)
    table.add_column("Date", style="cyan", width=20)
    table.add_column("Message", style="white")
    table.add_column("By Agent", style="green", width=8)

    for i, c in enumerate(commits):
        is_agent = AGENT_COMMIT_PREFIX in c.message
        date_str = datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            str(i),
            c.hexsha[:7],
            date_str,
            c.message.strip()[:80],
            "✓" if is_agent else "",
        )

    console.print(table)


@app.command()
def revert() -> None:
    """Interactively revert the vault to a previous commit."""
    repo = _get_repo()
    commits = list(repo.iter_commits(max_count=30))

    console.print(Panel.fit("↩️   Vault Revert Tool", style="bold yellow"))
    console.print("\n[bold]Recent commits:[/bold]")

    for i, c in enumerate(commits):
        date_str = datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d %H:%M")
        marker = "[yellow][AI][/yellow] " if AGENT_COMMIT_PREFIX in c.message else "       "
        console.print(f"  [dim]{i:2}[/dim]  {marker} [cyan]{c.hexsha[:7]}[/cyan]  {date_str}  {c.message.strip()[:60]}")

    choice = Prompt.ask("\nEnter the [bold]number[/bold] of the commit to restore to", default="0")
    try:
        idx = int(choice)
        target = commits[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/red]")
        sys.exit(1)

    console.print(f"\n[yellow]You are about to revert the vault to:[/yellow]")
    console.print(f"  [cyan]{target.hexsha[:7]}[/cyan] — {target.message.strip()}")
    console.print("[red]This will overwrite all changes after that commit in your vault![/red]")

    if Confirm.ask("Are you sure?", default=False):
        repo.git.checkout(target.hexsha, "--", ".")
        console.print(f"[green]✓ Vault restored to commit {target.hexsha[:7]}[/green]")
        console.print("[dim]Tip: Your current state is preserved in Git history.[/dim]")
    else:
        console.print("[yellow]Revert cancelled.[/yellow]")


if __name__ == "__main__":
    app()
