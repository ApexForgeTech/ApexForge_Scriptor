"""Keep the vault index current while Obsidian notes change.

Run with ``python -m src.ingestion.watcher``. File events are debounced so a
single Obsidian save (which can generate several filesystem events) results in
one safe incremental indexing pass. The embedder also removes stale chunks,
therefore note deletions are handled by the same pass.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from rich.console import Console
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from src.ingestion.embedder import ingest_vault

console = Console()


class VaultIndexWatcher(FileSystemEventHandler):
    """Debounce Markdown changes and run one incremental vault re-index."""

    def __init__(self, vault_path: Path, debounce_seconds: float = 2.0):
        self.vault_path = vault_path.resolve()
        self.debounce_seconds = debounce_seconds
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._index_lock = threading.Lock()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory or not event.src_path.lower().endswith(".md"):
            return
        self._schedule(Path(event.src_path))

    def _schedule(self, changed_path: Path) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._reindex, args=(changed_path,))
            self._timer.daemon = True
            self._timer.start()

    def _reindex(self, changed_path: Path) -> None:
        if not self._index_lock.acquire(blocking=False):
            return
        try:
            try:
                relative = changed_path.resolve().relative_to(self.vault_path)
            except ValueError:
                relative = changed_path
            console.print(f"[cyan]↻ Vault change detected:[/cyan] {relative}")
            ingest_vault(vault_path=self.vault_path)
        except Exception as exc:
            console.print(f"[red]✗ Automatic indexing failed: {exc}[/red]")
        finally:
            self._index_lock.release()


def watch_vault(vault_path: Path | None = None, debounce_seconds: float = 2.0) -> None:
    """Watch a vault until interrupted with Ctrl+C."""
    vault = (vault_path or settings.vault_path).resolve()
    if not vault.is_dir():
        raise FileNotFoundError(f"Vault not found: {vault}")

    handler = VaultIndexWatcher(vault, debounce_seconds)
    observer = Observer()
    observer.schedule(handler, str(vault), recursive=True)
    observer.start()
    console.print(f"[green]✓ Watching vault:[/green] {vault}")
    console.print(f"  Markdown changes re-index after {debounce_seconds:g}s. Press Ctrl+C to stop.")

    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping vault watcher...[/yellow]")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch an Obsidian vault and incrementally re-index it.")
    parser.add_argument("--vault", type=Path, default=None, help="Override vault path")
    parser.add_argument("--debounce", type=float, default=2.0, help="Seconds to wait after a change")
    args = parser.parse_args()
    watch_vault(args.vault, args.debounce)
