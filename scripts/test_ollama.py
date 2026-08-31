"""
scripts/test_ollama.py — Verify the Ollama server and required models are working.

Run:  python scripts/test_ollama.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from config import settings

console = Console()


def check_ollama_server() -> bool:
    """Ping the Ollama REST API."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return True
    except Exception as exc:
        console.print(f"[red]✗ Cannot reach Ollama at {settings.ollama_base_url}: {exc}[/red]")
        return False


def list_models() -> list[str]:
    """Return the names of all locally available models."""
    resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
    data = resp.json()
    return [m["name"] for m in data.get("models", [])]


def test_llm_generation() -> bool:
    """Send a minimal prompt to the LLM and check we get a response."""
    console.print(f"\n[cyan]Testing LLM: {settings.llm_model}[/cyan]")
    try:
        payload = {
            "model": settings.llm_model,
            "prompt": "Reply with exactly one sentence: confirm you are working.",
            "stream": False,
        }
        start = time.perf_counter()
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        response_text = resp.json().get("response", "").strip()
        console.print(f"  [green]✓ Response ({elapsed:.1f}s):[/green] {response_text[:200]}")
        return True
    except Exception as exc:
        console.print(f"  [red]✗ LLM test failed: {exc}[/red]")
        return False


def test_embeddings() -> bool:
    """Generate an embedding and verify its shape."""
    console.print(f"\n[cyan]Testing Embeddings: {settings.embed_model}[/cyan]")
    try:
        payload = {
            "model": settings.embed_model,
            "input": "This is a test sentence for embedding.",
        }
        start = time.perf_counter()
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/embed",
            json=payload,
            timeout=60,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings", [[]])
        dim = len(embeddings[0]) if embeddings else 0
        console.print(f"  [green]✓ Embedding vector ({elapsed:.1f}s):[/green] dimension = {dim}")
        return dim > 0
    except Exception as exc:
        console.print(f"  [red]✗ Embedding test failed: {exc}[/red]")
        return False


def main() -> None:
    console.print(Panel.fit("🤖  Ollama Connection Test", style="bold blue"))

    # 1. Server reachability
    console.print("\n[cyan]Checking Ollama server...[/cyan]")
    if not check_ollama_server():
        console.print("\n[bold red]Ollama is not running. Start it with: ollama serve[/bold red]")
        sys.exit(1)
    console.print(f"  [green]✓ Ollama is reachable at {settings.ollama_base_url}[/green]")

    # 2. List available models
    models = list_models()
    table = Table(title="Available Models", show_header=True)
    table.add_column("Model Name", style="cyan")
    table.add_column("Status", style="green")
    required = {settings.llm_model, settings.embed_model}
    for m in models:
        status = "✓ required" if m in required else "available"
        table.add_row(m, status)
    console.print(table)

    # Check required models are present
    missing = required - set(models)
    if missing:
        console.print(f"\n[bold red]Missing required models: {missing}[/bold red]")
        for m in missing:
            console.print(f"  Run: [yellow]ollama pull {m}[/yellow]")
        sys.exit(1)

    # 3. LLM generation test
    llm_ok = test_llm_generation()

    # 4. Embedding test
    embed_ok = test_embeddings()

    # Summary
    console.print("\n" + "─" * 50)
    if llm_ok and embed_ok:
        console.print(Panel.fit("✅  All tests passed! Ollama is ready.", style="bold green"))
    else:
        console.print(Panel.fit("❌  Some tests failed. Check output above.", style="bold red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
