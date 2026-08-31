"""
scripts/test_chromadb.py — Verify ChromaDB is working with local persistence.

Run:  python scripts/test_chromadb.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from rich.console import Console
from rich.panel import Panel

from config import settings

console = Console()

TEST_COLLECTION = "test-collection-delete-me"


def test_chromadb() -> bool:
    console.print(Panel.fit("🗄️   ChromaDB Connection Test", style="bold blue"))

    # 1. Initialize persistent client
    console.print(f"\n[cyan]Initialising ChromaDB at:[/cyan] {settings.chroma_persist_dir}")
    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        console.print("  [green]✓ Client created[/green]")
    except Exception as exc:
        console.print(f"  [red]✗ Failed to create client: {exc}[/red]")
        return False

    # 2. Create a temporary test collection
    try:
        col = client.get_or_create_collection(TEST_COLLECTION)
        console.print(f"  [green]✓ Collection created: {TEST_COLLECTION}[/green]")
    except Exception as exc:
        console.print(f"  [red]✗ Failed to create collection: {exc}[/red]")
        return False

    # 3. Insert sample documents
    sample_docs = [
        "The agent reads Obsidian markdown notes.",
        "ChromaDB stores vector embeddings locally.",
        "qwen3-vl can understand both text and images.",
    ]
    sample_ids = [f"doc_{i}" for i in range(len(sample_docs))]
    # We use simple incremental fake embeddings for the test (dim=4) to avoid needing Ollama here
    fake_embeddings = [[float(i), float(i + 1), float(i + 2), float(i + 3)] for i in range(len(sample_docs))]

    try:
        col.add(documents=sample_docs, ids=sample_ids, embeddings=fake_embeddings)
        console.print(f"  [green]✓ Inserted {len(sample_docs)} documents[/green]")
    except Exception as exc:
        console.print(f"  [red]✗ Insert failed: {exc}[/red]")
        return False

    # 4. Query the collection
    try:
        start = time.perf_counter()
        results = col.query(
            query_embeddings=[[0.0, 1.0, 2.0, 3.0]],
            n_results=2,
        )
        elapsed = time.perf_counter() - start
        returned = results.get("documents", [[]])[0]
        console.print(f"  [green]✓ Query succeeded ({elapsed*1000:.0f}ms) — top result:[/green]")
        console.print(f"    [dim]{returned[0] if returned else 'N/A'}[/dim]")
    except Exception as exc:
        console.print(f"  [red]✗ Query failed: {exc}[/red]")
        return False

    # 5. Count documents
    count = col.count()
    console.print(f"  [green]✓ Collection has {count} document(s)[/green]")

    # 6. Clean up test collection
    try:
        client.delete_collection(TEST_COLLECTION)
        console.print(f"  [green]✓ Test collection cleaned up[/green]")
    except Exception as exc:
        console.print(f"  [yellow]⚠ Could not delete test collection: {exc}[/yellow]")

    # 7. List real collections
    existing = client.list_collections()
    console.print(f"\n[cyan]Existing collections:[/cyan] {[c.name for c in existing] or '(none yet)'}")

    console.print("\n" + "─" * 50)
    console.print(Panel.fit("✅  ChromaDB is working correctly!", style="bold green"))
    return True


if __name__ == "__main__":
    ok = test_chromadb()
    sys.exit(0 if ok else 1)
