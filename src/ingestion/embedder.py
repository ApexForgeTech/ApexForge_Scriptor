"""
src/ingestion/embedder.py — Embeds vault chunks and stores them in ChromaDB.

Usage (CLI):
    python -m src.ingestion.embedder
    python -m src.ingestion.embedder --vault /path/to/vault --reset

Flow:
  1. Parse all .md notes  (parser.py)
  2. Chunk them            (chunker.py)
  3. Generate embeddings via nomic-embed-text (Ollama)
  4. Upsert into ChromaDB (only re-embed files that changed)
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import Optional

import httpx
import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

# Allow running as script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from src.ingestion.parser import parse_vault, VaultDocument
from src.ingestion.chunker import chunk_documents, DocumentChunk

console = Console()

COLLECTION_NAME = "obsidian_vault"
# ChromaDB metadata key used to track file modification time
_META_MTIME = "_mtime"
_META_HASH  = "_content_hash"


# ── Embedding via Ollama ──────────────────────────────────────────────────────

def _embed_texts(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """
    Call Ollama's /api/embed endpoint to get embeddings for a batch of texts.
    Returns a list of float vectors.
    """
    payload = {"model": model, "input": texts}
    resp = httpx.post(
        f"{base_url}/api/embed",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _content_hash(text: str) -> str:
    """MD5 hash of content — used to detect if a note changed."""
    return hashlib.md5(text.encode()).hexdigest()


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _get_existing_hashes(collection: chromadb.Collection) -> dict[str, str]:
    """Return {chunk_id: content_hash} for all currently stored chunks."""
    try:
        result = collection.get(include=["metadatas"])
        return {
            id_: meta.get(_META_HASH, "")
            for id_, meta in zip(result["ids"], result["metadatas"])
        }
    except Exception:
        return {}


def _delete_stale_chunks(collection: chromadb.Collection, current_ids: set[str]) -> int:
    """Remove vectors whose source chunks no longer exist in the vault."""
    try:
        stored_ids = set(collection.get(include=[])["ids"])
        stale_ids = list(stored_ids - current_ids)
        if stale_ids:
            collection.delete(ids=stale_ids)
        return len(stale_ids)
    except Exception as exc:
        console.print(f"[yellow]⚠ Could not clean stale index chunks: {exc}[/yellow]")
        return 0


# ── Main ingest function ──────────────────────────────────────────────────────

def ingest_vault(
    vault_path: Optional[Path] = None,
    reset: bool = False,
    batch_size: int = 32,
) -> dict:
    """
    Parse, chunk, embed, and store all vault notes in ChromaDB.

    Args:
        vault_path: Override the vault path from settings.
        reset: If True, delete and recreate the collection from scratch.
        batch_size: Number of chunks to embed per Ollama request.

    Returns:
        Summary dict with counts.
    """
    vault_path = vault_path or settings.vault_path
    console.print(Panel.fit("🧠  Obsidian Vault Ingestion", style="bold blue"))

    # ── Step 1: Parse vault ───────────────────────────────────────────────────
    docs = parse_vault(vault_path, verbose=True)
    if not docs:
        console.print("[yellow]No non-empty documents found in vault.[/yellow]")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    console.print(f"\n[cyan]Chunking {len(docs)} notes...[/cyan]")
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    console.print(f"  [green]✓ {len(chunks)} chunks created[/green]")

    # ── Step 3: ChromaDB setup ────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))

    if reset:
        console.print("\n[yellow]Resetting collection...[/yellow]")
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = _get_collection(client)
    existing_hashes = _get_existing_hashes(collection)

    # Notes may have been deleted or shortened since the last run. Remove
    # their obsolete vectors before searching or reporting the final count.
    deleted = _delete_stale_chunks(collection, {chunk.chunk_id for chunk in chunks})

    # ── Step 4: Filter — only embed what changed ───────────────────────────────
    new_chunks: list[DocumentChunk] = []
    skipped = 0

    for chunk in chunks:
        h = _content_hash(chunk.text)
        if existing_hashes.get(chunk.chunk_id) == h:
            skipped += 1
        else:
            new_chunks.append(chunk)

    console.print(f"\n[cyan]Embedding delta:[/cyan]")
    console.print(f"  ⏭  Skipped (unchanged): {skipped}")
    console.print(f"  🆕 To embed:            {len(new_chunks)}")
    console.print(f"  🗑️  Removed (stale):     {deleted}")

    if not new_chunks:
        console.print("\n[green]✓ Vault index is already up to date![/green]")
        return {
            "total_chunks": len(chunks), "embedded": 0, "skipped": skipped,
            "deleted": deleted, "collection_size": collection.count(),
        }

    # ── Step 5: Embed in batches ──────────────────────────────────────────────
    embedded_count = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(new_chunks))

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            try:
                embeddings = _embed_texts(texts, settings.embed_model, settings.ollama_base_url)
            except Exception as exc:
                console.print(f"[red]Embedding batch {i//batch_size} failed: {exc}[/red]")
                errors += len(batch)
                progress.advance(task, len(batch))
                continue

            # Build metadata — add content hash for change detection
            ids = [c.chunk_id for c in batch]
            metadatas = []
            for c in batch:
                m = c.metadata
                m[_META_HASH] = _content_hash(c.text)
                metadatas.append(m)

            try:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                embedded_count += len(batch)
            except Exception as exc:
                console.print(f"[red]ChromaDB upsert failed: {exc}[/red]")
                errors += len(batch)

            progress.advance(task, len(batch))

    # ── Summary ───────────────────────────────────────────────────────────────
    final_count = collection.count()
    table = Table(title="Ingestion Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Notes parsed", str(len(docs)))
    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Newly embedded", str(embedded_count))
    table.add_row("Skipped (cached)", str(skipped))
    table.add_row("Removed (stale)", str(deleted))
    table.add_row("Errors", str(errors) if errors else "0 ✓")
    table.add_row("Total in ChromaDB", str(final_count))
    console.print("\n")
    console.print(table)

    if errors == 0:
        console.print(Panel.fit("✅  Vault indexed successfully!", style="bold green"))
    else:
        console.print(Panel.fit(f"⚠️  Done with {errors} errors.", style="bold yellow"))

    return {
        "total_chunks": len(chunks),
        "embedded": embedded_count,
        "skipped": skipped,
        "deleted": deleted,
        "errors": errors,
        "collection_size": final_count,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Obsidian vault into ChromaDB.")
    parser.add_argument("--vault", type=Path, default=None, help="Override vault path")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the index")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    ingest_vault(vault_path=args.vault, reset=args.reset, batch_size=args.batch_size)
