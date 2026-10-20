"""
src/retrieval/search.py — Semantic search over the embedded vault.

Provides:
  - search(query, top_k, filters) → list of SearchResult
  - SearchResult: matched chunk text + source info + relevance score
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

COLLECTION_NAME = "obsidian_vault"


# ── Fuzzy matching for character variations ───────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Normalize text to handle different character spellings.
    Handles Azerbaijani and other character variations.
    """
    # Azerbaijani character mappings
    char_map = {
        'ə': 'e', 'ş': 'sh', 'ç': 'ch', 'ğ': 'gh', 'ı': 'i', 'ö': 'o', 'ü': 'u',
        'Ə': 'E', 'Ş': 'SH', 'Ç': 'CH', 'Ğ': 'GH', 'İ': 'I', 'Ö': 'O', 'Ü': 'U',
    }
    
    # Apply character mappings
    normalized = text
    for char, replacement in char_map.items():
        normalized = normalized.replace(char, replacement)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def expand_query_variants(query: str) -> list[str]:
    """
    Generate multiple query variants to handle different spellings.
    Returns original query + normalized version.
    """
    normalized = normalize_text(query)
    
    # If normalized is different from original, return both
    if normalized.lower() != query.lower():
        return [query, normalized]
    
    return [query]


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search result from ChromaDB."""

    text: str                   # The chunk text
    source_path: str            # Relative vault path
    source_title: str           # Note title
    note_type: str              # daily | learning | general
    tags: list[str]
    date: Optional[str]
    folder_category: str
    folder_subcategory: str
    heading: str
    distance: float             # Cosine distance (lower = more similar)

    @property
    def relevance_score(self) -> float:
        """Convert cosine distance to a 0-1 relevance score."""
        return round(1.0 - self.distance, 4)

    @property
    def source_display(self) -> str:
        """Human-readable source label for UI display."""
        parts = [self.source_title]
        if self.heading:
            parts.append(self.heading)
        if self.date:
            parts.append(self.date)
        return " › ".join(parts)

    def to_context_string(self) -> str:
        """Format this result as a context block for the LLM prompt."""
        lines = [f"[Source: {self.source_display} | Type: {self.note_type}]"]
        if self.tags:
            lines.append(f"[Tags: {', '.join(self.tags[:10])}]")
        lines.append("")
        lines.append(self.text)
        return "\n".join(lines)


# ── Embedding helper ──────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    """Embed a single query string using nomic-embed-text."""
    payload = {"model": settings.embed_model, "input": [query]}
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/embed",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ── Main search function ───────────────────────────────────────────────────────

def search(
    query: str,
    top_k: Optional[int] = None,
    note_type: Optional[str] = None,       # "daily" | "learning" | "general"
    folder_category: Optional[str] = None,  # e.g. "LEARNING"
    folder_subcategory: Optional[str] = None,  # e.g. "RUST", "January"
    date_from: Optional[str] = None,        # "2025-01-01"
    date_to: Optional[str] = None,          # "2025-01-31"
    tag: Optional[str] = None,              # Filter by a single tag name
) -> list[SearchResult]:
    """
    Perform a semantic search over the embedded Obsidian vault.

    Args:
        query: Natural language question or keywords.
        top_k: Number of results to return (defaults to settings.retrieval_top_k).
        note_type: Restrict to a note type ("daily", "learning", "general").
        folder_category: Restrict to a top-level folder (e.g. "LEARNING").
        folder_subcategory: Restrict to a subfolder (e.g. "RUST", "January").
        date_from / date_to: Filter daily notes by date range.
        tag: Filter by a specific tag name.

    Returns:
        List of SearchResult objects sorted by relevance (best first).
    """
    top_k = top_k or settings.retrieval_top_k

    # Generate query variants for fuzzy matching
    query_variants = expand_query_variants(query)
    
    # Search with all variants and combine results
    all_results: list[SearchResult] = []
    
    for query_variant in query_variants:
        # Embed the query variant
        query_embedding = _embed_query(query_variant)

        # Build ChromaDB where clause
        where: dict = {}
        conditions = []

        if note_type:
            conditions.append({"note_type": {"$eq": note_type}})
        if folder_category:
            conditions.append({"folder_category": {"$eq": folder_category}})
        if folder_subcategory:
            conditions.append({"folder_subcategory": {"$eq": folder_subcategory}})

        # Date filtering (only for daily notes which have YYYY-MM-DD dates)
        if date_from:
            conditions.append({"date": {"$gte": date_from}})
        if date_to:
            conditions.append({"date": {"$lte": date_to}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if collection.count() == 0:
            return []

        # Query
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        # Parse results
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for doc_text, meta, dist in zip(docs_list, metas_list, dists_list):
            raw_tags = meta.get("tags", "")
            tags = [t for t in raw_tags.split(",") if t] if raw_tags else []

            # Post-filter by tag if requested (ChromaDB can't filter inside comma-joined string)
            if tag and tag.upper() not in tags:
                continue

            sr = SearchResult(
                text=doc_text,
                source_path=meta.get("source_path", ""),
                source_title=meta.get("source_title", ""),
                note_type=meta.get("note_type", "general"),
                tags=tags,
                date=meta.get("date") or None,
                folder_category=meta.get("folder_category", ""),
                folder_subcategory=meta.get("folder_subcategory", ""),
                heading=meta.get("heading", ""),
                distance=float(dist),
            )
            all_results.append(sr)
    
    # Deduplicate results by source_path and heading, keeping best (lowest distance)
    seen = set()
    deduplicated_results = []
    for result in all_results:
        key = (result.source_path, result.heading)
        if key not in seen:
            seen.add(key)
            deduplicated_results.append(result)
        else:
            # Keep the result with better (lower) distance
            for i, existing in enumerate(deduplicated_results):
                if (existing.source_path, existing.heading) == key:
                    if result.distance < existing.distance:
                        deduplicated_results[i] = result
                    break
    
    # Sort by relevance (lowest distance first) and return top_k
    deduplicated_results.sort(key=lambda x: x.distance)
    return deduplicated_results[:top_k]


def build_context(results: list[SearchResult], max_chars: int = 2500) -> str:
    """
    Combine search results into a single context string for the LLM prompt.
    Truncates to max_chars to avoid exceeding the model's context window.
    """
    parts: list[str] = []
    total = 0

    for i, r in enumerate(results, 1):
        block = f"--- Context {i} (relevance: {r.relevance_score:.2f}) ---\n{r.to_context_string()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    test_queries = [
        "Rust ownership and borrowing",
        "daily routine morning exercise",
        "ASM x86 assembly lessons",
    ]

    for q in test_queries:
        console.print(Panel.fit(f"🔍 Query: {q}", style="cyan"))
        results = search(q, top_k=3)
        if not results:
            console.print("[yellow]No results (is the vault indexed?)[/yellow]")
            continue
        for r in results:
            console.print(f"  [{r.relevance_score:.2f}] [yellow]{r.source_display}[/yellow]")
            console.print(f"         {r.text[:150].strip()!r}")
        console.print()
