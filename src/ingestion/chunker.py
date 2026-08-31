"""
src/ingestion/chunker.py — Smart text chunker for Obsidian notes.

Splits VaultDocuments into smaller chunks suitable for embedding.
Strategy:
  1. First split by Markdown headings (## , ### etc.) to keep sections together.
  2. If a section still exceeds CHUNK_SIZE tokens, further split by paragraph.
  3. Add overlap between chunks for context continuity.
  4. Preserve all metadata (source, title, tags, date, note_type) in each chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .parser import VaultDocument

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single embeddable chunk derived from a VaultDocument."""

    # Content
    text: str                   # The actual text to embed

    # Traceability — always link back to the source note
    source_path: str            # Relative vault path of the original note
    source_title: str
    chunk_index: int
    total_chunks: int           # Total chunks from this document

    # Inherited metadata (used for filtering in ChromaDB)
    note_type: str              # "daily" | "learning" | "general"
    tags: list[str] = field(default_factory=list)
    date: Optional[str] = None
    folder_category: str = ""
    folder_subcategory: str = ""
    heading: str = ""           # The heading this chunk falls under

    @property
    def chunk_id(self) -> str:
        """Unique ID for ChromaDB."""
        # Sanitise path to avoid ChromaDB ID issues
        safe_path = self.source_path.replace("/", "_").replace(" ", "_").replace(".", "_")
        return f"{safe_path}__chunk_{self.chunk_index}"

    @property
    def metadata(self) -> dict:
        """Flat dict for ChromaDB metadata (values must be str/int/float/bool)."""
        return {
            "source_path": self.source_path,
            "source_title": self.source_title,
            "note_type": self.note_type,
            "tags": ",".join(self.tags[:20]),   # ChromaDB wants strings
            "date": self.date or "",
            "folder_category": self.folder_category,
            "folder_subcategory": self.folder_subcategory,
            "heading": self.heading,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _approx_token_count(text: str) -> int:
    """Rough token estimate: ~4 characters per token (works for English & Azerbaijani)."""
    return len(text) // 4


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """
    Split text into (heading_label, section_text) pairs.
    The first section before any heading gets an empty heading label.
    """
    sections: list[tuple[str, str]] = []
    pos = 0
    current_heading = ""

    for match in _RE_HEADING.finditer(text):
        section_text = text[pos:match.start()].strip()
        if section_text:
            sections.append((current_heading, section_text))
        current_heading = match.group(2).strip()
        pos = match.end()

    # Last section after final heading
    remaining = text[pos:].strip()
    if remaining:
        sections.append((current_heading, remaining))

    return sections if sections else [("", text)]


def _split_by_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping paragraph-based chunks.
    Used when a heading section is too large.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_token_count(para)

        if current_tokens + para_tokens > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraph(s) up to `overlap` tokens
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current):
                t = _approx_token_count(p)
                if overlap_tokens + t > overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += t
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text]


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_document(
    doc: VaultDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunk]:
    """
    Chunk a single VaultDocument into embeddable DocumentChunk objects.

    Args:
        doc: Parsed vault note.
        chunk_size: Max tokens per chunk (approximate).
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        List of DocumentChunk objects with full metadata.
    """
    if not doc.clean_text.strip():
        return []

    raw_chunks: list[tuple[str, str]] = []  # (heading, text)

    # 1. Split by headings first
    heading_sections = _split_by_headings(doc.clean_text)

    for heading, section_text in heading_sections:
        if _approx_token_count(section_text) <= chunk_size:
            raw_chunks.append((heading, section_text))
        else:
            # 2. Section too large — further split by paragraphs
            sub_chunks = _split_by_paragraphs(section_text, chunk_size, chunk_overlap)
            for sub in sub_chunks:
                raw_chunks.append((heading, sub))

    # 3. Build DocumentChunk objects
    total = len(raw_chunks)
    result: list[DocumentChunk] = []

    for i, (heading, text) in enumerate(raw_chunks):
        if not text.strip():
            continue

        # Prepend document title + heading for richer embedding context
        context_prefix = f"Note: {doc.title}"
        if heading:
            context_prefix += f" | Section: {heading}"
        if doc.date:
            context_prefix += f" | Date: {doc.date}"
        if doc.note_type == "learning" and doc.folder_subcategory:
            context_prefix += f" | Topic: {doc.folder_subcategory}"

        enriched_text = f"{context_prefix}\n\n{text}"

        chunk = DocumentChunk(
            text=enriched_text,
            source_path=doc.relative_path,
            source_title=doc.title,
            chunk_index=i,
            total_chunks=total,
            note_type=doc.note_type,
            tags=doc.tags,
            date=doc.date,
            folder_category=doc.folder_category,
            folder_subcategory=doc.folder_subcategory,
            heading=heading,
        )
        result.append(chunk)

    return result


def chunk_documents(
    docs: list[VaultDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunk]:
    """Chunk an entire list of VaultDocuments."""
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import settings
    from .parser import parse_vault

    docs = parse_vault(settings.vault_path, verbose=True)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)

    print(f"\nTotal chunks: {len(chunks)}")
    print("\n── Sample chunk ──")
    if chunks:
        c = chunks[0]
        print(f"ID:       {c.chunk_id}")
        print(f"Source:   {c.source_path}")
        print(f"Heading:  {c.heading!r}")
        print(f"Type:     {c.note_type}")
        print(f"Text:\n{c.text[:400]}")
