"""
src/ingestion/parser.py — Obsidian Markdown parser.

Reads all .md files from the vault, extracts:
  - Plain text content
  - YAML frontmatter (title, date, tags, aliases)
  - Obsidian-specific syntax: wikilinks [[Note]], inline tags #tag, backlinks
  - Folder-level metadata (category, subcategory)

Returns a list of `VaultDocument` objects ready for chunking and embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional

import frontmatter as fm
from rich.console import Console

console = Console()


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VaultDocument:
    """A single parsed Obsidian note."""

    # Location
    file_path: Path
    relative_path: str          # Relative to vault root

    # Content
    raw_text: str               # Full original markdown
    clean_text: str             # Text with Obsidian syntax stripped

    # Extracted metadata
    title: str                  # Filename stem if no frontmatter title
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)   # [[Note Name]]
    date: Optional[str] = None  # ISO date string if detectable

    # Structural metadata
    folder_category: str = ""   # e.g. "LEARNING", "Months"
    folder_subcategory: str = ""  # e.g. "RUST", "January"
    note_type: str = "general"  # "daily" | "learning" | "general"

    # File stats
    size_bytes: int = 0
    modified_at: Optional[datetime] = None

    def __repr__(self) -> str:
        return (
            f"VaultDocument(title={self.title!r}, type={self.note_type!r}, "
            f"tags={len(self.tags)}, wikilinks={len(self.wikilinks)}, "
            f"size={self.size_bytes}B)"
        )


# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_RE_INLINE_TAG = re.compile(r"(?<!\w)#([A-Za-z0-9_\-/]+)")
_RE_EMBEDDED = re.compile(r"!\[\[([^\]]+)\]\]")         # ![[image.png]]
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")  # [text](url)
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_RE_INLINE_CODE = re.compile(r"`[^`]+`")
_RE_HTML = re.compile(r"<[^>]+>")
_RE_CHECKBOX = re.compile(r"- \[[ xX]\]\s*")

# Date pattern for daily notes: 2025-01-17
_RE_DATE_FILENAME = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """
    Strip Obsidian/Markdown syntax to get readable plain text.
    Preserves the semantic content for embedding.
    """
    # Remove embedded files first (before wikilinks pattern)
    text = _RE_EMBEDDED.sub("", text)
    # Convert wikilinks to plain text: [[Note Name]] → Note Name
    text = _RE_WIKILINK.sub(lambda m: m.group(1), text)
    # Convert markdown links to plain text: [text](url) → text
    text = _RE_MARKDOWN_LINK.sub(lambda m: m.group(1), text)
    # Remove code blocks (keep a placeholder so we don't merge sentences)
    text = _RE_CODE_BLOCK.sub("\n[code block]\n", text)
    # Remove inline code backticks but keep the text
    text = _RE_INLINE_CODE.sub(lambda m: m.group(0).strip("`"), text)
    # Remove HTML tags
    text = _RE_HTML.sub("", text)
    # Remove checkbox markers but keep the task text
    text = _RE_CHECKBOX.sub("", text)
    # Remove heading markers but keep text
    text = _RE_HEADING.sub("", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_tags(text: str) -> list[str]:
    """Extract all #tags from the raw text."""
    return list({tag.upper() for tag in _RE_INLINE_TAG.findall(text)})


def _extract_wikilinks(text: str) -> list[str]:
    """Extract all [[wikilink]] targets from the raw text."""
    return list({link.strip() for link in _RE_WIKILINK.findall(text)})


def _infer_note_type(relative_path: str) -> str:
    """
    Classify the note based on its path:
    - daily: lives under Months/ (date-named files)
    - learning: lives under LEARNING/
    - general: everything else
    """
    p = relative_path.lower()
    if "months" in p or "january" in p or "february" in p or "march" in p \
       or "april" in p or "may" in p or "june" in p or "july" in p \
       or "august" in p or "september" in p or "october" in p \
       or "november" in p or "december" in p:
        return "daily"
    if "learning" in p:
        return "learning"
    return "general"


def _infer_date(stem: str) -> Optional[str]:
    """If the filename looks like 2025-01-17, return it as a date string."""
    if _RE_DATE_FILENAME.match(stem):
        return stem
    return None


def _folder_parts(relative_path: str) -> tuple[str, str]:
    """
    Extract (category, subcategory) from the folder path.
    e.g. "Daily/LEARNING/LOW_LEVEL/RUST/lesson_1.md" → ("LEARNING", "RUST")
         "Daily/Months/January/2025-01-17.md" → ("Months", "January")
    """
    parts = Path(relative_path).parts
    category = parts[1] if len(parts) > 1 else ""
    subcategory = parts[2] if len(parts) > 2 else ""
    return category, subcategory


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_note(file_path: Path, vault_root: Path) -> Optional[VaultDocument]:
    """
    Parse a single .md file into a VaultDocument.
    Returns None if the file cannot be read or is trivially empty.
    """
    try:
        raw_bytes = file_path.read_bytes()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        console.print(f"[yellow]⚠ Cannot read {file_path.name}: {exc}[/yellow]")
        return None

    if len(raw_text.strip()) < 10:
        return None  # Skip nearly-empty files

    # Parse frontmatter (YAML at top of file)
    try:
        post = fm.loads(raw_text)
        fm_meta = post.metadata          # dict of frontmatter keys
        body = post.content              # text after frontmatter
    except Exception:
        fm_meta = {}
        body = raw_text

    relative_path = str(file_path.relative_to(vault_root))
    stem = file_path.stem

    # Tags: combine frontmatter tags + inline #tags
    fm_tags = fm_meta.get("tags", [])
    if isinstance(fm_tags, str):
        fm_tags = [fm_tags]
    inline_tags = _extract_tags(raw_text)
    all_tags = list({t.upper() for t in fm_tags + inline_tags})

    # Wikilinks
    wikilinks = _extract_wikilinks(raw_text)

    # Title
    title = fm_meta.get("title") or stem

    # Date
    date = fm_meta.get("date") or _infer_date(stem)
    if date and not isinstance(date, str):
        date = str(date)

    # Clean text for embedding
    clean_text = _clean_markdown(body)

    # Structural info
    category, subcategory = _folder_parts(relative_path)
    note_type = _infer_note_type(relative_path)

    stat = file_path.stat()

    return VaultDocument(
        file_path=file_path,
        relative_path=relative_path,
        raw_text=raw_text,
        clean_text=clean_text,
        title=title,
        tags=all_tags,
        wikilinks=wikilinks,
        date=date,
        folder_category=category,
        folder_subcategory=subcategory,
        note_type=note_type,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime),
    )


def parse_vault(vault_path: Path, verbose: bool = True) -> list[VaultDocument]:
    """
    Recursively parse all .md files in the vault.

    Args:
        vault_path: Absolute path to the Obsidian vault root.
        verbose: Print progress to console.

    Returns:
        List of VaultDocument objects, sorted by modified date (newest first).
    """
    if not vault_path.exists():
        raise FileNotFoundError(f"Vault not found: {vault_path}")

    md_files = list(vault_path.rglob("*.md"))

    if verbose:
        console.print(f"\n[cyan]📂 Vault:[/cyan] {vault_path}")
        console.print(f"[cyan]📄 Found {len(md_files)} markdown files[/cyan]\n")

    docs: list[VaultDocument] = []

    for f in md_files:
        doc = parse_note(f, vault_path)
        if doc:
            docs.append(doc)

    # Sort by modification date, newest first
    docs.sort(key=lambda d: d.modified_at or datetime.min, reverse=True)

    if verbose:
        daily = sum(1 for d in docs if d.note_type == "daily")
        learning = sum(1 for d in docs if d.note_type == "learning")
        general = sum(1 for d in docs if d.note_type == "general")
        all_tags = {t for d in docs for t in d.tags}

        console.print(f"[green]✓ Parsed {len(docs)} notes[/green]")
        console.print(f"  📅 Daily notes:    {daily}")
        console.print(f"  📚 Learning notes: {learning}")
        console.print(f"  📝 General notes:  {general}")
        console.print(f"  🏷️  Unique tags:    {len(all_tags)}")

    return docs


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import settings

    docs = parse_vault(settings.vault_path)

    console.print("\n[bold]── Sample Documents ──[/bold]")
    for doc in docs[:5]:
        console.print(f"\n  [yellow]{doc.title}[/yellow] ({doc.note_type})")
        console.print(f"    Tags: {doc.tags[:8]}")
        console.print(f"    Links: {doc.wikilinks}")
        console.print(f"    Preview: {doc.clean_text[:120]!r}")
