"""
src/tools/read_tools.py — Read-only agent tools for vault interaction.

These tools are always safe — they never modify the vault.
  - ReadNoteTool        : Read a specific note by title or path
  - SearchVaultTool     : Semantic search across the vault
  - GetDailyNoteTool    : Get the daily note for a given date
  - ListTagsTool        : List all tags and their frequency
  - ListNotesTool       : List notes by folder/type/tag
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.retrieval.search import search, SearchResult, build_context
from src.tools.system_tools import now


def _inside_vault(path: Path, vault: Path) -> bool:
    """Return whether a resolved path remains inside the configured vault."""
    try:
        path.resolve().relative_to(vault.resolve())
        return True
    except ValueError:
        return False


# ── ReadNoteTool ──────────────────────────────────────────────────────────────

def read_note(title_or_path: str) -> str:
    """
    Read the full content of a note by its title (partial match) or relative path.

    Returns the raw markdown content, or an error message if not found.
    """
    vault = settings.vault_path

    # Try exact path match first
    candidate = vault / title_or_path
    if candidate.exists() and candidate.is_file() and _inside_vault(candidate, vault):
        return candidate.read_text(encoding="utf-8")

    # Search by title (case-insensitive filename match)
    matches = list(vault.rglob(f"*{title_or_path}*.md"))
    if not matches:
        # Try without .md
        matches = list(vault.rglob(f"*{title_or_path}*"))
        matches = [m for m in matches if m.suffix == ".md"]

    if not matches:
        return f"❌ Note not found: '{title_or_path}'"

    if len(matches) == 1:
        return matches[0].read_text(encoding="utf-8")

    # Multiple matches — return list of candidates
    paths = "\n".join(f"  - {m.relative_to(vault)}" for m in matches[:10])
    return f"Multiple notes match '{title_or_path}':\n{paths}\n\nBe more specific."


# ── SearchVaultTool ───────────────────────────────────────────────────────────

def search_vault(
    query: str,
    top_k: int = 5,
    note_type: Optional[str] = None,
    folder_subcategory: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    """
    Perform a semantic search over the embedded vault.
    Returns a formatted context string ready for the LLM prompt.
    """
    results = search(
        query=query,
        top_k=top_k,
        note_type=note_type,
        folder_subcategory=folder_subcategory,
        tag=tag,
    )

    if not results:
        return f"No relevant notes found for: '{query}'"

    return build_context(results)


# ── GetDailyNoteTool ──────────────────────────────────────────────────────────

def get_daily_note(target_date: Optional[str] = None) -> str:
    """
    Get the content of a daily note.

    Args:
        target_date: Date string "YYYY-MM-DD", or "today" / "yesterday".
                     Defaults to today.

    Returns the note content or an error message.
    """
    if target_date is None or target_date.lower() == "today":
        d = now().date()
    elif target_date.lower() == "yesterday":
        d = now().date() - timedelta(days=1)
    else:
        try:
            d = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return f"❌ Invalid date format: '{target_date}'. Use YYYY-MM-DD."

    date_str = d.strftime("%Y-%m-%d")
    # Search in Months folder
    vault = settings.vault_path
    candidates = list(vault.rglob(f"{date_str}.md"))

    if not candidates:
        return f"No daily note found for {date_str}."

    content = candidates[0].read_text(encoding="utf-8")
    return f"📅 Daily note for {date_str}:\n\n{content}"


# ── ListTagsTool ──────────────────────────────────────────────────────────────

def list_tags(min_count: int = 1) -> str:
    """
    List all tags used across the vault with their frequency.

    Args:
        min_count: Only show tags used at least this many times.

    Returns a formatted string of tags and counts.
    """
    import re
    _RE_TAG = re.compile(r"(?<!\w)#([A-Za-z0-9_\-/]+)")

    vault = settings.vault_path
    tag_counts: dict[str, int] = {}

    for md_file in vault.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            for tag in _RE_TAG.findall(text):
                t = tag.upper()
                tag_counts[t] = tag_counts.get(t, 0) + 1
        except Exception:
            continue

    filtered = {t: c for t, c in tag_counts.items() if c >= min_count}
    sorted_tags = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    if not sorted_tags:
        return "No tags found in vault."

    lines = [f"🏷️  Vault Tags (total unique: {len(sorted_tags)}):\n"]
    for tag, count in sorted_tags[:50]:
        bar = "█" * min(count, 20)
        lines.append(f"  #{tag:<25} {count:>3}  {bar}")

    return "\n".join(lines)


# ── ListNotesTool ─────────────────────────────────────────────────────────────

def list_notes(
    note_type: Optional[str] = None,
    folder: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List notes in the vault, optionally filtered by type or folder.

    Args:
        note_type: "daily" | "learning" | "general"
        folder:    Folder name fragment to filter by (e.g. "RUST", "January")
        limit:     Max number of notes to return.
    """
    vault = settings.vault_path
    all_files = sorted(vault.rglob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)

    filtered = []
    for f in all_files:
        rel = str(f.relative_to(vault)).lower()

        if folder and folder.lower() not in rel:
            continue

        if note_type == "daily" and "months" not in rel:
            continue
        if note_type == "learning" and "learning" not in rel:
            continue

        filtered.append(f)
        if len(filtered) >= limit:
            break

    if not filtered:
        return "No notes found matching the filter."

    lines = [f"📂 Notes ({len(filtered)} shown):\n"]
    for f in filtered:
        rel = f.relative_to(vault)
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")
        size_kb = f.stat().st_size // 1024
        lines.append(f"  [{mtime}] {rel}  ({size_kb}KB)")

    return "\n".join(lines)
