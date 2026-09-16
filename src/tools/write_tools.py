"""
src/tools/write_tools.py — Write tools for vault modification.

IMPORTANT: Every function here calls request_permission() before executing.
           If the user denies, NO changes are made.
           Before any write, vault_backup() is called to create a Git commit.

Available tools:
  - create_note    : Create a new .md note in the vault
  - update_note    : Append content to an existing note
  - add_link       : Add a wikilink [[Target]] to a note
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.tools.permission import request_permission


def _resolve_vault_path(vault: Path, relative_path: str) -> Path:
    """Resolve a user-provided path and reject any attempt to leave the vault."""
    vault = vault.resolve()
    resolved = (vault / relative_path).resolve()
    try:
        resolved.relative_to(vault)
    except ValueError as exc:
        raise ValueError("Path must stay inside the configured vault.") from exc
    return resolved


def _backup_before_write(action_label: str) -> bool:
    """Auto-commit vault state before any write. Returns True on success."""
    try:
        # Import here to avoid circular deps and to allow missing gitpython gracefully
        from scripts.vault_backup import backup as git_backup
        return git_backup(message=action_label)
    except Exception as exc:
        print(f"[yellow]⚠ Git backup failed (continuing anyway): {exc}[/yellow]")
        return False


# ── CreateNoteTool ────────────────────────────────────────────────────────────

def create_note(
    title: str,
    content: str,
    folder: Optional[str] = None,
    tags: Optional[list[str]] = None,
    ui_mode: bool = False,
) -> str:
    """
    Create a new Markdown note in the vault.

    Args:
        title:   Note title (becomes the filename).
        content: Markdown content of the note.
        folder:  Relative folder path inside vault (defaults to vault root).
        tags:    List of tags to add to frontmatter.
        ui_mode: If True, skips CLI prompt (permission handled by UI).

    Returns:
        Success or error message string.
    """
    vault = settings.vault_path
    try:
        target_folder = _resolve_vault_path(vault, folder or ".")
    except ValueError as exc:
        return f"❌ Invalid target folder: {exc}"
    safe_title = title.replace("/", "-").replace("\\", "-").strip()
    file_path = target_folder / f"{safe_title}.md"

    # Build frontmatter
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fm_tags = " ".join(f"#{t}" for t in (tags or []))
    frontmatter = f"---\ntitle: {title}\ncreated: {now}\ntags: [{', '.join(tags or [])}]\n---\n\n"
    full_content = frontmatter + content

    approved = request_permission(
        action="Create new note",
        target=str(file_path.relative_to(vault)),
        details=f"Title: '{title}' | Tags: {tags or []} | Length: {len(content)} chars",
        ui_mode=ui_mode,
    )

    if not approved:
        return f"❌ Cancelled — note '{title}' was NOT created."

    # Backup before writing
    _backup_before_write(f"Pre-create: {title}")

    # Write the file
    try:
        target_folder.mkdir(parents=True, exist_ok=True)
        if file_path.exists():
            return f"❌ Note already exists: {file_path.relative_to(vault)}"
        file_path.write_text(full_content, encoding="utf-8")
        return f"✅ Note created: {file_path.relative_to(vault)}"
    except Exception as exc:
        return f"❌ Failed to create note: {exc}"


# ── UpdateNoteTool ────────────────────────────────────────────────────────────

def update_note(
    title_or_path: str,
    append_content: str,
    section_heading: Optional[str] = None,
    ui_mode: bool = False,
) -> str:
    """
    Append content to an existing note.

    Args:
        title_or_path:  Note title or relative path.
        append_content: Markdown content to append.
        section_heading: If given, append under this heading (creates it if absent).
        ui_mode:        If True, skips CLI prompt.

    Returns:
        Success or error message.
    """
    vault = settings.vault_path

    # Find the note
    candidates = list(vault.rglob(f"*{title_or_path}*.md"))
    if not candidates:
        return f"❌ Note not found: '{title_or_path}'"
    if len(candidates) > 1:
        paths = "\n".join(f"  - {c.relative_to(vault)}" for c in candidates[:5])
        return f"Multiple matches — be more specific:\n{paths}"

    file_path = candidates[0]
    rel_path = str(file_path.relative_to(vault))

    approved = request_permission(
        action="Append to note",
        target=rel_path,
        details=f"Adding {len(append_content)} chars"
                + (f" under heading '{section_heading}'" if section_heading else ""),
        ui_mode=ui_mode,
    )

    if not approved:
        return f"❌ Cancelled — note '{rel_path}' was NOT modified."

    _backup_before_write(f"Pre-update: {rel_path}")

    try:
        existing = file_path.read_text(encoding="utf-8")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        if section_heading:
            addition = f"\n\n## {section_heading}\n_{timestamp}_\n\n{append_content}"
        else:
            addition = f"\n\n---\n_{timestamp} — Agent addition_\n\n{append_content}"

        file_path.write_text(existing + addition, encoding="utf-8")
        return f"✅ Note updated: {rel_path}"
    except Exception as exc:
        return f"❌ Failed to update note: {exc}"


# ── AddLinkTool ───────────────────────────────────────────────────────────────

def add_link(
    source_title: str,
    target_title: str,
    context_sentence: Optional[str] = None,
    ui_mode: bool = False,
) -> str:
    """
    Add a wikilink [[target_title]] to a source note.

    Args:
        source_title:      Note to add the link to.
        target_title:      Note being linked to.
        context_sentence:  Optional sentence to add the link within.
        ui_mode:           If True, skips CLI prompt.

    Returns:
        Success or error message.
    """
    vault = settings.vault_path
    candidates = list(vault.rglob(f"*{source_title}*.md"))

    if not candidates:
        return f"❌ Source note not found: '{source_title}'"

    file_path = candidates[0]
    rel_path = str(file_path.relative_to(vault))
    link_text = f"[[{target_title}]]"

    # Check if link already exists
    existing_content = file_path.read_text(encoding="utf-8")
    if link_text in existing_content:
        return f"ℹ️ Link {link_text} already exists in '{rel_path}'"

    approved = request_permission(
        action="Add wikilink",
        target=rel_path,
        details=f"Adding {link_text}" + (f" in context: '{context_sentence}'" if context_sentence else ""),
        ui_mode=ui_mode,
    )

    if not approved:
        return f"❌ Cancelled — no link added."

    _backup_before_write(f"Pre-link: {rel_path} → {target_title}")

    try:
        if context_sentence:
            addition = f"\n\n_{context_sentence}_ {link_text}"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            addition = f"\n\nRelated: {link_text} _(linked by AI Agent on {timestamp})_"

        file_path.write_text(existing_content + addition, encoding="utf-8")
        return f"✅ Link {link_text} added to '{rel_path}'"
    except Exception as exc:
        return f"❌ Failed to add link: {exc}"
