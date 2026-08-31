"""
src/agent/agent.py — Unified Agent class.

The Agent is the single entry-point for all interactions.
It orchestrates:
  - Text queries (RAG chain)
  - Image analysis (vision module)
  - Write actions (tools with permission gate)
  - Memory (conversation history)

Usage:
    agent = ObsidianAgent()
    response = agent.chat("Rust-da ownership haqqında nə öyrənmişəm?")
    print(response)

    response = agent.chat_with_image("/path/to/diagram.png", "Bu diaqramı izah et")
    print(response)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.agent.rag_chain import RAGChain, AgentResponse, ConversationMemory
from src.agent.vision import analyze_image_with_rag
from src.tools.read_tools import (
    read_note,
    search_vault,
    get_daily_note,
    list_tags,
    list_notes,
)
from src.tools.write_tools import create_note, update_note, add_link


class ObsidianAgent:
    """
    The main AI agent — your Obsidian second brain assistant.

    Capabilities:
      - Answer questions from your notes (RAG)
      - Summarise daily notes
      - Explain learning materials (Rust, ASM, SOC...)
      - Analyse images (diagrams, screenshots, handwritten notes)
      - Create / update notes (with permission)
      - Connect related notes with wikilinks (with permission)
    """

    def __init__(self):
        self.memory = ConversationMemory(max_turns=settings.memory_window)
        self.rag = RAGChain(memory=self.memory)

    # ── Text chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        note_type: Optional[str] = None,
        folder_subcategory: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> AgentResponse:
        """
        Send a text message to the agent and get a full response.

        Args:
            message:            User's message or question.
            note_type:          Optionally restrict retrieval to "daily", "learning", or "general".
            folder_subcategory: Optionally restrict to a folder e.g. "RUST", "ASM", "January".
            tag:                Optionally filter by a specific tag.

        Returns:
            AgentResponse with .answer and .sources_text.
        """
        return self.rag.ask(
            query=message,
            note_type=note_type,
            folder_subcategory=folder_subcategory,
            tag=tag,
        )

    def chat_stream(
        self,
        message: str,
        note_type: Optional[str] = None,
        folder_subcategory: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Generator[str, None, AgentResponse]:
        """
        Stream the agent's response token by token.
        Yields string tokens. Returns AgentResponse when complete.
        """
        return self.rag.ask_stream(
            query=message,
            note_type=note_type,
            folder_subcategory=folder_subcategory,
            tag=tag,
        )

    # ── Vision ─────────────────────────────────────────────────────────────

    def chat_with_image(self, image_path: str, question: str) -> str:
        """
        Analyse an image with vault context from RAG.

        Args:
            image_path: Absolute path to image file.
            question:   User's question about the image.

        Returns:
            Model's analysis as a string.
        """
        return analyze_image_with_rag(image_path, question, top_k=3)

    # ── Read shortcuts ──────────────────────────────────────────────────────

    def get_today(self) -> str:
        """Get today's daily note."""
        return get_daily_note("today")

    def get_yesterday(self) -> str:
        """Get yesterday's daily note."""
        return get_daily_note("yesterday")

    def get_note(self, title: str) -> str:
        """Read a specific note by title."""
        return read_note(title)

    def search(self, query: str, **kwargs) -> str:
        """Semantic search over the vault, returns formatted context string."""
        return search_vault(query, **kwargs)

    def get_tags(self) -> str:
        """List all tags in the vault."""
        return list_tags()

    def get_notes(self, note_type: Optional[str] = None, folder: Optional[str] = None) -> str:
        """List notes filtered by type or folder."""
        return list_notes(note_type=note_type, folder=folder)

    # ── Write shortcuts (all gated by permission) ───────────────────────────

    def create(self, title: str, content: str, folder: Optional[str] = None,
               tags: Optional[list[str]] = None, ui_mode: bool = False) -> str:
        """Create a new note (asks permission first)."""
        return create_note(title, content, folder, tags, ui_mode=ui_mode)

    def update(self, title: str, content: str, section: Optional[str] = None,
               ui_mode: bool = False) -> str:
        """Append to an existing note (asks permission first)."""
        return update_note(title, content, section, ui_mode=ui_mode)

    def link(self, source: str, target: str, context: Optional[str] = None,
             ui_mode: bool = False) -> str:
        """Add a wikilink from source note to target note (asks permission first)."""
        return add_link(source, target, context, ui_mode=ui_mode)

    # ── Memory control ──────────────────────────────────────────────────────

    def clear_memory(self) -> None:
        """Reset the conversation history."""
        self.memory.clear()

    def memory_summary(self) -> str:
        """Return a summary of the current conversation."""
        turns = self.memory.to_messages()
        if not turns:
            return "No conversation history."
        lines = [f"Conversation history ({len(turns)//2} turns):"]
        for m in turns[-6:]:  # Last 3 turns
            role = "👤" if m["role"] == "user" else "🤖"
            lines.append(f"  {role} {m['content'][:100]}...")
        return "\n".join(lines)
