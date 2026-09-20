from __future__ import annotations

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

from src.agent.memory.session_manager import SessionManager

import sys
import threading
import unicodedata
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
from src.tools.system_tools import current_time


def _is_direct_time_question(message: str) -> bool:
    """Identify questions answerable from the local clock, not the vault."""
    normalized = unicodedata.normalize("NFD", message.casefold())
    text = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    text = " ".join(text.split())
    exact_questions = {
        "saat neçədir", "indi saat neçədir", "saat necədir", "indi saat necədir",
        "saat necedir", "indi saat necedir",
        "bu gün ayın neçəsidir", "bu gün ayin necesidir", "bugün ayın neçəsidir",
        "bugun ayin necesidir", "bu gün hansı gündür", "bugun hansi gundur",
        "what time is it", "what is the time", "what day is it", "what is today's date",
    }
    return text.rstrip("?!.") in exact_questions


def _response_language(message: str) -> str:
    """Use English only for clear English time questions; Azerbaijani is the default."""
    return "en" if message.casefold().startswith(("what ", "what's ")) else "az"


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
        self.session_manager = SessionManager()
        self.current_session_id = self.session_manager.create_new_session()

    def set_session(self, session_id: str) -> None:
        """Switch to an existing session."""
        self.current_session_id = session_id
        history = self.session_manager.load_session(session_id)
        self.memory.clear()
        for msg in history:
            self.memory.add(msg["role"], msg["content"])

    def create_new_session(self) -> None:
        """Start a new chat session."""
        self.current_session_id = self.session_manager.create_new_session()
        self.memory.clear()

    def _save_current_session(self, first_message: Optional[str] = None) -> None:
        """Save current memory to the active session. Optionally generates a title if it's the first message."""
        history = self.memory.to_messages()
        if not history:
            return
            
        title = None
        if first_message and len(history) <= 2:
            title = self.session_manager.generate_title(first_message)
            
        self.session_manager.save_session(self.current_session_id, history, title=title)

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
        if _is_direct_time_question(message):
            answer = current_time(_response_language(message))
            self.memory.add("user", message)
            self.memory.add("assistant", answer)
            return AgentResponse(answer=answer, query=message)

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
        # Update global memory in the background so it doesn't block UI response
        def _update_memory():
            self.rag.global_memory.update_from_chat(message)
        threading.Thread(target=_update_memory, daemon=True).start()

        if _is_direct_time_question(message):
            answer = current_time(_response_language(message))
            self.memory.add("user", message)
            self.memory.add("assistant", answer)
            yield answer
            return AgentResponse(answer=answer, sources=[], query=message)

        response = yield from self.rag.ask_stream(
            query=message,
            note_type=note_type,
            folder_subcategory=folder_subcategory,
            tag=tag,
        )
        
        self._save_current_session(first_message=message)
        return response

    # ── Vision ─────────────────────────────────────────────────────────────

    def chat_with_image(self, image_path: str, question: str):
        """
        Analyse an image with vault context from RAG.

        Args:
            image_path: Absolute path to image file.
            question:   User's question about the image.

        Yields:
            Model tokens as strings (streaming).
        """
        yield from analyze_image_with_rag(image_path, question, top_k=3)

    # ── Read shortcuts ──────────────────────────────────────────────────────

    def get_today(self) -> str:
        """Get today's daily note."""
        return get_daily_note("today")

    def get_yesterday(self) -> str:
        """Get yesterday's daily note."""
        return get_daily_note("yesterday")

    def get_current_time(self, language: str = "az") -> str:
        """Return the local time, date, and weekday without calling the LLM."""
        return current_time(language)

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
