"""
src/agent/rag_chain.py — Core RAG (Retrieval-Augmented Generation) chain.

This is the "brain" of the agent. For each user message:
  1. Embed the query
  2. Retrieve relevant note chunks from ChromaDB
  3. Build a prompt with the retrieved context
  4. Stream the response from qwen3-vl via Ollama
  5. Return the answer + source attribution
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.retrieval.search import search, SearchResult, build_context
from src.agent.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
from src.tools.system_tools import temporal_context
from src.agent.llm_provider import get_provider


# ── Response model ────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """Full response from one RAG query."""

    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    query: str = ""

    @property
    def sources_text(self) -> str:
        if not self.sources:
            return ""
        seen = set()
        lines = ["\n\n📎 **Sources:**"]
        for s in self.sources:
            key = s.source_title
            if key not in seen:
                seen.add(key)
                path_info = f" `{s.source_path}`" if s.source_path else ""
                date_info = f" ({s.date})" if s.date else ""
                lines.append(f"  - **{s.source_title}**{date_info}{path_info}")
        return "\n".join(lines)

    @property
    def full_response(self) -> str:
        return self.answer + self.sources_text


# ── Conversation memory ───────────────────────────────────────────────────────

class ConversationMemory:
    """Simple sliding-window conversation buffer."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._turns: list[dict] = []  # [{"role": "user"|"assistant", "content": str}]

    def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})
        # Keep only the last N turns (each turn = 2 messages: user + assistant)
        if len(self._turns) > self.max_turns * 2:
            self._turns = self._turns[-(self.max_turns * 2):]

    def to_messages(self) -> list[dict]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def is_empty(self) -> bool:
        return len(self._turns) == 0


# ── RAG Chain ─────────────────────────────────────────────────────────────────

class RAGChain:
    """
    The core question-answering chain.

    Usage:
        chain = RAGChain()
        response = chain.ask("Rust-da lifetime nədir?")
        print(response.full_response)
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        self.top_k = top_k or settings.retrieval_top_k
        self.memory = memory or ConversationMemory(max_turns=settings.memory_window)
        self._provider = get_provider()

    # ── Retrieval ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        note_type: Optional[str] = None,
        folder_subcategory: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[SearchResult]:
        """Retrieve relevant note chunks for a query."""
        return search(
            query=query,
            top_k=self.top_k,
            note_type=note_type,
            folder_subcategory=folder_subcategory,
            tag=tag,
        )

    # ── Prompt building ────────────────────────────────────────────────────

    def _build_messages(self, query: str, context: str) -> list[dict]:
        """Build the full message list for the Ollama /api/chat endpoint."""
        user_content = RAG_PROMPT_TEMPLATE.format(
            context=context if context else "(No relevant notes found in vault.)",
            question=query,
        )

        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{temporal_context()}"}]
        messages.extend(self.memory.to_messages())
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── Non-streaming ask ──────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        note_type: Optional[str] = None,
        folder_subcategory: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> AgentResponse:
        """
        Ask a question and get a full response (non-streaming).

        Args:
            query: The user's question.
            note_type: Restrict retrieval to "daily", "learning", or "general".
            folder_subcategory: Restrict to a specific topic folder (e.g. "RUST").
            tag: Restrict to notes containing a specific tag.

        Returns:
            AgentResponse with answer text and source list.
        """
        sources = self.retrieve(query, note_type, folder_subcategory, tag)
        context = build_context(sources)
        messages = self._build_messages(query, context)

        stop = ["</s>", "<|im_end|>", "User question:"]
        raw_answer = self._provider.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            stop=stop,
        )

        # Remove any inline "📎 Sources:" block the model wrote itself (we add our own)
        import re
        raw_answer = re.sub(r"\n*📎\s*Sources?:.*", "", raw_answer, flags=re.DOTALL).strip()

        answer = raw_answer

        # Update memory
        self.memory.add("user", query)
        self.memory.add("assistant", answer)

        return AgentResponse(answer=answer, sources=sources, query=query)

    # ── Streaming ask ──────────────────────────────────────────────────────

    def ask_stream(
        self,
        query: str,
        note_type: Optional[str] = None,
        folder_subcategory: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Generator[str, None, AgentResponse]:
        """
        Ask a question and stream the response token by token.

        Yields token strings. Returns an AgentResponse when done.

        Usage:
            full_answer = ""
            result = chain.ask_stream("What did I study yesterday?")
            try:
                while True:
                    token = next(result)
                    print(token, end="", flush=True)
            except StopIteration as e:
                response = e.value
        """
        sources = self.retrieve(query, note_type, folder_subcategory, tag)
        context = build_context(sources)
        messages = self._build_messages(query, context)

        stop = ["</s>", "<|im_end|>", "User question:"]
        full_answer = ""

        for token in self._provider.chat_stream(
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            stop=stop,
        ):
            full_answer += token
            yield token

        # Update memory after streaming completes
        self.memory.add("user", query)
        self.memory.add("assistant", full_answer)

        return AgentResponse(answer=full_answer, sources=sources, query=query)

    def clear_memory(self) -> None:
        """Clear the conversation history."""
        self.memory.clear()


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()
    chain = RAGChain()

    test_questions = [
        "Rust-da ownership nədir? Qeydlərimdə nə var bu barədə?",
        "Mənim gündəlik vərdişlərim barədə nə bilə bilərsən?",
        "ASM x86_64 haqqında nələri öyrənmişəm?",
    ]

    for q in test_questions:
        console.print(Panel.fit(f"❓ {q}", style="cyan"))
        console.print("[dim]Thinking...[/dim]")

        response = chain.ask(q)
        console.print(Markdown(response.full_response))
        console.print()
