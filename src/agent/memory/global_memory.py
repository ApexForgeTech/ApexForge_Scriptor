"""
src/agent/memory/global_memory.py — Long-term user memory system.

Persists facts, preferences, and rules about the user across all chat sessions.
"""

import json
from pathlib import Path
from typing import Optional
from src.agent.llm_provider import get_provider

class GlobalMemory:
    def __init__(self, persist_path: str = "data/global_memory.json"):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory: list[str] = self._load()
        self._provider = get_provider()

    def _load(self) -> list[str]:
        if not self.persist_path.exists():
            return []
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("facts", [])
        except Exception:
            return []

    def _save(self) -> None:
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump({"facts": self.memory}, f, ensure_ascii=False, indent=4)

    def get_memory_string(self) -> str:
        """Return the memory formatted for the system prompt."""
        if not self.memory:
            return "No specific long-term memories recorded yet."
        return "\n".join(f"- {fact}" for fact in self.memory)
        
    def get_raw_memory(self) -> str:
        """Return the memory as a raw string for UI editing."""
        return "\n".join(self.memory)
        
    def set_raw_memory(self, memory_str: str) -> None:
        """Update memory from UI text area."""
        self.memory = [line.strip("- ") for line in memory_str.split("\n") if line.strip()]
        self._save()

    def update_from_chat(self, user_message: str) -> bool:
        """
        Check if the user message contains a request to remember or forget something.
        Uses a cheap LLM call to extract facts.
        Returns True if memory was updated.
        """
        prompt = (
            "You are a memory extractor. Analyze the user's message.\n"
            "If the user is asking you to remember a fact, preference, rule, or detail about them, "
            "extract it as a concise statement (e.g. 'User's name is Neo', 'User prefers Rust over C++').\n"
            "If they ask to forget something, extract a statement prefixed with 'DELETE: ' (e.g. 'DELETE: User's name is Neo').\n"
            "If the message does NOT contain any instructions to remember or forget, output EXACTLY 'NONE'.\n\n"
            f"User message: '{user_message}'\n\n"
            "Output only the extracted fact or 'NONE'."
        )
        
        try:
            response = self._provider.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
            result = response.strip()
            
            if result == "NONE" or not result:
                return False
                
            if result.startswith("DELETE: "):
                fact_to_delete = result[8:].strip()
                new_mem = [m for m in self.memory if fact_to_delete.lower() not in m.lower()]
                if len(new_mem) != len(self.memory):
                    self.memory = new_mem
                    self._save()
                    return True
            else:
                if result not in self.memory:
                    self.memory.append(result)
                    self._save()
                    return True
                    
        except Exception:
            pass
            
        return False
