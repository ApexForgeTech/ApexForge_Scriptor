"""
src/agent/memory/session_manager.py — Manage multiple chat sessions and histories.
"""

import json
import uuid
import time
from pathlib import Path
from typing import Optional
from src.agent.llm_provider import get_provider

class SessionManager:
    def __init__(self, sessions_dir: str = "data/sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._provider = get_provider()

    def get_all_sessions(self) -> list[dict]:
        """Return a list of all sessions sorted by last updated."""
        sessions = []
        for file_path in self.sessions_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": file_path.stem,
                        "title": data.get("title", "New Chat"),
                        "updated_at": data.get("updated_at", 0)
                    })
            except Exception:
                continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def load_session(self, session_id: str) -> list[dict]:
        """Load conversation history for a specific session."""
        file_path = self.sessions_dir / f"{session_id}.json"
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("history", [])
        except Exception:
            return []

    def save_session(self, session_id: str, history: list[dict], title: str = None) -> None:
        """Save conversation history."""
        file_path = self.sessions_dir / f"{session_id}.json"
        
        # Preserve title if already exists and no new title provided
        if not title and file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    title = json.load(f).get("title", "New Chat")
            except Exception:
                title = "New Chat"
        elif not title:
            title = "New Chat"
            
        data = {
            "title": title,
            "updated_at": time.time(),
            "history": history
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def delete_session(self, session_id: str) -> None:
        """Delete a session file."""
        file_path = self.sessions_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()

    def create_new_session(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())

    def generate_title(self, first_message: str) -> str:
        """Generate a short title using LLM based on the first message."""
        prompt = (
            "You are a title generator. Generate a very short (3-4 words max) "
            "title for a chat session that starts with this message:\n"
            f"'{first_message}'\n\n"
            "Output ONLY the title without quotes or punctuation."
        )
        try:
            response = self._provider.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=20)
            return response.strip(' "\'')
        except Exception:
            # Fallback
            words = first_message.split()
            return " ".join(words[:4]) + ("..." if len(words) > 4 else "")
