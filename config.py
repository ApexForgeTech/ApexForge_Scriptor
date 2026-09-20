"""
config.py — Central configuration loader.

Reads all settings from .env (or environment variables) and exposes
them as typed attributes through the `Settings` object.
All other modules import from here: `from config import settings`
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists (silently skip if not present)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


class Settings:
    """Typed, validated settings loaded from environment variables."""

    # ── Obsidian Vault ────────────────────────────────────────────────────────
    vault_path: Path = Path(os.getenv("VAULT_PATH", "~/Documents/Obsidian")).expanduser()

    # ── LLM Provider ─────────────────────────────────────────────────────────
    # Which provider to use:  ollama | openai | mistral | gemini | grok | qwen | claude
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3-vl:2b-instruct")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

    # ── External API Keys ────────────────────────────────────────────────────
    # Set the key for whichever provider you selected above.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    grok_api_key: str = os.getenv("GROK_API_KEY", "")
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))

    # ── Memory ────────────────────────────────────────────────────────────────
    memory_window: int = int(os.getenv("MEMORY_WINDOW", "10"))
    agent_memory_folder: str = os.getenv("AGENT_MEMORY_FOLDER", "Agent/Memory")

    # ── Local time ──────────────────────────────────────────────────────────
    # IANA timezone name used for date-sensitive questions such as "today".
    timezone: str = os.getenv("TIMEZONE", "Asia/Baku")

    # ── Permissions ───────────────────────────────────────────────────────────
    write_permission: str = os.getenv("WRITE_PERMISSION", "ask")  # ask | allow | deny

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = Path(os.getenv("LOG_DIR", "./logs"))

    def validate(self) -> None:
        """Check that critical settings point to real locations."""
        if not self.vault_path.exists():
            raise FileNotFoundError(
                f"VAULT_PATH does not exist: {self.vault_path}\n"
                "Please set VAULT_PATH in your .env file."
            )
        if self.write_permission not in ("ask", "allow", "deny"):
            raise ValueError(
                f"WRITE_PERMISSION must be 'ask', 'allow', or 'deny'. Got: {self.write_permission}"
            )
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(self.timezone)
        except Exception as exc:
            raise ValueError(f"TIMEZONE must be a valid IANA timezone. Got: {self.timezone}") from exc
        # Ensure output directories exist
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  llm_provider     = {self.llm_provider}\n"
            f"  vault_path       = {self.vault_path}\n"
            f"  llm_model        = {self.llm_model}\n"
            f"  embed_model      = {self.embed_model}\n"
            f"  ollama_base_url  = {self.ollama_base_url}\n"
            f"  chroma_dir       = {self.chroma_persist_dir}\n"
            f"  retrieval_top_k  = {self.retrieval_top_k}\n"
            f"  timezone         = {self.timezone}\n"
            f"  write_permission = {self.write_permission}\n"
            f")"
        )


# Single shared instance — import this everywhere.
settings = Settings()
