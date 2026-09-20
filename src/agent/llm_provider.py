"""
src/agent/llm_provider.py — Unified LLM provider adapter.

Abstracts away the differences between AI providers behind a single
interface.  Every provider exposes two methods:

    chat(messages, **kw)        → full response string
    chat_stream(messages, **kw) → Generator[str]   (token-by-token)

Supported providers
───────────────────
  ollama   — Local Ollama instance  (default)
  openai   — OpenAI  /v1/chat/completions
  mistral  — Mistral AI  /v1/chat/completions
  gemini   — Google Gemini  (OpenAI-compatible gateway)
  grok     — xAI Grok  /v1/chat/completions
  qwen     — Alibaba DashScope  (OpenAI-compatible gateway)
  claude   — Anthropic Messages API  /v1/messages

All external calls are made with ``httpx`` — no extra SDK dependencies.
Provider selection is controlled by the ``LLM_PROVIDER`` env-var in
``.env`` and surfaced through ``config.settings``.
"""

from __future__ import annotations

import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator, Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Base class
# ═══════════════════════════════════════════════════════════════════════════════

class BaseLLMProvider(ABC):
    """Abstract base for every LLM provider."""

    provider_name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Send messages and return the full response as a string."""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Send messages and yield tokens one by one."""
        ...

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports image input."""
        ...

    def vision_chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream a vision (multimodal) request.  Falls back to text-only."""
        yield from self.chat_stream(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Ollama provider  (local)
# ═══════════════════════════════════════════════════════════════════════════════

class OllamaProvider(BaseLLMProvider):
    """Ollama REST API — ``/api/chat``."""

    provider_name = "ollama"

    def __init__(self) -> None:
        self._base = settings.ollama_base_url.rstrip("/")
        self._url = f"{self._base}/api/chat"

    # -- helpers -----------------------------------------------------------

    def _payload(
        self,
        messages: list[dict],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        stream: bool,
        **extra: Any,
    ) -> dict:
        m = model or settings.llm_model
        opts: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "repeat_penalty": 1.3,
            "repeat_last_n": 64,
        }
        if stop:
            opts["stop"] = stop
        payload: dict[str, Any] = {
            "model": m,
            "messages": messages,
            "stream": stream,
            "options": opts,
            "think": False,
        }
        payload.update(extra)
        return payload

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._payload(messages, model, temperature, max_tokens, stop, stream=False, **kwargs)
        resp = httpx.post(self._url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        payload = self._payload(messages, model, temperature, max_tokens, stop, stream=True, **kwargs)
        with httpx.stream("POST", self._url, json=payload, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done", False):
                    break

    def supports_vision(self) -> bool:
        return True  # Ollama VL models support images

    def vision_chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Ollama vision: messages already contain ``images`` key."""
        payload = self._payload(messages, model, temperature, max_tokens, stop=None, stream=True, **kwargs)
        try:
            with httpx.stream("POST", self._url, json=payload, timeout=180) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except httpx.HTTPStatusError as exc:
            yield f"\u274c Vision API error: {exc.response.status_code} — {exc.response.text[:200]}"
        except Exception as exc:
            yield f"\u274c Vision request failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenAI-compatible provider  (OpenAI · Mistral · Gemini · Grok · Qwen)
# ═══════════════════════════════════════════════════════════════════════════════

# Map provider name → (default base URL, env-var for the key)
_OPENAI_COMPAT_REGISTRY: dict[str, tuple[str, str]] = {
    "openai":  ("https://api.openai.com/v1",                                               "OPENAI_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1",                                                "MISTRAL_API_KEY"),
    "gemini":  ("https://generativelanguage.googleapis.com/v1beta/openai",                   "GEMINI_API_KEY"),
    "grok":    ("https://api.x.ai/v1",                                                      "GROK_API_KEY"),
    "qwen":    ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",                    "QWEN_API_KEY"),
}


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Any provider that follows the OpenAI ``/v1/chat/completions`` contract.

    The SSE stream format is:
        data: {"choices":[{"delta":{"content":"token"}}]}
        ...
        data: [DONE]
    """

    def __init__(self, provider: str, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.provider_name = provider
        if base_url and api_key:
            self._base = base_url.rstrip("/")
            self._api_key = api_key
        else:
            base_url_reg, key_env = _OPENAI_COMPAT_REGISTRY[provider]
            self._base = base_url_reg.rstrip("/")
            self._api_key = self._resolve_key(key_env)
        self._url = f"{self._base}/chat/completions"

    @staticmethod
    def _resolve_key(env_var: str) -> str:
        """Pull the API key from settings (populated from .env)."""
        key_map = {
            "OPENAI_API_KEY":  settings.openai_api_key,
            "MISTRAL_API_KEY": settings.mistral_api_key,
            "GEMINI_API_KEY":  settings.gemini_api_key,
            "GROK_API_KEY":    settings.grok_api_key,
            "QWEN_API_KEY":    settings.qwen_api_key,
            "CLOUDFLARE_API_KEY": settings.cloudflare_api_key,
        }
        key = key_map.get(env_var, "")
        if not key:
            raise ValueError(
                f"API key not set.  Please set {env_var} in your .env file."
            )
        return key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: list[dict],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        stream: bool,
    ) -> dict:
        m = model or settings.llm_model
        body: dict[str, Any] = {
            "model": m,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stop:
            body["stop"] = stop
        return body

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._payload(messages, model, temperature, max_tokens, stop, stream=False)
        resp = httpx.post(
            self._url, headers=self._headers(), json=payload, timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        payload = self._payload(messages, model, temperature, max_tokens, stop, stream=True)
        is_reasoning = False
        try:
            with httpx.stream(
                "POST", self._url,
                headers=self._headers(),
                json=payload,
                timeout=120,
            ) as resp:
                if resp.status_code == 429:
                    raise RuntimeError(
                        f"⚠️ {self.provider_name.upper()} rate limit exceeded. "
                        f"Please wait a moment and try again."
                    )
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"🔑 {self.provider_name.upper()} authentication failed. "
                        f"Check your API key in .env."
                    )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        raw = line[len("data: "):]
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        
                        # Handle reasoning_content -> <think> mapping
                        if "reasoning_content" in delta:
                            r_token = delta.get("reasoning_content")
                            if r_token is not None:
                                if not is_reasoning:
                                    is_reasoning = True
                                    yield "<think>\n"
                                if r_token:
                                    yield r_token
                                continue
                                
                        # Handle normal content
                        if "content" in delta:
                            c_token = delta.get("content")
                            if c_token is not None:
                                if is_reasoning:
                                    is_reasoning = False
                                    yield "\n</think>\n"
                                if c_token:
                                    yield c_token
        except RuntimeError:
            raise
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"❌ {self.provider_name.upper()} API error: "
                f"{exc.response.status_code} — {exc.response.text[:200]}"
            ) from exc

    def supports_vision(self) -> bool:
        # OpenAI (gpt-4o etc.) and Gemini support vision via content parts.
        # Mistral (pixtral), Grok, Qwen also have vision models.
        return self.provider_name in ("openai", "gemini", "grok", "mistral", "qwen", "cloudflare")

    def vision_chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        For OpenAI-compatible vision, messages should already contain
        ``content`` as a list of parts:
            [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:...;base64,..."}}]
        """
        yield from self.chat_stream(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Claude (Anthropic) provider
# ═══════════════════════════════════════════════════════════════════════════════

class ClaudeProvider(BaseLLMProvider):
    """
    Anthropic Messages API — ``/v1/messages``.

    Claude uses a different request/response shape:
      - ``system`` is a top-level field, not in ``messages``
      - Streaming events are ``content_block_delta`` with ``delta.text``
    """

    provider_name = "claude"

    def __init__(self) -> None:
        self._url = "https://api.anthropic.com/v1/messages"
        self._api_key = settings.claude_api_key
        if not self._api_key:
            raise ValueError(
                "API key not set.  Please set CLAUDE_API_KEY in your .env file."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """
        Anthropic expects ``system`` as a separate field.
        Return (system_prompt, user_messages).
        """
        system = ""
        user_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                user_msgs.append({"role": m["role"], "content": m["content"]})
        return system.strip(), user_msgs

    def _payload(
        self,
        messages: list[dict],
        model: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[list[str]],
        stream: bool,
    ) -> dict:
        m = model or settings.llm_model
        system, user_msgs = self._convert_messages(messages)
        body: dict[str, Any] = {
            "model": m,
            "max_tokens": max_tokens,
            "messages": user_msgs,
            "stream": stream,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop_sequences"] = stop
        return body

    # -- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        payload = self._payload(messages, model, temperature, max_tokens, stop, stream=False)
        resp = httpx.post(
            self._url, headers=self._headers(), json=payload, timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        # Claude response: {"content": [{"type":"text","text":"..."}]}
        return data["content"][0]["text"].strip()

    def chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        payload = self._payload(messages, model, temperature, max_tokens, stop, stream=True)
        with httpx.stream(
            "POST", self._url,
            headers=self._headers(),
            json=payload,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    raw = line[len("data: "):]
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type", "")
                    if etype == "content_block_delta":
                        token = event.get("delta", {}).get("text", "")
                        if token:
                            yield token
                    elif etype == "message_stop":
                        break

    def supports_vision(self) -> bool:
        return True  # Claude 3+ supports vision

    def vision_chat_stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Claude vision expects content as a list of parts:
            [{"type":"text","text":"..."}, {"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}]
        """
        yield from self.chat_stream(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_PROVIDERS = {"ollama", "openai", "mistral", "gemini", "grok", "qwen", "claude", "cloudflare"}

_provider_cache: dict[str, BaseLLMProvider] = {}


def get_provider(provider: Optional[str] = None) -> BaseLLMProvider:
    """
    Return a cached provider instance.

    Args:
        provider: Override the provider name.  When ``None``, reads from
                  ``settings.llm_provider``.

    Raises:
        ValueError: If the provider name is unknown.
    """
    name = (provider or settings.llm_provider).lower().strip()

    if name in _provider_cache:
        return _provider_cache[name]

    if name not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: '{name}'.  "
            f"Valid providers: {sorted(_VALID_PROVIDERS)}"
        )

    instance: BaseLLMProvider
    if name == "claude":
        instance = ClaudeProvider()
    elif name == "ollama":
        instance = OllamaProvider()
    elif name == "cloudflare":
        if not settings.cloudflare_api_key or not settings.cloudflare_account_id:
            raise ValueError("API key or Account ID not set. Please set CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID in your .env file.")
        instance = OpenAICompatibleProvider(
            provider="cloudflare",
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{settings.cloudflare_account_id}/ai/v1",
            api_key=settings.cloudflare_api_key,
        )
    else:
        # OpenAI, Mistral, Gemini, Grok, Qwen
        instance = OpenAICompatibleProvider(provider=name)

    _provider_cache[name] = instance
    logger.info("LLM provider initialised: %s", name)
    return instance
