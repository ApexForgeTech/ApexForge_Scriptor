"""
src/agent/vision.py — Multimodal image analysis.

Supports:
  - Image path on disk
  - Base64-encoded image string
  - URL to an image

The image is sent to the active LLM provider via the unified LLMProvider
adapter.  Provider-specific message formats are handled automatically:
  - Ollama:              messages[].images = [base64]
  - OpenAI-compatible:   content = [{type:image_url, ...}]
  - Claude:              content = [{type:image, source:{type:base64, ...}}]

Optionally, vault context (from RAG) is injected into the prompt.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.agent.prompts import VISION_PROMPT_TEMPLATE
from src.agent.llm_provider import get_provider


def _encode_image(image_path: Path) -> tuple[str, str]:
    """
    Read an image file and return (base64_data, mime_type).
    Supported: .png, .jpg, .jpeg, .gif, .webp
    """
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return data, mime


def analyze_image(
    image_path: str | Path,
    question: str,
    vault_context: str = "",
) -> str:
    """
    Analyse an image using the active LLM provider.

    Args:
        image_path:    Absolute or relative path to the image file.
        question:      User's question or instruction about the image.
        vault_context: Optional relevant note context from RAG retrieval.

    Returns:
        The model's analysis as a plain text string.
    """
    img = Path(image_path)
    if not img.exists():
        return f"❌ Image not found: {image_path}"

    img_data, mime_type = _encode_image(img)

    # Build the prompt using the template
    context_section = vault_context if vault_context else "(No relevant vault notes found for this image.)"
    prompt_text = VISION_PROMPT_TEMPLATE.format(
        vault_context=context_section,
        question=question,
    )

    provider = get_provider()

    # Build provider-specific messages
    if provider.provider_name == "ollama":
        # Ollama format: images as a separate key
        messages = [
            {
                "role": "user",
                "content": prompt_text,
                "images": [img_data],
            }
        ]
    elif provider.provider_name == "claude":
        # Claude format: content as list of parts
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
    else:
        # OpenAI-compatible format (OpenAI, Gemini, Mistral, Grok, Qwen)
        data_uri = f"data:{mime_type};base64,{img_data}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

    try:
        if provider.supports_vision():
            yield from provider.vision_chat_stream(messages)
        else:
            # Fallback: text-only analysis without the image
            text_messages = [{"role": "user", "content": prompt_text}]
            yield "⚠️ *Provider does not support vision. Answering text-only.*\n\n"
            yield from provider.chat_stream(text_messages)
    except Exception as exc:
        yield f"❌ Vision request failed: {exc}"


def analyze_image_with_rag(
    image_path: str | Path,
    question: str,
    top_k: int = 3,
) -> str:
    """
    Analyse an image, enriching the prompt with relevant vault notes via RAG.

    Args:
        image_path: Path to the image.
        question:   The user's question.
        top_k:      Number of vault chunks to retrieve for context.

    Yields:
        Tokens from the model response (streaming).
    """
    from src.retrieval.search import search, build_context

    # Retrieve relevant vault context using the question as the query
    sources = search(question, top_k=top_k)
    vault_context = build_context(sources) if sources else ""

    yield from analyze_image(image_path, question, vault_context)

