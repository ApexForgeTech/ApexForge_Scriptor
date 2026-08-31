"""
src/agent/vision.py — Multimodal image analysis using qwen3-vl.

Supports:
  - Image path on disk
  - Base64-encoded image string
  - URL to an image

The image is sent alongside a text prompt to qwen3-vl:2b-instruct via Ollama.
Optionally, vault context (from RAG) is injected into the prompt.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.agent.prompts import VISION_PROMPT_TEMPLATE


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
    Analyse an image using qwen3-vl:2b-instruct.

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

    # Ollama multimodal message format
    messages = [
        {
            "role": "user",
            "content": prompt_text,
            "images": [img_data],
        }
    ]

    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        },
    }

    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=180,  # Vision models take longer
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except httpx.HTTPStatusError as exc:
        return f"❌ Vision API error: {exc.response.status_code} — {exc.response.text[:200]}"
    except Exception as exc:
        return f"❌ Vision request failed: {exc}"


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

    Returns:
        The model's response string.
    """
    from src.retrieval.search import search, build_context

    # Retrieve relevant vault context using the question as the query
    sources = search(question, top_k=top_k)
    vault_context = build_context(sources) if sources else ""

    return analyze_image(image_path, question, vault_context)
