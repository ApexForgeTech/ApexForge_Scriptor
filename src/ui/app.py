"""
src/ui/app.py — Gradio web interface for the Obsidian Local AI Agent.

Features:
  - Chat tab: Streaming conversation with vault-aware RAG
  - Image tab: Upload image + ask question (multimodal)
  - Vault tab: Browse notes, tags, daily logs
  - Settings tab: Configure vault path, model, permissions

Run: python -m src.ui.app
Opens at: http://localhost:7860
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Generator

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings
from src.agent.agent import ObsidianAgent

# ── Global agent instance (shared across tabs) ────────────────────────────────
_agent: ObsidianAgent | None = None
_agent_lock = threading.Lock()


def _get_agent() -> ObsidianAgent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = ObsidianAgent()
    return _agent


# ── Chat tab functions ────────────────────────────────────────────────────────

def chat_respond(
    message: str,
    history: list[dict],
    note_type_filter: str,
    topic_filter: str,
    tag_filter: str,
) -> Generator[tuple[list, str], None, None]:
    """Streaming chat response for the Chat tab."""
    if not message.strip():
        yield history, ""
        return

    agent = _get_agent()

    note_type = note_type_filter if note_type_filter != "All" else None
    topic = topic_filter.strip() or None
    tag = tag_filter.strip().lstrip("#") or None

    # Add user message to history immediately (openai-style)
    history = history + [{"role": "user", "content": message}]
    history.append({"role": "assistant", "content": ""})

    full_answer = ""

    try:
        stream_gen = agent.chat_stream(
            message=message,
            note_type=note_type,
            folder_subcategory=topic,
            tag=tag,
        )

        try:
            while True:
                token = next(stream_gen)
                full_answer += token
                history[-1] = {"role": "assistant", "content": full_answer}
                yield history, ""
        except StopIteration as e:
            response = e.value
            if response and response.sources:
                sources_text = response.sources_text
                history[-1] = {"role": "assistant", "content": full_answer + sources_text}
                yield history, ""

    except Exception as exc:
        error_msg = f"❌ Error: {exc}"
        history[-1] = {"role": "assistant", "content": error_msg}
        yield history, ""


def clear_chat() -> tuple[list, str]:
    """Clear conversation history and agent memory."""
    try:
        _get_agent().clear_memory()
    except Exception:
        pass
    return [], ""


# ── Image tab functions ───────────────────────────────────────────────────────

def analyze_image(image_path: str | None, question: str) -> str:
    """Analyse an uploaded image with optional vault context."""
    if not image_path:
        return "❌ Please upload an image first."
    if not question.strip():
        return "❌ Please enter a question about the image."

    agent = _get_agent()
    result = agent.chat_with_image(image_path, question)
    return result


# ── Vault tab functions ───────────────────────────────────────────────────────

def get_vault_stats() -> str:
    """Return a markdown summary of the vault."""
    vault = settings.vault_path
    if not vault.exists():
        return f"❌ Vault not found at: {vault}"

    md_files = list(vault.rglob("*.md"))
    daily = sum(1 for f in md_files if "months" in str(f).lower() or
                any(m in str(f).lower() for m in ["january","february","march","april","may",
                                                    "june","july","august","september",
                                                    "october","november","december"]))
    learning = sum(1 for f in md_files if "learning" in str(f).lower())
    total_size = sum(f.stat().st_size for f in md_files) // 1024

    return f"""## 📊 Vault Statistics

| Metric | Value |
|--------|-------|
| 📁 Vault Path | `{vault}` |
| 📄 Total Notes | **{len(md_files)}** |
| 📅 Daily Notes | **{daily}** |
| 📚 Learning Notes | **{learning}** |
| 💾 Total Size | **{total_size} KB** |
| 🤖 Embed Model | `{settings.embed_model}` |
| 🧠 LLM Model | `{settings.llm_model}` |
| 🔐 Write Permission | `{settings.write_permission}` |
"""


def get_tags_display() -> str:
    from src.tools.read_tools import list_tags
    return list_tags(min_count=2)


def get_daily_display(date_str: str) -> str:
    from src.tools.read_tools import get_daily_note
    return get_daily_note(date_str.strip() or "today")


def get_learning_notes() -> str:
    from src.tools.read_tools import list_notes
    return list_notes(note_type="learning", limit=30)


def search_vault_display(query: str, top_k: int) -> str:
    if not query.strip():
        return "Enter a search query above."
    from src.tools.read_tools import search_vault
    return search_vault(query.strip(), top_k=int(top_k))


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    custom_css = """
    :root {
        --ink: #f7fbff;
        --panel: rgba(255, 255, 255, .83);
        --panel-soft: rgba(245, 250, 255, .9);
        --line: rgba(83, 134, 204, .20);
        --cyan: #0079e8;
        --violet: #7141df;
        --pink: #e2478c;
        --text: #14213b;
        --muted: #58708f;
    }
    .gradio-container {
        max-width: 1500px !important;
        min-height: 100vh;
        color: var(--text) !important;
        font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
        background:
          radial-gradient(circle at 5% 4%, rgba(41, 189, 255, .23), transparent 27rem),
          radial-gradient(circle at 98% 14%, rgba(150, 112, 255, .20), transparent 29rem),
          radial-gradient(circle at 50% 105%, rgba(255, 133, 178, .18), transparent 32rem),
          var(--ink) !important;
    }
    .gradio-container:before {
        content: ""; position: fixed; inset: 0; pointer-events: none; opacity: .5;
        background-image: linear-gradient(rgba(119,177,255,.09) 1px, transparent 1px), linear-gradient(90deg, rgba(119,177,255,.09) 1px, transparent 1px);
        background-size: 42px 42px; mask-image: linear-gradient(to bottom, black, transparent 80%);
    }
    #hero { position: relative; overflow: hidden; border: 1px solid rgba(85,125,220,.24); border-radius: 24px; margin: 10px 0 22px; padding: 28px 32px; background: linear-gradient(112deg, #e5f8ff 0%, #f4ebff 52%, #fff0f7 100%); box-shadow: 0 18px 42px rgba(75, 123, 190, .15); }
    #hero:after { content: ""; position: absolute; width: 320px; height: 320px; right: -90px; top: -210px; border-radius: 50%; background: #68cfff; filter: blur(85px); opacity: .30; }
    .eyebrow { color: var(--cyan); letter-spacing: .16em; font-size: .72rem; font-weight: 800; text-transform: uppercase; }
    .hero-title { margin: 7px 0 5px; font-size: clamp(1.7rem, 4vw, 2.75rem); font-weight: 800; letter-spacing: -.045em; color: #18294a; }
    .hero-copy { color: var(--muted); font-size: 1rem; max-width: 680px; }
    .status-pill { display: inline-flex; margin-top: 17px; padding: 7px 11px; border: 1px solid rgba(0,121,232,.24); border-radius: 999px; color: #1762a8; background: rgba(255,255,255,.68); font-size: .79rem; font-weight: 700; }
    .status-dot { width: 7px; height: 7px; background: #5ff3ac; border-radius: 50%; margin: 5px 7px 0 0; box-shadow: 0 0 12px #5ff3ac; }
    #main-tabs { border: 1px solid var(--line); padding: 8px; border-radius: 18px; background: rgba(255,255,255,.65); backdrop-filter: blur(16px); }
    #main-tabs button { border-radius: 11px !important; color: var(--muted) !important; font-weight: 700 !important; }
    #main-tabs button.selected { color: #18335b !important; background: linear-gradient(100deg, #c9f2ff, #e5d7ff 55%, #ffe0ef) !important; }
    #chat-layout, #image-layout { gap: 18px; padding: 12px 4px 4px; }
    #chat-panel, #side-panel, #image-panel, #result-panel, #explorer-panel, #settings-panel { border: 1px solid var(--line); border-radius: 18px; background: var(--panel); box-shadow: 0 12px 35px rgba(56,103,160,.12); }
    #chat-panel { padding: 15px; }
    #side-panel, #image-panel, #result-panel, #explorer-panel, #settings-panel { padding: 18px; }
    #side-panel h3, #explorer-panel h2 { color: #1c3156; }
    #chatbot { border: 1px solid rgba(105,156,224,.20) !important; border-radius: 14px !important; overflow: hidden; background: rgba(242, 249, 255, .76) !important; }
    #chatbot .message { border-radius: 14px !important; border: 1px solid rgba(105,156,224,.13) !important; box-shadow: none !important; color: #172946 !important; }
    #chatbot .user { background: linear-gradient(115deg, #d5f4ff, #e8dcff) !important; }
    #chatbot .bot { background: rgba(255, 255, 255, .92) !important; }
    #composer textarea { color: #183154 !important; background: rgba(255,255,255,.96) !important; }
    #composer { border: 1px solid rgba(0,121,232,.30) !important; border-radius: 14px !important; background: rgba(255,255,255,.86) !important; }
    button.primary { border: 0 !important; background: linear-gradient(100deg, #27c8f1, #8170f5 55%, #d061d7) !important; box-shadow: 0 8px 25px rgba(86,138,255,.28) !important; font-weight: 800 !important; }
    button.secondary { background: #edf5ff !important; border-color: var(--line) !important; color: #4d6688 !important; }
    label span, .gradio-container .prose { color: var(--muted) !important; }
    input, textarea { border-color: rgba(148,192,255,.17) !important; }
    #quick-questions { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--muted); }
    #quick-questions li { margin: 8px 0; color: #415c7e; }
    #quick-questions strong { color: var(--cyan); }
    .tabitem { padding-top: 12px !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="🧠 Obsidian AI Agent",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=custom_css,
    ) as demo:

        gr.HTML("""
<section id="hero">
  <div class="eyebrow">Private knowledge workspace</div>
  <div class="hero-title">ApexForge Scriptor</div>
  <div class="hero-copy">Sənin qeydlərini anlayan, əlaqələndirən və yalnız lokal işləyən şəxsi AI assistant.</div>
  <div class="status-pill"><span class="status-dot"></span>LOCAL MODE · OLLAMA CONNECTED</div>
</section>
""")

        with gr.Tabs(elem_id="main-tabs"):

            # ── Tab 1: Chat ────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Sənin ikinci beynin\nQeydlərin, öyrənmə materialların və gündəliklərin üzərində soruş. Cavabların altında istifadə olunan mənbələr göstərilir.")

                with gr.Row(elem_id="chat-layout"):
                    with gr.Column(scale=3, elem_id="chat-panel"):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=520,
                            show_copy_button=True,
                            render_markdown=True,
                            type="messages",
                            avatar_images=(None, "🤖"),
                            elem_id="chatbot",
                        )
                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="Ask a question about your notes... (Azerbaijani or English)",
                                label="",
                                scale=5,
                                container=False,
                                lines=2,
                                elem_id="composer",
                            )
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

                    with gr.Column(scale=1, elem_id="side-panel"):
                        gr.Markdown("### 🔍 Filters")
                        note_type = gr.Radio(
                            choices=["All", "daily", "learning", "general"],
                            value="All",
                            label="Note Type",
                        )
                        topic_filter = gr.Textbox(
                            label="Topic Folder",
                            placeholder="e.g. RUST, ASM, January",
                        )
                        tag_filter = gr.Textbox(
                            label="Tag Filter",
                            placeholder="e.g. SLEEP, EXAM, RUST",
                        )
                        gr.Markdown("""<div id="quick-questions"><strong>Tez suallar</strong>
- *Rust-da ownership nədir?*
- *Bu həftə nə etdim?*
- *ASM lesson 3 haqqında nə var?*
- *Sabr nə deməkdir?*
</div>""")

                # Wire up chat
                send_btn.click(
                    chat_respond,
                    inputs=[msg_box, chatbot, note_type, topic_filter, tag_filter],
                    outputs=[chatbot, msg_box],
                ).then(lambda: "", outputs=msg_box)

                msg_box.submit(
                    chat_respond,
                    inputs=[msg_box, chatbot, note_type, topic_filter, tag_filter],
                    outputs=[chatbot, msg_box],
                ).then(lambda: "", outputs=msg_box)

                clear_btn.click(clear_chat, outputs=[chatbot, msg_box])

            # ── Tab 2: Image Analysis ──────────────────────────────────────
            with gr.Tab("🖼️ Image Analysis"):
                gr.Markdown("""
*Upload an image — a screenshot, diagram, handwritten notes, or error message.
The agent analyses it using `qwen3-vl` and enriches its answer with your vault context.*
""")
                with gr.Row(elem_id="image-layout"):
                    with gr.Column(elem_id="image-panel"):
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            height=300,
                        )
                        image_question = gr.Textbox(
                            label="Your question",
                            placeholder="Bu şəkli izah et / What does this diagram show? / Extract the text",
                            lines=2,
                        )
                        analyze_btn = gr.Button("🔍 Analyse Image", variant="primary")

                    with gr.Column(elem_id="result-panel"):
                        image_result = gr.Markdown(label="Analysis Result")

                analyze_btn.click(
                    analyze_image,
                    inputs=[image_input, image_question],
                    outputs=image_result,
                )

            # ── Tab 3: Vault Explorer ──────────────────────────────────────
            with gr.Tab("📂 Vault Explorer", elem_id="explorer-panel"):
                with gr.Tabs():

                    with gr.Tab("📊 Stats"):
                        stats_btn = gr.Button("Refresh Stats", variant="primary")
                        stats_out = gr.Markdown()
                        stats_btn.click(get_vault_stats, outputs=stats_out)
                        demo.load(get_vault_stats, outputs=stats_out)

                    with gr.Tab("🏷️ Tags"):
                        tags_btn = gr.Button("Load Tags", variant="primary")
                        tags_out = gr.Textbox(label="All Tags (used 2+ times)", lines=20, interactive=False)
                        tags_btn.click(get_tags_display, outputs=tags_out)

                    with gr.Tab("📅 Daily Notes"):
                        with gr.Row():
                            daily_date = gr.Textbox(
                                label="Date",
                                placeholder="today, yesterday, or 2025-01-17",
                                value="today",
                            )
                            daily_btn = gr.Button("Load", variant="primary")
                        daily_out = gr.Markdown()
                        daily_btn.click(get_daily_display, inputs=daily_date, outputs=daily_out)

                    with gr.Tab("📚 Learning Notes"):
                        learning_btn = gr.Button("Load Learning Notes", variant="primary")
                        learning_out = gr.Textbox(label="Learning Notes", lines=25, interactive=False)
                        learning_btn.click(get_learning_notes, outputs=learning_out)

                    with gr.Tab("🔍 Search"):
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="Rust ownership, ASM registers, sabr...",
                        )
                        top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K Results")
                        search_btn = gr.Button("Search", variant="primary")
                        search_out = gr.Textbox(label="Results", lines=20, interactive=False)
                        search_btn.click(
                            search_vault_display,
                            inputs=[search_query, top_k_slider],
                            outputs=search_out,
                        )

            # ── Tab 4: Settings ────────────────────────────────────────────
            with gr.Tab("⚙️ Settings", elem_id="settings-panel"):
                gr.Markdown(f"""
## Current Configuration

| Setting | Value |
|---------|-------|
| **Vault Path** | `{settings.vault_path}` |
| **LLM Model** | `{settings.llm_model}` |
| **Embed Model** | `{settings.embed_model}` |
| **Ollama URL** | `{settings.ollama_base_url}` |
| **ChromaDB Path** | `{settings.chroma_persist_dir}` |
| **Retrieval Top-K** | `{settings.retrieval_top_k}` |
| **Write Permission** | `{settings.write_permission}` |
| **Memory Window** | `{settings.memory_window} turns` |

---

To change these settings, edit the `.env` file and restart the application:

```bash
nano .env
python -m src.ui.app
```
""")

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Starting Obsidian Local AI Agent...")
    print(f"   Vault: {settings.vault_path}")
    print(f"   LLM:   {settings.llm_model}")
    print(f"   Embed: {settings.embed_model}")
    print()

    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )
