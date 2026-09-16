# 🧠 Obsidian Local AI Agent

<div align="center">

**A fully local, private, multimodal AI assistant built for your Obsidian second brain.**

*Powered by Ollama · qwen3-vl · nomic-embed-text · ChromaDB · Python*

</div>

---

## ✨ What Is This?

You take notes in Obsidian every day — daily journals, lecture notes, project logs, random thoughts. But notes are only as useful as your ability to *find and connect* them.

This project builds a **local AI agent** that:
1. **Reads and understands** your entire Obsidian vault
2. **Answers your questions** based on your actual notes, not general internet knowledge
3. **Analyzes images** you show it (screenshots, diagrams, handwritten notes)
4. **Helps you learn** by identifying gaps, explaining concepts, and connecting ideas
5. **Can create and update notes** — but only with your explicit permission, and always with a revert safety net
6. **Knows the local date and time** — direct questions such as “İndi saat neçədir?” are answered from the configured local clock, not guessed by the model

> **Zero cloud. Zero API keys. Your data never leaves your machine.**

---

## 🌟 Feature Highlights

### 🔒 100% Local & Private
All AI inference runs through **Ollama** on your own hardware. No subscriptions, no data leaks, no rate limits. Your daily journal stays yours.

### 🖼️ Multimodal (Text + Images)
Using `qwen3-vl:2b-instruct`, a Vision-Language model, you can:
- Upload a screenshot of a textbook diagram → get an explanation
- Take a photo of handwritten notes → extract and organize key points
- Show an error message → get debugging help in context of your project notes

### 🔍 Intelligent Retrieval (RAG)
The agent doesn't just search keywords. It uses `nomic-embed-text` to understand the *meaning* of your question and find the most relevant notes — even across **Azerbaijani**, **English**, or mixed-language content.

### 🛡️ Read-Only by Default (Human-in-the-Loop)
The agent operates in **read-only mode** by default. If it wants to create a note, add a link, or update content, it **asks for your permission first** and clearly explains what it intends to do.

### ↩️ Undo System (Git-Backed)
Before the agent makes any approved change to your vault, it automatically creates a **Git commit** as a restore point. If you change your mind, reverting is a single command.

### 🌐 Multilingual & Personalized
Write in Azerbaijani, English, or switch mid-note — the system handles it naturally. The agent learns your writing style, your topics of interest, and your habits over time.

---

## 🛠️ Technology Stack

| Layer              | Technology                    | Purpose                                  |
|--------------------|-------------------------------|------------------------------------------|
| LLM Inference      | `Ollama`                      | Local model runtime                      |
| Language & Vision  | `qwen3-vl:2b-instruct`        | Text understanding + image analysis      |
| Embeddings         | `nomic-embed-text:latest`     | Converting text to searchable vectors    |
| Vector Database    | `ChromaDB`                    | Local semantic search storage            |
| Agent Orchestration | Native Python + HTTPX         | Agent logic, tools, memory               |
| Vault Parser       | `python-frontmatter`          | Reading Obsidian frontmatter & metadata  |
| File Watcher       | `watchdog`                    | Auto-updating the index on note changes  |
| Version Control    | `Git` + `gitpython`           | Vault backup & undo mechanism            |
| User Interface     | `Streamlit` / `Gradio`        | Local web UI (browser-accessible)        |
| Language           | `Python 3.11+`                | Core backend language                    |

---

## 🏗️ Architecture (Quick Overview)

```
Your Question
      │
      ▼
  [Embed Query]  ──── nomic-embed-text
      │
      ▼
  [Search Vault] ──── ChromaDB (local vectors of your notes)
      │
      ▼
  [Build Prompt] ──── Relevant note chunks + conversation history
      │
      ▼
  [Generate Answer] ── qwen3-vl:2b-instruct (via Ollama)
      │
      ▼
   Response + Sources (which notes were used)
```

For the full architecture with diagrams, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 📁 Project Structure

```
LLM2/
├── README.md                # You are here
├── requirements.txt         # Python dependencies
├── .env.example             # Configuration template (vault path, model names, etc.)
├── config.py                # Loads settings from .env
│
├── docs/
│   ├── ROADMAP.md           # Detailed development plan & milestones
│   ├── ARCHITECTURE.md      # System design, data flow, component diagram
│   └── SETUP.md             # Step-by-step installation guide (coming soon)
│
├── src/
│   ├── ingestion/           # Vault parser, chunker, and embedder
│   ├── retrieval/           # ChromaDB semantic search
│   ├── agent/               # RAG chain, vision handler, unified agent
│   ├── tools/               # Agent tools: Read, Create, Update, Link notes
│   ├── memory/              # Short-term and long-term conversation memory
│   └── ui/                  # Web interface (Streamlit / Gradio)
│
└── scripts/
    ├── test_ollama.py        # Verify Ollama is running correctly
    ├── test_chromadb.py      # Verify vector DB is working
    └── vault_backup.py       # CLI backup, log, and restore utility
```

---

## 🚀 Getting Started

> **Prerequisites:** Python 3.11+, Ollama installed and running, `qwen3-vl:2b-instruct` and `nomic-embed-text` pulled.

### 1. Verify your Ollama models

```bash
ollama list
# Should show: qwen3-vl:2b-instruct, nomic-embed-text:latest
```

### 2. Clone the project and install dependencies

```bash
cd /path/to/LLM2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure your vault path

```bash
cp .env.example .env
# Edit .env and set VAULT_PATH=/path/to/your/obsidian/vault
```

### 4. Run connection tests

```bash
python scripts/test_ollama.py
python scripts/test_chromadb.py
```

### 5. Index your vault

```bash
python -m src.ingestion.embedder --vault $VAULT_PATH
```

Optional: keep the index updated while Obsidian is open:

```bash
python -m src.ingestion.watcher --vault $VAULT_PATH
```

### 6. Launch the agent

```bash
python -m src.ui.app   # Opens a web UI in your browser
# OR
python -m src.agent.agent  # Terminal/CLI mode
```

---

## 🗺️ Roadmap

| Phase | Description                         | Status         |
|-------|-------------------------------------|----------------|
| 0     | Research & Scaffolding              | ✅ Complete     |
| 1     | Infrastructure & Tooling            | ✅ Implemented |
| 2     | Vault Ingestion & RAG Pipeline      | ✅ Core implemented |
| 3     | Multimodal Brain & Session Memory   | 🟡 Core implemented |
| 4     | Agentic Write & Permission System   | 🟡 CLI implemented |
| 5     | UI & UX Polish                      | 🟡 Core UI implemented |
| 6     | Advanced Features (Habits, Gaps...) | 🔭 Future      |

👉 See the full, detailed roadmap in [`docs/ROADMAP.md`](docs/ROADMAP.md).

### What still needs work

- The vault watcher, persistent long-term memory, and UI write-confirmation flow are not implemented yet.
- Indexing removes stale chunks when notes are deleted or shortened. Run the optional watcher to keep it updated while Obsidian is open.
- Configure `.env`, start Ollama, and index a real vault before treating the assistant as ready for daily use.

---

## 🧪 Current Models in Use

```
neo@debian ~ % ollama list
NAME                       ID              SIZE      MODIFIED
nomic-embed-text:latest    0a109f422b47    274 MB    2 days ago
qwen3-vl:2b-instruct       ea422f1e7365    1.9 GB    2 days ago
```

---

## 🤝 Contributing

This is a personal project, but the architecture is modular by design. Each component (ingestion, retrieval, agent, tools, UI) can be swapped or extended independently.

---

## 📄 License

MIT License — use, modify, and share freely.
