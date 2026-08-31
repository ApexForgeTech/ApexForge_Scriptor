# 🗺️ Project Roadmap — Obsidian Local AI Agent

> **Vision:** Build a fully local, private, multimodal AI agent that deeply understands your Obsidian vault — your daily notes, study materials, project logs, and habits — and acts as a true second brain that advises, creates, and connects knowledge without ever sending your data to the cloud.

---

## 📊 Overview Table

| Phase | Name                              | Status      | Estimated Effort |
|-------|-----------------------------------|-------------|------------------|
| 0     | Research & Scaffolding            | ✅ Done      | ~1 day           |
| 1     | Infrastructure & Tooling          | 🔄 In Progress | ~2-3 days     |
| 2     | Vault Ingestion & RAG Pipeline    | ⏳ Upcoming | ~3-4 days        |
| 3     | Multimodal Brain & Memory         | ⏳ Upcoming | ~4-5 days        |
| 4     | Agentic Write & Permission System | ⏳ Upcoming | ~3-4 days        |
| 5     | Interface & UX Polish             | ⏳ Upcoming | ~3-5 days        |
| 6     | Advanced Features & Automation    | 🔭 Future   | Ongoing          |

---

## ✅ Phase 0 — Research & Project Scaffolding
**Status: Completed**

This phase covers the initial discovery, requirements gathering, and basic project structure.

### Goals
- Understand what the user needs from the AI agent (advising, studying, journaling, image analysis, etc.)
- Evaluate available open-source tools and models
- Create initial documentation

### Tasks
- [x] Decide on the core tech stack (Python + LangChain + ChromaDB + Ollama)
- [x] Choose the right models: `qwen3-vl:2b-instruct` for inference/vision, `nomic-embed-text` for embeddings
- [x] Create `README.md` with project overview
- [x] Create `docs/ROADMAP.md` (this file)
- [x] Create `docs/ARCHITECTURE.md` with system diagram and data flow

### Decisions Made
- **Why Ollama?** It is the easiest, most mature local LLM runtime. It exposes an OpenAI-compatible REST API, which means we can swap models easily.
- **Why qwen3-vl?** It is a Vision-Language model, meaning it can understand both text and images. This is critical for analyzing screenshots, diagrams from study notes, etc.
- **Why nomic-embed-text?** It is a lightweight, high-quality embedding model that runs fast locally and produces good multilingual vectors, essential for Azerbaijani + English notes.
- **Why ChromaDB?** It is the simplest, most developer-friendly local vector database. No server setup needed — it runs as a Python library.

---

## 🔄 Phase 1 — Infrastructure & Tooling
**Status: In Progress**

This phase sets up the development environment and confirms that all systems can communicate with each other before writing any real logic.

### Goals
- Create a reproducible development environment
- Verify the entire local AI stack is operational
- Set up Git on the Obsidian vault as the "undo" mechanism

### Tasks

#### 1.1 — Python Environment Setup
- [ ] Create `requirements.txt` with all dependencies:
  - `langchain`, `langchain-community`, `langchain-ollama`
  - `chromadb`
  - `ollama` (Python SDK)
  - `python-frontmatter` (for reading Obsidian `.md` metadata)
  - `watchdog` (for monitoring vault file changes in real time)
  - `streamlit` or `gradio` (for the UI — decision in Phase 5)
  - `pillow` (for image handling)
  - `gitpython` (for programmatic Git commits from Python)
- [ ] Create a Python virtual environment: `python -m venv .venv`
- [ ] Create a `.env.example` file for configurable settings (vault path, Ollama URL, etc.)
- [ ] Create a `config.py` or `settings.py` to load configuration cleanly

#### 1.2 — Ollama Connection Test
- [ ] Write `scripts/test_ollama.py`:
  - Ping the Ollama API (`http://localhost:11434`)
  - Send a sample prompt to `qwen3-vl:2b-instruct` and print the response
  - Generate an embedding for a sample text using `nomic-embed-text`
  - Confirm both models respond correctly

#### 1.3 — ChromaDB Initialization Test
- [ ] Write `scripts/test_chromadb.py`:
  - Initialize a local ChromaDB collection
  - Insert 3 sample document chunks with metadata
  - Run a similarity query and verify results are returned correctly
  - Delete the test collection

#### 1.4 — Git-Based Vault Safety Setup
- [ ] Document the process for the user to run `git init` inside their Obsidian vault directory
- [ ] Write `scripts/vault_backup.py`:
  - A function `backup_vault(message: str)` that runs `git add -A && git commit -m message` using `gitpython`
  - This will be called automatically before any write operation by the agent
- [ ] Test the backup and revert cycle manually

#### 1.5 — Project Structure
- [ ] Finalize the directory layout:
  ```
  LLM2/
  ├── README.md
  ├── requirements.txt
  ├── .env.example
  ├── config.py
  ├── docs/
  │   ├── ROADMAP.md
  │   ├── ARCHITECTURE.md
  │   └── SETUP.md
  ├── src/
  │   ├── ingestion/       # Vault reading & embedding
  │   ├── retrieval/       # ChromaDB search
  │   ├── agent/           # LangChain agent logic
  │   ├── tools/           # Agent tools (read, write, backup)
  │   ├── memory/          # Session & long-term memory
  │   └── ui/              # Streamlit / Gradio interface
  └── scripts/
      ├── test_ollama.py
      ├── test_chromadb.py
      └── vault_backup.py
  ```

---

## ⏳ Phase 2 — Vault Ingestion & RAG Pipeline
**Status: Upcoming**

This is the most critical phase. The agent needs to "read" and "understand" your entire Obsidian vault so it can answer questions based on your real notes.

### Goals
- Parse all Obsidian Markdown files correctly (including tags, wikilinks, frontmatter)
- Chunk documents intelligently for embedding
- Store and retrieve vectors from ChromaDB
- Handle multilingual content (Azerbaijani + English)

### Tasks

#### 2.1 — Obsidian Markdown Parser
- [ ] Write `src/ingestion/parser.py`:
  - Read all `.md` files recursively from the vault path
  - Parse YAML frontmatter (date, tags, title, aliases) using `python-frontmatter`
  - Extract plain text content, preserving structure
  - Identify and extract Obsidian-specific syntax:
    - Wikilinks: `[[Note Name]]` and `[[Note Name|Display Text]]`
    - Tags: `#tag`, `#nested/tag`
    - Embedded files: `![[image.png]]`
    - Block references: `^block-id`
  - Return a structured `Document` object: `{content, metadata: {path, title, tags, date, links}}`

#### 2.2 — Smart Text Chunking
- [ ] Write `src/ingestion/chunker.py`:
  - Implement a **Heading-Aware Chunker**: Split notes by `## Heading` sections to keep logical units together
  - Implement a **Size-Aware Chunker**: If a section is too long (>512 tokens), further split it with overlap
  - Add metadata to each chunk: `{source_file, heading, chunk_index, total_chunks}`
  - Why this matters: Small, targeted chunks make retrieval much more precise

#### 2.3 — Embedding Pipeline
- [ ] Write `src/ingestion/embedder.py`:
  - Take a list of document chunks
  - Generate embeddings in batches using `nomic-embed-text` via Ollama
  - Store each chunk + its embedding + its metadata into ChromaDB
  - Handle incremental updates: Only re-embed files that have changed (compare file modification timestamps)
  - Create a CLI command: `python -m src.ingestion.embedder --vault /path/to/vault`

#### 2.4 — Semantic Search / Retrieval
- [ ] Write `src/retrieval/search.py`:
  - Function `search(query: str, top_k: int = 5) -> list[Document]`
  - Embed the query using `nomic-embed-text`
  - Query ChromaDB for the `top_k` most similar chunks
  - Apply optional filters: by tag, by date range, by folder
  - Return results with relevance scores

#### 2.5 — Real-Time Vault Watcher (Optional but very useful)
- [ ] Write `src/ingestion/watcher.py`:
  - Use `watchdog` library to monitor the vault directory for file changes
  - On file save/create: Automatically re-chunk and re-embed the changed file
  - On file delete: Remove its vectors from ChromaDB
  - This keeps the vector database always up-to-date without manual re-ingestion

---

## ⏳ Phase 3 — Multimodal Brain & Memory
**Status: Upcoming**

This phase connects the retrieval system with the language model and adds memory and image understanding.

### Goals
- Build a working chat loop with full RAG support
- Enable image input and analysis
- Add session memory so the agent understands conversation context

### Tasks

#### 3.1 — Basic RAG Chain
- [ ] Write `src/agent/rag_chain.py`:
  - Build a LangChain `RetrievalQA` or `ConversationalRetrievalChain`
  - System prompt: Define the agent's persona — it knows the user's notes, habits, and learning goals
  - On each user message: retrieve relevant notes → inject into prompt → generate response
  - Stream output token-by-token for a responsive feel

#### 3.2 — System Prompt Engineering
- [ ] Write `src/agent/prompts.py`:
  - Design a detailed system prompt covering:
    - Role: "You are a personal AI assistant integrated with the user's Obsidian vault..."
    - Behavior: "You speak directly and concisely. You ask clarifying questions when uncertain."
    - Language: "Respond in the language the user writes in (Azerbaijani or English)."
    - Safety: "You never make changes to files without explicit user permission."
    - Advice: "When you notice learning gaps or unfinished topics in the notes, proactively flag them."
  - Make prompts easily configurable without touching code

#### 3.3 — Conversation Memory
- [ ] Write `src/memory/session_memory.py`:
  - Implement **Short-term memory**: Keep the last N turns of conversation in context using LangChain `ConversationBufferWindowMemory`
  - Implement **Long-term memory**: Periodically summarize the session and store a `session_summary.md` note in a dedicated folder in the vault
  - On startup: Load the last session summary to give the agent awareness of past conversations

#### 3.4 — Image Input & Multimodal Analysis
- [ ] Write `src/agent/vision.py`:
  - Accept an image path or base64-encoded image from the user
  - Build a multimodal prompt with the image + text question
  - Call `qwen3-vl:2b-instruct` via Ollama's multimodal API
  - Use cases:
    - "Here is a screenshot of a diagram from my textbook, explain it"
    - "I took a photo of my handwritten notes, extract the key points"
    - "Here's an error message screenshot, what does it mean?"
  - Return the model's analysis as a structured response

#### 3.5 — Multi-turn Contextual Chat
- [ ] Integrate memory + RAG + vision into one unified `Agent` class in `src/agent/agent.py`
- [ ] The agent must correctly handle:
  - Pure text questions
  - Follow-up questions referring to previous turns ("What about that other thing you mentioned?")
  - Image + text inputs
  - Questions that span multiple notes ("What's the connection between my note on X and my note on Y?")

---

## ⏳ Phase 4 — Agentic Write & Permission System
**Status: Upcoming**

This phase gives the agent the ability to actively modify your vault — but only with your explicit approval and always with a safety net.

### Goals
- Implement LangChain tools for vault manipulation
- Build a Human-in-the-Loop (HITL) permission gate
- Ensure all write operations are Git-backed and reversible

### Tasks

#### 4.1 — Define Agent Tools
- [ ] Write `src/tools/`:
  - **`ReadNoteTool`**: Read the full contents of a specific note by name or path
  - **`SearchVaultTool`**: Semantic search across the vault (wrapper around Phase 2 retrieval)
  - **`CreateNoteTool`**: Create a new `.md` file in the vault with given title, content, and tags
  - **`UpdateNoteTool`**: Append to or modify a specific section of an existing note
  - **`LinkNotesTool`**: Add a wikilink `[[Target Note]]` to a note
  - **`ListTagsTool`**: List all tags currently in use across the vault
  - **`GetDailyNoteTool`**: Fetch today's daily note (based on the Obsidian daily notes format)

#### 4.2 — Human-in-the-Loop Permission Gate
- [ ] Write `src/tools/permission.py`:
  - Every write tool (Create, Update, Link) must call `request_permission(action_description: str) -> bool` before executing
  - **CLI mode:** Print a clear description of what the agent intends to do, ask `[y/N]:`
  - **UI mode:** Show a confirmation modal/dialog before proceeding
  - If the user denies: the agent is informed and must either abandon the action or propose an alternative
  - Log all permission requests and outcomes to `logs/permissions.log`

#### 4.3 — Automatic Git Backup Before Write
- [ ] Integrate `vault_backup.py` (from Phase 1) into every write tool:
  - Before any `CreateNoteTool` or `UpdateNoteTool` executes, call `backup_vault(f"Agent pre-write backup: {action}")`
  - This creates a restore point the user can always go back to with `git checkout` or `git revert`

#### 4.4 — Revert System
- [ ] Write `scripts/revert_vault.py`:
  - List recent agent-made commits: `git log --oneline --author="AI Agent"`
  - Allow the user to select a commit to revert to
  - Run `git checkout <commit_hash> -- .` or `git revert <commit_hash>`
  - Provide a simple CLI interface for this: `python scripts/revert_vault.py`

---

## ⏳ Phase 5 — Interface & UX Polish
**Status: Upcoming**

Build a clean, easy-to-use interface so you don't have to work purely in the terminal.

### Goals
- Create a web-based UI accessible from a local browser
- Make it intuitive enough to use daily without friction
- Display vault context, related notes, and source attribution

### Tasks

#### 5.1 — Choose & Build the UI Framework
- [ ] Evaluate and decide between **Streamlit** (simpler, faster) or **Gradio** (better multimodal support)
  - Decision criteria: image upload ease, chat streaming, session state management
- [ ] Write `src/ui/app.py`:
  - A sidebar showing the currently active vault path and indexed document count
  - A main chat window with streaming response support
  - An image upload button next to the chat input
  - A "Sources" panel: After each AI response, show which notes were used as context (with clickable links to open them in Obsidian)

#### 5.2 — Obsidian Deep Links
- [ ] When displaying source notes in the UI, render them as `obsidian://open?vault=<name>&file=<path>` links so they open directly in Obsidian with one click

#### 5.3 — Settings Panel
- [ ] A UI settings page where the user can configure:
  - Vault path
  - Ollama base URL
  - Model selection (for both LLM and embeddings)
  - Default agent permissions (always ask / always allow / always deny)
  - Number of retrieved context chunks (top_k)

#### 5.4 — Tag & Note Browser
- [ ] A read-only panel in the UI showing:
  - All tags in the vault with note counts
  - Recent notes
  - Orphan notes (notes with no incoming wikilinks) — useful for identifying forgotten content

---

## 🔭 Phase 6 — Advanced Features & Automation
**Status: Future / Continuous**

These are longer-term ideas that will be built after the core system is stable.

### 6.1 — Habit Tracking & Personalized Insights
- [ ] Analyze daily notes for habit patterns (e.g., "I notice you've been studying Rust for 3 days without review — shall I generate a quiz?")
- [ ] Weekly/monthly summary generation: The agent reads all weekly notes and creates a concise progress report
- [ ] Goal tracking: If you define goals in a note, the agent monitors your daily notes for progress evidence

### 6.2 — Study & Learning Assistance
- [ ] **Spaced Repetition Suggestions:** Identify notes that haven't been visited in a while and flag them for review
- [ ] **Concept Explanation Mode:** If a term in your notes is unclear, ask the agent to expand on it with examples — and optionally save the explanation back to the note
- [ ] **Flashcard Generation:** From a study note, generate Q&A pairs and save them as a linked note or export to Anki format

### 6.3 — Note Graph Intelligence
- [ ] Analyze the Obsidian note graph for missing connections: "Your note on `Machine Learning` and your note on `Statistics` reference the same concepts but are not linked."
- [ ] Suggest new notes based on gap analysis: "You have notes on A and C, but no note on B which connects them."

### 6.4 — Proactive Agent Mode
- [ ] Instead of only reacting to user questions, the agent can run scheduled background tasks:
  - Every morning: Summarize yesterday's notes and suggest today's focus
  - Every week: Generate a knowledge graph report showing what topics you've explored

### 6.5 — Plugin / Extension System
- [ ] Allow adding custom Python "skills" or tools to the agent without modifying core code
- [ ] Example skills: Calendar integration, web search (via DuckDuckGo), weather lookup for daily notes

---

## 📝 Notes & Principles

- **Privacy First:** Nothing is sent to external APIs. All models run 100% locally via Ollama.
- **Reversibility:** No change to the vault is permanent. Every agent write operation has a Git backup.
- **Modularity:** Each component (ingestion, retrieval, agent, UI) is independent and can be upgraded or replaced.
- **Multilingual:** The system is designed from the ground up to handle Azerbaijani, English, and mixed-language notes naturally.
- **Lightweight:** The entire stack runs on a consumer machine. No GPU required (though it speeds things up significantly).
