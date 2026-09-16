# Setup Guide

This document walks you through setting up the Obsidian Local AI Agent from scratch on a Linux (Debian/Ubuntu) machine.

## Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.11+ | `python3 --version` |
| Ollama | Any recent | `ollama --version` |
| Git | Any | `git --version` |

## Step 1: Clone / Download the Project

```bash
cd ~/Documents/simpleProjects
# Already done — you are in LLM2/
```

## Step 2: Pull Required Ollama Models

Make sure Ollama is running and the two required models are available:

```bash
# Start Ollama in the background (if not already running as a service)
ollama serve &

# Pull the language + vision model (~1.9 GB)
ollama pull qwen3-vl:2b-instruct

# Pull the embedding model (~274 MB)
ollama pull nomic-embed-text:latest

# Verify both are listed
ollama list
```

## Step 3: Create Python Virtual Environment

```bash
cd ~/Documents/simpleProjects/LLM2

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Step 4: Configure Your Environment

```bash
cp .env.example .env
```

Open `.env` and at minimum set your Obsidian vault path:

```
VAULT_PATH=/home/yourname/Documents/YourVaultName
TIMEZONE=Asia/Baku
```

## Step 5: Run Connection Tests

```bash
source .venv/bin/activate

# Test that Ollama + both models are working
python scripts/test_ollama.py

# Test that ChromaDB can write and query locally
python scripts/test_chromadb.py
```

Both should end with a green "✅ All tests passed!" message.

## Step 6: Initialise Git in Your Vault (for Undo Support)

```bash
cd /path/to/your/obsidian/vault
git init
git add -A
git commit -m "Baseline — before AI Agent integration"
cd -
```

This is what enables the revert/undo system if the agent ever makes a change you don't like.

## Step 7: Index Your Vault

Once Phase 2 is implemented:

```bash
python -m src.ingestion.embedder --vault $VAULT_PATH
```

Optional, in a separate terminal, start continuous indexing while you edit notes:

```bash
python -m src.ingestion.watcher --vault $VAULT_PATH
```

## Step 8: Launch the Agent

```bash
python -m src.ui.app
# Opens a web UI at http://localhost:7860
```

The agent can answer direct time/date questions using `TIMEZONE`; for example: `İndi saat neçədir?`.

---

## Troubleshooting

### "Cannot reach Ollama at http://localhost:11434"
Ollama is not running. Start it:
```bash
ollama serve
```

### "VAULT_PATH does not exist"
Edit your `.env` file and set the correct absolute path to your Obsidian vault directory.

### Slow responses
`qwen3-vl:2b-instruct` is a 2B parameter model. It runs fine on CPU, but is significantly faster with a GPU. You can also try reducing `RETRIEVAL_TOP_K` in `.env` to give the model a smaller context.
