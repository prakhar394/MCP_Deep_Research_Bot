# 🧠 MCP Research Assistant

A modular, verifiable, multi-agent research system using the **Model Context Protocol (MCP)**, OpenAI tools, semantic verification, arXiv retrieval, and FastAPI.

---

## 🚀 Overview

This project implements a **production-grade research agent** that can:

* Retrieve academic papers using MCP tools (`arxiv_search`, `web_search`, `fetch_paper`)
* Summarize them using an LLM
* Extract atomic factual claims
* Verify each claim using:

  * Semantic similarity
  * An NLI model (`deberta-large-mnli`)
  * Optional external search
* Iterate until a high-confidence answer is produced
* Serve everything over a clean **FastAPI API**

You get:

* Research retrieval
* Automated synthesis
* Structured verification
* Deterministic refinement
* Full logs
* Plug-and-play backend

---

## 🏗️ System Architecture

```
User Query
    ↓
FastAPI (api.py)
    ↓
MCPResearchAssistant
    ↓
├── MCPRetrieverAgent
│     └── arxiv_search, embeddings
├── SummarizerAgent
│     └── openai summarization
└── ThoroughMCPVerifier
      ├── semantic verification
      ├── NLI entailment
      └── confidence scoring
    ↓
Final Verified Answer
```

---

## 📁 Project Structure

```
mcp-research-assistant/
│
├── api.py                     # FastAPI app entrypoint
├── main.py                    # CLI runner (optional)
├── requirements.txt
├── README.md                  ← you are here
├── venv/
│
└── src/
    ├── agents/
    │   ├── base_agent.py
    │   ├── summarizer.py
    │   ├── mcp_retriever.py
    │   └── thorough_mcp_verifier.py
    │
    ├── mcp/
    │   ├── tool_definitions.py
    │   └── tool_executors.py
    │
    ├── utils/
    │   ├── logger.py
    │   └── cache.py
    │
    └── mcp_research_assistant.py
```

---

## 🔧 Installation

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you are missing loguru:

```bash
pip install loguru
```

---

## ⚙️ Environment Variables

Create `.env`:

```
OPENAI_API_KEY=your-key
TAVILY_API_KEY=your-key
```

---

## ▶️ Running the API

Because your FastAPI file is **in the project root** (`api.py`), start the server with:

```bash
python -m uvicorn api:app --reload
```

✔ This ensures uvicorn uses your **virtualenv python** and does not try to import the wrong module.

### DO NOT run:

```bash
uvicorn api:app
```

because this will use the **Anaconda uvicorn**, not the venv one.

---

## ▶️ Running the CLI tool

```bash
python main.py
```

This will execute a full MCP query run:

* arxiv search →
* summarization →
* verification →
* final answer

---

## 🔍 API Endpoints

### `POST /research`

Request:

```json
{
  "query": "What are recent advances in transformer efficiency?"
}
```

Response:

```json
{
  "answer": "... final verified summary ...",
  "confidence": 0.81,
  "sources": [...],
  "verification_details": {...}
}
```

---

## 🧪 How Verification Works

Verification is hybrid:

### 1. **Semantic similarity**

* Claims compared to paper abstracts
* High similarity → more confidence

### 2. **NLI model**

* `microsoft/deberta-large-mnli`
* Determines entailment/contradiction/neutral

### 3. **Aggregate confidence**

Final confidence:

```
0.5 * semantic_score + 0.5 * NLI_score
```

No external web search → fully deterministic, grounded in provided papers.

---

## 🧩 Features

* ✔ MCP tool framework
* ✔ arxiv retrieval
* ✔ LLM summarizer
* ✔ claim extractor
* ✔ semantic verification
* ✔ NLI verification
* ✔ FastAPI interface
* ✔ CLI interface
* ✔ Caching layer
* ✔ Logging with Loguru
* ✔ Modular agent architecture

---
