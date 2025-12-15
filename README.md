# 🧠 Multi-Mode MCP Research Assistant

A research assistant with **3 operational modes** for testing accuracy and latency trade-offs:

1. **Simple Web-RAG** - Direct web search + basic summarization (fastest, no verification)
2. **MCP-Basic** - MCP tools + summarization (moderate speed, no verification)
3. **MCP-Verified** - Full pipeline with verification loop (slowest, highest accuracy)

---

## 🎯 Key Features

- **3 Operational Modes** for performance comparison
- **Benchmarking Suite** for accuracy and latency testing
- **FastAPI Endpoints** for programmatic access
- **CLI Interface** for quick testing
- **Comprehensive Metrics** (latency, confidence, hallucination detection)
- **Source Quality Analysis** across modes

---

## 🏗️ System Architecture

### Mode 1: Simple Web-RAG
```
User Query → Web Search (Tavily) → Basic LLM Summarization → Response
```
- ⚡ **Fastest** (~5-10s)
- ⚠️ No verification
- 🔍 Web search only

### Mode 2: MCP-Basic
```
User Query → MCP Retriever (arXiv + PubMed) → MCP Summarizer → Response
```
- ⚡ **Moderate** (~10-20s)
- ✅ Academic sources
- ⚠️ No verification loop

### Mode 3: MCP-Verified (Original Implementation)
```
User Query → MCP Retriever → MCP Summarizer → Verifier → 
  ↓ (if low confidence)
  Enhanced Retrieval / Revision → Re-verify → Final Response
```
- 🐌 **Slowest** (~20-60s)
- ✅ Academic sources
- ✅ Verification loop
- ✅ Confidence scoring
- ✅ Hallucination detection

---

## 📁 Project Structure

```
mcp-research-assistant/
│
├── main_multimode.py          # CLI entry point with mode selection
├── api_multimode.py            # FastAPI with mode support
├── requirements.txt
├── README_MULTIMODE.md         ← you are here
│
└── src/
    ├── multi_mode_assistant.py # Multi-mode research assistant
    ├── benchmark.py            # Benchmarking framework
    ├── agents/
    │   ├── mcp_retriever.py
    │   ├── summarizer.py
    │   └── thorough_mcp_verifier.py
    ├── mcp/
    │   ├── tool_definitions.py
    │   └── tool_executors.py
    └── utils/
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY and TAVILY_API_KEY to .env
```

### 2. Run Single Query

```bash
# Simple Web-RAG mode (fastest)
python main_multimode.py --query "What is RAG?" --mode simple_web_rag

# MCP-Basic mode (moderate)
python main_multimode.py --query "What is RAG?" --mode mcp_basic

# MCP-Verified mode (most accurate)
python main_multimode.py --query "What is RAG?" --mode mcp_verified
```

### 3. Compare All Modes

```bash
# Run same query in all 3 modes
python main_multimode.py --query "What are recent advances in transformer efficiency?" --all-modes
```

### 4. Run Benchmarks

```bash
# Quick benchmark (2 queries × 3 modes)
python main_multimode.py --benchmark quick

# Full benchmark (5 queries × 3 modes)
python main_multimode.py --benchmark full
```

---

## 🔌 API Usage

### Start API Server

```bash
python -m uvicorn api_multimode:app --reload
```

### Endpoints

#### 1. Single Mode Research

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "mode": "simple_web_rag",
    "max_papers": 10
  }'
```

#### 2. Compare All Modes

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "max_papers": 10
  }'
```

#### 3. Run Benchmark

```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is RAG?",
      "How does RLHF work?"
    ],
    "max_papers": 10
  }'
```

---

## 📊 Benchmarking

### Metrics Collected

For each query and mode, the benchmark collects:

- **Latency**: Total execution time (seconds)
- **Confidence**: Verifier confidence score (MCP-Verified only)
- **Source Count**: Number of sources retrieved
- **Answer Length**: Length of generated response
- **Hallucination Ratio**: Estimated hallucination rate (MCP-Verified only)
- **Success Rate**: Percentage of successful completions

### Benchmark Output

```
COMPARATIVE SUMMARY ACROSS MODES
================================================================================

SIMPLE_WEB_RAG
--------------------------------------------------------------------------------
Queries: 5
Avg Latency: 7.45s
Latency Range: 5.23s - 9.87s
Avg Sources: 8.2
Avg Answer Length: 1247 chars
Success Rate: 100.0%

MCP_BASIC
--------------------------------------------------------------------------------
Queries: 5
Avg Latency: 15.67s
Latency Range: 12.34s - 18.91s
Avg Sources: 9.4
Avg Answer Length: 1589 chars
Success Rate: 100.0%

MCP_VERIFIED
--------------------------------------------------------------------------------
Queries: 5
Avg Latency: 34.21s
Latency Range: 25.67s - 42.88s
Avg Sources: 9.6
Avg Answer Length: 1678 chars
Success Rate: 100.0%
Avg Confidence: 92.3%
```

### Comparison Table

| Metric | Simple Web-RAG | MCP-Basic | MCP-Verified |
|--------|----------------|-----------|--------------|
| Avg Latency (s) | 7.45 | 15.67 | 34.21 |
| Min Latency (s) | 5.23 | 12.34 | 25.67 |
| Max Latency (s) | 9.87 | 18.91 | 42.88 |
| Avg Sources | 8.2 | 9.4 | 9.6 |
| Success Rate | 100.0% | 100.0% | 100.0% |
| Avg Confidence | N/A | N/A | 92.3% |

---

## 🧪 Testing Accuracy

### Manual Evaluation

After running benchmarks, manually evaluate:

1. **Factual Accuracy**: Are claims correct?
2. **Source Quality**: Are sources authoritative?
3. **Completeness**: Does it cover key aspects?
4. **Citation Quality**: Are citations accurate? (MCP-Verified only)

### Automated Metrics (MCP-Verified)

- **Confidence Score**: 0-100% verifier confidence
- **Hallucination Ratio**: % of claims without evidence
- **Citation Coverage**: % of claims with citations

---

## 📈 When to Use Each Mode

### Simple Web-RAG
- ✅ Quick answers needed (<10s)
- ✅ General web information sufficient
- ✅ Latency is critical
- ❌ Not for high-stakes decisions
- ❌ Not for academic research

### MCP-Basic
- ✅ Academic sources needed
- ✅ Moderate latency acceptable (~15s)
- ✅ Basic reliability sufficient
- ❌ Not for high-stakes decisions
- ❌ No confidence guarantees

### MCP-Verified
- ✅ High accuracy required
- ✅ Academic sources essential
- ✅ Citations needed
- ✅ Latency not critical (30-60s)
- ✅ Confidence scoring needed
- ✅ High-stakes decisions

---

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required
TAVILY_API_KEY=tvly-...         # Required for web search
```

### Adjusting Parameters

In `src/multi_mode_assistant.py`:

```python
# Verification loop settings
self.max_iterations = 5  # Max verification loops
self.accept_confidence_threshold = 0.97  # Min confidence to accept
```

---

## 📝 Example CLI Session

```bash
$ python main_multimode.py --query "What is retrieval augmented generation?" --all-modes

================================================================================
MODE: SIMPLE_WEB_RAG
================================================================================

📚 Research Summary (SIMPLE_WEB_RAG):
What is retrieval augmented generation?

Retrieval Augmented Generation (RAG) is a technique that enhances large 
language models by retrieving relevant information from external knowledge 
bases before generating responses...

[Sources listed...]

Metrics:
  Latency: 6.23s
  Sources: 8
================================================================================

MODE: MCP_BASIC
================================================================================

📚 Research Summary (MCP_BASIC):
What is retrieval augmented generation?

RAG combines information retrieval with neural text generation. Recent 
papers on arXiv demonstrate improvements in factual accuracy...

[Academic sources listed...]

Metrics:
  Latency: 14.56s
  Sources: 9
================================================================================

MODE: MCP_VERIFIED
================================================================================

📚 Research Summary (MCP-VERIFIED):
What is retrieval augmented generation?

RAG enhances language models through external knowledge retrieval [1,3]. 
Studies show 40% improvement in factual accuracy [2]...

[Academic sources with inline citations...]

Metrics:
  Latency: 31.89s
  Confidence: 94.2%
  Sources: 10
================================================================================

COMPARISON SUMMARY
================================================================================
Mode                 Latency (s)     Sources    Confidence     
--------------------------------------------------------------------------------
simple_web_rag       6.23            8          N/A            
mcp_basic            14.56           9          N/A            
mcp_verified         31.89           10         94.2%          
================================================================================
```

---

## 🤝 Contributing

When adding new modes or features:

1. Add mode to `ResearchMode` enum
2. Implement mode handler in `MultiModeResearchAssistant`
3. Update CLI and API interfaces
4. Add benchmark support
5. Update documentation

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- OpenAI for GPT models
- Tavily for web search API
- arXiv and PubMed for academic sources
- Sentence Transformers for embeddings
- Microsoft DeBERTa for NLI verification
