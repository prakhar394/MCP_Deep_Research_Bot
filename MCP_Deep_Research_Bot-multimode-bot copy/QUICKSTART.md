# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 3. Test All 3 Modes

```bash
python test_modes.py
```

This will run a test query through all 3 modes and show you the differences.

---

## Common Use Cases

### Use Case 1: Quick Answer (Fastest)

When you need a fast answer and don't need verification:

```bash
python main_multimode.py \
  --query "What is RAG?" \
  --mode simple_web_rag
```

**Expected Time:** 5-10 seconds
**Best For:** Quick lookups, general questions

---

### Use Case 2: Academic Research (Moderate Speed)

When you need academic sources but can skip verification:

```bash
python main_multimode.py \
  --query "What are recent advances in transformer efficiency?" \
  --mode mcp_basic
```

**Expected Time:** 10-20 seconds
**Best For:** Academic overviews, literature exploration

---

### Use Case 3: High-Stakes Research (Most Accurate)

When accuracy is critical and you need verification:

```bash
python main_multimode.py \
  --query "What are the latest techniques for LLM quantization?" \
  --mode mcp_verified
```

**Expected Time:** 20-60 seconds
**Best For:** Research papers, critical decisions, high-stakes work

---

### Use Case 4: Compare All Modes

See how all modes perform on the same query:

```bash
python main_multimode.py \
  --query "How does RLHF improve language model alignment?" \
  --all-modes
```

---

### Use Case 5: Benchmark Performance

Run systematic benchmarks to compare accuracy and latency:

```bash
# Quick test (2 queries × 3 modes)
python main_multimode.py --benchmark quick

# Full test (5 queries × 3 modes)
python main_multimode.py --benchmark full
```

Then visualize the results:

```bash
python visualize_benchmark.py
```

---

## API Usage

### Start the API

```bash
python -m uvicorn api_multimode:app --reload
```

### Call the API

#### Simple Web-RAG Mode
```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "mode": "simple_web_rag"}'
```

#### Compare All Modes
```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

#### Run Benchmark
```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is RAG?",
      "How does RLHF work?"
    ]
  }'
```

---

## Understanding the Modes

### 🚀 Simple Web-RAG
- **Speed:** ⚡⚡⚡ (5-10s)
- **Accuracy:** ⭐⭐ (Basic)
- **Sources:** Web search only
- **Verification:** None
- **Use When:** Speed matters most

### 🎯 MCP-Basic
- **Speed:** ⚡⚡ (10-20s)
- **Accuracy:** ⭐⭐⭐ (Good)
- **Sources:** Academic (arXiv, PubMed)
- **Verification:** None
- **Use When:** Need academic sources, moderate speed

### 🔬 MCP-Verified
- **Speed:** ⚡ (20-60s)
- **Accuracy:** ⭐⭐⭐⭐⭐ (Excellent)
- **Sources:** Academic (arXiv, PubMed)
- **Verification:** Full loop with confidence scoring
- **Use When:** Accuracy is critical

---

## Interpreting Results

### Confidence Score (MCP-Verified only)

- **>95%**: High confidence, answer is well-supported
- **85-95%**: Good confidence, mostly reliable
- **70-85%**: Moderate confidence, some uncertainty
- **<70%**: Low confidence, treat with caution

### Hallucination Ratio (MCP-Verified only)

- **<5%**: Excellent, very few unsupported claims
- **5-15%**: Good, some minor issues
- **15-30%**: Concerning, many unsupported claims
- **>30%**: Poor, major reliability issues

---

## Troubleshooting

### "No results found"
- Check your API keys in `.env`
- Try a different query or increase `--max-papers`

### "Request timeout"
- MCP-Verified mode can take 30-60s for complex queries
- Try MCP-Basic or Simple Web-RAG for faster results

### "Confidence is low"
- The query might be too specific or ambiguous
- Try rephrasing the query
- Check if there are relevant papers in the sources

---

## Next Steps

1. **Run benchmarks** to understand performance characteristics
2. **Visualize results** to compare modes
3. **Integrate the API** into your application
4. **Customize settings** in `src/multi_mode_assistant.py`

For detailed documentation, see [README_MULTIMODE.md](README_MULTIMODE.md)
