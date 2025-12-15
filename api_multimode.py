# api_multimode.py

"""
FastAPI Multi-Mode Research Assistant

Endpoints:
    POST /research  - Run research query in specified mode
    POST /compare   - Run same query across all modes
    POST /benchmark - Run benchmark suite (open or GT AddHealth)
    GET  /health    - Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

from src.multi_mode_assistant import MultiModeResearchAssistant, ResearchMode
from src.benchmark import ResearchBenchmark, BenchmarkSummary
from src.utils.logger import get_logger

load_dotenv()

app = FastAPI(
    title="Multi-Mode Research Assistant API",
    description=(
        "Research assistant with 3 operational modes: "
        "Simple Web-RAG, MCP-Basic, and MCP-Verified"
    ),
    version="2.1.0",
)

# CORS for the React app on Vite dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")

if not OPENAI_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

assistant = MultiModeResearchAssistant(OPENAI_KEY, TAVILY_KEY)


# ============================================================================
# Request/Response Models
# ============================================================================

class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research question to answer")
    mode: str = Field(
        default="mcp_verified",
        description="Mode: simple_web_rag, mcp_basic, or mcp_verified",
    )
    max_papers: int = Field(
        default=10, ge=1, le=20, description="Max papers to retrieve"
    )


class ResearchResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    sources: List[dict]
    query: str
    mode: str
    metrics: dict
    verification_details: Optional[dict] = None


class CompareRequest(BaseModel):
    query: str = Field(..., description="Research question to answer")
    max_papers: int = Field(
        default=10, ge=1, le=20, description="Max papers to retrieve"
    )


class CompareResponse(BaseModel):
    query: str
    results: Dict[str, dict]  # mode -> ResearchResponse or error payload


class BenchmarkRequest(BaseModel):
    """
    BenchmarkRequest

    eval_set:
        - "open"      → use `queries` provided
        - "addhealth" → ignore `queries` and use Add Health GT JSONL
    """
    eval_set: Optional[str] = Field(
        default="open",
        description="Benchmark type: 'open' or 'addhealth'",
    )
    queries: Optional[List[str]] = Field(
        default=None,
        description="List of free-form queries (required for eval_set='open')",
    )
    modes: Optional[List[str]] = Field(
        default=None, description="Modes to test (defaults to all 3)"
    )
    max_papers: int = Field(
        default=10, ge=1, le=20, description="Max papers per query"
    )


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multi-Mode Research Assistant",
        "version": "2.1.0",
    }


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """
    Run research query in specified mode.

    Modes:
    - simple_web_rag: Direct web search + basic summarization
    - mcp_basic: MCP tools without verification
    - mcp_verified: Full pipeline with verification loop
    """
    try:
        mode = ResearchMode(request.mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid mode: {request.mode}. Must be one of: "
                "simple_web_rag, mcp_basic, mcp_verified"
            ),
        )

    logger.info(f"Processing research request in {mode} mode: {request.query}")

    try:
        result = await assistant.answer_query(
            query=request.query, mode=mode, max_papers=request.max_papers
        )
        return ResearchResponse(**result)
    except Exception as e:
        logger.error(f"Error processing /research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare_modes(request: CompareRequest):
    """
    Run the same query across all 3 modes for comparison.

    Returns results from all modes with timing and quality metrics.
    """
    logger.info(f"Running comparative analysis for: {request.query}")

    results: Dict[str, dict] = {}

    for mode in [
        ResearchMode.SIMPLE_WEB_RAG,
        ResearchMode.MCP_BASIC,
        ResearchMode.MCP_VERIFIED,
    ]:
        try:
            result = await assistant.answer_query(
                query=request.query, mode=mode, max_papers=request.max_papers
            )
            results[mode.value] = result
        except Exception as e:
            logger.error(f"Error in {mode.value} mode: {str(e)}")
            results[mode.value] = {
                "error": str(e),
                "mode": mode.value,
                "query": request.query,
            }

    return CompareResponse(query=request.query, results=results)


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """
    Run benchmark suite.

    Modes:
    - eval_set="open":
        uses free-form `queries` from the request (no ground truth)
    - eval_set="addhealth":
        uses Add Health GT JSONL specified by ADDHEALTH_GT_PATH env var
        (default: scripts/gt_addhealth_qa.jsonl)
    """
    # Parse modes into ResearchMode enums
    modes_enum: Optional[List[ResearchMode]] = None
    if request.modes:
        try:
            modes_enum = [ResearchMode(m) for m in request.modes]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"Starting benchmark: eval_set={request.eval_set}, "
        f"queries={len(request.queries or [])}, "
        f"modes={request.modes or 'ALL'}"
    )

    benchmark_runner = ResearchBenchmark(OPENAI_KEY, TAVILY_KEY)

    try:
        # Ground-truth AddHealth benchmark
        if request.eval_set == "addhealth":
            gt_path = os.getenv("ADDHEALTH_GT_PATH", "scripts/gt_addhealth_qa.jsonl")
            summaries = await benchmark_runner.run_addhealth_benchmark(
                jsonl_path=gt_path,
                modes=modes_enum,
                max_papers=request.max_papers,
            )

        # Open benchmark using ad-hoc queries
        else:
            if not request.queries:
                raise HTTPException(
                    status_code=400,
                    detail="For open benchmark, please provide at least one query in 'queries'.",
                )

            summaries = await benchmark_runner.run_comparative_benchmark(
                queries=request.queries,
                modes=modes_enum,
                max_papers=request.max_papers,
            )

        # Convert summaries to primitive dicts
        summaries_dict = {
            mode_name: {
                "mode": summary.mode,
                "num_queries": summary.num_queries,
                "avg_latency": summary.avg_latency,
                "min_latency": summary.min_latency,
                "max_latency": summary.max_latency,
                "avg_confidence": summary.avg_confidence,
                "avg_sources": summary.avg_sources,
                "avg_answer_length": summary.avg_answer_length,
                "success_rate": summary.success_rate,
                "total_time": summary.total_time,
                # GT metrics (A / B / C style)
                "avg_f1": summary.avg_f1,
                "avg_semantic": summary.avg_semantic,
                "avg_bert": summary.avg_bert,
                "avg_rouge_l": summary.avg_rouge_l,
            }
            for mode_name, summary in summaries.items()
        }

        # Individual per-query results
        individual_results = [
            {
                "query": r.query,
                "mode": r.mode,
                "latency_seconds": r.latency_seconds,
                "confidence": r.confidence,
                "num_sources": r.num_sources,
                "answer_length": r.answer_length,
                "error": r.error,
                # GT metrics exposed to UI if needed
                "gt_f1": r.gt_f1,
                "gt_semantic": r.gt_semantic,
                "gt_bert": r.gt_bert,
                "gt_rouge_l": r.gt_rouge_l,
            }
            for r in benchmark_runner.results
        ]

        return {
            "summaries": summaries_dict,
            "results": individual_results,
            "comparison_table": benchmark_runner.get_comparison_table(summaries),
        }

    except HTTPException:
        # Bubble up our own HTTP errors
        raise
    except Exception as e:
        logger.error(f"Error during /benchmark: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Multi-Mode Research Assistant API starting up")
    logger.info(f"Available modes: {[m.value for m in ResearchMode]}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Multi-Mode Research Assistant API shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)