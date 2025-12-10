# api_multimode.py

"""
FastAPI Multi-Mode Research Assistant

Endpoints:
    POST /research - Run research query in specified mode
    POST /compare - Run same query across all modes
    POST /benchmark - Run benchmark suite
    GET /health - Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from dotenv import load_dotenv

from src.multi_mode_assistant import MultiModeResearchAssistant, ResearchMode
from src.benchmark import ResearchBenchmark, BenchmarkSummary
from src.utils.logger import get_logger

load_dotenv()

app = FastAPI(
    title="Multi-Mode Research Assistant API",
    description="Research assistant with 3 operational modes: Simple Web-RAG, MCP-Basic, and MCP-Verified",
    version="2.0.0",
)

logger = get_logger(__name__)

# Initialize assistant
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
    max_papers: int = Field(default=10, ge=1, le=20, description="Max papers to retrieve")


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
    max_papers: int = Field(default=10, ge=1, le=20, description="Max papers to retrieve")


class CompareResponse(BaseModel):
    query: str
    results: dict  # mode -> ResearchResponse


class BenchmarkRequest(BaseModel):
    queries: List[str] = Field(
        ..., min_items=1, max_items=10, description="List of queries to benchmark"
    )
    modes: Optional[List[str]] = Field(
        default=None, description="Modes to test (defaults to all)"
    )
    max_papers: int = Field(default=10, ge=1, le=20, description="Max papers per query")


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multi-Mode Research Assistant",
        "version": "2.0.0",
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
            detail=f"Invalid mode: {request.mode}. Must be one of: simple_web_rag, mcp_basic, mcp_verified",
        )

    logger.info(f"Processing research request in {mode} mode: {request.query}")

    try:
        result = await assistant.answer_query(
            query=request.query, mode=mode, max_papers=request.max_papers
        )
        return ResearchResponse(**result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=CompareResponse)
async def compare_modes(request: CompareRequest):
    """
    Run the same query across all 3 modes for comparison.

    Returns results from all modes with timing and quality metrics.
    """
    logger.info(f"Running comparative analysis for: {request.query}")

    results = {}

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
            logger.error(f"Error in {mode} mode: {str(e)}")
            results[mode.value] = {
                "error": str(e),
                "mode": mode.value,
                "query": request.query,
            }

    return CompareResponse(query=request.query, results=results)


@app.post("/benchmark")
async def benchmark(request: BenchmarkRequest):
    """
    Run benchmark suite across multiple queries and modes.

    Returns aggregated performance metrics including:
    - Average latency per mode
    - Confidence scores (where applicable)
    - Success rates
    - Source quality metrics
    """
    logger.info(
        f"Starting benchmark with {len(request.queries)} queries "
        f"across {len(request.modes) if request.modes else 3} modes"
    )

    benchmark_runner = ResearchBenchmark(OPENAI_KEY, TAVILY_KEY)

    # Parse modes
    modes = None
    if request.modes:
        try:
            modes = [ResearchMode(m) for m in request.modes]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        summaries = await benchmark_runner.run_comparative_benchmark(
            queries=request.queries, modes=modes, max_papers=request.max_papers
        )

        # Convert summaries to dict format
        summaries_dict = {
            mode: {
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
            }
            for mode, summary in summaries.items()
        }

        # Also include individual results
        individual_results = [
            {
                "query": r.query,
                "mode": r.mode,
                "latency_seconds": r.latency_seconds,
                "confidence": r.confidence,
                "num_sources": r.num_sources,
                "answer_length": r.answer_length,
                "error": r.error,
            }
            for r in benchmark_runner.results
        ]

        return {
            "summaries": summaries_dict,
            "individual_results": individual_results,
            "comparison_table": benchmark_runner.get_comparison_table(summaries),
        }

    except Exception as e:
        logger.error(f"Error during benchmark: {str(e)}")
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
