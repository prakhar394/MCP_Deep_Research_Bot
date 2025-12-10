# src/benchmark.py

"""
Benchmarking Module for Multi-Mode Research Assistant

Evaluates performance across 3 modes:
- Simple Web-RAG
- MCP-Basic
- MCP-Verified

Metrics:
- Latency (execution time)
- Confidence scores (where applicable)
- Source quality
- Response completeness
"""

import asyncio
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from .multi_mode_assistant import MultiModeResearchAssistant, ResearchMode
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    query: str
    mode: str
    latency_seconds: float
    confidence: Optional[float]
    num_sources: int
    answer_length: int
    timestamp: str
    hallucination_ratio: Optional[float] = None
    verification_iterations: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Aggregated results across multiple queries and modes"""

    mode: str
    num_queries: int
    avg_latency: float
    min_latency: float
    max_latency: float
    avg_confidence: Optional[float]
    avg_sources: float
    avg_answer_length: float
    success_rate: float
    total_time: float


class ResearchBenchmark:
    """
    Benchmark harness for evaluating research assistant performance
    across different modes.
    """

    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.assistant = MultiModeResearchAssistant(openai_api_key, tavily_api_key)
        self.results: List[BenchmarkResult] = []

    async def run_single_query(
        self, query: str, mode: ResearchMode, max_papers: int = 10
    ) -> BenchmarkResult:
        """
        Run a single query in the specified mode and collect metrics.
        """
        logger.info(f"Benchmarking query in {mode} mode: {query}")

        start_time = time.time()
        error = None

        try:
            result = await self.assistant.answer_query(
                query=query, mode=mode, max_papers=max_papers
            )

            latency = time.time() - start_time
            answer = result.get("answer", "")

            # Extract verification details if available
            verification = result.get("verification_details", {})
            halluc_ratio = verification.get("hallucination_ratio")

            benchmark_result = BenchmarkResult(
                query=query,
                mode=mode.value,
                latency_seconds=latency,
                confidence=result.get("confidence"),
                num_sources=len(result.get("sources", [])),
                answer_length=len(answer),
                hallucination_ratio=halluc_ratio,
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error during benchmark: {str(e)}")
            latency = time.time() - start_time
            error = str(e)

            benchmark_result = BenchmarkResult(
                query=query,
                mode=mode.value,
                latency_seconds=latency,
                confidence=None,
                num_sources=0,
                answer_length=0,
                timestamp=datetime.utcnow().isoformat(),
                error=error,
            )

        self.results.append(benchmark_result)
        return benchmark_result

    async def run_comparative_benchmark(
        self,
        queries: List[str],
        modes: Optional[List[ResearchMode]] = None,
        max_papers: int = 10,
    ) -> Dict[str, BenchmarkSummary]:
        """
        Run comparative benchmark across all modes for multiple queries.

        Args:
            queries: List of research queries to test
            modes: List of modes to test (defaults to all 3)
            max_papers: Max papers per query

        Returns:
            Dict mapping mode name to BenchmarkSummary
        """
        if modes is None:
            modes = [
                ResearchMode.SIMPLE_WEB_RAG,
                ResearchMode.MCP_BASIC,
                ResearchMode.MCP_VERIFIED,
            ]

        logger.info(
            f"Starting comparative benchmark: {len(queries)} queries × {len(modes)} modes"
        )

        total_start = time.time()

        # Run all combinations
        for query in queries:
            for mode in modes:
                await self.run_single_query(query, mode, max_papers)
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

        total_time = time.time() - total_start

        # Compute summaries
        summaries = {}
        for mode in modes:
            mode_results = [r for r in self.results if r.mode == mode.value]
            summaries[mode.value] = self._compute_summary(mode_results, total_time)

        logger.info(f"Benchmark complete in {total_time:.2f}s")
        return summaries

    def _compute_summary(
        self, results: List[BenchmarkResult], total_time: float
    ) -> BenchmarkSummary:
        """Compute aggregate statistics for a set of results"""
        if not results:
            return BenchmarkSummary(
                mode="unknown",
                num_queries=0,
                avg_latency=0.0,
                min_latency=0.0,
                max_latency=0.0,
                avg_confidence=None,
                avg_sources=0.0,
                avg_answer_length=0.0,
                success_rate=0.0,
                total_time=total_time,
            )

        latencies = [r.latency_seconds for r in results]
        confidences = [r.confidence for r in results if r.confidence is not None]
        sources = [r.num_sources for r in results]
        answer_lengths = [r.answer_length for r in results]
        successes = [r for r in results if r.error is None]

        return BenchmarkSummary(
            mode=results[0].mode,
            num_queries=len(results),
            avg_latency=sum(latencies) / len(latencies),
            min_latency=min(latencies),
            max_latency=max(latencies),
            avg_confidence=sum(confidences) / len(confidences) if confidences else None,
            avg_sources=sum(sources) / len(sources),
            avg_answer_length=sum(answer_lengths) / len(answer_lengths),
            success_rate=len(successes) / len(results),
            total_time=total_time,
        )

    def print_results(self):
        """Print formatted benchmark results"""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS - INDIVIDUAL QUERIES")
        print("=" * 80)

        for result in self.results:
            print(f"\nQuery: {result.query}")
            print(f"Mode: {result.mode}")
            print(f"Latency: {result.latency_seconds:.2f}s")
            print(f"Sources: {result.num_sources}")
            print(f"Answer Length: {result.answer_length} chars")

            if result.confidence is not None:
                print(f"Confidence: {result.confidence:.2%}")
            if result.hallucination_ratio is not None:
                print(f"Hallucination Ratio: {result.hallucination_ratio:.2%}")
            if result.error:
                print(f"ERROR: {result.error}")

            print("-" * 80)

    def print_summary(self, summaries: Dict[str, BenchmarkSummary]):
        """Print comparative summary across modes"""
        print("\n" + "=" * 80)
        print("COMPARATIVE SUMMARY ACROSS MODES")
        print("=" * 80)

        for mode_name, summary in summaries.items():
            print(f"\n{mode_name.upper()}")
            print("-" * 80)
            print(f"Queries: {summary.num_queries}")
            print(f"Avg Latency: {summary.avg_latency:.2f}s")
            print(f"Latency Range: {summary.min_latency:.2f}s - {summary.max_latency:.2f}s")
            print(f"Avg Sources: {summary.avg_sources:.1f}")
            print(f"Avg Answer Length: {summary.avg_answer_length:.0f} chars")
            print(f"Success Rate: {summary.success_rate:.1%}")

            if summary.avg_confidence is not None:
                print(f"Avg Confidence: {summary.avg_confidence:.2%}")

        print("\n" + "=" * 80)

    def export_results(self, filename: str = "benchmark_results.json"):
        """Export results to JSON file"""
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "results": [asdict(r) for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Results exported to {filename}")

    def get_comparison_table(self, summaries: Dict[str, BenchmarkSummary]) -> str:
        """Generate a markdown comparison table"""
        table = "| Metric | Simple Web-RAG | MCP-Basic | MCP-Verified |\n"
        table += "|--------|----------------|-----------|---------------|\n"

        # Get summaries in order
        simple = summaries.get(ResearchMode.SIMPLE_WEB_RAG.value)
        basic = summaries.get(ResearchMode.MCP_BASIC.value)
        verified = summaries.get(ResearchMode.MCP_VERIFIED.value)

        if simple and basic and verified:
            table += f"| Avg Latency (s) | {simple.avg_latency:.2f} | {basic.avg_latency:.2f} | {verified.avg_latency:.2f} |\n"
            table += f"| Min Latency (s) | {simple.min_latency:.2f} | {basic.min_latency:.2f} | {verified.min_latency:.2f} |\n"
            table += f"| Max Latency (s) | {simple.max_latency:.2f} | {basic.max_latency:.2f} | {verified.max_latency:.2f} |\n"
            table += f"| Avg Sources | {simple.avg_sources:.1f} | {basic.avg_sources:.1f} | {verified.avg_sources:.1f} |\n"
            table += f"| Success Rate | {simple.success_rate:.1%} | {basic.success_rate:.1%} | {verified.success_rate:.1%} |\n"

            # Confidence only for verified
            if verified.avg_confidence:
                table += f"| Avg Confidence | N/A | N/A | {verified.avg_confidence:.2%} |\n"

        return table


# ============================================================================
# Predefined Test Queries
# ============================================================================

DEFAULT_TEST_QUERIES = [
    "What are recent advances in transformer efficiency?",
    "How does RLHF improve language model alignment?",
    "What are the latest techniques for LLM quantization?",
    "What is retrieval augmented generation (RAG)?",
    "How do vision transformers differ from CNNs?",
]


async def run_quick_benchmark(openai_key: str, tavily_key: str):
    """Quick benchmark with a small set of queries"""
    benchmark = ResearchBenchmark(openai_key, tavily_key)

    queries = DEFAULT_TEST_QUERIES[:2]  # Just 2 queries for quick test

    summaries = await benchmark.run_comparative_benchmark(queries, max_papers=8)

    benchmark.print_results()
    benchmark.print_summary(summaries)
    benchmark.export_results("quick_benchmark_results.json")

    print("\n" + benchmark.get_comparison_table(summaries))


async def run_full_benchmark(openai_key: str, tavily_key: str):
    """Full benchmark with all test queries"""
    benchmark = ResearchBenchmark(openai_key, tavily_key)

    summaries = await benchmark.run_comparative_benchmark(
        DEFAULT_TEST_QUERIES, max_papers=10
    )

    benchmark.print_results()
    benchmark.print_summary(summaries)
    benchmark.export_results("full_benchmark_results.json")

    print("\n" + benchmark.get_comparison_table(summaries))
