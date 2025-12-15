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
- Response length
- Ground-truth metrics (when GT is available):
    - token F1
    - cosine similarity over BoW
    - BERTScore-F1 (optional, if bert-score is installed)
    - ROUGE-L F1
"""

import asyncio
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json

from .multi_mode_assistant import MultiModeResearchAssistant, ResearchMode
from .utils.logger import get_logger

from .benchmark_accuracy import (
    token_f1,
    semantic_cosine,
    bert_score_f1,
    rouge_l_f1,
)

logger = get_logger(__name__)


# ============================================================================
# Data classes
# ============================================================================


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

    # Ground-truth metrics (per query, per mode)
    gt_f1: Optional[float] = None
    gt_semantic: Optional[float] = None
    gt_bert: Optional[float] = None
    gt_rouge_l: Optional[float] = None


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

    # GT metrics at mode level (averaged)
    avg_f1: Optional[float] = None
    avg_semantic: Optional[float] = None
    avg_bert: Optional[float] = None
    avg_rouge_l: Optional[float] = None


# ============================================================================
# Benchmark harness
# ============================================================================


class ResearchBenchmark:
    """
    Benchmark harness for evaluating research assistant performance
    across different modes.
    """

    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.assistant = MultiModeResearchAssistant(openai_api_key, tavily_api_key)
        self.results: List[BenchmarkResult] = []

    async def run_single_query(
        self,
        query: str,
        mode: ResearchMode,
        max_papers: int = 10,
        gold_answer: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run a single query in the specified mode and collect metrics.

        If gold_answer is provided ⇒ also compute GT metrics.
        """
        logger.info(f"Benchmarking query in {mode} mode: {query}")

        start_time = time.time()

        gt_f1_val = gt_sem_val = gt_bert_val = gt_rouge_val = None

        try:
            result = await self.assistant.answer_query(
                query=query, mode=mode, max_papers=max_papers
            )

            latency = time.time() - start_time
            answer = result.get("answer", "")

            verification = result.get("verification_details") or {}
            halluc_ratio = verification.get("hallucination_ratio")
            verification_iterations = verification.get("iterations")

            # Ground-truth metrics if we have a gold answer
            if gold_answer is not None:
                gt_f1_val = token_f1(answer, gold_answer)
                gt_sem_val = semantic_cosine(answer, gold_answer)
                gt_bert_val = bert_score_f1(answer, gold_answer)
                gt_rouge_val = rouge_l_f1(answer, gold_answer)

            benchmark_result = BenchmarkResult(
                query=query,
                mode=mode.value,
                latency_seconds=latency,
                confidence=result.get("confidence"),
                num_sources=len(result.get("sources", [])),
                answer_length=len(answer),
                hallucination_ratio=halluc_ratio,
                verification_iterations=verification_iterations,
                timestamp=datetime.utcnow().isoformat(),
                gt_f1=gt_f1_val,
                gt_semantic=gt_sem_val,
                gt_bert=gt_bert_val,
                gt_rouge_l=gt_rouge_val,
            )

        except Exception as e:
            logger.error(f"Error during benchmark: {str(e)}")
            latency = time.time() - start_time

            benchmark_result = BenchmarkResult(
                query=query,
                mode=mode.value,
                latency_seconds=latency,
                confidence=None,
                num_sources=0,
                answer_length=0,
                timestamp=datetime.utcnow().isoformat(),
                error=str(e),
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
        Regular “open” benchmark: uses arbitrary queries (no GT).
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

        for query in queries:
            for mode in modes:
                await self.run_single_query(query, mode, max_papers=max_papers)
                await asyncio.sleep(1)

        total_time = time.time() - total_start

        summaries: Dict[str, BenchmarkSummary] = {}
        for mode in modes:
            mode_results = [r for r in self.results if r.mode == mode.value]
            summaries[mode.value] = self._compute_summary(mode_results, total_time)

        logger.info(f"Benchmark complete in {total_time:.2f}s")
        return summaries

    async def run_addhealth_benchmark(
        self,
        jsonl_path: str,
        modes: Optional[List[ResearchMode]] = None,
        max_papers: int = 10,
    ) -> Dict[str, BenchmarkSummary]:
        """
        Special GT benchmark using the Add Health JSONL file.

        JSONL format (one per line):
            {
                "id": "...",
                "dataset": "addhealth",
                "query": "...",
                "gold_answer": "...",
                "meta": {...}
            }
        Only `query` and `gold_answer` are used here.
        """
        if modes is None:
            modes = [
                ResearchMode.SIMPLE_WEB_RAG,
                ResearchMode.MCP_BASIC,
                ResearchMode.MCP_VERIFIED,
            ]

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"GT file not found: {jsonl_path}")

        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("query", "")
                gold = obj.get("gold_answer", "")
                if not q or not gold:
                    continue
                items.append((obj.get("id"), q, gold))

        logger.info(
            f"Running Add Health benchmark on {len(items)} QA pairs × {len(modes)} modes"
        )

        total_start = time.time()

        for ex_id, q, gold in items:
            for mode in modes:
                await self.run_single_query(
                    q,
                    mode,
                    max_papers=max_papers,
                    gold_answer=gold,
                )
                await asyncio.sleep(1)

        total_time = time.time() - total_start

        summaries: Dict[str, BenchmarkSummary] = {}
        for mode in modes:
            mode_results = [r for r in self.results if r.mode == mode.value]
            summaries[mode.value] = self._compute_summary(mode_results, total_time)

        logger.info(f"Add Health GT benchmark complete in {total_time:.2f}s")
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

        f1_vals = [r.gt_f1 for r in results if r.gt_f1 is not None]
        sem_vals = [r.gt_semantic for r in results if r.gt_semantic is not None]
        bert_vals = [r.gt_bert for r in results if r.gt_bert is not None]
        rouge_vals = [r.gt_rouge_l for r in results if r.gt_rouge_l is not None]

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
            avg_f1=sum(f1_vals) / len(f1_vals) if f1_vals else None,
            avg_semantic=sum(sem_vals) / len(sem_vals) if sem_vals else None,
            avg_bert=sum(bert_vals) / len(bert_vals) if bert_vals else None,
            avg_rouge_l=sum(rouge_vals) / len(rouge_vals) if rouge_vals else None,
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
            if result.gt_f1 is not None:
                print(f"F1 (GT): {result.gt_f1:.3f}")
            if result.gt_semantic is not None:
                print(f"Cosine (GT): {result.gt_semantic:.3f}")
            if result.gt_bert is not None:
                print(f"BERTScore-F1 (GT): {result.gt_bert:.3f}")
            if result.gt_rouge_l is not None:
                print(f"ROUGE-L F1 (GT): {result.gt_rouge_l:.3f}")

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
            if summary.avg_f1 is not None:
                print(f"Avg F1 (GT): {summary.avg_f1:.3f}")
            if summary.avg_semantic is not None:
                print(f"Avg Cosine (GT): {summary.avg_semantic:.3f}")
            if summary.avg_bert is not None:
                print(f"Avg BERTScore-F1 (GT): {summary.avg_bert:.3f}")
            if summary.avg_rouge_l is not None:
                print(f"Avg ROUGE-L F1 (GT): {summary.avg_rouge_l:.3f}")

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
        """Generate a markdown comparison table (latency + GT cols if present)"""
        table = "| Metric | Simple Web-RAG | MCP-Basic | MCP-Verified |\n"
        table += "|--------|----------------|-----------|---------------|\n"

        simple = summaries.get(ResearchMode.SIMPLE_WEB_RAG.value)
        basic = summaries.get(ResearchMode.MCP_BASIC.value)
        verified = summaries.get(ResearchMode.MCP_VERIFIED.value)

        if simple and basic and verified:
            table += (
                f"| Avg Latency (s) | {simple.avg_latency:.2f} | "
                f"{basic.avg_latency:.2f} | {verified.avg_latency:.2f} |\n"
            )
            table += (
                f"| Avg Sources | {simple.avg_sources:.1f} | "
                f"{basic.avg_sources:.1f} | {verified.avg_sources:.1f} |\n"
            )
            table += (
                f"| Success Rate | {simple.success_rate:.1%} | "
                f"{basic.success_rate:.1%} | {verified.success_rate:.1%} |\n"
            )

            if (
                simple.avg_f1 is not None
                and basic.avg_f1 is not None
                and verified.avg_f1 is not None
            ):
                table += (
                    f"| Avg F1 (GT) | {simple.avg_f1:.3f} | "
                    f"{basic.avg_f1:.3f} | {verified.avg_f1:.3f} |\n"
                )
            if (
                simple.avg_semantic is not None
                and basic.avg_semantic is not None
                and verified.avg_semantic is not None
            ):
                table += (
                    f"| Avg Cosine (GT) | {simple.avg_semantic:.3f} | "
                    f"{basic.avg_semantic:.3f} | {verified.avg_semantic:.3f} |\n"
                )
            if (
                simple.avg_bert is not None
                and basic.avg_bert is not None
                and verified.avg_bert is not None
            ):
                table += (
                    f"| Avg BERTScore-F1 (GT) | {simple.avg_bert:.3f} | "
                    f"{basic.avg_bert:.3f} | {verified.avg_bert:.3f} |\n"
                )

        return table


# ============================================================================
# Quick helpers (optional CLI)
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

    queries = DEFAULT_TEST_QUERIES[:2]

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