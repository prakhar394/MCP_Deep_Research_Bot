#!/usr/bin/env python3
# main_multimode.py

"""
Multi-Mode Research Assistant Entry Point

Run research queries in 3 different modes:
1. Simple Web-RAG - Direct web search + basic summarization
2. MCP-Basic - MCP tools without verification
3. MCP-Verified - Full pipeline with verification loop

Usage:
    # Single query with specific mode
    python main_multimode.py --query "What is RAG?" --mode simple_web_rag
    
    # Run all modes for comparison
    python main_multimode.py --query "What is RAG?" --all-modes
    
    # Run benchmarks
    python main_multimode.py --benchmark quick
    python main_multimode.py --benchmark full
"""

import asyncio
import argparse
import os
from dotenv import load_dotenv

from src.multi_mode_assistant import MultiModeResearchAssistant, ResearchMode
from src.benchmark import ResearchBenchmark, run_quick_benchmark, run_full_benchmark
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def run_single_query(
    query: str, mode: ResearchMode, openai_key: str, tavily_key: str, max_papers: int = 10
):
    """Run a single query in the specified mode"""
    assistant = MultiModeResearchAssistant(openai_key, tavily_key)

    print(f"\n{'='*80}")
    print(f"Running query in {mode.value.upper()} mode")
    print(f"{'='*80}\n")

    result = await assistant.answer_query(query, mode=mode, max_papers=max_papers)

    print(result["answer"])
    print(f"\n{'='*80}")
    print("Metrics:")
    print(f"  Latency: {result['metrics']['latency_seconds']:.2f}s")
    print(f"  Mode: {result['metrics']['mode']}")
    if result.get("confidence") is not None:
        print(f"  Confidence: {result['confidence']:.2%}")
    print(f"{'='*80}\n")

    return result


async def run_all_modes(query: str, openai_key: str, tavily_key: str, max_papers: int = 10):
    """Run the same query in all 3 modes for comparison"""
    assistant = MultiModeResearchAssistant(openai_key, tavily_key)

    modes = [
        ResearchMode.SIMPLE_WEB_RAG,
        ResearchMode.MCP_BASIC,
        ResearchMode.MCP_VERIFIED,
    ]

    results = {}

    for mode in modes:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.value.upper()}")
        print(f"{'='*80}")

        result = await assistant.answer_query(query, mode=mode, max_papers=max_papers)

        print(result["answer"])
        print(f"\n{'='*80}")
        print("Metrics:")
        print(f"  Latency: {result['metrics']['latency_seconds']:.2f}s")
        if result.get("confidence") is not None:
            print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Sources: {len(result['sources'])}")
        print(f"{'='*80}\n")

        results[mode.value] = result

    # Print comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Mode':<20} {'Latency (s)':<15} {'Sources':<10} {'Confidence':<15}")
    print("-"*80)

    for mode in modes:
        mode_result = results[mode.value]
        latency = mode_result['metrics']['latency_seconds']
        sources = len(mode_result['sources'])
        confidence = mode_result.get('confidence')
        conf_str = f"{confidence:.2%}" if confidence is not None else "N/A"

        print(f"{mode.value:<20} {latency:<15.2f} {sources:<10} {conf_str:<15}")

    print("="*80 + "\n")

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Mode Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single query with Simple Web-RAG
  python main_multimode.py --query "What is RAG?" --mode simple_web_rag
  
  # Run query in all modes for comparison
  python main_multimode.py --query "What is RAG?" --all-modes
  
  # Run quick benchmark (2 queries × 3 modes)
  python main_multimode.py --benchmark quick
  
  # Run full benchmark (5 queries × 3 modes)
  python main_multimode.py --benchmark full
        """,
    )

    parser.add_argument(
        "--query", type=str, help="Research query to process"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple_web_rag", "mcp_basic", "mcp_verified"],
        default="mcp_verified",
        help="Research mode to use (default: mcp_verified)",
    )

    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Run the query in all 3 modes for comparison",
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=10,
        help="Maximum number of papers to retrieve (default: 10)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["quick", "full"],
        help="Run benchmark suite (quick=2 queries, full=5 queries)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if not openai_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return

    # Handle benchmark mode
    if args.benchmark:
        if args.benchmark == "quick":
            await run_quick_benchmark(openai_key, tavily_key)
        else:
            await run_full_benchmark(openai_key, tavily_key)
        return

    # Require query if not benchmarking
    if not args.query:
        parser.print_help()
        print("\nERROR: --query is required unless running --benchmark")
        return

    # Handle query modes
    if args.all_modes:
        await run_all_modes(args.query, openai_key, tavily_key, args.max_papers)
    else:
        mode = ResearchMode(args.mode)
        await run_single_query(args.query, mode, openai_key, tavily_key, args.max_papers)


if __name__ == "__main__":
    asyncio.run(main())
