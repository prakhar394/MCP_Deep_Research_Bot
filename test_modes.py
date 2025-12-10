#!/usr/bin/env python3
# test_modes.py

"""
Test script to demonstrate all 3 modes of the research assistant.

This script runs a simple test query through all modes and displays
the results with timing information.
"""

import asyncio
import os
from dotenv import load_dotenv

from src.multi_mode_assistant import MultiModeResearchAssistant, ResearchMode


async def test_all_modes():
    """Test all 3 modes with a simple query"""
    
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return
    
    # Initialize assistant
    assistant = MultiModeResearchAssistant(openai_key, tavily_key)
    
    # Test query
    query = "What is retrieval augmented generation?"
    
    print("\n" + "="*80)
    print("TESTING ALL 3 MODES")
    print("="*80)
    print(f"Query: {query}\n")
    
    # Test each mode
    modes = [
        (ResearchMode.SIMPLE_WEB_RAG, "Simple Web-RAG (No MCP, No Verification)"),
        (ResearchMode.MCP_BASIC, "MCP-Basic (MCP Tools, No Verification)"),
        (ResearchMode.MCP_VERIFIED, "MCP-Verified (MCP Tools + Verification Loop)"),
    ]
    
    results = {}
    
    for mode, description in modes:
        print(f"\n{'='*80}")
        print(f"Testing: {description}")
        print(f"{'='*80}\n")
        
        try:
            result = await assistant.answer_query(
                query=query,
                mode=mode,
                max_papers=8
            )
            
            results[mode.value] = result
            
            # Print summary
            print(f"✓ Success!")
            print(f"  Latency: {result['metrics']['latency_seconds']:.2f}s")
            print(f"  Sources: {len(result['sources'])}")
            
            if result.get('confidence') is not None:
                print(f"  Confidence: {result['confidence']:.2%}")
                
            if result.get('verification_details'):
                halluc = result['verification_details'].get('hallucination_ratio', 0)
                print(f"  Hallucination Ratio: {halluc:.2%}")
            
            # Print first 300 chars of answer
            answer_preview = result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer']
            print(f"\nAnswer Preview:\n{answer_preview}\n")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}\n")
            results[mode.value] = {"error": str(e)}
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Mode':<25} {'Latency':<12} {'Sources':<10} {'Confidence':<12}")
    print("-"*80)
    
    for mode, description in modes:
        result = results.get(mode.value)
        if result and "error" not in result:
            latency = f"{result['metrics']['latency_seconds']:.2f}s"
            sources = str(len(result['sources']))
            conf = result.get('confidence')
            conf_str = f"{conf:.2%}" if conf is not None else "N/A"
            
            print(f"{description:<25} {latency:<12} {sources:<10} {conf_str:<12}")
        else:
            print(f"{description:<25} {'ERROR':<12} {'-':<10} {'-':<12}")
    
    print("="*80)
    
    # Speed comparison
    if all(mode.value in results and "error" not in results[mode.value] for mode, _ in modes):
        simple_time = results[ResearchMode.SIMPLE_WEB_RAG.value]['metrics']['latency_seconds']
        basic_time = results[ResearchMode.MCP_BASIC.value]['metrics']['latency_seconds']
        verified_time = results[ResearchMode.MCP_VERIFIED.value]['metrics']['latency_seconds']
        
        print("\nSpeed Comparison:")
        print(f"  MCP-Basic is {basic_time/simple_time:.1f}x slower than Simple Web-RAG")
        print(f"  MCP-Verified is {verified_time/simple_time:.1f}x slower than Simple Web-RAG")
        print(f"  MCP-Verified is {verified_time/basic_time:.1f}x slower than MCP-Basic")
        
        if results[ResearchMode.MCP_VERIFIED.value].get('confidence'):
            print(f"\nTrade-off: {verified_time-simple_time:.1f}s extra for {results[ResearchMode.MCP_VERIFIED.value]['confidence']:.0%} verified confidence")
    
    print("\n" + "="*80 + "\n")


async def test_single_mode_detailed():
    """Test a single mode with detailed output"""
    
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    
    assistant = MultiModeResearchAssistant(openai_key, tavily_key)
    
    query = "What are the latest techniques for LLM quantization?"
    mode = ResearchMode.MCP_VERIFIED
    
    print("\n" + "="*80)
    print("DETAILED TEST - MCP-VERIFIED MODE")
    print("="*80)
    print(f"Query: {query}\n")
    
    result = await assistant.answer_query(
        query=query,
        mode=mode,
        max_papers=10
    )
    
    # Print full answer
    print(result['answer'])
    
    # Print detailed metrics
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(f"Latency: {result['metrics']['latency_seconds']:.2f}s")
    print(f"Mode: {result['mode']}")
    print(f"Sources: {len(result['sources'])}")
    
    if result.get('confidence'):
        print(f"Confidence: {result['confidence']:.4f}")
    
    if result.get('verification_details'):
        vd = result['verification_details']
        print(f"\nVerification Details:")
        print(f"  Status: {vd.get('status', 'N/A')}")
        print(f"  Hallucination Ratio: {vd.get('hallucination_ratio', 0):.4f}")
        print(f"  Claims Verified: {len(vd.get('claims', []))}")
        
        if vd.get('claims'):
            print(f"\n  Sample Claims:")
            for i, claim in enumerate(vd['claims'][:3], 1):
                print(f"    {i}. {claim.get('text', 'N/A')[:80]}...")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MCP RESEARCH ASSISTANT - MODE TESTING")
    print("="*80)
    
    print("\nRunning Test 1: All Modes Comparison...")
    asyncio.run(test_all_modes())
    
    print("\nWould you like to run Test 2: Detailed Single Mode Test? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        asyncio.run(test_single_mode_detailed())
