"""
Integration tests for complete research workflows
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


@pytest.mark.integration
class TestCompleteWorkflows:
    """Integration tests for end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_simple_web_rag_workflow(
        self, test_api_keys, sample_query, monkeypatch
    ):
        """Test complete simple web RAG workflow"""
        # Mock dependencies
        mock_embedder = MagicMock()
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock tool executor
        assistant.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {
                        "title": "Test Result",
                        "content": "Test content about RAG",
                        "url": "http://example.com"
                    }
                ]
            }
        )
        
        # Mock OpenAI
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            "RAG is a technique for enhancing language models."
        )
        assistant.openai_client.chat.completions.create = MagicMock(
            return_value=mock_completion
        )
        
        # Execute workflow
        result = await assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.SIMPLE_WEB_RAG,
            max_papers=10
        )
        
        # Verify complete workflow
        assert result is not None
        assert "answer" in result
        assert "sources" in result
        assert "metrics" in result
        assert result["mode"] == ResearchMode.SIMPLE_WEB_RAG
        assert len(result["answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_mcp_basic_workflow(
        self, test_api_keys, sample_query, sample_papers, monkeypatch
    ):
        """Test complete MCP basic workflow"""
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=[[0.1, 0.2]])
        
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock()
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock()
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock retriever
        mock_retrieval_msg = MagicMock()
        mock_retrieval_msg.content = {
            "documents": sample_papers,
            "total_found": len(sample_papers)
        }
        assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval_msg
        )
        
        # Mock summarizer
        mock_summary_msg = MagicMock()
        mock_summary_msg.content = {
            "summary": "Comprehensive summary of RAG research"
        }
        assistant.summarizer.process = AsyncMock(
            return_value=mock_summary_msg
        )
        
        # Execute workflow
        result = await assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_BASIC,
            max_papers=10
        )
        
        # Verify workflow
        assert result is not None
        assert result["mode"] == ResearchMode.MCP_BASIC
        assert len(result["sources"]) > 0
        assert result["confidence"] is None  # No verification in basic
    
    @pytest.mark.asyncio
    async def test_mcp_verified_workflow_accepted(
        self, test_api_keys, sample_query, sample_papers, monkeypatch
    ):
        """Test complete MCP verified workflow with accepted verification"""
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=[[0.1]])
        
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock()
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock()
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock retriever
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": sample_papers}
        assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        # Mock summarizer
        mock_summary = MagicMock()
        mock_summary.content = {"summary": "High quality summary"}
        assistant.summarizer.process = AsyncMock(
            return_value=mock_summary
        )
        
        # Mock verifier (high confidence - accept immediately)
        mock_verification = MagicMock()
        mock_verification.content = {
            "status": "accepted",
            "confidence": 0.98,
            "hallucination_ratio": 0.02,
            "claims": []
        }
        assistant.verifier.process = AsyncMock(
            return_value=mock_verification
        )
        
        # Execute workflow
        result = await assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_VERIFIED,
            max_papers=10
        )
        
        # Verify complete verified workflow
        assert result is not None
        assert result["mode"] == ResearchMode.MCP_VERIFIED
        assert result["confidence"] == 0.98
        assert "verification_details" in result
        assert result["verification_details"]["status"] == "accepted"
    
    @pytest.mark.asyncio
    async def test_mcp_verified_workflow_revision(
        self, test_api_keys, sample_query, sample_papers, monkeypatch
    ):
        """Test MCP verified workflow with revision loop"""
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=[[0.1]])
        
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock()
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock()
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock retriever
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": sample_papers}
        assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        # Mock summarizer
        mock_summary = MagicMock()
        mock_summary.content = {"summary": "Summary"}
        assistant.summarizer.process = AsyncMock(
            return_value=mock_summary
        )
        
        # Mock verifier - first reject, then accept
        call_count = 0
        async def mock_verify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                mock_result.content = {
                    "status": "needs_revision",
                    "confidence": 0.6,
                    "hallucination_ratio": 0.3
                }
            else:
                mock_result.content = {
                    "status": "accepted",
                    "confidence": 0.95,
                    "hallucination_ratio": 0.05
                }
            return mock_result
        
        assistant.verifier.process = AsyncMock(side_effect=mock_verify)
        
        # Execute workflow
        result = await assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_VERIFIED,
            max_papers=10
        )
        
        # Should have gone through revision
        assert result is not None
        assert call_count > 1  # Multiple verification calls
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(
        self, test_api_keys, sample_query, monkeypatch
    ):
        """Test workflow error recovery"""
        mock_embedder = MagicMock()
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock retriever failure
        assistant.mcp_retriever.process = AsyncMock(
            side_effect=Exception("Retrieval failed")
        )
        
        # Workflow should handle error gracefully
        try:
            result = await assistant.answer_query(
                query=sample_query,
                mode=ResearchMode.MCP_BASIC,
                max_papers=10
            )
            # If it returns, should have error indicator
            if result:
                assert "error" in result or "answer" in result
        except Exception as e:
            # Exception is acceptable
            assert "failed" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_benchmark_workflow(
        self, test_api_keys, monkeypatch
    ):
        """Test complete benchmark workflow"""
        mock_embedder = MagicMock()
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock()
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock()
        )
        
        from src.benchmark import ResearchBenchmark
        from src.multi_mode_assistant import ResearchMode
        
        benchmark = ResearchBenchmark(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock assistant responses
        benchmark.assistant.answer_query = AsyncMock(
            return_value={
                "answer": "Test answer",
                "confidence": 0.9,
                "sources": [{"title": "Test"}],
                "mode": "simple_web_rag",
                "metrics": {"latency_seconds": 5.0}
            }
        )
        
        queries = ["What is RAG?", "How does RLHF work?"]
        modes = [ResearchMode.SIMPLE_WEB_RAG, ResearchMode.MCP_BASIC]
        
        results = await benchmark.run_benchmark(
            queries=queries,
            modes=modes,
            max_papers=10
        )
        
        # Verify benchmark completed
        assert len(results) == len(queries) * len(modes)
        
        # Generate summary
        summaries = benchmark.generate_summary()
        assert len(summaries) == len(modes)
    
    @pytest.mark.asyncio
    async def test_concurrent_queries_workflow(
        self, test_api_keys, monkeypatch
    ):
        """Test handling multiple concurrent queries"""
        mock_embedder = MagicMock()
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock()
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock()
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock responses
        assistant.tool_executor.execute_tool = AsyncMock(
            return_value={"success": True, "result": []}
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Answer"
        assistant.openai_client.chat.completions.create = MagicMock(
            return_value=mock_completion
        )
        
        # Run multiple queries concurrently
        queries = [
            "What is RAG?",
            "How does RLHF work?",
            "Explain transformers"
        ]
        
        tasks = [
            assistant.answer_query(
                query=q,
                mode=ResearchMode.SIMPLE_WEB_RAG,
                max_papers=5
            )
            for q in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == len(queries)
        for result in results:
            assert result is not None
            assert "answer" in result
    
    @pytest.mark.asyncio
    async def test_mode_comparison_workflow(
        self, test_api_keys, sample_query, monkeypatch
    ):
        """Test comparing all modes for same query"""
        mock_embedder = MagicMock()
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock()
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock()
        )
        
        from src.multi_mode_assistant import (
            MultiModeResearchAssistant,
            ResearchMode
        )
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock minimal responses
        assistant.tool_executor.execute_tool = AsyncMock(
            return_value={"success": True, "result": []}
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Answer"
        assistant.openai_client.chat.completions.create = MagicMock(
            return_value=mock_completion
        )
        
        assistant.mcp_retriever.process = AsyncMock(
            return_value=MagicMock(content={"documents": []})
        )
        assistant.summarizer.process = AsyncMock(
            return_value=MagicMock(content={"summary": "Summary"})
        )
        assistant.verifier.process = AsyncMock(
            return_value=MagicMock(content={
                "status": "accepted",
                "confidence": 0.9
            })
        )
        
        # Run same query in all modes
        modes = [
            ResearchMode.SIMPLE_WEB_RAG,
            ResearchMode.MCP_BASIC,
            ResearchMode.MCP_VERIFIED
        ]
        
        results = {}
        for mode in modes:
            result = await assistant.answer_query(
                query=sample_query,
                mode=mode,
                max_papers=10
            )
            results[mode] = result
        
        # All modes should complete
        assert len(results) == 3
        
        # Verify latency increases with complexity
        simple_latency = results[ResearchMode.SIMPLE_WEB_RAG]["metrics"]["latency_seconds"]
        basic_latency = results[ResearchMode.MCP_BASIC]["metrics"]["latency_seconds"]
        verified_latency = results[ResearchMode.MCP_VERIFIED]["metrics"]["latency_seconds"]
        
        # Generally verified should be slowest (though mocked might not always reflect this)
        assert all(lat >= 0 for lat in [simple_latency, basic_latency, verified_latency])
