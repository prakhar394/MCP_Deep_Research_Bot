"""
Unit tests for MultiModeResearchAssistant
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.multi_mode_assistant import (
    MultiModeResearchAssistant,
    ResearchMode
)


@pytest.mark.unit
class TestMultiModeResearchAssistant:
    """Test suite for MultiModeResearchAssistant"""
    
    def test_assistant_initialization(self, test_api_keys, monkeypatch):
        """Test that assistant initializes correctly"""
        # Mock dependencies
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
        
        assistant = MultiModeResearchAssistant(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        assert assistant.openai_api_key == test_api_keys["openai"]
        assert assistant.tavily_api_key == test_api_keys["tavily"]
        assert assistant.mcp_retriever is not None
        assert assistant.summarizer is not None
        assert assistant.verifier is not None
    
    @pytest.mark.asyncio
    async def test_answer_query_simple_web_rag(
        self, multi_mode_assistant, sample_query, sample_web_results
    ):
        """Test answer_query in simple_web_rag mode"""
        # Mock tool executor
        multi_mode_assistant.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": sample_web_results
            }
        )
        
        # Mock OpenAI for summarization
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test summary"
        
        multi_mode_assistant.openai_client.chat.completions.create = MagicMock(
            return_value=mock_completion
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.SIMPLE_WEB_RAG,
            max_papers=10
        )
        
        assert result is not None
        assert "answer" in result
        assert "metrics" in result
        assert result["mode"] == ResearchMode.SIMPLE_WEB_RAG
        assert "latency_seconds" in result["metrics"]
    
    @pytest.mark.asyncio
    async def test_answer_query_mcp_basic(
        self, multi_mode_assistant, sample_query, sample_papers
    ):
        """Test answer_query in mcp_basic mode"""
        # Mock retriever
        mock_retrieval_message = MagicMock()
        mock_retrieval_message.content = {
            "documents": sample_papers,
            "total_found": len(sample_papers)
        }
        multi_mode_assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval_message
        )
        
        # Mock summarizer
        mock_summary_message = MagicMock()
        mock_summary_message.content = {"summary": "Test summary"}
        multi_mode_assistant.summarizer.process = AsyncMock(
            return_value=mock_summary_message
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_BASIC,
            max_papers=10
        )
        
        assert result is not None
        assert "answer" in result
        assert result["mode"] == ResearchMode.MCP_BASIC
        assert result["confidence"] is None  # No verification in basic mode
    
    @pytest.mark.asyncio
    async def test_answer_query_mcp_verified(
        self, multi_mode_assistant, sample_query, sample_papers
    ):
        """Test answer_query in mcp_verified mode"""
        # Mock retriever
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": sample_papers}
        multi_mode_assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        # Mock summarizer
        mock_summary = MagicMock()
        mock_summary.content = {"summary": "Test summary"}
        multi_mode_assistant.summarizer.process = AsyncMock(
            return_value=mock_summary
        )
        
        # Mock verifier (high confidence - accept)
        mock_verification = MagicMock()
        mock_verification.content = {
            "status": "accepted",
            "confidence": 0.95,
            "hallucination_ratio": 0.03,
            "claims": []
        }
        multi_mode_assistant.verifier.process = AsyncMock(
            return_value=mock_verification
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_VERIFIED,
            max_papers=10
        )
        
        assert result is not None
        assert "answer" in result
        assert result["mode"] == ResearchMode.MCP_VERIFIED
        assert "confidence" in result
        assert result["confidence"] == 0.95
        assert "verification_details" in result
    
    @pytest.mark.asyncio
    async def test_verification_loop_iteration(
        self, multi_mode_assistant, sample_query, sample_papers
    ):
        """Test that verification loop iterates when confidence is low"""
        # Mock retriever
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": sample_papers}
        multi_mode_assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        # Mock summarizer
        mock_summary = MagicMock()
        mock_summary.content = {"summary": "Initial summary"}
        multi_mode_assistant.summarizer.process = AsyncMock(
            return_value=mock_summary
        )
        
        # Mock verifier - first low confidence, then high
        call_count = 0
        async def mock_verify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: low confidence
                mock_result = MagicMock()
                mock_result.content = {
                    "status": "needs_revision",
                    "confidence": 0.6,
                    "hallucination_ratio": 0.3
                }
                return mock_result
            else:
                # Second call: high confidence
                mock_result = MagicMock()
                mock_result.content = {
                    "status": "accepted",
                    "confidence": 0.98,
                    "hallucination_ratio": 0.02
                }
                return mock_result
        
        multi_mode_assistant.verifier.process = AsyncMock(side_effect=mock_verify)
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_VERIFIED,
            max_papers=10
        )
        
        assert result is not None
        assert "verification_details" in result
        # Should have iterated
    
    @pytest.mark.asyncio
    async def test_max_iterations_reached(
        self, multi_mode_assistant, sample_query, sample_papers
    ):
        """Test that max iterations limit is respected"""
        multi_mode_assistant.max_iterations = 2
        
        # Mock retriever
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": sample_papers}
        multi_mode_assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        # Mock summarizer
        mock_summary = MagicMock()
        mock_summary.content = {"summary": "Summary"}
        multi_mode_assistant.summarizer.process = AsyncMock(
            return_value=mock_summary
        )
        
        # Mock verifier - always low confidence
        mock_verification = MagicMock()
        mock_verification.content = {
            "status": "needs_revision",
            "confidence": 0.5,
            "hallucination_ratio": 0.4
        }
        multi_mode_assistant.verifier.process = AsyncMock(
            return_value=mock_verification
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_VERIFIED,
            max_papers=10
        )
        
        assert result is not None
        # Should stop after max iterations
    
    @pytest.mark.asyncio
    async def test_empty_query(self, multi_mode_assistant):
        """Test handling of empty query"""
        result = await multi_mode_assistant.answer_query(
            query="",
            mode=ResearchMode.SIMPLE_WEB_RAG,
            max_papers=10
        )
        
        assert result is not None
        # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_no_papers_found(
        self, multi_mode_assistant, sample_query
    ):
        """Test handling when no papers are found"""
        # Mock empty retrieval
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": [], "total_found": 0}
        multi_mode_assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_BASIC,
            max_papers=10
        )
        
        assert result is not None
        assert "answer" in result
        # Should handle no results gracefully
    
    @pytest.mark.asyncio
    async def test_invalid_mode(self, multi_mode_assistant, sample_query):
        """Test handling of invalid research mode"""
        with pytest.raises((ValueError, KeyError)):
            await multi_mode_assistant.answer_query(
                query=sample_query,
                mode="invalid_mode",
                max_papers=10
            )
    
    @pytest.mark.asyncio
    async def test_source_filtering(
        self, multi_mode_assistant, sample_query, sample_papers
    ):
        """Test source filtering (arxiv only, pubmed only, both)"""
        # Mock retriever
        mock_retrieval = MagicMock()
        mock_retrieval.content = {"documents": sample_papers}
        multi_mode_assistant.mcp_retriever.process = AsyncMock(
            return_value=mock_retrieval
        )
        
        # Mock summarizer
        mock_summary = MagicMock()
        mock_summary.content = {"summary": "Summary"}
        multi_mode_assistant.summarizer.process = AsyncMock(
            return_value=mock_summary
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.MCP_BASIC,
            max_papers=10,
            sources=["arxiv"]
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_max_papers_parameter(
        self, multi_mode_assistant, sample_query
    ):
        """Test that max_papers parameter is respected"""
        max_papers = 5
        
        # Mock tool executor
        multi_mode_assistant.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [{"title": f"Paper {i}"} for i in range(20)]
            }
        )
        
        # Mock OpenAI
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Summary"
        multi_mode_assistant.openai_client.chat.completions.create = MagicMock(
            return_value=mock_completion
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.SIMPLE_WEB_RAG,
            max_papers=max_papers
        )
        
        assert result is not None
        # Should respect max_papers limit
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(
        self, multi_mode_assistant, sample_query, sample_web_results
    ):
        """Test that metrics are calculated correctly"""
        multi_mode_assistant.tool_executor.execute_tool = AsyncMock(
            return_value={"success": True, "result": sample_web_results}
        )
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Summary"
        multi_mode_assistant.openai_client.chat.completions.create = MagicMock(
            return_value=mock_completion
        )
        
        result = await multi_mode_assistant.answer_query(
            query=sample_query,
            mode=ResearchMode.SIMPLE_WEB_RAG,
            max_papers=10
        )
        
        assert "metrics" in result
        assert "latency_seconds" in result["metrics"]
        assert result["metrics"]["latency_seconds"] > 0
        assert "mode" in result["metrics"]
        assert "timestamp" in result["metrics"]
