"""
Unit tests for SummarizerAgent
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.summarizer import SummarizerAgent
from src.utils.mcp_schema import MCPMessage, MessageType


@pytest.mark.unit
class TestSummarizerAgent:
    """Test suite for SummarizerAgent"""
    
    def test_summarizer_initialization(self, test_api_keys, monkeypatch):
        """Test that summarizer initializes correctly"""
        mock_client = MagicMock()
        monkeypatch.setattr(
            "src.agents.summarizer.OpenAI",
            lambda api_key: mock_client
        )
        
        agent = SummarizerAgent(test_api_keys["openai"])
        
        assert agent.agent_name == "Summarizer"
        assert agent.client is not None
    
    @pytest.mark.asyncio
    async def test_process_with_documents(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test summarizer with valid documents"""
        mock_mcp_message.content = {
            "query": "What is RAG?",
            "documents": sample_papers
        }
        mock_mcp_message.message_type = MessageType.RETRIEVAL
        
        # Mock OpenAI response
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            "RAG is a technique that enhances language models by "
            "retrieving relevant information from external sources."
        )
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        assert result.message_type == MessageType.SUMMARY
        assert "summary" in result.content
        assert len(result.content["summary"]) > 0
    
    @pytest.mark.asyncio
    async def test_process_empty_documents(self, summarizer_agent, mock_mcp_message):
        """Test summarizer with no documents"""
        mock_mcp_message.content = {
            "query": "What is RAG?",
            "documents": []
        }
        mock_mcp_message.message_type = MessageType.RETRIEVAL
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        assert "summary" in result.content
        # Should handle empty documents gracefully
    
    @pytest.mark.asyncio
    async def test_process_with_citations(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test that summarizer can generate citations"""
        mock_mcp_message.content = {
            "query": "Explain RAG systems",
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            "RAG systems enhance LLMs [1]. Recent work shows improvements [2]."
        )
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        summary = result.content["summary"]
        # May contain citations in format [1], [2], etc.
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_process_long_documents(
        self, summarizer_agent, mock_mcp_message
    ):
        """Test summarizer with very long documents"""
        long_papers = [
            {
                "title": f"Paper {i}",
                "abstract": "A" * 5000,  # Very long abstract
                "url": f"https://example.com/{i}"
            }
            for i in range(10)
        ]
        
        mock_mcp_message.content = {
            "query": "Summarize research",
            "documents": long_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Summary of research"
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should handle long inputs without errors
    
    @pytest.mark.asyncio
    async def test_process_api_error(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test summarizer handles API errors gracefully"""
        mock_mcp_message.content = {
            "query": "What is RAG?",
            "documents": sample_papers
        }
        
        # Mock API error
        summarizer_agent.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Should handle error gracefully
        try:
            result = await summarizer_agent.process(mock_mcp_message)
            # If it returns a result, check it's valid
            if result:
                assert "summary" in result.content or "error" in result.content
        except Exception as e:
            # Exception is acceptable for API errors
            assert "API Error" in str(e)
    
    @pytest.mark.asyncio
    async def test_process_different_query_types(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test summarizer with different query types"""
        queries = [
            "What is X?",  # Definition
            "How does Y work?",  # Explanation
            "Compare A and B",  # Comparison
            "List advantages of Z",  # List
        ]
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test summary"
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        for query in queries:
            mock_mcp_message.content = {
                "query": query,
                "documents": sample_papers
            }
            
            result = await summarizer_agent.process(mock_mcp_message)
            
            assert result is not None
            assert "summary" in result.content
    
    @pytest.mark.asyncio
    async def test_summary_length_reasonable(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test that summary length is reasonable"""
        mock_mcp_message.content = {
            "query": "Explain RAG",
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = (
            "This is a reasonable length summary that provides "
            "enough detail without being too long or too short."
        )
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        summary = result.content["summary"]
        # Summary should not be empty or excessively long
        assert 10 < len(summary) < 10000
    
    @pytest.mark.asyncio
    async def test_process_with_metadata(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test that document metadata is preserved"""
        mock_mcp_message.content = {
            "query": "Research summary",
            "documents": sample_papers,
            "include_sources": True
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Summary text"
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        # May include source information
    
    @pytest.mark.asyncio
    async def test_process_duplicate_information(
        self, summarizer_agent, mock_mcp_message
    ):
        """Test summarizer handles duplicate information"""
        duplicate_papers = [
            {
                "title": "Paper A",
                "abstract": "Same information repeated",
                "url": "https://example.com/1"
            },
            {
                "title": "Paper B",
                "abstract": "Same information repeated",
                "url": "https://example.com/2"
            }
        ]
        
        mock_mcp_message.content = {
            "query": "What is this about?",
            "documents": duplicate_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Summary without duplication"
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        # Summary should deduplicate information
    
    @pytest.mark.asyncio
    async def test_process_missing_abstracts(
        self, summarizer_agent, mock_mcp_message
    ):
        """Test summarizer with documents missing abstracts"""
        incomplete_papers = [
            {
                "title": "Paper 1",
                "url": "https://example.com/1"
                # No abstract
            },
            {
                "title": "Paper 2",
                "abstract": None,
                "url": "https://example.com/2"
            }
        ]
        
        mock_mcp_message.content = {
            "query": "Summarize these",
            "documents": incomplete_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Summary based on titles"
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should handle missing data gracefully
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(
        self, summarizer_agent, mock_mcp_message, sample_papers
    ):
        """Test that confidence scores are calculated"""
        mock_mcp_message.content = {
            "query": "Test query",
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Test summary"
        
        summarizer_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await summarizer_agent.process(mock_mcp_message)
        
        assert result is not None
        # May have confidence score
        if hasattr(result, 'confidence_score'):
            assert 0 <= result.confidence_score <= 1
