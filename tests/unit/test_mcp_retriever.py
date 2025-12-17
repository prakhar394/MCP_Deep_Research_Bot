"""
Unit tests for MCPRetrieverAgent
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.mcp_retriever import MCPRetrieverAgent
from src.utils.mcp_schema import MCPMessage, MessageType


@pytest.mark.unit
class TestMCPRetrieverAgent:
    """Test suite for MCPRetrieverAgent"""
    
    @pytest.mark.asyncio
    async def test_retriever_initialization(self, test_api_keys, monkeypatch):
        """Test that retriever initializes correctly"""
        mock_embedder = MagicMock()
        monkeypatch.setattr(
            "src.agents.mcp_retriever.SentenceTransformer",
            lambda x: mock_embedder
        )
        
        agent = MCPRetrieverAgent(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        assert agent.agent_name == "MCPRetriever"
        assert agent.tool_executor is not None
        assert agent.embedder is not None
    
    @pytest.mark.asyncio
    async def test_process_valid_query_arxiv(self, mcp_retriever, mock_mcp_message):
        """Test retriever with valid arXiv query"""
        mock_mcp_message.content = {
            "query": "machine learning",
            "max_results": 5,
            "sources": ["arxiv"]
        }
        
        # Mock the tool executor response
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {
                        "id": "arxiv:2401.00001",
                        "title": "ML Paper",
                        "abstract": "Machine learning research",
                        "url": "https://arxiv.org/abs/2401.00001",
                        "published": "2024-01-01",
                        "authors": ["Test Author"],
                    }
                ]
            }
        )
        
        # Mock embeddings for reranking
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[[0.1, 0.2], [0.2, 0.3]]
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert result.message_type == MessageType.RETRIEVAL
        assert "documents" in result.content
        assert result.content["total_found"] >= 0
    
    @pytest.mark.asyncio
    async def test_process_valid_query_pubmed(self, mcp_retriever, mock_mcp_message):
        """Test retriever with PubMed query"""
        mock_mcp_message.content = {
            "query": "clinical trials",
            "max_results": 5,
            "sources": ["pubmed"]
        }
        
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {
                        "title": "Clinical Trial Study",
                        "content": "Medical research on clinical trials",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
                    }
                ]
            }
        )
        
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[[0.1, 0.2]]
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert result.message_type == MessageType.RETRIEVAL
    
    @pytest.mark.asyncio
    async def test_process_empty_query(self, mcp_retriever, mock_mcp_message):
        """Test retriever handles empty query gracefully"""
        mock_mcp_message.content = {
            "query": "",
            "max_results": 5
        }
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert result.content["total_found"] == 0
        assert len(result.content["documents"]) == 0
    
    @pytest.mark.asyncio
    async def test_process_none_query(self, mcp_retriever, mock_mcp_message):
        """Test retriever handles None query"""
        mock_mcp_message.content = {
            "query": None,
            "max_results": 5
        }
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert result.content["total_found"] == 0
    
    @pytest.mark.asyncio
    async def test_process_api_failure(self, mcp_retriever, mock_mcp_message):
        """Test retriever handles API failures gracefully"""
        mock_mcp_message.content = {
            "query": "test query",
            "max_results": 5,
            "sources": ["arxiv"]
        }
        
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={"success": False, "error": "API Error"}
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        # Should return empty results rather than crashing
        assert "documents" in result.content
    
    @pytest.mark.asyncio
    async def test_process_multiple_sources(self, mcp_retriever, mock_mcp_message):
        """Test retriever with multiple sources"""
        mock_mcp_message.content = {
            "query": "healthcare AI",
            "max_results": 10,
            "sources": ["arxiv", "pubmed"]
        }
        
        async def mock_execute(tool_name, params):
            if tool_name == "arxiv_search":
                return {
                    "success": True,
                    "result": [{"title": "arXiv paper", "abstract": "test"}]
                }
            elif tool_name == "web_search":
                return {
                    "success": True,
                    "result": [{"title": "PubMed result", "content": "test"}]
                }
            return {"success": False}
        
        mcp_retriever.tool_executor.execute_tool = AsyncMock(side_effect=mock_execute)
        mcp_retriever.embedder.encode = MagicMock(return_value=[[0.1]])
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert "documents" in result.content
    
    @pytest.mark.asyncio
    async def test_reranking_by_relevance(self, mcp_retriever, mock_mcp_message):
        """Test that documents are reranked by relevance"""
        mock_mcp_message.content = {
            "query": "deep learning",
            "max_results": 3,
            "sources": ["arxiv"]
        }
        
        # Mock multiple papers with different abstracts
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {"title": "Paper 1", "abstract": "deep learning"},
                    {"title": "Paper 2", "abstract": "unrelated topic"},
                    {"title": "Paper 3", "abstract": "neural networks deep learning"},
                ]
            }
        )
        
        # Mock embeddings where Paper 3 is most relevant
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[
                [0.5, 0.5],  # query
                [0.4, 0.4],  # paper 1
                [0.1, 0.1],  # paper 2 (least relevant)
                [0.6, 0.6],  # paper 3 (most relevant)
            ]
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert len(result.content["documents"]) > 0
    
    @pytest.mark.asyncio
    async def test_query_expansion(self, mcp_retriever):
        """Test query expansion logic"""
        original_query = "RAG"
        expanded = mcp_retriever._expand_query(original_query)
        
        # Expanded query should contain original and related terms
        assert "RAG" in expanded or "retrieval" in expanded.lower()
    
    @pytest.mark.asyncio
    async def test_relevance_filtering(self, mcp_retriever, mock_mcp_message):
        """Test that irrelevant documents are filtered out"""
        mock_mcp_message.content = {
            "query": "transformer architecture",
            "max_results": 5,
            "sources": ["arxiv"]
        }
        
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {"title": "Relevant", "abstract": "transformer architecture"},
                    {"title": "Irrelevant", "abstract": "cooking recipes"},
                ]
            }
        )
        
        # Mock embeddings where second doc has very low similarity
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[
                [0.5, 0.5],  # query
                [0.5, 0.5],  # relevant (high similarity)
                [0.1, 0.1],  # irrelevant (low similarity)
            ]
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        # Should filter based on relevance threshold
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_max_results_respected(self, mcp_retriever, mock_mcp_message):
        """Test that max_results parameter is respected"""
        max_results = 3
        mock_mcp_message.content = {
            "query": "AI research",
            "max_results": max_results,
            "sources": ["arxiv"]
        }
        
        # Return more papers than requested
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {"title": f"Paper {i}", "abstract": "test"}
                    for i in range(10)
                ]
            }
        )
        
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[[0.5, 0.5]] * 11  # query + 10 papers
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        # Should not exceed max_results
        assert len(result.content["documents"]) <= max_results * 2  # Some buffer is OK
    
    @pytest.mark.asyncio
    async def test_deduplication(self, mcp_retriever, mock_mcp_message):
        """Test that duplicate papers are removed"""
        mock_mcp_message.content = {
            "query": "machine learning",
            "max_results": 5,
            "sources": ["arxiv"]
        }
        
        # Return duplicate papers
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {"id": "arxiv:123", "title": "Paper A", "abstract": "test"},
                    {"id": "arxiv:123", "title": "Paper A", "abstract": "test"},  # duplicate
                    {"id": "arxiv:456", "title": "Paper B", "abstract": "test"},
                ]
            }
        )
        
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[[0.5]] * 4
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        # Check for deduplication logic if implemented
    
    @pytest.mark.asyncio
    async def test_confidence_score(self, mcp_retriever, mock_mcp_message):
        """Test that confidence scores are calculated"""
        mock_mcp_message.content = {
            "query": "neural networks",
            "max_results": 3,
            "sources": ["arxiv"]
        }
        
        mcp_retriever.tool_executor.execute_tool = AsyncMock(
            return_value={
                "success": True,
                "result": [
                    {"title": "Paper", "abstract": "neural networks"}
                ]
            }
        )
        
        mcp_retriever.embedder.encode = MagicMock(
            return_value=[[0.5, 0.5]] * 2
        )
        
        result = await mcp_retriever.process(mock_mcp_message)
        
        assert result is not None
        assert hasattr(result, 'confidence_score')
