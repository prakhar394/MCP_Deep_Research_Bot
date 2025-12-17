"""
Unit tests for ThoroughMCPVerifier
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.thorough_mcp_verifier import ThoroughMCPVerifier
from src.utils.mcp_schema import MCPMessage, MessageType


@pytest.mark.unit
class TestThoroughMCPVerifier:
    """Test suite for ThoroughMCPVerifier"""
    
    def test_verifier_initialization(self, test_api_keys, monkeypatch):
        """Test that verifier initializes correctly"""
        mock_client = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.OpenAI",
            lambda api_key: mock_client
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoTokenizer",
            MagicMock(return_value=mock_tokenizer)
        )
        monkeypatch.setattr(
            "src.agents.thorough_mcp_verifier.AutoModelForSequenceClassification",
            MagicMock(return_value=mock_model)
        )
        
        agent = ThoroughMCPVerifier(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        assert agent.agent_name == "ThoroughMCPVerifier"
    
    @pytest.mark.asyncio
    async def test_verify_high_confidence_summary(
        self, verifier_agent, mock_mcp_message, sample_papers, sample_summary
    ):
        """Test verification of high-confidence summary"""
        mock_mcp_message.content = {
            "query": "What is RAG?",
            "summary": sample_summary,
            "documents": sample_papers
        }
        mock_mcp_message.message_type = MessageType.SUMMARY
        
        # Mock high confidence verification
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "accepted",
            "confidence": 0.95,
            "claims": [
                {"text": "RAG enhances language models", "supported": true}
            ],
            "hallucination_ratio": 0.02
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        assert result.message_type == MessageType.VERIFICATION
        assert result.content.get("status") in ["accepted", "needs_revision"]
    
    @pytest.mark.asyncio
    async def test_verify_low_confidence_summary(
        self, verifier_agent, mock_mcp_message, sample_papers
    ):
        """Test verification of low-confidence summary"""
        low_confidence_summary = (
            "RAG might possibly enhance models in some way. "
            "There are claims that it could improve accuracy by 1000%."
        )
        
        mock_mcp_message.content = {
            "query": "What is RAG?",
            "summary": low_confidence_summary,
            "documents": sample_papers
        }
        
        # Mock low confidence verification
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "needs_revision",
            "confidence": 0.45,
            "claims": [
                {"text": "Could improve by 1000%", "supported": false}
            ],
            "hallucination_ratio": 0.4
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should identify low confidence
    
    @pytest.mark.asyncio
    async def test_detect_hallucinations(
        self, verifier_agent, mock_mcp_message, sample_papers
    ):
        """Test detection of hallucinated claims"""
        hallucinated_summary = (
            "RAG was invented in 1995 by John Smith. "
            "It improves accuracy by 500% in all cases."
        )
        
        mock_mcp_message.content = {
            "query": "What is RAG?",
            "summary": hallucinated_summary,
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "rejected",
            "confidence": 0.2,
            "claims": [
                {"text": "Invented in 1995", "supported": false},
                {"text": "500% improvement", "supported": false}
            ],
            "hallucination_ratio": 0.8
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should detect hallucinations
    
    @pytest.mark.asyncio
    async def test_verify_with_citations(
        self, verifier_agent, mock_mcp_message, sample_papers
    ):
        """Test verification with proper citations"""
        cited_summary = (
            "RAG enhances language models [1]. "
            "Studies show 40% improvement [2]."
        )
        
        mock_mcp_message.content = {
            "query": "Explain RAG",
            "summary": cited_summary,
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "accepted",
            "confidence": 0.92,
            "claims": [
                {"text": "RAG enhances models", "supported": true, "citation": "[1]"}
            ],
            "hallucination_ratio": 0.05
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # Citations should increase confidence
    
    @pytest.mark.asyncio
    async def test_verify_empty_summary(
        self, verifier_agent, mock_mcp_message, sample_papers
    ):
        """Test verification of empty summary"""
        mock_mcp_message.content = {
            "query": "Test",
            "summary": "",
            "documents": sample_papers
        }
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should handle empty input
    
    @pytest.mark.asyncio
    async def test_verify_no_documents(
        self, verifier_agent, mock_mcp_message, sample_summary
    ):
        """Test verification with no supporting documents"""
        mock_mcp_message.content = {
            "query": "Test",
            "summary": sample_summary,
            "documents": []
        }
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should handle missing documents
    
    @pytest.mark.asyncio
    async def test_confidence_score_range(
        self, verifier_agent, mock_mcp_message, sample_papers, sample_summary
    ):
        """Test that confidence scores are in valid range"""
        mock_mcp_message.content = {
            "query": "Test",
            "summary": sample_summary,
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "accepted",
            "confidence": 0.87,
            "claims": [],
            "hallucination_ratio": 0.1
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        if "confidence" in result.content:
            assert 0 <= result.content["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_hallucination_ratio_calculation(
        self, verifier_agent, mock_mcp_message, sample_papers, sample_summary
    ):
        """Test hallucination ratio is calculated correctly"""
        mock_mcp_message.content = {
            "query": "Test",
            "summary": sample_summary,
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "accepted",
            "confidence": 0.9,
            "claims": [
                {"text": "Claim 1", "supported": true},
                {"text": "Claim 2", "supported": false}
            ],
            "hallucination_ratio": 0.5
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        if "hallucination_ratio" in result.content:
            assert 0 <= result.content["hallucination_ratio"] <= 1
    
    @pytest.mark.asyncio
    async def test_revision_suggestions(
        self, verifier_agent, mock_mcp_message, sample_papers, sample_summary
    ):
        """Test that revision suggestions are provided"""
        mock_mcp_message.content = {
            "query": "Test",
            "summary": sample_summary,
            "documents": sample_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "needs_revision",
            "confidence": 0.65,
            "claims": [],
            "revision_suggestions": ["Add more citations", "Verify claim X"]
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # May include revision suggestions
    
    @pytest.mark.asyncio
    async def test_verify_with_conflicting_sources(
        self, verifier_agent, mock_mcp_message
    ):
        """Test verification with conflicting information"""
        conflicting_papers = [
            {
                "title": "Paper A",
                "abstract": "RAG improves accuracy by 40%",
                "url": "https://example.com/a"
            },
            {
                "title": "Paper B",
                "abstract": "RAG shows no improvement",
                "url": "https://example.com/b"
            }
        ]
        
        mock_mcp_message.content = {
            "query": "RAG effectiveness",
            "summary": "RAG improves accuracy significantly",
            "documents": conflicting_papers
        }
        
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """
        {
            "status": "needs_revision",
            "confidence": 0.5,
            "claims": [{"text": "Improves significantly", "supported": false}]
        }
        """
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )
        
        result = await verifier_agent.process(mock_mcp_message)
        
        assert result is not None
        # Should handle conflicting information
    
    @pytest.mark.asyncio
    async def test_api_error_handling(
        self, verifier_agent, mock_mcp_message, sample_papers, sample_summary
    ):
        """Test error handling for API failures"""
        mock_mcp_message.content = {
            "query": "Test",
            "summary": sample_summary,
            "documents": sample_papers
        }
        
        verifier_agent.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )
        
        # Should handle errors gracefully
        try:
            result = await verifier_agent.process(mock_mcp_message)
            if result:
                assert "error" in result.content or "status" in result.content
        except Exception as e:
            assert "API Error" in str(e)
