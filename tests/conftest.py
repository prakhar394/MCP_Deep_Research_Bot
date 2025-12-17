"""
Pytest configuration and fixtures for MCP Research Assistant tests.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()


# ============================================================================
# Session-level fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_api_keys():
    """Provide test API keys"""
    return {
        "openai": os.getenv("OPENAI_API_KEY", "test_openai_key"),
        "tavily": os.getenv("TAVILY_API_KEY", "test_tavily_key"),
    }


# ============================================================================
# Mock data fixtures
# ============================================================================

@pytest.fixture
def sample_query():
    """Sample research query"""
    return "What is retrieval augmented generation?"


@pytest.fixture
def sample_papers():
    """Sample paper data"""
    return [
        {
            "id": "arxiv:2401.00001",
            "title": "Advances in Retrieval Augmented Generation",
            "abstract": (
                "This paper presents novel techniques for improving RAG systems. "
                "We demonstrate significant improvements in accuracy and efficiency."
            ),
            "url": "https://arxiv.org/abs/2401.00001",
            "published": "2024-01-01",
            "authors": ["John Doe", "Jane Smith"],
            "source": "arxiv",
        },
        {
            "id": "arxiv:2401.00002",
            "title": "RAG for Domain-Specific Applications",
            "abstract": (
                "We explore RAG applications in specialized domains. "
                "Our approach shows promise for medical and legal contexts."
            ),
            "url": "https://arxiv.org/abs/2401.00002",
            "published": "2024-01-02",
            "authors": ["Alice Johnson"],
            "source": "arxiv",
        },
        {
            "id": "pubmed:12345678",
            "title": "Clinical Applications of RAG",
            "abstract": (
                "RAG systems show potential for clinical decision support. "
                "We evaluate performance on medical question answering."
            ),
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "published": "2024-01-03",
            "authors": ["Dr. Bob Wilson"],
            "source": "pubmed",
        },
    ]


@pytest.fixture
def sample_web_results():
    """Sample web search results"""
    return [
        {
            "title": "What is RAG? A Complete Guide",
            "content": (
                "Retrieval Augmented Generation (RAG) is a technique that "
                "enhances language models by retrieving relevant information "
                "from external sources before generating responses."
            ),
            "url": "https://example.com/rag-guide",
            "score": 0.95,
        },
        {
            "title": "RAG Systems in Production",
            "content": (
                "Implementing RAG in production requires careful consideration "
                "of latency, accuracy, and cost trade-offs."
            ),
            "url": "https://example.com/rag-production",
            "score": 0.87,
        },
    ]


@pytest.fixture
def sample_summary():
    """Sample summary text"""
    return (
        "Retrieval Augmented Generation (RAG) is an advanced technique that "
        "enhances large language models by incorporating external knowledge retrieval. "
        "The approach combines information retrieval with neural text generation, "
        "allowing models to access up-to-date information beyond their training data. "
        "Studies show RAG systems can improve factual accuracy by 40% compared to "
        "standard language models."
    )


@pytest.fixture
def sample_verification_result():
    """Sample verification result"""
    return {
        "status": "accepted",
        "confidence": 0.92,
        "hallucination_ratio": 0.05,
        "claims": [
            {
                "text": "RAG enhances large language models",
                "supported": True,
                "evidence_score": 0.95,
            },
            {
                "text": "RAG improves factual accuracy by 40%",
                "supported": True,
                "evidence_score": 0.88,
            },
            {
                "text": "RAG combines retrieval with generation",
                "supported": True,
                "evidence_score": 0.92,
            },
        ],
        "iterations": 1,
    }


# ============================================================================
# Mock objects fixtures
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = MagicMock()
    
    # Mock chat completion
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = "This is a test response."
    
    client.chat.completions.create = AsyncMock(return_value=completion)
    
    return client


@pytest.fixture
def mock_embedder():
    """Mock sentence transformer embedder"""
    embedder = MagicMock()
    
    # Mock encode method
    embedder.encode = MagicMock(return_value=[
        [0.1, 0.2, 0.3, 0.4],  # Mock embedding vectors
        [0.2, 0.3, 0.4, 0.5],
    ])
    
    return embedder


@pytest.fixture
def mock_tool_executor():
    """Mock MCP tool executor"""
    executor = MagicMock()
    
    async def mock_execute(tool_name: str, params: Dict) -> Dict:
        if tool_name == "arxiv_search":
            return {
                "success": True,
                "result": [
                    {
                        "id": "arxiv:2401.00001",
                        "title": "Test Paper",
                        "abstract": "Test abstract",
                        "url": "https://arxiv.org/abs/2401.00001",
                        "published": "2024-01-01",
                        "authors": ["Test Author"],
                    }
                ],
            }
        elif tool_name == "web_search":
            return {
                "success": True,
                "result": [
                    {
                        "title": "Test Result",
                        "content": "Test content",
                        "url": "https://example.com",
                    }
                ],
            }
        return {"success": False, "error": "Unknown tool"}
    
    executor.execute_tool = AsyncMock(side_effect=mock_execute)
    
    return executor


@pytest.fixture
def mock_mcp_message():
    """Create a mock MCP message"""
    from src.utils.mcp_schema import MCPMessage, MessageType
    
    return MCPMessage(
        message_type=MessageType.QUERY,
        sender_agent="test_agent",
        context_id="test_context_123",
        content={"query": "test query", "max_results": 10},
    )


# ============================================================================
# Component fixtures
# ============================================================================

@pytest.fixture
def mcp_retriever(test_api_keys, mock_embedder, mock_tool_executor, monkeypatch):
    """Create MCPRetrieverAgent with mocked dependencies"""
    from src.agents.mcp_retriever import MCPRetrieverAgent
    
    # Patch SentenceTransformer to avoid loading the actual model
    monkeypatch.setattr(
        "src.agents.mcp_retriever.SentenceTransformer",
        lambda x: mock_embedder
    )
    
    agent = MCPRetrieverAgent(
        test_api_keys["openai"],
        test_api_keys["tavily"]
    )
    
    # Replace tool executor with mock
    agent.tool_executor = mock_tool_executor
    agent.embedder = mock_embedder
    
    return agent


@pytest.fixture
def summarizer_agent(test_api_keys, mock_openai_client, monkeypatch):
    """Create SummarizerAgent with mocked dependencies"""
    from src.agents.summarizer import SummarizerAgent
    
    # Patch OpenAI client
    monkeypatch.setattr(
        "src.agents.summarizer.OpenAI",
        lambda api_key: mock_openai_client
    )
    
    agent = SummarizerAgent(test_api_keys["openai"])
    agent.client = mock_openai_client
    
    return agent


@pytest.fixture
def verifier_agent(test_api_keys, mock_openai_client, monkeypatch):
    """Create ThoroughMCPVerifier with mocked dependencies"""
    from src.agents.thorough_mcp_verifier import ThoroughMCPVerifier
    
    # Patch OpenAI and other dependencies
    monkeypatch.setattr(
        "src.agents.thorough_mcp_verifier.OpenAI",
        lambda api_key: mock_openai_client
    )
    
    agent = ThoroughMCPVerifier(
        test_api_keys["openai"],
        test_api_keys["tavily"]
    )
    agent.client = mock_openai_client
    
    return agent


@pytest.fixture
def multi_mode_assistant(test_api_keys, monkeypatch):
    """Create MultiModeResearchAssistant with mocked dependencies"""
    from src.multi_mode_assistant import MultiModeResearchAssistant
    from unittest.mock import MagicMock
    
    # Mock the SentenceTransformer to avoid loading models
    mock_embedder = MagicMock()
    mock_embedder.encode = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    
    monkeypatch.setattr(
        "src.agents.mcp_retriever.SentenceTransformer",
        lambda x: mock_embedder
    )
    
    # Mock transformers AutoTokenizer and AutoModelForSequenceClassification
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
    
    return assistant


# ============================================================================
# Utility fixtures
# ============================================================================

@pytest.fixture
def temp_results_dir(tmp_path):
    """Create temporary directory for test results"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture
def sample_benchmark_results():
    """Sample benchmark results for testing"""
    from src.benchmark import BenchmarkResult
    
    return [
        BenchmarkResult(
            query="What is RAG?",
            mode="mcp_verified",
            latency_seconds=25.3,
            confidence=0.92,
            num_sources=8,
            answer_length=1234,
            timestamp="2024-01-01T00:00:00",
            hallucination_ratio=0.05,
            verification_iterations=2,
            gt_f1=0.78,
            gt_semantic=0.82,
            gt_bert=0.85,
            gt_rouge_l=0.80,
        ),
        BenchmarkResult(
            query="How does RLHF work?",
            mode="mcp_verified",
            latency_seconds=28.7,
            confidence=0.89,
            num_sources=9,
            answer_length=1456,
            timestamp="2024-01-01T00:05:00",
            hallucination_ratio=0.08,
            verification_iterations=1,
            gt_f1=0.75,
            gt_semantic=0.79,
            gt_bert=0.82,
            gt_rouge_l=0.77,
        ),
    ]


# ============================================================================
# Markers for test categorization
# ============================================================================

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring API keys"
    )
