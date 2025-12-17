"""
Unit tests for ResearchBenchmark and benchmark_accuracy
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.benchmark import ResearchBenchmark, BenchmarkResult, BenchmarkSummary
from src.benchmark_accuracy import (
    token_f1, semantic_cosine, rouge_l_f1, bert_score_f1
)
from src.multi_mode_assistant import ResearchMode


@pytest.mark.unit
class TestBenchmarkAccuracyMetrics:
    """Test suite for accuracy metrics"""
    
    def test_token_f1_identical_strings(self):
        """Test F1 score for identical strings"""
        text = "This is a test sentence"
        score = token_f1(text, text)
        
        assert score == 1.0
    
    def test_token_f1_completely_different(self):
        """Test F1 score for completely different strings"""
        pred = "apple banana cherry"
        gold = "dog elephant fox"
        score = token_f1(pred, gold)
        
        assert score == 0.0
    
    def test_token_f1_partial_overlap(self):
        """Test F1 score for partial overlap"""
        pred = "the cat sat on the mat"
        gold = "the dog sat on the floor"
        score = token_f1(pred, gold)
        
        # Should have partial overlap
        assert 0 < score < 1
    
    def test_token_f1_case_insensitive(self):
        """Test that F1 is case-insensitive"""
        pred = "HELLO WORLD"
        gold = "hello world"
        score = token_f1(pred, gold)
        
        assert score == 1.0
    
    def test_token_f1_punctuation_handling(self):
        """Test that punctuation is handled correctly"""
        pred = "Hello, world!"
        gold = "hello world"
        score = token_f1(pred, gold)
        
        # Should ignore punctuation
        assert score == 1.0
    
    def test_token_f1_empty_strings(self):
        """Test F1 with empty strings"""
        assert token_f1("", "") == 1.0
        assert token_f1("test", "") == 0.0
        assert token_f1("", "test") == 0.0
    
    def test_semantic_cosine_identical(self):
        """Test cosine similarity for identical texts"""
        text = "machine learning models"
        score = semantic_cosine(text, text)
        
        assert score == 1.0
    
    def test_semantic_cosine_different(self):
        """Test cosine similarity for different texts"""
        text1 = "artificial intelligence"
        text2 = "cooking recipes"
        score = semantic_cosine(text1, text2)
        
        assert score == 0.0
    
    def test_semantic_cosine_partial(self):
        """Test cosine similarity for partial overlap"""
        text1 = "deep learning neural networks"
        text2 = "neural networks for vision"
        score = semantic_cosine(text1, text2)
        
        assert 0 < score < 1
    
    def test_semantic_cosine_empty(self):
        """Test cosine similarity with empty strings"""
        assert semantic_cosine("", "") == 1.0
        assert semantic_cosine("test", "") == 0.0
    
    def test_rouge_l_f1_identical(self):
        """Test ROUGE-L for identical strings"""
        text = "the quick brown fox"
        score = rouge_l_f1(text, text)
        
        assert score == 1.0
    
    def test_rouge_l_f1_subsequence(self):
        """Test ROUGE-L for texts with common subsequence"""
        pred = "the quick fox jumps"
        gold = "the brown fox sleeps"
        score = rouge_l_f1(pred, gold)
        
        # Should find common subsequence "the ... fox"
        assert score > 0
    
    def test_rouge_l_f1_no_overlap(self):
        """Test ROUGE-L with no overlap"""
        pred = "apple banana"
        gold = "cat dog"
        score = rouge_l_f1(pred, gold)
        
        assert score == 0.0
    
    def test_rouge_l_f1_empty(self):
        """Test ROUGE-L with empty strings"""
        assert rouge_l_f1("", "") == 1.0
        assert rouge_l_f1("test", "") == 0.0
    
    def test_bert_score_f1_returns_valid_or_none(self):
        """Test that BERTScore returns valid score or None"""
        score = bert_score_f1("test text", "test text")
        
        # Either returns valid score or None (if not installed)
        assert score is None or (0 <= score <= 1)
    
    def test_bert_score_f1_empty_strings(self):
        """Test BERTScore with empty strings"""
        score = bert_score_f1("", "")
        
        if score is not None:
            assert score == 1.0


@pytest.mark.unit
class TestResearchBenchmark:
    """Test suite for ResearchBenchmark"""
    
    def test_benchmark_initialization(self, test_api_keys, monkeypatch):
        """Test benchmark initialization"""
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
        
        benchmark = ResearchBenchmark(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        assert benchmark.assistant is not None
        assert benchmark.results == []
    
    @pytest.mark.asyncio
    async def test_run_single_query_simple_mode(
        self, test_api_keys, sample_query, monkeypatch
    ):
        """Test running single query benchmark"""
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
        
        benchmark = ResearchBenchmark(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock assistant response
        benchmark.assistant.answer_query = AsyncMock(
            return_value={
                "answer": "Test answer",
                "confidence": 0.9,
                "sources": [{"title": "Test"}],
                "mode": ResearchMode.SIMPLE_WEB_RAG,
                "metrics": {"latency_seconds": 5.0}
            }
        )
        
        result = await benchmark.run_single_query(
            query=sample_query,
            mode=ResearchMode.SIMPLE_WEB_RAG,
            max_papers=10
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.query == sample_query
        assert result.mode == ResearchMode.SIMPLE_WEB_RAG.value
        assert result.latency_seconds > 0
    
    @pytest.mark.asyncio
    async def test_run_single_query_with_ground_truth(
        self, test_api_keys, sample_query, monkeypatch
    ):
        """Test benchmark with ground truth comparison"""
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
        
        benchmark = ResearchBenchmark(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        benchmark.assistant.answer_query = AsyncMock(
            return_value={
                "answer": "Test answer about RAG",
                "confidence": 0.85,
                "sources": [],
                "mode": ResearchMode.MCP_VERIFIED,
                "metrics": {"latency_seconds": 25.0}
            }
        )
        
        gold_answer = "RAG is a technique for enhancing language models"
        
        result = await benchmark.run_single_query(
            query=sample_query,
            mode=ResearchMode.MCP_VERIFIED,
            max_papers=10,
            gold_answer=gold_answer
        )
        
        assert result.gt_f1 is not None
        assert 0 <= result.gt_f1 <= 1
        assert result.gt_semantic is not None
        assert result.gt_rouge_l is not None
    
    @pytest.mark.asyncio
    async def test_run_benchmark_multiple_queries(
        self, test_api_keys, monkeypatch
    ):
        """Test running benchmark with multiple queries"""
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
        
        benchmark = ResearchBenchmark(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        benchmark.assistant.answer_query = AsyncMock(
            return_value={
                "answer": "Test",
                "confidence": 0.9,
                "sources": [],
                "mode": ResearchMode.SIMPLE_WEB_RAG,
                "metrics": {"latency_seconds": 5.0}
            }
        )
        
        queries = [
            "What is RAG?",
            "How does RLHF work?",
            "Explain transformers"
        ]
        
        results = await benchmark.run_benchmark(
            queries=queries,
            modes=[ResearchMode.SIMPLE_WEB_RAG],
            max_papers=10
        )
        
        assert len(results) >= len(queries)
        for result in results:
            assert isinstance(result, BenchmarkResult)
    
    def test_compute_summary(self, sample_benchmark_results):
        """Test computing benchmark summary"""
        from src.benchmark import ResearchBenchmark
        
        summary = ResearchBenchmark._compute_summary(
            sample_benchmark_results
        )
        
        assert isinstance(summary, BenchmarkSummary)
        assert summary.num_queries == len(sample_benchmark_results)
        assert summary.avg_latency > 0
        assert 0 <= summary.success_rate <= 1
    
    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass"""
        result = BenchmarkResult(
            query="test",
            mode="mcp_verified",
            latency_seconds=10.0,
            confidence=0.9,
            num_sources=5,
            answer_length=500,
            timestamp="2024-01-01T00:00:00"
        )
        
        assert result.query == "test"
        assert result.latency_seconds == 10.0
        assert result.confidence == 0.9
    
    def test_benchmark_summary_dataclass(self):
        """Test BenchmarkSummary dataclass"""
        summary = BenchmarkSummary(
            mode="mcp_verified",
            num_queries=10,
            avg_latency=25.0,
            min_latency=15.0,
            max_latency=35.0,
            avg_confidence=0.9,
            avg_sources=8.5,
            avg_answer_length=1200,
            success_rate=1.0,
            total_time=250.0
        )
        
        assert summary.num_queries == 10
        assert summary.avg_latency == 25.0
        assert summary.success_rate == 1.0
