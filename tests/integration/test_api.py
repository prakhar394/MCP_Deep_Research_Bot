"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self, monkeypatch):
        """Create test client with mocked dependencies"""
        # Mock dependencies before importing app
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
        
        # Set test environment variables
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        monkeypatch.setenv("TAVILY_API_KEY", "test_key")
        
        # Import and create client
        from api_multimode import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_research_endpoint_simple_web_rag(self, client, monkeypatch):
        """Test research endpoint with simple_web_rag mode"""
        # Mock the assistant's answer_query method
        from api_multimode import assistant
        
        mock_result = {
            "answer": "Test answer",
            "confidence": None,
            "sources": [{"title": "Test", "url": "http://test.com"}],
            "query": "What is RAG?",
            "mode": "simple_web_rag",
            "metrics": {
                "latency_seconds": 5.0,
                "mode": "simple_web_rag",
                "timestamp": "2024-01-01T00:00:00"
            }
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            response = client.post(
                "/research",
                json={
                    "query": "What is RAG?",
                    "mode": "simple_web_rag",
                    "max_papers": 10
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["mode"] == "simple_web_rag"
    
    def test_research_endpoint_mcp_verified(self, client, monkeypatch):
        """Test research endpoint with mcp_verified mode"""
        from api_multimode import assistant
        
        mock_result = {
            "answer": "Detailed answer",
            "confidence": 0.92,
            "sources": [{"title": "Paper", "url": "http://arxiv.org/abs/123"}],
            "query": "What is RAG?",
            "mode": "mcp_verified",
            "metrics": {
                "latency_seconds": 25.0,
                "mode": "mcp_verified",
                "timestamp": "2024-01-01T00:00:00"
            },
            "verification_details": {
                "status": "accepted",
                "confidence": 0.92,
                "hallucination_ratio": 0.05
            }
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            response = client.post(
                "/research",
                json={
                    "query": "What is RAG?",
                    "mode": "mcp_verified",
                    "max_papers": 10
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert "verification_details" in data
    
    def test_research_endpoint_invalid_mode(self, client):
        """Test research endpoint with invalid mode"""
        response = client.post(
            "/research",
            json={
                "query": "What is RAG?",
                "mode": "invalid_mode",
                "max_papers": 10
            }
        )
        
        assert response.status_code == 400
    
    def test_research_endpoint_missing_query(self, client):
        """Test research endpoint with missing query"""
        response = client.post(
            "/research",
            json={
                "mode": "simple_web_rag",
                "max_papers": 10
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_research_endpoint_max_papers_validation(self, client, monkeypatch):
        """Test max_papers parameter validation"""
        from api_multimode import assistant
        
        mock_result = {
            "answer": "Test",
            "sources": [],
            "query": "test",
            "mode": "simple_web_rag",
            "metrics": {"latency_seconds": 1.0}
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            # Test invalid max_papers (too high)
            response = client.post(
                "/research",
                json={
                    "query": "test",
                    "mode": "simple_web_rag",
                    "max_papers": 100  # Should be limited to 20
                }
            )
            
            assert response.status_code == 422
    
    def test_compare_endpoint(self, client, monkeypatch):
        """Test compare endpoint (all modes)"""
        from api_multimode import assistant
        
        mock_result = {
            "answer": "Test answer",
            "sources": [],
            "query": "What is RAG?",
            "mode": "simple_web_rag",
            "metrics": {"latency_seconds": 5.0}
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            response = client.post(
                "/compare",
                json={
                    "query": "What is RAG?",
                    "max_papers": 10
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query"] == "What is RAG?"
    
    def test_benchmark_endpoint_open_eval(self, client, monkeypatch):
        """Test benchmark endpoint with open evaluation"""
        from api_multimode import assistant
        
        mock_result = {
            "answer": "Test",
            "sources": [],
            "query": "test",
            "mode": "simple_web_rag",
            "metrics": {"latency_seconds": 5.0}
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            response = client.post(
                "/benchmark",
                json={
                    "eval_set": "open",
                    "queries": ["What is RAG?", "How does RLHF work?"],
                    "max_papers": 10
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "summaries" in data
    
    def test_benchmark_endpoint_missing_queries(self, client):
        """Test benchmark endpoint with missing queries for open eval"""
        response = client.post(
            "/benchmark",
            json={
                "eval_set": "open",
                "max_papers": 10
                # Missing queries
            }
        )
        
        assert response.status_code == 400
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options(
            "/research",
            headers={"Origin": "http://localhost:5173"}
        )
        
        # CORS should be configured
        assert response.status_code in [200, 204]
    
    def test_api_error_handling(self, client, monkeypatch):
        """Test API error handling"""
        from api_multimode import assistant
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = Exception("Internal error")
            
            response = client.post(
                "/research",
                json={
                    "query": "test",
                    "mode": "simple_web_rag",
                    "max_papers": 10
                }
            )
            
            # Should return error response, not crash
            assert response.status_code in [500, 200]
    
    def test_concurrent_requests(self, client, monkeypatch):
        """Test handling of concurrent requests"""
        from api_multimode import assistant
        import concurrent.futures
        
        mock_result = {
            "answer": "Test",
            "sources": [],
            "query": "test",
            "mode": "simple_web_rag",
            "metrics": {"latency_seconds": 1.0}
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            def make_request():
                return client.post(
                    "/research",
                    json={
                        "query": "test",
                        "mode": "simple_web_rag",
                        "max_papers": 5
                    }
                )
            
            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(5)]
                responses = [f.result() for f in futures]
            
            # All should succeed
            assert all(r.status_code == 200 for r in responses)
    
    def test_response_time_reasonable(self, client, monkeypatch):
        """Test that API response time is reasonable"""
        from api_multimode import assistant
        import time
        
        mock_result = {
            "answer": "Test",
            "sources": [],
            "query": "test",
            "mode": "simple_web_rag",
            "metrics": {"latency_seconds": 0.1}
        }
        
        with patch.object(assistant, 'answer_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = mock_result
            
            start = time.time()
            response = client.post(
                "/research",
                json={
                    "query": "test",
                    "mode": "simple_web_rag",
                    "max_papers": 5
                }
            )
            elapsed = time.time() - start
            
            assert response.status_code == 200
            # API overhead should be minimal
            assert elapsed < 5.0
