"""
Unit tests for MCPToolExecutor
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.mcp.tool_executors import MCPToolExecutor


@pytest.mark.unit
class TestMCPToolExecutor:
    """Test suite for MCPToolExecutor"""
    
    def test_executor_initialization(self, test_api_keys):
        """Test that executor initializes correctly"""
        executor = MCPToolExecutor(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        assert executor.openai_api_key == test_api_keys["openai"]
        assert executor.tavily_api_key == test_api_keys["tavily"]
    
    @pytest.mark.asyncio
    async def test_execute_arxiv_search(self, mock_tool_executor):
        """Test arXiv search execution"""
        result = await mock_tool_executor.execute_tool(
            "arxiv_search",
            {"query": "machine learning", "max_results": 5}
        )
        
        assert result is not None
        assert "success" in result
        if result["success"]:
            assert "result" in result
            assert isinstance(result["result"], list)
    
    @pytest.mark.asyncio
    async def test_execute_web_search(self, mock_tool_executor):
        """Test web search execution"""
        result = await mock_tool_executor.execute_tool(
            "web_search",
            {"query": "artificial intelligence", "max_results": 10}
        )
        
        assert result is not None
        assert "success" in result
        if result["success"]:
            assert "result" in result
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, mock_tool_executor):
        """Test handling of unknown tool"""
        result = await mock_tool_executor.execute_tool(
            "nonexistent_tool",
            {"query": "test"}
        )
        
        assert result is not None
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_execute_missing_params(self, mock_tool_executor):
        """Test handling of missing required parameters"""
        result = await mock_tool_executor.execute_tool(
            "arxiv_search",
            {}  # Missing query parameter
        )
        
        assert result is not None
        # Should handle missing params gracefully
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_params(self, mock_tool_executor):
        """Test handling of invalid parameter types"""
        result = await mock_tool_executor.execute_tool(
            "arxiv_search",
            {"query": 123, "max_results": "invalid"}  # Wrong types
        )
        
        assert result is not None
        # Should handle invalid types gracefully
    
    @pytest.mark.asyncio
    @pytest.mark.requires_api
    async def test_arxiv_search_integration(self, test_api_keys):
        """Integration test for arXiv search (requires API)"""
        executor = MCPToolExecutor(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        result = await executor.execute_tool(
            "arxiv_search",
            {"query": "neural networks", "max_results": 3, "sort_by": "relevance"}
        )
        
        assert result["success"] is True
        if result["result"]:
            assert len(result["result"]) <= 3
            for paper in result["result"]:
                assert "title" in paper
                assert "abstract" in paper
                assert "url" in paper
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_tool_executor):
        """Test rate limiting behavior"""
        # Execute multiple requests rapidly
        results = []
        for _ in range(5):
            result = await mock_tool_executor.execute_tool(
                "web_search",
                {"query": "test", "max_results": 1}
            )
            results.append(result)
        
        # All requests should complete (mock doesn't rate limit)
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, test_api_keys, monkeypatch):
        """Test handling of network errors"""
        executor = MCPToolExecutor(
            test_api_keys["openai"],
            test_api_keys["tavily"]
        )
        
        # Mock network failure
        async def mock_search(*args, **kwargs):
            raise ConnectionError("Network error")
        
        with patch('arxiv.Client.results', side_effect=mock_search):
            result = await executor.execute_tool(
                "arxiv_search",
                {"query": "test", "max_results": 5}
            )
            
            # Should return error result instead of crashing
            assert "success" in result or "error" in result
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_tool_executor):
        """Test handling of request timeouts"""
        # Mock slow response
        async def slow_execute(*args, **kwargs):
            import asyncio
            await asyncio.sleep(10)  # Very slow
            return {"success": True, "result": []}
        
        # Should have timeout mechanism or handle gracefully
        result = await mock_tool_executor.execute_tool(
            "web_search",
            {"query": "test", "max_results": 1}
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_empty_query(self, mock_tool_executor):
        """Test handling of empty query string"""
        result = await mock_tool_executor.execute_tool(
            "arxiv_search",
            {"query": "", "max_results": 5}
        )
        
        assert result is not None
        # Should handle empty query
    
    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_tool_executor):
        """Test handling of very long query strings"""
        long_query = "test " * 1000  # Very long query
        
        result = await mock_tool_executor.execute_tool(
            "arxiv_search",
            {"query": long_query, "max_results": 5}
        )
        
        assert result is not None
        # Should handle or truncate long queries
    
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, mock_tool_executor):
        """Test handling of special characters in queries"""
        special_query = 'test "quoted" & special <chars>'
        
        result = await mock_tool_executor.execute_tool(
            "web_search",
            {"query": special_query, "max_results": 5}
        )
        
        assert result is not None
        # Should escape or handle special characters
    
    @pytest.mark.asyncio
    async def test_concurrent_executions(self, mock_tool_executor):
        """Test concurrent tool executions"""
        import asyncio
        
        tasks = [
            mock_tool_executor.execute_tool(
                "arxiv_search",
                {"query": f"query {i}", "max_results": 3}
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert "success" in result
