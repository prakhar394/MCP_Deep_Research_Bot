from typing import Dict, List, Any
from pydantic import BaseModel


class MCPToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class ArxivSearchTool(MCPToolDefinition):
    name: str = "arxiv_search"
    description: str = "Search arXiv for academic papers."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Max results",
                "default": 10,
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                "default": "relevance",
            },
        },
        "required": ["query"],
    }


class WebSearchTool(MCPToolDefinition):
    name: str = "web_search"
    description: str = "Web search for fact-checking (via Tavily)."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "advanced",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
            },
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }


class FetchPaperTool(MCPToolDefinition):
    name: str = "fetch_paper"
    description: str = "Download and extract basic text from paper PDF."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["url"],
    }


class ExtractClaimsTool(MCPToolDefinition):
    name: str = "extract_claims"
    description: str = "Extract atomic, verifiable claims from text."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "claim_type": {
                "type": "string",
                "enum": ["factual", "numerical", "comparative", "all"],
                "default": "all",
            },
        },
        "required": ["text"],
    }


class VerifyClaimTool(MCPToolDefinition):
    name: str = "verify_claim"
    description: str = "Verify a claim using NLI model + optional web search."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
            },
            "use_external_search": {
                "type": "boolean",
                "default": True,
            },
        },
        "required": ["claim", "evidence"],
    }


class CacheResultTool(MCPToolDefinition):
    name: str = "cache_result"
    description: str = "Cache results in a simple in-memory store."
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "object"},
            "ttl": {"type": "integer", "default": 3600},
        },
        "required": ["key", "value"],
    }


MCP_TOOLS: List[MCPToolDefinition] = [
    ArxivSearchTool(),
    WebSearchTool(),
    FetchPaperTool(),
    ExtractClaimsTool(),
    VerifyClaimTool(),
    CacheResultTool(),
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in MCP_TOOLS
    ]
