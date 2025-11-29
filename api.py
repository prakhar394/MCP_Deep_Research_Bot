import os
import asyncio
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.mcp_research_assistant import MCPResearchAssistant

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

assistant = MCPResearchAssistant(OPENAI_API_KEY, TAVILY_API_KEY)

app = FastAPI(title="MCP Research Assistant API")


class QueryRequest(BaseModel):
    query: str
    max_papers: int = 10


@app.post("/ask")
async def ask(req: QueryRequest):
    result = await assistant.answer_query(req.query, max_papers=req.max_papers)
    return result
