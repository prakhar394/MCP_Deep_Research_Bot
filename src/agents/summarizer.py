from typing import Dict, List
from openai import OpenAI

from .base_agent import BaseAgent
from ..utils.mcp_schema import MCPMessage, MessageType


class SummarizerAgent(BaseAgent):
    def __init__(self, openai_api_key: str, model: str = "gpt-4.1-mini"):
        super().__init__("Summarizer")
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    async def process(self, message: MCPMessage) -> MCPMessage:
        query: str = message.content.get("query", "")
        docs: List[Dict] = message.content.get("documents", [])

        bullets = []
        for p in docs[:8]:
            bullets.append(f"- {p['title']} — {p['abstract'][:400]}")

        prompt = f"""You are a precise research assistant.

User question:
{query}

Relevant papers:
{chr(10).join(bullets)}

Write a detailed, structured answer that:
- Synthesizes key ideas
- Explains trade-offs and limitations
- Clearly answers the question
- Stays grounded ONLY in the given papers.
"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a careful research summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        summary = resp.choices[0].message.content

        return await self.send_message(
            MessageType.SUMMARY,
            message.context_id,
            {"summary": summary, "query": query, "source_documents": docs},
            confidence_score=0.8,
            parent_message_id=message.message_id,
        )
