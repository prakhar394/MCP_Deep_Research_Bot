from typing import Dict, List, Tuple
from datetime import datetime

from .agents.mcp_retriever import MCPRetrieverAgent
from .agents.summarizer import SummarizerAgent
from .agents.thorough_mcp_verifier import ThoroughMCPVerifier
from .utils.mcp_schema import MCPMessage, MessageType
from .utils.logger import get_logger

logger = get_logger(__name__)


class MCPResearchAssistant:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.retriever = MCPRetrieverAgent(openai_api_key, tavily_api_key)
        self.summarizer = SummarizerAgent(openai_api_key)
        self.verifier = ThoroughMCPVerifier(openai_api_key, tavily_api_key)
        self.max_iterations = 1
        logger.info("MCPResearchAssistant initialized")

    async def answer_query(
        self, query: str, max_papers: int = 10, sources: List[str] = None
    ) -> Dict[str, any]:
        if sources is None:
            sources = ["arxiv"]

        context_id = f"ctx_{datetime.utcnow().timestamp()}"

        papers = await self._retrieve_papers(query, max_papers, sources, context_id)
        if not papers:
            return self._no_results_response(query)

        summary, verification = await self._generate_and_verify(
            query, papers, context_id
        )

        response = self._format_response(query, summary, papers, verification)
        return response

    async def _retrieve_papers(
        self, query: str, max_papers: int, sources: List[str], context_id: str
    ) -> List[Dict]:
        msg = MCPMessage(
            message_type=MessageType.QUERY,
            sender_agent="MCPResearchAssistant",
            context_id=context_id,
            content={"query": query, "max_results": max_papers, "sources": sources},
        )
        res = await self.retriever.process(msg)
        return res.content.get("documents", []) if res else []

    async def _generate_and_verify(
        self,
        query: str,
        papers: List[Dict],
        context_id: str,
        iteration: int = 0,
    ) -> Tuple[str, Dict]:
        if iteration >= self.max_iterations:
            return "", {"confidence": 0.3, "status": "max_iterations"}

        summary, meta = await self._generate_summary(query, papers, context_id)
        verification = await self._verify_summary(query, summary, papers, context_id)

        conf = verification.get("confidence", 0.0)
        status = verification.get("status", "accept")

        if conf >= 0.30 or status == "accept":
            return summary, verification

        if status == "re_retrieve":
            extra = await self._enhanced_retrieval(query, verification, context_id)
            papers.extend(extra)
        elif status == "revise":
            summary = await self._revise_summary(summary, verification, papers)

        return await self._generate_and_verify(
            query, papers, context_id, iteration + 1
        )

    async def _generate_summary(
        self, query: str, papers: List[Dict], context_id: str
    ) -> Tuple[str, Dict]:
        msg = MCPMessage(
            message_type=MessageType.RETRIEVAL,
            sender_agent="MCPResearchAssistant",
            context_id=context_id,
            content={"query": query, "documents": papers},
        )
        res = await self.summarizer.process(msg)
        return res.content.get("summary", ""), {"confidence": res.confidence_score}

    async def _verify_summary(
        self, query: str, summary: str, papers: List[Dict], context_id: str
    ) -> Dict:
        msg = MCPMessage(
            message_type=MessageType.SUMMARY,
            sender_agent="MCPResearchAssistant",
            context_id=context_id,
            content={"query": query, "summary": summary, "source_documents": papers},
        )
        res = await self.verifier.process(msg)
        return res.content

    async def _enhanced_retrieval(
        self, original_query: str, verification: Dict, context_id: str
    ) -> List[Dict]:
        suggested = verification.get("suggested_queries", [])
        if not suggested:
            return []
        enhanced_query = f"{original_query} {' '.join(suggested[:2])}"
        return await self._retrieve_papers(enhanced_query, 5, ["arxiv"], context_id)

    async def _revise_summary(
        self, original_summary: str, verification: Dict, papers: List[Dict]
    ) -> str:
        from openai import OpenAI

        client = OpenAI()
        suggestions = verification.get("revision_suggestions", [])
        sug_text = "\n".join(f"- {s}" for s in suggestions)
        prompt = f"""Revise the following summary to address the issues.

Summary:
{original_summary}

Issues:
{sug_text}

Return the revised summary:"""
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You revise summaries to be accurate."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    def _format_response(
        self, query: str, summary: str, papers: List[Dict], verification: Dict
    ) -> Dict:
        conf = verification.get("confidence", 0.0)
        answer = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"📚 Research Summary for:\n{query}\n\n"
        answer += summary + "\n\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += "📄 Top Sources:\n\n"
        for i, p in enumerate(papers[:5], start=1):
            answer += f"[{i}] {p['title']}\n"
            answer += f"    {p['url']}\n\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"✓ Confidence: {conf:.0%} | {len(papers)} papers\n"
        answer += f"✓ Semantic Consistency Confidence: {verification['confidence']*100:.1f}%\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        return {
            "answer": answer,
            "confidence": conf,
            "sources": papers[:10],
            "query": query,
            "verification_details": verification,
        }

    def _no_results_response(self, query: str) -> Dict:
        return {
            "answer": f"No relevant papers found for: {query}",
            "confidence": 0.0,
            "sources": [],
            "query": query,
            "verification_details": {},
        }