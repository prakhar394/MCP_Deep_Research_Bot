# src/multi_mode_assistant.py

"""
Multi-Mode Research Assistant

Supports 3 modes:
1. Simple Web-RAG: Direct web search + basic summarization (no MCP, no verification)
2. MCP-Basic: MCP tools + summarization (no verification loop)
3. MCP-Verified: Full pipeline with MCP tools + verification loop (current implementation)
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
from enum import Enum

from openai import OpenAI
from .agents.mcp_retriever import MCPRetrieverAgent
from .agents.summarizer import SummarizerAgent
from .agents.thorough_mcp_verifier import ThoroughMCPVerifier
from .mcp.tool_executors import MCPToolExecutor
from .utils.mcp_schema import MCPMessage, MessageType
from .utils.logger import get_logger

logger = get_logger(__name__)


class ResearchMode(str, Enum):
    """Research assistant operational modes"""
    SIMPLE_WEB_RAG = "simple_web_rag"
    MCP_BASIC = "mcp_basic"
    MCP_VERIFIED = "mcp_verified"


class MultiModeResearchAssistant:
    """
    Research assistant that can operate in 3 different modes for
    accuracy and latency testing.
    """

    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize components for MCP-based modes
        self.mcp_retriever = MCPRetrieverAgent(openai_api_key, tavily_api_key)
        self.summarizer = SummarizerAgent(openai_api_key)
        self.verifier = ThoroughMCPVerifier(openai_api_key, tavily_api_key)
        self.tool_executor = MCPToolExecutor(openai_api_key, tavily_api_key)

        # Verification settings (for MCP-Verified mode)
        self.max_iterations = 5
        self.accept_confidence_threshold = 0.97

        logger.info("MultiModeResearchAssistant initialized")

    async def answer_query(
        self,
        query: str,
        mode: ResearchMode = ResearchMode.MCP_VERIFIED,
        max_papers: int = 10,
        sources: List[str] = None,
    ) -> Dict:
        """
        Answer a research query using the specified mode.

        Args:
            query: Research question
            mode: Operating mode (simple_web_rag, mcp_basic, mcp_verified)
            max_papers: Maximum number of papers to retrieve
            sources: List of sources to search (arxiv, pubmed)

        Returns:
            Dict containing answer, confidence, sources, and metrics
        """
        start_time = time.time()

        logger.info(f"Processing query in {mode} mode: {query}")

        # Route to appropriate mode handler
        if mode == ResearchMode.SIMPLE_WEB_RAG:
            result = await self._simple_web_rag_mode(query, max_papers)
        elif mode == ResearchMode.MCP_BASIC:
            result = await self._mcp_basic_mode(query, max_papers, sources)
        elif mode == ResearchMode.MCP_VERIFIED:
            result = await self._mcp_verified_mode(query, max_papers, sources)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Add execution metrics
        result["metrics"] = {
            "mode": mode,
            "latency_seconds": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat(),
        }

        confidence = result.get('confidence') or 0.0
        logger.info(
            f"Query completed in {result['metrics']['latency_seconds']:.2f}s "
            f"with confidence {confidence:.2%}"
        )

        return result

    # ================================================================
    # MODE 1: Simple Web-RAG (No MCP, No Judge)
    # ================================================================

    async def _simple_web_rag_mode(self, query: str, max_results: int = 10) -> Dict:
        """
        Simple web search + basic RAG summarization.
        No MCP framework, no verification loop.
        """
        logger.info("[Simple Web-RAG] Starting basic web search")

        # Direct Tavily web search
        search_result = await self.tool_executor.execute_tool(
            "web_search",
            {"query": query, "max_results": max_results},
        )

        if not search_result["success"] or not search_result["result"]:
            return self._no_results_response(query, ResearchMode.SIMPLE_WEB_RAG)

        # Convert web results to document format
        documents = []
        for item in search_result["result"]:
            doc = {
                "title": item.get("title", "Untitled"),
                "abstract": item.get("content") or item.get("snippet", ""),
                "url": item.get("url", ""),
                "source": "web_search",
            }
            documents.append(doc)

        logger.info(f"[Simple Web-RAG] Retrieved {len(documents)} web results")

        # Basic summarization using OpenAI
        summary = await self._basic_summarize(query, documents)

        # Format response (no verification metrics)
        return self._format_simple_response(
            query=query,
            summary=summary,
            sources=documents[:8],
            mode=ResearchMode.SIMPLE_WEB_RAG,
        )

    async def _basic_summarize(self, query: str, documents: List[Dict]) -> str:
        """Simple summarization without verification"""
        # Concatenate document content
        context = "\n\n".join(
            [
                f"Title: {doc.get('title', 'N/A')}\n"
                f"Content: {doc.get('abstract', 'N/A')[:500]}"
                for doc in documents[:8]
            ]
        )

        prompt = f"""Based on the following sources, provide a comprehensive answer to this question:

Question: {query}

Sources:
{context}

Provide a clear, factual answer synthesizing information from these sources.
Keep your response focused and well-organized."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant that synthesizes information from multiple sources.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content

    # ================================================================
    # MODE 2: MCP-Basic (MCP Tools, No Judge)
    # ================================================================

    async def _mcp_basic_mode(
        self, query: str, max_papers: int = 10, sources: List[str] = None
    ) -> Dict:
        """
        Use MCP tools for retrieval and summarization, but skip verification.
        """
        if sources is None:
            sources = ["arxiv", "pubmed"]

        context_id = f"ctx_{datetime.utcnow().timestamp()}"

        logger.info("[MCP-Basic] Retrieving papers using MCP tools")

        # Use MCP retriever
        papers = await self._retrieve_papers_mcp(query, max_papers, sources, context_id)

        if not papers:
            return self._no_results_response(query, ResearchMode.MCP_BASIC)

        logger.info(f"[MCP-Basic] Retrieved {len(papers)} papers")

        # Use MCP summarizer (no verification)
        summary = await self._summarize_mcp(query, papers, context_id)

        logger.info("[MCP-Basic] Summary generated (no verification)")

        return self._format_simple_response(
            query=query,
            summary=summary,
            sources=papers[:8],
            mode=ResearchMode.MCP_BASIC,
        )

    async def _retrieve_papers_mcp(
        self, query: str, max_papers: int, sources: List[str], context_id: str
    ) -> List[Dict]:
        """Retrieve papers using MCP retriever"""
        msg = MCPMessage(
            message_type=MessageType.QUERY,
            sender_agent="MultiModeAssistant",
            context_id=context_id,
            content={"query": query, "max_results": max_papers, "sources": sources},
        )
        res = await self.mcp_retriever.process(msg)
        return res.content.get("documents", []) if res else []

    async def _summarize_mcp(
        self, query: str, papers: List[Dict], context_id: str
    ) -> str:
        """Summarize using MCP summarizer agent"""
        msg = MCPMessage(
            message_type=MessageType.RETRIEVAL,
            sender_agent="MultiModeAssistant",
            context_id=context_id,
            content={"query": query, "documents": papers},
        )
        res = await self.summarizer.process(msg)
        return res.content.get("summary", "") if res else ""

    # ================================================================
    # MODE 3: MCP-Verified (Full Pipeline)
    # ================================================================

    async def _mcp_verified_mode(
        self, query: str, max_papers: int = 10, sources: List[str] = None
    ) -> Dict:
        """
        Full pipeline with MCP tools and verification loop.
        This is the original implementation.
        """
        if sources is None:
            sources = ["arxiv", "pubmed"]

        context_id = f"ctx_{datetime.utcnow().timestamp()}"

        logger.info("[MCP-Verified] Starting full verification pipeline")

        # Retrieve papers
        papers = await self._retrieve_papers_mcp(query, max_papers, sources, context_id)

        if not papers:
            return self._no_results_response(query, ResearchMode.MCP_VERIFIED)

        logger.info(f"[MCP-Verified] Retrieved {len(papers)} papers")

        # Generate and verify with feedback loop
        summary, verification = await self._generate_and_verify(
            query, papers, context_id
        )

        logger.info(
            f"[MCP-Verified] Verification complete. "
            f"Confidence: {verification.get('confidence', 0.0):.2%}"
        )

        # Format with verification details
        return self._format_verified_response(query, summary, papers, verification)

    async def _generate_and_verify(
        self, query: str, papers: List[Dict], context_id: str
    ) -> Tuple[str, Dict]:
        """
        Run verification loop with up to max_iterations.
        Returns best summary and verification details.
        """
        best_summary = ""
        best_verification = {"confidence": 0.0, "status": "none"}
        best_conf = -1.0

        current_summary = ""

        for iteration in range(self.max_iterations):
            # Generate summary if needed
            if not current_summary:
                current_summary = await self._summarize_mcp(query, papers, context_id)

            # Verify current summary
            verification = await self._verify_summary(
                query, current_summary, papers, context_id
            )
            conf = verification.get("confidence", 0.0)
            status = verification.get("status", "accept")

            logger.info(
                f"[Verification Loop {iteration}] status={status}, confidence={conf:.3f}"
            )

            # Track best
            if conf >= best_conf:
                best_conf = conf
                best_summary = current_summary
                best_verification = verification

            # Accept if confidence threshold met
            if status == "accept" and conf >= self.accept_confidence_threshold:
                logger.info(f"[Verification Loop] Accepted at iteration {iteration}")
                return current_summary, verification

            # Adjust status based on confidence
            if status == "accept" and conf < self.accept_confidence_threshold:
                status = "re_retrieve" if conf < 0.60 else "revise"

            # Take action
            if status == "re_retrieve":
                extra_docs = await self._enhanced_retrieval(
                    query, verification, context_id
                )
                if extra_docs:
                    papers.extend(extra_docs)
                current_summary = ""  # Force regeneration

            elif status == "revise":
                current_summary = await self._revise_summary(
                    current_summary, verification, papers
                )

            else:
                logger.warning(f"Unknown status '{status}', stopping loop")
                break

        logger.warning(
            f"Max iterations ({self.max_iterations}) reached. "
            f"Best confidence: {best_conf:.3f}"
        )

        return best_summary or "", best_verification

    async def _verify_summary(
        self, query: str, summary: str, papers: List[Dict], context_id: str
    ) -> Dict:
        """Verify summary using verifier agent"""
        msg = MCPMessage(
            message_type=MessageType.SUMMARY,
            sender_agent="MultiModeAssistant",
            context_id=context_id,
            content={"query": query, "summary": summary, "source_documents": papers},
        )
        res = await self.verifier.process(msg)
        return res.content if res else {}

    async def _enhanced_retrieval(
        self, original_query: str, verification: Dict, context_id: str
    ) -> List[Dict]:
        """Enhanced retrieval based on verification suggestions"""
        suggested = verification.get("suggested_queries", [])
        if not suggested:
            return []

        enhanced_query = f"{original_query} {' '.join(suggested[:2])}"
        return await self._retrieve_papers_mcp(
            enhanced_query, 5, ["arxiv", "pubmed"], context_id
        )

    async def _revise_summary(
        self, original_summary: str, verification: Dict, papers: List[Dict]
    ) -> str:
        """Revise summary based on verification feedback"""
        suggestions = verification.get("revision_suggestions", [])
        sug_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None."

        prompt = f"""Revise the following summary to address the listed issues.

Summary:
{original_summary}

Issues (from factuality verifier):
{sug_text}

Make the answer more accurate and faithful to the source documents.
Return ONLY the revised summary:
"""

        resp = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You revise scientific summaries for accuracy and faithfulness.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    # ================================================================
    # Response Formatting
    # ================================================================

    def _format_simple_response(
        self, query: str, summary: str, sources: List[Dict], mode: ResearchMode
    ) -> Dict:
        """Format response for Simple Web-RAG and MCP-Basic modes"""
        answer = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"📚 Research Summary ({mode.value.upper()}):\n{query}\n\n"
        answer += summary + "\n\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += "📄 Top Sources:\n\n"

        for i, source in enumerate(sources[:8], start=1):
            title = source.get("title", "Untitled")
            url = source.get("url", "N/A")
            answer += f"[{i}] {title}\n"
            answer += f"    {url}\n\n"

        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"Mode: {mode.value} | {len(sources)} sources\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        return {
            "answer": answer,
            "confidence": None,  # No verification in these modes
            "sources": sources,
            "query": query,
            "mode": mode.value,
        }

    def _format_verified_response(
        self, query: str, summary: str, papers: List[Dict], verification: Dict
    ) -> Dict:
        """Format response for MCP-Verified mode with verification details"""
        import re

        conf = verification.get("confidence", 0.0)
        halluc_ratio = verification.get("hallucination_ratio", 0.0)

        display_papers = papers[:8]

        # Inject inline citations
        summary_with_citations = self._inject_inline_citations(
            summary, verification, display_papers
        )

        answer = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"📚 Research Summary (MCP-VERIFIED):\n{query}\n\n"
        answer += summary_with_citations + "\n\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += "📄 Top Sources:\n\n"

        for i, p in enumerate(display_papers, start=1):
            title = p.get("title", "Untitled")
            url = p.get("url", "N/A")
            answer += f"[{i}] {title}\n"
            answer += f"    {url}\n\n"

        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"✓ Verifier Confidence: {conf:.0%} | {len(papers)} papers\n"
        answer += f"✓ Estimated Hallucination Ratio: {halluc_ratio*100:.1f}%\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        return {
            "answer": answer,
            "confidence": conf,
            "sources": papers[:10],
            "query": query,
            "mode": ResearchMode.MCP_VERIFIED.value,
            "verification_details": verification,
        }

    def _inject_inline_citations(
        self, summary: str, verification: Dict, papers: List[Dict]
    ) -> str:
        """Inject inline citations based on claim evidence"""
        import re

        if not summary:
            return summary

        claims = verification.get("claims") or []
        if not claims:
            return summary

        sentences = re.split(r"(?<=[.!?])\s+", summary)
        if not sentences:
            return summary

        updated_sentences = sentences.copy()

        # Build claim map
        claim_map = []
        for c in claims:
            claim_text = (c.get("text") or "").strip()
            if not claim_text:
                continue

            ev_idxs = c.get("evidence_indices") or []
            cit_nums = sorted(
                {
                    idx + 1
                    for idx in ev_idxs
                    if isinstance(idx, int) and 0 <= idx < len(papers)
                }
            )
            if not cit_nums:
                continue

            claim_map.append((claim_text, cit_nums))

        if not claim_map:
            return summary

        # Helper for keyword extraction
        def get_keywords(text: str) -> set:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            return {t for t in tokens if len(t) > 4}

        sentence_keywords = [get_keywords(s) for s in sentences]

        # Match claims to sentences
        for claim_text, cit_nums in claim_map:
            claim_keywords = get_keywords(claim_text)
            if not claim_keywords:
                continue

            best_idx = None
            best_score = 0

            for idx, sk in enumerate(sentence_keywords):
                overlap = len(claim_keywords & sk)
                if overlap > best_score:
                    best_score = overlap
                    best_idx = idx

            if best_idx is not None and best_score > 0:
                citation_str = "[" + ",".join(str(n) for n in cit_nums) + "]"
                if citation_str not in updated_sentences[best_idx]:
                    updated_sentences[best_idx] = (
                        updated_sentences[best_idx].rstrip() + " " + citation_str
                    )

        return " ".join(updated_sentences)

    def _no_results_response(self, query: str, mode: ResearchMode) -> Dict:
        """Response when no results found"""
        return {
            "answer": f"No relevant sources found for: {query}",
            "confidence": 0.0,
            "sources": [],
            "query": query,
            "mode": mode.value,
        }
