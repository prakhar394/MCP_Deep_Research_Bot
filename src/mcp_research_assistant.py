# src/mcp_research_assistant.py

from typing import Dict, List, Tuple
from datetime import datetime
import re

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

        # Feedback loop settings
        self.max_iterations = 5
        self.accept_confidence_threshold = 0.97

        logger.info("MCPResearchAssistant initialized")

    async def answer_query(
        self, query: str, max_papers: int = 10, sources: List[str] = None
    ) -> Dict[str, any]:
        # Default: use both arXiv and PubMed
        if sources is None:
            sources = ["arxiv", "pubmed"]

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
    ) -> Tuple[str, Dict]:
        """
        Run up to self.max_iterations verification loops.
        - If status == "accept" and confidence >= threshold → return immediately.
        - Otherwise, keep track of the best (highest-confidence) result.
        - If we hit max_iterations without meeting the threshold, return best-so-far.
        """

        best_summary: str = ""
        best_verification: Dict = {"confidence": 0.0, "status": "none"}
        best_conf: float = -1.0  # ensure first iteration always becomes best

        current_summary: str = ""  # will be generated or revised

        for iteration in range(self.max_iterations):
            # STEP 1 — Generate summary if we don't have one yet
            if not current_summary:
                current_summary, _ = await self._generate_summary(
                    query, papers, context_id
                )

            # STEP 2 — Verify current summary
            verification = await self._verify_summary(
                query, current_summary, papers, context_id
            )
            conf = verification.get("confidence", 0.0)
            status = verification.get("status", "accept")

            logger.info(
                f"[Loop {iteration}] Verification → status={status}, confidence={conf:.3f}"
            )

            # Track best-so-far
            if conf >= best_conf:
                best_conf = conf
                best_summary = current_summary
                best_verification = verification

            # STEP 3 — If verifier says accept AND confidence above threshold → done
            if status == "accept" and conf >= self.accept_confidence_threshold:
                return current_summary, verification

            # STEP 4 — Interpret low/mid confidence as actions if verifier didn't
            if status == "accept" and conf < self.accept_confidence_threshold:
                if conf < 0.60:
                    status = "re_retrieve"
                else:
                    status = "revise"

            # STEP 5 — Take action based on (possibly adjusted) status
            if status == "re_retrieve":
                extra_docs = await self._enhanced_retrieval(
                    query, verification, context_id
                )
                if extra_docs:
                    papers.extend(extra_docs)
                # force regeneration with new evidence on next loop
                current_summary = ""

            elif status == "revise":
                current_summary = await self._revise_summary(
                    current_summary, verification, papers
                )
                # next loop will verify this revised summary

            else:
                # Unknown status: just break and return best-so-far
                logger.warning(
                    f"Unknown verifier status '{status}', returning best-so-far."
                )
                break

        # If we’re here, we hit max_iterations without meeting the threshold
        logger.warning(
            f"Max verification iterations ({self.max_iterations}) reached. "
            f"Returning best-so-far with confidence={best_conf:.3f}"
        )

        if not best_summary:
            return "", {"confidence": 0.3, "status": "max_iterations"}

        return best_summary, best_verification

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
        """
        Use verifier.suggested_queries to refine retrieval.
        """
        suggested = verification.get("suggested_queries", [])
        if not suggested:
            return []
        enhanced_query = f"{original_query} {' '.join(suggested[:2])}"
        # On re-retrieval we also use both arxiv+pubmed
        return await self._retrieve_papers(
            enhanced_query, 5, ["arxiv", "pubmed"], context_id
        )

    async def _revise_summary(
        self, original_summary: str, verification: Dict, papers: List[Dict]
    ) -> str:
        """
        Ask the LLM to revise the summary using the verifier's revision_suggestions.
        """
        from openai import OpenAI

        client = OpenAI()
        suggestions = verification.get("revision_suggestions", [])
        sug_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None."

        prompt = f"""Revise the following summary to address the listed issues.

Summary:
{original_summary}

Issues (from a factuality/faithfulness verifier):
{sug_text}

Make the answer more accurate and faithful to the kind of source documents described (scientific papers),
without introducing new claims that are not supported.

Return ONLY the revised summary:
"""

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You revise scientific summaries to be accurate, faithful, and well-calibrated.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    # ------------------------------------------------------------------ #
    # Inline citation injection (fuzzy matching)
    # ------------------------------------------------------------------ #

    def _inject_inline_citations(
        self,
        summary: str,
        verification: Dict,
        papers: List[Dict],
    ) -> str:
        """
        Fuzzy citation injection:
        - Break summary into sentences
        - For each claim, match it to the most similar sentence using keyword overlap
        - Append [1], [1,3] (based on evidence indices) to that sentence

        'papers' must be exactly the list of documents you will display
        in the Top Sources section, so citation numbers stay consistent.
        """
        if not summary:
            return summary

        claims = verification.get("claims") or []
        if not claims:
            return summary

        # Split summary into sentences (very simple heuristic)
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        if not sentences:
            return summary

        updated_sentences = sentences.copy()

        # Build list of (claim_text, [citation_numbers])
        claim_map = []
        for c in claims:
            claim_text = (c.get("text") or "").strip()
            if not claim_text:
                continue

            ev_idxs = c.get("evidence_indices") or []
            # Map 0-based doc indices to 1-based citation numbers,
            # but only for docs that exist in this 'papers' subset.
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

        # Helper: keyword extraction
        def get_keywords(text: str) -> set:
            tokens = re.findall(r"[a-zA-Z]+", text.lower())
            # Keep words with length > 4 to focus on content words
            return {t for t in tokens if len(t) > 4}

        # Precompute sentence keywords
        sentence_keywords = [get_keywords(s) for s in sentences]

        # Process each claim and attach citations to the best-matching sentence
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

            # Only attach citation if we have some overlap
            if best_idx is not None and best_score > 0:
                citation_str = "[" + ",".join(str(n) for n in cit_nums) + "]"
                # Avoid double-adding citation if it's already there
                if citation_str not in updated_sentences[best_idx]:
                    updated_sentences[best_idx] = (
                        updated_sentences[best_idx].rstrip() + " " + citation_str
                    )

        # Rejoin sentences
        return " ".join(updated_sentences)

    # ------------------------------------------------------------------ #
    # Formatting / final response
    # ------------------------------------------------------------------ #

    def _format_response(
        self, query: str, summary: str, papers: List[Dict], verification: Dict
    ) -> Dict:
        conf = verification.get("confidence", 0.0)
        halluc_ratio = verification.get("hallucination_ratio", 0.0)

        # Only display (and therefore cite) the top N papers
        display_papers = papers[:8]

        # Inject inline citations based on claim evidence,
        # using ONLY the display_papers to keep numbers consistent
        summary_with_citations = self._inject_inline_citations(
            summary, verification, display_papers
        )

        answer = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        answer += f"📚 Research Summary for:\n{query}\n\n"
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
        answer += f"✓ Estimated Hallucination Ratio: {halluc_ratio*100:.1f}% of claims\n"
        answer += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        return {
            "answer": answer,
            "confidence": conf,
            "sources": papers[:10],  # still return more raw sources if needed
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