import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional

import httpx
import arxiv
from transformers import pipeline
from pathlib import Path

from openai import OpenAI

from ..utils.logger import get_logger
from ..utils.cache import get_cache

logger = get_logger(__name__)


class MCPToolExecutor:
    def __init__(self, openai_api_key: str, tavily_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        self.cache = get_cache()

        logger.info("Loading NLI model for claim verification...")
        self.nli_model = pipeline(
            "text-classification",
            model="microsoft/deberta-large-mnli",
            device=-1,
        )

        logger.info("NLI model loaded")

        self.client = OpenAI(api_key=openai_api_key)

    async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Executing tool: {tool_name}")
            if tool_name == "arxiv_search":
                result = await self._arxiv_search(**tool_input)
            elif tool_name == "web_search":
                result = await self._web_search(**tool_input)
            elif tool_name == "fetch_paper":
                result = await self._fetch_paper(**tool_input)
            elif tool_name == "extract_claims":
                result = await self._extract_claims(**tool_input)
            elif tool_name == "verify_claim":
                result = await self._verify_claim(**tool_input)
            elif tool_name == "cache_result":
                result = await self._cache_result(**tool_input)
            else:
                return {"success": False, "result": None, "error": f"Unknown tool: {tool_name}"}

            return {"success": True, "result": result, "error": None}
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return {"success": False, "result": None, "error": str(e)}

    # ------------------ Tools ------------------ #

    async def _arxiv_search(
        self, query: str, max_results: int = 10, sort_by: str = "relevance"
    ) -> List[Dict[str, Any]]:
        cache_key = f"arxiv_{hashlib.md5(query.encode()).hexdigest()}_{max_results}"
        cached = self.cache.get("api", cache_key)
        if cached:
            logger.info("Cache hit for arXiv search")
            return cached

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
        )

        results: List[Dict[str, Any]] = []
        for paper in search.results():
            results.append(
                {
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published.isoformat(),
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                }
            )

        self.cache.set("api", cache_key, results)
        logger.info(f"Found {len(results)} papers on arXiv")
        return results

    async def _web_search(
        self,
        query: str,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.tavily_api_key:
            logger.warning("Tavily API key not provided, web_search returns empty list")
            return []

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": search_depth,
                        "include_domains": include_domains or [],
                        "max_results": max_results,
                    },
                    timeout=30.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", [])
                logger.error(f"Tavily error: {resp.status_code}")
                return []
            except Exception as e:
                logger.error(f"Web search error: {e}")
                return []

    async def _fetch_paper(self, url: str, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        logger.info(f"Fetching paper PDF: {url}")
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=30.0)
                pdf_content = resp.content
            except Exception as e:
                logger.error(f"Failed to download PDF: {e}")
                return {"error": str(e)}

        try:
            import fitz  # PyMuPDF

            tmp = Path("/tmp") / f"{hashlib.md5(url.encode()).hexdigest()}.pdf"
            tmp.write_bytes(pdf_content)

            doc = fitz.open(tmp)
            text_chunks = []
            for page in doc:
                text_chunks.append(page.get_text())
            doc.close()
            full_text = "\n".join(text_chunks)

            if sections:
                # naive: just return first 5000 chars tagged
                return {
                    "url": url,
                    "sections": {s: full_text[:1000] for s in sections},
                    "full_text_length": len(full_text),
                }

            return {"url": url, "text": full_text[:5000], "full_text_length": len(full_text)}
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            return {"error": str(e)}

    async def _extract_claims(self, text: str, claim_type: str = "all") -> List[Dict[str, Any]]:
        logger.info("Extracting claims via OpenAI")
        resp = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Extract atomic, verifiable claims from the text.

Focus on: {claim_type}

Return JSON with key "claims": [
  {{
    "claim": "...",
    "type": "factual|numerical|comparative",
    "entities": ["..."],
    "verifiability": "easy|medium|hard"
  }}
]""",
                },
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        claims = parsed.get("claims", [])
        logger.info(f"Extracted {len(claims)} claims")
        return claims

    async def _verify_claim(
        self, claim: str, evidence: List[str], use_external_search: bool = True
    ) -> Dict[str, Any]:
        logger.info(f"Verifying claim: {claim[:80]}")

        nli_scores: List[float] = []
        for ev in evidence[:5]:
            pair = f"{ev[:512]} [SEP] {claim[:256]}"
            result = self.nli_model(pair)[0]
            label = result["label"]
            score = result["score"]
            if label == "ENTAILMENT":
                nli_scores.append(score)
            elif label == "CONTRADICTION":
                nli_scores.append(-score)
            else:
                nli_scores.append(0.0)

        avg_nli = sum(nli_scores) / len(nli_scores) if nli_scores else 0.0

        external_evidence: List[str] = []
        external_scores: List[float] = []
        if use_external_search:
            web_results = await self._web_search(
                claim, include_domains=["arxiv.org", "scholar.google.com"], max_results=3
            )
            external_evidence = [r.get("content", "") for r in web_results]
            for ev in external_evidence:
                pair = f"{ev[:512]} [SEP] {claim[:256]}"
                r = self.nli_model(pair)[0]
                label = r["label"]
                score = r["score"]
                if label == "ENTAILMENT":
                    external_scores.append(score)
                elif label == "CONTRADICTION":
                    external_scores.append(-score)
                else:
                    external_scores.append(0.0)

        avg_ext = sum(external_scores) / len(external_scores) if external_scores else 0.0

        if external_scores:
            final_score = 0.7 * avg_nli + 0.3 * avg_ext
        else:
            final_score = avg_nli

        if final_score > 0.6:
            verdict = "SUPPORTED"
        elif final_score < -0.6:
            verdict = "REFUTED"
        else:
            verdict = "NOT_ENOUGH_INFO"

        confidence = abs(final_score)
        logger.info(f"Claim verdict: {verdict} (confidence={confidence:.2f})")

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": float(confidence),
            "nli_score": float(avg_nli),
            "external_score": float(avg_ext),
            "evidence_count": len(evidence),
            "external_evidence_count": len(external_evidence),
            "external_evidence": external_evidence if use_external_search else [],
        }

    async def _cache_result(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> Dict[str, Any]:
        self.cache.set("results", key, value, ttl=ttl)
        return {"cached": True, "key": key}
