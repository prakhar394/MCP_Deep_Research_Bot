# src/agents/thorough_mcp_verifier.py

"""
ThoroughMCPVerifier (LLM-based Evidence & Claim-Level Verifier)

This verifier:
- Decomposes the answer into atomic claims.
- Checks each claim against the provided documents (titles/abstracts/snippets).
- Detects hallucinations or contradictions.
- Computes:
    - per-claim support labels,
    - overall confidence,
    - hallucination ratio,
    - a next-step decision:
        * "accept"
        * "revise"
        * "re_retrieve"
- Returns structured feedback for iterative improvement.
"""

from typing import Dict, Any, List, Optional
import json

from openai import OpenAI

from .base_agent import BaseAgent
from ..utils.mcp_schema import MCPMessage, MessageType
from ..utils.logger import get_logger

logger = get_logger(__name__)


VERIFIER_SYSTEM_PROMPT = """
You are an expert verification model acting as a factuality and faithfulness judge for a research assistant.

You are given:
- a user query,
- a proposed answer (summary),
- a set of source documents (e.g., paper titles, abstracts, snippets).

YOUR JOB (HIGH LEVEL):
1. Decompose the answer into a set of short, atomic claims.
2. For each claim, judge how well it is supported by the documents ONLY.
3. Detect hallucinations:
   - unsupported claims
   - contradicted claims
4. Decide what the pipeline should do next:
   - "accept" → answer is very well supported and needs no substantive changes.
   - "revise" → answer is mostly grounded, but needs corrections, caveats, or clarification.
   - "re_retrieve" → evidence is missing, weak, or clearly insufficient to answer the query.

IMPORTANT RULES:
- You MUST rely ONLY on the provided documents.
- Ignore your own parametric knowledge if it conflicts with the documents.
- If the answer says something that is not clearly supported, treat it as a hallucination.
- Be strict: do not give high confidence unless key claims are clearly grounded in the sources.
- If there are NO documents or they are very weak, prefer "re_retrieve" with low confidence.

REASONING PROCEDURE (DO THIS INTERNALLY, BUT DO NOT OUTPUT IT DIRECTLY):
1. GLOBAL UNDERSTANDING
   - Read the user query carefully.
   - Read the proposed answer from start to end.
   - Skim the documents to understand the main topics and evidence.

2. CLAIM EXTRACTION
   - Go through the answer paragraph by paragraph.
   - List short, atomic factual claims that the answer is making.
   - Each claim should be a standalone sentence that could be true or false.

3. EVIDENCE CHECKING PER CLAIM
   For EACH claim:
   - Search through the provided documents for sentences that support or contradict it.
   - If you find clear supporting evidence: note the document indices.
   - If you find clear contradicting evidence: note those indices as well.
   - If evidence is mixed or partial, treat this carefully as "partially_supported".
   - If you find no clear evidence either way, treat it as "unsupported".

4. LABEL AND NOTE PER CLAIM
   For EACH claim:
   - Decide on "support" in {"supported", "partially_supported", "unsupported", "contradicted"}.
   - Be strict and pessimistic: if you are unsure, choose "unsupported".
   - Write a short 1–2 sentence explanation note for why you chose that label.

5. SUMMARY-LEVEL JUDGMENT
   - Mentally compute the proportion of claims that are unsupported or contradicted.
   - Reflect on whether the main conclusions of the answer are reliable given the evidence.
   - Decide on:
       * overall "confidence" in [0.0, 1.0]
       * "status" in {"accept", "revise", "re_retrieve"}
   - If many central claims are unsupported/contradicted → prefer "re_retrieve".
   - If only some wording or caveats are missing → prefer "revise".
   - Only if hallucinations are very low and evidence is strong should you "accept".

6. OUTPUT
   - After completing all reasoning steps above, output ONLY the final JSON object.
   - Do NOT output your intermediate reasoning or thought process.

CLAIM-LEVEL ANALYSIS:
For each claim in the answer:
- Extract a SHORT, standalone sentence.
- Assign a "support" label, one of:
  - "supported"            → clearly backed by the documents.
  - "partially_supported"  → some aspects are supported, others are not.
  - "unsupported"          → not backed by the documents.
  - "contradicted"         → actively contradicted by the documents.
- Optionally cite which document indices (0-based) provide evidence.
- Add a brief note explaining your judgment.

HARD CONSTRAINTS ON "support" FIELD:
- For EVERY claim you output, you MUST include a "support" field.
- "support" MUST be EXACTLY ONE of:
  - "supported"
  - "partially_supported"
  - "unsupported"
  - "contradicted"
- Use ONLY lowercase, no spaces, no extra words.
- If the documents do NOT clearly back the claim, you MUST choose "unsupported" or "contradicted".
- If you are unsure whether the claim is truly supported, treat it as "unsupported".
- Do NOT label everything as "supported": be strict and pessimistic.

CONFIDENCE & STATUS POLICY (HIGH LEVEL):
- Think of "confidence" as your calibrated belief that the answer is overall factually correct and well supported.
- Define hallucination_ratio as:
    (# of claims with support in {"unsupported", "contradicted"}) / (total number of claims)

Use the following mapping between confidence, hallucination_ratio, and status:

- If hallucination_ratio <= 0.10 and most key claims are supported:
    * confidence can be high (e.g., 0.90–1.0).
    * If confidence >= 0.97 → status MUST be "accept".

- If 0.10 < hallucination_ratio <= 0.35 OR there are important partially_supported claims:
    * confidence should be moderate (e.g., 0.60–0.90).
    * status SHOULD usually be "revise".
    * Provide precise "revision_suggestions" to fix or qualify problematic claims.

- If hallucination_ratio > 0.35 OR many key claims are unsupported/contradicted OR documents seem weak/irrelevant:
    * confidence should be low (e.g., 0.0–0.60).
    * status SHOULD be "re_retrieve".
    * Provide targeted "suggested_queries" that will help retrieve better evidence.

OUTPUT FORMAT (STRICT JSON):
You MUST respond with a single valid JSON object, with NO extra commentary, in this exact schema:

{
  "confidence": float between 0.0 and 1.0,
  "status": "accept" | "revise" | "re_retrieve",
  "issues": [
    "short description of a factual or coverage issue, if any"
  ],
  "revision_suggestions": [
    "actionable instructions for how to fix or improve the answer"
  ],
  "suggested_queries": [
    "refined retrieval queries to get better evidence, if needed"
  ],
  "claims": [
    {
      "text": "short atomic claim",
      "support": "supported" | "partially_supported" | "unsupported" | "contradicted",
      "evidence_indices": [0, 2],
      "note": "brief explanation"
    }
  ]
}

Additional guidelines:
- "issues" should briefly describe mismatches, gaps, or hallucinations at the answer level.
- "revision_suggestions" should be concrete edits, caveats, or restructuring advice.
- "suggested_queries" should be short, targeted phrases for a retriever.
- If you are unsure, lower the confidence and lean toward "revise" or "re_retrieve".
"""


class ThoroughMCPVerifier(BaseAgent):
    """
    LLM-based verifier that fits into the existing MCPResearchAssistant flow.

    It expects an MCPMessage with:
        message_type = SUMMARY
        content = {
            "query": str,
            "summary": str,
            "source_documents": List[Dict]
        }

    It returns an MCPMessage with:
        message_type = VERIFICATION
        content = {
            "status": "accept" | "revise" | "re_retrieve",
            "confidence": float,
            "issues": List[str],
            "revision_suggestions": List[str],
            "suggested_queries": List[str],
            "claims": List[Dict],
            "hallucination_ratio": float,
        }
    """

    VALID_SUPPORT_LABELS = {
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    }

    # tavily_api_key is kept ONLY for backward compatibility with signatures like
    # ThoroughMCPVerifier(openai_api_key, tavily_api_key).
    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
    ):
        super().__init__("ThoroughMCPVerifier")
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def process(self, message: MCPMessage) -> MCPMessage:
        query: str = message.content.get("query", "")
        summary: str = message.content.get("summary", "")
        docs: List[Dict[str, Any]] = message.content.get("source_documents", [])

        logger.info(
            f"[Verifier] Verifying summary for query='{query[:80]}...' "
            f"with {len(docs)} source docs"
        )

        if not summary:
            # If there's literally nothing to verify, force re-retrieval.
            logger.warning("[Verifier] Empty summary; forcing re_retrieve.")
            return await self._build_verification_message(
                original_message=message,
                confidence=0.0,
                status="re_retrieve",
                issues=["No summary was provided to verify."],
                revision_suggestions=[
                    "Generate an initial answer based on the available documents."
                ],
                suggested_queries=[],
                claims=[],
                hallucination_ratio=1.0,
            )

        if not docs:
            logger.warning("[Verifier] No documents provided; recommending re_retrieve.")
            return await self._build_verification_message(
                original_message=message,
                confidence=0.0,
                status="re_retrieve",
                issues=["No documents were provided to support the answer."],
                revision_suggestions=[],
                suggested_queries=[
                    "refine the query and retrieve relevant papers for this topic"
                ],
                claims=[],
                hallucination_ratio=1.0,
            )

        # Build the JSON "case" for the verifier LLM
        verifier_input = {
            "query": query,
            "answer": summary,
            "documents": docs,
        }

        user_prompt = json.dumps(verifier_input, ensure_ascii=False, indent=2)

        # Call the LLM as judge
        raw_output = self._call_verifier_model(user_prompt)

        parsed = self._parse_verifier_output(raw_output)

        confidence = float(parsed.get("confidence", 0.0))
        status = parsed.get("status", "accept")
        issues = parsed.get("issues") or []
        revision_suggestions = parsed.get("revision_suggestions") or []
        suggested_queries = parsed.get("suggested_queries") or []
        claims = parsed.get("claims") or []

        # Compute hallucination ratio from claims (strict)
        hallucination_ratio = self._compute_hallucination_ratio(claims)

        # Adjust confidence conservatively based on hallucinations
        if hallucination_ratio > 0.5:
            confidence *= 0.3
        elif hallucination_ratio > 0.25:
            confidence *= 0.6

        # Clamp confidence
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0

        # Enforce a conservative mapping if the model is too optimistic
        status = self._enforce_status_policy(status, confidence, hallucination_ratio)

        # DEBUG: see what labels we are actually getting
        try:
            labels = {
                self._normalize_support_label(c.get("support"))
                for c in claims
            }
            logger.info(f"[Verifier] Claim support labels seen: {labels}")
        except Exception as e:
            logger.warning(f"[Verifier] Failed to inspect claim labels: {e}")

        logger.info(
            f"[Verifier] status={status}, confidence={confidence:.2f}, "
            f"hallucination_ratio={hallucination_ratio:.2f}, "
            f"issues={len(issues)}, suggested_queries={len(suggested_queries)}, "
            f"claims={len(claims)}"
        )

        return await self._build_verification_message(
            original_message=message,
            confidence=confidence,
            status=status,
            issues=issues,
            revision_suggestions=revision_suggestions,
            suggested_queries=suggested_queries,
            claims=claims,
            hallucination_ratio=hallucination_ratio,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _call_verifier_model(self, user_prompt: str) -> str:
        """
        Synchronous OpenAI call (like in SummarizerAgent).
        This is called from an async context but uses the blocking client
        for simplicity, matching the rest of the codebase.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    def _parse_verifier_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Try hard to extract a JSON object from the model output.
        If parsing fails, fall back to a low-confidence 're_retrieve'.
        """
        if not raw_output:
            logger.error("[Verifier] Empty LLM output; falling back to default.")
            return {
                "confidence": 0.0,
                "status": "re_retrieve",
                "issues": ["Verifier returned no output."],
                "revision_suggestions": [],
                "suggested_queries": [],
                "claims": [],
            }

        text = raw_output.strip()

        def try_parse(s: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(s)
            except Exception:
                return None

        parsed = try_parse(text)
        if parsed is not None:
            return parsed

        # Fallback: extract first JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            parsed = try_parse(candidate)
            if parsed is not None:
                return parsed

        logger.error(
            "[Verifier] Failed to parse verifier JSON. Raw output was:\n%s", text
        )
        return {
            "confidence": 0.0,
            "status": "re_retrieve",
            "issues": ["Could not parse verifier JSON output."],
            "revision_suggestions": [],
            "suggested_queries": [],
            "claims": [],
        }

    def _normalize_support_label(self, raw: Any) -> str:
        """
        Normalize the 'support' field from a claim.
        - Lowercase
        - Fix common formatting variants
        - Return "" for unknown/invalid labels
        """
        if raw is None:
            return ""
        text = str(raw).strip().lower()

        # Fix common LLM variants
        if text == "partially supported":
            return "partially_supported"

        if text in self.VALID_SUPPORT_LABELS:
            return text

        # Anything else is treated as invalid
        return ""

    def _compute_hallucination_ratio(self, claims: List[Dict[str, Any]]) -> float:
        """
        Compute hallucination ratio as:
            (# unsupported or contradicted claims, or claims with invalid/missing support)
            / (total claims)

        We treat missing/unknown support as risky (hallucinated) rather than safe.
        """
        if not claims:
            # No claims = verifier basically failed to analyze → high risk
            return 1.0

        bad_labels = {"unsupported", "contradicted"}
        total = len(claims)
        bad = 0

        for c in claims:
            label = self._normalize_support_label(c.get("support"))

            # Missing/invalid label counts as bad
            if label == "" or label in bad_labels:
                bad += 1

        return bad / total if total > 0 else 1.0

    def _enforce_status_policy(
        self, status: str, confidence: float, hallucination_ratio: float
    ) -> str:
        """
        Safety net to ensure status is consistent with confidence and hallucination_ratio.
        """
        # Strong hallucinations → re_retrieve
        if hallucination_ratio > 0.4:
            return "re_retrieve"

        # Moderate hallucinations → revise
        if 0.15 < hallucination_ratio <= 0.4 and status == "accept":
            return "revise"

        # Confidence-based enforcement
        if confidence >= 0.97 and hallucination_ratio <= 0.10:
            return "accept"
        if confidence < 0.60:
            return "re_retrieve"
        if 0.60 <= confidence < 0.97 and status == "accept":
            return "revise"

        return status

    async def _build_verification_message(
        self,
        original_message: MCPMessage,
        confidence: float,
        status: str,
        issues: List[str],
        revision_suggestions: List[str],
        suggested_queries: List[str],
        claims: List[Dict[str, Any]],
        hallucination_ratio: float,
    ) -> MCPMessage:
        """
        Wraps the verifier result into an MCPMessage so that
        MCPResearchAssistant._verify_summary can consume it.
        """
        content = {
            "status": status,
            "confidence": confidence,
            "issues": issues,
            "revision_suggestions": revision_suggestions,
            "suggested_queries": suggested_queries,
            "claims": claims,
            "hallucination_ratio": hallucination_ratio,
        }

        return await self.send_message(
            message_type=MessageType.VERIFICATION,
            context_id=original_message.context_id,
            content=content,
            confidence_score=confidence,
            parent_message_id=original_message.message_id,
        )