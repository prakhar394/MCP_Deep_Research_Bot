# src/benchmark_accuracy.py

"""
Utility functions for evaluating generated answers against ground truth.

Metrics:
- token_f1          : classic F1 over word tokens
- semantic_cosine   : cosine similarity over simple bag-of-words vectors
- bert_score_f1     : BERTScore-F1 (optional, requires `pip install bert-score`)
- rouge_l_f1        : ROUGE-L F1 (LCS-based)
"""

import math
import re
from collections import Counter
from typing import List, Optional

# Optional BERTScore dependency
try:
    from bert_score import score as bert_score_fn  # type: ignore
except ImportError:
    bert_score_fn = None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip, remove extra spaces and basic punctuation."""
    text = text.lower().strip()
    # remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> List[str]:
    return _normalize_text(text).split() if text else []


# ---------------------------------------------------------------------------
# F1 (token-level)
# ---------------------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer.

    This ignores case, punctuation, and extra spaces.
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Simple cosine similarity over bag-of-words
# ---------------------------------------------------------------------------

def _bow_vector(text: str) -> Counter:
    return Counter(_tokenize(text))


def semantic_cosine(prediction: str, gold: str) -> float:
    """
    Very simple semantic metric: cosine similarity over bag-of-words.

    Not fancy embeddings – but:
    - symmetric
    - cheap
    - somewhat captures content overlap
    """
    v1 = _bow_vector(prediction)
    v2 = _bow_vector(gold)

    if not v1 and not v2:
        return 1.0
    if not v1 or not v2:
        return 0.0

    # dot product
    dot = 0.0
    for token, c1 in v1.items():
        dot += c1 * v2.get(token, 0.0)

    # norms
    norm1 = math.sqrt(sum(c * c for c in v1.values()))
    norm2 = math.sqrt(sum(c * c for c in v2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# ROUGE-L (Longest Common Subsequence based F1)
# ---------------------------------------------------------------------------

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Classic dynamic programming LCS length."""
    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len_a][len_b]


def rouge_l_f1(prediction: str, gold: str) -> float:
    """
    ROUGE-L F1: uses LCS length normalized as F1 over token sequences.
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BERTScore F1
# ---------------------------------------------------------------------------

def bert_score_f1(prediction: str, gold: str) -> Optional[float]:
    """
    Wrapper around bert_score.score(…, lang='en') to get F1.

    Returns:
        float in [0, 1] if bert-score is installed,
        otherwise None (so caller can skip / ignore).
    """
    if bert_score_fn is None:
        # bert-score not installed; gracefully degrade
        return None

    if not prediction.strip() and not gold.strip():
        return 1.0
    if not prediction.strip() or not gold.strip():
        return 0.0

    # bert_score.score expects lists of strings
    try:
        _, _, f1 = bert_score_fn(
            [prediction],
            [gold],
            lang="en",
            rescale_with_baseline=True,
        )
        return float(f1[0].item())
    except Exception:
        # If anything weird happens, just return None
        return None