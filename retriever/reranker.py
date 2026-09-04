"""
Local Reranker: re-scores a candidate set against the ACTUAL user query
using a cross-encoder, which reads (query, candidate) pairs together and
scores relevance directly - a stronger relevance signal than the
independent embed-then-cosine-similarity approach vector search uses.

Runs entirely locally (CPU, no API calls) - same privacy/cost profile as
the rest of the retrieval stack.

Model note: 'cross-encoder/ms-marco-MiniLM-L-6-v2' is a general
text-relevance model (trained on search-query/passage pairs), not
code-specific - there isn't a widely-standardized code-specialized
cross-encoder equivalent to it yet. It's still a real improvement over
no reranking, but treat it as a solid V2 baseline, not a final answer -
swapping in a code-specific reranker later is a reasonable future step.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        candidates: results from HybridRetriever's RRF fusion step (each
        with at least a "text" key). Returns the top_k re-scored,
        re-sorted, with a "rerank_score" field added to each.
        """
        if not candidates:
            return []

        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]