"""
Day 6b — Hybrid Retriever: Fuses dense vector search and sparse BM25 search
using Reciprocal Rank Fusion (RRF), with an OPTIONAL local reranking
stage on top.

V2 change: retrieve() now pulls a larger RRF candidate POOL than top_k
(so there's real material for the reranker to work with), and if a
reranker is configured, re-scores that pool against the actual query
before truncating to top_k. If no reranker is configured (the default),
behavior is IDENTICAL to before - a larger internal RRF pool doesn't
change the final top_k result, since it's still just the top-ranked
RRF items truncated to top_k.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: Optional[Any] = None,
        rerank_pool_size: int = 15,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.rerank_pool_size = rerank_pool_size

    def retrieve(self, query: str, top_k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        # Pull a larger pool than top_k when a reranker is configured, so
        # there's real material to rerank - otherwise, same sizing as
        # before (top_k * 2), preserving exact prior behavior.
        candidate_k = max(self.rerank_pool_size, top_k * 2) if self.reranker else max(top_k * 2, top_k)

        if hasattr(self.vector_store, "query"):
            vector_results = self.vector_store.query(query, top_k=candidate_k)
        elif hasattr(self.vector_store, "search"):
            vector_results = self.vector_store.search(query, top_k=candidate_k)
        else:
            vector_results = []

        if hasattr(self.bm25_index, "query"):
            bm25_results = self.bm25_index.query(query, top_k=candidate_k)
        elif hasattr(self.bm25_index, "search"):
            bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        else:
            bm25_results = []

        rrf_scores: Dict[str, float] = {}
        unit_map: Dict[str, Dict[str, Any]] = {}

        for rank, res in enumerate(vector_results, start=1):
            uid = res["id"]
            unit_map[uid] = dict(res)
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (k_rrf + rank))

        for rank, res in enumerate(bm25_results, start=1):
            uid = res["id"]
            if uid not in unit_map:
                unit_map[uid] = dict(res)
            else:
                if "bm25_score" in res:
                    unit_map[uid]["bm25_score"] = res["bm25_score"]
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (k_rrf + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        pool_size = candidate_k if self.reranker else top_k
        pool = []
        for uid in sorted_ids[:pool_size]:
            item = unit_map[uid]
            item["rrf_score"] = float(rrf_scores[uid])
            pool.append(item)

        if self.reranker is not None and pool:
            try:
                return self.reranker.rerank(query, pool, top_k=top_k)
            except Exception as e:
                logger.warning(f"Reranking failed, falling back to RRF-only results: {e}")

        return pool[:top_k]

    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(prompt, top_k=top_k)

    def search(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(prompt, top_k=top_k)
    def retrieve_metadata_only(self, query: str, top_k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        full_results = self.retrieve(query, top_k=top_k, k_rrf=k_rrf)
        stripped = []
        for r in full_results:
            meta = r.get("metadata", {})
            stripped.append({
                "id": r.get("id"),
                "metadata": {
                    "filepath": meta.get("filepath"),
                    "name": meta.get("name"),
                    "type": meta.get("type"),  # "function"/"class"/"method" - structural, not code
                },
                "score": r.get("rerank_score", r.get("rrf_score", 0.0)),
            })
        return stripped