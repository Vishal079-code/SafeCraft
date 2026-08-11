"""
Day 6b — Hybrid Retriever: Fuses dense vector search and sparse BM25 search
using Reciprocal Rank Fusion (RRF).
"""

import json
import logging
from typing import List, Dict, Any
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        """
        Executes both Vector and BM25 queries, combining results via Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (k_rrf + rank_vector) + 1 / (k_rrf + rank_bm25)
        """
        if not query or not query.strip():
            return []

        # Fetch extra candidates (top_k * 2) from both indexers to maximize fusion recall
        candidate_k = max(top_k * 2, top_k)
        
        # Safely invoke .query or .search on vector store
        if hasattr(self.vector_store, "query"):
            vector_results = self.vector_store.query(query, top_k=candidate_k)
        elif hasattr(self.vector_store, "search"):
            vector_results = self.vector_store.search(query, top_k=candidate_k)
        else:
            vector_results = []

        # Safely invoke .query or .search on BM25 store
        if hasattr(self.bm25_index, "query"):
            bm25_results = self.bm25_index.query(query, top_k=candidate_k)
        elif hasattr(self.bm25_index, "search"):
            bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        else:
            bm25_results = []

        rrf_scores: Dict[str, float] = {}
        unit_map: Dict[str, Dict[str, Any]] = {}

        # Process vector ranks
        for rank, res in enumerate(vector_results, start=1):
            uid = res["id"]
            unit_map[uid] = dict(res)  # Shallow copy to preserve isolated result metadata
            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (k_rrf + rank))

        # Process BM25 ranks
        for rank, res in enumerate(bm25_results, start=1):
            uid = res["id"]
            if uid not in unit_map:
                unit_map[uid] = dict(res)
            else:
                # Retain BM25 score metadata alongside vector score metadata
                if "bm25_score" in res:
                    unit_map[uid]["bm25_score"] = res["bm25_score"]

            rrf_scores[uid] = rrf_scores.get(uid, 0.0) + (1.0 / (k_rrf + rank))

        # Sort combined results by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        final_results = []
        for uid in sorted_ids[:top_k]:
            item = unit_map[uid]
            item["rrf_score"] = float(rrf_scores[uid])
            final_results.append(item)

        return final_results

    # Interface aliases for multi-retriever benchmark support
    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(prompt, top_k=top_k)

    def search(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.retrieve(prompt, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        with open("sample_repo/repo_map.json") as f:
            repo_map = json.load(f)

        v_store = VectorStore()
        v_store.index_repo_map(repo_map)

        b_index = BM25Index()
        b_index.index_repo_map(repo_map)

        retriever = HybridRetriever(vector_store=v_store, bm25_index=b_index)

        print("\n=== Hybrid Query Test: 'check user password' ===")
        hits = retriever.retrieve("check user password", top_k=3)
        for h in hits:
            print(f"  [RRF: {h['rrf_score']:.4f}] {h['id']}")
            print(f"    -> {h['text']}")
    except FileNotFoundError:
        print("Error: sample_repo/repo_map.json not found.")