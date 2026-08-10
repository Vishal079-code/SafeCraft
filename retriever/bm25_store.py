"""
Day 6a — BM25 Lexical Index for exact keyword matching.
Complements dense vector embeddings by capturing exact function names,
variable names, and specific identifier strings.
"""

import json
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from retriever.summary_builder import build_summary_units


def tokenize(text: str) -> List[str]:
    """
    Code-aware tokenizer that handles snake_case, camelCase, PascalCase,
    and punctuation-separated symbols. Preserves original full identifier names as well.
    """
    if not text:
        return []

    # 1. Expand camelCase / PascalCase (e.g., getUserById -> get User By Id)
    expanded = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)

    # 2. Extract alphanumeric sub-tokens
    sub_tokens = re.findall(r'[a-zA-Z0-9]+', expanded.lower())

    # 3. Preserve raw identifier tokens (e.g., hash_password)
    raw_tokens = re.findall(r'[a-zA-Z0-9_]+', text.lower())

    # Merge unique tokens preserving order
    all_tokens = list(dict.fromkeys(sub_tokens + raw_tokens))
    return all_tokens


class BM25Index:
    def __init__(self):
        self.units: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_repo_map(self, repo_map: Dict[str, Any]):
        """Build the BM25 index from repo_map summary units."""
        self.units = build_summary_units(repo_map)
        if not self.units:
            self.bm25 = None
            return

        # Tokenize unit texts along with their IDs and metadata for keyword coverage
        self.corpus_tokens = [
            tokenize(f"{u['id']} {u['text']}") for u in self.units
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the BM25 index and return the top_k relevant units."""
        if not self.bm25 or not self.units:
            return []

        query_tokens = tokenize(prompt)
        if not query_tokens:  # Guard against empty/punctuation-only queries
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with units and sort descending
        scored_units = list(zip(scores, self.units))
        scored_units.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, unit in scored_units[:top_k]:
            if score > 0:  # Only return units with non-zero keyword match score
                results.append({
                    "id": unit["id"],
                    "text": unit["text"],
                    "metadata": unit["metadata"],
                    "score": float(score),
                    "bm25_score": float(score),  # Standardized key for hybrid retriever
                })
        return results


if __name__ == "__main__":
    try:
        with open("sample_repo/repo_map.json") as f:
            repo_map = json.load(f)

        bm25_index = BM25Index()
        bm25_index.index_repo_map(repo_map)

        print("--- Testing BM25 Keyword Search ---")
        results = bm25_index.query("hash_password raw_password", top_k=3)
        for r in results:
            print(f"[{r['score']:.2f}] {r['id']} -> {r['text']}")
    except FileNotFoundError:
        print("Error: sample_repo/repo_map.json not found.")