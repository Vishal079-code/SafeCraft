import os
import json
from typing import List, Dict, Set, Any
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index
from retriever.hybrid_retriever import HybridRetriever
from indexer.graph_builder import (
    build_dependency_graph,
    expand_with_dependencies,
    load_repo_map,
)

os.environ["HF_HUB_OFFLINE"] = "1"

BENCHMARK_TASKS = [
    {
        "query": "authenticate user with password hash and JWT token",
        "expected_files": ["sample_repo/auth/login.py", "sample_repo/models/user.py"]
    },
    {
        "query": "process credit card payment and refund transaction",
        "expected_files": ["sample_repo/utils/payments.py"]
    },
    {
        "query": "validate email format and strong password complexity",
        "expected_files": ["sample_repo/utils/validators.py"]
    },
    {
        "query": "user database schema and SQL model definition",
        "expected_files": ["sample_repo/models/user.py"]
    },
    {
        "query": "REST API route handlers for HTTP requests",
        "expected_files": ["sample_repo/routes/api.py", "sample_repo/main.py"]
    },
    {
        "query": "application initialization and middleware setup",
        "expected_files": ["sample_repo/main.py"]
    }
]


class RetrievalBenchmark:
    def __init__(self, repo_dir: str = "sample_repo", k: int = 3):
        self.repo_dir = repo_dir
        self.k = max(1, k)

        self.vector_store = VectorStore()

        repo_map_path = os.path.join(self.repo_dir, "repo_map.json")
        if os.path.exists(repo_map_path):
            self.repo_map = load_repo_map(repo_map_path)
        else:
            self.repo_map = {"modules": {}}

        self.vector_store.index_repo_map(self.repo_map)

        self.bm25_store = BM25Index()
        if self.repo_map.get("modules"):
            self.bm25_store.index_repo_map(self.repo_map)

        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25_store)
        self.graph = build_dependency_graph(self.repo_map)

    def _query_store(self, store_obj, query: str, top_k: int) -> List[Dict]:
        if not store_obj or not query:
            return []
        if hasattr(store_obj, "retrieve"):
            return store_obj.retrieve(query, top_k=top_k) or []
        elif hasattr(store_obj, "query"):
            return store_obj.query(query, top_k=top_k) or []
        elif hasattr(store_obj, "search"):
            return store_obj.search(query, top_k=top_k) or []
        return []

    def _normalize_path(self, path: str) -> str:
        """
        Normalizes a path to the 'sample_repo/...' prefixed format used by
        BENCHMARK_TASKS' expected_files, so retrieved paths (which come back
        unprefixed, e.g. 'auth/login.py') can be compared against ground truth.
        """
        if not path or not isinstance(path, str):
            return ""
        path = path.replace("\\", "/").strip("/")
        if not path.startswith("sample_repo/") and self.repo_dir in path:
            idx = path.find(self.repo_dir)
            path = path[idx:]
        elif not path.startswith("sample_repo/"):
            path = f"sample_repo/{path}"
        return path

    def _extract_raw_filepath(self, hit: Any) -> str:
        """
        Returns the RAW (unprefixed) filepath as stored in the dependency
        graph - e.g. 'auth/login.py', NOT 'sample_repo/auth/login.py'.

        This is used specifically for graph lookups. self.graph's nodes are
        the unprefixed relative_path keys that RepoIndexer originally
        produced (Day 3/4), so expand_with_dependencies() must be called
        with paths in that same unprefixed format or every lookup silently
        fails to match any node.

        This is intentionally kept separate from _extract_file_path() below,
        which normalizes (prefixes) paths for METRIC comparison against
        expected_files. The two must not be conflated - the graph and the
        ground-truth list use different path conventions.
        """
        if isinstance(hit, dict):
            meta = hit.get("metadata")
            if isinstance(meta, dict):
                fp = meta.get("filepath") or meta.get("file_path")
                if fp:
                    return fp
            unit_id = str(hit.get("id", ""))
            if "::" in unit_id:
                return unit_id.split("::")[0]
        elif isinstance(hit, str):
            return hit
        return ""

    def _extract_file_path(self, hit: Any) -> str:
        if isinstance(hit, str):
            raw = hit
        elif isinstance(hit, dict):
            if hit.get("file_path"):
                raw = hit["file_path"]
            elif isinstance(hit.get("metadata"), dict) and hit["metadata"].get("filepath"):
                raw = hit["metadata"]["filepath"]
            elif isinstance(hit.get("metadata"), dict) and hit["metadata"].get("file_path"):
                raw = hit["metadata"]["file_path"]
            else:
                unit_id = str(hit.get("id", ""))
                raw = unit_id.split("::")[0] if "::" in unit_id else unit_id
        else:
            raw = ""

        return self._normalize_path(raw)

    def _calculate_metrics(self, retrieved_files: List[str], expected_files: List[str]) -> Dict[str, float]:
        unique_retrieved = []
        for f in retrieved_files:
            norm_f = self._normalize_path(f)
            if norm_f and norm_f not in unique_retrieved:
                unique_retrieved.append(norm_f)

        norm_expected = set(self._normalize_path(f) for f in expected_files if f)
        if not norm_expected:
            return {"precision": 0.0, "recall": 0.0, "rr": 0.0}

        top_k_retrieved = set(unique_retrieved[:self.k])
        hits = top_k_retrieved.intersection(norm_expected)

        precision = len(hits) / self.k if self.k > 0 else 0.0
        recall = len(hits) / len(norm_expected) if len(norm_expected) > 0 else 0.0

        rr = 0.0
        for idx, file_path in enumerate(unique_retrieved, start=1):
            if file_path in norm_expected:
                rr = 1.0 / idx
                break

        return {"precision": precision, "recall": recall, "rr": rr}

    def run_benchmark(self):
        print("=" * 65)
        print(f"  SAFECRAFT RETRIEVAL BENCHMARK (K = {self.k})")
        print("=" * 65)

        modes = ["Vector Only", "BM25 Only", "Hybrid RRF", "Hybrid + Graph Expansion"]
        results = {mode: {"precision": [], "recall": [], "mrr": []} for mode in modes}

        for idx, task in enumerate(BENCHMARK_TASKS):
            query = task["query"]
            expected = task["expected_files"]

            vec_hits = self._query_store(self.vector_store, query, top_k=self.k)
            vec_files = [self._extract_file_path(h) for h in vec_hits]

            bm25_hits = self._query_store(self.bm25_store, query, top_k=self.k)
            bm25_files = [self._extract_file_path(h) for h in bm25_hits]

            hybrid_hits = self._query_store(self.hybrid_retriever, query, top_k=self.k)
            hybrid_files = [self._extract_file_path(h) for h in hybrid_hits]

            # FIX: graph expansion must use RAW (unprefixed) paths, since
            # self.graph's nodes are unprefixed (e.g. "auth/login.py"), not
            # the "sample_repo/..." prefixed format used for metric comparison.
            # Previously this passed the already-normalized `hybrid_files`,
            # which never matched any graph node - expansion silently did
            # nothing, making this column identical to "Hybrid RRF".
            raw_hybrid_paths = [self._extract_raw_filepath(h) for h in hybrid_hits]
            raw_hybrid_paths = [p for p in raw_hybrid_paths if p]

            expanded_deps = expand_with_dependencies(self.graph, raw_hybrid_paths, hops=1)
            raw_expanded_files = [self._normalize_path(f) for f in expanded_deps]

            combined_files = []
            for f in hybrid_files:
                if f and f not in combined_files:
                    combined_files.append(f)
            for f in raw_expanded_files:
                if f and f not in combined_files:
                    combined_files.append(f)

            retrievals = {
                "Vector Only": vec_files,
                "BM25 Only": bm25_files,
                "Hybrid RRF": hybrid_files,
                "Hybrid + Graph Expansion": combined_files
            }

            for mode, files in retrievals.items():
                m = self._calculate_metrics(files, expected)
                results[mode]["precision"].append(m["precision"])
                results[mode]["recall"].append(m["recall"])
                results[mode]["mrr"].append(m["rr"])

        print(f"{'Mode':<28} | {'Precision@K':<12} | {'Recall@K':<10} | {'MRR':<8}")
        print("-" * 65)

        for mode in modes:
            avg_p = sum(results[mode]["precision"]) / len(BENCHMARK_TASKS)
            avg_r = sum(results[mode]["recall"]) / len(BENCHMARK_TASKS)
            avg_mrr = sum(results[mode]["mrr"]) / len(BENCHMARK_TASKS)
            print(f"{mode:<28} | {avg_p:<12.3f} | {avg_r:<10.3f} | {avg_mrr:<8.3f}")

        print("=" * 65)


if __name__ == "__main__":
    benchmark = RetrievalBenchmark()
    benchmark.run_benchmark()