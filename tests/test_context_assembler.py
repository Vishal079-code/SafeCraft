"""
Test ContextAssembler integration with HybridRetriever output
"""

import os
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index
from retriever.hybrid_retriever import HybridRetriever
from retriever.context_assembler import ContextAssembler
from indexer.graph_builder import load_repo_map


def test_pipeline():
    repo_map_path = "sample_repo/repo_map.json"
    if not os.path.exists(repo_map_path):
        print(f"Warning: '{repo_map_path}' not found. Test running with fallback mock data.")
        repo_map = {"modules": {}}
    else:
        repo_map = load_repo_map(repo_map_path)

    vector_store = VectorStore()
    vector_store.index_repo_map(repo_map)

    bm25_store = BM25Index()
    bm25_store.index_repo_map(repo_map)

    hybrid_retriever = HybridRetriever(vector_store, bm25_store)

    query = "authenticate user with password hash and JWT token"
    hits = hybrid_retriever.retrieve(query, top_k=3)

    assembler = ContextAssembler(max_tokens=1500)
    prompt_context = assembler.assemble_context(hits, query=query)

    print("=" * 60)
    print("  GENERATED PROMPT CONTEXT BLOCK")
    print("=" * 60)
    print(prompt_context)


if __name__ == "__main__":
    test_pipeline()