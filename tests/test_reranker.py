
from retriever.reranker import Reranker
from retriever.hybrid_retriever import HybridRetriever
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index
import json

repo_map = json.load(open("big_repo/repo_map.json"))
v_store = VectorStore(db_path="./chroma_db_bigrepo", collection_name="big_repo_index")
b_index = BM25Index()
b_index.index_repo_map(repo_map)

print("Loading reranker model (first run downloads it)...")
reranker = Reranker()

retriever_no_rerank = HybridRetriever(vector_store=v_store, bm25_index=b_index)
retriever_reranked = HybridRetriever(vector_store=v_store, bm25_index=b_index, reranker=reranker)

query = "apply a discount to a product's price"
print("\n--- WITHOUT reranker ---")
for r in retriever_no_rerank.retrieve(query, top_k=5):
    print(f"  [RRF {r['rrf_score']:.4f}] {r['id']}")

print("\n--- WITH reranker ---")
for r in retriever_reranked.retrieve(query, top_k=5):
    print(f"  [rerank {r['rerank_score']:.4f}] {r['id']}")