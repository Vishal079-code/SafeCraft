from indexer.repo_indexer import build_repo_map
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index
from retriever.hybrid_retriever import HybridRetriever
from agent.orchestrator import AgentOrchestrator
from eval.inspect_pipeline import load_repo_as_workspace

print("=" * 60)
print("STEP 2: Index big_repo and test retrieval")
print("=" * 60)

repo_map = build_repo_map("big_repo", "big_repo/repo_map.json")
print(f"Indexed {repo_map['files_parsed']} files from big_repo")

v_store = VectorStore(db_path="./chroma_db_bigrepo", collection_name="big_repo_index")
v_store.index_repo_map(repo_map)

b_index = BM25Index()
b_index.index_repo_map(repo_map)

retriever = HybridRetriever(vector_store=v_store, bm25_index=b_index)

print("\n--- Query: 'apply a discount to a product's price' ---")
results = retriever.retrieve("apply a discount to a product's price", top_k=5)
for r in results:
    print(f"  [RRF: {r['rrf_score']:.4f}] {r['id']}")
    print(f"    -> {r['text']}")

print("\n" + "=" * 60)
print("STEP 3: Integration test with AgentOrchestrator")
print("=" * 60)

workspace = load_repo_as_workspace("big_repo")

def mock_llm(prompt):
    return 'File: models/product.py\n<<<<<<< SEARCH\n...\n=======\n...\n>>>>>>> REPLACE'

orchestrator = AgentOrchestrator(
    protected_identifiers=["apply_discount"],
    retriever=retriever,
    retrieval_top_k=3,
)
result = orchestrator.run_cycle(
    user_request="apply a discount to a product's price",
    target_filepath="models/product.py",
    workspace_files=workspace,
    llm_caller_fn=mock_llm,
)

print("bundled_files:", result.bundled_files)
print("retrieval_added_files:", result.retrieval_added_files)

print("\n--- Second query, testing a NON-import-connected file: 'format currency for display' ---")
result2 = orchestrator.run_cycle(
    user_request="format currency for display",
    target_filepath="models/product.py",
    workspace_files=workspace,
    llm_caller_fn=mock_llm,
)
print("bundled_files:", result2.bundled_files)
print("retrieval_added_files:", result2.retrieval_added_files)