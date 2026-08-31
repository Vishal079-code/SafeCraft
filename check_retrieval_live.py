from dotenv import load_dotenv
load_dotenv()

from indexer.repo_indexer import build_repo_map
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index
from retriever.hybrid_retriever import HybridRetriever
from agent.orchestrator import AgentOrchestrator
from agent.llm_client import build_llm_caller
from eval.inspect_pipeline import load_repo_as_workspace

# Reuse the already-built big_repo index - no need to re-index
repo_map = build_repo_map("big_repo", "big_repo/repo_map.json")
v_store = VectorStore(db_path="./chroma_db_bigrepo", collection_name="big_repo_index")
b_index = BM25Index()
b_index.index_repo_map(repo_map)
retriever = HybridRetriever(vector_store=v_store, bm25_index=b_index)

workspace = load_repo_as_workspace("big_repo")

orchestrator = AgentOrchestrator(
    protected_identifiers=["apply_discount"],
    retriever=retriever,
    retrieval_top_k=3,
)

real_llm = build_llm_caller(provider="gemini")

print("Sending ONE real request to Gemini...\n")
result = orchestrator.run_cycle(
    user_request="Add a check so apply_discount rejects any percent over 90",
    target_filepath="models/product.py",
    workspace_files=workspace,
    llm_caller_fn=real_llm,
    max_attempts=1,   # deliberately capped at 1 - don't spend 2 requests confirming this
)

print("success:", result.success)
print("attempts_used:", result.attempts_used)
print("bundled_files:", result.bundled_files)
print("retrieval_added_files:", result.retrieval_added_files)

if result.success:
    print("\n--- Final models/product.py ---")
    print(result.staged_workspace["models/product.py"])
else:
    print("\nerror_message:", result.error_message)