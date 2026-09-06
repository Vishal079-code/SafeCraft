from indexer.repo_onboarder import onboard_repo
from planner.subtask_planner import generate_plan
from planner.multi_step_executor import execute_plan
from agent.orchestrator import AgentOrchestrator
from agent.llm_client import build_llm_caller
from eval.inspect_pipeline import load_repo_as_workspace

onboard_result = onboard_repo("big_repo")
retriever = onboard_result["retriever"]
workspace = load_repo_as_workspace("big_repo")

# Deliberately ambiguous - doesn't name a file or function directly.
user_task = "Make sure discounts and stock reservations both reject invalid negative quantities or percentages"

candidates = retriever.retrieve_metadata_only(user_task, top_k=8)
print("Retrieval candidates:")
for c in candidates:
    print(f"  {c['id']} (score={c['score']:.3f})")

llm_caller = build_llm_caller(provider="gemini")
plan = generate_plan(user_task, list(workspace.keys()), llm_caller, retrieval_candidates=candidates)
print("\nPlan:")
for s in plan:
    print(f"  [{s.order}] {s.target_filepath} :: {s.target_symbol} - {s.description}")

orchestrator = AgentOrchestrator(
    protected_identifiers=["apply_discount", "reserve_stock"],
    retriever=retriever,
)
result = execute_plan(plan, workspace, orchestrator, llm_caller, commit_to_disk=False)

print(f"\nsuccess: {result['success']}")
print(f"completed: {result['completed_subtasks']}/{result['total_subtasks']}")
for r in result["subtask_results"]:
    print(f"  [{r['order']}] {r['target_filepath']}::{r['target_symbol']} - success={r['success']} replanned={r['replanned']}")
if not result["success"]:
    print(f"error: {result['error_message']}")
else:
    for s in plan:
        print(f"\n--- {s.target_filepath} (final) ---")
        print(result["final_workspace"][s.target_filepath])