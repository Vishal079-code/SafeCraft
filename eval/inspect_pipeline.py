"""
Pipeline Inspector: runs a full user request through the planner +
multi-step executor, printing EVERY intermediate step - the generated
plan, the exact prompt sent for each subtask, the raw LLM response, the
parsed patch, and the final diff - so you can verify the whole system
is doing the right thing at each stage, not just trust the final result.
"""
import json
from pathlib import Path
from typing import Dict

from planner.subtask_planner import generate_plan
from agent.orchestrator import AgentOrchestrator


def load_repo_as_workspace(repo_root: str) -> Dict[str, str]:
    """Loads every .py file under repo_root into a {relative_path: content} dict."""
    root = Path(repo_root)
    workspace = {}
    for path in root.rglob("*.py"):
        rel = str(path.relative_to(root)).replace("\\", "/")
        workspace[rel] = path.read_text(encoding="utf-8")
    return workspace


def inspect_run(
    user_request: str,
    workspace_files: Dict[str, str],
    planner_llm_fn,
    orchestrator_llm_fn,
    protected_identifiers=None,
    protected_literals=None,
):
    print("=" * 70)
    print("  STEP 1: PLANNER OUTPUT")
    print("=" * 70)
    plan = generate_plan(user_request, list(workspace_files.keys()), planner_llm_fn)
    for s in plan:
        print(f"  [{s.order}] target={s.target_filepath!r}  -> {s.description}")

    orchestrator = AgentOrchestrator(protected_identifiers=protected_identifiers, protected_literals=protected_literals)
    current_workspace = dict(workspace_files)

    for subtask in plan:
        print()
        print("=" * 70)
        print(f"  STEP 2.{subtask.order}: EXECUTING SUBTASK -> {subtask.target_filepath}")
        print("=" * 70)

        result = orchestrator.run_cycle(
            user_request=subtask.description,
            target_filepath=subtask.target_filepath,
            workspace_files=current_workspace,
            llm_caller_fn=orchestrator_llm_fn,
        )

        print(f"--- Prompt SENT (first 800 chars) ---")
        print(result.anonymized_prompt[:800])
        print("...(truncated)" if len(result.anonymized_prompt) > 800 else "")

        print()
        print(f"--- Result: success={result.success}, attempts_used={result.attempts_used} ---")
        if not result.success:
            print(f"ERROR: {result.error_message}")
            print()
            print("STOPPING inspection - pipeline halted here, same as it would in real execution.")
            return

        print()
        print("--- Patches applied (de-anonymized, real names) ---")
        for p in result.patches:
            print(f"  File: {p.filepath}")
            print(f"    SEARCH:  {p.search_text!r}")
            print(f"    REPLACE: {p.replace_text!r}")

        current_workspace = result.staged_workspace

    print()
    print("=" * 70)
    print("  FINAL STATE OF ALL MODIFIED FILES")
    print("=" * 70)
    for subtask in plan:
        print(f"\n--- {subtask.target_filepath} ---")
        print(current_workspace[subtask.target_filepath])


if __name__ == "__main__":
    workspace = load_repo_as_workspace("big_repo")
    print(f"Loaded {len(workspace)} files from big_repo\n")

    def planner_mock(prompt):
        return json.dumps([
            {"description": "Add a MIN_DISCOUNT_PERCENT constant check: apply_discount should reject percent > 90", "target_filepath": "models/product.py"},
        ])

    def orchestrator_mock(prompt):
        return (
            "File: models/product.py\n<<<<<<< SEARCH\n"
            "    def __ANON_ATTR_1__(self, percent):\n"
            "        \"\"\"Apply a percentage discount to the product's price.\"\"\"\n"
            "        self.price = round(self.price * (1 - percent / 100), 2)\n"
            "        return self.price\n=======\n"
            "    def __ANON_ATTR_1__(self, percent):\n"
            "        \"\"\"Apply a percentage discount to the product's price.\"\"\"\n"
            "        if percent > 90:\n"
            "            raise ValueError(\"Discount cannot exceed 90 percent\")\n"
            "        self.price = round(self.price * (1 - percent / 100), 2)\n"
            "        return self.price\n>>>>>>> REPLACE"
        )

    inspect_run(
        user_request="Add a safety check preventing discounts over 90%",
        workspace_files=workspace,
        planner_llm_fn=planner_mock,
        orchestrator_llm_fn=orchestrator_mock,
        protected_identifiers=["apply_discount"],
        protected_literals=[],
    )