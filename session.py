"""
Interactive session: knows which repo it's working on (remembers the
last one used), suggests likely-relevant files instead of requiring you
to name one, verifies each proposed change BEFORE showing it to you,
then you accept or reject before anything touches disk.
"""
from dotenv import load_dotenv
load_dotenv()
import json
from pathlib import Path

from indexer.repo_onboarder import onboard_repo
from agent.orchestrator import AgentOrchestrator
from agent.llm_client import build_llm_caller
from planner.subtask_planner import generate_plan
from verifier.self_corrector import validate_staged_workspace
from eval.inspect_pipeline import load_repo_as_workspace

LAST_REPO_FILE = Path(".safecraft_data/last_repo.json")


def print_diff(patch):
    print(f"\n  --- {patch.filepath} ---")
    print("  SEARCH:")
    for line in patch.search_text.splitlines():
        print(f"    - {line}")
    print("  REPLACE:")
    for line in patch.replace_text.splitlines():
        print(f"    + {line}")


def load_last_repo():
    if LAST_REPO_FILE.exists():
        try:
            return json.loads(LAST_REPO_FILE.read_text(encoding="utf-8")).get("repo_path")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_last_repo(repo_path: str):
    LAST_REPO_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_REPO_FILE.write_text(json.dumps({"repo_path": repo_path}), encoding="utf-8")


def load_session_config(data_dir: Path) -> dict:
    config_path = data_dir / "session_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_session_config(data_dir: Path, protected_ids: list, protected_lits: list):
    config_path = data_dir / "session_config.json"
    config_path.write_text(
        json.dumps({"protected_identifiers": protected_ids, "protected_literals": protected_lits}, indent=2),
        encoding="utf-8",
    )


def build_single_file_plan(user_request: str, target_filepath: str):
    """One explicit file, zero extra API cost - reuses generate_plan's
    parsing via a mock that never calls the real LLM."""
    def forced_mock(prompt):
        return json.dumps([{"description": user_request, "target_filepath": target_filepath}])
    return generate_plan(user_request, [target_filepath], forced_mock)


def choose_target_files(user_task: str, retriever, workspace: dict) -> list:
    """Suggests candidate files via retrieval, presented as a numbered
    list - the terminal equivalent of tick/untick checkboxes, since a
    real GUI checkbox isn't available here. Returns the chosen filepaths
    (possibly empty if the user picks none)."""
    try:
        hits = retriever.retrieve(user_task, top_k=6)
    except Exception as e:
        print(f"  Retrieval failed ({e}). You'll need to type a filepath manually.")
        hits = []

    candidates = []
    for h in hits:
        fp = (h.get("metadata") or {}).get("filepath")
        if fp and fp in workspace and fp not in candidates:
            candidates.append(fp)

    if not candidates:
        manual = input("  No suggestions found. Type a target filepath manually (or leave blank to cancel): ").strip()
        return [manual] if manual and manual in workspace else []

    print("\n  Suggested files for this task:")
    for i, fp in enumerate(candidates, start=1):
        print(f"    [{i}] {fp}")
    print(f"    [0] none of these - type a filepath manually")

    choice = input("  Select file number(s), comma-separated, or 'all': ").strip().lower()

    if choice == "all":
        return candidates
    if choice == "0":
        manual = input("  Type a target filepath manually: ").strip()
        return [manual] if manual and manual in workspace else []

    selected = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(candidates):
                selected.append(candidates[idx - 1])
    return selected


def run_session():
    print("=" * 60)
    print("  SAFECRAFT INTERACTIVE SESSION")
    print("=" * 60)

    last_repo = load_last_repo()
    repo_path = None
    if last_repo and Path(last_repo).is_dir():
        use_last = input(f"\nContinue with last repo '{last_repo}'? (Y/n): ").strip().lower()
        if use_last != "n":
            repo_path = last_repo

    if not repo_path:
        repo_path = input("\nRepo path to work on: ").strip()
        if not repo_path or not Path(repo_path).is_dir():
            print("Invalid repo path. Exiting.")
            return

    save_last_repo(repo_path)

    print("\nOnboarding repo (indexing/updating as needed)...")
    onboard_result = onboard_repo(repo_path)
    print(
        f"Ready: {onboard_result['files_total']} files "
        f"(+{onboard_result['files_added']} added, ~{onboard_result['files_changed']} changed, "
        f"-{onboard_result['files_removed']} removed, {onboard_result['files_unchanged']} unchanged)"
    )
    retriever = onboard_result["retriever"]
    data_dir = Path(onboard_result["data_dir"])

    saved_config = load_session_config(data_dir)
    saved_ids = saved_config.get("protected_identifiers", [])
    saved_lits = saved_config.get("protected_literals", [])

    if saved_ids or saved_lits:
        print(f"\nSaved protected identifiers for this repo: {saved_ids or '(none)'}")
        use_saved = input("Reuse these? (Y/n): ").strip().lower()
        protected_ids, protected_lits = (saved_ids, saved_lits) if use_saved != "n" else (None, None)
    else:
        protected_ids, protected_lits = None, None

    if protected_ids is None:
        ids_input = input("\nProtected identifiers (comma-separated, Enter to skip): ").strip()
        protected_ids = [i.strip() for i in ids_input.split(",") if i.strip()]
        lits_input = input("Protected literals (comma-separated, Enter to skip): ").strip()
        protected_lits = []
        for l in lits_input.split(","):
            l_str = l.strip()
            if l_str:
                try:
                    protected_lits.append(float(l_str) if "." in l_str else int(l_str))
                except ValueError:
                    protected_lits.append(l_str)
        save_session_config(data_dir, protected_ids, protected_lits)

    orchestrator = AgentOrchestrator(
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
        retriever=retriever,
        retrieval_top_k=5,
    )
    llm_caller = build_llm_caller(provider="gemini")
    workspace = load_repo_as_workspace(repo_path)

    print("\nSession ready. Type any task, in your own words, or 'exit' to quit.\n")

    while True:
        user_task = input("\n> Task: ").strip()
        if not user_task:
            continue
        if user_task.lower() in ("exit", "quit"):
            print("Session ended.")
            return

        chosen_files = choose_target_files(user_task, retriever, workspace)
        if not chosen_files:
            print("  No file selected. Skipping this task.")
            continue

        if len(chosen_files) == 1:
            plan = build_single_file_plan(user_task, chosen_files[0])
        else:
            print(f"  Planning across {len(chosen_files)} files...")
            try:
                plan = generate_plan(user_task, chosen_files, llm_caller)
            except ValueError as e:
                print(f"  Planning failed: {e}")
                continue
            if not plan:
                print("  Planner returned no subtasks. Nothing to do.")
                continue
            print(f"  Plan: {[(s.order, s.target_filepath, s.description) for s in plan]}")

        for subtask in plan:
            print(f"\n  Working on: {subtask.target_filepath} - {subtask.description}")

            result = orchestrator.run_cycle(
                user_request=subtask.description,
                target_filepath=subtask.target_filepath,
                workspace_files=workspace,
                llm_caller_fn=llm_caller,
            )

            if not result.success:
                print(f"  FAILED: {result.error_message}")
                print("  Stopping the rest of this task.")
                break

            show_prompt = input("  View the exact prompt sent to the LLM? (y/N): ").strip().lower()
            if show_prompt == "y":
                print("\n  ----- PROMPT SENT -----")
                print(result.anonymized_prompt)
                print("  ----- END PROMPT -----\n")
            # Verification happens BEFORE the diff is shown - a
            # syntactically broken result never even reaches your
            # accept/reject decision.
            verification = validate_staged_workspace(
                result.staged_workspace,
                modified_files=[subtask.target_filepath],
                syntax_only=True,
            )
            if not verification["valid"]:
                print(f"  VERIFICATION FAILED: {verification.get('error_message', verification)}")
                print("  Discarding this change automatically. Stopping the rest of this task.")
                break

            print("\n  Proposed change (verified, valid Python):")
            for patch in result.patches:
                print_diff(patch)

            decision = input("\n  Accept this change? (y/N): ").strip().lower()
            if decision == "y":
                new_content = result.staged_workspace[subtask.target_filepath]
                Path(repo_path, subtask.target_filepath).write_text(new_content, encoding="utf-8")
                workspace[subtask.target_filepath] = new_content
                print(f"  Written to disk: {subtask.target_filepath}")

                print("  Re-indexing changed file...")
                onboard_repo(repo_path)
            else:
                print(f"  Rejected. {subtask.target_filepath} left unchanged.")
                if len(plan) > 1:
                    print("  Stopping the rest of this multi-file task.")
                break


if __name__ == "__main__":
    run_session()