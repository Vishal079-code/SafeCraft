"""
Main CLI Entrypoint: Interactive Developer Agent with Privacy-Preserving LLM Integration
"""

from pathlib import Path
from typing import Optional

from agent.orchestrator import AgentOrchestrator


def read_file_safely(filepath: str) -> Optional[str]:
    try:
        return Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ Error reading file '{filepath}': {e}")
        return None


def mock_llm_provider(prompt: str) -> str:
    """
    ⚠️ DEMO PLACEHOLDER ONLY:
    This mock returns hardcoded SEARCH/REPLACE blocks formatted for specific test identifiers
    and target file names. Replace this function with a live LLM API handler (e.g. Anthropic,
    OpenAI, or Gemini API) before running against arbitrary user files.
    """
    return """
File: target.py
<<<<<<< SEARCH
def calculate_salary(employee_grade, __ANON_VAR_1__):
    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__
    return __ANON_VAR_1__ + __ANON_VAR_2__
=======
def calculate_salary(employee_grade, __ANON_VAR_1__):
    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__
    return (__ANON_VAR_1__ + __ANON_VAR_2__) * 0.95
>>>>>>> REPLACE
"""


def run_cli():
    print("=" * 60)
    print("  🔒 PRIVACY-PRESERVING AI CODE AGENT")
    print("=" * 60)

    target_file = input("Enter target python file path (e.g., auth/payroll.py): ").strip()
    if not target_file:
        print("No target file specified. Exiting.")
        return

    content = read_file_safely(target_file)
    if content is None:
        return

    user_request = input("Enter task instruction for LLM: ").strip()
    if not user_request:
        print("No instruction provided. Exiting.")
        return

    # Configuration for protected terms
    protected_ids_input = input("Protected identifiers (comma-separated, optional): ").strip()
    protected_ids = [i.strip() for i in protected_ids_input.split(",") if i.strip()]

    protected_lits_input = input("Protected literals (comma-separated float/int, optional): ").strip()
    protected_lits = []
    for l in protected_lits_input.split(","):
        l_str = l.strip()
        if l_str:
            try:
                protected_lits.append(float(l_str) if "." in l_str else int(l_str))
            except ValueError:
                protected_lits.append(l_str)

    orchestrator = AgentOrchestrator(
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
    )

    workspace = {target_file: content}

    print("\n⏳ Processing request through local privacy pipeline...")
    result = orchestrator.run_cycle(
        user_request=user_request,
        target_filepath=target_file,
        workspace_files=workspace,
        llm_caller_fn=mock_llm_provider,
    )

    if not result.success:
        print(f"\n❌ Task Failed: {result.error_message}")
        return

    print("\n" + "=" * 60)
    print("  🔍 SANITIZED PROMPT (SENT TO REMOTE LLM)")
    print("=" * 60)
    print(result.anonymized_prompt)

    print("\n" + "=" * 60)
    print("  ✨ STAGED CHANGES (RESTORED LOCALLY)")
    print("=" * 60)
    print(result.staged_workspace[target_file])

    confirm = input("\n💾 Write changes to disk? (y/N): ").strip().lower()
    if confirm == "y":
        Path(target_file).write_text(result.staged_workspace[target_file], encoding="utf-8")
        print(f"✅ Successfully written changes to '{target_file}'!")
    else:
        print("🚫 Changes discarded.")


if __name__ == "__main__":
    run_cli()