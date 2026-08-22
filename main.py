"""
Main CLI Entrypoint: Interactive Developer Agent with Privacy-Preserving LLM Integration.
Uses the real, retry-enabled AgentOrchestrator and (by default) the real
Gemini API - falls back to a demo mock only if no GEMINI_API_KEY is set,
so the CLI is still runnable/testable without a live key.
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from agent.orchestrator import AgentOrchestrator
from agent.llm_client import build_llm_caller


def read_file_safely(filepath: str) -> Optional[str]:
    try:
        return Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        return None


def demo_fallback_mock(prompt: str) -> str:
    """
    Used ONLY when no GEMINI_API_KEY is configured, so the CLI stays
    runnable for a quick demo/test without a live key. This is NOT
    representative of real model behavior beyond the exact toy scenario
    it was written for - real usage should always go through the actual
    API via build_llm_caller.
    """
    return (
        "File: target.py\n<<<<<<< SEARCH\n"
        "def __ANON_FUNC_1__(employee_grade, __ANON_VAR_1__):\n"
        "    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__\n"
        "    return __ANON_VAR_1__ + __ANON_VAR_2__\n=======\n"
        "def __ANON_FUNC_1__(employee_grade, __ANON_VAR_1__):\n"
        "    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__\n"
        "    return (__ANON_VAR_1__ + __ANON_VAR_2__) * 0.95\n>>>>>>> REPLACE"
    )


def run_cli():
    print("=" * 60)
    print("  PRIVACY-PRESERVING AI CODE AGENT")
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

    orchestrator = AgentOrchestrator(protected_identifiers=protected_ids, protected_literals=protected_lits)
    workspace = {target_file: content}

    has_key = bool(os.getenv("GEMINI_API_KEY"))
    if has_key:
        print("\nUsing live Gemini API...")
        llm_caller = build_llm_caller(provider="gemini")
    else:
        print("\nNo GEMINI_API_KEY found - using demo fallback mock (not real model output).")
        llm_caller = demo_fallback_mock

    print("\nProcessing request through local privacy pipeline (up to 2 attempts on patch mismatch)...")
    result = orchestrator.run_cycle(
        user_request=user_request,
        target_filepath=target_file,
        workspace_files=workspace,
        llm_caller_fn=llm_caller,
    )

    if not result.success:
        print(f"\nTask Failed: {result.error_message}")
        return

    print(f"\nSucceeded on attempt {result.attempts_used} of up to 2.")

    print("\n" + "=" * 60)
    print("  SANITIZED PROMPT (SENT TO REMOTE LLM)")
    print("=" * 60)
    print(result.anonymized_prompt)

    print("\n" + "=" * 60)
    print("  STAGED CHANGES (RESTORED LOCALLY)")
    print("=" * 60)
    print(result.staged_workspace[target_file])

    confirm = input("\nWrite changes to disk? (y/N): ").strip().lower()
    if confirm == "y":
        Path(target_file).write_text(result.staged_workspace[target_file], encoding="utf-8")
        print(f"Successfully written changes to '{target_file}'!")
    else:
        print("Changes discarded.")


if __name__ == "__main__":
    run_cli()