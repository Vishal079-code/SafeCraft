"""
Day 18 Evaluation Suite: real side-by-side comparison of RAW (unanonymized,
whole-workspace, unscoped) vs SAFECRAFT (anonymized, dependency-scoped)
code generation, using the ACTUAL pipeline components already built and
tested (anonymizer, orchestrator, context bundler, diff parser, verifier).
Both paths call the real, live LLM API - nothing here is hardcoded/mocked.
"""
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

import tiktoken
from dotenv import load_dotenv

load_dotenv()

from agent.orchestrator import AgentOrchestrator, build_system_prompt
from agent.diff_parser import process_llm_diff_response
from verifier.self_corrector import validate_staged_workspace
from eval.inspect_pipeline import load_repo_as_workspace

BENCHMARK_TASKS = [
    {
        "task_id": "T001",
        "description": "Add a late fee waiver: if days_overdue is 0, calculate_late_fee should return 0.0 (it already does - verify this behavior is preserved after a refactor request)",
        "target_file": "utils/payments.py",
        "protected_identifiers": ["calculate_late_fee", "days_overdue", "base_amount"],
        "protected_literals": [0.02],
        "expected_substring": "0.0",
    },
    {
        "task_id": "T002",
        "description": "Add input validation: raise ValueError if raw_password is shorter than 8 characters in verify_password",
        "target_file": "auth/login.py",
        "protected_identifiers": ["verify_password", "raw_password", "password_hash"],
        "protected_literals": [],
        "expected_substring": "ValueError",
    },
    {
        "task_id": "T003",
        "description": "Add a docstring to the User.to_dict method explaining it excludes password_hash",
        "target_file": "models/user.py",
        "protected_identifiers": ["to_dict", "password_hash"],
        "protected_literals": [],
        "expected_substring": "password_hash",
    },
]

try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    _TOKENIZER_AVAILABLE = True
except Exception:
    TOKENIZER = None
    _TOKENIZER_AVAILABLE = False


def count_tokens(text: str) -> int:
    if _TOKENIZER_AVAILABLE and TOKENIZER:
        return len(TOKENIZER.encode(text))
    return max(1, int(len(text.split()) * 1.3))


class BenchmarkRunner:
    def __init__(self, repo_path: str, raw_llm_caller: Callable[[str], str], anon_llm_caller: Callable[[str], str]):
        self.repo_path = Path(repo_path)
        self.raw_llm_caller = raw_llm_caller
        self.anon_llm_caller = anon_llm_caller
        self.results: List[Dict[str, Any]] = []
        self.full_workspace = load_repo_as_workspace(str(self.repo_path))

    def _load_workspace_file(self, rel_path: str) -> str:
        return self.full_workspace[rel_path]

    def run_raw_baseline(self, task: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        original_code = self._load_workspace_file(task["target_file"])

        file_blocks = [f"File: {path}\n```python\n{code}\n```" for path, code in self.full_workspace.items()]
        workspace_context = "\n\n".join(file_blocks)
        prompt = (
            f"{build_system_prompt()}\n\n"
            "CRITICAL REQUIREMENT: Respond ONLY with SEARCH/REPLACE diff blocks:\n"
            "<<<<<<< SEARCH\noriginal lines\n=======\nnew lines\n>>>>>>> REPLACE\n\n"
            f"User Task: {task['description']}\n\n"
            f"Primary Target File: {task['target_file']}\n\nWorkspace Files:\n{workspace_context}"
        )
        input_tokens = count_tokens(prompt)

        try:
            llm_response = self.raw_llm_caller(prompt)
        except Exception as e:
            return self._failed_result("RAW", start_time, input_tokens, error=f"LLM call failed: {e}")

        output_tokens = count_tokens(llm_response)

        try:
            patches = process_llm_diff_response(
                llm_response=llm_response,
                privacy_maps={"reverse_identifier_map": {}, "reverse_literal_map": {}},
                default_filepath=task["target_file"],
            )
        except Exception as e:
            return self._failed_result("RAW", start_time, input_tokens, output_tokens, error=f"Diff parsing failed: {e}")

        staged = self._apply_patches_safely(original_code, patches, task["target_file"])
        if staged is None:
            return self._failed_result("RAW", start_time, input_tokens, output_tokens, error="Patch application failed")

        verification = validate_staged_workspace(
            {task["target_file"]: staged},
            modified_files=[task["target_file"]],
            syntax_only=True,
        )
        expected_present = task["expected_substring"] in staged

        return {
            "mode": "RAW",
            "passed": verification["valid"] and expected_present,
            "ast_valid": verification["valid"],
            "expected_behavior_present": expected_present,
            "latency_sec": round(time.time() - start_time, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "staged_code": staged,
        }

    def run_anonymized_safecraft(self, task: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        original_code = self._load_workspace_file(task["target_file"])

        orchestrator = AgentOrchestrator(
            protected_identifiers=task["protected_identifiers"],
            protected_literals=task["protected_literals"],
        )
        result = orchestrator.run_cycle(
            user_request=task["description"],
            target_filepath=task["target_file"],
            workspace_files=self.full_workspace,
            llm_caller_fn=self.anon_llm_caller,
        )

        input_tokens = count_tokens(result.anonymized_prompt) if result.anonymized_prompt else 0

        if not result.success:
            # bundled_files is populated by run_cycle on EVERY return path,
            # success or failure - pass it through instead of losing it in
            # _failed_result, so a failure still tells us what was scoped.
            return self._failed_result(
                "SAFECRAFT_ANONYMIZED", start_time, input_tokens,
                error=result.error_message, bundled_files=result.bundled_files,
            )

        staged = result.staged_workspace[task["target_file"]]
        output_tokens = max(count_tokens(staged) - count_tokens(original_code), 0)

        verification = validate_staged_workspace(
            {task["target_file"]: staged},
            modified_files=[task["target_file"]],
            syntax_only=True,
        )
        expected_present = task["expected_substring"] in staged
        deanon_success = "__ANON_" not in staged

        return {
            "mode": "SAFECRAFT_ANONYMIZED",
            "passed": verification["valid"] and expected_present,
            "ast_valid": verification["valid"],
            "expected_behavior_present": expected_present,
            "deanon_success": deanon_success,
            "bundled_files": result.bundled_files,
            "latency_sec": round(time.time() - start_time, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "staged_code": staged,
        }

    def _apply_patches_safely(self, original_code: str, patches, target_file: str) -> Optional[str]:
        from agent.patch_applier import apply_patch_to_content, PatchApplyError
        content = original_code
        try:
            for patch in patches:
                if patch.filepath == target_file:
                    content = apply_patch_to_content(content, patch)
            return content
        except PatchApplyError:
            return None

    def _failed_result(self, mode, start_time, input_tokens, output_tokens=0, error="", bundled_files=None):
        return {
            "mode": mode, "passed": False, "ast_valid": False, "expected_behavior_present": False,
            "deanon_success": None, "bundled_files": bundled_files,
            "latency_sec": round(time.time() - start_time, 2),
            "input_tokens": input_tokens, "output_tokens": output_tokens, "error": error, "staged_code": "",
        }

    def execute_eval(self):
        print("Starting SafeCraft Day 18 Quality & Token Benchmark...\n")
        print(f"Full workspace loaded: {len(self.full_workspace)} files from {self.repo_path}\n")
        for task in BENCHMARK_TASKS:
            print(f"Executing Task [{task['task_id']}]: {task['description']}")

            raw_res = self.run_raw_baseline(task)
            anon_res = self.run_anonymized_safecraft(task)

            token_savings_pct = 0.0
            if raw_res["input_tokens"] > 0:
                token_savings_pct = round(
                    ((raw_res["input_tokens"] - anon_res["input_tokens"]) / raw_res["input_tokens"]) * 100, 2
                )

            comparison = {
                "task_id": task["task_id"],
                "raw_passed": raw_res["passed"],
                "anon_passed": anon_res["passed"],
                "raw_input_tokens": raw_res["input_tokens"],
                "anon_input_tokens": anon_res["input_tokens"],
                "token_savings_pct": token_savings_pct,
                "bundled_files": anon_res.get("bundled_files"),
                "deanon_integrity": anon_res.get("deanon_success", False),
                "raw_error": raw_res.get("error", ""),
                "anon_error": anon_res.get("error", ""),
            }
            self.results.append(comparison)
            print(f"  Raw Pass: {raw_res['passed']} | Anon Pass: {anon_res['passed']} | "
                  f"Tokens: {raw_res['input_tokens']} -> {anon_res['input_tokens']} "
                  f"({token_savings_pct}% change) | Bundled: {comparison['bundled_files']} | "
                  f"Deanon OK: {comparison['deanon_integrity']}")
            # Print errors immediately to console - don't make the JSON file
            # the only place a failure's real cause is visible.
            if comparison["raw_error"]:
                print(f"    RAW error: {comparison['raw_error']}")
            if comparison["anon_error"]:
                print(f"    ANON error: {comparison['anon_error']}")
            print()

        self.save_report()

    def save_report(self):
        report_dir = Path(".safecraft_data")
        report_dir.mkdir(exist_ok=True)
        # Timestamped filename - previous runs' results (and their error
        # detail) are otherwise silently overwritten by the next run, which
        # cost us the actual error text on three separate runs in a row.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"benchmark_results_{timestamp}.json"
        report_path.write_text(json.dumps(self.results, indent=2))
        print(f"Benchmark Report Saved to: {report_path.resolve()}")


if __name__ == "__main__":
    from agent.llm_client import build_llm_caller

    real_caller = build_llm_caller(provider="gemini")
    runner = BenchmarkRunner(repo_path="./sample_repo", raw_llm_caller=real_caller, anon_llm_caller=real_caller)
    runner.execute_eval()