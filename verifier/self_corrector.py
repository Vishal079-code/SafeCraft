"""
Self-Correction Feedback Loop: Validates staged code changes using AST
syntax parsing and automated pytest execution. If failures occur, builds
a diagnostic prompt to feed back into AgentToolLoop for a repair round.
"""
import logging
from typing import Dict, Any, List, Optional
from verifier.ast_verifier import verify_code_syntax
from verifier.test_runner import run_pytest_suite

logger = logging.getLogger(__name__)


def validate_staged_workspace(
    staged_workspace: Dict[str, str],
    modified_files: List[str],
    test_dir: str = "tests/",
    syntax_only: bool = False,
) -> Dict[str, Any]:
    """
    Validates staged workspace files in two phases:
    1. AST syntax check on all modified files.
    2. Pytest execution against the test suite (skipped entirely if
       syntax_only=True - useful for benchmarks/tasks with no matching
       test file, where running the full suite would be slow and largely
       irrelevant to what's being measured).
    """
    # Phase 1: AST Syntax Validation
    for filepath in modified_files:
        if filepath not in staged_workspace:
            # FIX: previously defaulted to "" via .get(filepath, ""), and
            # ast.parse("") is VALID Python (an empty module) - so a file
            # that was supposed to be modified but is missing from
            # staged_workspace entirely silently passed validation instead
            # of surfacing a real upstream problem (e.g. a bug in whatever
            # produced modified_files, or a filepath typo).
            logger.warning(f"'{filepath}' listed in modified_files but missing from staged_workspace.")
            return {
                "valid": False,
                "error_type": "MISSING_FILE",
                "failed_file": filepath,
                "feedback": f"'{filepath}' was expected to be modified but is not present in the staged workspace.",
            }

        code_str = staged_workspace[filepath]
        is_valid, err_msg = verify_code_syntax(code_str, filename=filepath)
        if not is_valid:
            logger.warning(f"AST syntax error in {filepath}: {err_msg}")
            return {
                "valid": False,
                "error_type": "AST_SYNTAX_ERROR",
                "failed_file": filepath,
                "feedback": f"Syntax error detected in staged changes for '{filepath}':\n{err_msg}\n"
                            f"Please generate a corrected patch fixing this syntax error.",
            }

    # Phase 2: Pytest Execution (skipped if syntax_only=True)
    if syntax_only:
        return {"valid": True, "error_type": None, "feedback": None}

    test_results = run_pytest_suite(test_dir=test_dir)
    if not test_results["success"]:
        logger.warning(f"Pytest suite failed with return code {test_results['returncode']}")
        return {
            "valid": False,
            "error_type": "TEST_FAILURE",
            "failed_file": None,
            "feedback": f"Automated tests failed after applying patch:\n{test_results['traceback']}\n"
                        f"Please analyze the traceback above and adjust the patch to fix the failing tests.",
        }

    return {"valid": True, "error_type": None, "feedback": None}