"""
Sequential Subtask Executor: runs a subtask plan through AgentOrchestrator,
one file at a time, in order, against a shared evolving workspace.

V2 change - explicit verification, bounded replan, single final commit:
Each subtask's staged result is now EXPLICITLY verified (validate_staged_workspace),
not just implicitly checked via ast.parse inside patch application. On
verification failure OR a patch/LLM failure, ONE replan attempt is made -
the subtask's description is revised with the failure reason and re-run
through run_cycle once. If that ALSO fails, the ENTIRE plan stops -
later subtasks are never attempted, and nothing already-succeeded in
this call is written to disk (see single-commit note below).

This is a genuine behavior change from the prior per-subtask-commit
version: commit_to_disk is no longer passed to individual run_cycle
calls at all. If commit_to_disk=True is passed to execute_plan itself,
a SINGLE commit happens only after EVERY subtask has passed - all or
nothing, rather than each success writing immediately. This matches the
project's human-in-the-loop principle for its interactive use
(session.py) by extension: this non-interactive executor is meant for
programmatic/batch use (eval scripts, tests, a future automated mode)
where there's no per-step human review to catch a bad intermediate
state - all-or-nothing commit is the safety net standing in for that
review. session.py's own interactive loop is STRICTER than this (a
human approves every individual file before it's written) and
continues to call AgentOrchestrator.run_cycle directly per file rather
than through this function - the two are separate, deliberately
different-strictness code paths for different situations, not a
duplication to be merged.
"""
import logging
from typing import Dict, Any, List, Optional

from planner.subtask_planner import Subtask
from agent.orchestrator import AgentOrchestrator, AgentResult
from verifier.self_corrector import validate_staged_workspace
from agent.disk_committer import commit_staged_workspace_to_disk, DiskCommitError

logger = logging.getLogger(__name__)


def _run_and_verify(orchestrator, user_request, subtask, current_workspace, llm_caller_fn):
    """One attempt: run_cycle + explicit verification. Returns
    (passed: bool, result: AgentResult, failure_reason: Optional[str])."""
    result: AgentResult = orchestrator.run_cycle(
        user_request=user_request,
        target_filepath=subtask.target_filepath,
        target_symbol=subtask.target_symbol,
        workspace_files=current_workspace,
        llm_caller_fn=llm_caller_fn,
        commit_to_disk=False,  # never per-subtask now - see module docstring
    )

    if not result.success:
        return False, result, result.error_message

    verification = validate_staged_workspace(
        result.staged_workspace,
        modified_files=[subtask.target_filepath],
        syntax_only=True,
    )
    if not verification["valid"]:
        reason = verification.get("error_message", "Verification failed (syntax check).")
        return False, result, reason

    return True, result, None


def execute_plan(
    subtasks: List[Subtask],
    workspace_files: Dict[str, str],
    orchestrator: AgentOrchestrator,
    llm_caller_fn,
    commit_to_disk: bool = False,
    disk_root: Optional[str] = None,
) -> Dict[str, Any]:
    current_workspace = dict(workspace_files)
    subtask_results = []

    for subtask in subtasks:
        passed, result, failure_reason = _run_and_verify(
            orchestrator, subtask.description, subtask, current_workspace, llm_caller_fn
        )

        replanned = False
        if not passed:
            replanned = True
            revised_request = (
                f"{subtask.description}\n\n"
                f"NOTE: a previous attempt at this failed with: {failure_reason}. "
                f"Please produce a corrected fix that avoids this issue."
            )
            logger.info(f"Subtask {subtask.order} failed ({failure_reason}) - attempting ONE replan.")
            passed, result, failure_reason = _run_and_verify(
                orchestrator, revised_request, subtask, current_workspace, llm_caller_fn
            )

        subtask_results.append({
            "order": subtask.order,
            "description": subtask.description,
            "target_filepath": subtask.target_filepath,
            "target_symbol": subtask.target_symbol,
            "success": passed,
            "replanned": replanned,
            "error_message": None if passed else failure_reason,
        })

        if not passed:
            logger.warning(
                f"Subtask {subtask.order} failed{' even after replan' if replanned else ''} "
                f"- stopping ENTIRE plan. No further subtasks attempted."
            )
            return {
                "success": False,
                "completed_subtasks": subtask.order - 1,
                "total_subtasks": len(subtasks),
                "subtask_results": subtask_results,
                "final_workspace": current_workspace,  # state BEFORE this failed subtask
                "committed_to_disk": False,
                "files_written": 0,
                "error_message": f"Stopped at subtask {subtask.order}: {failure_reason}",
            }

        current_workspace = result.staged_workspace

    # Every subtask passed - single final commit, only if requested.
    committed_to_disk = False
    files_written = 0
    if commit_to_disk:
        if not disk_root:
            return {
                "success": False, "completed_subtasks": len(subtasks), "total_subtasks": len(subtasks),
                "subtask_results": subtask_results, "final_workspace": current_workspace,
                "committed_to_disk": False, "files_written": 0,
                "error_message": "commit_to_disk=True but no disk_root was provided.",
            }
        try:
            committed_to_disk, files_written = commit_staged_workspace_to_disk(current_workspace, disk_root)
        except DiskCommitError as e:
            return {
                "success": False, "completed_subtasks": len(subtasks), "total_subtasks": len(subtasks),
                "subtask_results": subtask_results, "final_workspace": current_workspace,
                "committed_to_disk": False, "files_written": 0,
                "error_message": f"Disk Commit Blocked: {e}",
            }

    return {
        "success": True,
        "completed_subtasks": len(subtasks),
        "total_subtasks": len(subtasks),
        "subtask_results": subtask_results,
        "final_workspace": current_workspace,
        "committed_to_disk": committed_to_disk,
        "files_written": files_written,
        "error_message": None,
    }