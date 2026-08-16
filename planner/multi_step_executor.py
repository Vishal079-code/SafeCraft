"""
Multi-Step Executor: runs a sequential subtask plan through
AgentOrchestrator, one file at a time, in order.

Key design point (ties back to the atomic-subtask discussion from early
in this project): each subtask's edit is staged and verified INDEPENDENTLY.
If subtask 3 of 5 fails, subtasks 1-2 remain committed (if commit_to_disk
was used) and subtasks 4-5 are never attempted - this gives a clean,
resumable state rather than a half-applied multi-file mess, and matches
the "atomic per-subtask" design decided on much earlier in this project.
"""
import logging
from typing import Dict, Any, List, Optional

from planner.subtask_planner import Subtask
from agent.orchestrator import AgentOrchestrator, AgentResult

logger = logging.getLogger(__name__)


def execute_plan(
    subtasks: List[Subtask],
    workspace_files: Dict[str, str],
    orchestrator: AgentOrchestrator,
    llm_caller_fn,
    commit_to_disk: bool = False,
    disk_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes each subtask in order against a SHARED, evolving workspace -
    subtask 2 sees the file changes subtask 1 made (important if two
    subtasks touch the same file, or if a later subtask's context should
    reflect earlier edits).

    Stops at the FIRST failure - does not attempt remaining subtasks.
    Returns a summary with per-subtask results so the caller can see
    exactly how far execution got and why it stopped, if it did.
    """
    current_workspace = dict(workspace_files)  # working copy, don't mutate caller's dict
    subtask_results = []

    for subtask in subtasks:
        result: AgentResult = orchestrator.run_cycle(
            user_request=subtask.description,
            target_filepath=subtask.target_filepath,
            workspace_files=current_workspace,
            llm_caller_fn=llm_caller_fn,
            commit_to_disk=commit_to_disk,
            disk_root=disk_root,
        )

        subtask_results.append({
            "order": subtask.order,
            "description": subtask.description,
            "target_filepath": subtask.target_filepath,
            "success": result.success,
            "error_message": result.error_message,
            "committed_to_disk": result.committed_to_disk,
        })

        if not result.success:
            logger.warning(f"Subtask {subtask.order} failed: {result.error_message} - stopping execution.")
            return {
                "success": False,
                "completed_subtasks": subtask.order - 1,
                "total_subtasks": len(subtasks),
                "subtask_results": subtask_results,
                "final_workspace": current_workspace,
                "error_message": f"Stopped at subtask {subtask.order}: {result.error_message}",
            }

        current_workspace = result.staged_workspace

    return {
        "success": True,
        "completed_subtasks": len(subtasks),
        "total_subtasks": len(subtasks),
        "subtask_results": subtask_results,
        "final_workspace": current_workspace,
        "error_message": None,
    }