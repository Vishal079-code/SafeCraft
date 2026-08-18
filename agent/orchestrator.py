"""
Agent Orchestrator: End-to-end pipeline connecting Privacy Anonymization,
LLM Prompting, Diff Parsing, De-anonymization, In-Memory Patch Staging,
and (optionally) Atomic Disk Commit.

Includes a retry loop for patch-staging failures - motivated by real,
observed evidence that LLM output is not perfectly deterministic even at
temperature=0: the exact same task/prompt can produce a SEARCH block that
matches on one call and doesn't on another. Retrying with a fresh LLM
call is a cheap, evidence-backed mitigation for this specific failure mode.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from privacy.anonymizer import anonymize_code
from agent.diff_parser import process_llm_diff_response, CodePatch
from agent.patch_applier import apply_patches_staged, PatchApplyError
from agent.disk_committer import commit_staged_workspace_to_disk, DiskCommitError


@dataclass
class AgentResult:
    success: bool
    staged_workspace: Dict[str, str]
    patches: List[CodePatch]
    anonymized_prompt: str
    error_message: Optional[str] = None
    committed_to_disk: bool = False
    files_written: int = 0
    attempts_used: int = 1


def build_system_prompt() -> str:
    return (
        "You are an expert AI code assistant. Update the provided code according to the user request.\n"
        "You MUST output code modifications strictly using SEARCH/REPLACE blocks in the following format:\n\n"
        "File: <filepath>\n"
        "<<<<<<< SEARCH\n"
        "<exact code to replace>\n"
        "=======\n"
        "<new updated code>\n"
        ">>>>>>> REPLACE\n\n"
        "Do not alter variable names or placeholders unless explicitly asked."
    )


class AgentOrchestrator:
    def __init__(self, protected_identifiers=None, protected_literals=None):
        self.protected_identifiers = protected_identifiers or []
        self.protected_literals = protected_literals or []

    def run_cycle(
        self,
        user_request: str,
        target_filepath: str,
        workspace_files: Dict[str, str],
        llm_caller_fn,
        commit_to_disk: bool = False,
        disk_root: Optional[str] = None,
        max_attempts: int = 2,
    ) -> AgentResult:
        """
        Runs a complete end-to-end cycle, retrying the LLM call + patch
        staging step up to max_attempts times if staging fails.

        Anonymization happens ONCE (identifiers/literals don't change
        between retries - only the LLM's response might). Each retry
        makes a fresh LLM call with the SAME anonymized prompt, since
        we've observed that a repeat call can succeed where a prior one
        didn't, even with no change to the input.
        """
        if target_filepath not in workspace_files:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt="", error_message=f"Target file '{target_filepath}' not found in workspace.")

        original_code = workspace_files[target_filepath]

        anonymized_code, privacy_maps = anonymize_code(
            code_snippet=original_code, filepath=target_filepath,
            protected_identifiers=self.protected_identifiers, protected_literals=self.protected_literals,
        )
        if "error" in privacy_maps:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt="", error_message=f"Anonymization failed: {privacy_maps['error']}")

        system_prompt = build_system_prompt()
        user_prompt = f"User Task: {user_request}\n\nFile: {target_filepath}\n```python\n{anonymized_code}\n```"
        full_llm_prompt = f"{system_prompt}\n\n{user_prompt}"

        last_error = None
        last_patches: List[CodePatch] = []

        for attempt in range(1, max_attempts + 1):
            try:
                llm_response = llm_caller_fn(full_llm_prompt)
            except Exception as e:
                last_error = f"LLM Invocation Error: {str(e)}"
                continue  # retry - a transient API error may not recur

            try:
                patches = process_llm_diff_response(llm_response=llm_response, privacy_maps=privacy_maps, default_filepath=target_filepath)
            except Exception as e:
                last_error = f"Diff Parsing Failed: {str(e)}"
                continue  # retry - the model may format correctly next time

            last_patches = patches

            try:
                staged_workspace = apply_patches_staged(workspace_files, patches)
            except PatchApplyError as e:
                # This is the exact failure mode observed in practice:
                # SEARCH block didn't match due to LLM non-determinism.
                # A fresh call with the SAME prompt has a real chance of
                # succeeding, since nothing about the input changed.
                last_error = f"Patch Staging Failed: {str(e)}"
                continue

            # Staging succeeded.
            if not commit_to_disk:
                return AgentResult(success=True, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt, error_message=None, attempts_used=attempt)

            if not disk_root:
                return AgentResult(success=False, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt, error_message="commit_to_disk=True but no disk_root was provided.", attempts_used=attempt)

            try:
                commit_success, files_written = commit_staged_workspace_to_disk(staged_workspace, disk_root)
            except DiskCommitError as e:
                return AgentResult(success=False, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt, error_message=f"Disk Commit Blocked: {str(e)}", committed_to_disk=False, files_written=0, attempts_used=attempt)

            return AgentResult(success=True, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt, error_message=None, committed_to_disk=commit_success, files_written=files_written, attempts_used=attempt)

        # All attempts exhausted.
        return AgentResult(
            success=False, staged_workspace=workspace_files, patches=last_patches, anonymized_prompt=full_llm_prompt,
            error_message=f"Failed after {max_attempts} attempt(s). Last error: {last_error}", attempts_used=max_attempts,
        )