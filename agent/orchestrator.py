"""
Agent Orchestrator: End-to-end pipeline connecting Privacy Anonymization,
LLM Prompting, Diff Parsing, De-anonymization, In-Memory Patch Staging,
and (optionally) Atomic Disk Commit.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from privacy.anonymizer import anonymize_code
from agent.diff_parser import process_llm_diff_response, CodePatch
from agent.patch_applier import apply_patches_staged
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
    ) -> AgentResult:
        """
        Runs a complete end-to-end cycle:
        1. Anonymizes the target file content locally.
        2. Constructs the prompt and sends it to the LLM handler.
        3. Parses and de-anonymizes the returned SEARCH/REPLACE diff.
        4. Safely applies patches in memory (stage).
        5. If commit_to_disk=True, verifies path safety and writes the
           staged workspace to disk_root. Staging (steps 1-4) always
           happens in memory first regardless of commit_to_disk - disk
           writing is strictly opt-in and is the LAST step, only reached
           if every prior step succeeded.
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

        try:
            llm_response = llm_caller_fn(full_llm_prompt)
        except Exception as e:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt=full_llm_prompt, error_message=f"LLM Invocation Error: {str(e)}")

        try:
            patches = process_llm_diff_response(llm_response=llm_response, privacy_maps=privacy_maps, default_filepath=target_filepath)
        except Exception as e:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt=full_llm_prompt, error_message=f"Diff Parsing Failed: {str(e)}")

        try:
            staged_workspace = apply_patches_staged(workspace_files, patches)
        except Exception as e:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=patches, anonymized_prompt=full_llm_prompt, error_message=f"Patch Staging Failed: {str(e)}")

        # Staging succeeded. If disk commit wasn't requested, we're done.
        if not commit_to_disk:
            return AgentResult(success=True, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt, error_message=None)

        if not disk_root:
            return AgentResult(
                success=False, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt,
                error_message="commit_to_disk=True but no disk_root was provided.",
            )

        try:
            commit_success, files_written = commit_staged_workspace_to_disk(staged_workspace, disk_root)
        except DiskCommitError as e:
            # Staging is still valid/returned - just the disk write was refused.
            # The caller can inspect staged_workspace even though nothing was written.
            return AgentResult(
                success=False, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt,
                error_message=f"Disk Commit Blocked: {str(e)}", committed_to_disk=False, files_written=0,
            )

        return AgentResult(
            success=True, staged_workspace=staged_workspace, patches=patches, anonymized_prompt=full_llm_prompt,
            error_message=None, committed_to_disk=commit_success, files_written=files_written,
        )