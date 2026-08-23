"""
Agent Orchestrator: Multi-file pipeline connecting Privacy Anonymization,
LLM Prompting, Diff Parsing, De-anonymization, In-Memory Patch Staging,
and Atomic Disk Commit.

Multi-file anonymization uses anonymize_workspace() (a SHARED transformer
across files), not per-file anonymize_code() calls merged afterward -
this is what guarantees the same real identifier maps to the same
placeholder across file boundaries, and avoids counter collisions.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from privacy.anonymizer import anonymize_workspace
from privacy.deanonymizer import deanonymize_code
from agent.diff_parser import parse_search_replace_blocks, CodePatch
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
        "You are an expert AI code assistant. Update the provided workspace files according to the user request.\n"
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
        if not workspace_files:
            return AgentResult(success=False, staged_workspace={}, patches=[], anonymized_prompt="", error_message="Workspace files map is empty.")

        if target_filepath not in workspace_files:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt="", error_message=f"Target file '{target_filepath}' not found in workspace.")

        anonymized_workspace, privacy_maps = anonymize_workspace(
            workspace_files, protected_identifiers=self.protected_identifiers, protected_literals=self.protected_literals,
        )
        if "error" in privacy_maps:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt="", error_message=f"Anonymization failed: {privacy_maps['error']}")

        system_prompt = build_system_prompt()
        file_blocks = [f"File: {path}\n```python\n{code}\n```" for path, code in anonymized_workspace.items()]
        workspace_context = "\n\n".join(file_blocks)
        user_prompt = f"User Task: {user_request}\n\nPrimary Target File: {target_filepath}\n\nWorkspace Files:\n{workspace_context}"
        full_llm_prompt = f"{system_prompt}\n\n{user_prompt}"

        last_error = None
        last_patches: List[CodePatch] = []

        for attempt in range(1, max_attempts + 1):
            try:
                llm_response = llm_caller_fn(full_llm_prompt)
            except Exception as e:
                last_error = f"LLM Invocation Error: {str(e)}"
                continue

            raw_blocks = parse_search_replace_blocks(llm_response, default_filepath=target_filepath)
            if not raw_blocks:
                last_error = "Malformed patch: No valid SEARCH/REPLACE blocks found in LLM response."
                continue

            still_anonymized_patches = []
            malformed = False
            for filepath, search_block, replace_block in raw_blocks:
                target_path = filepath or target_filepath
                if not search_block.strip():
                    last_error = f"Invalid patch for '{target_path}': SEARCH block cannot be empty."
                    malformed = True
                    break
                still_anonymized_patches.append(CodePatch(filepath=target_path, search_text=search_block, replace_text=replace_block))
            if malformed:
                continue

            try:
                staged_anonymized = apply_patches_staged(anonymized_workspace, still_anonymized_patches)
            except PatchApplyError as e:
                last_error = f"Patch Staging Failed: {str(e)}"
                continue

            staged_workspace = dict(workspace_files)
            for path, anon_content in staged_anonymized.items():
                staged_workspace[path] = deanonymize_code(anon_content, privacy_maps)

            last_patches = [
                CodePatch(
                    filepath=p.filepath,
                    search_text=deanonymize_code(p.search_text, privacy_maps),
                    replace_text=deanonymize_code(p.replace_text, privacy_maps),
                )
                for p in still_anonymized_patches
            ]

            if not commit_to_disk:
                return AgentResult(success=True, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message=None, attempts_used=attempt)

            if not disk_root:
                return AgentResult(success=False, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message="commit_to_disk=True but no disk_root was provided.", attempts_used=attempt)

            try:
                commit_success, files_written = commit_staged_workspace_to_disk(staged_workspace, disk_root)
            except DiskCommitError as e:
                return AgentResult(success=False, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message=f"Disk Commit Blocked: {str(e)}", committed_to_disk=False, files_written=0, attempts_used=attempt)

            return AgentResult(success=True, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message=None, committed_to_disk=commit_success, files_written=files_written, attempts_used=attempt)

        return AgentResult(success=False, staged_workspace=workspace_files, patches=last_patches, anonymized_prompt=full_llm_prompt,
                            error_message=f"Failed after {max_attempts} attempt(s). Last error: {last_error}", attempts_used=max_attempts)