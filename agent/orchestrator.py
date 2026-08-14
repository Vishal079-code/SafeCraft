"""
Agent Orchestrator: End-to-end pipeline connecting Privacy Anonymization,
LLM Prompting, Diff Parsing, De-anonymization, and In-Memory Patch Staging.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from privacy.anonymizer import anonymize_code
from agent.diff_parser import process_llm_diff_response, CodePatch
from agent.patch_applier import apply_patches_staged


@dataclass
class AgentResult:
    success: bool
    staged_workspace: Dict[str, str]
    patches: List[CodePatch]
    anonymized_prompt: str
    error_message: Optional[str] = None


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
    def __init__(
        self,
        protected_identifiers: Optional[List[str]] = None,
        protected_literals: Optional[List[Any]] = None,
    ):
        self.protected_identifiers = protected_identifiers or []
        self.protected_literals = protected_literals or []

    def run_cycle(
        self,
        user_request: str,
        target_filepath: str,
        workspace_files: Dict[str, str],
        llm_caller_fn,
    ) -> AgentResult:
        """
        Runs a complete end-to-end cycle:
        1. Anonymizes the target file content locally.
        2. Constructs the prompt and sends it to the LLM handler.
        3. Parses and de-anonymizes the returned SEARCH/REPLACE diff.
        4. Safely applies patches in memory.
        """
        if target_filepath not in workspace_files:
            return AgentResult(
                success=False,
                staged_workspace=workspace_files,
                patches=[],
                anonymized_prompt="",
                error_message=f"Target file '{target_filepath}' not found in workspace.",
            )

        original_code = workspace_files[target_filepath]

        # 1. Anonymize target code locally
        anonymized_code, privacy_maps = anonymize_code(
            code_snippet=original_code,
            filepath=target_filepath,
            protected_identifiers=self.protected_identifiers,
            protected_literals=self.protected_literals,
        )

        if "error" in privacy_maps:
            return AgentResult(
                success=False,
                staged_workspace=workspace_files,
                patches=[],
                anonymized_prompt="",
                error_message=f"Anonymization failed: {privacy_maps['error']}",
            )

        # 2. Build local sanitized prompt
        system_prompt = build_system_prompt()
        user_prompt = (
            f"User Task: {user_request}\n\n"
            f"File: {target_filepath}\n"
            f"```python\n{anonymized_code}\n```"
        )
        full_llm_prompt = f"{system_prompt}\n\n{user_prompt}"

        # 3. Call LLM (or mock handler in tests)
        try:
            llm_response = llm_caller_fn(full_llm_prompt)
        except Exception as e:
            return AgentResult(
                success=False,
                staged_workspace=workspace_files,
                patches=[],
                anonymized_prompt=full_llm_prompt,
                error_message=f"LLM Invocation Error: {str(e)}",
            )

        # 4. Parse SEARCH/REPLACE blocks & de-anonymize back to original code
        try:
            patches = process_llm_diff_response(
                llm_response=llm_response,
                privacy_maps=privacy_maps,
                default_filepath=target_filepath,
            )
        except Exception as e:
            return AgentResult(
                success=False,
                staged_workspace=workspace_files,
                patches=[],
                anonymized_prompt=full_llm_prompt,
                error_message=f"Diff Parsing Failed: {str(e)}",
            )

        # 5. Apply patches safely in memory
        try:
            staged_workspace = apply_patches_staged(workspace_files, patches)
        except Exception as e:
            return AgentResult(
                success=False,
                staged_workspace=workspace_files,
                patches=patches,
                anonymized_prompt=full_llm_prompt,
                error_message=f"Patch Staging Failed: {str(e)}",
            )

        return AgentResult(
            success=True,
            staged_workspace=staged_workspace,
            patches=patches,
            anonymized_prompt=full_llm_prompt,
            error_message=None,
        )