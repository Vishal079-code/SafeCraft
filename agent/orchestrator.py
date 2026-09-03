"""
Agent Orchestrator: Multi-file pipeline connecting Context Bundling
(V2: target-scope extraction), (optional) Semantic Retrieval, Privacy
Anonymization, LLM Prompting, Diff Parsing, De-anonymization, Patch
Staging, Splice-back, and Atomic Disk Commit.

=== V2 change: target_symbol and the splice-back requirement ===

When target_symbol is provided, ContextBundler sends the target file
with ONLY that function/method's real body - everything else in the
file is skeletonized. This means the anonymized/patched version of the
target file that comes back from the LLM is NOT safe to write to disk
directly - it would contain `...` in place of every OTHER function in
that file, destroying them.

So when target_symbol is set, after the normal anonymize -> LLM ->
patch -> de-anonymize flow produces a de-anonymized version of the
target file, splice_function_into_original() is used to extract ONLY
the target function's new source from that result and merge it into
the PRISTINE original file via a line-range text splice - never via
ast.unparse() on the whole file, which would silently reformat every
untouched function too (see bug #5 in the project README).

If the splice fails for any reason (target symbol not found in either
the original or the patched result), the run is treated as a FAILURE,
not silently written - see the code below. A failed splice must never
result in an accepted, partially-destructive write.

target_symbol=None (the default) preserves prior behavior exactly:
whole target file sent, whole target file staged, no splicing.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from agent.context_bundler import ContextBundler, splice_function_into_original
from agent.skeletonizer import skeletonize_code
from privacy.anonymizer import anonymize_workspace
from privacy.deanonymizer import deanonymize_code
from agent.diff_parser import parse_search_replace_blocks, CodePatch
from agent.patch_applier import apply_patches_staged, PatchApplyError
from agent.disk_committer import commit_staged_workspace_to_disk, DiskCommitError

logger = logging.getLogger(__name__)


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
    bundled_files: Optional[List[str]] = None
    retrieval_added_files: Optional[List[str]] = None


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
    def __init__(
        self,
        protected_identifiers=None,
        protected_literals=None,
        max_dependency_depth: int = 2,
        retriever=None,
        retrieval_top_k: int = 3,
    ):
        self.protected_identifiers = protected_identifiers or []
        self.protected_literals = protected_literals or []
        self.max_dependency_depth = max_dependency_depth
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k

    def _get_retrieval_additions(self, user_request, workspace_files, already_bundled):
        additions: Dict[str, str] = {}
        if self.retriever is None:
            return additions
        try:
            hits = self.retriever.retrieve(user_request, top_k=self.retrieval_top_k)
        except Exception as e:
            logger.warning(f"Semantic retrieval query failed, continuing without it: {e}")
            return additions
        for hit in hits or []:
            meta = (hit or {}).get("metadata") or {}
            filepath = meta.get("filepath")
            if not filepath or filepath in already_bundled or filepath in additions:
                continue
            if filepath not in workspace_files:
                continue
            additions[filepath] = skeletonize_code(workspace_files[filepath])
        return additions

    def run_cycle(
        self,
        user_request: str,
        target_filepath: str,
        workspace_files: Dict[str, str],
        llm_caller_fn,
        target_symbol: Optional[str] = None,
        commit_to_disk: bool = False,
        disk_root: Optional[str] = None,
        max_attempts: int = 2,
    ) -> AgentResult:
        if not workspace_files:
            return AgentResult(success=False, staged_workspace={}, patches=[], anonymized_prompt="", error_message="Workspace files map is empty.")

        if target_filepath not in workspace_files:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt="", error_message=f"Target file '{target_filepath}' not found in workspace.")

        bundler = ContextBundler(workspace_files, max_depth=self.max_dependency_depth)
        bundled_files = bundler.build_bundled_context(
            target_filepath, target_symbol=target_symbol, skeletonize_dependencies=True
        )

        retrieval_additions = self._get_retrieval_additions(user_request, workspace_files, bundled_files)
        bundled_files.update(retrieval_additions)

        bundled_file_list = list(bundled_files.keys())
        retrieval_added_list = list(retrieval_additions.keys()) if self.retriever is not None else None

        anonymized_workspace, privacy_maps = anonymize_workspace(
            bundled_files, protected_identifiers=self.protected_identifiers, protected_literals=self.protected_literals,
        )
        if "error" in privacy_maps:
            return AgentResult(success=False, staged_workspace=workspace_files, patches=[], anonymized_prompt="", error_message=f"Anonymization failed: {privacy_maps['error']}", bundled_files=bundled_file_list, retrieval_added_files=retrieval_added_list)

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

            # V2 splice-back: if target_symbol was used, the target
            # file's staged content above is a partially-skeletonized
            # view, NOT safe to write as-is. Splice just the target
            # function into the pristine original instead.
            if target_symbol is not None and target_filepath in staged_anonymized:
                try:
                    spliced = splice_function_into_original(
                        original_content=workspace_files[target_filepath],
                        target_symbol=target_symbol,
                        patched_content_with_target=staged_workspace[target_filepath],
                    )
                    staged_workspace[target_filepath] = spliced
                except ValueError as e:
                    last_error = f"Splice-back failed: {str(e)}"
                    continue  # do NOT return/write - retry or fail cleanly, never write a partial result

            last_patches = [
                CodePatch(
                    filepath=p.filepath,
                    search_text=deanonymize_code(p.search_text, privacy_maps),
                    replace_text=deanonymize_code(p.replace_text, privacy_maps),
                )
                for p in still_anonymized_patches
            ]

            if not commit_to_disk:
                return AgentResult(success=True, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message=None, attempts_used=attempt, bundled_files=bundled_file_list, retrieval_added_files=retrieval_added_list)

            if not disk_root:
                return AgentResult(success=False, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message="commit_to_disk=True but no disk_root was provided.", attempts_used=attempt, bundled_files=bundled_file_list, retrieval_added_files=retrieval_added_list)

            try:
                commit_success, files_written = commit_staged_workspace_to_disk(staged_workspace, disk_root)
            except DiskCommitError as e:
                return AgentResult(success=False, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message=f"Disk Commit Blocked: {str(e)}", committed_to_disk=False, files_written=0, attempts_used=attempt, bundled_files=bundled_file_list, retrieval_added_files=retrieval_added_list)

            return AgentResult(success=True, staged_workspace=staged_workspace, patches=last_patches, anonymized_prompt=full_llm_prompt, error_message=None, committed_to_disk=commit_success, files_written=files_written, attempts_used=attempt, bundled_files=bundled_file_list, retrieval_added_files=retrieval_added_list)

        return AgentResult(success=False, staged_workspace=workspace_files, patches=last_patches, anonymized_prompt=full_llm_prompt,
                            error_message=f"Failed after {max_attempts} attempt(s). Last error: {last_error}", attempts_used=max_attempts, bundled_files=bundled_file_list, retrieval_added_files=retrieval_added_list)