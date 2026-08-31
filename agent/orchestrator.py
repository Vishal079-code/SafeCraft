"""
Agent Orchestrator: Multi-file pipeline connecting Context Bundling,
(optional) Semantic Retrieval, Privacy Anonymization, LLM Prompting,
Diff Parsing, De-anonymization, In-Memory Patch Staging, and Atomic
Disk Commit.

=== Context sourcing strategy (read this before changing run_cycle) ===

Two independent, complementary sources decide which files get sent to
the LLM for a given task:

1. ContextBundler (deterministic, always runs): starts from
   target_filepath and walks its real Python import graph up to
   max_dependency_depth hops. Cheap (no embeddings, no API calls),
   exact, and catches "this file literally cannot be edited correctly
   without also seeing that file" cases - e.g. a base class, a shared
   constant, a function being called.

2. HybridRetriever (semantic, OPTIONAL - only runs if a retriever is
   passed into AgentOrchestrator): given the user's natural-language
   task description, returns the top_k files whose CONTENT is most
   relevant by meaning/keyword, regardless of whether they're
   import-connected to the target at all. This catches cases
   ContextBundler structurally cannot - e.g. "there's a similar
   validation pattern in an unrelated file that the LLM should follow
   for consistency," where there's no import relationship at all.

The two results are UNIONED, not one replacing the other: bundled_files
starts as ContextBundler's output, then any retriever hits not already
present are added (skeletonized, same as dependency files - only the
target file itself is ever sent full-source). This is deliberate:
either source alone misses real cases the other catches.

Retrieval is entirely optional and safe to omit:
- AgentOrchestrator(retriever=None) (the default) behaves EXACTLY as
  before retrieval existed - ContextBundler-only, no behavior change,
  no test breakage.
- If a retriever IS passed but its query fails for any reason (e.g. an
  empty/not-yet-built index for this repo), that failure is caught and
  logged, and the run proceeds with ContextBundler's results alone
  rather than failing the whole task over an optional enhancement.

Only the target file plus everything in the final unioned bundle is
anonymized and sent to the LLM - NOT the entire workspace. Files
outside the bundle are never anonymized, reformatted, or touched in
any way; they pass through untouched via a plain dict copy.

Multi-file anonymization uses anonymize_workspace() (a SHARED
transformer across files), not per-file anonymize_code() calls merged
afterward - this is what guarantees the same real identifier maps to
the same placeholder across file boundaries, and avoids counter
collisions.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from agent.context_bundler import ContextBundler
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
    # Files added specifically by semantic retrieval, i.e. present in
    # bundled_files but NOT reachable via ContextBundler's import graph.
    # Tracked separately so a caller/benchmark can see retrieval's actual
    # marginal contribution, not just the final merged set. Empty list
    # (not None) whenever a retriever was configured but contributed
    # nothing new; None only when no retriever was configured at all.
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
        """
        retriever: optional object exposing .retrieve(query, top_k) ->
            List[Dict] with each dict containing at least
            {"metadata": {"filepath": ...}}. HybridRetriever already
            satisfies this interface directly. None (default) disables
            semantic retrieval entirely - ContextBundler-only behavior.
        retrieval_top_k: how many semantic hits to request per task.
            Only relevant when retriever is not None.
        """
        self.protected_identifiers = protected_identifiers or []
        self.protected_literals = protected_literals or []
        self.max_dependency_depth = max_dependency_depth
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k

    def _get_retrieval_additions(
        self, user_request: str, workspace_files: Dict[str, str], already_bundled: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Queries self.retriever (if configured) for files semantically
        relevant to user_request, and returns just the NEW ones -
        skeletonized, same treatment as ContextBundler's dependency
        files - that aren't already in already_bundled. Returns {} if no
        retriever is configured, the query returns nothing new, or the
        query fails for any reason (failure is logged and swallowed,
        never raised - retrieval is an enhancement, not a requirement).
        """
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
            if not filepath:
                continue
            if filepath in already_bundled or filepath in additions:
                continue
            if filepath not in workspace_files:
                # Index may be stale relative to the current workspace
                # (file renamed/deleted since last onboarding) - skip
                # rather than KeyError.
                continue
            additions[filepath] = skeletonize_code(workspace_files[filepath])

        return additions

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

        # 1. Deterministic import-graph scoping (always runs).
        bundler = ContextBundler(workspace_files, max_depth=self.max_dependency_depth)
        bundled_files = bundler.build_bundled_context(target_filepath, skeletonize_dependencies=True)

        # 2. Semantic scoping (only if a retriever is configured) -
        # unioned in, never replacing what ContextBundler already found.
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