"""
Agent Patch Applier: In-memory application and validation of structured CodePatch objects.
"""

import ast
from typing import List
from agent.diff_parser import CodePatch


class PatchApplyError(Exception):
    """Raised when a patch cannot be applied safely to target file content."""
    pass


def apply_patch_to_content(original_content: str, patch: CodePatch) -> str:
    """
    Applies a single CodePatch to an in-memory source string safely.
    Validates exact match count and post-patch AST syntax.
    """
    search_text = patch.search_text
    replace_text = patch.replace_text

    match_count = original_content.count(search_text)

    if match_count == 0:
        raise PatchApplyError(
            f"Failed to apply patch to '{patch.filepath}': SEARCH block not found in file content."
        )
    if match_count > 1:
        raise PatchApplyError(
            f"Failed to apply patch to '{patch.filepath}': SEARCH block is ambiguous (found {match_count} occurrences)."
        )

    # Perform the in-memory replacement
    updated_content = original_content.replace(search_text, replace_text, 1)

    # Validate post-patch AST syntax if editing Python code
    if patch.filepath.endswith(".py"):
        try:
            ast.parse(updated_content)
        except SyntaxError as e:
            raise PatchApplyError(
                f"Patch for '{patch.filepath}' introduced a syntax error: {str(e)}"
            )

    return updated_content


def apply_patches_staged(
    file_map: dict[str, str], patches: List[CodePatch]
) -> dict[str, str]:
    """
    Applies multiple patches across a dictionary of {filepath: file_content}.
    Returns a copy of the updated file map if all patches apply cleanly.
    """
    staged_files = file_map.copy()

    for patch in patches:
        if patch.filepath not in staged_files:
            raise PatchApplyError(
                f"Target file '{patch.filepath}' not found in provided workspace context."
            )

        current_content = staged_files[patch.filepath]
        staged_files[patch.filepath] = apply_patch_to_content(current_content, patch)

    return staged_files