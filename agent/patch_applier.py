"""
Agent Patch Applier: In-memory application and validation of structured CodePatch objects.
"""
import ast
import difflib
import re
from typing import List, Dict, Optional
from agent.diff_parser import CodePatch


class PatchApplyError(Exception):
    """Raised when a patch cannot be applied safely to target file content."""
    pass


def _normalize_for_comparison(text: str) -> str:
    """
    Normalizes text for FUZZY comparison only - never used to actually
    modify file content. Collapses quote style (' <-> ") and trailing
    whitespace per line, since these are the most common, systematic
    (not random) ways an LLM's SEARCH block fails to exact-match real
    file content, confirmed via real diagnostic evidence: models very
    commonly default to double-quoted strings regardless of what's
    actually in the file.
    """
    lines = text.splitlines()
    normalized_lines = [line.rstrip().replace('"', "'") for line in lines]
    return "\n".join(normalized_lines)


def _find_normalized_match(original_content: str, search_text: str) -> Optional[str]:
    """
    Attempts to find the EXACT original substring in original_content
    whose normalized form matches the normalized form of search_text.
    Returns the real, unmodified substring from the file (never the
    LLM's version) so the actual replace() always operates on genuine
    file content - normalization is used only to LOCATE the match, never
    to determine what gets written.
    """
    normalized_search = _normalize_for_comparison(search_text)
    file_lines = original_content.splitlines(keepends=True)
    search_line_count = len(search_text.splitlines())

    for i in range(len(file_lines) - search_line_count + 1):
        candidate_lines = file_lines[i:i + search_line_count]
        candidate = "".join(candidate_lines)
        candidate_stripped = candidate[:-1] if candidate.endswith("\n") else candidate
        if _normalize_for_comparison(candidate_stripped) == normalized_search:
            return candidate_stripped

    return None


def _find_closest_match_diagnostic(original_content: str, search_text: str) -> str:
    """
    When even a normalized match fails, find the closest actual region in
    the file and show a line-by-line diff against it, so the failure is
    self-diagnosing instead of a bare "not found".
    """
    file_lines = original_content.splitlines()
    search_lines = search_text.splitlines()
    if not search_lines or not file_lines:
        return "(unable to generate diagnostic - empty content)"

    window_size = len(search_lines)
    best_ratio = 0.0
    best_window = []
    for i in range(max(1, len(file_lines) - window_size + 1)):
        window = file_lines[i:i + window_size]
        ratio = difflib.SequenceMatcher(None, window, search_lines).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_window = window

    if not best_window:
        return "(no similar region found in file)"

    diff = difflib.unified_diff(
        best_window, search_lines,
        fromfile="closest_match_in_file", tofile="llm_search_block",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    return f"Closest match found ({best_ratio:.0%} similar):\n{diff_text}"


def apply_patch_to_content(original_content: str, patch: CodePatch) -> str:
    """
    Applies a single CodePatch to an in-memory source string safely.

    Match strategy, in order:
    1. Exact match (fast path, zero risk of misinterpretation).
    2. Normalized match (quote-style + trailing-whitespace tolerant) -
       if found, the REAL file text is used for the replace, never the
       LLM's version, so this cannot corrupt content, only relax how
       strictly the SEARCH block's surface form must match.
    3. If neither matches, fail with a diagnostic diff instead of a bare
       "not found" error.

    Validates match count (no ambiguity) and post-patch AST syntax.
    """
    search_text = patch.search_text
    replace_text = patch.replace_text

    match_count = original_content.count(search_text)

    if match_count == 0:
        normalized_match = _find_normalized_match(original_content, search_text)
        if normalized_match is not None:
            occurrence_count = original_content.count(normalized_match)
            if occurrence_count > 1:
                raise PatchApplyError(
                    f"Failed to apply patch to '{patch.filepath}': normalized SEARCH match is "
                    f"ambiguous (found {occurrence_count} occurrences after normalization)."
                )
            search_text = normalized_match
            match_count = 1
        else:
            diagnostic = _find_closest_match_diagnostic(original_content, patch.search_text)
            raise PatchApplyError(
                f"Failed to apply patch to '{patch.filepath}': SEARCH block not found in file content "
                f"(tried exact match and normalized quote/whitespace match).\n"
                f"{diagnostic}"
            )
    elif match_count > 1:
        raise PatchApplyError(
            f"Failed to apply patch to '{patch.filepath}': SEARCH block is ambiguous (found {match_count} occurrences)."
        )

    updated_content = original_content.replace(search_text, replace_text, 1)

    if patch.filepath.endswith(".py"):
        try:
            ast.parse(updated_content)
        except SyntaxError as e:
            raise PatchApplyError(
                f"Patch for '{patch.filepath}' introduced a syntax error: {str(e)}"
            )

    return updated_content


def apply_patches_staged(file_map: Dict[str, str], patches: List[CodePatch]) -> Dict[str, str]:
    """
    Applies multiple patches across a dictionary of {filepath: file_content}.

    Returns the complete updated file map. Callers must use the patch
    filepaths to determine which files were actually modified.
    """
    staged_files = file_map.copy()

    for patch in patches:
        if patch.filepath not in staged_files:
            raise PatchApplyError(
                f"Target file '{patch.filepath}' not found in provided workspace context."
            )

        current_content = staged_files[patch.filepath]
        staged_files[patch.filepath] = apply_patch_to_content(
            current_content,
            patch,
        )

    return staged_files