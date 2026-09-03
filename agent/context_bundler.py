"""Agent Context Bundler: Resolves local Python imports and bundles multi-file
workspace context.

V2 changes (SafeCraft V2, Phase 1):
- Target-scope extraction: when target_symbol is given, the target file
  is sent with ONLY that function/method's body intact - every other
  function in the same file is skeletonized (signature + docstring
  only). Class structure and imports are always preserved. This is a
  prompt-construction-only transformation - see splice_function_into_original()
  below for the corresponding, mandatory write-back safety mechanism.
- Token budget is now actually ENFORCED here (not just estimated and
  reported): if the bundled set exceeds token_budget, dependency files
  are dropped starting from the ones added LAST (i.e. the deepest / most
  indirect hops first), never the target file, until under budget.

Dependency resolution is RECURSIVE up to max_depth hops via breadth-first
frontier expansion - a file's imports, then its imports' imports, and so
on - not just the target file's direct imports.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.skeletonizer import skeletonize_code, skeletonize_except

logger = logging.getLogger(__name__)


def extract_local_imports(code_content: str, current_filepath: str = "") -> List[str]:
    imported_modules = []
    try:
        tree = ast.parse(code_content)
    except SyntaxError:
        return imported_modules

    current_dir_parts = Path(current_filepath).parent.parts if current_filepath else ()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level == 0:
                base_module = node.module
            else:
                base_parts = current_dir_parts[: max(0, len(current_dir_parts) - (level - 1))]
                prefix = ".".join(base_parts)
                if node.module:
                    base_module = f"{prefix}.{node.module}" if prefix else node.module
                else:
                    base_module = prefix

            if not base_module:
                continue

            imported_modules.append(base_module)
            for alias in node.names:
                if alias.name != "*":
                    imported_modules.append(f"{base_module}.{alias.name}")

    return imported_modules


def resolve_import_to_filepath(module_name: str, workspace_files: Dict[str, str]) -> Optional[str]:
    if not module_name:
        return None

    path_prefix = module_name.replace(".", "/")
    possible_paths = [f"{path_prefix}.py", f"{path_prefix}/__init__.py"]

    for path in possible_paths:
        if path in workspace_files:
            return path

    parts = module_name.split(".")
    while len(parts) > 1:
        parts.pop()
        submodule = "/".join(parts)
        for path in [f"{submodule}.py", f"{submodule}/__init__.py"]:
            if path in workspace_files:
                return path

    return None


def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free estimate (~4 chars/token). Good enough for
    budget enforcement; exact tokenization isn't needed here, only
    relative sizing to decide what to drop."""
    return max(1, len(text) // 4)


def _find_function_node(tree: ast.Module, target_symbol: str) -> Optional[Tuple[ast.AST, Optional[str]]]:
    """Returns (function_node, enclosing_class_name_or_None) for the
    first function/method matching target_symbol (bare name OR
    'ClassName.method'), or None if not found."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = f"{node.name}.{child.name}"
                    if child.name == target_symbol or qualified == target_symbol:
                        return child, node.name
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_symbol:
            return node, None
    return None


def splice_function_into_original(
    original_content: str, target_symbol: str, patched_content_with_target: str
) -> str:
    """
    The mandatory counterpart to skeletonize_except(). Given:
      - original_content: the REAL, full, untouched original file
      - target_symbol: which function/method was edited
      - patched_content_with_target: the file as it looked AFTER the LLM's
        patch was applied against the (partially skeletonized) version
        that was actually sent

    Locates target_symbol's new source in patched_content_with_target,
    and splices ONLY that function's text into the ORIGINAL file at the
    original function's exact line range - via direct line-based text
    replacement, NOT via ast.unparse() on the whole file. This is
    deliberate: ast.unparse() on the full file would reformat every
    other function's quote style and layout (see bug #5 in the project
    README) even though nothing else was touched. A line-range text
    splice leaves every byte outside the target function exactly as it
    was.

    Raises ValueError if target_symbol can't be located in EITHER file -
    this must never silently fall back to something that could discard
    real code, so callers should treat a raised error as "do not write
    this to disk, something is wrong" rather than catching and proceeding.
    """
    try:
        original_tree = ast.parse(original_content)
    except SyntaxError as e:
        raise ValueError(f"Original file no longer parses: {e}")

    original_lines = original_content.splitlines(keepends=True)
    found = _find_function_node(original_tree, target_symbol)
    if found is None:
        raise ValueError(f"Target symbol '{target_symbol}' not found in original file - refusing to splice.")
    orig_node, _ = found

    try:
        patched_tree = ast.parse(patched_content_with_target)
    except SyntaxError as e:
        raise ValueError(f"Patched content is not valid Python: {e}")

    patched_lines = patched_content_with_target.splitlines(keepends=True)
    patched_found = _find_function_node(patched_tree, target_symbol)
    if patched_found is None:
        raise ValueError(f"Target symbol '{target_symbol}' not found in patched content - refusing to splice.")
    patched_node, _ = patched_found

    new_func_start = patched_node.lineno - 1
    # include leading decorators, if any, since they're part of "the function"
    if patched_node.decorator_list:
        new_func_start = patched_node.decorator_list[0].lineno - 1
    new_func_end = patched_node.end_lineno
    new_func_text = "".join(patched_lines[new_func_start:new_func_end])

    orig_start = orig_node.lineno - 1
    if orig_node.decorator_list:
        orig_start = orig_node.decorator_list[0].lineno - 1
    orig_end = orig_node.end_lineno

    spliced_lines = original_lines[:orig_start] + [new_func_text] + original_lines[orig_end:]
    return "".join(spliced_lines)


class ContextBundler:
    def __init__(self, workspace_files: Dict[str, str], max_depth: int = 2, token_budget: int = 50000):
        self.workspace_files = workspace_files
        self.max_depth = max_depth
        self.token_budget = token_budget

    def build_bundled_context(
        self,
        target_filepath: str,
        target_symbol: Optional[str] = None,
        skeletonize_dependencies: bool = True,
    ) -> Dict[str, str]:
        """
        target_symbol=None (the default): unchanged prior behavior -
        target file sent in full.
        target_symbol="function_name" or "Class.method": target file is
        sent with ONLY that function/method's real body; everything else
        in the file is skeletonized (signature + docstring only).

        Dependencies (from other files) are always fully skeletonized,
        same as before - they're context, never the thing being edited.

        Enforces token_budget: if the total exceeds it, dependency files
        are dropped starting from the ones discovered LAST (deepest/most
        indirect), until under budget. The target file is NEVER dropped.
        """
        if target_filepath not in self.workspace_files:
            return {}

        target_content = self.workspace_files[target_filepath]
        if target_symbol is not None:
            target_content = skeletonize_except(target_content, target_symbol)

        bundled = {target_filepath: target_content}
        discovery_order = [target_filepath]  # tracks order added, for budget-drop priority
        frontier = [target_filepath]

        for _ in range(self.max_depth):
            next_frontier = []
            for filepath in frontier:
                code = self.workspace_files.get(filepath, "")
                imports = extract_local_imports(code, current_filepath=filepath)

                for module in imports:
                    dep_path = resolve_import_to_filepath(module, self.workspace_files)
                    if dep_path and dep_path not in bundled:
                        dep_content = self.workspace_files[dep_path]
                        bundled[dep_path] = (
                            skeletonize_code(dep_content) if skeletonize_dependencies else dep_content
                        )
                        discovery_order.append(dep_path)
                        next_frontier.append(dep_path)

            if not next_frontier:
                break
            frontier = next_frontier

        total_tokens = sum(_estimate_tokens(c) for c in bundled.values())
        if total_tokens > self.token_budget:
            dropped = []
            # drop from the END of discovery_order (deepest/last-found)
            # backward, but NEVER the target file itself (index 0).
            for filepath in reversed(discovery_order[1:]):
                if total_tokens <= self.token_budget:
                    break
                total_tokens -= _estimate_tokens(bundled[filepath])
                del bundled[filepath]
                dropped.append(filepath)
            if dropped:
                logger.warning(
                    f"Token budget ({self.token_budget}) exceeded - dropped "
                    f"{len(dropped)} lowest-priority dependency file(s): {dropped}"
                )

        return bundled