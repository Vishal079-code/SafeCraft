"""
Skeletonizer: reduces function/method bodies to just their docstring +
an Ellipsis, for sending as low-cost/low-exposure structural context.

skeletonize_code() collapses EVERY function/method - used for whole
dependency files, which are context only, never edited directly.

skeletonize_except() collapses every function/method EXCEPT one named
target - used for the TARGET file, so the model sees the real
implementation of the function actually being edited, plus the real
signatures/docstrings of everything else in the file (for context: is
this a method that needs to call a sibling? what's this class's other
structure?), without paying the token/exposure cost of every other
function's full body.
"""
import ast
from typing import Optional


def skeletonize_code(code_str: str) -> str:
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return code_str

    class _FullSkeletonizer(ast.NodeTransformer):
        def _skeletonize(self, node):
            docstring = ast.get_docstring(node)
            new_body = []
            if docstring:
                new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
            new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
            node.body = new_body
            return node

        def visit_FunctionDef(self, node):
            return self._skeletonize(node)

        def visit_AsyncFunctionDef(self, node):
            return self._skeletonize(node)

    try:
        transformed = _FullSkeletonizer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return code_str


def skeletonize_except(code_str: str, target_symbol: Optional[str]) -> str:
    """
    target_symbol may be a bare function name ("apply_discount") or a
    qualified "ClassName.method_name" for a method - qualification is
    only used to disambiguate two methods with the same name in
    different classes; a bare name matches regardless of which class
    (or no class) it's in.

    target_symbol=None returns code_str completely unchanged - this is
    the safe default, preserving exact prior behavior (whole file, no
    skeletonization) for any caller that doesn't opt into target-scope
    extraction.

    IMPORTANT: this function's OUTPUT is for building the PROMPT only.
    It must never be used to construct what gets written back to disk -
    see splice_function_into_original() in context_bundler.py, which is
    the only safe way to merge an edit made against this skeletonized
    view back into the real, full original file.
    """
    if target_symbol is None:
        return code_str
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return code_str

    class _SelectiveSkeletonizer(ast.NodeTransformer):
        def __init__(self):
            self.current_class = None

        def visit_ClassDef(self, node):
            prev = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = prev
            return node

        def _maybe_skeletonize(self, node):
            qualified = f"{self.current_class}.{node.name}" if self.current_class else node.name
            if node.name == target_symbol or qualified == target_symbol:
                return node  # keep the target's real body, untouched
            docstring = ast.get_docstring(node)
            new_body = []
            if docstring:
                new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
            new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
            node.body = new_body
            return node

        def visit_FunctionDef(self, node):
            return self._maybe_skeletonize(node)

        def visit_AsyncFunctionDef(self, node):
            return self._maybe_skeletonize(node)

    try:
        transformed = _SelectiveSkeletonizer().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return code_str