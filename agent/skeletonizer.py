import ast


class ASTSkeletonTransformer(ast.NodeTransformer):
    """AST transformer that strips function and method bodies while preserving

    signatures, class structures, return annotations, and docstrings.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._skeletonize_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._skeletonize_function(node)

    def _skeletonize_function(self, node: ast.AST) -> ast.AST:
        docstring = ast.get_docstring(node)
        new_body = []

        if docstring:
            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))

        # Replace function implementation with Ellipsis (...)
        new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
        node.body = new_body
        return node


def skeletonize_code(code_str: str) -> str:
    """Parses source code into an AST and returns a signature-only skeleton string.

    Falls back to the raw string if parsing fails (e.g., non-Python syntax).
    """
    try:
        tree = ast.parse(code_str)
        transformer = ASTSkeletonTransformer()
        skeleton_tree = transformer.visit(tree)
        ast.fix_missing_locations(skeleton_tree)
        return ast.unparse(skeleton_tree)
    except Exception:
        return code_str