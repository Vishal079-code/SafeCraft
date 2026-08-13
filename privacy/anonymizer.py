"""
Privacy Anonymizer: Scope-aware AST Transformer for Identifiers
and Occurrence-Based Literal Protection.
"""

import ast
from typing import Dict, Any, List, Set, Tuple, Optional


class CodeAnonymizer(ast.NodeTransformer):
    def __init__(
        self,
        filepath: str,
        protected_identifiers: Optional[List[str]] = None,
        protected_literals: Optional[List[Any]] = None,
    ):
        self.filepath = filepath
        self.protected_identifiers: Set[str] = set(protected_identifiers or [])
        self.protected_literals: List[Any] = protected_literals or []

        # Local state maps
        self.identifier_map: Dict[str, str] = {}  # original_name -> placeholder
        self.reverse_identifier_map: Dict[str, str] = {}  # placeholder -> original_name

        # Occurrence-based literal mapping
        # occurrence_id ("filepath:line:col") -> metadata dict
        self.literal_map: Dict[str, Dict[str, Any]] = {}
        self.reverse_literal_map: Dict[str, Any] = {}  # placeholder -> original_value

        self._func_counter = 0
        self._var_counter = 0
        self._const_counter = 0
        self._attr_counter = 0

    def _get_next_func_placeholder(self) -> str:
        self._func_counter += 1
        return f"__ANON_FUNC_{self._func_counter}__"

    def _get_next_var_placeholder(self) -> str:
        self._var_counter += 1
        return f"__ANON_VAR_{self._var_counter}__"

    def _get_next_const_placeholder(self) -> str:
        self._const_counter += 1
        return f"__ANON_CONST_{self._const_counter}__"

    def _get_next_attr_placeholder(self) -> str:
        self._attr_counter += 1
        return f"__ANON_ATTR_{self._attr_counter}__"

    def _get_or_create_mapping(self, name: str, placeholder_fn) -> str:
        """
        Shared helper - ensures the SAME original name always maps to the
        SAME placeholder no matter where it's encountered (function
        parameter, body reference, attribute access, keyword argument,
        etc). This is what keeps anonymization consistent across a
        function's signature and its body - without it, a parameter could
        end up with one placeholder in the signature and a different one
        in the body, or worse, be left visible in one place while hidden
        in another.
        """
        if name not in self.identifier_map:
            placeholder = placeholder_fn()
            self.identifier_map[name] = placeholder
            self.reverse_identifier_map[placeholder] = name
        return self.identifier_map[name]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Anonymize the function name itself
        if node.name in self.protected_identifiers:
            node.name = self._get_or_create_mapping(node.name, self._get_next_func_placeholder)

        # Anonymize function PARAMETERS. These are ast.arg nodes, NOT
        # ast.Name nodes - visit_Name never sees them. Without this fix,
        # a protected parameter name stayed fully visible in the
        # signature while every USE of it in the body got replaced with
        # a placeholder - which is worse than not anonymizing at all,
        # since the correlation between the visible name and the
        # placeholder becomes trivially obvious to anyone reading it.
        all_args = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        if node.args.vararg:
            all_args.append(node.args.vararg)
        if node.args.kwarg:
            all_args.append(node.args.kwarg)

        for arg in all_args:
            if arg.arg in self.protected_identifiers:
                arg.arg = self._get_or_create_mapping(arg.arg, self._get_next_var_placeholder)

        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Anonymize variable references (both read and write/assignment)
        if node.id in self.protected_identifiers:
            node.id = self._get_or_create_mapping(node.id, self._get_next_var_placeholder)
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # Anonymize attribute access, e.g. `employee.grade` -> `employee.__ANON_ATTR_1__`.
        # The attribute name is stored as a plain string on
        # ast.Attribute.attr, NOT as an ast.Name node, so visit_Name
        # never fires on it - this was previously unhandled entirely.
        if node.attr in self.protected_identifiers:
            node.attr = self._get_or_create_mapping(node.attr, self._get_next_attr_placeholder)
        self.generic_visit(node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        # Anonymize keyword arguments in calls, e.g. `foo(salary=x)`.
        # Same gap as function parameters: node.arg is a plain string.
        # node.arg is None for **kwargs unpacking (e.g. foo(**data)) -
        # guard against that so we don't try to anonymize `None`.
        if node.arg and node.arg in self.protected_identifiers:
            node.arg = self._get_or_create_mapping(node.arg, self._get_next_var_placeholder)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # Check if value matches protected literals policy
        if node.value in self.protected_literals:
            lineno = getattr(node, "lineno", 0)
            col_offset = getattr(node, "col_offset", 0)
            occurrence_id = f"{self.filepath}:{lineno}:{col_offset}"

            placeholder = self._get_next_const_placeholder()

            # Track occurrence location uniquely - each OCCURRENCE gets
            # its own placeholder, even if the value is identical to
            # another occurrence elsewhere. This avoids ambiguously
            # merging two unrelated uses of the same number/string into
            # one placeholder.
            self.literal_map[occurrence_id] = {
                "occurrence_id": occurrence_id,
                "placeholder": placeholder,
                "original_value": node.value,
                "value_type": type(node.value).__name__,
            }
            self.reverse_literal_map[placeholder] = node.value

            # Replace AST constant with placeholder string token
            return ast.Name(id=placeholder, ctx=ast.Load())

        return node


def anonymize_code(
    code_snippet: str,
    filepath: str,
    protected_identifiers: Optional[List[str]] = None,
    protected_literals: Optional[List[Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Parses code snippet, transforms identifiers and occurrences of protected literals,
    and returns anonymized source code alongside local privacy maps.
    """
    try:
        parsed_ast = ast.parse(code_snippet)
    except SyntaxError as e:
        return code_snippet, {"error": f"Syntax error during parsing: {str(e)}"}

    transformer = CodeAnonymizer(
        filepath=filepath,
        protected_identifiers=protected_identifiers,
        protected_literals=protected_literals,
    )
    transformed_ast = transformer.visit(parsed_ast)
    ast.fix_missing_locations(transformed_ast)

    anonymized_code = ast.unparse(transformed_ast)

    privacy_maps = {
        "identifier_map": transformer.identifier_map,
        "reverse_identifier_map": transformer.reverse_identifier_map,
        "literal_map": transformer.literal_map,
        "reverse_literal_map": transformer.reverse_literal_map,
    }

    return anonymized_code, privacy_maps