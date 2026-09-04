"""
Parses a .py file and extracts top-level functions, classes (with base classes),
methods, import statements, docstrings, and rich argument metadata.
"""
import ast
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup logger to prevent stdout pollution in downstream pipeline stages
logger = logging.getLogger(__name__)


def _unparse_node(node: Optional[ast.AST]) -> Optional[str]:
    """Helper to convert any AST expression (generics, defaults, annotations) to string."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def extract_argument_info(args_node: ast.arguments) -> List[Dict[str, Any]]:
    """
    Extracts structured argument metadata preserving positional vs kwarg kinds,
    default values, and type annotations.
    """
    structured_args: List[Dict[str, Any]] = []

    # 1. Positional & Positional-Only Arguments
    positional_nodes = args_node.posonlyargs + args_node.args
    num_positionals = len(positional_nodes)
    num_defaults = len(args_node.defaults)
    default_offset = num_positionals - num_defaults

    for i, arg in enumerate(positional_nodes):
        has_default = i >= default_offset
        default_val = _unparse_node(args_node.defaults[i - default_offset]) if has_default else None
        
        kind = "positional_only" if i < len(args_node.posonlyargs) else "positional_or_keyword"
        structured_args.append({
            "name": arg.arg,
            "kind": kind,
            "has_default": has_default,
            "default": default_val,
            "annotation": _unparse_node(arg.annotation)
        })

    # 2. Variable Positional (*args)
    if args_node.vararg:
        structured_args.append({
            "name": args_node.vararg.arg,
            "kind": "vararg",
            "has_default": False,
            "default": None,
            "annotation": _unparse_node(args_node.vararg.annotation)
        })

    # 3. Keyword-Only Arguments
    for arg, default_node in zip(args_node.kwonlyargs, args_node.kw_defaults):
        has_default = default_node is not None
        default_val = _unparse_node(default_node) if has_default else None
        
        structured_args.append({
            "name": arg.arg,
            "kind": "keyword_only",
            "has_default": has_default,
            "default": default_val,
            "annotation": _unparse_node(arg.annotation)
        })

    # 4. Variable Keyword (**kwargs)
    if args_node.kwarg:
        structured_args.append({
            "name": args_node.kwarg.arg,
            "kind": "kwarg",
            "has_default": False,
            "default": None,
            "annotation": _unparse_node(args_node.kwarg.annotation)
        })

    return structured_args



def extract_imports(tree: ast.AST) -> List[Dict[str, Any]]:
    """Collect import statements including module names, aliases, and relative levels."""
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "standard",
                    "module": alias.name,
                    "alias": alias.asname,
                    "imported_names": []
                })
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            aliases = {alias.name: alias.asname for alias in node.names if alias.asname}
            imports.append({
                "type": "from",
                "module": node.module or "",
                "level": node.level,  # 0 = absolute, 1 = '.', 2 = '..'
                "imported_names": names,
                "aliases": aliases
            })

    return imports


def extract_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Pull out function metadata including structured arguments and decorators.

    source_lines (V2): the FULL file's source, split into lines. When
    provided, captures the function's first ~8 real source lines as
    source_snippet - used as embedding text for functions with NO
    docstring, since "Function foo(a, b) in filepath" alone carries
    almost no semantic signal for retrieval. Capped at 8 lines to keep
    embedding text short and avoid biasing toward long functions.
    """
    decorators = [_unparse_node(d) for d in node.decorator_list if _unparse_node(d) is not None]

    source_snippet = None
    if source_lines:
        start = node.lineno - 1
        end = min(node.end_lineno, start + 8)
        source_snippet = "\n".join(source_lines[start:end])

    return {
        "name": node.name,
        "args": extract_argument_info(node.args),
        "return_annotation": _unparse_node(node.returns),
        "docstring": ast.get_docstring(node),
        "source_snippet": source_snippet,
        "lineno": node.lineno,
        "end_lineno": getattr(node, "end_lineno", node.lineno),
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "is_method": False,
        "decorators": decorators,
    }


def extract_class_info(node: ast.ClassDef, source_lines: Optional[List[str]] = None) -> Dict[str, Any]:
    """Pull out class metadata, base classes (handling generics), and methods."""
    methods = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_info = extract_function_info(item, source_lines=source_lines)
            method_info["is_method"] = True
            methods.append(method_info)

    bases = [_unparse_node(base) for base in node.bases if _unparse_node(base) is not None]

    return {
        "name": node.name,
        "docstring": ast.get_docstring(node),
        "bases": bases,
        "methods": methods,
        "lineno": node.lineno,
        "end_lineno": getattr(node, "end_lineno", node.lineno),
    }


def extract_file(filepath: str | Path) -> Optional[Dict[str, Any]]:
    """Parse a single Python file into structured metadata safely."""
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None

    source_lines = source.splitlines()
    functions = []
    classes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(extract_function_info(node, source_lines=source_lines))
        elif isinstance(node, ast.ClassDef):
            classes.append(extract_class_info(node, source_lines=source_lines))

    return {
        "filepath": str(filepath),
        "module_docstring": ast.get_docstring(tree),
        "imports": extract_imports(tree),
        "functions": functions,
        "classes": classes,
    }

if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    test_path = "sample_repo/models/user.py"
    result = extract_file(test_path)
    
    if result:
        print(json.dumps(result, indent=2))