"""
Agent Context Bundler: Resolves local Python imports and bundles multi-file 
workspace context.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional


def extract_local_imports(code_content: str, current_filepath: str = "") -> List[str]:
    """
    Parses Python code AST to extract imported module strings.
    Handles relative imports (e.g. from .models import User) by resolving relative paths.
    """
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
                if node.module:
                    imported_modules.append(node.module)
            else:
                # Relative import handling (e.g. level=1 is current directory)
                base_parts = current_dir_parts[: max(0, len(current_dir_parts) - (level - 1))]
                prefix = ".".join(base_parts)
                if node.module:
                    full_module = f"{prefix}.{node.module}" if prefix else node.module
                else:
                    full_module = prefix
                if full_module:
                    imported_modules.append(full_module)

    return imported_modules


def resolve_import_to_filepath(module_name: str, workspace_files: Dict[str, str]) -> Optional[str]:
    """
    Attempts to map an imported module (e.g., 'auth.models') to a workspace filepath 
    (e.g., 'auth/models.py' or 'auth/models/__init__.py').
    """
    if not module_name:
        return None

    path_prefix = module_name.replace(".", "/")
    possible_paths = [
        f"{path_prefix}.py",
        f"{path_prefix}/__init__.py",
    ]

    # Exact match check
    for path in possible_paths:
        if path in workspace_files:
            return path

    # Submodule fallback check (e.g., 'auth.models.User')
    parts = module_name.split(".")
    while len(parts) > 1:
        parts.pop()
        submodule = "/".join(parts)
        for path in [f"{submodule}.py", f"{submodule}/__init__.py"]:
            if path in workspace_files:
                return path

    return None


class ContextBundler:
    def __init__(self, workspace_files: Dict[str, str], token_budget: int = 50000):
        self.workspace_files = workspace_files
        self.token_budget = token_budget

    def build_bundled_context(self, target_filepath: str) -> Dict[str, str]:
        """
        Bundles the target file alongside its resolved local import dependencies.
        Returns a dictionary of {filepath: file_content} included in the context.
        """
        if target_filepath not in self.workspace_files:
            return {}

        bundled = {target_filepath: self.workspace_files[target_filepath]}
        target_code = self.workspace_files[target_filepath]

        # Extract dependencies considering current file location
        imports = extract_local_imports(target_code, current_filepath=target_filepath)

        for module in imports:
            dep_path = resolve_import_to_filepath(module, self.workspace_files)
            if dep_path and dep_path not in bundled:
                bundled[dep_path] = self.workspace_files[dep_path]

        return bundled