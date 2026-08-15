"""
Agent Context Bundler: Resolves local Python imports and bundles multi-file
workspace context.
"""
import ast
from pathlib import Path
from typing import Dict, List, Optional


def extract_local_imports(code_content: str, current_filepath: str = "") -> List[str]:
    """
    Parses Python code AST to extract imported module strings, resolved to
    full dotted paths (relative imports are expanded using current_filepath).

    Also expands 'from package import submodule' style imports into
    'package.submodule' - without this, `from auth import models` only
    ever yields "auth" (the module), silently dropping which submodule
    was actually imported, which made it unresolvable to auth/models.py.
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

            # FIX: also add "<base_module>.<imported_name>" candidates -
            # covers 'from auth import models' (base_module="auth",
            # imported name="models") -> "auth.models", which
            # resolve_import_to_filepath can then correctly map to
            # auth/models.py. The plain base_module is still added too,
            # since some imported names are attributes/classes (not
            # files) and only the package's __init__.py should resolve
            # for those - resolve_import_to_filepath already tries both
            # forms and only keeps whichever actually exists on disk.
            imported_modules.append(base_module)
            for alias in node.names:
                if alias.name != "*":
                    imported_modules.append(f"{base_module}.{alias.name}")

    return imported_modules


def resolve_import_to_filepath(module_name: str, workspace_files: Dict[str, str]) -> Optional[str]:
    """
    Attempts to map an imported module (e.g., 'auth.models') to a workspace filepath
    (e.g., 'auth/models.py' or 'auth/models/__init__.py').
    """
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
        imports = extract_local_imports(target_code, current_filepath=target_filepath)

        for module in imports:
            dep_path = resolve_import_to_filepath(module, self.workspace_files)
            if dep_path and dep_path not in bundled:
                bundled[dep_path] = self.workspace_files[dep_path]

        return bundled