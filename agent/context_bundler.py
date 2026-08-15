"""Agent Context Bundler: Resolves local Python imports and bundles multi-file

workspace context. Primary target file (being edited) is sent full-source;
dependency files are skeletonized (signatures + docstrings only, bodies
stripped) to reduce both token cost and privacy exposure. Final output is
passed through ContextAssembler (Day 8) to enforce a hard token ceiling.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional

from agent.skeletonizer import skeletonize_code
from retriever.context_assembler import ContextAssembler


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


class ContextBundler:
    def __init__(self, workspace_files: Dict[str, str], token_budget: int = 50000):
        self.workspace_files = workspace_files
        self.token_budget = token_budget
        self.assembler = ContextAssembler(max_tokens=token_budget)

    def build_bundled_context(
        self, target_filepath: str, skeletonize_dependencies: bool = True
    ) -> Dict[str, str]:
        """Bundles the target file (FULL source) alongside its resolved local

        import dependencies (skeletonized by default). Returns raw {filepath: content} dict.
        """
        if target_filepath not in self.workspace_files:
            return {}

        bundled = {target_filepath: self.workspace_files[target_filepath]}
        target_code = self.workspace_files[target_filepath]
        imports = extract_local_imports(target_code, current_filepath=target_filepath)

        for module in imports:
            dep_path = resolve_import_to_filepath(module, self.workspace_files)
            if dep_path and dep_path not in bundled:
                dep_content = self.workspace_files[dep_path]
                if skeletonize_dependencies:
                    bundled[dep_path] = skeletonize_code(dep_content)
                else:
                    bundled[dep_path] = dep_content

        return bundled

    def build_bundled_prompt(
        self, target_filepath: str, user_request: str = "", skeletonize_dependencies: bool = True
    ) -> str:
        """Bundles the context and passes it through ContextAssembler to enforce

        self.token_budget. The primary target file is prioritized first.
        """
        bundled = self.build_bundled_context(
            target_filepath, skeletonize_dependencies=skeletonize_dependencies
        )
        if not bundled:
            return "No relevant context found."

        # Adapt {filepath: content} into units expected by ContextAssembler
        units = [
            {"id": filepath, "text": content, "metadata": {"filepath": filepath}}
            for filepath, content in bundled.items()
        ]
        return self.assembler.assemble_context(units, query=user_request)