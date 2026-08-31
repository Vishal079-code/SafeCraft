"""Agent Context Bundler: Resolves local Python imports and bundles multi-file
workspace context. Primary target file (being edited) is sent full-source;
dependency files are skeletonized (signatures + docstrings only, bodies
stripped) to reduce both token cost and privacy exposure. Final output is
passed through ContextAssembler (Day 8) to enforce a hard token ceiling.

Dependency resolution is RECURSIVE up to max_depth hops via breadth-first
frontier expansion - a file's imports, then its imports' imports, and so
on - not just the target file's direct imports.
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
    def __init__(self, workspace_files: Dict[str, str], max_depth: int = 2, token_budget: int = 50000):
        self.workspace_files = workspace_files
        self.max_depth = max_depth
        self.token_budget = token_budget
        self.assembler = ContextAssembler(max_tokens=token_budget)

    def build_bundled_context(
        self, target_filepath: str, skeletonize_dependencies: bool = True
    ) -> Dict[str, str]:
        """Bundles the target file (FULL source) alongside its resolved local
        import dependencies, resolved RECURSIVELY up to self.max_depth hops
        via breadth-first frontier expansion (dependencies of the target,
        then dependencies of those dependencies, etc). Dependencies are
        skeletonized by default. Returns raw {filepath: content} dict.
        """
        if target_filepath not in self.workspace_files:
            return {}

        bundled = {target_filepath: self.workspace_files[target_filepath]}
        frontier = [target_filepath]

        for _ in range(self.max_depth):
            next_frontier = []
            for filepath in frontier:
                # Always resolve imports against the file's FULL original
                # source, never against a skeletonized version already in
                # `bundled` - a skeletonized body has its imports intact
                # (only function bodies are stripped) but resolving from
                # workspace_files keeps this correct regardless.
                code = self.workspace_files.get(filepath, "")
                imports = extract_local_imports(code, current_filepath=filepath)

                for module in imports:
                    dep_path = resolve_import_to_filepath(module, self.workspace_files)
                    if dep_path and dep_path not in bundled:
                        dep_content = self.workspace_files[dep_path]
                        bundled[dep_path] = (
                            skeletonize_code(dep_content) if skeletonize_dependencies else dep_content
                        )
                        next_frontier.append(dep_path)

            if not next_frontier:
                # Nothing new resolved this hop - stop early rather than
                # looping through empty frontiers for the remaining depth.
                break
            frontier = next_frontier

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

        units = [
            {"id": filepath, "text": content, "metadata": {"filepath": filepath}}
            for filepath, content in bundled.items()
        ]
        return self.assembler.assemble_context(units, query=user_request)