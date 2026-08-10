"""
Day 4 — Call Graph Construction (v2: relative + submodule import resolution)

Reads repo_map.json (from Day 3) and builds a directed dependency graph:
an edge A -> B means "file A imports something from file B".

This graph is what lets Day 7's retrieval step expand beyond just the
files a semantic search finds, to also include their direct dependencies.

v2 changes: handles relative imports (from .sibling import x,
from ..pkg import x) and 'from package import submodule' style imports,
which v1 could not resolve correctly.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

import networkx as nx

logger = logging.getLogger(__name__)


def resolve_import_to_file(
    current_file: str,
    module_name: Optional[str],
    imported_names: List[str],
    level: int,
    known_files: Set[str],
) -> List[str]:
    """
    Resolves both absolute and relative imports, including
    'from package import submodule' style. Returns a list because a
    single import statement can resolve to MORE than one real
    dependency — e.g. 'from models import user' depends on both
    models/__init__.py (which runs as a side effect) AND models/user.py.
    """
    resolved = []

    # 1. Resolve relative import base path (level > 0 means relative,
    #    e.g. "from . import x" is level=1, "from .. import x" is level=2)
    if level > 0:
        current_parts = list(Path(current_file).parent.parts)
        trim_count = level - 1
        if trim_count <= len(current_parts):
            base_parts = (
                current_parts[: len(current_parts) - trim_count]
                if trim_count > 0
                else current_parts
            )
            target_dot = ".".join(base_parts + ([module_name] if module_name else []))
        else:
            return []  # would go above repo root -> can't resolve, skip safely
    else:
        target_dot = module_name or ""

    if not target_dot and not imported_names:
        return []

    # 2. Match exact module file or package __init__
    path_str = target_dot.replace(".", "/")
    if f"{path_str}.py" in known_files:
        resolved.append(f"{path_str}.py")
    elif f"{path_str}/__init__.py" in known_files:
        resolved.append(f"{path_str}/__init__.py")

    # 3. Match 'from package import submodule' (e.g. from models import user)
    for name in imported_names:
        submodule_path = f"{path_str}/{name}.py" if path_str else f"{name}.py"
        if submodule_path in known_files:
            resolved.append(submodule_path)

    return resolved


def build_dependency_graph(repo_map: Dict[str, Any]) -> nx.DiGraph:
    """
    Build a directed graph from a repo_map (Day 3 output).
    One node per file; edges represent "imports from / depends on".
    """
    if "modules" not in repo_map:
        raise ValueError(
            "repo_map is missing the 'modules' key — expected output from "
            "RepoIndexer.index_repository(). Got keys: " + str(list(repo_map.keys()))
        )

    graph = nx.DiGraph()
    known_files = set(repo_map["modules"].keys())

    # Add every file as a node first, even if it has no edges yet —
    # otherwise isolated files (like utils/validators.py) wouldn't
    # appear in the graph at all.
    for filepath in known_files:
        graph.add_node(filepath)

    unresolved_imports = []  # for visibility/debugging, not an error

    for filepath, file_data in repo_map["modules"].items():
        for imp in file_data.get("imports", []):
            module_name = imp.get("module")
            imported_names = imp.get("imported_names", [])
            level = imp.get("level", 0)  # "standard" imports have no "level" key -> default 0

            resolved_list = resolve_import_to_file(
                filepath, module_name, imported_names, level, known_files
            )

            if resolved_list:
                for resolved in resolved_list:
                    if resolved != filepath:
                        graph.add_edge(filepath, resolved)
            elif module_name:
                unresolved_imports.append((filepath, module_name))

    logger.info(f"Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    logger.info(f"{len(unresolved_imports)} imports were external (not part of this repo)")

    return graph


def get_dependencies(graph: nx.DiGraph, filepath: str) -> List[str]:
    """Files that `filepath` directly depends on (what it imports)."""
    if filepath not in graph:
        return []
    return list(graph.successors(filepath))


def get_dependents(graph: nx.DiGraph, filepath: str) -> List[str]:
    """Files that directly depend on `filepath` (what imports it)."""
    if filepath not in graph:
        return []
    return list(graph.predecessors(filepath))


def expand_with_dependencies(graph: nx.DiGraph, filepaths: List[str], hops: int = 1) -> Set[str]:
    """
    Given a starting set of files (e.g. from Day 6/7's semantic search),
    expand it to include files within `hops` steps in the dependency
    graph. This is the function Day 7's retrieval pipeline will call
    to pull in structurally-related files a pure text search would miss.
    """
    expanded = set(filepaths)
    frontier = set(filepaths)

    for _ in range(hops):
        next_frontier = set()
        for f in frontier:
            if f in graph:
                next_frontier.update(graph.successors(f))
                next_frontier.update(graph.predecessors(f))
        next_frontier -= expanded
        expanded.update(next_frontier)
        frontier = next_frontier

    return expanded


def load_repo_map(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    repo_map = load_repo_map("sample_repo/repo_map.json")
    graph = build_dependency_graph(repo_map)

    print("\n--- Graph edges (A -> B means A imports from B) ---")
    for source, target in graph.edges():
        print(f"  {source}  ->  {target}")

    print("\n--- Dependency check: routes/api.py ---")
    print("Depends on:", get_dependencies(graph, "routes/api.py"))
    print("Depended on by:", get_dependents(graph, "routes/api.py"))

    print("\n--- Isolated file check: utils/validators.py ---")
    print("Depends on:", get_dependencies(graph, "utils/validators.py"))
    print("Depended on by:", get_dependents(graph, "utils/validators.py"))

    print("\n--- Expansion test: starting from models/user.py, 1 hop ---")
    print(expand_with_dependencies(graph, ["models/user.py"], hops=1))