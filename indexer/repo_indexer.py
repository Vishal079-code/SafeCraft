"""
Recursively scans a project directory, parses all Python files using
ASTExtractor, ignores non-source/virtualenv folders, and outputs a 
unified repository map.
"""
import fnmatch
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from indexer.ast_extractor import extract_file

logger = logging.getLogger(__name__)

# Default folders and patterns to ignore during repository indexing
DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
    "*.egg-info",
}


class RepoIndexer:
    """Recursively parses a Python repository into a structured map."""

    def __init__(self, root_dir: str | Path, ignore_dirs: Optional[Set[str]] = None):
        self.root_dir = Path(root_dir).resolve()
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS

    def should_ignore(self, path: Path) -> bool:
        """Determines if a directory or file should be skipped using pattern matching."""
        for part in path.parts:
            # Skip hidden folders/files (e.g., .git, .env)
            if part.startswith("."):
                return True
            # Match against explicit names or wildcard patterns (e.g., *.egg-info)
            for pattern in self.ignore_dirs:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    def index_repository(self) -> Dict[str, Any]:
        """Walks the directory tree and builds the aggregated AST map."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Repository root not found: {self.root_dir}")

        repo_map: Dict[str, Any] = {
            "root_path": str(self.root_dir),
            "files_parsed": 0,
            "modules": {}
        }

        python_files = [
            p for p in self.root_dir.rglob("*.py")
            if not self.should_ignore(p.relative_to(self.root_dir))
        ]

        for file_path in python_files:
            # Use POSIX path separators to ensure cross-platform consistency in JSON
            relative_path = file_path.relative_to(self.root_dir).as_posix()
            extracted_data = extract_file(file_path)

            if extracted_data:
                extracted_data["relative_path"] = relative_path
                repo_map["modules"][relative_path] = extracted_data
                repo_map["files_parsed"] += 1

        return repo_map


def build_repo_map(repo_path: str, output_json_path: Optional[str] = None) -> Dict[str, Any]:
    """Helper entry point to index a repo and optionally write out a JSON map."""
    indexer = RepoIndexer(repo_path)
    result = indexer.index_repository()

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved repository map to {out_p}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    summary = build_repo_map("sample_repo", "sample_repo/repo_map.json")
    print(f"\nIndexed {summary['files_parsed']} files successfully.")