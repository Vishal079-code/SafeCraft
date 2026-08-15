"""Atomic Disk Writer: Flushes in-memory workspace changes to physical disk safely."""

from pathlib import Path
from typing import Dict, Tuple


class DiskCommitError(Exception):
    """Raised when path validation or file writing fails."""
    pass


def commit_staged_workspace_to_disk(
    staged_workspace: Dict[str, str],
    disk_root: str,
) -> Tuple[bool, int]:
    """Validates target paths for path containment safety before writing.
    Flushes staged in-memory workspace to physical disk atomically.
    """
    root_path = Path(disk_root).resolve()
    files_to_write = []

    # Phase 1: Pre-validate all target paths (Prevent Path Traversal)
    for rel_filepath, content in staged_workspace.items():
        target_path = (root_path / rel_filepath).resolve()

        try:
            target_path.relative_to(root_path)
        except ValueError:
            raise DiskCommitError(
                f"Refusing to write outside workspace root: '{rel_filepath}' "
                f"resolved to '{target_path}', which escapes '{root_path}'."
            )

        files_to_write.append((target_path, content))

    # Phase 2: Write all verified files to disk
    files_written = 0
    for target_path, content in files_to_write:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        files_written += 1

    return True, files_written