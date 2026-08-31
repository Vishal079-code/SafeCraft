"""
Repo Onboarder: one-time (and incremental re-run) setup for a target repo.
Given a local repo path, ensures repo_map.json, the vector store
(ChromaDB), and the BM25 index are all present, current, and ready to
use for retrieval - safe to call every time before working on a repo.

Storage convention: each repo gets its own folder under
.safecraft_data/repos/<repo_key>/ so multiple repos can be onboarded
independently without colliding. repo_key = <folder_name>_<8-char
sha256 of the absolute resolved path> - the hash suffix exists
specifically to avoid collisions between two different repos that
happen to share a folder name (e.g. two different projects both named
"backend" living at different paths on disk).

Change detection: each file's content is hashed (sha256) and stored in
repo_map.json alongside its extracted AST metadata. On a re-run:
  - unchanged files (hash matches) are copied through verbatim - never
    re-parsed, never re-embedded. This is the whole point: embeddings
    are the expensive part, so skipping untouched files is what makes
    re-onboarding cheap.
  - changed or newly added files are re-extracted; their OLD summary
    unit ids (if any existed) are deleted from the vector store FIRST,
    then the new units are embedded and upserted. Deleting first
    matters for renamed/removed functions WITHIN a changed file - a
    plain upsert alone would leave orphaned old ids behind.
  - files that existed in the previous repo_map but no longer exist on
    disk have their unit ids deleted from the vector store and are
    dropped from the new repo_map entirely.

BM25 is always FULLY rebuilt from the final, up-to-date repo_map on
every call, incremental or not - see module docstring reasoning above.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from indexer.repo_indexer import RepoIndexer
from indexer.ast_extractor import extract_file
from retriever.summary_builder import build_summary_units
from retriever.vector_store import VectorStore
from retriever.bm25_store import BM25Index
from retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

SAFECRAFT_DATA_ROOT = Path(".safecraft_data/repos")


def _repo_key(repo_path: str) -> str:
    resolved = str(Path(repo_path).resolve())
    folder_name = Path(resolved).name
    path_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]
    return f"{folder_name}_{path_hash}"


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _repo_data_dir(repo_path: str) -> Path:
    return SAFECRAFT_DATA_ROOT / _repo_key(repo_path)


def _load_previous_repo_map(data_dir: Path) -> Optional[Dict[str, Any]]:
    repo_map_path = data_dir / "repo_map.json"
    if not repo_map_path.exists():
        return None
    try:
        with open(repo_map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Previous repo_map.json unreadable ({e}) - treating as no previous index.")
        return None


def _unit_ids_for_file(repo_map: Dict[str, Any], filepath: str) -> list:
    """Reconstructs the summary unit ids that would exist for a SINGLE
    file's entry in a repo_map, without needing the whole repo - lets us
    know exactly which vector store ids to delete for one changed/removed
    file, without touching anyone else's entries."""
    if filepath not in repo_map.get("modules", {}):
        return []
    single_file_map = {"modules": {filepath: repo_map["modules"][filepath]}}
    units = build_summary_units(single_file_map)
    return [u["id"] for u in units]


def onboard_repo(repo_path: str, force_full_reindex: bool = False) -> Dict[str, Any]:
    """
    Ensures repo_path is indexed and ready for retrieval. Safe to call
    every time before working on a repo - detects what's actually
    changed since the last onboarding and does the minimum necessary
    work, rather than re-embedding everything from scratch each time.

    force_full_reindex=True ignores any previous index entirely and
    rebuilds from zero - useful if the stored index is ever suspected
    to be corrupted or badly out of sync.

    Returns:
        {
            "repo_key": str, "data_dir": str, "files_total": int,
            "files_added": int, "files_changed": int,
            "files_removed": int, "files_unchanged": int,
            "retriever": HybridRetriever,   # ready to use immediately
        }
    """
    repo_path = str(Path(repo_path).resolve())
    data_dir = _repo_data_dir(repo_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    previous_map = None if force_full_reindex else _load_previous_repo_map(data_dir)

    indexer = RepoIndexer(repo_path)
    python_files = [
        p for p in indexer.root_dir.rglob("*.py")
        if not indexer.should_ignore(p.relative_to(indexer.root_dir))
    ]

    new_map: Dict[str, Any] = {"root_path": repo_path, "files_parsed": 0, "modules": {}}
    added, changed, unchanged = [], [], []

    for file_path in python_files:
        relative_path = file_path.relative_to(indexer.root_dir).as_posix()
        content = file_path.read_text(encoding="utf-8")
        content_hash = _file_hash(content)

        prev_entry = (previous_map or {}).get("modules", {}).get(relative_path)
        if prev_entry is not None and prev_entry.get("content_hash") == content_hash:
            new_map["modules"][relative_path] = prev_entry
            unchanged.append(relative_path)
            continue

        extracted = extract_file(file_path)
        if not extracted:
            continue
        extracted["relative_path"] = relative_path
        extracted["content_hash"] = content_hash
        new_map["modules"][relative_path] = extracted

        if prev_entry is None:
            added.append(relative_path)
        else:
            changed.append(relative_path)

    removed = []
    if previous_map:
        removed = [f for f in previous_map.get("modules", {}) if f not in new_map["modules"]]

    new_map["files_parsed"] = len(new_map["modules"])

    # Vector store: delete stale ids for changed/removed files FIRST,
    # then upsert fresh embeddings only for added/changed files.
    # Unchanged files are never touched - no re-embedding, no API cost.
    v_store = VectorStore(
        db_path=str(data_dir / "chroma_db"),
        collection_name="repo_index",
    )

    for filepath in changed + removed:
        if previous_map:
            stale_ids = _unit_ids_for_file(previous_map, filepath)
            if stale_ids:
                try:
                    v_store.collection.delete(ids=stale_ids)
                except Exception as e:
                    logger.warning(f"Could not delete stale vector ids for {filepath}: {e}")

    files_to_embed = added + changed
    if files_to_embed:
        partial_map = {"modules": {f: new_map["modules"][f] for f in files_to_embed}}
        v_store.index_repo_map(partial_map)

    # BM25: always fully rebuilt - cheap, no embedding cost, no reliable
    # incremental path exists for BM25Okapi anyway.
    b_index = BM25Index()
    b_index.index_repo_map(new_map)

    (data_dir / "repo_map.json").write_text(json.dumps(new_map, indent=2), encoding="utf-8")

    retriever = HybridRetriever(vector_store=v_store, bm25_index=b_index)

    summary = {
        "repo_key": _repo_key(repo_path),
        "data_dir": str(data_dir),
        "files_total": len(new_map["modules"]),
        "files_added": len(added),
        "files_changed": len(changed),
        "files_removed": len(removed),
        "files_unchanged": len(unchanged),
        "retriever": retriever,
    }

    logger.info(
        f"Onboarded '{repo_path}': {summary['files_total']} files total "
        f"(+{summary['files_added']} added, ~{summary['files_changed']} changed, "
        f"-{summary['files_removed']} removed, {summary['files_unchanged']} unchanged)"
    )
    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "big_repo"
    result = onboard_repo(target)
    print(f"\nRepo key: {result['repo_key']}")
    print(f"Data dir: {result['data_dir']}")
    print(f"Files: {result['files_total']} total | "
          f"+{result['files_added']} added | ~{result['files_changed']} changed | "
          f"-{result['files_removed']} removed | {result['files_unchanged']} unchanged")