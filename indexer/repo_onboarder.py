"""
Repo Onboarder: one-time (and incremental re-run) setup for a target repo.
...
[docstring unchanged from before]
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
from retriever.reranker import Reranker

logger = logging.getLogger(__name__)

SAFECRAFT_DATA_ROOT = Path(".safecraft_data/repos")

# Loaded once per process, not once per repo - the reranker model is the
# same regardless of which repo you're onboarding, so there's no reason
# to reload it on every onboard_repo() call within one session.
_shared_reranker: Optional[Reranker] = None


def _get_shared_reranker(use_reranker: bool) -> Optional[Reranker]:
    global _shared_reranker
    if not use_reranker:
        return None
    if _shared_reranker is None:
        logger.info("Loading local reranker model (first call only)...")
        _shared_reranker = Reranker()
    return _shared_reranker


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



def onboard_repo(repo_path: str, force_full_reindex: bool = False, use_reranker: bool = True) -> Dict[str, Any]:
    """
    ... [rest of docstring unchanged] ...

    use_reranker=True (default): the returned retriever includes a local
    cross-encoder reranker on top of RRF fusion - meaningfully sharper
    relevance scores (see Step 2 validation: RRF alone produced results
    clustered in a ~0.003 band; reranking produced a ~17-point spread
    between clearly relevant and clearly irrelevant results on the same
    query). Set False to fall back to RRF-only behavior, e.g. for a fast
    smoke test where reranker model load time isn't worth it.
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

    b_index = BM25Index()
    b_index.index_repo_map(new_map)

    (data_dir / "repo_map.json").write_text(json.dumps(new_map, indent=2), encoding="utf-8")

    reranker = _get_shared_reranker(use_reranker)
    retriever = HybridRetriever(vector_store=v_store, bm25_index=b_index, reranker=reranker)

    summary = {
        "repo_key": _repo_key(repo_path),
        "data_dir": str(data_dir),
        "files_total": len(new_map["modules"]),
        "files_added": len(added),
        "files_changed": len(changed),
        "files_removed": len(removed),
        "files_unchanged": len(unchanged),
        "reranker_enabled": reranker is not None,
        "retriever": retriever,
    }

    logger.info(
        f"Onboarded '{repo_path}': {summary['files_total']} files total "
        f"(+{summary['files_added']} added, ~{summary['files_changed']} changed, "
        f"-{summary['files_removed']} removed, {summary['files_unchanged']} unchanged, "
        f"reranker={'on' if reranker else 'off'})"
    )
    return summary


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "big_repo"
    result = onboard_repo(target)
    print(f"\nRepo key: {result['repo_key']}")
    print(f"Data dir: {result['data_dir']}")
    print(f"Reranker: {'enabled' if result['reranker_enabled'] else 'disabled'}")
    print(f"Files: {result['files_total']} total | "
          f"+{result['files_added']} added | ~{result['files_changed']} changed | "
          f"-{result['files_removed']} removed | {result['files_unchanged']} unchanged")