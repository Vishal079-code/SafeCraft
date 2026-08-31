"""Test: agent/orchestrator.py - optional semantic retrieval integration.

Uses a fake retriever (no real embeddings/ChromaDB/API calls needed) so
this stays fast and fully offline, same as the rest of the suite. Real
HybridRetriever behavior was already confirmed separately against
big_repo; this test is purely about the ORCHESTRATOR's union/wiring
logic - does it call the retriever correctly, merge results correctly,
and stay identical to pre-retrieval behavior when no retriever is given.
"""
from agent.orchestrator import AgentOrchestrator


class FakeRetriever:
    """Returns a fixed, pre-scripted set of hits regardless of query -
    lets us test the orchestrator's handling of retrieval results without
    depending on real embeddings ever agreeing with a specific query."""

    def __init__(self, hits):
        self._hits = hits
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query, top_k=3):
        self.last_query = query
        self.last_top_k = top_k
        return self._hits


class FailingRetriever:
    """Simulates a broken/unindexed retriever - e.g. querying a repo that
    was never onboarded. Orchestrator must swallow this, not crash."""

    def retrieve(self, query, top_k=3):
        raise RuntimeError("Collection does not exist")


WORKSPACE = {
    "models/product.py": (
        "from models.base import BaseModel\n\n"
        "class Product(BaseModel):\n"
        "    def apply_discount(self, percent):\n"
        "        self.price = round(self.price * (1 - percent / 100), 2)\n"
        "        return self.price\n"
    ),
    "models/base.py": (
        "class BaseModel:\n"
        "    def __init__(self, id=None):\n"
        "        self.id = id\n"
    ),
    "apis/routes.py": (
        "def handle_apply_discount_request(product, percent):\n"
        "    return product.apply_discount(percent)\n"
    ),
    "utils/formatting.py": (
        "def format_currency(amount):\n"
        "    return f'${amount:.2f}'\n"
    ),
}


def mock_llm(prompt):
    # IMPORTANT: protected_identifiers=["apply_discount"] means the real
    # anonymizer rewrites `apply_discount` to `__ANON_FUNC_1__` before this
    # prompt is ever built - the mock must echo back that SAME placeholder,
    # not the real name, or apply_patches_staged correctly rejects it as a
    # non-match (this is the exact bug that broke this file's first draft).
    return (
        "File: models/product.py\n<<<<<<< SEARCH\n"
        "    def __ANON_FUNC_1__(self, percent):\n"
        "        self.price = round(self.price * (1 - percent / 100), 2)\n"
        "        return self.price\n=======\n"
        "    def __ANON_FUNC_1__(self, percent):\n"
        "        if percent > 90:\n"
        "            raise ValueError('Discount cannot exceed 90 percent')\n"
        "        self.price = round(self.price * (1 - percent / 100), 2)\n"
        "        return self.price\n>>>>>>> REPLACE"
    )


def test_no_retriever_behaves_exactly_as_before():
    """Default AgentOrchestrator() - no retriever passed - must be
    identical to pre-retrieval behavior: bundled_files only from
    ContextBundler, retrieval_added_files stays None (not [])."""
    orchestrator = AgentOrchestrator(protected_identifiers=["apply_discount"])
    result = orchestrator.run_cycle(
        user_request="Add a safety check over 90%",
        target_filepath="models/product.py",
        workspace_files=WORKSPACE,
        llm_caller_fn=mock_llm,
    )

    assert result.success is True, result.error_message
    assert set(result.bundled_files) == {"models/product.py", "models/base.py"}
    assert result.retrieval_added_files is None
    print("[PASS] No retriever configured - bundling-only behavior unchanged")


def test_retriever_adds_files_not_reachable_by_imports():
    """apis/routes.py imports FROM models.product, not the other way
    round, so ContextBundler (which only walks target_filepath's OWN
    imports) will never find it starting from models/product.py.
    Retrieval should surface it anyway, and it should be unioned in."""
    fake_hits = [
        {"metadata": {"filepath": "apis/routes.py"}, "text": "handle_apply_discount_request"},
    ]
    retriever = FakeRetriever(fake_hits)
    orchestrator = AgentOrchestrator(
        protected_identifiers=["apply_discount"],
        retriever=retriever,
        retrieval_top_k=3,
    )
    result = orchestrator.run_cycle(
        user_request="Add a safety check over 90%",
        target_filepath="models/product.py",
        workspace_files=WORKSPACE,
        llm_caller_fn=mock_llm,
    )

    assert result.success is True, result.error_message
    assert set(result.bundled_files) == {"models/product.py", "models/base.py", "apis/routes.py"}
    assert result.retrieval_added_files == ["apis/routes.py"]
    # confirm the orchestrator actually queried with the real user request
    assert retriever.last_query == "Add a safety check over 90%"
    assert retriever.last_top_k == 3
    print("[PASS] Retriever result unioned in, not reachable via imports alone")


def test_retriever_does_not_duplicate_files_already_bundled():
    """models/base.py IS already reachable via ContextBundler. If the
    retriever ALSO returns it, it must not appear twice in bundled_files,
    and must not appear in retrieval_added_files (it added nothing new)."""
    fake_hits = [
        {"metadata": {"filepath": "models/base.py"}, "text": "BaseModel"},
    ]
    retriever = FakeRetriever(fake_hits)
    orchestrator = AgentOrchestrator(
        protected_identifiers=["apply_discount"],
        retriever=retriever,
    )
    result = orchestrator.run_cycle(
        user_request="task",
        target_filepath="models/product.py",
        workspace_files=WORKSPACE,
        llm_caller_fn=mock_llm,
    )

    assert result.success is True, result.error_message
    assert result.bundled_files.count("models/base.py") == 1
    assert result.retrieval_added_files == []
    print("[PASS] Overlapping retriever hit deduplicated correctly")


def test_retriever_hit_for_nonexistent_file_is_skipped_safely():
    """A stale/unindexed retriever result pointing at a file that isn't
    actually in the current workspace (e.g. renamed/deleted since the
    repo was last onboarded) must be skipped, not crash the run."""
    fake_hits = [
        {"metadata": {"filepath": "does/not/exist.py"}, "text": "ghost"},
    ]
    retriever = FakeRetriever(fake_hits)
    orchestrator = AgentOrchestrator(
        protected_identifiers=["apply_discount"],
        retriever=retriever,
    )
    result = orchestrator.run_cycle(
        user_request="task",
        target_filepath="models/product.py",
        workspace_files=WORKSPACE,
        llm_caller_fn=mock_llm,
    )

    assert result.success is True, result.error_message
    assert "does/not/exist.py" not in result.bundled_files
    assert result.retrieval_added_files == []
    print("[PASS] Stale retriever hit for missing file skipped safely")


def test_failing_retriever_does_not_crash_the_run():
    """If querying the retriever raises (e.g. repo was never onboarded,
    ChromaDB collection doesn't exist yet), the run must still succeed
    using ContextBundler's results alone - retrieval is an enhancement,
    never a hard requirement."""
    orchestrator = AgentOrchestrator(
        protected_identifiers=["apply_discount"],
        retriever=FailingRetriever(),
    )
    result = orchestrator.run_cycle(
        user_request="task",
        target_filepath="models/product.py",
        workspace_files=WORKSPACE,
        llm_caller_fn=mock_llm,
    )

    assert result.success is True, result.error_message
    assert set(result.bundled_files) == {"models/product.py", "models/base.py"}
    assert result.retrieval_added_files == []
    print("[PASS] Failing retriever query swallowed, run succeeds via bundling alone")


if __name__ == "__main__":
    test_no_retriever_behaves_exactly_as_before()
    test_retriever_adds_files_not_reachable_by_imports()
    test_retriever_does_not_duplicate_files_already_bundled()
    test_retriever_hit_for_nonexistent_file_is_skipped_safely()
    test_failing_retriever_does_not_crash_the_run()
    print("\nALL RETRIEVAL INTEGRATION TESTS PASSED")