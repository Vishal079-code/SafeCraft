"""
Test Day 14: Non-interactive CLI & Disk Modification Flow
"""
import tempfile
from pathlib import Path
from agent.orchestrator import AgentOrchestrator


def mock_llm_service(prompt: str) -> str:
    return (
        "File: dummy.py\n<<<<<<< SEARCH\n"
        "def __ANON_FUNC_1__(val):\n    return val * __ANON_CONST_1__\n=======\n"
        "def __ANON_FUNC_1__(val):\n    return val * __ANON_CONST_1__ + 10\n>>>>>>> REPLACE"
    )
def test_cli_disk_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "dummy.py"
        test_file.write_text("def process_data(val):\n    return val * 42", encoding="utf-8")

        logical_name = "dummy.py"
        orchestrator = AgentOrchestrator(protected_identifiers=["process_data"], protected_literals=[42])
        workspace = {logical_name: test_file.read_text(encoding="utf-8")}
        result = orchestrator.run_cycle(
            user_request="Add 10 to output", target_filepath=logical_name,
            workspace_files=workspace, llm_caller_fn=mock_llm_service,
        )
        assert result.success, f"Orchestrator failed: {result.error_message}"

        test_file.write_text(result.staged_workspace[logical_name], encoding="utf-8")
        updated_disk_content = test_file.read_text(encoding="utf-8")
        assert "return val * 42 + 10" in updated_disk_content
        print("[PASS] Disk update verified! Final pipeline complete.")


if __name__ == "__main__":
    test_cli_disk_write()
    print("ALL DAY 14 TESTS PASSED")