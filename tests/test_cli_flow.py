"""
Test Day 14: Non-interactive CLI & Disk Modification Flow
"""

import tempfile
from pathlib import Path
from agent.orchestrator import AgentOrchestrator


def mock_llm_service(prompt: str) -> str:
    return """
File: dummy.py
<<<<<<< SEARCH
def process_data(val):
    return val * __ANON_CONST_1__
=======
def process_data(val):
    return val * __ANON_CONST_1__ + 10
>>>>>>> REPLACE
"""


def test_cli_disk_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = "dummy.py"
        test_file = Path(tmpdir) / filename
        test_file.write_text("def process_data(val):\n    return val * 42", encoding="utf-8")

        orchestrator = AgentOrchestrator(
            protected_identifiers=["process_data"],
            protected_literals=[42],
        )

        # Use filename matching the mock's SEARCH block header ("File: dummy.py")
        workspace = {filename: test_file.read_text(encoding="utf-8")}

        result = orchestrator.run_cycle(
            user_request="Add 10 to output",
            target_filepath=filename,
            workspace_files=workspace,
            llm_caller_fn=mock_llm_service,
        )

        assert result.success, f"Orchestrator failed: {result.error_message}"

        # Write staged content to actual temporary file on disk
        test_file.write_text(result.staged_workspace[filename], encoding="utf-8")

        updated_disk_content = test_file.read_text(encoding="utf-8")

        print("=" * 60)
        print("  TEST 1: End-to-End Disk Write Test")
        print("=" * 60)
        print(updated_disk_content)

        assert "return val * 42 + 10" in updated_disk_content
        print("\n[PASS] Disk update verified! Final Day 14 pipeline complete.")


if __name__ == "__main__":
    test_cli_disk_write()
    print("\n" + "=" * 60)
    print("  ALL DAY 14 TESTS PASSED")
    print("=" * 60)