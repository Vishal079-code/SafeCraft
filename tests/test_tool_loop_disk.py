"""Test: AgentToolLoop.run_and_commit - the tool loop wired to staging + disk commit."""
import tempfile
from pathlib import Path
from agent.tool_loop import AgentToolLoop, run_and_commit


class FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeGeminiResponse:
    def __init__(self, function_calls=None, text=""):
        self.function_calls = function_calls or []
        self.text = text


def test_run_and_commit_full_pipeline_with_disk_write():
    call_count = [0]

    def mock_caller(contents, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return FakeGeminiResponse(function_calls=[FakeFunctionCall("read_file", {"filepath": "auth/utils.py"})])
        return FakeGeminiResponse(text=(
            "File: main.py\n<<<<<<< SEARCH\n"
            "def process_data(val):\n    return val * __ANON_CONST_1__\n=======\n"
            "def process_data(val):\n    return val * __ANON_CONST_1__ + 5\n>>>>>>> REPLACE"
        ))

    workspace = {
        "main.py": "def process_data(val):\n    return val * 42",
        "auth/utils.py": "def helper():\n    return True",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        loop = AgentToolLoop(workspace_files=workspace, protected_identifiers=["process_data"], protected_literals=[42])
        result = run_and_commit(
            loop, user_request="Add 5", target_filepath="main.py", llm_tool_caller_fn=mock_caller,
            commit_to_disk=True, disk_root=tmpdir,
        )

        assert result["success"] is True
        assert result["committed_to_disk"] is True
        assert result["files_written"] == 2  # both files in the workspace get written

        written_content = Path(tmpdir, "main.py").read_text()
        assert "return val * 42 + 5" in written_content
        print("[PASS] run_and_commit correctly staged, verified, and wrote to disk")


def test_run_and_commit_stops_before_disk_write_on_tool_loop_failure():
    def always_calling_caller(contents, tools):
        # Model never stops calling tools -> loop hits max_rounds and fails
        return FakeGeminiResponse(function_calls=[FakeFunctionCall("read_file", {"filepath": "main.py"})])

    workspace = {"main.py": "x = 1"}
    with tempfile.TemporaryDirectory() as tmpdir:
        loop = AgentToolLoop(workspace_files=workspace, max_rounds=2)
        result = run_and_commit(
            loop, user_request="task", target_filepath="main.py", llm_tool_caller_fn=always_calling_caller,
            commit_to_disk=True, disk_root=tmpdir,
        )
        assert result["success"] is False
        assert result["committed_to_disk"] is False
        assert result["files_written"] == 0
        assert not any(Path(tmpdir).iterdir())  # nothing written to disk
        print("[PASS] Tool loop failure correctly prevents any disk write")


if __name__ == "__main__":
    test_run_and_commit_full_pipeline_with_disk_write()
    test_run_and_commit_stops_before_disk_write_on_tool_loop_failure()
    print("\nALL TOOL LOOP + DISK TESTS PASSED")