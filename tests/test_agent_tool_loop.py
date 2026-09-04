import copy
import json
import pytest
from unittest.mock import MagicMock
from agent.tool_loop import AgentToolLoop


class MockFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class MockGeminiResponse:
    def __init__(self, function_calls=None, text=None):
        self.function_calls = function_calls or []
        self.text = text


@pytest.fixture
def sample_workspace():
    return {
        "main.py": "def process_data(secret_user):\n    return secret_user.get_name()\n",
        "utils/helpers.py": "class SecretUser:\n    def get_name(self):\n        return 'Alice'\n",
    }


def test_full_tool_loop_multi_turn_with_inspection(sample_workspace):
    turn1_resp = MockGeminiResponse(
        function_calls=[MockFunctionCall("read_file", {"filepath": "utils/helpers.py"})]
    )
    diff_text = (
        "File: main.py\n<<<<<<< SEARCH\n"
        "def __ANON_FUNC_1__(secret_user):\n=======\n"
        "def __ANON_FUNC_1__(validated_user):\n>>>>>>> REPLACE"
    )
    turn2_resp = MockGeminiResponse(text=diff_text)
    mock_llm_caller = MagicMock(side_effect=[turn1_resp, turn2_resp])

    loop = AgentToolLoop(
        workspace_files=sample_workspace,
        protected_identifiers=["process_data"]
    )
    result = loop.run(
        user_request="Update process_data parameter name",
        target_filepath="main.py",
        llm_tool_caller_fn=mock_llm_caller
    )

    assert result["success"] is True
    assert result["rounds_used"] == 2
    assert len(result["patches"]) == 1

    search_text = result["patches"][0].search_text
    replace_text = result["patches"][0].replace_text

    assert "__ANON_FUNC_1__" not in search_text, "Placeholder leaked into final patch"
    assert "process_data" in search_text, "Real function name was not correctly restored"
    assert "process_data" in replace_text
    print("\n[PASS] De-anonymization genuinely substituted __ANON_FUNC_1__ -> process_data")


def test_dedup_on_repeat_read_file(sample_workspace):
    turn1_resp = MockGeminiResponse(
        function_calls=[MockFunctionCall("read_file", {"filepath": "utils/helpers.py"})]
    )
    turn2_resp = MockGeminiResponse(
        function_calls=[MockFunctionCall("read_file", {"filepath": "utils/helpers.py"})]
    )
    turn3_resp = MockGeminiResponse(text="File: main.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE")
    mock_llm_caller = MagicMock(side_effect=[turn1_resp, turn2_resp, turn3_resp])

    loop = AgentToolLoop(workspace_files=sample_workspace)
    result = loop.run(
        user_request="task",
        target_filepath="main.py",
        llm_tool_caller_fn=mock_llm_caller
    )

    assert result["rounds_used"] == 3

    second_call_contents = mock_llm_caller.call_args_list[1].args[0]
    last_turn_parts = second_call_contents[-1]["parts"]
    dedup_response = last_turn_parts[0]["function_response"]["response"]["result"]

    assert "Already provided" in dedup_response, f"Dedup did not fire, got: {dedup_response!r}"
    print("\n[PASS] Deduplication correctly fired on repeat read_file")

def test_shared_anonymizer_prevents_cross_file_placeholder_collision():
    """Two files, each with a DIFFERENT protected identifier that would
    each independently become __ANON_FUNC_1__ under a fresh-transformer-
    per-file scheme (the bug this test guards against). With a properly
    shared transformer, they must get DIFFERENT placeholders, and
    de-anonymization must restore each correctly."""
    workspace = {
        "main.py": "def alpha_task(x):\n    return x + 1\n",
        "utils/helpers.py": "def beta_task(y):\n    return y * 2\n",
    }

    # main.py is read first (it's the target), THEN helpers.py via read_file
    turn1_resp = MockGeminiResponse(
        function_calls=[MockFunctionCall("read_file", {"filepath": "utils/helpers.py"})]
    )
    turn2_resp = MockGeminiResponse(
        text="File: main.py\n<<<<<<< SEARCH\nx + 1\n=======\nx + 2\n>>>>>>> REPLACE"
    )
    mock_llm_caller = MagicMock(side_effect=[turn1_resp, turn2_resp])

    loop = AgentToolLoop(
        workspace_files=workspace,
        protected_identifiers=["alpha_task", "beta_task"],
    )
    result = loop.run(
        user_request="task",
        target_filepath="main.py",
        llm_tool_caller_fn=mock_llm_caller,
    )

    assert result["success"] is True
    reverse_map = loop._shared_transformer.reverse_identifier_map
    # both real names must be present, mapped to DIFFERENT placeholders -
    # this is the actual assertion that would have failed under the bug
    assert "alpha_task" in reverse_map.values()
    assert "beta_task" in reverse_map.values()
    placeholders_for_names = [k for k, v in reverse_map.items() if v in ("alpha_task", "beta_task")]
    assert len(set(placeholders_for_names)) == 2, "Both names collapsed onto the same placeholder!"
    print("\n[PASS] Shared transformer assigns distinct placeholders across files")
def test_symbol_scoped_read_file_hides_other_functions():
    """A symbol-scoped read_file call must return the target function's
    real body, while every OTHER function in the same file is
    skeletonized - directly verifying content minimization, not just
    that the call succeeds."""
    workspace = {
        "billing.py": (
            "def calculate_tax(amount):\n"
            "    return amount * 0.08\n\n"
            "def calculate_shipping_secret_formula(weight):\n"
            "    return weight * 2.5 + 4.99\n"
        ),
    }

    turn1_resp = MockGeminiResponse(
        function_calls=[MockFunctionCall("read_file", {"filepath": "billing.py", "symbol": "calculate_tax"})]
    )
    turn2_resp = MockGeminiResponse(text="File: billing.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE")
    mock_llm_caller = MagicMock(side_effect=[turn1_resp, turn2_resp])

    loop = AgentToolLoop(workspace_files={"main.py": "x = 1\n", **workspace})
    loop.run(user_request="task", target_filepath="main.py", llm_tool_caller_fn=mock_llm_caller)

    read_call_result = mock_llm_caller.call_args_list[1].args[0][-1]["parts"][0]
    tool_output = read_call_result["function_response"]["response"]["result"]

    assert "amount * 0.08" in tool_output, "Requested symbol's real body should be present"
    assert "weight * 2.5" not in tool_output, "Non-requested function's body leaked - not skeletonized!"
    assert "..." in tool_output, "Non-requested function should show as a skeleton stub"
    print("\n[PASS] Symbol-scoped read_file exposes only the requested function")


def test_symbol_scoped_dedup_allows_different_symbol_same_file():
    """Reading a DIFFERENT symbol from a file already partially read
    should NOT be deduped - only a repeat of the SAME (file, symbol), or
    a file already sent in full, should be."""
    workspace = {
        "main.py": "x = 1\n",
        "billing.py": (
            "def calculate_tax(amount):\n    return amount * 0.08\n\n"
            "def calculate_shipping(weight):\n    return weight * 2.5\n"
        ),
    }
    turn1 = MockGeminiResponse(function_calls=[MockFunctionCall("read_file", {"filepath": "billing.py", "symbol": "calculate_tax"})])
    turn2 = MockGeminiResponse(function_calls=[MockFunctionCall("read_file", {"filepath": "billing.py", "symbol": "calculate_shipping"})])
    turn3 = MockGeminiResponse(text="File: main.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE")
    mock_llm_caller = MagicMock(side_effect=[turn1, turn2, turn3])

    loop = AgentToolLoop(workspace_files=workspace)
    result = loop.run(user_request="task", target_filepath="main.py", llm_tool_caller_fn=mock_llm_caller)

    assert result["rounds_used"] == 3, "Second, different-symbol read should NOT have been deduped"
    second_tool_output = mock_llm_caller.call_args_list[2].args[0][-1]["parts"][0]["function_response"]["response"]["result"]
    assert "weight * 2.5" in second_tool_output, "The second, distinct symbol's real body should be present"
    assert "Already provided" not in second_tool_output
    print("\n[PASS] Different symbol from same file is not incorrectly deduped")
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])