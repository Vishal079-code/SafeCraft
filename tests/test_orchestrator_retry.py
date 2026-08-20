"""Test: AgentOrchestrator.run_cycle retry logic - motivated by real observed
LLM non-determinism (same prompt, different output across calls)."""
from agent.orchestrator import AgentOrchestrator


def test_retry_recovers_from_first_attempt_mismatch():
    call_count = [0]

    def flaky_llm(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            # Attempt 1: SEARCH block deliberately does not match
            return (
                "File: main.py\n<<<<<<< SEARCH\n"
                "def process_data(val):\n    return val * __ANON_CONST_1__ MISMATCH\n=======\n"
                "def process_data(val):\n    return val * __ANON_CONST_1__ + 5\n>>>>>>> REPLACE"
            )
        # Attempt 2: correct
        return (
            "File: main.py\n<<<<<<< SEARCH\n"
            "def process_data(val):\n    return val * __ANON_CONST_1__\n=======\n"
            "def process_data(val):\n    return val * __ANON_CONST_1__ + 5\n>>>>>>> REPLACE"
        )

    workspace = {"main.py": "def process_data(val):\n    return val * 42"}
    orchestrator = AgentOrchestrator(protected_identifiers=["process_data"], protected_literals=[42])
    result = orchestrator.run_cycle(
        user_request="task", target_filepath="main.py", workspace_files=workspace,
        llm_caller_fn=flaky_llm, max_attempts=2,
    )

    assert result.success is True
    assert result.attempts_used == 2
    assert call_count[0] == 2
    assert "return val * 42 + 5" in result.staged_workspace["main.py"]
    print("[PASS] Retry correctly recovered from a first-attempt SEARCH mismatch")


def test_retry_exhausts_and_reports_failure_cleanly():
    call_count = [0]

    def always_broken_llm(prompt):
        call_count[0] += 1
        return "File: main.py\n<<<<<<< SEARCH\nNEVER MATCHES\n=======\nnew\n>>>>>>> REPLACE"

    workspace = {"main.py": "def process_data(val):\n    return val * 42"}
    orchestrator = AgentOrchestrator(protected_identifiers=["process_data"], protected_literals=[42])
    result = orchestrator.run_cycle(
        user_request="task", target_filepath="main.py", workspace_files=workspace,
        llm_caller_fn=always_broken_llm, max_attempts=3,
    )

    assert result.success is False
    assert result.attempts_used == 3
    assert call_count[0] == 3  # exactly max_attempts calls, not more, not fewer
    assert "Failed after 3 attempt(s)" in result.error_message
    print("[PASS] Retry correctly exhausts after max_attempts and reports failure cleanly")


def test_first_attempt_success_does_not_trigger_extra_calls():
    call_count = [0]

    def working_llm(prompt):
        call_count[0] += 1
        return (
            "File: main.py\n<<<<<<< SEARCH\n"
            "def process_data(val):\n    return val * __ANON_CONST_1__\n=======\n"
            "def process_data(val):\n    return val * __ANON_CONST_1__ + 5\n>>>>>>> REPLACE"
        )

    workspace = {"main.py": "def process_data(val):\n    return val * 42"}
    orchestrator = AgentOrchestrator(protected_identifiers=["process_data"], protected_literals=[42])
    result = orchestrator.run_cycle(
        user_request="task", target_filepath="main.py", workspace_files=workspace,
        llm_caller_fn=working_llm, max_attempts=3,
    )

    assert result.success is True
    assert result.attempts_used == 1
    assert call_count[0] == 1  # no wasted retry calls when the first attempt already succeeds
    print("[PASS] Successful first attempt does not waste additional LLM calls")


if __name__ == "__main__":
    test_retry_recovers_from_first_attempt_mismatch()
    test_retry_exhausts_and_reports_failure_cleanly()
    test_first_attempt_success_does_not_trigger_extra_calls()
    print("\nALL ORCHESTRATOR RETRY TESTS PASSED")