"""
Test Day 13: End-to-End Orchestrator Pipeline Integration
"""

from agent.orchestrator import AgentOrchestrator


def mock_llm_service(prompt: str) -> str:
    """
    Simulates a remote cloud LLM receiving anonymized prompt text
    and returning an anonymized SEARCH/REPLACE diff response.
    """
    # Verify that raw identifiers NEVER reached the external LLM call
    assert "calculate_salary" not in prompt, "LEAK DETECTED: 'calculate_salary' sent to LLM!"
    assert "base_pay" not in prompt, "LEAK DETECTED: 'base_pay' sent to LLM!"
    assert "0.5" not in prompt, "LEAK DETECTED: Raw literal '0.5' sent to LLM!"

    return """
I have updated the payroll function to apply a 10% tax adjustment.

File: auth/payroll.py
<<<<<<< SEARCH
def __ANON_FUNC_1__(employee_grade, __ANON_VAR_1__):
    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__
    return __ANON_VAR_1__ + __ANON_VAR_2__
=======
def __ANON_FUNC_1__(employee_grade, __ANON_VAR_1__):
    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__
    return (__ANON_VAR_1__ + __ANON_VAR_2__) * 0.90
>>>>>>> REPLACE
"""


def test_full_agent_orchestration_cycle():
    workspace = {
        "auth/payroll.py": """def calculate_salary(employee_grade, base_pay):
    incentive = 0.5 * base_pay
    return base_pay + incentive"""
    }

    orchestrator = AgentOrchestrator(
        protected_identifiers=["calculate_salary", "base_pay", "incentive"],
        protected_literals=[0.5],
    )

    result = orchestrator.run_cycle(
        user_request="Apply a 10% tax adjustment to the total payout.",
        target_filepath="auth/payroll.py",
        workspace_files=workspace,
        llm_caller_fn=mock_llm_service,
    )

    print("=" * 60)
    print("  TEST 1: Full Agent Orchestration Cycle")
    print("=" * 60)

    assert result.success, f"Orchestrator failed with error: {result.error_message}"
    
    updated_file_content = result.staged_workspace["auth/payroll.py"]
    
    print("\n--- STAGED UPDATED FILE CONTENT ---")
    print(updated_file_content)

    # Verify that local code restored original identifiers cleanly
    assert "calculate_salary" in updated_file_content
    assert "base_pay" in updated_file_content
    assert "0.5" in updated_file_content
    assert "0.90" in updated_file_content
    assert "__ANON_FUNC_1__" not in updated_file_content

    print("\n[PASS] Pipeline executed cleanly: Privacy maintained in prompt, original code restored in staged output!")


if __name__ == "__main__":
    test_full_agent_orchestration_cycle()
    print("\n" + "=" * 60)
    print("  ALL DAY 13 TESTS PASSED")
    print("=" * 60)