"""Test: planner/multi_step_executor.py - sequential multi-file plan execution."""
from planner.subtask_planner import generate_plan
from planner.multi_step_executor import execute_plan
from agent.orchestrator import AgentOrchestrator


def test_execute_plan_success_across_two_files():
    workspace = {
        "routes/api.py": 'def get_routes():\n    return ["/home"]',
        "main.py": 'def get_nav_links():\n    return ["Home"]',
    }

    def planner_mock(prompt):
        return '[{"description": "Add settings route", "target_filepath": "routes/api.py"}, {"description": "Add settings nav link", "target_filepath": "main.py"}]'

    def orchestrator_mock(prompt):
        if "routes/api.py" in prompt:
            return 'File: routes/api.py\n<<<<<<< SEARCH\n    return ["/home"]\n=======\n    return ["/home", "/settings"]\n>>>>>>> REPLACE'
        return 'File: main.py\n<<<<<<< SEARCH\n    return ["Home"]\n=======\n    return ["Home", "Settings"]\n>>>>>>> REPLACE'

    plan = generate_plan("Add settings route and nav link", list(workspace.keys()), planner_mock)
    orchestrator = AgentOrchestrator()
    result = execute_plan(plan, workspace, orchestrator, orchestrator_mock)

    assert result["success"] is True
    assert result["completed_subtasks"] == 2
    assert '"/settings"' in result["final_workspace"]["routes/api.py"]
    assert '"Settings"' in result["final_workspace"]["main.py"]
    print("[PASS] Two-file sequential plan executed correctly, both files updated")


def test_execute_plan_stops_at_first_failure_without_corrupting_later_files():
    workspace = {
        "routes/api.py": 'def get_routes():\n    return ["/home"]',
        "main.py": 'def get_nav_links():\n    return ["Home"]',
    }

    def planner_mock(prompt):
        return '[{"description": "Add route", "target_filepath": "routes/api.py"}, {"description": "Add nav link", "target_filepath": "main.py"}]'

    def failing_mock(prompt):
        if "routes/api.py" in prompt:
            # subtask 1 succeeds
            return 'File: routes/api.py\n<<<<<<< SEARCH\n    return ["/home"]\n=======\n    return ["/home", "/settings"]\n>>>>>>> REPLACE'
        # subtask 2's SEARCH deliberately does not match -> fails
        return 'File: main.py\n<<<<<<< SEARCH\n    return ["TEXT THAT DOES NOT EXIST"]\n=======\n    return ["Home", "Settings"]\n>>>>>>> REPLACE'

    plan = generate_plan("task", list(workspace.keys()), planner_mock)
    orchestrator = AgentOrchestrator()
    result = execute_plan(plan, workspace, orchestrator, failing_mock)

    assert result["success"] is False
    assert result["completed_subtasks"] == 1
    # subtask 1's change survives in final_workspace even though overall failed
    assert '"/settings"' in result["final_workspace"]["routes/api.py"]
    # subtask 2 never touched main.py
    assert result["final_workspace"]["main.py"] == workspace["main.py"]
    print("[PASS] Execution stopped cleanly at failure, subtask 1 preserved, subtask 2 untouched")


if __name__ == "__main__":
    test_execute_plan_success_across_two_files()
    test_execute_plan_stops_at_first_failure_without_corrupting_later_files()
    print("\nALL MULTI-STEP EXECUTOR TESTS PASSED")