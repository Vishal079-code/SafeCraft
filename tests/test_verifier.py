import pytest
from verifier.ast_verifier import verify_code_syntax
from verifier.test_runner import run_pytest_suite

def test_ast_verifier_valid_code():
    valid_py = "def add(a, b):\n    return a + b\n"
    is_valid, err = verify_code_syntax(valid_py)
    assert is_valid is True
    assert err is None

def test_ast_verifier_syntax_error():
    invalid_py = "def add(a, b\n    return a + b"
    is_valid, err = verify_code_syntax(invalid_py)
    assert is_valid is False
    assert "SyntaxError" in err

def test_run_pytest_suite_execution():
    result = run_pytest_suite(test_dir="tests/test_agent_tool_loop.py")
    assert "success" in result
    assert result["success"] is True