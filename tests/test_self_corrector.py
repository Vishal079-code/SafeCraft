from verifier.self_corrector import validate_staged_workspace


def test_validate_staged_workspace_syntax_failure():
    staged = {"main.py": "def invalid_func(\n    return 42\n"}
    result = validate_staged_workspace(staged, modified_files=["main.py"])
    assert result["valid"] is False
    assert result["error_type"] == "AST_SYNTAX_ERROR"
    assert "Syntax error detected" in result["feedback"]
    print("[PASS] Syntax failure correctly detected")


def test_validate_staged_workspace_valid_code():
    staged = {"main.py": "def valid_func():\n    return 42\n"}
    result = validate_staged_workspace(staged, modified_files=["main.py"], test_dir="tests/test_verifier.py")
    assert result["valid"] is True
    assert result["feedback"] is None
    print("[PASS] Valid code correctly passes both phases")


def test_validate_staged_workspace_missing_file():
    """NEW: a file in modified_files that doesn't actually exist in
    staged_workspace must fail loudly, not silently pass as 'valid empty code'."""
    staged = {"other.py": "x = 1"}
    result = validate_staged_workspace(staged, modified_files=["main.py"], test_dir="tests/test_verifier.py")
    assert result["valid"] is False
    assert result["error_type"] == "MISSING_FILE"
    print("[PASS] Missing file correctly rejected instead of silently passing")


if __name__ == "__main__":
    test_validate_staged_workspace_syntax_failure()
    test_validate_staged_workspace_valid_code()
    test_validate_staged_workspace_missing_file()
    print("\nALL SELF-CORRECTOR TESTS PASSED")