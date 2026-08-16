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
    # NOTE: after renaming, this now points at ITSELF ("tests/test_verifier.py"),
    # not a distinct file - this is the exact self-reference case the
    # env-var recursion guard in test_runner.py was built to handle safely.
    result = validate_staged_workspace(staged, modified_files=["main.py"], test_dir="tests/test_verifier.py")
    assert result["valid"] is True
    assert result["feedback"] is None
    print("[PASS] Valid code correctly passes both phases")


def test_validate_staged_workspace_missing_file():
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