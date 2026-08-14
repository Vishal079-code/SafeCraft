"""
Test Day 12: In-Memory Patch Staging & Safety Engine
"""

import pytest
from agent.diff_parser import CodePatch
from agent.patch_applier import apply_patch_to_content, apply_patches_staged, PatchApplyError


def test_successful_patch_application():
    original_code = """def calculate_salary(employee_grade, base_pay):
    incentive = 0.5 * base_pay
    return base_pay + incentive"""

    patch = CodePatch(
        filepath="auth/payroll.py",
        search_text="return base_pay + incentive",
        replace_text="return (base_pay + incentive) * 1.10",
    )

    updated_code = apply_patch_to_content(original_code, patch)

    print("=" * 60)
    print("  TEST 1: Clean In-Memory Patch Application")
    print("=" * 60)
    print(updated_code)

    assert "1.10" in updated_code
    print("\n[PASS] Patch applied cleanly and validated AST syntax!")


def test_missing_search_block_rejection():
    original_code = "def foo(): pass"
    patch = CodePatch(
        filepath="test.py",
        search_text="def bar(): pass",
        replace_text="def bar(): return True",
    )

    print("\n" + "=" * 60)
    print("  TEST 2: Missing Search Block Rejection")
    print("=" * 60)

    try:
        apply_patch_to_content(original_code, patch)
        assert False, "FAIL: Should have raised PatchApplyError!"
    except PatchApplyError as e:
        print(f"[PASS] Correctly rejected missing search block: {e}")


def test_ambiguous_search_block_rejection():
    original_code = """x = 1
x = 1"""
    patch = CodePatch(
        filepath="test.py",
        search_text="x = 1",
        replace_text="x = 2",
    )

    print("\n" + "=" * 60)
    print("  TEST 3: Ambiguous Search Block Rejection")
    print("=" * 60)

    try:
        apply_patch_to_content(original_code, patch)
        assert False, "FAIL: Should have raised PatchApplyError for multiple matches!"
    except PatchApplyError as e:
        print(f"[PASS] Correctly rejected ambiguous search block: {e}")


def test_syntax_error_rejection():
    original_code = "def foo():\n    return True"
    patch = CodePatch(
        filepath="test.py",
        search_text="return True",
        replace_text="return (",  # Invalid syntax
    )

    print("\n" + "=" * 60)
    print("  TEST 4: Syntax Error Rejection")
    print("=" * 60)

    try:
        apply_patch_to_content(original_code, patch)
        assert False, "FAIL: Should have caught syntax error in post-patch code!"
    except PatchApplyError as e:
        print(f"[PASS] Correctly rejected patch that causes syntax error: {e}")


if __name__ == "__main__":
    test_successful_patch_application()
    test_missing_search_block_rejection()
    test_ambiguous_search_block_rejection()
    test_syntax_error_rejection()
    print("\n" + "=" * 60)
    print("  ALL DAY 12 TESTS PASSED")
    print("=" * 60)