"""
Test Day 10 De-anonymizer: Round-Trip Integrity & Value Restoration
"""

from privacy.anonymizer import anonymize_code
from privacy.deanonymizer import deanonymize_code, verify_roundtrip


def test_roundtrip_fidelity():
    original_code = """def calculate_salary(employee_grade, base_pay):
    incentive = 0.5 * base_pay
    tax_deduction = 0.5 * incentive
    return base_pay + incentive - tax_deduction"""

    protected_ids = ["calculate_salary", "incentive", "base_pay"]
    protected_lits = [0.5]

    # Step 1: Anonymize
    anonymized_code, privacy_maps = anonymize_code(
        code_snippet=original_code,
        filepath="auth/payroll.py",
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
    )

    print("=" * 60)
    print("  TEST 1: Standard AST Round-Trip Validation")
    print("=" * 60)
    print("ANONYMIZED CODE:")
    print(anonymized_code)

    # Step 2: De-anonymize locally
    restored_code = deanonymize_code(anonymized_code, privacy_maps)

    print("\nRESTORED CODE:")
    print(restored_code)

    # Step 3: AST structural verification
    is_identical = verify_roundtrip(original_code, restored_code)
    assert is_identical, "FAIL: De-anonymized AST does not match original AST structure!"
    print("\n[PASS] AST structure match validated successfully!")


def test_backslash_literal_roundtrip():
    """
    Regression test for string literals containing backslashes
    (Windows paths, regex sequences, escape sequences).
    """
    # Note: Valid Python string literals with backslashes must use raw strings (r'...')
    original_code = r"""def get_path():
    path = r'C:\Users\Admin\data.txt'
    pattern = r'\d+\.\w+'
    return path, pattern"""

    protected_ids = ["get_path", "path", "pattern"]
    protected_lits = [r"C:\Users\Admin\data.txt", r"\d+\.\w+"]

    anonymized_code, privacy_maps = anonymize_code(
        code_snippet=original_code,
        filepath="utils/paths.py",
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
    )

    print("\n" + "=" * 60)
    print("  TEST 2: Backslash String Literal Edge Case")
    print("=" * 60)

    restored_code = deanonymize_code(anonymized_code, privacy_maps)
    
    is_identical = verify_roundtrip(original_code, restored_code)
    assert is_identical, "FAIL: Backslash string literal was corrupted during re.sub restoration!"
    print("\n[PASS] Backslash string literals restored perfectly without regex escape corruption!")


if __name__ == "__main__":
    test_roundtrip_fidelity()
    test_backslash_literal_roundtrip()
    print("\n" + "=" * 60)
    print("  ALL DAY 10 TESTS PASSED")
    print("=" * 60)