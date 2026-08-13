"""
Test Day 9 Anonymizer: Identifier & Occurrence-Based Literal Protection
"""
from privacy.anonymizer import anonymize_code


def test_anonymization():
    sample_code = """
def calculate_salary(employee_grade, base_pay):
    incentive = 0.5 * base_pay
    tax_deduction = 0.5 * incentive
    return base_pay + incentive - tax_deduction
"""
    protected_ids = ["calculate_salary", "incentive"]
    protected_lits = [0.5]
    anonymized_code, privacy_maps = anonymize_code(
        code_snippet=sample_code,
        filepath="auth/payroll.py",
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
    )
    print("=" * 60)
    print("  TEST 1: Basic identifier + occurrence-based literal protection")
    print("=" * 60)
    print(anonymized_code)
    print("\n" + "-" * 60)
    print("  OCCURRENCE-BASED LITERAL MAP")
    print("-" * 60)
    for occ_id, meta in privacy_maps["literal_map"].items():
        print(f"Location: {occ_id:<25} -> Placeholder: {meta['placeholder']}")
    print("\n" + "-" * 60)
    print("  IDENTIFIER MAP")
    print("-" * 60)
    for orig, anon in privacy_maps["identifier_map"].items():
        print(f"Original: {orig:<20} -> Placeholder: {anon}")

    # Regression check: two occurrences of 0.5 must get DIFFERENT placeholders
    assert len(privacy_maps["literal_map"]) == 2, "Expected 2 distinct literal occurrences"
    placeholders = [m["placeholder"] for m in privacy_maps["literal_map"].values()]
    assert len(set(placeholders)) == 2, "Each occurrence of 0.5 must get its own placeholder"
    print("\n[PASS] Occurrence-based literal protection: two 0.5's got distinct placeholders")


def test_parameter_anonymization():
    """
    Regression test for the parameter-leak bug: a protected function
    PARAMETER must be anonymized consistently in both the signature and
    the body - never left visible in one place while hidden in another.
    """
    sample_code = """
def calculate_salary(employee_grade, base_pay):
    incentive = 0.5 * base_pay
    return base_pay + incentive
"""
    protected_ids = ["calculate_salary", "incentive", "base_pay"]
    protected_lits = [0.5]
    anonymized_code, privacy_maps = anonymize_code(
        code_snippet=sample_code,
        filepath="auth/payroll.py",
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
    )
    print("\n" + "=" * 60)
    print("  TEST 2: Parameter anonymization consistency")
    print("=" * 60)
    print(anonymized_code)

    assert "base_pay" not in anonymized_code, (
        "FAIL: 'base_pay' leaked into anonymized output - parameter was "
        "not anonymized consistently with its body usages"
    )
    print("\n[PASS] Protected parameter does not leak into anonymized output")


def test_attribute_anonymization():
    """
    Regression test for attribute access (e.g. employee.grade) - this was
    part of the original spec and previously completely unhandled, since
    ast.Attribute.attr is a plain string, not an ast.Name node.
    """
    sample_code = """
def check_bonus(employee):
    if employee.grade > 2:
        return employee.incentive
    return 0
"""
    protected_ids = ["grade", "incentive"]
    protected_lits = [2]
    anonymized_code, privacy_maps = anonymize_code(
        code_snippet=sample_code,
        filepath="auth/payroll.py",
        protected_identifiers=protected_ids,
        protected_literals=protected_lits,
    )
    print("\n" + "=" * 60)
    print("  TEST 3: Attribute access anonymization")
    print("=" * 60)
    print(anonymized_code)

    assert "employee.grade" not in anonymized_code, "FAIL: 'employee.grade' leaked"
    assert "employee.incentive" not in anonymized_code, "FAIL: 'employee.incentive' leaked"
    print("\n[PASS] Protected attribute names do not leak into anonymized output")


if __name__ == "__main__":
    test_anonymization()
    test_parameter_anonymization()
    test_attribute_anonymization()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)