"""
Test Day 11: Structured Diff Parsing & Patch De-anonymization
"""

from agent.diff_parser import process_llm_diff_response


def test_diff_parsing_and_deanonymization():
    privacy_maps = {
        "reverse_identifier_map": {
            "__ANON_FUNC_1__": "calculate_salary",
            "__ANON_VAR_1__": "base_pay",
            "__ANON_VAR_2__": "incentive",
        },
        "reverse_literal_map": {
            "__ANON_CONST_1__": 0.5,
        },
    }

    simulated_llm_response = """
Here is the requested patch:

File: auth/payroll.py
<<<<<<< SEARCH
def __ANON_FUNC_1__(employee_grade, __ANON_VAR_1__):
    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__
    return __ANON_VAR_1__ + __ANON_VAR_2__
=======
def __ANON_FUNC_1__(employee_grade, __ANON_VAR_1__):
    __ANON_VAR_2__ = __ANON_CONST_1__ * __ANON_VAR_1__
    return (__ANON_VAR_1__ + __ANON_VAR_2__) * 1.10
>>>>>>> REPLACE
"""

    patches = process_llm_diff_response(simulated_llm_response, privacy_maps)

    print("=" * 60)
    print("  TEST 1: Diff Parsing & Local De-anonymization")
    print("=" * 60)
    assert len(patches) == 1
    patch = patches[0]

    assert patch.filepath == "auth/payroll.py"
    assert "calculate_salary" in patch.search_text
    assert "base_pay" in patch.search_text
    assert "0.5" in patch.search_text
    assert "__ANON_FUNC_1__" not in patch.search_text

    assert "calculate_salary" in patch.replace_text
    assert "1.1" in patch.replace_text
    print("\n[PASS] Patch successfully parsed and de-anonymized!")


def test_blank_lines_after_file_header():
    """
    Tests that blank lines inserted between 'File: path' and '<<<<<<< SEARCH'
    do not lose the target filepath.
    """
    privacy_maps = {"reverse_identifier_map": {}, "reverse_literal_map": {}}

    llm_response_with_blank_line = """
File: auth/payroll.py


<<<<<<< SEARCH
def foo():
    pass
=======
def foo():
    return True
>>>>>>> REPLACE
"""

    patches = process_llm_diff_response(llm_response_with_blank_line, privacy_maps)

    print("\n" + "=" * 60)
    print("  TEST 2: Blank Lines After File Header Tolerance")
    print("=" * 60)
    assert len(patches) == 1
    assert patches[0].filepath == "auth/payroll.py", (
        f"Expected 'auth/payroll.py', got '{patches[0].filepath}'"
    )
    print("\n[PASS] Successfully extracted filepath despite blank lines!")


def test_malformed_patch_rejection():
    privacy_maps = {"reverse_identifier_map": {}, "reverse_literal_map": {}}
    malformed_llm_response = "I updated calculate_salary to return 1.1 times base pay."

    print("\n" + "=" * 60)
    print("  TEST 3: Reject Malformed Diff Response")
    print("=" * 60)

    try:
        process_llm_diff_response(malformed_llm_response, privacy_maps)
        assert False, "FAIL: Malformed response should have raised a ValueError!"
    except ValueError as e:
        print(f"[PASS] Correctly rejected malformed response: {e}")


if __name__ == "__main__":
    test_diff_parsing_and_deanonymization()
    test_blank_lines_after_file_header()
    test_malformed_patch_rejection()
    print("\n" + "=" * 60)
    print("  ALL DAY 11 TESTS PASSED")
    print("=" * 60)