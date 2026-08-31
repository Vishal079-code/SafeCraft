"""
Privacy De-anonymizer: Reverses identifier and literal placeholders back
to their original identifiers and values using local privacy maps.
"""
import ast
import re
from typing import Dict, Any


class DeanonymizationError(Exception):
    """Raised when a placeholder-shaped token survives de-anonymization -
    i.e. the LLM emitted something matching the __ANON_*__ pattern that
    was never actually assigned in privacy_maps (hallucinated or mangled
    placeholder), so it couldn't be reversed."""
    pass


def deanonymize_code(anonymized_text: str, privacy_maps: Dict[str, Any]) -> str:
    """
    Restores original code by substituting placeholder tokens back to their
    original identifiers and literal values using local maps.
    """
    reverse_identifiers = privacy_maps.get("reverse_identifier_map", {})
    reverse_literals = privacy_maps.get("reverse_literal_map", {})
    result = anonymized_text

    # 1. Restore identifiers (exact word boundary replacement).
    # Identifier names are always plain alphanumeric/underscore text, so a
    # plain string replacement is safe here - no backslash risk.
    for placeholder, orig_name in reverse_identifiers.items():
        pattern = rf"\b{re.escape(placeholder)}\b"
        result = re.sub(pattern, orig_name, result)

    # 2. Restore literals (formatted as valid Python literal representations).
    #
    # Must use a replacement FUNCTION here, not a replacement string.
    # re.sub() treats backslashes in a STRING replacement specially (as
    # backreferences like \1, \g<name>). repr() of a string containing
    # backslashes (e.g. a Windows path or regex pattern) produces text
    # full of backslashes - re.sub would then misinterpret those as
    # backreference syntax and silently corrupt the restored value instead
    # of erroring, breaking the round-trip. A replacement function's
    # return value is inserted literally, with no escape re-interpretation.
    for placeholder, orig_val in reverse_literals.items():
        pattern = rf"\b{re.escape(placeholder)}\b"
        formatted_val = repr(orig_val)
        result = re.sub(pattern, lambda m, v=formatted_val: v, result)

    # 3. Safety net: confirm no placeholder-shaped token survived. This
    # catches an LLM emitting a placeholder that was never actually
    # assigned in privacy_maps (hallucinated, or a mangled copy of a real
    # one) - previously this passed through silently, leaving a literal
    # "__ANON_..._N__" string sitting in otherwise-normal code with no
    # warning to the caller.
    leftover = re.findall(r"__ANON_[A-Z]+_\d+__", result)
    if leftover:
        raise DeanonymizationError(
            f"Unresolved placeholder(s) survived de-anonymization: {sorted(set(leftover))}"
        )

    return result


def verify_roundtrip(original_code: str, deanonymized_code: str) -> bool:
    """
    Validates structural identity by comparing AST dumps of the original
    and de-anonymized code snippets.
    """
    try:
        orig_ast = ast.parse(original_code)
        deanon_ast = ast.parse(deanonymized_code)
        return ast.dump(orig_ast) == ast.dump(deanon_ast)
    except SyntaxError:
        return False