import ast
from typing import Tuple, Optional

def verify_code_syntax(code_str: str, filename: str = "<string>") -> Tuple[bool, Optional[str]]:
    """
    Parses code into a Python AST to verify syntactic validity.
    Returns (True, None) if valid, or (False, error_message) if invalid.
    """
    try:
        ast.parse(code_str, filename=filename)
        return True, None
    except SyntaxError as e:
        error_msg = f"SyntaxError in {filename}:{e.lineno}:{e.offset} - {e.msg}"
        return False, error_msg