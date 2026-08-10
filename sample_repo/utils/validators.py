"""Input validation helpers. Deliberately has no dependencies on other
modules — useful later for testing that retrieval doesn't over-fetch
unrelated files."""
import re


def is_valid_email(email):
    """Check whether a string looks like a valid email address."""
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None


def is_strong_password(password):
    """Check whether a password meets minimum strength requirements."""
    return len(password) >= 8 and any(c.isdigit() for c in password)


def sanitize_input(text):
    """Strip potentially dangerous characters from user input."""
    return re.sub(r"[<>;]", "", text).strip()
