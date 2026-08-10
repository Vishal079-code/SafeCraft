"""Authentication: login, password hashing, session tokens."""
import hashlib
from models.user import get_user_by_id, create_user


def hash_password(raw_password):
    """Hash a plaintext password using SHA-256 (demo only, not production-grade)."""
    return hashlib.sha256(raw_password.encode()).hexdigest()


def verify_password(raw_password, password_hash):
    """Check whether a plaintext password matches the stored hash."""
    return hash_password(raw_password) == password_hash


def login(db, email, raw_password, user_lookup_by_email):
    """Attempt to log a user in, returning the user object on success."""
    user = user_lookup_by_email(db, email)
    if user is None:
        return None
    if verify_password(raw_password, user.password_hash):
        return user
    return None


def register(db, email, raw_password):
    """Register a new user account with a hashed password."""
    password_hash = hash_password(raw_password)
    return create_user(db, email, password_hash)
