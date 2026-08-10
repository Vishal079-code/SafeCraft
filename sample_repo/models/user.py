"""User account model and related database operations."""


class User:
    """Represents a registered user account."""

    def __init__(self, user_id, email, password_hash, balance=0.0):
        self.user_id = user_id
        self.email = email
        self.password_hash = password_hash
        self.balance = balance

    def to_dict(self):
        """Serialize user object to a plain dict for API responses."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "balance": self.balance,
        }


def get_user_by_id(db, user_id):
    """Fetch a single user record by ID from the database."""
    return db.query(User).filter_by(user_id=user_id).first()


def create_user(db, email, password_hash):
    """Insert a new user record into the database."""
    new_user = User(user_id=None, email=email, password_hash=password_hash)
    db.add(new_user)
    db.commit()
    return new_user


def update_balance(db, user_id, amount):
    """Adjust a user's account balance by the given amount."""
    user = get_user_by_id(db, user_id)
    user.balance += amount
    db.commit()
    return user
