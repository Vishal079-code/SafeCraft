"""Payment processing utilities."""
from models.user import get_user_by_id, update_balance


def charge_user(db, user_id, amount):
    """Deduct an amount from a user's balance, raising if insufficient funds."""
    user = get_user_by_id(db, user_id)
    if user.balance < amount:
        raise ValueError("Insufficient balance")
    return update_balance(db, user_id, -amount)


def refund_user(db, user_id, amount):
    """Credit an amount back to a user's balance."""
    return update_balance(db, user_id, amount)


def calculate_late_fee(days_overdue, base_amount):
    """Calculate a late fee based on days overdue."""
    if days_overdue <= 0:
        return 0.0
    return round(base_amount * 0.02 * days_overdue, 2)
