"""API route handlers wiring together auth and payments."""
from auth.login import login, register
from utils.payments import charge_user, refund_user
from models.user import get_user_by_id


def handle_login_request(db, request_data):
    """Handle POST /login."""
    email = request_data.get("email")
    password = request_data.get("password")
    user = login(db, email, password, lambda d, e: None)
    if user is None:
        return {"error": "Invalid credentials"}, 401
    return {"user": user.to_dict()}, 200


def handle_register_request(db, request_data):
    """Handle POST /register."""
    email = request_data.get("email")
    password = request_data.get("password")
    user = register(db, email, password)
    return {"user": user.to_dict()}, 201


def handle_charge_request(db, request_data):
    """Handle POST /charge."""
    user_id = request_data.get("user_id")
    amount = request_data.get("amount")
    updated_user = charge_user(db, user_id, amount)
    return {"user": updated_user.to_dict()}, 200


def handle_profile_request(db, user_id):
    """Handle GET /profile/<user_id>."""
    user = get_user_by_id(db, user_id)
    if user is None:
        return {"error": "User not found"}, 404
    return {"user": user.to_dict()}, 200
