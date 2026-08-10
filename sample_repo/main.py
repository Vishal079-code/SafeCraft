"""Application entrypoint."""
from routes.api import (
    handle_login_request,
    handle_register_request,
    handle_charge_request,
    handle_profile_request,
)


def create_app(db):
    """Wire up route handlers to a minimal app dispatcher."""
    routes = {
        "/login": handle_login_request,
        "/register": handle_register_request,
        "/charge": handle_charge_request,
        "/profile": handle_profile_request,
    }
    return routes


if __name__ == "__main__":
    print("Sample app ready (this is a demo repo for parsing/testing, not runnable as-is).")
