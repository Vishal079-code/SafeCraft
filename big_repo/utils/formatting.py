"""Formatting helpers used across the API layer."""

def format_currency(amount):
    """Format a numeric amount as a currency string."""
    return f"${amount:.2f}"


def format_percentage(fraction):
    """Format a fraction as a percentage string."""
    return f"{fraction * 100:.1f}%"