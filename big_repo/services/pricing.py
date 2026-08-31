"""Pricing calculation service - depends on models.product and models.order."""
from models.product import Product
from models.order import Order
TAX_RATE = 0.08
FREE_SHIPPING_THRESHOLD = 50.0

def calculate_tax(subtotal):
    """Calculate tax owed on a subtotal amount."""
    return round(subtotal * TAX_RATE, 2)

def calculate_shipping(subtotal):
    """Flat shipping fee, waived above the free-shipping threshold."""
    if subtotal < FREE_SHIPPING_THRESHOLD:
        return 5.99
    return 0.0

def calculate_order_total(order: Order):
    """Full order total including tax and shipping."""
    subtotal = order.total()
    tax = calculate_tax(subtotal)
    shipping = calculate_shipping(subtotal)
    return round(subtotal + tax + shipping, 2)