"""Inventory management service."""
from models.product import Product


class InsufficientStockError(Exception):
    """Raised when an order requests more stock than available."""
    pass


def reserve_stock(product: Product, quantity):
    """Reserve stock for a product, raising if insufficient."""
    if product.stock_quantity < quantity:
        raise InsufficientStockError(
            f"Cannot reserve {quantity} of {product.name}, only {product.stock_quantity} available"
        )
    product.stock_quantity -= quantity
    return product.stock_quantity


def restock(product: Product, quantity):
    """Add quantity back to a product's available stock."""
    product.stock_quantity += quantity
    return product.stock_quantity