"""API route handlers wiring together services and models."""
from services.pricing import calculate_order_total
from services.inventory import reserve_stock, InsufficientStockError
from models.order import Order
from models.product import Product


def handle_checkout_request(order: Order):
    """Process a checkout request: reserve stock, calculate total."""
    try:
        for item in order.items:
            reserve_stock(item.product, item.quantity)
    except InsufficientStockError as e:
        return {"error": str(e)}, 400

    total = calculate_order_total(order)
    return {"order_summary": order.summary(), "total": total}, 200


def handle_apply_discount_request(product: Product, percent):
    """Apply a discount to a product and return its new price."""
    new_price = product.apply_discount(percent)
    return {"product_id": product.id, "new_price": new_price}, 200