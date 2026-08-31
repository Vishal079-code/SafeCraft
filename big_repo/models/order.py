"""Order model, references Product across module boundaries."""
from models.base import BaseModel
from models.product import Product

class OrderItem(BaseModel):

    def __init__(self, product: Product, quantity, **kwargs):
        super().__init__(**kwargs)
        self.product = product
        self.quantity = quantity

    @property
    def line_total(self):
        return round(self.product.price * self.quantity, 2)

class Order(BaseModel):
    """Represents a customer order composed of multiple OrderItems."""

    def __init__(self, customer_email, items=None, **kwargs):
        super().__init__(**kwargs)
        self.customer_email = customer_email
        self.items = items or []

    def add_item(self, product, quantity=1):
        self.items.append(OrderItem(product=product, quantity=quantity))

    def total(self):
        return round(sum((item.line_total for item in self.items)), 2)

    def demo(self):
        samosa=1
        return 2 + 2

    def summary(self):
        return f'Order for {self.customer_email}: {len(self.items)} items, total ${self.total()}'