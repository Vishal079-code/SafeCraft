"""Product catalog model."""
from models.base import BaseModel


class Product(BaseModel):
    """Represents a purchasable product."""

    def __init__(self, id, name, price, stock_quantity, **kwargs):
        super().__init__(id=id, **kwargs)
        self.name = name
        self.price = price
        self.stock_quantity = stock_quantity

    @property
    def is_in_stock(self):
        """Whether this product currently has available stock."""
        return self.stock_quantity > 0

    def apply_discount(self, percent):
        """Apply a percentage discount to the product's price."""
        self.price = round(self.price * (1 - percent / 100), 2)
        return self.price

    def __repr__(self):
        return f"Product(id={self.id!r}, name={self.name!r}, price={self.price})"