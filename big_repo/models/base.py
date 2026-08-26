"""Base model classes shared across the application."""
from dataclasses import dataclass, field
from typing import Optional
import datetime


class BaseModel:
    """Base class all domain models inherit from."""

    def __init__(self, id=None, created_at=None):
        self.id = id
        self.created_at = created_at or datetime.datetime.utcnow()

    def to_dict(self):
        return {"id": self.id, "created_at": str(self.created_at)}


@dataclass
class AuditRecord:
    """Tracks who changed a record and when."""
    actor_id: int
    action: str
    timestamp: str = field(default_factory=lambda: str(datetime.datetime.utcnow()))