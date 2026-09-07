"""LAND-JEPA — User and Role ORM models."""
from __future__ import annotations

import uuid
from typing import List

from sqlalchemy import Boolean, ForeignKey, Integer, String, Table, Column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDMixin


# ── Many-to-many: users ↔ roles ──────────────────────────────────────
user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class Role(Base):
    """System roles: admin | authority | field_officer | citizen."""

    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(String(255))

    users: Mapped[List["User"]] = relationship(
        "User", secondary=user_roles, back_populates="roles"
    )

    def __repr__(self) -> str:
        return f"<Role {self.name}>"


class User(Base, UUIDMixin, TimestampMixin):
    """Application user — all roles share this table."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(20))
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preferred_lang: Mapped[str] = mapped_column(String(10), nullable=False, default="en")

    roles: Mapped[List[Role]] = relationship(
        "Role", secondary=user_roles, back_populates="users", lazy="selectin"
    )

    def has_role(self, role_name: str) -> bool:
        return any(r.name == role_name for r in self.roles)

    def __repr__(self) -> str:
        return f"<User {self.email}>"
