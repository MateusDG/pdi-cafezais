from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from ..db.base import Base

class Farm(Base):
    __tablename__ = "farm"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    owner: Mapped[str | None] = mapped_column(String(200), nullable=True)
