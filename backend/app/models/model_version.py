from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from ..db.base import Base

class ModelVersion(Base):
    __tablename__ = "model_version"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    params_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(String, nullable=True)
