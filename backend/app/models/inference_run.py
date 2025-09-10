from sqlalchemy import Integer, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from ..db.base import Base

class InferenceRun(Base):
    __tablename__ = "inference_run"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_version.id"), nullable=False)
    mosaic_id: Mapped[int] = mapped_column(ForeignKey("mosaic.id"), nullable=False)
    metrics_json: Mapped[str | None] = mapped_column(String, nullable=True)
