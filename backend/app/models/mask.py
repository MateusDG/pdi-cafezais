from sqlalchemy import Integer, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry
from ..db.base import Base

class Mask(Base):
    __tablename__ = "mask"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    inference_run_id: Mapped[int] = mapped_column(ForeignKey("inference_run.id", ondelete="CASCADE"), nullable=False)
    uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    geom_polygon = mapped_column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    area_ha: Mapped[float | None] = mapped_column(Float, nullable=True)
