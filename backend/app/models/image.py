from sqlalchemy import Integer, String, Float
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry
from ..db.base import Base

class Image(Base):
    __tablename__ = "image"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    capture_dt: Mapped[str | None] = mapped_column(String(50), nullable=True)
    gsd_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    footprint_geom = mapped_column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
