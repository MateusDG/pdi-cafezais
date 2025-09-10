from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry
from ..db.base import Base

class Mosaic(Base):
    __tablename__ = "mosaic"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    footprint_geom = mapped_column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
