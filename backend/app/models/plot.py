from sqlalchemy import Integer, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry
from ..db.base import Base

class Plot(Base):
    __tablename__ = "plot"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    farm_id: Mapped[int] = mapped_column(ForeignKey("farm.id", ondelete="CASCADE"), nullable=False)
    geom_polygon = mapped_column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    area_ha: Mapped[float] = mapped_column(Float, nullable=True)
