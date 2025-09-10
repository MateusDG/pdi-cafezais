from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geometry
from ..db.base import Base

class Label(Base):
    __tablename__ = "label"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    plot_id: Mapped[int] = mapped_column(ForeignKey("plot.id", ondelete="CASCADE"), nullable=False)
    geom_polygon = mapped_column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    annotator: Mapped[str | None] = mapped_column(String(100), nullable=True)
    version: Mapped[str | None] = mapped_column(String(50), nullable=True)
