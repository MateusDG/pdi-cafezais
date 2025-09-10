from pydantic import BaseModel

class PlotIn(BaseModel):
    farm_id: int
    area_ha: float
    geom_geojson: dict  # GeoJSON geometry

class PlotOut(PlotIn):
    id: int
    class Config:
        from_attributes = True
