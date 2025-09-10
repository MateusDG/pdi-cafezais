from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import shape
from ...db.session import get_db
from ... import models, schemas

router = APIRouter()

@router.get("/", response_model=list[schemas.PlotOut])
def list_plots(db: Session = Depends(get_db)):
    return db.query(models.Plot).all()

@router.post("/", response_model=schemas.PlotOut)
def create_plot(payload: schemas.PlotIn, db: Session = Depends(get_db)):
    try:
        geom = from_shape(shape(payload.geom_geojson), srid=4326)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid geometry: {e}")
    plot = models.Plot(farm_id=payload.farm_id, geom_polygon=geom, area_ha=payload.area_ha)
    db.add(plot)
    db.commit()
    db.refresh(plot)
    return plot
