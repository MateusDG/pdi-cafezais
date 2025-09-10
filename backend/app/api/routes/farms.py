from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ...db.session import get_db
from ... import models, schemas

router = APIRouter()

@router.get("/", response_model=list[schemas.FarmOut])
def list_farms(db: Session = Depends(get_db)):
    return db.query(models.Farm).all()

@router.post("/", response_model=schemas.FarmOut)
def create_farm(payload: schemas.FarmIn, db: Session = Depends(get_db)):
    farm = models.Farm(name=payload.name, owner=payload.owner)
    db.add(farm)
    db.commit()
    db.refresh(farm)
    return farm
