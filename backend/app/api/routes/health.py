from fastapi import APIRouter
from ..schemas.common import HealthOut

router = APIRouter()

@router.get("/", response_model=HealthOut)
def health():
    return HealthOut(status="ok")
