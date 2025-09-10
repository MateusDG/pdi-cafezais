from fastapi import APIRouter
router = APIRouter()

@router.get("/")
def metrics():
    # TODO: integrate Prometheus or similar
    return {"detail": "metrics endpoint - TODO"}
