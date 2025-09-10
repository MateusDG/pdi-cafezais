from pydantic import BaseModel

class ImageOut(BaseModel):
    id: int
    source: str | None = None
    capture_dt: str | None = None
    gsd_cm: float | None = None
    uri: str | None = None
    class Config:
        from_attributes = True
