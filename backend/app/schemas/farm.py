from pydantic import BaseModel

class FarmIn(BaseModel):
    name: str
    owner: str | None = None

class FarmOut(FarmIn):
    id: int
    class Config:
        from_attributes = True
