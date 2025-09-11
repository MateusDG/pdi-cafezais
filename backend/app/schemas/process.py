from pydantic import BaseModel

class ProcessResponse(BaseModel):
    result_image_url: str
    notes: str | None = None
