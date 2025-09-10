from pydantic import BaseModel

class InferenceIn(BaseModel):
    model_version_id: int
    mosaic_id: int

class InferenceOut(BaseModel):
    run_id: int
    task_id: str
    status: str
