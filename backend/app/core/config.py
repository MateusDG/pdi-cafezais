from pydantic import BaseModel
import os

class _Settings(BaseModel):
    ENV: str = os.getenv("ENV", "dev")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    OUTPUTS_DIR: str = os.getenv("OUTPUTS_DIR", "data/outputs")
    STATIC_RESULTS: str = os.getenv("STATIC_RESULTS", "backend/app/static/results")

settings = _Settings()
