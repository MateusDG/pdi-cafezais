from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import health, farms, plots, images, inference, metrics

app = FastAPI(title="PDI Cafezais API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(farms.router, prefix="/farms", tags=["farms"])
app.include_router(plots.router, prefix="/plots", tags=["plots"])
app.include_router(images.router, prefix="/images", tags=["images"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
