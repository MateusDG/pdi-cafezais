from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ...db.session import get_db
from ... import models, schemas
from ...celery_app import run_inference_task

router = APIRouter()

@router.post("/", response_model=schemas.InferenceOut)
def trigger_inference(req: schemas.InferenceIn, db: Session = Depends(get_db)):
    # Create record
    run = models.InferenceRun(model_version_id=req.model_version_id, mosaic_id=req.mosaic_id)
    db.add(run)
    db.commit()
    db.refresh(run)

    # Async task (stub)
    task = run_inference_task.delay(run_id=run.id)

    return schemas.InferenceOut(run_id=run.id, task_id=task.id, status="queued")


@router.post("/geo")
def trigger_geo():
    task = run_inference_task.apply_async(kwargs={"run_id": 0}, queue="ml")
    # also call specific geo task
    from ...celery_app import run_inference_geo
    t2 = run_inference_geo.apply_async(kwargs={"image_path": None})
    return {"default_task_id": task.id, "geo_task_id": t2.id}

@router.post("/onnx")
def trigger_onnx():
    from ...celery_app import run_inference_onnx
    t = run_inference_onnx.apply_async(kwargs={"model_path": "ml_models/weights/model.onnx"})
    return {"task_id": t.id}

@router.post("/torch")
def trigger_torch():
    from ...celery_app import run_inference_torch
    t = run_inference_torch.apply_async(kwargs={"weights": "ml_models/weights/yolov8n.pt"})
    return {"task_id": t.id}
