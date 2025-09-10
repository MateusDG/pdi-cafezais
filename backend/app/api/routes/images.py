from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from ...db.session import get_db
from ... import models, schemas
from ...services.s3 import S3Client

router = APIRouter()

@router.post("/upload", response_model=schemas.ImageOut)
async def upload_image(
    source: str,
    capture_dt: str,
    gsd_cm: float,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    s3: S3Client = Depends(S3Client.depends)
):
    # Simple stub upload -> S3 (MinIO)
    key = f"raw/{file.filename}"
    try:
        content = await file.read()
        s3.put_object(key, content, content_type=file.content_type)
    except Exception as e:
        raise HTTPException(500, f"S3 upload failed: {e}")
    image = models.Image(
        source=source,
        capture_dt=capture_dt,
        gsd_cm=gsd_cm,
        footprint_geom=None,
        uri=key
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image
