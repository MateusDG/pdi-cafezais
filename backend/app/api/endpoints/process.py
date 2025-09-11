from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil, uuid

from app.core.config import settings
from app.services.processing import weed, utils

router = APIRouter()

@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Envie uma imagem .jpg, .jpeg ou .png")

    upload_dir = Path(settings.OUTPUTS_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result_path = Path(settings.STATIC_RESULTS) / f"{tmp_path.stem}_annotated.png"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    img = utils.imread(str(tmp_path))
    img_annot = weed.placeholder_annotate(img)  # TODO: substituir por lógica real
    utils.imwrite(str(result_path), img_annot)

    return JSONResponse({
        "result_image_url": f"/static/results/{result_path.name}",
        "notes": "Stub de processamento executado. Substitua por algoritmos reais em services/processing."
    })
