from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil, uuid, time
import logging
from typing import Optional

from app.core.config import settings
from app.services.processing import weed, utils, robust_detection

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/process")
async def process_image(
    file: UploadFile = File(...), 
    sensitivity: float = Query(0.5, ge=0.0, le=1.0, description="Sensibilidade de detecção (0.0-1.0)"),
    algorithm: str = Query("vegetation_indices", description="Algoritmo: 'vegetation_indices' (robusto) ou 'hsv_fallback'"),
    normalize_illumination: bool = Query(True, description="Aplicar normalização de iluminação"),
    primary_index: str = Query("ExGR", description="Índice primário: 'ExG', 'ExGR', ou 'CIVE'"),
    row_spacing_px: Optional[int] = Query(None, description="Espaçamento entre linhas em pixels (auto se None)")
):
    """
    Processa imagem de cafezal para detectar ervas daninhas.
    
    Args:
        file: Arquivo de imagem (JPG, JPEG, PNG, BMP, TIFF)
        sensitivity: Sensibilidade de detecção (0.0 = menos sensível, 1.0 = mais sensível)
        
    Returns:
        JSON com resultado da análise e URL da imagem anotada
    """
    start_time = time.time()
    
    # Validate file format
    if not utils.validate_image_format(file.filename):
        raise HTTPException(
            status_code=400, 
            detail="Formato não suportado. Use: JPG, JPEG, PNG, BMP, TIFF"
        )
    
    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Tamanho máximo: 50MB. Tamanho atual: {file_size/1024/1024:.1f}MB"
        )
    
    # Reset file pointer
    await file.seek(0)
    
    try:
        # Create directories
        upload_dir = Path(settings.OUTPUTS_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        unique_filename = utils.generate_unique_filename("upload", "jpg")
        tmp_path = upload_dir / unique_filename
        
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Processing image: {tmp_path.name}, size: {file_size/1024/1024:.1f}MB")
        
        # Load and analyze image
        img_bgr = utils.imread(str(tmp_path))
        
        # Resize if too large for processing
        img_bgr, scale_factor = utils.resize_image_if_needed(img_bgr, max_size=2048)
        
        # Convert BGR to RGB for processing
        img_rgb = utils.bgr_to_rgb(img_bgr)
        
        # Calculate image statistics
        image_stats = utils.calculate_image_stats(img_bgr)
        
        # Perform robust weed detection
        weed_data = robust_detection.detect_weeds_robust_pipeline(
            img=img_rgb, 
            sensitivity=sensitivity,
            algorithm=algorithm,
            normalize_illumination=normalize_illumination,
            primary_index=primary_index,
            row_spacing_px=row_spacing_px
        )
        
        # Convert annotated image back to BGR for saving
        img_annotated_bgr = utils.rgb_to_bgr(weed_data['annotated_image'])
        
        # Save result
        result_filename = utils.generate_unique_filename("weed_analysis", "jpg")
        result_path = Path(settings.STATIC_RESULTS) / result_filename
        result_path.parent.mkdir(parents=True, exist_ok=True)
        
        utils.imwrite(str(result_path), img_annotated_bgr)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create comprehensive summary
        summary = utils.create_processing_summary(weed_data, processing_time, image_stats)
        summary['weed_detection']['detection_sensitivity'] = sensitivity
        summary['scale_factor'] = scale_factor
        
        # Convert contours to polygons for frontend
        polygons = weed.get_contour_polygons(weed_data['contours'])
        
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)
        
        logger.info(f"Processing completed in {processing_time:.2f}s. Detected {weed_data['weed_count']} weed areas.")
        
        response_data = {
            "success": True,
            "result_image_url": f"/static/results/{result_filename}",
            "summary": summary,
            "weed_polygons": polygons,
            "analysis_notes": f"Análise concluída em {processing_time:.1f}s. Detectadas {weed_data['weed_count']} áreas de ervas daninhas ({weed_data['weed_percentage']:.1f}% da imagem).",
            "processing_parameters": {
                "sensitivity": sensitivity,
                "algorithm": algorithm,
                "normalize_illumination": normalize_illumination,
                "primary_index": primary_index,
                "row_spacing_px": row_spacing_px,
                "version": "2.0_robust"
            }
        }
        
        return JSONResponse(response_data)
        
    except FileNotFoundError as e:
        logger.error(f"File error: {str(e)}")
        raise HTTPException(status_code=404, detail="Erro ao processar arquivo de imagem")
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        
        # Clean up on error
        if 'tmp_path' in locals():
            tmp_path.unlink(missing_ok=True)
            
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno no processamento: {str(e)}"
        )


@router.get("/process/status")
async def get_processing_status():
    """
    Retorna informações sobre o status do sistema de processamento.
    """
    algorithm_info = robust_detection.get_algorithm_info()
    
    return JSONResponse({
        "status": "operational",
        "algorithms_available": list(algorithm_info['algorithms'].keys()),
        "algorithm_details": algorithm_info['algorithms'],
        "supported_formats": ["JPG", "JPEG", "PNG", "BMP", "TIFF"],
        "max_file_size_mb": 50,
        "max_image_dimension": 2048,
        "vegetation_indices": ["ExG", "ExGR", "CIVE"],
        "normalization_pipeline": algorithm_info['normalization_pipeline'],
        "parameters": algorithm_info['parameters'],
        "version": "2.0.0_robust"
    })
