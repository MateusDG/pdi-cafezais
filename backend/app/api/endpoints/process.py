from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil, uuid, time
import logging
from typing import Optional

from app.core.config import settings
from app.services.processing import weed, utils, robust_detection, oblique_pipeline
from app.services.processing.ml_classifiers import ClassicalMLWeedDetector
from app.services.processing.ml_training import MLTrainingPipeline, TrainingConfig

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/process")
async def process_image(
    file: UploadFile = File(...), 
    sensitivity: float = Query(0.5, ge=0.0, le=1.0, description="Sensibilidade de detecção (0.0-1.0)"),
    algorithm: str = Query("oblique_pipeline", description="Algoritmo: 'oblique_pipeline' (completo), 'robust_exgr' (ExGR+Otsu permissivo), 'robust_exgr_v1' (ExGR+Otsu conservador), 'vegetation_indices' (robusto), 'hsv_fallback', 'ml_svm', 'ml_random_forest', 'ml_knn', 'ml_naive_bayes'"),
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
        
        # Perform weed detection based on algorithm choice
        if algorithm == "oblique_pipeline":
            weed_data = oblique_pipeline.oblique_weed_detection_pipeline(
                img=img_rgb, 
                sensitivity=sensitivity,
                normalize_illumination=normalize_illumination,
                primary_index=primary_index,
                row_spacing_px=row_spacing_px
            )
        elif algorithm == "robust_exgr":
            # Use new robust ExGR pipeline (permissive version)
            weed_data = weed.detect_weeds_robust(img_rgb, sensitivity)
        elif algorithm == "robust_exgr_v1":
            # Use conservative version of robust ExGR pipeline  
            weed_data = weed.detect_weeds_robust_v1(img_rgb, sensitivity)
        elif algorithm in ["ml_svm", "ml_random_forest", "ml_knn", "ml_naive_bayes"]:
            # Use classical ML algorithms
            ml_detector = ClassicalMLWeedDetector()
            ml_detector.load_models()
            model_name = algorithm.replace("ml_", "")
            weed_data = ml_detector.detect_weeds_ml(img_rgb, model_name, confidence_threshold=0.6)
        else:
            # Use robust detection for backward compatibility
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


@router.post("/ml/train")
async def train_ml_models(
    files: list[UploadFile] = File(...),
    samples_per_class: int = Query(500, ge=100, le=2000, description="Número de amostras por classe"),
    patch_size: int = Query(64, ge=32, le=128, description="Tamanho do patch quadrado"),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Proporção para teste")
):
    """
    Treina modelos de Machine Learning clássico para detecção de ervas daninhas.
    
    Args:
        files: Lista de imagens para treinamento (JPG, JPEG, PNG)
        samples_per_class: Número de amostras por classe (weed, coffee, soil)
        patch_size: Tamanho dos patches quadrados para extração
        test_size: Proporção dos dados para teste
        
    Returns:
        Resultados do treinamento de todos os modelos
    """
    try:
        # Validar arquivos
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for file in files:
            if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Formato não suportado: {file.filename}. Use: JPG, JPEG, PNG, BMP, TIFF"
                )
        
        if len(files) < 3:
            raise HTTPException(
                status_code=400,
                detail="Mínimo de 3 imagens necessárias para treinamento"
            )
        
        # Criar diretórios temporários
        upload_dir = Path(settings.OUTPUTS_DIR) / "ml_training"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar arquivos temporários
        temp_paths = []
        for file in files:
            unique_filename = utils.generate_unique_filename(f"train_{file.filename}", "jpg")
            temp_path = upload_dir / unique_filename
            
            with temp_path.open("wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append(str(temp_path))
        
        logger.info(f"Starting ML training with {len(temp_paths)} images")
        
        # Configurar treinamento
        config = TrainingConfig(
            patch_size=(patch_size, patch_size),
            samples_per_class=samples_per_class,
            test_size=test_size,
            random_state=42
        )
        
        # Criar pipeline e treinar
        pipeline = MLTrainingPipeline()
        results = pipeline.train_models_from_images(temp_paths, config)
        
        # Limpar arquivos temporários
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)
        
        # Preparar resposta
        training_summary = {}
        for model_name, model_result in results.items():
            training_summary[model_name] = {
                "accuracy": round(model_result['accuracy'], 3),
                "cross_validation_mean": round(model_result['cv_mean'], 3),
                "cross_validation_std": round(model_result['cv_std'], 3),
                "best_parameters": model_result.get('best_params', {}),
                "top_features": dict(list(model_result.get('feature_importance', {}).items())[:5]) if 'feature_importance' in model_result else None
            }
        
        logger.info(f"ML training completed successfully")
        
        return JSONResponse({
            "success": True,
            "message": "Modelos treinados com sucesso",
            "models_trained": list(results.keys()),
            "training_summary": training_summary,
            "configuration": {
                "samples_per_class": samples_per_class,
                "patch_size": patch_size,
                "test_size": test_size,
                "images_used": len(files)
            }
        })
        
    except Exception as e:
        logger.error(f"ML training error: {str(e)}")
        
        # Limpar arquivos em caso de erro
        if 'temp_paths' in locals():
            for temp_path in temp_paths:
                Path(temp_path).unlink(missing_ok=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro no treinamento ML: {str(e)}"
        )


@router.get("/ml/status")
async def get_ml_status():
    """
    Retorna status dos modelos de Machine Learning.
    """
    try:
        ml_detector = ClassicalMLWeedDetector()
        ml_detector.load_models()
        
        # Verificar quais modelos estão disponíveis
        available_models = []
        model_info = {}
        
        for model_name, model in ml_detector.models.items():
            if model is not None:
                available_models.append(model_name)
                model_info[model_name] = {
                    "type": type(model).__name__,
                    "ready": True
                }
            else:
                model_info[model_name] = {
                    "type": "Not trained",
                    "ready": False
                }
        
        return JSONResponse({
            "ml_models_available": len(available_models) > 0,
            "available_models": available_models,
            "model_details": model_info,
            "total_features_extracted": len(ml_detector.feature_extractor.get_all_feature_names()),
            "feature_categories": {
                "color_features": len(ml_detector.feature_extractor._get_color_feature_names()),
                "texture_features": len(ml_detector.feature_extractor._get_texture_feature_names()),
                "shape_features": len(ml_detector.feature_extractor._get_shape_feature_names())
            }
        })
        
    except Exception as e:
        logger.error(f"ML status error: {str(e)}")
        return JSONResponse({
            "ml_models_available": False,
            "error": str(e),
            "available_models": [],
            "model_details": {}
        })


@router.post("/ml/evaluate")
async def evaluate_ml_models(
    files: list[UploadFile] = File(...),
    model_name: str = Query("random_forest", description="Modelo para avaliação: svm, random_forest, knn, naive_bayes")
):
    """
    Avalia modelo de ML em imagens de teste.
    
    Args:
        files: Lista de imagens para avaliação
        model_name: Nome do modelo a avaliar
        
    Returns:
        Resultados da avaliação
    """
    try:
        # Validar modelo
        valid_models = ['svm', 'random_forest', 'knn', 'naive_bayes']
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Modelo inválido. Use: {', '.join(valid_models)}"
            )
        
        # Validar arquivos
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for file in files:
            if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
                raise HTTPException(
                    status_code=400,
                    detail=f"Formato não suportado: {file.filename}"
                )
        
        # Criar diretório temporário
        eval_dir = Path(settings.OUTPUTS_DIR) / "ml_evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar arquivos temporários
        temp_paths = []
        for file in files:
            unique_filename = utils.generate_unique_filename(f"eval_{file.filename}", "jpg")
            temp_path = eval_dir / unique_filename
            
            with temp_path.open("wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append(str(temp_path))
        
        logger.info(f"Evaluating ML model {model_name} on {len(temp_paths)} images")
        
        # Avaliar modelo
        pipeline = MLTrainingPipeline()
        eval_results = pipeline.evaluate_models(temp_paths)
        
        # Limpar arquivos temporários
        for temp_path in temp_paths:
            Path(temp_path).unlink(missing_ok=True)
        
        # Extrair resultados do modelo específico
        if model_name in eval_results['summary']:
            model_results = eval_results['summary'][model_name]
            detailed_results = eval_results['detailed_results'][model_name]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {model_name} não encontrado ou não treinado"
            )
        
        return JSONResponse({
            "success": True,
            "model_evaluated": model_name,
            "summary": {
                "average_weed_count": round(model_results['avg_weed_count'], 2),
                "average_weed_percentage": round(model_results['avg_weed_percentage'], 2),
                "average_confidence": round(model_results['avg_confidence'], 3),
                "images_processed": model_results['images_processed']
            },
            "detailed_results": detailed_results
        })
        
    except Exception as e:
        logger.error(f"ML evaluation error: {str(e)}")
        
        # Limpar arquivos em caso de erro
        if 'temp_paths' in locals():
            for temp_path in temp_paths:
                Path(temp_path).unlink(missing_ok=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro na avaliação ML: {str(e)}"
        )
