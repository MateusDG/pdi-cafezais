import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from . import vegetation_indices as vi


def detect_weeds_robust_pipeline(img: np.ndarray, 
                                sensitivity: float = 0.5,
                                algorithm: str = 'vegetation_indices',
                                normalize_illumination: bool = True,
                                primary_index: str = 'ExGR',
                                row_spacing_px: Optional[int] = None,
                                row_width_px: Optional[int] = None) -> Dict[str, Any]:
    """
    Detecção robusta de ervas daninhas usando índices de vegetação e geometria de plantio.
    
    Args:
        img: Imagem RGB de entrada
        sensitivity: Sensibilidade de detecção (0.0-1.0) - afeta thresholds
        algorithm: Tipo de algoritmo ('vegetation_indices' ou 'hsv_fallback')
        normalize_illumination: Se aplicar normalização de iluminação
        primary_index: Índice principal ('ExG', 'ExGR', 'CIVE')
        row_spacing_px: Espaçamento entre linhas (auto se None)
        row_width_px: Largura das copas (auto se None)
        
    Returns:
        Dict com resultados detalhados da análise
    """
    
    # Auto-estimate parameters based on image size and sensitivity
    h, w = img.shape[:2]
    
    if row_spacing_px is None:
        # Estimate based on image size (typical coffee rows)
        row_spacing_px = int(max(60, min(120, w // 8)) * (0.8 + sensitivity * 0.4))
    
    if row_width_px is None:
        # Estimate coffee tree crown width
        row_width_px = int(row_spacing_px * 0.4 * (0.7 + sensitivity * 0.6))
    
    try:
        if algorithm == 'vegetation_indices':
            # Use robust vegetation indices pipeline
            pipeline_results = vi.robust_weed_detection_pipeline(
                img=img,
                normalize_illumination=normalize_illumination,
                primary_index=primary_index,
                row_spacing_px=row_spacing_px,
                row_width_px=row_width_px
            )
            
            # Extract results
            weed_stats = pipeline_results['weed_statistics']
            vegetation_mask = pipeline_results['vegetation_mask']
            row_mask = pipeline_results['row_mask']
            
            # Calculate enhanced statistics
            image_area = h * w
            vegetation_area = cv2.countNonZero(vegetation_mask)
            row_area = cv2.countNonZero(row_mask)
            
            vegetation_percentage = (vegetation_area / image_area) * 100
            crop_percentage = (row_area / image_area) * 100
            bare_soil_percentage = 100 - vegetation_percentage
            
            # Calculate area statistics for weeds
            contour_areas = [cv2.contourArea(c) for c in weed_stats['contours']]
            area_stats = {}
            if contour_areas:
                area_stats = {
                    'min_area': int(min(contour_areas)),
                    'max_area': int(max(contour_areas)),
                    'avg_area': round(sum(contour_areas) / len(contour_areas), 1),
                    'median_area': int(np.median(contour_areas))
                }
            
            # Severity classification
            severity = classify_weed_severity_robust(weed_stats['weed_percentage'], crop_percentage)
            
            # Density calculation
            density_per_sqm = len(weed_stats['contours']) / (image_area / 10000) if image_area > 0 else 0
            
            return {
                'annotated_image': pipeline_results['annotated_image'],
                'contours': weed_stats['contours'],
                'weed_count': weed_stats['weed_count'],
                'total_weed_area': weed_stats['total_weed_area'],
                'weed_percentage': round(weed_stats['weed_percentage'], 2),
                'coffee_percentage': round(crop_percentage, 2),
                'vegetation_percentage': round(vegetation_percentage, 2),
                'bare_soil_percentage': round(bare_soil_percentage, 2),
                'image_area': image_area,
                'area_stats': area_stats,
                'severity_level': severity,
                'density_per_sqm': round(density_per_sqm, 2),
                'sensitivity_used': sensitivity,
                'algorithm': f'{algorithm}_{primary_index}',
                'processing_details': {
                    'illumination_normalized': normalize_illumination,
                    'primary_index': primary_index,
                    'dominant_angle': pipeline_results['dominant_angle'],
                    'row_spacing_px': row_spacing_px,
                    'row_width_px': row_width_px,
                    'vegetation_indices_used': list(pipeline_results['vegetation_indices'].keys())
                },
                'mask': pipeline_results['weed_mask'],
                'debug_masks': {
                    'vegetation_mask': pipeline_results['vegetation_mask'],
                    'row_mask': pipeline_results['row_mask'],
                    'processed_image': pipeline_results['processed_image']
                }
            }
            
    except Exception as e:
        # Fallback to HSV method if vegetation indices fail
        print(f"Vegetation indices failed: {e}, falling back to HSV")
        return detect_weeds_hsv_fallback(img, sensitivity)


def detect_weeds_hsv_fallback(img: np.ndarray, sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    Método HSV de fallback caso o pipeline robusto falhe.
    """
    from . import weed
    return weed.detect_weeds_hsv(img, sensitivity)


def classify_weed_severity_robust(weed_percentage: float, crop_percentage: float) -> str:
    """
    Classifica severidade considerando tanto invasoras quanto cobertura de cultivo.
    
    Args:
        weed_percentage: Porcentagem de ervas daninhas
        crop_percentage: Porcentagem de cobertura de cultivo
        
    Returns:
        Classificação: 'Baixa', 'Moderada', 'Alta', 'Crítica'
    """
    # Consider both weed infestation and crop coverage
    if weed_percentage > 20 or crop_percentage < 20:
        return 'Crítica'
    elif weed_percentage > 10 or crop_percentage < 35:
        return 'Alta' 
    elif weed_percentage > 5 or crop_percentage < 50:
        return 'Moderada'
    else:
        return 'Baixa'


def get_algorithm_info() -> Dict[str, Any]:
    """
    Retorna informações sobre os algoritmos disponíveis.
    """
    return {
        'algorithms': {
            'vegetation_indices': {
                'name': 'Índices de Vegetação + Geometria',
                'description': 'Pipeline robusto com ExG/ExGR/CIVE + detecção de linhas',
                'indices': ['ExG', 'ExGR', 'CIVE'],
                'features': ['illumination_normalization', 'crop_row_detection', 'inter_row_analysis'],
                'robustness': 'high',
                'recommended': True
            },
            'hsv_fallback': {
                'name': 'HSV Color Segmentation',
                'description': 'Segmentação HSV tradicional (fallback)',
                'features': ['color_based', 'morphological_cleanup'],
                'robustness': 'medium',
                'recommended': False
            }
        },
        'parameters': {
            'sensitivity': 'Detection sensitivity (0.0-1.0)',
            'normalize_illumination': 'Apply illumination normalization pipeline',
            'primary_index': 'Primary vegetation index (ExG, ExGR, CIVE)',
            'row_spacing_px': 'Coffee row spacing in pixels (auto if None)',
            'row_width_px': 'Coffee tree crown width in pixels (auto if None)'
        },
        'normalization_pipeline': [
            'gamma_correction',
            'shades_of_gray_white_balance', 
            'simple_retinex',
            'clahe_contrast_enhancement'
        ]
    }


def create_visualization_comparison(original_img: np.ndarray,
                                  hsv_result: Dict[str, Any],
                                  robust_result: Dict[str, Any]) -> np.ndarray:
    """
    Cria visualização comparativa entre métodos HSV e robusto.
    
    Args:
        original_img: Imagem original
        hsv_result: Resultado do método HSV
        robust_result: Resultado do método robusto
        
    Returns:
        Imagem com comparação lado a lado
    """
    h, w = original_img.shape[:2]
    
    # Create side-by-side comparison
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Original image
    comparison[:, :w] = original_img
    
    # HSV result
    comparison[:, w:2*w] = hsv_result['annotated_image']
    
    # Robust result  
    comparison[:, 2*w:3*w] = robust_result['annotated_image']
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "HSV Method", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "Robust Pipeline", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # Add statistics
    cv2.putText(comparison, f"Weeds: {hsv_result['weed_count']} ({hsv_result['weed_percentage']:.1f}%)", 
               (w + 10, h - 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(comparison, f"Weeds: {robust_result['weed_count']} ({robust_result['weed_percentage']:.1f}%)", 
               (2*w + 10, h - 20), font, 0.5, (255, 255, 255), 1)
    
    return comparison