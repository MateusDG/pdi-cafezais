import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import uuid
import os
from pathlib import Path


def imread(path: str) -> np.ndarray:
    """
    Lê uma imagem do disco com validação.
    
    Args:
        path: Caminho para a imagem
        
    Returns:
        Imagem como array numpy (BGR)
        
    Raises:
        FileNotFoundError: Se não conseguir ler a imagem
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    """
    Salva uma imagem no disco com validação.
    
    Args:
        path: Caminho de destino
        img: Imagem como array numpy
        
    Raises:
        RuntimeError: Se não conseguir salvar
    """
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Falha ao salvar imagem em {path}")


def generate_unique_filename(prefix: str = "result", extension: str = "jpg") -> str:
    """
    Gera um nome de arquivo único usando UUID.
    
    Args:
        prefix: Prefixo do arquivo
        extension: Extensão do arquivo (sem ponto)
        
    Returns:
        Nome de arquivo único
    """
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{unique_id}.{extension}"


def validate_image_format(file_path: str) -> bool:
    """
    Valida se o arquivo é uma imagem suportada.
    
    Args:
        file_path: Caminho para o arquivo
        
    Returns:
        True se for um formato válido
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    extension = Path(file_path).suffix.lower()
    return extension in valid_extensions


def calculate_image_stats(img: np.ndarray) -> Dict[str, Any]:
    """
    Calcula estatísticas básicas da imagem.
    
    Args:
        img: Imagem como array numpy
        
    Returns:
        Dicionário com estatísticas
    """
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'total_pixels': height * width,
        'file_size_mb': (img.nbytes / (1024 * 1024)),
        'mean_brightness': np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if channels > 1 else np.mean(img),
        'std_brightness': np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if channels > 1 else np.std(img)
    }


def resize_image_if_needed(img: np.ndarray, max_size: int = 2048) -> Tuple[np.ndarray, float]:
    """
    Redimensiona imagem se for muito grande, mantendo proporção.
    
    Args:
        img: Imagem original
        max_size: Tamanho máximo para a maior dimensão
        
    Returns:
        Tupla (imagem_redimensionada, fator_escala)
    """
    height, width = img.shape[:2]
    max_dim = max(height, width)
    
    if max_dim <= max_size:
        return img, 1.0
    
    scale_factor = max_size / max_dim
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale_factor


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Converte imagem de BGR (OpenCV) para RGB.
    
    Args:
        img: Imagem em formato BGR
        
    Returns:
        Imagem em formato RGB
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """
    Converte imagem de RGB para BGR (OpenCV).
    
    Args:
        img: Imagem em formato RGB
        
    Returns:
        Imagem em formato BGR
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def normalize_image_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para melhorar a detecção.
    
    Args:
        img: Imagem RGB de entrada
        clip_limit: Limite de corte para evitar over-amplification
        grid_size: Tamanho da grade para equalização adaptativa
        
    Returns:
        Imagem normalizada
    """
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    
    # Convert back to RGB
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return normalized


def create_processing_summary(
    weed_data: Dict[str, Any], 
    processing_time: float,
    image_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Cria um resumo consolidado do processamento com estatísticas detalhadas.
    
    Args:
        weed_data: Dados da detecção de ervas daninhas
        processing_time: Tempo de processamento em segundos
        image_stats: Estatísticas da imagem
        
    Returns:
        Resumo consolidado com métricas avançadas
    """
    # Convert area_stats if present
    area_statistics = None
    if weed_data.get('area_stats'):
        area_statistics = {
            'min_area': weed_data['area_stats']['min_area'],
            'max_area': weed_data['area_stats']['max_area'],
            'avg_area': weed_data['area_stats']['avg_area'],
            'median_area': weed_data['area_stats']['median_area']
        }
    
    # Generate detected issues based on analysis
    detected_issues = []
    weed_percentage = weed_data.get('weed_percentage', 0)
    
    if weed_percentage > 25:
        detected_issues.append("Alta infestação de ervas daninhas detectada")
    elif weed_percentage > 10:
        detected_issues.append("Infestação moderada de ervas daninhas")
    
    if weed_data.get('bare_soil_percentage', 0) > 30:
        detected_issues.append("Significativa área de solo exposto")
    
    if weed_data.get('coffee_percentage', 0) < 40:
        detected_issues.append("Cobertura de café abaixo do ideal")
    
    return {
        'processing_time_seconds': round(processing_time, 2),
        'image_stats': image_stats,
        'weed_detection': {
            'areas_detected': weed_data.get('weed_count', 0),
            'total_weed_area_pixels': weed_data.get('total_weed_area', 0),
            'weed_coverage_percentage': weed_data.get('weed_percentage', 0),
            'coffee_coverage_percentage': weed_data.get('coffee_percentage', 0),
            'vegetation_coverage_percentage': weed_data.get('vegetation_percentage', 0),
            'bare_soil_percentage': weed_data.get('bare_soil_percentage', 0),
            'detection_sensitivity': weed_data.get('sensitivity_used', 0.5),
            'severity_level': weed_data.get('severity_level', 'Baixa'),
            'density_per_sqm': weed_data.get('density_per_sqm', 0),
            'area_statistics': area_statistics
        },
        'analysis_status': 'completed',
        'detected_issues': detected_issues
    }
