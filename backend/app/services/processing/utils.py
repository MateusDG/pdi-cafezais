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


def create_processing_summary(
    weed_data: Dict[str, Any], 
    processing_time: float,
    image_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Cria um resumo consolidado do processamento.
    
    Args:
        weed_data: Dados da detecção de ervas daninhas
        processing_time: Tempo de processamento em segundos
        image_stats: Estatísticas da imagem
        
    Returns:
        Resumo consolidado
    """
    return {
        'processing_time_seconds': round(processing_time, 2),
        'image_stats': image_stats,
        'weed_detection': {
            'areas_detected': weed_data.get('weed_count', 0),
            'total_weed_area_pixels': weed_data.get('total_weed_area', 0),
            'weed_coverage_percentage': weed_data.get('weed_percentage', 0),
            'detection_sensitivity': 0.5  # Default sensitivity
        },
        'analysis_status': 'completed',
        'detected_issues': []
    }
