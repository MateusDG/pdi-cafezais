import cv2
import numpy as np
from typing import Tuple, Dict, Any
from scipy import ndimage
from skimage import measure, morphology
from sklearn.cluster import DBSCAN


def gamma_correction(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """
    Aplica correção gamma para melhorar sombras e highlights.
    
    Args:
        img: Imagem RGB normalizada [0-1]
        gamma: Valor gamma (2.2 é padrão)
    
    Returns:
        Imagem com correção gamma aplicada
    """
    return np.power(img, 1.0 / gamma)


def shades_of_gray_white_balance(img: np.ndarray, power: float = 6.0) -> np.ndarray:
    """
    Balanceamento de branco usando Shades-of-Gray (mais robusto que Gray-World).
    
    Args:
        img: Imagem RGB [0-255]
        power: Potência para o algoritmo (6.0 é robusto)
    
    Returns:
        Imagem com balanceamento de branco aplicado
    """
    img_float = img.astype(np.float64) / 255.0
    
    # Calculate illuminant using Shades-of-Gray
    illuminant = np.zeros(3)
    for c in range(3):
        channel = img_float[:, :, c]
        # Avoid division by zero
        channel_power = np.power(channel + 1e-10, power)
        illuminant[c] = np.power(np.mean(channel_power), 1.0 / power)
    
    # Avoid division by zero
    illuminant = np.maximum(illuminant, 1e-10)
    
    # Normalize by gray illuminant
    gray_illuminant = np.mean(illuminant)
    correction_factors = gray_illuminant / illuminant
    
    # Apply correction
    corrected = img_float * correction_factors[np.newaxis, np.newaxis, :]
    corrected = np.clip(corrected, 0, 1)
    
    return (corrected * 255).astype(np.uint8)


def simple_retinex(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """
    Aplica Single-Scale Retinex simples para remover variações de iluminação.
    
    Args:
        img: Imagem RGB [0-255]
        sigma: Desvio padrão do filtro Gaussiano
    
    Returns:
        Imagem com Retinex aplicado
    """
    img_float = img.astype(np.float64) + 1.0  # Avoid log(0)
    
    # Apply to each channel
    retinex = np.zeros_like(img_float)
    for c in range(3):
        # Gaussian blur approximates center/surround
        blurred = cv2.GaussianBlur(img_float[:, :, c], (0, 0), sigma)
        retinex[:, :, c] = np.log(img_float[:, :, c]) - np.log(blurred + 1e-10)
    
    # Normalize to [0, 255]
    retinex = retinex - np.min(retinex)
    retinex = retinex / (np.max(retinex) + 1e-10) * 255
    
    return retinex.astype(np.uint8)


def calculate_vegetation_indices(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcula índices de vegetação ExG, ExGR e CIVE.
    
    Args:
        img: Imagem RGB [0-255]
    
    Returns:
        Dict com os índices calculados
    """
    # Normalize to [0, 1] for ExG and ExGR
    r = img[:, :, 0].astype(np.float32) / 255.0
    g = img[:, :, 1].astype(np.float32) / 255.0  
    b = img[:, :, 2].astype(np.float32) / 255.0
    
    # Calculate vegetation indices
    indices = {}
    
    # ExG (Excess Green): ExG = 2g - r - b
    indices['ExG'] = 2 * g - r - b
    
    # ExR (Excess Red): ExR = 1.4r - b  
    exr = 1.4 * r - b
    indices['ExR'] = exr
    
    # ExGR (ExG - ExR): More robust for weed detection
    indices['ExGR'] = indices['ExG'] - exr
    
    # CIVE (Color Index of Vegetation Extraction) - using [0-255] values
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)
    indices['CIVE'] = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    
    return indices


def normalize_index_to_uint8(index: np.ndarray) -> np.ndarray:
    """
    Normaliza índice para [0, 255] para binarização.
    
    Args:
        index: Índice de vegetação
    
    Returns:
        Índice normalizado para uint8
    """
    index_norm = index - np.min(index)
    index_norm = index_norm / (np.max(index_norm) + 1e-10) * 255
    return index_norm.astype(np.uint8)


def create_vegetation_mask(indices: Dict[str, np.ndarray], 
                          primary_index: str = 'ExGR',
                          min_area_ratio: float = 0.00002) -> np.ndarray:
    """
    Cria máscara de vegetação usando binarização Otsu.
    
    Args:
        indices: Dicionário com índices de vegetação
        primary_index: Índice principal para usar ('ExG', 'ExGR', ou 'CIVE')
        min_area_ratio: Área mínima como fração da imagem total
    
    Returns:
        Máscara binária de vegetação
    """
    # Get primary index and normalize
    index = indices[primary_index]
    index_uint8 = normalize_index_to_uint8(index)
    
    # Otsu thresholding
    _, binary = cv2.threshold(index_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleaning: opening + small area removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Remove small areas
    min_area = int(cleaned.shape[0] * cleaned.shape[1] * min_area_ratio)
    cleaned = morphology.remove_small_objects(
        cleaned > 0, min_size=min_area, connectivity=2
    ).astype(np.uint8) * 255
    
    return cleaned


def detect_crop_row_orientation(vegetation_mask: np.ndarray, 
                              angle_range: Tuple[int, int] = (-30, 30),
                              angle_step: int = 1) -> float:
    """
    Detecta a orientação dominante das linhas de cultivo usando transformada de Hough.
    
    Args:
        vegetation_mask: Máscara binária de vegetação
        angle_range: Faixa de ângulos para buscar (graus)
        angle_step: Passo entre ângulos
    
    Returns:
        Ângulo dominante em graus
    """
    # Edge detection on vegetation mask
    edges = cv2.Canny(vegetation_mask, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle_deg = np.degrees(theta) - 90  # Convert to standard orientation
            # Normalize to [-90, 90]
            while angle_deg > 90:
                angle_deg -= 180
            while angle_deg < -90:
                angle_deg += 180
            
            if angle_range[0] <= angle_deg <= angle_range[1]:
                angles.append(angle_deg)
    
    if not angles:
        # Fallback: try different morphological orientations
        best_angle = 0
        best_response = 0
        
        for angle in range(angle_range[0], angle_range[1] + 1, angle_step):
            # Create oriented kernel
            kernel_size = 15
            kernel = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1.0)
            oriented_kernel = cv2.warpAffine(
                np.ones((3, kernel_size), dtype=np.uint8), 
                kernel, (kernel_size, kernel_size)
            )
            
            # Apply morphological opening
            opened = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, oriented_kernel)
            response = np.sum(opened)
            
            if response > best_response:
                best_response = response
                best_angle = angle
        
        return best_angle
    
    # Use mode of detected angles
    angles = np.array(angles)
    hist, bins = np.histogram(angles, bins=20)
    dominant_angle = bins[np.argmax(hist)]
    
    return dominant_angle


def create_crop_row_mask(vegetation_mask: np.ndarray, 
                        dominant_angle: float,
                        row_spacing_px: int = 80,
                        row_width_px: int = 40) -> np.ndarray:
    """
    Cria máscara das linhas de cultivo baseada na orientação dominante.
    
    Args:
        vegetation_mask: Máscara de vegetação
        dominant_angle: Ângulo dominante das linhas
        row_spacing_px: Espaçamento aproximado entre linhas em pixels
        row_width_px: Largura aproximada das copas em pixels
    
    Returns:
        Máscara binária das linhas de cultivo
    """
    h, w = vegetation_mask.shape
    
    # Rotate image to align rows horizontally
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -dominant_angle, 1.0)
    rotated_mask = cv2.warpAffine(vegetation_mask, rotation_matrix, (w, h))
    
    # Project along Y axis to find row positions
    y_projection = np.sum(rotated_mask > 0, axis=1)
    
    # Find peaks (row centers) with minimum distance
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y_projection, distance=row_spacing_px//2, height=w//10)
    
    # Create row mask
    row_mask_rotated = np.zeros_like(rotated_mask)
    for peak in peaks:
        y_start = max(0, peak - row_width_px // 2)
        y_end = min(h, peak + row_width_px // 2)
        row_mask_rotated[y_start:y_end, :] = 255
    
    # Rotate back to original orientation
    inv_rotation_matrix = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
    row_mask = cv2.warpAffine(row_mask_rotated, inv_rotation_matrix, (w, h))
    
    return row_mask


def detect_inter_row_weeds(vegetation_mask: np.ndarray, 
                          row_mask: np.ndarray,
                          min_weed_area: int = 100) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detecta ervas daninhas nas entrelinhas.
    
    Args:
        vegetation_mask: Máscara total de vegetação
        row_mask: Máscara das linhas de cultivo
        min_weed_area: Área mínima para considerar como erva daninha
    
    Returns:
        Tupla (weed_mask, statistics)
    """
    # Inter-row vegetation = vegetation NOT in crop rows
    inter_row_vegetation = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(row_mask))
    
    # Clean up with closing and small area removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(inter_row_vegetation, cv2.MORPH_CLOSE, kernel)
    
    # Remove small areas
    cleaned = morphology.remove_small_objects(
        cleaned > 0, min_size=min_weed_area, connectivity=2
    ).astype(np.uint8) * 255
    
    # Calculate statistics
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_weed_area = sum(cv2.contourArea(c) for c in contours)
    image_area = vegetation_mask.shape[0] * vegetation_mask.shape[1]
    
    statistics = {
        'weed_count': len(contours),
        'total_weed_area': int(total_weed_area),
        'weed_percentage': (total_weed_area / image_area) * 100,
        'inter_row_weeds': cleaned,
        'contours': contours
    }
    
    return cleaned, statistics


def robust_weed_detection_pipeline(img: np.ndarray, 
                                 normalize_illumination: bool = True,
                                 primary_index: str = 'ExGR',
                                 row_spacing_px: int = 80,
                                 row_width_px: int = 40) -> Dict[str, Any]:
    """
    Pipeline completo de detecção robusta de ervas daninhas.
    
    Args:
        img: Imagem RGB [0-255]
        normalize_illumination: Se aplicar normalização de iluminação
        primary_index: Índice principal ('ExG', 'ExGR', 'CIVE')
        row_spacing_px: Espaçamento entre linhas
        row_width_px: Largura das copas
    
    Returns:
        Dicionário com todos os resultados
    """
    results = {}
    
    # 1. Illumination normalization (optional but recommended)
    processed_img = img.copy()
    if normalize_illumination:
        # Gamma correction
        img_normalized = processed_img.astype(np.float32) / 255.0
        img_gamma = gamma_correction(img_normalized, gamma=2.2)
        processed_img = (img_gamma * 255).astype(np.uint8)
        
        # White balance
        processed_img = shades_of_gray_white_balance(processed_img)
        
        # Simple Retinex
        processed_img = simple_retinex(processed_img, sigma=15.0)
    
    results['processed_image'] = processed_img
    
    # 2. Calculate vegetation indices
    indices = calculate_vegetation_indices(processed_img)
    results['vegetation_indices'] = indices
    
    # 3. Create vegetation mask
    vegetation_mask = create_vegetation_mask(indices, primary_index)
    results['vegetation_mask'] = vegetation_mask
    
    # 4. Detect crop row orientation
    dominant_angle = detect_crop_row_orientation(vegetation_mask)
    results['dominant_angle'] = dominant_angle
    
    # 5. Create crop row mask
    row_mask = create_crop_row_mask(vegetation_mask, dominant_angle, row_spacing_px, row_width_px)
    results['row_mask'] = row_mask
    
    # 6. Detect inter-row weeds
    weed_mask, weed_stats = detect_inter_row_weeds(vegetation_mask, row_mask)
    results['weed_mask'] = weed_mask
    results['weed_statistics'] = weed_stats
    
    # 7. Create annotated visualization
    annotated = create_robust_annotation(img, vegetation_mask, row_mask, weed_mask, weed_stats['contours'])
    results['annotated_image'] = annotated
    
    return results


def create_robust_annotation(original_img: np.ndarray,
                           vegetation_mask: np.ndarray,
                           row_mask: np.ndarray, 
                           weed_mask: np.ndarray,
                           weed_contours) -> np.ndarray:
    """
    Cria visualização anotada do pipeline robusto.
    
    Args:
        original_img: Imagem original
        vegetation_mask: Máscara de vegetação total
        row_mask: Máscara das linhas de cultivo
        weed_mask: Máscara de ervas daninhas
        weed_contours: Contornos das ervas daninhas
    
    Returns:
        Imagem anotada
    """
    annotated = original_img.copy()
    
    # Create overlay
    overlay = np.zeros_like(original_img)
    
    # Crop rows in green (semi-transparent)
    overlay[row_mask > 0] = [0, 255, 0]
    
    # Inter-row weeds in red
    overlay[weed_mask > 0] = [255, 0, 0]
    
    # Blend overlay with original
    annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
    
    # Draw weed contours
    cv2.drawContours(annotated, weed_contours, -1, (255, 0, 0), 2)
    
    # Add statistics overlay
    weed_count = len(weed_contours)
    total_area = sum(cv2.contourArea(c) for c in weed_contours)
    image_area = original_img.shape[0] * original_img.shape[1]
    weed_percentage = (total_area / image_area) * 100
    
    # Info box
    cv2.rectangle(annotated, (10, 10), (450, 100), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (450, 100), (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, f"Pipeline Robusto - Indices de Vegetacao", 
               (15, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Ervas daninhas: {weed_count} areas ({weed_percentage:.1f}%)", 
               (15, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Verde=Linhas Cultivo, Vermelho=Invasoras", 
               (15, 70), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Deteccao geometrica + ExGR/CIVE", 
               (15, 90), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return annotated