import cv2
import numpy as np
from typing import Tuple, Dict, Any, List
from scipy import ndimage
from skimage import measure, morphology
import logging

logger = logging.getLogger(__name__)


def gamma_correction_enhanced(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Correção gamma aprimorada."""
    img_float = img.astype(np.float32) / 255.0
    corrected = np.power(img_float, 1.0 / gamma)
    return (corrected * 255).astype(np.uint8)


def shades_of_gray_white_balance_enhanced(img: np.ndarray, p_norm: float = 6.0) -> np.ndarray:
    """White balance por Shades-of-Gray com p-norm=6 (mais robusto)."""
    img_float = img.astype(np.float64) / 255.0
    
    illuminant = np.zeros(3)
    for c in range(3):
        channel = img_float[:, :, c]
        channel_power = np.power(channel + 1e-10, p_norm)
        illuminant[c] = np.power(np.mean(channel_power), 1.0 / p_norm)
    
    illuminant = np.maximum(illuminant, 1e-10)
    gray_illuminant = np.mean(illuminant)
    correction_factors = gray_illuminant / illuminant
    
    corrected = img_float * correction_factors[np.newaxis, np.newaxis, :]
    corrected = np.clip(corrected, 0, 1)
    
    return (corrected * 255).astype(np.uint8)


def multi_scale_retinex(img: np.ndarray, sigmas: List[float] = [15.0, 80.0]) -> np.ndarray:
    """Multi-Scale Retinex com 2 escalas."""
    img_float = img.astype(np.float64) + 1.0
    
    retinex_sum = np.zeros_like(img_float)
    for sigma in sigmas:
        for c in range(3):
            blurred = cv2.GaussianBlur(img_float[:, :, c], (0, 0), sigma)
            retinex_sum[:, :, c] += np.log(img_float[:, :, c]) - np.log(blurred + 1e-10)
    
    # Average over scales
    retinex = retinex_sum / len(sigmas)
    
    # Normalize to [0, 255]
    for c in range(3):
        channel = retinex[:, :, c]
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-10) * 255
        retinex[:, :, c] = channel
    
    return retinex.astype(np.uint8)


def clahe_enhance_v_channel(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """CLAHE no canal V (HSV) para contraste local."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    
    # Apply CLAHE to V channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return enhanced


def detect_sky_and_ground_roi(img: np.ndarray, margin_percent: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detecta céu e cria ROI de solo/lavoura para foto oblíqua.
    
    Args:
        img: Imagem RGB
        margin_percent: Margem adicional abaixo da transição céu-terra
    
    Returns:
        Tupla (sky_mask, ground_roi)
    """
    h, w = img.shape[:2]
    
    # Convert to HSV for sky detection
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    
    # Sky criteria: low saturation + high value
    sky_mask = (s < 0.15) & (v > 0.7)
    
    # For each column, find first non-sky pixel (sky-to-ground transition)
    ground_transition = np.zeros(w, dtype=int)
    
    for col in range(w):
        column_sky = sky_mask[:, col]
        # Find first False (non-sky) pixel
        non_sky_pixels = np.where(~column_sky)[0]
        if len(non_sky_pixels) > 0:
            ground_transition[col] = non_sky_pixels[0]
        else:
            ground_transition[col] = h - 1  # Fallback to bottom
    
    # Smooth transition line (moving average)
    kernel_size = max(5, w // 20)
    smoothed_transition = np.convolve(ground_transition, 
                                     np.ones(kernel_size)/kernel_size, 
                                     mode='same')
    
    # Add margin below transition
    margin_pixels = int(h * margin_percent)
    adjusted_transition = np.clip(smoothed_transition + margin_pixels, 0, h-1).astype(int)
    
    # Create ground ROI mask
    ground_roi = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        ground_roi[adjusted_transition[col]:, col] = 255
    
    # Refine sky mask using transition
    refined_sky_mask = np.zeros((h, w), dtype=np.uint8)
    for col in range(w):
        refined_sky_mask[:adjusted_transition[col], col] = 255
    
    return refined_sky_mask, ground_roi


def create_green_gate_hsv(img: np.ndarray, 
                         hue_range: Tuple[int, int] = (25, 105),
                         min_saturation: float = 0.20,
                         min_value: float = 0.12) -> np.ndarray:
    """
    Cria gate cromático para tons de verde.
    
    Args:
        img: Imagem RGB
        hue_range: Faixa de matizes em graus (35°-95°)
        min_saturation: Saturação mínima
        min_value: Valor mínimo
    
    Returns:
        Máscara do gate verde
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    h = hsv[:, :, 0].astype(np.float32)  # 0-359 degrees
    s = hsv[:, :, 1].astype(np.float32) / 255.0  # 0-1
    v = hsv[:, :, 2].astype(np.float32) / 255.0  # 0-1
    
    # Convert to degrees (OpenCV uses 0-179 for 0-358°)
    h_degrees = h * 2.0
    
    # Create green gate
    green_gate = (
        (h_degrees >= hue_range[0]) & (h_degrees <= hue_range[1]) &
        (s > min_saturation) &
        (v > min_value)
    )
    
    return green_gate.astype(np.uint8) * 255


def calculate_vegetation_index_with_gate(img: np.ndarray, green_gate: np.ndarray, index_type: str = 'ExGR', otsu_offset: int = -15) -> Tuple[np.ndarray, float]:
    """
    Calcula índice de vegetação apenas dentro do gate verde com Otsu local.
    
    Args:
        img: Imagem RGB
        green_gate: Máscara do gate verde
        index_type: Tipo do índice ('ExG', 'ExGR', 'CIVE')
    
    Returns:
        Tupla (vegetation_mask, threshold_used)
    """
    # Normalize to [0, 1]
    r = img[:, :, 0].astype(np.float32) / 255.0
    g = img[:, :, 1].astype(np.float32) / 255.0  
    b = img[:, :, 2].astype(np.float32) / 255.0
    
    # Calculate index based on type
    if index_type == 'ExG':
        index = 2 * g - r - b
    elif index_type == 'ExGR':
        exg = 2 * g - r - b
        exr = 1.4 * r - b
        index = exg - exr
    elif index_type == 'CIVE':
        R = img[:, :, 0].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        B = img[:, :, 2].astype(np.float32)
        index = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
        # Normalize CIVE to [0, 1] range
        index = (index - np.min(index)) / (np.max(index) - np.min(index) + 1e-10)
    else:
        # Default to ExGR
        exg = 2 * g - r - b
        exr = 1.4 * r - b
        index = exg - exr
    
    # Get pixels within green gate only
    gate_pixels = index[green_gate > 0]
    
    if len(gate_pixels) < 100:
        # Not enough green pixels, return empty mask
        return np.zeros_like(green_gate), 0.0
    
    # Normalize gate pixels to [0, 255] for Otsu
    gate_min, gate_max = np.min(gate_pixels), np.max(gate_pixels)
    if gate_max - gate_min < 1e-6:
        return np.zeros_like(green_gate), 0.0
    
    gate_normalized = ((gate_pixels - gate_min) / (gate_max - gate_min) * 255).astype(np.uint8)
    
    # Apply Otsu only to gate pixels
    threshold_otsu, _ = cv2.threshold(gate_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply negative offset to make threshold more permissive
    threshold_otsu_adjusted = max(0, threshold_otsu + otsu_offset)
    
    # Convert back to original scale
    threshold_original = gate_min + (threshold_otsu_adjusted / 255.0) * (gate_max - gate_min)
    
    # Create vegetation mask
    vegetation_mask = (index > threshold_original) & (green_gate > 0)
    
    return vegetation_mask.astype(np.uint8) * 255, float(threshold_original)


def morphological_cleanup(mask: np.ndarray, 
                         opening_kernel_size: int = 3,
                         min_area_ratio: float = 2e-5) -> np.ndarray:
    """
    Limpeza morfológica: abertura + remoção de componentes pequenos.
    
    Args:
        mask: Máscara binária
        opening_kernel_size: Tamanho do kernel de abertura
        min_area_ratio: Área mínima como fração da imagem total
    
    Returns:
        Máscara limpa
    """
    # Morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (opening_kernel_size, opening_kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small components
    total_pixels = mask.shape[0] * mask.shape[1]
    min_area = int(total_pixels * min_area_ratio)
    
    cleaned = morphology.remove_small_objects(
        cleaned > 0, min_size=min_area, connectivity=2
    ).astype(np.uint8) * 255
    
    return cleaned


def detect_crop_rows_reliable(vegetation_mask: np.ndarray, 
                             min_lines: int = 30,
                             max_angle_std: float = 6.0) -> Tuple[np.ndarray, bool, float]:
    """
    Detecta linhas de cultivo apenas se confiável (foto nadir).
    
    Args:
        vegetation_mask: Máscara de vegetação
        min_lines: Número mínimo de linhas Hough para confiar
        max_angle_std: Desvio padrão máximo dos ângulos
    
    Returns:
        Tupla (row_mask, is_reliable, dominant_angle)
    """
    # Edge detection
    edges = cv2.Canny(vegetation_mask, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=20)
    
    if lines is None or len(lines) < min_lines:
        # Not enough lines - oblique photo mode
        return vegetation_mask.copy(), False, 0.0
    
    # Extract angles
    angles = []
    for rho, theta in lines[:, 0]:
        angle_deg = np.degrees(theta) - 90
        # Normalize to [-90, 90]
        while angle_deg > 90:
            angle_deg -= 180
        while angle_deg < -90:
            angle_deg += 180
        angles.append(angle_deg)
    
    angles = np.array(angles)
    angle_std = np.std(angles)
    
    if angle_std > max_angle_std:
        # Inconsistent angles - oblique photo mode
        return vegetation_mask.copy(), False, 0.0
    
    # Reliable - create row mask
    dominant_angle = np.median(angles)
    
    # Simple row mask creation (can be improved)
    h, w = vegetation_mask.shape
    row_mask = np.zeros_like(vegetation_mask)
    
    # Create parallel lines at detected orientation
    center_x, center_y = w // 2, h // 2
    row_spacing = 60  # pixels
    
    for offset in range(-w, w, row_spacing):
        m = np.tan(np.radians(dominant_angle))
        
        for x in range(w):
            y = int(center_y + m * (x - center_x) + offset * np.cos(np.radians(dominant_angle)))
            if 0 <= y < h:
                # Draw thick line
                for dy in range(-20, 21):  # 40px width
                    if 0 <= y + dy < h:
                        row_mask[y + dy, x] = 255
    
    # Restrict to vegetation areas only
    row_mask = cv2.bitwise_and(row_mask, vegetation_mask)
    
    return row_mask, True, dominant_angle


def create_soil_mask(img: np.ndarray) -> np.ndarray:
    """
    Cria máscara de solo para filtro "toca solo" - versão expandida.
    
    Args:
        img: Imagem RGB
    
    Returns:
        Máscara de solo
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    h = hsv[:, :, 0].astype(np.float32) * 2.0  # Convert to degrees
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    
    # Expanded soil criteria for various soil types
    soil_mask = (
        (((h >= 0) & (h <= 35)) |   # Brown/red/orange hues (expanded)
         (s < 0.25)) &              # Low saturation (gray/light soil - more permissive)
        (v > 0.15) & (v < 0.95)     # Wider brightness range
    )
    
    return soil_mask.astype(np.uint8) * 255


def oblique_weed_detection_pipeline(img: np.ndarray,
                                   sensitivity: float = 0.5,
                                   normalize_illumination: bool = True,
                                   primary_index: str = 'ExGR',
                                   row_spacing_px: int = None,
                                   enable_row_detection: bool = True) -> Dict[str, Any]:
    """
    Pipeline completo para detecção robusta em fotos oblíquas.
    
    Args:
        img: Imagem RGB [0-255]
        sensitivity: Sensibilidade (afeta áreas mínimas)
        normalize_illumination: Aplicar normalização
        primary_index: Índice de vegetação principal ('ExG', 'ExGR', 'CIVE')
        row_spacing_px: Espaçamento entre fileiras em pixels (auto se None)
        enable_row_detection: Tentar detectar linhas (pode ser desabilitado)
    
    Returns:
        Dicionário com resultados e flags de qualidade
    """
    results = {}
    quality_flags = {}
    
    h, w = img.shape[:2]
    total_pixels = h * w
    
    try:
        # 1. Normalização de iluminação
        processed_img = img.copy()
        if normalize_illumination:
            logger.info("Applying illumination normalization")
            processed_img = gamma_correction_enhanced(processed_img, gamma=2.2)
            processed_img = shades_of_gray_white_balance_enhanced(processed_img, p_norm=6.0)
            processed_img = multi_scale_retinex(processed_img, sigmas=[15.0, 80.0])
            processed_img = clahe_enhance_v_channel(processed_img, clip_limit=2.0, grid_size=8)
        
        results['processed_image'] = processed_img
        
        # 2. ROI de solo/lavoura
        logger.info("Detecting sky and ground ROI")
        sky_mask, ground_roi = detect_sky_and_ground_roi(processed_img, margin_percent=0.05)
        results['sky_mask'] = sky_mask
        results['ground_roi'] = ground_roi
        
        # 3. Gate cromático de verde
        logger.info("Creating green gate")
        green_gate = create_green_gate_hsv(processed_img, 
                                          hue_range=(25, 105),
                                          min_saturation=0.20,
                                          min_value=0.12)
        # Intersect with ROI
        green_gate = cv2.bitwise_and(green_gate, ground_roi)
        results['green_gate'] = green_gate
        
        # Debug counts
        roi_pixels = cv2.countNonZero(ground_roi)
        gate_pixels = cv2.countNonZero(green_gate)
        logger.info(f"ROI pixels: {roi_pixels}, Green gate pixels: {gate_pixels} ({100*gate_pixels/roi_pixels:.1f}% of ROI)")
        
        # 4. Índice de vegetação + limiarização
        logger.info(f"Calculating {primary_index} index with Otsu thresholding")
        # Sensitivity affects Otsu offset: higher sensitivity = more negative offset (more permissive)
        otsu_offset = int(-10 - (sensitivity * 8))  # Range: -10 to -18
        vegetation_mask, threshold_used = calculate_vegetation_index_with_gate(processed_img, green_gate, primary_index, otsu_offset=otsu_offset)
        # Intersect with ROI
        vegetation_mask = cv2.bitwise_and(vegetation_mask, ground_roi)
        results['vegetation_mask'] = vegetation_mask
        results['vegetation_threshold'] = threshold_used
        
        # Debug counts
        veg_pixels_raw = cv2.countNonZero(vegetation_mask)
        logger.info(f"Vegetation pixels (raw): {veg_pixels_raw} ({100*veg_pixels_raw/roi_pixels:.1f}% of ROI), threshold: {threshold_used:.3f}")
        
        # 5. Pós-processamento morfológico
        logger.info("Morphological cleanup")
        # Reduced minimum area - more sensitive to small weeds
        # Base area is ~18px for 720x1280 image (0.002%)
        min_area_ratio = 0.002e-2 * (1.0 - sensitivity * 0.6)  # More aggressive reduction with sensitivity
        vegetation_mask = morphological_cleanup(vegetation_mask, 
                                              opening_kernel_size=3,
                                              min_area_ratio=min_area_ratio)
        # Re-intersect with ROI
        vegetation_mask = cv2.bitwise_and(vegetation_mask, ground_roi)
        results['vegetation_mask_cleaned'] = vegetation_mask
        
        # Debug counts after morphological cleanup
        veg_pixels_cleaned = cv2.countNonZero(vegetation_mask)
        logger.info(f"Vegetation pixels (after morphology): {veg_pixels_cleaned} ({100*veg_pixels_cleaned/roi_pixels:.1f}% of ROI), min_area: {int(total_pixels * min_area_ratio)}px")
        
        # 6. Fileiras do café (opcional/condicional)
        logger.info("Attempting crop row detection")
        if enable_row_detection:
            # Auto-estimate row spacing if not provided
            if row_spacing_px is None:
                h, w = img.shape[:2]
                row_spacing_px = int(max(60, min(120, w // 8)) * (0.8 + sensitivity * 0.4))
            
            row_mask, rows_reliable, dominant_angle = detect_crop_rows_reliable(
                vegetation_mask, min_lines=30, max_angle_std=6.0
            )
            # Always intersect with ROI
            row_mask = cv2.bitwise_and(row_mask, ground_roi)
        else:
            row_mask = vegetation_mask.copy()
            rows_reliable = False
            dominant_angle = 0.0
        
        results['row_mask'] = row_mask
        results['rows_reliable'] = rows_reliable
        results['dominant_angle'] = dominant_angle
        quality_flags['oblique_mode'] = not rows_reliable
        
        # 7. Margem de segurança nas fileiras
        logger.info("Applying safety margins to crop rows")
        safety_margin = int(5 + sensitivity * 5)  # 5-10 pixels based on sensitivity
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safety_margin, safety_margin))
        dilated_rows = cv2.dilate(row_mask, kernel)
        # Keep within ROI
        dilated_rows = cv2.bitwise_and(dilated_rows, ground_roi)
        
        # 8. Entrelinhas = vegetação fora das fileiras
        logger.info("Detecting inter-row weeds")
        weed_mask_base = cv2.bitwise_and(vegetation_mask, cv2.bitwise_not(dilated_rows))
        
        # Closing to unite fragments
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        weed_mask_base = cv2.morphologyEx(weed_mask_base, cv2.MORPH_CLOSE, closing_kernel)
        
        # Remove small components again (use same reduced min area)
        weed_mask_base = morphological_cleanup(weed_mask_base, 
                                              opening_kernel_size=3,
                                              min_area_ratio=min_area_ratio)
        # Keep within ROI
        weed_mask_base = cv2.bitwise_and(weed_mask_base, ground_roi)
        
        # 9. Filtro "toca solo"
        logger.info("Applying 'touches soil' filter")
        soil_mask = create_soil_mask(processed_img)
        # Dilate soil mask for better contact detection
        # Sensitivity affects dilation: higher sensitivity = more dilation (more permissive contact)
        dilation_size = int(6 + (sensitivity * 4))  # Range: 6-10 pixels
        soil_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        soil_mask_dilated = cv2.dilate(soil_mask, soil_dilate_kernel)
        # Keep within ROI
        soil_mask_dilated = cv2.bitwise_and(soil_mask_dilated, ground_roi)
        
        # Debug counts before soil filter
        weed_pixels_base = cv2.countNonZero(weed_mask_base)
        soil_pixels = cv2.countNonZero(soil_mask_dilated)
        logger.info(f"Weed pixels (base): {weed_pixels_base}, Soil pixels (dilated): {soil_pixels}")
        
        # Final weed mask: only weeds that touch soil
        weed_mask_final = cv2.bitwise_and(weed_mask_base, soil_mask_dilated)
        results['weed_mask_final'] = weed_mask_final
        results['soil_mask'] = soil_mask_dilated
        
        # Debug final counts
        weed_pixels_final = cv2.countNonZero(weed_mask_final)
        logger.info(f"Weed pixels (final): {weed_pixels_final} ({100*weed_pixels_final/roi_pixels:.2f}% of ROI)")
        
        # 10. Métricas e classificação
        logger.info("Calculating metrics and classification")
        
        # Find contours
        contours, _ = cv2.findContours(weed_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug blob information
        if contours:
            blob_areas = [cv2.contourArea(c) for c in contours]
            logger.info(f"Found {len(contours)} blobs, areas: min={min(blob_areas):.0f}, max={max(blob_areas):.0f}, avg={np.mean(blob_areas):.1f}")
        else:
            logger.info("No blobs found after all filters")
        
        # Calculate areas
        roi_area = cv2.countNonZero(ground_roi)
        total_weed_area = sum(cv2.contourArea(c) for c in contours)
        weed_coverage_percent = (total_weed_area / roi_area) * 100 if roi_area > 0 else 0
        
        # Classify blobs by size
        small_threshold = roi_area * 0.0002  # 0.02% of ROI
        medium_threshold = roi_area * 0.001   # 0.1% of ROI
        
        small_weeds, medium_weeds, large_weeds = [], [], []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < small_threshold:
                small_weeds.append(contour)
            elif area < medium_threshold:
                medium_weeds.append(contour)
            else:
                large_weeds.append(contour)
        
        # Quality checks
        if contours:
            largest_blob_area = max(cv2.contourArea(c) for c in contours)
            quality_flags['dominant_blob_suspicious'] = largest_blob_area >= (total_weed_area * 0.5)
        else:
            quality_flags['dominant_blob_suspicious'] = False
        
        # Check for sky leakage
        quality_flags['sky_leak'] = cv2.countNonZero(cv2.bitwise_and(weed_mask_final, sky_mask)) > 0
        
        # Coverage warning for oblique photos
        quality_flags['high_coverage_warning'] = (not rows_reliable) and (weed_coverage_percent > 40)
        
        # Compile statistics
        statistics = {
            'total_weeds': len(contours),
            'small_weeds': len(small_weeds),
            'medium_weeds': len(medium_weeds),
            'large_weeds': len(large_weeds),
            'total_weed_area_pixels': int(total_weed_area),
            'weed_coverage_percent': round(weed_coverage_percent, 2),
            'roi_area_pixels': roi_area,
            'contours': contours,
            'small_contours': small_weeds,
            'medium_contours': medium_weeds,
            'large_contours': large_weeds
        }
        
        results['statistics'] = statistics
        results['quality_flags'] = quality_flags
        
        # Add compatibility keys for API
        results['weed_count'] = len(contours)
        results['weed_percentage'] = round(weed_coverage_percent, 2)
        results['contours'] = contours
        results['total_weed_area'] = int(total_weed_area)
        
        # 11. Visualização
        annotated = create_oblique_annotation(img, results)
        results['annotated_image'] = annotated
        
        logger.info(f"Pipeline completed: {len(contours)} weeds detected, {weed_coverage_percent:.2f}% coverage")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return {
            'error': str(e),
            'statistics': {'total_weeds': 0, 'weed_coverage_percent': 0.0},
            'quality_flags': {'error': True},
            'annotated_image': img.copy()
        }


def create_oblique_annotation(original_img: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
    """
    Cria visualização anotada para pipeline oblíquo.
    
    Args:
        original_img: Imagem original
        results: Resultados do pipeline
    
    Returns:
        Imagem anotada
    """
    annotated = original_img.copy()
    
    if 'statistics' not in results:
        return annotated
    
    stats = results['statistics']
    quality_flags = results.get('quality_flags', {})
    
    # Create overlay
    overlay = np.zeros_like(original_img)
    
    # Sky region (light blue tint)
    if 'sky_mask' in results:
        overlay[results['sky_mask'] > 0] = [200, 220, 255]
    
    # ROI border (green outline)
    if 'ground_roi' in results:
        roi_contours, _ = cv2.findContours(results['ground_roi'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, roi_contours, -1, (0, 255, 0), 2)
    
    # Crop rows (dark green)
    if 'row_mask' in results and results.get('rows_reliable', False):
        overlay[results['row_mask'] > 0] = [0, 150, 0]
    
    # Weeds by size (yellow/orange/red)
    if 'small_contours' in stats:
        for contour in stats['small_contours']:
            cv2.drawContours(overlay, [contour], -1, (255, 255, 0), -1)  # Yellow fill
    
    if 'medium_contours' in stats:
        for contour in stats['medium_contours']:
            cv2.drawContours(overlay, [contour], -1, (255, 165, 0), -1)  # Orange fill
    
    if 'large_contours' in stats:
        for contour in stats['large_contours']:
            cv2.drawContours(overlay, [contour], -1, (255, 0, 0), -1)  # Red fill
    
    # Blend overlay
    annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
    
    # Draw contours
    if 'small_contours' in stats:
        for contour in stats['small_contours']:
            cv2.drawContours(annotated, [contour], -1, (255, 255, 0), 2)
    
    if 'medium_contours' in stats:
        for contour in stats['medium_contours']:
            cv2.drawContours(annotated, [contour], -1, (255, 165, 0), 2)
    
    if 'large_contours' in stats:
        for contour in stats['large_contours']:
            cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)
    
    # Info box
    box_height = 160
    cv2.rectangle(annotated, (10, 10), (600, box_height), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (600, box_height), (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 25
    line_height = 18
    
    # Title
    cv2.putText(annotated, "Pipeline Oblíquo - Detecção Robusta", 
               (15, y_offset), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y_offset += line_height + 5
    
    # Statistics
    cv2.putText(annotated, f"Ervas: {stats['total_weeds']} ({stats['weed_coverage_percent']:.1f}% da área útil)",
               (15, y_offset), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += line_height
    
    cv2.putText(annotated, f"Pequenas: {stats['small_weeds']} | Médias: {stats['medium_weeds']} | Grandes: {stats['large_weeds']}",
               (15, y_offset), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += line_height
    
    # Mode indicator
    mode_text = "Modo Oblíquo (sem fileiras)" if quality_flags.get('oblique_mode', False) else "Modo Nadir (com fileiras)"
    cv2.putText(annotated, mode_text,
               (15, y_offset), font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    y_offset += line_height
    
    # Quality flags
    if quality_flags.get('sky_leak', False):
        cv2.putText(annotated, "AVISO: Sky leak detectado",
                   (15, y_offset), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        y_offset += line_height
    
    if quality_flags.get('dominant_blob_suspicious', False):
        cv2.putText(annotated, "AVISO: Blob dominante suspeito (posível copa)",
                   (15, y_offset), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        y_offset += line_height
    
    # Legend
    cv2.putText(annotated, "Amarelo=Pequeno | Laranja=Médio | Vermelho=Grande | Verde=ROI",
               (15, y_offset), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    
    return annotated