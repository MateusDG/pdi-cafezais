import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def detect_weeds_hsv(img: np.ndarray, sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    Detecta ervas daninhas usando segmentação HSV com estatísticas detalhadas.
    
    Args:
        img: Imagem RGB de entrada
        sensitivity: Sensibilidade de detecção (0.0-1.0)
        
    Returns:
        Dict com imagem anotada, contornos e estatísticas detalhadas
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define HSV ranges for different vegetation types
    # Coffee plants: dark green, high saturation
    # Weeds: light green, yellowish, variable saturation
    
    # Adjust thresholds based on sensitivity
    base_lower_weed = np.array([25, 40, 40])   # Light green/yellow
    base_upper_weed = np.array([85, 255, 255])
    
    base_lower_coffee = np.array([35, 80, 30])  # Dark green
    base_upper_coffee = np.array([85, 255, 120])
    
    # Apply sensitivity adjustments
    sensitivity_factor = 1.0 + (sensitivity - 0.5) * 0.4
    
    # Create masks
    weed_mask = cv2.inRange(hsv, base_lower_weed, base_upper_weed)
    coffee_mask = cv2.inRange(hsv, base_lower_coffee, base_upper_coffee)
    vegetation_mask = cv2.bitwise_or(weed_mask, coffee_mask)
    
    # Remove coffee areas from weed mask to reduce false positives
    weed_mask = cv2.bitwise_and(weed_mask, cv2.bitwise_not(coffee_mask))
    
    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and calculate detailed stats
    min_area = int(100 * sensitivity_factor)
    max_area = img.shape[0] * img.shape[1] * 0.1  # Max 10% of image
    
    valid_contours = []
    contour_areas = []
    total_weed_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            contour_areas.append(area)
            total_weed_area += area
    
    # Create annotated image
    annotated = annotate_weeds(img, valid_contours, contour_areas)
    
    # Calculate comprehensive statistics
    image_area = img.shape[0] * img.shape[1]
    weed_percentage = (total_weed_area / image_area) * 100
    
    # Calculate vegetation coverage
    vegetation_area = cv2.countNonZero(vegetation_mask)
    coffee_area = cv2.countNonZero(coffee_mask)
    vegetation_percentage = (vegetation_area / image_area) * 100
    coffee_percentage = (coffee_area / image_area) * 100
    
    # Calculate area statistics
    area_stats = {}
    if contour_areas:
        area_stats = {
            'min_area': int(min(contour_areas)),
            'max_area': int(max(contour_areas)),
            'avg_area': round(sum(contour_areas) / len(contour_areas), 1),
            'median_area': int(np.median(contour_areas))
        }
    
    # Severity classification
    severity = classify_infestation_severity(weed_percentage)
    
    # Density calculation (weeds per m²)
    # Assuming 1 pixel = 1cm² (rough estimate for drone imagery)
    density_per_sqm = len(valid_contours) / (image_area / 10000) if image_area > 0 else 0
    
    return {
        'annotated_image': annotated,
        'contours': valid_contours,
        'weed_count': len(valid_contours),
        'total_weed_area': int(total_weed_area),
        'weed_percentage': round(weed_percentage, 2),
        'coffee_percentage': round(coffee_percentage, 2),
        'vegetation_percentage': round(vegetation_percentage, 2),
        'bare_soil_percentage': round(100 - vegetation_percentage, 2),
        'image_area': image_area,
        'area_stats': area_stats,
        'severity_level': severity,
        'density_per_sqm': round(density_per_sqm, 2),
        'sensitivity_used': sensitivity,
        'algorithm': 'HSV Color Segmentation',
        'mask': weed_mask
    }


def classify_infestation_severity(weed_percentage: float) -> str:
    """
    Classifica o nível de infestação baseado na porcentagem de ervas daninhas.
    
    Args:
        weed_percentage: Porcentagem de área infestada
        
    Returns:
        String com classificação: 'Baixa', 'Moderada', 'Alta', 'Crítica'
    """
    if weed_percentage < 5:
        return 'Baixa'
    elif weed_percentage < 15:
        return 'Moderada'
    elif weed_percentage < 30:
        return 'Alta'
    else:
        return 'Crítica'


def annotate_weeds(img: np.ndarray, contours: List[np.ndarray], areas: List[float] = None) -> np.ndarray:
    """
    Anota a imagem com contornos de ervas daninhas detectadas.
    
    Args:
        img: Imagem original
        contours: Lista de contornos detectados
        areas: Lista opcional das áreas correspondentes
        
    Returns:
        Imagem anotada com estatísticas
    """
    annotated = img.copy()
    
    # Color coding by area size
    def get_color_by_size(area):
        if area < 500:
            return (255, 255, 0)  # Yellow for small weeds
        elif area < 2000:
            return (255, 165, 0)  # Orange for medium weeds
        else:
            return (255, 0, 0)    # Red for large weeds
    
    for i, contour in enumerate(contours):
        # Get area
        area = areas[i] if areas else cv2.contourArea(contour)
        color = get_color_by_size(area)
        
        # Draw contour with size-based coloring
        cv2.drawContours(annotated, [contour], -1, color, 2)
        
        # Add area text for larger weeds
        if area > 300:  # Only show text for significant weeds
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Add area label with background
                text = f"{int(area)}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                # Background rectangle for better readability
                cv2.rectangle(annotated, 
                            (cx - text_size[0]//2 - 2, cy - text_size[1] - 2),
                            (cx + text_size[0]//2 + 2, cy + 2), 
                            (255, 255, 255), -1)
                
                cv2.putText(annotated, text, (cx - text_size[0]//2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Add comprehensive summary overlay
    total_detected = len(contours)
    total_area = sum(areas) if areas else sum(cv2.contourArea(c) for c in contours)
    image_area = img.shape[0] * img.shape[1]
    coverage_percent = (total_area / image_area) * 100
    
    # Create info box background
    info_height = 90
    cv2.rectangle(annotated, (10, 10), (400, info_height), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (400, info_height), (255, 255, 255), 2)
    
    # Add text with better formatting
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, f"Ervas daninhas: {total_detected}", 
               (15, 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Area infestada: {coverage_percent:.1f}%", 
               (15, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Legenda: Pequena=Amarelo, Media=Laranja, Grande=Vermelho", 
               (15, 75), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return annotated


def get_contour_polygons(contours: List[np.ndarray]) -> List[List[Tuple[int, int]]]:
    """
    Converte contornos OpenCV em listas de pontos para GeoJSON.
    
    Args:
        contours: Lista de contornos OpenCV
        
    Returns:
        Lista de polígonos como listas de pontos (x, y)
    """
    polygons = []
    
    for contour in contours:
        # Simplify contour to reduce points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of tuples
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
        
        # Close polygon if not already closed
        if len(polygon) > 2 and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
            
        polygons.append(polygon)
    
    return polygons


# Legacy function for backward compatibility
def detect_weeds_robust_v2(img: np.ndarray, sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    Versão 2: Mais permissiva para detectar ervas em imagens reais.
    Parâmetros ajustados baseados no feedback de que não estava detectando.
    """
    height, width = img.shape[:2]
    image_area = height * width
    
    # 1. GATE DE VERDE AINDA MAIS RELAXADO
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Parâmetros mais permissivos
    hue_min = 20   # Ainda mais amplo: 20°-110°
    hue_max = 110  
    sat_min = 0.15 + (sensitivity - 0.5) * 0.10  # 0.15-0.25 (mais baixo)
    val_min = 0.08 + (sensitivity - 0.5) * 0.06  # 0.08-0.14 (mais baixo)
    
    # Garantir limites mínimos
    sat_min = max(0.05, sat_min)
    val_min = max(0.05, val_min)
    
    lower_green = np.array([hue_min, int(sat_min * 255), int(val_min * 255)])
    upper_green = np.array([hue_max, 255, 255])
    
    green_gate = cv2.inRange(hsv, lower_green, upper_green)
    gate_pixels = cv2.countNonZero(green_gate)
    
    # 2. DETECÇÃO DE SOLO MAIS INCLUSIVA
    soil_mask = create_robust_soil_mask_v2(img, hsv)
    
    # Dilatação maior ainda
    soil_dilate_radius = max(8, int(min(width, height) * 0.012))  # Ainda maior
    kernel_soil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (soil_dilate_radius*2+1, soil_dilate_radius*2+1))
    soil_mask_dilated = cv2.dilate(soil_mask, kernel_soil, iterations=1)
    
    # 3. ExGR + OTSU COM OFFSET MAIS NEGATIVO
    veg_mask = apply_exgr_with_otsu_offset_v2(img, green_gate, offset=-18)  # Mais negativo
    
    # 4. FILTRO SOLO MENOS RESTRITIVO
    final_weed_mask = filter_vegetation_by_soil_contact_v2(veg_mask, soil_mask_dilated, 
                                                          max_distance=20, sensitivity=sensitivity)
    
    # 5. MORFOLOGIA MÍNIMA
    final_weed_mask = apply_minimal_morphology(final_weed_mask, sensitivity)
    
    # ÁREA MÍNIMA MUITO PEQUENA
    min_area = max(8, int(image_area * 0.00001 * (2.0 - sensitivity)))  # Ainda menor
    
    # Find contours
    contours, _ = cv2.findContours(final_weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    contour_areas = []
    total_weed_area = 0
    
    max_area = image_area * 0.20  # Permite até 20%
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            contour_areas.append(area)
            total_weed_area += area
    
    # Criar imagem anotada
    annotated = annotate_weeds_robust(img, valid_contours, contour_areas, 
                                    gate_pixels, soil_mask, veg_mask)
    
    # Estatísticas
    weed_percentage = (total_weed_area / image_area) * 100
    gate_percentage = (gate_pixels / image_area) * 100
    soil_percentage = (cv2.countNonZero(soil_mask) / image_area) * 100
    
    return {
        'annotated_image': annotated,
        'contours': valid_contours,
        'weed_count': len(valid_contours),
        'total_weed_area': int(total_weed_area),
        'weed_percentage': round(weed_percentage, 2),
        'gate_percentage': round(gate_percentage, 2),
        'soil_percentage': round(soil_percentage, 2),
        'vegetation_percentage': round(gate_percentage, 2),
        'bare_soil_percentage': round(100 - gate_percentage, 2),
        'image_area': image_area,
        'area_stats': calculate_area_stats(contour_areas),
        'severity_level': classify_infestation_severity(weed_percentage),
        'sensitivity_used': sensitivity,
        'algorithm': 'Robust ExGR+Otsu Pipeline V2 (Permissive)',
        'mask': final_weed_mask,
        'debug_info': {
            'gate_pixels': gate_pixels,
            'min_area_used': min_area,
            'soil_dilate_radius': soil_dilate_radius,
            'hsv_params': {
                'hue_range': f"{hue_min}-{hue_max}°",
                'sat_min': f"{sat_min:.2f}",
                'val_min': f"{val_min:.2f}"
            },
            'version': 'v2_permissive'
        }
    }


def create_robust_soil_mask_v2(img: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Versão ainda mais inclusiva da detecção de solo.
    """
    # Parâmetros ainda mais amplos
    lower_soil1 = np.array([0, 10, 10])    # Muito mais amplo
    upper_soil1 = np.array([40, 200, 220])
    
    # Solo claro/acinzentado - ainda mais inclusivo
    lower_soil2 = np.array([0, 0, 60])     # V ainda mais baixo
    upper_soil2 = np.array([40, 80, 255])
    
    # Adicionar detecção de tons acinzentados (sem cor)
    lower_soil3 = np.array([0, 0, 40])     # Cinza escuro
    upper_soil3 = np.array([180, 30, 180]) # Qualquer matiz, S muito baixo
    
    mask1 = cv2.inRange(hsv, lower_soil1, upper_soil1)
    mask2 = cv2.inRange(hsv, lower_soil2, upper_soil2)
    mask3 = cv2.inRange(hsv, lower_soil3, upper_soil3)
    soil_mask = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
    
    # Morfologia mais suave
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
    
    return soil_mask


def apply_exgr_with_otsu_offset_v2(img: np.ndarray, green_gate: np.ndarray, offset: int = -18) -> np.ndarray:
    """
    Versão mais permissiva do ExGR+Otsu.
    """
    # Mesmo cálculo ExGR
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)  
    b = img[:, :, 2].astype(np.float32)
    
    total = r + g + b + 1e-6
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total
    
    exgr = 2.0 * g_norm - r_norm - b_norm
    exgr_scaled = np.clip((exgr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    # Aplicar dentro do gate
    masked_exgr = cv2.bitwise_and(exgr_scaled, green_gate)
    
    # Otsu mais permissivo
    if cv2.countNonZero(green_gate) > 50:  # Limite menor
        gate_values = masked_exgr[green_gate > 0]
        if len(gate_values) > 0:
            # Usar percentil ao invés de Otsu em casos difíceis
            if np.std(gate_values) < 20:  # Histograma "chato"
                final_thresh = max(0, int(np.percentile(gate_values, 70)) + offset)
            else:
                thresh_otsu, _ = cv2.threshold(gate_values, 0, 255, 
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                final_thresh = max(0, thresh_otsu + offset)
        else:
            final_thresh = 80  # Mais baixo que antes
    else:
        final_thresh = 80
    
    # Aplicar limiar
    _, veg_mask = cv2.threshold(masked_exgr, final_thresh, 255, cv2.THRESH_BINARY)
    
    return veg_mask


def filter_vegetation_by_soil_contact_v2(veg_mask: np.ndarray, soil_mask_dilated: np.ndarray, 
                                        max_distance: int = 20, sensitivity: float = 0.5) -> np.ndarray:
    """
    Filtro menos restritivo para contato com solo.
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(soil_mask_dilated), 
                                         cv2.DIST_L2, 5)
    
    # Distância bem maior
    adjusted_max_dist = max_distance * (2.0 - sensitivity * 0.5)  # Até 40px
    
    # Manter vegetação próxima ao solo
    near_soil_mask = dist_transform <= adjusted_max_dist
    filtered_veg = cv2.bitwise_and(veg_mask, near_soil_mask.astype(np.uint8) * 255)
    
    return filtered_veg


def apply_minimal_morphology(mask: np.ndarray, sensitivity: float) -> np.ndarray:
    """
    Morfologia mínima para preservar detalhes.
    """
    # Apenas abertura muito suave se sensibilidade for baixa
    if sensitivity < 0.4:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fechamento suave para conectar
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    return mask


def detect_weeds_robust(img: np.ndarray, sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    Detecção robusta de ervas daninhas - versão balanceada (meio termo).
    """
    height, width = img.shape[:2]
    image_area = height * width
    
    # PARÂMETROS BALANCEADOS (meio termo entre conservador e permissivo)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Gate HSV balanceado
    hue_min = 22   # 22°-107° (meio termo)
    hue_max = 107  
    sat_min = 0.16 + (sensitivity - 0.5) * 0.09  # 0.16-0.24
    val_min = 0.09 + (sensitivity - 0.5) * 0.05  # 0.09-0.13
    
    # Garantir limites
    sat_min = max(0.08, min(0.30, sat_min))
    val_min = max(0.06, min(0.20, val_min))
    
    lower_green = np.array([hue_min, int(sat_min * 255), int(val_min * 255)])
    upper_green = np.array([hue_max, 255, 255])
    
    green_gate = cv2.inRange(hsv, lower_green, upper_green)
    gate_pixels = cv2.countNonZero(green_gate)
    
    # Solo - versão balanceada
    soil_mask = create_robust_soil_mask(img, hsv)  # Usar versão original (menos inclusiva)
    
    # Dilatação moderada
    soil_dilate_radius = max(7, int(min(width, height) * 0.010))
    kernel_soil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (soil_dilate_radius*2+1, soil_dilate_radius*2+1))
    soil_mask_dilated = cv2.dilate(soil_mask, kernel_soil, iterations=1)
    
    # ExGR+Otsu com offset moderado
    veg_mask = apply_exgr_with_otsu_offset_balanced(img, green_gate, offset=-15)
    
    # Filtro solo moderado
    final_weed_mask = filter_vegetation_by_soil_contact_balanced(veg_mask, soil_mask_dilated, 
                                                               max_distance=16, sensitivity=sensitivity)
    
    # Morfologia balanceada
    final_weed_mask = apply_balanced_morphology(final_weed_mask, sensitivity)
    
    # Área mínima balanceada
    min_area = max(12, int(image_area * 0.000015 * (1.8 - sensitivity)))
    
    # Find contours
    contours, _ = cv2.findContours(final_weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    contour_areas = []
    total_weed_area = 0
    
    max_area = image_area * 0.18
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            contour_areas.append(area)
            total_weed_area += area
    
    # Criar imagem anotada
    annotated = annotate_weeds_robust(img, valid_contours, contour_areas, 
                                    gate_pixels, soil_mask, veg_mask)
    
    # Estatísticas
    weed_percentage = (total_weed_area / image_area) * 100
    gate_percentage = (gate_pixels / image_area) * 100
    soil_percentage = (cv2.countNonZero(soil_mask) / image_area) * 100
    
    return {
        'annotated_image': annotated,
        'contours': valid_contours,
        'weed_count': len(valid_contours),
        'total_weed_area': int(total_weed_area),
        'weed_percentage': round(weed_percentage, 2),
        'gate_percentage': round(gate_percentage, 2),
        'soil_percentage': round(soil_percentage, 2),
        'vegetation_percentage': round(gate_percentage, 2),
        'bare_soil_percentage': round(100 - gate_percentage, 2),
        'image_area': image_area,
        'area_stats': calculate_area_stats(contour_areas),
        'severity_level': classify_infestation_severity(weed_percentage),
        'sensitivity_used': sensitivity,
        'algorithm': 'Robust ExGR+Otsu Pipeline (Balanced)',
        'mask': final_weed_mask,
        'debug_info': {
            'gate_pixels': gate_pixels,
            'min_area_used': min_area,
            'soil_dilate_radius': soil_dilate_radius,
            'hsv_params': {
                'hue_range': f"{hue_min}-{hue_max}°",
                'sat_min': f"{sat_min:.2f}",
                'val_min': f"{val_min:.2f}"
            },
            'version': 'balanced'
        }
    }


def apply_exgr_with_otsu_offset_balanced(img: np.ndarray, green_gate: np.ndarray, offset: int = -15) -> np.ndarray:
    """
    ExGR+Otsu balanceado.
    """
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)  
    b = img[:, :, 2].astype(np.float32)
    
    total = r + g + b + 1e-6
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total
    
    exgr = 2.0 * g_norm - r_norm - b_norm
    exgr_scaled = np.clip((exgr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    masked_exgr = cv2.bitwise_and(exgr_scaled, green_gate)
    
    if cv2.countNonZero(green_gate) > 75:  # Meio termo entre 50 e 100
        gate_values = masked_exgr[green_gate > 0]
        if len(gate_values) > 0:
            if np.std(gate_values) < 25:  # Meio termo
                final_thresh = max(0, int(np.percentile(gate_values, 75)) + offset)  # P75
            else:
                thresh_otsu, _ = cv2.threshold(gate_values, 0, 255, 
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                final_thresh = max(0, thresh_otsu + offset)
        else:
            final_thresh = 90
    else:
        final_thresh = 90
    
    _, veg_mask = cv2.threshold(masked_exgr, final_thresh, 255, cv2.THRESH_BINARY)
    return veg_mask


def filter_vegetation_by_soil_contact_balanced(veg_mask: np.ndarray, soil_mask_dilated: np.ndarray, 
                                             max_distance: int = 16, sensitivity: float = 0.5) -> np.ndarray:
    """
    Filtro solo balanceado.
    """
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(soil_mask_dilated), 
                                         cv2.DIST_L2, 5)
    
    adjusted_max_dist = max_distance * (1.7 - sensitivity * 0.4)  # 13-27px range
    
    near_soil_mask = dist_transform <= adjusted_max_dist
    filtered_veg = cv2.bitwise_and(veg_mask, near_soil_mask.astype(np.uint8) * 255)
    
    return filtered_veg


def apply_balanced_morphology(mask: np.ndarray, sensitivity: float) -> np.ndarray:
    """
    Morfologia balanceada.
    """
    # Abertura suave
    if sensitivity < 0.6:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fechamento moderado
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    return mask


def detect_weeds_robust_v1(img: np.ndarray, sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    Versão original mais conservadora (mantida para comparação).
    Implementa as correções sugeridas: gate verde relaxado, ExGR+Otsu, 
    filtro solo aprimorado e morfologia ajustada.
    
    Args:
        img: Imagem RGB de entrada
        sensitivity: Sensibilidade de detecção (0.0-1.0)
        
    Returns:
        Dict com imagem anotada, contornos e estatísticas detalhadas
    """
    height, width = img.shape[:2]
    image_area = height * width
    
    # 1. GATE DE VERDE RELAXADO (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Parâmetros relaxados conforme sugerido
    hue_min = 25   # Ampliado para 25°-105° (pega verdes amarelados/acinzentados)
    hue_max = 105  
    sat_min = 0.18 + (sensitivity - 0.5) * 0.08  # 0.18-0.22 (antes 0.25)
    val_min = 0.10 + (sensitivity - 0.5) * 0.04  # 0.10-0.12 (antes 0.15)
    
    # Conversão para escala 0-255
    lower_green = np.array([hue_min, int(sat_min * 255), int(val_min * 255)])
    upper_green = np.array([hue_max, 255, 255])
    
    green_gate = cv2.inRange(hsv, lower_green, upper_green)
    gate_pixels = cv2.countNonZero(green_gate)
    
    # 2. DETECÇÃO DE SOLO MELHORADA
    soil_mask = create_robust_soil_mask(img, hsv)
    
    # Dilatação aumentada conforme sugerido (6-10 px ao invés de 2)
    soil_dilate_radius = max(6, int(min(width, height) * 0.008))  # Adaptativo ao tamanho
    kernel_soil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (soil_dilate_radius*2+1, soil_dilate_radius*2+1))
    soil_mask_dilated = cv2.dilate(soil_mask, kernel_soil, iterations=1)
    
    # 3. ExGR + OTSU COM OFFSET NEGATIVO
    veg_mask = apply_exgr_with_otsu_offset(img, green_gate, offset=-12)
    
    # 4. FILTRO "TOCA SOLO" APRIMORADO
    # Usar distance transform ao invés de simples dilatação
    final_weed_mask = filter_vegetation_by_soil_contact(veg_mask, soil_mask_dilated, 
                                                       max_distance=12, sensitivity=sensitivity)
    
    # 5. MORFOLOGIA E ÁREA MÍNIMA AJUSTADAS
    final_weed_mask = apply_conservative_morphology(final_weed_mask, sensitivity)
    
    # Área mínima: 0.002% do total (≈ 18 px em 720×1280)
    min_area = max(18, int(image_area * 0.00002 * (1.5 - sensitivity)))
    
    # Find contours e filtragem
    contours, _ = cv2.findContours(final_weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    contour_areas = []
    total_weed_area = 0
    
    max_area = image_area * 0.15  # Max 15% da imagem
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            contour_areas.append(area)
            total_weed_area += area
    
    # Criar imagem anotada
    annotated = annotate_weeds_robust(img, valid_contours, contour_areas, 
                                    gate_pixels, soil_mask, veg_mask)
    
    # Estatísticas
    weed_percentage = (total_weed_area / image_area) * 100
    gate_percentage = (gate_pixels / image_area) * 100
    soil_percentage = (cv2.countNonZero(soil_mask) / image_area) * 100
    
    return {
        'annotated_image': annotated,
        'contours': valid_contours,
        'weed_count': len(valid_contours),
        'total_weed_area': int(total_weed_area),
        'weed_percentage': round(weed_percentage, 2),
        'gate_percentage': round(gate_percentage, 2),
        'soil_percentage': round(soil_percentage, 2),
        'vegetation_percentage': round(gate_percentage, 2),  # Aproximação
        'bare_soil_percentage': round(100 - gate_percentage, 2),
        'image_area': image_area,
        'area_stats': calculate_area_stats(contour_areas),
        'severity_level': classify_infestation_severity(weed_percentage),
        'sensitivity_used': sensitivity,
        'algorithm': 'Robust ExGR+Otsu Pipeline',
        'mask': final_weed_mask,
        'debug_info': {
            'gate_pixels': gate_pixels,
            'min_area_used': min_area,
            'soil_dilate_radius': soil_dilate_radius,
            'hsv_params': {
                'hue_range': f"{hue_min}-{hue_max}°",
                'sat_min': f"{sat_min:.2f}",
                'val_min': f"{val_min:.2f}"
            }
        }
    }


def create_robust_soil_mask(img: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """
    Cria máscara de solo mais robusta, incluindo solos claros/acinzentados.
    """
    height, width = img.shape[:2]
    
    # Parâmetros expandidos para solo
    # Inclui faixa com S baixo e V médio/alto (solo claro)
    # Amplia Hue de 5°-30° para 0°-35° se o solo for mais avermelhado
    
    # Solo marrom tradicional
    lower_soil1 = np.array([0, 20, 20])    # Ampliado: 0°-35° ao invés de 5°-30°
    upper_soil1 = np.array([35, 180, 200])
    
    # Solo claro/acinzentado (S baixo, V médio/alto)
    lower_soil2 = np.array([0, 0, 80])     # Solos claros com baixa saturação
    upper_soil2 = np.array([30, 60, 255])
    
    mask1 = cv2.inRange(hsv, lower_soil1, upper_soil1)
    mask2 = cv2.inRange(hsv, lower_soil2, upper_soil2)
    soil_mask = cv2.bitwise_or(mask1, mask2)
    
    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
    
    return soil_mask


def apply_exgr_with_otsu_offset(img: np.ndarray, green_gate: np.ndarray, offset: int = -12) -> np.ndarray:
    """
    Aplica ExGR (Excess Green minus Excess Red) com Otsu + offset negativo.
    """
    # Normalizar canais para float32
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)  
    b = img[:, :, 2].astype(np.float32)
    
    # Evitar divisão por zero
    total = r + g + b + 1e-6
    
    # Normalizar
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total
    
    # Calcular ExGR = 2*g_norm - r_norm - b_norm
    exgr = 2.0 * g_norm - r_norm - b_norm
    
    # Converter para escala 0-255
    exgr_scaled = np.clip((exgr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    
    # Aplicar apenas dentro do gate verde
    masked_exgr = cv2.bitwise_and(exgr_scaled, green_gate)
    
    # Aplicar Otsu apenas nos pixels do gate
    if cv2.countNonZero(green_gate) > 100:
        # Otsu no histograma dos pixels válidos
        gate_values = masked_exgr[green_gate > 0]
        if len(gate_values) > 0:
            thresh_otsu, _ = cv2.threshold(gate_values, 0, 255, 
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Aplicar offset negativo conforme sugerido
            final_thresh = max(0, thresh_otsu + offset)
        else:
            final_thresh = 100
    else:
        final_thresh = 100
    
    # Aplicar limiar
    _, veg_mask = cv2.threshold(masked_exgr, final_thresh, 255, cv2.THRESH_BINARY)
    
    return veg_mask


def filter_vegetation_by_soil_contact(veg_mask: np.ndarray, soil_mask_dilated: np.ndarray, 
                                    max_distance: int = 12, sensitivity: float = 0.5) -> np.ndarray:
    """
    Filtra vegetação baseado na proximidade com o solo usando distance transform.
    """
    # Distance transform da máscara de solo
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(soil_mask_dilated), 
                                         cv2.DIST_L2, 5)
    
    # Ajustar distância máxima baseada na sensibilidade
    adjusted_max_dist = max_distance * (1.5 - sensitivity * 0.5)
    
    # Manter apenas vegetação próxima ao solo
    near_soil_mask = dist_transform <= adjusted_max_dist
    filtered_veg = cv2.bitwise_and(veg_mask, near_soil_mask.astype(np.uint8) * 255)
    
    return filtered_veg


def apply_conservative_morphology(mask: np.ndarray, sensitivity: float) -> np.ndarray:
    """
    Aplica morfologia conservativa: abertura 3x3 e fechamento 5x5.
    """
    # Abertura para ruído fino (3x3)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Fechamento para unir pedaços (5x5)  
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Sensibilidade alta = menos erosão
    if sensitivity < 0.7:
        # Erosão leve apenas para sensibilidades baixas
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.erode(mask, kernel_erode, iterations=1)
    
    return mask


def calculate_area_stats(areas: List[float]) -> Dict[str, Any]:
    """Calcula estatísticas das áreas dos contornos."""
    if not areas:
        return {}
    
    return {
        'min_area': int(min(areas)),
        'max_area': int(max(areas)),
        'avg_area': round(sum(areas) / len(areas), 1),
        'median_area': int(np.median(areas)),
        'std_area': round(np.std(areas), 1)
    }


def annotate_weeds_robust(img: np.ndarray, contours: List[np.ndarray], areas: List[float],
                         gate_pixels: int, soil_mask: np.ndarray, veg_mask: np.ndarray) -> np.ndarray:
    """
    Anotação robusta com informações de debug do pipeline.
    """
    annotated = img.copy()
    
    # Desenhar contornos com cores por tamanho
    for i, contour in enumerate(contours):
        area = areas[i]
        
        if area < 500:
            color = (255, 255, 0)  # Amarelo - pequena
        elif area < 2000:
            color = (255, 165, 0)  # Laranja - média  
        else:
            color = (255, 0, 0)    # Vermelho - grande
            
        cv2.drawContours(annotated, [contour], -1, color, 2)
        
        # Adicionar área para manchas maiores
        if area > 300:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                text = f"{int(area)}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                
                cv2.rectangle(annotated, 
                            (cx - text_size[0]//2 - 2, cy - text_size[1] - 2),
                            (cx + text_size[0]//2 + 2, cy + 2), 
                            (255, 255, 255), -1)
                
                cv2.putText(annotated, text, (cx - text_size[0]//2, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Info box com estatísticas do pipeline
    image_area = img.shape[0] * img.shape[1]
    total_area = sum(areas)
    coverage_percent = (total_area / image_area) * 100
    gate_percent = (gate_pixels / image_area) * 100
    
    info_height = 120
    cv2.rectangle(annotated, (10, 10), (450, info_height), (0, 0, 0), -1)
    cv2.rectangle(annotated, (10, 10), (450, info_height), (255, 255, 255), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, f"Ervas detectadas: {len(contours)} | Area: {coverage_percent:.1f}%", 
               (15, 30), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Gate verde: {gate_percent:.1f}% | Pipeline: ExGR+Otsu", 
               (15, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Solo detectado: {cv2.countNonZero(soil_mask)/image_area*100:.1f}%", 
               (15, 70), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Cores: Pequena=Amarelo, Media=Laranja, Grande=Vermelho", 
               (15, 95), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return annotated


def placeholder_annotate(img: np.ndarray) -> np.ndarray:
    """
    Legacy placeholder function - now uses robust detection.
    """
    result = detect_weeds_robust(img)
    return result['annotated_image']
