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
def placeholder_annotate(img: np.ndarray) -> np.ndarray:
    """
    Legacy placeholder function - now uses real HSV detection.
    """
    result = detect_weeds_hsv(img)
    return result['annotated_image']
