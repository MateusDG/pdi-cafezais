import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


def detect_weeds_hsv(img: np.ndarray, sensitivity: float = 0.5) -> Dict[str, Any]:
    """
    Detecta ervas daninhas usando segmentação HSV.
    
    Args:
        img: Imagem RGB de entrada
        sensitivity: Sensibilidade de detecção (0.0-1.0)
        
    Returns:
        Dict com imagem anotada, contornos e estatísticas
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
    
    # Remove coffee areas from weed mask to reduce false positives
    weed_mask = cv2.bitwise_and(weed_mask, cv2.bitwise_not(coffee_mask))
    
    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_CLOSE, kernel)
    weed_mask = cv2.morphologyEx(weed_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = int(100 * sensitivity_factor)
    max_area = img.shape[0] * img.shape[1] * 0.1  # Max 10% of image
    
    valid_contours = []
    total_weed_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            total_weed_area += area
    
    # Create annotated image
    annotated = annotate_weeds(img, valid_contours)
    
    # Calculate statistics
    image_area = img.shape[0] * img.shape[1]
    weed_percentage = (total_weed_area / image_area) * 100
    
    return {
        'annotated_image': annotated,
        'contours': valid_contours,
        'weed_count': len(valid_contours),
        'total_weed_area': int(total_weed_area),
        'weed_percentage': round(weed_percentage, 2),
        'image_area': image_area,
        'mask': weed_mask
    }


def annotate_weeds(img: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """
    Anota a imagem com contornos de ervas daninhas detectadas.
    
    Args:
        img: Imagem original
        contours: Lista de contornos detectados
        
    Returns:
        Imagem anotada
    """
    annotated = img.copy()
    
    for i, contour in enumerate(contours):
        # Draw contour in red
        cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)
        
        # Add area text
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Add small label with area
            cv2.putText(annotated, f"{int(area)}", (cx-20, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    
    # Add summary text
    total_detected = len(contours)
    cv2.putText(annotated, f"Ervas daninhas detectadas: {total_detected}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
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
