import cv2
import numpy as np
from typing import Tuple


def detect_sky_mask(img: np.ndarray) -> np.ndarray:
    """
    Detecta e remove céu da imagem usando critérios HSV.
    
    Args:
        img: Imagem RGB [0-255]
    
    Returns:
        Máscara binária do céu (255 = céu, 0 = não-céu)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    
    # Sky criteria: low saturation + high value (clouds/blue sky)
    # Also consider upper portion of image (perspective photos)
    sky_mask = (s < 0.2) & (v > 0.6)
    
    # Additional constraint: focus on upper 40% of image for sky
    h, w = img.shape[:2]
    upper_region = np.zeros_like(sky_mask, dtype=bool)
    upper_region[:int(h * 0.4), :] = True
    
    # Sky is likely in upper region OR very low saturation everywhere
    sky_mask = sky_mask & (upper_region | (s < 0.1))
    
    return sky_mask.astype(np.uint8) * 255


def get_working_region_mask(img: np.ndarray, bottom_fraction: float = 0.7) -> np.ndarray:
    """
    Cria máscara da região útil de trabalho (porção inferior da imagem).
    
    Args:
        img: Imagem RGB
        bottom_fraction: Fração da parte inferior a considerar (0.7 = 70% de baixo)
    
    Returns:
        Máscara da região de trabalho
    """
    h, w = img.shape[:2]
    working_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Work only in bottom fraction
    start_row = int(h * (1.0 - bottom_fraction))
    working_mask[start_row:, :] = 255
    
    return working_mask


def create_vegetation_gate(img: np.ndarray) -> np.ndarray:
    """
    Cria gate inicial baseado em cor verde para evitar falsos positivos.
    
    Args:
        img: Imagem RGB [0-255]
    
    Returns:
        Máscara de gate verde
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    h = hsv[:, :, 0].astype(np.float32)  # [0-360]
    s = hsv[:, :, 1].astype(np.float32) / 255.0  # [0-1]
    v = hsv[:, :, 2].astype(np.float32) / 255.0  # [0-1]
    
    # Green gate: H in [35°, 95°], S > 0.25, V > 0.15
    # Convert OpenCV H (0-180) to degrees (0-360)
    h_degrees = h * 2.0  # OpenCV H is 0-179 for 0-358 degrees
    
    green_gate = (
        (h_degrees >= 35) & (h_degrees <= 95) &  # Green hues
        (s > 0.25) &  # Sufficient saturation
        (v > 0.15)    # Not too dark
    )
    
    return green_gate.astype(np.uint8) * 255


def apply_conservative_vegetation_indices(img: np.ndarray, 
                                        green_gate: np.ndarray,
                                        primary_index: str = 'ExGR') -> np.ndarray:
    """
    Aplica índices de vegetação apenas dentro do gate verde.
    
    Args:
        img: Imagem RGB [0-255]
        green_gate: Máscara do gate verde
        primary_index: Índice primário a usar
    
    Returns:
        Máscara de vegetação conservativa
    """
    # Normalize to [0, 1]
    r = img[:, :, 0].astype(np.float32) / 255.0
    g = img[:, :, 1].astype(np.float32) / 255.0  
    b = img[:, :, 2].astype(np.float32) / 255.0
    
    # Calculate index within green gate only
    if primary_index == 'ExG':
        index = 2 * g - r - b
    elif primary_index == 'ExGR':
        exg = 2 * g - r - b
        exr = 1.4 * r - b
        index = exg - exr
    else:  # CIVE
        R = img[:, :, 0].astype(np.float32)
        G = img[:, :, 1].astype(np.float32)
        B = img[:, :, 2].astype(np.float32)
        index = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
        # Normalize CIVE to [0, 1] range like others
        index = (index - np.min(index)) / (np.max(index) - np.min(index) + 1e-10)
    
    # Apply conservative threshold: mean + 0.5*std (not global min-max + Otsu)
    gate_pixels = index[green_gate > 0]
    if len(gate_pixels) > 100:  # Enough pixels for statistics
        threshold = np.mean(gate_pixels) + 0.5 * np.std(gate_pixels)
        vegetation_mask = (index > threshold) & (green_gate > 0)
    else:
        # Fallback if not enough green pixels
        vegetation_mask = green_gate > 0
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(vegetation_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    
    return cleaned


def detect_row_orientation_conservative(vegetation_mask: np.ndarray) -> float:
    """
    Detecta orientação das linhas usando apenas a máscara de vegetação.
    
    Args:
        vegetation_mask: Máscara conservativa de vegetação
    
    Returns:
        Ângulo dominante em graus (0 se não detectado)
    """
    # Edge detection only on vegetation
    edges = cv2.Canny(vegetation_mask, 50, 150, apertureSize=3)
    
    # Check if we have enough edges
    if np.sum(edges > 0) < 100:
        return 0.0  # Not enough structure for line detection
    
    # Hough line detection with stricter parameters
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
    
    if lines is None or len(lines) < 3:
        return 0.0  # Not enough lines detected
    
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
    
    # Use mode of angles (cluster analysis)
    if len(angles) > 0:
        angles = np.array(angles)
        # Simple clustering: find most common angle range
        hist, bins = np.histogram(angles, bins=18, range=(-90, 90))  # 10-degree bins
        dominant_bin = np.argmax(hist)
        return (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
    
    return 0.0


def create_row_mask_restricted(vegetation_mask: np.ndarray, 
                             angle: float,
                             row_spacing_px: int = 60) -> np.ndarray:
    """
    Cria máscara de linhas restrita à vegetação detectada.
    
    Args:
        vegetation_mask: Máscara de vegetação
        angle: Ângulo das linhas
        row_spacing_px: Espaçamento entre linhas
    
    Returns:
        Máscara de linhas restrita à vegetação
    """
    if abs(angle) < 5:  # Nearly horizontal or no strong orientation
        return vegetation_mask.copy()  # Don't try to detect rows
    
    h, w = vegetation_mask.shape
    
    # Create oriented lines at the detected angle
    row_mask = np.zeros_like(vegetation_mask)
    
    # Simple approach: create parallel lines at detected orientation
    center_x, center_y = w // 2, h // 2
    
    # Create multiple parallel lines
    for offset in range(-w, w, row_spacing_px):
        # Line equation: y = mx + b, where m = tan(angle)
        m = np.tan(np.radians(angle))
        
        for x in range(w):
            y = int(center_y + m * (x - center_x) + offset * np.cos(np.radians(angle)))
            if 0 <= y < h:
                # Draw thick line (simulate crown width)
                for dy in range(-15, 16):  # 30px width
                    if 0 <= y + dy < h:
                        row_mask[y + dy, x] = 255
    
    # Restrict to actual vegetation areas
    row_mask = cv2.bitwise_and(row_mask, vegetation_mask)
    
    return row_mask