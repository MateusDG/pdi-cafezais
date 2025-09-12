#!/usr/bin/env python3
"""
Script para debugar passo a passo o pipeline de detecção de ervas daninhas
e identificar onde está o problema.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.append('backend')

from backend.app.services.processing.weed import detect_weeds_robust


def debug_detection_pipeline(img, sensitivity=0.5):
    """
    Debugar cada etapa do pipeline para identificar onde está falhando.
    """
    print("=== DEBUG PIPELINE STEP BY STEP ===")
    height, width = img.shape[:2]
    image_area = height * width
    
    print(f"1. Imagem de entrada: {width}x{height} ({image_area} pixels)")
    
    # 1. GATE DE VERDE
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    hue_min = 25
    hue_max = 105  
    sat_min = 0.18 + (sensitivity - 0.5) * 0.08
    val_min = 0.10 + (sensitivity - 0.5) * 0.04
    
    lower_green = np.array([hue_min, int(sat_min * 255), int(val_min * 255)])
    upper_green = np.array([hue_max, 255, 255])
    
    green_gate = cv2.inRange(hsv, lower_green, upper_green)
    gate_pixels = cv2.countNonZero(green_gate)
    
    print(f"2. Gate HSV: {lower_green} - {upper_green}")
    print(f"   Gate pixels: {gate_pixels} ({gate_pixels/image_area*100:.1f}%)")
    
    if gate_pixels < 100:
        print("   ⚠️  PROBLEMA: Muito poucos pixels passaram pelo gate!")
        print("   → Tente relaxar mais os limites HSV ou verificar se há vegetação na imagem")
        return
    
    # 2. SOLO
    # Solo marrom tradicional
    lower_soil1 = np.array([0, 20, 20])
    upper_soil1 = np.array([35, 180, 200])
    
    # Solo claro/acinzentado
    lower_soil2 = np.array([0, 0, 80])
    upper_soil2 = np.array([30, 60, 255])
    
    mask1 = cv2.inRange(hsv, lower_soil1, upper_soil1)
    mask2 = cv2.inRange(hsv, lower_soil2, upper_soil2)
    soil_mask = cv2.bitwise_or(mask1, mask2)
    
    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
    
    soil_pixels = cv2.countNonZero(soil_mask)
    print(f"3. Solo detectado: {soil_pixels} pixels ({soil_pixels/image_area*100:.1f}%)")
    
    # 3. ExGR
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
    
    print(f"4. ExGR calculado e mascarado")
    
    # 4. OTSU
    if cv2.countNonZero(green_gate) > 100:
        gate_values = masked_exgr[green_gate > 0]
        if len(gate_values) > 0:
            thresh_otsu, _ = cv2.threshold(gate_values, 0, 255, 
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            offset = -12
            final_thresh = max(0, thresh_otsu + offset)
            print(f"5. Otsu: {thresh_otsu} → Final (com offset): {final_thresh}")
        else:
            final_thresh = 100
            print(f"5. Otsu: falhou, usando thresh=100")
    else:
        final_thresh = 100
        print(f"5. Otsu: pulado, usando thresh=100")
    
    # Aplicar limiar
    _, veg_mask = cv2.threshold(masked_exgr, final_thresh, 255, cv2.THRESH_BINARY)
    veg_pixels = cv2.countNonZero(veg_mask)
    print(f"6. Vegetação após Otsu: {veg_pixels} pixels ({veg_pixels/image_area*100:.1f}%)")
    
    if veg_pixels == 0:
        print("   ⚠️  PROBLEMA: Nenhum pixel passou pelo ExGR+Otsu!")
        print("   → O offset pode estar muito negativo ou a imagem não tem vegetação suficiente")
        return
    
    # 5. FILTRO SOLO
    soil_dilate_radius = max(6, int(min(width, height) * 0.008))
    kernel_soil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (soil_dilate_radius*2+1, soil_dilate_radius*2+1))
    soil_mask_dilated = cv2.dilate(soil_mask, kernel_soil, iterations=1)
    
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(soil_mask_dilated), 
                                         cv2.DIST_L2, 5)
    
    max_distance = 12
    adjusted_max_dist = max_distance * (1.5 - sensitivity * 0.5)
    near_soil_mask = dist_transform <= adjusted_max_dist
    filtered_veg = cv2.bitwise_and(veg_mask, near_soil_mask.astype(np.uint8) * 255)
    
    filtered_pixels = cv2.countNonZero(filtered_veg)
    print(f"7. Após filtro solo (dist≤{adjusted_max_dist:.1f}): {filtered_pixels} pixels ({filtered_pixels/image_area*100:.1f}%)")
    
    if filtered_pixels == 0:
        print("   ⚠️  PROBLEMA: Filtro de solo removeu toda a vegetação!")
        print("   → Distância máxima muito restritiva ou solo mal detectado")
        return
    
    # 6. MORFOLOGIA
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered_veg = cv2.morphologyEx(filtered_veg, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered_veg = cv2.morphologyEx(filtered_veg, cv2.MORPH_CLOSE, kernel_close)
    
    if sensitivity < 0.7:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        filtered_veg = cv2.erode(filtered_veg, kernel_erode, iterations=1)
    
    morph_pixels = cv2.countNonZero(filtered_veg)
    print(f"8. Após morfologia: {morph_pixels} pixels ({morph_pixels/image_area*100:.1f}%)")
    
    # 7. CONTORNOS E ÁREA MÍNIMA
    contours, _ = cv2.findContours(filtered_veg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = max(18, int(image_area * 0.00002 * (1.5 - sensitivity)))
    max_area = image_area * 0.15
    
    valid_contours = []
    total_area = 0
    
    print(f"9. Contornos encontrados: {len(contours)}")
    print(f"   Área mínima: {min_area} pixels")
    print(f"   Área máxima: {max_area} pixels")
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
            total_area += area
            print(f"     Contorno {i}: área={area:.0f} → ACEITO")
        else:
            print(f"     Contorno {i}: área={area:.0f} → REJEITADO (fora dos limites)")
    
    print(f"10. RESULTADO FINAL:")
    print(f"    Ervas detectadas: {len(valid_contours)}")
    print(f"    Área total: {total_area} pixels ({total_area/image_area*100:.1f}%)")
    
    return {
        'gate_pixels': gate_pixels,
        'soil_pixels': soil_pixels, 
        'veg_pixels': veg_pixels,
        'filtered_pixels': filtered_pixels,
        'morph_pixels': morph_pixels,
        'contours_found': len(contours),
        'valid_contours': len(valid_contours),
        'final_area': total_area
    }


def create_test_image_with_weeds():
    """Criar imagem de teste que definitivamente deve detectar ervas."""
    height, width = 600, 800
    
    # Solo marrom
    img = np.full((height, width, 3), [120, 80, 50], dtype=np.uint8)
    
    # Ervas bem verdes, grandes, tocando o solo
    cv2.rectangle(img, (100, 500), (200, 600), [50, 180, 50], -1)   # Verde forte no solo
    cv2.rectangle(img, (300, 480), (450, 600), [80, 200, 80], -1)   # Verde médio no solo  
    cv2.rectangle(img, (500, 450), (650, 600), [100, 220, 100], -1) # Verde claro no solo
    
    return img


def main():
    print("DEBUG: Detecção de Ervas Daninhas")
    print("=" * 50)
    
    # Teste 1: Imagem sintética com ervas óbvias
    print("\nTESTE 1: Imagem sintética com ervas óbvias")
    test_img = create_test_image_with_weeds()
    debug_stats = debug_detection_pipeline(test_img, sensitivity=0.7)
    
    # Teste 2: Comparar com detecção real
    print("\n" + "="*50)
    print("TESTE 2: Comparando com função real")
    result = detect_weeds_robust(test_img, 0.7)
    print(f"Função real detectou: {result['weed_count']} ervas")
    print(f"Cobertura: {result['weed_percentage']}%")
    
    # Salvar imagem de teste para inspeção visual
    output_dir = Path("data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "debug_test_image.jpg"), 
                cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
    print(f"\nImagem de teste salva em: {output_dir / 'debug_test_image.jpg'}")


if __name__ == "__main__":
    main()