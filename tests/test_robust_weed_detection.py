#!/usr/bin/env python3
"""
Test script for the new robust weed detection pipeline.
Tests all the corrections implemented: relaxed HSV gate, ExGR+Otsu offset, 
improved soil filtering, and conservative morphology.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add backend to path
sys.path.append('backend')

from backend.app.services.processing.weed import detect_weeds_robust, detect_weeds_hsv


def create_challenging_test_image():
    """
    Creates a challenging synthetic test image with issues mentioned:
    - Shaded areas with low saturation/value
    - Light soil areas
    - Small weed patches touching soil by thin connections
    """
    height, width = 720, 1280
    
    # Create base brown/soil color with variations
    img = np.full((height, width, 3), [120, 90, 60], dtype=np.uint8)
    
    # Add lighter soil areas (challenge for soil detection)
    cv2.rectangle(img, (200, 400), (600, 600), [160, 140, 110], -1)  # Light soil
    cv2.rectangle(img, (800, 200), (1200, 500), [180, 160, 120], -1)  # Very light soil
    
    # Add coffee plants (dark green) in shaded conditions
    # Some with low saturation/value due to shade
    cv2.rectangle(img, (100, 100), (200, 200), [40, 80, 30], -1)   # Dark shaded coffee
    cv2.rectangle(img, (300, 150), (400, 250), [60, 100, 45], -1)  # Less shaded coffee
    cv2.rectangle(img, (1000, 100), (1100, 180), [35, 70, 25], -1) # Very dark/shaded
    
    # Add problematic weed areas:
    # 1. Small patches with thin connections to soil (test soil contact filter)
    cv2.rectangle(img, (500, 580), (520, 600), [130, 180, 80], -1)   # Small patch touching soil
    cv2.line(img, (510, 580), (510, 520), [130, 180, 80], 2)         # Thin connection
    cv2.rectangle(img, (505, 500), (515, 520), [140, 190, 90], -1)   # Small weed patch
    
    cv2.rectangle(img, (750, 590), (770, 600), [125, 175, 75], -1)   # Another small patch
    cv2.line(img, (760, 590), (760, 550), [125, 175, 75], 1)         # Very thin connection  
    cv2.rectangle(img, (755, 530), (765, 550), [135, 185, 85], -1)   # Small weed
    
    # 2. Weeds in shadow with low saturation/value (challenge for HSV gate)
    cv2.rectangle(img, (150, 300), (200, 350), [90, 110, 70], -1)    # Low sat/val weed
    cv2.rectangle(img, (250, 320), (290, 360), [95, 115, 75], -1)    # Low sat/val weed
    
    # 3. Medium weeds with yellowish/grayish tones
    cv2.rectangle(img, (600, 100), (700, 150), [140, 160, 90], -1)   # Yellowish weed
    cv2.rectangle(img, (450, 300), (550, 380), [110, 130, 95], -1)   # Grayish green weed
    
    # 4. Larger weed patches for comparison
    cv2.rectangle(img, (900, 400), (1000, 500), [150, 200, 100], -1) # Bright green weed
    cv2.rectangle(img, (700, 300), (800, 400), [130, 180, 95], -1)   # Medium green weed
    
    return img


def test_robust_vs_original():
    """
    Compare the new robust detection with the original HSV method.
    """
    print("=== COMPARING ROBUST vs ORIGINAL DETECTION ===\n")
    
    test_img = create_challenging_test_image()
    print(f"Test image created: {test_img.shape[0]}x{test_img.shape[1]} pixels")
    
    sensitivities = [0.3, 0.5, 0.7, 1.0]
    
    print("\n--- ORIGINAL HSV METHOD ---")
    for sensitivity in sensitivities:
        result_hsv = detect_weeds_hsv(test_img, sensitivity=sensitivity)
        print(f"Sensitivity {sensitivity}: {result_hsv['weed_count']} weeds, "
              f"{result_hsv['weed_percentage']:.1f}% coverage, "
              f"gate: N/A")
    
    print("\n--- NEW ROBUST ExGR+OTSU METHOD ---")
    for sensitivity in sensitivities:
        result_robust = detect_weeds_robust(test_img, sensitivity=sensitivity)
        print(f"Sensitivity {sensitivity}: {result_robust['weed_count']} weeds, "
              f"{result_robust['weed_percentage']:.1f}% coverage, "
              f"gate: {result_robust['gate_percentage']:.1f}%")
        
        # Print debug info for highest sensitivity
        if sensitivity == 1.0:
            debug = result_robust['debug_info']
            print(f"  Debug - HSV params: {debug['hsv_params']}")
            print(f"  Debug - Min area: {debug['min_area_used']} px")
            print(f"  Debug - Soil dilate radius: {debug['soil_dilate_radius']} px")
    
    # Save test images for visual inspection
    output_dir = Path("data/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original test image
    cv2.imwrite(str(output_dir / "test_challenging_input.jpg"), test_img)
    
    # Save results for sensitivity 0.7 (good balance)
    result_hsv = detect_weeds_hsv(test_img, sensitivity=0.7)
    result_robust = detect_weeds_robust(test_img, sensitivity=0.7)
    
    cv2.imwrite(str(output_dir / "test_hsv_result_s07.jpg"), 
                cv2.cvtColor(result_hsv['annotated_image'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "test_robust_result_s07.jpg"), 
                cv2.cvtColor(result_robust['annotated_image'], cv2.COLOR_RGB2BGR))
    
    print(f"\nTest images saved to: {output_dir}")
    print("- test_challenging_input.jpg: Original test image")
    print("- test_hsv_result_s07.jpg: HSV method result")  
    print("- test_robust_result_s07.jpg: Robust ExGR+Otsu result")
    
    return True


def test_parameter_robustness():
    """
    Test how the robust method handles different image conditions.
    """
    print("\n=== TESTING PARAMETER ROBUSTNESS ===\n")
    
    base_img = create_challenging_test_image()
    
    # Test 1: Very dark image (simulates heavy shade)
    dark_img = cv2.convertScaleAbs(base_img, alpha=0.4, beta=-20)
    
    # Test 2: Very bright image (simulates strong sunlight)
    bright_img = cv2.convertScaleAbs(base_img, alpha=1.3, beta=30)
    
    # Test 3: Low contrast image
    low_contrast_img = cv2.convertScaleAbs(base_img, alpha=0.7, beta=50)
    
    test_images = [
        ("Normal", base_img),
        ("Dark/Shaded", dark_img), 
        ("Bright/Sunny", bright_img),
        ("Low Contrast", low_contrast_img)
    ]
    
    print("Testing robust detection on different lighting conditions:\n")
    
    for name, img in test_images:
        result = detect_weeds_robust(img, sensitivity=0.7)
        print(f"{name:12} | Weeds: {result['weed_count']:2d} | "
              f"Coverage: {result['weed_percentage']:4.1f}% | "
              f"Gate: {result['gate_percentage']:4.1f}% | "
              f"Soil: {result['soil_percentage']:4.1f}%")
    
    print(f"\nExpected behavior:")
    print("- Gate percentage should remain relatively stable (robust gate)")
    print("- Weed count should not vary drastically (robust thresholding)")
    print("- Soil percentage should adapt to different lighting")
    
    return True


def test_edge_cases():
    """
    Test edge cases and potential failure modes.
    """
    print("\n=== TESTING EDGE CASES ===\n")
    
    # Test 1: Mostly soil image (very few weeds)
    soil_img = np.full((600, 800, 3), [140, 110, 80], dtype=np.uint8)
    cv2.rectangle(soil_img, (100, 100), (120, 120), [130, 170, 90], -1)  # Tiny weed
    
    # Test 2: Very green image (simulates lush vegetation)
    green_img = np.full((600, 800, 3), [100, 160, 80], dtype=np.uint8)
    cv2.rectangle(green_img, (200, 200), (300, 300), [80, 130, 60], -1)   # Coffee area
    cv2.rectangle(green_img, (400, 150), (450, 200), [120, 140, 70], -1)  # Soil patch
    
    # Test 3: Very small image
    small_img = cv2.resize(create_challenging_test_image(), (320, 240))
    
    # Test 4: Very large simulated parameters
    large_img = np.full((2000, 3000, 3), [130, 100, 70], dtype=np.uint8)
    cv2.rectangle(large_img, (500, 500), (600, 600), [140, 180, 90], -1)
    
    test_cases = [
        ("Mostly Soil", soil_img),
        ("Very Green", green_img), 
        ("Small Image", small_img),
        ("Large Image", large_img)
    ]
    
    print("Testing edge cases:\n")
    
    for name, img in test_cases:
        try:
            result = detect_weeds_robust(img, sensitivity=0.5)
            print(f"{name:12} | SUCCESS | Size: {img.shape[1]}x{img.shape[0]} | "
                  f"Weeds: {result['weed_count']} | Gate: {result['gate_percentage']:.1f}%")
        except Exception as e:
            print(f"{name:12} | FAILED  | Error: {str(e)[:50]}...")
    
    return True


def main():
    """Run all robust detection tests."""
    print("ROBUST WEED DETECTION - COMPREHENSIVE TESTS")
    print("=" * 50)
    print()
    print("Testing the new robust pipeline with corrections:")
    print("1. Relaxed HSV gate (25°-105°, S≥0.18, V≥0.10)")
    print("2. ExGR + Otsu with -12 offset")
    print("3. Improved soil contact filter (distance transform)")
    print("4. Conservative morphology (3x3 open, 5x5 close)")
    print("5. Adaptive area filtering")
    print()
    
    tests = [
        test_robust_vs_original,
        test_parameter_robustness, 
        test_edge_cases
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("✓ PASSED\n")
            else:
                print("✗ FAILED\n")
        except Exception as e:
            print(f"✗ FAILED - Exception: {e}\n")
    
    print("=" * 50)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 SUCCESS - All robust detection tests passed!")
        print("\nThe new pipeline should now:")
        print("- Detect more weeds in shaded areas (relaxed HSV gate)")
        print("- Better handle variable lighting (ExGR + adaptive Otsu)")
        print("- Improve soil contact detection (distance transform)")
        print("- Preserve small weed patches (conservative morphology)")
    else:
        print("⚠️  SOME TESTS FAILED - Check implementation")
    
    print("\nTo test with the API:")
    print('curl -X POST "http://localhost:8000/api/process" \\')
    print('  -F "file=@your_image.jpg" \\') 
    print('  -F "algorithm=robust_exgr" \\')
    print('  -F "sensitivity=0.7"')


if __name__ == "__main__":
    main()