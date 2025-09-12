#!/usr/bin/env python3
"""
Test script for Week 1 implementation
"""

import sys
import os
import requests
import numpy as np
import cv2
from pathlib import Path

# Add backend to path
sys.path.append('backend')

from backend.app.services.processing import weed, utils


def create_test_image():
    """Creates a synthetic test image with green areas"""
    height, width = 600, 800
    
    # Create base brown/soil color
    img = np.full((height, width, 3), [139, 102, 69], dtype=np.uint8)
    
    # Add some coffee plant areas (dark green)
    cv2.rectangle(img, (100, 100), (200, 200), (34, 139, 34), -1)  # Dark green
    cv2.rectangle(img, (300, 150), (400, 250), (50, 150, 50), -1)
    
    # Add some weed areas (lighter green/yellowish)
    cv2.rectangle(img, (500, 100), (600, 150), (154, 205, 50), -1)  # Yellow green
    cv2.rectangle(img, (200, 300), (300, 350), (173, 255, 47), -1)  # Green yellow
    cv2.rectangle(img, (450, 300), (550, 400), (144, 238, 144), -1) # Light green
    
    return img


def test_weed_detection():
    """Test the weed detection algorithm"""
    print("Testing weed detection algorithm...")
    
    # Create test image
    test_img = create_test_image()
    
    # Test with different sensitivities
    for sensitivity in [0.3, 0.5, 0.7]:
        print(f"  Testing with sensitivity: {sensitivity}")
        
        result = weed.detect_weeds_hsv(test_img, sensitivity=sensitivity)
        
        print(f"    Areas detected: {result['weed_count']}")
        print(f"    Coverage: {result['weed_percentage']:.1f}%")
        print(f"    Total weed area: {result['total_weed_area']} pixels")
        
        assert isinstance(result, dict)
        assert 'annotated_image' in result
        assert 'weed_count' in result
        assert result['weed_count'] >= 0
        
    print("PASS - Weed detection tests passed!")
    return True


def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    test_img = create_test_image()
    
    # Test image stats
    stats = utils.calculate_image_stats(test_img)
    print(f"  Image stats: {stats['width']}x{stats['height']}, {stats['channels']} channels")
    
    assert stats['width'] == 800
    assert stats['height'] == 600
    assert stats['channels'] == 3
    
    # Test resize functionality
    resized, scale = utils.resize_image_if_needed(test_img, max_size=400)
    print(f"  Resize: scale factor = {scale:.2f}, new size = {resized.shape[:2]}")
    
    # Test color conversion
    rgb_img = utils.bgr_to_rgb(test_img)
    bgr_img = utils.rgb_to_bgr(rgb_img)
    
    assert rgb_img.shape == test_img.shape
    assert bgr_img.shape == test_img.shape
    
    print("PASS - Utility function tests passed!")
    return True


def test_api_endpoint():
    """Test the API endpoint if server is running"""
    print("Testing API endpoint...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("  PASS - Health endpoint working")
        
        # Test status endpoint
        response = requests.get("http://localhost:8000/api/process/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            print(f"  PASS - Status endpoint working: {status_data['status']}")
            print(f"    Supported formats: {status_data['supported_formats']}")
            print(f"    Max file size: {status_data['max_file_size_mb']}MB")
        
        return True
        
    except requests.exceptions.RequestException:
        print("  WARN - API server not running. Start with: uvicorn app.main:app --reload")
        return False


def save_test_image():
    """Save a test image for manual testing"""
    print("Creating test image for manual testing...")
    
    test_img = create_test_image()
    output_path = Path("data/samples/test_coffee_field.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), test_img)
    print(f"  Test image saved to: {output_path}")
    print("  You can use this image to test the web interface!")
    
    return True


def main():
    """Run all tests"""
    print("Starting Week 1 Implementation Tests\n")
    
    tests = [
        test_weed_detection,
        test_utils,
        save_test_image,
        test_api_endpoint
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  FAIL - Test failed: {e}")
        print()
    
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("SUCCESS - All tests passed! Week 1 implementation is ready!")
    else:
        print("WARN - Some tests failed. Check the issues above.")
    
    print("\nNext steps:")
    print("1. Start backend: cd backend && uvicorn app.main:app --reload")
    print("2. Start frontend: cd frontend && npm run dev")
    print("3. Open http://localhost:5173 and test with the sample image")


if __name__ == "__main__":
    main()