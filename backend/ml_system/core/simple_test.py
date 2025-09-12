#!/usr/bin/env python3
"""
Simple test for ML implementation
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        from ml_features import FeatureExtractor
        print("  [OK] FeatureExtractor imported")
        
        from ml_classifiers import ClassicalMLWeedDetector
        print("  [OK] ClassicalMLWeedDetector imported")
        
        from ml_training import MLTrainingPipeline
        print("  [OK] MLTrainingPipeline imported")
        
        return True
        
    except ImportError as e:
        print(f"  [ERROR] Import failed: {e}")
        return False

def test_feature_extractor():
    """Test feature extraction on a simple image."""
    print("Testing feature extraction...")
    
    try:
        from ml_features import FeatureExtractor
        
        # Create a simple test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        extractor = FeatureExtractor()
        features = extractor.extract_region_features(img, mask)
        
        print(f"  [OK] Extracted {len(features)} features")
        
        # Check for key features
        key_features = ['color_r_mean', 'color_exg', 'texture_glcm_contrast', 'shape_area']
        for feature in key_features:
            if feature in features:
                print(f"    {feature}: {features[feature]:.3f}")
            else:
                print(f"    [WARNING] Missing feature: {feature}")
                
        return True
        
    except Exception as e:
        print(f"  [ERROR] Feature extraction failed: {e}")
        return False

def test_detector_creation():
    """Test ML detector creation."""
    print("Testing detector creation...")
    
    try:
        from ml_classifiers import ClassicalMLWeedDetector
        
        detector = ClassicalMLWeedDetector()
        print("  [OK] Detector created successfully")
        
        feature_count = len(detector.feature_extractor.get_all_feature_names())
        print(f"  [OK] Total features available: {feature_count}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Detector creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Simple ML Implementation Test")
    print("=" * 60)
    
    tests = [
        ("Import test", test_imports),
        ("Feature extraction test", test_feature_extractor),
        ("Detector creation test", test_detector_creation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Summary:")
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results):
        print("\nAll tests passed! ML implementation is working.")
    else:
        print("\nSome tests failed. Check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()