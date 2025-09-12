#!/usr/bin/env python3
import cv2
import sys
import os
import numpy as np

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.processing.oblique_pipeline import oblique_weed_detection_pipeline

def test_oblique_pipeline():
    image_path = r"C:\Users\mateu\Desktop\pdi-cafezais\data\samples\test_coffee_field.jpg"
    
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded successfully. Shape: {img_rgb.shape}")
    
    try:
        print("Running oblique pipeline...")
        results = oblique_weed_detection_pipeline(img_rgb)
        
        print(f"\n=== RESULTS ===")
        print(f"Weeds detected: {results['weed_count']}")
        print(f"Coverage: {results['weed_percentage']:.2f}%")
        print(f"Quality flags: {results['quality_flags']}")
        
        if 'statistics' in results:
            stats = results['statistics']
            print(f"Small weeds: {stats['small_weeds']}")
            print(f"Medium weeds: {stats['medium_weeds']}") 
            print(f"Large weeds: {stats['large_weeds']}")
        
        # Save result
        result_path = "test_coffee_field_oblique_result.jpg"
        cv2.imwrite(result_path, cv2.cvtColor(results['annotated_image'], cv2.COLOR_RGB2BGR))
        print(f"\nResult saved to: {result_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_oblique_pipeline()