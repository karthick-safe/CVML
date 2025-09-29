#!/usr/bin/env python3
"""
Test image processing directly
"""

import sys
sys.path.append('/Users/karthickrajamurugan/Safe/CVML/backend')

from services.real_model_manager import RealModelManager
import asyncio
import numpy as np
from PIL import Image

async def test_image_processing():
    """Test image processing directly"""
    
    try:
        # Load the image
        image_path = "data/cardio_chek/images/cardio_chek_sample_01.jpg"
        image = Image.open(image_path)
        image_array = np.array(image)
        
        print(f"Image loaded: {image.size}")
        print(f"Image array shape: {image_array.shape}")
        
        # Initialize model manager
        model_manager = RealModelManager()
        await model_manager.load_models()
        
        print("Models loaded successfully")
        
        # Test detection
        print("Testing detection...")
        detection_result = await model_manager.detect_cardio_chek_kit(image_array)
        print(f"Detection result: {detection_result}")
        
        if detection_result["detected"]:
            # Test OCR
            print("Testing OCR...")
            bbox = detection_result["bounding_box"]
            x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
            kit_region = image_array[y:y+h, x:x+w]
            
            ocr_result = await model_manager.extract_screen_values(kit_region)
            print(f"OCR result: {ocr_result}")
            
            if ocr_result["success"]:
                # Test classification
                print("Testing classification...")
                classification_result = await model_manager.classify_cardio_chek_result(ocr_result["values"])
                print(f"Classification result: {classification_result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_image_processing())
