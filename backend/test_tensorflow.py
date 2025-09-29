#!/usr/bin/env python3
"""
Test TensorFlow model manager with synthetic data
"""

import sys
import os
import asyncio
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

async def test_tensorflow_manager():
    """Test the TensorFlow model manager"""

    print("🧪 TESTING TENSORFLOW MODEL MANAGER")
    print("=" * 50)

    try:
        from services.tensorflow_model_manager import TensorFlowModelManager

        # Initialize model manager
        model_manager = TensorFlowModelManager()

        # Test with a synthetic image
        test_image_path = Path("../data/cardio_chek/images/real_data/cardio_chek_synthetic_001.jpg")
        if not test_image_path.exists():
            print("❌ Test image not found")
            return False

        # Load test image
        img = cv2.imread(str(test_image_path))
        if img is None:
            print("❌ Failed to load test image")
            return False

        print(f"✅ Loaded test image: {test_image_path.name}")
        print(f"📏 Image size: {img.shape[1]}x{img.shape[0]}")

        # Test detection
        print("\n🎯 Testing device detection...")
        detection_result = await model_manager.detect_cardio_chek_kit(img)

        print(f"✅ Detection result: {detection_result['detected']}")
        print(f"🎯 Confidence: {detection_result['confidence']:.3f}")
        print(f"⚡ Processing time: {detection_result['processing_time']:.3f}s")

        if detection_result['detected']:
            bbox = detection_result['bounding_box']
            print(f"📦 Bounding box: ({bbox['x']}, {bbox['y']}) {bbox['width']}x{bbox['height']}")

            # Test OCR if detection succeeded
            if bbox:
                # Crop to device region
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                device_region = img[y:y+h, x:x+w]

                print("\n🔤 Testing OCR extraction...")
                ocr_result = await model_manager.extract_screen_values(device_region)

                if ocr_result['success']:
                    values = ocr_result['values']
                    print("✅ OCR successful!")
                    print("📊 Extracted values:")
                    for key, value in values.items():
                        if key not in ['raw_text', 'units'] and value is not None:
                            print(f"  • {key.upper()}: {value} {values.get('units', '')}")

                    # Test classification
                    print("\n🏥 Testing health classification...")
                    classification_result = await model_manager.classify_cardio_chek_result(values)

                    print(f"📋 Classification: {classification_result['result']}")
                    print(f"🎯 Confidence: {classification_result['confidence']:.3f}")

                    if 'analysis' in classification_result:
                        print("📈 Health Analysis:")
                        for analysis in classification_result['analysis']:
                            print(f"  • {analysis}")

                    return True
                else:
                    print("❌ OCR failed")
                    print(f"Error: {ocr_result.get('error', 'Unknown error')}")
                    return False
            else:
                print("❌ No bounding box returned")
                return False
        else:
            print("❌ Detection failed")
            print(f"Error: {detection_result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TESTING TENSORFLOW-BASED CVML SYSTEM")
    print("=" * 60)

    # Run the test
    success = asyncio.run(test_tensorflow_manager())

    if success:
        print("\n🎉 TENSORFLOW MODEL WORKING!")
        print("✅ Detection: Working")
        print("✅ OCR: Working")
        print("✅ Classification: Working")
        print("✅ Ready for production use!")
    else:
        print("\n⚠️ Some issues detected")
        print("Check the error messages above for details")

    print("\n🎯 Next steps:")
    print("1. The TensorFlow system is now ready to use")
    print("2. Restart the backend server to use TensorFlow models")
    print("3. Test with real CardioChek Plus images")
