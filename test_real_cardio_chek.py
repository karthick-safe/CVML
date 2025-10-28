#!/usr/bin/env python3
"""
Test script for real CardioChek Plus device detection and OCR
"""

import cv2
import numpy as np
import asyncio
import logging
from pathlib import Path

# Import our enhanced model manager
from backend.services.tensorflow_model_manager import TensorFlowModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_cardio_chek():
    """Test the trained model with real CardioChek Plus device"""
    try:
        # Initialize model manager
        model_manager = TensorFlowModelManager()
        await model_manager.initialize()

        # Test with multiple real device images
        test_images = [
            "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_01.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_02.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_03.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_04.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_05.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/backend/data/cardio_chek/images/IMG_20250815_123634073.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/backend/data/cardio_chek/images/IMG_20250815_123638229.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/backend/data/cardio_chek/images/IMG_20250815_123640167.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/backend/data/cardio_chek/images/IMG_20250815_123642707.jpg",
            "/Users/karthickrajamurugan/Safe/CVML/backend/data/cardio_chek/images/IMG_20250815_123644183.jpg"
        ]

        logger.info("🧪 Testing Enhanced Model with Real CardioChek Plus Devices")
        logger.info("=" * 60)

        results = []

        for i, image_path in enumerate(test_images, 1):
            if not Path(image_path).exists():
                logger.warning(f"⚠️  Test image {i} not found: {image_path}")
                continue

            logger.info(f"\n📷 Testing Image {i}: {Path(image_path).name}")

            # Load and test image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Failed to load image: {image_path}")
                continue

            logger.info(f"   Image size: {image.shape}")

            # Step 1: Test YOLO Detection
            detection_result = await model_manager.detect_cardio_chek_kit(image)

            if detection_result["detected"]:
                logger.info(f"   ✅ DETECTION: Success ({detection_result['confidence']:.3f} confidence)")
                logger.info(f"   📍 Bounding Box: {detection_result['bounding_box']}")
                logger.info(f"   ⏱️  Processing: {detection_result['processing_time']:.3f}s")

                # Step 2: Test OCR on detected region
                bbox = detection_result["bounding_box"]
                if bbox:
                    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
                    screen_region = image[y:y+h, x:x+w]

                    if screen_region.size > 0:
                        ocr_result = await model_manager.extract_screen_values(screen_region)

                        if ocr_result["success"]:
                            logger.info(f"   ✅ OCR: Success - {len([k for k, v in ocr_result['values'].items() if v is not None])} values extracted")

                            # Show extracted values
                            values = ocr_result["values"]
                            if values.get("cholesterol"):
                                logger.info(f"      🩸 Cholesterol: {values['cholesterol']} {values.get('units', 'mg/dL')}")
                            if values.get("hdl"):
                                logger.info(f"      🫀 HDL: {values['hdl']} {values.get('units', 'mg/dL')}")
                            if values.get("triglycerides"):
                                logger.info(f"      🩸 Triglycerides: {values['triglycerides']} {values.get('units', 'mg/dL')}")
                            if values.get("glucose"):
                                logger.info(f"      🩸 Glucose: {values['glucose']} {values.get('units', 'mg/dL')}")

                            # Test classification
                            if any(values.get(key) for key in ["cholesterol", "hdl", "triglycerides", "glucose"]):
                                classification_result = await model_manager.classify_cardio_chek_result(values)
                                logger.info(f"   🏥 CLASSIFICATION: {classification_result['result']} ({classification_result['confidence']:.1%})")

                        else:
                            logger.warning(f"   ❌ OCR: Failed to extract values")
                            logger.info(f"      💡 Try adjusting camera angle, lighting, or device position")

                    else:
                        logger.warning("   ❌ Could not extract screen region for OCR")

            else:
                logger.warning(f"   ❌ DETECTION: Failed (confidence: {detection_result['confidence']:.3f})")
                logger.info(f"      💡 Ensure device is clearly visible and well-lit")

            results.append({
                "image": Path(image_path).name,
                "detection_success": detection_result["detected"],
                "detection_confidence": detection_result["confidence"],
                "ocr_success": detection_result["detected"] and bbox is not None,
                "values_extracted": len([k for k, v in ocr_result.get("values", {}).items() if v is not None]) if detection_result["detected"] and bbox else 0
            })

        # Summary
        logger.info(f"\n📊 TEST SUMMARY")
        logger.info("=" * 60)

        total_tests = len(results)
        detection_successes = sum(1 for r in results if r["detection_success"])
        ocr_successes = sum(1 for r in results if r["ocr_success"])
        avg_values = sum(r["values_extracted"] for r in results) / total_tests if total_tests > 0 else 0

        logger.info(f"📈 Detection Accuracy: {detection_successes}/{total_tests} ({detection_successes/total_tests*100:.1f}%)")
        logger.info(f"📈 OCR Success Rate: {ocr_successes}/{total_tests} ({ocr_successes/total_tests*100:.1f}%)")
        logger.info(f"📈 Average Values Extracted: {avg_values:.1f} per image")

        if detection_successes / total_tests > 0.8:
            logger.info("✅ MODEL PERFORMANCE: EXCELLENT - Ready for production!")
        elif detection_successes / total_tests > 0.6:
            logger.info("⚠️  MODEL PERFORMANCE: GOOD - May need fine-tuning for some devices")
        else:
            logger.info("❌ MODEL PERFORMANCE: NEEDS IMPROVEMENT - Consider retraining with more diverse data")

        return results

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🔬 Starting Comprehensive CardioChek Plus Model Test...")
    asyncio.run(test_real_cardio_chek())
    logger.info("✨ Test completed!")
