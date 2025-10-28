#!/usr/bin/env python3
"""
Test script to verify the fix for the '60' OCR issue
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

async def test_60_issue_fix():
    """Test that the '60' issue has been fixed"""
    try:
        # Initialize model manager
        model_manager = TensorFlowModelManager()
        await model_manager.initialize()

        # Test with a sample image that previously gave "60" results
        test_image_path = "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_01.jpg"

        if Path(test_image_path).exists():
            # Load test image
            image = cv2.imread(test_image_path)
            if image is None:
                logger.error(f"Failed to load image: {test_image_path}")
                return

            logger.info(f"Testing OCR improvements with image: {test_image_path}")

            # For testing OCR, simulate screen region extraction
            height, width = image.shape[:2]
            screen_x = int(width * 0.2)
            screen_y = int(height * 0.3)
            screen_w = int(width * 0.6)
            screen_h = int(height * 0.4)

            kit_region = image[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w]

            if kit_region.size == 0:
                logger.error("Failed to extract screen region")
                return

            logger.info(f"Extracted screen region: {kit_region.shape}")

            # Run OCR on the screen region
            ocr_result = await model_manager.extract_screen_values(kit_region)

            logger.info(f"OCR result: {ocr_result}")

            # Check if OCR was successful and medically reasonable
            if ocr_result["success"]:
                logger.info("✅ SUCCESS: OCR extracted values!")
                values = ocr_result["values"]

                # Check for any suspicious "60" values that shouldn't be there
                suspicious_values = []
                for key, value in values.items():
                    if key in ['cholesterol', 'hdl', 'triglycerides', 'glucose'] and value is not None:
                        if value == 60:
                            suspicious_values.append(f"{key}: {value}")

                if suspicious_values:
                    logger.warning(f"⚠️  Found suspicious '60' values: {suspicious_values}")
                    logger.warning("This indicates the OCR is still picking up false positives")
                else:
                    logger.info("✅ No suspicious '60' values found - issue appears to be fixed!")

                # Show all extracted values
                logger.info("📊 Extracted Values:")
                for key in ['cholesterol', 'hdl', 'triglycerides', 'glucose']:
                    value = values.get(key)
                    if value is not None:
                        logger.info(f"   {key.upper()}: {value} {values.get('units', 'mg/dL')}")

            else:
                logger.warning("❌ OCR failed to extract values")

        else:
            logger.error(f"Test image not found: {test_image_path}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("🔍 Testing fix for '60' OCR issue...")
    asyncio.run(test_60_issue_fix())
    logger.info("✨ Test completed!")
