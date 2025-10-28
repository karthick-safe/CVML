#!/usr/bin/env python3
"""
Test script for improved OCR functionality
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

async def test_ocr_improvements():
    """Test the improved OCR with sample images"""
    try:
        # Initialize model manager
        model_manager = TensorFlowModelManager()
        await model_manager.initialize()

        # Test with a sample image that should contain screen values
        test_image_path = "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_01.jpg"

        if Path(test_image_path).exists():
            # Load test image
            image = cv2.imread(test_image_path)
            if image is None:
                logger.error(f"Failed to load image: {test_image_path}")
                return

            logger.info(f"Testing OCR with image: {test_image_path} (shape: {image.shape})")

            # For testing OCR, we need to simulate the screen region extraction
            # Let's assume the screen is roughly in the center of the image
            height, width = image.shape[:2]
            # Simulate a bounding box around the center (typical CardioChek screen location)
            screen_x = int(width * 0.2)
            screen_y = int(height * 0.3)
            screen_w = int(width * 0.6)
            screen_h = int(height * 0.4)

            # Extract screen region
            kit_region = image[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w]

            if kit_region.size == 0:
                logger.error("Failed to extract screen region")
                return

            logger.info(f"Extracted screen region: {kit_region.shape}")

            # Run OCR on the screen region
            ocr_result = await model_manager.extract_screen_values(kit_region)

            logger.info(f"OCR result: {ocr_result}")

            # Check if OCR was successful
            if ocr_result["success"]:
                logger.info("✅ SUCCESS: OCR extracted values!")
                logger.info(f"   Values: {ocr_result['values']}")
                logger.info(f"   Processing time: {ocr_result['processing_time']:.3f}s")

                # Test classification
                if ocr_result["values"]:
                    classification_result = await model_manager.classify_cardio_chek_result(ocr_result["values"])
                    logger.info(f"Classification result: {classification_result}")

            else:
                logger.warning("❌ OCR failed to extract values")
                logger.info(f"   Raw OCR results: {len(ocr_result.get('raw_ocr', []))} items")
                if ocr_result.get("raw_ocr"):
                    ocr_text = [item.get("text", "") for item in ocr_result["raw_ocr"] if item.get("confidence", 0) > 0.1]
                    logger.info(f"   Detected text: {ocr_text}")
                logger.info(f"   Processing time: {ocr_result['processing_time']:.3f}s")

        else:
            logger.error(f"Test image not found: {test_image_path}")

    except Exception as e:
        logger.error(f"OCR test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting OCR improvements test...")
    asyncio.run(test_ocr_improvements())
    logger.info("OCR test completed!")
