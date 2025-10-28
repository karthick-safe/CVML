#!/usr/bin/env python3
"""
Test script for the enhanced CardioChek Plus detection model
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

async def test_enhanced_model():
    """Test the enhanced model with sample images"""
    try:
        # Initialize model manager
        model_manager = TensorFlowModelManager()
        await model_manager.initialize()

        # Check model status
        status = await model_manager.get_model_status()
        logger.info(f"Model status: {status}")

        # Test with a sample image
        test_image_path = "/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/yolo/images/cardio_chek_sample_01.jpg"

        if Path(test_image_path).exists():
            # Load test image
            image = cv2.imread(test_image_path)
            if image is None:
                logger.error(f"Failed to load image: {test_image_path}")
                return

            logger.info(f"Testing with image: {test_image_path} (shape: {image.shape})")

            # Run detection
            result = await model_manager.detect_cardio_chek_kit(image)

            logger.info(f"Detection result: {result}")

            # Check if detection was successful
            if result["detected"]:
                logger.info("✅ SUCCESS: CardioChek Plus detected!")
                logger.info(f"   Confidence: {result['confidence']:.3f}")
                logger.info(f"   Bounding box: {result['bounding_box']}")
                logger.info(f"   Processing time: {result['processing_time']:.3f}s")
                logger.info(f"   Method: {result['method']}")
            else:
                logger.warning("❌ No CardioChek Plus detected")
                if "error" in result:
                    logger.error(f"   Error: {result['error']}")
        else:
            logger.error(f"Test image not found: {test_image_path}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting enhanced model test...")
    asyncio.run(test_enhanced_model())
    logger.info("Test completed!")
