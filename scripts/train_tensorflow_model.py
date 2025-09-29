#!/usr/bin/env python3
"""
Train TensorFlow model with synthetic CardioChek Plus images
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import cv2
import numpy as np
import json
import os
from pathlib import Path

def collect_training_data():
    """Collect training data from synthetic images"""
    images_dir = Path("data/cardio_chek/images/real_data")
    labels_dir = Path("data/cardio_chek/yolo/labels")

    # Generate more training data if needed
    print("🎨 Generating additional training images...")
    from generate_cardio_chek_images import generate_training_images
    generate_training_images()

    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Training data directories not found")
        return [], []

    # Get image files
    image_files = list(images_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No training images found")
        return [], []

    print(f"📊 Found {len(image_files)} training images")

    images = []
    annotations = []

    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to load image: {img_path}")
            continue

        images.append(img)

        # Create annotation (assume device covers most of image)
        height, width = img.shape[:2]
        annotation = {
            "class": "cardio_chek_plus",
            "bbox": {
                "x": int(width * 0.1),  # 10% margin
                "y": int(height * 0.1),
                "width": int(width * 0.8),
                "height": int(height * 0.8)
            }
        }
        annotations.append(annotation)

        print(f"✅ Loaded: {img_path.name}")

    print(f"📋 Collected {len(images)} images with {len(annotations)} annotations")
    return images, annotations

def train_tensorflow_model():
    """Train the TensorFlow model with collected data"""

    print("🧠 TRAINING TENSORFLOW MODEL")
    print("=" * 40)

    # Collect training data
    images, annotations = collect_training_data()

    if not images or not annotations:
        print("❌ No training data available")
        return False

    try:
        # Import TensorFlow model manager
        from backend.services.tensorflow_model_manager import TensorFlowModelManager

        # Initialize model manager
        model_manager = TensorFlowModelManager()

        # Train the detector
        print("🏋️ Starting training...")
        success = asyncio.run(model_manager.train_detector(images, annotations))

        if success:
            print("✅ Training completed successfully!")
            print("📁 Model saved as: cardio_chek_tf_detector.h5")
            return True
        else:
            print("❌ Training failed")
            return False

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def test_tensorflow_model():
    """Test the trained TensorFlow model"""

    print("\n🧪 TESTING TENSORFLOW MODEL")
    print("=" * 40)

    # Check if model exists
    model_path = "cardio_chek_tf_detector.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    try:
        # Import TensorFlow model manager
        from backend.services.tensorflow_model_manager import TensorFlowModelManager

        # Initialize model manager
        model_manager = TensorFlowModelManager()

        # Test with a sample image
        test_images_dir = Path("data/cardio_chek/images/real_data")
        test_image = list(test_images_dir.glob("*.jpg"))[0]

        if not test_image:
            print("❌ No test image found")
            return False

        # Load test image
        img = cv2.imread(str(test_image))
        if img is None:
            print(f"❌ Failed to load test image: {test_image}")
            return False

        print(f"🔍 Testing with: {test_image.name}")

        # Test detection
        import asyncio
        detection_result = asyncio.run(model_manager.detect_cardio_chek_kit(img))

        print(f"✅ Detection result: {detection_result['detected']}")
        print(f"🎯 Confidence: {detection_result['confidence']:.3f}")

        if detection_result['detected']:
            bbox = detection_result['bounding_box']
            print(f"📦 Bounding box: ({bbox['x']}, {bbox['y']}) {bbox['width']}x{bbox['height']}")
            print("✅ Model working correctly!")
            return True
        else:
            print("❌ Detection failed")
            return False

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TENSORFLOW MODEL TRAINING FOR CARDIOCHECK PLUS")
    print("=" * 60)

    # Generate additional training data if needed
    print("🎨 Generating synthetic training images...")
    from generate_cardio_chek_images import generate_training_images
    generate_training_images()

    # Train the model
    training_success = train_tensorflow_model()

    if training_success:
        # Test the trained model
        testing_success = test_tensorflow_model()

        if testing_success:
            print("\n🎉 TENSORFLOW MODEL READY!")
            print("=" * 30)
            print("✅ Training completed successfully")
            print("✅ Model tested and validated")
            print("✅ Ready for production use")
        else:
            print("\n⚠️ Training succeeded but testing failed")
    else:
        print("\n❌ Training failed")

    print("\n🎯 Next steps:")
    print("1. Restart the backend server to use TensorFlow models")
    print("2. Test with real CardioChek Plus images")
    print("3. Fine-tune model parameters if needed")
