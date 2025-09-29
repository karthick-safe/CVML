#!/usr/bin/env python3
"""
Train TensorFlow model with synthetic CardioChek Plus images
Run this from the backend directory
"""

import sys
import os
import asyncio
import cv2
import numpy as np
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def collect_training_data():
    """Collect training data from annotated images"""
    annotations_file = Path("../data/cardio_chek/annotations/annotations_summary.json")
    images_base_dir = Path("../data/cardio_chek/images")

    if not annotations_file.exists():
        print("❌ Annotations file not found")
        return [], []

    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    print(f"📊 Found {data['total_annotations']} annotated images")

    images = []
    annotations = []

    for annotation in data['annotations']:
        # Find image file
        image_name = annotation['image']
        
        # Try different possible locations
        possible_paths = [
            images_base_dir / image_name,
            images_base_dir / "real_data" / image_name,
            Path("../backend/data/cardio_chek/images") / image_name,
            Path("..") / image_name
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            print(f"⚠️  Image not found: {image_name}")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Failed to load image: {img_path}")
            continue

        images.append(img)
        annotations.append(annotation)

        print(f"✅ Loaded: {image_name}")

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
        from services.tensorflow_model_manager import TensorFlowCardioChekDetector

        # Initialize detector directly
        detector = TensorFlowCardioChekDetector()
        
        # Extract bounding boxes for training
        bboxes = [ann['bbox'] for ann in annotations]

        # Train the detector
        print("🏋️ Starting training...")
        print(f"Training with {len(images)} images...")
        
        history = detector.train(images, bboxes, epochs=50)

        print("✅ Training completed successfully!")
        print("📁 Model saved as: cardio_chek_tf_detector.h5")
        return True

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 TENSORFLOW MODEL TRAINING FOR CARDIOCHECK PLUS")
    print("=" * 60)

    # Train the model
    training_success = train_tensorflow_model()

    if training_success:
        print("\n🎉 TRAINING COMPLETED!")
        print("✅ TensorFlow model trained successfully")
        print("✅ Ready for use in the API")
    else:
        print("\n❌ Training failed")
