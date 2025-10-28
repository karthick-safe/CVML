#!/usr/bin/env python3
"""
Enhanced training script for CardioChek Plus YOLO detection model
"""

import os
import yaml
import json
from ultralytics import YOLO
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_training_data():
    """Validate and prepare training data"""
    base_dir = "/Users/karthickrajamurugan/Safe/CVML"
    yolo_dir = Path(base_dir) / "data" / "cardio_chek" / "yolo"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"

    # Check if directories exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Get image files
    image_files = list(images_dir.glob("*.jpg"))
    label_files = list(labels_dir.glob("*.txt"))

    logger.info(f"Found {len(image_files)} images and {len(label_files)} label files")

    # Validate that each image has a corresponding label file
    valid_pairs = []
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img_file, label_file))
        else:
            logger.warning(f"No label file for image: {img_file}")

    logger.info(f"Valid image-label pairs: {len(valid_pairs)}")

    if len(valid_pairs) < 3:
        raise ValueError("Not enough valid training data. Need at least 3 image-label pairs.")

    return valid_pairs

def create_enhanced_dataset():
    """Create enhanced dataset with data augmentation"""
    base_dir = "/Users/karthickrajamurugan/Safe/CVML"
    yolo_dir = Path(base_dir) / "data" / "cardio_chek" / "yolo"

    # Create enhanced dataset directory
    enhanced_dir = yolo_dir / "enhanced"
    enhanced_images = enhanced_dir / "images"
    enhanced_labels = enhanced_dir / "labels"

    enhanced_images.mkdir(parents=True, exist_ok=True)
    enhanced_labels.mkdir(parents=True, exist_ok=True)

    # Copy existing data and create augmented versions
    image_files = list((yolo_dir / "images").glob("*.jpg"))

    for img_file in image_files:
        # Copy original
        label_file = yolo_dir / "labels" / f"{img_file.stem}.txt"

        if label_file.exists():
            # Copy original image and label
            import shutil
            shutil.copy2(img_file, enhanced_images / img_file.name)
            shutil.copy2(label_file, enhanced_labels / label_file.name)

            # Create augmented versions (brightness, contrast, rotation)
            img = cv2.imread(str(img_file))

            # Brightness augmentation
            bright_img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            cv2.imwrite(str(enhanced_images / f"{img_file.stem}_bright.jpg"), bright_img)
            shutil.copy2(label_file, enhanced_labels / f"{img_file.stem}_bright.txt")

            # Contrast augmentation
            contrast_img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
            cv2.imwrite(str(enhanced_images / f"{img_file.stem}_contrast.jpg"), contrast_img)
            shutil.copy2(label_file, enhanced_labels / f"{img_file.stem}_contrast.txt")

    logger.info(f"Enhanced dataset created in {enhanced_dir}")
    return enhanced_dir

def train_cardio_chek_detector():
    """Train YOLO model for CardioChek Plus detection with enhanced data"""

    # Validate training data
    valid_pairs = validate_training_data()

    # Create enhanced dataset
    enhanced_dir = create_enhanced_dataset()

    # Create dataset YAML for enhanced data
    enhanced_yaml = enhanced_dir / "dataset.yaml"
    dataset_config = {
        "names": ["cardio_chek_plus"],
        "nc": 1,
        "path": str(enhanced_dir),
        "train": "images",
        "val": "images"
    }

    with open(enhanced_yaml, 'w') as f:
        yaml.dump(dataset_config, f)

    # Paths
    base_dir = "/Users/karthickrajamurugan/Safe/CVML"
    model_path = os.path.join(base_dir, "backend", "cardio_chek_models", "cardio_chek_detector3", "weights", "best.pt")
    output_dir = os.path.join(base_dir, "backend", "cardio_chek_models", "cardio_chek_detector_enhanced")

    # Verify existing model exists
    if not os.path.exists(model_path):
        logger.warning(f"Existing model not found: {model_path}, starting from scratch")
        model_path = "yolov8n.pt"  # Use base YOLO model

    try:
        # Load the model
        logger.info(f"Loading model from: {model_path}")
        model = YOLO(model_path)

        # Enhanced training configuration for better accuracy
        training_args = {
            "data": str(enhanced_yaml),
            "epochs": 100,  # More epochs for better training
            "imgsz": 640,
            "batch": 16,  # Larger batch size for better convergence
            "lr0": 0.001,  # Lower learning rate for fine-tuning
            "lrf": 0.0001,  # Lower final learning rate
            "patience": 30,  # More patience for convergence
            "save": True,
            "save_period": 20,
            "project": "cardio_chek_models",
            "name": "cardio_chek_detector_enhanced",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "cos_lr": True,  # Cosine annealing for better convergence
            "close_mosaic": 10,
            "plots": True,
            "val": True,
            "augment": True,  # Enable data augmentation
            "hsv_h": 0.02,  # HSV hue augmentation
            "hsv_s": 0.3,  # HSV saturation augmentation
            "hsv_v": 0.3,  # HSV value augmentation
            "degrees": 10,  # Rotation augmentation
            "translate": 0.2,  # Translation augmentation
            "scale": 0.3,  # Scale augmentation
            "shear": 5,  # Shear augmentation
            "flipud": 0.1,  # Vertical flip augmentation
            "fliplr": 0.5,  # Horizontal flip augmentation
            "mosaic": 1.0,  # Mosaic augmentation
            "mixup": 0.2,  # Mixup augmentation
            "device": "cpu"  # Use CPU for compatibility
        }

        logger.info("Starting enhanced CardioChek Plus detector training...")
        logger.info(f"Training arguments: {training_args}")

        # Train the model
        results = model.train(**training_args)

        logger.info("Training completed successfully!")
        logger.info(f"Enhanced model saved to: {output_dir}")

        # Save training results and configuration
        results_path = os.path.join(output_dir, "training_results.json")
        config_path = os.path.join(output_dir, "training_config.json")

        with open(results_path, 'w') as f:
            json.dump(str(results), f, indent=2)

        with open(config_path, 'w') as f:
            json.dump(training_args, f, indent=2)

        # Save enhanced dataset info
        dataset_info = {
            "original_pairs": len(valid_pairs),
            "enhanced_pairs": len(list(enhanced_dir.glob("images/*.jpg"))),
            "augmentation_factor": 3,
            "total_training_images": len(list(enhanced_dir.glob("images/*.jpg")))
        }

        with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_cardio_chek_detector()
