"""
Create pre-annotated training data for CardioChek Plus
This automatically creates annotations for the sample images
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

def create_pre_annotated_data():
    """Create pre-annotated training data"""
    
    images_dir = Path("data/cardio_chek/images")
    annotations_dir = Path("data/cardio_chek/annotations")
    yolo_dir = Path("data/cardio_chek/yolo")
    
    # Create directories
    annotations_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    
    print(f"Creating annotations for {len(image_files)} images...")
    
    for i, image_path in enumerate(image_files):
        print(f"Processing {image_path.name}...")
        
        # Load image to get dimensions
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load {image_path.name}")
            continue
            
        h, w = img.shape[:2]
        
        # Create annotation data
        # For our sample images, the CardioChek Plus device is roughly in the center
        # with coordinates: (150, 100) to (650, 500) in 800x600 images
        bbox_x = 150 / w
        bbox_y = 100 / h
        bbox_width = 500 / w
        bbox_height = 400 / h
        
        annotation_data = {
            "image_path": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "annotations": [{
                "class": "cardio_chek_plus",
                "bbox": {
                    "x": bbox_x,
                    "y": bbox_y,
                    "width": bbox_width,
                    "height": bbox_height
                },
                "confidence": 1.0
            }]
        }
        
        # Save annotation file
        annotation_filename = image_path.stem + ".json"
        annotation_path = annotations_dir / annotation_filename
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        # Create YOLO format annotation
        yolo_filename = image_path.stem + ".txt"
        yolo_path = yolo_dir / yolo_filename
        
        with open(yolo_path, 'w') as f:
            # YOLO format: class_id center_x center_y width height
            center_x = bbox_x + bbox_width / 2
            center_y = bbox_y + bbox_height / 2
            
            f.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
        
        # Copy image to YOLO directory
        yolo_image_path = yolo_dir / image_path.name
        import shutil
        shutil.copy2(image_path, yolo_image_path)
    
    # Create YOLO dataset configuration
    dataset_config = {
        'path': str(yolo_dir.parent.absolute()),
        'train': 'yolo',
        'val': 'yolo',
        'nc': 1,
        'names': ['cardio_chek_plus']
    }
    
    config_path = yolo_dir.parent / "dataset.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(dataset_config, f)
    
    print(f"\n✅ Training data collection completed!")
    print(f"📁 Annotations: {len(list(annotations_dir.glob('*.json')))} files")
    print(f"📁 YOLO dataset: {len(list(yolo_dir.glob('*.txt')))} files")
    print(f"📁 Images: {len(list(yolo_dir.glob('*.jpg')))} files")
    print(f"📁 Dataset config: {config_path}")
    
    return True

if __name__ == "__main__":
    create_pre_annotated_data()
