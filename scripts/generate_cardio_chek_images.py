#!/usr/bin/env python3
"""
Generate synthetic CardioChek Plus images for training
Based on the provided real image characteristics
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import random

def create_cardio_chek_display(readings, test_id=None, time_str=None, date_str=None):
    """Create a CardioChek Plus display image"""
    
    # Image dimensions (based on typical CardioChek Plus screen)
    width, height = 640, 480
    
    # Create background (light greenish-blue)
    bg_color = (180, 200, 190)  # Light greenish-blue
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Add device frame (darker border)
    cv2.rectangle(img, (20, 20), (width-20, height-20), (50, 50, 50), 3)
    
    # Screen area (recessed)
    screen_x, screen_y = 40, 50
    screen_w, screen_h = width-80, height-100
    cv2.rectangle(img, (screen_x, screen_y), (screen_x+screen_w, screen_y+screen_h), (30, 30, 30), 2)
    
    # Screen background (slightly darker)
    screen_bg = (160, 180, 170)
    cv2.rectangle(img, (screen_x+5, screen_y+5), (screen_x+screen_w-5, screen_y+screen_h-5), screen_bg, -1)
    
    # Header
    header_y = screen_y + 30
    cv2.putText(img, "eGLU+LIPIDS", (screen_x + 20, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    
    # Test ID
    test_id_text = test_id or f"{random.randint(1000, 9999)}"
    cv2.putText(img, test_id_text, (screen_x + screen_w - 100, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    
    # Navigation arrow
    arrow_points = np.array([[screen_x + screen_w - 30, header_y - 5], 
                            [screen_x + screen_w - 20, header_y], 
                            [screen_x + screen_w - 30, header_y + 5]], np.int32)
    cv2.fillPoly(img, [arrow_points], (20, 20, 20))
    
    # Main readings
    readings_y = screen_y + 80
    line_height = 40
    
    # CHOL
    cv2.putText(img, "CHOL", (screen_x + 30, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, f"{readings['chol']} mg/dL", (screen_x + 150, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    
    # HDL CHOL
    readings_y += line_height
    cv2.putText(img, "HDL CHOL", (screen_x + 30, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, f"{readings['hdl']} mg/dL", (screen_x + 150, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    
    # TRIG
    readings_y += line_height
    cv2.putText(img, "TRIG", (screen_x + 30, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, f"{readings['trig']} mg/dL", (screen_x + 150, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    
    # eGLU
    readings_y += line_height
    cv2.putText(img, "eGLU", (screen_x + 30, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, f"{readings['glu']} mg/dL", (screen_x + 150, readings_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    
    # Footer - Time and Date
    footer_y = screen_y + screen_h - 30
    time_text = time_str or f"{random.randint(1, 12):02d}:{random.randint(0, 59):02d} {'AM' if random.random() > 0.5 else 'PM'}"
    date_text = date_str or f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/2025"
    
    cv2.putText(img, time_text, (screen_x + 20, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, date_text, (screen_x + screen_w - 120, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    
    return img

def generate_training_images():
    """Generate multiple CardioChek Plus images for training"""
    
    # Create directories
    images_dir = Path("data/cardio_chek/images/real_data")
    labels_dir = Path("data/cardio_chek/yolo/labels")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training variations
    variations_path = Path(__file__).parent.parent / "data" / "cardio_chek" / "training_variations.json"
    with open(variations_path, "r") as f:
        variations = json.load(f)
    
    print("🎨 GENERATING SYNTHETIC CARDIOCHECK PLUS IMAGES")
    print("=" * 50)
    
    generated_images = []
    
    for i, variation in enumerate(variations):
        # Generate image
        img = create_cardio_chek_display(
            readings=variation,
            test_id=f"{random.randint(1000, 9999)}",
            time_str=f"{random.randint(1, 12):02d}:{random.randint(0, 59):02d} {'AM' if random.random() > 0.5 else 'PM'}",
            date_str=f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/2025"
        )
        
        # Add some noise and variations
        if random.random() > 0.5:
            # Add dust particles
            for _ in range(random.randint(1, 5)):
                x, y = random.randint(50, 590), random.randint(50, 430)
                cv2.circle(img, (x, y), random.randint(1, 3), (255, 255, 255), -1)
        
        if random.random() > 0.5:
            # Add slight blur
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Save image
        filename = f"cardio_chek_synthetic_{i+1:03d}.jpg"
        img_path = images_dir / filename
        cv2.imwrite(str(img_path), img)
        
        # Create YOLO annotation (device detection)
        # Device bounding box (entire image area)
        img_h, img_w = img.shape[:2]
        x_center = 0.5
        y_center = 0.5
        width = 0.9
        height = 0.8
        
        yolo_annotation = f"0 {x_center} {y_center} {width} {height}\n"
        
        # Save YOLO annotation
        label_filename = f"cardio_chek_synthetic_{i+1:03d}.txt"
        label_path = labels_dir / label_filename
        with open(label_path, "w") as f:
            f.write(yolo_annotation)
        
        generated_images.append({
            "filename": filename,
            "readings": variation,
            "label": variation.get("label", "unknown")
        })
        
        print(f"✅ Generated: {filename} - {variation.get('label', 'unknown')}")
    
    # Save generation log
    with open("data/cardio_chek/generated_images_log.json", "w") as f:
        json.dump(generated_images, f, indent=2)
    
    print(f"\n📊 GENERATION SUMMARY:")
    print(f"  • Images generated: {len(generated_images)}")
    print(f"  • Labels created: {len(generated_images)}")
    print(f"  • Directory: {images_dir}")
    print(f"  • Labels: {labels_dir}")
    
    return generated_images

def update_dataset_config():
    """Update the dataset configuration for training"""
    
    # Create comprehensive dataset config
    dataset_config = {
        "path": str(Path("data/cardio_chek").absolute()),
        "train": "yolo/images",
        "val": "yolo/images",
        "test": "yolo/images",
        "nc": 1,  # Number of classes
        "names": ["cardio_chek_plus"]
    }
    
    # Save config
    config_path = "data/cardio_chek/real_dataset.yaml"
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Dataset config updated: {config_path}")
    
    return dataset_config

if __name__ == "__main__":
    print("🚀 GENERATING CARDIOCHECK PLUS TRAINING IMAGES")
    print("=" * 60)
    
    # Generate images
    images = generate_training_images()
    
    # Update dataset config
    config = update_dataset_config()
    
    print("\n🎯 READY FOR YOLO TRAINING!")
    print("=" * 30)
    print("Next steps:")
    print("1. Copy real image to data/cardio_chek/images/real_data/")
    print("2. Run YOLO training with new dataset")
    print("3. Test detection on provided image")
    
    print("\n✅ Image generation complete!")
