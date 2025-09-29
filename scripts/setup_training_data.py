"""
Setup training data for CardioChek Plus CVML system
"""

import os
import shutil
from pathlib import Path

def setup_training_data():
    """Setup training data directory structure"""
    
    # Base directories
    data_dir = Path("data/cardio_chek")
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    yolo_dir = data_dir / "yolo"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    print("CardioChek Plus Training Data Setup")
    print("=" * 50)
    print(f"Images directory: {images_dir.absolute()}")
    print(f"Annotations directory: {annotations_dir.absolute()}")
    print(f"YOLO directory: {yolo_dir.absolute()}")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Copy your CardioChek Plus images to the images directory")
    print("2. Run the annotation tool to label the images")
    print("3. Convert to YOLO format for training")
    print()
    print("Commands:")
    print(f"# Copy images to: {images_dir.absolute()}")
    print("# Then run annotation tool:")
    print("cd backend && source venv/bin/activate")
    print("python models/annotate_existing_images.py --images-dir ../data/cardio_chek/images")
    print()
    print("# Convert to YOLO format:")
    print("python models/annotate_existing_images.py --mode convert --images-dir ../data/cardio_chek/images")
    print()
    print("# Train YOLO model:")
    print("python models/train_cardio_chek_yolo.py")

if __name__ == "__main__":
    setup_training_data()
