"""
Train YOLOv8 model for CardioChek Plus detection
"""

import os
import yaml
from ultralytics import YOLO
import torch

def train_cardio_chek_detector():
    """Train YOLOv8 model for CardioChek Plus detection"""
    
    # Use absolute path for dataset
    dataset_path = '/Users/karthickrajamurugan/Safe/CVML/data/cardio_chek/dataset.yaml'
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Use nano model for speed
    
    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='cardio_chek_models',
        name='cardio_chek_detector'
    )
    
    print("Training completed!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_cardio_chek_detector()
