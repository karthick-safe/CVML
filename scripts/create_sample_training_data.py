"""
Create sample training data for CardioChek Plus
This creates synthetic images for demonstration purposes
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_sample_cardio_chek_image():
    """Create a sample CardioChek Plus image for training"""
    
    # Create a white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw CardioChek Plus device (silver-gray rectangle)
    device_color = (200, 200, 200)
    device_rect = (150, 100, 500, 400)
    cv2.rectangle(img, device_rect, device_color, -1)
    cv2.rectangle(img, device_rect, (100, 100, 100), 3)
    
    # Draw screen (dark rectangle)
    screen_rect = (180, 150, 440, 280)
    cv2.rectangle(img, screen_rect, (50, 50, 50), -1)
    
    # Add text on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0)  # Green text
    
    # Brand name
    cv2.putText(img, "CardioChek Plus", (200, 180), font, 0.8, text_color, 2)
    
    # Values
    cv2.putText(img, "CHOL 170 mg/dL", (200, 220), font, 0.6, text_color, 2)
    cv2.putText(img, "HDL 45 mg/dL", (200, 250), font, 0.6, text_color, 2)
    cv2.putText(img, "TRIG 120 mg/dL", (200, 280), font, 0.6, text_color, 2)
    cv2.putText(img, "eGLU 90 mg/dL", (200, 310), font, 0.6, text_color, 2)
    
    # Add buttons
    cv2.circle(img, (250, 350), 15, (0, 0, 0), -1)
    cv2.circle(img, (400, 350), 20, (0, 0, 255), -1)  # Red heart button
    cv2.circle(img, (550, 350), 15, (0, 0, 0), -1)
    
    # Add test strip slot
    cv2.rectangle(img, (200, 450), (600, 480), (0, 0, 0), -1)
    
    return img

def create_training_samples():
    """Create multiple sample images with different values"""
    
    # Create images directory
    images_dir = Path("data/cardio_chek/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Different CardioChek Plus configurations
    configurations = [
        {"cholesterol": 170, "hdl": 45, "triglycerides": 120, "glucose": 90},
        {"cholesterol": 120, "hdl": 106, "triglycerides": 157, "glucose": 52},
        {"cholesterol": 215, "hdl": 67, "triglycerides": 113, "glucose": 100},
        {"cholesterol": 174, "hdl": 34, "triglycerides": 114, "glucose": 116},
        {"cholesterol": 209, "hdl": 71, "triglycerides": 117, "glucose": 72},
    ]
    
    for i, config in enumerate(configurations):
        img = create_cardio_chek_with_values(config)
        filename = f"cardio_chek_sample_{i+1:02d}.jpg"
        filepath = images_dir / filename
        cv2.imwrite(str(filepath), img)
        print(f"Created: {filename}")
    
    print(f"\nCreated {len(configurations)} sample images in {images_dir.absolute()}")
    print("You can now run the annotation tool to label these images.")

def create_cardio_chek_with_values(config):
    """Create CardioChek Plus image with specific values"""
    
    # Create a white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw CardioChek Plus device (silver-gray rectangle)
    device_color = (200, 200, 200)
    device_rect = (150, 100, 500, 400)
    cv2.rectangle(img, device_rect, device_color, -1)
    cv2.rectangle(img, device_rect, (100, 100, 100), 3)
    
    # Draw screen (dark rectangle)
    screen_rect = (180, 150, 440, 280)
    cv2.rectangle(img, screen_rect, (50, 50, 50), -1)
    
    # Add text on screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0)  # Green text
    
    # Brand name
    cv2.putText(img, "CardioChek Plus", (200, 180), font, 0.8, text_color, 2)
    
    # Values from config
    cv2.putText(img, f"CHOL {config['cholesterol']} mg/dL", (200, 220), font, 0.6, text_color, 2)
    cv2.putText(img, f"HDL {config['hdl']} mg/dL", (200, 250), font, 0.6, text_color, 2)
    cv2.putText(img, f"TRIG {config['triglycerides']} mg/dL", (200, 280), font, 0.6, text_color, 2)
    cv2.putText(img, f"eGLU {config['glucose']} mg/dL", (200, 310), font, 0.6, text_color, 2)
    
    # Add buttons
    cv2.circle(img, (250, 350), 15, (0, 0, 0), -1)
    cv2.circle(img, (400, 350), 20, (0, 0, 255), -1)  # Red heart button
    cv2.circle(img, (550, 350), 15, (0, 0, 0), -1)
    
    # Add test strip slot
    cv2.rectangle(img, (200, 450), (600, 480), (0, 0, 0), -1)
    
    return img

if __name__ == "__main__":
    create_training_samples()
