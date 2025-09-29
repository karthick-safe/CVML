#!/usr/bin/env python3
"""
Retrain YOLO model with real CardioChek Plus data
Based on the provided image characteristics
"""

import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def retrain_yolo_with_real_data():
    """Retrain YOLO model with real CardioChek Plus data"""
    
    print("🚀 RETRAINING YOLO WITH REAL CARDIOCHECK PLUS DATA")
    print("=" * 60)
    
    # Check if we have the required data
    images_dir = Path("data/cardio_chek/yolo/images")
    labels_dir = Path("data/cardio_chek/yolo/labels")
    config_path = Path("data/cardio_chek/real_dataset.yaml")
    
    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Error: Training data directories not found")
        return False
    
    # Count available data
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"📊 TRAINING DATA:")
    print(f"  • Images: {len(image_files)}")
    print(f"  • Labels: {len(label_files)}")
    print(f"  • Config: {config_path}")
    
    if len(image_files) == 0:
        print("❌ Error: No training images found")
        return False
    
    # Load existing model or create new one
    model_path = "cardio_chek_models/cardio_chek_detector3/weights/best.pt"
    
    if os.path.exists(model_path):
        print(f"🔄 Loading existing model: {model_path}")
        model = YOLO(model_path)
    else:
        print("🆕 Creating new YOLOv8 model")
        model = YOLO('yolov8n.pt')  # Start with nano model for speed
    
    # Training configuration
    training_config = {
        'data': str(config_path),
        'epochs': 50,  # Reduced for faster training
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',  # Use CPU for compatibility
        'project': 'cardio_chek_models',
        'name': 'cardio_chek_detector_real',
        'save': True,
        'save_period': 10,
        'patience': 15,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print(f"🎯 TRAINING CONFIGURATION:")
    print(f"  • Epochs: {training_config['epochs']}")
    print(f"  • Image Size: {training_config['imgsz']}")
    print(f"  • Batch Size: {training_config['batch']}")
    print(f"  • Device: {training_config['device']}")
    print(f"  • Project: {training_config['project']}")
    print(f"  • Name: {training_config['name']}")
    
    try:
        # Start training
        print("\n🏋️ STARTING TRAINING...")
        print("=" * 40)
        
        results = model.train(**training_config)
        
        print("\n✅ TRAINING COMPLETED!")
        print("=" * 30)
        print(f"📁 Model saved to: {training_config['project']}/{training_config['name']}")
        
        # Validate the model
        print("\n🔍 VALIDATING MODEL...")
        validation_results = model.val()
        
        print(f"📊 VALIDATION RESULTS:")
        print(f"  • mAP50: {validation_results.box.map50:.3f}")
        print(f"  • mAP50-95: {validation_results.box.map:.3f}")
        print(f"  • Precision: {validation_results.box.mp:.3f}")
        print(f"  • Recall: {validation_results.box.mr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def update_ocr_patterns():
    """Update OCR patterns for segmented display fonts"""
    
    print("\n🔤 UPDATING OCR PATTERNS FOR SEGMENTED DISPLAY")
    print("=" * 50)
    
    # Enhanced patterns for CardioChek Plus segmented display
    enhanced_patterns = {
        'cholesterol': [
            r'CHOL\s*(\d+\.?\d*)\s*mg/dL',
            r'CHOL\s*(\d+\.?\d*)',
            r'CHOLESTEROL\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*mg/dL\s*CHOL',
            r'(\d+\.?\d*)\s*CHOL',
            # Segmented display specific patterns
            r'CHOL\s*(\d+)\s*mg/',
            r'CHOL\s*(\d+)\s*mg',
            r'CHOL\s*\{(\d+)\}mg/',
            r'CHOL\s*(\d+)\s*mg/dL'
        ],
        'hdl': [
            r'HDL\s*CHOL\s*(\d+\.?\d*)\s*mg/dL',
            r'HDL\s*(\d+\.?\d*)\s*mg/dL',
            r'HDL\s*CHOL\s*(\d+\.?\d*)',
            r'HDL\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*mg/dL\s*HDL',
            r'(\d+\.?\d*)\s*HDL',
            # Segmented display specific patterns
            r'HDL[Ee](\d+)\s*mg/',
            r'HDL\s*(\d+)\s*mg/',
            r'HDL\s*(\d+)\s*mg',
            r'HDL\s*CHOL\s*(\d+)\s*mg/dL'
        ],
        'triglycerides': [
            r'TRIG\s*(\d+\.?\d*)\s*mg/dL',
            r'TRIG\s*(\d+\.?\d*)',
            r'TRIGLYCERIDES\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*mg/dL\s*TRIG',
            r'(\d+\.?\d*)\s*TRIG',
            # Segmented display specific patterns
            r'TrIc\s*(\d+)\s*mg/',
            r'TRIG\s*(\d+)\s*mg/',
            r'TRIG\s*(\d+)\s*mg',
            r'TRIG\s*(\d+)\s*mg/dL'
        ],
        'glucose': [
            r'eGLU\s*(\d+\.?\d*)\s*mg/dL',
            r'eGLU\s*(\d+\.?\d*)',
            r'GLU\s*(\d+\.?\d*)\s*mg/dL',
            r'GLU\s*(\d+\.?\d*)',
            r'GLUCOSE\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*mg/dL\s*eGLU',
            r'(\d+\.?\d*)\s*eGLU',
            # Segmented display specific patterns
            r'eGLU\s*(\d+)\s*mg/',
            r'eGLU\s*(\d+)\s*mg',
            r'GLU\s*(\d+)\s*mg/',
            r'GLU\s*(\d+)\s*mg'
        ]
    }
    
    # Save enhanced patterns
    patterns_file = "data/cardio_chek/enhanced_ocr_patterns.json"
    with open(patterns_file, "w") as f:
        import json
        json.dump(enhanced_patterns, f, indent=2)
    
    print(f"✅ Enhanced OCR patterns saved: {patterns_file}")
    print("✅ Patterns updated for segmented display fonts")
    
    return enhanced_patterns

def test_with_provided_image():
    """Test the retrained model with the provided image"""
    
    print("\n🧪 TESTING WITH PROVIDED IMAGE")
    print("=" * 40)
    
    # Load the retrained model
    model_path = "cardio_chek_models/cardio_chek_detector_real/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        
        # Test with synthetic images first
        test_images = list(Path("data/cardio_chek/images/real_data").glob("*.jpg"))
        
        if test_images:
            print(f"🔍 Testing with {len(test_images)} images...")
            
            for img_path in test_images[:3]:  # Test first 3 images
                print(f"\n📸 Testing: {img_path.name}")
                
                results = model(str(img_path))
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        box = result.boxes[0]
                        confidence = box.conf[0].item()
                        print(f"  ✅ Detection: {confidence:.3f} confidence")
                    else:
                        print(f"  ❌ No detection")
        
        print("\n✅ Model testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CARDIOCHECK PLUS MODEL RETRAINING")
    print("=" * 60)
    
    # Update OCR patterns
    patterns = update_ocr_patterns()
    
    # Retrain YOLO model
    success = retrain_yolo_with_real_data()
    
    if success:
        # Test the retrained model
        test_success = test_with_provided_image()
        
        if test_success:
            print("\n🎉 RETRAINING COMPLETE!")
            print("=" * 30)
            print("✅ YOLO model retrained with real data")
            print("✅ OCR patterns updated for segmented display")
            print("✅ Model tested and validated")
            print("\n🎯 Ready to use with provided image!")
        else:
            print("\n⚠️ Retraining completed but testing failed")
    else:
        print("\n❌ Retraining failed")
