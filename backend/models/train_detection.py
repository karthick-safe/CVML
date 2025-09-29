"""
Training script for YOLOv8 object detection model
Trains model to detect cardio health check kits in images
"""

import os
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KitDetectionTrainer:
    """Trainer for kit detection using YOLOv8"""
    
    def __init__(self, data_path: str = "data", model_size: str = "n"):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data
            model_size: YOLOv8 model size (n, s, m, l, x)
        """
        self.data_path = Path(data_path)
        self.model_size = model_size
        self.model = None
        
    def prepare_dataset(self):
        """Prepare dataset structure for YOLOv8 training"""
        try:
            # Create dataset structure
            dataset_path = self.data_path / "kit_detection"
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            for split in ['train', 'val', 'test']:
                (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
                (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Create dataset configuration file
            config = {
                'path': str(dataset_path.absolute()),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'test/images',
                'nc': 1,  # Number of classes
                'names': ['cardio_kit']  # Class names
            }
            
            config_path = dataset_path / 'dataset.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            logger.info(f"Dataset structure created at {dataset_path}")
            return str(config_path)
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    def train_model(self, config_path: str, epochs: int = 100, batch_size: int = 16):
        """
        Train YOLOv8 model for kit detection
        
        Args:
            config_path: Path to dataset configuration
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        try:
            # Load YOLOv8 model
            model_name = f"yolov8{self.model_size}.pt"
            self.model = YOLO(model_name)
            
            logger.info(f"Starting training with {model_name}")
            
            # Train the model
            results = self.model.train(
                data=config_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                workers=4,
                project='models',
                name='kit_detection',
                exist_ok=True,
                save=True,
                save_period=10,
                cache=True,
                augment=True,
                mixup=0.15,
                copy_paste=0.3,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                shear=2.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                val=True
            )
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def validate_model(self, model_path: str, test_data_path: str):
        """
        Validate trained model
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data
        """
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(
                data=test_data_path,
                imgsz=640,
                batch=1,
                conf=0.001,
                iou=0.6,
                max_det=300,
                save_json=True,
                save_hybrid=False,
                plots=True
            )
            
            logger.info(f"Validation mAP50: {results.box.map50:.3f}")
            logger.info(f"Validation mAP50-95: {results.box.map:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def export_model(self, model_path: str, format: str = "tflite"):
        """
        Export model to different formats for deployment
        
        Args:
            model_path: Path to trained model
            format: Export format (tflite, onnx, coreml, etc.)
        """
        try:
            model = YOLO(model_path)
            
            # Export to specified format
            exported_path = model.export(
                format=format,
                imgsz=640,
                optimize=True,
                half=False,
                int8=False,
                dynamic=False,
                simplify=True,
                opset=None,
                workspace=4,
                nms=False
            )
            
            logger.info(f"Model exported to {exported_path}")
            return exported_path
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise

def main():
    """Main training pipeline"""
    trainer = KitDetectionTrainer()
    
    try:
        # Prepare dataset
        config_path = trainer.prepare_dataset()
        logger.info("Dataset preparation completed")
        
        # Train model
        results = trainer.train_model(config_path, epochs=50, batch_size=8)
        logger.info("Model training completed")
        
        # Validate model
        model_path = "models/kit_detection/weights/best.pt"
        if os.path.exists(model_path):
            validation_results = trainer.validate_model(model_path, config_path)
            logger.info("Model validation completed")
        
        # Export for mobile deployment
        if os.path.exists(model_path):
            tflite_path = trainer.export_model(model_path, "tflite")
            logger.info(f"Mobile-optimized model saved to {tflite_path}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
