"""
Model optimization script for CVML Cardio Health Check Kit Analyzer
Converts trained models to optimized formats for deployment
"""

import os
import logging
import tensorflow as tf
import torch
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimizes models for different deployment scenarios"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.optimized_dir = self.models_dir / "optimized"
        self.optimized_dir.mkdir(exist_ok=True)
    
    def optimize_classification_model(self, model_path: str, output_formats: list = ["tflite", "onnx"]):
        """
        Optimize classification model for deployment
        
        Args:
            model_path: Path to trained Keras model
            output_formats: List of output formats to generate
        """
        try:
            # Load the model
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded classification model from {model_path}")
            
            # Convert to TensorFlow Lite
            if "tflite" in output_formats:
                self._convert_to_tflite(model, "classification")
            
            # Convert to ONNX
            if "onnx" in output_formats:
                self._convert_to_onnx(model, "classification")
            
            # Quantize model for mobile deployment
            if "quantized" in output_formats:
                self._quantize_model(model, "classification")
            
            logger.info("Classification model optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing classification model: {e}")
            raise
    
    def optimize_detection_model(self, model_path: str, output_formats: list = ["tflite", "onnx"]):
        """
        Optimize object detection model for deployment
        
        Args:
            model_path: Path to trained YOLO model
            output_formats: List of output formats to generate
        """
        try:
            from ultralytics import YOLO
            
            # Load YOLO model
            model = YOLO(model_path)
            logger.info(f"Loaded detection model from {model_path}")
            
            # Export to different formats
            for format_type in output_formats:
                if format_type == "tflite":
                    self._export_yolo_to_tflite(model, "detection")
                elif format_type == "onnx":
                    self._export_yolo_to_onnx(model, "detection")
                elif format_type == "coreml":
                    self._export_yolo_to_coreml(model, "detection")
            
            logger.info("Detection model optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing detection model: {e}")
            raise
    
    def _convert_to_tflite(self, model: tf.keras.Model, model_name: str):
        """Convert Keras model to TensorFlow Lite"""
        try:
            # Standard conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save standard TFLite model
            tflite_path = self.optimized_dir / f"{model_name}_standard.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"Standard TFLite model saved to {tflite_path}")
            
            # Optimized conversion with quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_optimized = converter.convert()
            optimized_path = self.optimized_dir / f"{model_name}_optimized.tflite"
            with open(optimized_path, 'wb') as f:
                f.write(tflite_optimized)
            logger.info(f"Optimized TFLite model saved to {optimized_path}")
            
        except Exception as e:
            logger.error(f"Error converting to TFLite: {e}")
            raise
    
    def _convert_to_onnx(self, model: tf.keras.Model, model_name: str):
        """Convert Keras model to ONNX"""
        try:
            import tf2onnx
            
            # Convert to ONNX
            spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            
            # Save ONNX model
            onnx_path = self.optimized_dir / f"{model_name}.onnx"
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"ONNX model saved to {onnx_path}")
            
        except Exception as e:
            logger.error(f"Error converting to ONNX: {e}")
            raise
    
    def _quantize_model(self, model: tf.keras.Model, model_name: str):
        """Quantize model for mobile deployment"""
        try:
            # Post-training quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Representative dataset for quantization
            def representative_data_gen():
                for _ in range(100):
                    data = np.random.random((1, 224, 224, 3)).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            quantized_model = converter.convert()
            quantized_path = self.optimized_dir / f"{model_name}_quantized.tflite"
            with open(quantized_path, 'wb') as f:
                f.write(quantized_model)
            logger.info(f"Quantized model saved to {quantized_path}")
            
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            raise
    
    def _export_yolo_to_tflite(self, model, model_name: str):
        """Export YOLO model to TensorFlow Lite"""
        try:
            # Export to TFLite
            tflite_path = self.optimized_dir / f"{model_name}.tflite"
            model.export(format='tflite', imgsz=640, optimize=True)
            logger.info(f"YOLO TFLite model exported to {tflite_path}")
            
        except Exception as e:
            logger.error(f"Error exporting YOLO to TFLite: {e}")
            raise
    
    def _export_yolo_to_onnx(self, model, model_name: str):
        """Export YOLO model to ONNX"""
        try:
            # Export to ONNX
            onnx_path = self.optimized_dir / f"{model_name}.onnx"
            model.export(format='onnx', imgsz=640, optimize=True)
            logger.info(f"YOLO ONNX model exported to {onnx_path}")
            
        except Exception as e:
            logger.error(f"Error exporting YOLO to ONNX: {e}")
            raise
    
    def _export_yolo_to_coreml(self, model, model_name: str):
        """Export YOLO model to CoreML"""
        try:
            # Export to CoreML
            coreml_path = self.optimized_dir / f"{model_name}.mlmodel"
            model.export(format='coreml', imgsz=640)
            logger.info(f"YOLO CoreML model exported to {coreml_path}")
            
        except Exception as e:
            logger.error(f"Error exporting YOLO to CoreML: {e}")
            raise
    
    def benchmark_models(self):
        """Benchmark optimized models for performance"""
        try:
            import time
            
            # Benchmark TFLite models
            tflite_models = list(self.optimized_dir.glob("*.tflite"))
            
            for model_path in tflite_models:
                logger.info(f"Benchmarking {model_path.name}...")
                
                # Load TFLite model
                interpreter = tf.lite.Interpreter(model_path=str(model_path))
                interpreter.allocate_tensors()
                
                # Get input details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Prepare test input
                input_shape = input_details[0]['shape']
                test_input = np.random.random(input_shape).astype(np.float32)
                
                # Benchmark inference
                times = []
                for _ in range(100):
                    start_time = time.time()
                    interpreter.set_tensor(input_details[0]['index'], test_input)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details[0]['index'])
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                logger.info(f"{model_path.name}: {avg_time:.4f}s ± {std_time:.4f}s")
                
        except Exception as e:
            logger.error(f"Error benchmarking models: {e}")
            raise
    
    def create_model_manifest(self):
        """Create manifest file for optimized models"""
        try:
            manifest = {
                "models": {},
                "optimization_date": str(Path().cwd()),
                "formats": ["tflite", "onnx", "coreml"]
            }
            
            # Scan for optimized models
            for model_file in self.optimized_dir.glob("*"):
                if model_file.is_file():
                    model_info = {
                        "path": str(model_file),
                        "size_mb": model_file.stat().st_size / (1024 * 1024),
                        "format": model_file.suffix[1:],
                        "optimized": True
                    }
                    manifest["models"][model_file.stem] = model_info
            
            # Save manifest
            import json
            manifest_path = self.optimized_dir / "model_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Model manifest saved to {manifest_path}")
            return manifest
            
        except Exception as e:
            logger.error(f"Error creating model manifest: {e}")
            raise

def main():
    """Main optimization pipeline"""
    optimizer = ModelOptimizer()
    
    try:
        # Optimize classification model
        classification_model = "models/classification/best_model.h5"
        if os.path.exists(classification_model):
            logger.info("Optimizing classification model...")
            optimizer.optimize_classification_model(
                classification_model, 
                ["tflite", "onnx", "quantized"]
            )
        
        # Optimize detection model
        detection_model = "models/kit_detection/weights/best.pt"
        if os.path.exists(detection_model):
            logger.info("Optimizing detection model...")
            optimizer.optimize_detection_model(
                detection_model,
                ["tflite", "onnx"]
            )
        
        # Benchmark models
        logger.info("Benchmarking optimized models...")
        optimizer.benchmark_models()
        
        # Create manifest
        manifest = optimizer.create_model_manifest()
        logger.info("Model optimization completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL OPTIMIZATION SUMMARY")
        print("="*50)
        for model_name, info in manifest["models"].items():
            print(f"{model_name}: {info['size_mb']:.2f} MB ({info['format']})")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()
