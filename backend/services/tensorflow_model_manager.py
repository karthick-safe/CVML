#!/usr/bin/env python3
"""
TensorFlow-based CVML Model Manager for CardioChek Plus Analysis
This provides a more robust alternative to PyTorch/YOLOv8 for detection and OCR
"""

import cv2
import numpy as np
import time
import logging
import re
import json
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from pathlib import Path

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import easyocr

logger = logging.getLogger(__name__)

class TensorFlowCardioChekDetector:
    """TensorFlow-based CardioChek Plus device detector"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.input_size = (224, 224)  # Standard input size for classification
        self.confidence_threshold = 0.4  # Lower threshold for better detection

        # Load or create model
        if model_path and Path(model_path).exists():
            self._load_model()
        else:
            self._create_model()

    def _create_model(self):
        """Create a custom CNN model for CardioChek Plus detection"""
        try:
            # Input layer
            inputs = Input(shape=(self.input_size[0], self.input_size[1], 3))

            # Convolutional layers
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2))(x)

            # Flatten and dense layers
            x = Flatten()(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)

            # Output layers
            # Bounding box regression (x, y, width, height)
            bbox_output = Dense(4, activation='linear', name='bbox')(x)
            # Classification (device present/absent)
            class_output = Dense(1, activation='sigmoid', name='class')(x)

            # Create model
            self.model = Model(inputs=inputs, outputs=[bbox_output, class_output])

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'bbox': 'mse',
                    'class': 'binary_crossentropy'
                },
                metrics={
                    'bbox': 'mse',
                    'class': 'accuracy'
                }
            )

            logger.info("Created new TensorFlow CardioChek detector model")

        except Exception as e:
            logger.error(f"Failed to create TensorFlow model: {e}")
            raise

    def _load_model(self):
        """Load pre-trained TensorFlow model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Loaded TensorFlow model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TensorFlow model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)

        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)

        return batched

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect CardioChek Plus device in image"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Make prediction
            bbox_pred, class_pred = self.model.predict(processed_image, verbose=0)

            # Extract results
            confidence = float(class_pred[0][0])
            bbox = bbox_pred[0]

            # Convert normalized bbox to pixel coordinates
            height, width = image.shape[:2]
            x = int(bbox[0] * width)
            y = int(bbox[1] * height)
            w = int(bbox[2] * width)
            h = int(bbox[3] * height)

            # Ensure bbox is within image bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))

            result = {
                "detected": confidence > self.confidence_threshold,
                "confidence": confidence,
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": confidence
                }
            }

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
                "error": str(e)
            }

    def train(self, images: List[np.ndarray], bboxes: List[Dict], epochs: int = 50):
        """Train the model with provided data"""
        try:
            # Prepare training data
            X_train = []
            y_bbox = []
            y_class = []

            for img, bbox in zip(images, bboxes):
                # Preprocess image
                processed = self.preprocess_image(img)
                X_train.append(processed[0])  # Remove batch dimension

                # Prepare bbox targets (normalized coordinates)
                height, width = img.shape[:2]
                norm_bbox = [
                    bbox['x'] / width,
                    bbox['y'] / height,
                    bbox['width'] / width,
                    bbox['height'] / height
                ]
                y_bbox.append(norm_bbox)
                y_class.append(1.0)  # Device present

            X_train = np.array(X_train)
            y_bbox = np.array(y_bbox)
            y_class = np.array(y_class)

            # Train model
            callbacks = [
                ModelCheckpoint('cardio_chek_tf_detector.keras', save_best_only=True),
                EarlyStopping(patience=10, restore_best_weights=True)
            ]

            history = self.model.fit(
                X_train,
                {'bbox': y_bbox, 'class': y_class},
                epochs=epochs,
                batch_size=8,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Model training completed successfully")
            return history

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

class TensorFlowModelManager:
    """TensorFlow-based model manager for CardioChek Plus analysis"""

    def __init__(self):
        self.detector = None
        self.ocr_reader = None
        self.initialized = False

    async def initialize(self):
        """Initialize TensorFlow models"""
        try:
            # Initialize CardioChek detector
            # Try new .keras format first, then fall back to .h5
            keras_path = "cardio_chek_tf_detector.keras"
            h5_path = "cardio_chek_tf_detector.h5"
            
            if Path(keras_path).exists():
                self.detector = TensorFlowCardioChekDetector(keras_path)
            elif Path(h5_path).exists():
                # Convert old h5 to new keras format
                logger.info("Converting .h5 model to .keras format...")
                try:
                    old_model = tf.keras.models.load_model(h5_path, compile=False)
                    old_model.save(keras_path)
                    self.detector = TensorFlowCardioChekDetector(keras_path)
                    logger.info("Model converted successfully")
                except Exception as e:
                    logger.warning(f"Conversion failed: {e}, creating new model")
                    self.detector = TensorFlowCardioChekDetector()
            else:
                self.detector = TensorFlowCardioChekDetector()
                logger.info("Created new TensorFlow detector (needs training)")

            # Initialize OCR reader
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility

            self.initialized = True
            logger.info("TensorFlow model manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TensorFlow models: {e}")
            raise
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            "detector_loaded": self.detector is not None,
            "ocr_loaded": self.ocr_reader is not None,
            "all_loaded": self.initialized,
            "detector_type": "tensorflow_cnn",
            "ocr_type": "easyocr"
        }

    async def detect_cardio_chek_kit(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect CardioChek Plus kit using TensorFlow model"""
        start_time = time.time()

        try:
            if not self.initialized:
                await self.initialize()

            # Use TensorFlow detector
            detection_result = self.detector.detect(image)

            processing_time = time.time() - start_time

            # Convert numpy types to native Python types
            # Ensure bounding box has confidence field
            bbox = detection_result["bounding_box"]
            if bbox and "confidence" not in bbox:
                bbox["confidence"] = float(detection_result["confidence"])
            
            result = {
                "detected": bool(detection_result["detected"]),
                "confidence": float(detection_result["confidence"]),
                "bounding_box": bbox,
                "processing_time": float(processing_time),
                "method": "tensorflow_detection"
            }

            if detection_result["detected"]:
                logger.info(f"TensorFlow detection: CardioChek Plus found (confidence: {detection_result['confidence']:.3f})")
            else:
                logger.info("TensorFlow detection: No CardioChek Plus device detected")

            return result

        except Exception as e:
            logger.error(f"TensorFlow detection error: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
                "processing_time": float(time.time() - start_time),
                "error": str(e)
            }

    async def extract_screen_values(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract CardioChek Plus screen values using OCR"""
        start_time = time.time()

        try:
            if not self.initialized:
                await self.initialize()

            # Preprocess image for OCR
            processed_images = self._get_multiple_preprocessed_images(image)

            all_ocr_results = []
            best_values = {}

            # Try OCR on each preprocessed image
            for i, processed_img in enumerate(processed_images):
                try:
                    ocr_results = self.ocr_reader.readtext(processed_img)
                    all_ocr_results.extend(ocr_results)

                    # Extract values from this attempt
                    values = self._parse_cardio_chek_values(ocr_results)

                    # Keep the best result (most values extracted)
                    if self._count_extracted_values(values) > self._count_extracted_values(best_values):
                        best_values = values

                except Exception as e:
                    logger.warning(f"OCR attempt {i} failed: {e}")
                    continue

            # If we have multiple OCR results, try combining them
            if len(all_ocr_results) > 0:
                combined_values = self._parse_cardio_chek_values(all_ocr_results)
                if self._count_extracted_values(combined_values) > self._count_extracted_values(best_values):
                    best_values = combined_values

            processing_time = time.time() - start_time

            # Convert all numpy types to native Python types
            best_values = self._convert_to_native_types(best_values)

            return {
                "success": self._count_extracted_values(best_values) > 0,
                "values": best_values,
                "raw_ocr": all_ocr_results,
                "processing_time": float(processing_time)
            }

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {
                "success": False,
                "values": {},
                "error": str(e),
                "processing_time": float(time.time() - start_time)
            }

    def _get_multiple_preprocessed_images(self, image: np.ndarray) -> List[np.ndarray]:
        """Get multiple preprocessed versions of the image for better OCR"""
        processed_images = []

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize for better OCR
        height, width = gray.shape
        if height < 200 or width < 200:
            scale_factor = max(200 / height, 200 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Method 1: CLAHE + Adaptive threshold
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        thresh1 = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(thresh1)

        # Method 2: Otsu thresholding
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh2)

        # Method 3: Denoising + Otsu
        denoised = cv2.fastNlMeansDenoising(gray)
        _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(thresh3)

        # Method 4: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)

        # Method 5: Inverted image (sometimes text is white on dark background)
        inverted = cv2.bitwise_not(thresh1)
        processed_images.append(inverted)

        return processed_images

    def _parse_cardio_chek_values(self, ocr_results: List) -> Dict[str, Any]:
        """Enhanced parsing of OCR results for CardioChek Plus values"""
        values = {
            "cholesterol": None,
            "hdl": None,
            "triglycerides": None,
            "glucose": None,
            "units": "mg/dL",
            "raw_text": []
        }

        # Combine all OCR text
        all_text = ""
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.3:  # Lower threshold for better coverage
                all_text += " " + text
                values["raw_text"].append({
                    "text": str(text),
                    "confidence": float(confidence),
                    "bbox": [[int(x), int(y)] for x, y in bbox] if isinstance(bbox, list) else bbox
                })

        all_text = all_text.upper()
        logger.info(f"Raw OCR text: {all_text}")

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

        # Extract values using enhanced patterns
        for key, pattern_list in enhanced_patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, all_text)
                if match:
                    try:
                        value = float(match.group(1))
                        if 0 < value < 1000:  # Reasonable range
                            values[key] = int(value) if value.is_integer() else float(value)
                            logger.info(f"Extracted {key}: {values[key]}")
                            break  # Use first valid match
                    except (ValueError, IndexError):
                        continue

        # If no specific values found, try to extract any numbers in order
        if not any(values[key] for key in ['cholesterol', 'hdl', 'triglycerides', 'glucose']):
            numbers = re.findall(r'\d+\.?\d*', all_text)
            if len(numbers) >= 4:
                try:
                    values['cholesterol'] = int(float(numbers[0])) if float(numbers[0]).is_integer() else float(numbers[0])
                    values['hdl'] = int(float(numbers[1])) if float(numbers[1]).is_integer() else float(numbers[1])
                    values['triglycerides'] = int(float(numbers[2])) if float(numbers[2]).is_integer() else float(numbers[2])
                    values['glucose'] = int(float(numbers[3])) if float(numbers[3]).is_integer() else float(numbers[3])
                    logger.info("Assigned numbers based on typical CardioChek Plus order")
                except (ValueError, IndexError):
                    pass

        # Determine units
        if "mmol/L" in all_text or "mmol/l" in all_text:
            values["units"] = "mmol/L"

        return values

    def _count_extracted_values(self, values: Dict[str, Any]) -> int:
        """Count how many values were successfully extracted"""
        count = 0
        for key in ['cholesterol', 'hdl', 'triglycerides', 'glucose']:
            if values.get(key) is not None:
                count += 1
        return count

    def _convert_to_native_types(self, obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj

    async def classify_cardio_chek_result(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Classify CardioChek Plus result based on extracted values"""
        start_time = time.time()

        try:
            # Check if we have valid values
            if not any([values.get("cholesterol"), values.get("hdl"), values.get("triglycerides"), values.get("glucose")]):
                return {
                    "result": "Invalid",
                    "confidence": 0.9,
                    "reason": "No valid values detected",
                    "processing_time": float(time.time() - start_time)
                }

            # Analyze values based on medical standards
            result_analysis = self._analyze_cardio_values(values)

            # Convert all numpy types to native Python types
            values = self._convert_to_native_types(values)
            result_analysis = self._convert_to_native_types(result_analysis)

            return {
                "result": str(result_analysis["result"]),
                "confidence": float(result_analysis["confidence"]),
                "values": values,
                "analysis": result_analysis["analysis"],
                "processing_time": float(time.time() - start_time)
            }

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "result": "Invalid",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": float(time.time() - start_time)
            }

    def _analyze_cardio_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cardiovascular health values"""
        analysis = []
        risk_factors = 0
        total_score = 0

        # Cholesterol analysis
        chol = values.get("cholesterol")
        if chol is not None:
            if chol < 200:
                analysis.append("Total Cholesterol: Normal")
                total_score += 1
            elif chol < 240:
                analysis.append("Total Cholesterol: Borderline High")
                total_score += 0
            else:
                analysis.append("Total Cholesterol: High")
                risk_factors += 1

        # HDL analysis
        hdl = values.get("hdl")
        if hdl is not None:
            if hdl >= 60:
                analysis.append("HDL: Good")
                total_score += 2
            elif hdl >= 40:
                analysis.append("HDL: Acceptable")
                total_score += 1
            else:
                analysis.append("HDL: Low")
                risk_factors += 1

        # Triglycerides analysis
        trig = values.get("triglycerides")
        if trig is not None:
            if trig < 150:
                analysis.append("Triglycerides: Normal")
                total_score += 1
            elif trig < 200:
                analysis.append("Triglycerides: Borderline High")
                total_score += 0
            else:
                analysis.append("Triglycerides: High")
                risk_factors += 1

        # Glucose analysis
        glu = values.get("glucose")
        if glu is not None:
            if glu < 100:
                analysis.append("Glucose: Normal")
                total_score += 1
            elif glu < 126:
                analysis.append("Glucose: Prediabetic")
                total_score += 0
            else:
                analysis.append("Glucose: High")
                risk_factors += 1

        # Overall assessment
        if risk_factors >= 2:
            result = "High Risk"
            confidence = 0.8
        elif risk_factors == 1:
            result = "Moderate Risk"
            confidence = 0.7
        elif total_score >= 4:
            result = "Good Health"
            confidence = 0.9
        else:
            result = "Borderline"
            confidence = 0.6

        return {
            "result": result,
            "confidence": confidence,
            "analysis": analysis,
            "risk_factors": risk_factors,
            "health_score": total_score
        }

    async def train_detector(self, images: List[np.ndarray], annotations: List[Dict]):
        """Train the TensorFlow detector with provided data"""
        try:
            # Extract bounding boxes and labels
            bboxes = []
            for annotation in annotations:
                if annotation.get("class") == "cardio_chek_plus":
                    bboxes.append(annotation["bbox"])

            # Train the model
            self.detector.train(images, bboxes, epochs=30)

            logger.info("TensorFlow detector training completed")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
