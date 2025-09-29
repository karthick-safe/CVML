"""
Real CVML Model Manager for CardioChek Plus Analysis
Implements actual computer vision and OCR models
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import easyocr
import re
import logging
from typing import Dict, Any, List, Tuple
import time
from .enhanced_detection import EnhancedCardioChekDetector

logger = logging.getLogger(__name__)

class RealModelManager:
    """
    Real CVML model manager with actual computer vision and OCR capabilities
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        
        # Initialize OCR reader for CardioChek Plus screen text
        self.ocr_reader = None
        
        # YOLO model for kit detection
        self.detection_model = None
        
        # Enhanced detection system
        self.enhanced_detector = EnhancedCardioChekDetector()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CardioChek Plus specific patterns
        self.cardio_chek_patterns = {
            'cholesterol': r'CHOL\s*(\d+\.?\d*)',
            'hdl': r'HDL\s*(\d+\.?\d*)',
            'triglycerides': r'TRIG\s*(\d+\.?\d*)',
            'glucose': r'eGLU\s*(\d+\.?\d*)',
            'mg_dl': r'(\d+\.?\d*)\s*mg/dL',
            'mmol_l': r'(\d+\.?\d*)\s*mmol/L'
        }
    
    async def load_models(self):
        """Load real CVML models"""
        try:
            logger.info("Loading real CVML models...")
            
            # Initialize OCR reader for CardioChek Plus
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("OCR reader initialized")
            
            # Load YOLO model for kit detection
            # In production, this would be a trained YOLOv8 model
            # For now, we'll use OpenCV-based detection with template matching
            self._load_detection_model()
            
            self.models_loaded = True
            logger.info("Real CVML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load real models: {e}")
            raise
    
    def _load_detection_model(self):
        """Load kit detection model"""
        try:
            # Load the retrained YOLOv8 model with real data
            from ultralytics import YOLO
            model_path = "/Users/karthickrajamurugan/Safe/CVML/cardio_chek_models/cardio_chek_detector_real/weights/best.pt"
            self.detection_model = YOLO(model_path)
            logger.info("Retrained YOLOv8 model with real data loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load retrained YOLO model: {e}")
            # Try fallback to original model
            try:
                fallback_path = "/Users/karthickrajamurugan/Safe/CVML/backend/cardio_chek_models/cardio_chek_detector3/weights/best.pt"
                self.detection_model = YOLO(fallback_path)
                logger.info("Fallback YOLO model loaded successfully")
            except Exception as e2:
                logger.warning(f"Failed to load fallback YOLO model: {e2}")
                logger.info("Falling back to OpenCV-based detection")
                self.detection_model = None
    
    async def detect_cardio_chek_kit(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect CardioChek Plus kit in image using trained YOLO model
        """
        start_time = time.time()
        
        try:
            # Try YOLO detection first
            if self.detection_model is not None:
                yolo_result = await self._yolo_detection(image, start_time)
                if yolo_result["detected"]:
                    return yolo_result
            
            # Fallback to OpenCV detection
            logger.info("YOLO detection failed, using OpenCV fallback")
            return await self._opencv_detection_fallback(image, start_time)
            
        except Exception as e:
            logger.error(f"Error in kit detection: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
                "processing_time": float(time.time() - start_time),
                "error": str(e)
            }
    
    async def _yolo_detection(self, image: np.ndarray, start_time: float) -> Dict[str, Any]:
        """Use trained YOLO model for detection"""
        try:
            results = self.detection_model(image, conf=0.5)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get the best detection
                    box = result.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    bbox = {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "confidence": float(confidence)
                    }
                    
                    logger.info(f"YOLO detection: CardioChek Plus found (confidence: {confidence:.3f})")
                    return {
                        "detected": True,
                        "confidence": float(confidence),
                        "bounding_box": bbox,
                        "processing_time": float(time.time() - start_time),
                        "method": "yolo_detection"
                    }
            
            return {"detected": False, "confidence": 0.0, "bounding_box": None, "processing_time": float(time.time() - start_time)}
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return {"detected": False, "confidence": 0.0, "bounding_box": None, "processing_time": float(time.time() - start_time)}
    
    async def _opencv_detection_fallback(self, image: np.ndarray, start_time: float) -> Dict[str, Any]:
        """Enhanced OpenCV-based detection using multiple strategies"""
        try:
            # Use enhanced detection system
            detection_result = self.enhanced_detector.detect_cardio_chek_plus(image)
            
            processing_time = time.time() - start_time
            
            # Convert all numpy types to native Python types
            detection_result = self._convert_to_native_types(detection_result)
            
            return {
                "detected": bool(detection_result["detected"]),
                "confidence": float(detection_result["confidence"]),
                "bounding_box": detection_result["bounding_box"],
                "processing_time": float(processing_time),
                "method": "enhanced_opencv_detection"
            }
            
        except Exception as e:
            logger.error(f"Enhanced detection error: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "bounding_box": None,
                "processing_time": float(time.time() - start_time),
                "error": str(e)
            }
    
    def _detect_cardio_chek_device(self, image: np.ndarray, gray: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """
        Detect CardioChek Plus device using computer vision techniques
        """
        height, width = image.shape[:2]
        
        # Look for rectangular device shape (CardioChek Plus is typically rectangular)
        # Use edge detection to find device boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            # Calculate contour area and aspect ratio
            area = cv2.contourArea(contour)
            if area < (width * height * 0.05):  # Too small
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # CardioChek Plus is typically rectangular with aspect ratio between 1.5 and 3.0
            if 1.5 <= aspect_ratio <= 3.0:
                # Calculate score based on area and aspect ratio
                score = area * (1.0 / abs(aspect_ratio - 2.0))  # Prefer aspect ratio around 2.0
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is not None and best_score > (width * height * 0.1):
            # Found potential device
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # Additional validation: check for screen-like area
            roi = image[y:y+h, x:x+w]
            screen_detected = self._detect_screen_area(roi)
            
            confidence = min(0.95, best_score / (width * height * 0.2))
            if screen_detected:
                confidence = min(0.98, confidence + 0.1)
            
            return {
                "detected": True,
                "confidence": confidence,
                "bounding_box": {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                    "confidence": confidence
                }
            }
        
        return {
            "detected": False,
            "confidence": 0.0,
            "bounding_box": None
        }
    
    def _detect_screen_area(self, roi: np.ndarray) -> bool:
        """
        Detect if the ROI contains a screen-like area (LCD display)
        """
        try:
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Look for rectangular screen area
            edges = cv2.Canny(gray_roi, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > (roi.shape[0] * roi.shape[1] * 0.1):  # Screen should be significant portion
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Screen should be roughly rectangular
                    if 1.2 <= aspect_ratio <= 3.0:
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def extract_screen_values(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract CardioChek Plus screen values using multiple OCR approaches
        """
        start_time = time.time()
        
        try:
            # Try multiple preprocessing approaches
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
            all_ocr_results = self._convert_to_native_types(all_ocr_results)
            
            return {
                "success": self._count_extracted_values(best_values) > 0,
                "values": best_values,
                "raw_ocr": all_ocr_results,
                "processing_time": float(processing_time)
            }
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return {
                "success": False,
                "values": {},
                "error": str(e),
                "processing_time": time.time() - start_time
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
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for CardioChek Plus screen OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image for better OCR (CardioChek Plus screens are typically small)
        height, width = gray.shape
        if height < 200 or width < 200:
            scale_factor = max(200 / height, 200 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(blurred)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Additional sharpening for better text recognition
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(cleaned, -1, kernel_sharpen)
        
        return sharpened
    
    def _parse_cardio_chek_values(self, ocr_results: List) -> Dict[str, Any]:
        """
        Enhanced parsing of OCR results to extract CardioChek Plus values
        """
        values = {
            "cholesterol": None,
            "hdl": None,
            "triglycerides": None,
            "glucose": None,
            "units": "mg/dL",
            "raw_text": []
        }
        
        # Combine all OCR text with lower confidence threshold
        all_text = ""
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.3:  # Lower threshold for better coverage
                all_text += " " + text
                # Convert numpy types to native Python types
                values["raw_text"].append({
                    "text": str(text),
                    "confidence": float(confidence),
                    "bbox": [[int(x), int(y)] for x, y in bbox] if isinstance(bbox, list) else bbox
                })
        
        all_text = all_text.upper()
        logger.info(f"Raw OCR text: {all_text}")
        
        # Enhanced patterns for CardioChek Plus segmented display (based on provided image)
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
                r'CHOL\s*(\d+)\s*mg/dL',
                # Additional patterns for segmented fonts
                r'CHOL\s*(\d+)\s*mg/',
                r'CHOL\s*(\d+)\s*mg',
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
                r'HDL\s*CHOL\s*(\d+)\s*mg/dL',
                # Additional patterns for segmented fonts
                r'HDL\s*(\d+)\s*mg/',
                r'HDL\s*(\d+)\s*mg',
                r'HDL\s*(\d+)\s*mg/dL'
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
                r'TRIG\s*(\d+)\s*mg/dL',
                # Additional patterns for segmented fonts
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
                r'GLU\s*(\d+)\s*mg',
                # Additional patterns for segmented fonts
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
                            # Convert to native Python types to avoid serialization issues
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
                    # Convert to native Python types
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
    
    async def classify_cardio_chek_result(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify CardioChek Plus result based on extracted values
        """
        start_time = time.time()
        
        try:
            # Check if we have valid values
            if not any([values.get("cholesterol"), values.get("hdl"), values.get("triglycerides"), values.get("glucose")]):
                return {
                    "result": "Invalid",
                    "confidence": 0.9,
                    "reason": "No valid values detected",
                    "processing_time": time.time() - start_time
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
            logger.error(f"Error in result classification: {e}")
            return {
                "result": "Invalid",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": float(time.time() - start_time)
            }
    
    def _analyze_cardio_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze CardioChek Plus values based on medical standards
        """
        units = values.get("units", "mg/dL")
        cholesterol = values.get("cholesterol")
        hdl = values.get("hdl")
        triglycerides = values.get("triglycerides")
        glucose = values.get("glucose")
        
        analysis = []
        risk_factors = 0
        
        # Cholesterol analysis (mg/dL)
        if cholesterol:
            if units == "mg/dL":
                if cholesterol < 200:
                    analysis.append("Total Cholesterol: Normal")
                elif cholesterol < 240:
                    analysis.append("Total Cholesterol: Borderline High")
                    risk_factors += 1
                else:
                    analysis.append("Total Cholesterol: High")
                    risk_factors += 2
            else:  # mmol/L
                if cholesterol < 5.2:
                    analysis.append("Total Cholesterol: Normal")
                elif cholesterol < 6.2:
                    analysis.append("Total Cholesterol: Borderline High")
                    risk_factors += 1
                else:
                    analysis.append("Total Cholesterol: High")
                    risk_factors += 2
        
        # HDL analysis
        if hdl:
            if units == "mg/dL":
                if hdl >= 60:
                    analysis.append("HDL: Good")
                elif hdl >= 40:
                    analysis.append("HDL: Borderline")
                    risk_factors += 1
                else:
                    analysis.append("HDL: Low")
                    risk_factors += 2
            else:  # mmol/L
                if hdl >= 1.55:
                    analysis.append("HDL: Good")
                elif hdl >= 1.04:
                    analysis.append("HDL: Borderline")
                    risk_factors += 1
                else:
                    analysis.append("HDL: Low")
                    risk_factors += 2
        
        # Triglycerides analysis
        if triglycerides:
            if units == "mg/dL":
                if triglycerides < 150:
                    analysis.append("Triglycerides: Normal")
                elif triglycerides < 200:
                    analysis.append("Triglycerides: Borderline High")
                    risk_factors += 1
                else:
                    analysis.append("Triglycerides: High")
                    risk_factors += 2
            else:  # mmol/L
                if triglycerides < 1.7:
                    analysis.append("Triglycerides: Normal")
                elif triglycerides < 2.3:
                    analysis.append("Triglycerides: Borderline High")
                    risk_factors += 1
                else:
                    analysis.append("Triglycerides: High")
                    risk_factors += 2
        
        # Glucose analysis
        if glucose:
            if units == "mg/dL":
                if glucose < 100:
                    analysis.append("Glucose: Normal")
                elif glucose < 126:
                    analysis.append("Glucose: Prediabetes")
                    risk_factors += 1
                else:
                    analysis.append("Glucose: Diabetes")
                    risk_factors += 2
            else:  # mmol/L
                if glucose < 5.6:
                    analysis.append("Glucose: Normal")
                elif glucose < 7.0:
                    analysis.append("Glucose: Prediabetes")
                    risk_factors += 1
                else:
                    analysis.append("Glucose: Diabetes")
                    risk_factors += 2
        
        # Determine overall result
        if risk_factors == 0:
            result = "Normal"
            confidence = 0.9
        elif risk_factors <= 2:
            result = "Borderline"
            confidence = 0.8
        else:
            result = "High Risk"
            confidence = 0.85
        
        return {
            "result": result,
            "confidence": confidence,
            "analysis": analysis,
            "risk_factors": risk_factors
        }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        return {
            "models_loaded": self.models_loaded,
            "ocr_available": self.ocr_reader is not None,
            "detection_model_loaded": self.detection_model is not None,
            "device": str(self.device),
            "all_loaded": self.models_loaded
        }
