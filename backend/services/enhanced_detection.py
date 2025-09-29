"""
Enhanced CardioChek Plus Detection Module
Implements multiple detection strategies for better accuracy
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class EnhancedCardioChekDetector:
    """
    Enhanced detector with multiple strategies for CardioChek Plus detection
    """
    
    def __init__(self):
        self.detection_methods = [
            self._detect_by_edges,
            self._detect_by_color,
            self._detect_by_template,
            self._detect_by_contours
        ]
    
    def detect_cardio_chek_plus(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main detection method that tries multiple strategies
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        best_result = {"detected": False, "confidence": 0.0, "bounding_box": None}
        
        for method in self.detection_methods:
            try:
                result = method(image, gray, hsv, width, height)
                if result["detected"] and result["confidence"] > best_result["confidence"]:
                    best_result = result
                    logger.info(f"Detection method {method.__name__} found CardioChek Plus with confidence {result['confidence']:.3f}")
            except Exception as e:
                logger.warning(f"Detection method {method.__name__} failed: {e}")
                continue
        
        return best_result
    
    def _detect_by_edges(self, image: np.ndarray, gray: np.ndarray, hsv: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Detect CardioChek Plus using edge detection"""
        # Use multiple edge detection parameters
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        
        # Combine edge images
        combined_edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 3000 or area > (width * height * 0.7):
                continue
            
            # Check for rectangular shape
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if 0.3 < aspect_ratio < 4.0:  # More flexible aspect ratio
                # Calculate solidity (how filled the contour is)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Score based on area, aspect ratio, and solidity
                score = area * solidity * (1 - abs(aspect_ratio - 1.5) * 0.05)
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is not None and best_score > 5000:
            x, y, w, h = cv2.boundingRect(best_contour)
            confidence = min(0.85, best_score / 15000)
            
            return {
                "detected": True,
                "confidence": confidence,
                "bounding_box": {
                    "x": int(x), 
                    "y": int(y), 
                    "width": int(w), 
                    "height": int(h), 
                    "confidence": float(confidence)
                }
            }
        
        return {"detected": False, "confidence": 0.0, "bounding_box": None}
    
    def _detect_by_color(self, image: np.ndarray, gray: np.ndarray, hsv: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Detect CardioChek Plus using color analysis"""
        # CardioChek Plus typically has dark gray/black body with green screen
        # Define color ranges for CardioChek Plus
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])  # Dark colors
        
        lower_green = np.array([40, 50, 50])   # Green screen
        upper_green = np.array([80, 255, 255])
        
        # Create masks
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(dark_mask, green_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000 or area > (width * height * 0.6):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if 0.4 < aspect_ratio < 3.5:
                # Check if this region contains both dark and green areas
                roi = hsv[y:y+h, x:x+w]
                dark_pixels = cv2.countNonZero(cv2.inRange(roi, lower_dark, upper_dark))
                green_pixels = cv2.countNonZero(cv2.inRange(roi, lower_green, upper_green))
                
                if dark_pixels > 100 and green_pixels > 50:  # Has both dark and green
                    score = area * (dark_pixels + green_pixels) / (w * h)
                    
                    if score > best_score:
                        best_score = score
                        best_contour = contour
        
        if best_contour is not None and best_score > 1000:
            x, y, w, h = cv2.boundingRect(best_contour)
            confidence = min(0.8, best_score / 5000)
            
            return {
                "detected": True,
                "confidence": confidence,
                "bounding_box": {
                    "x": int(x), 
                    "y": int(y), 
                    "width": int(w), 
                    "height": int(h), 
                    "confidence": float(confidence)
                }
            }
        
        return {"detected": False, "confidence": 0.0, "bounding_box": None}
    
    def _detect_by_template(self, image: np.ndarray, gray: np.ndarray, hsv: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Detect CardioChek Plus using template matching for screen area"""
        # Look for rectangular screen-like areas (CardioChek Plus has a distinctive screen)
        # Use morphological operations to find rectangular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > (width * height * 0.3):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Screen should be roughly rectangular
            if 1.5 < aspect_ratio < 4.0 and h > 50 and w > 100:
                confidence = min(0.75, area / 5000)
                
                return {
                    "detected": True,
                    "confidence": confidence,
                    "bounding_box": {
                    "x": int(x), 
                    "y": int(y), 
                    "width": int(w), 
                    "height": int(h), 
                    "confidence": float(confidence)
                }
                }
        
        return {"detected": False, "confidence": 0.0, "bounding_box": None}
    
    def _detect_by_contours(self, image: np.ndarray, gray: np.ndarray, hsv: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Enhanced contour-based detection"""
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000 or area > (width * height * 0.8):
                continue
            
            # Calculate various properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Check if contour is roughly rectangular
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4 and 0.3 < aspect_ratio < 4.0:
                # Calculate additional metrics
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Calculate extent (ratio of contour area to bounding box area)
                extent = area / (w * h)
                
                # Score based on multiple factors
                score = area * solidity * extent * (1 - abs(aspect_ratio - 1.5) * 0.1)
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is not None and best_score > 3000:
            x, y, w, h = cv2.boundingRect(best_contour)
            confidence = min(0.9, best_score / 20000)
            
            return {
                "detected": True,
                "confidence": confidence,
                "bounding_box": {
                    "x": int(x), 
                    "y": int(y), 
                    "width": int(w), 
                    "height": int(h), 
                    "confidence": float(confidence)
                }
            }
        
        return {"detected": False, "confidence": 0.0, "bounding_box": None}
    
    def extract_screen_roi(self, image: np.ndarray, bbox: Dict[str, Any]) -> np.ndarray:
        """
        Extract the screen region from the detected CardioChek Plus device
        """
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        
        # Extract the device region
        device_roi = image[y:y+h, x:x+w]
        
        # Look for the screen area within the device
        # CardioChek Plus screen is typically in the upper portion
        screen_height = int(h * 0.4)  # Screen takes about 40% of device height
        screen_roi = device_roi[:screen_height, :]
        
        return screen_roi
