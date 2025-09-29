"""
Image Processing Service for CVML Cardio Health Check Kit Analyzer
Handles image preprocessing, cropping, alignment, and ROI extraction
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import math

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing operations for kit analysis"""
    
    def __init__(self):
        self.target_kit_size = (400, 200)  # Expected kit dimensions
        self.roi_ratio = 0.3  # ROI should be 30% of kit height (bottom portion)
    
    def extract_kit_region(self, image: np.ndarray, bounding_box: Dict[str, float]) -> np.ndarray:
        """
        Extract the kit region from the full image using bounding box coordinates
        
        Args:
            image: Full input image
            bounding_box: Bounding box coordinates {x, y, width, height}
            
        Returns:
            Cropped kit region image
        """
        try:
            x, y, w, h = int(bounding_box['x']), int(bounding_box['y']), int(bounding_box['width']), int(bounding_box['height'])
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # Extract the region
            kit_region = image[y:y+h, x:x+w]
            
            logger.info(f"Extracted kit region: {kit_region.shape}")
            return kit_region
            
        except Exception as e:
            logger.error(f"Error extracting kit region: {e}")
            raise
    
    def extract_result_roi(self, kit_image: np.ndarray) -> np.ndarray:
        """
        Extract the result region of interest (ROI) from the kit image
        
        Args:
            kit_image: Cropped kit image
            
        Returns:
            ROI image containing the test result area
        """
        try:
            height, width = kit_image.shape[:2]
            
            # The result area is typically in the bottom portion of the kit
            # Calculate ROI coordinates
            roi_height = int(height * self.roi_ratio)
            roi_y = height - roi_height
            
            # Extract the bottom portion as ROI
            roi = kit_image[roi_y:height, :]
            
            # Apply additional preprocessing
            roi = self._preprocess_roi(roi)
            
            logger.info(f"Extracted ROI: {roi.shape}")
            return roi
            
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            raise
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess the ROI for better classification
        
        Args:
            roi: Raw ROI image
            
        Returns:
            Preprocessed ROI image
        """
        try:
            # Convert to grayscale for line detection
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding to enhance lines
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to 3-channel for model input
            if len(thresh.shape) == 2:
                thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Error preprocessing ROI: {e}")
            return roi  # Return original if preprocessing fails
    
    def align_kit(self, kit_image: np.ndarray) -> np.ndarray:
        """
        Align the kit image using perspective correction
        
        Args:
            kit_image: Input kit image
            
        Returns:
            Aligned kit image
        """
        try:
            # Find the kit edges using contour detection
            gray = cv2.cvtColor(kit_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found for alignment")
                return kit_image
            
            # Find the largest contour (likely the kit)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to get corners
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 4:
                # Get the four corners
                corners = self._order_points(approx.reshape(4, 2))
                
                # Calculate perspective transform
                width, height = self.target_kit_size
                dst_points = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Apply perspective transform
                matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
                aligned = cv2.warpPerspective(kit_image, matrix, self.target_kit_size)
                
                logger.info("Kit alignment completed")
                return aligned
            else:
                logger.warning("Could not find 4 corners for alignment")
                return kit_image
                
        except Exception as e:
            logger.error(f"Error aligning kit: {e}")
            return kit_image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the format: top-left, top-right, bottom-right, bottom-left
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered points array
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better analysis
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def detect_test_lines(self, roi: np.ndarray) -> Dict[str, Any]:
        """
        Detect test lines in the ROI for additional validation
        
        Args:
            roi: Region of interest image
            
        Returns:
            Dictionary with line detection results
        """
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect horizontal lines (test lines are typically horizontal)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find contours of horizontal lines
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter lines by area and aspect ratio
            valid_lines = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if aspect_ratio > 3:  # Horizontal lines should be wider than tall
                        valid_lines.append({
                            'contour': contour,
                            'area': area,
                            'bbox': (x, y, w, h),
                            'aspect_ratio': aspect_ratio
                        })
            
            # Sort lines by y-coordinate (top to bottom)
            valid_lines.sort(key=lambda line: line['bbox'][1])
            
            return {
                'lines_detected': len(valid_lines),
                'lines': valid_lines,
                'has_control_line': len(valid_lines) > 0,
                'has_test_line': len(valid_lines) > 1
            }
            
        except Exception as e:
            logger.error(f"Error detecting test lines: {e}")
            return {
                'lines_detected': 0,
                'lines': [],
                'has_control_line': False,
                'has_test_line': False,
                'error': str(e)
            }
