"""
Data Collection Script for CardioChek Plus Training
Collects and annotates images for CVML model training
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import argparse

class CardioChekDataCollector:
    """
    Data collection tool for CardioChek Plus training data
    """
    
    def __init__(self, data_dir: str = "data/cardio_chek"):
        self.data_dir = data_dir
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.images_dir = os.path.join(data_dir, "images")
        
        # Create directories
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Annotation data
        self.current_annotations = []
        self.current_image = None
        self.current_image_path = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_bbox = None
        
    def collect_data_interactive(self):
        """
        Interactive data collection with bounding box annotation
        """
        print("CardioChek Plus Data Collection Tool")
        print("=" * 50)
        print("Instructions:")
        print("1. Press 'c' to capture current frame")
        print("2. Click and drag to draw bounding box around CardioChek Plus device")
        print("3. Press 's' to save annotation")
        print("4. Press 'n' for next image")
        print("5. Press 'q' to quit")
        print("=" * 50)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        cv2.namedWindow('CardioChek Data Collection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('CardioChek Data Collection', self._mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display current frame with any existing annotations
            display_frame = frame.copy()
            if self.temp_bbox:
                cv2.rectangle(display_frame, self.temp_bbox[0], self.temp_bbox[1], (0, 255, 0), 2)
            
            # Show existing annotations
            for bbox in self.current_annotations:
                cv2.rectangle(display_frame, bbox[0], bbox[1], (255, 0, 0), 2)
            
            cv2.imshow('CardioChek Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self._capture_frame(frame)
            elif key == ord('s'):
                self._save_annotation()
            elif key == ord('n'):
                self._next_image()
            elif key == ord('r'):
                self._reset_annotations()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bounding box drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.temp_bbox = (self.start_point, self.end_point)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.temp_bbox = (self.start_point, self.end_point)
    
    def _capture_frame(self, frame):
        """Capture current frame for annotation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cardio_chek_{timestamp}.jpg"
        self.current_image_path = os.path.join(self.images_dir, filename)
        
        cv2.imwrite(self.current_image_path, frame)
        print(f"Captured: {filename}")
        
        # Reset annotations for new image
        self.current_annotations = []
        self.temp_bbox = None
    
    def _save_annotation(self):
        """Save current annotations"""
        if not self.current_image_path or not self.current_annotations:
            print("No image or annotations to save")
            return
        
        # Create annotation data
        annotation_data = {
            "image_path": self.current_image_path,
            "timestamp": datetime.now().isoformat(),
            "annotations": []
        }
        
        for bbox in self.current_annotations:
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            
            # Normalize coordinates
            img = cv2.imread(self.current_image_path)
            h, w = img.shape[:2]
            
            annotation_data["annotations"].append({
                "class": "cardio_chek_plus",
                "bbox": {
                    "x": min(x1, x2) / w,
                    "y": min(y1, y2) / h,
                    "width": abs(x2 - x1) / w,
                    "height": abs(y2 - y1) / h
                },
                "confidence": 1.0
            })
        
        # Save annotation file
        annotation_filename = os.path.basename(self.current_image_path).replace('.jpg', '.json')
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"Saved annotation: {annotation_filename}")
        print(f"Annotations: {len(self.current_annotations)}")
    
    def _next_image(self):
        """Move to next image"""
        if self.temp_bbox:
            self.current_annotations.append(self.temp_bbox)
            self.temp_bbox = None
        print("Ready for next image")
    
    def _reset_annotations(self):
        """Reset current annotations"""
        self.current_annotations = []
        self.temp_bbox = None
        print("Annotations reset")
    
    def create_yolo_dataset(self):
        """Convert annotations to YOLO format"""
        print("Converting to YOLO format...")
        
        yolo_dir = os.path.join(self.data_dir, "yolo")
        os.makedirs(yolo_dir, exist_ok=True)
        
        # Create class names file
        with open(os.path.join(yolo_dir, "classes.txt"), 'w') as f:
            f.write("cardio_chek_plus\n")
        
        # Process all annotation files
        for annotation_file in os.listdir(self.annotations_dir):
            if annotation_file.endswith('.json'):
                self._convert_to_yolo(annotation_file, yolo_dir)
        
        print(f"YOLO dataset created in: {yolo_dir}")
    
    def _convert_to_yolo(self, annotation_file: str, yolo_dir: str):
        """Convert single annotation to YOLO format"""
        annotation_path = os.path.join(self.annotations_dir, annotation_file)
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Create YOLO annotation file
        yolo_filename = annotation_file.replace('.json', '.txt')
        yolo_path = os.path.join(yolo_dir, yolo_filename)
        
        with open(yolo_path, 'w') as f:
            for annotation in data["annotations"]:
                bbox = annotation["bbox"]
                # YOLO format: class_id center_x center_y width height
                center_x = bbox["x"] + bbox["width"] / 2
                center_y = bbox["y"] + bbox["height"] / 2
                width = bbox["width"]
                height = bbox["height"]
                
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Copy image to YOLO directory
        image_filename = os.path.basename(data["image_path"])
        yolo_image_path = os.path.join(yolo_dir, image_filename)
        
        import shutil
        shutil.copy2(data["image_path"], yolo_image_path)

def main():
    parser = argparse.ArgumentParser(description='CardioChek Plus Data Collection')
    parser.add_argument('--data-dir', default='data/cardio_chek', help='Data directory')
    parser.add_argument('--mode', choices=['collect', 'convert'], default='collect', help='Mode: collect or convert')
    
    args = parser.parse_args()
    
    collector = CardioChekDataCollector(args.data_dir)
    
    if args.mode == 'collect':
        collector.collect_data_interactive()
    elif args.mode == 'convert':
        collector.create_yolo_dataset()

if __name__ == "__main__":
    main()
