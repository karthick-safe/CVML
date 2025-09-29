"""
Annotate existing CardioChek Plus images for training data
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import argparse

class CardioChekImageAnnotator:
    """
    Interactive tool for annotating existing CardioChek Plus images
    """
    
    def __init__(self, images_dir: str = "data/cardio_chek/images"):
        self.images_dir = images_dir
        self.annotations_dir = os.path.join(os.path.dirname(images_dir), "annotations")
        
        # Create directories
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Annotation data
        self.current_annotations = []
        self.current_image = None
        self.current_image_path = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_bbox = None
        self.current_image_index = 0
        self.image_files = []
        
    def load_images(self):
        """Load all images from the directory"""
        if not os.path.exists(self.images_dir):
            print(f"Images directory not found: {self.images_dir}")
            return False
            
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not self.image_files:
            print(f"No images found in {self.images_dir}")
            return False
            
        print(f"Found {len(self.image_files)} images")
        return True
    
    def annotate_images(self):
        """
        Interactive annotation of existing images
        """
        if not self.load_images():
            return
            
        print("CardioChek Plus Image Annotation Tool")
        print("=" * 50)
        print("Instructions:")
        print("1. Click and drag to draw bounding box around CardioChek Plus device")
        print("2. Press 's' to save annotation and move to next image")
        print("3. Press 'n' to skip current image")
        print("4. Press 'r' to reset current annotations")
        print("5. Press 'q' to quit")
        print("=" * 50)
        
        cv2.namedWindow('CardioChek Annotation', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('CardioChek Annotation', self._mouse_callback)
        
        while self.current_image_index < len(self.image_files):
            self._load_current_image()
            
            if self.current_image is None:
                self.current_image_index += 1
                continue
                
            while True:
                # Display current image with annotations
                display_image = self.current_image.copy()
                
                # Draw existing annotations
                for bbox in self.current_annotations:
                    cv2.rectangle(display_image, bbox[0], bbox[1], (0, 255, 0), 2)
                    cv2.putText(display_image, "CardioChek Plus", 
                              (bbox[0][0], bbox[0][1] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw temporary bounding box
                if self.temp_bbox:
                    cv2.rectangle(display_image, self.temp_bbox[0], self.temp_bbox[1], (255, 0, 0), 2)
                
                # Add instructions
                cv2.putText(display_image, f"Image {self.current_image_index + 1}/{len(self.image_files)}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_image, "Draw bounding box around CardioChek Plus device", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('CardioChek Annotation', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                elif key == ord('s'):
                    if self._save_annotation():
                        self.current_image_index += 1
                        break
                elif key == ord('n'):
                    print(f"Skipped image {self.current_image_index + 1}")
                    self.current_image_index += 1
                    break
                elif key == ord('r'):
                    self._reset_annotations()
                elif key == ord(' '):  # Spacebar to add current bbox
                    if self.temp_bbox:
                        self.current_annotations.append(self.temp_bbox)
                        self.temp_bbox = None
                        print(f"Added annotation. Total: {len(self.current_annotations)}")
        
        cv2.destroyAllWindows()
        print("Annotation complete!")
    
    def _load_current_image(self):
        """Load current image"""
        if self.current_image_index >= len(self.image_files):
            return
            
        image_path = os.path.join(self.images_dir, self.image_files[self.current_image_index])
        self.current_image_path = image_path
        
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            print(f"Could not load image: {image_path}")
            return
            
        # Reset annotations for new image
        self.current_annotations = []
        self.temp_bbox = None
        
        print(f"Loaded: {self.image_files[self.current_image_index]}")
    
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
    
    def _save_annotation(self):
        """Save current annotations"""
        if not self.current_image_path or not self.current_annotations:
            print("No image or annotations to save")
            return False
        
        # Create annotation data
        annotation_data = {
            "image_path": self.current_image_path,
            "timestamp": datetime.now().isoformat(),
            "annotations": []
        }
        
        h, w = self.current_image.shape[:2]
        
        for bbox in self.current_annotations:
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            
            # Normalize coordinates
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
        annotation_filename = os.path.basename(self.current_image_path).replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')
        annotation_path = os.path.join(self.annotations_dir, annotation_filename)
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        
        print(f"Saved annotation: {annotation_filename}")
        print(f"Annotations: {len(self.current_annotations)}")
        return True
    
    def _reset_annotations(self):
        """Reset current annotations"""
        self.current_annotations = []
        self.temp_bbox = None
        print("Annotations reset")
    
    def create_yolo_dataset(self):
        """Convert annotations to YOLO format"""
        print("Converting to YOLO format...")
        
        yolo_dir = os.path.join(os.path.dirname(self.images_dir), "yolo")
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
    parser = argparse.ArgumentParser(description='CardioChek Plus Image Annotation')
    parser.add_argument('--images-dir', default='data/cardio_chek/images', help='Images directory')
    parser.add_argument('--mode', choices=['annotate', 'convert'], default='annotate', help='Mode: annotate or convert')
    
    args = parser.parse_args()
    
    annotator = CardioChekImageAnnotator(args.images_dir)
    
    if args.mode == 'annotate':
        annotator.annotate_images()
    elif args.mode == 'convert':
        annotator.create_yolo_dataset()

if __name__ == "__main__":
    main()
