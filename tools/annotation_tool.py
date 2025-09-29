#!/usr/bin/env python3
"""
Interactive annotation tool for CardioChek Plus images
Allows users to draw bounding boxes for device detection training
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

class AnnotationTool:
    def __init__(self, images_dir: str, output_dir: str):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_image = None
        self.current_image_path = None
        self.bbox_start = None
        self.bbox_end = None
        self.drawing = False
        self.annotations = []
        
        print("🎨 CARDIOCHECK PLUS ANNOTATION TOOL")
        print("=" * 50)
        print("Controls:")
        print("  • Click and drag to draw bounding box")
        print("  • Press 's' to save annotation")
        print("  • Press 'r' to reset current box")
        print("  • Press 'n' for next image")
        print("  • Press 'q' to quit")
        print("=" * 50)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.bbox_start = (x, y)
            self.bbox_end = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.bbox_end = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox_end = (x, y)
            print(f"✓ Box drawn: {self.bbox_start} to {self.bbox_end}")
    
    def draw_current_box(self, image):
        """Draw the current bounding box on the image"""
        display_image = image.copy()
        
        if self.bbox_start and self.bbox_end:
            cv2.rectangle(display_image, self.bbox_start, self.bbox_end, (0, 255, 0), 2)
            
            # Show dimensions
            width = abs(self.bbox_end[0] - self.bbox_start[0])
            height = abs(self.bbox_end[1] - self.bbox_start[1])
            text = f"{width}x{height}"
            cv2.putText(display_image, text, (self.bbox_start[0], self.bbox_start[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return display_image
    
    def save_annotation(self):
        """Save the current annotation"""
        if not self.bbox_start or not self.bbox_end:
            print("❌ No bounding box drawn")
            return False
        
        # Calculate bounding box coordinates
        x1, y1 = self.bbox_start
        x2, y2 = self.bbox_end
        
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        annotation = {
            "image": str(self.current_image_path.name),
            "class": "cardio_chek_plus",
            "bbox": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "image_size": {
                "width": self.current_image.shape[1],
                "height": self.current_image.shape[0]
            }
        }
        
        self.annotations.append(annotation)
        
        # Save to JSON file
        output_file = self.output_dir / f"{self.current_image_path.stem}_annotation.json"
        with open(output_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Save YOLO format
        img_h, img_w = self.current_image.shape[:2]
        x_center = (x + width/2) / img_w
        y_center = (y + height/2) / img_h
        norm_width = width / img_w
        norm_height = height / img_h
        
        yolo_file = self.output_dir / f"{self.current_image_path.stem}.txt"
        with open(yolo_file, 'w') as f:
            f.write(f"0 {x_center} {y_center} {norm_width} {norm_height}\n")
        
        print(f"✅ Saved annotation: {output_file}")
        print(f"✅ Saved YOLO format: {yolo_file}")
        
        return True
    
    def annotate_images(self):
        """Main annotation loop"""
        # Get all images
        image_files = list(self.images_dir.glob("*.jpg")) + \
                     list(self.images_dir.glob("*.png")) + \
                     list(self.images_dir.glob("*.jpeg"))
        
        if not image_files:
            print("❌ No images found in directory")
            return
        
        print(f"📸 Found {len(image_files)} images to annotate")
        
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        for i, img_path in enumerate(image_files):
            self.current_image_path = img_path
            self.current_image = cv2.imread(str(img_path))
            
            if self.current_image is None:
                print(f"❌ Failed to load: {img_path}")
                continue
            
            self.bbox_start = None
            self.bbox_end = None
            
            print(f"\n📷 Image {i+1}/{len(image_files)}: {img_path.name}")
            
            while True:
                display_image = self.draw_current_box(self.current_image)
                cv2.imshow('Annotation Tool', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # Save
                    if self.save_annotation():
                        break
                
                elif key == ord('r'):  # Reset
                    self.bbox_start = None
                    self.bbox_end = None
                    print("🔄 Reset bounding box")
                
                elif key == ord('n'):  # Next without saving
                    print("⏭️  Skipping image")
                    break
                
                elif key == ord('q'):  # Quit
                    print("👋 Quitting annotation tool")
                    cv2.destroyAllWindows()
                    return
        
        cv2.destroyAllWindows()
        
        # Save summary
        summary_file = self.output_dir / "annotations_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "total_annotations": len(self.annotations),
                "annotations": self.annotations
            }, f, indent=2)
        
        print(f"\n✅ Annotation complete!")
        print(f"📊 Total annotations: {len(self.annotations)}")
        print(f"💾 Summary saved: {summary_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotate CardioChek Plus images')
    parser.add_argument('--images', type=str, default='data/cardio_chek/images',
                       help='Directory containing images to annotate')
    parser.add_argument('--output', type=str, default='data/cardio_chek/annotations',
                       help='Directory to save annotations')
    
    args = parser.parse_args()
    
    tool = AnnotationTool(args.images, args.output)
    tool.annotate_images()

if __name__ == "__main__":
    main()
