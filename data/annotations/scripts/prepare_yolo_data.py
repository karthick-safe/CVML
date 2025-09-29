"""
Data preparation script for YOLO object detection training
Converts various annotation formats to YOLO format and prepares dataset
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple
import random
from sklearn.model_selection import train_test_split

class YOLODataPreparer:
    """Prepares data for YOLO object detection training"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ['cardio_kit']
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
    def create_directory_structure(self):
        """Create YOLO dataset directory structure"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_config, f)
    
    def convert_labelimg_to_yolo(self, labelimg_dir: str):
        """Convert LabelImg XML annotations to YOLO format"""
        labelimg_path = Path(labelimg_dir)
        
        for xml_file in labelimg_path.glob('*.xml'):
            # Parse XML and convert to YOLO format
            # Implementation would parse XML and extract bounding boxes
            pass
    
    def convert_via_to_yolo(self, via_json: str):
        """Convert VGG Image Annotator JSON to YOLO format"""
        with open(via_json, 'r') as f:
            via_data = json.load(f)
        
        for image_id, image_data in via_data.items():
            if 'regions' not in image_data:
                continue
            
            image_name = image_data['filename']
            image_path = self.source_dir / 'images' / image_name
            
            if not image_path.exists():
                continue
            
            # Get image dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            # Convert regions to YOLO format
            yolo_annotations = []
            for region in image_data['regions']:
                if 'shape_attributes' not in region:
                    continue
                
                shape = region['shape_attributes']
                if shape['name'] != 'rect':
                    continue
                
                # Convert to YOLO format (normalized center coordinates)
                x = shape['x']
                y = shape['y']
                w = shape['width']
                h = shape['height']
                
                # Convert to center coordinates and normalize
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_width = w / width
                norm_height = h / height
                
                # Class ID (assuming cardio_kit = 0)
                class_id = 0
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
            
            # Save YOLO annotation file
            label_file = self.output_dir / 'labels' / f"{Path(image_name).stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Split dataset into train/validation/test sets"""
        # Get all image files
        image_files = list(self.source_dir.glob('images/*.jpg')) + list(self.source_dir.glob('images/*.png'))
        
        # Split the dataset
        train_files, temp_files = train_test_split(
            image_files, test_size=(val_ratio + test_ratio), random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
        )
        
        # Copy files to respective directories
        for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for img_file in files:
                # Copy image
                dst_img = self.output_dir / split / 'images' / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label
                label_file = self.source_dir / 'labels' / f"{img_file.stem}.txt"
                if label_file.exists():
                    dst_label = self.output_dir / split / 'labels' / f"{img_file.stem}.txt"
                    shutil.copy2(label_file, dst_label)
    
    def validate_dataset(self):
        """Validate the prepared dataset"""
        issues = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            # Check if directories exist
            if not images_dir.exists():
                issues.append(f"Missing images directory: {images_dir}")
                continue
            
            if not labels_dir.exists():
                issues.append(f"Missing labels directory: {labels_dir}")
                continue
            
            # Check for missing labels
            image_files = set(f.stem for f in images_dir.glob('*'))
            label_files = set(f.stem for f in labels_dir.glob('*.txt'))
            
            missing_labels = image_files - label_files
            if missing_labels:
                issues.append(f"Missing labels in {split}: {missing_labels}")
            
            # Check for orphaned labels
            orphaned_labels = label_files - image_files
            if orphaned_labels:
                issues.append(f"Orphaned labels in {split}: {orphaned_labels}")
        
        if issues:
            print("Dataset validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Dataset validation passed!")
        
        return len(issues) == 0
    
    def generate_statistics(self):
        """Generate dataset statistics"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_dir / split / 'images'
            labels_dir = self.output_dir / split / 'labels'
            
            if not images_dir.exists():
                continue
            
            image_count = len(list(images_dir.glob('*')))
            label_count = len(list(labels_dir.glob('*.txt')))
            
            # Count annotations
            annotation_count = 0
            for label_file in labels_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    annotation_count += len(f.readlines())
            
            stats[split] = {
                'images': image_count,
                'labels': label_count,
                'annotations': annotation_count
            }
        
        print("Dataset Statistics:")
        for split, data in stats.items():
            print(f"  {split}: {data['images']} images, {data['annotations']} annotations")
        
        return stats

def main():
    """Main data preparation pipeline"""
    preparer = YOLODataPreparer(
        source_dir="data/raw",
        output_dir="data/kit_detection"
    )
    
    try:
        # Create directory structure
        preparer.create_directory_structure()
        print("Directory structure created")
        
        # Convert annotations (example with VIA format)
        via_json = "data/annotations/via/annotations.json"
        if os.path.exists(via_json):
            preparer.convert_via_to_yolo(via_json)
            print("VIA annotations converted")
        
        # Split dataset
        preparer.split_dataset()
        print("Dataset split completed")
        
        # Validate dataset
        if preparer.validate_dataset():
            print("Dataset validation passed")
        else:
            print("Dataset validation failed")
            return
        
        # Generate statistics
        preparer.generate_statistics()
        
        print("Data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error in data preparation: {e}")
        raise

if __name__ == "__main__":
    main()
