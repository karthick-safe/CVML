# CVML Training Data

This directory contains the training data and annotations for the CVML Cardio Health Check Kit Analyzer.

## Directory Structure

```
data/
├── kit_detection/           # Object detection training data
│   ├── train/
│   │   ├── images/         # Training images
│   │   └── labels/         # YOLO format annotations
│   ├── val/
│   │   ├── images/         # Validation images
│   │   └── labels/         # YOLO format annotations
│   └── test/
│       ├── images/         # Test images
│       └── labels/         # YOLO format annotations
├── classification/          # Result classification training data
│   ├── positive/           # Positive result images
│   ├── negative/           # Negative result images
│   └── invalid/            # Invalid result images
└── annotations/            # Annotation files and tools
    ├── labelimg/           # LabelImg annotation files
    ├── via/               # VGG Image Annotator files
    └── scripts/           # Data processing scripts
```

## Data Collection Guidelines

### Kit Detection Data
- **Minimum Images**: 1000+ images per split (train/val/test)
- **Diversity**: Different lighting conditions, angles, backgrounds
- **Quality**: High resolution (minimum 640x640), clear images
- **Variations**: Different kit brands, orientations, partial views

### Classification Data
- **Positive Results**: 500+ images showing positive test lines
- **Negative Results**: 500+ images showing negative test lines  
- **Invalid Results**: 300+ images showing invalid/faulty results
- **Quality**: Clear ROI images focused on test result area

## Annotation Guidelines

### Object Detection (YOLO Format)
- Bounding box coordinates: `class_id x_center y_center width height`
- All coordinates normalized to [0, 1]
- Class ID: 0 for cardio_kit
- Tight bounding boxes around the entire kit

### Classification Labels
- **Positive**: Clear test line visible in result area
- **Negative**: Only control line visible, no test line
- **Invalid**: No lines, unclear results, or test failure

## Data Augmentation

The training pipeline includes:
- Random rotation (±15°)
- Random brightness/contrast adjustment
- Random horizontal flip
- Random zoom/crop
- Color jittering
- Gaussian noise

## Quality Control

- Manual review of all annotations
- Cross-validation between annotators
- Regular data quality audits
- Balanced class distribution
- Representative test set

## Usage

1. Place raw images in appropriate directories
2. Use annotation tools to create labels
3. Run data processing scripts
4. Train models using provided scripts
5. Validate on test set
