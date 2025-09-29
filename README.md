# CVML Cardio Health Check Kit Analyzer

A comprehensive Computer Vision + Machine Learning project for analyzing cardio health check kits using modern web technologies.

## Project Overview

This project combines:
- **Computer Vision**: Object detection and image classification
- **Machine Learning**: YOLOv8 for kit detection, CNN for result classification
- **Web Development**: Next.js frontend with Python FastAPI backend
- **Mobile Optimization**: TensorFlow Lite for on-device processing

## Architecture

```
CVML/
├── backend/                 # Python FastAPI backend
│   ├── models/             # ML model files and training scripts
│   ├── api/                # API endpoints and routes
│   ├── services/           # Business logic and image processing
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js React frontend
│   ├── components/         # React components
│   ├── pages/             # Next.js pages
│   └── package.json       # Node.js dependencies
└── data/                   # Training data and annotations
    ├── images/            # Raw training images
    └── annotations/       # Bounding box annotations
```

## Features

- 📱 **Mobile-First**: Camera access for real-time kit scanning
- 🔍 **Object Detection**: YOLOv8-based kit localization
- 🧠 **AI Classification**: CNN-based result interpretation
- ⚡ **Fast Processing**: Optimized for mobile inference
- 🎯 **High Accuracy**: Trained on diverse dataset
- 📊 **Confidence Scoring**: Result reliability indicators

## Quick Start

### Automated Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd CVML

# Run the automated setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Start the application
./start_all.sh
```

### Manual Setup

#### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Docker Setup
```bash
# Build and start all services
docker-compose up --build

# Or start individual services
docker-compose up backend
docker-compose up frontend
```

## Technology Stack

- **Backend**: Python, FastAPI, OpenCV, TensorFlow/PyTorch
- **Frontend**: Next.js, React, TypeScript
- **ML Models**: YOLOv8, MobileNet, TensorFlow Lite
- **Image Processing**: OpenCV, PIL
- **Deployment**: Docker, Vercel (frontend), Railway/Heroku (backend)

## Training and Model Development

### Data Preparation
1. **Collect Images**: Gather diverse cardio health check kit images
2. **Annotate Data**: Use LabelImg or VGG Image Annotator for bounding boxes
3. **Organize Dataset**: Follow the structure in `data/README.md`

### Model Training
```bash
# Train object detection model
cd backend/models
python train_detection.py

# Train classification model
python train_classification.py
```

### Model Optimization
```bash
# Convert to TensorFlow Lite for mobile deployment
python -c "
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

## API Documentation

### Endpoints
- `POST /api/scan-kit` - Analyze kit image
- `POST /api/validate-image` - Validate image suitability
- `GET /health` - API health check
- `GET /docs` - Interactive API documentation

### Example Usage
```python
import requests

# Analyze an image
with open('kit_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/scan-kit',
        files={'file': f}
    )
    result = response.json()
    print(f"Result: {result['result']} (Confidence: {result['confidence']})")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
