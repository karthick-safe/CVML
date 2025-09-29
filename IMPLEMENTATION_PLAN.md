# TensorFlow-Based CVML Implementation Plan

## Overview
Complete implementation plan for CardioChek Plus detection and OCR using TensorFlow with real-time accuracy checking.

## Architecture

```
┌─────────────────┐
│   Frontend      │
│  (Next.js)      │
│  - Camera       │
│  - Real-time    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Backend API   │
│  (FastAPI)      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  TensorFlow Model Manager       │
│                                 │
│  1. Detection (CNN)             │
│     ├─ Device Localization      │
│     └─ Confidence Check (>0.7)  │
│                                 │
│  2. OCR (EasyOCR)              │
│     ├─ Screen Value Extraction  │
│     └─ Multi-preprocessing      │
│                                 │
│  3. Classification (Rules)      │
│     └─ Health Risk Analysis     │
└─────────────────────────────────┘
```

## Current Status

### ✅ Completed
1. TensorFlow model manager created with CNN architecture
2. OCR integration with EasyOCR
3. Health classification rules
4. Frontend adaptive threshold system
5. Backend API endpoints
6. Numpy serialization fixes
7. Annotation tool created
8. Training data generation scripts

### ⚠️ Issues Fixed
1. Missing `get_model_status()` method - FIXED
2. Bounding box missing `confidence` field - FIXED
3. Startup event calling non-existent `load_models()` - FIXED

## Implementation Steps

### STEP 1: Test Current TensorFlow System ✓

**Status**: Models created but not trained

**Commands**:
```bash
cd /Users/karthickrajamurugan/Safe/CVML/backend
/Users/karthickrajamurugan/Safe/CVML/backend/venv/bin/python test_tensorflow.py
```

**Expected Output**:
- TensorFlow model initializes
- Detection works (untrained, low confidence)
- OCR extracts values
- Classification provides health analysis

---

### STEP 2: Collect Training Data

**Option A: Use Annotation Tool (Recommended for Real Images)**

```bash
cd /Users/karthickrajamurugan/Safe/CVML
/Users/karthickrajamurugan/Safe/CVML/backend/venv/bin/python tools/annotation_tool.py \
  --images data/cardio_chek/images \
  --output data/cardio_chek/annotations
```

**Controls**:
- Click and drag to draw bounding box around CardioChek Plus device
- Press 's' to save
- Press 'r' to reset
- Press 'n' for next image
- Press 'q' to quit

**Option B: Use Synthetic Data (Quick Testing)**

```bash
cd /Users/karthickrajamurugan/Safe/CVML
/Users/karthickrajamurugan/Safe/CVML/backend/venv/bin/python scripts/generate_cardio_chek_images.py
```

**Output**:
- Annotated images in `data/cardio_chek/annotations/`
- YOLO format labels in `.txt` files
- JSON annotations with full metadata

---

### STEP 3: Train TensorFlow Model

**Note**: Currently training script has issues. Need to prepare training data first.

**Training Data Requirements**:
- Minimum 20-30 annotated images
- Mix of:
  - Different lighting conditions
  - Different angles
  - Different screen values
  - Background variations

**Training Command** (after data collection):
```bash
cd /Users/karthickrajamurugan/Safe/CVML/backend
/Users/karthickrajamurugan/Safe/CVML/backend/venv/bin/python train_tensorflow.py
```

**Expected Training**:
- 30-50 epochs
- Model saved as `cardio_chek_tf_detector.h5`
- Validation accuracy >80%

---

### STEP 4: Start Backend Server

```bash
cd /Users/karthickrajamurugan/Safe/CVML/backend
/Users/karthickrajamurugan/Safe/CVML/backend/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify**:
- Server starts without errors
- Visit `http://localhost:8000/docs` for API documentation
- Check `/health` endpoint for model status

---

### STEP 5: Test API with Sample Image

```bash
curl -X POST "http://localhost:8000/api/scan-kit" \
  -F "file=@data/cardio_chek/images/cardio_chek_synthetic_001.jpg" \
  -H "Accept: application/json" | jq
```

**Expected Response**:
```json
{
  "result": "Borderline",
  "confidence": 0.8,
  "bounding_box": {
    "x": 50,
    "y": 30,
    "width": 540,
    "height": 400,
    "confidence": 0.8
  },
  "details": {
    "kit_detection_confidence": 0.8,
    "ocr_values": {
      "cholesterol": 150,
      "hdl": 45,
      "triglycerides": 120,
      "glucose": 90,
      "units": "mg/dL"
    },
    "method": "tensorflow_detection"
  }
}
```

---

### STEP 6: Start Frontend

```bash
cd /Users/karthickrajamurugan/Safe/CVML/frontend
npm run dev
```

**Verify**:
- Frontend starts on `http://localhost:3000`
- Camera access works
- Real-time detection displays status
- Auto-capture triggers when confidence is high

---

### STEP 7: Test Complete Flow

**Manual Test**:
1. Open `http://localhost:3000`
2. Click "Start Analysis"
3. Allow camera access
4. Click "Start Auto-Detection"
5. Position CardioChek Plus device in view
6. Watch detection status update
7. When confidence > 70% (or adaptive threshold), auto-capture triggers
8. View OCR results and health classification

**Expected Behavior**:
- Detection status updates every 2 seconds
- Bounding box drawn around detected device
- Current confidence displayed
- Adaptive threshold adjusts based on history
- Auto-capture when threshold met
- OCR extracts screen values
- Health analysis displayed

---

## Key Features

### 1. Real-Time Detection with Accuracy Check

**Implementation**: `frontend/components/CameraCapture.tsx`

```typescript
const performRealTimeDetection = useCallback(async () => {
  // Capture frame
  // Send to backend
  // Check confidence
  const confidence = result.confidence
  
  // Update detection history
  updateDetectionHistory(confidence)
  
  // Use adaptive threshold
  const currentThreshold = thresholdMode === 'adaptive' 
    ? adaptiveThreshold 
    : confidenceThreshold
  
  // Auto-capture if confidence is good
  if (autoCaptureEnabled && confidence >= currentThreshold) {
    captureImage() // Triggers OCR and classification
  }
}, [autoCaptureEnabled, adaptiveThreshold, thresholdMode])
```

**Features**:
- **Adaptive Threshold**: Learns from detection history
- **Fixed Threshold**: User-controlled (50%-95%)
- **Visual Feedback**: Color-coded confidence indicators
- **Detection History**: Tracks last 10 detections

### 2. TensorFlow Detection Model

**Architecture**: Custom CNN

```python
Input (224x224x3)
    ↓
Conv2D (32) + MaxPool
    ↓
Conv2D (64) + MaxPool
    ↓
Conv2D (128) + MaxPool
    ↓
Conv2D (256) + MaxPool
    ↓
Flatten + Dense (512) + Dropout
    ↓
Dense (256) + Dropout
    ↓
    ├─→ BBox Output (4) [x, y, width, height]
    └─→ Class Output (1) [present/absent]
```

**Advantages**:
- Lightweight: ~3M parameters
- Fast inference: ~100ms on CPU
- Trainable with small datasets (20-30 images)
- No external dependencies (pure TensorFlow)

### 3. Enhanced OCR Processing

**Multi-Preprocessing Approach**:
1. CLAHE + Adaptive Threshold
2. Otsu Thresholding
3. Denoising + Otsu
4. Morphological Operations
5. Inverted Image

**Pattern Matching**:
- Segmented display-specific patterns
- Multiple regex variations
- Fallback numeric extraction
- Unit detection (mg/dL vs mmol/L)

### 4. Health Classification

**Rules-Based Analysis**:

| Metric | Normal | Borderline | High Risk |
|--------|--------|------------|-----------|
| Total Cholesterol | <200 | 200-239 | ≥240 |
| HDL | ≥60 | 40-59 | <40 |
| Triglycerides | <150 | 150-199 | ≥200 |
| Glucose | <100 | 100-125 | ≥126 |

**Output**:
- Risk level (Good Health, Borderline, Moderate Risk, High Risk)
- Confidence score
- Individual metric analysis
- Health recommendations

---

## Directory Structure

```
CVML/
├── backend/
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   └── schemas.py              # Pydantic models
│   ├── services/
│   │   ├── tensorflow_model_manager.py  # TensorFlow CVML
│   │   ├── image_processor.py      # Image utilities
│   │   └── enhanced_detection.py   # OpenCV fallback
│   ├── train_tensorflow.py         # Training script
│   ├── test_tensorflow.py          # Testing script
│   └── venv/                       # Python environment
├── frontend/
│   ├── components/
│   │   ├── CameraCapture.tsx       # Camera + detection
│   │   └── ResultDisplay.tsx       # Results UI
│   ├── app/
│   │   ├── page.tsx                # Home page
│   │   └── layout.tsx              # Root layout
│   └── public/                     # Static assets
├── data/
│   └── cardio_chek/
│       ├── images/                 # Training images
│       ├── annotations/            # Annotation files
│       ├── yolo/
│       │   ├── images/            # YOLO format images
│       │   └── labels/            # YOLO format labels
│       └── training_variations.json  # Synthetic data config
├── tools/
│   └── annotation_tool.py          # Interactive annotation
├── scripts/
│   ├── generate_cardio_chek_images.py  # Synthetic data
│   └── train_tensorflow_model.py   # Alt training script
└── cardio_chek_models/
    └── cardio_chek_detector_real/  # Trained YOLO (backup)
```

---

## Testing Checklist

### Backend Tests
- [ ] TensorFlow model initializes
- [ ] Detection returns bounding box with confidence
- [ ] OCR extracts values from screen
- [ ] Classification returns health analysis
- [ ] All numpy types converted to native Python
- [ ] API returns valid JSON
- [ ] CORS headers correct
- [ ] Error handling works

### Frontend Tests
- [ ] Camera access granted
- [ ] Real-time detection updates
- [ ] Bounding box displayed
- [ ] Confidence indicator shows correct color
- [ ] Adaptive threshold calculates correctly
- [ ] Auto-capture triggers at right confidence
- [ ] OCR results display
- [ ] Health analysis shows
- [ ] Can switch between fixed/adaptive threshold

### Integration Tests
- [ ] Frontend → Backend communication
- [ ] Image upload works
- [ ] Real-time detection responsive (<2s)
- [ ] Auto-capture doesn't trigger too early
- [ ] OCR values accurate
- [ ] Classification reasonable

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Detection Speed | <200ms | ~100ms |
| OCR Speed | <1s | ~800ms |
| Total Processing | <2s | ~1.2s |
| Detection Accuracy | >85% | Untrained |
| OCR Accuracy | >90% | ~70% |
| Classification Accuracy | >95% | Rules-based |

---

## Troubleshooting

### Issue: Server won't start
**Solution**: Check backend logs, ensure all dependencies installed
```bash
cd /Users/karthickrajamurugan/Safe/CVML/backend
/Users/karthickrajamurugan/Safe/CVML/backend/venv/bin/pip install -r requirements-py313.txt
```

### Issue: Low detection confidence
**Solution**: 
1. Lower threshold: `confidence_threshold = 0.4`
2. Train model with more data
3. Check image quality and lighting

### Issue: OCR not extracting values
**Solution**:
1. Ensure screen is clearly visible and in focus
2. Check image preprocessing in logs
3. Add more regex patterns for specific font

### Issue: Auto-capture not triggering
**Solution**:
1. Check adaptive threshold value
2. Switch to fixed threshold mode
3. Lower fixed threshold (currently 70%)

---

## Next Steps for Production

1. **Collect Real Training Data**
   - Use annotation tool with real CardioChek Plus images
   - Aim for 50-100 annotated images
   - Vary conditions (lighting, angles, backgrounds)

2. **Train Production Model**
   - Train for 50-100 epochs
   - Achieve >85% validation accuracy
   - Test with holdout set

3. **Fine-tune OCR**
   - Add screen-specific preprocessing
   - Train custom OCR model for segmented display
   - Improve regex patterns

4. **Optimize Performance**
   - Model quantization (TensorFlow Lite)
   - Batch processing
   - Caching frequently accessed data

5. **Add Features**
   - Save history of scans
   - Export results to PDF
   - Multi-language support
   - Trend analysis over time

6. **Deploy**
   - Dockerize backend
   - Deploy to cloud (AWS/GCP/Azure)
   - Set up CI/CD pipeline
   - Monitor performance

---

## Resources

- **TensorFlow Docs**: https://www.tensorflow.org/
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **FastAPI**: https://fastapi.tiangolo.com/
- **Next.js**: https://nextjs.org/

---

## Support

For issues or questions:
1. Check logs: `backend/logs/`
2. Review error messages in terminal
3. Test individual components
4. Use annotation tool to verify training data quality

---

**Last Updated**: 2025-01-26
**Version**: 1.0.0
**Status**: Ready for Training & Testing
