# 🔧 Setup Instructions for New Users

## ⚠️ Important: Model Files Not Included

The trained model files are **not included** in the Git repository due to GitHub's 100MB file size limit.

### Model Files (Generated Locally)
- `backend/cardio_chek_tf_detector.keras` (300MB)
- `backend/cardio_chek_tf_detector.h5` (300MB)
- `cardio_chek_models/` (YOLO models, 200MB+)

---

## 🚀 First-Time Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/karthick-safe/CVML.git
cd CVML
```

### **2. Backend Setup**
```bash
cd backend

# Create virtual environment
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements-py313.txt
```

### **3. Frontend Setup**
```bash
cd ../frontend

# Install dependencies
npm install
```

---

## 🎓 **Training the Model (Required for First-Time Users)**

### **Option 1: Use Annotation Tool (Real Images)**

**Step 1: Collect CardioChek Plus Images**
- Take 20-50 photos of CardioChek Plus devices
- Vary lighting, angles, backgrounds
- Save to `data/cardio_chek/images/`

**Step 2: Annotate Images**
```bash
cd /path/to/CVML
venv/bin/python tools/annotation_tool.py \
  --images data/cardio_chek/images \
  --output data/cardio_chek/annotations
```

**Controls**:
- Click & drag to draw bounding box
- Press 's' to save
- Press 'n' for next image
- Press 'q' to quit

**Step 3: Train Model**
```bash
cd backend
venv/bin/python train_tensorflow.py
```

**Expected**:
- Training: 30-50 epochs (~2-5 minutes)
- Output: `cardio_chek_tf_detector.keras` (300MB)
- Validation accuracy: >85%

---

### **Option 2: Use Synthetic Data (Quick Testing)**

**Generate Synthetic Images**:
```bash
venv/bin/python scripts/generate_cardio_chek_images.py
```

**Train with Synthetic Data**:
```bash
cd backend
venv/bin/python train_tensorflow.py
```

**Note**: Synthetic data works for testing but real images give better accuracy.

---

## ▶️ **Running the Application**

### **Option 1: Automated (Recommended)**
```bash
./start.sh
```

### **Option 2: Manual**

**Terminal 1 - Backend**:
```bash
cd backend
venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
```

**Browser**:
```
http://localhost:3000
```

---

## 🛑 **Stopping the Application**

```bash
./stop.sh
```

Or manually:
```bash
pkill -f uvicorn
pkill -f "next dev"
```

---

## 📦 **What Gets Generated Locally**

After setup and training:
```
CVML/
├── backend/
│   ├── venv/ (Virtual environment - ~500MB)
│   ├── cardio_chek_tf_detector.keras (Trained model - 300MB)
│   └── logs/ (Runtime logs)
│
├── frontend/
│   ├── node_modules/ (Dependencies - ~300MB)
│   └── .next/ (Build cache)
│
├── data/cardio_chek/
│   ├── images/ (Your training images)
│   └── annotations/ (Annotation files)
│
└── cardio_chek_models/ (YOLO models if using legacy system)
```

**Total disk space needed**: ~1.5GB

---

## ✅ **Verification**

After setup, verify everything works:

```bash
# Test backend
curl http://localhost:8000/health | jq

# Expected: "status": "healthy"

# Test with sample image (after generating synthetic data)
curl -X POST "http://localhost:8000/api/scan-kit" \
  -F "file=@data/cardio_chek/images/real_data/cardio_chek_synthetic_001.jpg" | jq
```

---

## 📚 **Documentation**

- **START_HERE.txt** - Quick start guide
- **QUICKSTART.md** - Essential commands
- **README.md** - Project overview
- **RUN_APPLICATION_GUIDE.md** - Complete guide (500+ lines)
- **TODO_IMPLEMENTATION_GUIDE.md** - Future improvements

---

## 🆘 **Troubleshooting**

### Python dependencies fail
```bash
# Try Python 3.13 specific requirements
pip install -r requirements-py313.txt
```

### Model not found error
```bash
# Generate synthetic data and train
venv/bin/python scripts/generate_cardio_chek_images.py
cd backend && venv/bin/python train_tensorflow.py
```

### Port already in use
```bash
# Change ports in start command
# Backend: --port 8001
# Frontend: PORT=3001 npm run dev
```

---

## 🎯 **System Requirements**

- **Python**: 3.13+ (3.9+ may work)
- **Node.js**: 18+
- **RAM**: 8GB minimum
- **Disk**: 2GB free space
- **OS**: macOS, Linux, or Windows
- **Camera**: For real-time detection

---

**After setup, see START_HERE.txt for running instructions!**
