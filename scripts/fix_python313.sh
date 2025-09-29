#!/bin/bash

# CVML Python 3.13 Compatibility Fix Script
# This script fixes common Python 3.13 compatibility issues

set -e

echo "🔧 Fixing Python 3.13 compatibility issues..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    print_error "Please run this script from the CVML root directory"
    exit 1
fi

cd backend

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
else
    print_status "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade essential tools
print_status "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install Pillow with specific flags for Python 3.13
print_status "Installing Pillow with Python 3.13 compatibility..."
pip install --no-cache-dir --force-reinstall pillow

# Install other dependencies one by one to avoid conflicts
print_status "Installing core dependencies..."

# Install numpy first
pip install "numpy>=1.24.0,<2.0.0"

# Install OpenCV
pip install opencv-python-headless

# Install FastAPI and related
pip install fastapi uvicorn python-multipart

# Install ML libraries
pip install tensorflow torch torchvision

# Install other dependencies
pip install ultralytics scikit-learn python-jose[cryptography] passlib[bcrypt] python-dotenv aiofiles httpx pydantic

print_success "Python 3.13 compatibility fix completed!"

# Test the installation
print_status "Testing installation..."
python -c "
import fastapi
import cv2
import numpy as np
import tensorflow as tf
import torch
print('✅ All core dependencies imported successfully!')
"

print_success "🎉 CVML backend is now ready for Python 3.13!"
print_status "You can now run: ./start_backend.sh"
