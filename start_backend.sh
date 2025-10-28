#!/bin/bash
echo "🚀 Starting CVML Backend..."
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
