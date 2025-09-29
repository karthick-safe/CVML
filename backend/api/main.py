"""
CVML Cardio Health Check Kit Analyzer - FastAPI Backend
Main API application with endpoints for image processing and kit analysis
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, Any
import io
from PIL import Image
import numpy as np

from services.image_processor import ImageProcessor
from services.real_model_manager import RealModelManager
from api.schemas import ScanResult, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CVML Cardio Health Check Kit Analyzer",
    description="AI-powered analysis of cardio health check kits using computer vision and machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_manager = RealModelManager()
image_processor = ImageProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        await model_manager.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        message="CVML Cardio Health Check Kit Analyzer API",
        status="healthy",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check with model status"""
    model_status = await model_manager.get_model_status()
    return HealthResponse(
        message="API is healthy",
        status="healthy" if model_status["all_loaded"] else "degraded",
        version="1.0.0",
        details=model_status
    )

@app.post("/api/scan-kit", response_model=ScanResult)
async def scan_kit(file: UploadFile = File(...)):
    """
    Main endpoint for kit analysis
    
    Args:
        file: Image file of the cardio health check kit
        
    Returns:
        ScanResult: Analysis results with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Step 1: Detect CardioChek Plus kit using real CVML
        kit_detection = await model_manager.detect_cardio_chek_kit(image_array)
        
        if not kit_detection["detected"]:
            return ScanResult(
                result="No CardioChek Plus detected",
                confidence=0.0,
                bounding_box=None,
                processing_time=kit_detection.get("processing_time", 0),
                error="No CardioChek Plus device detected in the image. Please ensure the device is clearly visible."
            )
        
        # Step 2: Extract kit region for OCR processing
        bbox = kit_detection["bounding_box"]
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"])
        kit_region = image_array[y:y+h, x:x+w]
        
        # Step 3: Extract screen values using OCR
        ocr_result = await model_manager.extract_screen_values(kit_region)
        
        if not ocr_result["success"]:
            return ScanResult(
                result="OCR Failed",
                confidence=0.0,
                bounding_box=bbox,
                processing_time=kit_detection.get("processing_time", 0) + ocr_result.get("processing_time", 0),
                error="Failed to read CardioChek Plus screen values. Please ensure the screen is clearly visible."
            )
        
        # Step 4: Classify result based on extracted values
        classification_result = await model_manager.classify_cardio_chek_result(ocr_result["values"])
        
        # Step 5: Combine results
        final_result = ScanResult(
            result=classification_result["result"],
            confidence=classification_result["confidence"],
            bounding_box=kit_detection["bounding_box"],
            processing_time=kit_detection.get("processing_time", 0) + ocr_result.get("processing_time", 0) + classification_result.get("processing_time", 0),
            details={
                "kit_detection_confidence": kit_detection["confidence"],
                "ocr_values": ocr_result["values"],
                "result_classification": classification_result,
                "image_dimensions": image.size,
                "extracted_values": classification_result.get("values", {}),
                "analysis": classification_result.get("analysis", [])
            }
        )
        
        logger.info(f"Analysis complete: {final_result.result} (confidence: {final_result.confidence})")
        return final_result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/validate-image")
async def validate_image(file: UploadFile = File(...)):
    """
    Validate if image is suitable for analysis
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Basic validation checks
        width, height = image.size
        min_size = 224  # Minimum size for model input
        
        if width < min_size or height < min_size:
            return JSONResponse({
                "valid": False,
                "reason": f"Image too small. Minimum size: {min_size}x{min_size}",
                "current_size": f"{width}x{height}"
            })
        
        return JSONResponse({
            "valid": True,
            "size": f"{width}x{height}",
            "format": image.format
        })
        
    except Exception as e:
        return JSONResponse({
            "valid": False,
            "reason": f"Invalid image file: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
