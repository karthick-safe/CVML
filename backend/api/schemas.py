"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class ResultType(str, Enum):
    """Possible test result types"""
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    INVALID = "Invalid"
    NO_KIT = "No kit detected"

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of the bounding box")
    height: float = Field(..., description="Height of the bounding box")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")

class ScanResult(BaseModel):
    """Main result schema for kit analysis"""
    result: str = Field(..., description="Final test result")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence score")
    bounding_box: Optional[BoundingBox] = Field(None, description="Detected kit location")
    processing_time: float = Field(..., description="Total processing time in seconds")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional analysis details")
    error: Optional[str] = Field(None, description="Error message if analysis failed")

class HealthResponse(BaseModel):
    """Health check response"""
    message: str
    status: str
    version: str
    details: Optional[Dict[str, Any]] = None

class ModelStatus(BaseModel):
    """Model loading status"""
    object_detection: bool = Field(..., description="Object detection model status")
    classification: bool = Field(..., description="Classification model status")
    all_loaded: bool = Field(..., description="All models loaded successfully")

class ClassificationResult(BaseModel):
    """Result from classification model"""
    result: str
    confidence: float
    processing_time: float
    probabilities: Optional[Dict[str, float]] = None

class KitDetectionResult(BaseModel):
    """Result from kit detection model"""
    detected: bool
    confidence: float
    bounding_box: Optional[BoundingBox]
    processing_time: float
