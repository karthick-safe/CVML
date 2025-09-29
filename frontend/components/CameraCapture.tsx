'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Camera, ArrowLeft, RotateCcw, Check, AlertCircle, Upload } from 'lucide-react'
import { analyzeImage } from '@/lib/api'

interface CameraCaptureProps {
  onAnalysisComplete: (result: any) => void
  onBack: () => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function CameraCapture({ onAnalysisComplete, onBack, isLoading, setIsLoading }: CameraCaptureProps) {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [cameraStarted, setCameraStarted] = useState(false)
  const [detectionStatus, setDetectionStatus] = useState<string>('')
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionBox, setDetectionBox] = useState<{x: number, y: number, width: number, height: number} | null>(null)
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const startCamera = useCallback(async () => {
    try {
      setError(null)
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported on this device')
      }
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 }
        }
      })
      
      setStream(mediaStream)
      setCameraStarted(true)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        
        // Wait for video to load
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play().catch(console.error)
          }
        }
      }
    } catch (err: any) {
      console.error('Error accessing camera:', err)
      let errorMessage = 'Unable to access camera. '
      
      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please allow camera permissions and try again.'
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No camera found on this device.'
      } else if (err.name === 'NotSupportedError') {
        errorMessage += 'Camera not supported on this device.'
      } else {
        errorMessage += 'Please check permissions and try again.'
      }
      
      setError(errorMessage)
    }
  }, [])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
  }, [stream])

  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return

    setIsCapturing(true)
    
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')

    if (!context) return

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    // Convert to data URL
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8)
    setCapturedImage(imageDataUrl)
    setIsCapturing(false)
  }, [])

  const retakeImage = useCallback(() => {
    setCapturedImage(null)
    setError(null)
  }, [])

  const performRealTimeDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    try {
      setIsDetecting(true)
      setDetectionStatus('Detecting CardioChek Plus...')

      // Capture current frame
      const canvas = canvasRef.current
      const video = videoRef.current
      const context = canvas.getContext('2d')
      
      if (!context) return

      // Set canvas size to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Draw current video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob)
        }, 'image/jpeg', 0.8)
      })

      // Quick detection check
      const result = await analyzeImage(blob)
      
      if (result.result === 'No CardioChek Plus detected') {
        setDetectionStatus('CardioChek Plus not detected. Please position the device clearly in view.')
        setDetectionBox(null)
      } else {
        setDetectionStatus(`CardioChek Plus detected! Confidence: ${(result.confidence * 100).toFixed(1)}%`)
        if (result.bounding_box) {
          setDetectionBox(result.bounding_box)
        }
      }

    } catch (err) {
      console.error('Real-time detection error:', err)
      setDetectionStatus('Detection failed. Please try again.')
    } finally {
      setIsDetecting(false)
    }
  }, [])

  const analyzeCapturedImage = useCallback(async () => {
    if (!capturedImage) return

    try {
      setIsLoading(true)
      setError(null)

      // Convert data URL to blob
      const response = await fetch(capturedImage)
      const blob = await response.blob()

      // Analyze image
      const result = await analyzeImage(blob)
      onAnalysisComplete(result)
    } catch (err) {
      console.error('Error analyzing image:', err)
      setError('Failed to analyze image. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [capturedImage, onAnalysisComplete])

  const startNewCapture = useCallback(() => {
    setCapturedImage(null)
    setError(null)
    if (!stream) {
      startCamera()
    }
  }, [stream, startCamera])

  // Auto-start camera when component mounts
  useEffect(() => {
    if (!cameraStarted && !stream) {
      startCamera()
    }
  }, [cameraStarted, stream, startCamera])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [stream])

  // Handle file upload
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        setCapturedImage(result)
        setError(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const triggerFileUpload = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <button
          onClick={onBack}
          className="flex items-center text-gray-600 hover:text-gray-900"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Home
        </button>
        <h1 className="text-2xl font-bold text-gray-900">Capture Kit Image</h1>
        <div className="w-20"></div> {/* Spacer for centering */}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center">
          <AlertCircle className="w-5 h-5 text-danger-600 mr-3" />
          <span className="text-danger-800">{error}</span>
        </div>
      )}

      {/* Detection Status */}
      {detectionStatus && (
        <div className={`mb-6 p-4 rounded-lg ${
          detectionStatus.includes('detected!') 
            ? 'bg-green-100 border border-green-300 text-green-800'
            : detectionStatus.includes('not detected')
            ? 'bg-yellow-100 border border-yellow-300 text-yellow-800'
            : 'bg-blue-100 border border-blue-300 text-blue-800'
        }`}>
          <div className="flex items-center">
            {isDetecting ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
            ) : detectionStatus.includes('detected!') ? (
              <Check className="w-4 h-4 mr-2" />
            ) : (
              <AlertCircle className="w-4 h-4 mr-2" />
            )}
            <span className="text-sm font-medium">{detectionStatus}</span>
          </div>
        </div>
      )}

      {/* Camera Interface */}
      {!capturedImage && (
        <div className="space-y-6">
          {/* Camera Controls */}
          <div className="flex justify-center space-x-4">
            {!stream ? (
              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={startCamera}
                  className="btn-primary"
                  disabled={isLoading}
                >
                  <Camera className="w-5 h-5 mr-2" />
                  Start Camera
                </button>
                <button
                  onClick={triggerFileUpload}
                  className="btn-secondary"
                  disabled={isLoading}
                >
                  <Upload className="w-5 h-5 mr-2" />
                  Upload Image
                </button>
              </div>
            ) : (
              <div className="flex flex-col sm:flex-row gap-4">
                <button
                  onClick={performRealTimeDetection}
                  className="btn-secondary flex items-center"
                  disabled={isDetecting || isLoading}
                >
                  {isDetecting ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current mr-2"></div>
                  ) : (
                    <Check className="w-4 h-4 mr-2" />
                  )}
                  {isDetecting ? 'Detecting...' : 'Check Detection'}
                </button>
                <button
                  onClick={captureImage}
                  className="btn-primary flex items-center"
                  disabled={isCapturing || isLoading}
                >
                  <Camera className="w-5 h-5 mr-2" />
                  {isCapturing ? 'Capturing...' : 'Capture Image'}
                </button>
                <button
                  onClick={stopCamera}
                  className="btn-secondary"
                >
                  Stop Camera
                </button>
              </div>
            )}
          </div>

          {/* Video Stream */}
          {stream && (
            <div className="camera-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-auto max-h-96 object-cover"
                style={{ transform: 'scaleX(-1)' }} // Mirror the video for better UX
              />
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="border-2 border-white border-dashed rounded-lg w-80 h-48 opacity-50"></div>
              </div>
              {/* Detection Bounding Box Overlay */}
              {detectionBox && videoRef.current && (
                <div 
                  className="absolute border-2 border-green-500 bg-green-500 bg-opacity-20 pointer-events-none"
                  style={{
                    left: `${(detectionBox.x / videoRef.current.videoWidth) * 100}%`,
                    top: `${(detectionBox.y / videoRef.current.videoHeight) * 100}%`,
                    width: `${(detectionBox.width / videoRef.current.videoWidth) * 100}%`,
                    height: `${(detectionBox.height / videoRef.current.videoHeight) * 100}%`,
                  }}
                >
                  <div className="absolute -top-6 left-0 bg-green-500 text-white text-xs px-2 py-1 rounded">
                    CardioChek Plus Detected
                  </div>
                </div>
              )}
              {/* Hidden canvas for real-time detection */}
              <canvas ref={canvasRef} className="hidden" />
            </div>
          )}

          {/* Capture Button */}
          {stream && (
            <div className="flex justify-center">
              <button
                onClick={captureImage}
                disabled={isCapturing}
                className="btn-primary text-lg px-8 py-4 shadow-lg"
              >
                <Camera className="w-6 h-6 mr-2" />
                {isCapturing ? 'Capturing...' : 'Capture Image'}
              </button>
            </div>
          )}

          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-2">Instructions</h3>
            <ul className="text-blue-800 space-y-2">
              <li>• Ensure good lighting for clear image quality</li>
              <li>• Position the kit in the center of the frame</li>
              <li>• Keep the camera steady and avoid blur</li>
              <li>• Make sure the test result area is clearly visible</li>
              <li>• If camera doesn't work, use "Upload Image" to select a photo</li>
            </ul>
          </div>
        </div>
      )}

      {/* Captured Image Preview */}
      {capturedImage && (
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Captured Image</h2>
            <div className="relative inline-block">
              <img
                src={capturedImage}
                alt="Captured kit image"
                className="max-w-full h-auto rounded-lg shadow-lg"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-center space-x-4">
            <button
              onClick={retakeImage}
              className="btn-secondary"
              disabled={isLoading}
            >
              <RotateCcw className="w-5 h-5 mr-2" />
              Retake
            </button>
            <button
              onClick={analyzeCapturedImage}
              disabled={isLoading}
              className="btn-primary text-lg px-8 py-4 shadow-lg"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Check className="w-6 h-6 mr-2" />
                  Analyze Image
                </>
              )}
            </button>
          </div>

          {/* Analysis Status */}
          {isLoading && (
            <div className="text-center">
              <div className="inline-flex items-center px-4 py-2 bg-primary-100 text-primary-800 rounded-lg">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600 mr-2"></div>
                AI is analyzing your image...
              </div>
            </div>
          )}
        </div>
      )}

      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  )
}
