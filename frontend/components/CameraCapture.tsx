'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import { Camera, ArrowLeft, RotateCcw, Check, AlertCircle, Upload } from 'lucide-react'
import { analyzeImage } from '@/lib/api'

// Simple logger for camera debugging
const logger = {
  info: (msg: string) => console.log(`[Camera] ${msg}`),
  error: (msg: string, err?: any) => console.error(`[Camera] ${msg}`, err),
  warn: (msg: string) => console.warn(`[Camera] ${msg}`)
}

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
  const [autoCaptureEnabled, setAutoCaptureEnabled] = useState(true)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5) // 50% confidence threshold for faster auto-capture
  const [currentConfidence, setCurrentConfidence] = useState(0)
  const [detectionInterval, setDetectionInterval] = useState<NodeJS.Timeout | null>(null)
  const [detectionHistory, setDetectionHistory] = useState<number[]>([])
  const [adaptiveThreshold, setAdaptiveThreshold] = useState(0.5) // Start with 50% for faster initial detection
  const [thresholdMode, setThresholdMode] = useState<'fixed' | 'adaptive'>('adaptive')
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (detectionInterval) {
        clearInterval(detectionInterval)
      }
    }
  }, [detectionInterval])

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
          width: { ideal: 640, min: 320 },  // More flexible constraints
          height: { ideal: 480, min: 240 }
        }
      })
      
      setStream(mediaStream)
      setCameraStarted(true)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        
        // Wait for video to load and ensure it plays
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play().then(() => {
              logger.info('Camera video started successfully')
              // Start continuous real-time detection
              startContinuousDetection()
            }).catch((playError) => {
              logger.error('Failed to play camera video:', playError)
              setError('Camera started but video playback failed. Please check your browser settings.')
            })
          }
        }

        // Add error handling for video loading
        videoRef.current.onerror = () => {
          logger.error('Video element error occurred')
          setError('Camera video failed to load. Please refresh and try again.')
        }
      }
    } catch (err: any) {
      logger.error('Error accessing camera:', err)
      let errorMessage = 'Unable to access camera. '

      if (err.name === 'NotAllowedError') {
        errorMessage += 'Please allow camera permissions in your browser settings and refresh the page.'
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No camera found on this device. Please connect a camera and try again.'
      } else if (err.name === 'NotSupportedError') {
        errorMessage += 'Camera not supported on this device or browser.'
      } else if (err.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application. Please close other apps and try again.'
      } else if (err.name === 'OverconstrainedError') {
        errorMessage += 'Camera constraints not supported. Trying with basic settings...'
        // Retry with basic constraints
        setTimeout(() => {
          startCameraBasic()
        }, 1000)
        return
      } else if (err.name === 'SecurityError') {
        errorMessage += 'Camera access blocked for security reasons. Please check your browser settings.'
      } else {
        errorMessage += `Error: ${err.message || 'Unknown camera error'}. Please refresh and try again.`
      }

      setError(errorMessage)
    }
  }, [])

  // Fallback camera start with basic constraints
  const startCameraBasic = useCallback(async () => {
    try {
      logger.info('Attempting basic camera constraints...')

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true // Most basic constraint possible
      })

      setStream(mediaStream)
      setCameraStarted(true)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream

        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            videoRef.current.play().then(() => {
              logger.info('Camera video started successfully with basic constraints')
              startContinuousDetection()
            }).catch((playError) => {
              logger.error('Failed to play camera video with basic constraints:', playError)
              setError('Camera started but video playback failed. Please check your browser settings.')
            })
          }
        }
      }
    } catch (err: any) {
      logger.error('Basic camera constraints also failed:', err)
      setError('Unable to access camera with any settings. Please check camera permissions and refresh the page.')
    }
  }, [])

  const stopCamera = useCallback(() => {
    // Stop continuous detection
    if (detectionInterval) {
      clearInterval(detectionInterval)
      setDetectionInterval(null)
    }
    
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    setCameraStarted(false)
    setDetectionBox(null)
    setCurrentConfidence(0)
  }, [stream, detectionInterval])

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

  const calculateAdaptiveThreshold = useCallback((history: number[]) => {
    if (history.length < 3) return 0.7 // Default threshold for insufficient data
    
    // Sort history to find patterns
    const sortedHistory = [...history].sort((a, b) => a - b)
    const median = sortedHistory[Math.floor(sortedHistory.length / 2)]
    const mean = history.reduce((sum, val) => sum + val, 0) / history.length
    
    // Calculate standard deviation
    const variance = history.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / history.length
    const stdDev = Math.sqrt(variance)
    
    // Adaptive threshold based on:
    // 1. Median confidence (more stable than mean)
    // 2. Standard deviation (lower std = more consistent = lower threshold)
    // 3. Recent trend (last few detections)
    const recentDetections = history.slice(-3)
    const recentMean = recentDetections.reduce((sum, val) => sum + val, 0) / recentDetections.length
    
    // Base threshold on median, adjusted by consistency
    let adaptiveThreshold = median * 0.85 // Start at 85% of median
    
    // If very consistent (low std dev), lower threshold
    if (stdDev < 0.1) {
      adaptiveThreshold = Math.max(0.5, median * 0.8)
    }
    
    // If recent detections are higher, be more aggressive
    if (recentMean > median) {
      adaptiveThreshold = Math.max(0.5, recentMean * 0.8)
    }
    
    // If recent detections are lower, be more conservative
    if (recentMean < median * 0.9) {
      adaptiveThreshold = Math.min(0.9, median * 0.9)
    }
    
    // Ensure threshold is within reasonable bounds
    return Math.max(0.5, Math.min(0.95, adaptiveThreshold))
  }, [])

  const updateDetectionHistory = useCallback((confidence: number) => {
    setDetectionHistory(prev => {
      const newHistory = [...prev, confidence].slice(-10) // Keep last 10 detections
      const newAdaptiveThreshold = calculateAdaptiveThreshold(newHistory)
      setAdaptiveThreshold(newAdaptiveThreshold)
      return newHistory
    })
  }, [calculateAdaptiveThreshold])

  const performRealTimeDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return

    try {
      setIsDetecting(true)

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

      // Convert to blob with optimized quality for faster processing
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob)
        }, 'image/jpeg', 0.7)  // Reduced quality for faster processing
      })

      // Quick detection check with timeout to prevent hanging
      const detectionPromise = analyzeImage(blob)
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Detection timeout')), 3000) // Increased to 3s for reliability
      )

      const result = await Promise.race([detectionPromise, timeoutPromise]) as any

      if (result.result === 'No CardioChek Plus detected') {
        setDetectionStatus('CardioChek Plus not detected. Please position the device clearly in view.')
        setDetectionBox(null)
        setCurrentConfidence(0)
      } else {
        const confidence = result.confidence || 0
        setCurrentConfidence(confidence)

        // Update detection history for adaptive threshold
        updateDetectionHistory(confidence)

        // Use adaptive threshold if enabled, otherwise use fixed threshold
        const currentThreshold = thresholdMode === 'adaptive' ? adaptiveThreshold : confidenceThreshold

        setDetectionStatus(`CardioChek Plus detected! Confidence: ${(confidence * 100).toFixed(1)}% (Threshold: ${(currentThreshold * 100).toFixed(1)}%)`)

        if (result.bounding_box) {
          setDetectionBox(result.bounding_box)
        }

        // Auto-capture if confidence is above current threshold
        if (autoCaptureEnabled && confidence >= currentThreshold) {
          // More lenient stability check for faster auto-capture
          const recentConfidences = detectionHistory.slice(-2) // Reduced from 3 to 2
          const isStable = recentConfidences.length >= 1 &&
            recentConfidences.every(c => c >= currentThreshold * 0.8) // Reduced from 0.9 to 0.8

          if (isStable) {
            setDetectionStatus(`🎯 Auto-capturing... (${(confidence * 100).toFixed(1)}% ≥ ${(currentThreshold * 100).toFixed(1)}%)`)
            // Minimal delay for instant capture
            setTimeout(() => {
              if (videoRef.current && !isCapturing) {
                captureImage()
              }
            }, 150) // Reduced from 300ms to 150ms
          } else {
            setDetectionStatus(`✅ Device detected! Confidence: ${(confidence * 100).toFixed(1)}% (Auto-capture ready)`)
          }
        }
      }

    } catch (err) {
      console.error('Real-time detection error:', err)
      setDetectionStatus('Detection failed. Please try again.')

      // Reset confidence on error
      setCurrentConfidence(0)
      setDetectionBox(null)
    } finally {
      setIsDetecting(false)
    }
  }, [autoCaptureEnabled, confidenceThreshold, adaptiveThreshold, thresholdMode, updateDetectionHistory, detectionHistory])

  const startContinuousDetection = useCallback(() => {
    if (detectionInterval) {
      clearInterval(detectionInterval)
    }

    // Run detection every 500ms for ultra-responsive auto-capture
    const interval = setInterval(() => {
      if (videoRef.current && canvasRef.current && !isDetecting && !isCapturing && cameraStarted) {
        performRealTimeDetection()
      }
    }, 500)  // Ultra-fast interval for instant auto-capture

    setDetectionInterval(interval)
    setDetectionStatus('🚀 Auto-capture active! Position the CardioChek Plus device in view...')
  }, [detectionInterval, isDetecting, isCapturing, performRealTimeDetection, cameraStarted])

  const stopContinuousDetection = useCallback(() => {
    if (detectionInterval) {
      clearInterval(detectionInterval)
      setDetectionInterval(null)
    }
    setDetectionStatus('Real-time detection stopped.')
  }, [detectionInterval])

  const resetAdaptiveThreshold = useCallback(() => {
    setDetectionHistory([])
    setAdaptiveThreshold(0.7)
    setDetectionStatus('Adaptive threshold reset to default.')
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

  // Auto-start camera when component mounts (immediate start)
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

      {/* Auto-Capture Controls */}
      {cameraStarted && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-700">Auto-Capture Settings</h3>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={autoCaptureEnabled}
                onChange={(e) => setAutoCaptureEnabled(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-gray-600">Enable Auto-Capture</span>
            </label>
          </div>
          
          {/* Threshold Mode Selection */}
          <div className="mb-4">
            <div className="flex items-center space-x-4 mb-2">
              <label className="text-sm text-gray-600">Threshold Mode:</label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="thresholdMode"
                  value="adaptive"
                  checked={thresholdMode === 'adaptive'}
                  onChange={(e) => setThresholdMode(e.target.value as 'adaptive' | 'fixed')}
                  className="mr-1"
                />
                <span className="text-sm text-gray-600">Adaptive (Recommended)</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="thresholdMode"
                  value="fixed"
                  checked={thresholdMode === 'fixed'}
                  onChange={(e) => setThresholdMode(e.target.value as 'adaptive' | 'fixed')}
                  className="mr-1"
                />
                <span className="text-sm text-gray-600">Fixed</span>
              </label>
            </div>
          </div>
          
          {/* Fixed Threshold Controls */}
          {thresholdMode === 'fixed' && (
            <div className="flex items-center space-x-4 mb-4">
              <label className="text-sm text-gray-600">
                Fixed Threshold: {(confidenceThreshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="0.95"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="flex-1"
              />
            </div>
          )}
          
          {/* Adaptive Threshold Display */}
          {thresholdMode === 'adaptive' && (
            <div className="mb-4 p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-blue-800">Adaptive Threshold</span>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-blue-600">
                    {(adaptiveThreshold * 100).toFixed(1)}%
                  </span>
                  {detectionHistory.length > 0 && (
                    <button
                      onClick={resetAdaptiveThreshold}
                      className="text-xs text-blue-600 hover:text-blue-800 underline"
                    >
                      Reset
                    </button>
                  )}
                </div>
              </div>
              <div className="text-xs text-blue-600">
                Based on {detectionHistory.length} recent detections
                {detectionHistory.length > 0 && (
                  <span className="ml-2">
                    (Range: {(Math.min(...detectionHistory) * 100).toFixed(1)}% - {(Math.max(...detectionHistory) * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
            </div>
          )}
          
          {/* Current Status */}
          {currentConfidence > 0 && (
            <div className="mt-2 text-sm text-gray-600">
              <div className="flex items-center justify-between">
                <span>
                  Current Confidence: <span className={`font-medium ${
                    currentConfidence >= (thresholdMode === 'adaptive' ? adaptiveThreshold : confidenceThreshold) 
                      ? 'text-green-600' : 'text-yellow-600'
                  }`}>
                    {(currentConfidence * 100).toFixed(1)}%
                  </span>
                </span>
                <span className="text-xs text-gray-500">
                  vs {(thresholdMode === 'adaptive' ? adaptiveThreshold : confidenceThreshold) * 100}% threshold
                </span>
              </div>
              {currentConfidence >= (thresholdMode === 'adaptive' ? adaptiveThreshold : confidenceThreshold) && autoCaptureEnabled && (
                <div className="mt-1 text-green-600 font-medium text-sm">
                  ✓ Auto-capture ready!
                </div>
              )}
            </div>
          )}
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
                {error && (
                  <button
                    onClick={startCamera}
                    className="btn-warning flex items-center"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Retry Camera
                  </button>
                )}

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
                  <Camera className="w-4 h-4 mr-2" />
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

            {/* Auto-Detection Controls */}
            <div className="flex justify-center space-x-4 mt-4">
              {detectionInterval ? (
                <button
                  onClick={stopContinuousDetection}
                  className="btn-warning flex items-center"
                >
                  <AlertCircle className="w-4 h-4 mr-2" />
                  Stop Auto-Detection
                </button>
              ) : (
                <button
                  onClick={startContinuousDetection}
                  className="btn-success flex items-center"
                  disabled={isDetecting || isLoading}
                >
                  <Check className="w-4 h-4 mr-2" />
                  Start Auto-Detection
                </button>
              )}

              <button
                onClick={captureImage}
                className="btn-primary flex items-center"
                disabled={isCapturing || isLoading}
              >
                <Camera className="w-4 h-4 mr-2" />
                {isCapturing ? 'Capturing...' : 'Capture Image'}
              </button>
              <button
                onClick={stopCamera}
                className="btn-secondary"
              >
                Stop Camera
              </button>
            </div>
          </div>

          {/* Camera Loading State */}
          {stream && !cameraStarted && (
            <div className="camera-container flex items-center justify-center bg-gray-900 rounded-lg" style={{ height: '384px' }}>
              <div className="text-center text-white">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                <p className="text-lg font-medium">Starting camera...</p>
                <p className="text-sm opacity-75">Please allow camera permissions if prompted</p>
              </div>
            </div>
          )}

          {/* Video Stream */}
          {stream && cameraStarted && (
            <div className="camera-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-auto max-h-96 object-cover border-2 border-gray-300 rounded-lg"
                style={{
                  transform: 'scaleX(-1)', // Mirror the video for better UX
                  backgroundColor: '#000' // Black background while loading
                }}
                onLoadedData={() => logger.info('Video data loaded')}
                onCanPlay={() => logger.info('Video can play')}
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
