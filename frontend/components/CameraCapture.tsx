'use client'

import { useState, useRef, useCallback } from 'react'
import { Camera, ArrowLeft, RotateCcw, Check, AlertCircle } from 'lucide-react'
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
  
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const startCamera = useCallback(async () => {
    try {
      setError(null)
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      
      setStream(mediaStream)
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
      }
    } catch (err) {
      console.error('Error accessing camera:', err)
      setError('Unable to access camera. Please check permissions.')
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

      {/* Camera Interface */}
      {!capturedImage && (
        <div className="space-y-6">
          {/* Camera Controls */}
          <div className="flex justify-center space-x-4">
            {!stream ? (
              <button
                onClick={startCamera}
                className="btn-primary"
                disabled={isLoading}
              >
                <Camera className="w-5 h-5 mr-2" />
                Start Camera
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="btn-secondary"
              >
                Stop Camera
              </button>
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
                className="w-full h-auto"
              />
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="border-2 border-white border-dashed rounded-lg w-80 h-48 opacity-50"></div>
              </div>
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
    </div>
  )
}
