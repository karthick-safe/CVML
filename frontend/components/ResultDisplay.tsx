'use client'

import { useState } from 'react'
import { ArrowLeft, Camera, Download, Share2, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react'

interface ResultDisplayProps {
  result: any
  onBack: () => void
  onNewScan: () => void
}

export default function ResultDisplay({ result, onBack, onNewScan }: ResultDisplayProps) {
  const [showDetails, setShowDetails] = useState(false)

  const getResultIcon = (resultType: string) => {
    switch (resultType.toLowerCase()) {
      case 'positive':
        return <XCircle className="w-8 h-8 text-danger-600" />
      case 'negative':
        return <CheckCircle className="w-8 h-8 text-success-600" />
      case 'invalid':
        return <AlertTriangle className="w-8 h-8 text-warning-600" />
      default:
        return <Info className="w-8 h-8 text-gray-600" />
    }
  }

  const getResultBadge = (resultType: string) => {
    switch (resultType.toLowerCase()) {
      case 'positive':
        return 'result-positive'
      case 'negative':
        return 'result-negative'
      case 'invalid':
        return 'result-invalid'
      default:
        return 'result-error'
    }
  }

  const getResultColor = (resultType: string) => {
    switch (resultType.toLowerCase()) {
      case 'positive':
        return 'danger'
      case 'negative':
        return 'success'
      case 'invalid':
        return 'warning'
      default:
        return 'gray'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-success-600'
    if (confidence >= 0.7) return 'text-warning-600'
    return 'text-danger-600'
  }

  const formatConfidence = (confidence: number) => {
    return `${Math.round(confidence * 100)}%`
  }

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
        <h1 className="text-2xl font-bold text-gray-900">Analysis Results</h1>
        <div className="w-20"></div> {/* Spacer for centering */}
      </div>

      {/* Main Result Card */}
      <div className="card p-8 mb-8">
        <div className="text-center">
          {/* Result Icon */}
          <div className="flex justify-center mb-6">
            {getResultIcon(result.result)}
          </div>

          {/* Result Text */}
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            {result.result}
          </h2>
          
          {/* Confidence Score */}
          <div className="mb-6">
            <span className={`text-2xl font-semibold ${getConfidenceColor(result.confidence)}`}>
              {formatConfidence(result.confidence)} Confidence
            </span>
          </div>

          {/* Result Badge */}
          <div className="mb-8">
            <span className={getResultBadge(result.result)}>
              {result.result} Result
            </span>
          </div>

          {/* Processing Time */}
          <div className="text-sm text-gray-500 mb-6">
            Analysis completed in {result.processing_time?.toFixed(2)}s
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={onNewScan}
              className="btn-primary text-lg px-8 py-4"
            >
              <Camera className="w-6 h-6 mr-2" />
              Scan Another Kit
            </button>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="btn-secondary text-lg px-8 py-4"
            >
              <Info className="w-6 h-6 mr-2" />
              {showDetails ? 'Hide' : 'Show'} Details
            </button>
          </div>
        </div>
      </div>

      {/* Detailed Results */}
      {showDetails && (
        <div className="space-y-6">
          {/* Technical Details */}
          <div className="card p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Technical Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <span className="text-sm font-medium text-gray-500">Result Type:</span>
                <p className="text-lg font-semibold text-gray-900">{result.result}</p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-500">Confidence Score:</span>
                <p className={`text-lg font-semibold ${getConfidenceColor(result.confidence)}`}>
                  {formatConfidence(result.confidence)}
                </p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-500">Processing Time:</span>
                <p className="text-lg font-semibold text-gray-900">
                  {result.processing_time?.toFixed(2)}s
                </p>
              </div>
              <div>
                <span className="text-sm font-medium text-gray-500">Analysis Status:</span>
                <p className="text-lg font-semibold text-success-600">Completed</p>
              </div>
            </div>
          </div>

          {/* Bounding Box Information */}
          {result.bounding_box && (
            <div className="card p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Kit Detection</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <span className="text-sm font-medium text-gray-500">Kit Location:</span>
                  <p className="text-lg font-semibold text-gray-900">
                    X: {Math.round(result.bounding_box.x)}, Y: {Math.round(result.bounding_box.y)}
                  </p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Kit Size:</span>
                  <p className="text-lg font-semibold text-gray-900">
                    {Math.round(result.bounding_box.width)} × {Math.round(result.bounding_box.height)} px
                  </p>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-500">Detection Confidence:</span>
                  <p className={`text-lg font-semibold ${getConfidenceColor(result.bounding_box.confidence)}`}>
                    {formatConfidence(result.bounding_box.confidence)}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Additional Details */}
          {result.details && (
            <div className="card p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Additional Information</h3>
              <div className="space-y-3">
                {result.details.kit_detection_confidence && (
                  <div>
                    <span className="text-sm font-medium text-gray-500">Kit Detection Confidence:</span>
                    <p className={`text-lg font-semibold ${getConfidenceColor(result.details.kit_detection_confidence)}`}>
                      {formatConfidence(result.details.kit_detection_confidence)}
                    </p>
                  </div>
                )}
                {result.details.image_dimensions && (
                  <div>
                    <span className="text-sm font-medium text-gray-500">Image Dimensions:</span>
                    <p className="text-lg font-semibold text-gray-900">
                      {result.details.image_dimensions[0]} × {result.details.image_dimensions[1]} px
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Information */}
          {result.error && (
            <div className="card p-6 border-danger-200 bg-danger-50">
              <h3 className="text-xl font-semibold text-danger-900 mb-2">Analysis Error</h3>
              <p className="text-danger-800">{result.error}</p>
            </div>
          )}
        </div>
      )}

      {/* Result Interpretation */}
      <div className="card p-6 mt-8">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Result Interpretation</h3>
        <div className="space-y-4">
          {/* Extracted CardioChek Plus Values */}
        {result.details?.extracted_values && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <h4 className="font-semibold text-blue-900 mb-3">CardioChek Plus Screen Values</h4>
            <div className="grid grid-cols-2 gap-4">
              {result.details.extracted_values.cholesterol && (
                <div className="bg-white p-3 rounded border">
                  <span className="text-sm font-medium text-gray-600">Cholesterol</span>
                  <div className="text-lg font-bold text-gray-900">
                    {result.details.extracted_values.cholesterol} {result.details.extracted_values.units || 'mg/dL'}
                  </div>
                </div>
              )}
              {result.details.extracted_values.hdl && (
                <div className="bg-white p-3 rounded border">
                  <span className="text-sm font-medium text-gray-600">HDL</span>
                  <div className="text-lg font-bold text-gray-900">
                    {result.details.extracted_values.hdl} {result.details.extracted_values.units || 'mg/dL'}
                  </div>
                </div>
              )}
              {result.details.extracted_values.triglycerides && (
                <div className="bg-white p-3 rounded border">
                  <span className="text-sm font-medium text-gray-600">Triglycerides</span>
                  <div className="text-lg font-bold text-gray-900">
                    {result.details.extracted_values.triglycerides} {result.details.extracted_values.units || 'mg/dL'}
                  </div>
                </div>
              )}
              {result.details.extracted_values.glucose && (
                <div className="bg-white p-3 rounded border">
                  <span className="text-sm font-medium text-gray-600">Glucose</span>
                  <div className="text-lg font-bold text-gray-900">
                    {result.details.extracted_values.glucose} {result.details.extracted_values.units || 'mg/dL'}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analysis Results */}
        {result.details?.analysis && result.details.analysis.length > 0 && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
            <h4 className="font-semibold text-gray-900 mb-3">Analysis</h4>
            <ul className="space-y-2">
              {result.details.analysis.map((item: string, index: number) => (
                <li key={index} className="flex items-center text-gray-700">
                  <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Result Interpretation */}
        {result.result.toLowerCase() === 'normal' && (
          <div className="bg-success-50 border border-success-200 rounded-lg p-4">
            <h4 className="font-semibold text-success-900 mb-2">CardioChek Plus - Normal Result</h4>
            <p className="text-success-800">
              Your CardioChek Plus device shows normal levels for cardiovascular markers. 
              Continue to maintain a healthy lifestyle and follow your healthcare provider's recommendations for regular monitoring.
            </p>
          </div>
        )}
        
        {result.result.toLowerCase() === 'borderline' && (
          <div className="bg-warning-50 border border-warning-200 rounded-lg p-4">
            <h4 className="font-semibold text-warning-900 mb-2">CardioChek Plus - Borderline Result</h4>
            <p className="text-warning-800">
              Your CardioChek Plus device shows some borderline values. Consider lifestyle modifications and 
              consult with a healthcare professional for personalized guidance on managing your cardiovascular health.
            </p>
          </div>
        )}
        
        {result.result.toLowerCase() === 'high risk' && (
          <div className="bg-danger-50 border border-danger-200 rounded-lg p-4">
            <h4 className="font-semibold text-danger-900 mb-2">CardioChek Plus - High Risk Result</h4>
            <p className="text-danger-800">
              Your CardioChek Plus device shows elevated levels in one or more cardiovascular markers. 
              Please consult with a healthcare professional immediately for further evaluation and guidance on managing your cardiovascular health.
            </p>
          </div>
        )}
        
        {result.result.toLowerCase() === 'no cardioChek plus detected' && (
          <div className="bg-warning-50 border border-warning-200 rounded-lg p-4">
            <h4 className="font-semibold text-warning-900 mb-2">CardioChek Plus Not Detected</h4>
            <p className="text-warning-800">
              No CardioChek Plus device was detected in the image. Please ensure the device is clearly visible 
              and try capturing the image again with better lighting and positioning.
            </p>
          </div>
        )}
        
        {result.result.toLowerCase() === 'ocr failed' && (
          <div className="bg-warning-50 border border-warning-200 rounded-lg p-4">
            <h4 className="font-semibold text-warning-900 mb-2">CardioChek Plus - Screen Reading Failed</h4>
            <p className="text-warning-800">
              The CardioChek Plus screen could not be read properly. This may be due to poor image quality, 
              screen glare, or the device not being in the correct position. Please ensure the device screen is clearly visible 
              and try capturing the image again.
            </p>
          </div>
        )}
        </div>
      </div>

      {/* Footer Actions */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center mt-8">
        <button
          onClick={onNewScan}
          className="btn-primary text-lg px-8 py-4"
        >
          <Camera className="w-6 h-6 mr-2" />
          New Analysis
        </button>
        <button
          onClick={() => {/* Implement download functionality */}}
          className="btn-secondary text-lg px-8 py-4"
        >
          <Download className="w-6 h-6 mr-2" />
          Download Report
        </button>
        <button
          onClick={() => {/* Implement share functionality */}}
          className="btn-secondary text-lg px-8 py-4"
        >
          <Share2 className="w-6 h-6 mr-2" />
          Share Results
        </button>
      </div>
    </div>
  )
}
