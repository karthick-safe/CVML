'use client'

import { useState } from 'react'
import { Camera, Upload, Heart, Brain, Zap, Shield } from 'lucide-react'
import CameraCapture from '@/components/CameraCapture'
import ResultDisplay from '@/components/ResultDisplay'
import Header from '@/components/Header'

export default function Home() {
  const [currentStep, setCurrentStep] = useState<'home' | 'capture' | 'result'>('home')
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleAnalysisComplete = (result: any) => {
    setAnalysisResult(result)
    setCurrentStep('result')
  }

  const handleBackToHome = () => {
    setCurrentStep('home')
    setAnalysisResult(null)
  }

  const handleStartCapture = () => {
    setCurrentStep('capture')
  }

  if (currentStep === 'capture') {
    return (
      <div className="min-h-screen">
        <Header />
        <CameraCapture 
          onAnalysisComplete={handleAnalysisComplete}
          onBack={handleBackToHome}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />
      </div>
    )
  }

  if (currentStep === 'result') {
    return (
      <div className="min-h-screen">
        <Header />
        <ResultDisplay 
          result={analysisResult}
          onBack={handleBackToHome}
          onNewScan={handleStartCapture}
        />
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              <span className="text-gradient">CVML</span> Cardio Health Check
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
              AI-powered analysis of cardio health check kits using advanced computer vision and machine learning
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={handleStartCapture}
                className="btn-primary text-lg px-8 py-4 shadow-glow"
              >
                <Camera className="w-6 h-6 mr-2" />
                Start Analysis
              </button>
              <button className="btn-secondary text-lg px-8 py-4">
                <Upload className="w-6 h-6 mr-2" />
                Upload Image
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Advanced AI Technology
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our system combines cutting-edge computer vision and machine learning to provide accurate, fast, and reliable analysis
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="card p-6 text-center">
              <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-8 h-8 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Object Detection</h3>
              <p className="text-gray-600">
                YOLOv8-based kit localization with high accuracy and speed
              </p>
            </div>

            <div className="card p-6 text-center">
              <div className="w-16 h-16 bg-success-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Heart className="w-8 h-8 text-success-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Result Classification</h3>
              <p className="text-gray-600">
                CNN-based analysis for Positive, Negative, and Invalid results
              </p>
            </div>

            <div className="card p-6 text-center">
              <div className="w-16 h-16 bg-warning-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-warning-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Mobile Optimized</h3>
              <p className="text-gray-600">
                TensorFlow Lite models for fast on-device processing
              </p>
            </div>

            <div className="card p-6 text-center">
              <div className="w-16 h-16 bg-danger-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Shield className="w-8 h-8 text-danger-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">High Accuracy</h3>
              <p className="text-gray-600">
                Trained on diverse datasets for reliable results
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Simple three-step process for accurate kit analysis
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-20 h-20 bg-primary-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                1
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Capture Image</h3>
              <p className="text-gray-600">
                Use your camera to take a clear photo of the cardio health check kit
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-primary-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                2
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">AI Analysis</h3>
              <p className="text-gray-600">
                Our AI detects the kit and analyzes the test result automatically
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-primary-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                3
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Get Results</h3>
              <p className="text-gray-600">
                Receive instant results with confidence scores and detailed analysis
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-16 bg-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-primary-100 mb-8 max-w-2xl mx-auto">
            Experience the power of AI-driven health analysis with our advanced CVML system
          </p>
          <button
            onClick={handleStartCapture}
            className="btn bg-white text-primary-600 hover:bg-gray-100 text-lg px-8 py-4 shadow-lg"
          >
            <Camera className="w-6 h-6 mr-2" />
            Start Your Analysis
          </button>
        </div>
      </div>
    </div>
  )
}
