/**
 * API integration for CVML Cardio Health Check Kit Analyzer
 * Handles communication with the Python backend
 */

import axios from 'axios'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout for image processing
  headers: {
    'Content-Type': 'multipart/form-data',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export interface ScanResult {
  result: string
  confidence: number
  bounding_box?: {
    x: number
    y: number
    width: number
    height: number
    confidence: number
  }
  processing_time: number
  details?: {
    kit_detection_confidence: number
    result_classification: any
    image_dimensions: [number, number]
  }
  error?: string
}

export interface HealthResponse {
  message: string
  status: string
  version: string
  details?: any
}

export interface ValidationResponse {
  valid: boolean
  reason?: string
  size?: string
  format?: string
}

/**
 * Analyze a cardio health check kit image
 * @param imageBlob - Image file as Blob
 * @returns Promise<ScanResult> - Analysis results
 */
export async function analyzeImage(imageBlob: Blob): Promise<ScanResult> {
  try {
    const formData = new FormData()
    formData.append('file', imageBlob, 'kit_image.jpg')

    const response = await api.post<ScanResult>('/api/scan-kit', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  } catch (error: any) {
    console.error('Error analyzing image:', error)
    
    // Handle different error types
    if (error.response?.status === 400) {
      throw new Error('Invalid image file. Please ensure the image is in a supported format.')
    } else if (error.response?.status === 413) {
      throw new Error('Image file too large. Please use a smaller image.')
    } else if (error.response?.status === 500) {
      throw new Error('Server error during analysis. Please try again.')
    } else if (error.code === 'ECONNABORTED') {
      throw new Error('Analysis timed out. Please try again with a smaller image.')
    } else if (error.code === 'NETWORK_ERROR') {
      throw new Error('Network error. Please check your connection and try again.')
    } else {
      throw new Error('Failed to analyze image. Please try again.')
    }
  }
}

/**
 * Validate if an image is suitable for analysis
 * @param imageBlob - Image file as Blob
 * @returns Promise<ValidationResponse> - Validation results
 */
export async function validateImage(imageBlob: Blob): Promise<ValidationResponse> {
  try {
    const formData = new FormData()
    formData.append('file', imageBlob, 'validation_image.jpg')

    const response = await api.post<ValidationResponse>('/api/validate-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  } catch (error: any) {
    console.error('Error validating image:', error)
    return {
      valid: false,
      reason: 'Failed to validate image. Please try again.'
    }
  }
}

/**
 * Check API health status
 * @returns Promise<HealthResponse> - API health information
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await api.get<HealthResponse>('/health')
    return response.data
  } catch (error: any) {
    console.error('Error checking API health:', error)
    throw new Error('Unable to connect to the analysis service. Please try again later.')
  }
}

/**
 * Get API root information
 * @returns Promise<HealthResponse> - API information
 */
export async function getApiInfo(): Promise<HealthResponse> {
  try {
    const response = await api.get<HealthResponse>('/')
    return response.data
  } catch (error: any) {
    console.error('Error getting API info:', error)
    throw new Error('Unable to connect to the analysis service.')
  }
}

/**
 * Utility function to convert File to Blob
 * @param file - File object
 * @returns Blob
 */
export function fileToBlob(file: File): Blob {
  return new Blob([file], { type: file.type })
}

/**
 * Utility function to convert data URL to Blob
 * @param dataUrl - Data URL string
 * @returns Blob
 */
export function dataUrlToBlob(dataUrl: string): Blob {
  const arr = dataUrl.split(',')
  const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg'
  const bstr = atob(arr[1])
  let n = bstr.length
  const u8arr = new Uint8Array(n)
  
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n)
  }
  
  return new Blob([u8arr], { type: mime })
}

/**
 * Utility function to compress image before sending
 * @param blob - Image blob
 * @param maxWidth - Maximum width
 * @param maxHeight - Maximum height
 * @param quality - JPEG quality (0-1)
 * @returns Promise<Blob> - Compressed image blob
 */
export function compressImage(
  blob: Blob, 
  maxWidth: number = 1280, 
  maxHeight: number = 720, 
  quality: number = 0.8
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    
    if (!ctx) {
      reject(new Error('Could not get canvas context'))
      return
    }
    
    img.onload = () => {
      // Calculate new dimensions
      let { width, height } = img
      
      if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height)
        width *= ratio
        height *= ratio
      }
      
      // Set canvas dimensions
      canvas.width = width
      canvas.height = height
      
      // Draw and compress
      ctx.drawImage(img, 0, 0, width, height)
      canvas.toBlob(
        (compressedBlob) => {
          if (compressedBlob) {
            resolve(compressedBlob)
          } else {
            reject(new Error('Failed to compress image'))
          }
        },
        'image/jpeg',
        quality
      )
    }
    
    img.onerror = () => {
      reject(new Error('Failed to load image'))
    }
    
    img.src = URL.createObjectURL(blob)
  })
}

export default api
