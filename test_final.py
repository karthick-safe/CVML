#!/usr/bin/env python3
"""
Final test of the complete CVML system
"""

import requests
import json
import time

def test_complete_system():
    """Test the complete CVML system"""
    
    print("🧪 Testing Complete CVML System")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Backend healthy: {health['status']}")
            print(f"   📊 Models loaded: {health['details']['models_loaded']}")
            print(f"   🔍 OCR available: {health['details']['ocr_available']}")
            print(f"   🎯 Detection model: {health['details']['detection_model_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return
    
    # Test image analysis
    print("\n2. Testing image analysis...")
    try:
        with open('data/cardio_chek/images/cardio_chek_sample_01.jpg', 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(
                'http://localhost:8000/api/scan-kit',
                files=files,
                timeout=30
            )
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Analysis successful! ({processing_time:.2f}s)")
                print(f"   🎯 Result: {result['result']}")
                print(f"   📈 Confidence: {result['confidence']:.2f}")
                
                if result.get('details', {}).get('extracted_values'):
                    values = result['details']['extracted_values']
                    print(f"   📊 Extracted Values:")
                    for key, value in values.items():
                        if key != 'units' and value is not None:
                            print(f"      {key.upper()}: {value} {values.get('units', 'mg/dL')}")
                
                if result.get('details', {}).get('analysis'):
                    print(f"   🔍 Medical Analysis:")
                    for item in result['details']['analysis']:
                        print(f"      • {item}")
                        
            else:
                print(f"   ❌ Analysis failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_system()
