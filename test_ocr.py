#!/usr/bin/env python3
"""
Test OCR extraction for CardioChek Plus format
"""

import requests
import json

def test_cardio_chek_ocr():
    """Test the OCR with a sample CardioChek Plus image"""
    
    # Test with a sample image
    test_image_path = "data/cardio_chek/images/cardio_chek_sample_01.jpg"
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            
            response = requests.post(
                'http://localhost:8000/api/scan-kit',
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ OCR Test Successful!")
                print(f"Result: {result['result']}")
                print(f"Confidence: {result['confidence']:.2f}")
                
                if result.get('details', {}).get('extracted_values'):
                    values = result['details']['extracted_values']
                    print("\n📊 Extracted Values:")
                    for key, value in values.items():
                        if key != 'units' and value is not None:
                            print(f"  {key.upper()}: {value} {values.get('units', 'mg/dL')}")
                
                if result.get('details', {}).get('analysis'):
                    print("\n🔍 Analysis:")
                    for item in result['details']['analysis']:
                        print(f"  • {item}")
                        
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_cardio_chek_ocr()
