#!/usr/bin/env python3
"""
Debug test for the API
"""

import requests
import json
import sys

def test_api():
    """Test the API with a simple request"""
    
    # Create a simple test image (1x1 pixel)
    test_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
    
    try:
        files = {'file': ('test.png', test_data, 'image/png')}
        
        print("Testing API...")
        response = requests.post(
            'http://localhost:8000/api/scan-kit',
            files=files,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Error occurred")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api()
