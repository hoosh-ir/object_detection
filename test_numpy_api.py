#!/usr/bin/env python3
"""
Test script for the numpy array-only API
Demonstrates how to send numpy arrays via JSON requests
"""

import numpy as np
import base64
import requests
import json
import time

def test_lidar_api(base_url="http://localhost:8000"):
    """Test LiDAR inference with numpy array"""
    print("Testing LiDAR API with numpy array...")
    
    # Create sample LiDAR data (4 columns: x, y, z, intensity)
    num_points = 1000
    x = np.random.uniform(-10, 10, num_points)
    y = np.random.uniform(-10, 10, num_points)
    z = np.random.uniform(-2, 2, num_points)
    intensity = np.random.uniform(0, 1, num_points)
    
    lidar_data = np.column_stack([x, y, z, intensity]).astype(np.float32)
    print(f"Created LiDAR data with shape: {lidar_data.shape}")
    
    # Convert to base64
    lidar_bytes = lidar_data.tobytes()
    lidar_base64 = base64.b64encode(lidar_bytes).decode('utf-8')
    
    # Prepare request
    payload = {
        "model_name": "pointpillars",
        "score_threshold": 0.3,
        "lidar_data_base64": lidar_base64
    }
    
    # Send request
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/inference/lidar",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        print(f"✓ LiDAR inference successful in {elapsed:.2f}s")
        print(f"  Model: {result['model_name']}")
        print(f"  Detections: {result['num_detections']}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ LiDAR inference failed: {e}")
        return False

def test_image_api(base_url="http://localhost:8000"):
    """Test Image inference with numpy array"""
    print("\nTesting Image API with numpy array...")
    
    # Create sample image data (RGB)
    height, width, channels = 480, 640, 3
    image_data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    print(f"Created image data with shape: {image_data.shape}")
    
    # Convert to base64
    image_bytes = image_data.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Sample camera intrinsic matrix
    camera_intrinsic = [
        [2186.359688, 0.0, 968.712906],
        [0.0, 2332.160319, 542.356703],
        [0.0, 0.0, 1.0]
    ]
    
    # Prepare request
    payload = {
        "model_name": "imvoxelnet",
        "score_threshold": 0.3,
        "camera_intrinsic": camera_intrinsic,
        "image_data_base64": image_base64,
        "image_shape": [height, width, channels]
    }
    
    # Send request
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/inference/image",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        elapsed = time.time() - start_time
        
        print(f"✓ Image inference successful in {elapsed:.2f}s")
        print(f"  Model: {result['model_name']}")
        print(f"  Detections: {result['num_detections']}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Image inference failed: {e}")
        return False

def test_health_check(base_url="http://localhost:8000"):
    """Test health check endpoint"""
    print("Testing health check...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        
        health = response.json()
        print(f"✓ API is healthy")
        print(f"  Status: {health['status']}")
        print(f"  Available models: {health['models']}")
        print(f"  Loaded models: {health['loaded_models']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Numpy Array API Test ===\n")
    
    base_url = "http://localhost:8000"
    
    # Test health first
    if not test_health_check(base_url):
        print("\n❌ Health check failed. Make sure the API is running.")
        return 1
    
    # Test LiDAR
    lidar_success = test_lidar_api(base_url)
    
    # Test Image
    image_success = test_image_api(base_url)
    
    # Summary
    print(f"\n=== Test Results ===")
    print(f"Health Check: ✓")
    print(f"LiDAR API: {'✓' if lidar_success else '✗'}")
    print(f"Image API: {'✓' if image_success else '✗'}")
    
    if lidar_success and image_success:
        print("\n🎉 All tests passed! API is working correctly with numpy arrays.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the API logs for details.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
