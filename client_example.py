#!/usr/bin/env python3
"""
Example client for the Object Detection API
Demonstrates how to use both LiDAR and Image inference modes
"""

import requests
import numpy as np
import base64
import json
import argparse
import os
from typing import Optional

class ObjectDetectionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def lidar_inference_file(self, 
                           file_path: str, 
                           model_name: str = "pointpillars",
                           score_threshold: float = 0.3):
        """Run LiDAR inference by loading file and converting to array"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load file based on extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.npy':
            lidar_data = np.load(file_path).astype(np.float32)
        elif file_ext == '.pcd':
            # Note: This requires pypcd to be installed on client side
            try:
                from pypcd import pypcd
                pc = pypcd.PointCloud.from_path(file_path)
                np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
                np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
                np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
                np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255
                lidar_data = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
            except ImportError:
                raise ImportError("pypcd is required to load .pcd files. Please install it or convert to .npy format.")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use .npy or .pcd files.")
        
        return self.lidar_inference_array(lidar_data, model_name, score_threshold)
    
    def lidar_inference_array(self,
                            lidar_data: np.ndarray,
                            model_name: str = "pointpillars", 
                            score_threshold: float = 0.3):
        """Run LiDAR inference with numpy array"""
        
        if lidar_data.shape[1] != 4:
            raise ValueError(f"LiDAR data must have 4 columns (x,y,z,intensity), got {lidar_data.shape[1]}")
        
        # Convert to base64
        lidar_bytes = lidar_data.astype(np.float32).tobytes()
        lidar_base64 = base64.b64encode(lidar_bytes).decode('utf-8')
        
        payload = {
            'model_name': model_name,
            'score_threshold': score_threshold,
            'lidar_data_base64': lidar_base64
        }
        
        response = requests.post(
            f"{self.base_url}/inference/lidar",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    
    def image_inference_file(self,
                           image_path: str,
                           camera_intrinsic: list,
                           model_name: str = "imvoxelnet",
                           score_threshold: float = 0.3):
        """Run Image inference by loading file and converting to array"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        
        # Load image using a common library
        try:
            import cv2
            image_data = cv2.imread(image_path)
            if image_data is None:
                raise ValueError(f"Could not load image from {image_path}")
            # Convert BGR to RGB (OpenCV loads as BGR by default)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        except ImportError:
            try:
                from PIL import Image
                image = Image.open(image_path)
                image_data = np.array(image)
                if len(image_data.shape) == 2:  # Grayscale
                    image_data = np.stack([image_data, image_data, image_data], axis=-1)
            except ImportError:
                raise ImportError("Either cv2 or PIL is required to load images. Please install one of them.")
        
        return self.image_inference_array(image_data, camera_intrinsic, model_name, score_threshold)
    
    def image_inference_array(self,
                            image_data: np.ndarray,
                            camera_intrinsic: list,
                            model_name: str = "imvoxelnet",
                            score_threshold: float = 0.3):
        """Run Image inference with numpy array"""
        
        if len(image_data.shape) != 3:
            raise ValueError(f"Image data must be 3D (height, width, channels), got shape {image_data.shape}")
        
        # Convert to base64
        image_bytes = image_data.astype(np.uint8).tobytes()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            'model_name': model_name,
            'score_threshold': score_threshold,
            'camera_intrinsic': camera_intrinsic,
            'image_data_base64': image_base64,
            'image_shape': list(image_data.shape)
        }
        
        response = requests.post(
            f"{self.base_url}/inference/image",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()
    
    def image_inference_base64(self,
                             image_data: bytes,
                             camera_intrinsic: list,
                             model_name: str = "imvoxelnet",
                             score_threshold: float = 0.3):
        """Run Image inference with base64 encoded image bytes (deprecated - use image_inference_array)"""
        
        # This method is kept for backwards compatibility but is not recommended
        # since we need the image shape information
        raise NotImplementedError("This method is deprecated. Use image_inference_array with numpy arrays instead.")

def print_results(result: dict):
    """Pretty print inference results"""
    print(f"\n{'='*50}")
    print(f"Inference Results - {result['mode'].upper()} mode")
    print(f"{'='*50}")
    print(f"Model: {result['model_name']}")
    print(f"Success: {result['success']}")
    print(f"Processing Time: {result['processing_time']:.2f}s")
    print(f"Number of Detections: {result['num_detections']}")
    
    if result['error']:
        print(f"Error: {result['error']}")
        return
    
    for i, detection in enumerate(result['detections']):
        print(f"\nDetection {i+1}:")
        print(f"  Score: {detection['score']:.3f}")
        print(f"  Label: {detection['label']}")
        print(f"  Center: [{detection['center'][0]:.2f}, {detection['center'][1]:.2f}, {detection['center'][2]:.2f}]")
        print(f"  Dimensions: [{detection['dimensions'][0]:.2f}, {detection['dimensions'][1]:.2f}, {detection['dimensions'][2]:.2f}]")
        print(f"  Rotation: {detection['rotation']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Object Detection API Client Example")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--mode", choices=["lidar", "image"], required=True, help="Inference mode")
    parser.add_argument("--file", required=True, help="Input file path")
    parser.add_argument("--model", help="Model name (default: pointpillars for lidar, imvoxelnet for image)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold")
    
    # Image-specific arguments
    parser.add_argument("--intrinsic", help="Camera intrinsic matrix as JSON string (required for image mode)")
    
    args = parser.parse_args()
    
    # Initialize client
    client = ObjectDetectionClient(args.url)
    
    # Health check
    print("Checking API health...")
    health = client.health_check()
    if not health:
        print("API is not available!")
        return
    
    print(f"API Status: {health['status']}")
    print(f"Available models: {health['models']}")
    
    try:
        if args.mode == "lidar":
            model_name = args.model or "pointpillars"
            print(f"\nRunning LiDAR inference with {model_name}...")
            result = client.lidar_inference_file(
                args.file, 
                model_name=model_name,
                score_threshold=args.threshold
            )
            
        elif args.mode == "image":
            model_name = args.model or "imvoxelnet"
            
            if not args.intrinsic:
                # Use default intrinsic matrix from the example
                camera_intrinsic = [
                    [2186.359688, 0.0, 968.712906],
                    [0.0, 2332.160319, 542.356703],
                    [0.0, 0.0, 1.0]
                ]
                print("Using default camera intrinsic matrix")
            else:
                camera_intrinsic = json.loads(args.intrinsic)
            
            print(f"\nRunning Image inference with {model_name}...")
            result = client.image_inference_file(
                args.file,
                camera_intrinsic=camera_intrinsic,
                model_name=model_name,
                score_threshold=args.threshold
            )
        
        print_results(result)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
