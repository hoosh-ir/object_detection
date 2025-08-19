"""
FastAPI Object Detection Service
Supports both LiDAR and Image inference modes
"""

import os
import base64
import logging
from typing import Optional, List, Dict, Any
from enum import Enum

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mmcv

from mmdet3d.online_inference_plugin.inference_api import InferenceLidarAPI, InferenceCameraAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceMode(str, Enum):
    LIDAR = "lidar"
    IMAGE = "image"

class LidarInferenceRequest(BaseModel):
    """Request model for LiDAR inference"""
    model_name: str = "pointpillars"  # pointpillars or second
    score_threshold: Optional[float] = 0.3
    # LiDAR data as base64 encoded numpy array (4xN: x, y, z, intensity)
    lidar_data_base64: str

class ImageInferenceRequest(BaseModel):
    """Request model for Image inference"""
    model_name: str = "imvoxelnet"
    score_threshold: Optional[float] = 0.3
    # Camera intrinsic matrix (3x3 or 4x4)
    camera_intrinsic: List[List[float]]
    # Image data as base64 encoded numpy array
    image_data_base64: str
    # Image dimensions [height, width, channels] - required for proper reshaping
    image_shape: List[int]

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    success: bool
    mode: str
    model_name: str
    num_detections: int
    detections: List[Dict[str, Any]]
    processing_time: float
    error: Optional[str] = None

# Global inference APIs - initialized once for performance
lidar_apis = {}
image_apis = {}

app = FastAPI(
    title="Object Detection API",
    description="Infrastructure Object Detection Service supporting LiDAR and Image inference",
    version="1.0.0"
)

def get_lidar_api(model_name: str) -> InferenceLidarAPI:
    """Get or create LiDAR inference API"""
    if model_name not in lidar_apis:
        logger.info(f"Initializing LiDAR API for model: {model_name}")
        try:
            lidar_apis[model_name] = InferenceLidarAPI(model_name)
            logger.info(f"Successfully initialized LiDAR API for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LiDAR API for model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model {model_name}: {str(e)}")
    return lidar_apis[model_name]

def get_image_api(model_name: str) -> InferenceCameraAPI:
    """Get or create Image inference API"""
    if model_name not in image_apis:
        logger.info(f"Initializing Image API for model: {model_name}")
        try:
            image_apis[model_name] = InferenceCameraAPI(model_name)
            logger.info(f"Successfully initialized Image API for model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Image API for model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model {model_name}: {str(e)}")
    return image_apis[model_name]

def parse_detection_results(result, model_data) -> List[Dict[str, Any]]:
    """Parse model output into structured format"""
    detections = []
    
    if result and len(result) > 0:
        result_dict = result[0]
        
        # Extract bounding boxes, scores, and labels
        if 'boxes_3d' in result_dict:
            boxes = result_dict['boxes_3d'].tensor.cpu().numpy()
            scores = result_dict['scores_3d'].cpu().numpy()
            labels = result_dict['labels_3d'].cpu().numpy()
            
            for i in range(len(boxes)):
                detection = {
                    'box_3d': boxes[i].tolist(),  # [x, y, z, dx, dy, dz, yaw]
                    'score': float(scores[i]),
                    'label': int(labels[i]),
                    'center': boxes[i][:3].tolist(),  # [x, y, z]
                    'dimensions': boxes[i][3:6].tolist(),  # [dx, dy, dz]
                    'rotation': float(boxes[i][6]) if len(boxes[i]) > 6 else 0.0  # yaw
                }
                detections.append(detection)
    
    return detections

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Object Detection API is running", "modes": ["lidar", "image"]}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            "lidar": ["pointpillars", "second"],
            "image": ["imvoxelnet"]
        },
        "loaded_models": {
            "lidar": list(lidar_apis.keys()),
            "image": list(image_apis.keys())
        }
    }

@app.post("/inference/lidar", response_model=InferenceResponse)
async def lidar_inference(request: LidarInferenceRequest):
    """
    Perform LiDAR-based object detection
    
    Args:
        request: LidarInferenceRequest containing model_name, score_threshold, and base64 encoded LiDAR data
    """
    import time
    start_time = time.time()
    
    try:
        # Validate model name
        if request.model_name not in ["pointpillars", "second"]:
            raise HTTPException(status_code=400, detail="Invalid model name. Use 'pointpillars' or 'second'")
        
        # Decode base64 LiDAR data
        try:
            lidar_bytes = base64.b64decode(request.lidar_data_base64)
            lidar_data = np.frombuffer(lidar_bytes, dtype=np.float32).reshape(-1, 4)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 LiDAR data: {str(e)}")
        
        # Validate LiDAR data shape
        if lidar_data.shape[1] != 4:
            raise HTTPException(status_code=400, detail=f"LiDAR data must have 4 columns (x,y,z,intensity), got {lidar_data.shape[1]}")
        
        logger.info(f"Processing LiDAR data with shape: {lidar_data.shape}")
        
        # Get inference API
        api = get_lidar_api(request.model_name)
        
        # Run inference
        result, model_data = api(lidar_data)
        
        # Parse results
        detections = parse_detection_results(result, model_data)
        
        # Filter by score threshold
        filtered_detections = [d for d in detections if d['score'] >= request.score_threshold]
        
        processing_time = time.time() - start_time
        logger.info(f"LiDAR inference completed in {processing_time:.2f}s, found {len(filtered_detections)} detections")
        
        return InferenceResponse(
            success=True,
            mode="lidar",
            model_name=request.model_name,
            num_detections=len(filtered_detections),
            detections=filtered_detections,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LiDAR inference error: {str(e)}")
        return InferenceResponse(
            success=False,
            mode="lidar",
            model_name=request.model_name,
            num_detections=0,
            detections=[],
            processing_time=time.time() - start_time,
            error=str(e)
        )

@app.post("/inference/image", response_model=InferenceResponse)
async def image_inference(request: ImageInferenceRequest):
    """
    Perform Image-based object detection
    
    Args:
        request: ImageInferenceRequest containing model_name, score_threshold, camera_intrinsic, and base64 encoded image data
    """
    import time
    
    start_time = time.time()
    
    try:
        # Validate model name
        if request.model_name not in ["imvoxelnet"]:
            raise HTTPException(status_code=400, detail="Invalid model name. Use 'imvoxelnet'")
        
        # Parse camera intrinsic matrix
        try:
            camera_intrinsic_matrix = np.array(request.camera_intrinsic, dtype=np.float32)
            
            # Ensure 4x4 matrix format
            if camera_intrinsic_matrix.shape == (3, 3):
                # Extend 3x3 to 4x4
                temp = np.eye(4, dtype=np.float32)
                temp[:3, :3] = camera_intrinsic_matrix
                camera_intrinsic_matrix = temp
            elif camera_intrinsic_matrix.shape != (4, 4):
                raise ValueError(f"Camera intrinsic must be 3x3 or 4x4, got {camera_intrinsic_matrix.shape}")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid camera intrinsic matrix: {str(e)}")
        
        # Decode base64 image data
        try:
            image_bytes = base64.b64decode(request.image_data_base64)
            # Convert to numpy array and reshape using provided dimensions
            image_data = np.frombuffer(image_bytes, dtype=np.uint8)
            
            # Validate image shape
            if len(request.image_shape) != 3:
                raise ValueError("Image shape must have 3 dimensions: [height, width, channels]")
            
            expected_size = np.prod(request.image_shape)
            if len(image_data) != expected_size:
                raise ValueError(f"Image data size {len(image_data)} doesn't match expected size {expected_size} from shape {request.image_shape}")
            
            # Reshape the image data
            image_data = image_data.reshape(request.image_shape)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        logger.info(f"Processing image with shape: {image_data.shape}")
        
        # Get inference API
        api = get_image_api(request.model_name)
        
        # Run inference
        result, model_data = api(image_data, camera_intrinsic_matrix)
        
        # Parse results
        detections = parse_detection_results(result, model_data)
        
        # Filter by score threshold
        filtered_detections = [d for d in detections if d['score'] >= request.score_threshold]
        
        processing_time = time.time() - start_time
        logger.info(f"Image inference completed in {processing_time:.2f}s, found {len(filtered_detections)} detections")
        
        return InferenceResponse(
            success=True,
            mode="image",
            model_name=request.model_name,
            num_detections=len(filtered_detections),
            detections=filtered_detections,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image inference error: {str(e)}")
        return InferenceResponse(
            success=False,
            mode="image",
            model_name=request.model_name,
            num_detections=0,
            detections=[],
            processing_time=time.time() - start_time,
            error=str(e)
        )

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting Object Detection API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
