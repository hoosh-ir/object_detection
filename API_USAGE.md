# Object Detection API Usage Guide

This guide explains how to use the Docker-based Object Detection API that supports both LiDAR and Image inference modes using JSON requests with base64 encoded numpy arrays.

## Quick Start

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t object-detection-api .

# Run the container (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 object-detection-api
```

### Using Docker Compose (Recommended)

```bash
# Build and run with docker-compose
docker-compose up --build

# Run in background
docker-compose up -d --build
```

## API Endpoints

The API runs on port 8000 and provides the following endpoints:

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health check with model information

### LiDAR Inference
- `POST /inference/lidar` - Perform LiDAR-based object detection

### Image Inference  
- `POST /inference/image` - Perform Image-based object detection

## Supported Modes and Models

### LiDAR Inference
- **Endpoint**: `POST /inference/lidar`
- **Models**: 
  - `pointpillars` (default)
  - `second`
- **Input**: LiDAR point cloud data as base64 encoded numpy array (4xN: x, y, z, intensity)
- **Format**: JSON request with base64 encoded numpy array data

### Image Inference
- **Endpoint**: `POST /inference/image`
- **Models**: 
  - `imvoxelnet` (default)
- **Input**: Image as base64 encoded numpy array + camera intrinsic matrix + image dimensions
- **Format**: JSON request with base64 encoded numpy array data

## Usage Examples

### 1. LiDAR Inference

#### Using cURL with JSON:

```bash
# First, prepare your data (this would typically be done in a script)
# Assuming you have a base64 encoded numpy array in a variable

curl -X POST "http://localhost:8000/inference/lidar" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "pointpillars",
    "score_threshold": 0.3,
    "lidar_data_base64": "YOUR_BASE64_ENCODED_LIDAR_DATA"
  }'
```

#### Using Python with requests:

```python
import requests
import numpy as np
import base64

# Load your LiDAR data (4xN numpy array: x, y, z, intensity)
lidar_data = np.load("your_lidar_file.npy").astype(np.float32)

# Convert to base64
lidar_bytes = lidar_data.tobytes()
lidar_base64 = base64.b64encode(lidar_bytes).decode('utf-8')

payload = {
    "model_name": "pointpillars",  # or "second"
    "score_threshold": 0.3,
    "lidar_data_base64": lidar_base64
}

response = requests.post(
    "http://localhost:8000/inference/lidar",
    json=payload,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
print(f"Found {result['num_detections']} detections")
```

### 2. Image Inference

#### Using cURL with JSON:

```bash
curl -X POST "http://localhost:8000/inference/image" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "imvoxelnet",
    "score_threshold": 0.3,
    "camera_intrinsic": [[2186.359688, 0.0, 968.712906], [0.0, 2332.160319, 542.356703], [0.0, 0.0, 1.0]],
    "image_data_base64": "YOUR_BASE64_ENCODED_IMAGE_DATA",
    "image_shape": [480, 640, 3]
  }'
```

#### Using Python with requests:

```python
import requests
import numpy as np
import base64

# Load your image data (height x width x channels numpy array)
# You can use cv2, PIL, or any other image loading library
import cv2
image_data = cv2.imread("your_image.jpg")
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Camera intrinsic matrix (3x3 or 4x4)
camera_intrinsic = [
    [2186.359688, 0.0, 968.712906],
    [0.0, 2332.160319, 542.356703], 
    [0.0, 0.0, 1.0]
]

# Convert image to base64
image_bytes = image_data.astype(np.uint8).tobytes()
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

payload = {
    "model_name": "imvoxelnet",
    "score_threshold": 0.3,
    "camera_intrinsic": camera_intrinsic,
    "image_data_base64": image_base64,
    "image_shape": list(image_data.shape)  # [height, width, channels]
}

response = requests.post(
    "http://localhost:8000/inference/image",
    json=payload,
    headers={'Content-Type': 'application/json'}
)

result = response.json()
print(f"Found {result['num_detections']} detections")
```

### Using the Python Client

The included `client_example.py` provides a convenient wrapper:

```bash
# LiDAR inference
python client_example.py --mode lidar --file your_lidar_file.npy --model pointpillars

# Image inference  
python client_example.py --mode image --file your_image.jpg --model imvoxelnet --intrinsic "[[2186, 0, 968], [0, 2332, 542], [0, 0, 1]]"
```

Or use it programmatically:

```python
from client_example import ObjectDetectionClient
import numpy as np

client = ObjectDetectionClient("http://localhost:8000")

# LiDAR inference with numpy array
lidar_data = np.load("your_file.npy")
result = client.lidar_inference_array(lidar_data, model_name="pointpillars")

# Image inference with numpy array
import cv2
image_data = cv2.imread("your_image.jpg")
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
camera_intrinsic = [[2186, 0, 968], [0, 2332, 542], [0, 0, 1]]
result = client.image_inference_array(image_data, camera_intrinsic, model_name="imvoxelnet")
```

## Request/Response Format

### LiDAR Request
```json
{
  "model_name": "pointpillars",
  "score_threshold": 0.3,
  "lidar_data_base64": "base64_encoded_numpy_array"
}
```

### Image Request
```json
{
  "model_name": "imvoxelnet", 
  "score_threshold": 0.3,
  "camera_intrinsic": [[2186.359688, 0.0, 968.712906], [0.0, 2332.160319, 542.356703], [0.0, 0.0, 1.0]],
  "image_data_base64": "base64_encoded_numpy_array",
  "image_shape": [480, 640, 3]
}
```

### Response Format

All inference endpoints return a JSON response with the following structure:

```json
{
  "success": true,
  "mode": "lidar",
  "model_name": "pointpillars", 
  "num_detections": 5,
  "detections": [
    {
      "box_3d": [x, y, z, dx, dy, dz, yaw],
      "score": 0.85,
      "label": 0,
      "center": [x, y, z],
      "dimensions": [dx, dy, dz],
      "rotation": 0.5
    }
  ],
  "processing_time": 1.23,
  "error": null
}
```

## Input Requirements

### LiDAR Data
- Format: 4xN numpy array (x, y, z, intensity) as float32
- Encoding: Base64 encoded bytes from `numpy_array.tobytes()`

### Image Data
- Format: HxWxC numpy array (height, width, channels) as uint8
- Encoding: Base64 encoded bytes from `numpy_array.tobytes()`
- Required: image_shape parameter [height, width, channels]
- Camera intrinsic matrix (3x3 or 4x4)

## Environment Variables

- `HOST`: API host (default: 0.0.0.0)
- `PORT`: API port (default: 8000)
- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)

## Requirements

- NVIDIA GPU with CUDA 11.1+ support
- Docker with NVIDIA Container Runtime
- At least 8GB GPU memory recommended

## Troubleshooting

### Common Issues
1. **GPU not available**: Ensure NVIDIA drivers and Docker GPU support are installed
2. **Port already in use**: Change port with environment variable
3. **Out of memory**: Reduce input data size or use a GPU with more memory
4. **Model loading fails**: Ensure checkpoints are downloaded (run `scripts/download_checkpoints.sh`)
5. **Invalid base64 data**: Ensure numpy array is properly encoded using `.tobytes()` method
6. **Shape mismatch**: For images, ensure image_shape matches the actual numpy array dimensions

### Getting Help
- Check container logs: `docker logs <container_name>`
- Validate setup: `python test_api.py`
- Check API health: `curl http://localhost:8000/health`
- View interactive docs: http://localhost:8000/docs

## Interactive API Documentation

Once the service is running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.