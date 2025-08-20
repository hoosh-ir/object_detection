

## Options to run

- Docker-based (recommended): Run a ready-to-use FastAPI server with GPU support. See: [Docker-based Inference (Quickstart)](#docker-based-inference-quickstart)
- Run yourself (development): Create a local Python environment and run the code directly. See: [Manual Installation (Development)](#manual-installation-development)

## Docker-based Inference (Quickstart)

Get the API running with Docker and try inference in minutes.


#### Option 1: Using the convenience script (Recommended)

```bash
# Build and run the service
./run_docker.sh run

# Check service health
./run_docker.sh health

# View logs
./run_docker.sh logs

# Stop the service
./run_docker.sh stop
```

#### Option 2: Using Docker directly

```bash
# Build the Docker image
docker build -t object-detection-api .

# Run the container (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 object-detection-api
```


### API Usage

Once the service is running, you can access:
- Health Check: http://localhost:8000/health
- LiDAR Inference: POST http://localhost:8000/inference/lidar
- Image Inference: POST http://localhost:8000/inference/image

### Example API Calls

The API only accepts numpy arrays as base64 encoded JSON data (no file uploads).

#### LiDAR Inference
```bash
curl -X POST "http://localhost:8000/inference/lidar" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "pointpillars",
    "score_threshold": 0.3,
    "lidar_data_base64": "YOUR_BASE64_ENCODED_LIDAR_DATA"
  }'
```

#### Image Inference
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

#### Test with Sample Data
```bash
# Test the API with randomly generated numpy arrays
python test_numpy_api.py
```

### Available Models

**LiDAR Models:**
- `pointpillars` - PointPillars model for LiDAR object detection
- `second` - SECOND model for LiDAR object detection

**Image Models:**
- `imvoxelnet` - ImVoxelNet model for image-based object detection

### Requirements

- NVIDIA GPU with CUDA 11.1+ support
- Docker with NVIDIA Container Runtime
- At least 8GB GPU memory recommended

For more examples, see `client_example.py` and `test_numpy_api.py`.

### Use the Client with Your Data

```bash
# LiDAR (.npy with shape [N,4])
python client_example.py --mode lidar \
  --file /absolute/path/to/points.npy \
  --model pointpillars --threshold 0.3

# Image
python client_example.py --mode image \
  --file /absolute/path/to/image.jpg \
  --model imvoxelnet --threshold 0.3
```



### Commands explained (`run_docker.sh`)

- build: Build the Docker image. If `checkpoints/` is empty, runs `scripts/download_checkpoints.sh` first.
  - Example: `./run_docker.sh build`
- run: Build if needed and start the container with GPU, expose `PORT:8000`, mount `./results -> /app/results`.
  - Example: `./run_docker.sh run --port 8000 --gpu 0`
- stop: Stop and remove the running container.
  - Example: `./run_docker.sh stop`
- logs: Stream container logs (Ctrl+C to exit).
  - Example: `./run_docker.sh logs`
- shell: Open an interactive bash shell inside the running container.
  - Example: `./run_docker.sh shell`
- clean: Stop container (if any) and remove the image.
  - Example: `./run_docker.sh clean`
- health: Check `GET /health` and pretty-print the response.
  - Example: `./run_docker.sh health`

Notes:
- Requires NVIDIA GPU and NVIDIA Container Runtime.
- First build will download checkpoints via `scripts/download_checkpoints.sh`.


## Manual Installation (Development)

If you prefer to run the code without Docker:
### System Requirements

1) LLVM C++ (For visualization in Open3D)

```bash
sudo apt install libc++-dev
```

### Create environment

```bash
conda create --name mvxnet python==3.7
conda activate mvxnet
conda install cudatoolkit==11.1.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.14
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
cd object_detection
pip install -e . --user
pip install open3d==0.11
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
cd ..
```

### Download pre-trained models for infrastructure data

```bash
# install gdown with 'pip install gdown'
sh scripts/download_checkpoints.sh
```

### Inference on Custom Dataset

The `online_inference_plugin` provides a simple API for running inference on custom LiDAR data using pre-trained models.

#### Quick Start
The following code is also provided in `single_point_infer_example.py`

```python
from mmdet3d.online_inference_plugin.inference_api import InferenceLidarAPI
from mmdet3d.online_inference_plugin.data import load_pcd
import numpy as np

def rotate_shift_lidar(lidar, pitch_angle=5.0, shift_z_up=2.5):
    # Rotate lidar around y axis and shift z up
        
    pitch_angle = pitch_angle * np.pi / 180.0  # Convert 5 degrees to radians
    cos_pitch = np.cos(pitch_angle)
    sin_pitch = np.sin(pitch_angle)

    rotation_matrix = np.array([
        [cos_pitch,  0, sin_pitch],
        [0,          1, 0        ],
        [-sin_pitch, 0, cos_pitch]
    ])

    xyz_rotated = np.dot(lidar[:, :3], rotation_matrix.T)
    lidar[:, :3] = xyz_rotated

    lidar[:,2] += shift_z_up
    return lidar


# Create inference API. It can be slow, so we recommend to create it once and reuse it.
inference_api = InferenceLidarAPI("pointpillars")

# Load lidar data with any tool you want. It just needs to be a 4xN numpy array.
#lidar = load_pcd("online_inference_plugin/example_data_dairv2x_i/velodyne/000009.pcd")
lidar = np.load("_lidar3/_lidar3_1752394146165657614_000000.npy").astype(np.float32)
# lidar = ...

# The dairv2x assume lidar pitch is 0, and its height is around 2.5m - 3m. 
# Our lidar on the other hand is tilted 5 degrees toward ground, and its height is around 5.
# So we need to rotate and shift the lidar to make it compatible with the dairv2x.
lidar = rotate_shift_lidar(lidar)

# Add column of ones to the lidar data
lidar = np.concatenate([lidar, np.zeros((lidar.shape[0], 1))+0], axis=1)


# Run inference. For now, we only support lidar data.
result, model_data = inference_api(lidar)
print(result)
# Visualize the result
inference_api.visualize(model_data, result, score_thr=0.8)
```

#### Available Models

1. **PointPillars** ("pointpillars")

2. **SECOND** ("second")


   
#### Output Format

The inference returns a list of detection results, where each result contains:
- `boxes_3d`: 3D bounding boxes
- `scores_3d`: Confidence scores
- `labels_3d`: Object class labels

#### Visualization

Set `show=True` when calling the inference API to visualize results using Open3D:

```python
results = inference_api(lidar_data, show=True)
```

This will display the point cloud with detected 3D bounding boxes overlaid.



TODO:

- [x] Add Docker-based API service with FastAPI
- [x] Support both LiDAR and Image inference modes
- [ ] Add fusion models for multimodal inference
