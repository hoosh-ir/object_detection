

Code for inference on infrastructure model trained on [DAIR-V2X-I](https://thudair.baai.ac.cn/roadtest) (infrastructure-side 3d object detection)

This repo is based on **[DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)**, **[FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D)**, [mmdetection3d](https://github.com/open-mmlab/mmdetection3d). 

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
The following code is also provided in `infer_example.py`

```python
from online_inference_plugin.inference_api import InferenceLidarAPI
from online_inference_plugin.data import load_pcd

# Create inference API. It can be slow, so we recommend to create it once and reuse it.
inference_api = InferenceLidarAPI("pointpillars")

# Load lidar data with any tool you want. It just needs to be a N*4 numpy array.
lidar = load_pcd("online_inference_plugin/example_data_dairv2x_i/velodyne/000009.pcd")
# lidar = np.load("example-cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.npy")
# lidar = ...

# Run inference. For now, we only support lidar data.
# If you want to visualize the result, set show=True.
result = inference_api(lidar, show=True)
print(result)
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

- [ ] Add Image-only and fusion models.