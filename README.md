

Code for inference on MVXNet model trained on [DAIR-V2X-I](https://thudair.baai.ac.cn/roadtest) (infrastructure-side 3d object detection)

This repo is based on **[DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)**, **[FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D)**, [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) with almost no modifications:) 

### System Requirements

1) ## LLVM C++ (For visualization in Open3D)

```bash
sudo apt install libc++-dev
```

2) CUDA Toolkit 11.1.0 ([Installation guild](https://developer.nvidia.com/cuda-11.1.0-download-archive))

### Create environment

```bash
conda create --name mvxnet python==3.7
conda activate object_detection
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.14
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install -e . --user
pip install open3d==0.11
git clone https://github.com/klintan/pypcd.git
cd pypcd
python setup.py install
cd ..
```

### Download pre-trained MVXNet for infrastructure data

```bash
# install gdown with 'pip install gdown'
gdown https://drive.google.com/file/d/1dtTEuCzsj1I69vz6Hy2I6KZb515R-zoZ/view?usp=sharing --fuzzy -O checkpoints
```

### Inference on DAIV-V2I

1) Download DAIR-V2X-Example from [here](https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view?usp=drive_link) and unzip:

   ```bash
   # install gdown with 'pip install gdown'
   gdown https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view?usp=drive_link --fuzzy
   unzip example-cooperative-vehicle-infrastructure.zip
   ```

2)  Convert DAIR-V2I to KITTI-format data and prepare it for MMDet3D

   ```bash
   python tools/dataset_converter/dair2kitti.py --source-root example-cooperative-vehicle-infrastructure/infrastructure-side --target-root example-cooperative-vehicle-infrastructure/infrastructure-side --split-path ./data/dair-v2x/split_datas/example-single-infrastructure-split-data.json --label-type lidar --sensor-view infrastructure
   
   # Prepare KITTI dataset for MMDet3d
   python tools/create_data.py kitti --root-path example-cooperative-vehicle-infrastructure/infrastructure-side --out-dir example-cooperative-vehicle-infrastructure/infrastructure-side --extra-tag kitti
   ```

3) Run Inference

   You can evaluate your model on the validation set by visualizing results or computing 3D detection metrics

   ```bash
   # visualize
   python tools/test.py configs/sv3d-inf/mvxnet/trainval_config.py checkpoints/checkpointsp019_9v_tmp --show --show-dir out
   # Compute metrics
   python tools/test.py configs/sv3d-inf/mvxnet/trainval_config.py checkpoints/checkpointsp019_9v_tmp --eval bbox
   ```

   * The first argument of the above commands is the config of mvxnet model and the second is the path to the downloaded pre-trained model.

   **Inference on custom dataset**

1. Provide your data in KITTI format (see [here](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip) and [here](https://mmdetection3d.readthedocs.io/en/v0.17.1/datasets/kitti_det.html)) like this:

   ```
   ├── data
   │   ├── kitti
   │   │   ├── ImageSets
   │   │   ├── testing
   │   │   │   ├── calib
   │   │   │   ├── image_2
   │   │   │   ├── velodyne
   │   │   ├── training
   │   │   │   ├── calib
   │   │   │   ├── image_2
   │   │   │   ├── label_2
   │   │   │   ├── velodyne
   ```

2.  Prepare KITTI dataset for MMDet3d

   ```bash
   # Prepare KITTI dataset for MMDet3d
   python tools/create_data.py kitti --root-path {path-to-data} --out-dir {path-to-data} --extra-tag kitti
   ```

3. Change 'data_root' variable in ''configs/sv3d-inf/mvxnet/trainval_config.py'  to path to your dataset


4) Run Inference

   ```bash
   # visualize
   python tools/test.py configs/sv3d-inf/mvxnet/trainval_config.py checkpoints/checkpointsp019_9v_tmp --show --show-dir out
   # Compute metrics
   python tools/test.py configs/sv3d-inf/mvxnet/trainval_config.py checkpoints/checkpointsp019_9v_tmp --eval bbox
   ```

TODO:

- [ ] Add training scripts and new weights 