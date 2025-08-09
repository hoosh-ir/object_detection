import numpy as np
from typing import List, Dict, Any, Union
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.points import LiDARPoints
from mmdet3d.core.bbox import get_box_type
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from pypcd import pypcd

class ModelInput:
    def __init__(self, config: mmcv.Config):
        pipeline = config.data.test.pipeline
        # Remove LoadImageFromFile and LoadPointsFromFile from pipeline
        self.pipeline = Compose([p for p in pipeline if p['type'] not in ['LoadImageFromFile', 'LoadPointsFromFile']])
        self.multi_modal = config.input_modality.use_lidar and config.input_modality.use_camera
        # Get box type from config
        if config.input_modality.use_lidar:
            self.box_type_3d, self.box_mode_3d = get_box_type(config.data.test.box_type_3d)
        else:
            self.box_type_3d, self.box_mode_3d = get_box_type("LiDAR")
    
    def get_model_input(self, lidar: np.ndarray = None, camera: np.ndarray = None, camera_intrinsic: np.ndarray = None, lidar_extrinsic: np.ndarray = None) -> Dict[str, Any]:
        """
        Get data info for online inference.

        Args:
            lidar (np.ndarray): Lidar data.
            camera (np.ndarray): Camera data.
            camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3).
            lidar_extrinsic (np.ndarray): Lidar to camera transformation matrix (4x4).
        """
        
        results = {}
        
        if self.multi_modal:
            assert camera is not None and lidar is not None, "Multi-modal mode is enabled, but camera and/or lidar are not provided"
            assert camera_intrinsic is not None and lidar_extrinsic is not None, "Multi-modal data is provided, but camera_intrinsic and lidar_extrinsic are not provided"
            lidar2img = camera_intrinsic @ lidar_extrinsic
            results['lidar2img'] = lidar2img
        elif camera is not None:
            assert camera_intrinsic is not None, "Camera data is provided, but camera_intrinsic is not provided"
        else:
            assert lidar is not None or camera is not None, "Either lidar or camera must be provided"
        

        
        if camera is not None:
            # Simulate output of LoadImageFromFile pipeline
            results['img'] = camera
            results['img_info'] = {}
            results['img_shape'] = camera.shape
            results['ori_shape'] = camera.shape
            results['pad_shape'] = camera.shape

            results['lidar2img'] = camera_intrinsic

        if lidar is not None:
            # Simulate output of LoadPointsFromFile pipeline
            results['points'] = LiDARPoints(lidar, points_dim=4)
            results['pts_filename'] = "test.pcd"
            
        # Empty fields for compatibility with mmdet3d.datasets.custom_3d.Custom3DDataset
        results['img_fields'] = ['img']
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        # Add box type information
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
            
        results = self.pipeline(results)
        
        # Collate data into batch format
        data = collate([results], samples_per_gpu=1)
                
        return data
    
    def prepare_for_model(self, data: Dict[str, Any], model) -> Dict[str, Any]:
        """
        Prepare collated data for model inference by handling device scattering.
        
        Args:
            data: Collated data from get_model_input
            model: The model to check device from
            
        Returns:
            Dict[str, Any]: Data ready for model inference
        """
        device = next(model.parameters()).device
        
        if next(model.parameters()).is_cuda:
            # Scatter to specified GPU
            data = scatter(data, [device.index])[0]
        else:
            raise ValueError("CPU inference is not supported")
                
        return data
    
    
def load_pcd(pcd_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)

    np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    return points_32

def load_image(image_file_path):
    image = mmcv.imread(image_file_path)
    return image
            
            
if __name__ == "__main__":
    from mmdet3d.models import build_model
    from mmcv.parallel import MMDataParallel

    cfg = mmcv.Config.fromfile("configs/sv3d-inf/second/trainval_config.py")
    model_input = ModelInput(cfg)
    import open3d as o3d
    lidar = load_pcd("example-cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.pcd")
    # Add a column of ones to the lidar data
    print(lidar.shape)
    # Get collated data
    data = model_input.get_model_input(lidar=lidar)  
    print("Collated data keys:", data.keys())
    
    # Build model and prepare data for inference
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = MMDataParallel(model, device_ids=[0])
    checkpoint = load_checkpoint(model, "checkpoints/Second.pth", map_location='cpu')
    model.eval()
    model.to("cuda")
    
    # Prepare data for model
    model_data = model_input.prepare_for_model(data, model)
    
    # Run inference
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **model_data)
    print("Inference completed successfully!")
    print("Result:", result)
    
    from mmdet3d.apis.inference import show_det_result_meshlab
    show_det_result_meshlab(model_data, result, "results", show=True)