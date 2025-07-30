from .data import ModelInput
import os
import numpy as np
from typing import List, Dict, Any, Union
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.points import LiDARPoints
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from mmdet3d.apis.inference import show_det_result_meshlab
from copy import deepcopy
import time
import open3d as o3d
from open3d import geometry


model_to_config = {
    "pointpillars": "configs/sv3d-inf/pointpillars/trainval_config.py",
    "second": "configs/sv3d-inf/second/trainval_config.py",
}

model_to_checkpoint = {
    "pointpillars": "checkpoints/PointPillars.pth",
    "second": "checkpoints/Second.pth",
}


class InferenceLidarAPI:
    def __init__(self, model_name: str):
        assert model_name in model_to_config, f"Model {model_name} not found"
        
        config = model_to_config[model_name]
        self.cfg = mmcv.Config.fromfile(config)
        
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get('test_cfg'))
        self.checkpoint = model_to_checkpoint[model_name]
        
        checkpoint = load_checkpoint(self.model, self.checkpoint, map_location='cpu')
        self.model.cuda()
        self.model.to("cuda")
        
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"Checkpoint file {self.checkpoint} not found, Please run scripts/download_checkpoints.sh to download the checkpoint")
        
        self.model_input = ModelInput(self.cfg)
        
        # Initialize video visualizer (created when first needed)
        self.video_visualizer = None
        
    def __call__(self, lidar):
        
        data = self.model_input.get_model_input(lidar=lidar)  
        model_data = self.model_input.prepare_for_model(data, self.model)
        
         # Run inference
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **model_data)
        print("Inference completed successfully!")

        return result, model_data
    
    
    def visualize(self, model_data, result, score_thr=0.0):
        """Original visualization method using the standard implementation."""
        show_det_result_meshlab(model_data, result, "results", show=True, score_thr=score_thr)
   
