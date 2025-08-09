# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import re
import torch
from copy import deepcopy
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result)


def show_proj_det_result_meshlab(data,
                                 result,
                                 out_dir,
                                 score_thr=0.0,
                                 show=False,
                                 snapshot=False):
    """Show result of projecting 3D bbox to 2D image by meshlab."""
    assert 'img' in data.keys(), 'image data is not provided for visualization'

    file_name = 'test.jpg'


    img = result[0]['original_img']

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    # After creating pred_bboxes, add this debug code:
    print(f"Pred bboxes shape: {pred_bboxes.shape}")
    print(f"Pred bboxes content: {pred_bboxes}")

    # Check for invalid values
    if np.any(np.isnan(pred_bboxes)) or np.any(np.isinf(pred_bboxes)):
        print("WARNING: pred_bboxes contains NaN or infinity values!")
        # Remove invalid bboxes
        valid_mask = np.isfinite(pred_bboxes).all(axis=1)
        pred_bboxes = pred_bboxes[valid_mask]
        print(f"After filtering invalid: {pred_bboxes.shape}")

    if len(pred_bboxes) == 0:
        print("No valid bboxes for visualization!")
        return

    box_mode = data['img_metas'][0]['box_mode_3d']
    if box_mode == Box3DMode.LIDAR:
        if 'lidar2img' not in data['img_metas'][0]:
            raise NotImplementedError(
                'LiDAR to image transformation matrix is not provided')

        show_bboxes = LiDARInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0]['lidar2img'],
            out_dir,
            file_name,
            box_mode='lidar',
            show=show)
    elif box_mode == Box3DMode.DEPTH:
        show_bboxes = DepthInstance3DBoxes(pred_bboxes, origin=(0.5, 0.5, 0))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            None,
            out_dir,
            file_name,
            box_mode='depth',
            img_metas=data['img_metas'][0],
            show=show)
    elif box_mode == Box3DMode.CAM:
        if 'cam2img' not in data['img_metas'][0]:
            raise NotImplementedError(
                'camera intrinsic matrix is not provided')

        show_bboxes = CameraInstance3DBoxes(
            pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

        show_multi_modality_result(
            img,
            None,
            show_bboxes,
            data['img_metas'][0]['cam2img'],
            out_dir,
            file_name,
            box_mode='camera',
            show=show)
    else:
        raise NotImplementedError(
            f'visualization of {box_mode} bbox is not supported')

    return file_name
