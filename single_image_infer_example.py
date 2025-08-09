from online_inference_plugin.data import load_image
from online_inference_plugin.inference_api import InferenceCameraAPI
import numpy as np


# Create inference API. It can be slow, so we recommend to create it once and reuse it.
inference_api = InferenceCameraAPI("imvoxelnet")

# Load lidar data with any tool you want. It just needs to be a 4xN numpy array.
image = load_image("online_inference_plugin/example_data_dairv2x_i/image/000009.jpg")
rotation = np.array([[-0.0638033225610772, -0.9910914864003576, -0.04429948490729328], [-0.2102873406178483, 0.043997692433495696, -0.7987692871343754], [0.97575114561348, -0.06031492538699515, -0.17158543199893228]])
translation = np.array([[-5.779144404715124], [6.037615758600886], [1.0636424034755758]])
cam_K = np.array([[2186.359688, 0.0, 968.712906], [0.0, 2332.160319, 542.356703], [0.0, 0.0, 1.0]])
P2 = np.eye(4)
P2[:3, :3] = np.array(cam_K).reshape([3, 3])

Tr_velo_to_cam = np.concatenate((rotation, translation), axis=1)
Tr_velo_to_cam = np.concatenate((Tr_velo_to_cam, np.array([[0, 0, 0, 1]])), axis=0)
Tr_velo_to_cam = P2 @ Tr_velo_to_cam

#lidar = np.load("_lidar3/_lidar3_1752394146165657614_000000.npy").astype(np.float32)
print(Tr_velo_to_cam.shape)
# lidar = ...

# The dairv2x assume lidar pitch is 0, and its height is around 2.5m - 3m. 
# Our lidar on the other hand is tilted 5 degrees toward ground, and its height is around 5.
# So we need to rotate and shift the lidar to make it compatible with the dairv2x.
#lidar = rotate_shift_lidar(lidar)

# Add column of ones to the lidar data
#lidar = np.concatenate([lidar, np.zeros((lidar.shape[0], 1))+0], axis=1)


# Run inference. For now, we only support lidar data.
result, model_data = inference_api(image, Tr_velo_to_cam)
print(result)
# Visualize the result
inference_api.visualize(model_data, result, score_thr=0.0)