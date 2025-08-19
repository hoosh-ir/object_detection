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
lidar = load_pcd("mmdet3d/online_inference_plugin/example_data_dairv2x_i/velodyne/000009.pcd")
#lidar = np.load("_lidar3/_lidar3_1752394146165657614_000000.npy").astype(np.float32)
print(lidar.shape)
# lidar = ...

# The dairv2x assume lidar pitch is 0, and its height is around 2.5m - 3m. 
# Our lidar on the other hand is tilted 5 degrees toward ground, and its height is around 5.
# So we need to rotate and shift the lidar to make it compatible with the dairv2x.
#lidar = rotate_shift_lidar(lidar)

# Add column of ones to the lidar data
#lidar = np.concatenate([lidar, np.zeros((lidar.shape[0], 1))+0], axis=1)


# Run inference. For now, we only support lidar data.
result, model_data = inference_api(lidar)
print(result)
# Visualize the result
inference_api.visualize(model_data, result, score_thr=0.8)