from online_inference_plugin.inference_api import InferenceLidarAPI
from online_inference_plugin.data import load_pcd
import numpy as np

def rotate_shift_lidar(lidar, pitch_angle=5.0, shift_z_up=2.5):
        
    # Correct for 5-degree pitch toward ground
    pitch_angle = pitch_angle * np.pi / 180.0  # Convert 5 degrees to radians
    cos_pitch = np.cos(pitch_angle)
    sin_pitch = np.sin(pitch_angle)

    # Rotation matrix around Y-axis to correct pitch
    # R_y = [cos(θ)   0   sin(θ)]
    #       [0        1   0     ]
    #       [-sin(θ)  0   cos(θ)]
    rotation_matrix = np.array([
        [cos_pitch,  0, sin_pitch],
        [0,          1, 0        ],
        [-sin_pitch, 0, cos_pitch]
    ])

    # Apply rotation to X, Y, Z coordinates
    xyz_rotated = np.dot(lidar[:, :3], rotation_matrix.T)
    lidar[:, :3] = xyz_rotated

    ## Substract 2 from z of points
    lidar[:,2] += shift_z_up
    return lidar


# Create inference API. It can be slow, so we recommend to create it once and reuse it.
inference_api = InferenceLidarAPI("second")

# Load lidar data with any tool you want. It just needs to be a 4xN numpy array.
#lidar = load_pcd("online_inference_plugin/example_data_dairv2x_i/velodyne/000009.pcd")
lidar = np.load("_lidar3/_lidar3_1752394150421749713_000017.npy").astype(np.float32)
# lidar = ...
lidar = rotate_shift_lidar(lidar)

# Add column of ones to the lidar data
lidar = np.concatenate([lidar, np.zeros((lidar.shape[0], 1))+0], axis=1)


# Run inference. For now, we only support lidar data.
# If you want to visualize the result, set show=True.
result, model_data = inference_api(lidar)
print(result)

inference_api.visualize(model_data, result, score_thr=0.7)