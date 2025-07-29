from online_inference_plugin.inference_api import InferenceLidarAPI
from online_inference_plugin.data import load_pcd

# Create inference API. It can be slow, so we recommend to create it once and reuse it.
inference_api = InferenceLidarAPI("pointpillars")

# Load lidar data with any tool you want. It just needs to be a 4xN numpy array.
lidar = load_pcd("example-cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.pcd")
# lidar = np.load("example-cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.npy")
# lidar = ...

# Run inference. For now, we only support lidar data.
# If you want to visualize the result, set show=True.
result = inference_api(lidar, show=True)
print(result)