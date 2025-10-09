import argparse
from typing import Tuple

import numpy as np
import open3d as o3d


def depth_image_to_point_cloud(
    rgb_image: np.ndarray, 
    depth_image: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    Convert RGB and depth images to a point cloud.
    
    :param rgb_image: RGB image array with shape (height, width, 3)
    :param depth_image: Depth image array with shape (height, width)
    :return: Point cloud generated from the RGB-D images
    
    Note:
        - Camera uses pinhole model with resolution 512x512
        - Horizontal and vertical FOV are 90 degrees
        - depth_scale = 1000
    """
    fov = np.radians(90)
    width, height = 512, 512
    depth_scale = 1000.0

    # Intrinsic parameters
    focal_length_x = (width / 2) / np.tan(np.radians(fov) / 2)  # fx
    focal_length_y = (height / 2) / np.tan(np.radians(fov) / 2)  # fy
    center_x = width / 2  # cx
    center_y = height / 2  # cy

    # Calculate 3D coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image / depth_scale
    x = (u - center_x) * z / focal_length_x
    y = (v - center_y) * z / focal_length_y

    # Filter out points with zero depth
    mask = z > 0
    points = np.stack((x, y, z), axis=-1)[mask]
    colors = (rgb_image / 255.0)[mask]

    # Construct Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def preprocess_point_cloud(
    point_cloud: o3d.geometry.PointCloud, 
    voxel_size: float
) -> o3d.geometry.PointCloud:
    """
    Downsample point cloud using voxelization.
    
    :param point_cloud: Input point cloud to be downsampled
    :param voxel_size: Size of voxels for downsampling
    :return: Downsampled point cloud with reduced number of points
    """
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_point_cloud


def execute_global_registration(
    source_downsampled: o3d.geometry.PointCloud, 
    target_downsampled: o3d.geometry.PointCloud, 
    source_features: o3d.pipelines.registration.Feature,
    target_features: o3d.pipelines.registration.Feature, 
    voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform global registration between two point clouds.
    
    :param source_downsampled: Source point cloud (downsampled)
    :param target_downsampled: Target point cloud (downsampled)
    :param source_features: FPFH features of source point cloud
    :param target_features: FPFH features of target point cloud
    :param voxel_size: Voxel size used for downsampling
    :return: Registration result containing transformation matrix

    Note: The code is adapted from Open3D documentation
    (https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
    """
    distance_threshold = voxel_size * 1.5
    mutual_filter = True
    registration_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_downsampled, target_downsampled,
        source_features, target_features,
        mutual_filter,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            4000000,
            0.999
        )
    )

    return registration_result


def local_icp_algorithm(
    source_downsampled: o3d.geometry.PointCloud, 
    target_downsampled: o3d.geometry.PointCloud, 
    initial_transformation: np.ndarray, 
    distance_threshold: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform local ICP registration using Open3D library.
    
    :param source_downsampled: Source point cloud (downsampled)
    :param target_downsampled: Target point cloud (downsampled)
    :param initial_transformation: Initial transformation matrix (4x4)
    :param distance_threshold: Maximum correspondence distance threshold
    :return: Registration result containing refined transformation matrix
    """
    # TODO: Use Open3D ICP function to implement (point-to-point or point-to-plane)
    raise NotImplementedError
    return registration_result


def my_local_icp_algorithm(
    source_downsampled: o3d.geometry.PointCloud, 
    target_downsampled: o3d.geometry.PointCloud, 
    initial_transformation: np.ndarray, 
    voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform local ICP registration with custom implementation.
    
    :param source_downsampled: Source point cloud (downsampled)
    :param target_downsampled: Target point cloud (downsampled)
    :param initial_transformation: Initial transformation matrix (4x4)
    :param voxel_size: Voxel size for correspondence finding
    :return: Registration result containing refined transformation matrix
    
    Reference:
        https://cs.gmu.edu/~kosecka/cs685/cs685-icp.pdf
    """
    # TODO: Write your own ICP function
    raise NotImplementedError
    return registration_result


def reconstruct(args: argparse.Namespace) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Reconstruct 3D scene from RGB-D image sequence.
    
    :param args: Command line arguments containing version ('open3d' or 'my_icp') and data_root path
    :return: Tuple of (reconstructed_point_cloud, predicted_camera_positions)
    
    Pipeline:
        1. Unproject depth images at time t_i and t_{i+1} to point clouds
        2. Apply voxelization to downsample point clouds
        3. Apply global registration for initialization
        4. Apply local registration (ICP) to obtain transformation matrix
        5. Align point clouds and accumulate transformations along trajectory
    
    Example:
        if args.version == 'open3d':
            transformation = local_icp_algorithm(...)
        elif args.version == 'my_icp':
            transformation = my_local_icp_algorithm(...)
    """
    # TODO: Return results
    raise NotImplementedError
    return reconstructed_point_cloud, predicted_camera_positions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # TODO: Output result point cloud and estimated camera pose
    """
    Hint: Follow the steps on the spec
    """
    reconstructed_point_cloud, predicted_camera_positions = reconstruct(args)

    # TODO: Calculate and print L2 distance
    """
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    
    Steps:
        1. Load ground truth camera positions from data
        2. Compute L2 distance: ||ground_truth_position - predicted_position||_2 for each frame
        3. Calculate mean of all L2 distances
    """
    ground_truth_camera_positions = None  # TODO: Load ground truth
    mean_l2_distance = None  # TODO: Calculate mean L2 distance
    print("Mean L2 distance: ", mean_l2_distance)

    # TODO: Visualize result
    """
    Visualization requirements:
        1. Reconstructed point cloud (remove ceiling for better view)
        2. Red line: estimated camera trajectory
        3. Black line: ground truth camera trajectory
    
    Note: Transform trajectories and 3D scene to same coordinate system
          (be aware of coordinate direction and scale)
    """
    visualization_geometries = []  # TODO: Add geometries to visualize
    o3d.visualization.draw_geometries(visualization_geometries)
