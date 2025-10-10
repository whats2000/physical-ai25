import argparse

import numpy as np
import open3d as o3d


def depth_image_to_point_cloud(rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert RGB and depth images to point cloud
    :param rgb: HxWx3 RGB image
    :param depth: HxW depth image in millimeters
    :return: Open3D point cloud object
    """
    # Camera intrinsic parameters
    height, width = depth.shape
    focal_length = width / (2 * np.tan(np.radians(90) / 2))
    cx, cy = width / 2, height / 2
    depth_scale = 1000.0
    
    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert depth to meters
    z = depth.astype(np.float32) / depth_scale
    
    # Transform u, v back to 3D camera coordinate system
    x = (u - cx) * z / focal_length
    y = (v - cy) * z / focal_length
    
    # Filter out invalid depth
    valid = (z > 0) & (z < 10.0)
    
    # Stack coordinates
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = rgb[valid].astype(np.float32) / 255.0
    
    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float=0.005) -> o3d.geometry.PointCloud:
    """
    Down-sample the point cloud using voxel down-sampling
    :param pcd: Input point cloud
    :param voxel_size: Voxel size for down-sampling
    :return: Down-sampled point cloud
    """
    if voxel_size <= 0:
        # When the voxel size is non-positive, return the original point cloud
        print("[Warning] Voxel size should be positive. Returning original point cloud.")
        return pcd

    # Convert to numpy array (Nx3)
    cloud_point_np = np.asarray(pcd.points)
    has_color = pcd.has_colors()

    # Compute voxel indices for each point
    voxel_indices = np.floor(cloud_point_np / voxel_size).astype(np.int32)

    # Group points by voxel index
    unique_voxels, inverse, counts = np.unique(
        voxel_indices, axis=0, return_inverse=True, return_counts=True
    )

    # Compute mean point position per voxel
    downsampled_points = np.zeros((len(unique_voxels), 3), dtype=np.float32)
    np.add.at(downsampled_points, inverse, cloud_point_np)
    downsampled_points /= counts[:, None]

    # Build the down-sampled point cloud
    downsampled_points_cloud = o3d.geometry.PointCloud()
    downsampled_points_cloud.points = o3d.utility.Vector3dVector(downsampled_points)

    # Down-sample colors (if available)
    if has_color:
        cloud_color_np = np.asarray(pcd.colors)
        downsampled_colors = np.zeros((len(unique_voxels), 3), dtype=np.float32)
        np.add.at(downsampled_colors, inverse, cloud_color_np)
        downsampled_colors /= counts[:, None]
        downsampled_points_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors)

    return downsampled_points_cloud

def estimate_rigid_transformation(
    source_points: np.ndarray,
    target_points: np.ndarray
) -> np.ndarray:
    """
    Estimate rigid transformation aligning source_points to target_points using SVD.
    :param source_points: Nx3 array of source points
    :param target_points: Nx3 array of target points
    :return: 4x4 homogeneous transformation matrix
    """
    # Compute centroids of both point sets
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)

    # Subtract centroids to center the points
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    # Compute rotation using Singular Value Decomposition (SVD)
    correlation_matrix = source_centered.T @ target_centered
    u_matrix, _, v_transpose = np.linalg.svd(correlation_matrix)
    rotation_matrix = v_transpose.T @ u_matrix.T

    # Handle reflection case
    if np.linalg.det(rotation_matrix) < 0:
        v_transpose[2, :] *= -1
        rotation_matrix = v_transpose.T @ u_matrix.T

    # Compute translation vector
    translation_vector = target_center - rotation_matrix @ source_center

    # Assemble full 4x4 transformation matrix
    transformation_matrix = np.eye(4, dtype=np.float32)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def execute_global_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float=0.005,
):
    """
    Perform global registration between two down-sampled point clouds
    using RANSAC based on FPFH features.
    :param source_down: Down-sampled source point cloud
    :param target_down: Down-sampled target point cloud
    :param source_fpfh: FPFH feature of source point cloud
    :param target_fpfh: FPFH feature of target point cloud
    :param voxel_size: The voxel size used for down-sampling
    :return: RegistrationResult object containing the transformation
    """
    # Convert Open3D data to numpy arrays
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    source_feature = np.asarray(source_fpfh.data).T
    target_feature = np.asarray(target_fpfh.data).T

    # Match FPFH features (find nearest neighbor in target for each source)
    diff = source_feature[:, None, :] - target_feature[None, :, :]
    distance = np.sum(diff ** 2, axis=2)
    match_indices = np.argmin(distance, axis=1)
    matched_target_points = target_points[match_indices]

    # Initialize RANSAC parameters
    num_iterations = 2000
    distance_threshold = voxel_size * 1.5
    best_inlier_count = 0
    best_transform = np.eye(4)

    # RANSAC iterations
    for _ in range(num_iterations):
        # Randomly sample 3 unique points for minimal transform estimation
        indices = np.random.choice(len(source_points), 3, replace=False)
        source_sample = source_points[indices]
        target_sample = matched_target_points[indices]

        # Estimate rigid transform using SVD
        transform = estimate_rigid_transformation(source_sample, target_sample)

        # Apply transformation to all source points
        transformed_source = (transform[:3, :3] @ source_points.T).T + transform[:3, 3]

        # Compute Euclidean distances to matched target points
        diff = np.linalg.norm(transformed_source - matched_target_points, axis=1)

        # Count inliers (points within distance threshold)
        inlier_count = np.sum(diff < distance_threshold)

        # Keep the best transformation
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_transform = transform

    # Build result object compatible with Open3D
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = best_transform
    result.fitness = best_inlier_count / len(source_points)
    result.inlier_rmse = distance_threshold

    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # TODO: Use Open3D ICP function to implement
    raise NotImplementedError
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result


def reconstruct(args):
    """
    For example:
        ...
        args.version == 'open3d':
            trans = local_icp_algorithm()
        args.version == 'my_icp':
            trans = my_local_icp_algorithm()
        ...
    """
    raise NotImplementedError
    return result_pcd, pred_cam_pos


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
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_cam_pos = reconstruct()

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    print("Mean L2 distance: ", )

    # TODO: Visualize result
    '''
    Hint: Sould visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    o3d.visualization.draw_geometries()
