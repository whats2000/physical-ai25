import argparse
import glob
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm


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


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.005) -> o3d.geometry.PointCloud:
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
    voxel_size: float = 0.005,
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


def local_icp_algorithm(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    trans_init: np.ndarray,
    threshold: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Run ICP registration using Open3D built-in implementation.
    :param source_down: Source point cloud
    :param target_down: Target point cloud
    :param trans_init: Initial transformation (from global registration)
    :param threshold: Max correspondence distance
    :return: RegistrationResult object
    """
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


def my_local_icp_algorithm(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    trans_init: np.ndarray,
    voxel_size: float
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Implement your own ICP algorithm here.
    :param source_down: Source point cloud
    :param target_down: Target point cloud
    :param trans_init: Initial transformation (from global registration)
    :param voxel_size: Voxel size used for down-sampling
    :return: RegistrationResult object
    """
    # ICP parameters
    max_iterations = 20
    threshold = voxel_size * 1.5
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    transform = np.copy(trans_init)

    # Initialize mask for valid points
    valid_mask = np.ones(len(source_points), dtype=bool)

    # ICP iterations
    for _ in range(max_iterations):
        # Transform source
        transformed = (transform[:3, :3] @ source_points.T).T + transform[:3, 3]

        # Find the closest target point for each source
        diff = transformed[:, None, :] - target_points[None, :, :]
        distance = np.sum(diff ** 2, axis=2)
        closest_idx = np.argmin(distance, axis=1)

        # Keep only close matches
        valid_mask = np.sqrt(distance[np.arange(len(source_points)), closest_idx]) < threshold
        src_valid = transformed[valid_mask]
        tgt_valid = target_points[closest_idx[valid_mask]]

        # Estimate new transformation
        delta = estimate_rigid_transformation(src_valid, tgt_valid)
        transform = delta @ transform

    # Build result
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = transform
    result.fitness = np.mean(valid_mask)
    result.inlier_rmse = threshold

    return result


def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH feature for a point cloud.
    :param pcd: Input point cloud
    :param voxel_size: Voxel size used for down-sampling
    :return: FPFH feature
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def reconstruct(args: argparse.Namespace):
    """
    The main reconstruction function.
    :param args: Command line arguments
        - args.version: 'open3d' or 'my_icp' to choose ICP implementation
        - args.data_root: path to the dataset
    :return: Reconstructed point cloud and estimated camera poses
    """
    # Load ground truth poses (shape: Nx7 [x, y, z, qw, qx, qy, qz])
    ground_truth = np.load(os.path.join(args.data_root, 'GT_pose.npy'))

    # Get sorted lists of RGB and depth images
    rgb_files = sorted(glob.glob(os.path.join(args.data_root, 'rgb', '*.png')))
    depth_files = sorted(glob.glob(os.path.join(args.data_root, 'depth', '*.png')))
    voxel_size = 0.005

    # Initialize variables
    result_pcd = o3d.geometry.PointCloud()
    pred_cam_pos = [np.eye(4)]
    current_transform = np.eye(4)

    # Iterate through all consecutive frames
    for i in tqdm(range(1, len(rgb_files)), desc="Reconstructing"):
        # Load RGB-D pair
        rgb_prev = o3d.io.read_image(rgb_files[i - 1])
        depth_prev = o3d.io.read_image(depth_files[i - 1])
        rgb_curr = o3d.io.read_image(rgb_files[i])
        depth_curr = o3d.io.read_image(depth_files[i])

        # Convert to point clouds
        pcd_prev = depth_image_to_point_cloud(np.asarray(rgb_prev), np.asarray(depth_prev))
        pcd_curr = depth_image_to_point_cloud(np.asarray(rgb_curr), np.asarray(depth_curr))

        # Downsample and compute features
        pcd_prev_down = preprocess_point_cloud(pcd_prev, voxel_size)
        pcd_curr_down = preprocess_point_cloud(pcd_curr, voxel_size)
        fpfh_prev = compute_fpfh(pcd_prev_down, voxel_size)
        fpfh_curr = compute_fpfh(pcd_curr_down, voxel_size)

        # Step 1: Global registration (gives coarse alignment)
        global_result = execute_global_registration(pcd_curr_down, pcd_prev_down, fpfh_curr, fpfh_prev, voxel_size)

        # Step 2: Local refinement (ICP)
        if args.version == 'open3d':
            local_result = local_icp_algorithm(
                pcd_curr_down, pcd_prev_down,
                global_result.transformation, voxel_size * 1.5
            )
        else:
            local_result = my_local_icp_algorithm(
                pcd_curr_down, pcd_prev_down,
                global_result.transformation, voxel_size
            )

        # Accumulate transformation (current to world)
        current_transform = current_transform @ np.linalg.inv(local_result.transformation)
        pred_cam_pos.append(current_transform.copy())

        # Merge current frame into the global map
        pcd_curr_down.transform(current_transform)
        result_pcd += pcd_curr_down

    # Convert to numpy for trajectory
    pred_cam_pos = np.array(pred_cam_pos)
    return result_pcd, pred_cam_pos, ground_truth


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

    result_pcd, pred_cam_pos, ground_truth = reconstruct(args)

    # Evaluate result
    pred_positions = np.array([pose[:3, 3] for pose in pred_cam_pos])
    ground_truth_positions = ground_truth[:, :3]
    distances = np.linalg.norm(pred_positions - ground_truth_positions, axis=1)
    mean_l2 = np.mean(distances)
    print(f"\nMean L2 distance: {mean_l2:.6f}")

    # Visualize final result
    lines = [[i, i + 1] for i in range(len(pred_cam_pos) - 1)]

    # Estimated trajectory (red)
    estimated_points = [pose[:3, 3] for pose in pred_cam_pos]
    estimated_line_set = o3d.geometry.LineSet()
    estimated_line_set.points = o3d.utility.Vector3dVector(estimated_points)
    estimated_line_set.lines = o3d.utility.Vector2iVector(lines)
    estimated_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

    # Ground truth trajectory (black)
    ground_truth_points = ground_truth[:, :3]
    ground_truth_line_set = o3d.geometry.LineSet()
    ground_truth_line_set.points = o3d.utility.Vector3dVector(ground_truth_points)
    ground_truth_line_set.lines = o3d.utility.Vector2iVector(lines)
    ground_truth_line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in lines])

    # Visualize together
    o3d.visualization.draw_geometries(
        [result_pcd, estimated_line_set, ground_truth_line_set],
        window_name="Reconstruction Result"
    )
