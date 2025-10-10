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


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float=0.0015) -> o3d.geometry.PointCloud:
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

    # Compute voxel indices for each point
    voxel_indices = np.floor(cloud_point_np / voxel_size).astype(np.int32)

    # Find unique voxel indices and their corresponding point indices
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    # Select the first point in each voxel
    downsampled_points = cloud_point_np[unique_indices]

    # Build the down-sampled point cloud
    downsampled_points_cloud = o3d.geometry.PointCloud()
    downsampled_points_cloud.points = o3d.utility.Vector3dVector(downsampled_points)

    # Down-sample colors
    if pcd.has_colors():
        cloud_color_np = np.asarray(pcd.colors)
        downsampled_colors = cloud_color_np[unique_indices]
        downsampled_points_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors)

    return downsampled_points_cloud


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    raise NotImplementedError
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
