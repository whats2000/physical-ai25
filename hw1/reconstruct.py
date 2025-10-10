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


def preprocess_point_cloud(pcd, voxel_size):
    # TODO: Do voxelization to reduce the number of points for less memory usage and speedup
    raise NotImplementedError
    return pcd_down


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
