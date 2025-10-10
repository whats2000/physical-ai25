import cv2
import open3d as o3d
import numpy as np

from reconstruct import (
    preprocess_point_cloud,
    depth_image_to_point_cloud,
    execute_global_registration,
    my_local_icp_algorithm,
    local_icp_algorithm,
)

def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.pipelines.registration.Feature:
    """
    Compute FPFH feature for a point cloud.
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def test_my_local_icp_algorithm():
    """
    Test both global registration and local ICP refinement.
    """
    # Read RGB and depth images (ensure depth is 16-bit)
    rgb_image1 = cv2.imread('data_collection/first_floor/rgb/1.png')
    depth_image1 = cv2.imread('data_collection/first_floor/depth/1.png', cv2.IMREAD_UNCHANGED)
    rgb_image2 = cv2.imread('data_collection/first_floor/rgb/2.png')
    depth_image2 = cv2.imread('data_collection/first_floor/depth/2.png', cv2.IMREAD_UNCHANGED)

    # Convert to point clouds
    point_cloud1 = depth_image_to_point_cloud(rgb_image1, depth_image1)
    point_cloud2 = depth_image_to_point_cloud(rgb_image2, depth_image2)

    # Down-sample the point clouds
    voxel_size = 0.005
    point_cloud1_down = preprocess_point_cloud(point_cloud1, voxel_size)
    point_cloud2_down = preprocess_point_cloud(point_cloud2, voxel_size)

    # Compute FPFH features
    fpfh1 = compute_fpfh(point_cloud1_down, voxel_size)
    fpfh2 = compute_fpfh(point_cloud2_down, voxel_size)

    # Step 1: Global registration (NumPy)
    global_result = execute_global_registration(point_cloud1_down, point_cloud2_down, fpfh1, fpfh2, voxel_size)
    print("[Global Registration]")
    print("Transformation:\n", global_result.transformation)

    # Step 2: Local refinement (manual NumPy ICP)
    icp_result_manual = my_local_icp_algorithm(point_cloud1_down, point_cloud2_down,
                                               global_result.transformation, voxel_size)
    print("[Manual ICP Result]")
    print("Transformation:\n", icp_result_manual.transformation)
    print("Fitness:", icp_result_manual.fitness, "  RMSE:", icp_result_manual.inlier_rmse)

    # Optional: Compare with Open3D ICP for validation
    threshold = voxel_size * 1.5
    icp_result_open3d = local_icp_algorithm(point_cloud1_down, point_cloud2_down,
                                            global_result.transformation, threshold)
    print("[Open3D ICP Result]")
    print("Transformation:\n", icp_result_open3d.transformation)
    print("Fitness:", icp_result_open3d.fitness, "  RMSE:", icp_result_open3d.inlier_rmse)

    # Visualize results (apply the transformation)
    aligned_manual = point_cloud2_down.transform(icp_result_manual.transformation)
    o3d.visualization.draw_geometries([point_cloud1_down, point_cloud2_down],
                                      window_name="Manual ICP Alignment")

    # (Optional) Show Open3D ICP alignment for comparison
    aligned_open3d = point_cloud2_down.transform(icp_result_open3d.transformation)
    o3d.visualization.draw_geometries([point_cloud1_down, point_cloud2_down],
                                      window_name="Open3D ICP Alignment")


if __name__ == "__main__":
    test_my_local_icp_algorithm()
