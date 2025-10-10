import cv2
import open3d as o3d

from reconstruct import preprocess_point_cloud, depth_image_to_point_cloud, execute_global_registration

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

def test_execute_global_registration():
    # Read first image pair
    rgb_image1 = cv2.imread('data_collection/first_floor/rgb/1.png')
    depth_image1 = cv2.imread('data_collection/first_floor/depth/1.png', cv2.IMREAD_GRAYSCALE)
    rgb_image2 = cv2.imread('data_collection/first_floor/rgb/2.png')
    depth_image2 = cv2.imread('data_collection/first_floor/depth/2.png', cv2.IMREAD_GRAYSCALE)

    # Convert to point clouds
    point_cloud1 = depth_image_to_point_cloud(rgb_image1, depth_image1)
    point_cloud2 = depth_image_to_point_cloud(rgb_image2, depth_image2)

    # Down-sample with larger voxel size for testing to reduce memory usage
    voxel_size = 0.1
    point_cloud1_down = preprocess_point_cloud(point_cloud1, voxel_size)
    point_cloud2_down = preprocess_point_cloud(point_cloud2, voxel_size)
    
    print(f"Point cloud 1 size after downsampling: {len(point_cloud1_down.points)}")
    print(f"Point cloud 2 size after downsampling: {len(point_cloud2_down.points)}")

    # Compute FPFH features
    fpfh1 = compute_fpfh(point_cloud1_down, voxel_size)
    fpfh2 = compute_fpfh(point_cloud2_down, voxel_size)

    # Run NumPy-based global registration
    result = execute_global_registration(point_cloud1_down, point_cloud2_down, fpfh1, fpfh2, voxel_size)

    # Print and visualize result
    print("Transformation:\n", result.transformation)
    print(f"Fitness: {result.fitness}")
    o3d.visualization.draw_geometries([point_cloud1_down, point_cloud2_down])

if __name__ == "__main__":
    test_execute_global_registration()
