import cv2
import open3d as o3d

from reconstruct import preprocess_point_cloud, depth_image_to_point_cloud


def test_preprocess_point_cloud():
    # Read first image pair
    rgb_image = cv2.imread('data_collection/first_floor/rgb/1.png')
    depth_image = cv2.imread('data_collection/first_floor/depth/1.png', cv2.IMREAD_GRAYSCALE)

    # Convert depth image to point cloud
    point_cloud = depth_image_to_point_cloud(rgb_image, depth_image)

    # Preprocess point cloud with voxel down-sampling
    voxel_size = 0.0015
    point_cloud_down = preprocess_point_cloud(point_cloud, voxel_size)

    # Show the down-sampled point cloud
    o3d.visualization.draw_geometries([point_cloud_down])

if __name__ == "__main__":
    test_preprocess_point_cloud()
