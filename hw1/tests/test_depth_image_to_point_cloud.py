import cv2
import open3d as o3d

from reconstruct import depth_image_to_point_cloud

def test_depth_image_to_point_cloud():
    # Read first image pair
    rgb_image = cv2.imread('data_collection/first_floor/rgb/1.png')
    depth_image = cv2.imread('data_collection/first_floor/depth/1.png', cv2.IMREAD_GRAYSCALE)

    # Convert depth image to point cloud
    point_cloud = depth_image_to_point_cloud(rgb_image, depth_image)

    # Show the point cloud
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    test_depth_image_to_point_cloud()
