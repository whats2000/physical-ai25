import argparse

import numpy as np

from src.semetic_map_construction import remove_items_by_color, remove_top_and_bottom, save_semantic_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pointcloud_path', type=str, default='semantic_3d_pointcloud',
        help='Path to the 3D semantic point cloud directory'
    )
    args = parser.parse_args()

    # Load the 3D semantic map from npy files
    points = np.load(f'{args.pointcloud_path}/point.npy')     # Shape: (N, 3)
    colors = np.load(f'{args.pointcloud_path}/color0255.npy') # Shape: (N, 3), values in [0, 255]

    # Remove ceiling and floor points
    filtered_points, filtered_colors = remove_items_by_color(points, colors)

    print(f'Original points: {points.shape[0]}, Filtered points: {filtered_points.shape[0]}')

    # After removing ceiling and floor, also remove the top and bottom of points based on y-coordinates
    filtered_points, filtered_colors = remove_top_and_bottom(filtered_points, filtered_colors)
    print(f'After removing top and bottom points: {filtered_points.shape[0]}')

    # Save a 2D semantic map
    x_limit, y_limit, image_size = save_semantic_map(filtered_points, filtered_colors)
    print("2D semantic map saved as 'map.png'")
    print(f"Map x_limit: {x_limit}, y_limit: {y_limit}")
    print(f"Image size (width, height): {image_size}")
