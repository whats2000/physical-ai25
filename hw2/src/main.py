import argparse

import cv2
import numpy as np
import pandas as pd

from src.semetic_map_construction import remove_items_by_color, remove_top_and_bottom, save_semantic_map

SCALE_FACTOR = 0.2


def pick_starting_point(event: int, x: int, y: int, _flags, _param):
    """
    Mouse callback function to pick the starting point on the map.
    Args:
        event: Mouse event.
        x: x-coordinate of the mouse event.
        y: y-coordinate of the mouse event.
        _flags: Unused.
        _param: Unused.
    """
    global starting_point, map_image, original_map
    if event == cv2.EVENT_LBUTTONDOWN:
        starting_point = (x, y)
        map_image = original_map.copy()
        cv2.circle(map_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('map', map_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pointcloud_path', type=str, default='semantic_3d_pointcloud',
        help='Path to the 3D semantic point cloud directory'
    )
    parser.add_argument(
        '--mapping_file', type=str, default='color_coding_semantic_segmentation_classes.xlsx',
        help='Path to the Excel file containing color to item mapping'
    )
    args = parser.parse_args()

    # Load the 3D semantic map from npy files
    points = np.load(f'{args.pointcloud_path}/point.npy')  # Shape: (N, 3)
    colors = np.load(f'{args.pointcloud_path}/color0255.npy')  # Shape: (N, 3), values in [0, 255]

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

    # Read the mapping dataframe
    """
               | Unnamed: 0 | Color_Code (R,G,B) | Color_Code(hex) | Color | Name
    -----------|------------|--------------------|-----------------|-------|------
    0          | 1.0        | (120, 120, 120)    | #787878         | NaN   | backpack
    """
    mapping_dataframe = pd.read_excel(args.mapping_file)

    selected_target = None

    print("=" * 50)

    while selected_target is None:
        # Let user input the target item to search
        target_item = input("Enter the target item to search (e.g., 'chair', 'table'): ").strip().lower()

        # Check if the item exists in the mapping dataframe
        if target_item in mapping_dataframe['Name'].str.lower().values:
            selected_target = mapping_dataframe[mapping_dataframe['Name'].str.lower() == target_item].iloc[0]
            selected_target['Color_Code (R,G,B)'] = tuple(
                map(int, selected_target['Color_Code (R,G,B)'].strip('()').split(',')))
        else:
            print(f"Item '{target_item}' not found in the mapping file. Please try again.")

    print(f"Selected target item: {selected_target['Name']}, Color code: {selected_target['Color_Code (R,G,B)']}")

    print("=" * 50)

    # Open the map for pick the starting point
    map_image = cv2.imread('map.png')

    # Resize the map to make the window smaller
    new_width = int(image_size[0] * SCALE_FACTOR)
    new_height = int(image_size[1] * SCALE_FACTOR)
    map_image = cv2.resize(map_image, (new_width, new_height))
    original_map = map_image.copy()
    starting_point = None

    cv2.namedWindow('map')
    cv2.imshow('map', map_image)
    cv2.setMouseCallback('map', pick_starting_point)
    print("Click to select the starting point. Press any key to confirm and close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Now, check if starting point was selected
    if starting_point is not None:
        # Scale back to original coordinates
        original_x = int(starting_point[0] / SCALE_FACTOR)
        original_y = int(starting_point[1] / SCALE_FACTOR)
        starting_point = (original_x, original_y)
        print(f"Starting point: {starting_point}")
    else:
        raise ValueError("No starting point selected.")

    print("=" * 50)
