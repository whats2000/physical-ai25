import argparse

import cv2
import numpy as np
import pandas as pd

from src.path_finding import RRTPathFinder, find_target_points_on_map, draw_path_on_map
from src.semetic_map_construction import remove_items_by_color, remove_top_and_bottom, save_semantic_map

SCALE_FACTOR = 0.2
ROBOT_RADIUS = 10.0


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
        # Draw circle with robot radius scaled to display size
        scaled_radius = int(ROBOT_RADIUS * SCALE_FACTOR)
        cv2.circle(map_image, (x, y), scaled_radius, (0, 255, 0), 2)
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
    map_np_array, x_limit, y_limit, image_size = save_semantic_map(filtered_points, filtered_colors)
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
    target_points = []

    print("=" * 50)

    # Load the 2D semantic map image
    map_image = cv2.imread('map.png')

    while selected_target is None or not target_points:
        # Let user input the target item to search
        target_item = input("Enter the target item to search (e.g., 'chair', 'table'): ").strip().lower()

        # Check if the target item exists in the mapping file
        if target_item not in mapping_dataframe['Name'].str.lower().values:
            print(f"Item '{target_item}' not found in the mapping file. Please try again.")
            continue

        selected_target = mapping_dataframe[mapping_dataframe['Name'].str.lower() == target_item].iloc[0]
        selected_target['Color_Code (R,G,B)'] = tuple(
            map(int, selected_target['Color_Code (R,G,B)'].strip('()').split(','))
        )

        # Find multiple reachable points around the target
        target_points = find_target_points_on_map(
            map_image, 
            selected_target['Color_Code (R,G,B)'], 
            offset_distance=40,
            max_points=30
        )

        # If there are no reachable points for the target item, ask for a different item
        if not target_points:
            print(f"No reachable points found for item '{target_item}'. Please choose a different item.")
            selected_target = None
            continue

    # Highlight all candidate target points with circles
    for i, target_point in enumerate(target_points):
        cv2.circle(
            map_image,
            target_point,
            20,
            (255, 200, 0),
            3
        )

    print(f"Selected target item: {selected_target['Name']}, Color code: {selected_target['Color_Code (R,G,B)']}")

    print("=" * 50)

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

    # Load the map image for RRT
    map_image = cv2.imread('map.png')
    
    # Initialize RRT pathfinder with improved parameters
    path_finder = RRTPathFinder(
        map_image=map_image,
        step_size=80.0,
        max_iterations=5000,
        goal_sample_rate=0.15,
        robot_radius=ROBOT_RADIUS
    )
    
    # Find path from start to any of the target points
    print(f"Finding path from {starting_point} to one of {len(target_points)} target points...")
    path = path_finder.find_path(starting_point, target_points)
    
    if path is None:
        print("Failed to find a valid path!")
        exit(1)
    
    print("=" * 50)
    print(f"Path found with {len(path)} waypoints!")
    
    # Draw path on map with RRT exploration visualization
    path_map = draw_path_on_map(
        map_image, 
        path, 
        starting_point, 
        target_points,
        path_finder.explored_nodes,
        path_finder.explored_edges,
        'path_map.png'
    )
    
    # Display the path map with resizing for better visibility
    display_image = cv2.resize(path_map, (new_width, new_height))
    cv2.namedWindow('Path Result', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Path Result', display_image)
    print("Displaying path result... Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("=" * 50)
