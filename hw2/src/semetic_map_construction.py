from typing import Tuple, List, Union

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

plt.switch_backend('Agg')


def remove_items_by_color(
    points_cloud: np.ndarray,
    point_colors: np.ndarray,
    remove_colors: Union[List[Tuple[int, int, int]], None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove specified items from the point cloud based on their colors.
    Args:
        points_cloud: The input point cloud of shape (N, 3).
        point_colors: The colors of the points of shape (N, 3), values in [0, 255].
        remove_colors: List of items (Corresponding to their RGB colors) to be removed.
                       e.g., ceiling (8, 255, 214), floor (255, 194, 7).
                       If None, defaults to removing ceiling and floor.
    Returns:
        A tuple containing the filtered points and their corresponding colors.
    """
    if remove_colors is None:
        # Default colors for ceiling, floor and mat
        remove_colors = [(8, 255, 214), (255, 194, 7)]

    # Create a combined mask to filter out all specified colors
    combined_mask = np.ones(len(points_cloud), dtype=bool)

    # Remove points matching any of the specified colors
    for color in remove_colors:
        color_mask = np.all(point_colors == np.array(color), axis=1)
        combined_mask &= ~color_mask

    return points_cloud[combined_mask], point_colors[combined_mask]


def remove_top_and_bottom(
    points_cloud: np.ndarray,
    point_colors: np.ndarray,
    y_percentage_above: float = 0.6,
    y_percentage_below: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the top and bottom percentage of points based on their y-coordinates.
    Args:
        points_cloud: The input point cloud of shape (N, 3).
        point_colors: The colors of the points of shape (N, 3), values in [0, 255].
        y_percentage_above: The percentage of points to remove from the top.
        y_percentage_below: The percentage of points to remove from the bottom.
    Returns:
        A tuple containing the filtered points and their corresponding colors.
    """
    y_values = points_cloud[:, 1]

    # Find the min and max y-values
    y_min, y_max = np.min(y_values), np.max(y_values)

    # Compute the exact y-thresholds
    y_threshold_above = y_max - (y_max - y_min) * y_percentage_above
    y_threshold_below = y_min + (y_max - y_min) * y_percentage_below

    mask = (y_values <= y_threshold_above) & (y_values >= y_threshold_below)

    return points_cloud[mask], point_colors[mask]

def save_semantic_map(
    points_cloud: np.ndarray,
    point_colors: np.ndarray,
    save_path: str = 'map.png'
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int]]:
    """
    Save a 2D semantic map from the point cloud and colors.
    Args:
        points_cloud: The input point cloud of shape (N, 3).
        point_colors: The colors of the points of shape (N, 3), values in [0, 255].
        save_path: The path to save the generated map.
    Returns:
        A tuple containing (x_limit, y_limit, image_size) for coordinate mapping.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(
        points_cloud[:, 2] * 10000. / 255.,  # z-coordinate scaled (For same orientation as the homework spec)
        points_cloud[:, 0] * 10000. / 255.,  # x-coordinate scaled (Same as above)
        c=point_colors / 255.,               # Normalize colors to [0, 1]
        s=30,                                # Point size (I make a little larger to prevent holes)
        marker='.',                          # Point marker
    )
    # Empty background
    plt.gca().set_facecolor((1, 1, 1))
    plt.axis('equal')
    plt.axis('off')    # Turn off axis, so I can focus on the map only
    plt.tight_layout()
    
    # Get the data limits for coordinate transformation, this is useful for 3rd step of the homework
    x_limit = plt.xlim()
    y_limit = plt.ylim()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    
    # Get image size
    image = Image.open(save_path)
    image_size = image.size
    image.close()
    
    return x_limit, y_limit, image_size
