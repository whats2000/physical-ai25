import os
import sys
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

from src.semetic_map_construction import remove_items_by_color, remove_top_and_bottom


def transform_rgb_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB to BGR for OpenCV display.

    Args:
        image: RGB image array.

    Returns:
        BGR image array.
    """
    return image[:, :, [2, 1, 0]]


def transform_depth(image: np.ndarray) -> np.ndarray:
    """
    Transform depth image for visualization.

    Args:
        image: Depth image array.

    Returns:
        Transformed depth image array.
    """
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs: np.ndarray) -> np.ndarray:
    """
    Transform semantic observation for visualization.

    Args:
        semantic_obs: Semantic observation array.

    Returns:
        Transformed semantic image array.
    """
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img


def pixel_to_habitat_coords(
    pixel_x: float,
    pixel_y: float,
    x_limit: Tuple[float, float],
    y_limit: Tuple[float, float],
    image_size: Tuple[int, int],
    data_bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[float, float]:
    """
    Convert pixel coordinates on the 2D map to Habitat 3D coordinates.

    Args:
        pixel_x: X coordinate on the map image.
        pixel_y: Y coordinate on the map image.
        x_limit: Tuple of (x_min, x_max) from the map plotting (Z coordinates).
        y_limit: Tuple of (y_min, y_max) from the map plotting (X coordinates).
        image_size: Tuple of (width, height) of the map image.
        data_bounds: Optional tuple of (x_min, x_max, y_min, y_max) for actual data region.

    Returns:
        Tuple of (habitat_x, habitat_z) coordinates.
    """
    # Use data bounds if provided, otherwise use full image
    if data_bounds is not None:
        data_x_min, data_x_max, data_y_min, data_y_max = data_bounds
    else:
        data_x_min, data_x_max = 0, image_size[0]
        data_y_min, data_y_max = 0, image_size[1]

    # Map pixel coordinates to plot coordinates using the actual data bounds
    # In the plot: X-axis = Z coordinates (horizontal), Y-axis = X coordinates (vertical)

    # Map pixel X to plot Z (horizontal axis)
    plot_z = x_limit[0] + ((pixel_x - data_x_min) / (data_x_max - data_x_min)) * (x_limit[1] - x_limit[0])

    # Map pixel Y to plot X (vertical axis), inverting because image Y goes top-down but plot Y goes bottom-up
    plot_x = y_limit[1] - ((pixel_y - data_y_min) / (data_y_max - data_y_min)) * (y_limit[1] - y_limit[0])

    # Plot coordinates ARE the Habitat world coordinates
    habitat_z = plot_z
    habitat_x = plot_x

    return habitat_x, habitat_z


def habitat_to_pixel_coords(
    habitat_x: float,
    habitat_z: float,
    x_limit: Tuple[float, float],
    y_limit: Tuple[float, float],
    image_size: Tuple[int, int],
    data_bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[int, int]:
    """
    Convert Habitat 3D coordinates to pixel coordinates on the 2D map.

    Args:
        habitat_x: X coordinate in Habitat.
        habitat_z: Z coordinate in Habitat.
        x_limit: Tuple of (x_min, x_max) from the map plotting (Z coordinates).
        y_limit: Tuple of (y_min, y_max) from the map plotting (X coordinates).
        image_size: Tuple of (width, height) of the map image.
        data_bounds: Optional tuple of (x_min, x_max, y_min, y_max) for actual data region.

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates.
    """
    # Use data bounds if provided, otherwise use full image
    if data_bounds is not None:
        data_x_min, data_x_max, data_y_min, data_y_max = data_bounds
    else:
        data_x_min, data_x_max = 0, image_size[0]
        data_y_min, data_y_max = 0, image_size[1]

    # Habitat coordinates ARE the plot coordinates
    plot_z = habitat_z
    plot_x = habitat_x

    # Map plot Z to pixel X (horizontal axis)
    pixel_x = int(data_x_min + ((plot_z - x_limit[0]) / (x_limit[1] - x_limit[0])) * (data_x_max - data_x_min))

    # Map plot X to pixel Y (vertical axis), inverting because plot Y goes bottom-up but image Y goes top-down
    pixel_y = int(data_y_min + ((y_limit[1] - plot_x) / (y_limit[1] - y_limit[0])) * (data_y_max - data_y_min))

    return pixel_x, pixel_y

def calculate_data_bounds(
    map_image_path: str = 'map.png',
    threshold: int = 250
) -> Optional[Tuple[float, float, float, float]]:
    """
    Calculate the actual data region in the map image by finding non-white pixels.

    Args:
        map_image_path: Path to the map image.
        threshold: Pixel value threshold for detecting white background (default: 250).

    Returns:
        Tuple of (x_min, x_max, y_min, y_max) representing the data bounding box.
    """
    map_img = cv2.imread(map_image_path)
    if map_img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    # Find non-white pixels
    non_white = gray < threshold

    # Find bounding box of non-white pixels
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) > 0 and len(col_indices) > 0:
        y_min = row_indices[0]
        y_max = row_indices[-1]
        x_min = col_indices[0]
        x_max = col_indices[-1]
        return x_min, x_max, y_min, y_max

    return None


def load_map_limits(
    pointcloud_path: str = 'semantic_3d_pointcloud',
    map_image_path: str = 'map.png'
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Optional[Tuple[float, float, float, float]]]:
    """
    Reconstruct the map limits from the point cloud.
    This should match what was used in save_semantic_map.

    Args:
        pointcloud_path: Path to the point cloud data.
        map_image_path: Path to the map image.

    Returns:
        Tuple containing (x_limit, y_limit, image_size, data_bounds).
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load the 3D semantic map from npy files
    points = np.load(f'{pointcloud_path}/point.npy')
    colors = np.load(f'{pointcloud_path}/color0255.npy')

    # Apply the same filtering as in main.py
    filtered_points, filtered_colors = remove_items_by_color(points, colors)
    filtered_points, filtered_colors = remove_top_and_bottom(filtered_points, filtered_colors)

    # Calculate the limits (same as in save_semantic_map)
    z_coords = filtered_points[:, 2] * 10000.0 / 255.0
    x_coords = filtered_points[:, 0] * 10000.0 / 255.0

    x_limit = (np.min(z_coords), np.max(z_coords))
    y_limit = (np.min(x_coords), np.max(x_coords))

    # Load the map image to get its size
    map_image = cv2.imread(map_image_path)
    image_size = (map_image.shape[1], map_image.shape[0])  # (width, height)

    # Calculate the actual data bounds in the image
    data_bounds = calculate_data_bounds(map_image_path)

    return x_limit, y_limit, image_size, data_bounds
