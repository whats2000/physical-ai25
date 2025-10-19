import argparse
import os
import sys
from typing import Tuple, Optional

import cv2
import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

from src.semetic_map_construction import remove_items_by_color, remove_top_and_bottom

# Global variables
simulation: Optional[habitat_sim.Simulator] = None
agent: Optional[habitat_sim.Agent] = None
current_position: Optional[np.ndarray] = None
map_limit: Optional[Tuple[Tuple[float, float],  # x_limit
    Tuple[float, float],  # y_limit
    Tuple[int, int],  # image_size
    Optional[Tuple[float, float, float, float]]]  # data_bounds
] = None
map_display: Optional[np.ndarray] = None
original_map: Optional[np.ndarray] = None
SCALE_FACTOR = 0.2  # Scale factor for display (same as main.py)
floor_height = 0.0  # Will be set based on floor argument


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


def update_map_display(
    position: np.ndarray,
    x_limit: Tuple[float, float],
    y_limit: Tuple[float, float],
    image_size: Tuple[int, int],
    data_bounds: Optional[Tuple[float, float, float, float]]
) -> None:
    """
    Update the map display with the current agent position.
    
    Args:
        position: Agent's current position as (x, y, z).
        x_limit: Tuple of (x_min, x_max) from the map plotting (Z coordinates).
        y_limit: Tuple of (y_min, y_max) from the map plotting (X coordinates).
        image_size: Tuple of (width, height) of the map image.
        data_bounds: Optional tuple of (x_min, x_max, y_min, y_max) for actual data region.
    """
    global map_display, original_map

    # Reset to original map
    map_display = original_map.copy()

    # Convert agent position to pixel coordinates
    pixel_x, pixel_y = habitat_to_pixel_coords(
        position[0], position[2],
        x_limit, y_limit, image_size, data_bounds
    )

    # Scale for display
    scaled_x = int(pixel_x * SCALE_FACTOR)
    scaled_y = int(pixel_y * SCALE_FACTOR)

    # Draw agent position (blue dot with white outline)
    cv2.circle(map_display, (scaled_x, scaled_y), 5, (255, 0, 0), -1)  # Blue dot
    cv2.circle(map_display, (scaled_x, scaled_y), 8, (255, 255, 255), 2)  # White outline

    # Update the display
    cv2.imshow('Map - Click to Teleport', map_display)


def mouse_callback(event: int, x: int, y: int, _flags: int, _param: Optional[object]) -> None:
    """
    Mouse callback function for clicking on the map to teleport.
    """
    global simulation, agent, current_position, map_limits, map_display, original_map, floor_height

    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale coordinates back to original map size
        original_x = int(x / SCALE_FACTOR)
        original_y = int(y / SCALE_FACTOR)

        x_limit, y_limit, image_size, data_bounds = map_limits

        # Convert pixel coordinates to Habitat coordinates
        habitat_x, habitat_z = pixel_to_habitat_coords(
            original_x, original_y, x_limit, y_limit, image_size, data_bounds
        )

        # Snap to nearest navigable point to ensure we're on the correct floor
        pathfinder = simulation.pathfinder
        target_position = np.array([habitat_x, floor_height, habitat_z])
        snapped_position = pathfinder.snap_point(target_position)

        if snapped_position is None:
            print(f"Warning: Position is not navigable, finding nearest point...")
            # Try to find closest navigable point
            snapped_position = pathfinder.get_random_navigable_point()
            if snapped_position is None:
                print("Error: Could not find any navigable point!")
                return

        # Teleport agent to the snapped position
        agent_state = habitat_sim.AgentState()
        agent_state.position = snapped_position
        agent.set_state(agent_state)

        current_position = snapped_position

        # Update the map display with the current position
        update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)

        # Get and display observations from the new position
        observations = simulation.get_sensor_observations()

        cv2.imshow("RGB View", transform_rgb_bgr(observations["color_sensor"]))
        cv2.imshow("Depth View", transform_depth(observations["depth_sensor"]))
        cv2.imshow("Semantic View", transform_semantic(observations["semantic_sensor"]))


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
    pointcloud_path: str = 'semantic_3d_pointcloud'
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[int, int], Optional[Tuple[float, float, float, float]]]:
    """
    Reconstruct the map limits from the point cloud.
    This should match what was used in save_semantic_map.
    
    Args:
        pointcloud_path: Path to the point cloud data.
    
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
    map_image = cv2.imread('map.png')
    image_size = (map_image.shape[1], map_image.shape[0])  # (width, height)

    # Calculate the actual data bounds in the image
    data_bounds = calculate_data_bounds('map.png')

    return x_limit, y_limit, image_size, data_bounds


def make_simple_cfg(settings: dict) -> habitat_sim.Configuration:
    """
    Create simulator configuration.
    
    Args:
        settings: Dictionary containing simulator settings.
    
    Returns:
        Habitat simulator configuration.
    """
    simulation_config = habitat_sim.SimulatorConfiguration()
    simulation_config.scene_id = settings["scene"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # RGB sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # Semantic sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [settings["sensor_pitch"], 0.0, 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(simulation_config, [agent_cfg])


def main() -> None:
    """
    Main function for the interactive map teleportation tool.
    """
    global simulation, agent, current_position, map_limits, map_display, original_map, floor_height

    parser = argparse.ArgumentParser(
        description='Interactive map teleportation tool for Habitat simulator'
    )
    parser.add_argument(
        '--scene', type=str,
        default='replica_v1/apartment_0/habitat/mesh_semantic.ply',
        help='Path to the scene file'
    )
    parser.add_argument(
        '--floor', type=int, default=1,
        help='Floor number (affects Y coordinate)'
    )
    parser.add_argument(
        '--pointcloud_path', type=str,
        default='semantic_3d_pointcloud',
        help='Path to the point cloud data'
    )
    args = parser.parse_args()

    # Set floor height based on floor number
    # For floor 1, we need to check the actual navmesh height
    # The pointcloud was collected at floor 1, so we'll use a small offset
    if args.floor == 1:
        floor_height = 0.088  # Slightly above ground level for floor 1
    elif args.floor == 2:
        floor_height = 1.0  # Height for floor 2
    else:
        floor_height = 0.088

    # Simulator settings
    sim_settings = {
        "scene": args.scene,
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 512,
        "height": 512,
        "sensor_pitch": 0,
    }

    # Initialize simulator
    cfg = make_simple_cfg(sim_settings)
    simulation = habitat_sim.Simulator(cfg)
    agent = simulation.initialize_agent(sim_settings["default_agent"])

    # Check pathfinder for navigable points
    pathfinder = simulation.pathfinder

    # Find the actual navigable height for floor 1 by checking a navigable point near origin
    test_point = pathfinder.snap_point(np.array([0.0, 0.0, 0.0]))
    if test_point is not None:
        floor_height = test_point[1]
    else:
        # Try a random navigable point
        test_point = pathfinder.get_random_navigable_point()
        if test_point is not None:
            floor_height = test_point[1]

    # Set initial agent state with correct floor height
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, floor_height, 0.0])
    agent.set_state(agent_state)
    current_position = agent_state.position

    # Load map limits
    map_limits = load_map_limits(args.pointcloud_path)
    x_limit, y_limit, image_size, data_bounds = map_limits

    # Load and display the map
    map_image = cv2.imread('map.png')
    if map_image is None:
        print("Error: Could not load map.png. Please run the main script first to generate the map.")
        return

    # Resize for display
    new_width = int(image_size[0] * SCALE_FACTOR)
    new_height = int(image_size[1] * SCALE_FACTOR)
    original_map = cv2.resize(map_image, (new_width, new_height))
    map_display = original_map.copy()

    # Setup windows
    cv2.namedWindow('Map - Click to Teleport')
    cv2.setMouseCallback('Map - Click to Teleport', mouse_callback)

    # Draw initial position on map
    update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)

    # Get initial observations
    observations = simulation.get_sensor_observations()
    cv2.imshow("RGB View", transform_rgb_bgr(observations["color_sensor"]))
    cv2.imshow("Depth View", transform_depth(observations["depth_sensor"]))
    cv2.imshow("Semantic View", transform_semantic(observations["semantic_sensor"]))

    print("\n" + "=" * 50)
    print("Interactive Map Teleportation Tool")
    print("=" * 50)
    print("Click anywhere on the map to teleport to that location")
    print("Press 'q' to quit")
    print("Press 'w/a/s/d' to move forward/turn left/backward/turn right")
    print("=" * 50 + "\n")

    while True:
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w'):
            observations = simulation.step("move_forward")
            agent_state = agent.get_state()
            current_position = agent_state.position
            # Update map with new position
            update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
            cv2.imshow("RGB View", transform_rgb_bgr(observations["color_sensor"]))
            cv2.imshow("Depth View", transform_depth(observations["depth_sensor"]))
            cv2.imshow("Semantic View", transform_semantic(observations["semantic_sensor"]))
        elif key == ord('a'):
            observations = simulation.step("turn_left")
            # Turning doesn't change position, but update views
            cv2.imshow("RGB View", transform_rgb_bgr(observations["color_sensor"]))
            cv2.imshow("Depth View", transform_depth(observations["depth_sensor"]))
            cv2.imshow("Semantic View", transform_semantic(observations["semantic_sensor"]))
        elif key == ord('d'):
            observations = simulation.step("turn_right")
            # Turning doesn't change position, but update views
            cv2.imshow("RGB View", transform_rgb_bgr(observations["color_sensor"]))
            cv2.imshow("Depth View", transform_depth(observations["depth_sensor"]))
            cv2.imshow("Semantic View", transform_semantic(observations["semantic_sensor"]))
        elif key == ord('s'):
            observations = simulation.step("move_backward")
            agent_state = agent.get_state()
            current_position = agent_state.position
            # Update map with new position
            update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
            cv2.imshow("RGB View", transform_rgb_bgr(observations["color_sensor"]))
            cv2.imshow("Depth View", transform_depth(observations["depth_sensor"]))
            cv2.imshow("Semantic View", transform_semantic(observations["semantic_sensor"]))

    cv2.destroyAllWindows()
    simulation.close()


if __name__ == '__main__':
    main()
