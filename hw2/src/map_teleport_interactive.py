import argparse
import json
from typing import Tuple, Optional, Dict, List, Union

import cv2
import habitat_sim
import numpy as np
import pandas as pd
import quaternion as qt

from src.map_utils import (
    transform_semantic,
    transform_depth,
    transform_rgb_bgr,
    pixel_to_habitat_coords,
    habitat_to_pixel_coords,
    load_map_limits,
    apply_semantic_highlighting
)

# Global variables
simulation: Optional[habitat_sim.Simulator] = None
agent: Optional[habitat_sim.Agent] = None
current_position: Optional[np.ndarray] = None
current_rotation: Optional[Union[qt.quaternion, List, np.ndarray]] = None
map_limit: Optional[
    Tuple[Tuple[float, float],  # x_limit
    Tuple[float, float],  # y_limit
    Tuple[int, int],  # image_size
    Optional[Tuple[float, float, float, float]]]  # data_bounds
] = None
map_display: Optional[np.ndarray] = None
original_map: Optional[np.ndarray] = None
SCALE_FACTOR = 0.2  # Scale factor for display (same as main.py)
floor_height = 0.0  # Will be set based on floor argument

# Item highlighting variables
selected_item_name: Optional[str] = None
color_mapping: Dict[str, Tuple[int, int, int]] = {}  # name -> RGB color
semantic_mapping: Dict[int, str] = {}  # semantic_id -> name
name_to_instance_ids: Dict[str, List[int]] = {}  # name -> list of instance IDs
pointcloud_colors: Optional[np.ndarray] = None  # For 2D map highlighting
original_observations: Dict[str, np.ndarray] = {}  # Cache of original observations


def quaternion_to_yaw(rotation: Optional[Union[qt.quaternion, List, np.ndarray]]) -> float:
    """
    Convert a quaternion to yaw angle (rotation around Y-axis).
    
    Args:
        rotation: Quaternion object from habitat_sim (quaternion.quaternion).
    
    Returns:
        Yaw angle in radians.
    """
    # Extract quaternion components
    if hasattr(rotation, 'components'):
        w, x, y, z = rotation.components
    else:
        x, y, z, w = rotation

    # Calculate yaw (rotation around Y-axis)
    # yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + x^2))
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))

    return yaw


def update_map_display(
    position: np.ndarray,
    x_limit: Tuple[float, float],
    y_limit: Tuple[float, float],
    image_size: Tuple[int, int],
    data_bounds: Optional[Tuple[float, float, float, float]]
) -> None:
    """
    Update the map display with the current agent position and facing direction.
    
    Args:
        position: Agent's current position as (x, y, z).
        x_limit: Tuple of (x_min, x_max) from the map plotting (Z coordinates).
        y_limit: Tuple of (y_min, y_max) from the map plotting (X coordinates).
        image_size: Tuple of (width, height) of the map image.
        data_bounds: Optional tuple of (x_min, x_max, y_min, y_max) for actual data region.
    """
    global map_display, original_map, selected_item_name, color_mapping, pointcloud_colors, current_rotation

    # Reset to original map
    map_display = original_map.copy()

    # Apply item highlighting if an item is selected
    if selected_item_name and selected_item_name in color_mapping and pointcloud_colors is not None:
        target_color = color_mapping[selected_item_name]
        # Create mask for the selected item color in the original map (before scaling)
        original_full_map = cv2.imread('map.png')
        if original_full_map is not None:
            # Match colors (allowing some tolerance)
            mask = np.all(np.abs(original_full_map - target_color[::-1]) < 5, axis=2)

            # Create red overlay
            overlay = original_full_map.copy()
            overlay[mask] = [0, 0, 255]  # Red in BGR

            # Blend with opacity
            highlighted_map = cv2.addWeighted(original_full_map, 0.6, overlay, 0.4, 0)

            # Resize for display
            new_width = int(image_size[0] * SCALE_FACTOR)
            new_height = int(image_size[1] * SCALE_FACTOR)
            map_display = cv2.resize(highlighted_map, (new_width, new_height))

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

    # Draw facing direction arrow
    if current_rotation is not None:
        # Get yaw angle from quaternion
        yaw = quaternion_to_yaw(current_rotation)

        """
        In Habitat: +Z is forward initially, yaw rotates around Y-axis
        In 2D map: X-axis is habitat Z, Y-axis is habitat X (with Y inverted)
        When yaw=0: agent faces +Z direction (right on 2D map)
        Positive yaw rotates counter-clockwise around Y-axis
        """
        # Calculate arrow endpoint
        arrow_length = 30  # pixels
        # Map yaw to 2D: need to add 180 degrees (pi) to flip direction
        # and negate dy because image Y is inverted
        arrow_dx = arrow_length * np.cos(yaw + np.pi)
        arrow_dy = -arrow_length * np.sin(yaw + np.pi)  # Negative because image Y is inverted

        end_x = int(scaled_x + arrow_dx)
        end_y = int(scaled_y + arrow_dy)

        # Draw arrow
        cv2.arrowedLine(
            map_display,
            (scaled_x, scaled_y),
            (end_x, end_y),
            color=(0, 255, 0),   # Green arrow
            thickness=3,
            tipLength=0.3
        )

        # Draw text showing yaw angle
        yaw_degrees = np.degrees(yaw)
        text = f"Yaw: {yaw_degrees:.1f}°"
        cv2.putText(
            map_display, text, (scaled_x + 15, scaled_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
        cv2.putText(
            map_display, text, (scaled_x + 15, scaled_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

    # Update the display
    cv2.imshow('Map - Click to Teleport', map_display)


def update_3d_views(observations: Dict[str, np.ndarray]) -> None:
    """
    Update all 3D view windows with highlighting if an item is selected.
    
    Args:
        observations: Dictionary of sensor observations.
    """
    global selected_item_name, original_observations, name_to_instance_ids

    # Store original observations
    original_observations = {
        'rgb': transform_rgb_bgr(observations["color_sensor"]),
        'depth': transform_depth(observations["depth_sensor"]),
        'semantic': transform_semantic(observations["semantic_sensor"])
    }

    if selected_item_name:
        # Apply highlighting to all views
        rgb_highlighted = apply_semantic_highlighting(
            original_observations['rgb'].copy(),
            observations["semantic_sensor"],
            selected_item_name,
            name_to_instance_ids,
            is_depth=False
        )

        depth_highlighted = apply_semantic_highlighting(
            original_observations['depth'].copy(),
            observations["semantic_sensor"],
            selected_item_name,
            name_to_instance_ids,
            is_depth=True
        )

        semantic_highlighted = apply_semantic_highlighting(
            original_observations['semantic'].copy(),
            observations["semantic_sensor"],
            selected_item_name,
            name_to_instance_ids,
            is_depth=False
        )

        cv2.imshow("RGB View", rgb_highlighted)
        cv2.imshow("Depth View", depth_highlighted)
        cv2.imshow("Semantic View", semantic_highlighted)
    else:
        # Show original views
        cv2.imshow("RGB View", original_observations['rgb'])
        cv2.imshow("Depth View", original_observations['depth'])
        cv2.imshow("Semantic View", original_observations['semantic'])


def select_item_interactive() -> Optional[str]:
    """
    Let user select an item from the available items interactively.
    
    Returns:
        Selected item name or None if cancelled.
    """
    global color_mapping

    items = sorted(color_mapping.keys())

    if not items:
        print("No items available!")
        return None

    print("\n" + "=" * 50)
    print(f"Available items to highlight ({len(items)} items):")
    print("=" * 50)

    # Display in columns with numbers
    num_columns = 4
    for i, item in enumerate(items, 1):
        # Print item with number, padded for alignment
        print(f"{i:3d}. {item:25s}", end="")

        # Add newline after every num_columns items or at the end
        if i % num_columns == 0 or i == len(items):
            print()

    print("=" * 50)

    try:
        choice = input("Enter item name (or 'c' to cancel): ").strip()
        if choice.lower() == 'c':
            return None

        # Find exact match (case-insensitive)
        for item in items:
            if item.lower() == choice.lower():
                return item

        print(f"Item '{choice}' not found!")
        return None
    except (ValueError, KeyboardInterrupt):
        return None


def load_mappings(
    mapping_file: str,
    semantic_file: str
) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[int, str], Dict[str, List[int]]]:
    """
    Load color mapping from Excel and semantic mapping from JSON.
    Only include items that exist in both 2D map (Excel) and 3D semantic data.
    
    Args:
        mapping_file: Path to the Excel file with color mappings.
        semantic_file: Path to the semantic JSON file.
    
    Returns:
        Tuple of (color_mapping, semantic_mapping, name_to_instance_ids).
        - color_mapping: Maps item names to RGB colors
        - semantic_mapping: Maps instance IDs to class names
        - name_to_instance_ids: Maps item names to lists of instance IDs
    """
    # Load color mapping from Excel (2D map items)
    df = pd.read_excel(mapping_file)
    color_map = {}
    items_in_2d = set()

    for _, row in df.iterrows():
        name = row['Name']
        color_str = row['Color_Code (R,G,B)']
        # Parse color string like "(120, 120, 120)"
        color_str = color_str.strip('()')
        r, g, b = map(int, color_str.split(','))
        color_map[name] = (r, g, b)
        items_in_2d.add(name)

    # Load semantic mapping from JSON (3D semantic data)
    with open(semantic_file, 'r') as f:
        semantic_data = json.load(f)

    # Build mapping from instance IDs to class names
    # The semantic observation contains instance IDs, not class IDs
    instance_to_name = {}
    class_names_in_3d = set()

    for obj in semantic_data['objects']:
        instance_id = obj['id']
        class_name = obj['class_name']
        instance_to_name[instance_id] = class_name
        class_names_in_3d.add(class_name)

    # Build name_to_instance_ids mapping
    # This maps class names to a list of all instance IDs with that class
    name_to_instances = {}
    for instance_id, class_name in instance_to_name.items():
        if class_name not in name_to_instances:
            name_to_instances[class_name] = []
        name_to_instances[class_name].append(instance_id)

    # Filter to only include items that exist in both 2D and 3D
    filtered_color_map = {}
    name_to_semantic_id = {}

    for name in items_in_2d:
        if name in name_to_instances:
            filtered_color_map[name] = color_map[name]
            # Store all instance IDs for this class name
            name_to_semantic_id[name] = name_to_instances[name]

    print(f"Loaded {len(filtered_color_map)} items available for highlighting")

    return filtered_color_map, instance_to_name, name_to_semantic_id


def semantic_click_callback(event: int, x: int, y: int, _flags: int, _param: Optional[object]) -> None:
    """
    Mouse callback function for clicking on the semantic view to see item info.
    """
    global simulation, semantic_mapping

    if event == cv2.EVENT_LBUTTONDOWN:
        # Get current semantic observation
        observations = simulation.get_sensor_observations()
        semantic_obs = observations["semantic_sensor"]

        # Get the semantic ID at the clicked position
        if 0 <= y < semantic_obs.shape[0] and 0 <= x < semantic_obs.shape[1]:
            semantic_id = semantic_obs[y, x]

            # Look up the name
            if semantic_id in semantic_mapping:
                item_name = semantic_mapping[semantic_id]
                print(f"Clicked: {item_name}")
            else:
                print(f"Clicked: Unknown (ID: {semantic_id})")


def mouse_callback(event: int, x: int, y: int, _flags: int, _param: Optional[object]) -> None:
    """
    Mouse callback function for clicking on the map to teleport.
    """
    global simulation, agent, current_position, current_rotation, map_limits, map_display, original_map, floor_height

    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale coordinates back to original map size
        original_x = int(x / SCALE_FACTOR)
        original_y = int(y / SCALE_FACTOR)

        x_limit, y_limit, image_size, data_bounds = map_limits

        # Convert pixel coordinates to Habitat coordinates
        habitat_x, habitat_z = pixel_to_habitat_coords(
            original_x, original_y, x_limit, y_limit, data_bounds
        )

        # Snap to nearest navigable point to ensure we're on the correct floor
        pathfinder = simulation.pathfinder
        target_position = np.array([habitat_x, floor_height, habitat_z])
        snapped_position = pathfinder.snap_point(target_position)

        if snapped_position is None:
            print(f"Warning: Position is not navigable, finding nearest point...")
            # Try to find the closest navigable point
            snapped_position = pathfinder.get_random_navigable_point()
            if snapped_position is None:
                print("Error: Could not find any navigable point!")
                return

        # Teleport agent to the snapped position (preserve rotation)
        agent_state = habitat_sim.AgentState()
        agent_state.position = snapped_position
        agent_state.rotation = current_rotation if current_rotation is not None else agent.get_state().rotation
        agent.set_state(agent_state)

        current_position = snapped_position
        current_rotation = agent_state.rotation

        # Update the map display with the current position
        update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)

        # Get and display observations from the new position
        observations = simulation.get_sensor_observations()
        update_3d_views(observations)


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
    global simulation, agent, current_position, current_rotation, map_limits, map_display, original_map, floor_height
    global color_mapping, semantic_mapping, name_to_instance_ids, pointcloud_colors, selected_item_name

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
    parser.add_argument(
        '--mapping_file', type=str,
        default='color_coding_semantic_segmentation_classes.xlsx',
        help='Path to the Excel file with color mappings'
    )
    parser.add_argument(
        '--semantic_file', type=str,
        default='replica_v1/apartment_0/habitat/info_semantic.json',
        help='Path to the semantic JSON file'
    )
    args = parser.parse_args()

    # Load mappings
    print("Loading color and semantic mappings...")
    color_mapping, semantic_mapping, name_to_instance_ids = load_mappings(
        args.mapping_file,
        args.semantic_file
    )

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
    current_rotation = agent.get_state().rotation

    # Load map limits
    map_limits = load_map_limits(args.pointcloud_path)
    x_limit, y_limit, image_size, data_bounds = map_limits

    # Load pointcloud colors for 2D highlighting
    pointcloud_colors = np.load(f'{args.pointcloud_path}/color0255.npy')

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

    cv2.namedWindow('Semantic View')
    cv2.setMouseCallback('Semantic View', semantic_click_callback)

    # Draw initial position on map
    update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)

    # Get initial observations
    observations = simulation.get_sensor_observations()
    update_3d_views(observations)

    print("\n" + "=" * 50)
    print("Interactive Map Teleportation Tool")
    print("=" * 50)
    print("Click anywhere on the map to teleport to that location")
    print("Click on Semantic View to see item name")
    print("Press 'i' to select an item to highlight")
    print("Press 'c' to clear item highlighting")
    print("Press 'q' to quit")
    print("Press 'w/a/d' to move forward/turn left/turn right")
    print("=" * 50 + "\n")

    while True:
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('i'):
            # Select item to highlight
            selected_item_name = select_item_interactive()
            if selected_item_name:
                print(f"Highlighting: {selected_item_name}")
                # Update all views with highlighting
                update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
                observations = simulation.get_sensor_observations()
                update_3d_views(observations)
            else:
                print("No item selected")
        elif key == ord('c'):
            # Clear highlighting
            selected_item_name = None
            print("Cleared item highlighting")
            # Update all views without highlighting
            update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
            observations = simulation.get_sensor_observations()
            update_3d_views(observations)
        elif key == ord('w'):
            observations = simulation.step("move_forward")
            agent_state = agent.get_state()
            current_position = agent_state.position
            current_rotation = agent_state.rotation
            # Update map with new position
            update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
            update_3d_views(observations)
        elif key == ord('a'):
            observations = simulation.step("turn_left")
            agent_state = agent.get_state()
            current_rotation = agent_state.rotation
            # Turning changes rotation, update views and map
            update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
            update_3d_views(observations)
        elif key == ord('d'):
            observations = simulation.step("turn_right")
            agent_state = agent.get_state()
            current_rotation = agent_state.rotation
            # Turning changes rotation, update views and map
            update_map_display(current_position, x_limit, y_limit, image_size, data_bounds)
            update_3d_views(observations)

    cv2.destroyAllWindows()
    simulation.close()


if __name__ == '__main__':
    main()
