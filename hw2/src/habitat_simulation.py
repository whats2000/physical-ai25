import argparse
from typing import List, Tuple, Dict

import cv2
import habitat_sim
import numpy as np

from src.map_utils import (
    pixel_to_habitat_coords,
    load_map_limits,
    apply_semantic_highlighting,
    transform_rgb_bgr,
    transform_semantic
)


def make_navigation_cfg(settings: dict) -> habitat_sim.Configuration:
    """
    Create simulator configuration for navigation with discrete actions.

    Args:
        settings: Dictionary containing simulator settings.

    Returns:
        Habitat simulator configuration.
    """
    simulation_config = habitat_sim.SimulatorConfiguration()
    simulation_config.scene_id = settings["scene"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Define action space for navigation
    action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward",
            habitat_sim.ActuationSpec(amount=settings["forward_amount"])
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=settings["turn_amount"])
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=settings["turn_amount"])
        ),
    }
    agent_cfg.action_space = action_space

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


def load_semantic_mappings(
    semantic_file: str
) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """
    Load semantic mappings from JSON file.

    Args:
        semantic_file: Path to the semantic JSON file.

    Returns:
        Tuple of (semantic_mapping, name_to_instance_ids).
    """
    import json

    with open(semantic_file, 'r') as f:
        semantic_data = json.load(f)

    # Build mapping from instance IDs to class names
    semantic_mapping = {}
    name_to_instance_ids = {}

    for obj in semantic_data['objects']:
        class_name = obj['class_name']
        instance_id = obj['id']

        semantic_mapping[instance_id] = class_name

        if class_name not in name_to_instance_ids:
            name_to_instance_ids[class_name] = []
        name_to_instance_ids[class_name].append(instance_id)

    return semantic_mapping, name_to_instance_ids


def navigate_path(
    path: List[Tuple[int, int]],
    target_name: str,
    scene: str = 'replica_v1/apartment_0/habitat/mesh_semantic.ply',
    semantic_file: str = 'replica_v1/apartment_0/habitat/info_semantic.json',
    pointcloud_path: str = 'semantic_3d_pointcloud',
    floor_height: float = 0.0,
    forward_amount: float = 0.5,
    turn_amount: float = 10.0,
    video_fps: int = 10
) -> None:
    """
    Navigate the agent along the given path in Habitat simulation.

    Args:
        path: List of (pixel_x, pixel_y) waypoints.
        target_name: Name of the target item for highlighting.
        scene: Path to the scene file.
        semantic_file: Path to the semantic JSON file.
        pointcloud_path: Path to the point cloud data.
        floor_height: Y coordinate for the floor.
        forward_amount: Distance to move forward per step.
        turn_amount: Degrees to turn per step.
        video_fps: FPS for the output video.
    """
    # Load map limits
    x_limit, y_limit, image_size, data_bounds = load_map_limits(pointcloud_path)

    # Convert path to Habitat coordinates
    habitat_path = []
    for pixel_x, pixel_y in path:
        habitat_x, habitat_z = pixel_to_habitat_coords(
            pixel_x, pixel_y, x_limit, y_limit, image_size, data_bounds
        )
        habitat_path.append((habitat_x, habitat_z))

    print(f"Converted path to {len(habitat_path)} Habitat waypoints")

    # Simulator settings
    sim_settings = {
        "scene": scene,
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 512,
        "height": 512,
        "sensor_pitch": 0,
        "forward_amount": forward_amount,
        "turn_amount": np.deg2rad(turn_amount),  # Convert to radians
    }

    # Initialize simulator
    cfg = make_navigation_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])

    # Load semantic mappings
    semantic_mapping, name_to_instance_ids = load_semantic_mappings(semantic_file)

    # Set initial position
    start_x, start_z = habitat_path[0]
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([start_x, floor_height, start_z])
    agent.set_state(agent_state)

    # Initialize video writer
    video_filename = f"{target_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, video_fps, (512, 512))

    # Create window for display
    cv2.namedWindow('Navigation', cv2.WINDOW_NORMAL)

    print(f"Starting navigation to {target_name}")
    print(f"Saving video to {video_filename}")

    # Only render every N actions for smooth but fast visualization
    render_interval = 3  # Render every 3 actions (adjust for speed vs smoothness)
    action_count = 0

    # Navigate to each waypoint step by step
    for i, (target_x, target_z) in enumerate(habitat_path[1:], 1):
        print(f"Navigating to waypoint {i}/{len(habitat_path)-1}")

        waypoint_reached = False
        while not waypoint_reached:
            # Get current position
            current_pos = agent.get_state().position
            current_x, current_z = current_pos[0], current_pos[2]

            # Calculate direction to target
            dx = target_x - current_x
            dz = target_z - current_z
            distance = np.sqrt(dx**2 + dz**2)

            if distance < forward_amount / 2:  # Close enough to waypoint
                waypoint_reached = True
                break

            target_angle = np.arctan2(dx, dz)  # Angle in radians

            # Get current rotation
            current_rotation = agent.get_state().rotation
            # Convert quaternion to yaw angle
            # Quaternion is quaternion object
            x, y, z, w = current_rotation.x, current_rotation.y, current_rotation.z, current_rotation.w
            current_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

            # Calculate angle difference
            angle_diff = target_angle - current_yaw
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

            # Turn towards target (one step) if needed
            if abs(angle_diff) > 0.05:  # Threshold for alignment
                if angle_diff > 0:
                    agent.act("turn_right")
                else:
                    agent.act("turn_left")
            else:
                # Move forward
                agent.act("move_forward")

            # Get observations after each action
            observations = sim.get_sensor_observations()

            # Highlight target in semantic view
            semantic_obs = observations["semantic_sensor"]
            highlighted_semantic = apply_semantic_highlighting(
                transform_semantic(semantic_obs),
                semantic_obs,
                target_name,
                name_to_instance_ids,
                is_depth=False
            )

            # Get RGB image
            rgb_image = transform_rgb_bgr(observations["color_sensor"])

            # Combine RGB and highlighted semantic for visualization
            combined = cv2.addWeighted(rgb_image, 0.7, highlighted_semantic, 0.3, 0)

            # Write frame to video
            video_writer.write(combined)

            # Display in window
            cv2.imshow('Navigation', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                waypoint_reached = True
                break

    # Release video writer
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Navigation complete. Video saved as {video_filename}")

    sim.close()
