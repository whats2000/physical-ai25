from typing import Tuple, List, Optional

import cv2
import numpy as np


class TreeNode:
    """
    Data structure for RRT tree node.
    """

    def __init__(self, position: np.ndarray, parent: Optional['TreeNode'] = None):
        self.position = np.array(position, dtype=float)
        self.parent = parent


class RRTPathFinder:
    """
    RRT path finding algorithm.
    """

    def __init__(
        self,
        map_image: np.ndarray,
        step_size: float = 30.0,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.5,
        robot_radius: float = 15.0
    ):
        """
        Initialize the RRT pathfinder.
        
        Args:
            map_image: Map image where white is free space.
            step_size: Maximum distance to extend tree in each iteration.
            max_iterations: Maximum number of iterations.
            goal_sample_rate: Probability of sampling the goal.
            robot_radius: Safety margin around obstacles in pixels.
        """
        self.map_image = map_image
        self.map_height, self.map_width = map_image.shape[:2]
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.robot_radius = robot_radius
        self.occupancy_map = self._create_occupancy_map()
        self.explored_nodes = []
        self.explored_edges = []

    def _create_occupancy_map(self) -> np.ndarray:
        """
        Create a binary occupancy map with safety margin.
        """
        if len(self.map_image.shape) == 3:
            # The free space is white (255, 255, 255)
            is_free = np.all(self.map_image == 255, axis=2).astype(np.uint8)
        else:
            raise NotImplementedError("Map image must be a 3-channel RGB image.")

        # Add safety margin by eroding free space, this can be help the robot avoid obstacles
        kernel_size = int(2 * self.robot_radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        occupancy_map = cv2.erode(is_free, kernel, iterations=1)

        # Save occupancy map
        cv2.imwrite('occupancy_map.png', occupancy_map * 255)

        return occupancy_map

    def _is_collision_free(self, point: np.ndarray) -> bool:
        """
        Check if a point is collision-free with safety margin.
        Args:
            point: Point (x, y) in pixels.
        Returns:
            True if collision-free, False otherwise.
        """
        x, y = int(np.round(point[0])), int(np.round(point[1]))

        # Check bounds with margin
        margin = 5
        if x < margin or x >= self.map_width - margin or y < margin or y >= self.map_height - margin:
            return False

        # Check the point and a small area around it for extra safety
        check_radius = 5
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                check_x = x + dx
                check_y = y + dy

                if (0 > check_x or check_x >= self.map_width) or (0 > check_y or check_y >= self.map_height):
                    # Skip out-of-bounds
                    continue

                if self.occupancy_map[check_y, check_x] == 0:
                    # If any point in the area is occupied, return False
                    return False

        return True

    def _is_path_collision_free(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """
        Check if the path between two points is collision-free.
        Args:
            point1: Start point (x, y) in pixels.
            point2: End point (x, y) in pixels.
        Returns:
            True if path is collision-free, False otherwise.
        """
        distance = np.linalg.norm(point2 - point1)

        # Check with more samples for longer distances for safety
        num_samples = max(int(distance * 2), 10)

        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = point1 + t * (point2 - point1)
            if not self._is_collision_free(sample_point):
                return False

        return True

    def _steer(self, from_node: TreeNode, to_point: np.ndarray) -> TreeNode:
        """
        Steer from from_node towards to_point by step_size.
        Args:
            from_node: The node to steer from.
            to_point: The target point to steer towards.
        Returns:
            A new TreeNode in the direction of to_point.
        """
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            new_position = to_point
        else:
            new_position = from_node.position + (direction / distance) * self.step_size

        return TreeNode(new_position, parent=from_node)

    def _random_point(self) -> np.ndarray:
        """
        Sample a random point in the map.
        """
        x = np.random.uniform(0, self.map_width)
        y = np.random.uniform(0, self.map_height)
        return np.array([x, y])

    @staticmethod
    def _extract_path(node: TreeNode) -> List[Tuple[int, int]]:
        """
        Extract path from start to the given node.
        Args:
            node: The goal node.
        Returns:
            List of (x, y) tuples representing the path.
        """
        path = []
        current = node

        while current is not None:
            path.append(tuple(current.position.astype(int)))
            current = current.parent

        path.reverse()
        return path

    def _simplify_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Simplify the path by removing unnecessary waypoints where a direct straight line is collision-free.
        
        Args:
            path: The original path as list of (x, y) tuples.
        
        Returns:
            Simplified path with fewer waypoints.
        """
        if len(path) <= 2:
            return path

        simplified_path = [path[0]]
        current_index = 0

        while current_index < len(path) - 1:
            # Find the furthest point we can reach directly from current_index
            furthest_index = current_index + 1
            for candidate_index in range(current_index + 2, len(path)):
                if not self._is_path_collision_free(np.array(path[current_index]), np.array(path[candidate_index])):
                    # When we hit a collision, stop checking further
                    break

                # Update the furthest reachable index
                furthest_index = candidate_index

            # Add the furthest reachable point
            simplified_path.append(path[furthest_index])
            current_index = furthest_index

        return simplified_path

    def find_path(
        self,
        start: Tuple[int, int],
        goals: List[Tuple[int, int]]
    ) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Find a path from start to any of the goal points using RRT.
        
        Args:
            start: Start position (x, y) in pixels.
            goals: List of goal positions (x, y) in pixels. Path will be found to the nearest reachable goal.
        
        Returns:
            A tuple containing:
                - The full path as a list of (x, y) tuples.
                - The simplified path as a list of (x, y) tuples.
            Returns None if no path is found.
        """
        start_point = np.array(start, dtype=float)
        goal_points = [np.array(goal, dtype=float) for goal in goals]

        if not self._is_collision_free(start_point):
            print(f"Start point {start} is not collision-free!")
            return None

        # Filter out goals that are not collision-free
        valid_goal_points = [g for g in goal_points if self._is_collision_free(g)]
        if not valid_goal_points:
            print(f"No valid collision-free goal points!")
            return None

        print(f"Path finding with {len(valid_goal_points)} valid goal points")

        # Initialize tree
        root = TreeNode(start_point)
        nodes = [root]

        goal_threshold = self.step_size * 1.5  # 1.5x step size for goal reach

        for iteration in range(self.max_iterations):
            if (iteration + 1) % 500 == 0:
                # Print progress every 500 iterations
                print(f"RRT iteration {iteration + 1}/{self.max_iterations}")

            # Sample random point or one of the goals
            if np.random.random() < self.goal_sample_rate:
                # Randomly select one of the valid goal points
                random_point = valid_goal_points[np.random.randint(0, len(valid_goal_points))]
            else:
                random_point = self._random_point()

            # Find nearest node
            distances = [np.linalg.norm(node.position - random_point) for node in nodes]
            nearest_node = nodes[np.argmin(distances)]

            # Steer towards random point
            new_node = self._steer(nearest_node, random_point)

            # Check for collisions, if any, skip this node
            if not self._is_path_collision_free(nearest_node.position, new_node.position):
                continue

            # Add new node to tree
            nodes.append(new_node)

            # Store exploration data
            self.explored_nodes.append(tuple(new_node.position.astype(int)))
            self.explored_edges.append((
                tuple(nearest_node.position.astype(int)),
                tuple(new_node.position.astype(int))
            ))

            # Check if we reached any of the goals
            for goal_point in valid_goal_points:
                distance_to_goal = np.linalg.norm(new_node.position - goal_point)
                if distance_to_goal >= goal_threshold:
                    # Not close enough to this goal yet
                    continue

                # Check if path to goal is collision-free
                if not self._is_path_collision_free(new_node.position, goal_point):
                    # The path to this goal is not collision-free, try next goal
                    continue

                # Try connecting directly to the goal
                goal_node = TreeNode(goal_point, parent=new_node)
                nodes.append(goal_node)

                # Extract path
                path = self._extract_path(goal_node)

                # Simplify the path by removing unnecessary waypoints
                simplified_path = self._simplify_path(path)

                print(f"Path found in {iteration + 1} iterations with {len(path)} waypoints")
                print(f"Simplified to {len(simplified_path)} waypoints")
                print(f"Reached goal at {tuple(goal_point.astype(int))}")
                return path, simplified_path

        print("No path found!")
        return None


def find_target_points_on_map(
    map_image: np.ndarray,
    target_color: Tuple[int, int, int],
    offset_distance: int = 40,
    max_points: int = 10,
    min_point_separation: int = 30
) -> List[Tuple[int, int]]:
    """
    Find multiple well-distributed safe points around the target item for better path planning.
    Handles multiple instances of the target object by detecting separate clusters.
    
    Args:
        map_image: The semantic map image.
        target_color: RGB color of the target item.
        offset_distance: Distance to move away from the target.
        max_points: Maximum number of feasible target points to return.
        min_point_separation: Minimum distance between target points to avoid overlap.
    
    Returns:
        List of well-separated (x, y) points around the target, or empty list if target not found.
    """
    # Convert BGR to RGB and find target pixels
    target_bgr = (target_color[2], target_color[1], target_color[0])
    mask = cv2.inRange(map_image, np.array(target_bgr), np.array(target_bgr))

    # Get coordinates of target color
    target_coords = np.where(mask)
    if len(target_coords[0]) == 0:
        print(f"Target color {target_color} not found on map!")
        return []

    # Use connected components to find separate instances of the target
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Filter out background (label 0) and small noise components
    min_component_size = 5
    valid_instances = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_size:
            center_x = int(centroids[i][0])
            center_y = int(centroids[i][1])
            valid_instances.append((center_x, center_y, area))

    if not valid_instances:
        print(f"No valid target instances found!")
        return []

    print(f"Found {len(valid_instances)} target instance(s)")

    # Create occupancy map with safety margin
    gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    occupancy_map = (gray == 255).astype(np.uint8)

    # Create wall mask
    # I found that the occupancy map alone is not enough to avoid walls,
    # so additional wall collision checking is added.
    wall_colors_bgr = [
        (255, 112, 0),  # wall #0070FF
        (255, 0, 51),  # wall-cabinet #3300FF
        (255, 194, 0)  # wall-plug #00C2FF
    ]
    wall_mask = np.zeros(map_image.shape[:2], dtype=np.uint8)
    for wall_color in wall_colors_bgr:
        color_mask = cv2.inRange(map_image, np.array(wall_color), np.array(wall_color))
        wall_mask = cv2.bitwise_or(wall_mask, color_mask)
    wall_mask = (wall_mask > 0).astype(np.uint8)

    # Add safety margin by eroding free space
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    safe_occupancy = cv2.erode(occupancy_map, kernel, iterations=1)

    def is_far_enough_from_existing(
        new_point: Tuple[int, int],
        existing_points: List[Tuple[int, int]],
        min_distance: int
    ) -> bool:
        """
        Check if new point is far enough from all existing points.
        Args:
            new_point: The new point to check.
            existing_points: List of existing points.
            min_distance: Minimum required distance.
        Returns:
            True if far enough, False otherwise.
        """
        for existing_x, existing_y in existing_points:
            distance = np.sqrt((new_point[0] - existing_x) ** 2 + (new_point[1] - existing_y) ** 2)
            if distance < min_distance:
                return False
        return True

    # Collect multiple feasible target points with good separation
    # Process each target instance separately
    all_feasible_points = []

    # Calculate max points per instance (distribute evenly)
    max_points_per_instance = max(1, max_points // len(valid_instances))

    # For each valid target instance, find feasible points around it
    for instance_index, (center_x, center_y, area) in enumerate(valid_instances):
        instance_candidate_points = []

        # Try different directions and distances around this instance
        # Because too close points may be unsafe, we try a little away first!
        for distance_multiplier in [1.0, 1.5, 2.0, 0.7, 2.5]:
            current_offset = int(offset_distance * distance_multiplier)

            # Generate candidate directions in a circular pattern
            num_directions = 24  # More directions for better coverage (Try every 15 degrees)
            for i in range(num_directions):
                # Calculate vector direction
                angle = 2 * np.pi * i / num_directions
                dx = int(current_offset * np.cos(angle))
                dy = int(current_offset * np.sin(angle))

                # Calculate new candidate point
                new_x = center_x + dx
                new_y = center_y + dy

                # Check if within the map bounds
                if (0 > new_x or new_x >= safe_occupancy.shape[1]) or (0 > new_y or new_y >= safe_occupancy.shape[0]):
                    continue

                # Check if the new point is in free space
                if safe_occupancy[new_y, new_x] != 1:
                    continue

                # Check surrounding area for safety
                is_safe = True
                for check_dx in range(-5, 6):
                    for check_dy in range(-5, 6):
                        check_x = new_x + check_dx
                        check_y = new_y + check_dy

                        # Check bounds
                        if (0 > check_x or check_x >= safe_occupancy.shape[1]) or (
                            0 > check_y or check_y >= safe_occupancy.shape[0]):
                            continue

                        # Check occupancy
                        if safe_occupancy[check_y, check_x] == 0:
                            is_safe = False
                            break

                    if not is_safe:
                        break

                if is_safe:
                    # Check if line from candidate to target crosses a wall
                    # If crossing a wall, it must be avoided
                    crosses_wall = False
                    steps = int(np.sqrt((new_x - center_x) ** 2 + (new_y - center_y) ** 2))
                    for step in range(steps + 1):
                        t = step / max(steps, 1)
                        check_x = int(center_x + t * (new_x - center_x))
                        check_y = int(center_y + t * (new_y - center_y))
                        if (0 <= check_x < wall_mask.shape[1] and 0 <= check_y < wall_mask.shape[0] and
                            wall_mask[check_y, check_x] == 1):
                            crosses_wall = True
                            break

                    if crosses_wall:
                        # When line crosses a wall, skip this candidate
                        continue

                    # Add to candidate points with distance from center for sorting
                    distance_from_center = np.sqrt((new_x - center_x) ** 2 + (new_y - center_y) ** 2)
                    instance_candidate_points.append((new_x, new_y, distance_from_center))

        # Sort candidates by distance from center
        instance_candidate_points.sort(key=lambda p: p[2])

        # Select well-separated points from this instance's candidates
        instance_feasible_points = []
        for candidate in instance_candidate_points:
            point = (candidate[0], candidate[1])

            # Check separation from points within this instance
            if not is_far_enough_from_existing(point, instance_feasible_points, min_point_separation):
                continue

            # Also check separation from all previously added points from other instances
            if not is_far_enough_from_existing(point, all_feasible_points, min_point_separation):
                continue

            # Add this point as feasible
            instance_feasible_points.append(point)

            if len(instance_feasible_points) >= max_points_per_instance:
                # When we have enough points for this instance, stop adding more
                break

        # Add this instance's feasible points to the global list
        all_feasible_points.extend(instance_feasible_points)

        # Stop if we've reached the overall max
        if len(all_feasible_points) >= max_points:
            all_feasible_points = all_feasible_points[:max_points]
            break

    # If we found feasible points, return them
    if all_feasible_points:
        print(f"Total: {len(all_feasible_points)} well-separated feasible goal points across all instances")
        print(f"(Minimum separation: {min_point_separation} pixels)")
        return all_feasible_points

    # There are no safe points found
    print("No safe goal points found around the target!")
    return []


def draw_path_on_map(
    map_image: np.ndarray,
    path: List[Tuple[int, int]],
    start: Tuple[int, int],
    goals: List[Tuple[int, int]],
    explored_nodes: List[Tuple[int, int]],
    explored_edges: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    output_path: str = 'path_map.png'
) -> np.ndarray:
    """
    Draw the RRT exploration and final path on the map.
    
    Args:
        map_image: Original map image.
        path: List of (x, y) tuples representing the final path (red).
        start: Start position (x, y) - green circle.
        goals: List of goal positions (x, y) - light blue circles for candidates, dark blue for reached.
        explored_nodes: All explored nodes during RRT - purple squares.
        explored_edges: All explored edges during RRT - black thin lines.
        output_path: Path to save the output image.
    
    Returns:
        The map image with path drawn.
    """
    result_image = map_image.copy()

    # Draw explored edges (black thin lines)
    for pt1, pt2 in explored_edges:
        cv2.line(result_image, pt1, pt2, (0, 0, 0), 2)  # Black

    # Draw final path (red lines)
    for i in range(len(path) - 1):
        pt1 = tuple(path[i])
        pt2 = tuple(path[i + 1])
        cv2.line(result_image, pt1, pt2, (0, 0, 255), 5)  # Red

    # Draw explored nodes (purple squares)
    for node in explored_nodes:
        cv2.rectangle(
            result_image,
            (node[0] - 5, node[1] - 5),
            (node[0] + 5, node[1] + 5),
            (128, 0, 128), -1  # Purple
        )

    # Draw candidate goal points
    reached_goal = tuple(path[-1]) if path else None
    for goal in goals:
        if goal == reached_goal:
            # Draw the reached goal in light blue (larger)
            cv2.circle(result_image, goal, 15, (255, 200, 0), -1)  # Light Blue
        else:
            # Draw candidate goals in light blue (smaller)
            cv2.circle(result_image, goal, 10, (255, 200, 0), 2)  # Light Blue (Hollow)

    # Draw start point (green circle)
    cv2.circle(result_image, start, 15, (0, 255, 0), -1)  # Green

    cv2.imwrite(output_path, result_image)
    print(f"Path map saved to {output_path}")

    return result_image
