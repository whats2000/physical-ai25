import numpy as np
import cv2
from typing import Tuple, List, Optional


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
        goal_sample_rate: float = 0.15,
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
        self.explored_nodes = []  # Track all explored nodes
        self.explored_edges = []  # Track all explored edges

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
                if 0 <= check_x < self.map_width and 0 <= check_y < self.map_height:
                    if self.occupancy_map[check_y, check_x] == 0:
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

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find a path from start to goal using RRT.
        
        Args:
            start: Start position (x, y) in pixels.
            goal: Goal position (x, y) in pixels.
        
        Returns:
            List of (x, y) tuples representing the path, or None if no path found.
        """
        start_point = np.array(start, dtype=float)
        goal_point = np.array(goal, dtype=float)

        if not self._is_collision_free(start_point):
            print(f"Start point {start} is not collision-free!")
            return None

        if not self._is_collision_free(goal_point):
            print(f"Goal point {goal} is not collision-free!")
            return None

        # Initialize tree
        root = TreeNode(start_point)
        nodes = [root]

        goal_threshold = self.step_size * 1.5  # 1.5x step size for goal reach

        for iteration in range(self.max_iterations):
            if (iteration + 1) % 500 == 0:
                # Print progress every 500 iterations
                print(f"RRT iteration {iteration + 1}/{self.max_iterations}")

            # Sample random point or goal
            if np.random.random() < self.goal_sample_rate:
                random_point = goal_point
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

            # Check if we reached the goal
            distance_to_goal = np.linalg.norm(new_node.position - goal_point)
            if distance_to_goal >= goal_threshold:
                # Not close enough to goal yet
                continue

            # Check if path to goal is collision-free
            if not self._is_path_collision_free(new_node.position, goal_point):
                # The path to goal is not collision-free, we continue searching
                continue

            # Try connecting directly to the goal
            goal_node = TreeNode(goal_point, parent=new_node)
            nodes.append(goal_node)

            # Extract path
            path = self._extract_path(goal_node)
            print(f"Path found in {iteration + 1} iterations with {len(path)} waypoints")
            return path

        print("No path found!")
        return None


def find_target_point_on_map(
    map_image: np.ndarray,
    target_color: Tuple[int, int, int],
    offset_distance: int = 40
) -> Optional[Tuple[int, int]]:
    """
    Find a safe point in front of the target item.
    
    Args:
        map_image: The semantic map image.
        target_color: RGB color of the target item.
        offset_distance: Distance to move away from the target.
    
    Returns:
        A point (x, y) in front of the target, or None if target not found.
    """
    # Convert BGR to RGB and find target pixels
    target_bgr = (target_color[2], target_color[1], target_color[0])
    mask = cv2.inRange(map_image, np.array(target_bgr), np.array(target_bgr))

    # Get coordinates of target color
    target_coords = np.where(mask)
    if len(target_coords[0]) == 0:
        print(f"Target color {target_color} not found on map!")
        return None

    # Calculate centroid
    center_y = int(np.mean(target_coords[0]))
    center_x = int(np.mean(target_coords[1]))
    print(f"Target found at ({center_x}, {center_y})")

    # Create occupancy map with safety margin
    gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    occupancy_map = (gray > 200).astype(np.uint8)

    # Add safety margin by eroding free space
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    safe_occupancy = cv2.erode(occupancy_map, kernel, iterations=1)

    # Try different directions with increasing distance
    directions = [
        (offset_distance, 0), (-offset_distance, 0),
        (0, offset_distance), (0, -offset_distance),
        (offset_distance, offset_distance), (-offset_distance, offset_distance),
        (offset_distance, -offset_distance), (-offset_distance, -offset_distance)
    ]

    for dx, dy in directions:
        new_x = center_x + dx
        new_y = center_y + dy

        # Check if within the map bounds
        if (0 > new_x or new_x >= safe_occupancy.shape[1]) or (0 > new_y or new_y >= safe_occupancy.shape[0]):
            # Skip out-of-bounds points
            continue

        # Check if the new point is in free space
        if safe_occupancy[new_y, new_x] != 1:
            # Skip occupied points
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
                    # We skip out-of-bounds checks
                    continue

                # Check occupancy
                if safe_occupancy[check_y, check_x] == 0:
                    # We found an occupied cell in the surrounding area, we will consider this point unsafe
                    is_safe = False
                    break

            if not is_safe:
                # Early exit if already unsafe
                break

        if is_safe:
            # Early return if a safe point is found
            print(f"Goal point found at ({new_x}, {new_y})")
            return new_x, new_y

    # There is no safe point found
    print("No safe goal point found in front of the target!")
    return None


def draw_path_on_map(
    map_image: np.ndarray,
    path: List[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
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
        goal: Goal position (x, y) - blue circle.
        explored_nodes: All explored nodes during RRT - purple squares.
        explored_edges: All explored edges during RRT - black thin lines.
        output_path: Path to save the output image.
    
    Returns:
        The map image with path drawn.
    """
    result_image = map_image.copy()

    # 1. Draw explored edges (black thin lines)
    for pt1, pt2 in explored_edges:
        cv2.line(result_image, pt1, pt2, (0, 0, 0), 2)  # Black

    # 2. Draw final path (red lines)
    for i in range(len(path) - 1):
        pt1 = tuple(path[i])
        pt2 = tuple(path[i + 1])
        cv2.line(result_image, pt1, pt2, (0, 0, 255), 5)  # Red

    # 3. Draw explored nodes (purple squares)
    for node in explored_nodes:
        cv2.rectangle(
            result_image,
            (node[0] - 5, node[1] - 5),
            (node[0] + 5, node[1] + 5),
            (128, 0, 128), -1  # Purple
        )

    # 4. Draw start point (green circle)
    cv2.circle(result_image, start, 15, (0, 255, 0), -1)  # Green

    # 5. Draw goal point (blue circle)
    cv2.circle(result_image, goal, 15, (255, 0, 0), -1)  # Blue

    cv2.imwrite(output_path, result_image)
    print(f"Path map saved to {output_path}")

    return result_image
