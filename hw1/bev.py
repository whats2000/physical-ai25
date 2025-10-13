import os
from typing import List

import cv2
import numpy as np

points: List[List[int]] = []


class Projection(object):

    def __init__(self, image_path: str):
        """
        :param image_path: Path to the perspective(front) view image or the image array
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, alpha=0.0, beta=0.0, gamma=0.0, dx=0.0, dy=0.0, dz=0.0, fov=90.0) -> List[List[int]]:
        """
        Project the top view pixels to the front view pixels.
        :param alpha: Rotation angle around z-axis (in degrees)
        :param beta: Rotation angle around y-axis (in degrees)
        :param gamma: Rotation angle around x-axis (in degrees)
        :param dx: Translation along x-axis
        :param dy: Translation along y-axis
        :param dz: Translation along z-axis
        :param fov: Field of view of the camera (in degrees)
        :return: New pixels on perspective(front) view image
        """
        # Convert the points to numpy array
        points_np = np.array(points)

        # Intrinsic parameters
        focal_length_x = (self.width / 2) / np.tan(np.radians(fov) / 2)  # fx
        focal_length_y = (self.height / 2) / np.tan(np.radians(fov) / 2)  # fy
        center_x = self.width / 2  # cx
        center_y = self.height / 2  # cy
        # Intrinsic matrix (K)
        intrinsic_matrix = np.array([
            [focal_length_x, 0, center_x],
            [0, focal_length_y, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        # Inverse of intrinsic matrix (K^-1)
        intrinsic_matrix_inverse = np.linalg.inv(intrinsic_matrix)

        # Extrinsic parameters (For front view camera)
        rotation_world_to_front = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        translation_world_to_front = np.array([
            [0.0],
            [1.0],
            [0.0],
        ], dtype=np.float32)

        # Extrinsic parameters (For bev view camera)
        rotation_world_to_bev_z = np.array([
            [np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0],
            [np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        rotation_world_to_bev_y = np.array([
            [np.cos(np.radians(beta)), 0, np.sin(np.radians(beta))],
            [0, 1, 0],
            [-np.sin(np.radians(beta)), 0, np.cos(np.radians(beta))]
        ], dtype=np.float32)
        rotation_world_to_bev_x = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(gamma)), -np.sin(np.radians(gamma))],
            [0, np.sin(np.radians(gamma)), np.cos(np.radians(gamma))]
        ], dtype=np.float32)
        rotation_world_to_bev = rotation_world_to_bev_z @ rotation_world_to_bev_y @ rotation_world_to_bev_x
        translation_world_to_bev = np.array([
            [dx],
            [dy],
            [dz],
        ], dtype=np.float32)

        # Add the z-axis to the bev points
        # [u, v] -> [u, v, 1]
        num_points = points_np.shape[0]
        bev_points_homogeneous = np.hstack((points_np, np.ones((num_points, 1), dtype=np.float32)))

        # Convert the bev points to the bev camera coordinate system
        bev_camera_points = (intrinsic_matrix_inverse @ bev_points_homogeneous.T).T

        # Convert the camera direction vectors to world coordinates
        world_direction_vectors = (np.linalg.inv(rotation_world_to_bev) @ bev_camera_points.T).T

        # Find intersection with ground plane
        scale_values = -translation_world_to_bev[1, 0] / world_direction_vectors[:, 1]

        # Compute world points on the ground plane
        world_points_3d = translation_world_to_bev.T + (world_direction_vectors.T * scale_values).T

        # Convert the world points to the front camera coordinate system
        front_camera_points = (rotation_world_to_front @ (world_points_3d.T - translation_world_to_front)).T

        # Project the front camera points to the front image plane
        front_image_points_homogeneous = (intrinsic_matrix @ front_camera_points.T).T

        # Convert homogeneous coordinates to 2D pixel coordinates
        # [u', v', w] -> [u'/w, v'/w] = [u, v]
        front_image_points = front_image_points_homogeneous[:, :2] / front_image_points_homogeneous[:, 2:]

        return front_image_points.astype(np.int32).tolist()

    def show_image(self, new_pixels: List[List[int]], img_name='projection.png', color=(0, 0, 255),
                   alpha=0.4) -> np.ndarray:
        """
        Show the projection result and fill the selected area on perspective(front) view image.
        """
        if len(new_pixels) == 0:
            print("No points to project!")
            return self.image

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event: 'cv2.MouseEventTypes', x: int, y: int, _flags, _params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        points.append([x, y])
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":
    pitch_ang = 90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"
    
    # If output directory does not exist, create it
    if not os.path.exists('output'):
        os.makedirs('output')

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.imwrite('output/selected_pixels_front1.png', img)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb)
    new_transform_pixels = projection.top_to_front(gamma=pitch_ang, dy=2.5, fov=90.0)
    projection.show_image(new_transform_pixels, img_name='output/projection_front1.png')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    front_rgb = "bev_data/front2.png"
    top_rgb = "bev_data/bev2.png"
    points.clear()

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.imwrite('output/selected_pixels_front2.png', img)

    projection = Projection(front_rgb)
    new_transform_pixels = projection.top_to_front(gamma=pitch_ang, dy=2.5, fov=90.0)
    projection.show_image(new_transform_pixels, img_name='output/projection_front2.png')
