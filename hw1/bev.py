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

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
        Project the top view pixels to the front view pixels.
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
        ])
        # Inverse of intrinsic matrix (K^-1)
        intrinsic_matrix_inverse = np.linalg.inv(intrinsic_matrix)

        # Extrinsic parameters
        rotation_front_to_world = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        translation_front_to_world = np.array([
            [0],
            [1],
            [0]
        ])
        rotation_bev_to_world = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        translation_bev_to_world = np.array([
            [0],
            [2.5],
            [0]
        ])

        # Convert all points to homogeneous coordinates
        pixels_homogeneous = np.hstack([points_np, np.ones((points_np.shape[0], 1))])

        # Convert the pixel coordinates to camera direction vectors
        camera_direction_vectors = intrinsic_matrix_inverse.dot(pixels_homogeneous.T).T

        # Convert the camera direction vectors to world coordinates
        world_direction_vectors = rotation_bev_to_world.dot(camera_direction_vectors.T).T

        # Compute scaling factors (lambda) for intersection with ground y = 0
        scale_values = -translation_bev_to_world[1] / world_direction_vectors[:, 1]

        # Compute world points on the ground plane
        world_points = translation_bev_to_world.T + (world_direction_vectors.T * scale_values).T

        # Convert world coordinates to front camera coordinates
        points_front_camera = (rotation_front_to_world.T.dot((world_points - translation_front_to_world.T).T)).T

        # Project the front camera coordinates to pixel coordinates
        projected_pixels_homogeneous = intrinsic_matrix.dot(points_front_camera.T).T

        # Convert homogeneous coordinates to 2D pixel coordinates
        projected_pixels = projected_pixels_homogeneous[:, :2] / projected_pixels_homogeneous[:, 2][:, np.newaxis]

        return projected_pixels.astype(int).tolist()

    def show_image(self, new_pixels: List[List[int]], img_name='projection.png', color=(0, 0, 255), alpha=0.4):
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
    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.imwrite('output/selected_pixels_front1.png', img)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb)
    new_transform_pixels = projection.top_to_front(theta=pitch_ang)
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
    new_transform_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_transform_pixels, img_name='output/projection_front2.png')
