import argparse
import json
import os
import time
from typing import Union, List, Tuple, Optional

import numpy as np
# for simulator
import pybullet as p  # type: ignore
import pybullet_data
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation as R
from typing_extensions import TypedDict

# you may use your forward kinematic algorithm to compute
from fk import your_fk, get_ur5_dh_params
# for geometry information
from hw3_utils.bullet_utils import draw_coordinate
from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env

SIM_TIMESTEP = 1.0 / 240.0
TASK2_SCORE_MAX = 40
IK_ERROR_THRESH = 0.02

# Set to True to use Damped Least Squares method in your_ik function
USE_DLS = True

class InverseKinematicTestcaseDict(TypedDict):
    current_joint_poses: List[List[float]] # list of joint angles (6 DoF)
    next_poses: List[List[float]] # list of target end-effector poses (x, y, z, qx, qy, qz, qw)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate the cross product between two 3D vectors

    Args:
        a (np.ndarray): first vector
        b (np.ndarray): second vector
    Returns:
        np.ndarray: cross product
    """
    return np.cross(a, b)


# this is the pybullet version
def pybullet_ik(
    robot_id: int,
    new_pose: Union[List[float], Tuple[float, float, float, float, float, float], np.ndarray],
    max_iters: int = 1000,
    stop_thresh: float = .001,
    base_pos: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Pybullet Inverse Kinematic Solver of the robot.
    Compute the joint angles given the target end-effector pose.

    Args:
        robot_id (int): the unique ID of the robot in pybullet
        new_pose (Union[List[float], Tuple[float, float, float, float, float, float], np.ndarray]):
            target end-effector pose (position + orientation as quaternion)
        max_iters (int): maximum number of iterations
        stop_thresh (float): stopping threshold for the error norm
        base_pos (Optional[List[float]]): base position of the robot (not used here)

    Returns:
        np.ndarray: computed joint angles (6 DoF)
    """
    new_pos, new_rot = new_pose[:3], new_pose[3:]

    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=10,
        targetPosition=new_pos,
        targetOrientation=new_rot,
        lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
        upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
        jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
        restPoses=np.float32(np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi).tolist(),
        maxNumIterations=max_iters,
        residualThreshold=stop_thresh)
    joint_poses = np.array(joint_poses)

    return joint_poses


def your_ik(
    robot_id: int,
    new_pose: Union[List[float], Tuple[float, float, float, float, float, float], np.ndarray],
    max_iters: int = 1000,
    stop_thresh: float = .001,
    base_pos: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Your Inverse Kinematic Solver of the robot.
    Compute the joint angles given the target end-effector pose.
    Args:
        robot_id (int): the unique ID of the robot in pybullet
        new_pose (Union[List[float], Tuple[float, float, float, float, float, float], np.ndarray]):
            target end-effector pose (position + orientation as quaternion)
        max_iters (int): maximum number of iterations
        stop_thresh (float): stopping threshold for the error norm
        base_pos (Optional[List[float]]): base position of the robot (not used here)
    Returns:
        np.ndarray: computed joint angles (6 DoF)
    """
    if USE_DLS:
        print("Using Damped Least Squares method for IK")
        # For comparison purposes
        return your_ik_damped_least_squares(
            robot_id,
            new_pose,
            max_iters,
            stop_thresh,
            base_pos
        )

    print("Using Pseudo-Inverse method for IK")

    joint_limits = np.asarray([
        [-3 * np.pi / 2, -np.pi / 2],  # joint1
        [-2.3562, -1],  # joint2
        [-17, 17],  # joint3
        [-17, 17],  # joint4
        [-17, 17],  # joint5
        [-17, 17],  # joint6
    ])

    # get current joint angles and gripper pos, (gripper pos is fixed)
    num_q = p.getNumJoints(robot_id)
    q_states = p.getJointStates(robot_id, range(0, num_q))

    tmp_q = np.asarray([x[0] for x in q_states][2:8])  # current joint angles 6d (You only need to modify this)

    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 2 : Compute Inverse-Kinematic Solver of the robot by yourself.      --- #
    # ---          Try to implement `your_ik` function WITHOUT using ANY pybullet  --- #
    # ---          API. (40% for accuracy)                                         --- #
    # --- Note : please modify the code in `your_ik` function.                     --- #
    # -------------------------------------------------------------------------------- #

    #### your code ################################
    # TODO: update tmp_q
    # tmp_q = ? # may be more than one line

    # hint : 
    # 1. You may use `your_fk` function and jacobian matrix to do this
    # 2. Be careful when computing the delta x
    # 3. You may use some hyperparameters (i.e., step rate) in optimization loops
    ###############################################
    # Convert new_pose to numpy array
    new_pose = np.array(new_pose)

    # Get the Denavit–Hartenberg params for the UR5 robot
    dh_params = get_ur5_dh_params()

    # Set step rate for joint updates
    step_rate = 0.5

    # iterative optimization
    for i in range(max_iters):
        # Forward kinematics to get current end-effector pose
        pose_7d_current, j_matrix = your_fk(dh_params, tmp_q, base_pos)

        # Calculate position error
        position_current = pose_7d_current[:3]
        position_target = new_pose[:3]
        position_error = position_target - position_current

        # Calculate orientation error
        quaternion_current = pose_7d_current[3:]
        quaternion_target = new_pose[3:]
        rotation_current = R.from_quat(quaternion_current)
        rotation_target = R.from_quat(quaternion_target)
        rotation_error_vector = (rotation_target * rotation_current.inv()).as_rotvec()

        # Combine position and orientation errors
        delta_x = np.concatenate((position_error, rotation_error_vector))

        # Calculate the Jacobian Pseudo-Inverse
        jacobian_pseudo_inverse = np.array(pinv(j_matrix))

        # Compute the required joint angle changes
        delta_q = jacobian_pseudo_inverse @ delta_x

        # Update joint angles
        tmp_q += step_rate * delta_q

        # Enforce joint limits to keep angles within the physical range
        tmp_q = np.clip(tmp_q, joint_limits[:, 0], joint_limits[:, 1])

        # Check for convergence
        if np.linalg.norm(delta_x) < stop_thresh:
            break
    ###############################################

    return np.array(tmp_q)  # 6 DoF


def your_ik_damped_least_squares(
    robot_id: int,
    new_pose: Union[List[float], Tuple[float, float, float, float, float, float], np.ndarray],
    max_iters: int = 1000,
    stop_thresh: float = .001,
    base_pos: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Your Inverse Kinematic Solver of the robot using Damped Least Squares method.
    Compute the joint angles given the target end-effector pose.
    Args:
        robot_id (int): the unique ID of the robot in pybullet
        new_pose (Union[List[float], Tuple[float, float, float, float, float, float], np.ndarray]):
            target end-effector pose (position + orientation as quaternion)
        max_iters (int): maximum number of iterations
        stop_thresh (float): stopping threshold for the error norm
        base_pos (Optional[List[float]]): base position of the robot (not used here)
    Returns:
        np.ndarray: computed joint angles (6 DoF)
    """
    joint_limits = np.asarray([
        [-3 * np.pi / 2, -np.pi / 2],  # joint1
        [-2.3562, -1],  # joint2
        [-17, 17],  # joint3
        [-17, 17],  # joint4
        [-17, 17],  # joint5
        [-17, 17],  # joint6
    ])

    # get current joint angles and gripper pos, (gripper pos is fixed)
    num_q = p.getNumJoints(robot_id)
    q_states = p.getJointStates(robot_id, range(0, num_q))

    tmp_q = np.asarray([x[0] for x in q_states][2:8])  # current joint angles 6d (You only need to modify this)

    #### your code ################################
    # Convert new_pose to numpy array
    new_pose = np.array(new_pose)

    # Get the Denavit–Hartenberg params for the UR5 robot
    dh_params = get_ur5_dh_params()

    # Set step rate for joint updates
    step_rate = 0.5

    # Set the damping factor (lambda) for Damped Least Squares
    damping_lambda = 0.01

    # iterative optimization
    for i in range(max_iters):
        # Forward kinematics to get current end-effector pose
        pose_7d_current, j_matrix = your_fk(dh_params, tmp_q, base_pos)

        # Calculate position error
        position_current = pose_7d_current[:3]
        position_target = new_pose[:3]
        position_error = position_target - position_current

        # Calculate orientation error
        quaternion_current = pose_7d_current[3:]
        quaternion_target = new_pose[3:]
        rotation_current = R.from_quat(quaternion_current)
        rotation_target = R.from_quat(quaternion_target)
        rotation_error_vector = (rotation_target * rotation_current.inv()).as_rotvec()

        # Combine position and orientation errors
        delta_x = np.concatenate((position_error, rotation_error_vector))

        # Get Jacobian Transpose
        jacobian_transpose = j_matrix.T

        # Get the identity matrix (I), (6x6 for a 6-DOF task space)
        identity_matrix = np.identity(j_matrix.shape[0])

        # Calculate the term to invert
        term_to_invert = j_matrix @ jacobian_transpose + (damping_lambda ** 2) * identity_matrix

        # Invert the term
        term_inverted = np.linalg.inv(term_to_invert)

        # Compute the final joint angle changes
        delta_q = jacobian_transpose @ term_inverted @ delta_x

        # Update joint angles
        tmp_q += step_rate * delta_q

        # Enforce joint limits to keep angles within the physical range
        tmp_q = np.clip(tmp_q, joint_limits[:, 0], joint_limits[:, 1])

        # Check for convergence
        if np.linalg.norm(delta_x) < stop_thresh:
            break
    ###############################################
    return np.array(tmp_q)  # 6 DoF


# TODO: [for your information]
# This function is the scoring function, we will use the same code 
# to score your algorithm using all the testcases
def score_ik(
    robot: ur5Env,
    testcase_files: List[str],
    visualize: bool = False
):
    """
    Score your Inverse Kinematic function.
    This will compare your results with ground truth data stored in the testcase files
    And give you a score based on the accuracy of your results

    Args:
        robot (ur5Env): the ur5 robot environment
        testcase_files (List[str]): list of testcase file paths
        visualize (bool): whether visualize the end-effector poses
    """
    testcase_file_num = len(testcase_files)
    ik_score = [TASK2_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
    ik_error_cnt = [0 for _ in range(testcase_file_num)]

    p.addUserDebugText(
        text="Scoring Your Inverse Kinematic Algorithm ...",
        textPosition=[0.1, -0.6, 1.5],
        textColorRGB=[1, 1, 1],
        textSize=1.0,
        lifeTime=0
    )

    print("============================ Task 2 : Inverse Kinematic ============================\n")
    for file_id, testcase_file in enumerate(testcase_files):

        f_in = open(testcase_file, 'r')
        ik_dict: InverseKinematicTestcaseDict = json.load(f_in)
        f_in.close()

        test_case_name = os.path.split(testcase_file)[-1]

        poses = ik_dict['next_poses']
        cases_num = len(ik_dict['current_joint_poses'])

        penalty = (TASK2_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)
        ik_errors = []

        for i in range(cases_num):

            # TODO: check your default arguments of `max_iters` and `stop_thresh` are your best parameters.
            #       We will only pass default arguments of your `max_iters` and `stop_thresh`.
            your_joint_poses = your_ik(robot.robot_id, poses[i], base_pos=robot._base_position)

            # You can use `pybullet_ik` to see the correct version 
            # your_joint_poses = pybullet_ik(robot.robot_id, poses[i])

            gt_pose = poses[i]

            p.setJointMotorControlArray(bodyUniqueId=robot.robot_id,
                                        jointIndices=robot._joint_name_to_ids.values(),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=your_joint_poses,
                                        positionGains=[0.2] * len(your_joint_poses),
                                        velocityGains=[1] * len(your_joint_poses),
                                        physicsClientId=robot._physics_client_id)

            # warmup for 0.1 sec
            for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                p.stepSimulation()
                time.sleep(SIM_TIMESTEP)

            your_pose = robot.get_eef_pose()

            if visualize:
                color_yours = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
                color_gt = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
                draw_coordinate(your_pose, size=0.01, color=color_yours)
                draw_coordinate(gt_pose, size=0.01, color=color_gt)

            ik_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
            ik_errors.append(ik_error)
            if ik_error > IK_ERROR_THRESH:
                ik_score[file_id] -= penalty
                ik_error_cnt[file_id] += 1

        ik_score[file_id] = 0.0 if ik_score[file_id] < 0.0 else ik_score[file_id]
        ik_errors = np.asarray(ik_errors)

        score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                    "- Mean Error : {:0.06f}\n".format(np.mean(ik_errors)) + \
                    "- Error Count : {:3d} / {:3d}\n".format(ik_error_cnt[file_id], cases_num) + \
                    "- Your Score Of Inverse Kinematic : {:00.03f} / {:00.03f}\n".format(
                        ik_score[file_id], TASK2_SCORE_MAX / testcase_file_num)

        print(score_msg)
    p.removeAllUserDebugItems()

    total_ik_score = 0.0
    for file_id in range(testcase_file_num):
        total_ik_score += ik_score[file_id]

    print("====================================================================================")
    print("- Your Total Score : {:00.03f} / {:00.03f}".format(total_ik_score, TASK2_SCORE_MAX))
    print("====================================================================================")


def main(args: argparse.Namespace):
    # ------------------------ #
    # --- Setup simulation --- #
    # ------------------------ #

    # Create pybullet GUI
    physics_client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=90,
        cameraPitch=0,
        cameraTargetPosition=[0.5, 0.0, 1.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(SIM_TIMESTEP)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [0.9, 0.0, 0.0])

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # goto initial pose
    from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env
    robot = ur5Env(physics_client_id, use_IK=1)

    # -------------------------------------------- #
    # --- Test your Forward Kinematic function --- #
    # -------------------------------------------- #

    # warmup for 1 sec
    for _ in range(int(1 / SIM_TIMESTEP * 1)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)

    # ------------------------------------------------------------------ #
    # --- Test your Inverse Kinematic function using one target pose --- #
    # ------------------------------------------------------------------ #

    # warmup for 2 secs
    p.addUserDebugText(text="Warmup for 2 secs ...",
                       textPosition=[0.1, -0.6, 1.5],
                       textColorRGB=[1, 1, 1],
                       textSize=1.0,
                       lifeTime=0)
    for _ in range(int(1 / SIM_TIMESTEP * 2)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    p.removeAllUserDebugItems()

    # test your ik solver
    testcase_files = [
        'test_case/ik_test_case_easy.json',
        'test_case/ik_test_case_medium.json',
        'test_case/ik_test_case_hard.json',
        # 'test_case/ik_test_case_ta1.json',
        # 'test_case/ik_test_case_ta2.json',
    ]

    # ------------------------------------------------------------- #
    # --- Test your Inverse Kinematic function using test cases --- #
    # ------------------------------------------------------------- #

    # scoring your algorithm
    score_ik(robot, testcase_files, visualize=args.visualize_pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize-pose', '-vp', action='store_true', default=False,
                        help='whether show the poses of end effector')
    args = parser.parse_args()
    main(args)
