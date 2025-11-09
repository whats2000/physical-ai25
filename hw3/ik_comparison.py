"""
IK Comparison Script
This script compares three IK methods:
1. Your IK (Pseudo-inverse)
2. Your IK with Damped Least Squares (DLS)
3. PyBullet built-in IK

It generates comparison plots for:
- Position errors
- Orientation errors
- Joint velocities
- Computation time
- Convergence iterations
"""

import argparse
import json
import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import pybullet as p # type: ignore
import pybullet_data
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation as R

from fk import your_fk, get_ur5_dh_params
from hw3_utils.bullet_utils import draw_coordinate
from pybullet_robot_envs.envs.ur5_envs.ur5_env import ur5Env

SIM_TIMESTEP = 1.0 / 240.0


def your_ik_with_stats(
    robot_id: int,
    new_pose: np.ndarray,
    max_iters: int = 1000,
    stop_thresh: float = .001,
    base_pos: List[float] = None,
    method: str = "pseudoinverse"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Your IK solver with statistics tracking
    
    Args:
        robot_id: PyBullet robot ID
        new_pose: Target pose [x, y, z, qx, qy, qz, qw]
        max_iters: Maximum iterations
        stop_thresh: Convergence threshold
        base_pos: Base position of robot
        method: "pseudoinverse" or "dls"
    
    Returns:
        Joint angles and statistics dictionary
    """
    joint_limits = np.asarray([
        [-3 * np.pi / 2, -np.pi / 2],
        [-2.3562, -1],
        [-17, 17],
        [-17, 17],
        [-17, 17],
        [-17, 17],
    ])
    
    # Get current joint angles
    num_q = p.getNumJoints(robot_id)
    q_states = p.getJointStates(robot_id, range(0, num_q))
    tmp_q = np.asarray([x[0] for x in q_states][2:8])
    
    # Statistics tracking
    stats = {
        'iterations': 0,
        'position_errors': [],
        'orientation_errors': [],
        'total_errors': [],
        'joint_velocities': [],
        'computation_time': 0.0
    }
    
    dh_params = get_ur5_dh_params()
    step_rate = 0.5
    damping_lambda = 0.1 if method == "dls" else 0.0
    
    start_time = time.time()
    
    for i in range(max_iters):
        # Forward kinematics
        pose_7d_current, j_matrix = your_fk(dh_params, tmp_q, base_pos)
        
        # Position error
        position_current = pose_7d_current[:3]
        position_target = new_pose[:3]
        position_error = position_target - position_current
        
        # Orientation error
        quaternion_current = pose_7d_current[3:]
        quaternion_target = new_pose[3:]
        rotation_current = R.from_quat(quaternion_current)
        rotation_target = R.from_quat(quaternion_target)
        rotation_error_vector = (rotation_target * rotation_current.inv()).as_rotvec()
        
        # Combined error
        delta_x = np.concatenate((position_error, rotation_error_vector))
        
        # Calculate delta_q based on method
        if method == "dls":
            # Damped Least Squares
            jacobian_transpose = j_matrix.T
            identity_matrix = np.identity(j_matrix.shape[0])
            term_to_invert = j_matrix @ jacobian_transpose + (damping_lambda ** 2) * identity_matrix
            term_inverted = np.linalg.inv(term_to_invert)
            delta_q = jacobian_transpose @ term_inverted @ delta_x
        else:
            # Pseudo-inverse
            jacobian_pseudo_inverse = pinv(j_matrix)
            delta_q = jacobian_pseudo_inverse @ delta_x
        
        # Update joint angles
        tmp_q_old = tmp_q.copy()
        tmp_q += step_rate * delta_q
        tmp_q = np.clip(tmp_q, joint_limits[:, 0], joint_limits[:, 1])
        
        # Calculate joint velocity (change per iteration)
        joint_velocity = np.linalg.norm(tmp_q - tmp_q_old)
        
        # Store statistics
        stats['position_errors'].append(np.linalg.norm(position_error))
        stats['orientation_errors'].append(np.linalg.norm(rotation_error_vector))
        stats['total_errors'].append(np.linalg.norm(delta_x))
        stats['joint_velocities'].append(joint_velocity)
        stats['iterations'] = i + 1
        
        # Check convergence
        if np.linalg.norm(delta_x) < stop_thresh:
            break
    
    stats['computation_time'] = time.time() - start_time
    
    return tmp_q, stats


def pybullet_ik_with_stats(
    robot_id: int,
    new_pose: np.ndarray,
    max_iters: int = 1000,
    stop_thresh: float = .001,
    base_pos: List[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    PyBullet IK with statistics tracking
    """
    stats = {
        'iterations': max_iters,  # PyBullet doesn't expose this
        'position_errors': [],
        'orientation_errors': [],
        'total_errors': [],
        'joint_velocities': [],
        'computation_time': 0.0
    }
    
    new_pos, new_rot = new_pose[:3], new_pose[3:]
    
    start_time = time.time()
    
    joint_poses = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=10,
        targetPosition=new_pos,
        targetOrientation=new_rot,
        lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
        upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
        jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],
        restPoses=np.float32(np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi).tolist(),
        maxNumIterations=max_iters,
        residualThreshold=stop_thresh
    )
    
    stats['computation_time'] = time.time() - start_time
    joint_poses = np.array(joint_poses)
    
    return joint_poses, stats


def compare_ik_methods(
    robot: ur5Env,
    testcase_file: str,
    output_dir: str = "comparison_results"
):
    """
    Compare IK methods and generate plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test cases
    with open(testcase_file, 'r') as f:
        ik_dict = json.load(f)
    
    test_case_name = os.path.splitext(os.path.basename(testcase_file))[0]
    poses = ik_dict['next_poses']
    cases_num = min(len(poses), 20)  # Limit to 20 cases for clearer plots
    
    # Storage for results
    results = {
        'pseudoinverse': {
            'errors': [],
            'position_errors': [],
            'orientation_errors': [],
            'times': [],
            'iterations': [],
            'avg_velocities': [],
            'final_poses': []
        },
        'dls': {
            'errors': [],
            'position_errors': [],
            'orientation_errors': [],
            'times': [],
            'iterations': [],
            'avg_velocities': [],
            'final_poses': []
        },
        'pybullet': {
            'errors': [],
            'position_errors': [],
            'orientation_errors': [],
            'times': [],
            'iterations': [],
            'avg_velocities': [],
            'final_poses': []
        }
    }
    
    print(f"\n{'='*80}")
    print(f"Comparing IK Methods on: {test_case_name}")
    print(f"Number of test cases: {cases_num}")
    print(f"{'='*80}")
    print(f"\n{'Method Configurations:':<30}")
    print(f"  1. Pseudo-inverse:     Standard Jacobian pseudo-inverse (pinv)")
    print(f"  2. DLS (λ=0.1):        Damped Least Squares with damping factor 0.1")
    print(f"  3. PyBullet:           Built-in IK solver (reference)")
    print()
    
    for i in range(cases_num):
        target_pose = np.array(poses[i])
        
        print(f"Test case {i+1}/{cases_num}...", end='\r')
        
        # Test each method
        for method_name, method_key in [
            ("Pseudo-inverse", "pseudoinverse"),
            ("Damped Least Squares", "dls"),
            ("PyBullet", "pybullet")
        ]:
            # Reset robot to initial pose
            initial_joint_poses = ik_dict['current_joint_poses'][i]
            for joint_idx, joint_id in enumerate(robot._joint_name_to_ids.values()):
                p.resetJointState(robot.robot_id, joint_id, initial_joint_poses[joint_idx])
            
            # Compute IK
            if method_key == "pybullet":
                joint_angles, stats = pybullet_ik_with_stats(
                    robot.robot_id, target_pose, base_pos=robot._base_position
                )
            else:
                joint_angles, stats = your_ik_with_stats(
                    robot.robot_id, target_pose, base_pos=robot._base_position, method=method_key
                )
            
            # Set joint positions and simulate
            p.setJointMotorControlArray(
                bodyUniqueId=robot.robot_id,
                jointIndices=robot._joint_name_to_ids.values(),
                controlMode=p.POSITION_CONTROL,
                targetPositions=joint_angles,
                positionGains=[0.2] * len(joint_angles),
                velocityGains=[1] * len(joint_angles),
                physicsClientId=robot._physics_client_id
            )
            
            # Simulate
            for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                p.stepSimulation()
            
            # Get final pose
            final_pose = robot.get_eef_pose()
            
            # Calculate errors
            pos_error = np.linalg.norm(final_pose[:3] - target_pose[:3])
            
            # Orientation error
            quat_final = final_pose[3:]
            quat_target = target_pose[3:]
            rot_final = R.from_quat(quat_final)
            rot_target = R.from_quat(quat_target)
            ori_error = np.linalg.norm((rot_target * rot_final.inv()).as_rotvec())
            
            total_error = np.linalg.norm(final_pose - target_pose)
            
            # Store results
            results[method_key]['errors'].append(total_error)
            results[method_key]['position_errors'].append(pos_error)
            results[method_key]['orientation_errors'].append(ori_error)
            results[method_key]['times'].append(stats['computation_time'])
            results[method_key]['iterations'].append(stats['iterations'])
            
            # Average velocity
            if stats['joint_velocities']:
                avg_vel = np.mean(stats['joint_velocities'])
            else:
                avg_vel = 0.0
            results[method_key]['avg_velocities'].append(avg_vel)
            results[method_key]['final_poses'].append(final_pose)
    
    print(f"\nTest cases completed!{' '*30}")
    
    # Generate plots
    generate_comparison_plots(results, cases_num, test_case_name, output_dir)
    
    # Print summary statistics
    print_summary_statistics(results, test_case_name)
    
    return results


def generate_comparison_plots(
    results: Dict,
    cases_num: int,
    test_case_name: str,
    output_dir: str
):
    """
    Generate comparison plots
    """
    case_indices = np.arange(1, cases_num + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'IK Methods Comparison - {test_case_name}\n(Pseudo-inverse | DLS λ=0.1 | PyBullet)', 
                 fontsize=16, fontweight='bold')
    
    methods = ['pseudoinverse', 'dls', 'pybullet']
    labels = ['Pseudo-inverse', 'DLS', 'PyBullet']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # 1. Total Error
    ax = axes[0, 0]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        ax.plot(case_indices, results[method]['errors'], 
                marker=marker, label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Test Case', fontsize=11)
    ax.set_ylabel('Total Error (m)', fontsize=11)
    ax.set_title('Total Position & Orientation Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Position Error
    ax = axes[0, 1]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        ax.plot(case_indices, results[method]['position_errors'],
                marker=marker, label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Test Case', fontsize=11)
    ax.set_ylabel('Position Error (m)', fontsize=11)
    ax.set_title('Position Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Orientation Error
    ax = axes[1, 0]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        ax.plot(case_indices, results[method]['orientation_errors'],
                marker=marker, label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Test Case', fontsize=11)
    ax.set_ylabel('Orientation Error (rad)', fontsize=11)
    ax.set_title('Orientation Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Computation Time
    ax = axes[1, 1]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        ax.plot(case_indices, np.array(results[method]['times']) * 1000,
                marker=marker, label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Test Case', fontsize=11)
    ax.set_ylabel('Computation Time (ms)', fontsize=11)
    ax.set_title('Computation Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Average Joint Velocity
    ax = axes[2, 0]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        if method != 'pybullet':  # PyBullet doesn't provide velocity info
            ax.plot(case_indices, results[method]['avg_velocities'],
                    marker=marker, label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Test Case', fontsize=11)
    ax.set_ylabel('Avg Joint Velocity (rad/iter)', fontsize=11)
    ax.set_title('Average Joint Velocity per Iteration', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Iterations to Convergence
    ax = axes[2, 1]
    for method, label, color, marker in zip(methods, labels, colors, markers):
        if method != 'pybullet':  # PyBullet doesn't expose iteration count
            ax.plot(case_indices, results[method]['iterations'],
                    marker=marker, label=label, color=color, linewidth=2, markersize=5)
    ax.set_xlabel('Test Case', fontsize=11)
    ax.set_ylabel('Iterations', fontsize=11)
    ax.set_title('Iterations to Convergence', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{test_case_name}_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {plot_path}")
    plt.close()
    
    # Create bar chart for average statistics
    create_summary_bar_chart(results, test_case_name, output_dir)


def create_summary_bar_chart(results: Dict, test_case_name: str, output_dir: str):
    """
    Create bar chart comparing average metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Average Performance Metrics - {test_case_name}\n(3 Methods: Pseudo-inverse | DLS λ=0.1 | PyBullet)', 
                 fontsize=16, fontweight='bold')
    
    methods = ['pseudoinverse', 'dls', 'pybullet']
    labels = ['Pseudo-inv', 'DLS', 'PyBullet']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Calculate averages
    metrics = {
        'Total Error (m)': [np.mean(results[m]['errors']) for m in methods],
        'Position Error (m)': [np.mean(results[m]['position_errors']) for m in methods],
        'Orientation Error (rad)': [np.mean(results[m]['orientation_errors']) for m in methods],
        'Computation Time (ms)': [np.mean(results[m]['times']) * 1000 for m in methods],
        'Avg Joint Velocity': [np.mean(results[m]['avg_velocities']) if m != 'pybullet' else 0 for m in methods],
        'Iterations': [np.mean(results[m]['iterations']) if m != 'pybullet' else 0 for m in methods],
    }
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'Average {metric_name}', fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{test_case_name}_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary bar chart saved to: {plot_path}")
    plt.close()


def print_summary_statistics(results: Dict, test_case_name: str):
    """
    Print summary statistics table
    """
    print(f"\n{'='*80}")
    print(f"Summary Statistics - {test_case_name}")
    print(f"{'='*80}")
    
    methods = ['pseudoinverse', 'dls', 'pybullet']
    labels = ['Pseudo-inverse', 'DLS', 'PyBullet']
    
    # Header
    print(f"\n{'Metric':<30} {'Pseudo-inv':<15} {'DLS':<15} {'PyBullet':<15}")
    print(f"{'-'*75}")
    
    # Metrics
    metrics = [
        ('Total Error (m)', 'errors'),
        ('Position Error (m)', 'position_errors'),
        ('Orientation Error (rad)', 'orientation_errors'),
        ('Computation Time (ms)', 'times'),
        ('Iterations', 'iterations'),
        ('Avg Joint Velocity', 'avg_velocities'),
    ]
    
    for metric_name, metric_key in metrics:
        values = []
        for method in methods:
            data = results[method][metric_key]
            if not data or (method == 'pybullet' and metric_key in ['iterations', 'avg_velocities']):
                values.append('N/A')
            else:
                avg = np.mean(data)
                if metric_key == 'times':
                    avg *= 1000  # Convert to ms
                values.append(f"{avg:.6f}")
        
        print(f"{metric_name:<30} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
    
    print(f"{'-'*75}\n")


def main(args: argparse.Namespace):
    """
    Main comparison function
    """
    # Setup PyBullet
    if args.headless:
        physics_client_id = p.connect(p.DIRECT)
    else:
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
    
    # Setup robot
    robot = ur5Env(physics_client_id, use_IK=1)
    
    # Warmup
    for _ in range(int(1 / SIM_TIMESTEP * 1)):
        p.stepSimulation()
        time.sleep(SIM_TIMESTEP)
    
    # Test cases
    testcase_files = [
        'test_case/ik_test_case_easy.json',
        'test_case/ik_test_case_medium.json',
        'test_case/ik_test_case_hard.json',
    ]
    
    # Run comparisons
    for testcase_file in testcase_files:
        if os.path.exists(testcase_file):
            compare_ik_methods(robot, testcase_file, args.output_dir)
        else:
            print(f"Warning: Test case file not found: {testcase_file}")
    
    print(f"\n{'='*80}")
    print("Comparison complete! Check the output directory for plots.")
    print(f"{'='*80}\n")
    
    p.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare IK methods with visualization')
    parser.add_argument('--output-dir', '-o', type=str, default='comparison_results',
                        help='Output directory for comparison plots')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode (no GUI)')
    args = parser.parse_args()
    main(args)
