import os
import sys
import argparse
import numpy as np

def align(model, data):
    """
    Aligns model points to data points using the Umeyama method (SVD-based).
    Returns the rotation matrix, translation vector, and per-point translation error.
    """
    np.set_printoptions(precision=3, suppress=True)

    model_centered = model - model.mean(1, keepdims=True)
    data_centered = data - data.mean(1, keepdims=True)

    W = np.zeros((3, 3))
    for i in range(model.shape[1]):
        W += np.outer(model_centered[:, i], data_centered[:, i])

    U, _, Vh = np.linalg.svd(W.T, full_matrices=False)

    S = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1

    rotation = U @ S @ Vh
    translation = data.mean(1, keepdims=True) - rotation @ model.mean(1, keepdims=True)

    aligned_model = rotation @ model + translation
    error = aligned_model - data
    trans_error = np.sqrt(np.sum(error ** 2, axis=0))

    return rotation, translation, trans_error

def read_trajectory_file(filename):
    """
    Reads a trajectory file. Supports:
    - 4x4 matrix per line (16 floats)
    - Optional timestamp in the first column (17 values per line)
    Returns a list of 4x4 transformation matrices.
    """
    trajectory = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = [float(v) for v in line.strip().split()]
            if len(values) == 17:
                values = values[1:]  # Remove timestamp
            if len(values) != 16:
                continue
            transform = np.array(values).reshape(4, 4)
            trajectory.append(transform)
    return np.array(trajectory)

def evaluate_ate(gt_traj, est_traj):
    """
    Evaluates Absolute Trajectory Error (ATE) RMSE between two sets of 4x4 poses.
    """
    min_len = min(len(gt_traj), len(est_traj))
    if min_len == 0:
        raise ValueError("Empty trajectory input.")

    print(f"[INFO] GT trajectory length: {len(gt_traj)}, Estimated trajectory length: {len(est_traj)}")
    print(f"[INFO] Evaluating on first {min_len} frames.")

    valid_indices = [
        i for i in range(min_len)
        if not np.any(np.isinf(gt_traj[i])) and not np.any(np.isinf(est_traj[i]))
    ]
    if len(valid_indices) == 0:
        raise ValueError("No valid trajectory point pairs found.")

    print(f"[INFO] Valid points: {len(valid_indices)}")

    gt_points = np.array([gt_traj[i][:3, 3] for i in valid_indices]).T  # (3, N)
    est_points = np.array([est_traj[i][:3, 3] for i in valid_indices]).T  # (3, N)

    _, _, trans_error = align(gt_points, est_points)
    avg_error = np.mean(trans_error)

    return avg_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Compute the Absolute Trajectory Error (ATE) between ground truth and estimated pose trajectories.
    Input files must contain 4x4 transformation matrices, optionally prefixed by timestamps.
    """)
    parser.add_argument("first_file", help="Ground truth trajectory file")
    parser.add_argument("second_file", help="Estimated trajectory file")
    parser.add_argument('--save_path', help='Evalustion results will be saved to this directory (default: current directory)',default=".")
    args = parser.parse_args()

    try:
        gt_traj = read_trajectory_file(args.first_file)
        est_traj = read_trajectory_file(args.second_file)
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] File format error: {e}")
        sys.exit(1)

    try:
        ate = evaluate_ate(gt_traj, est_traj)
        output_dir = args.save_path
        os.makedirs(output_dir, exist_ok=True) 
        with open(os.path.join(output_dir, "result.txt"), "a") as f:
            f.write("--------\n")
            f.write(f"ATE RMSE: {ate * 100:.2f} cm\n")
        print(f"\n[RESULT] ATE RMSE: {ate * 100:.2f} cm")
    except ValueError as e:
        print(f"[ERROR] Evaluation failed: {e}")
        sys.exit(1)

