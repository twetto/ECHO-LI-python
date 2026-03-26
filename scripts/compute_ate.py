#!/usr/bin/env python3
"""Compute Absolute Trajectory Error (ATE) between aligned and ground truth.

Usage:
    python scripts/compute_ate.py eqvio_output_V1_01_easy/

Reads aligned_trajectory.txt and groundtruth_trajectory.txt from the directory.
"""
import sys
import numpy as np
from pathlib import Path


def load_tum(filepath):
    """Load TUM-format trajectory: timestamp tx ty tz qx qy qz qw"""
    data = np.loadtxt(filepath)
    stamps = data[:, 0]
    positions = data[:, 1:4]
    return stamps, positions


def associate(stamps_est, stamps_gt, max_dt=0.02):
    """Find nearest-timestamp pairs."""
    pairs = []
    for i, t_e in enumerate(stamps_est):
        diffs = np.abs(stamps_gt - t_e)
        j = np.argmin(diffs)
        if diffs[j] < max_dt:
            pairs.append((i, j))
    return pairs


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    aligned_file = d / "aligned_trajectory.txt"
    gt_file = d / "groundtruth_trajectory.txt"

    if not aligned_file.exists() or not gt_file.exists():
        print(f"Need both aligned_trajectory.txt and groundtruth_trajectory.txt in {d}")
        sys.exit(1)

    t_est, p_est = load_tum(aligned_file)
    t_gt, p_gt = load_tum(gt_file)

    pairs = associate(t_est, t_gt)
    if not pairs:
        print("No matching timestamps found")
        sys.exit(1)

    idx_est = [p[0] for p in pairs]
    idx_gt = [p[1] for p in pairs]

    errors = np.linalg.norm(p_est[idx_est] - p_gt[idx_gt], axis=1)

    print(f"Matched frames: {len(pairs)}")
    print(f"ATE RMSE:  {np.sqrt(np.mean(errors**2)):.4f} m")
    print(f"ATE mean:  {np.mean(errors):.4f} m")
    print(f"ATE median:{np.median(errors):.4f} m")
    print(f"ATE max:   {np.max(errors):.4f} m")
    print(f"ATE std:   {np.std(errors):.4f} m")

    # Per-axis breakdown
    diff = p_est[idx_est] - p_gt[idx_gt]
    for ax, name in enumerate(["x", "y", "z"]):
        print(f"  {name}-axis RMSE: {np.sqrt(np.mean(diff[:, ax]**2)):.4f} m")


if __name__ == "__main__":
    main()
