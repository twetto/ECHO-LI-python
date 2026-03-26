"""
Trajectory alignment utilities.

Port of: VIOVisualiser.cpp alignUmeyama() and alignTrajectories()

Provides SE(3) alignment between estimated and reference trajectories
for evaluation and visualization. Uses Umeyama's method (SVD-based).
"""

import numpy as np
from liepp import SO3, SE3


def align_umeyama(
    points1: np.ndarray, points2: np.ndarray
) -> SE3:
    """Find SE(3) alignment between two point sets.

    Finds (R, x) in SE(3) to minimize sum_i ||(R p_i + x) - q_i||^2
    where p_i = points1[i] and q_i = points2[i].

    Port of: alignUmeyama() in VIOVisualiser.cpp

    Args:
        points1: (N, 3) source points
        points2: (N, 3) target points

    Returns:
        SE3 transform mapping points1 towards points2
    """
    assert points1.shape == points2.shape
    n = points1.shape[0]

    mu1 = np.mean(points1, axis=0)
    mu2 = np.mean(points2, axis=0)

    sigma1_sq = np.mean(np.sum((points1 - mu1) ** 2, axis=1))
    sigma12 = (points2 - mu2).T @ (points1 - mu1) / n

    U, S_vals, Vt = np.linalg.svd(sigma12)

    # Ensure proper rotation (right-handed)
    S_mat = np.eye(3)
    if np.linalg.det(sigma12) < 0:
        S_mat[2, 2] = -1.0
        scale_sum = S_vals[0] + S_vals[1] - S_vals[2]
    else:
        scale_sum = np.sum(S_vals)

    R_mat = U @ S_mat @ Vt
    R = SO3(matrix=R_mat)

    s = scale_sum / sigma1_sq if sigma1_sq > 1e-12 else 1.0
    x = mu2 - s * (R * mu1)

    result = SE3()
    result.R = R
    result.x = x
    return result


def align_trajectories(est_trajectory, ref_trajectory) -> SE3:
    """Find SE(3) alignment between estimated and reference trajectories.

    Port of: alignTrajectories() in VIOVisualiser.cpp

    Matches trajectories by timestamp, then applies Umeyama alignment.
    Falls back to first-pose alignment if too few matched points.

    Args:
        est_trajectory: list of StampedPose (estimated)
        ref_trajectory: list of StampedPose (reference/ground truth)

    Returns:
        SE3 transform mapping estimated trajectory towards reference
    """
    if not est_trajectory or not ref_trajectory:
        return SE3.Identity()

    min_time = max(est_trajectory[0].t, ref_trajectory[0].t)
    max_time = min(est_trajectory[-1].t, ref_trajectory[-1].t)

    ref_period = ((ref_trajectory[-1].t - ref_trajectory[0].t)
                  / len(ref_trajectory))
    est_period = ((est_trajectory[-1].t - est_trajectory[0].t)
                  / max(len(est_trajectory), 1))
    use_period = max(ref_period, est_period)

    est_matched = []
    ref_matched = []
    est_it = 0
    ref_it = 0

    t = min_time
    while t < max_time:
        while est_it < len(est_trajectory) and est_trajectory[est_it].t < t:
            est_it += 1
        while ref_it < len(ref_trajectory) and ref_trajectory[ref_it].t < t:
            ref_it += 1

        if est_it >= len(est_trajectory) or ref_it >= len(ref_trajectory):
            break

        est_matched.append(est_trajectory[est_it].pose.x.copy())
        ref_matched.append(ref_trajectory[ref_it].pose.x.copy())

        t += use_period

    if len(est_matched) == 0:
        return SE3.Identity()

    if len(est_matched) <= 100:
        # Too few points for Umeyama — use first-pose alignment
        return ref_trajectory[0].pose * est_trajectory[0].pose.inverse()

    return align_umeyama(np.array(est_matched), np.array(ref_matched))
