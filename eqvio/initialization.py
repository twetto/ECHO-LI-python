"""
Static initialization for VIO from stationary IMU readings.

Estimates initial roll/pitch from gravity direction in the accelerometer.
Yaw is unobservable from IMU alone and remains at zero.

This replaces the GT initialization hack and matches how production VIO
systems bootstrap (e.g., VINS-Mono, OKVIS, ORB-SLAM3).
"""

import numpy as np
from liepp import SO3, SE3

from eqvio.mathematical.imu_velocity import IMUVelocity


def estimate_initial_pose(
    imu_readings: list,
    n_samples: int = 100,
    gravity_magnitude: float = 9.80665,
) -> SE3:
    """Estimate initial pose from stationary IMU accelerometer readings.

    Accumulates the first n_samples IMU readings, averages the accelerometer,
    and finds the rotation that aligns the measured gravity direction with
    the world gravity direction [0, 0, -g] (NED-like) or [0, 0, g] (ENU-like).

    EuRoC convention: gravity in world frame is [0, 0, -g].
    When stationary, acc measures -g_body = R^T * [0, 0, g_world].
    So the measured acc ≈ R^T * [0, 0, -(-g)] = R^T * [0, 0, g].
    Actually: acc_measured = R^T * (a_world - g_world) and a_world=0 when
    stationary, g_world = [0,0,-g], so acc = R^T * [0, 0, g].

    We find R such that R * acc_avg / ||acc_avg|| = [0, 0, 1].

    Args:
        imu_readings: list of IMUVelocity measurements
        n_samples: number of initial IMU samples to average
        gravity_magnitude: expected gravity magnitude (not used for direction)

    Returns:
        SE3 pose with estimated attitude and zero position
    """
    if len(imu_readings) < 5:
        return SE3.Identity()

    n = min(n_samples, len(imu_readings))

    # Average accelerometer readings
    acc_sum = np.zeros(3)
    for i in range(n):
        acc_sum += imu_readings[i].acc
    acc_avg = acc_sum / n

    # Measured gravity direction in body frame
    g_body = acc_avg / np.linalg.norm(acc_avg)

    # World gravity direction: when stationary, acc = R^T * [0, 0, g]
    # So g_body should map to [0, 0, 1] in world frame
    g_world = np.array([0.0, 0.0, 1.0])

    # Find rotation: R * g_body = g_world
    R = SO3.SO3FromVectors(g_body, g_world)

    pose = SE3()
    pose.R = R
    pose.x = np.zeros(3)

    return pose


def check_stationary(
    imu_readings: list,
    n_samples: int = 50,
    gyro_std_threshold: float = 0.1,  # rad/s std dev
    acc_std_threshold: float = 0.5,  # m/s^2 std dev
) -> bool:
    """Check if the first n_samples IMU readings indicate a stationary platform.

    Uses standard deviation of gyro and accelerometer, not the mean,
    so constant bias doesn't trip the detector.

    Args:
        imu_readings: list of IMUVelocity
        n_samples: number of samples to check
        gyro_std_threshold: max gyro std dev to consider stationary
        acc_std_threshold: max acc std dev to consider stationary

    Returns:
        True if platform appears stationary
    """
    if len(imu_readings) < 10:
        return False

    n = min(n_samples, len(imu_readings))

    gyro_vecs = np.array([imu_readings[i].gyr for i in range(n)])
    acc_vecs = np.array([imu_readings[i].acc for i in range(n)])

    # Check variance, not mean — bias is constant, motion adds variance
    gyro_std = np.std(gyro_vecs)
    acc_std = np.std(np.linalg.norm(acc_vecs, axis=1))

    return gyro_std < gyro_std_threshold and acc_std < acc_std_threshold
