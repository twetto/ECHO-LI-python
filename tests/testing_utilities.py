"""
Testing utilities for EqVIO Python tests.

Port of: test/testing_utilities.h / testing_utilities.cpp

Provides:
    - Random state/group/velocity generators
    - Distance functions (stateDistance, logNorm)
    - Simple pinhole camera for output tests
    - Finite-difference Jacobian checker
"""

from __future__ import annotations

import numpy as np
from liepp import SO3, SE3, SOT3

from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark, GRAVITY_CONSTANT,
)
from eqvio.mathematical.vio_group import VIOGroup, VIOAlgebra
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vision_measurement import VisionMeasurement


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_REPS = 20
NEAR_ZERO = 1e-8


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------

def random_sensor_state(rng: np.random.Generator = None) -> VIOSensorState:
    if rng is None:
        rng = np.random.default_rng()
    return VIOSensorState(
        input_bias=rng.standard_normal(6) * 0.01,
        pose=SE3(R=SO3.Random(), x=rng.standard_normal(3)),
        velocity=rng.standard_normal(3) * 0.5,
        camera_offset=SE3(R=SO3.Random(), x=rng.standard_normal(3) * 0.1),
    )


def reasonable_sensor_state(rng: np.random.Generator = None) -> VIOSensorState:
    """Sensor state with small biases and a forward-looking pose (more realistic)."""
    if rng is None:
        rng = np.random.default_rng()
    return VIOSensorState(
        input_bias=rng.standard_normal(6) * 0.001,
        pose=SE3(R=SO3.Random(), x=rng.standard_normal(3) * 0.5),
        velocity=rng.standard_normal(3) * 0.3,
        camera_offset=SE3(R=SO3.Random(), x=rng.standard_normal(3) * 0.05),
    )


def random_state_element(ids: list, rng: np.random.Generator = None) -> VIOState:
    """Random VIOState with landmarks at given ids.

    Reference: testing_utilities randomStateElement()
    """
    if rng is None:
        rng = np.random.default_rng()
    state = VIOState(sensor=random_sensor_state(rng))
    for i in ids:
        p = rng.standard_normal(3) + np.array([0.0, 0.0, 3.0])  # in front of camera
        state.camera_landmarks.append(Landmark(p=p, id=i))
    return state


def reasonable_state_element(ids: list, rng: np.random.Generator = None) -> VIOState:
    """State with reasonable landmark depths."""
    if rng is None:
        rng = np.random.default_rng()
    state = VIOState(sensor=reasonable_sensor_state(rng))
    for i in ids:
        p = rng.standard_normal(3) * 0.5 + np.array([0.0, 0.0, 5.0])
        state.camera_landmarks.append(Landmark(p=p, id=i))
    return state


def random_group_element(ids: list, rng: np.random.Generator = None) -> VIOGroup:
    """Random VIOGroup element with given landmark ids.

    Reference: testing_utilities randomGroupElement()
    """
    if rng is None:
        rng = np.random.default_rng()
    return VIOGroup(
        beta=rng.standard_normal(6) * 0.1,
        A=SE3(R=SO3.Random(), x=rng.standard_normal(3) * 0.5),
        w=rng.standard_normal(3) * 0.3,
        B=SE3(R=SO3.Random(), x=rng.standard_normal(3) * 0.2),
        Q=[SOT3(R=SO3.Random(), a=np.exp(rng.standard_normal() * 0.3))
           for _ in ids],
        id=list(ids),
        Q_planes=[],
        plane_id=[],
    )


def reasonable_group_element(ids: list, rng: np.random.Generator = None) -> VIOGroup:
    """Group element near identity (more realistic for filter testing)."""
    if rng is None:
        rng = np.random.default_rng()
    return VIOGroup(
        beta=rng.standard_normal(6) * 0.01,
        A=SE3(R=SO3.exp(rng.standard_normal(3) * 0.1),
              x=rng.standard_normal(3) * 0.1),
        w=rng.standard_normal(3) * 0.05,
        B=SE3(R=SO3.exp(rng.standard_normal(3) * 0.05),
              x=rng.standard_normal(3) * 0.05),
        Q=[SOT3(R=SO3.exp(rng.standard_normal(3) * 0.1),
                 a=np.exp(rng.standard_normal() * 0.1))
           for _ in ids],
        id=list(ids),
        Q_planes=[],
        plane_id=[],
    )


def random_velocity_element(rng: np.random.Generator = None) -> IMUVelocity:
    """Random IMU velocity.

    Reference: testing_utilities randomVelocityElement()
    """
    if rng is None:
        rng = np.random.default_rng()
    return IMUVelocity(
        stamp=0.0,
        gyr=rng.standard_normal(3) * 0.5,
        acc=rng.standard_normal(3) * 2.0 + np.array([0.0, 0.0, GRAVITY_CONSTANT]),
    )


# ---------------------------------------------------------------------------
# Distance / norm functions
# ---------------------------------------------------------------------------

def se3_log_norm(T: SE3) -> float:
    """Norm of SE3 log."""
    return np.linalg.norm(SE3.log(T))


def sot3_log_norm(Q: SOT3) -> float:
    """Norm of SOT3 log."""
    return np.linalg.norm(SOT3.log(Q))


def log_norm(X: VIOGroup) -> float:
    """Distance of VIOGroup from identity, measured via component log norms.

    Reference: testing_utilities logNorm()
    """
    total = 0.0
    total += np.linalg.norm(X.beta)
    total += se3_log_norm(X.A)
    total += np.linalg.norm(X.w)
    total += se3_log_norm(X.B)
    for Qi in X.Q:
        total += sot3_log_norm(Qi)
    for Qi in X.Q_planes:
        total += sot3_log_norm(Qi)
    return total


def state_distance(xi1: VIOState, xi2: VIOState) -> float:
    """Distance between two VIO states.

    Reference: testing_utilities stateDistance()
    """
    dist = 0.0
    # Sensor
    dist += np.linalg.norm(xi1.sensor.input_bias - xi2.sensor.input_bias)
    dist += se3_log_norm(xi1.sensor.pose.inverse() * xi2.sensor.pose)
    dist += np.linalg.norm(xi1.sensor.velocity - xi2.sensor.velocity)
    dist += se3_log_norm(xi1.sensor.camera_offset.inverse() * xi2.sensor.camera_offset)
    # Landmarks
    assert len(xi1.camera_landmarks) == len(xi2.camera_landmarks)
    for lm1, lm2 in zip(xi1.camera_landmarks, xi2.camera_landmarks):
        dist += np.linalg.norm(lm1.p - lm2.p)
    # Planes
    for pl1, pl2 in zip(xi1.plane_landmarks, xi2.plane_landmarks):
        dist += np.linalg.norm(pl1.q - pl2.q)
    return dist


def measurement_distance(y1: VisionMeasurement, y2: VisionMeasurement) -> float:
    """Distance between two vision measurements."""
    common_ids = sorted(set(y1.cam_coordinates.keys()) & set(y2.cam_coordinates.keys()))
    if not common_ids:
        return 0.0
    return sum(
        np.linalg.norm(y1.cam_coordinates[fid] - y2.cam_coordinates[fid])
        for fid in common_ids
    )


# ---------------------------------------------------------------------------
# Simple pinhole camera (no distortion)
# ---------------------------------------------------------------------------

class SimplePinholeCamera:
    """Minimal pinhole camera model for testing.

    Reference: creates the same camera as createDefaultCamera() in testing_utilities.

    EuRoC-like parameters: fx=458.654, fy=457.296, cx=367.215, cy=248.375
    """

    def __init__(self, fx=458.654, fy=457.296, cx=367.215, cy=248.375):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project_point(self, p: np.ndarray) -> np.ndarray:
        """Project 3D point to pixel coordinates: (3,) -> (2,)."""
        x, y, z = p[0], p[1], p[2]
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return np.array([u, v])

    def undistort_point(self, pixel: np.ndarray) -> np.ndarray:
        """Pixel to unit bearing vector: (2,) -> (3,)."""
        x = (pixel[0] - self.cx) / self.fx
        y = (pixel[1] - self.cy) / self.fy
        bearing = np.array([x, y, 1.0])
        return bearing / np.linalg.norm(bearing)

    def projection_jacobian(self, p: np.ndarray) -> np.ndarray:
        """Jacobian of project_point w.r.t. 3D point: (3,) -> (2,3)."""
        x, y, z = p[0], p[1], p[2]
        z2 = z * z
        return np.array([
            [self.fx / z, 0.0, -self.fx * x / z2],
            [0.0, self.fy / z, -self.fy * y / z2],
        ])


def create_default_camera() -> SimplePinholeCamera:
    """Create the default camera matching C++ test utilities."""
    return SimplePinholeCamera()


# ---------------------------------------------------------------------------
# Random vision measurement
# ---------------------------------------------------------------------------

def random_vision_measurement(ids: list, cam_ptr=None,
                               rng: np.random.Generator = None) -> VisionMeasurement:
    """Random VisionMeasurement for given ids."""
    if rng is None:
        rng = np.random.default_rng()
    if cam_ptr is None:
        cam_ptr = create_default_camera()

    y = VisionMeasurement(stamp=0.0, camera_ptr=cam_ptr)
    for fid in ids:
        # Random point in front of camera, project
        p = rng.standard_normal(3) * 0.5 + np.array([0.0, 0.0, 5.0])
        y.cam_coordinates[fid] = cam_ptr.project_point(p)
    return y


# ---------------------------------------------------------------------------
# Finite-difference Jacobian test
# ---------------------------------------------------------------------------

def check_differential(f, x0: np.ndarray, J_expected: np.ndarray,
                      step: float = 1e-6, atol: float = 1e-4):
    """Verify an analytical Jacobian against central finite differences.

    Reference: testing_utilities testDifferential()

    Args:
        f:          function x -> y
        x0:         point at which to test
        J_expected: analytical Jacobian (m, n)
        step:       finite-difference step size
        atol:       absolute tolerance
    """
    n = x0.shape[0]
    f0 = f(x0)
    m = f0.shape[0]
    J_numerical = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = step
        J_numerical[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2 * step)

    diff = np.linalg.norm(J_expected - J_numerical)
    assert diff < atol, (
        f"Jacobian mismatch: ||J_analytical - J_numerical|| = {diff:.2e} > {atol:.2e}\n"
        f"Max element diff: {np.max(np.abs(J_expected - J_numerical)):.2e}"
    )
