"""
VIO State definition.

Port of: VIOState.h / VIOState.cpp
Reference: Section 1.1 of porting guide.

Key types:
    Landmark         — point position + id
    PlaneLandmark    — plane CP (n/d) + id  [NEW]
    VIOSensorState   — biases, pose, velocity, camera extrinsics
    VIOState         — sensor state + camera-frame landmarks (point + plane)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List
import numpy as np

from liepp import SO3, SE3

GRAVITY_CONSTANT = 9.80665


# ---------------------------------------------------------------------------
# Landmark
# ---------------------------------------------------------------------------

@dataclass
class Landmark:
    """A VIO landmark: 3D position in camera frame + id.

    Reference: VIOState.h struct Landmark
    """
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    id: int = -1

    CDim: ClassVar[int] = 3


# ---------------------------------------------------------------------------
# PlaneLandmark (NEW — not in C++ codebase)
# ---------------------------------------------------------------------------

@dataclass
class PlaneLandmark:
    """A planar landmark in CP representation: q = n/d.

    The plane satisfies q^T x + 1 = 0 for points x on the plane.
    Transforms under dual SOT(3) action.

    Reference: Section 2.1 of porting guide.
    """
    q: np.ndarray = field(default_factory=lambda: np.zeros(3))
    id: int = -1
    point_ids: List[int] = field(default_factory=list)

    CDim: ClassVar[int] = 3


# ---------------------------------------------------------------------------
# VIOSensorState
# ---------------------------------------------------------------------------

@dataclass
class VIOSensorState:
    """IMU sensor state: biases, pose, velocity, camera extrinsics.

    Reference: VIOState.h struct VIOSensorState

    Fields:
        input_bias:    (6,) [gyro_bias(3), accel_bias(3)]
        pose:          SE3  — IMU/robot pose in world frame
        velocity:      (3,) — body-frame velocity
        camera_offset: SE3  — camera pose w.r.t. IMU
    """
    input_bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    pose: SE3 = field(default_factory=SE3.Identity)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    camera_offset: SE3 = field(default_factory=SE3.Identity)

    CDim: ClassVar[int] = 21  # 6 + 6 + 3 + 6

    def gravity_dir(self) -> np.ndarray:
        """Direction of gravity in the IMU frame: R^{-1} e_z.

        Reference: VIOSensorState::gravityDir()
        """
        return self.pose.R.inverse() * np.array([0.0, 0.0, 1.0])

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.input_bias[:3]

    @property
    def accel_bias(self) -> np.ndarray:
        return self.input_bias[3:]


# ---------------------------------------------------------------------------
# VIOState
# ---------------------------------------------------------------------------

@dataclass
class VIOState:
    """Full VIO state: sensor + point landmarks + plane landmarks.

    Reference: VIOState.h struct VIOState

    C++ stores cameraLandmarks as vector<Landmark> (ordered, with ids).
    Python keeps the same list structure for 1:1 correspondence with the
    group's Q vector. Plane landmarks are a new addition (Section 2.1).
    """
    sensor: VIOSensorState = field(default_factory=VIOSensorState)
    camera_landmarks: List[Landmark] = field(default_factory=list)
    plane_landmarks: List[PlaneLandmark] = field(default_factory=list)  # NEW

    def get_ids(self) -> List[int]:
        """Get point landmark ids (matching C++ getIds()).

        Reference: VIOState::getIds()
        """
        return [lm.id for lm in self.camera_landmarks]

    def get_all_ids(self) -> List[int]:
        """Get all landmark ids (points + planes)."""
        return ([lm.id for lm in self.camera_landmarks]
                + [pl.id for pl in self.plane_landmarks])

    def dim(self) -> int:
        """Total dimension of the state manifold.

        Reference: VIOState::Dim()
        """
        return (VIOSensorState.CDim
                + Landmark.CDim * len(self.camera_landmarks)
                + PlaneLandmark.CDim * len(self.plane_landmarks))


# ---------------------------------------------------------------------------
# StampedPose (utility)
# ---------------------------------------------------------------------------

@dataclass
class StampedPose:
    """Timestamped SE(3) pose.

    Reference: VIOState.h struct StampedPose
    """
    t: float = 0.0
    pose: SE3 = field(default_factory=SE3.Identity)


# ---------------------------------------------------------------------------
# System dynamics integration
# ---------------------------------------------------------------------------

def integrate_system_function(state: VIOState, velocity, dt: float) -> VIOState:
    """Integrate the VIO dynamics over one timestep.

    Reference: integrateSystemFunction() in VIOState.cpp

    Args:
        state:    current VIO state
        velocity: IMUVelocity measurement
        dt:       integration timestep

    Returns:
        New VIOState after integration
    """
    new_state = VIOState()
    sensor = state.sensor

    # Bias-corrected velocity
    v_est = velocity - sensor.input_bias

    # Integrate biases
    new_state.sensor.input_bias = np.zeros(6)
    new_state.sensor.input_bias[:3] = sensor.input_bias[:3] + dt * velocity.gyr_bias_vel
    new_state.sensor.input_bias[3:] = sensor.input_bias[3:] + dt * velocity.acc_bias_vel

    # Integrate pose (second-order, matching C++ exactly)
    pose_change = SE3()
    pose_change.R = SO3.exp(dt * v_est.gyr)
    x_world = (dt * (sensor.pose.R * sensor.velocity)
               + 0.5 * dt * dt * (sensor.pose.R * v_est.acc
                                   + np.array([0.0, 0.0, -GRAVITY_CONSTANT])))
    pose_change.x = sensor.pose.R.inverse() * x_world
    new_state.sensor.pose = sensor.pose * pose_change

    # Integrate velocity
    inertial_vel_diff = (sensor.pose.R.asMatrix() @ v_est.acc
                         + np.array([0.0, 0.0, -GRAVITY_CONSTANT]))
    new_vel_world = sensor.pose.R * sensor.velocity + dt * inertial_vel_diff
    new_state.sensor.velocity = new_state.sensor.pose.R.inverse() * new_vel_world

    # Transform landmarks in camera frame
    camera_pose_change_inv = (sensor.camera_offset.inverse()
                               * pose_change.inverse()
                               * sensor.camera_offset)
    new_state.camera_landmarks = [
        Landmark(p=camera_pose_change_inv * lm.p, id=lm.id)
        for lm in state.camera_landmarks
    ]

    # Plane landmarks: dual transform (NEW)
    # q_new = (R^T q) * d_old / d_new  via the (n,d) representation
    new_state.plane_landmarks = []
    for pl in state.plane_landmarks:
        q0 = pl.q
        q_norm = np.linalg.norm(q0)
        n0 = q0 / q_norm
        d0 = 1.0 / q_norm

        R_M = camera_pose_change_inv.R.asMatrix()
        t_M = camera_pose_change_inv.x
        n_new = R_M.T @ n0
        d_new = d0 + n0 @ t_M
        q_new = n_new / d_new

        new_state.plane_landmarks.append(
            PlaneLandmark(q=q_new, id=pl.id, point_ids=list(pl.point_ids))
        )

    # Camera offset is constant
    new_state.sensor.camera_offset = sensor.camera_offset

    return new_state
