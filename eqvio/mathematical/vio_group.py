"""
VIO symmetry group definition.

Port of: VIOGroup.h / VIOGroup.cpp
Reference: Section 1.1 of porting guide.

Key types:
    VIOGroup    — symmetry group element (SE3 x R^6 x R^3 x SE3 x SOT3^N x SOT3^M)
    VIOAlgebra  — Lie algebra element

Key functions:
    state_group_action         — right action of VIOGroup on VIOState
    sensor_state_group_action  — right action on sensor states only
    lift_velocity              — lift IMU velocity to Lie algebra
    lift_velocity_discrete     — discrete version of the lift
    vio_exp                    — Lie group exponential
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation

from liepp import SO3, SE3, SOT3, SEn3

from .vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark, GRAVITY_CONSTANT
)


# ---------------------------------------------------------------------------
# VIOGroup
# ---------------------------------------------------------------------------

@dataclass
class VIOGroup:
    """Symmetry group element for EqVIO.

    Reference: VIOGroup.h struct VIOGroup

    Fields (matching C++ exactly):
        beta: (6,)       — bias symmetry component
        A:    SE3        — pose symmetry component
        w:    (3,)       — velocity symmetry component
        B:    SE3        — camera offset symmetry component
        Q:    list[SOT3] — point landmark symmetry components
        id:   list[int]  — ids aligned with Q

    New for planar extension:
        Q_planes: list[SOT3] — plane landmark symmetry components
        plane_id: list[int]  — ids aligned with Q_planes
    """
    beta: np.ndarray = field(default_factory=lambda: np.zeros(6))
    A: SE3 = field(default_factory=SE3.Identity)
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    B: SE3 = field(default_factory=SE3.Identity)
    Q: List[SOT3] = field(default_factory=list)
    id: List[int] = field(default_factory=list)
    # Planar extension (NEW)
    Q_planes: List[SOT3] = field(default_factory=list)
    plane_id: List[int] = field(default_factory=list)

    @staticmethod
    def Identity(ids: Optional[List[int]] = None,
                 plane_ids: Optional[List[int]] = None) -> VIOGroup:
        """Group identity with given landmark ids.

        Reference: VIOGroup::Identity()
        """
        ids = ids or []
        plane_ids = plane_ids or []
        return VIOGroup(
            beta=np.zeros(6),
            A=SE3.Identity(),
            w=np.zeros(3),
            B=SE3.Identity(),
            Q=[SOT3.Identity() for _ in ids],
            id=list(ids),
            Q_planes=[SOT3.Identity() for _ in plane_ids],
            plane_id=list(plane_ids),
        )

    def inverse(self) -> VIOGroup:
        """Group inverse.

        Reference: VIOGroup::inverse()
        """
        return VIOGroup(
            beta=-self.beta,
            A=self.A.inverse(),
            w=-(self.A.R.inverse() * self.w),
            B=self.B.inverse(),
            Q=[Qi.inverse() for Qi in self.Q],
            id=list(self.id),
            Q_planes=[Qi.inverse() for Qi in self.Q_planes],
            plane_id=list(self.plane_id),
        )

    def __mul__(self, other: VIOGroup) -> VIOGroup:
        """Group multiplication: result = self * other.

        Reference: VIOGroup::operator*()
        """
        assert len(self.Q) == len(other.Q)
        assert len(self.Q_planes) == len(other.Q_planes)

        return VIOGroup(
            beta=self.beta + other.beta,
            A=self.A * other.A,
            w=self.w + self.A.R * other.w,
            B=self.B * other.B,
            Q=[Qi1 * Qi2 for Qi1, Qi2 in zip(self.Q, other.Q)],
            id=list(self.id),
            Q_planes=[Qi1 * Qi2 for Qi1, Qi2 in zip(self.Q_planes, other.Q_planes)],
            plane_id=list(self.plane_id),
        )

    def has_nan(self) -> bool:
        """Check for NaN in any component.

        Reference: VIOGroup::hasNaN()
        """
        if np.any(np.isnan(self.beta)):
            return True
        if np.any(np.isnan(self.A.asMatrix())):
            return True
        if np.any(np.isnan(self.B.asMatrix())):
            return True
        if np.any(np.isnan(self.w)):
            return True
        if any(np.any(np.isnan(Qi.asMatrix())) for Qi in self.Q):
            return True
        if any(np.any(np.isnan(Qi.asMatrix())) for Qi in self.Q_planes):
            return True
        return False


# ---------------------------------------------------------------------------
# VIOAlgebra
# ---------------------------------------------------------------------------

@dataclass
class VIOAlgebra:
    """Lie algebra element of VIOGroup.

    Reference: VIOGroup.h struct VIOAlgebra

    Fields:
        u_beta: (6,)      — bias algebra component
        U_A:    (6,)      — se(3) pose component [omega(3), v(3)]
        U_B:    (6,)      — se(3) camera offset component
        u_w:    (3,)      — velocity algebra component
        W:      list[(4,)] — sot(3) landmark components [omega(3), sigma(1)]
        id:     list[int]

    New for planar extension:
        W_planes: list[(4,)] — sot(3) plane landmark components
        plane_id: list[int]
    """
    u_beta: np.ndarray = field(default_factory=lambda: np.zeros(6))
    U_A: np.ndarray = field(default_factory=lambda: np.zeros(6))
    U_B: np.ndarray = field(default_factory=lambda: np.zeros(6))
    u_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    W: List[np.ndarray] = field(default_factory=list)
    id: List[int] = field(default_factory=list)
    # Planar extension (NEW)
    W_planes: List[np.ndarray] = field(default_factory=list)
    plane_id: List[int] = field(default_factory=list)

    def __mul__(self, c: float) -> VIOAlgebra:
        """Scale by constant: lambda * c.

        Reference: VIOAlgebra::operator*(double)
        """
        return VIOAlgebra(
            u_beta=self.u_beta * c,
            U_A=self.U_A * c,
            U_B=self.U_B * c,
            u_w=self.u_w * c,
            W=[Wi * c for Wi in self.W],
            id=list(self.id),
            W_planes=[Wi * c for Wi in self.W_planes],
            plane_id=list(self.plane_id),
        )

    def __rmul__(self, c: float) -> VIOAlgebra:
        return self.__mul__(c)

    def __neg__(self) -> VIOAlgebra:
        """Negate.

        Reference: VIOAlgebra::operator-()
        """
        return self * (-1.0)

    def __add__(self, other: VIOAlgebra) -> VIOAlgebra:
        """Add algebra elements.

        Reference: VIOAlgebra::operator+()
        """
        return VIOAlgebra(
            u_beta=self.u_beta + other.u_beta,
            U_A=self.U_A + other.U_A,
            U_B=self.U_B + other.U_B,
            u_w=self.u_w + other.u_w,
            W=[Wi1 + Wi2 for Wi1, Wi2 in zip(self.W, other.W)],
            id=list(self.id),
            W_planes=[Wi1 + Wi2 for Wi1, Wi2 in zip(self.W_planes, other.W_planes)],
            plane_id=list(self.plane_id),
        )

    def __sub__(self, other: VIOAlgebra) -> VIOAlgebra:
        """Subtract algebra elements.

        Reference: VIOAlgebra::operator-()
        """
        return self + (-other)


# ---------------------------------------------------------------------------
# Group actions
# ---------------------------------------------------------------------------

def sensor_state_group_action(X: VIOGroup, sensor: VIOSensorState) -> VIOSensorState:
    """Right group action of VIOGroup on sensor states.

    Reference: sensorStateGroupAction() in VIOGroup.cpp

    result.inputBias = sensor.inputBias + X.beta
    result.pose      = sensor.pose * X.A
    result.velocity   = X.A.R^{-1} * (sensor.velocity - X.w)
    result.cameraOffset = X.A^{-1} * sensor.cameraOffset * X.B
    """
    result = VIOSensorState()
    result.input_bias = sensor.input_bias + X.beta
    result.pose = sensor.pose * X.A
    result.velocity = X.A.R.inverse() * (sensor.velocity - X.w)
    result.camera_offset = X.A.inverse() * sensor.camera_offset * X.B
    return result


def state_group_action(X: VIOGroup, state: VIOState) -> VIOState:
    """Right group action of VIOGroup on VIOState.

    Reference: stateGroupAction() in VIOGroup.cpp

    Points use the inverse SOT(3) action: p' = Q^{-1} * p
    Planes use the dual inverse action: q' = a * R^T * q  (Section 2.1)
    """
    new_state = VIOState()
    new_state.sensor = sensor_state_group_action(X, state.sensor)

    # Check alignment
    assert len(X.Q) == len(state.camera_landmarks)
    assert all(Xi == lm.id for Xi, lm in zip(X.id, state.camera_landmarks))

    # Transform point landmarks: Q^{-1} * p
    '''
    # This is the reference logic
    new_state.camera_landmarks = [
        Landmark(p=Qi.inverse() * lm.p, id=lm.id)
        for Qi, lm in zip(X.Q, state.camera_landmarks)
    ]
    '''

    # --- Fast alternative logic begins --- #
    if state.camera_landmarks:
        quats = np.array([Qi.R.asQuaternion() for Qi in X.Q]) 
        scales = np.array([Qi.a for Qi in X.Q])
        points = np.array([lm.p for lm in state.camera_landmarks])
        batched_inv_rotations = Rotation.from_quat(quats).inv()
        rotated_points = batched_inv_rotations.apply(points)
        final_points = (1.0 / scales[:, np.newaxis]) * rotated_points
        new_state.camera_landmarks = [
            Landmark(p=final_points[i], id=state.camera_landmarks[i].id)
            for i in range(len(final_points))
        ]
    else:
        new_state.camera_landmarks = []
    # --- Fast alternative logic ends --- #

    # Transform plane landmarks: dual inverse action (NEW, Section 2.1)
    assert len(X.Q_planes) == len(state.plane_landmarks)
    new_state.plane_landmarks = [
        PlaneLandmark(
            q=Qi.a * (Qi.R.inverse() * pl.q),  # dual: scale by a, not 1/a
            id=pl.id,
            point_ids=list(pl.point_ids),
        )
        for Qi, pl in zip(X.Q_planes, state.plane_landmarks)
    ]

    return new_state


# ---------------------------------------------------------------------------
# Exponential map
# ---------------------------------------------------------------------------

def vio_exp(lam: VIOAlgebra) -> VIOGroup:
    """Lie group exponential: VIOAlgebra -> VIOGroup.

    Reference: VIOExp() in VIOGroup.cpp

    Uses SE_2(3) exponential for the coupled (A, w) block.
    """
    # Coupled exponential via SE_2(3) for pose + velocity
    ext_vel = np.concatenate([lam.U_A, lam.u_w])  # (9,)
    ext_pose = SEn3.exp(2, ext_vel)

    result = VIOGroup()
    result.beta = lam.u_beta.copy()
    result.A = SE3(R=ext_pose.R, x=ext_pose.x[0])
    result.w = ext_pose.x[1].copy()
    result.B = SE3.exp(lam.U_B)

    result.id = list(lam.id)
    result.Q = [SOT3.exp(Wi) for Wi in lam.W]

    result.plane_id = list(lam.plane_id)
    result.Q_planes = [SOT3.exp(Wi) for Wi in lam.W_planes]

    return result


# ---------------------------------------------------------------------------
# Lift velocity (continuous)
# ---------------------------------------------------------------------------

def lift_velocity(state: VIOState, velocity) -> VIOAlgebra:
    """Lift IMU velocity from state space to VIO Lie algebra.

    Reference: liftVelocity() in VIOGroup.cpp

    Args:
        state: Current VIO state
        velocity: IMUVelocity (must support subtraction by bias -> (gyr, acc))

    Returns:
        VIOAlgebra element
    """
    sensor = state.sensor
    v_est = velocity - sensor.input_bias  # bias-corrected

    lift = VIOAlgebra()

    # Bias lift
    lift.u_beta = np.concatenate([velocity.gyr_bias_vel, velocity.acc_bias_vel])

    # SE(3) pose velocity: U_A = [omega; v] = [gyr; body_velocity]
    lift.U_A = np.concatenate([v_est.gyr, sensor.velocity])

    # Camera offset velocity: U_B = Ad_{T_C^{-1}} U_A
    lift.U_B = sensor.camera_offset.inverse().Adjoint() @ lift.U_A

    # R^3 velocity component
    lift.u_w = -v_est.acc + sensor.gravity_dir() * GRAVITY_CONSTANT

    # Camera-frame velocity for landmarks
    U_C = sensor.camera_offset.inverse().Adjoint() @ lift.U_A
    omega_C = U_C[:3]
    v_C = U_C[3:]

    # Point landmark lifts
    lift.W = []
    lift.id = []
    for lm in state.camera_landmarks:
        p = lm.p
        pp = p @ p
        Wi = np.zeros(4)
        Wi[:3] = omega_C + np.cross(p, v_C) / pp    # omega_L (with parallax)
        Wi[3] = (p @ v_C) / pp                        # sigma_L
        lift.W.append(Wi)
        lift.id.append(lm.id)

    # Plane landmark lifts (NEW, Section 2.2)
    lift.W_planes = []
    lift.plane_id = []
    for pl in state.plane_landmarks:
        q = pl.q
        Wi = np.zeros(4)
        Wi[:3] = omega_C              # no parallax term for planes
        Wi[3] = -(q @ v_C)            # note: no normalization by ||q||^2
        lift.W_planes.append(Wi)
        lift.plane_id.append(pl.id)

    return lift


# ---------------------------------------------------------------------------
# Lift velocity (discrete)
# ---------------------------------------------------------------------------

def lift_velocity_discrete(state: VIOState, velocity, dt: float) -> VIOGroup:
    """Discrete version of the VIO lift.

    Reference: liftVelocityDiscrete() in VIOGroup.cpp

    Args:
        state: Current VIO state
        velocity: IMUVelocity
        dt: Integration timestep

    Returns:
        VIOGroup element (discrete lift over dt)
    """
    sensor = state.sensor
    v_est = velocity - sensor.input_bias

    lift = VIOGroup()

    # Bias
    lift.beta = dt * np.concatenate([velocity.gyr_bias_vel, velocity.acc_bias_vel])

    # Pose: discrete integration (matching C++ exactly)
    lift.A = SE3()
    lift.A.R = SO3.exp(dt * v_est.gyr)

    # Translation in body frame (second-order integration)
    x_world = (dt * (sensor.pose.R * sensor.velocity)
                + 0.5 * dt * dt * (sensor.pose.R * v_est.acc
                                    + np.array([0.0, 0.0, -GRAVITY_CONSTANT])))
    lift.A.x = sensor.pose.R.inverse() * x_world

    # Camera offset
    lift.B = sensor.camera_offset.inverse() * lift.A * sensor.camera_offset

    # Velocity change
    body_vel_diff = v_est.acc - sensor.gravity_dir() * GRAVITY_CONSTANT
    lift.w = sensor.velocity - (sensor.velocity + dt * body_vel_diff)

    # Point landmark discrete lifts
    camera_pose_change_inv = (sensor.camera_offset.inverse()
                               * lift.A.inverse()
                               * sensor.camera_offset)
    N = len(state.camera_landmarks)
    lift.Q = []
    lift.id = []
    '''
    # This is the reference logic
    for lm in state.camera_landmarks:
        p0 = lm.p
        p1 = camera_pose_change_inv * p0

        # SOT(3) element taking p1 -> p0
        Qi = SOT3()
        Qi.R = SO3.SO3FromVectors(p1 / np.linalg.norm(p1), p0 / np.linalg.norm(p0))
        Qi.a = np.linalg.norm(p0) / np.linalg.norm(p1)
        lift.Q.append(Qi)
        lift.id.append(lm.id)
    '''
    
    # --- Alternative fast logic begins --- #
    ids = [lm.id for lm in state.camera_landmarks]
    p0_array = np.array([lm.p for lm in state.camera_landmarks]) # Shape: N x 3
    R_mat = camera_pose_change_inv.R.asMatrix() # 3x3 array
    x_vec = camera_pose_change_inv.x            # 3-element array
    p1_array = (R_mat @ p0_array.T).T + x_vec
    norm_p0 = np.linalg.norm(p0_array, axis=1, keepdims=True)
    norm_p1 = np.linalg.norm(p1_array, axis=1, keepdims=True)
    a_array = (norm_p0 / norm_p1).flatten()
    o = p1_array / norm_p1 
    d = p0_array / norm_p0 
    axes = np.cross(o, d)
    axis_norms = np.linalg.norm(axes, axis=1, keepdims=True)
    dots = np.sum(o * d, axis=1)
    angles = np.arccos(np.clip(dots, -1.0, 1.0)) 
    rot_vecs = np.zeros_like(axes)
    mask_normal = (axis_norms.flatten() >= 1e-10)
    rot_vecs[mask_normal] = (axes[mask_normal] / axis_norms[mask_normal]) * angles[mask_normal, np.newaxis]
    
    # Edge cases (parallel/antiparallel)
    mask_zero = ~mask_normal
    for i in np.where(mask_zero)[0]:
        if angles[i] >= np.pi / 2:
            perp = np.array([1.0, 0.0, 0.0]) if abs(o[i, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            ax = np.cross(o[i], perp)
            ax = ax / np.linalg.norm(ax)
            rot_vecs[i] = np.pi * ax
    
    batched_rotations = Rotation.from_rotvec(rot_vecs)
    for i in range(len(ids)):
        r = SO3()
        r.rotation = batched_rotations[i] 
        
        t = SOT3()
        t.R = r
        t.a = a_array[i] 
        
        lift.Q.append(t)
        lift.id.append(ids[i])
    # --- Alternative fast logic ends --- #

    # Plane landmark discrete lifts (NEW)
    # For planes, the dual discrete lift mirrors the point lift
    # but with the dual action convention
    lift.Q_planes = []
    lift.plane_id = []
    for pl in state.plane_landmarks:
        q0 = pl.q
        # Dual forward action of camera_pose_change_inv on q:
        # The camera moves by M, so the plane in the new camera frame is
        # q1 = M^{-dual} q0. We need the SOT(3) taking q1 back to q0.
        M = camera_pose_change_inv
        # Dual action of SE3 on CP: q_new = (R^T q) / (d_new/d_old)
        # For simplicity, compute via (n,d) representation:
        q_norm = np.linalg.norm(q0)
        n0 = q0 / q_norm
        d0 = 1.0 / q_norm
        R_M = M.R.asMatrix()
        t_M = M.x
        n1 = R_M.T @ n0
        d1 = d0 + n0 @ t_M
        q1 = n1 / d1

        # SOT(3) element: dual action Q takes q1 -> q0
        # q0 = (1/a) R q1, so R maps q1_hat to q0_hat, a = ||q1||/||q0||
        Qi = SOT3()
        Qi.R = SO3.SO3FromVectors(q1 / np.linalg.norm(q1), q0 / np.linalg.norm(q0))
        Qi.a = np.linalg.norm(q1) / np.linalg.norm(q0)  # dual: inverted ratio
        lift.Q_planes.append(Qi)
        lift.plane_id.append(pl.id)

    return lift
