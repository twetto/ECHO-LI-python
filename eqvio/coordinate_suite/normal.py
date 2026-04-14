"""
Normal (Polar) coordinate chart implementation for the EqF.

Port of: src/mathematical/coordinateSuite/normal.cpp
         + sphereChart_normal / pointChart_normal / sensorChart_normal
           / coordinateDifferential_normal_euclid from VIOState.cpp

Landmark parameterization: [stereo_bearing_normal(2), log_depth(1)]
    bearing:   SO(3)-exp-based chart on S^2 (omega truncation)
    log_depth: log(rho / rho0) = -log(||p|| / ||p0||)

Sensor state uses the SE2(3)-based chart that couples pose and
velocity together (world-frame velocity difference expressed in the
origin body frame), and a conjugated SE(3) on the camera offset.

State vector layout (Normal chart):
    [0,  6):   input bias                            — same as Euclidean
    [6, 15):   SE23-log of (A.R, A.x, v_A)           — differs from Euclidean
    [15, 21):  SE(3)-log of B = Tc0.inv * A * Tc     — differs from Euclidean
    [21+3i, 21+3(i+1)):  landmark i (normal stereo + log-depth)

Note: the C++ reference computes the state_matrix_A, input_matrix_B and
continuous lift_innovation via an M-sandwich of the Euclidean-chart
implementations, where M = coordinateDifferential_normal_euclid is the
Jacobian of the chart-to-chart change at the origin. We follow the
same approach here and compute M by forward-difference numerical
differentiation, matching the C++ implementation.
"""

from __future__ import annotations

import numpy as np
from liepp import SO3, SE3, SOT3, SEn3

from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark,
)
from eqvio.mathematical.vio_group import VIOGroup, VIOAlgebra
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.eqf_matrices import EqFCoordinateSuite
from eqvio.coordinate_suite.euclid import (
    state_chart_euclid,
    state_chart_inv_euclid,
    state_matrix_A_euclid,
    input_matrix_B_euclid,
    output_matrix_Ci_star_euclid,
    lift_innovation_euclid,
    lift_innovation_discrete_euclid,
)


# ===========================================================================
# Sphere chart: normal (SO(3)-exp based)
# Port of: sphereChart_normal in VIOState.cpp
# ===========================================================================

_E3 = np.array([0.0, 0.0, 1.0])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def sphere_chart_normal(eta: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """Normal chart on S^2: (3,), (3,) -> (2,).

    Maps pole -> 0 using SO(3) exp coordinates about e3.
    Reference: sphereChart_normal in VIOState.cpp
    """
    R = SO3.SO3FromVectors(pole, _E3)
    y = R * eta
    sin_th = np.linalg.norm(_skew(y) @ _E3)
    cos_th = float(y @ _E3)
    th = np.arctan2(sin_th, cos_th)
    if abs(th) < 1e-8:
        omega = _skew(y) @ _E3
    else:
        omega = (_skew(y) @ _E3) * (th / sin_th)
    return omega[:2].copy()


def sphere_chart_normal_inv(eps: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """Inverse normal chart on S^2: (2,), (3,) -> (3,).

    Reference: sphereChart_normal.inv in VIOState.cpp
    """
    omega = np.array([eps[0], eps[1], 0.0])
    y = SO3.exp(-omega) * _E3
    R = SO3.SO3FromVectors(pole, _E3)
    return R.inverse() * y


def sphere_chart_normal_diff0(pole: np.ndarray) -> np.ndarray:
    """Jacobian of the normal chart at the pole: (3,) -> (2, 3).

    Reference: sphereChart_normal.chartDiff0 in VIOState.cpp
    """
    R = SO3.SO3FromVectors(pole, _E3)
    diff = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    return diff @ R.asMatrix()


def sphere_chart_normal_inv_diff0(pole: np.ndarray) -> np.ndarray:
    """Jacobian of the inverse normal chart at 0: (3,) -> (3, 2).

    Reference: sphereChart_normal.chartInvDiff0 in VIOState.cpp
    """
    R = SO3.SO3FromVectors(pole, _E3)
    diff = np.array([
        [0.0, -1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ])
    return R.inverse().asMatrix() @ diff


# ===========================================================================
# Point chart: normal (stereo_normal bearing + log depth)
# ===========================================================================

def point_chart_normal(q: Landmark, q0: Landmark) -> np.ndarray:
    """Normal chart for a single landmark: (q, q0) -> eps(3,).

    eps = [sphere_normal_stereo(2), log(rho/rho0)]

    Reference: pointChart_normal in VIOState.cpp
    """
    rho = 1.0 / np.linalg.norm(q.p)
    rho0 = 1.0 / np.linalg.norm(q0.p)
    y = q.p * rho
    y0 = q0.p * rho0

    eps = np.zeros(3)
    eps[0:2] = sphere_chart_normal(y, y0)
    eps[2] = np.log(rho / rho0)
    return eps


def point_chart_normal_inv(eps: np.ndarray, q0: Landmark) -> Landmark:
    """Inverse normal chart: (eps, q0) -> Landmark.

    Reference: pointChart_normal.inv in VIOState.cpp
    """
    rho0 = 1.0 / np.linalg.norm(q0.p)
    y0 = q0.p * rho0
    y = sphere_chart_normal_inv(eps[0:2], y0)
    rho = rho0 * np.exp(eps[2])
    return Landmark(p=y / rho, id=q0.id)


# ===========================================================================
# Coordinate change helpers: Normal <-> Euclidean (landmark slot)
# ===========================================================================

def conv_normal2euc(q0: np.ndarray) -> np.ndarray:
    """3x3 Jacobian of the Normal-to-Euclidean landmark change at q0.

    Normal coords: [sphere_normal_bearing(2), log(rho/rho0)]
    Euclidean coords: [dx, dy, dz]

    At eps=0, with p = y/rho, y = sphere_normal_inv(eps[0:2], y0),
    rho = rho0*exp(eps[2]):
        dp/d(eps[0:2]) = (1/rho0) * sphere_chart_normal_inv_diff0(y0)
        dp/d(eps[2])   = -p0
    """
    rho0 = 1.0 / np.linalg.norm(q0)
    y0 = q0 * rho0
    M = np.zeros((3, 3))
    M[:, 0:2] = sphere_chart_normal_inv_diff0(y0) / rho0
    M[:, 2] = -q0
    return M


def conv_euc2normal(q0: np.ndarray) -> np.ndarray:
    """3x3 Jacobian of the Euclidean-to-Normal landmark change at q0."""
    rho0 = 1.0 / np.linalg.norm(q0)
    y0 = q0 * rho0
    M = np.zeros((3, 3))
    # d(sphere_normal_bearing)/dp = sphere_chart_normal_diff0(y0) * dy/dp
    # with dy/dp = (1/||q0||) * (I - y0 y0^T)
    M[0:2, :] = rho0 * sphere_chart_normal_diff0(y0) @ (
        np.eye(3) - np.outer(y0, y0)
    )
    # d(log(rho/rho0))/dp = -(1/||q0||^2) * y0^T
    M[2, :] = -rho0 * y0
    return M


# ===========================================================================
# Sensor chart: normal (SE23-based pose+velocity, conjugated camera offset)
# Port of: sensorChart_normal in VIOState.cpp
# ===========================================================================

def _se23_log(R: SO3, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    g = SEn3(2, R, [x0, x1])
    return SEn3.log(g)


def _se23_exp(u: np.ndarray):
    return SEn3.exp(2, u)


def sensor_chart_normal(xi: VIOSensorState, xi0: VIOSensorState) -> np.ndarray:
    """Sensor chart (normal): VIOSensorState -> eps(21,)."""
    A = xi0.pose.inverse() * xi.pose
    v_xi0_world = xi0.pose.R * xi0.velocity
    v_xi_world = xi.pose.R * xi.velocity
    v_A = xi0.pose.R.inverse() * (v_xi_world - v_xi0_world)
    B = xi0.camera_offset.inverse() * A * xi.camera_offset

    eps = np.zeros(VIOSensorState.CDim)
    eps[0:6] = xi.input_bias - xi0.input_bias
    eps[6:15] = _se23_log(A.R, A.x, v_A)
    eps[15:21] = SE3.log(B)
    return eps


def sensor_chart_inv_normal(eps: np.ndarray, xi0: VIOSensorState) -> VIOSensorState:
    """Inverse sensor chart (normal): eps(21,) -> VIOSensorState."""
    X = _se23_exp(eps[6:15])
    A = SE3(R=X.R, x=X.x[0])
    v_A = X.x[1]
    B = SE3.exp(eps[15:21])

    xi = VIOSensorState()
    xi.input_bias = xi0.input_bias + eps[0:6]
    xi.pose = xi0.pose * A
    v_xi0_world = xi0.pose.R * xi0.velocity
    xi.velocity = xi.pose.R.inverse() * (v_xi0_world + xi0.pose.R * v_A)
    xi.camera_offset = A.inverse() * xi0.camera_offset * B
    return xi


# ===========================================================================
# State chart: normal (sensor_normal + point_normal)
# Planes are passed through unchanged (no coord change defined in C++).
# ===========================================================================

def state_chart_normal(xi: VIOState, xi0: VIOState) -> np.ndarray:
    N = len(xi.camera_landmarks)
    Mpl = len(xi.plane_landmarks)
    S = VIOSensorState.CDim
    eps = np.zeros(S + 3 * N + 3 * Mpl)

    eps[0:S] = sensor_chart_normal(xi.sensor, xi0.sensor)

    for i in range(N):
        eps[S + 3 * i:S + 3 * (i + 1)] = point_chart_normal(
            xi.camera_landmarks[i], xi0.camera_landmarks[i]
        )

    plane_offset = S + 3 * N
    for i in range(Mpl):
        eps[plane_offset + 3 * i:plane_offset + 3 * (i + 1)] = (
            xi.plane_landmarks[i].q - xi0.plane_landmarks[i].q
        )

    return eps


def state_chart_inv_normal(eps: np.ndarray, xi0: VIOState) -> VIOState:
    S = VIOSensorState.CDim
    N = len(xi0.camera_landmarks)

    xi = VIOState()
    xi.sensor = sensor_chart_inv_normal(eps[0:S], xi0.sensor)

    xi.camera_landmarks = []
    for i in range(N):
        lm = point_chart_normal_inv(
            eps[S + 3 * i:S + 3 * (i + 1)], xi0.camera_landmarks[i]
        )
        xi.camera_landmarks.append(lm)

    plane_offset = S + 3 * N
    xi.plane_landmarks = []
    for i, pl0 in enumerate(xi0.plane_landmarks):
        q_new = pl0.q + eps[plane_offset + 3 * i:plane_offset + 3 * (i + 1)]
        xi.plane_landmarks.append(
            PlaneLandmark(q=q_new, id=pl0.id, point_ids=list(pl0.point_ids))
        )

    return xi


# ===========================================================================
# Coordinate differential: normal <- euclid (analytical, cached)
# Port of: coordinateDifferential_normal_euclid in VIOState.cpp
#
# The Jacobian at eps=0 of  eps_euc -> state_chart_normal(state_chart_inv_euclid(eps_euc, xi0), xi0)
# has a block-diagonal structure across (bias / sensor-pose-vel-Tc / landmark_i / plane_j)
# and each block has a closed form:
#
#   bias block                     = I_6
#   sensor 15x15 block (6:21)      = lower-block-triangular with identity diagonal,
#                                    off-diagonal entries = -skew(vel0) (v_A <- theta_pose)
#                                    and Ad_{Tc0^-1}     (se3_log_B <- (theta_pose, x_pose))
#   landmark block i               = conv_euc2normal(q0_i)
#   plane block j                  = I_3
#
# All blocks are independent of each other (no cross-terms), so M is block-diagonal.
# The C++ reference computes M via finite differences; here we do it analytically
# because the numerical path is expensive (O(dim^2) liepp calls per filter step)
# AND suffers from catastrophic cancellation in liepp's SE23.log near identity.
# ===========================================================================

def _build_M_analytical(xi0: VIOState) -> np.ndarray:
    S = VIOSensorState.CDim
    dim = xi0.dim()
    M = np.eye(dim)

    # --- Sensor block (indices 6:21) ---
    vel0 = xi0.sensor.velocity
    # v_A(12:15) <- theta_pose(6:9): -skew(vel0)
    M[12, 7] = vel0[2]
    M[12, 8] = -vel0[1]
    M[13, 6] = -vel0[2]
    M[13, 8] = vel0[0]
    M[14, 6] = vel0[1]
    M[14, 7] = -vel0[0]
    # SE3.log(B) (15:21) <- (theta_pose, x_pose) (6:12): Ad_{Tc0^{-1}}
    M[15:21, 6:12] = xi0.sensor.camera_offset.inverse().Adjoint()

    # --- Landmark blocks ---
    N = len(xi0.camera_landmarks)
    for i in range(N):
        q0 = xi0.camera_landmarks[i].p
        M[S + 3 * i:S + 3 * i + 3, S + 3 * i:S + 3 * i + 3] = conv_euc2normal(q0)

    # Plane blocks stay identity (from np.eye)
    return M


_M_CACHE = {'key': None, 'M': None, 'Minv': None}


def _cache_key(xi0: VIOState):
    return (
        id(xi0),
        tuple(lm.id for lm in xi0.camera_landmarks),
        tuple(pl.id for pl in xi0.plane_landmarks),
    )


def _get_M_and_inv(xi0: VIOState):
    key = _cache_key(xi0)
    if _M_CACHE['key'] == key:
        return _M_CACHE['M'], _M_CACHE['Minv']
    M = _build_M_analytical(xi0)
    Minv = np.linalg.inv(M)
    _M_CACHE['key'] = key
    _M_CACHE['M'] = M
    _M_CACHE['Minv'] = Minv
    return M, Minv


def coordinate_differential_normal_euclid(xi0: VIOState) -> np.ndarray:
    """Analytical Jacobian M such that  eps_normal = M @ eps_euc  at eps=0.

    Reference: coordinateDifferential_normal_euclid in VIOState.cpp
    """
    return _get_M_and_inv(xi0)[0]


# ===========================================================================
# A0t, Bt — State and input matrices (Normal, via M-sandwich of Euclidean)
# Port of: EqFStateMatrixA_normal / EqFInputMatrixB_normal in normal.cpp
# ===========================================================================

def state_matrix_A_normal(
    X: VIOGroup, xi0: VIOState, imu_vel: IMUVelocity
) -> np.ndarray:
    M, Minv = _get_M_and_inv(xi0)
    A_euc = state_matrix_A_euclid(X, xi0, imu_vel)
    return M @ A_euc @ Minv


def input_matrix_B_normal(X: VIOGroup, xi0: VIOState) -> np.ndarray:
    M, _ = _get_M_and_inv(xi0)
    B_euc = input_matrix_B_euclid(X, xi0)
    return M @ B_euc


# ===========================================================================
# C*_ti — Per-landmark equivariant output matrix (Normal)
# Port of: EqFoutputMatrixCiStar_normal in normal.cpp
# ===========================================================================

def output_matrix_Ci_star_normal(
    q0: np.ndarray, Q_hat: SOT3, cam_ptr, y: np.ndarray
) -> np.ndarray:
    """Equivariant output matrix block for one landmark (Normal chart).

    The third column (log-depth) is identically zero because a bearing
    measurement carries no direct depth information in this chart.

    Reference: EqFoutputMatrixCiStar_normal in normal.cpp
    """
    y0 = q0 / np.linalg.norm(q0)
    y_hat = Q_hat.R.inverse() * y0
    proj_jac = cam_ptr.projection_jacobian(y_hat)  # (2, 3)
    C0i = np.zeros((2, 3))
    C0i[:, 0:2] = (
        proj_jac @ Q_hat.R.asMatrix().T @ sphere_chart_normal_inv_diff0(q0)
    )
    return C0i


# ===========================================================================
# Innovation lifting — continuous (Normal, via M^{-1} then Euclidean)
# Port of: liftInnovation_normal in normal.cpp
# ===========================================================================

def lift_innovation_normal(
    total_innovation: np.ndarray, xi0: VIOState
) -> VIOAlgebra:
    _, Minv = _get_M_and_inv(xi0)
    inn_euc = Minv @ total_innovation
    return lift_innovation_euclid(inn_euc, xi0)


# ===========================================================================
# Innovation lifting — discrete (Normal, via chart round-trip)
# Port of: liftInnovationDiscrete_normal in normal.cpp
# ===========================================================================

def lift_innovation_discrete_normal(
    total_innovation: np.ndarray, xi0: VIOState
) -> VIOGroup:
    xi = state_chart_inv_normal(total_innovation, xi0)
    inn_euc = state_chart_euclid(xi, xi0)
    return lift_innovation_discrete_euclid(inn_euc, xi0)


# ===========================================================================
# Assemble the coordinate suite
# ===========================================================================

EqFCoordinateSuite_normal = EqFCoordinateSuite(
    state_chart=state_chart_normal,
    state_chart_inv=state_chart_inv_normal,
    state_matrix_A=state_matrix_A_normal,
    input_matrix_B=input_matrix_B_normal,
    output_matrix_Ci_star=output_matrix_Ci_star_normal,
    lift_innovation=lift_innovation_normal,
    lift_innovation_discrete=lift_innovation_discrete_normal,
)
