"""
Inverse-depth coordinate chart implementation for the EqF.

Port of: src/mathematical/coordinateSuite/invdepth.cpp
         + sphere chart utilities from VIOState.cpp

Landmark parameterization: [bearing(2), inverse_depth(1)]
    bearing:       stereographic projection of unit direction
    inverse_depth: rho = 1/||p||

State vector layout (InvDepth chart):
    [0,  6):   input bias (gyr_bias, acc_bias)         — same as Euclidean
    [6, 12):   pose (SO(3) rotation, position)         — same as Euclidean
    [12, 15):  body-fixed velocity                     — same as Euclidean
    [15, 21):  camera offset (SE(3))                   — same as Euclidean
    [21+3i, 21+3(i+1)):  landmark i (bearing_stereo(2), inv_depth(1))
"""

from __future__ import annotations

import numpy as np
from liepp import SO3, SE3, SOT3

from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, GRAVITY_CONSTANT,
)
from eqvio.mathematical.vio_group import (
    VIOGroup, VIOAlgebra, state_group_action,
)
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.eqf_matrices import EqFCoordinateSuite
from eqvio.coordinate_suite.euclid import (
    _skew,
    output_matrix_Ci_star_euclid,
)


# ===========================================================================
# Sphere chart utilities (stereographic projection)
# Port of: VIOState.cpp sphere chart functions
# ===========================================================================

def e3_project_sphere(eta: np.ndarray) -> np.ndarray:
    """Stereographic projection from S^2 about e3: (3,) -> (2,).

    Reference: e3ProjectSphere() in VIOState.cpp
    """
    return eta[:2] / (1.0 - eta[2])


def e3_project_sphere_inv(y: np.ndarray) -> np.ndarray:
    """Inverse stereographic projection about e3: (2,) -> (3,).

    Reference: e3ProjectSphereInv() in VIOState.cpp
    """
    e3 = np.array([0.0, 0.0, 1.0])
    y_bar = np.array([y[0], y[1], 0.0])
    return e3 + 2.0 / (y_bar @ y_bar + 1.0) * (y_bar - e3)


def e3_project_sphere_diff(eta: np.ndarray) -> np.ndarray:
    """Jacobian of stereographic projection: (3,) -> (2,3).

    Reference: e3ProjectSphereDiff() in VIOState.cpp
    """
    I23 = np.eye(2, 3)
    e3 = np.array([0.0, 0.0, 1.0])
    num = I23 @ (np.eye(3) * (1.0 - eta[2]) + np.outer(eta - e3, e3))
    denom = (1.0 - eta[2]) ** 2
    return num / denom


def e3_project_sphere_inv_diff(y: np.ndarray) -> np.ndarray:
    """Jacobian of inverse stereographic projection: (2,) -> (3,2).

    Reference: e3ProjectSphereInvDiff() in VIOState.cpp
    """
    sq = y @ y
    Diff = np.zeros((3, 2))
    Diff[0:2, 0:2] = np.eye(2) * (sq + 1.0) - 2.0 * np.outer(y, y)
    Diff[2, :] = 2.0 * y
    Diff = 2.0 / (sq + 1.0) ** 2 * Diff
    return Diff


def sphere_chart_stereo(eta: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """Stereographic chart on S^2 centered at pole: (3,), (3,) -> (2,).

    Reference: sphereChart_stereo_impl() in VIOState.cpp
    """
    R = SO3.SO3FromVectors(-pole, np.array([0.0, 0.0, 1.0]))
    eta_rotated = R * eta
    return e3_project_sphere(eta_rotated)


def sphere_chart_stereo_inv(y: np.ndarray, pole: np.ndarray) -> np.ndarray:
    """Inverse stereographic chart: (2,), (3,) -> (3,).

    Reference: sphereChart_stereo_inv_impl() in VIOState.cpp
    """
    eta_rotated = e3_project_sphere_inv(y)
    R = SO3.SO3FromVectors(-pole, np.array([0.0, 0.0, 1.0]))
    return R.inverse() * eta_rotated


def sphere_chart_stereo_diff0(pole: np.ndarray) -> np.ndarray:
    """Jacobian of stereographic chart at the pole: (3,) -> (2,3).

    Reference: sphereChart_stereo_diff0_impl() in VIOState.cpp
    """
    R = SO3.SO3FromVectors(-pole, np.array([0.0, 0.0, 1.0]))
    eta_rotated = R * pole
    return e3_project_sphere_diff(eta_rotated) @ R.asMatrix()


def sphere_chart_stereo_inv_diff0(pole: np.ndarray) -> np.ndarray:
    """Jacobian of inverse stereographic chart at zero: (3,) -> (3,2).

    Reference: sphereChart_stereo_inv_diff0_impl() in VIOState.cpp
    """
    R = SO3.SO3FromVectors(-pole, np.array([0.0, 0.0, 1.0]))
    return R.inverse().asMatrix() @ e3_project_sphere_inv_diff(np.zeros(2))


# ===========================================================================
# Coordinate change matrices: Euclidean <-> InvDepth
# ===========================================================================

def conv_euc2ind(q0: np.ndarray) -> np.ndarray:
    """3x3 Jacobian of the Euclidean-to-InvDepth coordinate change at q0.

    InvDepth coords: [bearing_stereo(2), inverse_depth(1)]
    Euclidean coords: [dx, dy, dz]

    Reference: conv_euc2ind lambda in invdepth.cpp
    """
    rho0 = 1.0 / np.linalg.norm(q0)
    y0 = q0 * rho0  # unit direction

    M = np.zeros((3, 3))
    M[0:2, :] = rho0 * sphere_chart_stereo_diff0(y0) @ (
        np.eye(3) - np.outer(y0, y0)
    )
    M[2, :] = -rho0 * rho0 * y0
    return M


def conv_ind2euc(q0: np.ndarray) -> np.ndarray:
    """3x3 Jacobian of the InvDepth-to-Euclidean coordinate change at q0.

    Reference: conv_ind2euc lambda in invdepth.cpp
    """
    rho0 = 1.0 / np.linalg.norm(q0)
    y0 = q0 * rho0

    M = np.zeros((3, 3))
    M[:, 0:2] = sphere_chart_stereo_inv_diff0(y0) / rho0
    M[:, 2] = -y0 / (rho0 * rho0)
    return M


# ===========================================================================
# InvDepth point chart
# ===========================================================================

def point_chart_invdepth(q: Landmark, q0: Landmark) -> np.ndarray:
    """InvDepth chart for a single landmark: (q, q0) -> eps(3,).

    Reference: pointChart_invdepth in VIOState.cpp

    eps = [stereo_bearing(2), delta_invdepth(1)]
    """
    rho = 1.0 / np.linalg.norm(q.p)
    rho0 = 1.0 / np.linalg.norm(q0.p)
    y = q.p * rho
    y0 = q0.p * rho0

    eps = np.zeros(3)
    eps[0:2] = sphere_chart_stereo(y, y0)
    eps[2] = rho - rho0
    return eps


def point_chart_invdepth_inv(eps: np.ndarray, q0: Landmark) -> Landmark:
    """Inverse InvDepth chart: (eps, q0) -> Landmark.

    Reference: pointChart_invdepth.inv in VIOState.cpp
    """
    rho0 = 1.0 / np.linalg.norm(q0.p)
    y0 = q0.p * rho0

    y = sphere_chart_stereo_inv(eps[0:2], y0)
    rho = eps[2] + rho0
    if rho <= 0.0:
        rho = 1e-6

    return Landmark(p=y / rho, id=q0.id)


# ===========================================================================
# InvDepth state chart
# ===========================================================================

def state_chart_invdepth(xi: VIOState, xi0: VIOState) -> np.ndarray:
    """InvDepth chart: xi -> eps.

    Sensor states use the standard chart (same as Euclidean).
    Landmarks use the inverse-depth parameterization.

    Reference: constructVIOChart(sensorChart_std, pointChart_invdepth) in VIOState.cpp
    """
    N = len(xi.camera_landmarks)
    S = VIOSensorState.CDim
    eps = np.zeros(S + 3 * N)

    # Sensor state (identical to Euclidean)
    eps[0:6] = xi.sensor.input_bias - xi0.sensor.input_bias
    eps[6:12] = SE3.log(xi0.sensor.pose.inverse() * xi.sensor.pose)
    eps[12:15] = xi.sensor.velocity - xi0.sensor.velocity
    eps[15:21] = SE3.log(xi0.sensor.camera_offset.inverse() * xi.sensor.camera_offset)

    # Landmarks (inverse-depth)
    for i in range(N):
        eps[S + 3 * i:S + 3 * (i + 1)] = point_chart_invdepth(
            xi.camera_landmarks[i], xi0.camera_landmarks[i]
        )

    return eps


def state_chart_inv_invdepth(eps: np.ndarray, xi0: VIOState) -> VIOState:
    """Inverse InvDepth chart: eps -> xi.

    Reference: constructVIOChart(...).inv in VIOState.cpp
    """
    S = VIOSensorState.CDim
    xi = VIOState()

    # Sensor state (identical to Euclidean)
    xi.sensor.input_bias = xi0.sensor.input_bias + eps[0:6]
    xi.sensor.pose = xi0.sensor.pose * SE3.exp(eps[6:12])
    xi.sensor.velocity = xi0.sensor.velocity + eps[12:15]
    xi.sensor.camera_offset = xi0.sensor.camera_offset * SE3.exp(eps[15:21])

    # Landmarks (inverse-depth)
    N = len(xi0.camera_landmarks)
    xi.camera_landmarks = []
    for i in range(N):
        lm = point_chart_invdepth_inv(
            eps[S + 3 * i:S + 3 * (i + 1)], xi0.camera_landmarks[i]
        )
        xi.camera_landmarks.append(lm)

    # Planes (passthrough)
    xi.plane_landmarks = list(xi0.plane_landmarks)

    return xi


# ===========================================================================
# A0t — State matrix (InvDepth)
# ===========================================================================

def state_matrix_A_invdepth(
    X: VIOGroup, xi0: VIOState, imu_vel: IMUVelocity
) -> np.ndarray:
    """Equivariant filter state matrix A0t (InvDepth chart).

    Reference: EqFStateMatrixA_invdepth() in invdepth.cpp

    The sensor blocks [0:21, 0:21] are identical to Euclidean.
    The landmark blocks are transformed by the coordinate change matrix.
    """
    N = len(xi0.camera_landmarks)
    dim = xi0.dim()
    A0t = np.zeros((dim, dim))
    S = VIOSensorState.CDim

    # --- Effect of bias: A[:, 0:6] = -B[:, 0:6] ---
    Bt = input_matrix_B_invdepth(X, xi0)
    A0t[:, 0:6] = -Bt[:, 0:6]

    # --- Effect of velocity on translation: A[9:12, 12:15] = I ---
    A0t[9:12, 12:15] = np.eye(3)

    # --- Effect of gravity cov on velocity cov ---
    g_dir = xi0.sensor.gravity_dir()
    A0t[12:15, 6:9] = -GRAVITY_CONSTANT * _skew(g_dir)

    # --- Precompute common terms ---
    xi_hat = state_group_action(X, xi0)
    v_est = imu_vel - xi_hat.sensor.input_bias
    U_I = np.concatenate([v_est.gyr, xi_hat.sensor.velocity])

    Ad_Tc_inv = xi0.sensor.camera_offset.inverse().Adjoint()
    Ad_A = X.A.Adjoint()
    transformed = Ad_Tc_inv @ Ad_A @ U_I

    # --- Effect of camera offset cov on self ---
    A0t[15:21, 15:21] = SE3.adjoint(transformed)

    # --- Landmark blocks: apply coordinate change ---
    R_IC = xi_hat.sensor.camera_offset.R.asMatrix()
    R_Ahat = X.A.R.asMatrix()

    # Effect of velocity cov on landmarks
    for i in range(N):
        q0 = xi0.camera_landmarks[i].p
        Qi = X.Q[i]
        Qhat_i = Qi.R.asMatrix() * Qi.a
        M_e2i = conv_euc2ind(q0)
        A0t[S + 3 * i:S + 3 * (i + 1), 12:15] = (
            -M_e2i @ Qhat_i @ R_IC.T @ R_Ahat.T
        )

    # Effect of camera offset cov on landmarks
    common_term = (X.B.inverse().Adjoint()
                   @ SE3.adjoint(Ad_Tc_inv @ Ad_A @ U_I))
    for i in range(N):
        q0 = xi0.camera_landmarks[i].p
        Qi = X.Q[i]
        R_Qi = Qi.R.asMatrix()
        temp = np.zeros((3, 6))
        temp[:, 0:3] = _skew(q0) @ R_Qi
        temp[:, 3:6] = -Qi.a * R_Qi
        M_e2i = conv_euc2ind(q0)
        A0t[S + 3 * i:S + 3 * (i + 1), 15:21] = M_e2i @ temp @ common_term

    # Effect of landmark cov on landmark cov
    U_C_full = xi_hat.sensor.camera_offset.inverse().Adjoint() @ U_I
    v_C = U_C_full[3:6]
    for i in range(N):
        q0 = xi0.camera_landmarks[i].p
        Qi = X.Q[i]
        Qhat_i = Qi.R.asMatrix() * Qi.a
        Qhat_i_inv = Qi.inverse()
        Qhat_i_inv_mat = Qhat_i_inv.R.asMatrix() * Qhat_i_inv.a
        qhat_i = xi_hat.camera_landmarks[i].p
        qq = qhat_i @ qhat_i
        inner = (
            _skew(qhat_i) @ _skew(v_C)
            - 2.0 * np.outer(v_C, qhat_i)
            + np.outer(qhat_i, v_C)
        )
        A_qi_euc = -Qhat_i @ inner @ Qhat_i_inv_mat / qq

        M_e2i = conv_euc2ind(q0)
        M_i2e = conv_ind2euc(q0)
        A0t[S + 3 * i:S + 3 * (i + 1), S + 3 * i:S + 3 * (i + 1)] = (
            M_e2i @ A_qi_euc @ M_i2e
        )

    return A0t


# ===========================================================================
# Bt — Input matrix (InvDepth)
# ===========================================================================

def input_matrix_B_invdepth(X: VIOGroup, xi0: VIOState) -> np.ndarray:
    """Equivariant filter input matrix Bt (InvDepth chart).

    Reference: EqFInputMatrixB_invdepth() in invdepth.cpp
    """
    N = len(xi0.camera_landmarks)
    dim = xi0.dim()
    S = VIOSensorState.CDim
    Bt = np.zeros((dim, IMUVelocity.CDim))

    xi_hat = state_group_action(X, xi0)
    R_A = X.A.R.asMatrix()

    # Sensor blocks (identical to Euclidean)
    Bt[0:6, 6:12] = np.eye(6)
    Bt[6:9, 0:3] = R_A
    Bt[9:12, 0:3] = _skew(X.A.x) @ R_A
    Bt[12:15, 0:3] = R_A @ _skew(xi_hat.sensor.velocity)
    Bt[12:15, 3:6] = R_A

    # Landmarks: apply coordinate change
    R_IC_T = xi_hat.sensor.camera_offset.R.inverse().asMatrix()
    x_IC = xi_hat.sensor.camera_offset.x
    for i in range(N):
        q0 = xi0.camera_landmarks[i].p
        Qi = X.Q[i]
        Qhat_i = Qi.R.asMatrix() * Qi.a
        qhat_i = xi_hat.camera_landmarks[i].p
        M_e2i = conv_euc2ind(q0)
        Bt[S + 3 * i:S + 3 * (i + 1), 0:3] = (
            M_e2i @ Qhat_i @ (_skew(qhat_i) @ R_IC_T + R_IC_T @ _skew(x_IC))
        )

    return Bt


# ===========================================================================
# C*_ti — Per-landmark equivariant output matrix (InvDepth)
# ===========================================================================

def output_matrix_Ci_star_invdepth(
    q0: np.ndarray, Q_hat: SOT3, cam_ptr, y: np.ndarray
) -> np.ndarray:
    """Equivariant output matrix block for one landmark (InvDepth chart).

    Reference: EqFoutputMatrixCiStar_invdepth() in invdepth.cpp

    C*_ti_invdepth = C*_ti_euclid @ ind2euc
    """
    M_i2e = conv_ind2euc(q0)
    return output_matrix_Ci_star_euclid(q0, Q_hat, cam_ptr, y) @ M_i2e


# ===========================================================================
# Innovation lifting — continuous (InvDepth)
# ===========================================================================

def lift_innovation_invdepth(
    total_innovation: np.ndarray, xi0: VIOState
) -> VIOAlgebra:
    """Lift from InvDepth tangent coordinates to VIO Lie algebra.

    Reference: liftInnovation_invdepth() in invdepth.cpp

    Sensor part is identical to Euclidean.
    Landmark part converts InvDepth perturbation to Euclidean first,
    then applies the same SOT(3) lifting.
    """
    S = VIOSensorState.CDim
    assert total_innovation.shape[0] == xi0.dim()

    Delta = VIOAlgebra()

    # Sensor state (identical to Euclidean)
    Delta.u_beta = total_innovation[0:6].copy()
    Delta.U_A = total_innovation[6:12].copy()

    gamma_v = total_innovation[12:15]
    omega_A = Delta.U_A[0:3]
    Delta.u_w = -gamma_v - _skew(omega_A) @ xi0.sensor.velocity

    Delta.U_B = (
        total_innovation[15:21]
        + xi0.sensor.camera_offset.inverse().Adjoint() @ Delta.U_A
    )

    # Point landmarks: convert InvDepth -> Euclidean, then lift
    N = len(xi0.camera_landmarks)
    Delta.id = []
    Delta.W = []
    for i in range(N):
        qi0 = xi0.camera_landmarks[i].p
        r0 = np.linalg.norm(qi0)
        y0 = qi0 / r0

        # ind2euc matrix (matching C++ exactly)
        ind2euc = np.zeros((3, 3))
        ind2euc[:, 0:2] = r0 * sphere_chart_stereo_inv_diff0(y0)
        ind2euc[:, 2] = -r0 * qi0

        # Convert InvDepth perturbation to Euclidean
        gamma_qi_invdepth = total_innovation[S + 3 * i:S + 3 * (i + 1)]
        gamma_qi_euclid = ind2euc @ gamma_qi_invdepth

        # SOT(3) lift (same as Euclidean)
        qq = qi0 @ qi0
        Wi = np.zeros(4)
        Wi[0:3] = -np.cross(qi0, gamma_qi_euclid) / qq
        Wi[3] = -(qi0 @ gamma_qi_euclid) / qq

        Delta.W.append(Wi)
        Delta.id.append(xi0.camera_landmarks[i].id)

    # Plane landmarks (same as Euclidean — planes don't use InvDepth)
    Delta.plane_id = []
    Delta.W_planes = []
    plane_offset = S + 3 * N
    for i, pl in enumerate(xi0.plane_landmarks):
        gamma_qi = total_innovation[plane_offset + 3 * i:plane_offset + 3 * (i + 1)]
        q0 = pl.q
        qq = q0 @ q0
        Wi = np.zeros(4)
        Wi[0:3] = -np.cross(q0, gamma_qi) / qq
        Wi[3] = +(q0 @ gamma_qi) / qq
        Delta.W_planes.append(Wi)
        Delta.plane_id.append(pl.id)

    return Delta


# ===========================================================================
# Innovation lifting — discrete (InvDepth)
# ===========================================================================

def lift_innovation_discrete_invdepth(
    total_innovation: np.ndarray, xi0: VIOState
) -> VIOGroup:
    """Discrete lift from InvDepth tangent coordinates to VIOGroup.

    Reference: liftInnovationDiscrete_invdepth() in invdepth.cpp

    Sensor part identical to Euclidean.
    Landmark part uses pointChart_invdepth.inv to recover the Euclidean
    perturbed position, then constructs the SOT(3) element.
    """
    S = VIOSensorState.CDim

    lift = VIOGroup()

    # Sensor state (identical to Euclidean)
    lift.beta = total_innovation[0:6].copy()
    lift.A = SE3.exp(total_innovation[6:12])

    gamma_v = total_innovation[12:15]
    v0 = xi0.sensor.velocity
    lift.w = v0 - lift.A.R * (v0 + gamma_v)

    lift.B = (
        xi0.sensor.camera_offset.inverse()
        * lift.A
        * xi0.sensor.camera_offset
        * SE3.exp(total_innovation[15:21])
    )

    # Point landmarks: use InvDepth chart inverse to get perturbed position
    N = len(xi0.camera_landmarks)
    lift.id = []
    lift.Q = []
    for i in range(N):
        q0 = xi0.camera_landmarks[i]
        Gamma_qi = total_innovation[S + 3 * i:S + 3 * (i + 1)]
        q1 = point_chart_invdepth_inv(Gamma_qi, q0)

        Qi = SOT3()
        Qi.R = SO3.SO3FromVectors(
            q1.p / np.linalg.norm(q1.p),
            q0.p / np.linalg.norm(q0.p)
        )
        Qi.a = np.linalg.norm(q0.p) / np.linalg.norm(q1.p)

        lift.Q.append(Qi)
        lift.id.append(q0.id)

    # Plane landmarks (same as Euclidean)
    plane_offset = S + 3 * N
    lift.plane_id = []
    lift.Q_planes = []
    for i, pl in enumerate(xi0.plane_landmarks):
        qi = pl.q
        Gamma_qi = total_innovation[plane_offset + 3 * i:plane_offset + 3 * (i + 1)]
        qi1 = qi + Gamma_qi

        Qi = SOT3()
        Qi.R = SO3.SO3FromVectors(
            qi1 / np.linalg.norm(qi1), qi / np.linalg.norm(qi)
        )
        Qi.a = np.linalg.norm(qi1) / np.linalg.norm(qi)

        lift.Q_planes.append(Qi)
        lift.plane_id.append(pl.id)

    return lift


# ===========================================================================
# Assemble the coordinate suite
# ===========================================================================

EqFCoordinateSuite_invdepth = EqFCoordinateSuite(
    state_chart=state_chart_invdepth,
    state_chart_inv=state_chart_inv_invdepth,
    state_matrix_A=state_matrix_A_invdepth,
    input_matrix_B=input_matrix_B_invdepth,
    output_matrix_Ci_star=output_matrix_Ci_star_invdepth,
    lift_innovation=lift_innovation_invdepth,
    lift_innovation_discrete=lift_innovation_discrete_invdepth,
)
