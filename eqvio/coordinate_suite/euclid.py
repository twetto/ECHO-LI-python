"""
Euclidean coordinate chart implementation for the EqF.

Port of: src/mathematical/coordinateSuite/euclid.cpp

Provides the chart-specific functions for the EqFCoordinateSuite:
    - state_chart / state_chart_inv  (Euclidean local coordinates)
    - state_matrix_A                 (A0t)
    - input_matrix_B                 (Bt)
    - output_matrix_Ci_star          (C*_ti per landmark)
    - lift_innovation                (continuous)
    - lift_innovation_discrete       (discrete)

State vector layout (Euclidean chart):
    [0,  6):   input bias (gyr_bias, acc_bias)
    [6, 12):   pose (SO(3) rotation, position)
    [12, 15):  body-fixed velocity
    [15, 21):  camera offset (SE(3))
    [21+3i, 21+3(i+1)):  landmark i (Euclidean R^3)
"""

from __future__ import annotations

import numpy as np
from liepp import SO3, SE3, SOT3

from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark, GRAVITY_CONSTANT,
)
from eqvio.mathematical.vio_group import (
    VIOGroup, VIOAlgebra, state_group_action,
)
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.eqf_matrices import EqFCoordinateSuite


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def _batched_skew(v_arr: np.ndarray) -> np.ndarray:
    """Vectorized skew-symmetric matrix: (N, 3) -> (N, 3, 3)."""
    N = v_arr.shape[0]
    S = np.zeros((N, 3, 3), dtype=np.float64)
    S[:, 0, 1] = -v_arr[:, 2]
    S[:, 0, 2] =  v_arr[:, 1]
    S[:, 1, 0] =  v_arr[:, 2]
    S[:, 1, 2] = -v_arr[:, 0]
    S[:, 2, 0] = -v_arr[:, 1]
    S[:, 2, 1] =  v_arr[:, 0]
    return S


# ---------------------------------------------------------------------------
# Coordinate chart
# ---------------------------------------------------------------------------

def state_chart_euclid(xi: VIOState, xi0: VIOState) -> np.ndarray:
    """Euclidean chart: xi -> eps (deviation from xi0).

    Reference: sensorChart_std + pointChart_euclid in VIOState.cpp
    """
    N = len(xi.camera_landmarks)
    M = len(xi.plane_landmarks)
    S = VIOSensorState.CDim
    eps = np.zeros(S + 3 * N + 3 * M)

    # Sensor state
    eps[0:6] = xi.sensor.input_bias - xi0.sensor.input_bias
    eps[6:12] = SE3.log(xi0.sensor.pose.inverse() * xi.sensor.pose)
    eps[12:15] = xi.sensor.velocity - xi0.sensor.velocity
    eps[15:21] = SE3.log(xi0.sensor.camera_offset.inverse() * xi.sensor.camera_offset)

    # Point landmarks (Euclidean)
    for i in range(N):
        eps[S + 3 * i:S + 3 * (i + 1)] = (
            xi.camera_landmarks[i].p - xi0.camera_landmarks[i].p
        )

    # Plane landmarks (Euclidean on CP, NEW)
    plane_offset = S + 3 * N
    for i in range(M):
        eps[plane_offset + 3 * i:plane_offset + 3 * (i + 1)] = (
            xi.plane_landmarks[i].q - xi0.plane_landmarks[i].q
        )

    return eps


def state_chart_inv_euclid(eps: np.ndarray, xi0: VIOState) -> VIOState:
    """Inverse Euclidean chart: eps -> xi (state from deviation).

    Reference: sensorChart_std.inv + pointChart_euclid.inv in VIOState.cpp
    """
    from eqvio.mathematical.vio_state import VIOSensorState as VSS

    xi = VIOState()

    # Sensor state
    xi.sensor.input_bias = xi0.sensor.input_bias + eps[0:6]
    xi.sensor.pose = xi0.sensor.pose * SE3.exp(eps[6:12])
    xi.sensor.velocity = xi0.sensor.velocity + eps[12:15]
    xi.sensor.camera_offset = xi0.sensor.camera_offset * SE3.exp(eps[15:21])

    # Landmarks
    N = len(xi0.camera_landmarks)
    xi.camera_landmarks = []
    for i in range(N):
        p_new = xi0.camera_landmarks[i].p + eps[VSS.CDim + 3 * i:VSS.CDim + 3 * (i + 1)]
        xi.camera_landmarks.append(Landmark(p=p_new, id=xi0.camera_landmarks[i].id))

    # Plane landmarks (NEW)
    plane_offset = VSS.CDim + 3 * N
    xi.plane_landmarks = []
    for i, pl0 in enumerate(xi0.plane_landmarks):
        q_new = pl0.q + eps[plane_offset + 3 * i:plane_offset + 3 * (i + 1)]
        xi.plane_landmarks.append(
            PlaneLandmark(q=q_new, id=pl0.id, point_ids=list(pl0.point_ids))
        )

    return xi


# ---------------------------------------------------------------------------
# A0t — State matrix
# ---------------------------------------------------------------------------

def state_matrix_A_euclid(
    X: VIOGroup, xi0: VIOState, imu_vel: IMUVelocity
) -> np.ndarray:
    """Fully Vectorized EqF state matrix A0t (Euclidean chart)."""
    N = len(xi0.camera_landmarks)
    M = len(xi0.plane_landmarks)
    dim = xi0.dim()
    A0t = np.zeros((dim, dim))
    S = VIOSensorState.CDim

    # --- Sensor State (Same as before) ---
    Bt = input_matrix_B_euclid(X, xi0)
    A0t[:, 0:6] = -Bt[:, 0:6]
    A0t[9:12, 12:15] = np.eye(3)
    A0t[12:15, 6:9] = -GRAVITY_CONSTANT * _skew(xi0.sensor.gravity_dir())

    xi_hat = state_group_action(X, xi0)
    v_est = imu_vel - xi_hat.sensor.input_bias
    U_I = np.concatenate([v_est.gyr, xi_hat.sensor.velocity])

    Ad_Tc_inv = xi0.sensor.camera_offset.inverse().Adjoint()
    Ad_A = X.A.Adjoint()
    transformed = Ad_Tc_inv @ Ad_A @ U_I
    A0t[15:21, 15:21] = SE3.adjoint(transformed)

    R_IC = xi_hat.sensor.camera_offset.R.asMatrix()
    R_Ahat = X.A.R.asMatrix()
    common_term = X.B.inverse().Adjoint() @ SE3.adjoint(transformed)
    U_C_full = Ad_Tc_inv @ U_I
    v_C = U_C_full[3:6]

    # ===================================================================
    # Vectorized Point Landmarks
    # ===================================================================
    if N > 0:
        # Extract once to NumPy space
        a_arr = np.array([Q.a for Q in X.Q])  # (N,)
        R_arr = np.array([Q.R.asMatrix() for Q in X.Q])  # (N, 3, 3)
        Qhat_arr = R_arr * a_arr[:, None, None]

        # --- Velocity -> Landmarks (Block Assignment) ---
        M_vel = R_IC.T @ R_Ahat.T
        A0t[S:S+3*N, 12:15] = -(Qhat_arr @ M_vel).reshape(3*N, 3)

        # --- Camera Offset -> Landmarks ---
        q0_arr = np.array([lm.p for lm in xi0.camera_landmarks])  # (N, 3)
        skew_q0 = _batched_skew(q0_arr)  # (N, 3, 3)

        temp_arr = np.zeros((N, 3, 6))
        temp_arr[:, :, 0:3] = skew_q0 @ R_arr
        temp_arr[:, :, 3:6] = -Qhat_arr
        A0t[S:S+3*N, 15:21] = (temp_arr @ common_term).reshape(3*N, 6)

        # --- Landmark -> Landmark (Block Diagonal) ---
        qhat_arr = np.array([lm.p for lm in xi_hat.camera_landmarks])
        qq_arr = np.sum(qhat_arr**2, axis=1)
        skew_qhat = _batched_skew(qhat_arr)
        skew_vC = _skew(v_C)

        term2 = 2.0 * (v_C.reshape(1, 3, 1) * qhat_arr.reshape(N, 1, 3))
        term3 = qhat_arr.reshape(N, 3, 1) * v_C.reshape(1, 1, 3)
        inner_arr = (skew_qhat @ skew_vC) - term2 + term3

        Qhat_inv_arr = R_arr.transpose(0, 2, 1) / a_arr[:, None, None]
        A_qi_arr = -(Qhat_arr @ inner_arr @ Qhat_inv_arr) / qq_arr[:, None, None]

        for i in range(N):
            A0t[S+3*i:S+3*i+3, S+3*i:S+3*i+3] = A_qi_arr[i]

    # ===================================================================
    # Vectorized Plane Landmarks
    # ===================================================================
    P = S + 3 * N
    if M > 0:
        a_p_arr = np.array([Q.a for Q in X.Q_planes])
        R_p_arr = np.array([Q.R.asMatrix() for Q in X.Q_planes])
        Qhat_p_arr = R_p_arr * a_p_arr[:, None, None]
        Qhat_p_inv_arr = R_p_arr.transpose(0, 2, 1) / a_p_arr[:, None, None]

        qhat_p_arr = np.array([lm.q for lm in xi_hat.plane_landmarks])
        outer_qv = qhat_p_arr.reshape(M, 3, 1) * v_C.reshape(1, 1, 3)

        # --- Plane -> Plane (Block Diagonal) ---
        A_pj_arr = Qhat_p_arr @ outer_qv @ Qhat_p_inv_arr
        for j in range(M):
            A0t[P+3*j:P+3*j+3, P+3*j:P+3*j+3] = A_pj_arr[j]

        # --- Velocity -> Plane ---
        outer_qq = qhat_p_arr.reshape(M, 3, 1) * qhat_p_arr.reshape(M, 1, 3)
        A0t[P:P+3*M, 12:15] = (outer_qq @ Qhat_p_arr @ M_vel).reshape(3*M, 3)

        # --- Camera Offset -> Plane ---
        skew_qhat_p = _batched_skew(qhat_p_arr)
        temp_p = np.zeros((M, 3, 6))
        temp_p[:, :, 0:3] = -skew_qhat_p @ R_p_arr
        temp_p[:, :, 3:6] = outer_qq @ Qhat_p_arr
        A0t[P:P+3*M, 15:21] = (temp_p @ common_term).reshape(3*M, 6)

    return A0t


# ---------------------------------------------------------------------------
# Bt — Input matrix
# ---------------------------------------------------------------------------

def input_matrix_B_euclid(X: VIOGroup, xi0: VIOState) -> np.ndarray:
    """Fully Vectorized EqF input matrix Bt (Euclidean chart)."""
    N = len(xi0.camera_landmarks)
    M = len(xi0.plane_landmarks)
    dim = xi0.dim()
    S = VIOSensorState.CDim
    Bt = np.zeros((dim, IMUVelocity.CDim))

    xi_hat = state_group_action(X, xi0)
    R_A = X.A.R.asMatrix()

    # --- Sensor State ---
    Bt[0:6, 6:12] = np.eye(6)
    Bt[6:9, 0:3] = R_A
    Bt[9:12, 0:3] = _skew(X.A.x) @ R_A
    Bt[12:15, 0:3] = R_A @ _skew(xi_hat.sensor.velocity)
    Bt[12:15, 3:6] = R_A

    R_IC_T = xi_hat.sensor.camera_offset.R.inverse().asMatrix()
    x_IC = xi_hat.sensor.camera_offset.x
    term_x_IC = R_IC_T @ _skew(x_IC)  # (3,3)

    # --- Vectorized Point Landmarks ---
    if N > 0:
        a_arr = np.array([Q.a for Q in X.Q])
        R_arr = np.array([Q.R.asMatrix() for Q in X.Q])
        Qhat_arr = R_arr * a_arr[:, None, None]

        qhat_arr = np.array([lm.p for lm in xi_hat.camera_landmarks])
        skew_qhat = _batched_skew(qhat_arr)

        inner = (skew_qhat @ R_IC_T) + term_x_IC
        Bt[S:S+3*N, 0:3] = (Qhat_arr @ inner).reshape(3*N, 3)

    # --- Vectorized Plane Landmarks ---
    P = S + 3 * N
    if M > 0:
        a_p_arr = np.array([Q.a for Q in X.Q_planes])
        R_p_arr = np.array([Q.R.asMatrix() for Q in X.Q_planes])
        Qhat_p_arr = R_p_arr * a_p_arr[:, None, None]

        qhat_p_arr = np.array([lm.q for lm in xi_hat.plane_landmarks])
        skew_qhat_p = _batched_skew(qhat_p_arr)
        outer_qq = qhat_p_arr.reshape(M, 3, 1) * qhat_p_arr.reshape(M, 1, 3)

        inner_p = -(skew_qhat_p @ R_IC_T) - (outer_qq @ term_x_IC)
        Bt[P:P+3*M, 0:3] = (Qhat_p_arr @ inner_p).reshape(3*M, 3)

    return Bt


# ---------------------------------------------------------------------------
# C*_ti — Per-landmark equivariant output matrix
# ---------------------------------------------------------------------------

def output_matrix_Ci_star_euclid(
    q0: np.ndarray, Q_hat: SOT3, cam_ptr, y: np.ndarray
) -> np.ndarray:
    """Equivariant output matrix block for one landmark (Euclidean chart).

    Reference: EqFoutputMatrixCiStar_euclid() in euclid.cpp

    Args:
        q0:      landmark position at origin xi0 (3,)
        Q_hat:   SOT(3) observer element for this landmark
        cam_ptr: camera model with projection_jacobian, undistort_point
        y:       observed pixel coordinates (2,)

    Returns:
        (2, 3) matrix C*_ti
    """
    q_hat = Q_hat.inverse() * q0
    y_hat = q_hat / np.linalg.norm(q_hat)  # normalized bearing

    # Map from Euclidean R^3 perturbation to sot(3) (omega, sigma)
    # m2g = [-skew(q0); -q0^T] / ||q0||^2   shape (4, 3)
    qq = q0 @ q0
    m2g = np.zeros((4, 3))
    m2g[0:3, :] = -_skew(q0) / qq
    m2g[3, :] = -q0 / qq

    def D_rho(y_vec):
        """(2, 4) matrix: projection Jacobian composed with SOT(3) action.

        DRho = projJac(y) @ [skew(y), 0]
        """
        D_rho_vec = np.zeros((3, 4))
        D_rho_vec[0:3, 0:3] = _skew(y_vec)
        # D_rho_vec[:, 3] = 0  (already zero)
        proj_jac = cam_ptr.projection_jacobian(y_vec)  # (2, 3)
        return proj_jac @ D_rho_vec  # (2, 4)

    # True bearing from observation
    y_tru = cam_ptr.undistort_point(y)  # unit bearing (3,)

    # SOT(3) inverse adjoint (4x4)
    Q_inv_adj = Q_hat.inverse().Adjoint()

    # Equivariant average of true and predicted
    Cti = 0.5 * (D_rho(y_tru) + D_rho(y_hat)) @ Q_inv_adj @ m2g

    return Cti  # (2, 3)


# ---------------------------------------------------------------------------
# Innovation lifting — continuous
# ---------------------------------------------------------------------------

def lift_innovation_euclid(
    total_innovation: np.ndarray, xi0: VIOState
) -> VIOAlgebra:
    """Lift the correction from tangent coordinates to the Lie algebra.

    Reference: liftInnovation_euclid() in euclid.cpp

    Args:
        total_innovation: (dim,) vector in Euclidean chart coordinates
        xi0:              origin state

    Returns:
        VIOAlgebra element Delta
    """
    S = VIOSensorState.CDim
    assert total_innovation.shape[0] == xi0.dim()

    Delta = VIOAlgebra()

    # Bias
    Delta.u_beta = total_innovation[0:6].copy()

    # Pose (se(3))
    Delta.U_A = total_innovation[6:12].copy()

    # Velocity: Delta_w = -gamma_v - skew(omega_A) @ v0
    gamma_v = total_innovation[12:15]
    omega_A = Delta.U_A[0:3]
    Delta.u_w = -gamma_v - _skew(omega_A) @ xi0.sensor.velocity

    # Camera offset: Delta_B = gamma_B + Ad_{T_C^{-1}} @ Delta_A
    Delta.U_B = (
        total_innovation[15:21]
        + xi0.sensor.camera_offset.inverse().Adjoint() @ Delta.U_A
    )

    # Point landmarks
    N = len(xi0.camera_landmarks)
    Delta.id = []
    Delta.W = []
    for i in range(N):
        gamma_qi = total_innovation[S + 3 * i:S + 3 * (i + 1)]
        q0 = xi0.camera_landmarks[i].p
        qq = q0 @ q0

        Wi = np.zeros(4)
        Wi[0:3] = -np.cross(q0, gamma_qi) / qq   # rotation
        Wi[3] = -(q0 @ gamma_qi) / qq             # scale (negative for points)

        Delta.W.append(Wi)
        Delta.id.append(xi0.camera_landmarks[i].id)

    # Plane landmarks (NEW — Section 2.6 of porting guide)
    Delta.plane_id = []
    Delta.W_planes = []
    # Planes come after points in the state vector
    plane_offset = S + 3 * N
    for i, pl in enumerate(xi0.plane_landmarks):
        gamma_qi = total_innovation[plane_offset + 3 * i:plane_offset + 3 * (i + 1)]
        q0 = pl.q
        qq = q0 @ q0

        Wi = np.zeros(4)
        Wi[0:3] = -np.cross(q0, gamma_qi) / qq   # rotation (same formula)
        Wi[3] = +(q0 @ gamma_qi) / qq             # scale (POSITIVE for planes — dual)

        Delta.W_planes.append(Wi)
        Delta.plane_id.append(pl.id)

    return Delta


# ---------------------------------------------------------------------------
# Innovation lifting — discrete
# ---------------------------------------------------------------------------

def lift_innovation_discrete_euclid(
    total_innovation: np.ndarray, xi0: VIOState
) -> VIOGroup:
    """Discrete lift: correction from tangent coordinates to group element.

    Reference: liftInnovationDiscrete_euclid() in euclid.cpp
    """
    S = VIOSensorState.CDim

    lift = VIOGroup()

    # Bias
    lift.beta = total_innovation[0:6].copy()

    # Pose
    lift.A = SE3.exp(total_innovation[6:12])

    # Velocity
    gamma_v = total_innovation[12:15]
    v0 = xi0.sensor.velocity
    lift.w = v0 - lift.A.R * (v0 + gamma_v)

    # Camera offset
    lift.B = (
        xi0.sensor.camera_offset.inverse()
        * lift.A
        * xi0.sensor.camera_offset
        * SE3.exp(total_innovation[15:21])
    )

    # Point landmarks
    N = len(xi0.camera_landmarks)
    lift.id = []
    lift.Q = []
    for i in range(N):
        qi = xi0.camera_landmarks[i].p
        Gamma_qi = total_innovation[S + 3 * i:S + 3 * (i + 1)]
        qi1 = qi + Gamma_qi

        Qi = SOT3()
        Qi.R = SO3.SO3FromVectors(qi1 / np.linalg.norm(qi1), qi / np.linalg.norm(qi))
        Qi.a = np.linalg.norm(qi) / np.linalg.norm(qi1)

        lift.Q.append(Qi)
        lift.id.append(xi0.camera_landmarks[i].id)

    # Plane landmarks (NEW)
    plane_offset = S + 3 * N
    lift.plane_id = []
    lift.Q_planes = []
    for i, pl in enumerate(xi0.plane_landmarks):
        qi = pl.q
        Gamma_qi = total_innovation[plane_offset + 3 * i:plane_offset + 3 * (i + 1)]
        qi1 = qi + Gamma_qi

        # Dual discrete lift: direction maps qi1_hat -> qi_hat,
        # scale ratio inverted (dual convention)
        Qi = SOT3()
        Qi.R = SO3.SO3FromVectors(qi1 / np.linalg.norm(qi1), qi / np.linalg.norm(qi))
        Qi.a = np.linalg.norm(qi1) / np.linalg.norm(qi)  # inverted for dual

        lift.Q_planes.append(Qi)
        lift.plane_id.append(pl.id)

    return lift


# ---------------------------------------------------------------------------
# Assemble the coordinate suite
# ---------------------------------------------------------------------------

EqFCoordinateSuite_euclid = EqFCoordinateSuite(
    state_chart=state_chart_euclid,
    state_chart_inv=state_chart_inv_euclid,
    state_matrix_A=state_matrix_A_euclid,
    input_matrix_B=input_matrix_B_euclid,
    output_matrix_Ci_star=output_matrix_Ci_star_euclid,
    lift_innovation=lift_innovation_euclid,
    lift_innovation_discrete=lift_innovation_discrete_euclid,
)
