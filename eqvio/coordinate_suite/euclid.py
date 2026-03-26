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
    """Equivariant filter state matrix A0t (Euclidean chart).

    Reference: EqFStateMatrixA_euclid() in euclid.cpp

    State layout:
        [0,6) bias, [6,12) pose, [12,15) velocity,
        [15,21) camera offset, [21+3i,...) landmarks
    """
    N = len(xi0.camera_landmarks)
    dim = xi0.dim()
    A0t = np.zeros((dim, dim))
    S = VIOSensorState.CDim  # 21

    # --- Effect of bias on everything: A[:, 0:6] = -B[:, 0:6] ---
    Bt = input_matrix_B_euclid(X, xi0)
    A0t[:, 0:6] = -Bt[:, 0:6]

    # --- Effect of velocity on translation: A[9:12, 12:15] = I ---
    A0t[9:12, 12:15] = np.eye(3)

    # --- Effect of gravity cov on velocity: A[12:15, 6:9] ---
    g_dir = xi0.sensor.gravity_dir()
    A0t[12:15, 6:9] = -GRAVITY_CONSTANT * _skew(g_dir)

    # --- Effect of camera offset cov on self: A[15:21, 15:21] ---
    xi_hat = state_group_action(X, xi0)
    v_est = imu_vel - xi_hat.sensor.input_bias
    U_I = np.concatenate([v_est.gyr, xi_hat.sensor.velocity])  # se(3) velocity

    # ad(Ad_{T_C^{-1}} Ad_{A_hat} U_I)
    Ad_Tc_inv = xi0.sensor.camera_offset.inverse().Adjoint()
    Ad_A = X.A.Adjoint()
    transformed = Ad_Tc_inv @ Ad_A @ U_I
    A0t[15:21, 15:21] = SE3.adjoint(transformed)

    # --- Effect of velocity cov on landmarks ---
    R_IC = xi_hat.sensor.camera_offset.R.asMatrix()
    R_Ahat = X.A.R.asMatrix()
    for i in range(N):
        Qi = X.Q[i]
        Qhat_i = Qi.R.asMatrix() * Qi.a  # 3x3: a * R
        A0t[S + 3 * i:S + 3 * (i + 1), 12:15] = (
            -Qhat_i @ R_IC.T @ R_Ahat.T
        )

    # --- Effect of camera offset cov on landmarks ---
    Ad_Binv = X.B.inverse().Adjoint()
    common_term = Ad_Binv @ SE3.adjoint(Ad_Tc_inv @ Ad_A @ U_I)  # 6x6
    for i in range(N):
        Qi = X.Q[i]
        q0i = xi0.camera_landmarks[i].p
        R_Qi = Qi.R.asMatrix()
        # temp = [skew(q0i) @ R_Qi, -a_i * R_Qi]  shape (3, 6)
        temp = np.zeros((3, 6))
        temp[:, 0:3] = _skew(q0i) @ R_Qi
        temp[:, 3:6] = -Qi.a * R_Qi
        A0t[S + 3 * i:S + 3 * (i + 1), 15:21] = temp @ common_term

    # --- Effect of landmark cov on landmark cov ---
    U_C_full = xi_hat.sensor.camera_offset.inverse().Adjoint() @ U_I
    v_C = U_C_full[3:6]
    for i in range(N):
        Qi = X.Q[i]
        Qhat_i = Qi.R.asMatrix() * Qi.a
        Qhat_i_inv = Qi.inverse()
        Qhat_i_inv_mat = Qhat_i_inv.R.asMatrix() * Qhat_i_inv.a
        qhat_i = xi_hat.camera_landmarks[i].p
        qq = qhat_i @ qhat_i

        # A_qi = -Qhat * (skew(q)*skew(v) - 2*v*q^T + q*v^T) * Qhat^{-1} / ||q||^2
        inner = (
            _skew(qhat_i) @ _skew(v_C)
            - 2 * np.outer(v_C, qhat_i)
            + np.outer(qhat_i, v_C)
        )
        A_qi = -Qhat_i @ inner @ Qhat_i_inv_mat / qq
        A0t[S + 3 * i:S + 3 * (i + 1), S + 3 * i:S + 3 * (i + 1)] = A_qi

    # ===================================================================
    # Plane landmark blocks (NEW)
    # ===================================================================
    M = len(xi0.plane_landmarks)
    P = S + 3 * N  # plane block offset in state vector

    for j in range(M):
        Qj = X.Q_planes[j]
        Qhat_j = Qj.R.asMatrix() * Qj.a             # 3x3: a*R
        Qhat_j_inv = Qj.inverse()
        Qhat_j_inv_mat = Qhat_j_inv.R.asMatrix() * Qhat_j_inv.a  # (1/a)*R^{-1}
        qhat_j = xi_hat.plane_landmarks[j].q
        q0j = xi0.plane_landmarks[j].q
        R_Qj = Qj.R.asMatrix()

        # --- Plane self-block (validated to 7e-5 against numerical A) ---
        A_pj = Qhat_j @ np.outer(qhat_j, v_C) @ Qhat_j_inv_mat
        A0t[P + 3 * j:P + 3 * (j + 1), P + 3 * j:P + 3 * (j + 1)] = A_pj

        # --- Velocity → plane ---
        A0t[P + 3 * j:P + 3 * (j + 1), 12:15] = (
            np.outer(qhat_j, qhat_j) @ Qhat_j @ R_IC.T @ R_Ahat.T
        )

        # --- Camera offset → plane ---
        temp_plane = np.zeros((3, 6))
        temp_plane[:, 0:3] = -_skew(qhat_j) @ R_Qj
        temp_plane[:, 3:6] = np.outer(qhat_j, qhat_j) @ Qhat_j
        A0t[P + 3 * j:P + 3 * (j + 1), 15:21] = temp_plane @ common_term

    return A0t


# ---------------------------------------------------------------------------
# Bt — Input matrix
# ---------------------------------------------------------------------------

def input_matrix_B_euclid(X: VIOGroup, xi0: VIOState) -> np.ndarray:
    """Equivariant filter input matrix Bt (Euclidean chart).

    Reference: EqFInputMatrixB_euclid() in euclid.cpp

    Columns: [gyr(3), acc(3), gyr_bias_vel(3), acc_bias_vel(3)] = 12
    """
    N = len(xi0.camera_landmarks)
    dim = xi0.dim()
    S = VIOSensorState.CDim
    Bt = np.zeros((dim, IMUVelocity.CDim))

    xi_hat = state_group_action(X, xi0)
    R_A = X.A.R.asMatrix()

    # Biases: rows [0,6), cols [6,12)
    Bt[0:6, 6:12] = np.eye(6)

    # Attitude: rows [6,9), cols [0,3)
    Bt[6:9, 0:3] = R_A

    # Position: rows [9,12), cols [0,3)
    Bt[9:12, 0:3] = _skew(X.A.x) @ R_A

    # Body-fixed velocity: rows [12,15)
    Bt[12:15, 0:3] = R_A @ _skew(xi_hat.sensor.velocity)  # gyr
    Bt[12:15, 3:6] = R_A                                    # acc

    # Landmarks: rows [S+3i, S+3(i+1)), cols [0,3) (gyr only)
    R_IC_T = xi_hat.sensor.camera_offset.R.inverse().asMatrix()
    x_IC = xi_hat.sensor.camera_offset.x
    for i in range(N):
        Qi = X.Q[i]
        Qhat_i = Qi.R.asMatrix() * Qi.a
        qhat_i = xi_hat.camera_landmarks[i].p
        Bt[S + 3 * i:S + 3 * (i + 1), 0:3] = (
            Qhat_i @ (_skew(qhat_i) @ R_IC_T + R_IC_T @ _skew(x_IC))
        )

    # Plane landmark B rows (NEW)
    M = len(xi0.plane_landmarks)
    P = S + 3 * N
    for j in range(M):
        Qj = X.Q_planes[j]
        Qhat_j = Qj.R.asMatrix() * Qj.a
        qhat_j = xi_hat.plane_landmarks[j].q
        # Plane B: same structure as points but with outer(q,q) scaling
        Bt[P + 3 * j:P + 3 * (j + 1), 0:3] = (
            Qhat_j @ (-_skew(qhat_j) @ R_IC_T
                       - np.outer(qhat_j, qhat_j) @ R_IC_T @ _skew(x_IC))
        )

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
