"""
Plane constraint measurement for the EqF.

NEW — not in C++ codebase.

Implements the point-on-plane constraint h(p, q) = q^T p + 1 = 0
and its equivariant output matrix C*_t for the Euclidean coordinate chart.

The constraint is exactly invariant under the dual SOT(3) action:
    h(Q^{-1} * p, Q^{-1}_{dual} * q) = h(p, q)
which means C*_{t,pose} = 0 — no pose columns needed.

Key functions:
    constraint_residual          — innovation scalar for one (point, plane) pair
    constraint_Ci_star_euclid    — (1×3) C* blocks for point and plane columns
    build_stacked_update         — assemble bearing + constraint into one Kalman update
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np

from liepp import SOT3

from eqvio.mathematical.vio_state import VIOState, VIOSensorState, Landmark, PlaneLandmark


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


# ---------------------------------------------------------------------------
# Constraint residual
# ---------------------------------------------------------------------------

def constraint_residual(p_hat: np.ndarray, q_hat: np.ndarray) -> float:
    """Point-on-plane constraint innovation.

    Returns -(q_hat^T p_hat + 1) so the Kalman update drives it toward zero.

    Args:
        p_hat: estimated point position in camera frame (3,)
        q_hat: estimated plane CP in camera frame (3,)

    Returns:
        Scalar residual
    """
    return -(q_hat @ p_hat + 1.0)


# ---------------------------------------------------------------------------
# Equivariant output matrix for constraint (Euclidean chart)
# ---------------------------------------------------------------------------

def constraint_Ci_star_euclid(
    p0: np.ndarray, Q_p: SOT3,
    q0: np.ndarray, Q_q: SOT3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Equivariant output matrix blocks for the point-on-plane constraint.

    Computes C*_t for h(p, q) = q^T p + 1 in the Euclidean coordinate chart.

    The derivation follows the same chain rule as the bearing C*_ti:
        C*_chart = C*_algebra @ Q_inv_adj @ m2g

    where:
        C*_algebra_p = [cross(q, p)^T, -(q^T p)]    (1×4 in sot(3))
        C*_algebra_q = -C*_algebra_p                  (exact anti-symmetry)
        m2g_point    = [-skew(p0); -p0^T] / ||p0||^2  (4×3)
        m2g_plane    = [-skew(q0); +q0^T] / ||q0||^2  (4×3, sign flip on sigma row)

    Args:
        p0:   point position at origin xi0 (3,)
        Q_p:  SOT(3) observer element for this point
        q0:   plane CP at origin xi0 (3,)
        Q_q:  SOT(3) observer element for this plane

    Returns:
        C_p: (1, 3) — point columns in Euclidean chart
        C_q: (1, 3) — plane columns in Euclidean chart
    """
    # Estimated values via group action
    p_hat = Q_p.inverse() * p0           # standard inverse action
    q_hat = Q_q.a * (Q_q.R.inverse() * q0)  # dual inverse action

    # Algebra-level C* (1×4 in sot(3) = [omega(3), sigma(1)])
    qxp = np.cross(q_hat, p_hat)
    qdp = q_hat @ p_hat
    C_alg_p = np.array([qxp[0], qxp[1], qxp[2], -qdp])  # (4,)
    C_alg_q = -C_alg_p                                     # exact anti-symmetry

    # m2g: maps R^3 perturbation -> sot(3) algebra element
    # Point: from lift_innovation_euclid
    pp = p0 @ p0
    m2g_p = np.zeros((4, 3))
    m2g_p[0:3, :] = -_skew(p0) / pp
    m2g_p[3, :] = -p0 / pp

    # Plane: dual sign on sigma row (from lift_innovation_euclid, plane section)
    qq = q0 @ q0
    m2g_q = np.zeros((4, 3))
    m2g_q[0:3, :] = -_skew(q0) / qq
    m2g_q[3, :] = +q0 / qq  # positive for planes (dual)

    # SOT(3) inverse adjoint (4×4)
    Q_p_inv_adj = Q_p.inverse().Adjoint()
    Q_q_inv_adj = Q_q.inverse().Adjoint()

    # Chain rule: C*_chart = C*_alg @ Q_inv_adj @ m2g
    C_p = (C_alg_p @ Q_p_inv_adj @ m2g_p).reshape(1, 3)
    C_q = (C_alg_q @ Q_q_inv_adj @ m2g_q).reshape(1, 3)

    return C_p, C_q


# ---------------------------------------------------------------------------
# Chart-to-Euclidean Jacobians for constraint C* transformation
# ---------------------------------------------------------------------------

def conv_polar2euc(p0: np.ndarray) -> np.ndarray:
    """3×3 Jacobian of the Polar(SOT3)-to-Euclidean coordinate change at p0.

    The polar chart maps ε = [ε₁, ε₂, ε₃] to sot(3) via injection E:
        [ε₁, ε₂, 0, ε₃] → (rotation around x, y; scale)

    At ε=0 the Euclidean perturbation is:
        δp = -[ε₁,ε₂,0] × p₀ - ε₃ · p₀

    So the Jacobian is:
        J[:, 0] = -e₁ × p₀ = skew(p₀) @ e₁
        J[:, 1] = -e₂ × p₀ = skew(p₀) @ e₂
        J[:, 2] = -p₀

    This gives C*_polar = C*_euc @ J, which at convergence yields:
        C*_polar ≈ [(q×p)_x, (q×p)_y, 1]
    with the depth-independent third column — the key property.
    """
    J = np.zeros((3, 3))
    J[:, 0:2] = _skew(p0)[:, 0:2]  # first two columns of skew(p0)
    J[:, 2] = -p0
    return J


def constraint_Ci_star_for_chart(
    p0: np.ndarray, Q_p: SOT3,
    q0: np.ndarray, Q_q: SOT3,
    point_chart_jacobian=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Constraint C* blocks adapted to the active point coordinate chart.

    Computes C*_p in Euclidean coordinates first, then transforms to the
    active chart via the chain rule:
        C*_chart = C*_euc @ J_chart2euc(p0)

    The plane C*_q is unaffected (planes always use Euclidean CP coords).

    Args:
        p0, Q_p, q0, Q_q: same as constraint_Ci_star_euclid
        point_chart_jacobian: function(p0) -> (3,3) Jacobian mapping
            chart perturbation to Euclidean perturbation.
            None = Euclidean (identity transform).
            Use conv_ind2euc for InvDepth, conv_polar2euc for Polar.

    Returns:
        C_p: (1, 3) — point columns in active chart
        C_q: (1, 3) — plane columns (always Euclidean CP)
    """
    C_p_euc, C_q = constraint_Ci_star_euclid(p0, Q_p, q0, Q_q)

    if point_chart_jacobian is not None:
        J = point_chart_jacobian(p0)
        C_p = (C_p_euc @ J).reshape(1, 3)
    else:
        C_p = C_p_euc

    return C_p, C_q


# ---------------------------------------------------------------------------
# Stacked bearing + constraint update assembly
# ---------------------------------------------------------------------------

def build_stacked_update(
    xi0: VIOState,
    X,  # VIOGroup
    y_ids: List[int],
    y_coords: Dict[int, np.ndarray],
    cam_ptr,
    output_matrix_Ci_star,
    sigma_bearing: float,
    sigma_constraint: float,
    use_equivariance: bool = True,
    include_constraints: bool = True,
    eligible_constraint_ids: set = None,
    point_chart_jacobian=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the full stacked bearing + constraint Kalman update.

    For each observed point:
        - 2 bearing rows (existing EqVIO)
    For each observed point that lies on a plane in the state (if include_constraints):
        - 1 constraint row (NEW), only if point is in eligible_constraint_ids

    When include_constraints=False, only bearing rows are produced but the
    output matrix still has correct full-state width (including plane columns).

    Args:
        eligible_constraint_ids: if not None, only these point IDs get constraint
            rows. If None, all plane-associated points are eligible.
        point_chart_jacobian: function(p0) -> (3,3) Jacobian mapping chart
            perturbation to Euclidean perturbation. None = Euclidean chart.
            Use conv_ind2euc for InvDepth, conv_polar2euc for Polar/SOT(3).

    Returns:
        residual:   (n_rows,) stacked innovation vector
        C_star:     (n_rows, dim) stacked output matrix
        R_noise:    (n_rows, n_rows) stacked noise covariance
    """
    from eqvio.mathematical.vio_group import state_group_action
    from eqvio.mathematical.vision_measurement import measure_system_state

    M_pts = len(xi0.camera_landmarks)
    M_pls = len(xi0.plane_landmarks)
    S = VIOSensorState.CDim
    dim = xi0.dim()  # includes planes

    xi_hat = state_group_action(X, xi0)

    # --- Build point-on-plane lookup ---
    # Map: point_id -> (plane_index_in_xi0, plane_landmark)
    point_to_plane_idx: Dict[int, int] = {}
    for j, pl in enumerate(xi0.plane_landmarks):
        for pid in pl.point_ids:
            point_to_plane_idx[pid] = j

    # --- Count rows ---
    n_bearing = 2 * len(y_ids)

    def _is_eligible(fid):
        if not include_constraints:
            return False
        if fid not in point_to_plane_idx:
            return False
        if eligible_constraint_ids is not None and fid not in eligible_constraint_ids:
            return False
        return True

    n_constraints = sum(1 for fid in y_ids if _is_eligible(fid))
    n_rows = n_bearing + n_constraints

    if n_rows == 0:
        return np.array([]), np.zeros((0, dim)), np.zeros((0, 0))

    # --- Allocate ---
    residual = np.zeros(n_rows)
    C_star = np.zeros((n_rows, dim))
    R_noise = np.zeros((n_rows, n_rows))

    # --- Predicted measurement ---
    y_hat_meas = measure_system_state(xi_hat, cam_ptr)

    # --- Fill bearing rows ---
    constraint_row = n_bearing  # write pointer for constraint rows

    for obs_idx, fid in enumerate(y_ids):
        # Find this landmark in xi0
        lm_idx = next(
            i for i, lm in enumerate(xi0.camera_landmarks) if lm.id == fid
        )
        q0 = xi0.camera_landmarks[lm_idx].p

        # Find Q element
        k = X.id.index(fid)
        Q_k = X.Q[k]

        # --- Bearing rows ---
        if use_equivariance:
            y_pixel = y_coords[fid]
            Ci = output_matrix_Ci_star(q0, Q_k, cam_ptr, y_pixel)
        else:
            q_hat = Q_k.inverse() * q0
            y_hat_px = cam_ptr.project_point(q_hat)
            Ci = output_matrix_Ci_star(q0, Q_k, cam_ptr, y_hat_px)

        # Place bearing C* in point columns
        col_start = S + 3 * lm_idx
        C_star[2 * obs_idx:2 * obs_idx + 2, col_start:col_start + 3] = Ci

        # Bearing residual
        if fid in y_hat_meas.cam_coordinates:
            residual[2 * obs_idx:2 * obs_idx + 2] = (
                y_coords[fid] - y_hat_meas.cam_coordinates[fid]
            )

        # Bearing noise
        R_noise[2 * obs_idx, 2 * obs_idx] = sigma_bearing ** 2
        R_noise[2 * obs_idx + 1, 2 * obs_idx + 1] = sigma_bearing ** 2

        # --- Constraint row (if this point is eligible) ---
        if _is_eligible(fid):
            pl_idx = point_to_plane_idx[fid]
            pl0 = xi0.plane_landmarks[pl_idx]

            # Get plane Q element
            pk = X.plane_id.index(pl0.id)
            Q_pl = X.Q_planes[pk]

            # Estimated values
            p_hat = xi_hat.camera_landmarks[lm_idx].p
            q_hat_pl = xi_hat.plane_landmarks[pl_idx].q

            # Constraint C* blocks — chart-aware
            C_p, C_q = constraint_Ci_star_for_chart(
                q0, Q_k, pl0.q, Q_pl,
                point_chart_jacobian=point_chart_jacobian,
            )

            # Place in stacked matrix
            # Point columns
            C_star[constraint_row, col_start:col_start + 3] = C_p.ravel()
            # Plane columns (after all points)
            pl_col_start = S + 3 * M_pts + 3 * pl_idx
            C_star[constraint_row, pl_col_start:pl_col_start + 3] = C_q.ravel()

            # Constraint residual with 3σ clipping for linearization safety
            raw_residual = constraint_residual(p_hat, q_hat_pl)
            max_residual = 3.0 * sigma_constraint
            residual[constraint_row] = np.clip(raw_residual, -max_residual, max_residual)

            # Constraint noise
            R_noise[constraint_row, constraint_row] = sigma_constraint ** 2

            constraint_row += 1

    return residual, C_star, R_noise
