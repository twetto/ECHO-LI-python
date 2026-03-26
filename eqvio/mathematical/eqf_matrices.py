"""
Equivariant filter matrices: coordinate suite framework.

Port of: EqFMatrices.h / EqFMatrices.cpp

The EqFCoordinateSuite bundles all coordinate-chart-dependent functions:
    - A0t state matrix
    - Bt input matrix
    - C*_t output matrix (per-landmark and assembled)
    - Innovation lifting (continuous and discrete)

Chart-specific implementations live in eqvio/coordinate_suite/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List
import numpy as np

from .vio_state import VIOState, VIOSensorState, Landmark, GRAVITY_CONSTANT
from .vio_group import VIOGroup, VIOAlgebra, state_group_action, lift_velocity_discrete
from .imu_velocity import IMUVelocity


def numerical_differential(f: Callable, x0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute Jacobian of f at x0 via central finite differences.

    Used for stateMatrixADiscrete (matching C++ numericalDifferential).
    """
    n = x0.shape[0]
    f0 = f(x0)
    m = f0.shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        J[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2 * eps)
    return J


@dataclass
class EqFCoordinateSuite:
    """Suite of coordinate-chart-dependent EqF functions.

    Reference: EqFMatrices.h struct EqFCoordinateSuite

    All function signatures match the C++ exactly. Chart-specific
    implementations (euclid, invdepth, normal) are injected at construction.
    """

    # Coordinate chart: (xi, xi0) -> eps,  (eps, xi0) -> xi
    state_chart: Callable[[VIOState, VIOState], np.ndarray]
    state_chart_inv: Callable[[np.ndarray, VIOState], VIOState]

    # A0t: (X, xi0, imuVel) -> MatrixXd
    state_matrix_A: Callable[[VIOGroup, VIOState, IMUVelocity], np.ndarray]

    # Bt: (X, xi0) -> MatrixXd
    input_matrix_B: Callable[[VIOGroup, VIOState], np.ndarray]

    # C*_ti: (q0, QHat, cam_ptr, y) -> (2, 3) matrix
    output_matrix_Ci_star: Callable

    # Lift innovation: (Gamma, xi0) -> VIOAlgebra
    lift_innovation: Callable[[np.ndarray, VIOState], VIOAlgebra]

    # Lift innovation discrete: (Gamma, xi0) -> VIOGroup
    lift_innovation_discrete: Callable[[np.ndarray, VIOState], VIOGroup]

    # -----------------------------------------------------------------------
    # Generic methods (not chart-specific)
    # -----------------------------------------------------------------------

    def output_matrix_C(
        self, xi0: VIOState, X: VIOGroup,
        y_ids: List[int], y_coords: dict,
        cam_ptr, use_equivariance: bool = True,
    ) -> np.ndarray:
        """Assemble full output matrix from per-landmark C*_ti blocks.

        Reference: EqFCoordinateSuite::outputMatrixC() in EqFMatrices.cpp

        Args:
            xi0:               origin state
            X:                 observer group element
            y_ids:             list of observed landmark ids
            y_coords:          {id: pixel_coords} for observed landmarks
            cam_ptr:           camera model (project_point, undistort_point, projection_jacobian)
            use_equivariance:  use C*_t (True) or C_t (False)

        Returns:
            (2*N_obs, dim_state) output matrix
        """
        M = len(xi0.camera_landmarks)
        N = len(y_ids)
        dim = VIOSensorState.CDim + Landmark.CDim * M
        C_star = np.zeros((2 * N, dim))

        # Predicted measurement for non-equivariant fallback
        xi_hat = state_group_action(X, xi0)

        for i in range(M):
            id_num = xi0.camera_landmarks[i].id
            q0 = xi0.camera_landmarks[i].p

            if id_num not in y_ids:
                continue

            # Find index in observation list
            j = y_ids.index(id_num)

            # Find corresponding Q
            k = X.id.index(id_num)
            Q_k = X.Q[k]

            if use_equivariance:
                y_pixel = y_coords[id_num]
                Ci = self.output_matrix_Ci_star(q0, Q_k, cam_ptr, y_pixel)
            else:
                Ci = self._output_matrix_Ci(q0, Q_k, cam_ptr)

            C_star[2 * j:2 * j + 2,
                   VIOSensorState.CDim + 3 * i:VIOSensorState.CDim + 3 * i + 3] = Ci

        return C_star

    def _output_matrix_Ci(self, q0, Q_hat, cam_ptr):
        """Non-equivariant output matrix block.

        Reference: EqFCoordinateSuite::outputMatrixCi()
        Falls back to C*_ti evaluated at the predicted measurement.
        """
        q_hat = Q_hat.inverse() * q0
        y_hat = cam_ptr.project_point(q_hat)
        return self.output_matrix_Ci_star(q0, Q_hat, cam_ptr, y_hat)

    def state_matrix_A_discrete(
        self, X: VIOGroup, xi0: VIOState, imu_vel: IMUVelocity, dt: float
    ) -> np.ndarray:
        """Discrete state transition matrix via numerical differentiation.

        Reference: EqFCoordinateSuite::stateMatrixADiscrete() in EqFMatrices.cpp
        """
        def a0_discrete(epsilon):
            xi_e = self.state_chart_inv(epsilon, xi0)
            xi_hat = state_group_action(X, xi0)
            xi = state_group_action(X, xi_e)
            Lambda_tilde = (
                lift_velocity_discrete(xi, imu_vel, dt)
                * lift_velocity_discrete(xi_hat, imu_vel, dt).inverse()
            )
            xi_e1 = state_group_action(X * Lambda_tilde * X.inverse(), xi_e)
            return self.state_chart(xi_e1, xi0)

        return numerical_differential(a0_discrete, np.zeros(xi0.dim()))
