"""
Equivariant filter for VIO: observer state propagation, Riccati propagation, vision update.

Port of: VIO_eqf.h / VIO_eqf.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
import scipy.linalg
from copy import deepcopy

from liepp import SOT3

from .vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark,
    GRAVITY_CONSTANT,
)
from .vio_group import (
    VIOGroup, VIOAlgebra,
    state_group_action, lift_velocity, lift_velocity_discrete, vio_exp,
)
from .imu_velocity import IMUVelocity


def _remove_rows_cols(mat: np.ndarray, start: int, count: int) -> np.ndarray:
    """Remove rows and columns from a matrix (covariance marginalization)."""
    idx = list(range(start, start + count))
    mat = np.delete(mat, idx, axis=0)
    mat = np.delete(mat, idx, axis=1)
    return mat


@dataclass
class VIO_eqf:
    """The equivariant filter for VIO.

    Reference: VIO_eqf.h struct VIO_eqf

    Fields:
        xi0:          Fixed origin configuration
        X:            Observer group state
        Sigma:        Riccati covariance matrix
        current_time: Current filter time
    """
    xi0: VIOState = field(default_factory=VIOState)
    X: VIOGroup = field(default_factory=VIOGroup)
    Sigma: np.ndarray = field(
        default_factory=lambda: np.eye(VIOSensorState.CDim)
    )
    current_time: float = -1.0

    # Coordinate suite function pointers (set by filter wrapper)
    # These are typed as Optional[Callable] so the filter can be configured
    # with different coordinate charts (euclid, invdepth, normal).
    _state_matrix_A: Optional[Callable] = field(default=None, repr=False)
    _input_matrix_B: Optional[Callable] = field(default=None, repr=False)
    _output_matrix_C: Optional[Callable] = field(default=None, repr=False)
    _state_matrix_A_discrete: Optional[Callable] = field(default=None, repr=False)
    _lift_innovation: Optional[Callable] = field(default=None, repr=False)
    _lift_innovation_discrete: Optional[Callable] = field(default=None, repr=False)
    _state_chart: Optional[Callable] = field(default=None, repr=False)

    # -----------------------------------------------------------------------
    # State estimate
    # -----------------------------------------------------------------------

    def state_estimate(self) -> VIOState:
        """Current state estimate: xi_hat = X * xi0.

        Reference: VIO_eqf::stateEstimate()
        """
        return state_group_action(self.X, self.xi0)

    # -----------------------------------------------------------------------
    # Observer state integration
    # -----------------------------------------------------------------------

    def integrate_observer_state(
        self, imu: IMUVelocity, dt: float, discrete_lift: bool = True
    ):
        """Propagate the observer group state by one IMU step.

        Reference: VIO_eqf::integrateObserverState()
        """
        if discrete_lift:
            lifted = lift_velocity_discrete(self.state_estimate(), imu, dt)
        else:
            lifted_alg = lift_velocity(self.state_estimate(), imu)
            lifted = vio_exp(dt * lifted_alg)

        assert not lifted.has_nan(), "Lifted velocity contains NaN"
        self.X = self.X * lifted
        assert not self.X.has_nan(), "Observer state X contains NaN after propagation"

    def _enforce_spd(self):
        """Enforce symmetric positive-definite on Sigma.

        Symmetrize and clamp minimum eigenvalue to prevent
        numerical drift from accumulating over many steps.
        """
        # Symmetrize
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)

        # Clamp minimum eigenvalue (prevents negative eigenvalues from roundoff)
        # min_eig = 1e-12
        # eigvals = np.linalg.eigvalsh(self.Sigma)
        # if eigvals[0] < min_eig:
        #     self.Sigma += (min_eig - eigvals[0]) * np.eye(self.Sigma.shape[0])
        np.fill_diagonal(self.Sigma, self.Sigma.diagonal() + 1e-12)

    # -----------------------------------------------------------------------
    # Riccati propagation
    # -----------------------------------------------------------------------

    def integrate_riccati_fast(
        self, imu: IMUVelocity, dt: float,
        input_gain: np.ndarray, state_gain: np.ndarray,
    ):
        """Fast (Euler) Riccati propagation.

        Reference: VIO_eqf::integrateRiccatiStateFast()
        """
        A0t = self._state_matrix_A(self.X, self.xi0, imu)
        Bt = self._input_matrix_B(self.X, self.xi0)
        n = self.xi0.dim()
        F = np.eye(n) + dt * A0t
        self.Sigma = (F @ self.Sigma @ F.T
                      + dt * (Bt @ input_gain @ Bt.T + state_gain))
        self._enforce_spd()

    def integrate_riccati_discrete(
        self, imu: IMUVelocity, dt: float,
        input_gain: np.ndarray, state_gain: np.ndarray,
    ):
        """Discrete Riccati propagation.

        Reference: VIO_eqf::integrateRiccatiStateDiscrete()
        """
        Bt = self._input_matrix_B(self.X, self.xi0)
        A0d = self._state_matrix_A_discrete(self.X, self.xi0, imu, dt)
        self.Sigma = (A0d @ self.Sigma @ A0d.T
                      + dt * (Bt @ input_gain @ Bt.T + state_gain))
        self._enforce_spd()

    # -----------------------------------------------------------------------
    # Vision update
    # -----------------------------------------------------------------------

    def perform_vision_update(
        self, measurement, output_gain: np.ndarray,
        measure_fn: Callable,
        use_equivariant_output: bool = True,
        discrete_correction: bool = False,
    ):
        """EKF-style vision update with equivariant output.

        Reference: VIO_eqf::performVisionUpdate()
        Uses Joseph form: Σ = (I - KC) Σ (I - KC)^T + K R K^T
        """
        if not measurement:
            return

        # Predicted measurement from current estimate
        estimated = measure_fn(self.state_estimate(), measurement.camera_ptr)
        y_tilde = measurement - estimated  # innovation vector

        # Output matrix
        Ct = self._output_matrix_C(
            self.xi0, self.X, measurement, use_equivariant_output
        )

        # Kalman gain
        S = Ct @ self.Sigma @ Ct.T + output_gain
        K = scipy.linalg.solve(S, Ct @ self.Sigma, assume_a='pos').T

        # Correction in coordinates
        Gamma = K @ y_tilde
        assert not np.any(np.isnan(Gamma)), "Innovation Gamma contains NaN"

        # Lift to group correction
        if discrete_correction:
            Delta = self._lift_innovation_discrete(Gamma, self.xi0)
        else:
            Delta = vio_exp(self._lift_innovation(Gamma, self.xi0))

        assert not Delta.has_nan(), "Correction Delta contains NaN"

        # Apply correction: X <- Delta * X
        self.X = Delta * self.X

        # Joseph form: Σ = (I - KC) Σ (I - KC)^T + K R K^T
        n = self.Sigma.shape[0]
        IKC = np.eye(n) - K @ Ct
        self.Sigma = IKC @ self.Sigma @ IKC.T + K @ output_gain @ K.T
        self._enforce_spd()

    def perform_stacked_update(
        self, residual: np.ndarray, C_star: np.ndarray,
        R_noise: np.ndarray, discrete_correction: bool = False,
    ):
        """Kalman update with pre-assembled stacked matrices.

        Used for combined bearing + constraint updates when planes are in state.
        Same mechanics as perform_vision_update but accepts raw matrices.

        Args:
            residual:   (n_rows,) innovation vector
            C_star:     (n_rows, dim) output matrix
            R_noise:    (n_rows, n_rows) measurement noise covariance
            discrete_correction: use discrete lift (True) or continuous (False)
        """
        if residual.shape[0] == 0:
            return

        Ct = C_star
        y_tilde = residual

        # Kalman gain
        S = Ct @ self.Sigma @ Ct.T + R_noise
        K = scipy.linalg.solve(S, Ct @ self.Sigma, assume_a='pos').T

        # Correction in coordinates
        Gamma = K @ y_tilde
        assert not np.any(np.isnan(Gamma)), "Innovation Gamma contains NaN"

        # Lift to group correction
        if discrete_correction:
            Delta = self._lift_innovation_discrete(Gamma, self.xi0)
        else:
            Delta = vio_exp(self._lift_innovation(Gamma, self.xi0))

        assert not Delta.has_nan(), "Correction Delta contains NaN"

        # Apply correction: X <- Delta * X
        self.X = Delta * self.X

        # Joseph form: Σ = (I - KC) Σ (I - KC)^T + K R K^T
        n = self.Sigma.shape[0]
        IKC = np.eye(n) - K @ Ct
        self.Sigma = IKC @ self.Sigma @ IKC.T + K @ R_noise @ K.T
        self._enforce_spd()

    # -----------------------------------------------------------------------
    # Landmark management
    # -----------------------------------------------------------------------

    def add_new_landmarks(
        self, new_landmarks: List[Landmark], new_cov: np.ndarray
    ):
        """Add new point landmarks to the state and augment covariance.

        Reference: VIO_eqf::addNewLandmarks()
        """
        self.xi0.camera_landmarks.extend(new_landmarks)

        new_ids = [lm.id for lm in new_landmarks]
        self.X.id.extend(new_ids)
        self.X.Q.extend([SOT3.Identity() for _ in new_landmarks])

        n_old = self.Sigma.shape[0]
        n_new = 3 * len(new_landmarks)
        new_Sigma = np.zeros((n_old + n_new, n_old + n_new))
        new_Sigma[:n_old, :n_old] = self.Sigma
        new_Sigma[n_old:, n_old:] = new_cov
        self.Sigma = new_Sigma

    def add_new_plane_landmarks(
        self, new_planes: List[PlaneLandmark], new_cov: np.ndarray
    ):
        """Add new plane landmarks to the state and augment covariance.

        NEW — not in C++ codebase. Mirrors addNewLandmarks for planes.
        Plane SOT(3) dimension is 4 in the algebra, but 3 in the coordinate
        chart (same as points in the Euclidean chart).
        """
        self.xi0.plane_landmarks.extend(new_planes)

        new_ids = [pl.id for pl in new_planes]
        self.X.plane_id.extend(new_ids)
        self.X.Q_planes.extend([SOT3.Identity() for _ in new_planes])

        # Augment covariance (4 DOF per plane in sot(3), but typically
        # 3 DOF in the coordinate chart — caller decides)
        n_old = self.Sigma.shape[0]
        n_new = new_cov.shape[0]
        new_Sigma = np.zeros((n_old + n_new, n_old + n_new))
        new_Sigma[:n_old, :n_old] = self.Sigma
        new_Sigma[n_old:, n_old:] = new_cov
        self.Sigma = new_Sigma

    def remove_landmark_by_index(self, idx: int):
        """Remove a point landmark by its index.

        Reference: VIO_eqf::removeLandmarkByIndex()
        """
        del self.xi0.camera_landmarks[idx]
        del self.X.id[idx]
        del self.X.Q[idx]
        start = VIOSensorState.CDim + 3 * idx
        self.Sigma = _remove_rows_cols(self.Sigma, start, 3)

    def remove_landmark_by_id(self, lm_id: int):
        """Remove a point landmark by its id.

        Reference: VIO_eqf::removeLandmarkById()
        """
        idx = next(
            i for i, lm in enumerate(self.xi0.camera_landmarks) if lm.id == lm_id
        )
        self.remove_landmark_by_index(idx)

    def remove_invalid_landmarks(self):
        """Remove landmarks with degenerate SOT(3) scale.

        Reference: VIO_eqf::removeInvalidLandmarks()
        """
        invalid_ids = [
            self.X.id[i]
            for i in range(len(self.X.id))
            if self.X.Q[i].a <= 1e-8 or self.X.Q[i].a > 1e8
        ]
        for lm_id in invalid_ids:
            self.remove_landmark_by_id(lm_id)

    def remove_plane_by_index(self, idx: int):
        """Remove a plane landmark by its index in the plane list.

        NEW — not in C++ codebase.
        """
        del self.xi0.plane_landmarks[idx]
        del self.X.plane_id[idx]
        del self.X.Q_planes[idx]
        # Planes are stored after all points in the covariance
        n_pts = len(self.xi0.camera_landmarks)
        start = VIOSensorState.CDim + 3 * n_pts + 3 * idx
        self.Sigma = _remove_rows_cols(self.Sigma, start, 3)

    def remove_plane_by_id(self, plane_id: int):
        """Remove a plane landmark by its id.

        NEW — not in C++ codebase.
        """
        idx = next(
            i for i, pl in enumerate(self.xi0.plane_landmarks) if pl.id == plane_id
        )
        self.remove_plane_by_index(idx)

    # -----------------------------------------------------------------------
    # Covariance queries
    # -----------------------------------------------------------------------

    def get_velocity_cov(self) -> np.ndarray:
        """Get 3x3 marginal covariance for the velocity state."""
        return self.Sigma[12:15, 12:15].copy()

    def get_landmark_cov_by_id(self, lm_id: int) -> np.ndarray:
        """Get 3x3 marginal covariance for a point landmark.

        Reference: VIO_eqf::getLandmarkCovById()
        """
        idx = next(
            i for i, lm in enumerate(self.xi0.camera_landmarks) if lm.id == lm_id
        )
        start = VIOSensorState.CDim + 3 * idx
        return self.Sigma[start:start + 3, start:start + 3].copy()

    # -----------------------------------------------------------------------
    # State prediction
    # -----------------------------------------------------------------------

    def predict_state(
        self, stamp: float, imu_velocities: List[IMUVelocity],
        integrate_fn: Callable,
    ) -> VIOState:
        """Predict the state at a future timestamp.

        Reference: VIO_eqf::predictState()

        Args:
            stamp:          target time
            imu_velocities: IMU measurements covering [current_time, stamp]
            integrate_fn:   function(state, imu, dt) -> state
        """
        state = self.state_estimate()
        assert stamp >= self.current_time

        for i, imu in enumerate(imu_velocities):
            t0 = max(imu.stamp, self.current_time)
            if i + 1 < len(imu_velocities):
                t1 = min(imu_velocities[i + 1].stamp, stamp)
            else:
                t1 = stamp
            dt = max(t1 - t0, 0.0)
            state = integrate_fn(state, imu, dt)

        return state

    # -----------------------------------------------------------------------
    # NEES (for evaluation)
    # -----------------------------------------------------------------------

    def compute_nees(self, true_state: VIOState) -> float:
        """Compute Normalized Estimation Error Squared.

        Reference: VIO_eqf::computeNEES()
        """
        # Truncate true state to match filter's landmark set
        truncated = VIOState()
        truncated.sensor = true_state.sensor

        lm_by_id = {lm.id: lm for lm in true_state.camera_landmarks}
        for lm_id in self.X.id:
            assert lm_id in lm_by_id, f"True state missing landmark {lm_id}"
            truncated.camera_landmarks.append(lm_by_id[lm_id])

        # Error state
        state_error = state_group_action(self.X.inverse(), truncated)
        eps = self._state_chart(state_error, self.xi0)

        # NEES
        info = np.linalg.inv(self.Sigma)
        nees = eps @ info @ eps
        return nees / truncated.dim()
