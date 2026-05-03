"""
Sparse Vogiatzis Gaussian-Beta mixture filter for tracked features.

Runs a per-feature 1D depth filter on the LK-tracked feature pool (~300
features), parallel to the core EqF (~40). Converged features feed:
    - the core EqF as a depth warm-start for newly added landmarks
    - the PlaneDetector as a larger feature pool via feat_uvs /
      feat_positions_global

Canonical chart is configurable via DepthParametrization:
    - EUCLIDEAN  : z = scene depth (metric)
                  tau_sq = z^4 * sigma_norm^2 / drive^2
                  uniform prior: z in [0, uniform_z_max]
    - INVDEPTH   : rho = 1/z (metric)
                  tau_sq = sigma_norm^2 / drive^2  (no z^4 — InvDepth absorbs
                  the nonlinearity, giving more uniform convergence across
                  depth range. Matching FlowDep's default.)
                  uniform prior: rho in [0, uniform_rho_max]
    - POLAR      : d = -log(z) = log(ρ) (dimensionless, consistent with
                  InvDepth and Normal chart)
                  tau_sq = z^2 * sigma_norm^2 / drive^2
                  uniform prior: d in [uniform_d_min, uniform_d_max]

Prediction uses a 3D warp in Euclidean space (shared by all charts); the
canonical value and variance are then derived afterward via the chart's
Jacobian.

Per-feature Vogiatzis update follows the same moment-matched Gaussian-Beta
math as FlowDep._vogiatzis_update, specialised to scalar depth:
    mu          canonical depth (z, rho, or -log_z)
    sigma_sq    var(canonical)
    a, b        Beta shape of inlier prob pi
"""

from __future__ import annotations

import cv2
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np

from .mathematical.vision_measurement import VisionMeasurement
from .mathematical.vio_state import Landmark
from .coordinate_suite.normal import (
    conv_euc2normal,
    conv_normal2euc,
    point_chart_normal_inv,
    sphere_chart_normal_inv_diff0,
)


# ============================================================================
# Parametrization
# ============================================================================

class DepthParametrization(Enum):
    EUCLIDEAN = auto()
    INVDEPTH = auto()
    POLAR = auto()
    POLAR3D = auto()  # 3D Local IEKF on Normal chart


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix [v]×."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


# ============================================================================
# Settings
# ============================================================================

@dataclass
class SparseVogSettings:
    """Configuration for the sparse Vogiatzis filter."""

    # Canonical chart used for the filter's internal 1D state.
    parametrization: DepthParametrization = DepthParametrization.INVDEPTH

    # --- Pool management ---
    max_pool_size: int = 300
    min_track_length: int = 5   # frames before query() will return a value

    # --- Convergence gate ---
    conv_inlier_ratio: float = 0.7     # a/(a+b) threshold for query()
    conv_variance_threshold: float = 0.5  # var(z) threshold for query()

    # --- Initial state (Euclidean depth) ---
    init_depth_var: float = 1.0
    sigma_pixel: float = 0.5
    dist_coeffs: Optional[np.ndarray] = None  # if provided, features are undistorted

    # --- Vogiatzis Beta prior ---
    uniform_z_max: float = 20.0
    # Uniform prior upper bound for INVDEPTH (rho in [0, uniform_rho_max])
    uniform_rho_max: float = 10.0
    # Uniform prior bounds for POLAR (d in [uniform_d_min, uniform_d_max])
    uniform_d_min: float = -5.0
    uniform_d_max: float = 5.0
    a_init: float = 10.0
    b_init: float = 2.0
    ab_min: float = 1.0
    ab_max: float = 100.0
    min_inlier_ratio: float = 0.3  # also used for reset gating
    mahalanobis_reset_chi2: float = 9.0

    # --- Process model ---
    process_depth_var: float = 0.01  # per-frame fallback when P_vv unavailable

    # --- Triangulation gating ---
    min_parallax: float = 1e-4   # ideal parallax mag (normalised image coords)
    min_cos_sim: float = 0.95    # reject pixel motion misaligned with epipolar
    min_depth: float = 0.1
    max_depth: float = 100.0
    reanchor_flow_px: float = 3.0  # derotated flow (px) before update+re-anchor


# ============================================================================
# Per-feature state
# ============================================================================

@dataclass
class FeatureState:
    """One tracked feature's scalar Vogiatzis belief (canonical: z, rho, or -log_z)."""
    feat_id: int
    canonical: float = float('nan')  # z (EUCLIDEAN), rho=1/z (INVDEPTH), or -log(z) (POLAR)
    canonical_var: float = 1.0
    a: float = 10.0
    b: float = 2.0
    track_length: int = 0
    ref_T_WC: Optional[np.ndarray] = None   # reference frame pose (for triangulation)
    ref_uv: Optional[np.ndarray] = None     # reference frame pixel position
    ref_stamp: float = -1.0

    @property
    def depth(self) -> float:
        return self.canonical

    @depth.setter
    def depth(self, v: float) -> None:
        self.canonical = v

    @property
    def depth_var(self) -> float:
        return self.canonical_var

    @depth_var.setter
    def depth_var(self, v: float) -> None:
        self.canonical_var = v

    def inlier_ratio(self) -> float:
        ab = self.a + self.b
        if ab <= 0.0:
            return 0.0
        return self.a / ab


# ============================================================================
# Filter
# ============================================================================

class SparseVogiatzisFilter:
    """Per-feature 1D Vogiatzis mixture filter with configurable parametrization.

    Lifecycle:
        update()    per vision frame; propagates + updates each feature
        query(id)   returns (depth, depth_var) if converged, else (-1, inf)
        feat_uvs    property of current pixel coords (for PlaneDetector)
        feat_positions_global(state)  world-frame 3D points for converged feats
    """

    def __init__(self, K: np.ndarray, settings: SparseVogSettings):
        self.K = K.astype(np.float64)
        self.settings = settings
        self._param = settings.parametrization
        fx = float(self.K[0, 0])
        self._sigma_norm_sq = (settings.sigma_pixel / fx) ** 2

        self._dist_coeffs = (
            settings.dist_coeffs.astype(np.float64)
            if settings.dist_coeffs is not None
            else None
        )

        self._features: Dict[int, FeatureState] = {}
        self._pending: Dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self._prev_uvs: Dict[int, np.ndarray] = {}
        self._prev_T_WC: Optional[np.ndarray] = None
        self._prev_stamp: float = -1.0
        self._prev_uvs_undistorted: bool = False

    def _undistort_uvs(
        self,
        uvs: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """Undistort pixel coords in-place using cv2.undistortPoints."""
        if self._dist_coeffs is None:
            return uvs
        result = {}
        for fid, uv in uvs.items():
            pt = np.array([[uv]], dtype=np.float64)
            uv_undist = cv2.undistortPoints(
                pt, self.K, self._dist_coeffs, P=self.K,
            )
            result[fid] = uv_undist[0, 0]
        return result

    # ------------------------------------------------------------------
    # Main update entry point
    # ------------------------------------------------------------------

    def update(
        self,
        measurement: VisionMeasurement,
        T_WC: np.ndarray,
        P_vv: Optional[np.ndarray] = None,
        flowdep=None,
        v_C: Optional[np.ndarray] = None,
    ) -> None:
        """Propagate + update each tracked feature's depth belief.

        Parameters
        ----------
        P_vv : (3,3) or (6,6) array, optional
            Velocity covariance.  3×3 = translational only.
            6×6 = full spatial velocity [ω; v] (improvement 2).
        flowdep : optional FlowDepFilter
            When provided, new features are seeded with FlowDep's dense
            depth estimate instead of the first triangulation.
        v_C : (3,) array, optional
            EqF-filtered camera velocity in camera frame.  When provided,
            the cos_sim epipolar gate uses this instead of frame-to-frame
            t for a more stable direction (improvement 3).
        """
        stamp = measurement.stamp
        curr_uvs = {
            fid: uv.astype(np.float64).copy()
            for fid, uv in measurement.cam_coordinates.items()
        }
        curr_uvs = self._undistort_uvs(curr_uvs)

        # First frame — just record for next-frame baseline
        if self._prev_T_WC is None or self._prev_stamp < 0:
            self._prev_T_WC = T_WC.copy()
            self._prev_stamp = stamp
            self._prev_uvs = curr_uvs
            self._prev_uvs_undistorted = True
            return

        if not self._prev_uvs_undistorted:
            self._prev_uvs = self._undistort_uvs(self._prev_uvs)
        self._prev_uvs_undistorted = True

        dt = max(stamp - self._prev_stamp, 0.0)

        # Relative pose: current camera <- previous camera
        T_CW_curr = np.linalg.inv(T_WC)
        T_curr_prev = T_CW_curr @ self._prev_T_WC
        R = T_curr_prev[:3, :3]
        t = T_curr_prev[:3, 3]

        # Baseline uncertainty for τ² inflation (improvement 1).
        P_vv_trans = None
        if P_vv is not None:
            P_vv_trans = P_vv[3:6, 3:6] if P_vv.shape == (6, 6) else P_vv
        t_norm_sq = float(t[0] ** 2 + t[1] ** 2 + t[2] ** 2)
        baseline_tau_sq = 0.0
        if P_vv_trans is not None and t_norm_sq > 1e-16 and dt > 0:
            t_hat = t / math.sqrt(t_norm_sq)
            var_t_mag = dt * dt * float(t_hat @ P_vv_trans @ t_hat)
            baseline_tau_sq = var_t_mag / t_norm_sq

        for fid, uv_curr in curr_uvs.items():
            uv_prev = self._prev_uvs.get(fid)
            if uv_prev is None:
                continue

            feat = self._features.get(fid)

            # --- Pending features: coast until flow gate is met ---
            if feat is None:
                pending = self._pending.get(fid)
                if pending is None:
                    if len(self._features) + len(self._pending) < self.settings.max_pool_size:
                        self._pending[fid] = (
                            self._prev_T_WC.copy(), uv_prev.copy(),
                            self._prev_stamp,
                        )
                    continue

                ref_T, ref_uv, ref_stamp = pending
                T_curr_ref = T_CW_curr @ ref_T
                R_ref = T_curr_ref[:3, :3]
                t_ref = T_curr_ref[:3, 3]
                z_obs, drive = self._triangulate(
                    ref_uv, uv_curr, R_ref, t_ref, v_C,
                )
                if z_obs <= 0.0:
                    continue
                flow_px = drive * float(self.K[0, 0])
                if flow_px < self.settings.reanchor_flow_px:
                    continue

                t_ref_nsq = float(t_ref[0] ** 2 + t_ref[1] ** 2 + t_ref[2] ** 2)
                feat_btau = 0.0
                if P_vv_trans is not None and t_ref_nsq > 1e-16 and dt > 0:
                    t_ref_hat = t_ref / math.sqrt(t_ref_nsq)
                    dt_total = max(stamp - ref_stamp, dt)
                    var_t = dt_total * dt * float(
                        t_ref_hat @ P_vv_trans @ t_ref_hat
                    )
                    feat_btau = var_t / t_ref_nsq

                init_obs, init_tau = self._obs_and_tau(z_obs, drive, None, feat_btau)
                init_var = init_tau
                if flowdep is not None:
                    fd_depth, fd_var = flowdep.query_depth(
                        float(uv_curr[0]), float(uv_curr[1]),
                    )
                    if fd_depth > 0:
                        init_obs, init_var = self._canonical_from_depth(fd_depth, fd_var)
                feat = FeatureState(
                    feat_id=fid,
                    canonical=init_obs,
                    canonical_var=init_var,
                    a=self.settings.a_init,
                    b=self.settings.b_init,
                    ref_T_WC=T_WC.copy(),
                    ref_uv=uv_curr.copy(),
                    ref_stamp=stamp,
                )
                self._features[fid] = feat
                del self._pending[fid]
                feat.track_length += 1
                continue

            # --- Existing features ---
            # Triangulate from reference frame (growing baseline).
            T_curr_ref = T_CW_curr @ feat.ref_T_WC
            R_ref = T_curr_ref[:3, :3]
            t_ref = T_curr_ref[:3, 3]
            z_obs, drive = self._triangulate(
                feat.ref_uv, uv_curr, R_ref, t_ref, v_C,
            )
            t_ref_nsq = float(t_ref[0] ** 2 + t_ref[1] ** 2 + t_ref[2] ** 2)
            feat_btau = 0.0
            if P_vv_trans is not None and t_ref_nsq > 1e-16 and dt > 0:
                t_ref_hat = t_ref / math.sqrt(t_ref_nsq)
                dt_total = max(stamp - feat.ref_stamp, dt)
                var_t = dt_total * dt * float(
                    t_ref_hat @ P_vv_trans @ t_ref_hat
                )
                feat_btau = var_t / t_ref_nsq

            # Predict using frame-to-frame pose (independent of observation).
            if math.isfinite(feat.canonical):
                self._predict_feature(feat, uv_prev, R, t, P_vv, dt)

            if z_obs <= 0.0:
                continue

            flow_px = drive * float(self.K[0, 0])

            if not math.isfinite(feat.canonical):
                # Prediction invalidated — reset from obs + fresh reference.
                init_obs, init_tau = self._obs_and_tau(z_obs, drive, None, feat_btau)
                feat.canonical = init_obs
                feat.canonical_var = init_tau
                feat.a = self.settings.a_init
                feat.b = self.settings.b_init
                feat.ref_T_WC = T_WC.copy()
                feat.ref_uv = uv_curr.copy()
                feat.ref_stamp = stamp
                obs, tau_sq = self._obs_and_tau(z_obs, drive, feat, feat_btau)
                self._vogiatzis_update(feat, obs, tau_sq)
            elif flow_px >= self.settings.reanchor_flow_px:
                obs, tau_sq = self._obs_and_tau(z_obs, drive, feat, feat_btau)
                self._vogiatzis_update(feat, obs, tau_sq)
                feat.ref_T_WC = T_WC.copy()
                feat.ref_uv = uv_curr.copy()
                feat.ref_stamp = stamp
            else:
                continue

            feat.track_length += 1

        # Drop features and pending entries no longer tracked by LK.
        lost = set(self._features.keys()) - set(curr_uvs.keys())
        for fid in lost:
            del self._features[fid]
        lost_pending = set(self._pending.keys()) - set(curr_uvs.keys())
        for fid in lost_pending:
            del self._pending[fid]

        self._prev_T_WC = T_WC.copy()
        self._prev_stamp = stamp
        self._prev_uvs = curr_uvs

    # ------------------------------------------------------------------
    # Observation: canonical value and tau_sq for each parametrization
    # ------------------------------------------------------------------

    def _obs_and_tau(
        self,
        z_obs: float,
        drive: float,
        feat: FeatureState,
        baseline_tau_sq: float = 0.0,
    ) -> tuple[float, float]:
        """Return (canonical_observation, tau_sq) for the active parametrization.

        baseline_tau_sq = var(||t||) / ||t||² inflates tau when the
        triangulation baseline is uncertain (improvement 1).
        """
        pixel_tau_sq = self._sigma_norm_sq / (drive * drive)
        if self._param is DepthParametrization.INVDEPTH:
            rho_obs = 1.0 / z_obs
            if rho_obs < 0.0 or not math.isfinite(rho_obs):
                return -1.0, 0.0
            return rho_obs, pixel_tau_sq + rho_obs * rho_obs * baseline_tau_sq
        elif self._param is DepthParametrization.POLAR:
            d_obs = -math.log(z_obs)
            return d_obs, z_obs * z_obs * pixel_tau_sq + baseline_tau_sq
        else:
            return z_obs, (z_obs ** 4) * pixel_tau_sq + z_obs * z_obs * baseline_tau_sq

    def _canonical_from_depth(
        self,
        z: float,
        euclidean_var: float,
    ) -> tuple[float, float]:
        """Convert Euclidean (z, var_z) to (canonical, var_canonical).

        Jacobian: ∂canonical/∂z
        - INVDEPTH: ρ = 1/z → ∂ρ/∂z = -1/z² → var(ρ) = var(z) / z⁴
        - POLAR:    d = -log(z) → ∂d/∂z = -1/z → var(d) = var(z) / z²
        - EUCLIDEAN: var(z) = var(z)
        """
        if self._param is DepthParametrization.INVDEPTH:
            rho = 1.0 / z
            rho_var = euclidean_var / (z ** 4)
            return rho, rho_var
        elif self._param is DepthParametrization.POLAR:
            d = -math.log(z)
            d_var = euclidean_var / (z * z)
            return d, d_var
        else:
            return z, euclidean_var

    def _canonical_to_depth(self, canonical: float) -> float:
        """Convert canonical → Euclidean depth z."""
        if self._param is DepthParametrization.INVDEPTH:
            return 1.0 / canonical
        elif self._param is DepthParametrization.POLAR:
            return math.exp(-canonical)
        else:
            return canonical

    def _canonical_var_to_euclidean(
        self,
        canonical: float,
        canonical_var: float,
        z: float,
    ) -> float:
        """Convert var(canonical) → var(z) for query output.

        Jacobian: ∂z/∂canonical
        - INVDEPTH: z = 1/ρ → ∂z/∂ρ = -z² → var(z) = z⁴ · var(ρ)
        - POLAR:    z = exp(-d) → ∂z/∂d = -z  → var(z) = z² · var(d)
        - EUCLIDEAN: var(z) = var(z)
        """
        if self._param is DepthParametrization.INVDEPTH:
            return (z ** 4) * canonical_var
        elif self._param is DepthParametrization.POLAR:
            return (z * z) * canonical_var
        else:
            return canonical_var

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------

    def _triangulate(
        self,
        uv_prev: np.ndarray,
        uv_curr: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        v_C: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        """Derotated epipolar triangulation for a single pixel pair.

        Returns (z_curr, ideal_parallax_mag). z_curr <= 0 on failure.

        When v_C (EqF-filtered camera velocity in camera frame) is
        provided, the cos_sim direction gate uses -v_C instead of t,
        giving a more stable epipolar direction at low speed or high
        rotation rate (improvement 3).
        """
        s = self.settings
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        x_curr = (uv_curr[0] - cx) / fx
        y_curr = (uv_curr[1] - cy) / fy

        x_prev = (uv_prev[0] - cx) / fx
        y_prev = (uv_prev[1] - cy) / fy
        bearing_prev = np.array([x_prev, y_prev, 1.0])

        aligned = R @ bearing_prev
        if aligned[2] <= 1e-6:
            return -1.0, 0.0
        x_prev_rect = aligned[0] / aligned[2]
        y_prev_rect = aligned[1] / aligned[2]

        num_x = x_prev_rect * t[2] - t[0]
        num_y = y_prev_rect * t[2] - t[1]
        den_x = x_prev_rect - x_curr
        den_y = y_prev_rect - y_curr

        geom_mag_sq = num_x * num_x + num_y * num_y
        ideal_mag = math.sqrt(geom_mag_sq)
        obs_mag = math.sqrt(den_x * den_x + den_y * den_y)

        if ideal_mag < s.min_parallax or obs_mag < 1e-6:
            return -1.0, 0.0

        # Epipolar direction for cos_sim gate: use v_C if available.
        gate_x, gate_y = num_x, num_y
        gate_mag = ideal_mag
        if v_C is not None:
            v_norm = math.sqrt(v_C[0] ** 2 + v_C[1] ** 2 + v_C[2] ** 2)
            if v_norm > 1e-6:
                vd = -v_C
                gx = x_prev_rect * vd[2] - vd[0]
                gy = y_prev_rect * vd[2] - vd[1]
                gm = math.sqrt(gx * gx + gy * gy)
                if gm > 1e-6:
                    gate_x, gate_y, gate_mag = gx, gy, gm

        dot_gate = gate_x * den_x + gate_y * den_y
        if dot_gate <= 1e-6:
            return -1.0, 0.0
        cos_sim = dot_gate / (gate_mag * obs_mag)
        if cos_sim < s.min_cos_sim:
            return -1.0, 0.0

        # Depth from original t-based geometry (not the gating direction).
        dot = num_x * den_x + num_y * den_y
        if dot <= 1e-6:
            return -1.0, 0.0
        z_curr = geom_mag_sq / dot
        if z_curr < s.min_depth or z_curr > s.max_depth:
            return -1.0, 0.0

        return z_curr, ideal_mag

    # ------------------------------------------------------------------
    # Prediction (3D warp)
    # ------------------------------------------------------------------

    def _predict_feature(
        self,
        feat: FeatureState,
        uv_prev: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        P_vv: Optional[np.ndarray],
        dt: float,
    ) -> None:
        """Propagate (canonical, canonical_var) from prev to curr camera.

        3D warp always happens in Euclidean space (shared by all charts).
        The canonical value and variance are then derived afterward.
        """
        s = self.settings
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (uv_prev[0] - cx) / fx
        y = (uv_prev[1] - cy) / fy

        # Recover Euclidean depth and 3D point from canonical.
        if self._param is DepthParametrization.INVDEPTH:
            rho_old = feat.canonical
            z_old = 1.0 / rho_old
            p_prev = np.array([x / rho_old, y / rho_old, 1.0 / rho_old])
        elif self._param is DepthParametrization.POLAR:
            z_old = math.exp(-feat.canonical)
            p_prev = np.array([x * z_old, y * z_old, z_old])
        else:
            z_old = feat.canonical
            p_prev = np.array([x * z_old, y * z_old, z_old])

        p_curr = R @ p_prev + t
        z_new = p_curr[2]
        if z_new < s.min_depth:
            feat.canonical = float('nan')
            return

        # g = R[2,:] @ bearing = R[2,0]*x + R[2,1]*y + R[2,2]
        g = R[2, 0] * x + R[2, 1] * y + R[2, 2]

        # Propagation Jacobian J = d(State_new) / d(State_old).
        # EUCLIDEAN:   J = g
        # INVDEPTH:    J = g * (Z_old / Z_new)^2   (chain rule: d(1/Z)/dZ @ dZ/dZ_old @ dZ_old/d(1/Z))
        # POLAR:       J = g * (Z_old / Z_new)     (chain rule: d(log Z)/dZ @ dZ/dZ_old @ dZ_old/d(log Z))
        if self._param is DepthParametrization.INVDEPTH:
            J = g * ((z_old / z_new) ** 2)
        elif self._param is DepthParametrization.POLAR:
            J = g * (z_old / z_new)
        else:
            J = g
        var_new = (J * J) * feat.canonical_var

        # Process noise: physical depth variance Q_Z mapped into canonical chart.
        # When P_vv is 6×6 [ω; v], use the full depth-rate Jacobian
        #   J_ż = [Z·y, -Z·x, 0, 0, 0, -1]  (improvement 2).
        # When P_vv is 3×3, fall back to bearing-projected velocity variance.
        if P_vv is not None and dt > 0.0:
            if P_vv.shape == (6, 6):
                x_n = p_curr[0] / z_new
                y_n = p_curr[1] / z_new
                J_zdot = np.array([
                    z_new * y_n, -z_new * x_n, 0.0,
                    0.0, 0.0, -1.0,
                ])
                Q_Z = dt * dt * float(J_zdot @ P_vv @ J_zdot)
            else:
                bx, by = x, y
                b_norm_sq = bx * bx + by * by + 1.0
                num = (
                    bx * bx * P_vv[0, 0]
                    + by * by * P_vv[1, 1]
                    + P_vv[2, 2]
                    + 2.0 * bx * by * P_vv[0, 1]
                    + 2.0 * bx * P_vv[0, 2]
                    + 2.0 * by * P_vv[1, 2]
                )
                Q_Z = dt * dt * num / b_norm_sq
            if self._param is DepthParametrization.INVDEPTH:
                rho_new = 1.0 / z_new
                var_new += Q_Z * (rho_new ** 4)
            elif self._param is DepthParametrization.POLAR:
                var_new += Q_Z / (z_new * z_new)
            else:
                var_new += Q_Z
        else:
            var_new += s.process_depth_var * max(dt, 1e-3)

        # Update canonical state.
        if self._param is DepthParametrization.INVDEPTH:
            feat.canonical = 1.0 / z_new
        elif self._param is DepthParametrization.POLAR:
            feat.canonical = -math.log(z_new)
        else:
            feat.canonical = z_new
        feat.canonical_var = float(max(var_new, 1e-8))

    # ------------------------------------------------------------------
    # Vogiatzis moment-matched update (scalar)
    # ------------------------------------------------------------------

    def _vogiatzis_update(
        self,
        feat: FeatureState,
        obs: float,
        tau_sq: float,
    ) -> None:
        """1D Gaussian-Beta mixture update, matching FlowDep._vogiatzis_update."""
        s = self.settings
        mu = feat.canonical
        sigma_sq = feat.canonical_var
        a = feat.a
        b = feat.b

        s_total = sigma_sq + tau_sq
        diff = obs - mu
        m_dist_sq = (diff * diff) / s_total

        # Stuck-outlier recovery.
        ab_pre = a + b
        if (
            s.mahalanobis_reset_chi2 > 0.0
            and ab_pre > 0.0
            and (a / ab_pre) < s.min_inlier_ratio
            and m_dist_sq > s.mahalanobis_reset_chi2
        ):
            feat.canonical = obs
            feat.canonical_var = tau_sq
            feat.a = s.a_init
            feat.b = s.b_init
            return

        # Inlier Kalman branch.
        m = (mu * tau_sq + obs * sigma_sq) / s_total
        s_sq = sigma_sq * tau_sq / s_total

        # Gaussian evidence.
        exponent = -0.5 * m_dist_sq
        if exponent < -50.0:
            gauss_pdf = 0.0
        else:
            gauss_pdf = math.exp(exponent) / math.sqrt(2.0 * math.pi * s_total)

        # Uniform prior over the canonical interval.
        # EUCLIDEAN: z in [0, uniform_z_max]
        # INVDEPTH:  rho in [0, uniform_rho_max]
        # POLAR:     d in [uniform_d_min, uniform_d_max]
        if self._param is DepthParametrization.INVDEPTH:
            U_prior = 1.0 / s.uniform_rho_max
        elif self._param is DepthParametrization.POLAR:
            U_prior = 1.0 / (s.uniform_d_max - s.uniform_d_min)
        else:
            U_prior = 1.0 / s.uniform_z_max
        ab_sum = a + b
        C1 = (a / ab_sum) * gauss_pdf
        C2 = (b / ab_sum) * U_prior
        Z_norm = C1 + C2

        if Z_norm < 1e-30:
            feat.b = min(b + 1.0, s.ab_max)
            return

        w1 = C1 / Z_norm
        w2 = C2 / Z_norm

        new_mu = w1 * m + w2 * mu
        E_x2 = w1 * (s_sq + m * m) + w2 * (sigma_sq + mu * mu)
        new_sigma_sq = E_x2 - new_mu * new_mu
        if new_sigma_sq < 1e-8:
            new_sigma_sq = 1e-8

        denom1 = ab_sum + 1.0
        denom2 = denom1 * (ab_sum + 2.0)
        E_pi = (w1 * (a + 1.0) + w2 * a) / denom1
        E_pi2 = (
            w1 * (a + 1.0) * (a + 2.0)
            + w2 * a * (a + 1.0)
        ) / denom2
        v_pi = E_pi2 - E_pi * E_pi

        if v_pi < 1e-6 or E_pi <= 1e-6 or E_pi >= 1.0 - 1e-6:
            new_a = a + w1
            new_b = b + w2
        else:
            factor = E_pi * (1.0 - E_pi) / v_pi - 1.0
            if factor < 0.5:
                factor = 0.5
            new_a = E_pi * factor
            new_b = (1.0 - E_pi) * factor

        new_a = float(np.clip(new_a, s.ab_min, s.ab_max))
        new_b = float(np.clip(new_b, s.ab_min, s.ab_max))

        feat.canonical = float(new_mu)
        feat.canonical_var = float(new_sigma_sq)
        feat.a = new_a
        feat.b = new_b

    # ------------------------------------------------------------------
    # Query interface (for EqF warmstart)
    # ------------------------------------------------------------------

    def query(self, feat_id: int) -> tuple[float, float]:
        """Return (depth, depth_var) if converged, else (-1.0, inf).

        Consumers receive Euclidean depth regardless of canonical chart.
        """
        feat = self._features.get(feat_id)
        if feat is None or not math.isfinite(feat.canonical):
            return -1.0, float("inf")
        if feat.track_length < self.settings.min_track_length:
            return -1.0, float("inf")

        ab = feat.a + feat.b
        if ab <= 0.0:
            return -1.0, float("inf")
        inlier_ratio = feat.a / ab
        if inlier_ratio < self.settings.conv_inlier_ratio:
            return -1.0, float("inf")
        if feat.canonical_var > self.settings.conv_variance_threshold:
            return -1.0, float("inf")

        depth = self._canonical_to_depth(feat.canonical)
        depth_var = self._canonical_var_to_euclidean(
            feat.canonical, feat.canonical_var, depth
        )
        return float(depth), float(depth_var)

    # ------------------------------------------------------------------
    # PlaneDetector interface
    # ------------------------------------------------------------------

    @property
    def feat_uvs(self) -> Dict[int, tuple[float, float]]:
        """Pixel coords of all currently tracked features."""
        return {
            fid: (float(uv[0]), float(uv[1]))
            for fid, uv in self._prev_uvs.items()
        }

    def feat_positions_global(self, state) -> Dict[int, np.ndarray]:
        """Global-frame 3D positions for converged features.

        Uses the same rigid-transform chain as plane_detector.landmarks_to_global.
        """
        s = self.settings
        R_GtoI = state.sensor.pose.R.asMatrix()
        p_IinG = state.sensor.pose.x
        R_ItoC = state.sensor.camera_offset.R.asMatrix()
        p_IinC = state.sensor.camera_offset.x

        R_GtoC = R_ItoC @ R_GtoI
        R_CtoG = R_GtoC.T
        p_CinG = p_IinG - R_CtoG @ p_IinC

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        positions: Dict[int, np.ndarray] = {}
        for fid, uv in self._prev_uvs.items():
            feat = self._features.get(fid)
            if feat is None or not math.isfinite(feat.canonical):
                continue
            if feat.track_length < s.min_track_length:
                continue
            ab = feat.a + feat.b
            if ab <= 0.0 or (feat.a / ab) < s.conv_inlier_ratio:
                continue
            if feat.canonical_var > s.conv_variance_threshold:
                continue

            depth = self._canonical_to_depth(feat.canonical)
            x_norm = (uv[0] - cx) / fx
            y_norm = (uv[1] - cy) / fy
            p_cam = np.array(
                [x_norm * depth, y_norm * depth, depth]
            )
            positions[fid] = R_CtoG @ p_cam + p_CinG

        return positions

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def features(self) -> Dict[int, FeatureState]:
        return self._features

    def num_converged(self) -> int:
        s = self.settings
        n = 0
        for feat in self._features.values():
            if not math.isfinite(feat.canonical) or feat.track_length < s.min_track_length:
                continue
            ab = feat.a + feat.b
            if ab <= 0.0 or (feat.a / ab) < s.conv_inlier_ratio:
                continue
            if feat.canonical_var > s.conv_variance_threshold:
                continue
            n += 1
        return n


# ============================================================================
# 3D EqF variant (SOT(3) with full 3x3 covariance)
# ============================================================================


@dataclass
class FeatureState3D:
    """3D landmark state with full 3x3 covariance (Normal/Polar chart).

    State: landmark position q ∈ ℝ³ (camera frame)
    Covariance: Σ ∈ 𝕊₊(3) tracked in Normal chart coordinates

    The Normal chart parameterizes state as [sphere_bearing(2), log_invdepth]:
        - Indices 0,1: bearing on S^2 (invariant to scale)
        - Index 2: log(ρ/ρ₀) = -log(z/z₀), where ρ=1/z is inverse depth
    """
    feat_id: int
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 1.0)
    a: float = 10.0
    b: float = 2.0
    track_length: int = 0
    ref_T_WC: Optional[np.ndarray] = None
    ref_uv: Optional[np.ndarray] = None
    ref_stamp: float = -1.0

    @property
    def depth(self) -> float:
        return float(np.linalg.norm(self.position))

    @property
    def depth_var(self) -> float:
        """Extract depth variance from Normal chart covariance.

        Index 2 is log(ρ) = -log(z).  Since var(-log z) = var(log z),
        the transform is: var(z) = z² · cov[2,2]
        """
        if self.depth < 1e-6:
            return float("inf")
        z = self.depth
        var_log_depth = self.covariance[2, 2]
        return float(var_log_depth * (z * z))

    def inlier_ratio(self) -> float:
        ab = self.a + self.b
        if ab <= 0.0:
            return 0.0
        return self.a / ab


class SparseVogiatzisFilter3D(SparseVogiatzisFilter):
    """3D local IEKF on SOT(3) manifold with Normal chart.

    Maintains full 3×3 covariance in Normal chart coordinates.
    Sequential decoupled update: bearing IEKF + Vogiatzis depth.
    """

    def __init__(
        self,
        K: np.ndarray,
        settings: SparseVogSettings,
        eqf_suite=None,
        cam_ptr=None,
    ):
        super().__init__(K, settings)
        self._features3d: Dict[int, FeatureState3D] = {}
        self._eqf_suite = eqf_suite
        self._cam_ptr = cam_ptr
        self._last_T_WC: Optional[np.ndarray] = None

    def update(
        self,
        measurement: VisionMeasurement,
        T_WC: np.ndarray,
        P_vv: Optional[np.ndarray] = None,
        flowdep=None,
        v_C: Optional[np.ndarray] = None,
    ) -> None:
        """Propagate + update each tracked 3D feature's belief.

        This is the 3D equivalent of update() but maintains full
        3×3 covariance and uses Riccati update instead of scalar
        Gaussian-Beta mixture.
        """
        stamp = measurement.stamp
        curr_uvs = {
            fid: uv.astype(np.float64).copy()
            for fid, uv in measurement.cam_coordinates.items()
        }
        curr_uvs = self._undistort_uvs(curr_uvs)

        if self._prev_T_WC is None or self._prev_stamp < 0:
            self._prev_T_WC = T_WC.copy()
            self._prev_stamp = stamp
            self._prev_uvs = curr_uvs
            self._prev_uvs_undistorted = True
            return

        if not self._prev_uvs_undistorted:
            self._prev_uvs = self._undistort_uvs(self._prev_uvs)
        self._prev_uvs_undistorted = True

        dt = max(stamp - self._prev_stamp, 0.0)

        T_CW_curr = np.linalg.inv(T_WC)
        T_curr_prev = T_CW_curr @ self._prev_T_WC
        R = T_curr_prev[:3, :3]
        t = T_curr_prev[:3, 3]

        # Baseline uncertainty (improvement 1).
        P_vv_trans = None
        if P_vv is not None:
            P_vv_trans = P_vv[3:6, 3:6] if P_vv.shape == (6, 6) else P_vv
        t_norm_sq = float(t[0] ** 2 + t[1] ** 2 + t[2] ** 2)
        baseline_tau_sq = 0.0
        if P_vv_trans is not None and t_norm_sq > 1e-16 and dt > 0:
            t_hat = t / math.sqrt(t_norm_sq)
            var_t_mag = dt * dt * float(t_hat @ P_vv_trans @ t_hat)
            baseline_tau_sq = var_t_mag / t_norm_sq

        for fid, uv_curr in curr_uvs.items():
            uv_prev = self._prev_uvs.get(fid)
            if uv_prev is None:
                continue

            feat = self._features3d.get(fid)

            # --- Pending features: coast until flow gate is met ---
            if feat is None:
                pending = self._pending.get(fid)
                if pending is None:
                    if len(self._features3d) + len(self._pending) < self.settings.max_pool_size:
                        self._pending[fid] = (
                            self._prev_T_WC.copy(), uv_prev.copy(),
                            self._prev_stamp,
                        )
                    continue

                ref_T, ref_uv, ref_stamp = pending
                T_curr_ref = T_CW_curr @ ref_T
                R_ref = T_curr_ref[:3, :3]
                t_ref = T_curr_ref[:3, 3]
                z_obs, drive = self._triangulate(
                    ref_uv, uv_curr, R_ref, t_ref, v_C,
                )
                if z_obs <= 0.0:
                    continue
                flow_px = drive * float(self.K[0, 0])
                if flow_px < self.settings.reanchor_flow_px:
                    continue

                t_ref_nsq = float(t_ref[0] ** 2 + t_ref[1] ** 2 + t_ref[2] ** 2)
                feat_btau = 0.0
                if P_vv_trans is not None and t_ref_nsq > 1e-16 and dt > 0:
                    t_ref_hat = t_ref / math.sqrt(t_ref_nsq)
                    dt_total = max(stamp - ref_stamp, dt)
                    var_t = dt_total * dt * float(
                        t_ref_hat @ P_vv_trans @ t_ref_hat
                    )
                    feat_btau = var_t / t_ref_nsq

                init_pos = self._position_from_depth(
                    uv_curr, z_obs, self.settings.init_depth_var
                )
                feat = FeatureState3D(
                    feat_id=fid,
                    position=init_pos,
                    covariance=self._init_cov_3d(z_obs, drive, feat_btau),
                    a=self.settings.a_init,
                    b=self.settings.b_init,
                    ref_T_WC=T_WC.copy(),
                    ref_uv=uv_curr.copy(),
                    ref_stamp=stamp,
                )
                self._features3d[fid] = feat
                del self._pending[fid]
                self._bearing_update_3d(feat, uv_curr)
                self._depth_update_3d(feat, uv_curr, z_obs, drive, feat_btau)
                feat.track_length += 1
                continue

            # --- Existing features ---
            # Predict (frame-to-frame).
            if feat.depth > 0:
                self._predict_feature_3d(feat, R, t, P_vv, dt)

            # Per-frame bearing update (pixel obs is independent each frame).
            if feat.depth > 0:
                self._bearing_update_3d(feat, uv_curr)

            # Triangulate from reference for depth update.
            T_curr_ref = T_CW_curr @ feat.ref_T_WC
            R_ref = T_curr_ref[:3, :3]
            t_ref = T_curr_ref[:3, 3]
            z_obs, drive = self._triangulate(
                feat.ref_uv, uv_curr, R_ref, t_ref, v_C,
            )

            if z_obs <= 0.0:
                feat.track_length += 1
                continue

            flow_px = drive * float(self.K[0, 0])

            t_ref_nsq = float(t_ref[0] ** 2 + t_ref[1] ** 2 + t_ref[2] ** 2)
            feat_btau = 0.0
            if P_vv_trans is not None and t_ref_nsq > 1e-16 and dt > 0:
                t_ref_hat = t_ref / math.sqrt(t_ref_nsq)
                dt_total = max(stamp - feat.ref_stamp, dt)
                var_t = dt_total * dt * float(
                    t_ref_hat @ P_vv_trans @ t_ref_hat
                )
                feat_btau = var_t / t_ref_nsq

            if feat.depth <= 0:
                # Prediction invalidated — reinit.
                init_pos = self._position_from_depth(
                    uv_curr, z_obs, self.settings.init_depth_var
                )
                feat.position = init_pos
                feat.covariance = self._init_cov_3d(z_obs, drive, feat_btau)
                feat.a = self.settings.a_init
                feat.b = self.settings.b_init
                feat.ref_T_WC = T_WC.copy()
                feat.ref_uv = uv_curr.copy()
                feat.ref_stamp = stamp
                self._bearing_update_3d(feat, uv_curr)
                self._depth_update_3d(feat, uv_curr, z_obs, drive, feat_btau)
            elif flow_px >= self.settings.reanchor_flow_px:
                # Flow-gated depth update + re-anchor.
                self._depth_update_3d(feat, uv_curr, z_obs, drive, feat_btau)
                feat.ref_T_WC = T_WC.copy()
                feat.ref_uv = uv_curr.copy()
                feat.ref_stamp = stamp

            feat.track_length += 1

        lost = set(self._features3d.keys()) - set(curr_uvs.keys())
        for fid in lost:
            del self._features3d[fid]
        lost_pending = set(self._pending.keys()) - set(curr_uvs.keys())
        for fid in lost_pending:
            del self._pending[fid]

        self._prev_T_WC = T_WC.copy()
        self._prev_stamp = stamp
        self._prev_uvs = curr_uvs

    def _position_from_depth(
        self,
        uv: np.ndarray,
        depth: float,
        depth_var: float,
    ) -> np.ndarray:
        """Convert (u,v) + depth → camera-frame 3D position."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_norm = (uv[0] - cx) / fx
        y_norm = (uv[1] - cy) / fy
        return np.array([x_norm * depth, y_norm * depth, depth])

    def _init_cov_3d(self, z_obs: float, drive: float,
                     baseline_tau_sq: float) -> np.ndarray:
        """Initial Normal-chart covariance.

        Uses init_depth_var for all directions.  A wide isotropic prior
        is necessary because the coupled bearing update aggressively
        shrinks depth via cross-covariance — a tight depth init gets
        crushed to ~1e-6 after a single update, causing overconfidence.
        """
        v0 = self.settings.init_depth_var
        return np.eye(3) * v0

    def _predict_feature_3d(
        self,
        feat: FeatureState3D,
        R: np.ndarray,
        t: np.ndarray,
        P_vv: Optional[np.ndarray],
        dt: float,
    ) -> None:
        """Predict 3D landmark via Normal chart Jacobians (local IEKF)."""
        s = self.settings
        q_old = feat.position
        if feat.depth < s.min_depth:
            feat.position = np.zeros(3)
            return

        q_new = R @ q_old + t
        if q_new[2] < s.min_depth:
            feat.position = np.zeros(3)
            return

        M_norm2euc = conv_normal2euc(q_old)
        M_euc2norm = conv_euc2normal(q_new)

        J = M_euc2norm @ R @ M_norm2euc
        cov_new = J @ feat.covariance @ J.T

        if P_vv is not None and dt > 0.0:
            if P_vv.shape == (6, 6):
                P_ww = P_vv[0:3, 0:3]
                P_trans = P_vv[3:6, 3:6]
                qx = _skew(q_new)
                Q_euc = dt * dt * (P_trans + qx @ P_ww @ qx.T)
            else:
                Q_euc = P_vv * (dt * dt)
        else:
            Q_euc = np.eye(3) * s.process_depth_var * max(dt, 1e-3)

        Q_norm = M_euc2norm @ Q_euc @ M_euc2norm.T
        cov_new += Q_norm

        feat.position = q_new
        feat.covariance = cov_new

    def _bearing_update_3d(
            self,
            feat: FeatureState3D,
            y_observed: np.ndarray,
        ) -> None:
        """Per-frame bearing update (standard IEKF Kalman, no mixture)."""
        s = self.settings
        q = feat.position
        if np.linalg.norm(q) < 1e-4:
            return

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        if self._cam_ptr is not None:
            q_pred = q / np.linalg.norm(q)
            H_pred = self._cam_ptr.projection_jacobian(q_pred)

            y_obs_norm = np.array([
                (y_observed[0] - cx) / fx,
                (y_observed[1] - cy) / fy,
                1.0,
            ])
            y_obs_bearing = y_obs_norm / np.linalg.norm(y_obs_norm)
            H_obs = self._cam_ptr.projection_jacobian(y_obs_bearing)

            H_euc = 0.5 * (H_pred + H_obs)
        else:
            depth = feat.depth
            if depth < 1e-4:
                return
            x, y, z = q[0], q[1], q[2]
            H_euc = np.zeros((2, 3))
            H_euc[0, 0] = fx / z
            H_euc[0, 2] = -fx * x / (z * z)
            H_euc[1, 1] = fy / z
            H_euc[1, 2] = -fy * y / (z * z)

        M_norm2euc = conv_normal2euc(q)
        H_bearing = H_euc @ M_norm2euc

        y_pred = np.array([
            fx * q[0] / q[2] + cx,
            fy * q[1] / q[2] + cy,
        ])

        sigma_pixel_sq = s.sigma_pixel * s.sigma_pixel
        R_bearing = sigma_pixel_sq * np.eye(2)
        S_bearing = H_bearing @ feat.covariance @ H_bearing.T + R_bearing

        try:
            S_inv = np.linalg.inv(S_bearing + 1e-8 * np.eye(2))
        except np.linalg.LinAlgError:
            return

        K_bearing = feat.covariance @ H_bearing.T @ S_inv
        inn_bearing = y_observed - y_pred
        Delta_eps_bearing = K_bearing @ inn_bearing

        lm_new = point_chart_normal_inv(Delta_eps_bearing,
                                        Landmark(p=q, id=feat.feat_id))

        I_KH = np.eye(3) - K_bearing @ H_bearing
        P_new = I_KH @ feat.covariance @ I_KH.T + K_bearing @ R_bearing @ K_bearing.T

        eigvals = np.linalg.eigvalsh(P_new)
        if np.any(eigvals <= 0):
            return

        feat.position = lm_new.p
        feat.covariance = P_new

    def _depth_update_3d(
            self,
            feat: FeatureState3D,
            y_observed: np.ndarray,
            z_obs: float,
            drive: float,
            baseline_tau_sq: float = 0.0,
        ) -> None:
        """Flow-gated depth update (1D Vogiatzis mixture)."""
        s = self.settings
        q = feat.position
        z_cur = float(np.linalg.norm(q))
        if z_cur < s.min_depth:
            return

        d_obs = math.log(z_obs)
        d_pred = math.log(z_cur)
        inn_depth = d_obs - d_pred

        tau_d_sq = (z_obs * z_obs) * self._sigma_norm_sq / (drive * drive) + baseline_tau_sq
        S_dep = feat.covariance[2, 2] + tau_d_sq

        if S_dep < 1e-30:
            return

        maha_sq_dep = (inn_depth * inn_depth) / S_dep

        a, b = feat.a, feat.b
        ab = a + b

        if (
            s.mahalanobis_reset_chi2 > 0.0
            and ab > 0.0
            and (a / ab) < s.min_inlier_ratio
            and maha_sq_dep > s.mahalanobis_reset_chi2
        ):
            feat.position = self._position_from_depth(
                y_observed, z_obs, s.init_depth_var,
            )
            feat.covariance = np.eye(3) * s.init_depth_var
            feat.a = s.a_init
            feat.b = s.b_init
            return

        exponent = -0.5 * maha_sq_dep
        if exponent < -50.0:
            gauss_pdf = 0.0
        else:
            gauss_pdf = math.exp(exponent) / math.sqrt(2.0 * math.pi * S_dep)

        U_prior = 1.0 / (s.uniform_d_max - s.uniform_d_min)

        C1 = (a / ab) * gauss_pdf
        C2 = (b / ab) * U_prior
        Z_norm = C1 + C2

        if Z_norm < 1e-30:
            feat.b = min(b + 1.0, s.ab_max)
            return

        w1 = C1 / Z_norm
        w2 = C2 / Z_norm

        K_dep = -feat.covariance[:, 2] / S_dep

        Delta_eps_depth = (w1 * inn_depth) * K_dep

        lm_new = point_chart_normal_inv(Delta_eps_depth,
                                        Landmark(p=q, id=feat.feat_id))

        H_dep = np.array([[0.0, 0.0, -1.0]])
        KH_dep = np.outer(K_dep, H_dep)
        I_KH_dep = np.eye(3) - KH_dep
        P_kalman = I_KH_dep @ feat.covariance @ I_KH_dep.T + (tau_d_sq * np.outer(K_dep, K_dep))

        full_dep_eps = inn_depth * K_dep
        P_new = (
            w1 * P_kalman
            + w2 * feat.covariance
            + w1 * w2 * np.outer(full_dep_eps, full_dep_eps)
        )

        eigvals = np.linalg.eigvalsh(P_new)
        if np.any(eigvals <= 0):
            pass
        else:
            feat.position = lm_new.p
            feat.covariance = P_new

        denom1 = ab + 1.0
        denom2 = denom1 * (ab + 2.0)
        E_pi = (w1 * (a + 1.0) + w2 * a) / denom1
        E_pi2 = (
            w1 * (a + 1.0) * (a + 2.0)
            + w2 * a * (a + 1.0)
        ) / denom2
        v_pi = E_pi2 - E_pi * E_pi

        if v_pi < 1e-6 or E_pi <= 1e-6 or E_pi >= 1.0 - 1e-6:
            new_a = a + w1
            new_b = b + w2
        else:
            factor = E_pi * (1.0 - E_pi) / v_pi - 1.0
            if factor < 0.5:
                factor = 0.5
            new_a = E_pi * factor
            new_b = (1.0 - E_pi) * factor

        feat.a = float(np.clip(new_a, s.ab_min, s.ab_max))
        feat.b = float(np.clip(new_b, s.ab_min, s.ab_max))

    def _observed_bearing(self, uv: np.ndarray) -> np.ndarray:
        """Convert pixel → normalized bearing vector."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (uv[0] - cx) / fx
        y = (uv[1] - cy) / fy
        norm = np.sqrt(x * x + y * y + 1.0)
        return np.array([x / norm, y / norm, 1.0 / norm])

    def query(self, feat_id: int) -> tuple[float, float]:
        """Query (depth, depth_var) if converged, else (-1, inf)."""
        feat = self._features3d.get(feat_id)
        if feat is None or feat.depth <= 0:
            return -1.0, float("inf")
        if feat.track_length < self.settings.min_track_length:
            return -1.0, float("inf")
        if feat.inlier_ratio() < self.settings.conv_inlier_ratio:
            return -1.0, float("inf")
        if feat.depth_var > self.settings.conv_variance_threshold:
            return -1.0, float("inf")
        return feat.depth, feat.depth_var

    @property
    def features(self) -> Dict[int, FeatureState3D]:
        return self._features3d

    def feat_positions_global(self, state) -> Dict[int, np.ndarray]:
        """Global-frame 3D positions for converged 3D features."""
        R_GtoI = state.sensor.pose.R.asMatrix()
        p_IinG = state.sensor.pose.x
        R_ItoC = state.sensor.camera_offset.R.asMatrix()
        p_IinC = state.sensor.camera_offset.x

        R_GtoC = R_ItoC @ R_GtoI
        R_CtoG = R_GtoC.T
        p_CinG = p_IinG - R_CtoG @ p_IinC

        positions: Dict[int, np.ndarray] = {}
        for fid, feat in self._features3d.items():
            if feat.depth <= 0 or feat.track_length < self.settings.min_track_length:
                continue
            if feat.inlier_ratio() < self.settings.conv_inlier_ratio:
                continue
            if feat.depth_var > self.settings.conv_variance_threshold:
                continue

            positions[fid] = R_CtoG @ feat.position + p_CinG

        return positions
