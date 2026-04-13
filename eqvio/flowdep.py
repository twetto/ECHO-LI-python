"""
FlowDep: Dense depth Kalman filter via optical flow triangulation.

Phase (c) of ECHO-LI: loose coupling with the EqF sparse VIO.

Pipeline:
    EqF pose → DIS optical flow → per-pixel triangulation → Kalman predict/update
    → dense invdepth map → grid mesh plane detection (reuses PlaneDetector)
    → EqVIO landmark init + mesh plane masks

The filter maintains a per-pixel inverse-depth state and variance, propagated
via 3D warping and updated via epipolar triangulation from flow.

Keyframe management: a small pool of candidate keyframes scored by median
geometric drive (parallax quality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numba
import numpy as np


# ============================================================================
# Settings
# ============================================================================

@dataclass
class FlowDepSettings:
    """Configuration for the FlowDep dense depth filter."""

    # --- Inverse-depth filter ---
    init_invdepth_var: float = 1.0
    process_invdepth_var: float = 1e-1
    propagate_crit_var: float = 3.0
    max_inv_depth: float = 100.0
    border_crop_pixels: int = 5
    sigma_pixel: float = 0.1

    # --- DIS optical flow ---
    dis_preset: int = cv2.DISOpticalFlow_PRESET_MEDIUM
    dis_finest_scale: int = -1  # -1 = use preset default, 0 = full res
    # --- Keyframe pool ---
    max_keyframes: int = 5
    keyframe_flow_threshold: float = 3.0  # mean flow (px) to push frame as keyframe
    max_flow_pixels: float = 25.0  # retire keyframes whose flow exceeds this

    # --- Texture mask ---
    texture_mask: bool = False  # only observe textured pixels
    texture_threshold: int = 5  # absdiff / Laplacian threshold

    # --- Image downscale ---
    image_scale: float = 1.0  # state grid downscale (0.125 = 94x60 for 752x480)
    flow_scale: float = 0.0   # DIS flow downscale; 0 = same as image_scale

    # --- VIO warm-start ---
    enable_warmstart: bool = True  # allow query() to feed depth into VIO

    # --- Grid mesh plane detection ---
    grid_stride: int = 8
    grid_var_threshold: float = 0.1

    # --- Vogiatzis Gaussian-Beta mixture update ---
    uniform_rho_max: float = 1.0  # uniform outlier density upper bound (Z_min = 1m)
    a_init: float = 10.0
    b_init: float = 2.0  # asymmetric prior: ~85% inlier from DIS noise study
    ab_min: float = 1.0
    ab_max: float = 500.0
    min_inlier_ratio: float = 0.5  # gate query() on a/(a+b)
    mahalanobis_reset_chi2: float = 9.0  # reset stuck outlier pixels (<=0 disables)


# ============================================================================
# Numba kernels (from temp/visualize_convergence_pure_observation_260203.py)
# ============================================================================

@numba.jit(nopython=True)
def _depth_densification(K, dR, p, flow):
    """Per-pixel depth triangulation via derotated epipolar geometry.

    Args:
        K:    (3,3) intrinsic matrix
        dR:   R_curr_prev rotation from prev to curr camera frame
        p:    (3,) translation t_curr_prev in curr camera frame
        flow: (H,W,2) optical flow defined as u_curr - u_prev

    Returns:
        invdepth_map:  (H,W) inverse depth in curr frame (-1 = invalid)
        geom_drive_map: (H,W) parallax quality per pixel
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w, _ = flow.shape
    invdepth_map = np.ones((h, w), dtype=np.float32) * -1
    geom_drive_map = np.zeros((h, w), dtype=np.float32)

    for v in range(h):
        for u in range(w):
            x_curr_norm = (u - cx) / fx
            y_curr_norm = (v - cy) / fy

            u_prev = u - flow[v, u, 0]
            v_prev = v - flow[v, u, 1]
            if not (0 <= u_prev < w and 0 <= v_prev < h):
                continue

            x_prev_norm = (u_prev - cx) / fx
            y_prev_norm = (v_prev - cy) / fy
            bearing_prev_raw = np.array(
                [x_prev_norm, y_prev_norm, 1.0], dtype=np.float32
            )

            bearing_prev_aligned = dR @ bearing_prev_raw
            if bearing_prev_aligned[2] <= 1e-6:
                continue

            x_prev_rect = bearing_prev_aligned[0] / bearing_prev_aligned[2]
            y_prev_rect = bearing_prev_aligned[1] / bearing_prev_aligned[2]

            tx, ty, tz = p[0], p[1], p[2]
            num_x = x_prev_rect * tz - tx
            den_x = x_prev_rect - x_curr_norm
            num_y = y_prev_rect * tz - ty
            den_y = y_prev_rect - y_curr_norm

            geom_mag_sq = num_x**2 + num_y**2
            ideal_mag = np.sqrt(geom_mag_sq)
            obs_mag = np.sqrt(den_x**2 + den_y**2)

            if ideal_mag > 1e-2 and obs_mag > 1e-6:
                geom_drive_map[v, u] = ideal_mag
                dot_product = num_x * den_x + num_y * den_y
                cosine_sim = dot_product / (ideal_mag * obs_mag)

                if dot_product > 1e-6 and cosine_sim > 0.95:
                    Z_curr = geom_mag_sq / dot_product
                    if Z_curr > 0.1:
                        invdepth_map[v, u] = 1.0 / Z_curr

    return invdepth_map, geom_drive_map


@numba.jit(nopython=True)
def _bilinear_splatting(u_proj, v_proj, inv_z_proj, propagated_var, h, w):
    """Forward-warp inverse depth + variance via bilinear splatting."""
    predicted_invdepth_accum = np.zeros((h, w), dtype=np.float32)
    predicted_var_accum = np.zeros((h, w), dtype=np.float32)
    weights_accum = np.zeros((h, w), dtype=np.float32)
    for i in range(len(u_proj)):
        u_p, v_p = u_proj[i], v_proj[i]
        u_f, v_f = int(u_p), int(v_p)
        if u_f >= 0 and u_f < w - 1 and v_f >= 0 and v_f < h - 1:
            inv_z_p = inv_z_proj[i]
            var_p = propagated_var[i]
            dx, dy = u_p - u_f, v_p - v_f
            w_ll = (1 - dx) * (1 - dy)
            w_lr = dx * (1 - dy)
            w_ul = (1 - dx) * dy
            w_ur = dx * dy
            predicted_invdepth_accum[v_f, u_f] += w_ll * inv_z_p
            predicted_var_accum[v_f, u_f] += w_ll * var_p
            weights_accum[v_f, u_f] += w_ll
            predicted_invdepth_accum[v_f, u_f + 1] += w_lr * inv_z_p
            predicted_var_accum[v_f, u_f + 1] += w_lr * var_p
            weights_accum[v_f, u_f + 1] += w_lr
            predicted_invdepth_accum[v_f + 1, u_f] += w_ul * inv_z_p
            predicted_var_accum[v_f + 1, u_f] += w_ul * var_p
            weights_accum[v_f + 1, u_f] += w_ul
            predicted_invdepth_accum[v_f + 1, u_f + 1] += w_ur * inv_z_p
            predicted_var_accum[v_f + 1, u_f + 1] += w_ur * var_p
            weights_accum[v_f + 1, u_f + 1] += w_ur
    return predicted_invdepth_accum, predicted_var_accum, weights_accum


@numba.jit(nopython=True)
def _bilinear_splatting_ab(u_proj, v_proj, a_vals, b_vals, h, w):
    """Forward-warp Beta (a, b) counts via bilinear splatting."""
    a_accum = np.zeros((h, w), dtype=np.float32)
    b_accum = np.zeros((h, w), dtype=np.float32)
    weights_accum = np.zeros((h, w), dtype=np.float32)
    for i in range(len(u_proj)):
        u_p, v_p = u_proj[i], v_proj[i]
        u_f, v_f = int(u_p), int(v_p)
        if u_f >= 0 and u_f < w - 1 and v_f >= 0 and v_f < h - 1:
            a_p = a_vals[i]
            b_p = b_vals[i]
            dx, dy = u_p - u_f, v_p - v_f
            w_ll = (1 - dx) * (1 - dy)
            w_lr = dx * (1 - dy)
            w_ul = (1 - dx) * dy
            w_ur = dx * dy
            a_accum[v_f, u_f] += w_ll * a_p
            b_accum[v_f, u_f] += w_ll * b_p
            weights_accum[v_f, u_f] += w_ll
            a_accum[v_f, u_f + 1] += w_lr * a_p
            b_accum[v_f, u_f + 1] += w_lr * b_p
            weights_accum[v_f, u_f + 1] += w_lr
            a_accum[v_f + 1, u_f] += w_ul * a_p
            b_accum[v_f + 1, u_f] += w_ul * b_p
            weights_accum[v_f + 1, u_f] += w_ul
            a_accum[v_f + 1, u_f + 1] += w_ur * a_p
            b_accum[v_f + 1, u_f + 1] += w_ur * b_p
            weights_accum[v_f + 1, u_f + 1] += w_ur
    return a_accum, b_accum, weights_accum


@numba.jit(nopython=True)
def _vogiatzis_update(
    predicted_invdepth, predicted_var, predicted_a, predicted_b,
    observed_invdepth, geom_drive, sigma_norm, init_var,
    uniform_rho_max, a_init, b_init, ab_min, ab_max,
    min_inlier_ratio, mahal_reset_chi2,
):
    """Gaussian-Beta mixture update (SVO-style Vogiatzis filter).

    State per pixel: (mu, sigma^2, a, b). Likelihood is a mixture:
        p(x|rho,pi) = pi N(x;rho,tau^2) + (1-pi) U(x; 0, rho_max)
    with tau^2 = sigma_pixel^2 / geom_drive^2 (physically derived).
    Posterior is projected back to Gaussian-Beta via moment matching.

    Stuck-outlier recovery: if a/(a+b) has fallen below min_inlier_ratio AND
    the fresh observation is Mahalanobis-far (d^2 > mahal_reset_chi2), the
    pixel is reset to (x, tau^2, a_init, b_init). Passing mahal_reset_chi2<=0
    disables the reset.
    """
    h, w = predicted_invdepth.shape
    updated_invdepth = predicted_invdepth.copy()
    updated_var = predicted_var.copy()
    updated_a = predicted_a.copy()
    updated_b = predicted_b.copy()
    sigma_norm_sq = sigma_norm ** 2
    U_rho = 1.0 / uniform_rho_max  # uniform density (rho_min = 0)

    for r in range(h):
        for c in range(w):
            valid_pred = predicted_invdepth[r, c] >= 0
            valid_obs = observed_invdepth[r, c] >= 0

            drive = geom_drive[r, c]
            if drive > 1e-4:
                tau_sq = sigma_norm_sq / (drive ** 2)
            else:
                tau_sq = 1e6

            if valid_pred and valid_obs:
                mu = predicted_invdepth[r, c]
                sigma_sq = predicted_var[r, c]
                a = predicted_a[r, c]
                b = predicted_b[r, c]
                x = observed_invdepth[r, c]

                # Stuck-outlier recovery: if the pixel is already tagged as an
                # outlier (low inlier ratio) AND the fresh observation is far
                # in Mahalanobis, wipe the belief and restart from the obs.
                s_total = sigma_sq + tau_sq
                diff = x - mu
                m_dist_sq = diff * diff / s_total
                ab_pre = a + b
                if (
                    mahal_reset_chi2 > 0.0
                    and ab_pre > 0.0
                    and (a / ab_pre) < min_inlier_ratio
                    and m_dist_sq > mahal_reset_chi2
                ):
                    updated_invdepth[r, c] = x
                    updated_var[r, c] = tau_sq
                    updated_a[r, c] = a_init
                    updated_b[r, c] = b_init
                    continue

                # Inlier Kalman branch
                m = (mu * tau_sq + x * sigma_sq) / s_total
                s_sq = sigma_sq * tau_sq / s_total

                # Gaussian evidence N(x; mu, sigma^2 + tau^2)
                exponent = -0.5 * m_dist_sq
                if exponent < -50.0:
                    gauss_pdf = 0.0
                else:
                    gauss_pdf = math.exp(exponent) / math.sqrt(2.0 * math.pi * s_total)

                ab_sum = a + b
                C1 = (a / ab_sum) * gauss_pdf
                C2 = (b / ab_sum) * U_rho
                Z_norm = C1 + C2

                new_a = a
                new_b = b
                if Z_norm < 1e-30:
                    # Negligible evidence — treat as outlier, nudge b
                    new_mu = mu
                    new_sigma_sq = sigma_sq
                    new_b = b + 1.0
                else:
                    w1 = C1 / Z_norm
                    w2 = C2 / Z_norm

                    # Moment match rho
                    new_mu = w1 * m + w2 * mu
                    E_rho2 = w1 * (s_sq + m * m) + w2 * (sigma_sq + mu * mu)
                    new_sigma_sq = E_rho2 - new_mu * new_mu
                    if new_sigma_sq < 1e-8:
                        new_sigma_sq = 1e-8

                    # Moment match pi
                    denom1 = ab_sum + 1.0
                    denom2 = denom1 * (ab_sum + 2.0)
                    E_pi = (w1 * (a + 1.0) + w2 * a) / denom1
                    E_pi2 = (
                        w1 * (a + 1.0) * (a + 2.0)
                        + w2 * a * (a + 1.0)
                    ) / denom2
                    v_pi = E_pi2 - E_pi * E_pi

                    if v_pi < 1e-6 or E_pi <= 1e-6 or E_pi >= 1.0 - 1e-6:
                        # Degenerate moment match — fall back to soft counts
                        new_a = a + w1
                        new_b = b + w2
                    else:
                        factor = E_pi * (1.0 - E_pi) / v_pi - 1.0
                        if factor < 0.5:
                            factor = 0.5
                        new_a = E_pi * factor
                        new_b = (1.0 - E_pi) * factor

                # Clamp a, b
                if new_a < ab_min:
                    new_a = ab_min
                elif new_a > ab_max:
                    new_a = ab_max
                if new_b < ab_min:
                    new_b = ab_min
                elif new_b > ab_max:
                    new_b = ab_max

                updated_invdepth[r, c] = new_mu
                updated_var[r, c] = new_sigma_sq
                updated_a[r, c] = new_a
                updated_b[r, c] = new_b

            elif valid_obs:
                # New pixel — initialize from observation
                updated_invdepth[r, c] = observed_invdepth[r, c]
                updated_var[r, c] = init_var
                updated_a[r, c] = a_init
                updated_b[r, c] = b_init

    return updated_invdepth, updated_var, updated_a, updated_b


# ============================================================================
# DIS Optical Flow wrapper
# ============================================================================

class DISFlow:
    """Thin wrapper around OpenCV DISOpticalFlow."""

    def __init__(self, preset: int = cv2.DISOpticalFlow_PRESET_MEDIUM, finest_scale: int = -1):
        self._dis = cv2.DISOpticalFlow_create(preset)
        if finest_scale >= 0:
            self._dis.setFinestScale(finest_scale)

    def compute(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> np.ndarray:
        """Compute dense flow between two grayscale frames.

        Returns:
            (H, W, 2) float32 flow array (u_curr - u_prev convention).
        """
        flow = self._dis.calc(prev_gray, curr_gray, None)
        return flow


# ============================================================================
# Keyframe pool
# ============================================================================

@dataclass
class Keyframe:
    """A stored keyframe for FlowDep triangulation."""
    gray: np.ndarray
    T_WC: np.ndarray          # (4,4) camera-to-world SE3
    stamp: float
    score: float = 0.0        # latest geom_drive score


class KeyframePool:
    """Maintains a small pool of keyframes.

    Add:    when adjacent-frame mean flow exceeds a threshold.
    Select: keyframe with largest baseline to current frame.
    Retire: drop oldest when pool is full.
    """

    def __init__(self, settings: FlowDepSettings):
        self.settings = settings
        self._pool: list[Keyframe] = []

    @property
    def pool(self) -> list[Keyframe]:
        return self._pool

    def add_keyframe(self, gray: np.ndarray, T_WC: np.ndarray, stamp: float):
        if len(self._pool) >= self.settings.max_keyframes:
            self._pool.pop(0)  # drop oldest
        self._pool.append(Keyframe(gray=gray.copy(), T_WC=T_WC.copy(), stamp=stamp))

    def select_best(self, T_WC_curr: np.ndarray) -> Optional[Keyframe]:
        """Return keyframe with largest baseline to current pose, or None."""
        if not self._pool:
            return None
        T_CW_curr = np.linalg.inv(T_WC_curr)
        best_kf = None
        best_baseline = 0.0
        for kf in self._pool:
            baseline = np.linalg.norm((T_CW_curr @ kf.T_WC)[:3, 3])
            if baseline > best_baseline:
                best_baseline = baseline
                best_kf = kf
        return best_kf


# ============================================================================
# FlowDep Filter
# ============================================================================

class FlowDepFilter:
    """Dense per-pixel inverse-depth Kalman filter driven by optical flow.

    Usage:
        fdf = FlowDepFilter(K, settings)

        # Each vision frame (after EqF update):
        fdf.process_frame(curr_gray, T_WC_curr, stamp)

        # Query depth at a pixel:
        inv_d, var = fdf.query(u, v)

        # Get grid pseudo-features for mesh plane detection:
        grid_uvs, grid_positions = fdf.grid_features()
    """

    def __init__(self, K: np.ndarray, settings: Optional[FlowDepSettings] = None):
        self.settings = settings or FlowDepSettings()
        self._scale = self.settings.image_scale  # state grid scale
        self._flow_scale = (
            self.settings.flow_scale
            if self.settings.flow_scale > 0
            else self.settings.image_scale
        )

        # Scale intrinsics to match state grid (triangulation runs at state scale)
        K_scaled = K.astype(np.float32).copy()
        if self._scale != 1.0:
            K_scaled[0, 0] *= self._scale  # fx
            K_scaled[1, 1] *= self._scale  # fy
            K_scaled[0, 2] *= self._scale  # cx
            K_scaled[1, 2] *= self._scale  # cy
        self.K = K_scaled
        self._sigma_norm = self.settings.sigma_pixel / self.K[0, 0]

        # Dense state
        self.invdepth_state: Optional[np.ndarray] = None
        self.invdepth_var: Optional[np.ndarray] = None
        # Vogiatzis Beta parameters (inlier / outlier counts)
        self.a_state: Optional[np.ndarray] = None
        self.b_state: Optional[np.ndarray] = None

        # Components
        self.dis = DISFlow(self.settings.dis_preset, self.settings.dis_finest_scale)
        self.keyframe_pool = KeyframePool(self.settings)

        # Previous frame for adjacent-frame flow (fallback)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_T_WC: Optional[np.ndarray] = None
        self._prev_stamp: float = -1.0

    def reset(self):
        """Reset all state."""
        self.invdepth_state = None
        self.invdepth_var = None
        self.a_state = None
        self.b_state = None
        self._prev_gray = None
        self._prev_T_WC = None
        self._prev_stamp = -1.0
        self.keyframe_pool = KeyframePool(self.settings)

    def process_frame(
        self,
        curr_gray: np.ndarray,
        T_WC_curr: np.ndarray,
        stamp: float,
        P_vv: np.ndarray | None = None,
    ) -> bool:
        """Run one FlowDep cycle: predict (warp) + observe (triangulate) + update.

        Args:
            curr_gray: current grayscale image (H, W) uint8
            T_WC_curr: (4,4) camera-to-world SE3 from EqF
            stamp: timestamp
            P_vv: (3,3) velocity covariance from EqF Riccati, or None for fixed process noise

        Returns:
            True if depth map was updated this frame.
        """
        s = self.settings

        # Downscale image to FLOW scale (keyframes/prev/DIS all live here)
        h0, w0 = curr_gray.shape[:2]
        if self._flow_scale != 1.0:
            fw, fh = int(w0 * self._flow_scale), int(h0 * self._flow_scale)
            curr_gray = cv2.resize(
                curr_gray, (fw, fh), interpolation=cv2.INTER_AREA,
            )

        # State dimensions (triangulation + state grid)
        state_w = int(w0 * self._scale)
        state_h = int(h0 * self._scale)
        flow_h, flow_w = curr_gray.shape[:2]
        needs_flow_resize = (flow_w, flow_h) != (state_w, state_h)
        flow_to_state = self._scale / self._flow_scale

        if self._prev_gray is None:
            # First frame — just store
            self._prev_gray = curr_gray.copy()
            self._prev_T_WC = T_WC_curr.copy()
            self._prev_stamp = stamp
            self.keyframe_pool.add_keyframe(curr_gray, T_WC_curr, stamp)
            return False

        # --- Select source: best keyframe or adjacent ---
        use_keyframe = False
        best_kf = self.keyframe_pool.select_best(T_WC_curr)
        if best_kf is not None:
            kf_flow = self.dis.compute(best_kf.gray, curr_gray)
            kf_flow_mean = float(np.mean(np.sqrt(
                kf_flow[:, :, 0]**2 + kf_flow[:, :, 1]**2)))

            if kf_flow_mean > s.max_flow_pixels:
                # Keyframe too far — retire it and fall back to adjacent
                self.keyframe_pool.pool.remove(best_kf)
            elif kf_flow_mean >= s.keyframe_flow_threshold:
                # Good keyframe with enough parallax
                T_CW_curr = np.linalg.inv(T_WC_curr)
                T_curr_kf = T_CW_curr @ best_kf.T_WC
                R_curr_ref = T_curr_kf[:3, :3].astype(np.float32)
                t_curr_ref = T_curr_kf[:3, 3].astype(np.float32)
                flow = kf_flow
                use_keyframe = True

        if not use_keyframe:
            # Adjacent frame fallback
            flow = self.dis.compute(self._prev_gray, curr_gray)
            T_CW_curr = np.linalg.inv(T_WC_curr)
            T_curr_prev = T_CW_curr @ self._prev_T_WC
            R_curr_ref = T_curr_prev[:3, :3].astype(np.float32)
            t_curr_ref = T_curr_prev[:3, 3].astype(np.float32)

        # Push current frame as keyframe candidate (always available for future)
        self.keyframe_pool.add_keyframe(curr_gray, T_WC_curr, stamp)

        # --- Texture mask: suppress observations in textureless regions ---
        texture_valid = None
        if s.texture_mask:
            if not use_keyframe:
                # Adjacent frame: absdiff detects motion/texture
                diff = cv2.absdiff(self._prev_gray, curr_gray)
                tmask_flow = diff >= s.texture_threshold
            else:
                # Keyframe: Laplacian magnitude detects spatial texture
                lap = cv2.Laplacian(curr_gray, cv2.CV_16S)
                tmask_flow = np.abs(lap) >= s.texture_threshold
            if needs_flow_resize:
                texture_valid = cv2.resize(
                    tmask_flow.astype(np.uint8), (state_w, state_h),
                    interpolation=cv2.INTER_AREA,
                ) > 0
            else:
                texture_valid = tmask_flow

        # --- Downsample flow from flow resolution to state resolution ---
        if needs_flow_resize:
            flow_state = cv2.resize(
                flow, (state_w, state_h), interpolation=cv2.INTER_AREA,
            ) * flow_to_state
        else:
            flow_state = flow

        # --- Observe: triangulate per-pixel depth from flow (at state scale) ---
        observed_invdepth, geom_drive_map = _depth_densification(
            self.K, R_curr_ref, t_curr_ref, flow_state.astype(np.float32),
        )
        if texture_valid is not None:
            observed_invdepth[~texture_valid] = -1.0
        observed_invdepth[observed_invdepth > s.max_inv_depth] = -1.0
        n_valid = np.count_nonzero(observed_invdepth > 0)
        n_total = observed_invdepth.size
        print(f"[FlowDep] obs valid={n_valid}/{n_total} "
              f"t_norm={np.linalg.norm(t_curr_ref):.4f} "
              f"flow_med={np.median(np.linalg.norm(flow, axis=2)):.2f}")
        if s.border_crop_pixels > 0:
            b = s.border_crop_pixels
            observed_invdepth[:b, :] = -1.0
            observed_invdepth[-b:, :] = -1.0
            observed_invdepth[:, :b] = -1.0
            observed_invdepth[:, -b:] = -1.0

        # --- Predict: warp previous state to current frame ---
        dt = stamp - self._prev_stamp if self._prev_stamp >= 0 else 0.0
        predicted_invdepth, predicted_var, predicted_a, predicted_b = self._predict(
            T_WC_curr, P_vv, dt,
        )

        if predicted_invdepth is None:
            # First observation — initialize directly from triangulation
            self.invdepth_state = observed_invdepth.copy()
            self.invdepth_var = np.where(
                observed_invdepth > 0,
                (self._sigma_norm / np.maximum(geom_drive_map, 1e-8)) ** 2,
                s.init_invdepth_var,
            ).astype(np.float32)
            self.a_state = np.full_like(observed_invdepth, s.a_init, dtype=np.float32)
            self.b_state = np.full_like(observed_invdepth, s.b_init, dtype=np.float32)
        else:
            # --- Update: Vogiatzis Gaussian-Beta mixture fusion ---
            updated_invdepth, updated_var, updated_a, updated_b = _vogiatzis_update(
                predicted_invdepth, predicted_var, predicted_a, predicted_b,
                observed_invdepth, geom_drive_map,
                self._sigma_norm, s.init_invdepth_var,
                s.uniform_rho_max, s.a_init, s.b_init, s.ab_min, s.ab_max,
                s.min_inlier_ratio, s.mahalanobis_reset_chi2,
            )
            updated_invdepth[updated_invdepth > s.max_inv_depth] = -1.0
            self.invdepth_state = updated_invdepth
            self.invdepth_var = updated_var
            self.a_state = updated_a
            self.b_state = updated_b

        # Store for next frame (adjacent-frame fallback)
        self._prev_gray = curr_gray.copy()
        self._prev_T_WC = T_WC_curr.copy()
        self._prev_stamp = stamp

        return True

    def _predict(
        self,
        T_WC_curr: np.ndarray,
        P_vv: np.ndarray | None = None,
        dt: float = 0.0,
    ):
        """Propagate depth state from previous frame to current via 3D warping.

        P_vv: (3, 3) velocity covariance, assumed camera-frame (~body-frame on
        EuRoC where IMU-camera rotation is near identity). Process noise then
        becomes per-pixel: dt^2 * (b^T P_vv b) / Z^4 where b is the bearing.

        Returns:
            (predicted_invdepth, predicted_var, predicted_a, predicted_b).
            The a/b maps are None if Vogiatzis is disabled or state is empty.
        """
        s = self.settings

        if self.invdepth_state is None:
            return None, None, None, None

        h, w = self.invdepth_state.shape

        # Relative pose: current camera <- previous camera
        T_CW_curr = np.linalg.inv(T_WC_curr)
        T_curr_prev = T_CW_curr @ self._prev_T_WC
        R = T_curr_prev[:3, :3]
        t = T_curr_prev[:3, 3]

        valid_mask = self.invdepth_var.flatten() < s.propagate_crit_var
        if not np.any(valid_mask):
            return (
                np.full((h, w), -1.0, dtype=np.float32),
                np.full((h, w), s.init_invdepth_var, dtype=np.float32),
                np.full((h, w), s.a_init, dtype=np.float32),
                np.full((h, w), s.b_init, dtype=np.float32),
            )

        # Back-project valid pixels to 3D
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
        u_flat = u_grid.flatten()[valid_mask]
        v_flat = v_grid.flatten()[valid_mask]
        inv_depth_flat = self.invdepth_state.flatten()[valid_mask]

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        x = (u_flat - cx) / fx
        y = (v_flat - cy) / fy
        z = 1.0 / inv_depth_flat

        pts3d = np.stack((x * z, y * z, z), axis=0)
        transformed_pts = R @ pts3d + t.reshape(3, 1)
        z_new = transformed_pts[2, :]
        valid_proj = z_new > 0.1

        inv_z_proj = (1.0 / z_new[valid_proj]).astype(np.float32)
        var_old = self.invdepth_var.flatten()[valid_mask][valid_proj]
        Z_old = z[valid_proj]
        Z_new = z_new[valid_proj]

        # Jacobian for inverse-depth propagation
        z_axis_new_in_old = R[2, :]
        x_old_norm = x[valid_proj]
        y_old_norm = y[valid_proj]
        geometric_factor = (
            z_axis_new_in_old[0] * x_old_norm
            + z_axis_new_in_old[1] * y_old_norm
            + z_axis_new_in_old[2]
        )
        J_squared = ((Z_old / Z_new) ** 4) * (geometric_factor**2)
        # Process noise: project velocity covariance along each pixel's bearing,
        # scale by dt^2 / Z^4 (inverse-depth sensitivity to viewing-ray motion).
        if P_vv is not None and dt > 0.0:
            bx = x_old_norm
            by = y_old_norm
            b_norm_sq = bx * bx + by * by + 1.0
            # σ²_v_along = b^T P_vv b, with b = (bx, by, 1) / ||.||
            num = (
                bx * bx * P_vv[0, 0]
                + by * by * P_vv[1, 1]
                + P_vv[2, 2]
                + 2.0 * bx * by * P_vv[0, 1]
                + 2.0 * bx * P_vv[0, 2]
                + 2.0 * by * P_vv[1, 2]
            )
            sigma_v_along = num / b_norm_sq
            proc_noise = (dt * dt) * sigma_v_along / (Z_old ** 4)
        else:
            proc_noise = s.process_invdepth_var
        propagated_var = (J_squared * var_old + proc_noise).astype(np.float32)

        # Project to current image
        u_proj = (
            (transformed_pts[0, valid_proj] / z_new[valid_proj]) * fx + cx
        ).astype(np.float32)
        v_proj = (
            (transformed_pts[1, valid_proj] / z_new[valid_proj]) * fy + cy
        ).astype(np.float32)

        # Bilinear splatting
        pred_accum, var_accum, w_accum = _bilinear_splatting(
            u_proj, v_proj, inv_z_proj, propagated_var, h, w,
        )
        valid_pred = w_accum > 1e-6
        predicted_invdepth = np.full((h, w), -1.0, dtype=np.float32)
        predicted_var = np.full((h, w), s.init_invdepth_var, dtype=np.float32)
        predicted_invdepth[valid_pred] = pred_accum[valid_pred] / w_accum[valid_pred]
        predicted_var[valid_pred] = var_accum[valid_pred] / w_accum[valid_pred]

        a_flat = self.a_state.flatten()[valid_mask][valid_proj].astype(np.float32)
        b_flat = self.b_state.flatten()[valid_mask][valid_proj].astype(np.float32)
        a_accum, b_accum, _ = _bilinear_splatting_ab(
            u_proj, v_proj, a_flat, b_flat, h, w,
        )
        predicted_a = np.full((h, w), s.a_init, dtype=np.float32)
        predicted_b = np.full((h, w), s.b_init, dtype=np.float32)
        predicted_a[valid_pred] = a_accum[valid_pred] / w_accum[valid_pred]
        predicted_b[valid_pred] = b_accum[valid_pred] / w_accum[valid_pred]

        return predicted_invdepth, predicted_var, predicted_a, predicted_b

    def _median_depth(self) -> float:
        """Median depth from current state, for keyframe baseline threshold."""
        if self.invdepth_state is None:
            return 5.0
        valid = self.invdepth_state[self.invdepth_state > 0]
        if len(valid) == 0:
            return 5.0
        return float(1.0 / np.median(valid))

    # ------------------------------------------------------------------
    # Query interface (for EqF landmark init)
    # ------------------------------------------------------------------

    def query(self, u: float, v: float) -> tuple[float, float]:
        """Query inverse depth and variance at a pixel (full-res coords).

        Returns:
            (inv_depth, variance) or (-1.0, inf) if invalid/disabled.
        """
        if not self.settings.enable_warmstart:
            return -1.0, float("inf")
        if self.invdepth_state is None:
            return -1.0, float("inf")
        vi, ui = int(round(v * self._scale)), int(round(u * self._scale))
        h, w = self.invdepth_state.shape
        if not (0 <= vi < h and 0 <= ui < w):
            return -1.0, float("inf")
        inv_d = self.invdepth_state[vi, ui]
        var = self.invdepth_var[vi, ui]
        if inv_d <= 0:
            return -1.0, float("inf")
        # Vogiatzis inlier-ratio gate
        a = self.a_state[vi, ui]
        b = self.b_state[vi, ui]
        if a + b <= 0 or a / (a + b) < self.settings.min_inlier_ratio:
            return -1.0, float("inf")
        return float(inv_d), float(var)

    def query_depth(self, u: float, v: float) -> tuple[float, float]:
        """Query metric depth and its variance at a pixel.

        Returns:
            (depth, depth_var) or (-1.0, inf) if invalid.
        """
        inv_d, inv_var = self.query(u, v)
        if inv_d <= 0:
            return -1.0, float("inf")
        depth = 1.0 / inv_d
        # var(z) ≈ (dz/d(1/z))^2 * var(1/z) = z^4 * var(1/z)
        depth_var = (depth**4) * inv_var
        return float(depth), float(depth_var)

    # ------------------------------------------------------------------
    # Grid back-projection for mesh plane detection
    # ------------------------------------------------------------------

    def grid_features(
        self,
    ) -> tuple[dict[int, tuple[float, float]], dict[int, np.ndarray]]:
        """Back-project FlowDep depth grid to pseudo-feature dicts.

        Returns:
            feat_uvs:       {cell_id: (u, v)}
            feat_positions: {cell_id: np.array([x, y, z])} in camera frame
        """
        if self.invdepth_state is None:
            return {}, {}

        s = self.settings
        stride = s.grid_stride
        var_thresh = s.grid_var_threshold
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        H, W = self.invdepth_state.shape
        grid_cols = W // stride

        feat_uvs: dict[int, tuple[float, float]] = {}
        feat_positions: dict[int, np.ndarray] = {}

        for v0 in range(0, H, stride):
            for u0 in range(0, W, stride):
                v1 = min(v0 + stride, H)
                u1 = min(u0 + stride, W)
                cell_inv = self.invdepth_state[v0:v1, u0:u1]
                cell_var = self.invdepth_var[v0:v1, u0:u1]
                valid = (cell_inv > 0) & (cell_var < var_thresh)
                if not np.any(valid):
                    continue
                inv_d = float(np.median(cell_inv[valid]))
                if inv_d <= 0:
                    continue
                z = 1.0 / inv_d
                uc = (u0 + u1) * 0.5
                vc = (v0 + v1) * 0.5
                cell_id = (v0 // stride) * grid_cols + (u0 // stride)
                feat_uvs[cell_id] = (float(uc), float(vc))
                feat_positions[cell_id] = np.array([
                    (uc - cx) / fx * z,
                    (vc - cy) / fy * z,
                    z,
                ])

        return feat_uvs, feat_positions

    def grid_features_global(
        self, T_WC: np.ndarray,
    ) -> tuple[dict[int, tuple[float, float]], dict[int, np.ndarray]]:
        """Like grid_features() but positions are in world frame.

        Args:
            T_WC: (4,4) camera-to-world SE3.

        Returns:
            feat_uvs:       {cell_id: (u, v)}
            feat_positions: {cell_id: np.array([x, y, z])} in world frame
        """
        feat_uvs, feat_cam = self.grid_features()
        if not feat_cam:
            return {}, {}

        R = T_WC[:3, :3]
        t = T_WC[:3, 3]
        feat_world = {}
        for cid, p_cam in feat_cam.items():
            feat_world[cid] = R @ p_cam + t

        return feat_uvs, feat_world


# ============================================================================
# Helper: relabel EqVIO landmarks using grid plane mask
# ============================================================================

def relabel_landmarks_by_grid(
    eqvio_feat_uvs: dict[int, tuple[float, float]],
    grid_feat2plane: dict[int, int],
    grid_cols: int,
    stride: int,
    image_scale: float = 1.0,
) -> dict[int, int]:
    """Map EqVIO landmark IDs to plane IDs from the grid plane detector.

    Args:
        eqvio_feat_uvs:  {landmark_id: (u, v)} pixel positions of EqVIO features (full-res)
        grid_feat2plane: {grid_cell_id: plane_id} from PlaneDetector on grid
        grid_cols:       number of grid columns (W_scaled // stride)
        stride:          grid cell stride in pixels (in scaled space)
        image_scale:     FlowDep downscale factor (full-res UVs are scaled before lookup)

    Returns:
        {landmark_id: plane_id} for landmarks that fall in a labelled cell
    """
    eqvio_feat2plane: dict[int, int] = {}
    for fid, (u, v) in eqvio_feat_uvs.items():
        us, vs = u * image_scale, v * image_scale
        cell_id = (int(vs) // stride) * grid_cols + (int(us) // stride)
        if cell_id in grid_feat2plane:
            eqvio_feat2plane[fid] = grid_feat2plane[cell_id]
    return eqvio_feat2plane
