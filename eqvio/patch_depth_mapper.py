"""
Patch-grid direct depth mapper with sparse GB priors.

Lays a grid of patch centers on the current image, optimises a scalar
inverse-depth per patch via photometric + sparse-seed cost, then fuses
overlapping patches into output cells.  Maintains a 2-keyframe buffer
internally; the caller just feeds each new frame and the mapper picks
the best reference.

See docs/patch_grid_direct_depth_guide.md for the design rationale.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Union

import numba
import numpy as np

from .sparse_vogiatzis import (
    SparseVogiatzisFilter,
    SparseVogiatzisFilter3D,
)

FilterT = Union[SparseVogiatzisFilter, SparseVogiatzisFilter3D]


# ============================================================================
# Status enum
# ============================================================================

class PatchStatus(enum.IntEnum):
    UNKNOWN = 0
    SEED_ONLY = 1
    PHOTO_REFINED = 2
    REJECTED = 3


# ============================================================================
# Settings
# ============================================================================

@dataclass
class PatchDepthSettings:
    scale: float = 1.0

    patch_size: int = 8
    patch_stride: int = 4
    cell_size: int = 8

    max_depth: float = 20.0
    min_depth: float = 0.1

    # Photometric
    photo_huber_delta: float = 5.0
    sigma_photo: float = 5.0
    n_gn_iters: int = 5
    fd_eps: float = 1e-3

    # Seed term
    lambda_seed: float = 1.0
    seed_radius_px: float = 32.0
    sigma_seed_floor: float = 0.01

    # Discrete search around seed init
    n_search_candidates: int = 5
    search_half_range: float = 0.5

    # Keyframe management (fraction of median seed depth)
    min_baseline_ratio: float = 0.005
    max_baseline_ratio: float = 0.3

    # Status thresholds
    min_photo_curvature: float = 1e-6
    max_photo_residual: float = 20.0

    # Cell fusion
    var_floor: float = 1e-6
    status_weight_photo: float = 1.0
    status_weight_seed: float = 0.6


# ============================================================================
# Mapper
# ============================================================================

class PatchDepthMapper:
    """Patch-grid direct depth mapper with 2-keyframe buffer."""

    def __init__(
        self,
        K: np.ndarray,
        settings: Optional[PatchDepthSettings] = None,
    ):
        self.K = K.astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.s = settings or PatchDepthSettings()

        if self.s.patch_stride != self.s.cell_size // 2:
            raise ValueError("patch_stride must equal cell_size / 2")
        if self.s.patch_size < self.s.cell_size:
            raise ValueError("patch_size must be >= cell_size")

        self._keyframes: list[tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        sparse_vog: FilterT,
        curr_gray: np.ndarray,
        T_WC: np.ndarray,
        P_vv: Optional[np.ndarray] = None,
        dt: float = 0.0,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Run one frame of patch-grid direct depth.

        Args:
            sparse_vog: sparse Vogiatzis filter with current-frame seeds
            curr_gray:  (H, W) float32 current undistorted grayscale
            T_WC:       (4, 4) SE(3) camera-to-world pose
            P_vv:       (3,3) or (6,6) velocity covariance (optional)
            dt:         time since last frame (for P_vv scaling)

        Returns:
            None if no keyframe with sufficient baseline is available, else
            (depth_cells, var_cells, status_cells).
        """
        s = self.s

        # Gather seeds early so we can compute median depth for baseline check
        seed_uv, seed_rho, seed_var = self._gather_seeds(sparse_vog)
        if seed_rho.shape[0] > 0:
            median_depth = 1.0 / float(np.median(seed_rho))
        else:
            median_depth = s.max_depth

        # Pick best keyframe: largest baseline that's still within bounds
        ref_gray, T_ref_curr = self._select_keyframe(T_WC, median_depth)

        # Update keyframe buffer
        self._manage_keyframes(curr_gray, T_WC, median_depth)

        if ref_gray is None:
            return None

        # Compute sigma_warp_sq from pose uncertainty
        sigma_warp_sq = self._compute_sigma_warp_sq(
            T_ref_curr, P_vv, dt, median_depth,
        )

        return self._solve(
            curr_gray, ref_gray, T_ref_curr,
            seed_uv, seed_rho, seed_var, median_depth,
            sigma_warp_sq,
        )

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def _select_keyframe(
        self, T_WC: np.ndarray, median_depth: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Pick the keyframe with the largest usable baseline."""
        s = self.s
        min_bl = s.min_baseline_ratio * median_depth
        max_bl = s.max_baseline_ratio * median_depth

        best_kf = None
        best_baseline = -1.0

        T_CW = np.linalg.inv(T_WC)
        for kf_gray, kf_T_WC in self._keyframes:
            T_ref_curr = np.linalg.inv(kf_T_WC) @ T_WC
            baseline = float(np.linalg.norm(T_ref_curr[:3, 3]))
            if baseline >= min_bl and baseline <= max_bl and baseline > best_baseline:
                best_kf = (kf_gray, T_ref_curr)
                best_baseline = baseline

        if best_kf is None:
            return None, None
        return best_kf

    def _manage_keyframes(
        self, curr_gray: np.ndarray, T_WC: np.ndarray, median_depth: float,
    ):
        """Maintain 2 keyframes: push current frame, evict oldest if full."""
        s = self.s
        min_bl = s.min_baseline_ratio * median_depth

        if len(self._keyframes) < 2:
            self._keyframes.append((curr_gray.copy(), T_WC.copy()))
            return

        newest_T_WC = self._keyframes[-1][1]
        bl_from_newest = float(np.linalg.norm(
            (np.linalg.inv(newest_T_WC) @ T_WC)[:3, 3]
        ))
        if bl_from_newest >= min_bl:
            self._keyframes.pop(0)
            self._keyframes.append((curr_gray.copy(), T_WC.copy()))

    # ------------------------------------------------------------------
    # Pose-uncertainty weighting
    # ------------------------------------------------------------------

    def _compute_sigma_warp_sq(
        self,
        T_ref_curr: np.ndarray,
        P_vv: Optional[np.ndarray],
        dt: float,
        median_depth: float,
    ) -> float:
        """Pixel-space warp variance from velocity covariance.

        sigma_warp^2 = (f/z)^2 * (t_hat^T P_vv_trans t_hat) * dt^2
        """
        if P_vv is None or dt <= 0.0:
            return 0.0

        P_vv_trans = P_vv[3:6, 3:6] if P_vv.shape == (6, 6) else P_vv
        t = T_ref_curr[:3, 3]
        t_norm_sq = float(t @ t)
        if t_norm_sq < 1e-16:
            return 0.0

        t_hat = t / np.sqrt(t_norm_sq)
        var_t_mag = dt * dt * float(t_hat @ P_vv_trans @ t_hat)
        f = 0.5 * (self.fx + self.fy)
        sigma_warp_sq = (f / median_depth) ** 2 * var_t_mag
        return float(sigma_warp_sq)

    # ------------------------------------------------------------------
    # Core solve
    # ------------------------------------------------------------------

    def _solve(
        self,
        curr_gray: np.ndarray,
        ref_gray: np.ndarray,
        T_ref_curr: np.ndarray,
        seed_uv: np.ndarray,
        seed_rho: np.ndarray,
        seed_var: np.ndarray,
        median_depth: float,
        sigma_warp_sq: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.s
        H_orig, W_orig = curr_gray.shape[:2]
        sc = s.scale

        if sc < 1.0:
            import cv2
            H = int(H_orig * sc + 0.5)
            W = int(W_orig * sc + 0.5)
            curr_f32 = cv2.resize(
                curr_gray, (W, H), interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            ref_f32 = cv2.resize(
                ref_gray, (W, H), interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            fx = self.fx * sc
            fy = self.fy * sc
            cx = self.cx * sc
            cy = self.cy * sc
        else:
            H, W = H_orig, W_orig
            curr_f32 = curr_gray.astype(np.float32)
            ref_f32 = ref_gray.astype(np.float32)
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        if sc < 1.0 and seed_uv.shape[0] > 0:
            seed_uv = seed_uv * sc

        half = s.patch_size // 2
        grid_u = np.arange(half, W - half, s.patch_stride, dtype=np.float64)
        grid_v = np.arange(half, H - half, s.patch_stride, dtype=np.float64)
        gu, gv = np.meshgrid(grid_u, grid_v)
        patch_centers = np.stack([gu.ravel(), gv.ravel()], axis=1)
        N = patch_centers.shape[0]

        patch_rho = np.full(N, np.nan, dtype=np.float64)
        patch_var = np.full(N, np.inf, dtype=np.float64)
        patch_status = np.full(N, PatchStatus.UNKNOWN, dtype=np.int32)
        patch_residual = np.full(N, np.inf, dtype=np.float64)
        patch_curvature = np.zeros(N, dtype=np.float64)

        R = T_ref_curr[:3, :3].astype(np.float64)
        t = T_ref_curr[:3, 3].astype(np.float64)

        rho_min = 1.0 / s.max_depth
        rho_max = 1.0 / s.min_depth

        seed_radius_scaled = s.seed_radius_px * sc if sc < 1.0 else s.seed_radius_px
        sigma_photo_sq = s.sigma_photo * s.sigma_photo

        # Precompute reference image gradients (central difference)
        ref_grad_x = np.zeros_like(ref_f32)
        ref_grad_y = np.zeros_like(ref_f32)
        ref_grad_x[:, 1:-1] = (ref_f32[:, 2:] - ref_f32[:, :-2]) * 0.5
        ref_grad_y[1:-1, :] = (ref_f32[2:, :] - ref_f32[:-2, :]) * 0.5

        # Build spatial grid for seed lookup
        seed_grid_ids, seed_grid_starts, seed_grid_cols = _build_seed_grid(
            seed_uv, seed_radius_scaled, W, H,
        )

        _solve_all_patches(
            patch_centers, patch_rho, patch_var, patch_status,
            patch_residual, patch_curvature,
            seed_uv, seed_rho, seed_var,
            seed_grid_ids, seed_grid_starts, seed_grid_cols,
            curr_f32, ref_f32, ref_grad_x, ref_grad_y,
            R, t,
            fx, fy, cx, cy,
            s.patch_size, s.lambda_seed, seed_radius_scaled,
            s.sigma_seed_floor,
            s.photo_huber_delta,
            s.n_gn_iters,
            rho_min, rho_max,
            s.min_photo_curvature * s.patch_size * s.patch_size,
            s.max_photo_residual,
            sigma_photo_sq, sigma_warp_sq,
        )

        n_cells_u = max(1, W // s.cell_size)
        n_cells_v = max(1, H // s.cell_size)
        depth_cells, var_cells, status_cells = _fuse_cells(
            patch_centers, patch_rho, patch_var, patch_status,
            n_cells_u, n_cells_v,
            s.cell_size, s.var_floor,
            s.status_weight_photo, s.status_weight_seed,
        )

        return depth_cells, var_cells, status_cells

    # ------------------------------------------------------------------
    # Seed gathering
    # ------------------------------------------------------------------

    def _gather_seeds(
        self, sparse_vog: FilterT,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect (uv, inv_depth, inv_depth_var) for converged features only.

        Uses query() which enforces track length, inlier ratio, and variance
        convergence gates.
        """
        feat_uvs = sparse_vog.feat_uvs
        uvs, rhos, vars_ = [], [], []

        for fid, (u, v) in feat_uvs.items():
            z, z_var = sparse_vog.query(fid)
            if z <= 0 or z < self.s.min_depth:
                continue
            rho = 1.0 / z
            rho_var = z_var / (z ** 4)
            uvs.append((u, v))
            rhos.append(rho)
            vars_.append(rho_var)

        if not uvs:
            e2 = np.empty((0, 2), dtype=np.float64)
            e1 = np.empty((0,), dtype=np.float64)
            return e2, e1, e1

        return (
            np.array(uvs, dtype=np.float64),
            np.array(rhos, dtype=np.float64),
            np.array(vars_, dtype=np.float64),
        )


# ============================================================================
# Seed spatial grid
# ============================================================================

def _build_seed_grid(
    seed_uv: np.ndarray,
    cell_size: float,
    W: int,
    H: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Bin seeds into a spatial grid for O(1) neighborhood lookup.

    Returns:
        seed_grid_ids:    flat array of seed indices, grouped by cell
        seed_grid_starts: (n_cells+1,) start index into seed_grid_ids per cell
        n_cols:           number of grid columns
    """
    M = seed_uv.shape[0]
    n_cols = max(1, int(np.ceil(W / cell_size)))
    n_rows = max(1, int(np.ceil(H / cell_size)))
    n_cells = n_rows * n_cols

    # Count seeds per cell
    counts = np.zeros(n_cells, dtype=np.int64)
    cell_ids = np.empty(M, dtype=np.int64)
    for m in range(M):
        ci = min(int(seed_uv[m, 0] / cell_size), n_cols - 1)
        cj = min(int(seed_uv[m, 1] / cell_size), n_rows - 1)
        idx = cj * n_cols + ci
        cell_ids[m] = idx
        counts[idx] += 1

    # Build starts array (prefix sum)
    starts = np.zeros(n_cells + 1, dtype=np.int64)
    for i in range(n_cells):
        starts[i + 1] = starts[i] + counts[i]

    # Fill seed ids
    grid_ids = np.empty(M, dtype=np.int64)
    offsets = starts[:n_cells].copy()
    for m in range(M):
        idx = cell_ids[m]
        grid_ids[offsets[idx]] = m
        offsets[idx] += 1

    return grid_ids, starts, n_cols


# ============================================================================
# Numba kernels
# ============================================================================

@numba.jit(nopython=True)
def _warp_and_sample(
    u, v, rho,
    R, t,
    fx, fy, cx, cy,
    ref_img,
):
    """Warp pixel (u,v) at inverse depth rho from current to reference frame.

    Returns (I_ref_sampled, valid).
    """
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    # 3D point in current frame: bearing / rho = (x_n, y_n, 1) / rho
    qx = x_n / rho
    qy = y_n / rho
    qz = 1.0 / rho

    # Transform to reference frame
    rx = R[0, 0] * qx + R[0, 1] * qy + R[0, 2] * qz + t[0]
    ry = R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * qz + t[1]
    rz = R[2, 0] * qx + R[2, 1] * qy + R[2, 2] * qz + t[2]

    if rz <= 1e-6:
        return 0.0, False

    u_ref = fx * rx / rz + cx
    v_ref = fy * ry / rz + cy

    H, W = ref_img.shape
    ix = int(u_ref)
    iy = int(v_ref)
    if ix < 0 or ix >= W - 1 or iy < 0 or iy >= H - 1:
        return 0.0, False

    # Bilinear interpolation
    dx = u_ref - ix
    dy = v_ref - iy
    val = (
        (1.0 - dx) * (1.0 - dy) * ref_img[iy, ix]
        + dx * (1.0 - dy) * ref_img[iy, ix + 1]
        + (1.0 - dx) * dy * ref_img[iy + 1, ix]
        + dx * dy * ref_img[iy + 1, ix + 1]
    )
    return val, True


@numba.jit(nopython=True)
def _patch_cost(
    cu, cv, rho, half,
    R, t,
    fx, fy, cx, cy,
    curr_img, ref_img,
    huber_delta,
):
    """Compute robust photometric cost and Hessian for one patch at given rho.

    Returns (cost, n_valid).
    """
    cost = 0.0
    n_valid = 0
    for dy in range(-half, half):
        for dx in range(-half, half):
            pu = cu + dx
            pv = cv + dy
            i_curr = curr_img[int(pv), int(pu)]
            i_ref, valid = _warp_and_sample(
                float(pu), float(pv), rho,
                R, t, fx, fy, cx, cy, ref_img,
            )
            if not valid:
                continue
            r = i_ref - i_curr
            ar = abs(r)
            if ar <= huber_delta:
                cost += 0.5 * r * r
            else:
                cost += huber_delta * (ar - 0.5 * huber_delta)
            n_valid += 1
    return cost, n_valid


@numba.jit(nopython=True)
def _sample_ref(u_ref, v_ref, ref_img):
    """Bilinear sample from ref_img at (u_ref, v_ref). Returns (val, valid)."""
    H, W = ref_img.shape
    ix = int(u_ref)
    iy = int(v_ref)
    if ix < 0 or ix >= W - 1 or iy < 0 or iy >= H - 1:
        return 0.0, False
    dx = u_ref - ix
    dy = v_ref - iy
    val = (
        (1.0 - dx) * (1.0 - dy) * ref_img[iy, ix]
        + dx * (1.0 - dy) * ref_img[iy, ix + 1]
        + (1.0 - dx) * dy * ref_img[iy + 1, ix]
        + dx * dy * ref_img[iy + 1, ix + 1]
    )
    return val, True


@numba.jit(nopython=True)
def _warp_uv(u, v, rho, R, t, fx, fy, cx, cy):
    """Warp pixel (u,v) at inv-depth rho, return (u_ref, v_ref, valid)."""
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    qx = x_n / rho
    qy = y_n / rho
    qz = 1.0 / rho
    rx = R[0, 0] * qx + R[0, 1] * qy + R[0, 2] * qz + t[0]
    ry = R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * qz + t[1]
    rz = R[2, 0] * qx + R[2, 1] * qy + R[2, 2] * qz + t[2]
    if rz <= 1e-6:
        return 0.0, 0.0, False
    u_ref = fx * rx / rz + cx
    v_ref = fy * ry / rz + cy
    return u_ref, v_ref, True


@numba.jit(nopython=True)
def _patch_residual_jacobian(
    cu, cv, rho, half,
    R, t,
    fx, fy, cx, cy,
    curr_img, ref_img, ref_grad_x, ref_grad_y,
    huber_delta,
    sigma_photo_sq, sigma_warp_sq,
):
    """Compute GN gradient and Hessian via analytical Jacobian.

    J_rho = grad_I^T @ d(u_ref)/d(rho)

    Image gradients are sampled from precomputed gradient images.

    Returns (gradient, hessian, mean_residual, n_valid).
    """
    grad = 0.0
    hess = 0.0
    sum_abs_res = 0.0
    n_valid = 0

    rho_sq = rho * rho

    for dy in range(-half, half):
        for dx in range(-half, half):
            pu = float(cu + dx)
            pv = float(cv + dy)
            i_curr = curr_img[int(pv), int(pu)]

            x_n = (pu - cx) / fx
            y_n = (pv - cy) / fy

            qx = x_n / rho
            qy = y_n / rho
            qz = 1.0 / rho

            px = R[0, 0] * qx + R[0, 1] * qy + R[0, 2] * qz + t[0]
            py = R[1, 0] * qx + R[1, 1] * qy + R[1, 2] * qz + t[1]
            pz = R[2, 0] * qx + R[2, 1] * qy + R[2, 2] * qz + t[2]

            if pz <= 1e-6:
                continue

            u_ref = fx * px / pz + cx
            v_ref = fy * py / pz + cy

            i_ref, svalid = _sample_ref(u_ref, v_ref, ref_img)
            if not svalid:
                continue

            # Sample precomputed gradients at warped location
            gx, gx_valid = _sample_ref(u_ref, v_ref, ref_grad_x)
            gy, gy_valid = _sample_ref(u_ref, v_ref, ref_grad_y)
            if not (gx_valid and gy_valid):
                continue

            # d(p_ref)/d(rho) = R @ [-x_n, -y_n, -1] / rho^2
            dpx = -(R[0, 0] * x_n + R[0, 1] * y_n + R[0, 2]) / rho_sq
            dpy = -(R[1, 0] * x_n + R[1, 1] * y_n + R[1, 2]) / rho_sq
            dpz = -(R[2, 0] * x_n + R[2, 1] * y_n + R[2, 2]) / rho_sq

            inv_pz_sq = 1.0 / (pz * pz)
            du_drho = fx * (dpx * pz - px * dpz) * inv_pz_sq
            dv_drho = fy * (dpy * pz - py * dpz) * inv_pz_sq

            J = gx * du_drho + gy * dv_drho

            r = i_ref - i_curr

            grad_I_sq = gx * gx + gy * gy
            sigma_eff_sq = sigma_photo_sq + grad_I_sq * sigma_warp_sq
            inv_sigma_eff_sq = 1.0 / sigma_eff_sq

            ar = abs(r)
            if ar <= huber_delta:
                w = inv_sigma_eff_sq
            else:
                w = inv_sigma_eff_sq * huber_delta / ar

            grad += w * J * r
            hess += w * J * J
            sum_abs_res += ar
            n_valid += 1

    mean_res = sum_abs_res / max(n_valid, 1)
    return grad, hess, mean_res, n_valid


@numba.jit(nopython=True)
def _solve_one_patch(
    cu, cv,
    seed_uv, seed_rho, seed_var,
    seed_grid_ids, seed_grid_starts, seed_grid_cols,
    curr_img, ref_img, ref_grad_x, ref_grad_y,
    R, t,
    fx, fy, cx, cy,
    half,
    lambda_seed, seed_radius_px, sigma_seed_floor,
    huber_delta,
    n_gn_iters,
    rho_min, rho_max,
    min_curvature, max_residual,
    sigma_photo_sq, sigma_warp_sq,
):
    """Solve inverse depth for a single patch.

    Returns (rho, var, status, residual, curvature).
    """
    STATUS_UNKNOWN = 0
    STATUS_SEED_ONLY = 1
    STATUS_PHOTO_REFINED = 2
    STATUS_REJECTED = 3

    H_img = curr_img.shape[0]
    n_total_cells = seed_grid_starts.shape[0] - 1
    n_grid_rows = n_total_cells // seed_grid_cols

    # Find nearby seeds via spatial grid
    ci_min = max(0, int((cu - seed_radius_px) / seed_radius_px))
    ci_max = min(seed_grid_cols - 1, int((cu + seed_radius_px) / seed_radius_px))
    cj_min = max(0, int((cv - seed_radius_px) / seed_radius_px))
    cj_max = min(n_grid_rows - 1, int((cv + seed_radius_px) / seed_radius_px))

    seed_rho_init = 0.0
    seed_weight_total = 0.0
    seed_precision_sum = 0.0

    MAX_NEARBY = 64
    n_seeds = 0
    local_indices = np.empty(MAX_NEARBY, dtype=numba.int64)
    local_weights = np.empty(MAX_NEARBY, dtype=numba.float64)
    local_precisions = np.empty(MAX_NEARBY, dtype=numba.float64)

    for cj in range(cj_min, cj_max + 1):
        for ci in range(ci_min, ci_max + 1):
            cell_idx = cj * seed_grid_cols + ci
            for k in range(seed_grid_starts[cell_idx], seed_grid_starts[cell_idx + 1]):
                m = seed_grid_ids[k]
                du = seed_uv[m, 0] - cu
                dv = seed_uv[m, 1] - cv
                dist = (du * du + dv * dv) ** 0.5
                if dist > seed_radius_px:
                    continue
                if n_seeds >= MAX_NEARBY:
                    break
                w_spatial = 1.0 - dist / seed_radius_px
                var_capped = max(seed_var[m], sigma_seed_floor * sigma_seed_floor)
                prec = 1.0 / var_capped

                local_indices[n_seeds] = m
                local_weights[n_seeds] = w_spatial
                local_precisions[n_seeds] = prec
                seed_rho_init += w_spatial * prec * seed_rho[m]
                seed_weight_total += w_spatial * prec
                seed_precision_sum += w_spatial * prec
                n_seeds += 1

    if seed_weight_total > 0.0:
        rho_init = seed_rho_init / seed_weight_total
    else:
        return 0.0, 1e10, STATUS_UNKNOWN, 1e10, 0.0

    rho = max(rho_min, min(rho_max, rho_init))
    final_residual = 1e10
    final_curvature = 0.0

    for it in range(n_gn_iters):
        grad_photo, hess_photo, mean_res, nv = _patch_residual_jacobian(
            cu, cv, rho, half,
            R, t, fx, fy, cx, cy,
            curr_img, ref_img, ref_grad_x, ref_grad_y,
            huber_delta,
            sigma_photo_sq, sigma_warp_sq,
        )
        if nv == 0:
            break

        # Add seed gradient and Hessian
        grad_seed = 0.0
        hess_seed = 0.0
        for si in range(n_seeds):
            m = local_indices[si]
            dr = rho - seed_rho[m]
            wp = lambda_seed * local_weights[si] * local_precisions[si]
            grad_seed += wp * dr
            hess_seed += wp

        grad_total = grad_photo + grad_seed
        hess_total = hess_photo + hess_seed

        if hess_total < 1e-12:
            break

        delta = -grad_total / hess_total
        # Damped step
        rho_new = rho + delta
        rho_new = max(rho_min, min(rho_max, rho_new))
        rho = rho_new

        final_residual = mean_res
        final_curvature = hess_photo

    # Status classification
    if final_curvature >= min_curvature and final_residual <= max_residual:
        status = STATUS_PHOTO_REFINED
        hess_total = final_curvature + seed_precision_sum * lambda_seed
        var = 1.0 / max(hess_total, 1e-12)
    elif n_seeds > 0:
        if final_residual > max_residual and final_curvature >= min_curvature:
            status = STATUS_REJECTED
            var = 1e10
        else:
            status = STATUS_SEED_ONLY
            var = 1.0 / max(seed_precision_sum * lambda_seed, 1e-12)
    else:
        status = STATUS_UNKNOWN
        var = 1e10

    return rho, var, status, final_residual, final_curvature


@numba.jit(nopython=True, parallel=True)
def _seed_only_all_patches(
    patch_centers, patch_rho, patch_var, patch_status,
    seed_uv, seed_rho, seed_var,
    seed_radius_px, sigma_seed_floor,
    lambda_seed,
):
    """Assign seed-only depth to all patches (no photometric refinement)."""
    STATUS_UNKNOWN = 0
    STATUS_SEED_ONLY = 1
    N = patch_centers.shape[0]
    M = seed_uv.shape[0]
    for i in numba.prange(N):
        cu = patch_centers[i, 0]
        cv = patch_centers[i, 1]
        rho_acc = 0.0
        prec_acc = 0.0
        for m in range(M):
            du = seed_uv[m, 0] - cu
            dv = seed_uv[m, 1] - cv
            dist = (du * du + dv * dv) ** 0.5
            if dist > seed_radius_px:
                continue
            w_spatial = 1.0 - dist / seed_radius_px
            var_capped = max(seed_var[m], sigma_seed_floor * sigma_seed_floor)
            prec = w_spatial * lambda_seed / var_capped
            rho_acc += prec * seed_rho[m]
            prec_acc += prec
        if prec_acc > 0.0:
            patch_rho[i] = rho_acc / prec_acc
            patch_var[i] = 1.0 / prec_acc
            patch_status[i] = STATUS_SEED_ONLY
        else:
            patch_status[i] = STATUS_UNKNOWN


@numba.jit(nopython=True, parallel=True)
def _solve_all_patches(
    patch_centers, patch_rho, patch_var, patch_status,
    patch_residual, patch_curvature,
    seed_uv, seed_rho, seed_var,
    seed_grid_ids, seed_grid_starts, seed_grid_cols,
    curr_img, ref_img, ref_grad_x, ref_grad_y,
    R, t,
    fx, fy, cx, cy,
    patch_size, lambda_seed, seed_radius_px,
    sigma_seed_floor,
    huber_delta,
    n_gn_iters,
    rho_min, rho_max,
    min_curvature, max_residual,
    sigma_photo_sq, sigma_warp_sq,
):
    """Solve all patches in parallel."""
    N = patch_centers.shape[0]
    half = patch_size // 2
    for i in numba.prange(N):
        cu = patch_centers[i, 0]
        cv = patch_centers[i, 1]
        rho, var, status, res, curv = _solve_one_patch(
            cu, cv,
            seed_uv, seed_rho, seed_var,
            seed_grid_ids, seed_grid_starts, seed_grid_cols,
            curr_img, ref_img, ref_grad_x, ref_grad_y,
            R, t,
            fx, fy, cx, cy,
            half,
            lambda_seed, seed_radius_px, sigma_seed_floor,
            huber_delta,
            n_gn_iters,
            rho_min, rho_max,
            min_curvature, max_residual,
            sigma_photo_sq, sigma_warp_sq,
        )
        patch_rho[i] = rho
        patch_var[i] = var
        patch_status[i] = status
        patch_residual[i] = res
        patch_curvature[i] = curv


@numba.jit(nopython=True)
def _fuse_cells(
    patch_centers, patch_rho, patch_var, patch_status,
    n_cells_u, n_cells_v,
    cell_size, var_floor,
    w_photo, w_seed,
):
    """Assign each patch to the cell containing its center and fuse.

    With stride = cell_size / 2, each interior cell receives ~4 patch centers.
    """
    STATUS_UNKNOWN = 0
    STATUS_SEED_ONLY = 1
    STATUS_PHOTO_REFINED = 2

    depth_cells = np.full((n_cells_v, n_cells_u), np.nan, dtype=np.float64)
    var_cells = np.full((n_cells_v, n_cells_u), np.inf, dtype=np.float64)
    status_cells = np.zeros((n_cells_v, n_cells_u), dtype=np.int32)

    rho_acc = np.zeros((n_cells_v, n_cells_u), dtype=np.float64)
    w_acc = np.zeros((n_cells_v, n_cells_u), dtype=np.float64)
    best_status = np.zeros((n_cells_v, n_cells_u), dtype=np.int32)

    N = patch_centers.shape[0]
    for i in range(N):
        s = patch_status[i]
        if s == STATUS_UNKNOWN or s == 3:  # UNKNOWN or REJECTED
            continue

        cu = patch_centers[i, 0]
        cv = patch_centers[i, 1]
        ci = int(cu / cell_size)
        cj = int(cv / cell_size)
        if ci < 0 or ci >= n_cells_u or cj < 0 or cj >= n_cells_v:
            continue

        if s == STATUS_PHOTO_REFINED:
            sw = w_photo
        elif s == STATUS_SEED_ONLY:
            sw = w_seed
        else:
            sw = 0.0

        v = max(patch_var[i], var_floor)
        w = sw / v
        rho_acc[cj, ci] += w * patch_rho[i]
        w_acc[cj, ci] += w

        if s > best_status[cj, ci]:
            best_status[cj, ci] = s

    for j in range(n_cells_v):
        for i in range(n_cells_u):
            if w_acc[j, i] > 0.0:
                rho_fused = rho_acc[j, i] / w_acc[j, i]
                depth_cells[j, i] = 1.0 / rho_fused
                var_cells[j, i] = 1.0 / w_acc[j, i]
                status_cells[j, i] = best_status[j, i]

    return depth_cells, var_cells, status_cells
