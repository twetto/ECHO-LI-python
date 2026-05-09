"""
Patch-grid direct depth mapper with sparse GB priors.

Per-frame stateless densifier: lays a grid of patch centers on the current
image, optimises a scalar inverse-depth per patch via photometric + sparse-seed
cost, then fuses overlapping patches into output cells.

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
    n_gn_iters: int = 5
    fd_eps: float = 1e-3

    # Seed term
    lambda_seed: float = 1.0
    seed_radius_px: float = 32.0
    sigma_seed_floor: float = 0.01

    # Discrete search around seed init
    n_search_candidates: int = 5
    search_half_range: float = 0.5

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
    """Per-frame patch-grid direct depth mapper."""

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        sparse_vog: FilterT,
        curr_gray: np.ndarray,
        ref_gray: np.ndarray,
        T_ref_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run one frame of patch-grid direct depth.

        Args:
            sparse_vog: sparse Vogiatzis filter with current-frame seeds
            curr_gray:  (H, W) float32 current undistorted grayscale
            ref_gray:   (H, W) float32 reference undistorted grayscale
            T_ref_curr: (4, 4) SE(3) transform from current to reference frame

        Returns:
            depth_cells:  (Hc, Wc) float32 cell depth (NaN = unknown)
            var_cells:    (Hc, Wc) float32 cell inv-depth variance
            status_cells: (Hc, Wc) int32 PatchStatus enum
        """
        s = self.s
        H_orig, W_orig = curr_gray.shape[:2]
        sc = s.scale

        # 0. Downscale images and adjust intrinsics
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

        # 1. Gather sparse seeds as (M, 2) pixel, (M,) inv-depth, (M,) inv-depth var
        seed_uv, seed_rho, seed_var = self._gather_seeds(sparse_vog)
        if sc < 1.0 and seed_uv.shape[0] > 0:
            seed_uv = seed_uv * sc

        # 2. Build patch grid centers in scaled frame
        half = s.patch_size // 2
        grid_u = np.arange(half, W - half, s.patch_stride, dtype=np.float64)
        grid_v = np.arange(half, H - half, s.patch_stride, dtype=np.float64)
        gu, gv = np.meshgrid(grid_u, grid_v)
        patch_centers = np.stack([gu.ravel(), gv.ravel()], axis=1)  # (N, 2)
        N = patch_centers.shape[0]

        # 3. For each patch, find nearby seeds and initialise
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

        _solve_all_patches(
            patch_centers, patch_rho, patch_var, patch_status,
            patch_residual, patch_curvature,
            seed_uv, seed_rho, seed_var,
            curr_f32, ref_f32,
            R, t,
            fx, fy, cx, cy,
            s.patch_size, s.lambda_seed, seed_radius_scaled,
            s.sigma_seed_floor,
            s.photo_huber_delta, s.fd_eps,
            s.n_search_candidates, s.search_half_range,
            s.n_gn_iters,
            rho_min, rho_max,
            s.min_photo_curvature, s.max_photo_residual,
        )

        # 4. Fuse patches into output cells
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
def _patch_residual_jacobian(
    cu, cv, rho, half,
    R, t,
    fx, fy, cx, cy,
    curr_img, ref_img,
    huber_delta, fd_eps,
):
    """Compute GN gradient and Hessian via finite-difference Jacobian.

    Returns (gradient, hessian, mean_residual, n_valid).
    gradient = sum_p w_p * J_p * r_p
    hessian  = sum_p w_p * J_p^2
    """
    grad = 0.0
    hess = 0.0
    sum_abs_res = 0.0
    n_valid = 0

    rho_plus = rho + fd_eps
    rho_minus = rho - fd_eps

    for dy in range(-half, half):
        for dx in range(-half, half):
            pu = float(cu + dx)
            pv = float(cv + dy)
            i_curr = curr_img[int(pv), int(pu)]

            i_ref, valid = _warp_and_sample(
                pu, pv, rho, R, t, fx, fy, cx, cy, ref_img,
            )
            if not valid:
                continue

            i_ref_p, vp = _warp_and_sample(
                pu, pv, rho_plus, R, t, fx, fy, cx, cy, ref_img,
            )
            i_ref_m, vm = _warp_and_sample(
                pu, pv, rho_minus, R, t, fx, fy, cx, cy, ref_img,
            )
            if not vp or not vm:
                continue

            r = i_ref - i_curr
            J = (i_ref_p - i_ref_m) / (2.0 * fd_eps)

            ar = abs(r)
            if ar <= huber_delta:
                w = 1.0
            else:
                w = huber_delta / ar

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
    curr_img, ref_img,
    R, t,
    fx, fy, cx, cy,
    half,
    lambda_seed, seed_radius_px, sigma_seed_floor,
    huber_delta, fd_eps,
    n_search, search_half_range,
    n_gn_iters,
    rho_min, rho_max,
    min_curvature, max_residual,
):
    """Solve inverse depth for a single patch.

    Returns (rho, var, status, residual, curvature).
    """
    STATUS_UNKNOWN = 0
    STATUS_SEED_ONLY = 1
    STATUS_PHOTO_REFINED = 2
    STATUS_REJECTED = 3

    # Find nearby seeds
    M = seed_uv.shape[0]
    seed_rho_init = 0.0
    seed_weight_total = 0.0
    seed_precision_sum = 0.0

    # Accumulate weighted seed rho and precision for cost term
    n_seeds = 0
    seed_indices = np.empty(M, dtype=numba.int64)
    seed_weights = np.empty(M, dtype=numba.float64)
    seed_precisions = np.empty(M, dtype=numba.float64)

    for m in range(M):
        du = seed_uv[m, 0] - cu
        dv = seed_uv[m, 1] - cv
        dist = (du * du + dv * dv) ** 0.5
        if dist > seed_radius_px:
            continue
        w_spatial = 1.0 - dist / seed_radius_px
        var_capped = max(seed_var[m], sigma_seed_floor * sigma_seed_floor)
        prec = 1.0 / var_capped

        seed_indices[n_seeds] = m
        seed_weights[n_seeds] = w_spatial
        seed_precisions[n_seeds] = prec
        seed_rho_init += w_spatial * prec * seed_rho[m]
        seed_weight_total += w_spatial * prec
        seed_precision_sum += w_spatial * prec
        n_seeds += 1

    if seed_weight_total > 0.0:
        rho_init = seed_rho_init / seed_weight_total
    else:
        return 0.0, 1e10, STATUS_UNKNOWN, 1e10, 0.0

    rho_init = max(rho_min, min(rho_max, rho_init))

    # Discrete search
    best_rho = rho_init
    best_cost = 1e30

    for k in range(n_search):
        frac = -search_half_range + 2.0 * search_half_range * k / max(n_search - 1, 1)
        rho_k = rho_init * (1.0 + frac)
        rho_k = max(rho_min, min(rho_max, rho_k))

        photo_cost, nv = _patch_cost(
            cu, cv, rho_k, half,
            R, t, fx, fy, cx, cy,
            curr_img, ref_img, huber_delta,
        )
        if nv == 0:
            continue

        # Seed cost
        seed_cost = 0.0
        for si in range(n_seeds):
            m = seed_indices[si]
            dr = rho_k - seed_rho[m]
            seed_cost += lambda_seed * seed_weights[si] * seed_precisions[si] * dr * dr

        total = photo_cost + seed_cost
        if total < best_cost:
            best_cost = total
            best_rho = rho_k

    # GN refinement
    rho = best_rho
    final_residual = 1e10
    final_curvature = 0.0

    for it in range(n_gn_iters):
        grad_photo, hess_photo, mean_res, nv = _patch_residual_jacobian(
            cu, cv, rho, half,
            R, t, fx, fy, cx, cy,
            curr_img, ref_img,
            huber_delta, fd_eps,
        )
        if nv == 0:
            break

        # Add seed gradient and Hessian
        grad_seed = 0.0
        hess_seed = 0.0
        for si in range(n_seeds):
            m = seed_indices[si]
            dr = rho - seed_rho[m]
            wp = lambda_seed * seed_weights[si] * seed_precisions[si]
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
def _solve_all_patches(
    patch_centers, patch_rho, patch_var, patch_status,
    patch_residual, patch_curvature,
    seed_uv, seed_rho, seed_var,
    curr_img, ref_img,
    R, t,
    fx, fy, cx, cy,
    patch_size, lambda_seed, seed_radius_px,
    sigma_seed_floor,
    huber_delta, fd_eps,
    n_search, search_half_range,
    n_gn_iters,
    rho_min, rho_max,
    min_curvature, max_residual,
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
            curr_img, ref_img,
            R, t,
            fx, fy, cx, cy,
            half,
            lambda_seed, seed_radius_px, sigma_seed_floor,
            huber_delta, fd_eps,
            n_search, search_half_range,
            n_gn_iters,
            rho_min, rho_max,
            min_curvature, max_residual,
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
