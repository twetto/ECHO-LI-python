"""
Plane fitting for EqVIO-P.

Port of: ov_plane/src/track_plane/PlaneFitting.cpp

Given a set of 3D feature points (from the EqF state) that have been
grouped onto the same plane by :class:`PlaneDetector`, this module
recovers the plane's closest-point (CP) parameterisation:

    cp = n * d        (3-vector in global frame)

where *n* is the outward unit normal and *d* is the distance from the
origin to the plane.

Three levels of refinement:

1. :func:`fit_plane_linear`  — single SVD solve (fast, used inside RANSAC)
2. :func:`fit_plane_ransac`  — RANSAC wrapper with inlier selection
3. :func:`optimize_plane`    — joint least-squares refinement of CP + feature
   positions using reprojection + point-on-plane factors (scipy, replaces Ceres)

Typical pipeline:

    # After plane_detector.update(...):
    for pid in active_plane_ids:
        fids = [fid for fid, p in feat2plane.items() if p == pid]
        points = [feat_positions[fid] for fid in fids]
        ok, cp_inG, inlier_ids = fit_plane_ransac(fids, points)
        if ok:
            ok2, cp_inG = optimize_plane(cp_inG, fids, feat_positions, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class PlaneFittingSettings:
    """Configuration for RANSAC + optimization.

    Defaults match ov_plane's PlaneFitting.cpp.
    """
    # RANSAC
    ransac_n_points: int = 5
    """Number of points per RANSAC sample."""

    ransac_max_iters: int = 200
    """Maximum RANSAC iterations."""

    ransac_inlier_threshold: float = 0.05
    """Max point-to-plane distance (m) for an inlier."""

    ransac_min_inlier_ratio: float = 0.80
    """Minimum fraction of points that must be inliers."""

    ransac_min_point_separation: float = 0.05
    """Minimum 3D distance between RANSAC sample points."""

    ransac_seed: int = 8888
    """RNG seed for reproducibility."""

    # Linear fit
    max_condition_number: float = 200.0
    """Reject linear fits with condition number above this."""

    min_cp_distance: float = 0.02
    """Reject planes whose CP is closer to origin than this (degenerate)."""

    # Optimization
    opt_max_iters: int = 12
    """Max Levenberg-Marquardt iterations for joint refinement."""

    opt_sigma_px_norm: float = 0.005
    """Feature observation noise in normalised image coordinates (σ_raw / f)."""

    opt_sigma_c: float = 0.01
    """Point-on-plane constraint noise (m)."""

    opt_cauchy_scale: float = 1.0
    """Cauchy loss scale for robustness."""

    opt_max_inlier_distance: float = 0.03
    """Post-optimization inlier distance threshold."""

    opt_min_inlier_ratio: float = 0.80
    """Post-optimization minimum inlier fraction."""


# ---------------------------------------------------------------------------
# 1. Linear plane fit (used inside RANSAC)
# ---------------------------------------------------------------------------

def fit_plane_linear(
    points: np.ndarray,
    max_cond: float = 200.0,
    min_cp_dist: float = 0.02,
) -> tuple[bool, np.ndarray]:
    """Fit a plane to a set of 3D points via least-squares.

    Solves  a*x + b*y + c*z + 1 = 0  for (a, b, c), then normalises
    so that (a, b, c) is unit and d = 1/‖(a,b,c)‖_original.

    Port of PlaneFitting::fit_plane().

    Parameters
    ----------
    points : (N, 3) array of 3D points.
    max_cond : condition number threshold.
    min_cp_dist : reject if ‖cp‖ < this.

    Returns
    -------
    (success, abcd) where abcd = [nx, ny, nz, d] with n unit,
    and the plane equation is  n·p + d = 0.
    """
    n = points.shape[0]
    if n < 3:
        return False, np.zeros(4)

    A = points  # (N, 3)
    b = -np.ones(n)

    # Condition number check
    try:
        sv = np.linalg.svd(A, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-15 else np.inf
        if cond > max_cond:
            return False, np.zeros(4)
    except np.linalg.LinAlgError:
        return False, np.zeros(4)

    # Solve via QR
    try:
        abc, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return False, np.zeros(4)

    abcd = np.array([abc[0], abc[1], abc[2], 1.0])
    norm = np.linalg.norm(abcd[:3])
    if norm < 1e-12:
        return False, np.zeros(4)

    abcd /= norm  # now abcd[:3] is unit normal, abcd[3] = 1/norm_original

    # Check CP distance (plane too close to origin is degenerate)
    cp = -abcd[:3] * abcd[3]
    if np.linalg.norm(cp) < min_cp_dist:
        return False, np.zeros(4)

    return True, abcd


def point_to_plane_distance(point: np.ndarray, abcd: np.ndarray) -> float:
    """Signed distance from a point to a plane.

    Plane equation: n·p + d = 0, so distance = n·p + d.
    """
    return np.dot(point, abcd[:3]) + abcd[3]


def abcd_to_cp(abcd: np.ndarray) -> np.ndarray:
    """Convert (n, d) plane parameterisation to closest-point vector.

    cp = -n * d, where n·p + d = 0.
    """
    return -abcd[:3] * abcd[3]


def cp_to_abcd(cp: np.ndarray) -> np.ndarray:
    """Convert closest-point vector to (n, d) plane parameterisation."""
    d = np.linalg.norm(cp)
    if d < 1e-15:
        return np.array([0.0, 0.0, 1.0, 0.0])
    n = cp / d
    return np.array([n[0], n[1], n[2], -d])


# ---------------------------------------------------------------------------
# 2. RANSAC plane fitting
# ---------------------------------------------------------------------------

def fit_plane_ransac(
    feat_ids: list[int],
    points: np.ndarray,
    settings: Optional[PlaneFittingSettings] = None,
) -> tuple[bool, np.ndarray, list[int]]:
    """RANSAC plane fitting on 3D feature points.

    Port of PlaneFitting::plane_fitting().

    Parameters
    ----------
    feat_ids : list of feature IDs corresponding to rows of points.
    points : (N, 3) array of 3D positions in global frame.
    settings : fitting configuration.

    Returns
    -------
    (success, cp_inG, inlier_feat_ids)
    """
    if settings is None:
        settings = PlaneFittingSettings()

    n = len(feat_ids)
    if n < max(3, settings.ransac_n_points):
        return False, np.zeros(3), []

    pts = np.asarray(points, dtype=np.float64)
    min_inliers = max(
        settings.ransac_n_points,
        int(n * settings.ransac_min_inlier_ratio),
    )
    rng = np.random.RandomState(settings.ransac_seed)

    best_inlier_idx: list[int] = []
    best_error = np.inf
    best_abcd = np.zeros(4)

    for _ in range(settings.ransac_max_iters):
        # Draw sample with minimum separation
        sample_idx = _draw_separated_sample(
            pts, settings.ransac_n_points,
            settings.ransac_min_point_separation, rng,
        )
        if sample_idx is None:
            continue

        # Fit plane to sample
        ok, abcd = fit_plane_linear(
            pts[sample_idx], settings.max_condition_number,
        )
        if not ok:
            continue

        # Count inliers
        dists = np.abs(pts @ abcd[:3] + abcd[3])
        inlier_mask = dists < settings.ransac_inlier_threshold
        inlier_count = int(np.sum(inlier_mask))
        inlier_error = np.mean(dists[inlier_mask]) if inlier_count > 0 else np.inf

        if inlier_count < min_inliers:
            continue
        if inlier_error >= settings.ransac_inlier_threshold:
            continue

        # Keep if better (more inliers, or same count but lower error)
        better = (
            inlier_count > len(best_inlier_idx)
            or (inlier_count == len(best_inlier_idx) and inlier_error < best_error)
        )
        if better:
            best_inlier_idx = list(np.where(inlier_mask)[0])
            best_error = inlier_error
            best_abcd = abcd.copy()

    if not best_inlier_idx:
        return False, np.zeros(3), []

    # Refit on full inlier set
    ok, abcd = fit_plane_linear(
        pts[best_inlier_idx],
        settings.max_condition_number,
        settings.min_cp_distance,
    )
    if not ok:
        abcd = best_abcd

    cp_inG = abcd_to_cp(abcd)
    inlier_fids = [feat_ids[i] for i in best_inlier_idx]

    return True, cp_inG, inlier_fids


def _draw_separated_sample(
    points: np.ndarray,
    n_sample: int,
    min_sep: float,
    rng: np.random.RandomState,
) -> Optional[list[int]]:
    """Draw n_sample indices with minimum pairwise 3D separation."""
    n = len(points)
    order = rng.permutation(n)
    sample: list[int] = []

    for idx in order:
        if len(sample) >= n_sample:
            break
        p = points[idx]
        if not sample:
            sample.append(idx)
        else:
            # Check distance to all already-sampled points
            dists = np.linalg.norm(points[sample] - p, axis=1)
            if np.all(dists >= min_sep):
                sample.append(idx)

    return sample if len(sample) == n_sample else None


# ---------------------------------------------------------------------------
# 3. Joint optimization (replaces Ceres)
# ---------------------------------------------------------------------------

def optimize_plane(
    cp_init: np.ndarray,
    feat_ids: list[int],
    feat_positions: dict[int, np.ndarray],
    feat_observations: Optional[dict[int, list[tuple[np.ndarray, np.ndarray]]]] = None,
    fix_plane: bool = False,
    settings: Optional[PlaneFittingSettings] = None,
) -> tuple[bool, np.ndarray, dict[int, np.ndarray], list[int]]:
    """Jointly optimize plane CP and feature positions.

    Port of PlaneFitting::optimize_plane(), using scipy.optimize.least_squares
    instead of Ceres.

    Parameters
    ----------
    cp_init : (3,) initial closest-point estimate in global frame.
    feat_ids : feature IDs to optimize.
    feat_positions : {fid: (3,)} initial 3D positions in global frame.
    feat_observations : {fid: [(R_GtoC_i, p_CinG_i, uv_norm_i), ...]}
        Per-feature list of camera observations. Each entry is a tuple of
        (R_GtoC, p_CinG, uv_normalised). If None, only point-on-plane
        constraints are used (no reprojection).
    fix_plane : if True, only refine feature positions (plane held fixed).
    settings : fitting configuration.

    Returns
    -------
    (success, cp_refined, refined_positions, inlier_fids)
    """
    if settings is None:
        settings = PlaneFittingSettings()

    n_feats = len(feat_ids)
    if n_feats == 0:
        return False, cp_init, {}, []

    sigma_c = settings.opt_sigma_c
    sigma_px = settings.opt_sigma_px_norm

    # --- Pack variables into a single vector ---
    # Layout: [cp(3), feat0(3), feat1(3), ...]
    # If fix_plane, cp is not in the variable vector.
    x0_parts = []
    if not fix_plane:
        x0_parts.append(cp_init.copy())
    for fid in feat_ids:
        x0_parts.append(feat_positions[fid].copy())
    x0 = np.concatenate(x0_parts)

    cp_offset = 0 if not fix_plane else -3  # offset helper
    feat_offset = 3 if not fix_plane else 0

    def _get_cp(x):
        if fix_plane:
            return cp_init
        return x[0:3]

    def _get_feat(x, i):
        start = feat_offset + 3 * i
        return x[start:start + 3]

    # --- Build residual function ---
    def residual_fn(x):
        cp = _get_cp(x)
        d = np.linalg.norm(cp)
        if d < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
            d = 1e-12
        else:
            n = cp / d

        residuals = []

        for i, fid in enumerate(feat_ids):
            p = _get_feat(x, i)

            # Point-on-plane constraint: (n·p - d) / sigma_c
            constraint = (np.dot(n, p) - d) / sigma_c
            residuals.append(constraint)

            # Reprojection residuals (if observations available)
            if feat_observations is not None and fid in feat_observations:
                for R_GtoC, p_CinG, uv_meas in feat_observations[fid]:
                    # Project feature into camera
                    p_inC = R_GtoC @ (p - p_CinG)
                    if p_inC[2] < 0.01:
                        # Behind camera — large residual to push it away
                        residuals.extend([10.0, 10.0])
                        continue
                    uv_pred = p_inC[:2] / p_inC[2]
                    reproj = (uv_meas - uv_pred) / sigma_px
                    residuals.extend(reproj.tolist())

        return np.array(residuals)

    # --- Solve ---
    try:
        result = least_squares(
            residual_fn, x0,
            loss='cauchy',
            f_scale=settings.opt_cauchy_scale,
            max_nfev=settings.opt_max_iters * len(x0),
            method='trf',
        )
    except Exception:
        return False, cp_init, {}, []

    if not result.success and result.status < 0:
        return False, cp_init, {}, []

    # --- Unpack results ---
    cp_out = _get_cp(result.x)
    abcd = cp_to_abcd(cp_out)

    refined_positions: dict[int, np.ndarray] = {}
    inlier_fids: list[int] = []

    for i, fid in enumerate(feat_ids):
        p = _get_feat(result.x, i)
        if np.any(np.isnan(p)):
            continue
        dist = abs(point_to_plane_distance(p, abcd))
        if dist < settings.opt_max_inlier_distance:
            refined_positions[fid] = p
            inlier_fids.append(fid)

    # Check inlier ratio
    min_inliers = max(3, int(n_feats * settings.opt_min_inlier_ratio))
    if fix_plane:
        min_inliers = 1
    if len(inlier_fids) < min_inliers:
        return False, cp_init, {}, []

    return True, cp_out, refined_positions, inlier_fids


# ---------------------------------------------------------------------------
# 4. Convenience: fit all planes from detector output
# ---------------------------------------------------------------------------

def fit_detected_planes(
    feat2plane: dict[int, int],
    feat_positions: dict[int, np.ndarray],
    settings: Optional[PlaneFittingSettings] = None,
    min_features: int = 5,
) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    """Run RANSAC fitting on each plane cluster from the detector.

    Parameters
    ----------
    feat2plane : {feat_id: plane_id} from PlaneDetector.
    feat_positions : {feat_id: (3,)} global-frame positions.
    settings : fitting settings.
    min_features : skip planes with fewer features.

    Returns
    -------
    plane_cps : {plane_id: cp_inG} for successfully fitted planes.
    plane_inliers : {plane_id: [feat_ids]} RANSAC inlier feature IDs per plane.
    """
    if settings is None:
        settings = PlaneFittingSettings()

    # Group features by plane
    plane_feats: dict[int, list[int]] = {}
    for fid, pid in feat2plane.items():
        if fid in feat_positions:
            plane_feats.setdefault(pid, []).append(fid)

    plane_cps: dict[int, np.ndarray] = {}
    plane_inliers: dict[int, list[int]] = {}

    for pid, fids in plane_feats.items():
        if len(fids) < min_features:
            continue

        points = np.array([feat_positions[fid] for fid in fids])
        ok, cp, inlier_fids = fit_plane_ransac(fids, points, settings)
        if ok:
            plane_cps[pid] = cp
            plane_inliers[pid] = inlier_fids

    return plane_cps, plane_inliers
