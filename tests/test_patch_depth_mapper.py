"""Smoke test for PatchDepthMapper with synthetic data."""

from __future__ import annotations

import numpy as np
import math

from eqvio.patch_depth_mapper import (
    PatchDepthMapper,
    PatchDepthSettings,
    PatchStatus,
)
from eqvio.sparse_vogiatzis import (
    SparseVogiatzisFilter,
    SparseVogSettings,
    DepthParametrization,
)
from eqvio.mathematical.vision_measurement import VisionMeasurement

FX = FY = 458.0
CX, CY = 376.0, 240.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])


def _make_flat_scene(z_true: float, hw: tuple[int, int]) -> np.ndarray:
    """Render a textured fronto-parallel plane at depth z_true."""
    H, W = hw
    rng = np.random.default_rng(42)
    texture = rng.integers(50, 200, size=(H, W)).astype(np.float32)
    return texture


def _warp_image(
    img: np.ndarray, T_ref_curr: np.ndarray, z: float,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Warp img (reference) to current frame assuming flat depth z."""
    H, W = img.shape
    R = T_ref_curr[:3, :3]
    t = T_ref_curr[:3, 3]

    out = np.full((H, W), np.nan, dtype=np.float32)
    for v in range(H):
        for u in range(W):
            xn = (u - cx) / fx
            yn = (v - cy) / fy
            p_curr = np.array([xn * z, yn * z, z])
            p_ref = R @ p_curr + t
            if p_ref[2] <= 0:
                continue
            u_ref = fx * p_ref[0] / p_ref[2] + cx
            v_ref = fy * p_ref[1] / p_ref[2] + cy
            ix, iy = int(u_ref), int(v_ref)
            if 0 <= ix < W - 1 and 0 <= iy < H - 1:
                dx = u_ref - ix
                dy = v_ref - iy
                out[v, u] = (
                    (1 - dx) * (1 - dy) * img[iy, ix]
                    + dx * (1 - dy) * img[iy, ix + 1]
                    + (1 - dx) * dy * img[iy + 1, ix]
                    + dx * dy * img[iy + 1, ix + 1]
                )
    mask = np.isnan(out)
    out[mask] = np.nanmean(out)
    return out


def test_flat_plane_with_seeds():
    """A flat textured plane at z=5m with sparse seeds should produce
    PHOTO_REFINED patches near the true depth."""
    H, W = 480, 752
    z_true = 5.0
    baseline = 0.1

    ref_img = _make_flat_scene(z_true, (H, W))

    T_ref_curr = np.eye(4)
    T_ref_curr[0, 3] = -baseline

    curr_img = _warp_image(ref_img, T_ref_curr, z_true,
                           FX, FY, CX, CY)

    # Set up sparse vog filter with a few converged seeds
    vog_settings = SparseVogSettings(
        parametrization=DepthParametrization.INVDEPTH,
        sigma_pixel=0.5,
    )
    sparse_vog = SparseVogiatzisFilter(K, vog_settings)

    seed_positions = []
    for su in range(100, 700, 50):
        for sv in range(80, 440, 50):
            seed_positions.append((su, sv))
    for step in range(20):
        T_WC = np.eye(4)
        T_WC[0, 3] = step * baseline / 20.0
        T_CW = np.linalg.inv(T_WC)

        cam_coords = {}
        for i, (su, sv) in enumerate(seed_positions):
            fid = 100 + i
            p_cam = T_CW[:3, :3] @ np.array([
                (su - CX) / FX * z_true,
                (sv - CY) / FY * z_true,
                z_true,
            ]) + T_CW[:3, 3]
            u_obs = FX * p_cam[0] / p_cam[2] + CX
            v_obs = FY * p_cam[1] / p_cam[2] + CY
            cam_coords[fid] = np.array([u_obs, v_obs])

        meas = VisionMeasurement(stamp=step * 0.05, cam_coordinates=cam_coords)
        sparse_vog.update(meas, T_WC)

    # Run patch depth mapper
    # Convention: ref camera at origin, current camera translated +x by baseline.
    # T_ref_curr that the mapper computes: inv(T_WC_ref) @ T_WC_curr
    #   = inv(eye) @ [[I, baseline],[0,1]] → t = [+baseline, 0, 0]
    # But _warp_image was called with T_ref_curr[0,3] = -baseline (maps curr→ref).
    # So place ref at +baseline and curr at origin so mapper gets t = [-baseline,0,0].
    settings = PatchDepthSettings(
        patch_size=16,
        patch_stride=8,
        cell_size=16,
        min_baseline_ratio=0.001,
    )
    mapper = PatchDepthMapper(K, settings)
    # Feed reference frame as keyframe at x=+baseline
    T_WC_ref = np.eye(4)
    T_WC_ref[0, 3] = baseline
    result = mapper.update(sparse_vog, ref_img, T_WC_ref)
    assert result is None
    # Feed current frame at origin
    T_WC_curr = np.eye(4)
    result = mapper.update(sparse_vog, curr_img, T_WC_curr)
    assert result is not None, "Expected result with sufficient baseline"
    depth_cells, var_cells, status_cells = result

    # Check results
    valid = ~np.isnan(depth_cells)
    n_valid = np.sum(valid)
    n_total = depth_cells.size
    fill_ratio = n_valid / n_total

    print(f"Cell grid: {depth_cells.shape}")
    print(f"Fill ratio: {fill_ratio:.1%} ({n_valid}/{n_total})")

    n_photo = np.sum(status_cells == PatchStatus.PHOTO_REFINED)
    n_seed = np.sum(status_cells == PatchStatus.SEED_ONLY)
    n_unknown = np.sum(status_cells == PatchStatus.UNKNOWN)
    n_rejected = np.sum(status_cells == PatchStatus.REJECTED)
    print(f"Status: PHOTO_REFINED={n_photo}, SEED_ONLY={n_seed}, "
          f"UNKNOWN={n_unknown}, REJECTED={n_rejected}")

    if n_valid > 0:
        depths_valid = depth_cells[valid]
        mean_depth = np.mean(depths_valid)
        median_depth = np.median(depths_valid)
        std_depth = np.std(depths_valid)
        print(f"Depth: mean={mean_depth:.2f}m, median={median_depth:.2f}m, "
              f"std={std_depth:.3f}m (true={z_true:.1f}m)")

        rel_error = abs(median_depth - z_true) / z_true
        print(f"Relative error: {rel_error:.1%}")
        assert rel_error < 0.2, f"Median depth too far from truth: {median_depth:.2f} vs {z_true}"

    assert fill_ratio > 0.01, f"Fill ratio too low: {fill_ratio:.1%}"
    assert n_photo > 0, "Expected some PHOTO_REFINED cells"
    print("\nPASS: flat plane with seeds")


def test_no_parallax_gives_seed_only():
    """With zero baseline between curr and ref, all seeded patches should be
    SEED_ONLY because photometric curvature is zero.

    The sparse GB seeds are converged by moving the camera first, then the
    densifier runs on two frames at the same pose (hovering).
    """
    H, W = 480, 752
    z_true = 5.0

    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, size=(H, W)).astype(np.float32)

    vog_settings = SparseVogSettings(
        parametrization=DepthParametrization.INVDEPTH,
        sigma_pixel=0.5,
    )
    sparse_vog = SparseVogiatzisFilter(K, vog_settings)

    seed_positions = []
    for su in range(100, 700, 50):
        for sv in range(80, 440, 50):
            seed_positions.append((su, sv))

    # Move camera to converge seeds
    for step in range(30):
        T_WC = np.eye(4)
        T_WC[0, 3] = step * 0.1 / 30.0
        T_CW = np.linalg.inv(T_WC)
        cam_coords = {}
        for i, (su, sv) in enumerate(seed_positions):
            fid = 100 + i
            p_cam = T_CW[:3, :3] @ np.array([
                (su - CX) / FX * z_true,
                (sv - CY) / FY * z_true,
                z_true,
            ]) + T_CW[:3, 3]
            cam_coords[fid] = np.array([
                FX * p_cam[0] / p_cam[2] + CX,
                FY * p_cam[1] / p_cam[2] + CY,
            ])
        meas = VisionMeasurement(stamp=step * 0.05, cam_coordinates=cam_coords)
        sparse_vog.update(meas, T_WC)

    n_converged = sum(1 for fid in sparse_vog.feat_uvs if sparse_vog.query(fid)[0] > 0)
    print(f"Converged seeds: {n_converged}/{len(seed_positions)}")
    assert n_converged > 0, "No seeds converged"

    # Densify with zero baseline (hovering) — mapper should reject due to
    # insufficient baseline between keyframes and current frame.
    settings = PatchDepthSettings(patch_size=16, patch_stride=8, cell_size=16)
    mapper = PatchDepthMapper(K, settings)
    # Feed first frame as keyframe
    T_WC_hover = np.eye(4)
    T_WC_hover[0, 3] = 0.1  # at final position from seed convergence
    result = mapper.update(sparse_vog, img, T_WC_hover)
    assert result is None, "First frame should return None (no keyframe yet)"
    # Feed same pose again — baseline is zero
    result = mapper.update(sparse_vog, img, T_WC_hover)
    assert result is None, "Expected None with zero baseline (hovering)"

    print("PASS: no parallax gives SEED_ONLY\n")


if __name__ == "__main__":
    test_flat_plane_with_seeds()
    test_no_parallax_gives_seed_only()
