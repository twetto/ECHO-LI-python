"""Tests for plane fitting module.

Validates linear fitting, RANSAC, joint optimization, and the
convenience wrapper using synthetic planar features.
"""

import numpy as np
import pytest

from eqvio.plane_detection.plane_fitting import (
    fit_plane_linear,
    fit_plane_ransac,
    optimize_plane,
    fit_detected_planes,
    point_to_plane_distance,
    abcd_to_cp,
    cp_to_abcd,
    PlaneFittingSettings,
)


def _make_plane_points(
    normal: np.ndarray,
    d: float,
    n_points: int,
    spread: float = 0.5,
    seed: int = 42,
    z_offset: float = 3.0,
) -> np.ndarray:
    """Generate 3D points on a plane n·p = d, shifted to z ≈ z_offset."""
    rng = np.random.RandomState(seed)
    normal = normal / np.linalg.norm(normal)

    if abs(normal[2]) < 0.9:
        t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    p0 = normal * d
    z_shift = z_offset - p0[2]
    p0 = p0 + np.array([0, 0, z_shift])

    points = np.zeros((n_points, 3))
    for i in range(n_points):
        s1, s2 = rng.normal(0, spread), rng.normal(0, spread)
        points[i] = p0 + s1 * t1 + s2 * t2

    return points


# ---------------------------------------------------------------------------
# CP ↔ abcd round-trip
# ---------------------------------------------------------------------------

class TestConversions:

    def test_cp_abcd_roundtrip(self):
        cp = np.array([1.0, 2.0, 3.0])
        abcd = cp_to_abcd(cp)
        cp2 = abcd_to_cp(abcd)
        np.testing.assert_allclose(cp, cp2, atol=1e-12)

    def test_abcd_cp_roundtrip(self):
        abcd = np.array([0.0, 0.0, 1.0, -5.0])  # z = 5 plane
        cp = abcd_to_cp(abcd)
        assert np.allclose(cp, [0, 0, 5])
        abcd2 = cp_to_abcd(cp)
        np.testing.assert_allclose(abcd, abcd2, atol=1e-12)


# ---------------------------------------------------------------------------
# Linear fit
# ---------------------------------------------------------------------------

class TestLinearFit:

    def test_perfect_plane(self):
        """Points exactly on z=3 plane should fit perfectly."""
        pts = np.array([
            [0, 0, 3], [1, 0, 3], [0, 1, 3], [1, 1, 3], [-1, 0, 3],
        ], dtype=float)
        ok, abcd = fit_plane_linear(pts)
        assert ok
        # Normal should be (0, 0, ±1), d such that n·p + d = 0 on z=3
        assert abs(abs(abcd[2]) - 1.0) < 1e-6, f"normal z component: {abcd}"
        for p in pts:
            assert abs(point_to_plane_distance(p, abcd)) < 1e-10

    def test_tilted_plane(self):
        """Points on x + y + z = 6 plane."""
        n = np.array([1, 1, 1], dtype=float)
        n /= np.linalg.norm(n)
        d = 6.0
        pts = _make_plane_points(n, d / np.linalg.norm([1, 1, 1]),
                                 n_points=10, z_offset=2.0)
        ok, abcd = fit_plane_linear(pts)
        assert ok
        for p in pts:
            assert abs(point_to_plane_distance(p, abcd)) < 0.01

    def test_too_few_points(self):
        ok, _ = fit_plane_linear(np.array([[0, 0, 1], [1, 0, 1]], dtype=float))
        assert not ok

    def test_collinear_points_rejected(self):
        """Collinear points (degenerate) should fail condition check."""
        pts = np.array([[i, 0, 3.0] for i in range(5)], dtype=float)
        ok, _ = fit_plane_linear(pts, max_cond=10.0)
        assert not ok


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

class TestRANSAC:

    def test_clean_plane(self):
        """RANSAC on clean planar data should recover the plane."""
        pts = _make_plane_points(
            np.array([0, 0, 1.0]), d=5.0, n_points=30, seed=1, z_offset=5.0,
        )
        fids = list(range(30))
        ok, cp, inliers = fit_plane_ransac(fids, pts)
        assert ok
        assert len(inliers) >= 24  # ≥80% of 30
        # CP should point along z at ~5
        assert abs(np.linalg.norm(cp) - 5.0) < 0.5
        # Normal should be roughly +z
        n = cp / np.linalg.norm(cp)
        assert abs(abs(n[2]) - 1.0) < 0.1

    def test_with_outliers(self):
        """RANSAC should reject outlier points."""
        pts_good = _make_plane_points(
            np.array([0, 0, 1.0]), d=5.0, n_points=25, seed=1,
        )
        rng = np.random.RandomState(99)
        pts_bad = rng.uniform(-5, 5, (8, 3))
        pts_bad[:, 2] = rng.uniform(1, 10, 8)

        pts = np.vstack([pts_good, pts_bad])
        fids = list(range(len(pts)))

        ok, cp, inliers = fit_plane_ransac(fids, pts)
        assert ok
        # Inliers should be mostly from the good set (ids 0-24)
        good_inliers = [i for i in inliers if i < 25]
        bad_inliers = [i for i in inliers if i >= 25]
        assert len(good_inliers) >= 20
        assert len(bad_inliers) <= 3

    def test_too_few_points(self):
        fids = [0, 1]
        pts = np.array([[0, 0, 3], [1, 0, 3]], dtype=float)
        ok, _, _ = fit_plane_ransac(fids, pts)
        assert not ok

    def test_random_points_fail(self):
        """Non-coplanar random points should fail RANSAC."""
        rng = np.random.RandomState(77)
        pts = rng.uniform(-5, 5, (20, 3))
        pts[:, 2] = rng.uniform(1, 10, 20)
        fids = list(range(20))
        settings = PlaneFittingSettings(ransac_inlier_threshold=0.02)
        ok, _, inliers = fit_plane_ransac(fids, pts, settings)
        # Should either fail or have very few inliers
        assert not ok or len(inliers) < 10


# ---------------------------------------------------------------------------
# Joint optimization
# ---------------------------------------------------------------------------

class TestOptimize:

    def test_constraint_only(self):
        """Optimization with only point-on-plane constraints."""
        normal = np.array([0, 0, 1.0])
        pts = _make_plane_points(normal, d=5.0, n_points=10, seed=3, z_offset=5.0)

        # Add small noise to positions to give optimizer something to do
        rng = np.random.RandomState(10)
        noisy_pts = pts + rng.normal(0, 0.02, pts.shape)

        fids = list(range(len(pts)))
        feat_pos = {fid: noisy_pts[i] for i, fid in enumerate(fids)}
        cp_init = np.array([0.0, 0.0, 4.8])  # slightly off

        ok, cp_out, refined, inliers = optimize_plane(
            cp_init, fids, feat_pos,
            feat_observations=None,
            fix_plane=False,
        )
        assert ok
        # CP should converge toward (0, 0, 5)
        assert abs(np.linalg.norm(cp_out) - 5.0) < 0.3
        assert len(inliers) >= 8

    def test_fixed_plane(self):
        """With fix_plane=True, only features move."""
        normal = np.array([0, 0, 1.0])
        cp_true = np.array([0.0, 0.0, 5.0])

        rng = np.random.RandomState(5)
        pts = _make_plane_points(normal, d=5.0, n_points=8, seed=5, z_offset=5.0)
        noisy_pts = pts + rng.normal(0, 0.01, pts.shape)

        fids = list(range(len(pts)))
        feat_pos = {fid: noisy_pts[i] for i, fid in enumerate(fids)}

        ok, cp_out, refined, inliers = optimize_plane(
            cp_true.copy(), fids, feat_pos,
            fix_plane=True,
        )
        assert ok
        # CP should be unchanged
        np.testing.assert_allclose(cp_out, cp_true, atol=1e-10)
        assert len(inliers) >= 6


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class TestFitDetectedPlanes:

    def test_two_planes(self):
        """Fit two separate plane clusters."""
        pts_a = _make_plane_points(np.array([0, 0, 1.0]), d=5.0,
                                   n_points=15, seed=1, z_offset=5.0)
        pts_b = _make_plane_points(np.array([0, -1, 0.0]), d=2.0,
                                   n_points=15, seed=2, z_offset=4.0)

        feat2plane = {}
        feat_pos = {}
        for i in range(15):
            feat2plane[i] = 0
            feat_pos[i] = pts_a[i]
        for i in range(15):
            feat2plane[100 + i] = 1
            feat_pos[100 + i] = pts_b[i]

        cps, inliers = fit_detected_planes(feat2plane, feat_pos, min_features=5)
        assert 0 in cps
        assert 1 in cps

        # Plane 0 CP should be near (0, 0, 5)
        assert abs(np.linalg.norm(cps[0]) - 5.0) < 1.0

    def test_skip_small_cluster(self):
        """Planes with too few features are skipped."""
        feat2plane = {0: 0, 1: 0, 2: 0}  # only 3 features
        feat_pos = {i: np.array([i, 0, 3.0]) for i in range(3)}
        cps, inliers = fit_detected_planes(feat2plane, feat_pos, min_features=5)
        assert len(cps) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
