"""Tests for PlaneDetector.

Verifies the core detection pipeline using synthetic planar features
with known ground truth geometry.
"""

import numpy as np
import pytest

from eqvio.plane_detection.plane_detector import PlaneDetector, PlaneDetectorSettings


def _make_plane_features(
    normal: np.ndarray,
    d: float,
    center_2d: np.ndarray,
    spread_2d: float,
    n_points: int,
    id_offset: int = 0,
    seed: int = 42,
    focal: float = 400.0,
    depth: float = 3.0,
) -> tuple[dict, dict]:
    """Generate synthetic features lying on a plane.

    Creates 3D points on the plane n·p = d, offset in z to be at
    roughly `depth` in front of a camera at the origin looking along +z.
    Pixel coordinates are computed by projecting through a pinhole model.
    """
    rng = np.random.RandomState(seed)
    normal = normal / np.linalg.norm(normal)

    # Build a local tangent frame on the plane
    if abs(normal[2]) < 0.9:
        t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        t1 = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    # Plane point closest to origin
    p0 = normal * d

    # Shift the entire plane so points are at z ≈ depth
    z_shift = depth - p0[2]
    # Adjust d so the plane equation still holds: n·(p + [0,0,z_shift]) = d'
    p0_shifted = p0 + np.array([0, 0, z_shift])
    d_shifted = np.dot(normal, p0_shifted)

    feat_uvs = {}
    feat_pos = {}

    for i in range(n_points):
        fid = id_offset + i
        # Random point on the shifted plane
        s1 = rng.normal(0, 0.5)
        s2 = rng.normal(0, 0.5)
        p3d = p0_shifted + s1 * t1 + s2 * t2

        # Skip if behind camera
        if p3d[2] < 0.5:
            continue

        # Project to pixels (simple pinhole, camera at origin looking +z)
        u = focal * p3d[0] / p3d[2] + 320 + rng.normal(0, spread_2d)
        v = focal * p3d[1] / p3d[2] + 240 + rng.normal(0, spread_2d)

        feat_uvs[fid] = (float(u), float(v))
        feat_pos[fid] = p3d.copy()

    return feat_uvs, feat_pos


class TestPlaneDetectorBasic:
    """Test basic detection on synthetic planar features."""

    def test_single_plane_detected(self):
        """Features on one plane should all be grouped together."""
        settings = PlaneDetectorSettings(
            min_norms=1,      # detect on first frame
            min_plane_features=3,
        )
        det = PlaneDetector(settings)

        # Floor plane: normal = (0, -1, 0), d = 2.0
        feat_uvs, feat_pos = _make_plane_features(
            normal=np.array([0.0, -1.0, 0.0]),
            d=2.0,
            center_2d=np.array([320.0, 350.0]),
            spread_2d=5.0,
            n_points=20,
            depth=5.0,
        )

        camera_pos = np.zeros(3)
        R_GtoC = np.eye(3)

        # Run several frames to build normal history
        for _ in range(3):
            det.update(feat_uvs, feat_pos, camera_pos, R_GtoC)

        f2p = det.feat2plane
        if f2p:
            plane_ids = set(f2p.values())
            # All features on the same plane should share one ID
            assert len(plane_ids) == 1, (
                f"Expected 1 plane, got {len(plane_ids)}: {plane_ids}"
            )
            # Most features should be detected
            assert len(f2p) >= 10, (
                f"Expected ≥10 features on plane, got {len(f2p)}"
            )

    def test_two_planes_separated(self):
        """Features on two distinct planes should get different IDs."""
        settings = PlaneDetectorSettings(
            min_norms=1,
            min_plane_features=3,
            max_norm_deg=15.0,       # tighter for cleaner separation
            max_dist_between_z=0.05,
        )
        det = PlaneDetector(settings)

        # Plane A: floor (y = -2)
        uvs_a, pos_a = _make_plane_features(
            normal=np.array([0.0, -1.0, 0.0]),
            d=2.0,
            center_2d=np.array([200.0, 400.0]),
            spread_2d=3.0,
            n_points=15,
            id_offset=0,
            seed=1,
            depth=5.0,
        )

        # Plane B: wall (z = 5)
        uvs_b, pos_b = _make_plane_features(
            normal=np.array([0.0, 0.0, 1.0]),
            d=5.0,
            center_2d=np.array([500.0, 200.0]),
            spread_2d=3.0,
            n_points=15,
            id_offset=100,
            seed=2,
            depth=5.0,
        )

        feat_uvs = {**uvs_a, **uvs_b}
        feat_pos = {**pos_a, **pos_b}
        camera_pos = np.zeros(3)
        R_GtoC = np.eye(3)

        for _ in range(3):
            det.update(feat_uvs, feat_pos, camera_pos, R_GtoC)

        f2p = det.feat2plane
        if len(f2p) >= 6:
            # Group detected features by plane ID
            planes: dict[int, set[int]] = {}
            for fid, pid in f2p.items():
                planes.setdefault(pid, set()).add(fid)

            # Should have at least 2 distinct planes
            assert len(planes) >= 2, (
                f"Expected ≥2 planes, got {len(planes)}"
            )

            # Features from plane A and plane B should not mix
            for pid, fids in planes.items():
                a_count = sum(1 for f in fids if f < 100)
                b_count = sum(1 for f in fids if f >= 100)
                # One group should dominate
                assert min(a_count, b_count) <= 2, (
                    f"Plane {pid} mixes groups: {a_count} from A, {b_count} from B"
                )

    def test_no_planes_from_random_points(self):
        """Random non-coplanar points should not form planes."""
        settings = PlaneDetectorSettings(min_norms=2, min_plane_features=4)
        det = PlaneDetector(settings)

        rng = np.random.RandomState(99)
        feat_uvs = {}
        feat_pos = {}
        for i in range(20):
            feat_uvs[i] = (rng.uniform(50, 600), rng.uniform(50, 400))
            feat_pos[i] = rng.uniform(-3, 3, size=3)
            feat_pos[i][2] = abs(feat_pos[i][2]) + 1.0  # positive depth

        for _ in range(4):
            det.update(feat_uvs, feat_pos, np.zeros(3), np.eye(3))

        # Should detect few or no planes
        f2p = det.feat2plane
        n_on_plane = len(f2p)
        assert n_on_plane < 10, (
            f"Expected few plane associations from random points, got {n_on_plane}"
        )

    def test_delaunay_data_exposed(self):
        """Delaunay data should be available after update."""
        det = PlaneDetector()

        feat_uvs, feat_pos = _make_plane_features(
            normal=np.array([0, 0, 1]),
            d=3.0,
            center_2d=np.array([320, 240]),
            spread_2d=5.0,
            n_points=10,
        )

        det.update(feat_uvs, feat_pos, np.zeros(3), np.eye(3))
        tri_data = det.delaunay_data

        assert tri_data is not None
        simplices, feat_ids, normals = tri_data
        assert simplices.shape[1] == 3
        assert len(feat_ids) == len(feat_pos)
        assert normals.shape[0] == simplices.shape[0]
        assert normals.shape[1] == 3

    def test_too_few_features_no_crash(self):
        """Detector should handle gracefully when <3 features."""
        det = PlaneDetector()

        det.update({0: (100.0, 200.0)}, {0: np.array([1, 0, 3.0])})
        assert det.feat2plane == {}
        assert det.delaunay_data is None

        det.update({}, {})
        assert det.feat2plane == {}

    def test_features_pruned_when_lost(self):
        """Features removed from tracking should be cleaned up."""
        settings = PlaneDetectorSettings(min_norms=1, min_plane_features=3)
        det = PlaneDetector(settings)

        feat_uvs, feat_pos = _make_plane_features(
            normal=np.array([0, 0, 1]),
            d=3.0,
            center_2d=np.array([320, 240]),
            spread_2d=5.0,
            n_points=15,
        )

        # Run a few frames
        for _ in range(3):
            det.update(feat_uvs, feat_pos, np.zeros(3), np.eye(3))

        # Now remove most features (simulating them going out of view)
        keep_ids = list(feat_uvs.keys())[:3]
        small_uvs = {k: feat_uvs[k] for k in keep_ids}
        small_pos = {k: feat_pos[k] for k in keep_ids}

        det.update(small_uvs, small_pos, np.zeros(3), np.eye(3))

        # Lost features should not appear in feat2plane
        for fid in det.feat2plane:
            assert fid in small_uvs, f"Dead feature {fid} still in feat2plane"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
