"""
Plane detector for EqVIO-P.

Port of: ov_plane/src/track_plane/TrackPlane::perform_plane_detection_monocular

Given tracked features with 3D positions and pixel coordinates, detects
planar surfaces by:
    1. Delaunay triangulation in image space
    2. Surface normal estimation from 3D triangle cross-products
    3. Pairwise normal matching across Delaunay neighbours
    4. Spatial outlier filtering (z-test on neighbour distances)

The detector is stateful — it accumulates per-feature normal history
across frames for robust averaging, and maintains persistent plane IDs
with merging when planes are re-associated.

Usage:
    detector = PlaneDetector()

    # Each vision frame:
    detector.update(feat_uvs, feat_positions_global, camera_pos_global, camera_R_global_to_cam)

    # Read results:
    feat2plane = detector.feat2plane           # {feat_id: plane_id}
    tri_data   = detector.delaunay_data        # (simplices, feat_ids, normals) or None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class PlaneDetectorSettings:
    """Configuration for plane detection.

    Default values match ov_plane's TrackPlaneOptions.
    """
    # --- Delaunay filtering ---
    max_tri_side_px: float = 200.0
    """Max triangle edge length in pixels; longer edges are rejected."""

    # --- Normal history ---
    max_norm_count: int = 5
    """How many per-feature normals to keep for averaging."""

    max_norm_avg_var: float = 25.0
    """Max std-dev (degrees) of normal history to consider the average valid."""

    max_norm_avg_max: float = 25.0
    """Max single-sample deviation (degrees) from the mean normal."""

    # --- Pairwise matching ---
    max_norm_deg: float = 25.0
    """Max angle (degrees) between two features' average normals to match."""

    max_dist_between_z: float = 0.10
    """Max point-to-plane distance (metres) for two features to be co-planar."""

    max_pairwise_px: float = 100.0
    """Max pixel distance between features for pairwise comparison."""

    min_norms: int = 3
    """Min accumulated normal observations before a feature can be matched."""

    check_old_feats: bool = True
    """Whether to re-check features already assigned to a plane."""

    # --- Spatial filter ---
    filter_num_feat: int = 4
    """Number of nearest neighbours for the z-test spatial filter."""

    filter_z_thresh: float = 1.2
    """Z-test threshold to reject spatial outliers."""

    # --- Plane cleanup ---
    min_plane_features: int = 4
    """Minimum active features for a plane to survive cleanup."""


# ---------------------------------------------------------------------------
# Plane detector
# ---------------------------------------------------------------------------

class PlaneDetector:
    """Detects planar surfaces from tracked features with 3D positions.

    Call :meth:`update` once per vision frame. Results are available via
    :attr:`feat2plane` and :attr:`delaunay_data`.
    """

    def __init__(self, settings: Optional[PlaneDetectorSettings] = None):
        self.settings = settings or PlaneDetectorSettings()

        # Persistent state across frames
        self._feat_norms: dict[int, list[np.ndarray]] = {}
        """Per-feature accumulated surface normals in global frame."""

        self._feat2plane: dict[int, int] = {}
        """Current feature → plane mapping."""

        self._plane_merges: dict[int, set[int]] = {}
        """plane_id → set of old plane_ids merged into it."""

        self._next_plane_id: int = 0

        # Per-frame outputs (reset each update)
        self._tri_simplices: Optional[np.ndarray] = None
        self._tri_feat_ids: Optional[list[int]] = None
        self._tri_normals: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def feat2plane(self) -> dict[int, int]:
        """Feature-to-plane mapping. Only contains features on a detected plane."""
        return dict(self._feat2plane)

    @property
    def delaunay_data(self) -> Optional[tuple[np.ndarray, list[int], np.ndarray]]:
        """(simplices, feat_ids, normals_per_triangle) from the last frame,
        or None if Delaunay was not computed."""
        if self._tri_simplices is None:
            return None
        return (self._tri_simplices, self._tri_feat_ids, self._tri_normals)

    def update(
        self,
        feat_uvs: dict[int, tuple[float, float]],
        feat_positions: dict[int, np.ndarray],
        camera_pos: Optional[np.ndarray] = None,
        R_global_to_cam: Optional[np.ndarray] = None,
    ):
        """Run one frame of plane detection.

        Parameters
        ----------
        feat_uvs : {feat_id: (u, v)}
            Pixel coordinates of all tracked features.
        feat_positions : {feat_id: np.array([x,y,z])}
            3D positions in global frame for features with known depth
            (typically from EqF state landmarks).
        camera_pos : (3,) array, optional
            Camera position in global frame (for normal sign correction).
        R_global_to_cam : (3,3) array, optional
            Rotation from global to camera frame (for normal sign correction).
        """
        opts = self.settings

        # Reset per-frame data
        self._tri_simplices = None
        self._tri_feat_ids = None
        self._tri_normals = None

        # Only keep features that have both pixel and 3D positions
        common_ids = sorted(set(feat_uvs.keys()) & set(feat_positions.keys()))
        if len(common_ids) < 3:
            self._prune_dead_features(set(feat_uvs.keys()))
            return

        # Purge normal history for features no longer tracked
        self._prune_dead_features(set(feat_uvs.keys()))

        # Build ordered arrays for Delaunay
        n = len(common_ids)
        uvs = np.array([feat_uvs[fid] for fid in common_ids], dtype=np.float64)
        pos = np.array([feat_positions[fid] for fid in common_ids], dtype=np.float64)
        id_to_idx = {fid: i for i, fid in enumerate(common_ids)}

        # ----- 1. Delaunay triangulation in image space -----
        try:
            tri = Delaunay(uvs)
        except Exception:
            return

        simplices = tri.simplices  # (M, 3) indices into common_ids

        # ----- 2. Compute per-triangle normals -----
        # Build adjacency: for each feature, which other features are neighbours
        neighbours: dict[int, set[int]] = {fid: set() for fid in common_ids}
        tri_normals = []

        for simplex in simplices:
            i0, i1, i2 = simplex
            fid0, fid1, fid2 = common_ids[i0], common_ids[i1], common_ids[i2]

            # Record adjacency
            neighbours[fid0].update([fid1, fid2])
            neighbours[fid1].update([fid0, fid2])
            neighbours[fid2].update([fid0, fid1])

            # Check triangle edge lengths in pixels
            uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
            len01 = np.linalg.norm(uv0 - uv1)
            len12 = np.linalg.norm(uv1 - uv2)
            len20 = np.linalg.norm(uv2 - uv0)

            if max(len01, len12, len20) > opts.max_tri_side_px:
                tri_normals.append(np.zeros(3))
                continue

            # Cross product for surface normal in global frame
            p0, p1, p2 = pos[i0], pos[i1], pos[i2]
            d1 = p1 - p0
            d2 = p2 - p0
            n1 = np.linalg.norm(d1)
            n2 = np.linalg.norm(d2)
            if n1 < 1e-10 or n2 < 1e-10:
                tri_normals.append(np.zeros(3))
                continue

            normal = np.cross(d1 / n1, d2 / n2)
            nn = np.linalg.norm(normal)
            if nn < 1e-10:
                tri_normals.append(np.zeros(3))
                continue
            normal /= nn

            # Fix sign: normal should point toward the camera
            # In global frame: dot(R_GtoC @ normal, R_GtoC @ (p0 - cam_pos)) > 0
            # Simplifies to: dot(normal, p0 - cam_pos) > 0
            # (i.e. normal points from surface toward camera)
            if camera_pos is not None:
                view_dir = p0 - camera_pos
                if np.dot(normal, view_dir) < 0:
                    normal = -normal

            tri_normals.append(normal)

            # Accumulate normals for each vertex
            for fid in (fid0, fid1, fid2):
                if fid not in self._feat_norms:
                    self._feat_norms[fid] = []
                self._feat_norms[fid].append(normal.copy())
                # Trim to max history
                if len(self._feat_norms[fid]) > opts.max_norm_count:
                    self._feat_norms[fid] = self._feat_norms[fid][-opts.max_norm_count:]

        tri_normals = np.array(tri_normals)

        # Store for visualiser
        self._tri_simplices = simplices
        self._tri_feat_ids = common_ids
        self._tri_normals = tri_normals

        # ----- 3. Average normals per feature -----
        avg_norms: dict[int, np.ndarray] = {}
        for fid in common_ids:
            avg = self._average_normal(fid)
            if avg is not None:
                avg_norms[fid] = avg

        # ----- 4. Pairwise normal matching -----
        self._match_features(common_ids, avg_norms, neighbours, pos, uvs,
                             id_to_idx, feat_positions)

        # ----- 5. Spatial outlier filtering -----
        self._spatial_filter(common_ids, feat_positions)

        # ----- 6. Cleanup small / dead planes -----
        self._cleanup_planes(set(feat_uvs.keys()))

    # ------------------------------------------------------------------
    # Step 3: Average normal with variance check
    # ------------------------------------------------------------------

    def _average_normal(self, fid: int) -> Optional[np.ndarray]:
        """Compute average normal for a feature, rejecting high-variance ones.

        Port of TrackPlane::avg_norm().
        """
        opts = self.settings
        norms = self._feat_norms.get(fid, [])
        if len(norms) < opts.min_norms:
            return None

        # Mean direction (simple vector average + renormalise)
        valid = [n for n in norms if np.linalg.norm(n) > 1e-8]
        if len(valid) < 2:
            return None

        mean = np.sum(valid, axis=0)
        mn = np.linalg.norm(mean)
        if mn < 1e-8:
            return None
        mean /= mn

        # Check variance: angle of each sample from the mean
        angles_deg = []
        for n in valid:
            dot = np.clip(np.dot(n, mean), -1.0, 1.0)
            angles_deg.append(np.degrees(np.arccos(dot)))

        max_deg = max(angles_deg)
        var_deg = np.sqrt(np.sum((np.array(angles_deg)) ** 2) / (len(angles_deg) - 1))

        if var_deg > opts.max_norm_avg_var or max_deg > opts.max_norm_avg_max:
            return None

        return mean

    # ------------------------------------------------------------------
    # Step 4: Pairwise matching (core plane grouping)
    # ------------------------------------------------------------------

    def _match_features(
        self,
        common_ids: list[int],
        avg_norms: dict[int, np.ndarray],
        neighbours: dict[int, set[int]],
        pos_array: np.ndarray,
        uvs_array: np.ndarray,
        id_to_idx: dict[int, int],
        feat_positions: dict[int, np.ndarray],
    ):
        """Pairwise normal matching across Delaunay neighbours.

        Port of the matching section in perform_plane_detection_monocular().
        For each feature with a valid average normal, compare it against each
        Delaunay neighbour. If normals match and the two points are co-planar,
        assign them to the same plane (creating a new plane if needed, or
        merging existing planes).
        """
        opts = self.settings
        matched_this_frame: set[int] = set()

        for fid in common_ids:
            if fid not in avg_norms:
                continue
            norm = avg_norms[fid]

            # Skip already-matched if configured
            if not opts.check_old_feats and fid in self._feat2plane:
                continue

            # Plane distance of this feature: d = p · n
            p = feat_positions[fid]
            d = np.dot(p, norm)

            # Find matching neighbours
            matches: list[int] = []
            for fid2 in neighbours.get(fid, set()):
                if fid2 not in avg_norms:
                    continue
                norm2 = avg_norms[fid2]
                if fid2 in matched_this_frame:
                    continue

                # Pixel distance check
                idx1 = id_to_idx[fid]
                idx2 = id_to_idx[fid2]
                px_dist = np.linalg.norm(uvs_array[idx1] - uvs_array[idx2])
                if px_dist > opts.max_pairwise_px:
                    continue

                # Normal angle check
                dot = np.clip(np.dot(norm, norm2), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot))
                if np.isnan(angle_deg) or angle_deg > opts.max_norm_deg:
                    continue

                # Point-to-plane distance check
                p2 = feat_positions[fid2]
                plane_dist = np.dot(p2, norm) - d
                if abs(plane_dist) > opts.max_dist_between_z:
                    continue

                matches.append(fid2)

            if not matches:
                continue

            # Find the minimum plane ID among this feature and its matches
            min_pid = self._feat2plane.get(fid, None)
            for fid2 in matches:
                pid2 = self._feat2plane.get(fid2, None)
                if pid2 is not None:
                    if min_pid is None:
                        min_pid = pid2
                    else:
                        min_pid = min(min_pid, pid2)

            # If no existing plane, create a new one
            if min_pid is None:
                min_pid = self._next_plane_id
                self._next_plane_id += 1

            # Merge all involved planes into min_pid
            pids_to_merge: set[int] = set()
            if fid in self._feat2plane and self._feat2plane[fid] != min_pid:
                pids_to_merge.add(self._feat2plane[fid])
            for fid2 in matches:
                if fid2 in self._feat2plane and self._feat2plane[fid2] != min_pid:
                    pids_to_merge.add(self._feat2plane[fid2])

            for old_pid in pids_to_merge:
                self._merge_plane(old_pid, min_pid)

            # Assign all to min_pid
            self._feat2plane[fid] = min_pid
            for fid2 in matches:
                self._feat2plane[fid2] = min_pid
            matched_this_frame.add(fid)

    def _merge_plane(self, old_pid: int, new_pid: int):
        """Re-assign all features from old_pid to new_pid."""
        if old_pid == new_pid:
            return
        for fid in list(self._feat2plane.keys()):
            if self._feat2plane[fid] == old_pid:
                self._feat2plane[fid] = new_pid
        # Track merge history
        self._plane_merges.setdefault(new_pid, set()).add(old_pid)
        if old_pid in self._plane_merges:
            self._plane_merges[new_pid].update(self._plane_merges.pop(old_pid))

    # ------------------------------------------------------------------
    # Step 5: Spatial outlier filter
    # ------------------------------------------------------------------

    def _spatial_filter(
        self,
        common_ids: list[int],
        feat_positions: dict[int, np.ndarray],
    ):
        """Remove features whose nearest-neighbour distance is anomalous.

        Port of the ikd-tree z-test in perform_plane_detection_monocular().
        Uses brute-force kNN (fast enough for typical feature counts <200).
        """
        opts = self.settings
        k = opts.filter_num_feat

        # Group active features by plane
        plane_feats: dict[int, list[int]] = {}
        active_set = set(common_ids)
        for fid, pid in self._feat2plane.items():
            if fid in active_set and fid in feat_positions:
                plane_feats.setdefault(pid, []).append(fid)

        for pid, fids in plane_feats.items():
            if len(fids) <= k:
                continue

            # Build position array for this plane
            positions = np.array([feat_positions[fid] for fid in fids])

            # Pairwise distances
            # (n, n) matrix; diagonal is 0
            diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
            dists = np.linalg.norm(diff, axis=2)                  # (n, n)

            # For each point, average distance to k nearest neighbours
            avg_knn = np.zeros(len(fids))
            for i in range(len(fids)):
                sorted_d = np.sort(dists[i])  # ascending, [0] is self=0
                avg_knn[i] = np.mean(sorted_d[1:k + 1])  # skip self

            # Z-test
            mu = np.mean(avg_knn)
            sigma = np.std(avg_knn, ddof=1)
            if sigma < 1e-12:
                continue

            for i, fid in enumerate(fids):
                z = abs(avg_knn[i] - mu) / sigma
                if z > opts.filter_z_thresh:
                    self._feat2plane.pop(fid, None)

    # ------------------------------------------------------------------
    # Step 6: Cleanup
    # ------------------------------------------------------------------

    def _cleanup_planes(self, active_feat_ids: set[int]):
        """Remove planes with too few active features."""
        opts = self.settings

        # Only keep mappings for currently tracked features
        self._feat2plane = {
            fid: pid for fid, pid in self._feat2plane.items()
            if fid in active_feat_ids
        }

        # Count features per plane
        plane_counts: dict[int, int] = {}
        for pid in self._feat2plane.values():
            plane_counts[pid] = plane_counts.get(pid, 0) + 1

        # Remove features on planes that are too small
        small_planes = {pid for pid, ct in plane_counts.items()
                        if ct < opts.min_plane_features}
        if small_planes:
            self._feat2plane = {
                fid: pid for fid, pid in self._feat2plane.items()
                if pid not in small_planes
            }

        # Clean merge history for dead planes
        active_planes = set(self._feat2plane.values())
        dead_merges = [pid for pid in self._plane_merges if pid not in active_planes]
        for pid in dead_merges:
            del self._plane_merges[pid]

    def _prune_dead_features(self, active_feat_ids: set[int]):
        """Remove normal history and plane mappings for features no longer tracked."""
        dead = set(self._feat_norms.keys()) - active_feat_ids
        for fid in dead:
            del self._feat_norms[fid]
        dead2 = set(self._feat2plane.keys()) - active_feat_ids
        for fid in dead2:
            del self._feat2plane[fid]


# ---------------------------------------------------------------------------
# Helper: extract global-frame landmarks from VIOState
# ---------------------------------------------------------------------------

def landmarks_to_global(state) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Extract landmark positions in global frame from a VIOState.

    Parameters
    ----------
    state : VIOState
        Filter state estimate (landmarks in camera frame).

    Returns
    -------
    feat_positions : {feat_id: np.array([x,y,z])} in global frame
    camera_pos : (3,) camera position in global frame
    R_GtoC : (3,3) rotation from global to camera frame
    """
    # IMU pose: R_GtoI, p_IinG
    R_GtoI = state.sensor.pose.R.asMatrix()
    p_IinG = state.sensor.pose.x

    # Camera extrinsics: R_ItoC, p_IinC
    R_ItoC = state.sensor.camera_offset.R.asMatrix()
    p_IinC = state.sensor.camera_offset.x

    # Combined: global-to-camera
    R_GtoC = R_ItoC @ R_GtoI
    R_CtoG = R_GtoC.T
    p_CinG = p_IinG - R_CtoG @ p_IinC

    # Transform each landmark from camera frame to global frame
    feat_positions = {}
    for lm in state.camera_landmarks:
        feat_positions[lm.id] = R_CtoG @ lm.p + p_CinG

    return feat_positions, p_CinG, R_GtoC
