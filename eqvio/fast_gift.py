"""Runtime speedups for the Python GIFT point tracker.

The upstream tracker is intentionally straightforward, but two distance-pruning
passes and RANSAC inlier scoring become expensive with sparse-vog-sized pools.
These replacements keep the same public behavior while avoiding per-pair NumPy
calls in tight Python loops.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import numpy as np


def _cell_key(pt: np.ndarray, cell_size: float) -> tuple[int, int]:
    return (int(np.floor(float(pt[0]) / cell_size)), int(np.floor(float(pt[1]) / cell_size)))


def _nearby_cells(key: tuple[int, int]):
    x, y = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (x + dx, y + dy)


def _fast_remove_duplicates(self, proposed: list[np.ndarray]) -> list[np.ndarray]:
    dist = float(self.settings.feature_dist)
    if dist <= 0.0 or not proposed or not self._features:
        return proposed

    dist2 = dist * dist
    grid: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for f in self._features:
        pt = f.cam_coordinates
        grid[_cell_key(pt, dist)].append(pt)

    new_feats = []
    for p in proposed:
        too_close = False
        for cell in _nearby_cells(_cell_key(p, dist)):
            for q in grid.get(cell, ()):
                dx = float(p[0] - q[0])
                dy = float(p[1] - q[1])
                if dx * dx + dy * dy < dist2:
                    too_close = True
                    break
            if too_close:
                break
        if not too_close:
            new_feats.append(p)
    return new_feats


def _fast_remove_features_too_close(self, close_dist: float) -> None:
    if close_dist <= 0.0 or len(self._features) < 2:
        return

    close_dist = float(close_dist)
    close_dist2 = close_dist * close_dist
    grid: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = defaultdict(list)
    remove = [False] * len(self._features)

    for i, fi in enumerate(self._features):
        pt = fi.cam_coordinates
        lifetime = fi.lifetime
        for cell in _nearby_cells(_cell_key(pt, close_dist)):
            for q, q_lifetime in grid.get(cell, ()):
                if lifetime > q_lifetime:
                    continue
                dx = float(pt[0] - q[0])
                dy = float(pt[1] - q[1])
                if dx * dx + dy * dy < close_dist2:
                    remove[i] = True
                    break
            if remove[i]:
                break
        grid[_cell_key(pt, close_dist)].append((pt, lifetime))

    if any(remove):
        self._features = [f for f, should_remove in zip(self._features, remove) if not should_remove]


def _sample_indices(n_items: int, n_sample: int, rng: random.Random) -> list[int]:
    n_sample = min(n_items, n_sample)
    sample = list(range(n_sample))
    for i in range(n_sample, n_items):
        j = rng.randint(0, i)
        if j < n_sample:
            sample[j] = i
    return sample


def _fit_essential_matrix_arrays(p1_xy: np.ndarray, p2_xy: np.ndarray, indices: list[int]) -> np.ndarray:
    p1 = p1_xy[indices]
    p2 = p2_xy[indices]
    A = np.column_stack(
        (
            p1[:, 0] * p2[:, 0],
            p1[:, 0] * p2[:, 1],
            p1[:, 0],
            p1[:, 1] * p2[:, 0],
            p1[:, 1] * p2[:, 1],
            p1[:, 1],
            p2[:, 0],
            p2[:, 1],
            np.ones(len(indices)),
        )
    )

    _, _, vt = np.linalg.svd(A, full_matrices=True)
    f_mat = vt[-1, :].reshape(3, 3)

    u, s, vt2 = np.linalg.svd(f_mat)
    avg_s = 0.5 * (s[0] + s[1])
    return u @ np.diag([avg_s, avg_s, 0.0]) @ vt2


def _fast_determine_static_world_inliers(features, params, rng: Optional[random.Random] = None):
    if rng is None:
        rng = random.Random(0)

    n_features = len(features)
    if (
        n_features < params.min_inliers
        or n_features < params.min_data_points
        or params.max_iterations == 0
    ):
        return features

    p2_xy = np.array([f.cam_coordinates_norm() for f in features], dtype=np.float64)
    flow_xy = np.array([f.optical_flow_norm for f in features], dtype=np.float64)
    p1_xy = p2_xy - flow_xy

    p2_h = np.column_stack((p2_xy, np.ones(n_features)))
    p1_h = np.column_stack((p1_xy, np.ones(n_features)))
    best_mask = None
    best_count = 0

    for _ in range(params.max_iterations):
        indices = _sample_indices(n_features, params.min_data_points, rng)
        e_mat = _fit_essential_matrix_arrays(p1_xy, p2_xy, indices)
        errors = np.einsum("ij,jk,ik->i", p1_h, e_mat, p2_h)
        mask = errors < params.inlier_threshold
        count = int(np.count_nonzero(mask))
        if count > best_count and count > params.min_inliers:
            best_mask = mask
            best_count = count

    if best_mask is None or best_count < params.min_inliers:
        return features
    return [f for f, keep in zip(features, best_mask) if keep]


def install_fast_point_tracker() -> None:
    """Patch GIFT's point tracker with vectorized sparse-pool hot paths."""
    import gift.tracker as tracker_module

    tracker_module.PointFeatureTracker._remove_duplicates = _fast_remove_duplicates
    tracker_module.PointFeatureTracker._remove_features_too_close = _fast_remove_features_too_close
    tracker_module.determine_static_world_inliers = _fast_determine_static_world_inliers
