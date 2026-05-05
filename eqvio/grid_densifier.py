"""Sparse-vog grid densification for obstacle-avoidance depth queries.

This module is a stateless per-frame consumer of SparseVogiatzisFilter output.
It turns sparse tracked depths into a grid while preserving thin foreground
obstacles and avoiding strict z-buffer corruption from isolated close outliers.
Plane filling is intentionally stubbed for now; empty cells remain UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil, isfinite
from typing import Optional

import cv2
import numpy as np


class GridCellState(IntEnum):
    UNKNOWN = 0
    MAPPED = 1
    PLANE_FILLED = 2
    THIN_OBSTACLE = 3
    SUSPECT_CLOSE = 4
    REGION_FILLED = 5
    FAR_FILLED = 6


@dataclass
class GridDensifierSettings:
    stride: int = 16
    window_size: Optional[float] = None
    max_depth: float = 20.0
    z_abs_cluster: float = 0.25
    z_rel_cluster: float = 0.08
    mapped_min_support: int = 3
    thin_min_support: int = 2
    min_depth: float = 0.1
    region_grow_enabled: bool = True
    region_grow_max_steps: int = 2
    region_grow_max_depth_jump: float = 0.5
    region_grow_var_scale: float = 2.0
    region_grow_depth_step: float = 0.5
    region_grow_fill_far: bool = True


@dataclass
class _Candidate:
    fid: int
    uv: tuple[float, float]
    depth: float
    var: float
    cell: tuple[int, int]
    weight: float = 1.0


@dataclass
class _Cluster:
    candidates: list[_Candidate]

    @property
    def depth(self) -> float:
        weights = np.array([c.weight / max(c.var, 1e-12) for c in self.candidates])
        depths = np.array([c.depth for c in self.candidates])
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0:
            return float(np.median(depths))
        return float(np.sum(weights * depths) / w_sum)

    @property
    def var(self) -> float:
        weights = np.array([c.weight / max(c.var, 1e-12) for c in self.candidates])
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0:
            return float(np.median([c.var for c in self.candidates]))
        return float(1.0 / w_sum)

    @property
    def support(self) -> int:
        return len(self.candidates)


class GridDensifier:
    """Build a robust cell-depth grid from sparse Vogiatzis features."""

    def __init__(
        self,
        K: np.ndarray,
        image_size: tuple[int, int],
        settings: Optional[GridDensifierSettings] = None,
    ):
        self.K = K.astype(np.float64)
        self.image_size = image_size
        self.settings = settings or GridDensifierSettings()
        if self.settings.stride <= 0:
            raise ValueError("Grid stride must be positive")
        if self.settings.window_size is not None and self.settings.window_size <= 0:
            raise ValueError("Grid window_size must be positive")

        width, height = image_size
        self.grid_w = int(ceil(width / self.settings.stride))
        self.grid_h = int(ceil(height / self.settings.stride))

        self._depth = np.full((self.grid_h, self.grid_w), -1.0, dtype=np.float32)
        self._var = np.full((self.grid_h, self.grid_w), np.inf, dtype=np.float32)
        self._debug_depth = np.full((self.grid_h, self.grid_w), -1.0, dtype=np.float32)
        self._debug_var = np.full((self.grid_h, self.grid_w), np.inf, dtype=np.float32)
        self._state = np.full(
            (self.grid_h, self.grid_w),
            int(GridCellState.UNKNOWN),
            dtype=np.uint8,
        )
        self._unconverged_count = 0

    def update(self, sparse_vog, plane_detector=None) -> None:
        """Rebuild the grid from current sparse-vog features.

        ``plane_detector`` is accepted for API compatibility with the design
        sketch. Plane fill is intentionally not implemented in this first pass.
        """
        self._reset()
        candidates, preview_candidates = self._collect_candidates(sparse_vog)
        self._unconverged_count = len(preview_candidates)

        for row in range(self.grid_h):
            for col in range(self.grid_w):
                local = self._weighted_candidates_for_cell(candidates, row, col)
                if not local:
                    continue
                cluster = self._select_front_cluster(local)
                if cluster is None:
                    continue
                state = self._classify_cluster(cluster)
                if state is GridCellState.UNKNOWN:
                    continue
                self._set_cell(row, col, cluster.depth, cluster.var, state)

        if self.settings.region_grow_enabled:
            self._region_grow_empty_cells()

        self._build_debug_depth(preview_candidates)

    def dense_depth_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of (depth, variance, state) at cell resolution."""
        return self._depth.copy(), self._var.copy(), self._state.copy()

    def dense_debug_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (debug_depth, debug_var, state), including unconverged previews."""
        return self._debug_depth.copy(), self._debug_var.copy(), self._state.copy()

    @property
    def unconverged_count(self) -> int:
        return self._unconverged_count

    def min_depth_in_roi(self, u: float, v: float, w: float, h: float) -> tuple[float, bool]:
        """Return min depth in a pixel-space ROI and whether it is fully mapped."""
        if w <= 0 or h <= 0:
            return -1.0, False

        s = self.settings.stride
        c0 = max(0, int(np.floor(u / s)))
        r0 = max(0, int(np.floor(v / s)))
        c1 = min(self.grid_w, int(np.ceil((u + w) / s)))
        r1 = min(self.grid_h, int(np.ceil((v + h) / s)))
        if c0 >= c1 or r0 >= r1:
            return -1.0, False

        depths = self._depth[r0:r1, c0:c1]
        states = self._state[r0:r1, c0:c1]
        known = states != int(GridCellState.UNKNOWN)
        complete = bool(np.all(known))
        if not np.any(known):
            return -1.0, complete
        return float(np.min(depths[known])), complete

    def _reset(self) -> None:
        self._depth.fill(-1.0)
        self._var.fill(np.inf)
        self._debug_depth.fill(-1.0)
        self._debug_var.fill(np.inf)
        self._state.fill(int(GridCellState.UNKNOWN))
        self._unconverged_count = 0

    def _collect_candidates(self, sparse_vog) -> tuple[list[_Candidate], list[_Candidate]]:
        candidates: list[_Candidate] = []
        preview_candidates: list[_Candidate] = []
        for fid, uv in sparse_vog.feat_uvs.items():
            depth, var = sparse_vog.query(fid)
            if not self._valid_depth(depth, var):
                preview = self._preview_candidate(sparse_vog, fid, uv)
                if preview is not None:
                    preview_candidates.append(preview)
                continue
            cell = self._cell_for_uv(uv)
            if cell is None:
                continue
            candidates.append(
                _Candidate(
                    fid=int(fid),
                    uv=(float(uv[0]), float(uv[1])),
                    depth=float(depth),
                    var=float(var),
                    cell=cell,
                )
            )
        return candidates, preview_candidates

    def _preview_candidate(self, sparse_vog, fid: int, uv: tuple[float, float]) -> Optional[_Candidate]:
        feat = sparse_vog.features.get(fid)
        if feat is None:
            return None
        depth, var = self._preview_depth_var(sparse_vog, feat)
        if not self._valid_depth(depth, var):
            return None
        cell = self._cell_for_uv(uv)
        if cell is None:
            return None
        return _Candidate(
            fid=int(fid),
            uv=(float(uv[0]), float(uv[1])),
            depth=depth,
            var=var,
            cell=cell,
        )

    @staticmethod
    def _preview_depth_var(sparse_vog, feat) -> tuple[float, float]:
        if hasattr(sparse_vog, "_canonical_to_depth") and hasattr(feat, "canonical"):
            canonical = float(getattr(feat, "canonical"))
            canonical_var = float(getattr(feat, "canonical_var", 1.0))
            if not isfinite(canonical) or not isfinite(canonical_var):
                return -1.0, float("inf")
            depth = float(sparse_vog._canonical_to_depth(canonical))
            if hasattr(sparse_vog, "_canonical_var_to_euclidean"):
                var = float(
                    sparse_vog._canonical_var_to_euclidean(
                        canonical, canonical_var, depth
                    )
                )
            else:
                var = canonical_var
            return depth, var

        depth = float(getattr(feat, "depth", -1.0))
        var = float(getattr(feat, "depth_var", 1.0))
        return depth, var

    @property
    def _window_size(self) -> float:
        if self.settings.window_size is not None:
            return float(self.settings.window_size)
        return float(2 * self.settings.stride)

    def _valid_depth(self, depth: float, var: float) -> bool:
        if not isfinite(depth) or not isfinite(var):
            return False
        return self.settings.min_depth <= depth <= self.settings.max_depth and var >= 0.0

    def _cell_for_uv(self, uv: tuple[float, float]) -> Optional[tuple[int, int]]:
        u, v = float(uv[0]), float(uv[1])
        width, height = self.image_size
        if u < 0.0 or v < 0.0 or u >= width or v >= height:
            return None
        col = int(u // self.settings.stride)
        row = int(v // self.settings.stride)
        return row, col

    def _cell_center(self, row: int, col: int) -> tuple[float, float]:
        stride = self.settings.stride
        return (col + 0.5) * stride, (row + 0.5) * stride

    def _weighted_candidates_for_cell(
        self,
        candidates: list[_Candidate],
        row: int,
        col: int,
    ) -> list[_Candidate]:
        center_u, center_v = self._cell_center(row, col)
        radius = 0.5 * self._window_size
        weighted: list[_Candidate] = []

        for candidate in candidates:
            du = abs(candidate.uv[0] - center_u)
            dv = abs(candidate.uv[1] - center_v)
            if du > radius or dv > radius:
                continue
            wu = max(0.0, 1.0 - du / radius)
            wv = max(0.0, 1.0 - dv / radius)
            weight = wu * wv
            if weight <= 0.0:
                continue
            weighted.append(
                _Candidate(
                    fid=candidate.fid,
                    uv=candidate.uv,
                    depth=candidate.depth,
                    var=candidate.var,
                    cell=candidate.cell,
                    weight=weight,
                )
            )

        return weighted

    def _select_front_cluster(self, candidates: list[_Candidate]) -> Optional[_Cluster]:
        clusters = self._cluster_by_depth(candidates)
        if not clusters:
            return None

        for cluster in clusters:
            state = self._classify_cluster(cluster)
            if state is not GridCellState.UNKNOWN:
                return cluster
        return None

    def _cluster_by_depth(self, candidates: list[_Candidate]) -> list[_Cluster]:
        sorted_candidates = sorted(candidates, key=lambda c: c.depth)
        clusters: list[_Cluster] = []
        for candidate in sorted_candidates:
            if not clusters:
                clusters.append(_Cluster([candidate]))
                continue
            last = clusters[-1]
            threshold = self._depth_threshold(last.depth)
            if abs(candidate.depth - last.depth) <= threshold:
                last.candidates.append(candidate)
            else:
                clusters.append(_Cluster([candidate]))
        return clusters

    def _depth_threshold(self, depth: float) -> float:
        return max(self.settings.z_abs_cluster, self.settings.z_rel_cluster * depth)

    def _classify_cluster(self, cluster: _Cluster) -> GridCellState:
        if cluster.support >= self.settings.mapped_min_support:
            return GridCellState.MAPPED
        if (
            cluster.support >= self.settings.thin_min_support
            and self._has_adjacent_support(cluster.candidates)
        ):
            return GridCellState.THIN_OBSTACLE
        if cluster.support == 1:
            return GridCellState.SUSPECT_CLOSE
        return GridCellState.UNKNOWN

    def _has_adjacent_support(self, candidates: list[_Candidate]) -> bool:
        for i, a in enumerate(candidates):
            ar, ac = a.cell
            for b in candidates[i + 1:]:
                br, bc = b.cell
                if abs(ar - br) <= 1 and abs(ac - bc) <= 1:
                    return True
        return False

    def _set_cell(self, row: int, col: int, depth: float, var: float, state: GridCellState) -> None:
        self._depth[row, col] = depth
        self._var[row, col] = var
        self._state[row, col] = int(state)

    def _region_grow_empty_cells(self) -> None:
        max_steps = int(self.settings.region_grow_max_steps)
        if max_steps <= 0:
            return

        queue: list[tuple[int, int, int]] = []
        seed_states = {
            int(GridCellState.MAPPED),
            int(GridCellState.THIN_OBSTACLE),
            int(GridCellState.PLANE_FILLED),
        }
        for row in range(self.grid_h):
            for col in range(self.grid_w):
                if int(self._state[row, col]) in seed_states:
                    queue.append((row, col, 0))
        if not queue:
            return

        head = 0
        while head < len(queue):
            row, col, steps = queue[head]
            head += 1
            if steps >= max_steps:
                continue

            source_depth = float(self._depth[row, col])
            source_var = float(self._var[row, col])
            if not self._valid_depth(source_depth, source_var):
                continue

            for nr, nc in self._grid_neighbours4(row, col):
                if self._state[nr, nc] != int(GridCellState.UNKNOWN):
                    continue
                grow_steps = steps + 1
                filled_depth = min(
                    self.settings.max_depth,
                    source_depth + float(self.settings.region_grow_depth_step),
                )
                if not self._can_region_fill(nr, nc, filled_depth):
                    continue
                self._depth[nr, nc] = filled_depth
                self._var[nr, nc] = source_var * (
                    max(float(self.settings.region_grow_var_scale), 1.0) ** grow_steps
                )
                self._state[nr, nc] = int(GridCellState.REGION_FILLED)
                queue.append((nr, nc, grow_steps))

        if self.settings.region_grow_fill_far:
            self._fill_remaining_unknown_far()

    def _fill_remaining_unknown_far(self) -> None:
        unknown = self._state == int(GridCellState.UNKNOWN)
        self._depth[unknown] = self.settings.max_depth
        self._var[unknown] = float("inf")
        self._state[unknown] = int(GridCellState.FAR_FILLED)

    def _grid_neighbours4(self, row: int, col: int):
        if row > 0:
            yield row - 1, col
        if row + 1 < self.grid_h:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col + 1 < self.grid_w:
            yield row, col + 1

    def _can_region_fill(self, row: int, col: int, depth: float) -> bool:
        for nr, nc in self._grid_neighbours4(row, col):
            if self._state[nr, nc] == int(GridCellState.UNKNOWN):
                continue
            neighbour_depth = float(self._depth[nr, nc])
            neighbour_var = float(self._var[nr, nc])
            if not self._valid_depth(neighbour_depth, neighbour_var):
                continue
            threshold = max(
                float(self.settings.region_grow_max_depth_jump),
                self._depth_threshold(min(depth, neighbour_depth)),
            )
            if abs(depth - neighbour_depth) > threshold:
                return False
        return True

    def _build_debug_depth(self, preview_candidates: list[_Candidate]) -> None:
        self._debug_depth[:, :] = self._depth
        self._debug_var[:, :] = self._var

        for row in range(self.grid_h):
            for col in range(self.grid_w):
                if self._state[row, col] != int(GridCellState.UNKNOWN):
                    continue
                local = self._weighted_candidates_for_cell(preview_candidates, row, col)
                if not local:
                    continue
                cluster = self._select_front_preview_cluster(local)
                if cluster is None:
                    continue
                self._debug_depth[row, col] = cluster.depth
                self._debug_var[row, col] = cluster.var

    def _select_front_preview_cluster(self, candidates: list[_Candidate]) -> Optional[_Cluster]:
        clusters = self._cluster_by_depth(candidates)
        if not clusters:
            return None
        return clusters[0]


class GridDensifierDebugWindow:
    """Separate OpenCV window for the native sparse grid depth map."""

    WINDOW_NAME = "Sparse Grid Depth"

    def __init__(self, enabled: bool = True, cell_px: int = 12, wait_ms: int = 1):
        self.enabled = enabled
        self.cell_px = cell_px
        self.wait_ms = wait_ms
        self._window_created = False
        self._show_state_overlay = False

    def update(
        self,
        depth: np.ndarray,
        var: np.ndarray,
        state: np.ndarray,
        unconverged_count: int = 0,
    ) -> None:
        if not self.enabled:
            return

        img = self._render(depth, state, unconverged_count)
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self._window_created = True
        cv2.imshow(self.WINDOW_NAME, img)

        key = cv2.waitKey(self.wait_ms) & 0xFF
        if key == ord("q"):
            self.enabled = False
            self.close()
        elif key == ord("s"):
            self._show_state_overlay = not self._show_state_overlay

    def close(self) -> None:
        if self._window_created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._window_created = False

    def _render(
        self,
        depth: np.ndarray,
        state: np.ndarray,
        unconverged_count: int = 0,
    ) -> np.ndarray:
        known = np.isfinite(depth) & (depth > 0.0)
        scale = max(int(self.cell_px), 1)
        smooth_depth, smooth_known = self._smooth_depth_for_display(depth, known, scale)

        if np.any(smooth_known):
            valid_depths = smooth_depth[smooth_known]
            vmin = float(np.percentile(valid_depths, 2))
            vmax = float(np.percentile(valid_depths, 98))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, max(float(np.max(valid_depths)), 1.0)
        else:
            vmin, vmax = 0.0, 1.0

        norm = np.zeros_like(smooth_depth, dtype=np.uint8)
        if vmax > vmin:
            norm_f = 1.0 - np.clip((smooth_depth - vmin) / (vmax - vmin), 0.0, 1.0)
            norm = (norm_f * 255.0).astype(np.uint8)
        colour = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        colour[~smooth_known] = (35, 35, 35)

        if self._show_state_overlay:
            state_colours = {
                GridCellState.PLANE_FILLED: (220, 120, 40),
                GridCellState.THIN_OBSTACLE: (0, 220, 255),
                GridCellState.SUSPECT_CLOSE: (0, 80, 255),
                GridCellState.REGION_FILLED: (190, 130, 40),
                GridCellState.FAR_FILLED: (180, 80, 80),
            }
            for cell_state, bgr in state_colours.items():
                mask = self._resize_mask_nearest(state == int(cell_state), scale)
                colour[mask] = (
                    0.45 * colour[mask] + 0.55 * np.array(bgr, dtype=np.float32)
                ).astype(np.uint8)

        out = colour

        n_mapped = int(np.count_nonzero(state == int(GridCellState.MAPPED)))
        n_thin = int(np.count_nonzero(state == int(GridCellState.THIN_OBSTACLE)))
        n_suspect = int(np.count_nonzero(state == int(GridCellState.SUSPECT_CLOSE)))
        n_plane = int(np.count_nonzero(state == int(GridCellState.PLANE_FILLED)))
        n_region = int(np.count_nonzero(state == int(GridCellState.REGION_FILLED)))
        n_far = int(np.count_nonzero(state == int(GridCellState.FAR_FILLED)))
        state_hint = "state:on" if self._show_state_overlay else "state:off"
        stats = (
            f"M:{n_mapped} P:{n_plane} R:{n_region} F:{n_far} "
            f"T:{n_thin} ?:{n_suspect} "
            f"U:{unconverged_count}  z:{vmin:.1f}-{vmax:.1f}m  {state_hint}"
        )
        cv2.putText(
            out, stats, (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (245, 245, 245), 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, "'s' state overlay, 'q' close", (8, out.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (210, 210, 210), 1, cv2.LINE_AA,
        )
        return out

    @staticmethod
    def _smooth_depth_for_display(
        depth: np.ndarray,
        known: np.ndarray,
        scale: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        size = (depth.shape[1] * scale, depth.shape[0] * scale)
        known_f = known.astype(np.float32)
        weighted_depth = np.where(known, depth, 0.0).astype(np.float32)

        smooth_weight = cv2.resize(known_f, size, interpolation=cv2.INTER_LINEAR)
        smooth_sum = cv2.resize(weighted_depth, size, interpolation=cv2.INTER_LINEAR)
        smooth_known = smooth_weight > 1e-3

        smooth_depth = np.full(size[::-1], -1.0, dtype=np.float32)
        np.divide(
            smooth_sum,
            smooth_weight,
            out=smooth_depth,
            where=smooth_known,
        )
        return smooth_depth, smooth_known

    @staticmethod
    def _resize_mask_nearest(mask: np.ndarray, scale: int) -> np.ndarray:
        return cv2.resize(
            mask.astype(np.uint8),
            (mask.shape[1] * scale, mask.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
