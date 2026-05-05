import numpy as np
from dataclasses import dataclass

from eqvio.grid_densifier import (
    GridCellState,
    GridDensifier,
    GridDensifierDebugWindow,
    GridDensifierSettings,
)


@dataclass
class FakeFeature:
    depth: float
    depth_var: float


class FakeSparseVog:
    def __init__(self, points):
        self.feat_uvs = {
            fid: (uv[0], uv[1])
            for fid, uv, _depth, _var in points
        }
        self._depths = {
            fid: (depth, var)
            for fid, _uv, depth, var in points
        }
        self.features = {
            fid: FakeFeature(depth, var)
            for fid, _uv, depth, var in points
        }

    def query(self, fid):
        return self._depths.get(fid, (-1.0, float("inf")))


@dataclass
class FakeCanonicalFeature:
    canonical: float
    canonical_var: float


class FakeCanonicalSparseVog:
    def __init__(self):
        self.feat_uvs = {1: (18.0, 18.0)}
        self.features = {1: FakeCanonicalFeature(0.5, 0.01)}

    def query(self, fid):
        return -1.0, float("inf")

    def _canonical_to_depth(self, canonical):
        return 1.0 / canonical

    def _canonical_var_to_euclidean(self, canonical, canonical_var, z):
        return (z ** 4) * canonical_var


def _make_densifier(stride=16, region_grow_enabled=True):
    return GridDensifier(
        np.eye(3),
        image_size=(64, 64),
        settings=GridDensifierSettings(
            stride=stride,
            window_size=stride * 2,
            z_abs_cluster=0.25,
            z_rel_cluster=0.05,
            mapped_min_support=3,
            thin_min_support=2,
            max_depth=20.0,
            region_grow_enabled=region_grow_enabled,
        ),
    )


def test_isolated_closer_point_is_suspect_not_mapped():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (18.0, 18.0), 1.0, 0.01),
        (2, (20.0, 20.0), 5.0, 0.01),
        (3, (22.0, 19.0), 5.1, 0.01),
        (4, (19.0, 22.0), 4.9, 0.01),
    ])

    densifier.update(sparse)
    depth, _var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.SUSPECT_CLOSE
    assert depth[1, 1] == np.float32(1.0)


def test_two_adjacent_close_points_survive_as_thin_obstacle():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (18.0, 18.0), 1.0, 0.01),
        (2, (20.0, 20.0), 1.08, 0.02),
        (3, (22.0, 19.0), 5.0, 0.01),
        (4, (19.0, 22.0), 5.1, 0.01),
        (5, (21.0, 21.0), 4.9, 0.01),
    ])

    densifier.update(sparse)
    depth, _var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.THIN_OBSTACLE
    assert 1.0 <= depth[1, 1] <= 1.08


def test_three_supported_points_are_mapped():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (18.0, 18.0), 5.0, 0.01),
        (2, (20.0, 20.0), 5.1, 0.02),
        (3, (22.0, 19.0), 4.9, 0.01),
    ])

    densifier.update(sparse)
    depth, _var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.MAPPED
    assert 4.9 <= depth[1, 1] <= 5.1


def test_invalid_sparse_depths_are_unknown():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (18.0, 18.0), -1.0, float("inf")),
        (2, (20.0, 20.0), float("nan"), 0.01),
    ])

    densifier.update(sparse)
    depth, var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.UNKNOWN
    assert depth[1, 1] == np.float32(-1.0)
    assert np.isinf(var[1, 1])
    assert densifier.unconverged_count == 0


def test_unconverged_finite_depths_render_in_debug_grid_only():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (18.0, 18.0), 3.0, 0.2),
    ])

    sparse.query = lambda _fid: (-1.0, float("inf"))

    densifier.update(sparse)
    depth, _var, state = densifier.dense_depth_grid()
    debug_depth, _debug_var, debug_state = densifier.dense_debug_grid()

    assert state[1, 1] == GridCellState.UNKNOWN
    assert depth[1, 1] == np.float32(-1.0)
    assert debug_state[1, 1] == GridCellState.UNKNOWN
    assert np.isclose(debug_depth[1, 1], 3.0)
    assert densifier.unconverged_count == 1


def test_unconverged_unknown_cells_are_colored_by_debug_depth():
    window = GridDensifierDebugWindow(enabled=False, cell_px=1)
    depth = np.full((2, 2), -1.0, dtype=np.float32)
    state = np.full((2, 2), int(GridCellState.UNKNOWN), dtype=np.uint8)
    depth[0, 0] = 3.0

    image = window._render(depth, state, unconverged_count=1)

    assert tuple(image[0, 0]) != (35, 35, 35)
    assert tuple(image[1, 1]) == (35, 35, 35)


def test_debug_window_uses_smooth_visual_upsampling():
    window = GridDensifierDebugWindow(enabled=False, cell_px=4)
    depth = np.array([[1.0, 5.0]], dtype=np.float32)
    state = np.full((1, 2), int(GridCellState.MAPPED), dtype=np.uint8)

    image = window._render(depth, state)
    smooth_depth, smooth_known = window._smooth_depth_for_display(
        depth,
        np.isfinite(depth) & (depth > 0.0),
        scale=4,
    )

    assert np.all(smooth_known)
    assert 1.0 < smooth_depth[0, 3] < 5.0
    assert np.all(np.diff(smooth_depth[0]) >= 0.0)
    assert image.shape == (4, 8, 3)


def test_debug_window_renders_near_depths_redder_than_far_depths():
    window = GridDensifierDebugWindow(enabled=False, cell_px=4)
    depth = np.array([[1.0, 5.0]], dtype=np.float32)
    state = np.full((1, 2), int(GridCellState.MAPPED), dtype=np.uint8)

    image = window._render(depth, state)

    near_bgr = image[0, 0].astype(np.int16)
    far_bgr = image[0, -1].astype(np.int16)

    assert near_bgr[2] > far_bgr[2]
    assert near_bgr[0] < far_bgr[0]


def test_unconverged_canonical_depth_is_converted_for_debug_preview():
    densifier = _make_densifier()
    sparse = FakeCanonicalSparseVog()

    densifier.update(sparse)
    debug_depth, debug_var, _state = densifier.dense_debug_grid()

    assert np.isclose(debug_depth[1, 1], 2.0)
    assert np.isclose(debug_var[1, 1], 0.4096)
    assert densifier.unconverged_count == 1


def test_roi_query_reports_min_depth_and_incomplete_coverage():
    densifier = _make_densifier(region_grow_enabled=False)
    sparse = FakeSparseVog([
        (1, (18.0, 18.0), 2.0, 0.01),
        (2, (20.0, 20.0), 2.1, 0.01),
    ])

    densifier.update(sparse)

    min_depth, complete = densifier.min_depth_in_roi(16.0, 16.0, 32.0, 16.0)

    assert np.isclose(min_depth, 2.06, atol=0.02)
    assert complete is False


def test_window_contributes_to_neighboring_grid_centers():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (16.0, 24.0), 2.0, 0.01),
        (2, (16.0, 24.0), 2.05, 0.01),
    ])

    densifier.update(sparse)
    depth, _var, state = densifier.dense_depth_grid()

    assert state[1, 0] == GridCellState.THIN_OBSTACLE
    assert state[1, 1] == GridCellState.THIN_OBSTACLE
    assert np.isclose(depth[1, 0], 2.025)
    assert np.isclose(depth[1, 1], 2.025)


def test_ivif_depth_fusion_favors_lower_variance():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (24.0, 24.0), 4.0, 0.01),
        (2, (24.0, 24.0), 4.2, 1.0),
        (3, (24.0, 24.0), 4.2, 1.0),
    ])

    densifier.update(sparse)
    depth, var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.MAPPED
    assert depth[1, 1] < 4.05
    assert var[1, 1] < 0.02


def test_region_grow_fills_adjacent_empty_cells_from_supported_seed():
    densifier = _make_densifier()
    sparse = FakeSparseVog([
        (1, (24.0, 24.0), 4.0, 0.01),
        (2, (24.0, 24.0), 4.05, 0.01),
        (3, (24.0, 24.0), 3.95, 0.01),
    ])

    densifier.update(sparse)
    depth, var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.MAPPED
    assert state[1, 2] == GridCellState.REGION_FILLED
    assert depth[1, 2] > depth[1, 1]
    assert var[1, 2] > var[1, 1]


def test_region_grow_does_not_cross_sharp_depth_boundary():
    densifier = GridDensifier(
        np.eye(3),
        image_size=(80, 48),
        settings=GridDensifierSettings(
            stride=16,
            window_size=16,
            mapped_min_support=3,
            thin_min_support=2,
            z_abs_cluster=0.25,
            z_rel_cluster=0.05,
            region_grow_max_steps=2,
            region_grow_max_depth_jump=0.5,
        ),
    )
    sparse = FakeSparseVog([
        (1, (8.0, 24.0), 2.0, 0.01),
        (2, (8.0, 24.0), 2.05, 0.01),
        (3, (8.0, 24.0), 1.95, 0.01),
        (4, (56.0, 24.0), 8.0, 0.01),
        (5, (56.0, 24.0), 8.05, 0.01),
        (6, (56.0, 24.0), 7.95, 0.01),
    ])

    densifier.update(sparse)
    _depth, _var, state = densifier.dense_depth_grid()

    assert state[1, 1] == GridCellState.REGION_FILLED
    assert state[1, 2] == GridCellState.FAR_FILLED
    assert state[1, 3] == GridCellState.MAPPED


def test_region_grow_fills_unreached_cells_as_far_after_finding_seed():
    densifier = GridDensifier(
        np.eye(3),
        image_size=(80, 48),
        settings=GridDensifierSettings(
            stride=16,
            window_size=16,
            mapped_min_support=3,
            thin_min_support=2,
            region_grow_max_steps=1,
            max_depth=20.0,
        ),
    )
    sparse = FakeSparseVog([
        (1, (8.0, 24.0), 2.0, 0.01),
        (2, (8.0, 24.0), 2.05, 0.01),
        (3, (8.0, 24.0), 1.95, 0.01),
    ])

    densifier.update(sparse)
    depth, _var, state = densifier.dense_depth_grid()

    assert state[1, 0] == GridCellState.MAPPED
    assert state[1, 1] == GridCellState.REGION_FILLED
    assert state[1, 4] == GridCellState.FAR_FILLED
    assert depth[1, 4] == np.float32(20.0)
