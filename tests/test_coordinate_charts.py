"""
Tests for VIO coordinate charts (Euclidean).

Port of: test/test_CoordinateCharts.cpp (VIOChart_euclid portions)

Validates:
    1. Chart roundtrip: chart_inv(chart(xi, xi0), xi0) ≈ xi
    2. Chart at origin gives zero: chart(xi0, xi0) = 0
    3. Chart inverse at zero gives origin: chart_inv(0, xi0) = xi0
    4. Chart linearity for small perturbations

Only the Euclidean chart is tested (invdepth, normal not yet ported).

Run: pytest tests/test_coordinate_charts.py -v
"""

import numpy as np
import pytest

from .testing_utilities import (
    random_state_element, reasonable_state_element, state_distance,
    TEST_REPS, NEAR_ZERO,
)
from eqvio.mathematical.vio_state import VIOState, VIOSensorState, Landmark
from eqvio.coordinate_suite.euclid import (
    state_chart_euclid,
    state_chart_inv_euclid,
)


IDS = [0, 1, 2, 3, 4]


class TestVIOChartEuclid:
    """Port of TEST(CoordinateChartTest, VIOChart_euclid).

    The fundamental chart property:
        chart_inv(chart(xi, xi0), xi0) ≈ xi
    for any states xi, xi0.
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_roundtrip(self, seed):
        """chart_inv(chart(xi1, xi0), xi0) ≈ xi1"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        xi1 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi1, xi0)
        xi2 = state_chart_inv_euclid(eps, xi0)

        dist = state_distance(xi1, xi2)
        assert dist < 1e-8, f"Chart roundtrip error: dist={dist:.2e}"

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_origin_maps_to_zero(self, seed):
        """chart(xi0, xi0) = 0"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi0, xi0)
        assert np.linalg.norm(eps) < NEAR_ZERO, (
            f"chart(xi0, xi0) not zero: norm={np.linalg.norm(eps):.2e}"
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_zero_maps_to_origin(self, seed):
        """chart_inv(0, xi0) = xi0"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        dim = xi0.dim()

        xi_recovered = state_chart_inv_euclid(np.zeros(dim), xi0)
        dist = state_distance(xi0, xi_recovered)
        assert dist < NEAR_ZERO, (
            f"chart_inv(0, xi0) != xi0: dist={dist:.2e}"
        )


class TestChartDimensions:
    """Verify chart output dimensions are correct."""

    @pytest.mark.parametrize("n_landmarks", [0, 1, 3, 5, 10])
    def test_chart_output_size(self, n_landmarks):
        ids = list(range(n_landmarks))
        rng = np.random.default_rng(42)
        xi0 = random_state_element(ids, rng)
        xi1 = random_state_element(ids, rng)

        eps = state_chart_euclid(xi1, xi0)
        expected_dim = VIOSensorState.CDim + 3 * n_landmarks
        assert eps.shape == (expected_dim,), (
            f"Chart output shape {eps.shape} != ({expected_dim},)"
        )


class TestChartSensorComponents:
    """Verify each sensor state component is correctly charted."""

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_bias_component(self, seed):
        """Bias chart is simple subtraction: eps[0:6] = bias1 - bias0."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        xi1 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi1, xi0)
        np.testing.assert_allclose(
            eps[0:6],
            xi1.sensor.input_bias - xi0.sensor.input_bias,
            atol=1e-12,
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_velocity_component(self, seed):
        """Velocity chart is simple subtraction: eps[12:15] = v1 - v0."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        xi1 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi1, xi0)
        np.testing.assert_allclose(
            eps[12:15],
            xi1.sensor.velocity - xi0.sensor.velocity,
            atol=1e-12,
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_landmark_component(self, seed):
        """Landmark chart is Euclidean subtraction: eps[21+3i:24+3i] = p1 - p0."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        xi1 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi1, xi0)
        S = VIOSensorState.CDim
        for i in range(len(IDS)):
            np.testing.assert_allclose(
                eps[S + 3 * i:S + 3 * (i + 1)],
                xi1.camera_landmarks[i].p - xi0.camera_landmarks[i].p,
                atol=1e-12,
            )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_pose_component_roundtrip(self, seed):
        """Pose chart uses SE3 log: eps[6:12] = log(pose0^{-1} * pose1).

        Verify roundtrip: pose0 * exp(eps[6:12]) ≈ pose1.
        """
        from liepp import SE3

        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        xi1 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi1, xi0)
        pose_recovered = xi0.sensor.pose * SE3.exp(eps[6:12])

        np.testing.assert_allclose(
            pose_recovered.asMatrix(),
            xi1.sensor.pose.asMatrix(),
            atol=1e-8,
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_camera_offset_component_roundtrip(self, seed):
        """Camera offset chart: eps[15:21] = log(offset0^{-1} * offset1)."""
        from liepp import SE3

        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        xi1 = random_state_element(IDS, rng)

        eps = state_chart_euclid(xi1, xi0)
        offset_recovered = xi0.sensor.camera_offset * SE3.exp(eps[15:21])

        np.testing.assert_allclose(
            offset_recovered.asMatrix(),
            xi1.sensor.camera_offset.asMatrix(),
            atol=1e-8,
        )


class TestChartInverseComponents:
    """Verify chart inverse reconstructs each component correctly."""

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_small_perturbation_inverse(self, seed):
        """For small eps, chart_inv(eps, xi0) should be close to xi0."""
        rng = np.random.default_rng(seed)
        xi0 = reasonable_state_element(IDS, rng)
        dim = xi0.dim()

        eps = rng.standard_normal(dim) * 0.001
        xi_perturbed = state_chart_inv_euclid(eps, xi0)

        dist = state_distance(xi0, xi_perturbed)
        assert dist < 0.1, f"Small perturbation moved state by {dist:.2e}"

        # Re-chart should recover eps
        eps_recovered = state_chart_euclid(xi_perturbed, xi0)
        np.testing.assert_allclose(eps_recovered, eps, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
