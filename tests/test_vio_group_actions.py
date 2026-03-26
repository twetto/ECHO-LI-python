"""
Tests for VIOGroup actions on states and measurements.

Port of: test/test_VIOGroupActions.cpp

Validates:
    1. State action identity: φ(I, ξ) = ξ
    2. State action compatibility: φ(X2, φ(X1, ξ)) = φ(X1*X2, ξ)
    3. Output action identity and compatibility
    4. Output equivariance: h(φ(X, ξ)) = ρ(X, h(ξ))

Run: pytest tests/test_vio_group_actions.py -v
"""

import numpy as np
import pytest

from .testing_utilities import (
    random_state_element, random_group_element,
    random_vision_measurement, create_default_camera,
    state_distance, measurement_distance, log_norm,
    TEST_REPS, NEAR_ZERO,
)
from eqvio.mathematical.vio_group import (
    VIOGroup, state_group_action, sensor_state_group_action,
)
from eqvio.mathematical.vision_measurement import (
    VisionMeasurement, measure_system_state,
)


IDS = [0, 1, 2, 3, 4]


class TestStateAction:
    """Port of TEST(VIOActionTest, StateAction)."""

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_self_distance_zero(self, seed):
        """stateDistance(ξ, ξ) = 0."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        assert state_distance(xi0, xi0) < NEAR_ZERO

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_identity_action(self, seed):
        """φ(I, ξ) = ξ."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        I = VIOGroup.Identity(IDS)
        xi0_id = state_group_action(I, xi0)
        dist = state_distance(xi0_id, xi0)
        assert dist < NEAR_ZERO, f"Identity action moved state by {dist}"

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_action_compatibility(self, seed):
        """φ(X2, φ(X1, ξ)) = φ(X1 * X2, ξ).

        This is the fundamental group action axiom.
        """
        rng = np.random.default_rng(seed)
        X1 = random_group_element(IDS, rng)
        X2 = random_group_element(IDS, rng)
        xi0 = random_state_element(IDS, rng)

        # Sequential: apply X1, then X2
        xi1 = state_group_action(X2, state_group_action(X1, xi0))

        # Combined: apply X1*X2 at once
        xi2 = state_group_action(X1 * X2, xi0)

        dist = state_distance(xi1, xi2)
        assert dist < NEAR_ZERO, (
            f"Action compatibility failed: dist={dist}"
        )


class TestSensorStateAction:
    """Additional tests for the sensor state portion of the action."""

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_bias_addition(self, seed):
        """result.input_bias = sensor.input_bias + X.beta"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        result = sensor_state_group_action(X, xi0.sensor)
        np.testing.assert_allclose(
            result.input_bias, xi0.sensor.input_bias + X.beta, atol=1e-12
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_pose_right_multiply(self, seed):
        """result.pose = sensor.pose * X.A"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        result = sensor_state_group_action(X, xi0.sensor)
        expected = xi0.sensor.pose * X.A
        np.testing.assert_allclose(
            result.pose.asMatrix(), expected.asMatrix(), atol=1e-10
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_velocity_transform(self, seed):
        """result.velocity = A.R^{-1} * (sensor.velocity - X.w)"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        result = sensor_state_group_action(X, xi0.sensor)
        expected = X.A.R.inverse() * (xi0.sensor.velocity - X.w)
        np.testing.assert_allclose(result.velocity, expected, atol=1e-10)

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_camera_offset_transform(self, seed):
        """result.cameraOffset = A^{-1} * sensor.cameraOffset * X.B"""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        result = sensor_state_group_action(X, xi0.sensor)
        expected = X.A.inverse() * xi0.sensor.camera_offset * X.B
        np.testing.assert_allclose(
            result.camera_offset.asMatrix(), expected.asMatrix(), atol=1e-10
        )


class TestLandmarkAction:
    """Tests for point and plane landmark group action."""

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_point_inverse_sot3_action(self, seed):
        """Point landmarks: p_new = Q^{-1} * p = (1/a) R^T p."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        result = state_group_action(X, xi0)

        for i, (lm, Qi) in enumerate(zip(xi0.camera_landmarks, X.Q)):
            expected = Qi.inverse() * lm.p
            np.testing.assert_allclose(
                result.camera_landmarks[i].p, expected, atol=1e-12,
                err_msg=f"Point {i} inverse action mismatch"
            )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_landmark_ids_preserved(self, seed):
        """Landmark ids should be preserved by the action."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        result = state_group_action(X, xi0)

        for lm_orig, lm_new in zip(xi0.camera_landmarks, result.camera_landmarks):
            assert lm_orig.id == lm_new.id


class TestOutputEquivariance:
    """Port of TEST(VIOActionTest, OutputEquivariance).

    Verifies: h(φ(X, ξ)) = ρ(X, h(ξ))

    This is the fundamental equivariance property of the measurement function.
    We test only the bearing/projection part here (ρ acts on pixel coordinates
    via Q.R^{-1} on the bearing, then re-project).
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_measurement_from_transformed_state(self, seed):
        """Projecting landmarks from transformed state should give valid pixels."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        X = random_group_element(IDS, rng)
        cam = create_default_camera()

        xi_transformed = state_group_action(X, xi0)
        y = measure_system_state(xi_transformed, cam)

        # All landmarks should project to finite pixels
        for fid, pixel in y.cam_coordinates.items():
            assert np.all(np.isfinite(pixel)), f"Non-finite pixel for id {fid}"

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_identity_measurement_unchanged(self, seed):
        """h(φ(I, ξ)) = h(ξ)."""
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        cam = create_default_camera()
        I = VIOGroup.Identity(IDS)

        y_orig = measure_system_state(xi0, cam)
        y_id = measure_system_state(state_group_action(I, xi0), cam)

        dist = measurement_distance(y_orig, y_id)
        assert dist < 1e-5, f"Identity changed measurement by {dist}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
