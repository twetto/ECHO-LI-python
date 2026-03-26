"""
Tests for VIOGroup: Lie group axioms.

Port of: test/test_VIOGroup.cpp

Validates:
    1. Inverse: X^{-1} X = I and X X^{-1} = I
    2. Associativity: (X1 X2) X3 = X1 (X2 X3)
    3. Identity: I is neutral element

Run: pytest tests/test_vio_group.py -v
"""

import numpy as np
import pytest

from .testing_utilities import (
    random_group_element, log_norm,
    TEST_REPS, NEAR_ZERO,
)
from eqvio.mathematical.vio_group import VIOGroup


ALL_IDS = [0, 1, 2, 3, 4]


class TestVIOGroupBasicOperations:
    """Port of TEST(VIOGroupTest, BasicOperations)."""

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_inverse_left(self, seed):
        """X^{-1} * X ≈ I"""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        result = X.inverse() * X
        assert log_norm(result) < NEAR_ZERO, (
            f"||log(X^{{-1}} X)|| = {log_norm(result)}"
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_inverse_right(self, seed):
        """X * X^{-1} ≈ I"""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        result = X * X.inverse()
        assert log_norm(result) < NEAR_ZERO, (
            f"||log(X X^{{-1}})|| = {log_norm(result)}"
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_associativity(self, seed):
        """(X1 * X2) * X3 = X1 * (X2 * X3)"""
        rng = np.random.default_rng(seed)
        X1 = random_group_element(ALL_IDS, rng)
        X2 = random_group_element(ALL_IDS, rng)
        X3 = random_group_element(ALL_IDS, rng)

        lhs = (X1 * X2) * X3
        rhs = X1 * (X2 * X3)

        # Check all four ways (matching C++)
        assert log_norm(lhs.inverse() * rhs) < NEAR_ZERO
        assert log_norm(rhs.inverse() * lhs) < NEAR_ZERO
        assert log_norm(lhs * rhs.inverse()) < NEAR_ZERO
        assert log_norm(rhs * lhs.inverse()) < NEAR_ZERO

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_identity_is_neutral(self, seed):
        """I * X = X * I = X"""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        I = VIOGroup.Identity(ALL_IDS)

        # Identity has zero log-norm
        assert log_norm(I) < NEAR_ZERO

        # Left: I * X * X^{-1} ≈ I
        assert log_norm((I * X) * X.inverse()) < NEAR_ZERO

        # Right: X^{-1} * (X * I) ≈ I
        assert log_norm(X.inverse() * (X * I)) < NEAR_ZERO

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_identity_left_product(self, seed):
        """I * X should equal X (component-wise)."""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        I = VIOGroup.Identity(ALL_IDS)
        result = I * X

        np.testing.assert_allclose(result.beta, X.beta, atol=1e-12)
        np.testing.assert_allclose(result.w, X.w, atol=1e-12)
        np.testing.assert_allclose(result.A.asMatrix(), X.A.asMatrix(), atol=1e-12)
        np.testing.assert_allclose(result.B.asMatrix(), X.B.asMatrix(), atol=1e-12)
        for Qr, Qx in zip(result.Q, X.Q):
            np.testing.assert_allclose(Qr.asMatrix(), Qx.asMatrix(), atol=1e-12)

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_identity_right_product(self, seed):
        """X * I should equal X (component-wise)."""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        I = VIOGroup.Identity(ALL_IDS)
        result = X * I

        np.testing.assert_allclose(result.beta, X.beta, atol=1e-12)
        np.testing.assert_allclose(result.w, X.w, atol=1e-12)
        np.testing.assert_allclose(result.A.asMatrix(), X.A.asMatrix(), atol=1e-12)
        np.testing.assert_allclose(result.B.asMatrix(), X.B.asMatrix(), atol=1e-12)

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_double_inverse(self, seed):
        """(X^{-1})^{-1} = X"""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        result = X.inverse().inverse()
        diff = log_norm(result * X.inverse())
        assert diff < NEAR_ZERO, f"(X^-1)^-1 != X, diff={diff}"

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_no_nan(self, seed):
        """Random group elements should never contain NaN."""
        rng = np.random.default_rng(seed)
        X = random_group_element(ALL_IDS, rng)
        assert not X.has_nan()
        assert not X.inverse().has_nan()
        assert not (X * X.inverse()).has_nan()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
