"""
Tests for EqF matrices: A0t state matrix, Bt input matrix, Ct output matrix.

Port of: test/test_EqFMatrices.cpp

Validates (all for Euclidean chart):
    1. A0t: Jacobian of the state error propagation function matches analytical A0t
    2. Bt: Jacobian of the input error propagation function matches analytical Bt
    3. Ct: Jacobian of the output error function matches analytical Ct
    4. C*_t at predicted measurement equals Ct (non-equivariant case)

Run: pytest tests/test_eqf_matrices.py -v
"""

import numpy as np
import pytest

from .testing_utilities import (
    reasonable_state_element, reasonable_group_element,
    random_velocity_element, create_default_camera,
    check_differential,
    TEST_REPS, NEAR_ZERO,
)
from eqvio.mathematical.vio_state import VIOState, VIOSensorState
from eqvio.mathematical.vio_group import (
    VIOGroup, VIOAlgebra,
    state_group_action, lift_velocity, vio_exp,
)
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vision_measurement import VisionMeasurement, measure_system_state
from eqvio.coordinate_suite.euclid import (
    state_matrix_A_euclid,
    input_matrix_B_euclid,
    output_matrix_Ci_star_euclid,
    state_chart_euclid,
    state_chart_inv_euclid,
    EqFCoordinateSuite_euclid,
)


IDS = [0, 1, 2, 3, 4]
IDS_OUTPUT = [5, 0, 1, 2, 3, 4]  # C++ test uses this order for output tests


class TestStateMatrixA:
    """Port of TEST_P(EqFSuiteTest, stateMatrixA).

    The A0t matrix is the Jacobian of the state error dynamics at epsilon=0.
    We verify this by numerically differentiating the function:

        a0(epsilon) = chart(action(X^{-1}, action(exp(LambdaTilde), action(X, chart_inv(epsilon)))))

    where LambdaTilde = lift(action(X, chart_inv(eps)), vel) - lift(action(X, xi0), vel).
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_A0t_finite_difference(self, seed):
        rng = np.random.default_rng(seed)
        xi0 = reasonable_state_element(IDS, rng)
        X_hat = reasonable_group_element(IDS, rng)
        vel = random_velocity_element(rng)

        # Analytical A0t
        A0t = state_matrix_A_euclid(X_hat, xi0, vel)

        def a0(epsilon):
            xi_hat = state_group_action(X_hat, xi0)
            xi_e = state_chart_inv_euclid(epsilon, xi0)
            xi = state_group_action(X_hat, xi_e)
            Lambda_tilde = lift_velocity(xi, vel) - lift_velocity(xi_hat, vel)
            xi_hat1 = state_group_action(vio_exp(Lambda_tilde), xi_hat)
            xi_e1 = state_group_action(X_hat.inverse(), xi_hat1)
            return state_chart_euclid(xi_e1, xi0)

        dim = xi0.dim()

        # Check function at zero gives zero
        a0_at_zero = a0(np.zeros(dim))
        assert np.linalg.norm(a0_at_zero) < NEAR_ZERO, (
            f"a0(0) should be zero, got norm={np.linalg.norm(a0_at_zero):.2e}"
        )

        # Check Jacobian
        check_differential(a0, np.zeros(dim), A0t)


class TestInputMatrixB:
    """Port of TEST_P(EqFSuiteTest, inputMatrixB).

    The Bt matrix is the Jacobian of the state error w.r.t. input perturbation.
    We verify by differentiating:

        b0(vel_err) = chart(action(X^{-1}, action(exp(LambdaTilde), xi_hat)))

    where LambdaTilde = lift(xi_hat, vel + vel_err) - lift(xi_hat, vel).
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_Bt_finite_difference(self, seed):
        rng = np.random.default_rng(seed)
        xi0 = reasonable_state_element(IDS, rng)
        X_hat = reasonable_group_element(IDS, rng)
        vel = random_velocity_element(rng)

        # Analytical Bt
        Bt = input_matrix_B_euclid(X_hat, xi0)

        def b0(vel_err_vec):
            xi_hat = state_group_action(X_hat, xi0)
            vel_err = IMUVelocity.from_vec12(vel_err_vec)
            Lambda_tilde = lift_velocity(xi_hat, vel + vel_err) - lift_velocity(xi_hat, vel)
            xi_hat1 = state_group_action(vio_exp(Lambda_tilde), xi_hat)
            xi_e1 = state_group_action(X_hat.inverse(), xi_hat1)
            return state_chart_euclid(xi_e1, xi0)

        # Check function at zero gives zero
        b0_at_zero = b0(np.zeros(12))
        assert np.linalg.norm(b0_at_zero) < NEAR_ZERO, (
            f"b0(0) should be zero, got norm={np.linalg.norm(b0_at_zero):.2e}"
        )

        # Check Jacobian
        check_differential(b0, np.zeros(12), Bt)


class TestOutputMatrixC:
    """Port of TEST_P(EqFSuiteTest, outputMatrixC).

    The Ct matrix is the Jacobian of the output error function:

        ct(epsilon) = h(action(X, chart_inv(epsilon))) - h(action(X, xi0))

    We verify:
    1. Ct matches numerical Jacobian at epsilon=0
    2. Ct with equivariance (C*_t) equals Ct without when evaluated at predicted measurement
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_Ct_finite_difference(self, seed):
        rng = np.random.default_rng(seed)
        xi0 = reasonable_state_element(IDS_OUTPUT, rng)
        X_hat = reasonable_group_element(IDS_OUTPUT, rng)
        cam = create_default_camera()

        xi_hat = state_group_action(X_hat, xi0)
        y_hat = measure_system_state(xi_hat, cam)
        y_ids = y_hat.get_ids()
        y_coords = y_hat.cam_coordinates

        # Analytical Ct (equivariant, evaluated at predicted measurement)
        suite = EqFCoordinateSuite_euclid
        Ct = suite.output_matrix_C(xi0, X_hat, y_ids, y_coords, cam, use_equivariance=True)

        # Also compute non-equivariant version — at predicted measurement they should match
        Ct_noneq = suite.output_matrix_C(xi0, X_hat, y_ids, y_coords, cam, use_equivariance=False)
        np.testing.assert_allclose(
            Ct, Ct_noneq, atol=1e-8,
            err_msg="C*_t and Ct should be equal when evaluated at predicted measurement"
        )

        def ct(epsilon):
            xi_e = state_chart_inv_euclid(epsilon, xi0)
            xi = state_group_action(X_hat, xi_e)
            y = measure_system_state(xi, cam)
            # Innovation: difference for common features
            y_tilde = y - y_hat
            return y_tilde

        dim = xi0.dim()

        # Check function at zero gives zero
        ct_at_zero = ct(np.zeros(dim))
        assert np.linalg.norm(ct_at_zero) < NEAR_ZERO, (
            f"ct(0) should be zero, got norm={np.linalg.norm(ct_at_zero):.2e}"
        )

        # Check Jacobian (use float-epsilon step like C++)
        float_step = np.cbrt(np.finfo(np.float32).eps)
        check_differential(ct, np.zeros(dim), Ct, step=float_step, atol=1e-3)

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_Ct_equivariant_at_prediction(self, seed):
        """C*_t(y_hat) = C_t when y = y_hat (the equivariant and standard output
        matrices coincide at the predicted measurement).

        This is a key property: the equivariant approximation reduces to the
        standard linearization when no measurement innovation is available.
        """
        rng = np.random.default_rng(seed)
        xi0 = reasonable_state_element(IDS, rng)
        X_hat = reasonable_group_element(IDS, rng)
        cam = create_default_camera()

        xi_hat = state_group_action(X_hat, xi0)
        y_hat = measure_system_state(xi_hat, cam)

        suite = EqFCoordinateSuite_euclid
        y_ids = y_hat.get_ids()
        y_coords = y_hat.cam_coordinates

        Ct_star = suite.output_matrix_C(xi0, X_hat, y_ids, y_coords, cam, True)
        Ct = suite.output_matrix_C(xi0, X_hat, y_ids, y_coords, cam, False)

        np.testing.assert_allclose(Ct_star, Ct, atol=1e-8)


class TestOutputMatrixCStar:
    """Port of TEST(EqFSuiteTest, outputMatrixCStar).

    Verifies that C*_t (equivariant output matrix) is a better local
    approximation than Ct when the true measurement differs from predicted.
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_Cstar_better_approximation(self, seed):
        rng = np.random.default_rng(seed)
        cam = create_default_camera()

        # Random landmark and SOT(3) near identity
        q0 = rng.standard_normal(3) * 10 + np.array([0, 0, 20.0])
        Q_hat = SOT3(
            R=SO3.exp(rng.standard_normal(3) * 0.02),
            a=rng.uniform(1.0, 3.0),
        )

        q_hat = Q_hat.inverse() * q0
        y_hat = cam.project_point(q_hat)

        # Non-equivariant Ct (= C*_t at y_hat)
        Ct = output_matrix_Ci_star_euclid(q0, Q_hat, cam, y_hat)

        def h_func(epsilon):
            """Output function in Euclidean chart coordinates."""
            qq = q0 @ q0
            # Map R^3 epsilon to sot(3)
            eps_sot = np.zeros(4)
            eps_sot[:3] = -np.cross(q0, epsilon) / qq
            eps_sot[3] = -(q0 @ epsilon) / qq

            q_e = SOT3.exp(-eps_sot) * q0
            q = Q_hat.inverse() * q_e
            return cam.project_point(q)

        float_step = 100.0 * np.cbrt(np.finfo(np.float32).eps)

        for j in range(3):
            ej = np.zeros(3)
            ej[j] = 1.0
            eps = float_step * ej

            y_true = h_func(eps)
            y_tilde = y_true - y_hat

            # Equivariant approximation at true measurement
            Ct_star = output_matrix_Ci_star_euclid(q0, Q_hat, cam, y_true)
            y_tilde_star = Ct_star @ eps

            # Standard approximation
            y_tilde_est0 = Ct @ eps

            lin_error_star = np.linalg.norm(y_tilde_star - y_tilde)
            lin_error_est0 = np.linalg.norm(y_tilde_est0 - y_tilde)

            assert lin_error_star <= lin_error_est0 + 1e-10, (
                f"C*_t not better than Ct in direction {j}: "
                f"star={lin_error_star:.2e}, standard={lin_error_est0:.2e}"
            )


class TestMatrixDimensions:
    """Verify output dimensions of all matrices."""

    @pytest.mark.parametrize("n_landmarks", [1, 3, 5, 10])
    def test_A0t_dimensions(self, n_landmarks):
        ids = list(range(n_landmarks))
        rng = np.random.default_rng(42)
        xi0 = reasonable_state_element(ids, rng)
        X = reasonable_group_element(ids, rng)
        vel = random_velocity_element(rng)

        A0t = state_matrix_A_euclid(X, xi0, vel)
        dim = xi0.dim()
        assert A0t.shape == (dim, dim), f"A0t shape {A0t.shape} != ({dim},{dim})"

    @pytest.mark.parametrize("n_landmarks", [1, 3, 5, 10])
    def test_Bt_dimensions(self, n_landmarks):
        ids = list(range(n_landmarks))
        rng = np.random.default_rng(42)
        xi0 = reasonable_state_element(ids, rng)
        X = reasonable_group_element(ids, rng)

        Bt = input_matrix_B_euclid(X, xi0)
        dim = xi0.dim()
        assert Bt.shape == (dim, 12), f"Bt shape {Bt.shape} != ({dim},12)"

    @pytest.mark.parametrize("n_landmarks", [1, 3, 5])
    def test_Ct_dimensions(self, n_landmarks):
        ids = list(range(n_landmarks))
        rng = np.random.default_rng(42)
        xi0 = reasonable_state_element(ids, rng)
        X = reasonable_group_element(ids, rng)
        cam = create_default_camera()

        xi_hat = state_group_action(X, xi0)
        y_hat = measure_system_state(xi_hat, cam)
        y_ids = y_hat.get_ids()

        suite = EqFCoordinateSuite_euclid
        Ct = suite.output_matrix_C(xi0, X, y_ids, y_hat.cam_coordinates, cam)
        dim = xi0.dim()
        n_obs = len(y_ids)
        assert Ct.shape == (2 * n_obs, dim), f"Ct shape {Ct.shape} != ({2*n_obs},{dim})"


# Need SOT3 and SO3 imports for the C*_t test
from liepp import SO3, SOT3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
