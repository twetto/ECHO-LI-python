"""
Tests for VIO lift velocity and innovation lifting.

Port of: test/test_VIOLift.cpp

Validates:
    1. Continuous lift convergence: as dt->0, stateGroupAction(exp(dt*lambda), xi)
       converges to integrateSystemFunction(xi, vel, dt)
    2. Discrete lift exactness: stateGroupAction(liftDiscrete(xi, vel, dt), xi)
       exactly equals integrateSystemFunction(xi, vel, dt)
    3. Innovation lift roundtrip (continuous): lifting eps to the algebra, exponentiating,
       applying the action, then charting back gives eps (to first order)
    4. Innovation lift roundtrip (discrete): same but exact for finite eps

Run: pytest tests/test_vio_lift.py -v
"""

import numpy as np
import pytest

from .testing_utilities import (
    random_state_element, reasonable_state_element,
    random_group_element, random_velocity_element,
    state_distance, log_norm,
    TEST_REPS, NEAR_ZERO,
)
from eqvio.mathematical.vio_state import VIOState, VIOSensorState, integrate_system_function
from eqvio.mathematical.vio_group import (
    VIOGroup, VIOAlgebra,
    state_group_action, lift_velocity, lift_velocity_discrete, vio_exp,
)
from eqvio.coordinate_suite.euclid import (
    state_chart_euclid, state_chart_inv_euclid,
    lift_innovation_euclid, lift_innovation_discrete_euclid,
)


IDS = [0, 1, 2, 3, 4]


class TestLiftConvergence:
    """Port of TEST(VIOLiftTest, Lift).

    Verify that the continuous lift converges: as dt -> 0,
    stateGroupAction(exp(dt * lambda), xi) approaches
    integrateSystemFunction(xi, vel, dt) with decreasing error/dt.
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_lift_convergence(self, seed):
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        velocity = random_velocity_element(rng)

        lam = lift_velocity(xi0, velocity)

        prev_dist = 1e8
        for i in range(8):
            dt = 10.0 ** (-i)

            # Ground truth: integrate dynamics
            xi1 = integrate_system_function(xi0, velocity, dt)

            # Lift: apply group action
            lam_exp = vio_exp(dt * lam)
            xi2 = state_group_action(lam_exp, xi0)

            # Error per unit time should decrease
            diff_dist = state_distance(xi1, xi2) / dt if dt > 0 else 0
            assert diff_dist <= prev_dist + 1e-6, (
                f"Lift convergence failed at dt=1e-{i}: "
                f"dist/dt={diff_dist:.2e} > prev={prev_dist:.2e}"
            )
            prev_dist = diff_dist


class TestDiscreteLift:
    """Port of TEST(VIOLiftTest, DiscreteLift).

    Verify that the discrete lift exactly reproduces system integration:
    stateGroupAction(liftVelocityDiscrete(xi, vel, dt), xi) = integrateSystemFunction(xi, vel, dt)
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_discrete_lift_exact(self, seed):
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        velocity = random_velocity_element(rng)
        dt = 0.1

        # Ground truth
        xi1 = integrate_system_function(xi0, velocity, dt)

        # Discrete lift
        X = lift_velocity_discrete(xi0, velocity, dt)
        xi2 = state_group_action(X, xi0)

        dist = state_distance(xi1, xi2)
        assert dist < NEAR_ZERO, (
            f"Discrete lift mismatch: dist={dist:.2e}"
        )


class TestInnovationLiftEuclid:
    """Port of TEST(VIOLiftTest, InnovationLifts_euclid).

    Test 1 (continuous): The reprojection function
        eps -> chart(action(exp(liftInnovation(eps)), xi0), xi0)
    should have identity Jacobian at eps=0.

    Test 2 (discrete): The reprojection function
        eps -> chart(action(liftInnovationDiscrete(eps), xi0), xi0)
    should be exactly the identity for any eps (not just at zero).
    """

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_continuous_innovation_roundtrip(self, seed):
        """Jacobian of reprojection at zero should be identity.

        Reference: innovationLift_test() in test_VIOLift.cpp
        """
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        dim = xi0.dim()

        def reprojection(eps):
            lifted = lift_innovation_euclid(eps, xi0)
            Delta = vio_exp(lifted)
            xi1 = state_group_action(Delta, xi0)
            return state_chart_euclid(xi1, xi0)

        # Check function at zero gives zero
        zero_val = reprojection(np.zeros(dim))
        assert np.linalg.norm(zero_val) < NEAR_ZERO, (
            f"Reprojection at zero != 0: norm={np.linalg.norm(zero_val):.2e}"
        )

        # Check Jacobian is identity via finite differences
        step = 1e-6
        J = np.zeros((dim, dim))
        for j in range(dim):
            ej = np.zeros(dim)
            ej[j] = step
            J[:, j] = (reprojection(ej) - reprojection(-ej)) / (2 * step)

        diff = np.linalg.norm(J - np.eye(dim))
        assert diff < 1e-4, (
            f"Innovation lift Jacobian not identity: ||J - I|| = {diff:.2e}"
        )

    @pytest.mark.parametrize("seed", range(TEST_REPS))
    def test_discrete_innovation_roundtrip(self, seed):
        """Discrete lift should be exactly identity for any eps.

        Reference: discreteInnovationLift_test() in test_VIOLift.cpp
        Tests: for each basis vector ej, reprojection(ej) == ej
        """
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        dim = xi0.dim()

        def reprojection(eps):
            Delta = lift_innovation_discrete_euclid(eps, xi0)
            xi1 = state_group_action(Delta, xi0)
            return state_chart_euclid(xi1, xi0)

        # Test along each basis vector (matching C++)
        for j in range(dim):
            ej = np.zeros(dim)
            ej[j] = 1.0
            result = reprojection(ej)
            np.testing.assert_allclose(
                result, ej, atol=1e-8,
                err_msg=f"Discrete innovation roundtrip failed for basis {j}"
            )


class TestDiscreteLiftMultipleTimesteps:
    """Additional: verify discrete lift works across a range of dt values."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("dt", [0.001, 0.01, 0.05, 0.1, 0.5])
    def test_various_dt(self, seed, dt):
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        velocity = random_velocity_element(rng)

        xi1 = integrate_system_function(xi0, velocity, dt)
        X = lift_velocity_discrete(xi0, velocity, dt)
        xi2 = state_group_action(X, xi0)

        dist = state_distance(xi1, xi2)
        assert dist < 1e-6, f"Discrete lift at dt={dt}: dist={dist:.2e}"


class TestLiftAtZeroVelocity:
    """Verify lift at zero velocity gives identity."""

    @pytest.mark.parametrize("seed", range(5))
    def test_zero_velocity_continuous(self, seed):
        rng = np.random.default_rng(seed)
        xi0 = random_state_element(IDS, rng)
        from eqvio.mathematical.imu_velocity import IMUVelocity

        # Zero IMU velocity (but with gravity in acc to cancel)
        xi0.sensor.velocity = np.zeros(3)  # must also be zero
        vel = IMUVelocity(
            gyr=xi0.sensor.input_bias[:3].copy(),  # bias-corrected gyr = 0
            acc=xi0.sensor.input_bias[3:].copy() + xi0.sensor.gravity_dir() * 9.80665,
        )

        lam = lift_velocity(xi0, vel)

        # All algebra components should be near zero
        assert np.linalg.norm(lam.u_w) < 1e-10, f"u_w not zero: {lam.u_w}"
        for Wi in lam.W:
            # sigma should be near zero (no translational velocity in camera frame)
            # omega should be near zero (no rotation)
            assert np.linalg.norm(Wi) < 1e-10, f"W not zero: {Wi}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
