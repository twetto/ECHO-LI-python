"""
Validate analytical plane A/B blocks against numerical state_matrix_A_discrete.

The numerical differentiator computes the EXACT A_discrete by perturbing each
chart coordinate and observing the resulting error dynamics. We compare our
analytical candidate blocks against this ground truth.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from liepp import SO3, SE3, SOT3

from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark,
)
from eqvio.mathematical.vio_group import VIOGroup, state_group_action
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.eqf_matrices import EqFCoordinateSuite
from eqvio.coordinate_suite.euclid import (
    state_matrix_A_euclid,
    input_matrix_B_euclid,
    state_chart_euclid,
    state_chart_inv_euclid,
    lift_innovation_euclid,
    lift_innovation_discrete_euclid,
    EqFCoordinateSuite_euclid,
)


def make_state_with_planes():
    """Create a state with 2 points + 1 plane, nonzero velocity and rotation."""
    xi0 = VIOState()
    # Give the sensor a nontrivial pose and velocity
    xi0.sensor.pose = SE3.exp(np.array([0.1, -0.05, 0.02, 0.5, -0.3, 0.1]))
    xi0.sensor.velocity = np.array([0.3, -0.1, 0.05])
    xi0.sensor.camera_offset = SE3.exp(np.array([0.01, 0.02, -0.01, 0.05, 0.0, -0.02]))
    xi0.sensor.input_bias = np.array([0.001, -0.002, 0.001, 0.01, -0.01, 0.005])

    xi0.camera_landmarks = [
        Landmark(p=np.array([0.5, 0.2, 3.0]), id=10),
        Landmark(p=np.array([-0.3, 0.1, 2.5]), id=11),
    ]
    # Plane at z≈3 in camera frame: q^T p + 1 = 0 → q = [0, 0, -1/3]
    xi0.plane_landmarks = [
        PlaneLandmark(q=np.array([0.02, -0.01, -0.33]), id=100,
                      point_ids=[10]),
    ]
    return xi0


def make_imu():
    """Create a nontrivial IMU velocity."""
    imu = IMUVelocity()
    imu.gyr = np.array([0.1, -0.05, 0.02])
    imu.acc = np.array([0.3, -0.1, 9.81])
    imu.gyr_bias_vel = np.array([0.0001, -0.0002, 0.0001])
    imu.acc_bias_vel = np.array([0.001, -0.001, 0.0005])
    return imu


class TestPlaneAMatrix:
    """Compare analytical A blocks against numerical A_discrete.

    Note: state_matrix_A_euclid returns continuous A (small values).
    state_matrix_A_discrete returns discrete F ≈ I + dt*A (near-identity).
    We compare F_analytical = I + dt*A_analytical against F_numerical.
    """

    def _get_matrices(self, xi0=None, X=None, imu=None, dt=0.005):
        if xi0 is None:
            xi0 = make_state_with_planes()
        if X is None:
            X = VIOGroup.Identity(ids=[10, 11], plane_ids=[100])
        if imu is None:
            imu = make_imu()

        A_cont = state_matrix_A_euclid(X, xi0, imu)
        dim = xi0.dim()
        F_analytical = np.eye(dim) + dt * A_cont
        F_numerical = EqFCoordinateSuite_euclid.state_matrix_A_discrete(X, xi0, imu, dt)
        return F_analytical, F_numerical, xi0

    def test_self_block_identity_Q(self):
        """Plane self-block matches numerical at identity Q."""
        F_ana, F_num, xi0 = self._get_matrices()
        S, P = VIOSensorState.CDim, VIOSensorState.CDim + 6
        np.testing.assert_allclose(
            F_ana[P:P+3, P:P+3], F_num[P:P+3, P:P+3], atol=2e-4,
            err_msg="Plane self-block mismatch")

    def test_self_block_nontrivial_Q(self):
        """Plane self-block matches numerical with random Q."""
        xi0 = make_state_with_planes()
        X = VIOGroup.Identity(ids=[10, 11], plane_ids=[100])
        for i in range(len(X.Q)):
            X.Q[i] = SOT3(SO3.exp(np.random.randn(3) * 0.1),
                           np.exp(np.random.randn() * 0.1))
        X.Q_planes[0] = SOT3(SO3.exp(np.random.randn(3) * 0.1),
                               np.exp(np.random.randn() * 0.1))
        F_ana, F_num, _ = self._get_matrices(xi0=xi0, X=X)
        P = VIOSensorState.CDim + 6
        np.testing.assert_allclose(
            F_ana[P:P+3, P:P+3], F_num[P:P+3, P:P+3], atol=2e-4,
            err_msg="Plane self-block (nontrivial Q) mismatch")

    def test_velocity_to_plane_block(self):
        F_ana, F_num, _ = self._get_matrices()
        P = VIOSensorState.CDim + 6
        np.testing.assert_allclose(
            F_ana[P:P+3, 12:15], F_num[P:P+3, 12:15], atol=2e-4,
            err_msg="Plane velocity cross-term mismatch")

    def test_camera_offset_to_plane_block(self):
        F_ana, F_num, _ = self._get_matrices()
        P = VIOSensorState.CDim + 6
        np.testing.assert_allclose(
            F_ana[P:P+3, 15:21], F_num[P:P+3, 15:21], atol=2e-4,
            err_msg="Plane camera-offset cross-term mismatch")

    def test_bias_to_plane_block(self):
        F_ana, F_num, _ = self._get_matrices()
        P = VIOSensorState.CDim + 6
        np.testing.assert_allclose(
            F_ana[P:P+3, 0:6], F_num[P:P+3, 0:6], atol=2e-4,
            err_msg="Plane bias cross-term mismatch")

    def test_point_blocks_unchanged(self):
        """Existing point blocks NOT broken by plane additions."""
        F_ana, F_num, _ = self._get_matrices()
        S = VIOSensorState.CDim
        for i in range(2):
            np.testing.assert_allclose(
                F_ana[S+3*i:S+3*(i+1), S+3*i:S+3*(i+1)],
                F_num[S+3*i:S+3*(i+1), S+3*i:S+3*(i+1)],
                atol=2e-4, err_msg=f"Point {i} self-block broken")

    def test_full_matrix(self):
        """Full F matrix matches numerical within tolerance."""
        F_ana, F_num, xi0 = self._get_matrices()

        S = VIOSensorState.CDim
        P = S + 6
        block_names = ["bias(0:6)", "pose(6:12)", "vel(12:15)", "cam(15:21)",
                       "pt0(21:24)", "pt1(24:27)", "plane0(27:30)"]
        block_ranges = [(0,6), (6,12), (12,15), (15,21), (21,24), (24,27), (27,30)]

        print("\nMax |F_analytical - F_numerical| per block:")
        for rn, (r0, r1) in zip(block_names, block_ranges):
            for cn, (c0, c1) in zip(block_names, block_ranges):
                diff = np.max(np.abs(F_ana[r0:r1, c0:c1] - F_num[r0:r1, c0:c1]))
                #if diff > 1e-5:
                if True:
                    print(f"  [{rn}][{cn}]: {diff:.6e}")

        np.testing.assert_allclose(F_ana, F_num, atol=1e-3,
            err_msg="Full F matrix mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
