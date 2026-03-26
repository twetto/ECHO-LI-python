"""
Core math tests for EqVIO planar landmark extension.

Validates the new math NOT in the C++ codebase (Sections 2.1–2.6 of porting guide).
These tests are self-contained — SOT(3) operations are defined inline so they serve
as independent validators, not circular tests of the Lie group library.

Run: pytest test_planar_landmark_math.py -v
"""

import numpy as np
from scipy.spatial.transform import Rotation
import pytest

# ---------------------------------------------------------------------------
# Helpers: SOT(3) operations defined from scratch for independent validation
# ---------------------------------------------------------------------------

def random_rotation():
    """Random SO(3) matrix via scipy."""
    return Rotation.random().as_matrix()


def random_sot3():
    """Random SOT(3) element (R, a) with a > 0."""
    R = random_rotation()
    a = np.exp(np.random.randn() * 0.5)  # log-normal, stays positive
    return R, a


def sot3_action_point(R, a, p):
    """SOT(3) action on a point: Q * p = a * R @ p."""
    return a * R @ p


def sot3_inverse_action_point(R, a, p):
    """SOT(3) inverse action on a point: Q^{-1} * p = (1/a) R^T @ p."""
    return (1.0 / a) * R.T @ p


def sot3_inverse_action_plane(R, a, q):
    """SOT(3) dual inverse action on a plane CP: Q^{-1} * q = a * R^T @ q.

    Dual because planes transform contravariantly — the scale exponent
    flips sign relative to the point action.
    """
    return a * R.T @ q


def random_point_on_plane(q):
    """Generate a random point satisfying q^T p + 1 = 0.

    Given plane CP q = n/d, points on the plane satisfy n^T p = d,
    equivalently q^T p = -1.
    """
    # Pick two random tangent vectors on the plane
    n_hat = q / np.linalg.norm(q)
    # Arbitrary vector not parallel to n_hat
    v = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n_hat, v)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n_hat, t1)

    # Base point on plane: p0 = -q / ||q||^2 satisfies q^T p0 = -1
    p0 = -q / (q @ q)
    # Add tangent displacement
    p = p0 + np.random.randn() * t1 + np.random.randn() * t2
    return p


def skew(v):
    """3x3 skew-symmetric matrix from vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def so3_exp(omega):
    """Exponential map SO(3): omega in R^3 -> R in SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + skew(omega)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def sot3_exp(omega, sigma):
    """Exponential map SOT(3): (omega, sigma) -> (R, a)."""
    R = so3_exp(omega)
    a = np.exp(sigma)
    return R, a


# ---------------------------------------------------------------------------
# Test 1: CP incidence preserved under dual group action (Section 2.1)
# ---------------------------------------------------------------------------

class TestDualGroupAction:
    """Verify q'^T p' + 1 = 0 is preserved when (p, q) are transformed
    by (Q^{-1}_point, Q^{-1}_plane) respectively."""

    @pytest.mark.parametrize("seed", range(20))
    def test_incidence_preserved(self, seed):
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # Random plane CP: q = n/d, with d > 0
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        d = rng.uniform(0.5, 5.0)
        q = n / d

        # Random point on that plane
        p = random_point_on_plane(q)
        assert abs(q @ p + 1) < 1e-12, "Setup: point not on plane"

        # Random SOT(3) element
        R, a = random_sot3()

        # Transform both
        p_new = sot3_inverse_action_point(R, a, p)
        q_new = sot3_inverse_action_plane(R, a, q)

        # Incidence must be preserved
        residual = q_new @ p_new + 1
        assert abs(residual) < 1e-10, f"Incidence broken: q'^T p' + 1 = {residual}"

    @pytest.mark.parametrize("seed", range(10))
    def test_forward_action_incidence(self, seed):
        """Also verify using the forward action Q * p, Q_dual * q."""
        np.random.seed(seed)

        q = np.random.randn(3)
        q = q / np.linalg.norm(q) / np.random.uniform(0.5, 5.0)
        p = random_point_on_plane(q)

        R, a = random_sot3()

        # Forward actions
        p_new = sot3_action_point(R, a, p)  # a R p
        q_new = (1.0 / a) * R @ q           # dual forward: (1/a) R q

        residual = q_new @ p_new + 1
        assert abs(residual) < 1e-10, f"Forward incidence broken: {residual}"


# ---------------------------------------------------------------------------
# Test 2: Plane lift finite-difference (Section 2.2)
# ---------------------------------------------------------------------------

class TestPlaneLift:
    """Verify the lift velocity for planes:
        omega_L = -Omega_C
        sigma_L = -(q . v_C)
    by checking that the resulting SOT(3) algebra element reproduces
    the physical time derivative of q in camera frame.
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_lift_reproduces_q_dot(self, seed):
        rng = np.random.default_rng(seed)

        # Camera-frame velocity
        Omega_C = rng.standard_normal(3) * 0.3  # angular velocity
        v_C = rng.standard_normal(3) * 0.5       # linear velocity

        # Plane CP in camera frame
        q = rng.standard_normal(3)
        q_norm = np.linalg.norm(q)
        if q_norm < 0.1:
            q = np.array([0.0, 0.0, 1.0])

        # Lift (Section 2.2)
        omega_L = -Omega_C
        sigma_L = -(q @ v_C)

        # The SOT(3) algebra acts on q via the dual adjoint:
        #   q_dot = sigma_L * q + omega_L x q
        # (dual: +sigma instead of -sigma for points)
        q_dot_lift = sigma_L * q + np.cross(omega_L, q)

        # Physical derivative: camera rotates with Omega_C and translates with v_C.
        # A plane CP q = n/d transforms as:
        #   q_dot = -Omega_C x q - (q . v_C) * q     ... wait, let's derive carefully.
        #
        # In camera frame, the plane's (n,d) evolves as:
        #   n_dot = -Omega_C x n
        #   d_dot = n . v_C     (camera moving toward plane increases d)
        # Since q = n/d:
        #   q_dot = n_dot/d - n * d_dot / d^2
        #         = (-Omega_C x n)/d - n(n.v_C)/d^2
        #         = -Omega_C x q - (q . v_C) * q      [since n/d = q, n.v_C/d = q.v_C * d/d... no]
        #
        # More carefully: q = n/d, so n = q*d.
        #   n.v_C = q.v_C * d
        #   d_dot = q.v_C * d
        #   q_dot = (-Omega x (qd))/d - qd * (q.v_C * d) / d^2
        #         = -Omega x q - (q.v_C) * q
        q_dot_physics = -np.cross(Omega_C, q) - (q @ v_C) * q

        np.testing.assert_allclose(
            q_dot_lift, q_dot_physics, atol=1e-12,
            err_msg="Lift does not reproduce physical q_dot"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_lift_finite_difference(self, seed):
        """Numerical finite-difference check: evolve q with a small dt,
        compare against lift prediction."""
        rng = np.random.default_rng(seed)
        dt = 1e-7

        Omega_C = rng.standard_normal(3) * 0.3
        v_C = rng.standard_normal(3) * 0.5
        q = rng.standard_normal(3)
        if np.linalg.norm(q) < 0.1:
            q = np.array([0.0, 0.0, 1.0])

        # Decompose q = n/d with unit normal
        d_val = 1.0 / np.linalg.norm(q)
        n_vec = q * d_val  # unit normal

        # Camera evolves: R(dt) = exp(Omega*dt), translate by v_C*dt
        R_dt = so3_exp(Omega_C * dt)

        # Plane in new camera frame via homogeneous transform:
        #   pi = [n; d] satisfies pi^T [x; 1] = 0  (i.e., n^T x + d = 0)
        #   pi' = M^{-T} pi  where  M = [R^T, -R^T v dt; 0, 1]
        #   Gives: n' = R^T n,  d' = d + n·v dt  (note: + sign, not -)
        n_new = R_dt.T @ n_vec
        d_new = d_val + n_vec @ (v_C * dt)  # CP convention: d_dot = +n·v
        q_new = n_new / d_new

        q_dot_numerical = (q_new - q) / dt

        # Analytical from lift
        omega_L = -Omega_C
        sigma_L = -(q @ v_C)
        q_dot_analytical = sigma_L * q + np.cross(omega_L, q)

        np.testing.assert_allclose(
            q_dot_numerical, q_dot_analytical, atol=1e-5,
            err_msg="Finite-difference q_dot doesn't match lift"
        )


# ---------------------------------------------------------------------------
# Test 3: C*_t symmetry — C_tp = -C_tq (Section 2.3)
# ---------------------------------------------------------------------------

class TestConstraintCstar:
    """Validate the equivariant output matrix for h(p, q) = q^T p + 1."""

    @staticmethod
    def compute_C_star_landmark_blocks(p, q):
        """Compute the (1x4) SOT(3) blocks for point and plane.

        Returns (C_p, C_q) where:
            C_p = [q x p | -(q . p)]   shape (1, 4)
            C_q = -C_p                  shape (1, 4)
        """
        qxp = np.cross(q, p)
        qdp = q @ p
        C_p = np.array([[qxp[0], qxp[1], qxp[2], -qdp]])
        C_q = -C_p
        return C_p, C_q

    @pytest.mark.parametrize("seed", range(20))
    def test_symmetry(self, seed):
        """C*_p = -C*_q for all (p, q) pairs."""
        rng = np.random.default_rng(seed)
        p = rng.standard_normal(3) * 2
        q = rng.standard_normal(3)

        C_p, C_q = self.compute_C_star_landmark_blocks(p, q)
        np.testing.assert_allclose(C_p, -C_q, atol=1e-15)

    @pytest.mark.parametrize("seed", range(20))
    def test_finite_difference_point(self, seed):
        """Numerical Jacobian of h = q^T p + 1 w.r.t. SOT(3) perturbation of p.

        EqF convention: stateGroupAction uses the INVERSE action on points,
        so the perturbation is p_pert = exp(epsilon)^{-1} * p = (1/a) R^T p.
        C*_t measures dh/d_epsilon with this convention.
        """
        rng = np.random.default_rng(seed)

        q = rng.standard_normal(3)
        q = q / np.linalg.norm(q) / rng.uniform(0.5, 3.0)
        p = random_point_on_plane(q)

        C_p_analytical, _ = self.compute_C_star_landmark_blocks(p, q)

        eps = 1e-7
        C_p_numerical = np.zeros((1, 4))

        for i in range(3):
            omega = np.zeros(3)
            omega[i] = eps
            R_eps, a_eps = sot3_exp(omega, 0.0)
            # Inverse action: (1/a) R^T p
            p_pert = sot3_inverse_action_point(R_eps, a_eps, p)
            h_pert = q @ p_pert + 1
            h_nom = q @ p + 1
            C_p_numerical[0, i] = (h_pert - h_nom) / eps

        R_eps, a_eps = sot3_exp(np.zeros(3), eps)
        p_pert = sot3_inverse_action_point(R_eps, a_eps, p)
        h_pert = q @ p_pert + 1
        h_nom = q @ p + 1
        C_p_numerical[0, 3] = (h_pert - h_nom) / eps

        np.testing.assert_allclose(
            C_p_analytical, C_p_numerical, atol=1e-5,
            err_msg="C*_p doesn't match finite difference"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_finite_difference_plane(self, seed):
        """Numerical Jacobian of h = q^T p + 1 w.r.t. dual SOT(3) perturbation of q.

        EqF convention: stateGroupAction uses the dual inverse on planes,
        so the perturbation is q_pert = exp(epsilon)^{dual,-1} * q = a R^T q.
        """
        rng = np.random.default_rng(seed)

        q = rng.standard_normal(3)
        q = q / np.linalg.norm(q) / rng.uniform(0.5, 3.0)
        p = random_point_on_plane(q)

        _, C_q_analytical = self.compute_C_star_landmark_blocks(p, q)

        eps = 1e-7
        C_q_numerical = np.zeros((1, 4))

        for i in range(3):
            omega = np.zeros(3)
            omega[i] = eps
            R_eps, a_eps = sot3_exp(omega, 0.0)
            # Dual inverse action: a R^T q
            q_pert = sot3_inverse_action_plane(R_eps, a_eps, q)
            h_pert = q_pert @ p + 1
            h_nom = q @ p + 1
            C_q_numerical[0, i] = (h_pert - h_nom) / eps

        R_eps, a_eps = sot3_exp(np.zeros(3), eps)
        q_pert = sot3_inverse_action_plane(R_eps, a_eps, q)
        h_pert = q_pert @ p + 1
        h_nom = q @ p + 1
        C_q_numerical[0, 3] = (h_pert - h_nom) / eps

        np.testing.assert_allclose(
            C_q_analytical, C_q_numerical, atol=1e-5,
            err_msg="C*_q doesn't match finite difference"
        )


# ---------------------------------------------------------------------------
# Test 4: Innovation lift sign (Section 2.6)
# ---------------------------------------------------------------------------

class TestInnovationLift:
    """Verify structural properties of the innovation lift for points and planes.

    The innovation lift maps chart coordinates gamma in R^3 to sot(3) elements.
    We test:
    1. omega is perpendicular to p (resp. q) — minimal rotation
    2. Sign relationship: plane sigma = +, point sigma = - (dual)
    3. Finite-difference: the lift composed with the group action reproduces gamma
    """

    @staticmethod
    def point_innovation_lift(p, gamma):
        pp = p @ p
        omega = np.cross(p, gamma) / pp
        sigma = -(p @ gamma) / pp
        return omega, sigma

    @staticmethod
    def plane_innovation_lift(q, gamma):
        qq = q @ q
        omega = np.cross(q, gamma) / qq
        sigma = +(q @ gamma) / qq
        return omega, sigma

    @pytest.mark.parametrize("seed", range(20))
    def test_omega_perpendicular_to_landmark_point(self, seed):
        """For points, omega should be perpendicular to p."""
        rng = np.random.default_rng(seed)
        p = rng.standard_normal(3) + np.array([0, 0, 3.0])
        gamma = rng.standard_normal(3) * 0.1

        omega, _ = self.point_innovation_lift(p, gamma)
        assert abs(omega @ p) < 1e-12, f"omega not perpendicular to p: {omega @ p}"

    @pytest.mark.parametrize("seed", range(20))
    def test_omega_perpendicular_to_landmark_plane(self, seed):
        """For planes, omega should be perpendicular to q."""
        rng = np.random.default_rng(seed)
        q = rng.standard_normal(3)
        if np.linalg.norm(q) < 0.3:
            q += np.array([0, 0, 1.0])
        gamma = rng.standard_normal(3) * 0.1

        omega, _ = self.plane_innovation_lift(q, gamma)
        assert abs(omega @ q) < 1e-12, f"omega not perpendicular to q: {omega @ q}"

    @pytest.mark.parametrize("seed", range(20))
    def test_sigma_sign_duality(self, seed):
        """Point and plane lifts should have opposite sigma signs
        for the same landmark vector and gamma."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(3) + np.array([0, 0, 2.0])
        gamma = rng.standard_normal(3) * 0.1

        _, sigma_point = self.point_innovation_lift(v, gamma)
        _, sigma_plane = self.plane_innovation_lift(v, gamma)

        # Same omega formula, but sigma signs are flipped
        assert abs(sigma_point + sigma_plane) < 1e-12, \
            f"sigma signs not dual: point={sigma_point}, plane={sigma_plane}"

    @pytest.mark.parametrize("seed", range(20))
    def test_point_lift_fd_consistency(self, seed):
        """Verify point innovation lift via finite difference.

        Apply exp(omega, sigma) via the FORWARD group action (aRp) to p,
        check that the resulting displacement approximates gamma
        (forward convention for the lift).
        """
        rng = np.random.default_rng(seed)
        p = rng.standard_normal(3) + np.array([0, 0, 3.0])
        gamma = rng.standard_normal(3) * 1e-6  # very small for first-order accuracy

        omega, sigma = self.point_innovation_lift(p, gamma)
        R, a = sot3_exp(omega, sigma)

        # Forward action: the lift formula corresponds to aRp ≈ p + gamma
        p_corrected = sot3_action_point(R, a, p)
        delta = p_corrected - p

        np.testing.assert_allclose(
            delta, gamma, atol=1e-5,
            err_msg="Point lift (forward action) doesn't reproduce gamma"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_plane_lift_fd_consistency(self, seed):
        """Verify plane innovation lift via finite difference.

        Apply exp(omega, sigma) via the dual FORWARD action ((1/a)Rq) to q,
        check that the resulting displacement approximates gamma.
        """
        rng = np.random.default_rng(seed)
        q = rng.standard_normal(3)
        if np.linalg.norm(q) < 0.3:
            q += np.array([0, 0, 1.0])
        gamma = rng.standard_normal(3) * 1e-6

        omega, sigma = self.plane_innovation_lift(q, gamma)
        R, a = sot3_exp(omega, sigma)

        # Dual forward action: (1/a) R q
        q_corrected = (1.0 / a) * R @ q
        delta = q_corrected - q

        np.testing.assert_allclose(
            delta, gamma, atol=1e-5,
            err_msg="Plane lift (dual forward action) doesn't reproduce gamma"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_constraint_residual_shrinks(self, seed):
        """End-to-end: start with a perturbed (p, q) pair where q^T p + 1 != 0,
        apply EqF-style correction, verify residual decreases.

        The Kalman update produces delta in the SOT(3) Lie algebra directly
        (omega, sigma). We apply exp(delta) via the group action — no need
        to re-process through the innovation lift.
        """
        rng = np.random.default_rng(seed)

        # True plane and point (on the plane)
        q_true = rng.standard_normal(3)
        q_true = q_true / np.linalg.norm(q_true) / rng.uniform(1.0, 3.0)
        p_true = random_point_on_plane(q_true)

        # Small perturbations for linearization to hold
        p_hat = p_true + rng.standard_normal(3) * 0.005
        q_hat = q_true + rng.standard_normal(3) * 0.002

        residual_before = abs(q_hat @ p_hat + 1)

        # Compute constraint C* blocks (validated by FD tests above)
        C_p, C_q = TestConstraintCstar.compute_C_star_landmark_blocks(p_hat, q_hat)
        C_full = np.hstack([C_p, C_q])  # (1, 8)

        # Kalman-like correction: delta = K * innovation
        h = q_hat @ p_hat + 1
        R_noise = 0.0001  # small noise for this test
        S = C_full @ C_full.T + R_noise
        K = C_full.T / S[0, 0]
        delta = (K * h).flatten()

        # delta is in sot(3) x sot(3): [omega_p(3), sigma_p(1), omega_q(3), sigma_q(1)]
        omega_p, sigma_p = delta[:3], delta[3]
        omega_q, sigma_q = delta[4:7], delta[7]

        # Apply correction via FORWARD group action
        # (C*_t was derived for the inverse action convention, so the correction
        # that reduces the residual uses the forward action)
        R_p, a_p = sot3_exp(omega_p, sigma_p)
        p_corrected = sot3_action_point(R_p, a_p, p_hat)

        R_q, a_q = sot3_exp(omega_q, sigma_q)
        # Dual forward: (1/a) R q
        q_corrected = (1.0 / a_q) * R_q @ q_hat

        residual_after = abs(q_corrected @ p_corrected + 1)

        assert residual_after < residual_before, (
            f"Residual didn't shrink: {residual_before:.6f} -> {residual_after:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: CP convention companion (Section 5, last paragraph)
# ---------------------------------------------------------------------------

class TestCPConvention:
    """Companion to test_sot3_official_match.py.
    Validates that the CP representation q = n/d with the dual SOT(3) action
    is consistent with the 4x4 matrix representation of planes.
    """

    @pytest.mark.parametrize("seed", range(20))
    def test_cp_vs_4x4_matrix(self, seed):
        """A plane in homogeneous coordinates is pi = [n; d] such that
        pi^T [p; 1] = 0 for points on the plane.

        Under SE(3) transform T = [R t; 0 1], the plane transforms as:
            pi' = T^{-T} pi

        Under SOT(3) element (R, a) acting on points as p' = aRp,
        the plane must transform dually to preserve pi^T [p;1] = 0.

        Verify consistency between:
            1. CP transform: q' = a R^T q  (dual inverse)
            2. Homogeneous 4-vector transform
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # Random plane
        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        d = rng.uniform(0.5, 5.0)
        q = n / d  # CP representation

        # Homogeneous plane vector
        pi = np.array([n[0], n[1], n[2], d])

        # Random SOT(3) element
        R, a = random_sot3()

        # Method 1: CP dual inverse action
        q_new = sot3_inverse_action_plane(R, a, q)

        # Method 2: The SOT(3) inverse action on points is p' = (1/a) R^T p.
        # In homogeneous coordinates as a 4x4 matrix:
        #   M = [ (1/a) R^T   0 ]
        #       [    0^T      1 ]
        # Plane transforms as pi' = M^{-T} pi
        M = np.eye(4)
        M[:3, :3] = (1.0 / a) * R.T
        M_inv_T = np.linalg.inv(M).T
        pi_new = M_inv_T @ pi

        # Extract CP from transformed homogeneous vector
        n_new_hom = pi_new[:3]
        d_new_hom = pi_new[3]
        q_new_hom = n_new_hom / d_new_hom

        np.testing.assert_allclose(
            q_new, q_new_hom, atol=1e-12,
            err_msg="CP dual action inconsistent with 4x4 homogeneous transform"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_cp_incidence_via_homogeneous(self, seed):
        """Cross-check: incidence in homogeneous coords and CP coords agree."""
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        n = rng.standard_normal(3)
        n /= np.linalg.norm(n)
        d = rng.uniform(0.5, 5.0)
        q = n / d

        p = random_point_on_plane(q)

        # CP incidence
        assert abs(q @ p + 1) < 1e-12

        # Homogeneous incidence: [n; d]^T [p; 1] = n^T p + d = 0
        # Since q = n/d, we have q^T p + 1 = (n^T p + d) / d
        # So q^T p + 1 = 0 iff n^T p + d = 0
        homogeneous_incidence = n @ p + d
        assert abs(homogeneous_incidence) < 1e-12


# ---------------------------------------------------------------------------
# Test 6: Point lift cross-check (Section 1.3 — existing math, sanity check)
# ---------------------------------------------------------------------------

class TestPointLift:
    """Sanity check for the point lift (existing C++ math).

    NOTE: The point lift velocity involves cross-coupling with the camera pose
    dynamics (parallax term p×v/||p||²). The full EqF tracking condition cannot
    be verified from landmark kinematics alone — it requires the complete state
    propagation. These tests will be completed in step 4 when porting from C++.

    For now we verify the lift's structural properties.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_parallax_term_vanishes_for_distant_point(self, seed):
        """For ||p|| >> ||v||, the parallax term p×v/||p||² → 0,
        so omega_L → -Omega_C (pure rotation tracking)."""
        rng = np.random.default_rng(seed)

        Omega_C = rng.standard_normal(3) * 0.3
        v_C = rng.standard_normal(3) * 0.5
        p = rng.standard_normal(3) * 100 + np.array([0, 0, 100])  # far away

        pp = p @ p
        omega_L = -Omega_C + np.cross(p, v_C) / pp
        sigma_L = -(p @ v_C) / pp

        np.testing.assert_allclose(omega_L, -Omega_C, atol=1e-2,
            err_msg="Parallax term should be negligible for distant points")
        assert abs(sigma_L) < 0.01, "Scale velocity should be small for distant points"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
