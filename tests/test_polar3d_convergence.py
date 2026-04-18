"""Side-by-side comparison of POLAR (1D) vs POLAR3D (3D sequential) filter.

After the sequential decoupled update fix:
  Step 1: bearing IEKF (standard Kalman, no mixture) → q_mid, P_mid
  Step 2: depth Vogiatzis (1D scalar mixture on log-depth) → q_new, P_new

This test verifies that:
  - 3D depth_var now converges comparably to 1D
  - a/b tracks inlier status correctly
  - Cross-covariance from Step 1 improves depth via Step 2
"""

import math
import numpy as np
from eqvio.sparse_vogiatzis import (
    SparseVogiatzisFilter,
    SparseVogiatzisFilter3D,
    SparseVogSettings,
    DepthParametrization,
)
from eqvio.mathematical.vision_measurement import VisionMeasurement


class FakePinholeCamera:
    def __init__(self, fx, fy, cx, cy, w, h):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.image_size = (w, h)

    def K_matrix(self):
        K = np.eye(3)
        K[0, 0], K[1, 1] = self.fx, self.fy
        K[0, 2], K[1, 2] = self.cx, self.cy
        return K

    def project_point(self, p):
        return np.array([
            self.fx * p[0] / p[2] + self.cx,
            self.fy * p[1] / p[2] + self.cy,
        ])

    def undistort_point(self, uv):
        r = np.array([
            (uv[0] - self.cx) / self.fx,
            (uv[1] - self.cy) / self.fy,
            1.0,
        ])
        return r / np.linalg.norm(r)

    def projection_jacobian(self, p):
        x, y, z = p
        return np.array([
            [self.fx / z, 0.0, -self.fx * x / (z * z)],
            [0.0, self.fy / z, -self.fy * y / (z * z)],
        ])


FX, FY, CX, CY = 458.0, 458.0, 376.0, 240.0
W, H = 752, 480


def make_K():
    return np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


def make_T_WC(position):
    T = np.eye(4)
    T[:3, 3] = position
    return T


def project(K, p_cam):
    return np.array([
        K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
        K[1, 0] * p_cam[1] / p_cam[2] + K[1, 2],
    ])


def make_settings(param):
    return SparseVogSettings(
        parametrization=param,
        max_pool_size=10,
        min_track_length=3,
        conv_inlier_ratio=0.7,
        conv_variance_threshold=0.5,
        init_depth_var=1.0,
        sigma_pixel=0.5,
        a_init=10.0,
        b_init=2.0,
        min_parallax=1e-4,
        min_cos_sim=0.95,
        min_depth=0.1,
        max_depth=100.0,
        process_depth_var=0.01,
        mahalanobis_reset_chi2=9.0,
        min_inlier_ratio=0.5,
        ab_min=1.0,
        ab_max=20.0,
    )


def generate_observations(n_frames=30, seed=42):
    """Generate synthetic pixel observations of a point at [1, 0.5, 3]."""
    K = make_K()
    p_world = np.array([1.0, 0.5, 3.0])
    baseline = 0.05
    np.random.seed(seed)
    frames = []
    for i in range(n_frames):
        cam_pos = np.array([i * baseline, 0.0, 0.0])
        T_WC = make_T_WC(cam_pos)
        p_cam = p_world - cam_pos
        uv = project(K, p_cam)
        uv_noisy = uv + np.random.randn(2) * 0.3
        frames.append((float(i) * 0.05, T_WC, uv_noisy))
    return frames


def run_filter(filt, frames, fid=42, is_3d=False):
    """Run filter on frames, return per-frame (depth, depth_var, a/(a+b))."""
    results = []
    for stamp, T_WC, uv_noisy in frames:
        meas = VisionMeasurement(
            stamp=stamp, cam_coordinates={fid: uv_noisy},
        )
        filt.update(meas, T_WC)

        if is_3d:
            feat = filt._features3d.get(fid)
            if feat is not None:
                ab = feat.a + feat.b
                q_z, q_var = filt.query(fid)
                results.append((feat.depth, feat.depth_var,
                                feat.a / ab if ab > 0 else 0,
                                np.diag(feat.covariance).copy(),
                                q_z > 0))
            else:
                results.append(None)
        else:
            feat = filt._features.get(fid)
            if feat is not None:
                ab = feat.a + feat.b
                z = filt._canonical_to_depth(feat.canonical)
                zvar = filt._canonical_var_to_euclidean(
                    feat.canonical, feat.canonical_var, z)
                q_z, _ = filt.query(fid)
                results.append((z, zvar, feat.a / ab if ab > 0 else 0,
                                np.array([feat.canonical_var]), q_z > 0))
            else:
                results.append(None)
    return results


def main():
    K = make_K()
    cam = FakePinholeCamera(FX, FY, CX, CY, W, H)
    frames = generate_observations()

    # --- 3D filter (cam_ptr, sequential update) ---
    filt3d = SparseVogiatzisFilter3D(K, make_settings(DepthParametrization.POLAR3D),
                                      cam_ptr=cam)
    res3d = run_filter(filt3d, frames, is_3d=True)

    # --- 1D Polar filter ---
    filt1d = SparseVogiatzisFilter(K, make_settings(DepthParametrization.POLAR))
    frames1d = generate_observations()  # same seed → same noise
    res1d = run_filter(filt1d, frames1d, is_3d=False)

    # --- Print ---
    print(f"{'':>3}  {'--- POLAR3D (sequential) ---':^50}  {'--- POLAR (1D) ---':^35}")
    print(f"{'fr':>3}  {'depth':>7} {'dvar':>8} {'a/ab':>6} "
          f"{'cov[0]':>8} {'cov[1]':>8} {'cov[2]':>8} {'conv':>4}"
          f"  {'depth':>7} {'dvar':>8} {'a/ab':>6} {'σ²_d':>10} {'conv':>4}")
    print("-" * 105)

    for i in range(len(frames)):
        r3 = res3d[i] if i < len(res3d) else None
        r1 = res1d[i] if i < len(res1d) else None

        s3 = ""
        if r3 is not None:
            d, dv, ab, cov, conv = r3
            s3 = (f"{d:7.3f} {dv:8.4f} {ab:6.3f} "
                  f"{cov[0]:8.4f} {cov[1]:8.4f} {cov[2]:8.4f} "
                  f"{'Y' if conv else 'n':>4}")
        else:
            s3 = f"{'(init)':>50}"

        s1 = ""
        if r1 is not None:
            d, dv, ab, cov, conv = r1
            s1 = (f"{d:7.3f} {dv:8.4f} {ab:6.3f} "
                  f"{cov[0]:10.2e} {'Y' if conv else 'n':>4}")
        else:
            s1 = f"{'(init)':>35}"

        print(f"{i:3d}  {s3}  {s1}")

    # --- Final summary ---
    print()
    f3 = filt3d._features3d.get(42)
    f1 = filt1d._features.get(42)
    if f3 and f1:
        z1 = math.exp(f1.canonical)
        zvar1 = z1 * z1 * f1.canonical_var
        print(f"Final POLAR3D:  depth={f3.depth:.3f}  depth_var={f3.depth_var:.4f}  "
              f"a/(a+b)={f3.a/(f3.a+f3.b):.3f}")
        print(f"Final POLAR:    depth={z1:.3f}  depth_var={zvar1:.4f}  "
              f"a/(a+b)={f1.a/(f1.a+f1.b):.3f}")
        print(f"True depth:     3.000")
        ratio = f3.depth_var / zvar1 if zvar1 > 0 else float('inf')
        print(f"depth_var ratio (3D/1D): {ratio:.2f}x")


if __name__ == "__main__":
    main()
