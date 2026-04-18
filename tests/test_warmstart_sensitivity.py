#!/usr/bin/env python3
"""Warmstart sensitivity diagnostic for sparse EqVIO.

Drives a VIOFilter with a known forward-moving trajectory, a handful of
well-observed anchor landmarks, and one "test" landmark whose initial
depth we sweep. The test probes whether the InvDepth chart is more
sensitive to a wrong initial depth seed than the Euclidean chart — which
is the leading hypothesis for why FlowDep warmstart regresses InvDepth
accuracy in production while improving Euclidean accuracy.

Run directly (not a pytest test yet):
    python tests/test_warmstart_sensitivity.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from liepp import SO3, SE3

from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vision_measurement import VisionMeasurement
from eqvio.vio_filter import VIOFilter, VIOFilterSettings


GRAV = 9.80665
# Camera looks along world +x (matches tests/test_synthetic_backend.py).
R_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
HOVER_ACC = R_BODY.T @ np.array([0.0, 0.0, GRAV])
V_FWD = 0.5  # world-frame forward velocity (m/s along +x)


def make_pose(x, y=0.0, z=1.5) -> SE3:
    return SE3(R=SO3(matrix=R_BODY), x=np.array([x, y, z], dtype=float))


class PinholeCamera:
    def __init__(self, fx=300.0, fy=300.0, cx=320.0, cy=240.0):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def project_point(self, p):
        # Clamp z so that a degenerate estimate never raises — production
        # cameras just return a (possibly garbage) projection and let the
        # filter's innovation handle it.
        z = p[2] if p[2] > 0.01 else 0.01
        return np.array([self.fx * p[0] / z + self.cx,
                         self.fy * p[1] / z + self.cy])

    def undistort_point(self, px):
        b = np.array([(px[0] - self.cx) / self.fx,
                      (px[1] - self.cy) / self.fy, 1.0])
        return b / np.linalg.norm(b)

    def projection_jacobian(self, p):
        z = max(p[2], 1e-10)
        return np.array([
            [self.fx / z, 0.0, -self.fx * p[0] / (z * z)],
            [0.0, self.fy / z, -self.fy * p[1] / (z * z)],
        ])


class FakeFlowDep:
    """Per-pixel warmstart source used by VIOFilter.process_vision.

    The real FlowDep is queried at the first frame a landmark is added.
    This fake returns a preset (inv_depth, inv_var) for registered pixels
    and (-1, inf) otherwise — which triggers the fallback to
    initial_scene_depth and the default initial_point_variance.
    """

    def __init__(self):
        self._map: dict = {}

    def set_pixel(self, pixel, depth: float, inv_var: float = 1e-4):
        key = (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
        self._map[key] = (1.0 / depth, inv_var)

    def query(self, u: float, v: float):
        key = (int(round(float(u))), int(round(float(v))))
        return self._map.get(key, (-1.0, float("inf")))


def _scene_points() -> dict:
    """6 anchors (well warmstarted) + 1 test landmark."""
    return {
        1: np.array([3.0, -0.4, 0.8]),
        2: np.array([3.0, 0.4, 0.8]),
        3: np.array([3.0, -0.4, 2.2]),
        4: np.array([3.0, 0.4, 2.2]),
        5: np.array([5.0, -0.6, 1.5]),
        6: np.array([5.0, 0.6, 1.5]),
        42: np.array([4.5, 0.2, 1.5]),  # test landmark
    }


TEST_LM_ID = 42


def run_scenario(chart: str, seed_mode: str, seed_factor: float,
                 n_vision: int = 40, *,
                 warm_all: bool = False,
                 pixel_noise: float = 0.0,
                 outlier_thresh: float = 1e6,
                 seed: int = 0,
                 verbose: bool = False) -> dict:
    """Run one scenario and return landmark & pose error history.

    chart:          'Euclidean' | 'InvDepth'
    seed_mode:      'none' — test landmark has no warmstart
                    'warm' — test landmark warmstarted to seed_factor * truth
    seed_factor:    only used when seed_mode='warm'
    warm_all:       if True, ALL landmarks (anchors + test) use seed_factor.
                    Matches the production scenario where every FlowDep query
                    returns a depth with similar systematic bias.
    pixel_noise:    Gaussian sigma added to pixel measurements
    outlier_thresh: pixel-space outlier threshold (vio_filter uses it × sigma_bearing)
    """
    rng = np.random.default_rng(seed)

    s = VIOFilterSettings()
    s.coordinate_choice = chart
    s.camera_offset = SE3.Identity()
    s.initial_scene_depth = 5.0
    s.initial_point_variance = 50.0
    s.sigma_bearing = 1.0
    s.max_landmarks = 20
    s.outlier_mahalanobis_threshold = outlier_thresh
    s.constraint_max_point_var = 0.0

    vio = VIOFilter(s)
    start_x = -2.0
    vio.eqf.xi0.sensor.pose = make_pose(start_x)
    vio.eqf.xi0.sensor.velocity = R_BODY.T @ np.array([V_FWD, 0.0, 0.0])

    camera = PinholeCamera()
    world_points = _scene_points()

    # Build FakeFlowDep keyed on the t=0 pixel projections.
    flowdep = FakeFlowDep()
    T_WtoC0 = make_pose(start_x).inverse()
    for lid, pw in world_points.items():
        pc = T_WtoC0 * pw
        if pc[2] < 0.1:
            continue
        px = camera.project_point(pc)
        true_depth = float(pc[2])
        if lid == TEST_LM_ID:
            if seed_mode == "warm":
                flowdep.set_pixel(px, seed_factor * true_depth)
        else:
            if warm_all and seed_mode == "warm":
                flowdep.set_pixel(px, seed_factor * true_depth)
            elif warm_all and seed_mode == "none":
                pass
            else:
                flowdep.set_pixel(px, true_depth)

    # --- Simulation loop ---
    dt_imu = 0.005
    imu_per_vision = 10  # 20 Hz vision, 200 Hz IMU
    n_imu = n_vision * imu_per_vision + 1
    err_history = []

    prev_active_ids: set = set()
    total_adds = 0
    total_removes = 0

    for i in range(n_imu):
        t = i * dt_imu

        imu = IMUVelocity(
            stamp=t,
            gyr=np.zeros(3),
            acc=HOVER_ACC.copy(),
            gyr_bias_vel=np.zeros(3),
            acc_bias_vel=np.zeros(3),
        )
        vio.process_imu(imu)

        if i % imu_per_vision != 0:
            continue

        true_pose = make_pose(start_x + V_FWD * t)
        T_WtoC = true_pose.inverse()
        cam_coords = {}
        for lid, pw in world_points.items():
            pc = T_WtoC * pw
            if pc[2] < 0.1:
                continue
            try:
                cam_coords[lid] = camera.project_point(pc)
            except ValueError:
                continue
        if not cam_coords:
            continue

        if pixel_noise > 0.0:
            for fid in cam_coords:
                cam_coords[fid] = cam_coords[fid] + rng.standard_normal(2) * pixel_noise

        meas = VisionMeasurement(stamp=t)
        meas.cam_coordinates = cam_coords
        meas.camera_ptr = camera
        try:
            vio.process_vision(meas, camera, flowdep=flowdep)
        except np.linalg.LinAlgError:
            break

        # --- Errors ---
        xi_hat = vio.state_estimate()
        pose_err = float(np.linalg.norm(xi_hat.sensor.pose.x - true_pose.x))

        # Track churn (IDs added/removed since previous frame)
        curr_ids = {lm.id for lm in xi_hat.camera_landmarks}
        adds = len(curr_ids - prev_active_ids)
        removes = len(prev_active_ids - curr_ids)
        total_adds += adds
        total_removes += removes
        prev_active_ids = curr_ids

        lm_est_cam = None
        for lm in xi_hat.camera_landmarks:
            if lm.id == TEST_LM_ID:
                lm_est_cam = lm.p
                break
        if lm_est_cam is not None:
            lm_true_cam = T_WtoC * world_points[TEST_LM_ID]
            total_err = float(np.linalg.norm(lm_est_cam - lm_true_cam))
            depth_err = float(abs(lm_est_cam[2] - lm_true_cam[2]))
        else:
            total_err = float("nan")
            depth_err = float("nan")

        n_active = len(xi_hat.camera_landmarks)
        err_history.append({
            "t": t,
            "total": total_err,
            "depth": depth_err,
            "pose": pose_err,
            "n_active": n_active,
        })

        if verbose:
            print(f"    t={t:.2f}  total={total_err:.3f}  "
                  f"depth_err={depth_err:.3f}  pose_err={pose_err:.3f}  "
                  f"n_active={n_active}")

    return {"chart": chart, "seed_mode": seed_mode, "seed_factor": seed_factor,
            "history": err_history,
            "total_adds": total_adds, "total_removes": total_removes}


def _err_at(history, t_target: float, key: str = "total") -> float:
    if not history:
        return float("nan")
    idx = min(range(len(history)), key=lambda i: abs(history[i]["t"] - t_target))
    return history[idx][key]


def _print_table(title, rows, configs):
    print()
    print(title)
    hdr = (f"{'chart':10s} {'seed':6s} {'factor':>7s} "
           f"{'tot@2s':>9s} {'pose@2s':>9s} {'n_act':>6s} "
           f"{'adds':>6s} {'rmvs':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for (chart, seed_mode, factor), res in zip(configs, rows):
        hist = res["history"]
        last = hist[-1] if hist else {"total": float("nan"),
                                      "pose": float("nan"), "n_active": 0}
        print(
            f"{chart:10s} {seed_mode:6s} {factor:7.2f} "
            f"{_err_at(hist, 2.0):9.4f} "
            f"{_err_at(hist, 2.0, 'pose'):9.4f} "
            f"{int(last['n_active']):6d} "
            f"{res['total_adds']:6d} {res['total_removes']:6d}"
        )


def run_depth_drift_scenario(
    chart: str,
    bias_px: float,
    n_clean: int = 15,
    n_biased: int = 30,
    outlier_thresh: float = 5.0,
    sigma_bearing: float = 1.0,
    bias_dir: tuple = (1.0, 0.0),
    sigma_lock_thresh: float = None,
    drift_m_thresh: float = 1.0,
) -> dict:
    """Drive a converged landmark with a constant pixel offset and watch
    whether the outlier gate fires and whether the depth estimate drifts.

    Design:
      1. Run `n_clean` frames with perfect data so all landmarks converge.
      2. Then for `n_biased` frames, inject a constant `bias_px` offset
         on the test landmark's observed pixel (in direction `bias_dir`)
         while keeping anchors clean.
      3. Log per biased frame: pre-update pixel innovation (the value the
         gate sees), whether the gate would fire (innov_norm > thresh*sigma),
         the landmark's estimated depth, and the pose error.
    """
    s = VIOFilterSettings()
    s.coordinate_choice = chart
    s.camera_offset = SE3.Identity()
    s.initial_scene_depth = 5.0
    s.initial_point_variance = 50.0
    s.sigma_bearing = sigma_bearing
    s.max_landmarks = 20
    s.outlier_mahalanobis_threshold = outlier_thresh
    s.constraint_max_point_var = 0.0

    vio = VIOFilter(s)
    start_x = -2.0
    vio.eqf.xi0.sensor.pose = make_pose(start_x)
    vio.eqf.xi0.sensor.velocity = R_BODY.T @ np.array([V_FWD, 0.0, 0.0])

    camera = PinholeCamera()
    world_points = _scene_points()

    dt_imu = 0.005
    imu_per_vision = 10
    bias_vec = np.array(bias_dir, dtype=float)
    if np.linalg.norm(bias_vec) > 0:
        bias_vec = bias_vec / np.linalg.norm(bias_vec) * bias_px

    threshold_px = outlier_thresh * sigma_bearing

    # Lock-in / drift guard state (observer-only — does not mutate filter).
    # sigma_lock_thresh is chart-native (see docs/chart_initial_cov.md).
    if sigma_lock_thresh is None:
        sigma_lock_thresh = 0.5 if chart == "Euclidean" else 0.05
    lock_depth_m = None
    lock_sigma_chart = None
    lock_frame = -1
    guard_fire_frame = -1

    history = []
    vision_idx = 0
    n_vision_total = n_clean + n_biased
    n_imu = n_vision_total * imu_per_vision + 1

    for i in range(n_imu):
        t = i * dt_imu

        imu = IMUVelocity(
            stamp=t, gyr=np.zeros(3), acc=HOVER_ACC.copy(),
            gyr_bias_vel=np.zeros(3), acc_bias_vel=np.zeros(3),
        )
        vio.process_imu(imu)

        if i % imu_per_vision != 0:
            continue

        true_pose = make_pose(start_x + V_FWD * t)
        T_WtoC = true_pose.inverse()
        cam_coords = {}
        for lid, pw in world_points.items():
            pc = T_WtoC * pw
            if pc[2] < 0.1:
                continue
            try:
                cam_coords[lid] = camera.project_point(pc)
            except ValueError:
                continue
        if not cam_coords:
            vision_idx += 1
            continue

        biased = vision_idx >= n_clean
        if biased and TEST_LM_ID in cam_coords:
            cam_coords[TEST_LM_ID] = cam_coords[TEST_LM_ID] + bias_vec

        # --- pre-update innovation + gate check on the test landmark ---
        xi_pre = vio.state_estimate()
        pre_lm_in_state = any(lm.id == TEST_LM_ID for lm in xi_pre.camera_landmarks)
        if pre_lm_in_state and TEST_LM_ID in cam_coords:
            lm_est = next(
                lm for lm in xi_pre.camera_landmarks if lm.id == TEST_LM_ID
            )
            pred_px = camera.project_point(lm_est.p)
            innov = cam_coords[TEST_LM_ID] - pred_px
            innov_norm = float(np.linalg.norm(innov))
            gate_fires = innov_norm > threshold_px
            est_depth_cam = float(lm_est.p[2])
        else:
            innov_norm = float("nan")
            gate_fires = False
            est_depth_cam = float("nan")

        meas = VisionMeasurement(stamp=t)
        meas.cam_coordinates = cam_coords
        meas.camera_ptr = camera
        try:
            vio.process_vision(meas, camera, flowdep=None)
        except np.linalg.LinAlgError:
            break

        xi_hat = vio.state_estimate()
        pose_err = float(np.linalg.norm(xi_hat.sensor.pose.x - true_pose.x))
        post_lm_in_state = any(lm.id == TEST_LM_ID for lm in xi_hat.camera_landmarks)
        if post_lm_in_state:
            lm_post = next(
                lm for lm in xi_hat.camera_landmarks if lm.id == TEST_LM_ID
            )
            post_depth_cam = float(lm_post.p[2])
            post_depth_range = float(np.linalg.norm(lm_post.p))
            try:
                lm_cov = vio.eqf.get_landmark_cov_by_id(TEST_LM_ID)
                sigma_depth_chart = float(np.sqrt(max(lm_cov[2, 2], 0.0)))
            except (StopIteration, KeyError):
                sigma_depth_chart = float("nan")
        else:
            post_depth_cam = float("nan")
            post_depth_range = float("nan")
            sigma_depth_chart = float("nan")

        # Lock-in / drift guard observer
        drift_from_lock = float("nan")
        locked = lock_depth_m is not None
        guard_would_fire = False
        if post_lm_in_state:
            if not locked and sigma_depth_chart == sigma_depth_chart \
                    and sigma_depth_chart < sigma_lock_thresh:
                lock_depth_m = post_depth_range
                lock_sigma_chart = sigma_depth_chart
                lock_frame = vision_idx
                locked = True
            if locked:
                drift_from_lock = abs(post_depth_range - lock_depth_m)
                if drift_from_lock > drift_m_thresh:
                    guard_would_fire = True
                    if guard_fire_frame < 0:
                        guard_fire_frame = vision_idx

        true_pc = T_WtoC * world_points[TEST_LM_ID]
        true_depth_cam = float(true_pc[2])

        history.append({
            "vidx": vision_idx,
            "biased": biased,
            "innov_norm": innov_norm,
            "gate_fires": gate_fires,
            "pre_depth": est_depth_cam,
            "post_depth": post_depth_cam,
            "post_range": post_depth_range,
            "true_depth": true_depth_cam,
            "pose_err": pose_err,
            "in_state": post_lm_in_state,
            "sigma_depth_chart": sigma_depth_chart,
            "locked": locked,
            "drift_from_lock": drift_from_lock,
            "guard_would_fire": guard_would_fire,
        })

        vision_idx += 1

    return {
        "chart": chart,
        "bias_px": bias_px,
        "history": history,
        "lock_frame": lock_frame,
        "lock_depth_m": lock_depth_m,
        "lock_sigma_chart": lock_sigma_chart,
        "guard_fire_frame": guard_fire_frame,
        "sigma_lock_thresh": sigma_lock_thresh,
        "drift_m_thresh": drift_m_thresh,
    }


def _print_drift_table(title, results):
    print()
    print(title)
    hdr = (f"{'chart':10s} {'bias':>5s} {'pxgate':>7s} "
           f"{'lockf':>5s} {'σlock':>7s} {'d_lock':>7s} "
           f"{'max_drift':>10s} {'guard@':>7s} {'r_end':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        hist = res["history"]
        if not hist:
            continue
        biased = [h for h in hist if h["biased"]]
        fires = sum(1 for h in biased if h["gate_fires"])
        max_drift = max(
            (h["drift_from_lock"] for h in hist
             if h["drift_from_lock"] == h["drift_from_lock"]),
            default=float("nan"),
        )
        last = hist[-1]
        lockf = res["lock_frame"]
        lock_d = res["lock_depth_m"]
        lock_s = res["lock_sigma_chart"]
        gfire = res["guard_fire_frame"]
        print(
            f"{res['chart']:10s} {res['bias_px']:5.1f} "
            f"{fires:>3d}/{len(biased):<2d}  "
            f"{lockf:>5d} "
            f"{(lock_s if lock_s is not None else float('nan')):7.4f} "
            f"{(lock_d if lock_d is not None else float('nan')):7.3f} "
            f"{max_drift:10.4f} "
            f"{gfire:>7d} "
            f"{last['post_range']:7.3f}"
        )


def main():
    charts = ["Euclidean", "InvDepth"]
    factors = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]

    print()
    print("Warmstart sensitivity diagnostic")
    print(f"  scene: 6 anchors + 1 test landmark (id={TEST_LM_ID})")
    print(f"  motion: camera +x at {V_FWD} m/s, 40 vision frames @ 20 Hz")

    # --- Suite 1: ONLY test landmark warmstart varies; anchors at truth. -----
    print()
    print("=" * 70)
    print("Suite 1: anchors warmstarted to truth; test LM factor swept")
    print("  noise = 0 px, outlier gate DISABLED")
    configs = [(c, "none", 1.0) for c in charts] + \
              [(c, "warm", f) for c in charts for f in factors]
    rows = [run_scenario(*cfg, n_vision=40) for cfg in configs]
    _print_table("", rows, configs)

    # --- Suite 2: ALL landmarks warmstarted with same factor. -----------------
    print()
    print("=" * 70)
    print("Suite 2: ALL landmarks warmstarted with seed_factor (warm_all=True)")
    print("  noise = 0 px, outlier gate DISABLED")
    configs = [(c, "none", 1.0) for c in charts] + \
              [(c, "warm", f) for c in charts for f in factors]
    rows = [run_scenario(*cfg, n_vision=40, warm_all=True) for cfg in configs]
    _print_table("", rows, configs)

    # --- Suite 3a: noise only (NO outlier gate) -------------------------------
    print()
    print("=" * 70)
    print("Suite 3a: ALL landmarks warmstarted; noise=1 px; outlier gate OFF")
    configs = [(c, "none", 1.0) for c in charts] + \
              [(c, "warm", f) for c in charts for f in factors]
    rows = [run_scenario(*cfg, n_vision=40, warm_all=True,
                         pixel_noise=1.0, outlier_thresh=1e6, seed=7)
            for cfg in configs]
    _print_table("", rows, configs)

    # --- Suite 3b: outlier gate only (no noise) -------------------------------
    print()
    print("=" * 70)
    print("Suite 3b: ALL landmarks warmstarted; noise OFF; outlier gate=5.0")
    configs = [(c, "none", 1.0) for c in charts] + \
              [(c, "warm", f) for c in charts for f in factors]
    rows = [run_scenario(*cfg, n_vision=40, warm_all=True,
                         pixel_noise=0.0, outlier_thresh=5.0, seed=7)
            for cfg in configs]
    _print_table("", rows, configs)

    # --- Suite 3c: BOTH noise + outlier gate (production-like) ----------------
    print()
    print("=" * 70)
    print("Suite 3c: ALL landmarks warmstarted; noise=1 px; outlier gate=5.0")
    configs = [(c, "none", 1.0) for c in charts] + \
              [(c, "warm", f) for c in charts for f in factors]
    rows = [run_scenario(*cfg, n_vision=40, warm_all=True,
                         pixel_noise=1.0, outlier_thresh=5.0, seed=7)
            for cfg in configs]
    _print_table("", rows, configs)

    # --- Suite 4: depth drift under sustained pixel bias ----------------------
    print()
    print("=" * 70)
    print("Suite 4: converge for 15 frames, then inject constant bias on LM 42")
    print("  sigma_bearing=1.0, outlier_thresh=5.0 (gate threshold = 5.0 px)")
    print("  'fires' counts the number of biased frames where gate would fire")
    biases = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    for c in charts:
        results = [
            run_depth_drift_scenario(
                chart=c, bias_px=b, n_clean=15, n_biased=30,
                outlier_thresh=5.0, sigma_bearing=1.0,
            )
            for b in biases
        ]
        _print_drift_table(f"  {c}", results)
    print()


if __name__ == "__main__":
    main()
