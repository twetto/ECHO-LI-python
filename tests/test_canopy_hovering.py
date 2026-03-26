#!/usr/bin/env python3
"""
Synthetic test: drone hovering at 60m above noisy tree canopy.

Two plane modes:
    GT:     Ground-truth plane association (bypasses RANSAC)
    RANSAC: Full pipeline — PlaneDetector + fit_plane_ransac + augment_planes

Compares point-only vs plane-augmented filter across canopy roughness levels.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from liepp import SO3, SE3, SOT3
from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark,
)
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vision_measurement import VisionMeasurement
from eqvio.vio_filter import VIOFilter, VIOFilterSettings
from eqvio.plane_detection.plane_detector import (
    PlaneDetector, PlaneDetectorSettings, landmarks_to_global,
)
from eqvio.plane_detection.plane_fitting import (
    fit_detected_planes, PlaneFittingSettings,
)

# Visualizer helper
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets
import time

class SyntheticVisualiser:
    def __init__(self, true_points, cam_dist=10.0, elevation=30):
        self.app = pg.mkQApp("Synthetic EqF Backend")
        pg.setConfigOptions(antialias=True)
        self.main_window = QtWidgets.QWidget()
        self.main_window.resize(800, 800)
        self.layout = QtWidgets.QVBoxLayout()
        self.main_window.setLayout(self.layout)

        self.w = gl.GLViewWidget()
        self.w.setCameraPosition(distance=cam_dist, elevation=elevation, azimuth=-90)
        self.layout.addWidget(self.w)

        self.w.addItem(gl.GLGridItem(size=pg.Vector(100, 100, 1)))

        self.gt_line = gl.GLLinePlotItem(color=(0, 1, 1, 0.7), width=2.0)
        self.est_line = gl.GLLinePlotItem(color=(1, 0, 0, 1.0), width=2.5)
        self.w.addItem(self.gt_line)
        self.w.addItem(self.est_line)

        # Plot true geometry in grey
        gt_pts = np.array(list(true_points.values()), dtype=np.float32)
        self.w.addItem(gl.GLScatterPlotItem(pos=gt_pts, color=(0.5, 0.5, 0.5, 0.5), size=3.0))

        # Estimated landmarks in yellow
        self.est_points = gl.GLScatterPlotItem(color=(1.0, 1.0, 0.0, 1.0), size=5.0)
        self.w.addItem(self.est_points)

        self.plane_items = {}
        self.plane_colors = [(0, 1, 1, 0.4), (1, 0, 1, 0.4), (0, 1, 0, 0.4)]
        self.main_window.show()

        self.gt_path = []
        self.est_path = []

        self.paused = False
        # Hook into the main window's key press event
        self.main_window.keyPressEvent = self._key_pressed

    def update(self, state, true_pose):
        self.gt_path.append(true_pose.x.copy())
        self.est_path.append(state.sensor.pose.x.copy())
        
        if len(self.gt_path) > 1:
            self.gt_line.setData(pos=np.array(self.gt_path, dtype=np.float32))
            self.est_line.setData(pos=np.array(self.est_path, dtype=np.float32))

        T_CtoG = state.sensor.pose * state.sensor.camera_offset
        R_CtoG = T_CtoG.R.asMatrix()
        t_CtoG = T_CtoG.x

        if state.camera_landmarks:
            pts = np.array([lm.p for lm in state.camera_landmarks], dtype=np.float32)
            self.est_points.setData(pos=(R_CtoG @ pts.T).T + t_CtoG)
        else:
            self.est_points.setData(pos=np.empty((0, 3)))

        current_pids = set()
        active_lm_dict = {lm.id: lm.p for lm in state.camera_landmarks}

        for plane in state.plane_landmarks:
            pid = plane.id
            current_pids.add(pid)
            q_cam = plane.q
            q_norm_sq = np.dot(q_cam, q_cam)
            if q_norm_sq < 1e-8: continue

            n_cam = q_cam / np.sqrt(q_norm_sq)
            cp_cam = -q_cam / q_norm_sq
            n_global = R_CtoG @ n_cam
            cp_global = R_CtoG @ cp_cam + t_CtoG
            d_global = -np.dot(n_global, cp_global)

            pts_global = []
            if hasattr(plane, 'point_ids'):
                for pt_id in plane.point_ids:
                    if pt_id in active_lm_dict:
                        pts_global.append(R_CtoG @ active_lm_dict[pt_id] + t_CtoG)

            if len(pts_global) < 3: continue
            pts_global = np.array(pts_global)

            dists = np.dot(pts_global, n_global) + d_global
            pts_proj = pts_global - np.outer(dists, n_global)
            centroid = np.mean(pts_proj, axis=0)

            z_axis = n_global
            up = np.array([1., 0., 0.]) if abs(z_axis[0]) < 0.9 else np.array([0., 1., 0.])
            x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            vecs = pts_proj - centroid
            angles = np.arctan2(np.dot(vecs, y_axis), np.dot(vecs, x_axis))
            sort_idx = np.argsort(angles)
            sorted_pts = pts_proj[sort_idx]

            verts = np.vstack((centroid, sorted_pts))
            n_pts = len(sorted_pts)

            faces = []
            for i in range(1, n_pts):
                faces.append([0, i, i + 1]) # Front
                faces.append([0, i + 1, i]) # Back
            faces.append([0, n_pts, 1])
            faces.append([0, 1, n_pts])
            faces = np.array(faces, dtype=np.uint)

            if pid not in self.plane_items:
                color = self.plane_colors[pid % len(self.plane_colors)]
                mesh = gl.GLMeshItem(
                    meshdata=gl.MeshData(vertexes=verts, faces=faces),
                    color=color, smooth=False, shader='shaded',
                    glOptions='translucent', drawEdges=True, edgeColor=(1, 1, 1, 0.6)
                )
                self.plane_items[pid] = mesh
                self.w.addItem(mesh)
            else:
                self.plane_items[pid].setMeshData(vertexes=verts, faces=faces)

        for pid, mesh in list(self.plane_items.items()):
            if pid not in current_pids:
                mesh.hide()

        self.app.processEvents()

    def wait(self, delay_ms: int):
        """
        Pauses the loop for `delay_ms` milliseconds while keeping the GUI responsive.
        Acts exactly like cv2.waitKey().
        """
        import time
        start_time = time.time()
        delay_sec = delay_ms / 1000.0
        
        #while time.time() - start_time < delay_sec:
        while (time.time() - start_time < delay_sec) or self.paused:
            # 1. Pump the Qt event loop so the 3D view remains interactive
            self.app.processEvents()
            # 2. Tiny sleep to prevent the while loop from maxing out a CPU core
            time.sleep(0.005)
            
            if self.paused:
                start_time = time.time()

    def _key_pressed(self, event):
        """Toggle pause when the Spacebar is pressed."""
        from PyQt5.QtCore import Qt
        if event.key() == Qt.Key_Space:
            self.paused = not self.paused
            if self.paused:
                print("Simulation paused. Press Space to resume.")

class PinholeCamera:
    def __init__(self, fx=300., fy=300., cx=320., cy=240.):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
    def project_point(self, p):
        if p[2] < 0.01: raise ValueError("Behind camera")
        return np.array([self.fx*p[0]/p[2]+self.cx, self.fy*p[1]/p[2]+self.cy])
    def undistort_point(self, px):
        b = np.array([(px[0]-self.cx)/self.fx, (px[1]-self.cy)/self.fy, 1.0])
        return b / np.linalg.norm(b)
    def projection_jacobian(self, p):
        z = max(p[2], 1e-10)
        return np.array([[self.fx/z, 0, -self.fx*p[0]/z**2],
                         [0, self.fy/z, -self.fy*p[1]/z**2]])


# cam_x->world_x, cam_y->world_-y, cam_z->world_-z (looking down)
R_DOWN = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
HOVER_ACC = R_DOWN.T @ np.array([0., 0., 9.80665])


def make_pose(x, y, altitude=60.0):
    return SE3(R=SO3(matrix=R_DOWN), x=np.array([x, y, altitude]))


def make_canopy_points(n_points=40, spread=30.0, tree_height_std=0.0, seed=None):
    rng = np.random.RandomState(seed if seed is not None else 12345)
    pts = {}
    for i in range(n_points):
        x = rng.uniform(-spread, spread)
        y = rng.uniform(-spread, spread)
        z = rng.randn() * tree_height_std
        pts[3000 + i] = np.array([x, y, z])
    return pts


def generate_measurements(pose, camera, world_points, sigma_px=0.5):
    T_WtoC = pose.inverse()
    cam_coords = {}
    for fid, pw in world_points.items():
        pc = T_WtoC * pw
        if pc[2] < 0.1: continue
        try:
            px = camera.project_point(pc)
            if 0 <= px[0] <= 640 and 0 <= px[1] <= 480:
                cam_coords[fid] = px + np.random.randn(2) * sigma_px
        except ValueError: continue
    return cam_coords


# ---------------------------------------------------------------------------
# GT plane augmentation (oracle, for comparison)
# ---------------------------------------------------------------------------

def augment_gt_plane(vio, canopy_pts, tree_height_std):
    """Augment with GT canopy plane."""
    xi_hat = vio.state_estimate()
    active = set(vio.eqf.X.id)
    existing = set(vio.eqf.X.plane_id)

    if 300 in existing:
        for pl in vio.eqf.xi0.plane_landmarks:
            if pl.id == 300:
                pl.point_ids = [f for f in canopy_pts if f in active]
        return

    T_WtoC = (xi_hat.sensor.pose * xi_hat.sensor.camera_offset).inverse()
    R_WtoC = T_WtoC.R.asMatrix()
    p_CinW = xi_hat.sensor.pose.x

    n_W, d_W = np.array([0., 0., 1.]), 0.0
    n_C = R_WtoC @ n_W
    d_C = d_W + n_W @ p_CinW

    if abs(d_C) < 0.1: return

    q_cam = n_C / d_C
    ids = [f for f in canopy_pts if f in active]
    if len(ids) < 4: return

    plane_var = max(0.01, (tree_height_std / 60.0) ** 2)
    vio.eqf.add_new_plane_landmarks(
        [PlaneLandmark(q=q_cam, id=300, point_ids=ids)],
        np.eye(3) * plane_var
    )
    vio._invalidate_gain_cache()


# ---------------------------------------------------------------------------
# RANSAC plane augmentation (real pipeline)
# ---------------------------------------------------------------------------

def make_ransac_settings(altitude, tree_height_std):
    """Altitude-scaled RANSAC settings."""
    s = PlaneFittingSettings()
    # Inlier threshold: at altitude h, bearing-initialized points deviate
    # from the plane by ~h * tan(FOV/2) * (angular_error).
    # Use 10% of altitude as baseline, plus tree roughness.
    s.ransac_inlier_threshold = max(altitude * 0.1, 2.0 * tree_height_std)
    s.ransac_min_point_separation = 1.0
    s.ransac_min_inlier_ratio = 0.50
    s.min_cp_distance = 1.0
    return s


def make_detector_settings():
    """Permissive detector for canopy at altitude."""
    s = PlaneDetectorSettings()
    s.max_norm_deg = 60.0        # canopy normals vary a lot
    s.max_dist_between_z = 20.0  # tall trees = large z gaps between co-planar points
    s.min_plane_features = 4
    s.max_tri_side_px = 500      # wide FOV at 60m
    s.max_pairwise_px = 300      # points spread over ~230px at 60m
    s.max_norm_avg_var = 60.0    # rough surface → high normal variance
    s.max_norm_avg_max = 60.0    # same
    s.min_norms = 1              # accept single-frame normals (stable geometry)
    return s


def augment_ransac_plane(vio, plane_detector, fitting_settings, camera):
    """Run RANSAC fitting directly on all landmarks, bypass detector.

    At high altitude, Delaunay normal matching fails because landmarks
    initialized along bearings form a sphere. Direct RANSAC on 3D
    positions still finds the plane.
    """
    state = vio.state_estimate()
    feat_pos, cam_pos, R_GtoC = landmarks_to_global(state)

    if len(feat_pos) < 5:
        return 0, 0

    # Run RANSAC directly on all landmarks as one candidate plane
    from eqvio.plane_detection.plane_fitting import fit_plane_ransac
    all_fids = sorted(feat_pos.keys())
    all_pts = np.array([feat_pos[fid] for fid in all_fids])

    ok, cp, inlier_fids = fit_plane_ransac(all_fids, all_pts, fitting_settings)
    if not ok:
        return 0, 0

    # Use a single synthetic plane_id
    plane_cps = {500: cp}
    plane_inliers = {500: inlier_fids}

    vio.augment_planes(plane_cps, plane_inliers)

    return 1, 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(use_planes=False, plane_mode='gt', sigma_constraint=0.5,
            tree_height_std=0.0, altitude=60.0, drift_speed=0.3,
            sigma_px=0.5, seed=42, verbose=False, visualize=False):
    np.random.seed(seed)
    camera = PinholeCamera()
    canopy = make_canopy_points(n_points=40, spread=25.0,
                                tree_height_std=tree_height_std, seed=99)

    vis = SyntheticVisualiser(canopy, cam_dist=120.0, elevation=45) if visualize else None

    s = VIOFilterSettings()
    s.camera_offset = SE3.Identity()
    s.initial_scene_depth = altitude
    s.initial_point_variance = altitude ** 2
    s.sigma_bearing = sigma_px
    s.max_landmarks = 50
    s.outlier_mahalanobis_threshold = 1e6
    s.sigma_constraint = sigma_constraint
    s.initial_plane_variance = 1.0

    vio = VIOFilter(s)
    vio.eqf.xi0.sensor.pose = make_pose(0.0, 0.0, altitude + 2.0)

    # RANSAC pipeline objects
    plane_detector = None
    fitting_settings = None
    if plane_mode == 'ransac':
        plane_detector = PlaneDetector(make_detector_settings())
        fitting_settings = make_ransac_settings(altitude, tree_height_std)

    pos_errors, z_errors, lm_errors = [], [], []
    vc = 0

    for i in range(300):
        t = i * 0.05
        x = drift_speed * t
        y = 0.5 * np.sin(0.3 * t)
        true_pose = make_pose(x, y, altitude)

        imu = IMUVelocity(stamp=t, gyr=np.zeros(3), acc=HOVER_ACC.copy(),
                          gyr_bias_vel=np.zeros(3), acc_bias_vel=np.zeros(3))
        vio.process_imu(imu)

        if i % 5 != 0: continue
        cam_coords = generate_measurements(true_pose, camera, canopy, sigma_px=sigma_px)
        if not cam_coords: continue

        meas = VisionMeasurement(stamp=t)
        meas.cam_coordinates = cam_coords
        meas.camera_ptr = camera

        try:
            vio.process_vision(meas, camera)
        except Exception as e:
            if verbose: print(f"  t={t:.1f}: {e}")
            continue

        vc += 1
        state = vio.state_estimate()

        if use_planes and vc >= 3:
            if plane_mode == 'gt':
                augment_gt_plane(vio, canopy, tree_height_std)
            elif plane_mode == 'ransac':
                n_det, n_fit = augment_ransac_plane(
                    vio, plane_detector, fitting_settings, camera
                )
                if verbose and vc <= 12:
                    n_pl = len(vio.eqf.xi0.plane_landmarks)
                    print(f"    RANSAC: detected={n_det}, fitted={n_fit}, "
                          f"in_filter={n_pl}")

        if vis:
            vis.update(state, true_pose)
            vis.wait(1000)

        state = vio.state_estimate()
        pe = np.linalg.norm(state.sensor.pose.x - true_pose.x)
        ze = abs(state.sensor.pose.x[2] - true_pose.x[2])
        pos_errors.append(pe)
        z_errors.append(ze)

        T = true_pose.inverse()
        errs = [np.linalg.norm(lm.p - T*canopy[lm.id])
                for lm in state.camera_landmarks if lm.id in canopy]
        lm_errors.append(np.mean(errs) if errs else float('nan'))

        if verbose and vc <= 12:
            np_ = len(state.plane_landmarks)
            nc = sum(len(pl.point_ids) for pl in state.plane_landmarks)
            print(f"  t={t:.1f}: lm={len(state.camera_landmarks):2d}, "
                  f"planes={np_}, constrained={nc:2d}, "
                  f"pos={pe:.2f}, z_err={ze:.2f}")

    if vis:
        print("Simulation complete. Close the window to continue.")
        pg.exec()
    
    return pos_errors, z_errors, lm_errors


def print_result(label, pe, ze, le, rmse_b=None, zrmse_b=None):
    rmse = np.sqrt(np.mean(np.array(pe)**2))
    zrmse = np.sqrt(np.mean(np.array(ze)**2))
    s = f"    {label}: Pos {rmse:.2f}m  Z {zrmse:.2f}m  Lm: {le[-1]:.2f}m"
    if rmse_b is not None:
        imp = (rmse_b - rmse) / rmse_b * 100
        zimp = (zrmse_b - zrmse) / zrmse_b * 100
        s += f"  ({imp:+.1f}% / Z {zimp:+.1f}%)"
    print(s)
    return rmse, zrmse


def main():
    sigma_px = 0.5

    print("=" * 70)
    print(f"  Canopy Hovering: 60m alt, σ_px={sigma_px}")
    print("=" * 70)

    for tree_std in [0.0, 2.0, 5.0]:
        print(f"\n{'='*70}")
        print(f"  Tree height std = {tree_std:.1f}m")
        print(f"{'='*70}")

        # Baseline
        print("\n  Baseline (point-only):")
        pe_b, ze_b, le_b = run_sim(
            use_planes=False, tree_height_std=tree_std,
            sigma_px=sigma_px, verbose=(tree_std == 0.0))
            #sigma_px=sigma_px, verbose=(tree_std == 0.0), visualize=True)
        rmse_b, zrmse_b = print_result("baseline", pe_b, ze_b, le_b)

        # GT planes
        print("\n  GT planes:")
        for sig in [1.0, 0.5, 0.1, 0.05]:
            pe, ze, le = run_sim(
                use_planes=True, plane_mode='gt', sigma_constraint=sig,
                tree_height_std=tree_std, sigma_px=sigma_px)
            print_result(f"GT σ={sig:.2f}", pe, ze, le, rmse_b, zrmse_b)

        # RANSAC planes
        print("\n  RANSAC planes:")
        for sig in [1.0, 0.5, 0.1, 0.05]:
            pe, ze, le = run_sim(
                use_planes=True, plane_mode='ransac', sigma_constraint=sig,
                tree_height_std=tree_std, sigma_px=sigma_px,
                #verbose=(sig == 0.1 and tree_std == 2.0))
                visualize=(sig == 0.1 and tree_std == 2.0))
            print_result(f"RANSAC σ={sig:.2f}", pe, ze, le, rmse_b, zrmse_b)


if __name__ == "__main__":
    main()
