#!/usr/bin/env python3
"""
Realistic long-flight canopy test with landmark lifecycle.

Scenario: drone at 60m traverses 200m over tree canopy.
Points enter/exit FOV naturally. New points are initialized from the
filter's current (possibly wrong) depth estimate, simulating real VIO.

Key questions:
    1. How does depth estimate degrade as original landmarks depart?
    2. Do new landmarks inherit the degraded depth?
    3. Does the plane constraint prevent drift through turnover?
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
from eqvio.plane_detection.plane_fitting import (
    fit_plane_ransac, PlaneFittingSettings,
)
from eqvio.plane_detection.plane_detector import landmarks_to_global


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
        
        while (time.time() - start_time < delay_sec) or self.paused:
            self.app.processEvents()
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


# Camera looking down
R_DOWN = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
HOVER_ACC = R_DOWN.T @ np.array([0., 0., 9.80665])

def rotate_camera_around_gravity(theta_radians):
    """
    Rotates the R_DOWN camera matrix around the Z-axis (gravity).
    """
    c = np.cos(theta_radians)
    s = np.sin(theta_radians)
    
    R_z = np.array([
        [ c, -s,  0],
        [ s,  c,  0],
        [ 0,  0,  1]
    ], dtype=float)
    
    R_new = R_z @ R_DOWN
    
    return R_new

def make_pose(x, y=0., altitude=60.0, yaw=0.):
    #return SE3(R=SO3(matrix=R_DOWN), x=np.array([x, y, altitude]))
    return SE3(R=SO3(matrix=rotate_camera_around_gravity(yaw)), x=np.array([x, y, altitude]))


def scatter_canopy(x_range=(-50, 250), y_range=(-30, 30),
                   density=0.05, tree_height_std=2.0, seed=42):
    """Dense random points over a large area. density = points per m^2."""
    rng = np.random.RandomState(seed)
    lx = x_range[1] - x_range[0]
    ly = y_range[1] - y_range[0]
    n = int(lx * ly * density)
    pts = {}
    for i in range(n):
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        z = rng.randn() * tree_height_std
        pts[5000 + i] = np.array([x, y, z])
    return pts


def visible_points(pose, camera, world_points):
    """Return {fid: pixel_coords} for points in FOV."""
    T_WtoC = pose.inverse()
    visible = {}
    for fid, pw in world_points.items():
        pc = T_WtoC * pw
        if pc[2] < 0.1:
            continue
        try:
            px = camera.project_point(pc)
            if 0 <= px[0] <= 640 and 0 <= px[1] <= 480:
                visible[fid] = px
        except ValueError:
            continue
    return visible


def ransac_augment(vio, camera, altitude, tree_height_std):
    """Direct RANSAC on all landmarks → plane augmentation."""
    state = vio.state_estimate()
    feat_pos, cam_pos, R_GtoC = landmarks_to_global(state)

    if len(feat_pos) < 5:
        return

    s = PlaneFittingSettings()
    s.ransac_inlier_threshold = max(altitude * 0.1, 2.0 * tree_height_std)
    s.ransac_min_inlier_ratio = 0.50
    s.ransac_min_point_separation = 1.0
    s.min_cp_distance = 1.0

    all_fids = sorted(feat_pos.keys())
    all_pts = np.array([feat_pos[fid] for fid in all_fids])

    ok, cp, inlier_fids = fit_plane_ransac(all_fids, all_pts, s)
    if not ok:
        return

    # Update existing plane or augment new one
    existing = set(vio.eqf.X.plane_id)
    if 500 in existing:
        # Update inlier associations
        active = set(vio.eqf.X.id)
        for pl in vio.eqf.xi0.plane_landmarks:
            if pl.id == 500:
                pl.point_ids = [f for f in inlier_fids if f in active]
    else:
        vio.augment_planes({500: cp}, {500: inlier_fids})


def gt_augment(vio, all_canopy_pts):
    """GT plane augmentation (oracle)."""
    xi_hat = vio.state_estimate()
    active = set(vio.eqf.X.id)
    existing = set(vio.eqf.X.plane_id)

    T_WtoC = (xi_hat.sensor.pose * xi_hat.sensor.camera_offset).inverse()
    R_WtoC = T_WtoC.R.asMatrix()
    p_CinW = xi_hat.sensor.pose.x

    n_W, d_W = np.array([0., 0., 1.]), 0.0
    n_C = R_WtoC @ n_W
    d_C = d_W + n_W @ p_CinW

    if abs(d_C) < 0.1:
        return

    q_cam = n_C / d_C
    ids = [f for f in all_canopy_pts if f in active]
    if len(ids) < 4:
        return

    if 300 in existing:
        for pl in vio.eqf.xi0.plane_landmarks:
            if pl.id == 300:
                pl.point_ids = ids
    else:
        vio.eqf.add_new_plane_landmarks(
            [PlaneLandmark(q=q_cam, id=300, point_ids=ids)],
            np.eye(3) * vio.settings.initial_plane_variance
        )
        vio._invalidate_gain_cache()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_sim(plane_mode='none', sigma_constraint=0.1,
            tree_height_std=2.0, altitude=60.0,
            speed=2.0, duration=100.0, dt=0.05,
            #sigma_px=0.5, seed=42, verbose=False, visualize=False):
            sigma_px=0.1, seed=42, verbose=False, visualize=False):
    np.random.seed(seed)
    camera = PinholeCamera()

    # Dense canopy spanning the full flight path
    canopy = scatter_canopy(
        x_range=(-30, speed * duration + 30),
        y_range=(-25, 25),
        density=0.04, tree_height_std=tree_height_std, seed=99
    )
    print(f"    Total canopy points: {len(canopy)}") if verbose else None

    vis = SyntheticVisualiser(canopy, cam_dist=120.0, elevation=45) if visualize else None

    s = VIOFilterSettings()
    s.camera_offset = SE3.Identity()
    s.initial_scene_depth = altitude
    s.initial_point_variance = altitude ** 2
    s.sigma_bearing = sigma_px
    s.max_landmarks = 40
    s.outlier_mahalanobis_threshold = 1e6
    s.sigma_constraint = sigma_constraint
    s.initial_plane_variance = 1.0

    vio = VIOFilter(s)
    vio.eqf.xi0.sensor.pose = make_pose(0.0, 0.0, altitude + 2.0)
    vio.eqf.xi0.sensor.velocity = np.array([speed, -1.0, 0])

    n_steps = int(duration / dt)
    pos_errors = []
    z_errors = []
    lm_errors = []
    lm_counts = []
    turnover_events = []
    vc = 0

    for i in range(n_steps):
        t = i * dt
        omega = 0.5
        yaw = omega * t
        x = speed * t
        y = 5.0 * np.sin(0.2 * t)  # slight lateral wobble
        
        true_pose = make_pose(x, y, altitude, yaw=yaw)
        R_WtoB = true_pose.R.asMatrix().T
        acc_world = np.array([0.0, -0.2 * np.sin(0.2 * t), 0.0])
        gravity_world = np.array([0.0, 0.0, 9.80665])
        specific_force_world = acc_world + gravity_world
        acc_body = R_WtoB @ specific_force_world
        w_world = np.array([0.0, 0.0, omega]) 
        gyr_body = R_WtoB @ w_world
        
        imu = IMUVelocity(stamp=t, gyr=gyr_body, acc=acc_body,
                          gyr_bias_vel=np.zeros(3), acc_bias_vel=np.zeros(3))

        vio.process_imu(imu)

        if i % 5 != 0:
            continue

        # What's visible from true pose
        true_visible = visible_points(true_pose, camera, canopy)
        if not true_visible:
            continue

        # Add noise to measurements
        cam_coords = {
            fid: px + np.random.randn(2) * sigma_px
            for fid, px in true_visible.items()
        }

        meas = VisionMeasurement(stamp=t)
        meas.cam_coordinates = cam_coords
        meas.camera_ptr = camera

        # Count landmarks before update (for turnover tracking)
        n_before = len(vio.eqf.X.id)

        try:
            vio.process_vision(meas, camera)
        except Exception as e:
            if verbose:
                print(f"  t={t:.1f}: {e}")
            continue

        vc += 1

        # Track turnover
        n_after = len(vio.eqf.X.id)
        ids_after = set(vio.eqf.X.id)
        n_new = len(ids_after - set(vio.eqf.X.id))  # approximate

        # Plane augmentation
        state = vio.state_estimate()
        if plane_mode == 'ransac' and vc >= 3:
            ransac_augment(vio, camera, altitude, tree_height_std)
        elif plane_mode == 'gt' and vc >= 3:
            gt_augment(vio, canopy)

        if vis:
            vis.update(state, true_pose)
            #vis.wait(100)

        # Record metrics
        state = vio.state_estimate()
        pe = np.linalg.norm(state.sensor.pose.x - true_pose.x)
        ze = abs(state.sensor.pose.x[2] - true_pose.x[2])
        pos_errors.append(pe)
        z_errors.append(ze)
        lm_counts.append(len(state.camera_landmarks))

        T = true_pose.inverse()
        errs = [np.linalg.norm(lm.p - T * canopy[lm.id])
                for lm in state.camera_landmarks if lm.id in canopy]
        lm_errors.append(np.mean(errs) if errs else float('nan'))

        if verbose and (vc <= 5 or vc % 50 == 0):
            n_pl = len(state.plane_landmarks)
            nc = sum(len(pl.point_ids) for pl in state.plane_landmarks)
            n_vis = len(true_visible)
            print(f"  t={t:5.1f}s x={x:6.1f}m: lm={len(state.camera_landmarks):2d} "
                  f"vis={n_vis:2d} planes={n_pl} constr={nc:2d} "
                  f"pos={pe:.2f} z={ze:.2f} lm_err={lm_errors[-1]:.1f}")

    if vis:
        print("Simulation complete. Close the window to continue.")
        pg.exec()

    return pos_errors, z_errors, lm_errors, lm_counts


def print_result(label, pe, ze, le, rmse_b=None, zrmse_b=None):
    rmse = np.sqrt(np.mean(np.array(pe)**2))
    zrmse = np.sqrt(np.mean(np.array(ze)**2))
    # Also compute late-flight (last 25%) metrics
    n = len(pe)
    late = slice(3*n//4, n)
    rmse_late = np.sqrt(np.mean(np.array(pe[late])**2))
    zrmse_late = np.sqrt(np.mean(np.array(ze[late])**2))
    s = (f"    {label:20s}: Pos {rmse:.2f}m (late {rmse_late:.2f}m)  "
         f"Z {zrmse:.2f}m (late {zrmse_late:.2f}m)  Lm: {le[-1]:.1f}m")
    if rmse_b is not None:
        imp = (rmse_b - rmse) / rmse_b * 100
        imp_l = ((np.sqrt(np.mean(np.array(pe[late])**2))) -
                  rmse_late) / np.sqrt(np.mean(np.array(pe[late])**2)) * 100
        s += f"  ({imp:+.1f}%)"
    print(s)
    return rmse, zrmse


def main():
    print("=" * 75)
    print("  Long-flight canopy: 60m alt, 200m traverse, landmark turnover")
    print("=" * 75)
    duration = 30.0

    for tree_std in [2.0]:
        print(f"\n{'='*75}")
        print(f"  Tree height std = {tree_std:.1f}m, speed=2m/s, {duration}s flight")
        print(f"{'='*75}")

        print("\n  Baseline:")
        pe_b, ze_b, le_b, lc_b = run_sim(
            plane_mode='none', tree_height_std=tree_std,
            #duration=duration, verbose=True)
            duration=duration, verbose=True, visualize=True)
        rmse_b, zrmse_b = print_result("point-only", pe_b, ze_b, le_b)

        print("\n  GT plane:")
        for sig in [0.5, 0.1]:
            pe, ze, le, lc = run_sim(
                plane_mode='gt', sigma_constraint=sig,
                tree_height_std=tree_std, duration=duration,
                verbose=(sig == 0.1))
                #verbose=(sig == 0.1), visualize=(sig==0.5))
            print_result(f"GT σ={sig}", pe, ze, le, rmse_b, zrmse_b)

        print("\n  RANSAC plane:")
        for sig in [0.5, 0.1]:
            pe, ze, le, lc = run_sim(
                plane_mode='ransac', sigma_constraint=sig,
                tree_height_std=tree_std, duration=duration,
                verbose=(sig == 0.1))
            print_result(f"RANSAC σ={sig}", pe, ze, le, rmse_b, zrmse_b)


if __name__ == "__main__":
    main()
