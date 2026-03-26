#!/usr/bin/env python3
"""
Synthetic EqF backend test: known geometry, no RANSAC, no tracking.

Camera at z=1.5m, looking along world +x, moving forward.
Floor at z=0, wall at y=-2. Points on both surfaces.
Compares point-only vs plane-augmented filter.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from liepp import SO3, SE3, SOT3
from eqvio.mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark,
)
from eqvio.mathematical.vio_group import VIOGroup, state_group_action
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vision_measurement import VisionMeasurement, measure_system_state
from eqvio.vio_filter import VIOFilter, VIOFilterSettings

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


# Camera looks along world +x: cam_z->world_x, cam_x->world_-y, cam_y->world_-z
R_BODY = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=float)
HOVER_ACC = R_BODY.T @ np.array([0., 0., 9.80665])  # body-frame gravity for hover

def make_pose(x, y=0., z=1.5):
    return SE3(R=SO3(matrix=R_BODY), x=np.array([x, y, z]))

def make_floor_points():
    pts = {}; fid = 1000
    #for xi in np.linspace(1.0, 5.0, 5):
    #    for yi in np.linspace(-1.0, 1.0, 4):
    for xi in np.linspace(0.0, 8.0, 6):
        for yi in np.linspace(-1.5, 1.5, 5):
            pts[fid] = np.array([xi, yi, 0.0]); fid += 1
    return pts

def make_wall_points():
    pts = {}; fid = 2000
    #for xi in np.linspace(1.0, 5.0, 5):
    #    for zi in np.linspace(0.5, 2.5, 4):
    for xi in np.linspace(0.0, 8.0, 6):
        for zi in np.linspace(0.3, 2.5, 5):
            pts[fid] = np.array([xi, -2.0, zi]); fid += 1
    return pts

def generate_measurements(pose, camera, world_points, sigma_px=1.0):
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


def augment_gt_planes(vio, floor_pts, wall_pts):
    """Add planes using known world geometry in estimated camera frame."""
    xi_hat = vio.state_estimate()
    active = set(vio.eqf.X.id)
    existing = set(vio.eqf.X.plane_id)
    T_WtoC = (xi_hat.sensor.pose * xi_hat.sensor.camera_offset).inverse()
    R_WtoC = T_WtoC.R.asMatrix()
    p_CinW = xi_hat.sensor.pose.x

    new_planes = []

    # Floor: z=0 => [0,0,1]·p + 0 = 0
    if 100 not in existing:
        n_W, d_W = np.array([0.,0.,1.]), 0.0
        n_C = R_WtoC @ n_W; d_C = d_W + n_W @ p_CinW
        if abs(d_C) > 0.1:
            ids = [f for f in floor_pts if f in active]
            if len(ids) >= 4:
                new_planes.append(PlaneLandmark(q=n_C/d_C, id=100, point_ids=ids))

    # Wall: y=-2 => [0,1,0]·p + 2 = 0
    if 200 not in existing:
        n_W, d_W = np.array([0.,1.,0.]), 2.0
        n_C = R_WtoC @ n_W; d_C = d_W + n_W @ p_CinW
        if abs(d_C) > 0.1:
            ids = [f for f in wall_pts if f in active]
            if len(ids) >= 4:
                new_planes.append(PlaneLandmark(q=n_C/d_C, id=200, point_ids=ids))

    if new_planes:
        n = len(new_planes)
        vio.eqf.add_new_plane_landmarks(new_planes, np.eye(3*n) * vio.settings.initial_plane_variance)
        vio._invalidate_gain_cache()

    # Update associations
    for pl in vio.eqf.xi0.plane_landmarks:
        if pl.id == 100: pl.point_ids = [f for f in floor_pts if f in active]
        elif pl.id == 200: pl.point_ids = [f for f in wall_pts if f in active]


def run_sim(use_planes=False, sigma_constraint=0.5, seed=42, verbose=False, visualize=False):
    np.random.seed(seed)
    camera = PinholeCamera()
    floor, wall = make_floor_points(), make_wall_points()
    all_pts = {**floor, **wall}

    vis = SyntheticVisualiser(all_pts, cam_dist=10.0) if visualize else None

    s = VIOFilterSettings()
    s.camera_offset = SE3.Identity()
    s.initial_scene_depth = 2.5
    s.initial_point_variance = 50.0
    s.sigma_bearing = 1.0
    s.max_landmarks = 60
    s.outlier_mahalanobis_threshold = 1e6
    s.sigma_constraint = sigma_constraint
    s.initial_plane_variance = 1.0

    vio = VIOFilter(s)
    #vio.eqf.xi0.sensor.pose = make_pose(0.05, -0.03, 1.52)
    vio.eqf.xi0.sensor.pose = make_pose(-1.95, -0.03, 1.52)

    pos_errors, lm_errors = [], []
    vc = 0

    for i in range(200):
        t = i * 0.05
        #true_pose = make_pose(x=0.5*t)
        true_pose = make_pose(x=-2.0 + 0.5*t)

        imu = IMUVelocity(stamp=t, gyr=np.zeros(3),
                          acc=HOVER_ACC.copy(),
                          gyr_bias_vel=np.zeros(3), acc_bias_vel=np.zeros(3))
        vio.process_imu(imu)

        if i % 5 != 0: continue
        cam_coords = generate_measurements(true_pose, camera, all_pts, sigma_px=1.0)
        if not cam_coords: continue

        meas = VisionMeasurement(stamp=t)
        meas.cam_coordinates = cam_coords
        meas.camera_ptr = camera

        try:
            vio.process_vision(meas, camera)
        except Exception as e:
            if verbose: print(f"  t={t:.2f}: {e}")
            continue

        vc += 1
        state = vio.state_estimate()

        if use_planes and vc >= 3:
            augment_gt_planes(vio, floor, wall)

        if vis:
            vis.update(state, true_pose)
            vis.wait(1000)

        state = vio.state_estimate()
        pe = np.linalg.norm(state.sensor.pose.x - true_pose.x)
        pos_errors.append(pe)

        T = true_pose.inverse()
        errs = [np.linalg.norm(lm.p - T*all_pts[lm.id]) for lm in state.camera_landmarks if lm.id in all_pts]
        lm_errors.append(np.mean(errs) if errs else float('nan'))

        if verbose and vc <= 12:
            np_ = len(state.plane_landmarks)
            nc = sum(len(pl.point_ids) for pl in state.plane_landmarks)
            print(f"  t={t:.2f}: lm={len(state.camera_landmarks):2d}, "
                  f"planes={np_}, pts_on_planes={nc:2d}, "
                  f"pos_err={pe:.4f}, lm_err={lm_errors[-1]:.4f}")

    if vis:
        print("Simulation complete. Close the window to continue.")
        pg.exec()

    return pos_errors, lm_errors


def main():
    print("=" * 60)
    print("Synthetic EqF Backend Test")
    print("=" * 60)

    print("\n--- Baseline (point-only) ---")
    pe_b, le_b = run_sim(use_planes=False, verbose=True)
    #pe_b, le_b = run_sim(use_planes=False, verbose=True, visualize=True)
    rmse_b = np.sqrt(np.mean(np.array(pe_b)**2))
    print(f"  Pos RMSE: {rmse_b:.4f}  final: {pe_b[-1]:.4f}  lm: {le_b[-1]:.4f}")

    for sig in [0.5, 0.1, 0.05]:
        print(f"\n--- Planes σ={sig} ---")
        #pe_p, le_p = run_sim(use_planes=True, sigma_constraint=sig, verbose=(sig==0.5))
        pe_p, le_p = run_sim(use_planes=True, sigma_constraint=sig, verbose=(sig==0.5), visualize=(sig==0.05))
        rmse_p = np.sqrt(np.mean(np.array(pe_p)**2))
        imp = (rmse_b - rmse_p) / rmse_b * 100
        print(f"  Pos RMSE: {rmse_p:.4f}  final: {pe_p[-1]:.4f}  lm: {le_p[-1]:.4f}  ({imp:+.1f}%)")


if __name__ == "__main__":
    main()
