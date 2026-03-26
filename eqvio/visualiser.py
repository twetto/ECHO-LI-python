"""
3-Tier Real-time Equivariant Filter Visualizer.
- Top: Global Manifold (Aligned Trajectory, Persistent Map, Active Points, & Planes)
- Middle: Fixed Origin Space (Local Points & Covariance Crosses)
- Bottom: Tangent Space (Offset Pose & Attitude Ellipsoids)
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets 

class TrajectoryVisualiser:
    def __init__(self, gt_poses=None, update_interval: int = 5):
        self.app = pg.mkQApp("EqVIO-P Visualiser")
        pg.setConfigOptions(antialias=True)
        
        self.main_window = QtWidgets.QWidget()
        self.main_window.setWindowTitle('EqVIO: Manifold -> Origin -> Tangent Space')
        self.main_window.resize(800, 1000)
        self.layout = QtWidgets.QVBoxLayout()
        self.main_window.setLayout(self.layout)

        # =========================================================
        # LEVEL 1: Global Manifold (Top)
        # =========================================================
        self.w_global = gl.GLViewWidget()
        self.w_global.setCameraPosition(distance=20, elevation=30, azimuth=-90)
        self.layout.addWidget(self.w_global)
        
        world_grid = gl.GLGridItem()
        world_grid.scale(2, 2, 2)
        self.w_global.addItem(world_grid)

        self.gt_line = gl.GLLinePlotItem(color=(0, 1, 1, 0.7), width=2.0)
        self.est_line = gl.GLLinePlotItem(color=(1, 0, 0, 1.0), width=2.5)
        
        # Persistent Map Points (Gray)
        self.map_points = gl.GLScatterPlotItem(color=(0.5, 0.5, 0.5, 0.8), size=3.0)
        # Active Points (Yellow)
        self.global_points = gl.GLScatterPlotItem(color=(1.0, 1.0, 0.0, 1.0), size=5.0)
        
        self.w_global.addItem(self.gt_line)
        self.w_global.addItem(self.est_line)
        self.w_global.addItem(self.map_points)
        self.w_global.addItem(self.global_points)

        # Plane tracking variables
        self.plane_items = {}
        self.plane_colors = [
            (0.0, 1.0, 1.0, 0.8), # Cyan
            (1.0, 0.0, 1.0, 0.8), # Magenta
            (0.0, 1.0, 0.0, 0.8), # Green
            (1.0, 0.5, 0.0, 0.8), # Orange
            (0.5, 0.0, 1.0, 0.8), # Purple
        ]

        # =========================================================
        # LEVEL 2: Fixed Origin Space / xi_0 (Middle)
        # =========================================================
        self.w_origin = gl.GLViewWidget()
        self.w_origin.setCameraPosition(distance=10, elevation=20, azimuth=-90)
        self.layout.addWidget(self.w_origin)

        cam_axis = gl.GLAxisItem()
        cam_axis.setSize(1, 1, 1)
        self.w_origin.addItem(cam_axis)

        self.xi0_points = gl.GLScatterPlotItem(color=(0.0, 1.0, 1.0, 1.0), size=6.0)
        self.w_origin.addItem(self.xi0_points)

        self.xi0_cov_lines = gl.GLLinePlotItem(color=(0.0, 1.0, 0.0, 0.6), width=1.5, mode='lines')
        self.w_origin.addItem(self.xi0_cov_lines)

        # =========================================================
        # LEVEL 3: Linearized Tangent Space (Bottom)
        # =========================================================
        self.w_tangent = gl.GLViewWidget()
        self.w_tangent.setCameraPosition(distance=8, elevation=15, azimuth=-90)
        self.layout.addWidget(self.w_tangent)

        tangent_grid = gl.GLGridItem(color=(255, 255, 255, 50))
        self.w_tangent.addItem(tangent_grid)

        pos_axis = gl.GLAxisItem(); pos_axis.setSize(0.5, 0.5, 0.5); pos_axis.translate(-2, 0, 0)
        att_axis = gl.GLAxisItem(); att_axis.setSize(0.5, 0.5, 0.5); att_axis.translate(2, 0, 0)
        self.w_tangent.addItem(pos_axis)
        self.w_tangent.addItem(att_axis)

        self.base_sphere = gl.MeshData.sphere(rows=30, cols=30)

        self.pos_ellipsoid = gl.GLMeshItem(
            meshdata=self.base_sphere, color=(0.0, 1.0, 1.0, 0.6), 
            smooth=True, shader='shaded', glOptions='translucent'
        )
        self.w_tangent.addItem(self.pos_ellipsoid)

        self.att_ellipsoid = gl.GLMeshItem(
            meshdata=self.base_sphere, color=(1.0, 0.5, 0.0, 0.6), 
            smooth=True, shader='shaded', glOptions='translucent'
        )
        self.w_tangent.addItem(self.att_ellipsoid)

        self.main_window.show()

        # --- Data storage ---
        self.gt_positions = []
        self.gt_times = []
        if gt_poses:
            for p in gt_poses:
                self.gt_positions.append(p.pose.x.copy())
                self.gt_times.append(p.t)
            self.gt_positions = np.array(self.gt_positions, dtype=np.float32)
            self.gt_times = np.array(self.gt_times)

        self.est_positions = []
        self.est_times = []
        
        # Point lifetime and persistent map storage
        self.point_lifetime = {}     
        self.persistent_points = {}

        self.update_interval = update_interval
        self._frame_count = 0
        self._T_align = None  

    def _calculate_alignment(self):
        """Recalculate Umeyama alignment to ground truth."""
        est = np.array(self.est_positions)
        t_est = np.array(self.est_times)

        if len(self.gt_positions) > 0 and len(est) > 100:
            try:
                from eqvio.alignment import align_umeyama
                matched_est = []
                matched_gt = []
                gt_idx = 0
                for i, t in enumerate(t_est):
                    while gt_idx < len(self.gt_times) - 1 and self.gt_times[gt_idx] < t:
                        gt_idx += 1
                    if gt_idx < len(self.gt_positions):
                        matched_est.append(est[i])
                        matched_gt.append(self.gt_positions[gt_idx])
                        
                if len(matched_est) > 10:
                    self._T_align = align_umeyama(np.array(matched_est), np.array(matched_gt))
            except Exception:
                pass

    def _get_ellipsoid_transform(self, cov: np.ndarray, center: list, sigma_scale: float = 3.0) -> pg.Transform3D:
        """Converts covariance matrix into an offset 4x4 OpenGL transform."""
        cov = 0.5 * (cov + cov.T)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-12) 
        
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = evecs @ np.diag(sigma_scale * np.sqrt(evals))
        transform[:3, 3] = center
        
        return pg.Transform3D(transform)

    def update(self, timestamp: float, state, eqf=None, predicted_state=None):
        pos = state.sensor.pose.x.copy()
        self.est_positions.append(pos)
        self.est_times.append(timestamp)
        self._frame_count += 1

        # Track point lifetime for the persistent map
        unaligned_cam_pose = state.sensor.pose * state.sensor.camera_offset
        R_cam = unaligned_cam_pose.R.asMatrix()
        x_cam = unaligned_cam_pose.x

        for lm in state.camera_landmarks:
            self.point_lifetime[lm.id] = self.point_lifetime.get(lm.id, 0) + 1
            if self.point_lifetime[lm.id] > 3:
                self.persistent_points[lm.id] = R_cam @ lm.p + x_cam

        if self._frame_count % self.update_interval == 0:
            self._calculate_alignment()
            self._redraw(state, eqf)
            
        self.app.processEvents()

    def _redraw(self, state, eqf):
        # --- 1. Top Window: Global Manifold ---
        if len(self.gt_positions) > 0:
            gt_mask = self.gt_times <= self.est_times[-1]
            if np.any(gt_mask):
                self.gt_line.setData(pos=self.gt_positions[gt_mask])

        R_align = np.eye(3)
        t_align = np.zeros(3)
        if self._T_align is not None:
            R_align = self._T_align.R.asMatrix()
            t_align = self._T_align.x

        est = np.array(self.est_positions, dtype=np.float32)
        if len(est) > 1:
            aligned_est = (R_align @ est.T).T + t_align
            self.est_line.setData(pos=aligned_est)

        # Draw Persistent Map Points (Gray)
        if self.persistent_points:
            pers_arr = np.array(list(self.persistent_points.values()), dtype=np.float32)
            aligned_pers = (R_align @ pers_arr.T).T + t_align
            self.map_points.setData(pos=aligned_pers)

        # Draw Active Global Points (Yellow)
        unaligned_cam = state.sensor.pose * state.sensor.camera_offset
        aligned_cam = self._T_align * unaligned_cam if self._T_align is not None else unaligned_cam
        R_cam_aligned = aligned_cam.R.asMatrix()
        t_cam_aligned = aligned_cam.x
        
        if state.camera_landmarks:
            curr_pts = np.array([lm.p for lm in state.camera_landmarks], dtype=np.float32)
            global_pts = (R_cam_aligned @ curr_pts.T).T + t_cam_aligned
            self.global_points.setData(pos=global_pts)
        else:
            self.global_points.setData(pos=np.empty((0, 3)))

        # Draw Plane Landmarks as dynamic bounded polygons
        current_plane_ids = set()
        
        # Quick lookup for active camera landmarks
        active_lm_dict = {lm.id: lm.p for lm in state.camera_landmarks}
        
        # Fallback to eqf manifold if state doesn't have planes yet
        plane_list = state.plane_landmarks
        if not plane_list and eqf is not None and hasattr(eqf.xi0, 'plane_landmarks'):
            plane_list = eqf.xi0.plane_landmarks

        for plane in plane_list:
            pid = plane.id
            current_plane_ids.add(pid)
            
            q_cam = plane.q
            q_norm_sq = np.dot(q_cam, q_cam)
            if q_norm_sq < 1e-8:
                continue
                
            n_cam = q_cam / np.sqrt(q_norm_sq)
            cp_cam = -q_cam / q_norm_sq
            
            # Global plane parameters
            n_global = R_cam_aligned @ n_cam
            cp_global = R_cam_aligned @ cp_cam + t_cam_aligned
            d_global = -np.dot(n_global, cp_global)
            
            # 1. Collect points (FIXED: Properly apply alignment to persistent points!)
            pts_global = []
            if hasattr(plane, 'point_ids'):
                for pt_id in plane.point_ids:
                    if pt_id in active_lm_dict:
                        # Align active landmark to global
                        p_global = R_cam_aligned @ active_lm_dict[pt_id] + t_cam_aligned
                        pts_global.append(p_global)
                    elif pt_id in self.persistent_points:
                        # Apply the Umeyama transform to the raw persistent point
                        p_raw = self.persistent_points[pt_id]
                        p_global = R_align @ p_raw + t_align
                        pts_global.append(p_global)
            
            if len(pts_global) < 3:
                continue
                
            pts_global = np.array(pts_global)
            
            # Project the points mathematically onto the plane
            dists = np.dot(pts_global, n_global) + d_global
            pts_proj = pts_global - np.outer(dists, n_global)
            
            # Calculate Centroid and Local 2D Axes
            centroid = np.mean(pts_proj, axis=0)
            z_axis = n_global
            up = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Sort points by angle around the centroid
            vecs = pts_proj - centroid
            angles = np.arctan2(np.dot(vecs, y_axis), np.dot(vecs, x_axis))
            sort_idx = np.argsort(angles)
            sorted_pts = pts_proj[sort_idx]
            
            verts = np.vstack((centroid, sorted_pts))
            n_pts = len(sorted_pts)
            
            # 2. Build Faces (FIXED: Add backfaces to prevent OpenGL culling!)
            faces = []
            for i in range(1, n_pts):
                faces.append([0, i, i + 1])       # Front Face
                faces.append([0, i + 1, i])       # Back Face
            faces.append([0, n_pts, 1])           # Front Face (Last slice)
            faces.append([0, 1, n_pts])           # Back Face (Last slice)
            faces = np.array(faces, dtype=np.uint)
            
            # Draw or Update the Mesh
            if pid not in self.plane_items:
                color = self.plane_colors[pid % len(self.plane_colors)]
                meshdata = gl.MeshData(vertexes=verts, faces=faces)
                plane_mesh = gl.GLMeshItem(
                    meshdata=meshdata, 
                    color=color, 
                    smooth=False, 
                    shader='shaded', 
                    glOptions='translucent',
                    drawEdges=True,                        
                    edgeColor=(1.0, 1.0, 1.0, 0.6)         
                )
                self.plane_items[pid] = plane_mesh
                self.w_global.addItem(plane_mesh)
            else:
                self.plane_items[pid].setMeshData(vertexes=verts, faces=faces)
            
            self.plane_items[pid].setTransform(pg.Transform3D())
            self.plane_items[pid].show()
            
        # Hide planes that were marginalized
        for pid, mesh in self.plane_items.items():
            if pid not in current_plane_ids:
                mesh.hide()

        # --- 2 & 3. Middle/Bottom Windows ---
        if eqf is not None and eqf.xi0.camera_landmarks:
            
            # LEVEL 2: Fixed Origin Points
            local_pts = np.array([lm.p for lm in eqf.xi0.camera_landmarks], dtype=np.float32)
            self.xi0_points.setData(pos=local_pts)
            
            # LEVEL 2: Landmark Covariance Crosses
            local_crosses = []
            for lm in eqf.xi0.camera_landmarks:
                try:
                    cov = eqf.get_landmark_cov_by_id(lm.id)
                    evals, evecs = np.linalg.eigh(cov)
                    for i in range(3):
                        length = 3.0 * np.sqrt(max(evals[i], 0.0))
                        axis = evecs[:, i]
                        local_crosses.append(lm.p - length * axis)
                        local_crosses.append(lm.p + length * axis)
                except (StopIteration, KeyError):
                    pass
            
            if local_crosses:
                self.xi0_cov_lines.setData(pos=np.array(local_crosses, dtype=np.float32))
            else:
                self.xi0_cov_lines.setData(pos=np.empty((0, 3)))

            # LEVEL 3: Offset Tangent Space Ellipsoids
            cov_pos = eqf.Sigma[9:12, 9:12]
            self.pos_ellipsoid.setTransform(self._get_ellipsoid_transform(cov_pos, [-2, 0, 0]))

            cov_att = eqf.Sigma[6:9, 6:9]
            self.att_ellipsoid.setTransform(self._get_ellipsoid_transform(cov_att, [2, 0, 0]))

    def finish(self):
        print("Dataset finished. Close the window to exit.")
        pg.exec()
