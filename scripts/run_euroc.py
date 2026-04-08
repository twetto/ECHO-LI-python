#!/usr/bin/env python3
"""
Run EqVIO-P on EuRoC MAV dataset.

Port of: src/main_opt.cpp

Usage:
    python scripts/run_euroc.py /path/to/V1_01_easy/ [--config configs/eqvio_euroc.yaml] [--output output/]

Pipeline:
    1. Load dataset (IMU + images + camera + extrinsics + ground truth)
    2. Initialize filter + feature tracker
    3. Main loop: interleave IMU propagation and vision updates
    4. Write estimated trajectory to file
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

# EqVIO imports
from eqvio.dataserver.asl_dataset import ASLDatasetReader
from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vision_measurement import VisionMeasurement
from eqvio.mathematical.vio_state import VIOState, StampedPose
from eqvio.vio_filter import VIOFilter, VIOFilterSettings

# GIFT imports
from gift.tracker import PointFeatureTracker
from gift.feature import Feature

# Plane detection
from eqvio.plane_detection import (
    CameraDebugWindow, PlaneDetector, PlaneDetectorSettings, landmarks_to_global, fit_detected_planes,
)


def convert_gift_features(features: list[Feature], stamp: float) -> VisionMeasurement:
    """Convert GIFT features to VisionMeasurement.

    Reference: convertGIFTFeatures() in main_opt.cpp
    """
    measurement = VisionMeasurement(stamp=stamp)
    for f in features:
        measurement.cam_coordinates[f.id_number] = f.cam_coordinates.copy()
    return measurement


def write_trajectory(filepath: Path, timestamps: list, states: list):
    """Write estimated trajectory in TUM format (for evo evaluation).

    Format: timestamp tx ty tz qx qy qz qw
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for t, state in zip(timestamps, states):
            R = state.sensor.pose.R
            x = state.sensor.pose.x
            q = R.asQuaternion()  # scipy: [x, y, z, w]
            writer.writerow([
                f"{t:.9f}",
                f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}",
                f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}",
            ])


def main():
    parser = argparse.ArgumentParser(description="EqVIO-P on EuRoC MAV dataset")
    parser.add_argument("dataset", type=Path, help="Path to EuRoC sequence (e.g., V1_01_easy)")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parent.parent / "configs" / "EQVIO_config_EuRoC_stationary.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start N seconds after dataset begins")
    parser.add_argument("--stop", type=float, default=-1.0,
                        help="Stop N seconds after dataset begins")
    parser.add_argument("--display", action="store_true",
                        help="Display feature tracking image")
    parser.add_argument("--planes", action="store_true",
                        help="Enable plane detection (shown in --display window)")
    parser.add_argument("--plot", action="store_true",
                        help="Live 3D trajectory plot vs ground truth")
    parser.add_argument("--profile", action="store_true",
                        help="Run cProfile and print top functions by cumulative time")
    parser.add_argument("--gt-init", action="store_true",
                        help="Initialize from ground truth pose (for debugging)")
    parser.add_argument("--max-features", type=int, default=40)
    parser.add_argument("--flowdep", action="store_true",
                        help="Enable FlowDep dense depth filter (Phase c)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Dataset: {args.dataset}")
    reader = ASLDatasetReader(str(args.dataset))

    if reader.camera is None:
        print("ERROR: Could not read camera from dataset.", file=sys.stderr)
        sys.exit(1)

    camera = reader.camera
    print(f"Camera: {type(camera).__name__}")

    if reader.camera_extrinsics is not None:
        print(f"Camera extrinsics (T_BS):\n{reader.camera_extrinsics.asMatrix()}")
    else:
        print("WARNING: No camera extrinsics found, using identity.")

    # ------------------------------------------------------------------
    # 2. Load config and initialize filter
    # ------------------------------------------------------------------
    if args.config.exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        settings = VIOFilterSettings.from_yaml(config)
    else:
        print(f"Config {args.config} not found, using defaults.")
        config = {}
        settings = VIOFilterSettings()

    # Override camera offset from dataset extrinsics
    if reader.camera_extrinsics is not None:
        settings.camera_offset = reader.camera_extrinsics

    settings.max_landmarks = args.max_features

    vio_filter = VIOFilter(settings)
    print(f"Filter initialized: max_landmarks={settings.max_landmarks}, "
          f"chart={settings.coordinate_choice}")

    # ------------------------------------------------------------------
    # 3. Initialize feature tracker
    # ------------------------------------------------------------------
    tracker = PointFeatureTracker(camera=camera)

    # Configure from GIFT section of config
    gift_config = config.get('GIFT', {}) if args.config.exists() else {}
    tracker.settings.max_features = gift_config.get('maxFeatures', args.max_features)
    tracker.settings.feature_dist = gift_config.get('featureDist', settings.min_feature_distance)
    tracker.settings.feature_search_threshold = gift_config.get('featureSearchThreshold', 0.8)
    tracker.settings.min_harris_quality = gift_config.get('minHarrisQuality', 0.1)
    tracker.settings.max_error = gift_config.get('maxError', 1e8)
    tracker.settings.win_size = gift_config.get('winSize', 21)
    tracker.settings.max_level = gift_config.get('maxLevel', 3)
    tracker.settings.tracked_feature_dist = gift_config.get('trackedFeatureDist', 0.0)
    tracker.settings.use_fast_features = gift_config.get('useFastFeatures', False)
    tracker.settings.equalise_image_histogram = gift_config.get('equaliseImageHistogram', False)

    # RANSAC settings
    ransac_config = gift_config.get('ransacParams', {})
    if ransac_config:
        from gift.ransac import RansacParameters
        tracker.settings.ransac_params = RansacParameters(
            max_iterations=ransac_config.get('maxIterations', 0),
            inlier_threshold=ransac_config.get('inlierThreshold', 0.003),
            min_data_points=ransac_config.get('minDataPoints', 5),
            min_inliers=ransac_config.get('minInliers', 10),
        )

    # Override from CLI if provided
    if args.max_features:
        tracker.settings.max_features = args.max_features
        settings.max_landmarks = args.max_features

    print(f"Tracker: PointFeatureTracker, max_features={tracker.settings.max_features}, "
          f"feature_dist={tracker.settings.feature_dist:.1f}")

    # ------------------------------------------------------------------
    # 4. Preload data into time-sorted stream
    # ------------------------------------------------------------------
    print("Loading IMU data...")
    imu_data = list(reader.imu_iter())
    print(f"  {len(imu_data)} IMU measurements")

    print("Loading image timestamps...")
    image_data = list(reader.image_iter())
    print(f"  {len(image_data)} images")

    # Build time-sorted event stream
    events = []
    for imu in imu_data:
        events.append(('imu', imu.stamp, imu))
    for img in image_data:
        events.append(('image', img.stamp, img))
    events.sort(key=lambda e: e[1])

    # ------------------------------------------------------------------
    # 4b. Pose initialization
    # ------------------------------------------------------------------
    if args.gt_init:
        gt_poses = reader.groundtruth()
        if gt_poses:
            vio_filter.eqf.xi0.sensor.pose = gt_poses[0].pose
            print(f"GT initialization at t={gt_poses[0].t:.3f}")
            print(f"  position: {gt_poses[0].pose.x}")
        else:
            print("WARNING: --gt-init requested but no ground truth available")
    else:
        from eqvio.initialization import estimate_initial_pose, check_stationary

        if check_stationary(imu_data):
            init_pose = estimate_initial_pose(imu_data)
            vio_filter.eqf.xi0.sensor.pose = init_pose
            R_euler = init_pose.R.asMatrix()
            pitch = np.arcsin(np.clip(-R_euler[2, 0], -1, 1)) * 180 / np.pi
            roll = np.arctan2(R_euler[2, 1], R_euler[2, 2]) * 180 / np.pi
            print(f"Static IMU initialization: roll={roll:.1f}°, pitch={pitch:.1f}°")
        else:
            print("WARNING: Platform not stationary at start, using identity pose")

    # Compute start/stop times
    initial_time = events[0][1] if events else 0.0
    start_time = initial_time + args.start if args.start > 0 else 0.0
    stop_time = initial_time + args.stop if args.stop > 0 else float('inf')

    # ------------------------------------------------------------------
    # 5. Main loop
    # ------------------------------------------------------------------
    print(f"\nRunning filter...")

    # Initialize visualizer if requested
    visualiser = None
    if args.plot:
        from eqvio.visualiser import TrajectoryVisualiser
        gt_poses = reader.groundtruth()
        visualiser = TrajectoryVisualiser(gt_poses, update_interval=5)
        print("Live trajectory plot enabled")

    # Initialize camera debug window if requested
    cam_debug = None
    if args.display:
        start_mode = 1 if args.planes else 0
        cam_debug = CameraDebugWindow(enabled=True, start_mode=start_mode)
        print("Camera debug window enabled (press 'm' to cycle modes, 'q' to close)")

    # Initialize plane detector if requested
    plane_detector = None
    if args.planes:
        plane_detector = PlaneDetector(PlaneDetectorSettings(
            max_tri_side_px=250,
            max_dist_between_z=0.1,
            #max_norm_deg=5))
            max_norm_deg=180))
        print("Plane detection enabled")

    # Initialize FlowDep dense depth filter
    flowdep_filter = None
    flowdep_plane_detector = None
    if args.flowdep:
        from eqvio.flowdep import FlowDepFilter, FlowDepSettings, relabel_landmarks_by_grid
        flowdep_cfg = config.get('FlowDep', {})
        _dis_presets = {
            'ultrafast': cv2.DISOpticalFlow_PRESET_ULTRAFAST,
            'fast': cv2.DISOpticalFlow_PRESET_FAST,
            'medium': cv2.DISOpticalFlow_PRESET_MEDIUM,
        }
        flowdep_settings = FlowDepSettings(
            image_scale=flowdep_cfg.get('image_scale', 1.0),
            enable_warmstart=flowdep_cfg.get('enable_warmstart', True),
            dis_preset=_dis_presets.get(flowdep_cfg.get('dis_preset', 'medium'),
                                        cv2.DISOpticalFlow_PRESET_MEDIUM),
            dis_finest_scale=flowdep_cfg.get('dis_finest_scale', -1),
            grid_stride=flowdep_cfg.get('grid_stride', 8),
            grid_var_threshold=flowdep_cfg.get('grid_var_threshold', 0.1),
            keyframe_flow_threshold=flowdep_cfg.get('keyframe_flow_threshold', 3.0),
            max_keyframes=flowdep_cfg.get('max_keyframes', 5),
            texture_mask=flowdep_cfg.get('texture_mask', False),
            texture_threshold=flowdep_cfg.get('texture_threshold', 5),
            process_invdepth_var=flowdep_cfg.get('process_invdepth_var', 0.1),
        )
        # Precompute undistortion maps (FlowDep assumes pinhole model)
        K_raw = camera.K_matrix()
        dist_coeffs = np.array(getattr(camera, 'dist', [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        w_cam, h_cam = camera.image_size
        K_undist, _ = cv2.getOptimalNewCameraMatrix(K_raw, dist_coeffs, (w_cam, h_cam), 0)
        flowdep_mapx, flowdep_mapy = cv2.initUndistortRectifyMap(
            K_raw, dist_coeffs, None, K_undist, (w_cam, h_cam), cv2.CV_32FC1,
        )
        K_flowdep = K_undist
        flowdep_filter = FlowDepFilter(K_flowdep, flowdep_settings)
        # Separate PlaneDetector instance for grid mesh plane detection
        grid_stride = flowdep_settings.grid_stride
        flowdep_plane_detector = PlaneDetector(PlaneDetectorSettings(
            max_tri_side_px=int(1.5 * grid_stride),
            max_pairwise_px=int(2.5 * grid_stride),
            max_dist_between_z=0.15,
            max_norm_deg=25,
            min_plane_features=3,
        ))
        print(f"FlowDep enabled (scale={flowdep_settings.image_scale}, "
              f"{int(w_cam * flowdep_settings.image_scale)}x{int(h_cam * flowdep_settings.image_scale)}, "
              f"grid_stride={grid_stride})")

    # FlowDep debug window
    flowdep_debug = None
    if args.flowdep and args.display:
        from eqvio.flowdep_visualiser import FlowDepDebugWindow
        flowdep_debug = FlowDepDebugWindow(enabled=True)
        print("FlowDep debug window enabled (press 'd' depth/var, 'q' quit)")

    timestamps_out = []
    from eqvio.loop_timer import LoopTimer
    timer = LoopTimer()

    states_out = []
    imu_count = 0
    vision_count = 0
    t_start = time.time()

    for event_type, stamp, data in events:
        if stamp < start_time:
            continue
        if stamp > stop_time:
            break

        if event_type == 'imu':
            timer.start("imu_total")
            vio_filter.process_imu(data)
            timer.stop("imu_total")
            imu_count += 1

        elif event_type == 'image':
            # Read image
            timer.start("image_io")
            img_path = data.image_path
            if not img_path.exists():
                timer.stop("image_io")
                continue
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            timer.stop("image_io")
            if image is None:
                continue

            # Get feature predictions for tracker guidance
            timer.start("prediction")
            if settings.use_feature_predictions:
                predictions = vio_filter.get_feature_predictions(camera, stamp)
                pred_cv = {
                    fid: np.array(px, dtype=np.float32)
                    for fid, px in predictions.items()
                }
            else:
                pred_cv = {}
            timer.stop("prediction")

            # Track features
            timer.start("tracking")
            tracker.process_image(image, predicted_features=pred_cv)
            features = tracker.output_features()
            timer.stop("tracking")

            # Convert to VisionMeasurement
            meas = convert_gift_features(features, stamp)
            meas.camera_ptr = camera

            # Capture the predicted state before the vision update for visualization
            predicted_state = vio_filter.state_estimate()

            # Vision update (pass FlowDep for landmark depth warm-start)
            timer.start("vision_update")
            vio_filter.process_vision(meas, camera, flowdep=flowdep_filter)
            timer.stop("vision_update")
            vision_count += 1

            # Record state
            state = vio_filter.state_estimate()
            timestamps_out.append(vio_filter.get_time())
            states_out.append(state)

            # FlowDep: feed EqF camera pose + grayscale image
            if flowdep_filter is not None:
                timer.start("flowdep")
                # T_WC via SE3 composition: T_{G←C} = T_{G←I} * T_{I←C}
                T_WC = (state.sensor.pose * state.sensor.camera_offset).asMatrix()
                image_undist = cv2.remap(image, flowdep_mapx, flowdep_mapy, cv2.INTER_LINEAR)
                flowdep_filter.process_frame(image_undist, T_WC, stamp)
                timer.stop("flowdep")

            # Live trajectory plot
            if visualiser is not None:
                #visualiser.update(vio_filter.get_time(), state)
                visualiser.update(vio_filter.get_time(), state, eqf=vio_filter.eqf, predicted_state=predicted_state)

            # Plane detection + fitting
            feat2plane = {}
            grid_feat2plane = {}
            tri_data = None
            plane_cps = {}
            feat_pos = {}
            cam_pos = None
            R_GtoC = None

            # FlowDep mesh plane detection: grid pseudo-features → PlaneDetector
            # → relabel EqVIO landmarks → fit CP on EqVIO landmarks
            if flowdep_plane_detector is not None and flowdep_filter is not None and flowdep_filter.invdepth_state is not None:
                timer.start("plane_detect")
                # Get grid pseudo-features in world frame
                grid_uvs, grid_pos = flowdep_filter.grid_features_global(T_WC)
                if len(grid_pos) >= 3:
                    if cam_pos is None:
                        feat_pos, cam_pos, R_GtoC = landmarks_to_global(state)
                    flowdep_plane_detector.update(grid_uvs, grid_pos, cam_pos, R_GtoC)
                    grid_feat2plane = flowdep_plane_detector.feat2plane

                    if grid_feat2plane:
                        # Relabel EqVIO landmarks by grid cell plane IDs
                        eqvio_uvs = {
                            f.id_number: (float(f.cam_coordinates[0]),
                                          float(f.cam_coordinates[1]))
                            for f in features
                        }
                        H_img, W_img = flowdep_filter.invdepth_state.shape
                        grid_cols = W_img // flowdep_settings.grid_stride
                        feat2plane = relabel_landmarks_by_grid(
                            eqvio_uvs, grid_feat2plane,
                            grid_cols, flowdep_settings.grid_stride,
                            image_scale=flowdep_settings.image_scale,
                        )
                timer.stop("plane_detect")

                # Fit CP using EqVIO landmark 3D positions
                if feat2plane:
                    timer.start("plane_fit")
                    if not feat_pos:
                        feat_pos, cam_pos, R_GtoC = landmarks_to_global(state)
                    plane_cps, plane_inliers = fit_detected_planes(feat2plane, feat_pos)
                    timer.stop("plane_fit")

                if plane_cps and args.planes:
                    timer.start("plane_augment")
                    vio_filter.augment_planes(plane_cps, plane_inliers)
                    timer.stop("plane_augment")

            elif plane_detector is not None:
                timer.start("plane_detect")
                # Build feature pixel positions
                feat_uvs = {
                    f.id_number: (float(f.cam_coordinates[0]),
                                  float(f.cam_coordinates[1]))
                    for f in features
                }
                # Get 3D landmarks in global frame
                if not feat_pos:
                    feat_pos, cam_pos, R_GtoC = landmarks_to_global(state)
                plane_detector.update(feat_uvs, feat_pos, cam_pos, R_GtoC)
                feat2plane = plane_detector.feat2plane
                tri_data = plane_detector.delaunay_data
                timer.stop("plane_detect")

                # Fit CP for each detected plane (RANSAC)
                if feat2plane:
                    timer.start("plane_fit")
                    plane_cps, plane_inliers = fit_detected_planes(feat2plane, feat_pos)
                    timer.stop("plane_fit")

                # Augment filter state with new planes
                if plane_cps:
                    timer.start("plane_augment")
                    vio_filter.augment_planes(plane_cps, plane_inliers)
                    timer.stop("plane_augment")

            # Camera debug window
            if cam_debug is not None:
                slam_ids = {lm.id for lm in state.camera_landmarks}
                tri_kw = {}
                if tri_data is not None:
                    tri_kw = dict(
                        tri_simplices=tri_data[0],
                        tri_feat_ids=tri_data[1],
                        tri_normals=tri_data[2],
                    )
                fd_gcols_cam = 0
                if flowdep_filter is not None and flowdep_filter.invdepth_state is not None:
                    fd_gcols_cam = flowdep_filter.invdepth_state.shape[1] // flowdep_settings.grid_stride
                cam_debug.update(
                    image,
                    features=features,
                    feat2plane=feat2plane,
                    slam_feat_ids=slam_ids,
                    grid_feat2plane=grid_feat2plane,
                    grid_feat_norms=flowdep_plane_detector._feat_norms if flowdep_plane_detector is not None else None,
                    grid_cols=fd_gcols_cam,
                    grid_stride=flowdep_settings.grid_stride if flowdep_filter is not None else 8,
                    grid_image_scale=flowdep_settings.image_scale if flowdep_filter is not None else 1.0,
                    **tri_kw,
                )
                if not cam_debug.enabled:   # user pressed 'q'
                    break

            # FlowDep debug window
            if flowdep_debug is not None and flowdep_filter is not None:
                flowdep_debug.update(
                    flowdep_filter.invdepth_state,
                    flowdep_filter.invdepth_var,
                )
                if not flowdep_debug.enabled:
                    break

            # Progress
            if vision_count % 50 == 0:
                pos = state.sensor.pose.x
                n_planes = len(set(feat2plane.values())) if feat2plane else 0
                n_fitted = len(plane_cps)
                n_filter_planes = len(state.plane_landmarks)
                print(f"  [{vision_count:4d}] t={stamp:.3f}  "
                      f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  "
                      f"landmarks={len(state.camera_landmarks)}  "
                      f"planes={n_planes} ({n_fitted} fitted, {n_filter_planes} in filter)")
                for pid, cp in plane_cps.items():
                    n_on = sum(1 for p in feat2plane.values() if p == pid)
                    in_state = "✓" if pid in {pl.id for pl in state.plane_landmarks} else " "
                    print(f"         [{in_state}] plane {pid}: "
                          f"cp=({cp[0]:+.2f},{cp[1]:+.2f},{cp[2]:+.2f})  "
                          f"d={np.linalg.norm(cp):.2f}m  feats={n_on}")

            timer.end_loop()

    elapsed = time.time() - t_start
    print(f"\nProcessed {imu_count} IMU + {vision_count} vision in {elapsed:.2f}s")
    print(f"\n{timer.summary()}")

    # ------------------------------------------------------------------
    # 6. Output
    # ------------------------------------------------------------------
    if args.output is None:
        args.output = Path(f"eqvio_output_{args.dataset.name}")

    traj_file = args.output / "estimated_trajectory.txt"
    write_trajectory(traj_file, timestamps_out, states_out)
    print(f"Trajectory written to {traj_file}")

    # Also write ground truth in same format for comparison
    gt_poses = reader.groundtruth()
    if gt_poses:
        gt_file = args.output / "groundtruth_trajectory.txt"
        gt_file.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_file, 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for pose in gt_poses:
                q = pose.pose.R.asQuaternion()
                x = pose.pose.x
                writer.writerow([
                    f"{pose.t:.9f}",
                    f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}",
                    f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}",
                ])
        print(f"Ground truth written to {gt_file}")

        # Write Umeyama-aligned trajectory
        if len(states_out) > 100:
            from eqvio.alignment import align_trajectories
            from eqvio.mathematical.vio_state import StampedPose

            est_stamped = [
                StampedPose(t=t, pose=s.sensor.pose)
                for t, s in zip(timestamps_out, states_out)
            ]
            T_align = align_trajectories(est_stamped, gt_poses)

            aligned_file = args.output / "aligned_trajectory.txt"
            with open(aligned_file, 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                for t, state in zip(timestamps_out, states_out):
                    aligned_pose = T_align * state.sensor.pose
                    x = aligned_pose.x
                    q = aligned_pose.R.asQuaternion()
                    writer.writerow([
                        f"{t:.9f}",
                        f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}",
                        f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}",
                    ])
            print(f"Aligned trajectory written to {aligned_file}")

    if cam_debug is not None:
        cam_debug.close()
    if flowdep_debug is not None:
        flowdep_debug.close()

    if visualiser is not None:
        visualiser.finish()

    print("Done.")


if __name__ == "__main__":
    # Check for --profile before argparse to decide wrapping
    import sys
    if '--profile' in sys.argv:
        import cProfile
        import pstats
        import io
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(40)
        print("\n" + s.getvalue())
    else:
        main()
