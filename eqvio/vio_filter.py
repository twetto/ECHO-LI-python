"""
VIO Filter wrapper: state management, feature augmentation/marginalization.

Port of: VIOFilter.h / VIOFilter.cpp

This is the high-level interface used by run_euroc.py. It wraps VIO_eqf and handles:
    - Filter initialization from settings
    - IMU propagation (observer state + Riccati)
    - Vision update (feature management + Kalman update)
    - Feature augmentation/marginalization
    - Feature prediction for tracker guidance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import yaml

from liepp import SO3, SE3, SOT3

from .mathematical.vio_state import (
    VIOState, VIOSensorState, Landmark, PlaneLandmark, GRAVITY_CONSTANT,
    integrate_system_function,
)
from .mathematical.vio_group import (
    VIOGroup, state_group_action, lift_velocity, vio_exp,
)
from .mathematical.imu_velocity import IMUVelocity
from .mathematical.vio_eqf import VIO_eqf
from .mathematical.vision_measurement import VisionMeasurement, measure_system_state
from .mathematical.plane_measurement import build_stacked_update
from .coordinate_suite.euclid import (
    EqFCoordinateSuite_euclid,
    state_matrix_A_euclid,
    input_matrix_B_euclid,
    state_chart_euclid,
    state_chart_inv_euclid,
    lift_innovation_euclid,
    lift_innovation_discrete_euclid,
)
from .coordinate_suite.invdepth import (
    EqFCoordinateSuite_invdepth,
    state_matrix_A_invdepth,
    input_matrix_B_invdepth,
    state_chart_invdepth,
    state_chart_inv_invdepth,
    lift_innovation_invdepth,
    lift_innovation_discrete_invdepth,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class VIOFilterSettings:
    """Filter configuration.

    Reference: VIOFilterSettings.h struct VIOFilter::Settings
    Matches the YAML structure from EQVIO_config_EuRoC_stationary.yaml.
    """
    # velocityNoise
    sigma_gyroscope: float = 0.000243153572917808
    sigma_accelerometer: float = 0.012438843268295521
    sigma_gyroscope_bias: float = 0.00013372703521098622
    sigma_accelerometer_bias: float = 0.004462289865453429

    # measurementNoise
    sigma_bearing: float = 1.9297839969591413

    # initialVariance
    initial_point_variance: float = 129.90415638150924
    initial_attitude_variance: float = 0.13565029126052572
    initial_position_variance: float = 0.1
    initial_velocity_variance: float = 8.974852995731e-08
    initial_bias_omega_variance: float = 97162.79515771076
    initial_bias_accel_variance: float = 1.5813333765300104
    initial_camera_attitude_variance: float = 0.0010228558965517584
    initial_camera_position_variance: float = 0.023501400846134893

    # initialValue
    initial_scene_depth: float = 5.0

    # processVariance
    process_attitude: float = 6.025875320811407e-05
    process_position: float = 9.981466095928483e-06
    process_velocity: float = 0.025317333863551263
    process_bias_gyr: float = 0.0
    process_bias_acc: float = 0.0
    process_camera_attitude: float = 5.075382174045239e-06
    process_camera_position: float = 1.2188313140115635e-05
    process_point: float = 0.00029845436136043135

    # settings
    use_discrete_lift: bool = True
    use_discrete_correction: bool = False
    use_equivariant_output: bool = True
    use_feature_predictions: bool = False
    coordinate_choice: str = "Euclidean"  # "Euclidean" or "InvDepth"

    # Feature management
    max_landmarks: int = 40
    min_feature_distance: float = 79.0

    # Outlier rejection
    outlier_mahalanobis_threshold: float = 5.0  # chi2 threshold per feature (2 DOF)

    # Plane settings (NEW)
    sigma_constraint: float = 0.50       # constraint noise std (meters)
    initial_plane_variance: float = 1.0  # initial CP covariance (per axis)
    process_plane: float = 0.0003        # plane process noise (similar to point)
    min_plane_points: int = 4            # remove plane if fewer points remain
    constraint_update_interval: int = 1  # apply constraints every N vision frames
    plane_max_point_var: float = 1.0     # only use points with var below this for plane detection
    constraint_max_point_var: float = 0.0  # only constrain points with var below this (0=disabled)

    # Camera offset (body-to-camera SE(3))
    camera_offset: SE3 = field(default_factory=SE3.Identity)

    @staticmethod
    def from_yaml(config: dict) -> VIOFilterSettings:
        """Load settings from the full YAML config dict.

        Expected structure (matching EQVIO_config_EuRoC_stationary.yaml):
            eqf:
                velocityNoise: {gyr, acc, gyrBias, accBias}
                measurementNoise: {feature, ...}
                initialVariance: {attitude, position, velocity, biasGyr, biasAcc, ...}
                initialValue: {sceneDepth}
                processVariance: {attitude, position, velocity, point, ...}
                settings: {useDiscreteVelocityLift, ...}
            GIFT:
                maxFeatures, featureDist, ...
        """
        s = VIOFilterSettings()
        eqf = config.get('eqf', config)

        # velocityNoise
        vn = eqf.get('velocityNoise', {})
        s.sigma_gyroscope = vn.get('gyr', s.sigma_gyroscope)
        s.sigma_accelerometer = vn.get('acc', s.sigma_accelerometer)
        s.sigma_gyroscope_bias = vn.get('gyrBias', s.sigma_gyroscope_bias)
        s.sigma_accelerometer_bias = vn.get('accBias', s.sigma_accelerometer_bias)

        # measurementNoise
        mn = eqf.get('measurementNoise', {})
        s.sigma_bearing = mn.get('feature', s.sigma_bearing)
        # featureOutlierAbs in C++ config maps to our Mahalanobis threshold
        s.outlier_mahalanobis_threshold = mn.get('featureOutlierAbs', s.outlier_mahalanobis_threshold)

        # initialVariance
        iv = eqf.get('initialVariance', {})
        s.initial_point_variance = iv.get('point', s.initial_point_variance)
        s.initial_attitude_variance = iv.get('attitude', s.initial_attitude_variance)
        s.initial_position_variance = iv.get('position', s.initial_position_variance)
        s.initial_velocity_variance = iv.get('velocity', s.initial_velocity_variance)
        s.initial_bias_omega_variance = iv.get('biasGyr', s.initial_bias_omega_variance)
        s.initial_bias_accel_variance = iv.get('biasAcc', s.initial_bias_accel_variance)
        s.initial_camera_attitude_variance = iv.get('cameraAttitude', s.initial_camera_attitude_variance)
        s.initial_camera_position_variance = iv.get('cameraPosition', s.initial_camera_position_variance)

        # initialValue
        ival = eqf.get('initialValue', {})
        s.initial_scene_depth = ival.get('sceneDepth', s.initial_scene_depth)

        # processVariance
        pv = eqf.get('processVariance', {})
        s.process_attitude = pv.get('attitude', s.process_attitude)
        s.process_position = pv.get('position', s.process_position)
        s.process_velocity = pv.get('velocity', s.process_velocity)
        s.process_bias_gyr = pv.get('biasGyr', s.process_bias_gyr)
        s.process_bias_acc = pv.get('biasAcc', s.process_bias_acc)
        s.process_camera_attitude = pv.get('cameraAttitude', s.process_camera_attitude)
        s.process_camera_position = pv.get('cameraPosition', s.process_camera_position)
        s.process_point = pv.get('point', s.process_point)

        # settings
        st = eqf.get('settings', {})
        s.use_discrete_lift = st.get('useDiscreteVelocityLift', s.use_discrete_lift)
        s.use_discrete_correction = st.get('useDiscreteInnovationLift', s.use_discrete_correction)
        s.use_equivariant_output = st.get('useEquivariantOutput', s.use_equivariant_output)
        s.use_feature_predictions = st.get('useFeaturePredictions', s.use_feature_predictions)
        s.coordinate_choice = st.get('coordinateChoice', s.coordinate_choice)

        # GIFT section for feature management
        gift = config.get('GIFT', {})
        s.max_landmarks = gift.get('maxFeatures', s.max_landmarks)
        s.min_feature_distance = gift.get('featureDist', s.min_feature_distance)

        return s

    def input_gain_matrix(self) -> np.ndarray:
        """12x12 input noise covariance.

        Cols: [gyr(3), acc(3), gyr_bias_vel(3), acc_bias_vel(3)]
        """
        Q = np.zeros((12, 12))
        Q[0:3, 0:3] = np.eye(3) * self.sigma_gyroscope ** 2
        Q[3:6, 3:6] = np.eye(3) * self.sigma_accelerometer ** 2
        Q[6:9, 6:9] = np.eye(3) * self.sigma_gyroscope_bias ** 2
        Q[9:12, 9:12] = np.eye(3) * self.sigma_accelerometer_bias ** 2
        return Q

    def output_gain_matrix(self, n_observations: int) -> np.ndarray:
        """(2*n_obs, 2*n_obs) measurement noise covariance."""
        return np.eye(2 * n_observations) * self.sigma_bearing ** 2

    def initial_covariance(self, n_landmarks: int) -> np.ndarray:
        """Initial Riccati covariance."""
        S = VIOSensorState.CDim
        dim = S + 3 * n_landmarks
        P = np.zeros((dim, dim))

        # Biases [0:6)
        P[0:3, 0:3] = np.eye(3) * self.initial_bias_omega_variance
        P[3:6, 3:6] = np.eye(3) * self.initial_bias_accel_variance
        # Attitude [6:9)
        P[6:9, 6:9] = np.eye(3) * self.initial_attitude_variance
        # Position [9:12)
        P[9:12, 9:12] = np.eye(3) * self.initial_position_variance
        # Velocity [12:15)
        P[12:15, 12:15] = np.eye(3) * self.initial_velocity_variance
        # Camera offset [15:21)
        P[15:18, 15:18] = np.eye(3) * self.initial_camera_attitude_variance
        P[18:21, 18:21] = np.eye(3) * self.initial_camera_position_variance
        # Landmarks
        for i in range(n_landmarks):
            P[S + 3*i:S + 3*(i+1), S + 3*i:S + 3*(i+1)] = (
                np.eye(3) * self.initial_point_variance
            )
        return P

    def state_gain_matrix(self, dim: int) -> np.ndarray:
        """State process noise covariance.

        Matches C++ constructStateGainMatrix().
        """
        S = VIOSensorState.CDim
        n_landmarks = (dim - S) // 3
        Q = np.zeros((dim, dim))

        Q[0:3, 0:3] = np.eye(3) * self.process_bias_gyr
        Q[3:6, 3:6] = np.eye(3) * self.process_bias_acc
        Q[6:9, 6:9] = np.eye(3) * self.process_attitude
        Q[9:12, 9:12] = np.eye(3) * self.process_position
        Q[12:15, 12:15] = np.eye(3) * self.process_velocity
        Q[15:18, 15:18] = np.eye(3) * self.process_camera_attitude
        Q[18:21, 18:21] = np.eye(3) * self.process_camera_position
        for i in range(n_landmarks):
            Q[S + 3*i:S + 3*(i+1), S + 3*i:S + 3*(i+1)] = (
                np.eye(3) * self.process_point
            )
        return Q


# ---------------------------------------------------------------------------
# VIOFilter
# ---------------------------------------------------------------------------

class VIOFilter:
    """High-level VIO filter interface.

    Port of: VIOFilter.h / VIOFilter.cpp

    Usage:
        filter = VIOFilter(settings)
        filter.process_imu(imu_velocity)
        filter.process_vision(vision_measurement, camera)
        state = filter.state_estimate()
    """

    def __init__(self, settings: VIOFilterSettings):
        self.settings = settings

        # Select coordinate suite
        if settings.coordinate_choice.lower() in ('invdepth', 'inv_depth'):
            self.suite = EqFCoordinateSuite_invdepth
            _A = state_matrix_A_invdepth
            _B = input_matrix_B_invdepth
            _lift = lift_innovation_invdepth
            _lift_d = lift_innovation_discrete_invdepth
            _chart = state_chart_invdepth
            # InvDepth chart: constraint C* needs conv_ind2euc
            from .coordinate_suite.invdepth import conv_ind2euc
            self._constraint_chart_jacobian = conv_ind2euc
        else:
            self.suite = EqFCoordinateSuite_euclid
            _A = state_matrix_A_euclid
            _B = input_matrix_B_euclid
            _lift = lift_innovation_euclid
            _lift_d = lift_innovation_discrete_euclid
            _chart = state_chart_euclid
            # Euclidean chart: no transform needed
            self._constraint_chart_jacobian = None

        # Build initial origin state
        xi0 = VIOState()
        xi0.sensor.camera_offset = settings.camera_offset

        # Initialize EqF
        self.eqf = VIO_eqf(
            xi0=xi0,
            X=VIOGroup.Identity(),
            Sigma=settings.initial_covariance(0),
            current_time=-1.0,
            _state_matrix_A=_A,
            _input_matrix_B=_B,
            _output_matrix_C=self._build_output_matrix,
            _lift_innovation=_lift,
            _lift_innovation_discrete=_lift_d,
            _state_chart=_chart,
        )

        self._pending_imu: List[IMUVelocity] = []
        self._cached_input_gain = settings.input_gain_matrix()
        self._cached_state_gain = settings.state_gain_matrix(xi0.dim())
        self._vision_count: int = 0

    def _invalidate_gain_cache(self):
        """Recompute gain matrices after landmark count changes."""
        self._cached_state_gain = self.settings.state_gain_matrix(self.eqf.xi0.dim())

    def get_time(self) -> float:
        return self.eqf.current_time

    def state_estimate(self) -> VIOState:
        return self.eqf.state_estimate()

    def get_velocity_cov(self) -> np.ndarray:
        """Get 3x3 velocity covariance from the Riccati matrix."""
        return self.eqf.get_velocity_cov()

    # -------------------------------------------------------------------
    # IMU processing
    # -------------------------------------------------------------------

    def process_imu(self, imu: IMUVelocity):
        """Process a single IMU measurement.

        Reference: VIOFilter::processIMUData()
        """
        if self.eqf.current_time < 0:
            # First IMU — just set the time
            self.eqf.current_time = imu.stamp
            self._pending_imu.append(imu)
            return

        dt = imu.stamp - self.eqf.current_time
        if dt <= 0:
            return

        # Propagate observer state
        self.eqf.integrate_observer_state(imu, dt, self.settings.use_discrete_lift)

        # Propagate Riccati
        self.eqf.integrate_riccati_fast(
            imu, dt, self._cached_input_gain, self._cached_state_gain
        )

        self.eqf.current_time = imu.stamp
        self._pending_imu.append(imu)

        # Keep only recent IMU for prediction
        if len(self._pending_imu) > 200:
            self._pending_imu = self._pending_imu[-100:]

    # -------------------------------------------------------------------
    # Vision processing
    # -------------------------------------------------------------------

    def process_vision(self, measurement: VisionMeasurement, camera, flowdep=None, tracker=None):
        """Process a vision measurement: manage features, then Kalman update.

        Reference: VIOFilter::processVisionData()

        Args:
            measurement: VisionMeasurement with tracked features
            camera:      GIFT camera model
            flowdep:     Optional FlowDepFilter for warm-starting landmark depth
            tracker:     Optional GIFT tracker. If provided, outlier-rejected
                         features are also discarded from the tracker so it
                         re-detects fresh corners next frame (VINS-FUSION
                         style). Avoids latching onto edges that survive LK
                         but drift in depth.
        """
        if self.eqf.current_time < 0:
            return

        # --- Feature management: add new, remove lost ---
        current_ids = set(self.eqf.X.id)
        observed_ids = set(measurement.cam_coordinates.keys())

        # Remove landmarks not observed
        lost_ids = current_ids - observed_ids
        for lm_id in lost_ids:
            try:
                self.eqf.remove_landmark_by_id(lm_id)
            except StopIteration:
                pass

        # Remove invalid landmarks (degenerate scale)
        self.eqf.remove_invalid_landmarks()

        # Update plane point_ids and remove orphaned planes (NEW)
        if self.eqf.xi0.plane_landmarks:
            active_pt_ids = set(self.eqf.X.id)
            planes_to_remove = []
            for pl in self.eqf.xi0.plane_landmarks:
                pl.point_ids = [pid for pid in pl.point_ids if pid in active_pt_ids]
                if len(pl.point_ids) < self.settings.min_plane_points:
                    planes_to_remove.append(pl.id)
            for pid in planes_to_remove:
                try:
                    self.eqf.remove_plane_by_id(pid)
                except StopIteration:
                    pass
            if planes_to_remove:
                lost_ids = lost_ids | set()  # ensure we invalidate cache

        if lost_ids:
            self._invalidate_gain_cache()

        # Add new landmarks (features not yet in state)
        new_ids = observed_ids - set(self.eqf.X.id)
        if new_ids:
            xi_hat = self.eqf.state_estimate()
            new_landmarks = []
            new_variances = []
            for fid in new_ids:
                if len(self.eqf.X.id) + len(new_landmarks) >= self.settings.max_landmarks:
                    break
                pixel = measurement.cam_coordinates[fid]
                bearing = camera.undistort_point(pixel)

                depth = self.settings.initial_scene_depth
                if flowdep is not None:
                    fd_inv_d, fd_inv_var = flowdep.query(pixel[0], pixel[1])
                    if fd_inv_d > 0:
                        depth = 1.0 / fd_inv_d

                p = bearing * depth
                # NOTE: initial_point_variance is interpreted directly in the
                # active chart's landmark slot (see docs/chart_initial_cov.md).
                point_cov_3x3 = np.eye(3) * self.settings.initial_point_variance

                new_landmarks.append(Landmark(p=p, id=fid))
                new_variances.append(point_cov_3x3)

            if new_landmarks:
                n_new = len(new_landmarks)
                new_cov = np.zeros((3 * n_new, 3 * n_new))
                for i, cov_3x3 in enumerate(new_variances):
                    new_cov[3*i:3*(i+1), 3*i:3*(i+1)] = cov_3x3
                self.eqf.add_new_landmarks(new_landmarks, new_cov)
                self._invalidate_gain_cache()

        # --- Kalman update ---
        if not measurement:
            return

        # Build output matrix and perform update
        y_ids = sorted(
            set(measurement.cam_coordinates.keys()) & set(self.eqf.X.id)
        )
        if not y_ids:
            return

        # --- Innovation-based outlier rejection ---
        # Compute per-feature innovation and reject outliers before update
        xi_hat = self.eqf.state_estimate()
        y_hat = measure_system_state(xi_hat, camera)
        outlier_ids = set()

        for fid in y_ids:
            if fid not in y_hat.cam_coordinates:
                continue
            innov = measurement.cam_coordinates[fid] - y_hat.cam_coordinates[fid]
            innov_norm = np.linalg.norm(innov)

            # Get per-feature innovation covariance for Mahalanobis test
            try:
                lm_cov = self.eqf.get_landmark_cov_by_id(fid)
                # Approximate output covariance: sigma_bearing^2 I + Ci @ P_lm @ Ci^T
                # Simplified: just use pixel innovation norm vs threshold * sigma
                threshold_px = self.settings.outlier_mahalanobis_threshold * self.settings.sigma_bearing
                if innov_norm > threshold_px:
                    outlier_ids.add(fid)
            except (StopIteration, KeyError):
                pass

        # Remove outlier features from measurement before update
        if outlier_ids:
            filtered_meas = VisionMeasurement(
                stamp=measurement.stamp,
                camera_ptr=measurement.camera_ptr,
            )
            for fid, px in measurement.cam_coordinates.items():
                if fid not in outlier_ids:
                    filtered_meas.cam_coordinates[fid] = px
            measurement = filtered_meas
            y_ids = [fid for fid in y_ids if fid not in outlier_ids]

            # Also remove outlier landmarks from state
            for fid in outlier_ids:
                try:
                    self.eqf.remove_landmark_by_id(fid)
                except (StopIteration, KeyError):
                    pass
            self._invalidate_gain_cache()

            if tracker is not None:
                try:
                    tracker.discard_features(outlier_ids)
                except AttributeError:
                    pass

        if not y_ids:
            return

        n_obs = len(y_ids)
        self._vision_count += 1

        if self.eqf.xi0.plane_landmarks:
            # When planes are in state, always use stacked path for correct
            # dimensions. Only include constraint rows every Nth frame.
            use_constraints = (
                self._vision_count % self.settings.constraint_update_interval == 0
            )
            self._stacked_vision_update(
                y_ids, measurement.cam_coordinates,
                measurement.camera_ptr or camera,
                include_constraints=use_constraints,
            )
        else:
            # Standard bearing-only update
            output_gain = self.settings.output_gain_matrix(n_obs)
            self.eqf.perform_vision_update(
                measurement=measurement,
                output_gain=output_gain,
                measure_fn=measure_system_state,
                use_equivariant_output=self.settings.use_equivariant_output,
                discrete_correction=self.settings.use_discrete_correction,
            )

    # -------------------------------------------------------------------
    # Feature prediction (for tracker guidance)
    # -------------------------------------------------------------------

    def get_feature_predictions(self, camera, stamp: float) -> Dict[int, np.ndarray]:
        """Predict feature pixel locations at a future timestamp.

        The filter's observer state is already propagated to current_time
        via IMU, so we just use the current state estimate directly.
        Only integrate forward if stamp is significantly ahead.
        """
        if self.eqf.current_time < 0:
            return {}

        # Current estimate is already at current_time — close enough for
        # tracker seeding. Avoids expensive re-integration of all pending IMU.
        state = self.eqf.state_estimate()

        predictions = {}
        for lm in state.camera_landmarks:
            try:
                pixel = camera.project_point(lm.p)
                predictions[lm.id] = pixel
            except Exception:
                pass

        return predictions

    # -------------------------------------------------------------------
    # Stacked vision update (bearing + constraint, NEW)
    # -------------------------------------------------------------------

    def _stacked_vision_update(
        self, y_ids: List[int], y_coords: Dict[int, np.ndarray], cam_ptr,
        include_constraints: bool = True,
    ):
        """Combined bearing + constraint Kalman update.

        Called when plane landmarks are in the state. Assembles bearing rows
        for all observed points plus (optionally) constraint rows for points
        on planes, then performs a single Kalman update.

        Points with covariance above constraint_max_point_var are excluded
        from constraint rows (but still get bearing rows) to prevent
        unconverged points from pulling the plane.
        """
        # Filter: only constrain well-converged points
        eligible_ids = None
        if include_constraints and self.settings.constraint_max_point_var > 0:
            eligible_ids = set()
            for fid in y_ids:
                try:
                    pcov = self.eqf.get_landmark_cov_by_id(fid)
                    var = np.trace(pcov) / 3.0
                    if var <= self.settings.constraint_max_point_var:
                        eligible_ids.add(fid)
                except (StopIteration, KeyError):
                    pass

        residual, C_star, R_noise = build_stacked_update(
            xi0=self.eqf.xi0,
            X=self.eqf.X,
            y_ids=y_ids,
            y_coords=y_coords,
            cam_ptr=cam_ptr,
            output_matrix_Ci_star=self.suite.output_matrix_Ci_star,
            sigma_bearing=self.settings.sigma_bearing,
            sigma_constraint=self.settings.sigma_constraint,
            use_equivariance=self.settings.use_equivariant_output,
            include_constraints=include_constraints,
            eligible_constraint_ids=eligible_ids,
            point_chart_jacobian=self._constraint_chart_jacobian,
        )

        self.eqf.perform_stacked_update(
            residual, C_star, R_noise,
            discrete_correction=self.settings.use_discrete_correction,
        )

    # -------------------------------------------------------------------
    # Plane augmentation (NEW)
    # -------------------------------------------------------------------

    def augment_planes(
        self,
        plane_cps: Dict[int, np.ndarray],
        plane_inliers: Dict[int, List[int]],
    ):
        """Augment filter state with new plane landmarks.

        Called from the main loop after plane detection + fitting.
        Only augments planes not already in the state.

        Args:
            plane_cps:      {plane_id: cp_inG} from fit_detected_planes
            plane_inliers:  {plane_id: [feat_ids]} RANSAC inlier IDs per plane
        """
        existing_plane_ids = set(self.eqf.X.plane_id)
        active_pt_ids = set(self.eqf.X.id)

        new_planes = []
        for pid, cp_global in plane_cps.items():
            if pid not in plane_inliers:
                continue

            inlier_ids = plane_inliers[pid]

            if pid in existing_plane_ids:
                # Update point_ids for existing planes (only add inliers)
                for pl in self.eqf.xi0.plane_landmarks:
                    if pl.id == pid:
                        new_pts = [fid for fid in inlier_ids if fid in active_pt_ids]
                        pl.point_ids = list(set(pl.point_ids) | set(new_pts))
                        break
                continue

            # Use RANSAC inliers that are in the filter state
            point_ids_on_plane = [fid for fid in inlier_ids if fid in active_pt_ids]

            if len(point_ids_on_plane) < self.settings.min_plane_points:
                continue

            # ----------------------------------------------------------
            # Convert fitting CP to filter CP and transform to camera frame
            #
            # plane_fitting returns cp = -n*d  where  n·p + d = 0
            #   so ||cp|| = distance, n = cp_hat direction
            # Filter needs q = n/d  where  q^T p + 1 = 0
            #   (divide plane eq by d)
            #
            # IMPORTANT: The CP is in the "global" frame produced by
            # landmarks_to_global(), which uses a specific rotation convention.
            # We MUST use the same convention to transform back to camera frame.
            # ----------------------------------------------------------
            from eqvio.plane_detection.plane_fitting import cp_to_abcd
            from eqvio.plane_detection import landmarks_to_global

            abcd = cp_to_abcd(cp_global)
            n_G = abcd[:3]    # unit normal
            d_G = abcd[3]     # signed scalar (n·p + d = 0)

            xi_hat = self.eqf.state_estimate()

            # Use landmarks_to_global's convention for consistency
            _, p_CinG, R_GtoC = landmarks_to_global(xi_hat)

            # Transform plane to camera frame:
            #   p_global = R_GtoC^T @ p_cam + p_CinG
            #   n_G · (R_GtoC^T p_cam + p_CinG) + d_G = 0
            #   => n_C = R_GtoC @ n_G,  d_C = d_G + n_G · p_CinG
            n_C = R_GtoC @ n_G
            d_C = d_G + n_G @ p_CinG

            if abs(d_C) < 0.1:
                continue  # plane behind or too close to camera

            # Filter CP: q_cam = n_C / d_C
            q_cam = n_C / d_C

            # Sanity check: verify constraint is approximately satisfied
            # for associated points already in the state
            max_residual = 0.0
            for fid in point_ids_on_plane:
                lm_idx = next(
                    (i for i, lm in enumerate(xi_hat.camera_landmarks) if lm.id == fid),
                    None
                )
                if lm_idx is not None:
                    p_cam = xi_hat.camera_landmarks[lm_idx].p
                    res = abs(q_cam @ p_cam + 1.0)
                    max_residual = max(max_residual, res)
            if max_residual > 0.5:
                # CP doesn't match the points — skip this plane
                continue

            new_planes.append(PlaneLandmark(
                q=q_cam, id=pid, point_ids=point_ids_on_plane,
            ))

        if not new_planes:
            return

        n_new = len(new_planes)
        new_cov = np.zeros((3 * n_new, 3 * n_new))
        for k, pl in enumerate(new_planes):
            # Scale covariance to ||q||²: a fraction of the CP magnitude squared.
            # initial_plane_variance acts as the relative uncertainty (e.g. 0.01 = 10%)
            q_norm_sq = np.dot(pl.q, pl.q)
            var = self.settings.initial_plane_variance * q_norm_sq
            new_cov[3*k:3*k+3, 3*k:3*k+3] = np.eye(3) * var
        self.eqf.add_new_plane_landmarks(new_planes, new_cov)
        self._invalidate_gain_cache()

    # -------------------------------------------------------------------
    # Internal: output matrix builder (adapts suite to eqf interface)
    # -------------------------------------------------------------------

    def _build_output_matrix(self, xi0, X, measurement, use_equivariance):
        """Adapter: VIO_eqf calls this as _output_matrix_C."""
        y_ids = sorted(
            set(measurement.cam_coordinates.keys()) & set(X.id)
        )
        return self.suite.output_matrix_C(
            xi0, X, y_ids, measurement.cam_coordinates,
            measurement.camera_ptr, use_equivariance,
        )
