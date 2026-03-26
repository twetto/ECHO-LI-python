"""Plane detection and visualisation for EqVIO-P."""

from .plane_visualiser import (
    overlay_planes,
    overlay_delaunay,
    overlay_full_diagnostic,
    highlight_slam_features,
)
from .camera_debugger import CameraDebugWindow
from .plane_detector import PlaneDetector, PlaneDetectorSettings, landmarks_to_global
from .plane_fitting import (
    fit_plane_linear,
    fit_plane_ransac,
    optimize_plane,
    fit_detected_planes,
    point_to_plane_distance,
    abcd_to_cp,
    cp_to_abcd,
    PlaneFittingSettings,
)

__all__ = [
    # Visualisation
    "overlay_planes",
    "overlay_delaunay",
    "overlay_full_diagnostic",
    "highlight_slam_features",
    "CameraDebugWindow",
    # Detection
    "PlaneDetector",
    "PlaneDetectorSettings",
    "landmarks_to_global",
    # Fitting
    "fit_plane_linear",
    "fit_plane_ransac",
    "optimize_plane",
    "fit_detected_planes",
    "point_to_plane_distance",
    "abcd_to_cp",
    "cp_to_abcd",
    "PlaneFittingSettings",
]
