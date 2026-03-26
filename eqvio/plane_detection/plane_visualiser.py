"""Plane detection debug visualiser for EqVIO-P.

Overlays plane associations, convex hulls, Delaunay triangulations,
and surface-normal coloring onto camera images. Designed to be called
from either GIFT's or EqVIO's existing visualisation loops.

All functions are pure: image_in → image_out (no state mutation).
Uses OpenCV for drawing to match GIFT/ov_plane conventions.

Typical usage
-------------
>>> from eqvio.plane_detection.plane_visualiser import overlay_planes
>>> img_out = overlay_planes(img_gray, feat_uvs, feat2plane)

Or for the full diagnostic view including Delaunay + normals:
>>> img_out = overlay_full_diagnostic(
...     img_gray, feat_uvs, feat2plane,
...     tri_simplices=tri.simplices,  # from scipy.spatial.Delaunay
...     tri_normals=normals_per_triangle,
... )
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _plane_color(plane_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR color for a plane ID.

    Uses a seeded RNG (same approach as ov_plane) so colours are
    stable across frames.  Rejects very dark colours.
    """
    rng = np.random.RandomState(int(plane_id) & 0xFFFFFFFF)
    color = np.zeros(3)
    while np.linalg.norm(color) < 0.45:          # reject near-black
        color = rng.uniform(0.25, 1.0, size=3)
    bgr = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    return bgr


def _normal_to_rgb(n: np.ndarray) -> tuple[int, int, int]:
    """Map a unit normal vector to a BGR colour (ov_plane's unit2rgb).

    n ∈ ℝ³ with ‖n‖≈1  →  mapped to [0,255]³ via (n+1)/2·255.
    """
    if np.linalg.norm(n) < 1e-8:
        return (0, 0, 0)
    n_hat = n / np.linalg.norm(n)
    rgb01 = 0.5 * (n_hat + 1.0)
    return (int(rgb01[0] * 255), int(rgb01[1] * 255), int(rgb01[2] * 255))


# ---------------------------------------------------------------------------
# Core overlay: features coloured by plane + convex hulls
# ---------------------------------------------------------------------------

def overlay_planes(
    img: np.ndarray,
    feat_uvs: dict[int, tuple[float, float]],
    feat2plane: dict[int, int],
    *,
    draw_hulls: bool = True,
    draw_ids: bool = True,
    point_radius: int = 4,
    hull_thickness: int = 2,
    alpha_hull: float = 0.25,
    min_hull_points: int = 3,
) -> np.ndarray:
    """Overlay plane-coloured features and convex hulls on a camera image.

    Parameters
    ----------
    img : (H, W) or (H, W, 3) uint8
        Camera image (gray or BGR). Will be converted to BGR if gray.
    feat_uvs : {feat_id: (u, v), ...}
        Pixel coordinates of currently tracked features.
    feat2plane : {feat_id: plane_id, ...}
        Mapping from feature ID to plane ID. Features not in this dict
        are drawn in white (unassociated).
    draw_hulls : bool
        Whether to draw semi-transparent convex hulls around each plane.
    draw_ids : bool
        Whether to label each hull with its plane ID.
    point_radius, hull_thickness : int
        Drawing sizes.
    alpha_hull : float
        Opacity of the filled convex hull overlay.
    min_hull_points : int
        Minimum features on a plane to draw its hull.

    Returns
    -------
    img_out : (H, W, 3) uint8  BGR annotated image.
    """
    img_out = _ensure_bgr(img)

    # Group features by plane
    plane_feats: dict[int, list[tuple[int, float, float]]] = {}
    for fid, uv in feat_uvs.items():
        pid = feat2plane.get(fid, -1)
        plane_feats.setdefault(pid, []).append((fid, uv[0], uv[1]))

    # Draw unassociated features first (white dots)
    for fid, u, v in plane_feats.get(-1, []):
        cv2.circle(img_out, (int(u), int(v)), point_radius - 1,
                   (255, 255, 255), cv2.FILLED)

    # Draw each plane
    for pid, feats in plane_feats.items():
        if pid == -1:
            continue
        color = _plane_color(pid)
        pts_px = np.array([(u, v) for _, u, v in feats], dtype=np.float32)

        # Draw feature dots
        for _, u, v in feats:
            cv2.circle(img_out, (int(u), int(v)), point_radius,
                       color, cv2.FILLED)

        # Convex hull overlay
        if draw_hulls and len(pts_px) >= min_hull_points:
            hull = cv2.convexHull(pts_px)
            if hull is not None and len(hull) >= 3:
                hull_int = hull.astype(np.int32)
                # Semi-transparent fill
                overlay = img_out.copy()
                cv2.fillConvexPoly(overlay, hull_int, color)
                cv2.addWeighted(overlay, alpha_hull, img_out,
                                1.0 - alpha_hull, 0, img_out)
                # Solid border
                cv2.polylines(img_out, [hull_int], isClosed=True,
                              color=color, thickness=hull_thickness)

        # Plane ID label at centroid
        if draw_ids and len(pts_px) >= min_hull_points:
            cx, cy = pts_px.mean(axis=0)
            cv2.putText(img_out, str(pid), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)

    return img_out


# ---------------------------------------------------------------------------
# Delaunay triangulation overlay with normal colouring
# ---------------------------------------------------------------------------

def overlay_delaunay(
    img: np.ndarray,
    feat_uvs_ordered: np.ndarray,
    tri_simplices: np.ndarray,
    *,
    tri_normals: Optional[np.ndarray] = None,
    edge_color: tuple[int, int, int] = (255, 0, 0),
    edge_thickness: int = 1,
    alpha_fill: float = 0.35,
) -> np.ndarray:
    """Overlay Delaunay triangulation, optionally coloured by surface normal.

    Parameters
    ----------
    img : (H, W) or (H, W, 3)
        Camera image.
    feat_uvs_ordered : (N, 2) float
        Pixel coordinates in the same order as the Delaunay vertex indices.
    tri_simplices : (M, 3) int
        Triangle vertex indices (e.g. from scipy.spatial.Delaunay.simplices).
    tri_normals : (M, 3) float, optional
        Unit surface normal per triangle in global frame. If provided,
        triangles are filled with the normal-to-colour mapping.
    edge_color : BGR tuple
        Colour for triangle edges (used when normals not provided).
    edge_thickness : int
        Thickness of triangle edges.
    alpha_fill : float
        Opacity of normal-coloured triangle fill.

    Returns
    -------
    img_out : (H, W, 3) uint8
    """
    img_out = _ensure_bgr(img)
    pts = feat_uvs_ordered.astype(np.int32)

    # Normal-coloured fill layer
    if tri_normals is not None:
        fill_layer = np.zeros_like(img_out)
        for i, simplex in enumerate(tri_simplices):
            n = tri_normals[i]
            if np.linalg.norm(n) < 1e-8:
                continue
            triangle = pts[simplex].reshape((-1, 1, 2))
            color = _normal_to_rgb(n)
            cv2.fillConvexPoly(fill_layer, triangle, color)
        cv2.addWeighted(fill_layer, alpha_fill, img_out,
                        1.0 - alpha_fill, 0, img_out)

    # Draw edges
    for simplex in tri_simplices:
        p0, p1, p2 = pts[simplex]
        cv2.line(img_out, tuple(p0), tuple(p1), edge_color, edge_thickness)
        cv2.line(img_out, tuple(p1), tuple(p2), edge_color, edge_thickness)
        cv2.line(img_out, tuple(p2), tuple(p0), edge_color, edge_thickness)

    return img_out


# ---------------------------------------------------------------------------
# Full diagnostic: planes + Delaunay + normals combined
# ---------------------------------------------------------------------------

def overlay_full_diagnostic(
    img: np.ndarray,
    feat_uvs: dict[int, tuple[float, float]],
    feat2plane: dict[int, int],
    *,
    tri_simplices: Optional[np.ndarray] = None,
    tri_feat_ids: Optional[list[int]] = None,
    tri_normals: Optional[np.ndarray] = None,
    text_overlay: str = "",
) -> np.ndarray:
    """Combined diagnostic overlay: Delaunay + normals + planes + hulls.

    Parameters
    ----------
    img : camera image
    feat_uvs : {feat_id: (u,v)}
    feat2plane : {feat_id: plane_id}
    tri_simplices : (M,3) int — Delaunay triangle indices into tri_feat_ids
    tri_feat_ids : list of feat IDs matching Delaunay vertex ordering
    tri_normals : (M,3) — surface normal per triangle
    text_overlay : optional text in top-left corner

    Returns
    -------
    img_out : annotated BGR image
    """
    img_out = _ensure_bgr(img)

    # Layer 1: Delaunay + normals (if available)
    if tri_simplices is not None and tri_feat_ids is not None:
        # Build ordered UV array matching Delaunay vertex indices
        uvs_ordered = np.array(
            [feat_uvs.get(fid, (0.0, 0.0)) for fid in tri_feat_ids],
            dtype=np.float32,
        )
        img_out = overlay_delaunay(
            img_out, uvs_ordered, tri_simplices,
            tri_normals=tri_normals,
            alpha_fill=0.3,
        )

    # Layer 2: plane hulls + coloured features
    img_out = overlay_planes(
        img_out, feat_uvs, feat2plane,
        alpha_hull=0.15,      # lighter since Delaunay is already drawn
        point_radius=3,
    )

    # Text overlay
    if text_overlay:
        is_small = min(img_out.shape[:2]) < 400
        scale = 1.0 if is_small else 2.0
        thickness = 1 if is_small else 2
        cv2.putText(img_out, text_overlay, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0),
                    thickness, cv2.LINE_AA)

    return img_out


# ---------------------------------------------------------------------------
# Highlighted SLAM features (box around features in the EqF state)
# ---------------------------------------------------------------------------

def highlight_slam_features(
    img: np.ndarray,
    feat_uvs: dict[int, tuple[float, float]],
    slam_feat_ids: set[int],
    *,
    box_half_size: int = 5,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw green boxes around features that are SLAM landmarks in the state.

    Call this after overlay_planes to add an extra layer.
    """
    img_out = _ensure_bgr(img)
    for fid in slam_feat_ids:
        if fid not in feat_uvs:
            continue
        u, v = int(feat_uvs[fid][0]), int(feat_uvs[fid][1])
        pt1 = (u - box_half_size, v - box_half_size)
        pt2 = (u + box_half_size, v + box_half_size)
        cv2.rectangle(img_out, pt1, pt2, color, 1)
    return img_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Return a writable BGR copy of the input image."""
    if img.ndim == 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    return img.copy()
