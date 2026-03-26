"""
Camera debug window for EqVIO-P.

Composes GIFT feature visualisation with plane detection overlays
in a separate OpenCV window, running alongside the matplotlib
trajectory visualiser.

Usage in run_euroc.py:
    from eqvio.plane_detection import CameraDebugWindow

    cam_debug = CameraDebugWindow()

    # in the image event block, after tracking + vision update:
    cam_debug.update(
        image,
        features=features,
        feat2plane={},             # empty until plane_detector exists
        slam_feat_ids={lm.id for lm in state.camera_landmarks},
    )

    # at shutdown:
    cam_debug.close()
"""

from __future__ import annotations

from typing import Optional, Sequence, Set

import cv2
import numpy as np


class CameraDebugWindow:
    """Manages a live OpenCV debug window showing planes on camera images.

    Modes (cycled with 'm' key):
        0: GIFT features only (yellow dots + optional flow)
        1: Plane hulls + coloured features  (default)
        2: Full diagnostic (Delaunay + normals + hulls)
    """

    WINDOW_NAME = "EqVIO-P Camera Debug"

    def __init__(
        self,
        enabled: bool = True,
        start_mode: int = 1,
        wait_ms: int = 1,
    ):
        self.enabled = enabled
        self.mode = start_mode
        self.wait_ms = wait_ms
        self._window_created = False
        self._prev_features = None

    # ------------------------------------------------------------------

    def update(
        self,
        img: np.ndarray,
        features: Optional[Sequence] = None,
        feat2plane: Optional[dict[int, int]] = None,
        *,
        tri_simplices: Optional[np.ndarray] = None,
        tri_feat_ids: Optional[list[int]] = None,
        tri_normals: Optional[np.ndarray] = None,
        slam_feat_ids: Optional[Set[int]] = None,
        text_overlay: str = "",
    ):
        """Render one frame to the debug window.

        Parameters
        ----------
        img : (H,W) uint8 grayscale camera image
        features : list of GIFT Feature objects (.id_number, .cam_coordinates)
        feat2plane : {feat_id: plane_id} from plane detector (empty dict = no planes yet)
        tri_simplices, tri_feat_ids, tri_normals : Delaunay data (mode 2 only)
        slam_feat_ids : set of feature IDs currently in the EqF state
        text_overlay : extra text for top-left corner
        """
        if not self.enabled:
            return

        from .plane_visualiser import (
            overlay_planes,
            overlay_full_diagnostic,
            highlight_slam_features,
            _ensure_bgr,
        )

        # Build {feat_id: (u, v)} from GIFT features
        feat_uvs: dict[int, tuple[float, float]] = {}
        if features is not None:
            for f in features:
                feat_uvs[f.id_number] = (
                    float(f.cam_coordinates[0]),
                    float(f.cam_coordinates[1]),
                )

        if feat2plane is None:
            feat2plane = {}

        # --- Mode 0: plain GIFT features (yellow dots + flow lines) ---
        if self.mode == 0:
            img_out = self._draw_gift_features(img, features)

        # --- Mode 1: plane hulls + coloured features ---
        elif self.mode == 1:
            img_out = overlay_planes(img, feat_uvs, feat2plane)

        # --- Mode 2: full diagnostic ---
        elif self.mode == 2:
            img_out = overlay_full_diagnostic(
                img, feat_uvs, feat2plane,
                tri_simplices=tri_simplices,
                tri_feat_ids=tri_feat_ids,
                tri_normals=tri_normals,
                text_overlay=text_overlay,
            )
        else:
            img_out = _ensure_bgr(img)

        # SLAM feature highlights (green boxes, modes 1 & 2)
        if self.mode >= 1 and slam_feat_ids:
            img_out = highlight_slam_features(img_out, feat_uvs, slam_feat_ids)

        # Mode indicator
        mode_labels = {0: "GIFT", 1: "Planes", 2: "Full Diag"}
        label = mode_labels.get(self.mode, "?")
        h = img_out.shape[0]
        cv2.putText(
            img_out, f"[{self.mode}] {label}  ('m' cycle, 'q' quit)",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # Stats top-right
        n_feats = len(feat_uvs)
        n_planes = len(set(feat2plane.values())) if feat2plane else 0
        n_slam = len(slam_feat_ids) if slam_feat_ids else 0
        stats = f"F:{n_feats}  P:{n_planes}  S:{n_slam}"
        w = img_out.shape[1]
        cv2.putText(
            img_out, stats,
            (w - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # Show
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self._window_created = True
        cv2.imshow(self.WINDOW_NAME, img_out)

        # Keyboard
        key = cv2.waitKey(self.wait_ms) & 0xFF
        if key == ord('m'):
            self.mode = (self.mode + 1) % 3
        elif key == ord('q'):
            self.enabled = False
            self.close()

        # Save features for next frame's flow drawing
        self._prev_features = features

    # ------------------------------------------------------------------

    def close(self):
        """Destroy the OpenCV window."""
        if self._window_created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._window_created = False

    # ------------------------------------------------------------------
    # Internal: GIFT-style feature drawing (mode 0)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_gift_features(img: np.ndarray, features) -> np.ndarray:
        """Draw GIFT features as yellow circles with ID labels.

        Matches the style of the existing --display block in run_euroc.py.
        """
        if img.ndim == 2:
            out = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        else:
            out = img.copy()

        if features is None:
            return out

        for f in features:
            pt = (int(round(f.cam_coordinates[0])),
                  int(round(f.cam_coordinates[1])))
            cv2.circle(out, pt, 3, (0, 255, 255), cv2.FILLED)
            cv2.putText(out, str(f.id_number), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        return out
