"""
FlowDep debug window: dense depth and plane mask visualisation.

Shows two panels side by side:
    Left:  inverse-depth map (turbo colourmap, clipped to valid range)
    Right: plane mask from grid mesh detection (random colour per plane ID)

Keyboard:
    'd' — toggle depth/variance display on left panel
    'o' — toggle hiding outlier pixels (Vogiatzis a/(a+b) < threshold)
    'q' — close window
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np



class FlowDepDebugWindow:
    """Live OpenCV window for FlowDep dense depth and plane masks."""

    WINDOW_NAME = "FlowDep Debug"

    def __init__(self, enabled: bool = True, wait_ms: int = 1):
        self.enabled = enabled
        self.wait_ms = wait_ms
        self._window_created = False
        self._show_var = False  # False=depth, True=variance
        self._hide_outliers = True

    def update(
        self,
        invdepth: Optional[np.ndarray],
        invdepth_var: Optional[np.ndarray],
        a_state: Optional[np.ndarray] = None,
        b_state: Optional[np.ndarray] = None,
        min_inlier_ratio: float = 0.5,
    ):
        """Render one frame.

        Args:
            invdepth: (H, W) float32 inverse-depth map (<=0 means invalid).
            invdepth_var: (H, W) float32 variance map.
            a_state: (H, W) Vogiatzis inlier Beta shape (optional).
            b_state: (H, W) Vogiatzis outlier Beta shape (optional).
            min_inlier_ratio: pixels with a/(a+b) below this are hidden when
                outlier masking is on.
        """
        if not self.enabled:
            return

        if invdepth is None:
            return

        h, w = invdepth.shape

        inlier_mask: Optional[np.ndarray] = None
        if a_state is not None and b_state is not None:
            denom = a_state + b_state
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(denom > 0, a_state / denom, 0.0)
            inlier_mask = ratio >= min_inlier_ratio

        display_mask = inlier_mask if self._hide_outliers else None
        canvas = self._render_depth(invdepth, invdepth_var, display_mask)

        # Label
        label = "Variance" if self._show_var else "Inv-Depth"
        if inlier_mask is not None:
            valid = invdepth > 0
            n_valid = int(valid.sum())
            if n_valid > 0:
                n_in = int((inlier_mask & valid).sum())
                label += f"  inliers={n_in}/{n_valid} ({100.0 * n_in / n_valid:.0f}%)"
            if self._hide_outliers:
                label += "  [hiding outliers]"
        cv2.putText(
            canvas, label, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "'d' depth/var  'o' outliers  'q' quit",
            (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (180, 180, 180), 1, cv2.LINE_AA,
        )

        # Show
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self._window_created = True
        cv2.imshow(self.WINDOW_NAME, canvas)

        key = cv2.waitKey(self.wait_ms) & 0xFF
        if key == ord("d"):
            self._show_var = not self._show_var
        elif key == ord("o"):
            self._hide_outliers = not self._hide_outliers
        elif key == ord("q"):
            self.enabled = False
            self.close()

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._window_created = False

    # ------------------------------------------------------------------
    # Internal renderers
    # ------------------------------------------------------------------

    def _render_depth(
        self,
        invdepth: np.ndarray,
        invdepth_var: Optional[np.ndarray],
        extra_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Colourmap inverse-depth or variance."""
        valid = invdepth > 0
        if extra_mask is not None:
            valid = valid & extra_mask

        if self._show_var and invdepth_var is not None:
            data = invdepth_var.copy()
            # Log-scale variance for better visibility
            data[valid] = np.log1p(data[valid] * 1e3)
            vmin, vmax = 0.0, float(np.percentile(data[valid], 95)) if np.any(valid) else 1.0
        else:
            data = invdepth.copy()
            if np.any(valid):
                vmin = float(np.percentile(data[valid], 2))
                vmax = float(np.percentile(data[valid], 98))
            else:
                vmin, vmax = 0.0, 1.0

        if vmax <= vmin:
            vmax = vmin + 1e-6

        normed = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        normed[~valid] = 0
        gray8 = (normed * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(gray8, cv2.COLORMAP_TURBO)
        # Black out invalid pixels
        coloured[~valid] = 0
        return coloured

