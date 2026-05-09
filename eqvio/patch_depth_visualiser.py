"""
Patch-grid direct depth debug window.

Shows depth colourmap with status overlay:
    - PHOTO_REFINED cells in full colour
    - SEED_ONLY cells tinted blue
    - UNKNOWN / REJECTED cells black

Keyboard:
    'd' — toggle depth / log-variance display
    's' — toggle status overlay
    'q' — close window
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .patch_depth_mapper import PatchStatus


class PatchDepthDebugWindow:
    """Live OpenCV window for patch-grid direct depth."""

    WINDOW_NAME = "Patch-Grid Direct Depth"

    def __init__(
        self,
        enabled: bool = True,
        wait_ms: int = 1,
        depth_min: float = 0.0,
        depth_max: float = 5.0,
    ):
        self.enabled = enabled
        self.wait_ms = wait_ms
        self.depth_min = depth_min
        self.depth_max = depth_max
        self._window_created = False
        self._show_var = False
        self._show_status = True

    def update(
        self,
        depth_cells: np.ndarray,
        var_cells: np.ndarray,
        status_cells: np.ndarray,
        n_seeds: int = 0,
    ):
        if not self.enabled:
            return

        Hc, Wc = depth_cells.shape[:2]
        scale = max(1, 480 // max(Hc, 1))
        H_disp = Hc * scale
        W_disp = Wc * scale

        valid = np.isfinite(depth_cells) & (depth_cells > 0)

        if self._show_var:
            data = var_cells.copy()
            data[valid] = np.log1p(np.abs(data[valid]) * 1e3)
            label = "Log-Variance"
            if np.any(valid):
                vmin = float(np.percentile(data[valid], 2))
                vmax = float(np.percentile(data[valid], 98))
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-6
            normed = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        else:
            data = depth_cells.copy()
            label = f"Depth [{self.depth_min:.1f}-{self.depth_max:.1f}m]"
            vmin, vmax = self.depth_min, self.depth_max
            # Closer = higher value (brighter in turbo)
            normed = 1.0 - np.clip(
                (data - vmin) / (vmax - vmin), 0, 1,
            )

        normed[~valid] = 0
        gray8 = (normed * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(gray8, cv2.COLORMAP_JET)
        coloured[~valid] = 0

        if self._show_status:
            seed_mask = status_cells == PatchStatus.SEED_ONLY
            coloured[seed_mask] = (
                coloured[seed_mask] * 0.4
                + np.array([200, 100, 0], dtype=np.uint8) * 0.6
            ).astype(np.uint8)

        canvas = cv2.resize(
            coloured, (W_disp, H_disp), interpolation=cv2.INTER_NEAREST,
        )

        n_photo = int(np.sum(status_cells == PatchStatus.PHOTO_REFINED))
        n_seed = int(np.sum(status_cells == PatchStatus.SEED_ONLY))
        n_unk = int(np.sum(status_cells == PatchStatus.UNKNOWN))
        n_rej = int(np.sum(status_cells == PatchStatus.REJECTED))

        cv2.putText(
            canvas, f"Patch Depth ({label})", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        info = (
            f"photo={n_photo} seed={n_seed} unk={n_unk} rej={n_rej} "
            f"seeds={n_seeds}"
        )
        cv2.putText(
            canvas, info, (10, H_disp - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
        )

        if valid.any() and not self._show_var:
            depths_valid = depth_cells[valid]
            dmin = float(np.min(depths_valid))
            dmax = float(np.max(depths_valid))
            dmed = float(np.median(depths_valid))
            range_str = f"depth: {dmin:.1f}-{dmax:.1f}m  med={dmed:.1f}m"
            cv2.putText(
                canvas, range_str, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
            )

        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            self._window_created = True
        cv2.imshow(self.WINDOW_NAME, canvas)

        key = cv2.waitKey(self.wait_ms) & 0xFF
        if key == ord("d"):
            self._show_var = not self._show_var
        elif key == ord("s"):
            self._show_status = not self._show_status
        elif key == ord("q"):
            self.enabled = False
            self.close()

    def close(self):
        if self._window_created:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._window_created = False
