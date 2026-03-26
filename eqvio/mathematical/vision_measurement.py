"""
Vision measurement: feature pixel coordinates + camera model.

Port of: VisionMeasurement.h / VisionMeasurement.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class VisionMeasurement:
    """A measurement of features from an image.

    Reference: VisionMeasurement.h struct VisionMeasurement

    Fields:
        stamp:           timestamp (seconds)
        cam_coordinates: {id: pixel_coords (2,)} ordered by id
        camera_ptr:      camera model (project_point, undistort_point, projection_jacobian)
    """
    stamp: float = 0.0
    cam_coordinates: Dict[int, np.ndarray] = field(default_factory=dict)
    camera_ptr: object = None  # GIFT camera model

    def get_ids(self) -> List[int]:
        """Get sorted list of feature ids.

        Reference: VisionMeasurement::getIds()
        """
        return sorted(self.cam_coordinates.keys())

    def as_vector(self) -> np.ndarray:
        """Cast to (2*N,) vector, ordered by ascending id.

        Reference: VisionMeasurement::operator VectorXd()
        """
        ids = self.get_ids()
        result = np.zeros(2 * len(ids))
        for i, fid in enumerate(ids):
            result[2 * i:2 * i + 2] = self.cam_coordinates[fid]
        return result

    def __sub__(self, other: VisionMeasurement) -> np.ndarray:
        """Subtract two measurements -> innovation vector (2*N,).

        Reference: operator-(VisionMeasurement, VisionMeasurement)

        Only includes features present in both measurements.
        Returns a flat vector ordered by the ids present in self.
        """
        if isinstance(other, VisionMeasurement):
            common_ids = sorted(
                set(self.cam_coordinates.keys()) & set(other.cam_coordinates.keys())
            )
            result = np.zeros(2 * len(common_ids))
            for i, fid in enumerate(common_ids):
                result[2 * i:2 * i + 2] = (
                    self.cam_coordinates[fid] - other.cam_coordinates[fid]
                )
            return result
        return NotImplemented

    def __add__(self, eta: np.ndarray) -> VisionMeasurement:
        """Add noise vector to all pixel coordinates.

        Reference: operator+(VisionMeasurement, VectorXd)
        """
        ids = self.get_ids()
        assert eta.shape[0] == 2 * len(ids)
        result = VisionMeasurement(
            stamp=self.stamp,
            camera_ptr=self.camera_ptr,
        )
        for i, fid in enumerate(ids):
            result.cam_coordinates[fid] = (
                self.cam_coordinates[fid] + eta[2 * i:2 * i + 2]
            )
        return result

    def __bool__(self) -> bool:
        """Truth value: False if no features."""
        return len(self.cam_coordinates) > 0

    def __len__(self) -> int:
        return len(self.cam_coordinates)


def measure_system_state(state, camera_ptr) -> VisionMeasurement:
    """Produce a measurement from a VIO state by projecting landmarks.

    Reference: measureSystemState() in VIOState.cpp

    Args:
        state:      VIOState with camera_landmarks
        camera_ptr: camera model with project_point(p) -> (2,)

    Returns:
        VisionMeasurement with projected pixel coordinates
    """
    result = VisionMeasurement(camera_ptr=camera_ptr)
    for lm in state.camera_landmarks:
        result.cam_coordinates[lm.id] = camera_ptr.project_point(lm.p)
    return result
