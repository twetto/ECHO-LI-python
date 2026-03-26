"""
ASL (EuRoC) dataset reader.

Port of: ASLDatasetReader.h / ASLDatasetReader.cpp

EuRoC directory layout:
    mav0/
        imu0/
            data.csv          # timestamp[ns], gx, gy, gz, ax, ay, az
            sensor.yaml
        cam0/
            data.csv          # timestamp[ns], filename
            data/             # images
            sensor.yaml       # intrinsics, T_BS extrinsics
        state_groundtruth_estimate0/
            data.csv          # timestamp[ns], px,py,pz, qw,qx,qy,qz, vx,vy,vz, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Iterator, Optional, Tuple
import csv
import numpy as np
import yaml

from liepp import SO3, SE3

from eqvio.mathematical.imu_velocity import IMUVelocity
from eqvio.mathematical.vio_state import StampedPose


@dataclass
class CameraIntrinsics:
    """Minimal camera intrinsics from EuRoC sensor.yaml."""
    width: int = 0
    height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: List[float] = field(default_factory=list)
    dist_model: str = "radtan"


@dataclass
class StampedImagePath:
    """Timestamped image file path (no cv2 dependency at this level)."""
    stamp: float = 0.0
    image_path: Path = field(default_factory=Path)


class ASLDatasetReader:
    """Reader for EuRoC MAV datasets.

    Reference: ASLDatasetReader.h / ASLDatasetReader.cpp

    Usage:
        reader = ASLDatasetReader("/path/to/V1_01_easy/")
        for imu in reader.imu_iter():
            ...
        for img in reader.image_iter():
            ...
        gt = reader.groundtruth()
    """

    def __init__(self, dataset_dir: str, camera_lag: float = 0.0):
        self.dataset_dir = Path(dataset_dir)
        self.camera_lag = camera_lag

        self.imu_csv = self.dataset_dir / "mav0" / "imu0" / "data.csv"
        self.cam_dir = self.dataset_dir / "mav0" / "cam0"
        self.image_csv = self.cam_dir / "data.csv"
        self.gt_csv = (
            self.dataset_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        )
        self.camera_yaml = self.cam_dir / "sensor.yaml"

        # Read camera intrinsics and extrinsics
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.camera_extrinsics: Optional[SE3] = None
        self.camera = None  # GIFT GICamera instance
        if self.camera_yaml.exists():
            self._read_camera(self.camera_yaml)

    def _read_camera(self, camera_file: Path):
        """Parse cam0/sensor.yaml for intrinsics and T_BS extrinsics.

        Reference: ASLDatasetReader::readCamera()

        Uses gift.camera.read_euroc_camera() for the camera model (with
        proper distortion handling), and parses T_BS extrinsics separately.
        """
        from gift.camera import read_euroc_camera

        # Camera model via GIFT (handles radtan, equidistant, etc.)
        self.camera = read_euroc_camera(str(camera_file))

        # Also keep raw intrinsics for reference
        with open(camera_file) as f:
            node = yaml.safe_load(f)

        self.intrinsics = CameraIntrinsics(
            width=node["resolution"][0],
            height=node["resolution"][1],
            fx=node["intrinsics"][0],
            fy=node["intrinsics"][1],
            cx=node["intrinsics"][2],
            cy=node["intrinsics"][3],
            distortion=node.get("distortion_coefficients", []),
            dist_model=node.get("distortion_model", "radtan"),
        )

        # Extrinsics: T_BS (body-to-sensor transform)
        if "T_BS" in node and "data" in node["T_BS"]:
            data = node["T_BS"]["data"]
            # Row-major 4x4
            T = np.array(data, dtype=np.float64).reshape(4, 4)
            R = SO3(matrix=T[:3, :3])
            x = T[:3, 3]
            self.camera_extrinsics = SE3(R=R, x=x)

    def imu_iter(self) -> Iterator[IMUVelocity]:
        """Iterate over IMU measurements.

        Reference: ASLDatasetReader::nextIMU()

        Yields IMUVelocity with timestamps in seconds.
        """
        with open(self.imu_csv) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 7:
                    continue
                vals = [float(x) for x in row]
                yield IMUVelocity(
                    stamp=vals[0] * 1e-9,
                    gyr=np.array(vals[1:4]),
                    acc=np.array(vals[4:7]),
                )

    def image_iter(self) -> Iterator[StampedImagePath]:
        """Iterate over image timestamps and paths.

        Reference: ASLDatasetReader::nextImage()

        Yields StampedImagePath (no OpenCV dependency here).
        """
        with open(self.image_csv) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                stamp = float(row[0]) * 1e-9 - self.camera_lag
                fname = row[1].strip()
                yield StampedImagePath(
                    stamp=stamp,
                    image_path=self.cam_dir / "data" / fname,
                )

    def groundtruth(self) -> List[StampedPose]:
        """Read groundtruth trajectory.

        Reference: ASLDatasetReader::groundtruth()

        EuRoC GT format per line:
            timestamp[ns], px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...

        Returns list of StampedPose (position + orientation).
        """
        if not self.gt_csv.exists():
            return []

        poses = []
        prev_time = -1e8

        with open(self.gt_csv) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 8:
                    continue
                vals = [float(x) for x in row]
                t = vals[0] * 1e-9

                if t <= prev_time + 1e-8:
                    continue  # skip duplicate timestamps

                px, py, pz = vals[1], vals[2], vals[3]
                qw, qx, qy, qz = vals[4], vals[5], vals[6], vals[7]

                # scipy Rotation uses [x, y, z, w] ordering
                R = SO3(quaternion=np.array([qx, qy, qz, qw]))
                pose = SE3(R=R, x=np.array([px, py, pz]))

                poses.append(StampedPose(t=t, pose=pose))
                prev_time = t

        return poses

    def groundtruth_velocities(self) -> List[Tuple[float, np.ndarray]]:
        """Read groundtruth velocities (body-frame).

        EuRoC GT columns 8-10 are vx, vy, vz in world frame.
        Returns list of (timestamp, v_world).
        """
        if not self.gt_csv.exists():
            return []

        vels = []
        with open(self.gt_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 11:
                    continue
                vals = [float(x) for x in row]
                t = vals[0] * 1e-9
                v = np.array([vals[8], vals[9], vals[10]])
                vels.append((t, v))
        return vels
