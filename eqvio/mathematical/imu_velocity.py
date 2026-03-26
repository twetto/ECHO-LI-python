"""
IMU velocity and measurement representation.

Port of: IMUVelocity.h / IMUVelocity.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np


@dataclass
class IMUVelocity:
    """An Inertial Measurement Unit reading.

    Reference: IMUVelocity.h struct IMUVelocity

    Fields:
        stamp:        timestamp (seconds)
        gyr:          (3,) angular velocity from gyroscope
        acc:          (3,) linear acceleration from accelerometer
        gyr_bias_vel: (3,) gyroscope bias velocity (usually zero)
        acc_bias_vel: (3,) accelerometer bias velocity (usually zero)
    """
    stamp: float = 0.0
    gyr: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyr_bias_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc_bias_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    CDim: ClassVar[int] = 12

    @staticmethod
    def Zero() -> IMUVelocity:
        return IMUVelocity()

    @staticmethod
    def from_vec6(vec: np.ndarray) -> IMUVelocity:
        """Construct from (gyr, acc) vector."""
        return IMUVelocity(gyr=vec[:3].copy(), acc=vec[3:6].copy())

    @staticmethod
    def from_vec12(vec: np.ndarray) -> IMUVelocity:
        """Construct from (gyr, acc, gyr_bias_vel, acc_bias_vel) vector."""
        return IMUVelocity(
            gyr=vec[:3].copy(), acc=vec[3:6].copy(),
            gyr_bias_vel=vec[6:9].copy(), acc_bias_vel=vec[9:12].copy(),
        )

    def __add__(self, other: IMUVelocity) -> IMUVelocity:
        """Add two IMU velocities.

        Reference: IMUVelocity::operator+()
        Stamp taken from whichever has a positive stamp (self preferred).
        """
        return IMUVelocity(
            stamp=self.stamp if self.stamp > 0 else other.stamp,
            gyr=self.gyr + other.gyr,
            acc=self.acc + other.acc,
            gyr_bias_vel=self.gyr_bias_vel + other.gyr_bias_vel,
            acc_bias_vel=self.acc_bias_vel + other.acc_bias_vel,
        )

    def __sub__(self, other) -> IMUVelocity:
        """Subtract a bias vector or another IMUVelocity.

        Reference: IMUVelocity::operator-(vec6) and operator-(vec12)

        Accepts:
            np.ndarray of shape (6,):  subtract from (gyr, acc), keep bias vels
            np.ndarray of shape (12,): subtract from all components
            IMUVelocity: subtract all components
        """
        if isinstance(other, np.ndarray):
            if other.shape == (6,):
                return IMUVelocity(
                    stamp=self.stamp,
                    gyr=self.gyr - other[:3],
                    acc=self.acc - other[3:6],
                    gyr_bias_vel=self.gyr_bias_vel.copy(),
                    acc_bias_vel=self.acc_bias_vel.copy(),
                )
            elif other.shape == (12,):
                return IMUVelocity(
                    stamp=self.stamp,
                    gyr=self.gyr - other[:3],
                    acc=self.acc - other[3:6],
                    gyr_bias_vel=self.gyr_bias_vel - other[6:9],
                    acc_bias_vel=self.acc_bias_vel - other[9:12],
                )
            else:
                raise ValueError(f"Cannot subtract array of shape {other.shape}")
        elif isinstance(other, IMUVelocity):
            return IMUVelocity(
                stamp=self.stamp,
                gyr=self.gyr - other.gyr,
                acc=self.acc - other.acc,
                gyr_bias_vel=self.gyr_bias_vel - other.gyr_bias_vel,
                acc_bias_vel=self.acc_bias_vel - other.acc_bias_vel,
            )
        else:
            raise TypeError(f"Cannot subtract {type(other)} from IMUVelocity")

    def __mul__(self, c: float) -> IMUVelocity:
        """Scale by constant.

        Reference: IMUVelocity::operator*(double)
        """
        return IMUVelocity(
            stamp=self.stamp,
            gyr=self.gyr * c,
            acc=self.acc * c,
            gyr_bias_vel=self.gyr_bias_vel * c,
            acc_bias_vel=self.acc_bias_vel * c,
        )

    def __rmul__(self, c: float) -> IMUVelocity:
        return self.__mul__(c)

    def as_vec12(self) -> np.ndarray:
        """Return as (12,) vector: [gyr, acc, gyr_bias_vel, acc_bias_vel]."""
        return np.concatenate([self.gyr, self.acc, self.gyr_bias_vel, self.acc_bias_vel])
