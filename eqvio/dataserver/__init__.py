"""
Dataset readers for VIO evaluation.

Only ASL (EuRoC) is ported initially.
Maps to C++ src/dataserver/ASLDatasetReader.cpp
"""

from .asl_dataset import ASLDatasetReader, StampedImagePath, CameraIntrinsics
