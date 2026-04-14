"""
Coordinate chart implementations for landmark parameterization.

Maps to C++ src/mathematical/coordinateSuite/:
    euclid.py   <- euclid.cpp
    invdepth.py <- invdepth.cpp
    normal.py   <- normal.cpp
"""

from .euclid import EqFCoordinateSuite_euclid
from .invdepth import EqFCoordinateSuite_invdepth
from .normal import EqFCoordinateSuite_normal
