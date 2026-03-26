# ECHO-LI-python

**Equivariant Cortical Hybrid Odometry via Lie-group Inference**

A Python-first equivariant visual-inertial odometry system with planar landmark support, grounded in Lie group geometry and inspired by biological spatial perception.

ECHO-LI extends the [EqVIO](https://github.com/pvangoor/EqVIO) framework (van Goor & Mahony, 2023) with dual-action planar landmarks using a closest point (CP) parameterisation, enabling structure-aware state estimation within the equivariant filter (EqF) formalism.

---

## Motivation

Biological navigation systems -- from drosophila lobula plate tangential cells to mammalian head direction circuits — achieve robust ego-motion estimation by fusing visual and inertial cues through structured, low-dimensional representations. ECHO-LI takes this as motivation: rather than treating SLAM as a generic nonlinear estimation problem, we exploit the symmetry structure of the state space via Lie group actions, yielding trajectory-independent linearisation and geometrically consistent updates.

The planar landmark extension is particularly natural in this framework. The point-on-plane constraint $h(p, q) = q^Tp + 1$ is exactly invariant under the dual SOT(3) action (scales and rotations cancel), requiring no nullspace projection — both point and plane landmarks live in the filter state.

## Features

- **Equivariant filter (EqF)** on $SE_2(3) × SOT(3)^n$ with trajectory-independent linearisation
- **Dual-action planar landmarks** via CP parameterisation (q = n/d), with dramatically simplified plane lift $\Lambda_\pi$
- **Plane detection pipeline** — Delaunay triangulation, RANSAC + SVD plane fitting, joint CP + feature refinement (scipy `least_squares` with Cauchy loss)
- **Real-time visualisation** — per-plane feature colouring, convex hulls, Delaunay overlays with surface-normal colouring, multi-mode debug camera window
- **Python-first design** — separates mathematical debugging from compilation complexity; SymPy-based Jacobian validation; later language translation becomes mechanical once Python passes

## Research Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **(a)** | EqVIO + dual-action planar landmarks (CP parameterisation) | Active |
| **(c)** | Loose coupling with dense depth Kalman filter | Planned |
| **(b)** | ROVIO-style photometric tight coupling (wgpu) | Future maybe |
| — | Dual-axis orientation manifold ($T_1S^2$ replacing SO(3)) | Separate paper |

## Dependencies

- Python ≥ 3.10
- NumPy, SciPy
- OpenCV (visualization, KLT tracking)
- PyQtGraph (visualization)
- pytest (testing)

## Quick Start

```bash
# Clone
git clone https://github.com/twetto/echo-li-python.git
cd echo-li-python

# Install
pip install -e .

# Run on EuRoC MAV dataset
python scripts/run_euroc.py /path/to/V1_01_easy/ --config configs/eqvio_euroc_euclid.yaml --max-features 40 --display --plot --planes

# Enable plane detection
python scripts/run_euroc.py /path/to/V1_01_easy/ --config configs/eqvio_euroc_euclid.yaml --max-features 40 --display --plot
```

## Datasets

Currently tested on:
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/) (V1_01_easy and others)

## References

- P. van Goor and R. Mahony, "EqVIO: An Equivariant Filter for Visual-Inertial Odometry," *IEEE T-RO*, 2023.
- Y. Chen et al., "Monocular Visual-Inertial Odometry with Planar Regularities," *ICRA*, 2023.

## License

MIT

