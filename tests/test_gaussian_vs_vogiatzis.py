"""Convergence rate: Gaussian (Kalman) vs Gaussian-Beta (Vogiatzis) × 4 parametrizations.

Simulates a static feature at known depth with noisy triangulated
observations (20 % outliers).  Compares convergence of:

    Filter type          | Configuration
    ---------------------+------------------------------------------
    Pure Gaussian        | a_init → ∞, b_init → 0  ⇒  w1 ≈ 1 always
    Gaussian-Beta (Vog)  | a_init=10, b_init=2      ⇒  mixture

Across 4 parametrizations:

    Euclidean   z           τ² = z⁴·σ²/d²
    InvDepth    ρ = 1/z     τ² = σ²/d²
    Polar       d = log z   τ² = z²·σ²/d²
    Polar3D     Normal chart (3×3 cov, sequential bearing + Vog depth)

Output: convergence_gaussian_vs_vogiatzis.png  (depth error, canonical
variance, inlier ratio vs observation step).

Usage:
    python tests/test_gaussian_vs_vogiatzis.py
    python tests/test_gaussian_vs_vogiatzis.py --no-outliers
    python tests/test_gaussian_vs_vogiatzis.py --outlier-rate 0.4
"""

from __future__ import annotations

import argparse
import math
import sys
import numpy as np
import matplotlib.pyplot as plt

from eqvio.sparse_vogiatzis import (
    SparseVogiatzisFilter,
    SparseVogiatzisFilter3D,
    SparseVogSettings,
    DepthParametrization
)
from eqvio.mathematical.vision_measurement import VisionMeasurement

# ================================================================
# Shared constants
# ================================================================

# Z_TRUE = 5.0
Z_TRUE = 100.0
FX = FY = 458.0
CX, CY = 376.0, 240.0
# SIGMA_PIXEL = 0.5
SIGMA_PIXEL = 0.1
BASELINE_PER_FRAME = 0.05        # camera moves 5 cm per frame
DT = 0.05                        # 20 fps
N_STEPS = 100
SEED = 42

# True pixel of the feature (arbitrary, inside 752×480)
U_TRUE = 400.0
V_TRUE = 250.0

K_MAT = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
])

# ================================================================
# Generate data (shared across all filters)
# ================================================================

def _gen_data(rng, n_steps, z_true, outlier_rate):
    """Generate sequence of camera poses and pixel observations.

    Noise is applied in image space (pixels), then triangulated.
    Camera moves along X axis.
    """
    # Feature in world frame (at depth Z_TRUE from camera 0)
    x_n = (U_TRUE - CX) / FX
    y_n = (V_TRUE - CY) / FY
    P_W = np.array([x_n * z_true, y_n * z_true, z_true])
    
    data = []
    for i in range(n_steps + 1):
        # Camera moves laterally
        bl = i * BASELINE_PER_FRAME
        T_WC = np.eye(4)
        T_WC[0, 3] = bl
        
        # Point in current camera frame
        T_CW = np.linalg.inv(T_WC)
        P_C = T_CW[:3, :3] @ P_W + T_CW[:3, 3]
        
        # True pixel
        u_true = FX * P_C[0] / P_C[2] + CX
        v_true = FY * P_C[1] / P_C[2] + CY
        
        is_outlier = False
        if i > 0 and rng.random() < outlier_rate:
            is_outlier = True
            # For outlier, pick a random depth in [0.5, 30]
            z_out = rng.uniform(0.5, 30.0)
            
            # Find pixel that would produce this depth relative to previous noisy pixel
            u_prev = data[-1][1][0]
            # In pure lateral motion: z = bl / (x_prev - x_curr)
            # x_curr = x_prev - bl / z
            x_prev = (u_prev - CX) / FX
            x_curr = x_prev - BASELINE_PER_FRAME / z_out
            u_obs = x_curr * FX + CX
            v_obs = v_true + rng.normal(0, SIGMA_PIXEL)
        else:
            u_obs = u_true + rng.normal(0, SIGMA_PIXEL)
            v_obs = v_true + rng.normal(0, SIGMA_PIXEL)
            
        data.append((T_WC, np.array([u_obs, v_obs]), is_outlier))
    return data


# ================================================================
# Filter wrappers
# ================================================================

def _run_1d(param: DepthParametrization, is_vogiatzis: bool, data,
            P_vv=None, ab_max=20.0, min_inlier_ratio=0.5):
    s = SparseVogSettings(
        parametrization=param,
        a_init=10.0 if is_vogiatzis else 1e8, # Pure Gaussian
        b_init=2.0 if is_vogiatzis else 1e-8,
        ab_max=ab_max if is_vogiatzis else 1e9,
        min_inlier_ratio=min_inlier_ratio,
        min_track_length=1,
        sigma_pixel=SIGMA_PIXEL,
        process_depth_var=0.0, # static feature
    )

    filt = SparseVogiatzisFilter(K_MAT, s)

    n = len(data) - 1
    err, var_, ir = np.empty(n), np.empty(n), np.empty(n)

    for i in range(n + 1):
        T_WC, uv, _ = data[i]
        meas = VisionMeasurement(
            stamp=i * DT,
            cam_coordinates={42: uv},
        )
        filt.update(meas, T_WC, P_vv=P_vv)

        if i > 0:
            feat = filt.features.get(42)
            if feat:
                z_est = filt._canonical_to_depth(feat.canonical)
                err[i-1] = abs(z_est - Z_TRUE)
                var_[i-1] = feat.canonical_var
                ir[i-1] = feat.inlier_ratio()
            else:
                err[i-1] = np.nan
                var_[i-1] = np.nan
                ir[i-1] = 0.0
    return err, var_, ir


def _run_3d(is_vogiatzis: bool, data, P_vv=None, ab_max=20.0,
            min_inlier_ratio=0.5):
    s = SparseVogSettings(
        a_init=10.0 if is_vogiatzis else 1e8,
        b_init=2.0 if is_vogiatzis else 1e-8,
        ab_max=ab_max if is_vogiatzis else 1e9,
        min_inlier_ratio=min_inlier_ratio,
        min_track_length=1,
        sigma_pixel=SIGMA_PIXEL,
        process_depth_var=0.0,
    )
    filt = SparseVogiatzisFilter3D(K_MAT, s)

    n = len(data) - 1
    err, var_, ir = np.empty(n), np.empty(n), np.empty(n)

    for i in range(n + 1):
        T_WC, uv, _ = data[i]
        meas = VisionMeasurement(
            stamp=i * DT,
            cam_coordinates={42: uv},
        )
        filt.update(meas, T_WC, P_vv=P_vv)

        if i > 0:
            feat = filt.features.get(42)
            if feat:
                err[i-1] = abs(feat.depth - Z_TRUE)
                # In Normal chart, index 2 is log(ρ) = -log(z)
                var_[i-1] = feat.covariance[2, 2]
                ir[i-1] = feat.inlier_ratio()
            else:
                err[i-1] = np.nan
                var_[i-1] = np.nan
                ir[i-1] = 0.0
    return err, var_, ir


# ================================================================
# Main
# ================================================================

def _plot_results(axes, results, is_out, colors, title_suffix=""):
    """Plot depth error, variance, and inlier ratio on the given axes."""
    n = len(is_out)
    steps = np.arange(n)

    for key, (err, var_, ir) in results.items():
        p = key.split(" / ")[0]
        cfg_label = key.split(" / ")[1]
        c = colors[p]
        ls = "-" if "Gaussian-Beta" in cfg_label else "--"
        lw = 1.5 if "Gaussian-Beta" in cfg_label else 1.0

        axes[0].semilogy(steps, np.maximum(err, 1e-6),
                         color=c, ls=ls, lw=lw, alpha=0.85, label=key)
        axes[1].semilogy(steps, np.maximum(var_, 1e-12),
                         color=c, ls=ls, lw=lw, alpha=0.85, label=key)
        axes[2].plot(steps, ir,
                     color=c, ls=ls, lw=lw, alpha=0.85, label=key)

    out_idx = np.where(is_out)[0]
    for ax in axes:
        for oi in out_idx:
            ax.axvline(oi, color="red", alpha=0.08, lw=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier-rate", type=float, default=0.2)
    ap.add_argument("--no-outliers", action="store_true")
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save", type=str, default="convergence_gaussian_vs_vogiatzis.png")
    args = ap.parse_args()

    outlier_rate = 0.0 if args.no_outliers else args.outlier_rate
    n = args.n_steps
    rng = np.random.default_rng(args.seed)

    data = _gen_data(rng, n, Z_TRUE, outlier_rate)
    is_out = np.array([d[2] for d in data[1:]])

    n_outliers = int(is_out.sum())
    print(f"z*={Z_TRUE:.1f}m  N={n}  outlier_rate={outlier_rate:.0%}  "
          f"({n_outliers} outliers)  seed={args.seed}")

    # Velocity covariances for Part 1 improvements.
    P_vv_3x3 = np.diag([0.01, 0.01, 0.01])  # translational velocity cov
    P_UU_6x6 = np.diag([1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01])  # [\u03c9; v]

    params = [
        (DepthParametrization.EUCLIDEAN, "euclidean"),
        (DepthParametrization.INVDEPTH,  "invdepth"),
        (DepthParametrization.POLAR,     "polar"),
    ]

    colors = {
        "euclidean": "tab:blue",
        "invdepth":  "tab:orange",
        "polar":     "tab:green",
        "polar3d":   "tab:red",
    }

    # ---- Panel A: baseline (no P_vv) ----
    results_base: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p_enum, p_name in params:
        for vog in [True, False]:
            label = "Gaussian-Beta" if vog else "Gaussian"
            key = f"{p_name} / {label}"
            results_base[key] = _run_1d(p_enum, vog, data)
    for vog in [True, False]:
        label = "Gaussian-Beta" if vog else "Gaussian"
        key = f"polar3d / {label}"
        results_base[key] = _run_3d(vog, data)

    # ---- Panel B: with P_vv (3\u00d73) \u2014 baseline \u03c4\u00b2 inflation ----
    results_pvv: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p_enum, p_name in params:
        for vog in [True, False]:
            label = "Gaussian-Beta" if vog else "Gaussian"
            key = f"{p_name} / {label}"
            results_pvv[key] = _run_1d(p_enum, vog, data, P_vv=P_vv_3x3)
    for vog in [True, False]:
        label = "Gaussian-Beta" if vog else "Gaussian"
        key = f"polar3d / {label}"
        results_pvv[key] = _run_3d(vog, data, P_vv=P_vv_3x3)

    # ---- Panel C: with P_UU (6\u00d76) \u2014 baseline \u03c4\u00b2 + rotational Q ----
    results_puu: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p_enum, p_name in params:
        for vog in [True, False]:
            label = "Gaussian-Beta" if vog else "Gaussian"
            key = f"{p_name} / {label}"
            results_puu[key] = _run_1d(p_enum, vog, data, P_vv=P_UU_6x6)
    for vog in [True, False]:
        label = "Gaussian-Beta" if vog else "Gaussian"
        key = f"polar3d / {label}"
        results_puu[key] = _run_3d(vog, data, P_vv=P_UU_6x6)

    # ---- Print final state ----
    for tag, results in [("No P_vv", results_base),
                         ("P_vv 3\u00d73", results_pvv),
                         ("P_UU 6\u00d76", results_puu)]:
        print(f"\n=== {tag} ===")
        print(f"{'Config':<32s} {'final |err|':>10s} {'final var':>10s} "
              f"{'a/(a+b)':>8s}")
        print("-" * 66)
        for key, (err, var_, ir) in results.items():
            print(f"{key:<32s} {err[-1]:10.4f} {var_[-1]:10.2e} {ir[-1]:8.3f}")

    # ---- Plot: 3 panels \u00d7 3 rows ----
    fig, all_axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True)

    panel_data = [
        ("No P_vv (baseline)", results_base),
        ("P_vv 3\u00d73 (\u03c4\u00b2 inflation)", results_pvv),
        ("P_UU 6\u00d76 (\u03c4\u00b2 + rot. Q)", results_puu),
    ]

    for col, (panel_title, results) in enumerate(panel_data):
        axes = all_axes[:, col]
        _plot_results(axes, results, is_out, colors)

        axes[0].set_ylabel("Depth error |\u1e91 \u2212 z*| (m)")
        axes[0].set_title(
            f"{panel_title}\n"
            f"z*={Z_TRUE} m, \u03c3_px={SIGMA_PIXEL}, "
            f"outlier={outlier_rate:.0%}, N={n}",
            fontsize=10,
        )
        axes[0].legend(fontsize=6, ncol=2, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel("Canonical var \u03c3\u00b2")
        axes[1].legend(fontsize=6, ncol=2, loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel("Inlier ratio a/(a+b)")
        axes[2].set_xlabel("Observation step")
        axes[2].set_ylim(-0.02, 1.05)
        axes[2].legend(fontsize=6, ncol=2, loc="lower right")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
