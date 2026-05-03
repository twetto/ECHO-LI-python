"""NEES (Normalized Estimation Error Squared) consistency test.

A consistent estimator has E[NEES] = dim(state).  For the 1D filters
(EUCLIDEAN, INVDEPTH, POLAR) dim=1 so E[NEES]=1.  For Polar3D dim=3.

We run N_MC Monte Carlo trials with different noise seeds, record
per-step NEES, and plot the time-averaged NEES with 95% chi-squared
confidence bounds.

Usage:
    python tests/test_nees.py
    python tests/test_nees.py --no-outliers
    python tests/test_nees.py --n-mc 200
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from eqvio.sparse_vogiatzis import (
    SparseVogiatzisFilter,
    SparseVogiatzisFilter3D,
    SparseVogSettings,
    DepthParametrization,
)
from eqvio.coordinate_suite.normal import point_chart_normal
from eqvio.mathematical.vision_measurement import VisionMeasurement
from eqvio.mathematical.vio_state import Landmark

# ================================================================
# Constants (same as other convergence tests)
# ================================================================

Z_TRUE = 100.0
FX = FY = 458.0
CX, CY = 376.0, 240.0
SIGMA_PIXEL = 0.1
BASELINE_PER_FRAME = 0.05
DT = 0.05
N_STEPS = 100
U_TRUE, V_TRUE = 400.0, 250.0

K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
P_UU = np.diag([1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01])


# ================================================================
# Data generation (identical to other tests)
# ================================================================

def _gen_data(rng, n_steps, z_true, outlier_rate):
    x_n = (U_TRUE - CX) / FX
    y_n = (V_TRUE - CY) / FY
    P_W = np.array([x_n * z_true, y_n * z_true, z_true])

    data = []
    for i in range(n_steps + 1):
        bl = i * BASELINE_PER_FRAME
        T_WC = np.eye(4)
        T_WC[0, 3] = bl

        T_CW = np.linalg.inv(T_WC)
        P_C = T_CW[:3, :3] @ P_W + T_CW[:3, 3]

        u_true = FX * P_C[0] / P_C[2] + CX
        v_true = FY * P_C[1] / P_C[2] + CY

        is_outlier = False
        if i > 0 and rng.random() < outlier_rate:
            is_outlier = True
            z_out = rng.uniform(0.5, 30.0)
            u_prev = data[-1][1][0]
            x_prev = (u_prev - CX) / FX
            x_curr = x_prev - BASELINE_PER_FRAME / z_out
            u_obs = x_curr * FX + CX
            v_obs = v_true + rng.normal(0, SIGMA_PIXEL)
        else:
            u_obs = u_true + rng.normal(0, SIGMA_PIXEL)
            v_obs = v_true + rng.normal(0, SIGMA_PIXEL)

        data.append((T_WC, np.array([u_obs, v_obs]), P_C.copy(), is_outlier))
    return data


# ================================================================
# True canonical value for each parametrization
# ================================================================

def _true_canonical_1d(param, z_true):
    if param is DepthParametrization.INVDEPTH:
        return 1.0 / z_true
    elif param is DepthParametrization.POLAR:
        return -math.log(z_true)
    else:
        return z_true


# ================================================================
# Single-trial runners returning per-step NEES
# ================================================================

def _run_1d_nees(param, is_vog, data, z_true):
    s = SparseVogSettings(
        parametrization=param,
        a_init=10.0 if is_vog else 1e8,
        b_init=2.0 if is_vog else 1e-8,
        ab_max=100.0 if is_vog else 1e9,
        min_inlier_ratio=0.3,
        min_track_length=1,
        sigma_pixel=SIGMA_PIXEL,
        process_depth_var=0.0,
    )
    filt = SparseVogiatzisFilter(K_MAT, s)

    n = len(data) - 1
    nees = np.full(n, np.nan)

    for i in range(n + 1):
        T_WC, uv, P_C, _ = data[i]
        meas = VisionMeasurement(stamp=i * DT, cam_coordinates={42: uv})
        filt.update(meas, T_WC)

        if i > 0:
            feat = filt.features.get(42)
            if feat and math.isfinite(feat.canonical) and feat.canonical_var > 1e-30:
                z_curr = float(P_C[2])
                c_true = _true_canonical_1d(param, z_curr)
                err = feat.canonical - c_true
                nees[i - 1] = (err * err) / feat.canonical_var

    return nees


def _run_3d_nees(is_vog, data, z_true):
    s = SparseVogSettings(
        a_init=10.0 if is_vog else 1e8,
        b_init=2.0 if is_vog else 1e-8,
        ab_max=100.0 if is_vog else 1e9,
        min_inlier_ratio=0.3,
        min_track_length=1,
        sigma_pixel=SIGMA_PIXEL,
        process_depth_var=0.0,
    )
    filt = SparseVogiatzisFilter3D(K_MAT, s)

    n = len(data) - 1
    nees = np.full(n, np.nan)

    for i in range(n + 1):
        T_WC, uv, P_C, _ = data[i]
        meas = VisionMeasurement(stamp=i * DT, cam_coordinates={42: uv})
        filt.update(meas, T_WC)

        if i > 0:
            feat = filt.features.get(42)
            if feat and feat.depth > 1e-6:
                q_true = P_C
                q_est = feat.position
                eps = point_chart_normal(
                    Landmark(p=q_true, id=42),
                    Landmark(p=q_est, id=42),
                )
                try:
                    P_inv = np.linalg.inv(feat.covariance)
                    nees[i - 1] = float(eps @ P_inv @ eps)
                except np.linalg.LinAlgError:
                    pass

    return nees


# ================================================================
# Monte Carlo
# ================================================================

def _run_mc(param_enum, param_name, is_vog, n_mc, n_steps, z_true,
            outlier_rate, base_seed):
    all_nees = np.full((n_mc, n_steps), np.nan)
    for mc in range(n_mc):
        rng = np.random.default_rng(base_seed + mc)
        data = _gen_data(rng, n_steps, z_true, outlier_rate)
        if param_name == "polar3d":
            all_nees[mc] = _run_3d_nees(is_vog, data, z_true)
        else:
            all_nees[mc] = _run_1d_nees(param_enum, is_vog, data, z_true)
    return all_nees


# ================================================================
# Main
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier-rate", type=float, default=0.0)
    ap.add_argument("--no-outliers", action="store_true")
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--z-true", type=float, default=Z_TRUE)
    ap.add_argument("--save", type=str, default="nees_consistency.png")
    args = ap.parse_args()

    outlier_rate = 0.0 if args.no_outliers else args.outlier_rate
    n = args.n_steps
    n_mc = args.n_mc
    z_true = args.z_true

    print(f"NEES consistency test: z*={z_true:.1f}m  N={n}  "
          f"outlier_rate={outlier_rate:.0%}  N_MC={n_mc}  seed={args.seed}")

    configs = [
        (DepthParametrization.EUCLIDEAN, "euclidean"),
        (DepthParametrization.INVDEPTH, "invdepth"),
        (DepthParametrization.POLAR, "polar"),
        (None, "polar3d"),
    ]
    colors = {
        "euclidean": "tab:blue",
        "invdepth": "tab:orange",
        "polar": "tab:green",
        "polar3d": "tab:red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    steps = np.arange(n)

    for idx, (p_enum, p_name) in enumerate(configs):
        ax = axes[idx]
        dim = 3 if p_name == "polar3d" else 1

        for is_vog, label, ls, lw in [
            (True, "Gaussian-Beta", "-", 1.5),
            (False, "Gaussian", "--", 1.0),
        ]:
            all_nees = _run_mc(p_enum, p_name, is_vog, n_mc, n,
                               z_true, outlier_rate, args.seed)

            valid_count = np.sum(np.isfinite(all_nees), axis=0)
            mean_nees = np.nanmean(all_nees, axis=0)

            ax.plot(steps, mean_nees,
                    color=colors[p_name], ls=ls, lw=lw,
                    alpha=0.85, label=f"{label} (mean)")

            # Print summary
            settled = mean_nees[n // 2:]
            avg = np.nanmean(settled)
            print(f"  {p_name:10s} / {label:14s}: "
                  f"avg NEES (2nd half) = {avg:.3f}  "
                  f"(expected {dim})")

        # Chi-squared 95% bounds for the mean of n_mc independent χ²(dim)
        # Mean of n_mc χ²(dim) ~ Normal(dim, 2*dim/n_mc) for large n_mc
        from scipy.stats import chi2
        lo = chi2.ppf(0.025, dim * n_mc) / n_mc
        hi = chi2.ppf(0.975, dim * n_mc) / n_mc
        ax.axhline(dim, color="k", ls=":", lw=0.8, label=f"E[NEES]={dim}")
        ax.axhspan(lo, hi, color="gray", alpha=0.12, label="95% bounds")

        ax.set_title(f"{p_name}  (dim={dim})", fontsize=11)
        ax.set_ylabel("Mean NEES")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    for ax in axes[2:]:
        ax.set_xlabel("Observation step")

    fig.suptitle(
        f"NEES consistency  |  z*={z_true}m, σ_px={SIGMA_PIXEL}, "
        f"outlier={outlier_rate:.0%}, N_MC={n_mc}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
