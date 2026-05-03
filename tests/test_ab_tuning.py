"""Test effect of ab_max and min_inlier_ratio on GB outlier survival.

Compares old (ab_max=20, min_ir=0.5) vs tuned parameters with
reference-frame triangulation.  Both use P_UU (6×6).

Usage:
    python tests/test_ab_tuning.py
    python tests/test_ab_tuning.py --outlier-rate 0.4
    python tests/test_ab_tuning.py --ab-max 200 --min-ir 0.2
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from eqvio.sparse_vogiatzis import (
    SparseVogiatzisFilter,
    SparseVogiatzisFilter3D,
    SparseVogSettings,
    DepthParametrization,
)
from eqvio.mathematical.vision_measurement import VisionMeasurement

# Re-use constants from the main convergence test.
Z_TRUE = 100.0
FX = FY = 458.0
CX, CY = 376.0, 240.0
SIGMA_PIXEL = 0.1
BASELINE_PER_FRAME = 0.05
DT = 0.05
N_STEPS = 100
SEED = 42
U_TRUE, V_TRUE = 400.0, 250.0

K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
P_UU = np.diag([1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01])


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

        data.append((T_WC, np.array([u_obs, v_obs]), is_outlier))
    return data


def _run_1d(param, is_vog, data, ab_max, min_ir):
    s = SparseVogSettings(
        parametrization=param,
        a_init=10.0 if is_vog else 1e8,
        b_init=2.0 if is_vog else 1e-8,
        ab_max=ab_max if is_vog else 1e9,
        min_inlier_ratio=min_ir,
        min_track_length=1,
        sigma_pixel=SIGMA_PIXEL,
        process_depth_var=0.0,
    )
    filt = SparseVogiatzisFilter(K_MAT, s)
    n = len(data) - 1
    err, var_, ir = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n + 1):
        T_WC, uv, _ = data[i]
        meas = VisionMeasurement(stamp=i * DT, cam_coordinates={42: uv})
        filt.update(meas, T_WC, P_vv=P_UU)
        if i > 0:
            feat = filt.features.get(42)
            if feat:
                err[i - 1] = abs(filt._canonical_to_depth(feat.canonical) - Z_TRUE)
                var_[i - 1] = feat.canonical_var
                ir[i - 1] = feat.inlier_ratio()
            else:
                err[i - 1] = np.nan
                var_[i - 1] = np.nan
                ir[i - 1] = 0.0
    return err, var_, ir


def _run_3d(is_vog, data, ab_max, min_ir):
    s = SparseVogSettings(
        a_init=10.0 if is_vog else 1e8,
        b_init=2.0 if is_vog else 1e-8,
        ab_max=ab_max if is_vog else 1e9,
        min_inlier_ratio=min_ir,
        min_track_length=1,
        sigma_pixel=SIGMA_PIXEL,
        process_depth_var=0.0,
    )
    filt = SparseVogiatzisFilter3D(K_MAT, s)
    n = len(data) - 1
    err, var_, ir = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n + 1):
        T_WC, uv, _ = data[i]
        meas = VisionMeasurement(stamp=i * DT, cam_coordinates={42: uv})
        filt.update(meas, T_WC, P_vv=P_UU)
        if i > 0:
            feat = filt.features.get(42)
            if feat:
                err[i - 1] = abs(feat.depth - Z_TRUE)
                var_[i - 1] = feat.covariance[2, 2]
                ir[i - 1] = feat.inlier_ratio()
            else:
                err[i - 1] = np.nan
                var_[i - 1] = np.nan
                ir[i - 1] = 0.0
    return err, var_, ir


def _run_all(params, data, ab_max, min_ir):
    results = {}
    for p_enum, p_name in params:
        for vog in [True, False]:
            label = "Gaussian-Beta" if vog else "Gaussian"
            results[f"{p_name} / {label}"] = _run_1d(
                p_enum, vog, data, ab_max, min_ir)
    for vog in [True, False]:
        label = "Gaussian-Beta" if vog else "Gaussian"
        results[f"polar3d / {label}"] = _run_3d(vog, data, ab_max, min_ir)
    return results


def _plot_results(axes, results, is_out, colors):
    n = len(is_out)
    steps = np.arange(n)
    for key, (err, var_, ir) in results.items():
        p = key.split(" / ")[0]
        cfg = key.split(" / ")[1]
        c = colors[p]
        ls = "-" if "Gaussian-Beta" in cfg else "--"
        lw = 1.5 if "Gaussian-Beta" in cfg else 1.0
        axes[0].semilogy(steps, np.maximum(err, 1e-6),
                         color=c, ls=ls, lw=lw, alpha=0.85, label=key)
        axes[1].semilogy(steps, np.maximum(var_, 1e-12),
                         color=c, ls=ls, lw=lw, alpha=0.85, label=key)
        axes[2].plot(steps, ir, color=c, ls=ls, lw=lw, alpha=0.85, label=key)
    out_idx = np.where(is_out)[0]
    for ax in axes:
        for oi in out_idx:
            ax.axvline(oi, color="red", alpha=0.08, lw=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlier-rate", type=float, default=0.4)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ab-max", type=float, default=100.0)
    ap.add_argument("--min-ir", type=float, default=0.3)
    ap.add_argument("--save", type=str, default="convergence_ab_tuning.png")
    args = ap.parse_args()

    n = args.n_steps
    rng = np.random.default_rng(args.seed)
    data = _gen_data(rng, n, Z_TRUE, args.outlier_rate)
    is_out = np.array([d[2] for d in data[1:]])
    n_outliers = int(is_out.sum())
    print(f"z*={Z_TRUE:.1f}m  N={n}  outlier_rate={args.outlier_rate:.0%}  "
          f"({n_outliers} outliers)  seed={args.seed}")

    params = [
        (DepthParametrization.EUCLIDEAN, "euclidean"),
        (DepthParametrization.INVDEPTH, "invdepth"),
        (DepthParametrization.POLAR, "polar"),
    ]
    colors = {
        "euclidean": "tab:blue",
        "invdepth": "tab:orange",
        "polar": "tab:green",
        "polar3d": "tab:red",
    }

    results_old = _run_all(params, data, ab_max=20.0, min_ir=0.5)
    results_new = _run_all(params, data, ab_max=args.ab_max,
                           min_ir=args.min_ir)

    for tag, results in [
        ("Old (ab_max=20, min_ir=0.5)", results_old),
        (f"Tuned (ab_max={args.ab_max}, min_ir={args.min_ir})", results_new),
    ]:
        print(f"\n=== {tag} ===")
        print(f"{'Config':<32s} {'final |err|':>10s} {'final var':>10s} "
              f"{'a/(a+b)':>8s}")
        print("-" * 66)
        for key, (err, var_, ir) in results.items():
            print(f"{key:<32s} {err[-1]:10.4f} {var_[-1]:10.2e} {ir[-1]:8.3f}")

    fig, all_axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    panel_data = [
        ("Old (ab_max=20, min_ir=0.5)", results_old),
        (f"Tuned (ab_max={args.ab_max}, min_ir={args.min_ir})", results_new),
    ]
    for col, (title, results) in enumerate(panel_data):
        axes = all_axes[:, col]
        _plot_results(axes, results, is_out, colors)
        axes[0].set_ylabel("Depth error |ẑ − z*| (m)")
        axes[0].set_title(
            f"{title}\nz*={Z_TRUE} m, σ_px={SIGMA_PIXEL}, "
            f"outlier={args.outlier_rate:.0%}, N={n}", fontsize=10)
        axes[0].legend(fontsize=6, ncol=2, loc="upper right")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("Canonical var σ²")
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
