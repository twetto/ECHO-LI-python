"""NEES diagnostic with KLT-inspired heavy-tailed image residuals.

This script complements tests/test_nees.py.

tests/test_nees.py keeps the classical Gaussian + uniform-outlier stress
test.  This script models the residual shapes seen when sparse KLT tracks are
compared against MidAir geometry: a sub-pixel Gaussian core, rare heavy tails,
and optional along-epipolar drift bursts.

Usage:
    python tests/test_klt_noise_nees.py
    python tests/test_klt_noise_nees.py --noise-model student-t
    python tests/test_klt_noise_nees.py --noise-model epipolar-drift
    python tests/test_klt_noise_nees.py --noise-model burst-epipolar
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from eqvio.coordinate_suite.normal import point_chart_normal
from eqvio.coordinate_suite.invdepth import conv_ind2euc, point_chart_invdepth
from eqvio.mathematical.vision_measurement import VisionMeasurement
from eqvio.mathematical.vio_state import Landmark
from eqvio.sparse_vogiatzis import (
    DepthParametrization,
    SparseVogiatzisFilter,
    SparseVogiatzisFilter3D,
    SparseVogiatzisFilterInvDepth3D,
    SparseVogSettings,
)


Z_TRUE = 100.0
FX = FY = 458.0
CX, CY = 376.0, 240.0
BASELINE_PER_FRAME = 0.05
DT = 0.05
N_STEPS = 1500
U_TRUE, V_TRUE = 400.0, 250.0

K_MAT = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])


def _true_canonical_1d(param, z_true):
    if param is DepthParametrization.INVDEPTH:
        return 1.0 / z_true
    if param is DepthParametrization.POLAR:
        return -math.log(z_true)
    return z_true


def _sample_image_noise(
    rng: np.random.Generator,
    model: str,
    sigma_core: float,
    tail_prob: float,
    tail_scale: float,
    student_df: float,
    burst_remaining: int,
) -> tuple[np.ndarray, int, bool]:
    """Sample image-plane residual in pixels.

    Returns:
        noise: 2-vector pixel residual.
        burst_remaining: updated burst counter.
        tail_event: whether this sample used the tail/burst component.
    """
    noise = rng.normal(0.0, sigma_core, size=2)
    tail_event = False

    if model == "gaussian":
        return noise, burst_remaining, tail_event

    if model == "student-t":
        if rng.random() < tail_prob:
            noise += rng.standard_t(student_df, size=2) * tail_scale
            tail_event = True
        return noise, burst_remaining, tail_event

    if model == "epipolar-drift":
        if rng.random() < tail_prob:
            noise[0] += rng.standard_t(student_df) * tail_scale
            tail_event = True
        return noise, burst_remaining, tail_event

    if model == "burst-epipolar":
        if burst_remaining > 0:
            noise[0] += rng.standard_t(student_df) * tail_scale
            return noise, burst_remaining - 1, True
        if rng.random() < tail_prob:
            noise[0] += rng.standard_t(student_df) * tail_scale
            return noise, burst_remaining, True
        return noise, burst_remaining, tail_event

    raise ValueError(f"unknown noise model: {model}")


def _gen_data(
    rng: np.random.Generator,
    n_steps: int,
    z_true: float,
    noise_model: str,
    sigma_core: float,
    tail_prob: float,
    tail_scale: float,
    student_df: float,
    burst_prob: float,
    burst_len: int,
):
    x_n = (U_TRUE - CX) / FX
    y_n = (V_TRUE - CY) / FY
    P_W = np.array([x_n * z_true, y_n * z_true, z_true])

    data = []
    burst_remaining = 0
    for i in range(n_steps + 1):
        bl = i * BASELINE_PER_FRAME
        T_WC = np.eye(4)
        T_WC[0, 3] = bl

        T_CW = np.linalg.inv(T_WC)
        P_C = T_CW[:3, :3] @ P_W + T_CW[:3, 3]

        u_true = FX * P_C[0] / P_C[2] + CX
        v_true = FY * P_C[1] / P_C[2] + CY

        if noise_model == "burst-epipolar" and burst_remaining == 0 and i > 0:
            if rng.random() < burst_prob:
                burst_remaining = max(1, burst_len)

        noise, burst_remaining, tail_event = _sample_image_noise(
            rng,
            noise_model,
            sigma_core,
            tail_prob,
            tail_scale,
            student_df,
            burst_remaining,
        )
        uv = np.array([u_true, v_true]) + noise
        data.append((T_WC, uv, P_C.copy(), tail_event))

    return data


def _run_1d_nees(param, is_vog, data, sigma_pixel, init_depth_var):
    s = SparseVogSettings(
        parametrization=param,
        a_init=10.0 if is_vog else 1e8,
        b_init=2.0 if is_vog else 1e-8,
        ab_max=100.0 if is_vog else 1e9,
        min_inlier_ratio=0.3,
        min_track_length=1,
        sigma_pixel=sigma_pixel,
        process_depth_var=0.0,
        init_depth_var=init_depth_var,
    )
    filt = SparseVogiatzisFilter(K_MAT, s)

    n = len(data) - 1
    nees = np.full(n, np.nan)
    rel_err = np.full(n, np.nan)
    failed = np.zeros(n, dtype=np.float64)

    for i, (T_WC, uv, P_C, _) in enumerate(data):
        meas = VisionMeasurement(stamp=i * DT, cam_coordinates={42: uv})
        filt.update(meas, T_WC)

        if i == 0:
            continue

        feat = filt.features.get(42)
        j = i - 1
        if feat and math.isfinite(feat.canonical) and feat.canonical_var > 1e-30:
            z_curr = float(P_C[2])
            c_true = _true_canonical_1d(param, z_curr)
            err = feat.canonical - c_true
            nees[j] = (err * err) / feat.canonical_var
            z_est, _ = filt.query(42)
            if z_est > 0 and z_curr > 0:
                rel_err[j] = abs(z_est - z_curr) / z_curr
                failed[j] = float(rel_err[j] > 0.2)
            else:
                failed[j] = 1.0
        else:
            failed[j] = 1.0

    return nees, rel_err, failed


def _run_3d_nees(filter_cls, is_vog, data, sigma_pixel, init_depth_var):
    s = SparseVogSettings(
        a_init=10.0 if is_vog else 1e8,
        b_init=2.0 if is_vog else 1e-8,
        ab_max=100.0 if is_vog else 1e9,
        min_inlier_ratio=0.3,
        min_track_length=1,
        sigma_pixel=sigma_pixel,
        process_depth_var=0.0,
        init_depth_var=init_depth_var,
    )
    filt = filter_cls(K_MAT, s)

    n = len(data) - 1
    nees = np.full(n, np.nan)
    rel_err = np.full(n, np.nan)
    failed = np.zeros(n, dtype=np.float64)

    for i, (T_WC, uv, P_C, _) in enumerate(data):
        meas = VisionMeasurement(stamp=i * DT, cam_coordinates={42: uv})
        filt.update(meas, T_WC)

        if i == 0:
            continue

        feat = filt.features.get(42)
        j = i - 1
        if feat and feat.depth > 1e-6:
            if filter_cls is SparseVogiatzisFilterInvDepth3D:
                eps, cov_nees = _camera_invdepth3d_error_and_cov(feat, P_C)
            else:
                eps = point_chart_normal(
                    Landmark(p=P_C, id=42),
                    Landmark(p=feat.position, id=42),
                )
                cov_nees = feat.covariance
            try:
                P_inv = np.linalg.inv(cov_nees)
                nees[j] = float(eps @ P_inv @ eps)
            except np.linalg.LinAlgError:
                pass
            d_true = float(P_C[2]) if filter_cls is SparseVogiatzisFilterInvDepth3D else float(np.linalg.norm(P_C))
            rel_err[j] = abs(feat.depth - d_true) / d_true
            failed[j] = float(rel_err[j] > 0.2)
        else:
            failed[j] = 1.0

    return nees, rel_err, failed


def _camera_invdepth3d_error_and_cov(feat, q_true):
    q_est = feat.position
    eps = point_chart_invdepth(
        Landmark(p=q_true, id=feat.feat_id),
        Landmark(p=q_est, id=feat.feat_id),
    )
    eps[2] = 1.0 / float(q_true[2]) - 1.0 / float(q_est[2])

    M_ind2euc = conv_ind2euc(q_est)
    H_zinv = np.array([[0.0, 0.0, -1.0 / (q_est[2] * q_est[2])]])
    J = np.eye(3)
    J[2, :] = (H_zinv @ M_ind2euc).reshape(3)
    cov = J @ feat.covariance @ J.T
    return eps, cov


def _run_mc(
    param_enum,
    param_name,
    is_vog,
    args,
):
    all_nees = np.full((args.n_mc, args.n_steps), np.nan)
    all_rel_err = np.full((args.n_mc, args.n_steps), np.nan)
    all_failed = np.zeros((args.n_mc, args.n_steps), dtype=np.float64)

    for mc in range(args.n_mc):
        rng = np.random.default_rng(args.seed + mc)
        data = _gen_data(
            rng,
            args.n_steps,
            args.z_true,
            args.noise_model,
            args.sigma_core,
            args.tail_prob,
            args.tail_scale,
            args.student_df,
            args.burst_prob,
            args.burst_len,
        )
        if param_name == "polar3d":
            all_nees[mc], all_rel_err[mc], all_failed[mc] = _run_3d_nees(
                SparseVogiatzisFilter3D, is_vog, data, args.sigma_pixel,
                args.init_depth_var,
            )
        elif param_name == "invdepth3d":
            all_nees[mc], all_rel_err[mc], all_failed[mc] = _run_3d_nees(
                SparseVogiatzisFilterInvDepth3D, is_vog, data,
                args.sigma_pixel, args.init_depth_var,
            )
        else:
            all_nees[mc], all_rel_err[mc], all_failed[mc] = _run_1d_nees(
                param_enum, is_vog, data, args.sigma_pixel, args.init_depth_var,
            )

    return all_nees, all_rel_err, all_failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--noise-model",
        choices=["gaussian", "student-t", "epipolar-drift", "burst-epipolar"],
        default="epipolar-drift",
    )
    ap.add_argument("--sigma-core", type=float, default=0.35)
    ap.add_argument("--sigma-pixel", type=float, default=0.35)
    ap.add_argument("--tail-prob", type=float, default=0.03)
    ap.add_argument("--tail-scale", type=float, default=3.0)
    ap.add_argument("--student-df", type=float, default=2.5)
    ap.add_argument("--burst-prob", type=float, default=0.01)
    ap.add_argument("--burst-len", type=int, default=5)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--n-mc", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--z-true", type=float, default=Z_TRUE)
    ap.add_argument("--init-depth-var", type=float, default=SparseVogSettings.init_depth_var)
    ap.add_argument("--save", type=str, default="klt_noise_nees.png")
    args = ap.parse_args()

    print(
        "KLT-noise NEES diagnostic: "
        f"model={args.noise_model} z*={args.z_true:.1f}m "
        f"N={args.n_steps} N_MC={args.n_mc} "
        f"sigma_core={args.sigma_core:.2f}px tail_prob={args.tail_prob:.1%}"
    )

    configs = [
        (DepthParametrization.EUCLIDEAN, "euclidean"),
        (DepthParametrization.INVDEPTH, "invdepth"),
        (DepthParametrization.POLAR, "polar"),
        (None, "polar3d"),
        (None, "invdepth3d"),
    ]
    colors = {
        "euclidean": "tab:blue",
        "invdepth": "tab:orange",
        "polar": "tab:green",
        "polar3d": "tab:red",
        "invdepth3d": "tab:purple",
    }

    fig, axes = plt.subplots(3, 5, figsize=(22, 12), sharex=True)
    steps = np.arange(args.n_steps)

    for idx, (p_enum, p_name) in enumerate(configs):
        ax_nees = axes[0, idx]
        ax_err = axes[1, idx]
        ax_fail = axes[2, idx]
        dim = 3 if p_name in ("polar3d", "invdepth3d") else 1

        for is_vog, label, ls, lw in [
            (True, "Gaussian-Beta", "-", 1.5),
            (False, "Gaussian", "--", 1.0),
        ]:
            all_nees, all_rel_err, all_failed = _run_mc(
                p_enum, p_name, is_vog, args,
            )
            median_nees = np.nanmedian(all_nees, axis=0)
            q25_nees = np.nanpercentile(all_nees, 25, axis=0)
            q75_nees = np.nanpercentile(all_nees, 75, axis=0)
            median_rel_err = np.nanmedian(all_rel_err, axis=0)
            fail_rate = np.mean(all_failed, axis=0)

            ax_nees.plot(
                steps, median_nees,
                color=colors[p_name], ls=ls, lw=lw, label=label,
            )
            ax_nees.fill_between(
                steps, q25_nees, q75_nees,
                color=colors[p_name], alpha=0.12,
            )
            ax_err.plot(
                steps, median_rel_err * 100.0,
                color=colors[p_name], ls=ls, lw=lw, label=label,
            )
            ax_fail.plot(
                steps, fail_rate * 100.0,
                color=colors[p_name], ls=ls, lw=lw, label=label,
            )

            half = args.n_steps // 2
            print(
                f"  {p_name:10s} / {label:14s}: "
                f"median NEES = {np.nanmedian(median_nees[half:]):.3f} "
                f"(expected median {chi2.median(dim):.3f})  "
                f"median rel_err = {np.nanmedian(median_rel_err[half:]):.4f}  "
                f"fail = {np.mean(fail_rate[half:]):.1%}"
            )

        ax_nees.axhline(
            chi2.median(dim), color="k", ls=":", lw=0.8,
            label=f"med[chi2({dim})]",
        )
        ax_nees.set_title(f"{p_name} (dim={dim})")
        ax_nees.set_ylabel("Median NEES")
        ax_nees.grid(True, alpha=0.3)
        ax_nees.set_ylim(bottom=0)
        ax_nees.legend(fontsize=7, loc="upper right")

        ax_err.set_ylabel("Median depth err (%)")
        ax_err.grid(True, alpha=0.3)
        ax_err.set_ylim(bottom=0)
        ax_err.legend(fontsize=7, loc="upper right")

        ax_fail.set_ylabel("Failure rate (%)")
        ax_fail.set_xlabel("Observation step")
        ax_fail.grid(True, alpha=0.3)
        ax_fail.set_ylim(bottom=0)
        ax_fail.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"KLT-inspired NEES | {args.noise_model}, "
        f"sigma_core={args.sigma_core:.2f}px, tail_prob={args.tail_prob:.1%}, "
        f"N_MC={args.n_mc}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
