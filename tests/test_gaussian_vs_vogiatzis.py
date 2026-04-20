"""Convergence rate: Gaussian (Kalman) vs Gaussian-Beta (Vogiatzis) × 4 parametrizations.

Simulates a static feature at known depth with noisy triangulated
observations (20 % outliers).  Compares convergence of:

    Filter type          | How obtained from Vogiatzis code
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
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Shared constants
# ================================================================

Z_TRUE = 5.0
FX = FY = 458.0
CX, CY = 376.0, 240.0
SIGMA_PIXEL = 0.5
SIGMA_NORM_SQ = (SIGMA_PIXEL / FX) ** 2
BASELINE_PER_FRAME = 0.05        # camera moves 5 cm per frame
N_STEPS = 100
SEED = 42

# True pixel of the feature (arbitrary, inside 752×480)
U_TRUE = 400.0
V_TRUE = 250.0

# ================================================================
# Filter configs
# ================================================================

@dataclass
class FilterCfg:
    label: str
    a_init: float
    b_init: float
    ab_min: float
    ab_max: float


VOG = FilterCfg("Gaussian-Beta", a_init=10.0, b_init=2.0,
                ab_min=1.0, ab_max=20.0)
KAL = FilterCfg("Gaussian",      a_init=1e6,  b_init=1e-6,
                ab_min=1e-8, ab_max=1e8)

PARAMS_1D = ["euclidean", "invdepth", "polar"]
ALL_PARAMS = PARAMS_1D + ["polar3d"]

# ================================================================
# Parametrization helpers (1D)
# ================================================================

def _euc2can(z: float, p: str) -> float:
    if p == "invdepth":
        return 1.0 / z
    if p == "polar":
        return math.log(z)
    return z


def _can2euc(c: float, p: str) -> float:
    if p == "invdepth":
        return 1.0 / c
    if p == "polar":
        return math.exp(c)
    return c


def _tau_sq(z_obs: float, drive: float, p: str) -> float:
    base = SIGMA_NORM_SQ / (drive * drive)
    if p == "invdepth":
        return base
    if p == "polar":
        return z_obs * z_obs * base
    return (z_obs ** 4) * base


def _init_canonical_var(p: str) -> float:
    """Initial variance in canonical coordinates (loose prior)."""
    if p == "invdepth":
        return 1.0            # var(ρ)
    if p == "polar":
        return 1.0            # var(log z)
    return 10.0               # var(z)


def _uniform_range(p: str) -> float:
    if p == "invdepth":
        return 10.0
    if p == "polar":
        return 10.0            # d ∈ [-5, 5]
    return 20.0                # z ∈ [0, 20]


# ================================================================
# 1D Vogiatzis update (scalar, mirrors sparse_vogiatzis.py)
# ================================================================

def _vogiatzis_1d(mu, sig2, a, b, obs, tau2, U_range, cfg: FilterCfg):
    s_total = sig2 + tau2
    diff = obs - mu
    maha = diff * diff / s_total

    # Kalman branch
    m = (mu * tau2 + obs * sig2) / s_total
    s2 = sig2 * tau2 / s_total

    exp_arg = -0.5 * maha
    gpdf = 0.0 if exp_arg < -50 else math.exp(exp_arg) / math.sqrt(2 * math.pi * s_total)

    U = 1.0 / U_range
    ab = a + b
    C1 = (a / ab) * gpdf
    C2 = (b / ab) * U
    Z = C1 + C2
    if Z < 1e-30:
        return mu, sig2, a, min(b + 1, cfg.ab_max)

    w1 = C1 / Z
    w2 = C2 / Z

    new_mu = w1 * m + w2 * mu
    Ex2 = w1 * (s2 + m * m) + w2 * (sig2 + mu * mu)
    new_sig2 = max(Ex2 - new_mu * new_mu, 1e-12)

    # Beta moment matching
    d1 = ab + 1
    d2 = d1 * (ab + 2)
    Epi = (w1 * (a + 1) + w2 * a) / d1
    Epi2 = (w1 * (a + 1) * (a + 2) + w2 * a * (a + 1)) / d2
    vpi = Epi2 - Epi * Epi

    if vpi < 1e-6 or Epi <= 1e-6 or Epi >= 1 - 1e-6:
        na, nb = a + w1, b + w2
    else:
        fac = max(Epi * (1 - Epi) / vpi - 1, 0.5)
        na, nb = Epi * fac, (1 - Epi) * fac

    na = float(np.clip(na, cfg.ab_min, cfg.ab_max))
    nb = float(np.clip(nb, cfg.ab_min, cfg.ab_max))
    return float(new_mu), float(new_sig2), na, nb


# ================================================================
# Generate observations (shared across all filters)
# ================================================================

def _gen_obs(rng, n: int, z_true: float, outlier_rate: float):
    """Return (z_obs[n], drive[n], is_outlier[n]).

    Inlier noise ~ N(0, τ_euc) where τ_euc = z²·σ_norm/drive.
    Drive increases linearly with frame index (growing baseline).
    Outliers are uniform in [0.5, 30].
    """
    z_obs = np.empty(n)
    drives = np.empty(n)
    is_out = np.zeros(n, dtype=bool)

    for i in range(n):
        # Drive grows with baseline: first few frames are noisy,
        # later frames are informative.  Mimics real tracking.
        bl = (i + 1) * BASELINE_PER_FRAME
        x_norm = (U_TRUE - CX) / FX
        y_norm = (V_TRUE - CY) / FY
        # Ideal parallax ≈ baseline / depth (normalised image coords)
        drive = bl / z_true
        drives[i] = max(drive, 1e-6)

        # Observation noise std in depth
        tau_euc = (z_true ** 2) * math.sqrt(SIGMA_NORM_SQ) / drives[i]

        if rng.random() < outlier_rate:
            z_obs[i] = rng.uniform(0.5, 30.0)
            is_out[i] = True
        else:
            z_obs[i] = max(rng.normal(z_true, tau_euc), 0.05)

    return z_obs, drives, is_out


# ================================================================
# Run 1D filter
# ================================================================

def _run_1d(param: str, cfg: FilterCfg, z_obs, drives):
    n = len(z_obs)
    U_range = _uniform_range(param)

    # Initialise from a vague prior centred on first observation
    mu = _euc2can(z_obs[0], param)
    sig2 = _init_canonical_var(param)
    a, b = cfg.a_init, cfg.b_init

    err = np.empty(n)
    var_ = np.empty(n)
    ir = np.empty(n)

    for i in range(n):
        z = z_obs[i]
        if z <= 0:
            err[i] = err[max(i - 1, 0)]
            var_[i] = var_[max(i - 1, 0)]
            ir[i] = ir[max(i - 1, 0)]
            continue

        obs = _euc2can(z, param)
        tau2 = _tau_sq(z, drives[i], param)

        mu, sig2, a, b = _vogiatzis_1d(mu, sig2, a, b, obs, tau2, U_range, cfg)

        z_est = _can2euc(mu, param)
        err[i] = abs(z_est - Z_TRUE)
        var_[i] = sig2
        ir[i] = a / (a + b) if (a + b) > 0 else 0

    return err, var_, ir


# ================================================================
# 3D Normal-chart helpers (standalone, no liepp dependency)
# ================================================================

def _ortho_basis(n):
    v = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, v);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1); e2 /= np.linalg.norm(e2)
    return e1, e2


def _J_norm2euc(q):
    r = np.linalg.norm(q)
    if r < 1e-10:
        return np.eye(3)
    n = q / r
    e1, e2 = _ortho_basis(n)
    return np.column_stack([r * e1, r * e2, -q])


def _J_euc2norm(q):
    return np.linalg.inv(_J_norm2euc(q))


def _retract(eps, q):
    r = np.linalg.norm(q)
    if r < 1e-10:
        return q
    n = q / r
    e1, e2 = _ortho_basis(n)
    ang = math.sqrt(eps[0] ** 2 + eps[1] ** 2)
    if ang > 1e-8:
        ax = (eps[0] * e1 + eps[1] * e2)
        ax /= np.linalg.norm(ax)
        c, s = math.cos(ang), math.sin(ang)
        n_new = n * c + np.cross(ax, n) * s + ax * np.dot(ax, n) * (1 - c)
    else:
        n_new = n + eps[0] * e1 + eps[1] * e2
        n_new /= np.linalg.norm(n_new)
    return n_new * r * math.exp(-eps[2])


# ================================================================
# Run 3D filter
# ================================================================

def _run_3d(cfg: FilterCfg, z_obs, drives):
    n = len(z_obs)
    x_n = (U_TRUE - CX) / FX
    y_n = (V_TRUE - CY) / FY

    q = np.array([x_n * z_obs[0], y_n * z_obs[0], z_obs[0]])
    P = np.eye(3) * 1.0
    a, b = cfg.a_init, cfg.b_init
    U_d_range = 10.0
    sp2 = SIGMA_PIXEL ** 2

    err = np.empty(n)
    var_ = np.empty(n)
    ir = np.empty(n)

    for i in range(n):
        z = z_obs[i]
        if z <= 0.05 or np.linalg.norm(q) < 1e-4 or q[2] < 1e-4:
            err[i] = err[max(i - 1, 0)]
            var_[i] = var_[max(i - 1, 0)]
            ir[i] = ir[max(i - 1, 0)]
            continue

        # -- Step 1: bearing update (standard Kalman) --
        y_pred = np.array([FX * q[0] / q[2] + CX, FY * q[1] / q[2] + CY])
        y_obs = np.array([U_TRUE, V_TRUE])

        H_euc = np.zeros((2, 3))
        H_euc[0, 0] = FX / q[2]
        H_euc[0, 2] = -FX * q[0] / q[2] ** 2
        H_euc[1, 1] = FY / q[2]
        H_euc[1, 2] = -FY * q[1] / q[2] ** 2

        M = _J_norm2euc(q)
        Hb = H_euc @ M
        Rb = sp2 * np.eye(2)
        S = Hb @ P @ Hb.T + Rb
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            continue
        K = P @ Hb.T @ Si
        eps_b = K @ (y_obs - y_pred)
        q_mid = _retract(eps_b, q)
        IKH = np.eye(3) - K @ Hb
        P_mid = IKH @ P @ IKH.T + K @ Rb @ K.T
        if np.any(np.linalg.eigvalsh(P_mid) <= 0):
            P_mid, q_mid = P, q

        # -- Step 2: depth Vogiatzis --
        z_mid = np.linalg.norm(q_mid)
        if z_mid < 0.01:
            continue
        d_obs = math.log(z)
        d_pred = math.log(z_mid)
        inn = d_obs - d_pred

        tau_d2 = (z ** 2) * SIGMA_NORM_SQ / (drives[i] ** 2)
        S_d = P_mid[2, 2] + tau_d2
        if S_d < 1e-30:
            q, P = q_mid, P_mid
            continue

        maha = inn * inn / S_d
        exp_arg = -0.5 * maha
        gpdf = 0.0 if exp_arg < -50 else math.exp(exp_arg) / math.sqrt(2 * math.pi * S_d)

        U_pr = 1.0 / U_d_range
        ab = a + b
        C1 = (a / ab) * gpdf
        C2 = (b / ab) * U_pr
        Zn = C1 + C2
        if Zn < 1e-30:
            b = min(b + 1, cfg.ab_max)
            q, P = q_mid, P_mid
        else:
            w1, w2 = C1 / Zn, C2 / Zn
            K_d = -P_mid[:, 2] / S_d
            eps_d = (w1 * inn) * K_d
            q_new = _retract(eps_d, q_mid)

            H_d = np.array([[0.0, 0.0, -1.0]])
            KH_d = np.outer(K_d, H_d)
            IKH_d = np.eye(3) - KH_d
            P_k = IKH_d @ P_mid @ IKH_d.T + tau_d2 * np.outer(K_d, K_d)
            full_e = inn * K_d
            P_new = w1 * P_k + w2 * P_mid + w1 * w2 * np.outer(full_e, full_e)

            if np.any(np.linalg.eigvalsh(P_new) <= 0):
                P_new, q_new = P_mid, q_mid

            q, P = q_new, P_new

            # Beta moment matching
            d1 = ab + 1
            d2 = d1 * (ab + 2)
            Epi = (w1 * (a + 1) + w2 * a) / d1
            Epi2 = (w1 * (a + 1) * (a + 2) + w2 * a * (a + 1)) / d2
            vpi = Epi2 - Epi * Epi
            if vpi < 1e-6 or Epi <= 1e-6 or Epi >= 1 - 1e-6:
                na, nb = a + w1, b + w2
            else:
                fac = max(Epi * (1 - Epi) / vpi - 1, 0.5)
                na, nb = Epi * fac, (1 - Epi) * fac
            a = float(np.clip(na, cfg.ab_min, cfg.ab_max))
            b = float(np.clip(nb, cfg.ab_min, cfg.ab_max))

        depth_est = np.linalg.norm(q)
        err[i] = abs(depth_est - Z_TRUE)
        var_[i] = P[2, 2]   # log-depth variance
        ir[i] = a / (a + b) if (a + b) > 0 else 0

    return err, var_, ir


# ================================================================
# Main
# ================================================================

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
    z_obs, drives, is_out = _gen_obs(rng, n, Z_TRUE, outlier_rate)

    n_outliers = int(is_out.sum())
    print(f"z*={Z_TRUE:.1f}m  N={n}  outlier_rate={outlier_rate:.0%}  "
          f"({n_outliers} outliers)  seed={args.seed}")

    # Run 8 configs
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p in PARAMS_1D:
        for cfg in (VOG, KAL):
            key = f"{p} / {cfg.label}"
            results[key] = _run_1d(p, cfg, z_obs, drives)
    for cfg in (VOG, KAL):
        key = f"polar3d / {cfg.label}"
        results[key] = _run_3d(cfg, z_obs, drives)

    # ---- Print final state ----
    print(f"\n{'Config':<32s} {'final |err|':>10s} {'final var':>10s} "
          f"{'a/(a+b)':>8s}")
    print("-" * 66)
    for key, (err, var_, ir) in results.items():
        print(f"{key:<32s} {err[-1]:10.4f} {var_[-1]:10.2e} {ir[-1]:8.3f}")

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    colors = {
        "euclidean": "tab:blue",
        "invdepth":  "tab:orange",
        "polar":     "tab:green",
        "polar3d":   "tab:red",
    }
    steps = np.arange(n)

    for key, (err, var_, ir) in results.items():
        p = key.split(" / ")[0]
        cfg_label = key.split(" / ")[1]
        c = colors[p]
        ls = "-" if cfg_label == "Gaussian-Beta" else "--"
        lw = 1.5 if cfg_label == "Gaussian-Beta" else 1.0

        axes[0].semilogy(steps, np.maximum(err, 1e-6),
                         color=c, ls=ls, lw=lw, alpha=0.85, label=key)
        axes[1].semilogy(steps, np.maximum(var_, 1e-12),
                         color=c, ls=ls, lw=lw, alpha=0.85, label=key)
        axes[2].plot(steps, ir,
                     color=c, ls=ls, lw=lw, alpha=0.85, label=key)

    # Shade outlier steps
    out_idx = np.where(is_out)[0]
    for ax in axes:
        for oi in out_idx:
            ax.axvline(oi, color="red", alpha=0.08, lw=0.8)

    axes[0].set_ylabel("Depth error  |z\u0302 \u2212 z*|  (m)")
    axes[0].set_title(
        f"Gaussian vs Gaussian-Beta  \u00d7  4 parametrizations\n"
        f"z*={Z_TRUE} m,  \u03c3_px={SIGMA_PIXEL},  "
        f"outlier rate={outlier_rate:.0%},  N={n}",
        fontsize=11,
    )
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Canonical variance  \u03c3\u00b2")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Inlier ratio  a/(a+b)")
    axes[2].set_xlabel("Observation step")
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].legend(fontsize=7, ncol=2, loc="lower right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"\nSaved {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
