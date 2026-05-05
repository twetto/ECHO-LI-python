# 2D State Augmentation for Colored Re-Anchor Noise

## Why the scalar Vogiatzis filter is underconfident

The filter assumes white observation noise:
`Cov(n_t, n_s) = 0` for `t ≠ s`.

After commit 5eac92c we know this is wrong by construction. Every
triangulated depth observation reuses one of its two pixels in the next
triangulation — the *current* pixel of update `t` becomes the *reference*
pixel of update `t+1`. Consecutive observations therefore share a noise
component, violating whiteness with a one-step Markov structure.

## The noise model

Let `η_t` denote the pixel noise on the LK-tracked feature at frame `t`
(zero-mean, variance `σ_px²`). After derotation in normalised image
coordinates, the inverse-depth observation at update `t` is approximately

```
ρ̂_t = ρ_true + (η_{t-1} − η_t) / (f · B_t)
```

where `B_t` is the parallax baseline since the last anchor and `f` is the
focal length. Defining

```
n_t := (η_{t-1} − η_t) / (f · B_t)
```

gives

```
Var(n_t)             =  σ_px² (1/B_t² + 1/B_t²) / f²       (per obs)
Cov(n_t, n_{t+1})    = −σ_px² / (f² · B_t · B_{t+1})       (carry-over)
Cov(n_t, n_{t+k})    =  0                  for  k ≥ 2
```

So the noise sequence is **first-order Markov, not white**. The carry-over
sign is *negative* — that turns out to matter a lot.

## Why this causes underconfidence (NEES → 0, not ∞)

Sum the observations over `n` updates with constant baseline `B`:

```
Σ_{t=1}^{n} n_t  =  (η_0 − η_n) / (f · B)
```

All intermediate `η_t` cancel — telescoping. The mean error therefore
shrinks as

```
E[(mean error)²]  =  2 σ_px² / (n · f · B)²   ~  1/n²
```

A scalar Kalman filter, treating each `n_t` as independent with variance
`σ²_indep = 2σ_px²/(f²B²)`, instead estimates posterior variance shrinking
as

```
σ²_filter  ~  σ²_indep / n   ~  1/n
```

NEES is the ratio of squared error to filter variance:

```
NEES  =  E[(mean error)²] / σ²_filter   ~   (1/n²) / (1/n)   =  1/n   →  0
```

The filter is **underconfident**: actual error is much smaller than its
own covariance suggests, because the correlated noise *cancels* across
steps. Counterintuitively, breaking white noise here shrinks error
faster than independence — but only the augmented filter knows it.

## Fix: state augmentation

Standard Kalman filtering handles colored measurement noise by promoting
the noise process to part of the state. Augment the scalar depth state
`x_t` (canonical chart) with a second component `η_t^ref` that tracks the
*reference* pixel noise contribution in ρ-space:

```
state:    s_t = [ x_t , η_t^ref ]^T

dynamics: s_{t+1} = F s_t + w_t
          F = [ J  0 ]                            (J = chart Jacobian for prediction)
              [ 0  0 ]                            (η_t^ref is reset by re-anchor)

obs:      ρ̂_t = h(x_t) + η_t^ref + ν_t           (ν_t = current-pixel noise)
          H_t = [ ∂ρ/∂x | x_t ,  1 ]
          R_t = σ_px² / (f² B_t²)                 (current-pixel only)
```

The Kalman innovation variance becomes

```
S_t  =  H Σ H^T + R
     =  H_x² · σ²_x + 2 H_x · cov_{xη} + σ²_η + R
```

The cross term `2 H_x · cov_{xη}` is what the scalar filter was missing.
After the update, `η_t^ref` and its covariance with `x` are propagated
forward; on a re-anchor, `η_t^ref` is reset to the current-pixel noise
variance (the new reference *is* the just-observed pixel) and `cov_{xη}`
is reinitialised from the chart Jacobian.

In code (`SparseVogiatzisFilter._vogiatzis_update`):

```python
S       = H*H*σ²    + 2*H*cov_ce  + var_eta  + tau_rho
K_c     = (H*σ²     + cov_ce)     / S        # gain into canonical
K_e     = (H*cov_ce + var_eta)    / S        # gain into eta
new_σ²  = σ² − K_c*(H*σ² + cov_ce)
```

Empirically NEES at z=5 m goes from 0.09 → 0.93 (target 1).

## Is this "2D"?

Dimensionally yes — the Kalman update operates on a 2-vector with a 2×2
covariance. But only one of the two state components is a physical
quantity:

* `x_t` — depth in the canonical chart (real state)
* `η_t^ref` — the reference pixel's noise sample (auxiliary)

`η_t^ref` doesn't represent anything in the world; it exists solely to
let the Kalman update see the cross-frame noise correlation it was
otherwise blind to. On every re-anchor it is reset (its prior history is
discarded), so it carries no inertia of its own.

So **"1.5D"** is a fair tongue-in-cheek label: one full physical
dimension plus a half-dimension of noise bookkeeping. The classical name
is **state augmentation for colored measurement noise** (e.g., Bryson &
Henrikson 1968, *Estimation Using Sampled Data Containing Sequentially
Correlated Noise*). It's the textbook trick for MA(1) observation noise,
applied here to a depth filter rather than the usual aerospace example.

## Practical consequences

| Property                      | Scalar (1D)     | Augmented (1.5D)         |
|-------------------------------|-----------------|--------------------------|
| State dimension               | 1               | 2                        |
| Cov entries to track          | 1 (`σ²`)        | 3 (`σ², cov_{xη}, σ²_η`) |
| Obs Jacobian                  | scalar `H`      | `[H, 1]`                 |
| Innovation variance `S`       | `H²σ² + R`      | `H²σ² + 2H·cov + σ²_η + R` |
| Behaviour at re-anchor        | n/a             | reset `η`-row of cov     |
| NEES at z=5 m, no outliers    | ≈ 0.09 (under)  | ≈ 0.93 (correct)         |
| Asymptotic MSE shrink rate    | ~ 1/n           | ~ 1/n² (matches truth)   |
| Cost per update               | O(1)            | O(1) (still scalar inv)  |

The augmented filter is asymptotically *more* accurate than the white-
noise version, not just better-calibrated — it correctly extracts the
telescoping cancellation that the scalar filter conflates with noise.
