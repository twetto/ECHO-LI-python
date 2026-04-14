# Landmark initial covariance across coordinate charts

## TL;DR

`VIOFilterSettings.initial_point_variance` is interpreted **directly in the
active chart's landmark slot**. It is NOT transformed through any Jacobian.
If you change `coordinate_choice`, the numerical meaning of
`initial_point_variance` changes with it. Tune per chart.

## Why this matters

`VIOFilter.process_vision` builds a 3x3 covariance
`point_cov_3x3 = I * initial_point_variance` for each new landmark and hands
it to `VIO_eqf.add_new_landmarks`. That function writes the block straight
into the global `Sigma` at the landmark's slot — there is no chart-aware
transformation on the way in. Whatever units the diagonal has, that is what
the filter sees on the very first Kalman update.

The 3D landmark slot has different physical meaning per chart:

| Chart      | Slot coordinates                          | Units                    |
|------------|-------------------------------------------|--------------------------|
| Euclidean  | `[dx, dy, dz]`                            | meters                   |
| InvDepth   | `[bearing_stereo_u, bearing_stereo_v, 1/||p||]` | dimensionless, dimensionless, 1/m |
| Polar      | `[bearing_stereo_u, bearing_stereo_v, log(||p||)]` | dimensionless × 2, dimensionless | *(planned)*

In Euclidean, setting `initial_point_variance = 50` means std dev ≈ 7 m on
each Euclidean axis — plausible for a scene initialized at ~5 m depth.
In InvDepth, the same value means std dev ≈ 7 on the `1/||p||` slot, which
is nonsense (inverse depth for a 5 m landmark is only 0.2). The bearing
slots are dimensionless and also get the same 50.

## The mapping (for reference / future Polar work)

`eqvio/coordinate_suite/invdepth.py` provides the Jacobian of the
Euclidean-to-InvDepth coordinate change at a landmark position `q0`:

```python
def conv_euc2ind(q0):
    rho0 = 1.0 / np.linalg.norm(q0)
    y0 = q0 * rho0  # unit bearing
    M = np.zeros((3, 3))
    M[0:2, :] = rho0 * sphere_chart_stereo_diff0(y0) @ (np.eye(3) - np.outer(y0, y0))
    M[2, :] = -rho0 * rho0 * y0
    return M
```

If one ever wants to express a *Euclidean-sized* initial variance
`Σ_euc = σ² I` in the InvDepth chart, the principled transform is

```
Σ_ind = J Σ_euc Jᵀ    where  J = conv_euc2ind(p̂)
```

The equivalent `conv_ind2euc(q0)` is the inverse Jacobian. A Polar chart
would need its own `conv_euc2polar` (and its inverse) built the same way:
upper 2×3 rows are the stereographic-bearing Jacobian, the bottom row is
`∂log(||p||) / ∂p = pᵀ / ||p||²`.

## Current convention (2026-04-14)

We chose **not** to apply the chart Jacobian in
`VIOFilter.process_vision`. Reasons:

1. Keeps the landmark-add path chart-agnostic.
2. Avoids a second copy of `conv_euc2ind` (and a future `conv_euc2polar`)
   outside the coordinate suite.
3. `initial_point_variance` is a tuning knob anyway — the Euclidean default
   was already hand-tuned to the consequences of being written into chart
   slots unchanged. Applying a Jacobian would silently retune the filter
   in chart-specific ways.

If you port this filter to Polar, set `initial_point_variance` based on
what that chart actually needs. Do not assume the Euclidean default
transfers.

## What this is not

This note is about where `initial_point_variance` lives. It is **not**
about the `InvDepth` chart being fragile under warmstart (it isn't — see
`tests/test_warmstart_sensitivity.py` for a falsification), nor about the
production regression of FlowDep warmstart on InvDepth (that is an EqVIO
landmark-rejection issue at occlusion boundaries, unrelated to this file).
