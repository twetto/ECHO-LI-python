# NEES Fix: Re-Anchoring After Each Update

## Problem

All parametrizations show NEES >> expected (overconfident filter):

| Filter | Avg NEES (2nd half) | Expected |
|--------|-------------------:|----------|
| euclidean / GB | 76.6 | 1 |
| invdepth / GB | 32.6 | 1 |
| polar / GB | 66.6 | 1 |
| polar3d / GB | 28.6 | 3 |

(50 MC trials, z*=100m, no outliers, P_UU 6x6)

## Root cause: correlated observations

The filter assumes each observation is independent, but they are not.

Currently the reference frame (ref_T_WC, ref_uv) is set once at feature
initialization and never updated.  Every subsequent triangulation reuses
the same noisy reference pixel.  If the reference pixel has +0.05px error,
*every* triangulated z_obs inherits that bias — the observations are
correlated through the shared reference noise.

The Vogiatzis update treats each observation as independent with variance
tau_sq.  Correlated observations make the filter shrink its variance
faster than justified, producing NEES >> 1.

## Fix: re-anchor after each update

After each successful Vogiatzis update, overwrite the reference with
the current frame:

```python
feat.ref_T_WC = T_WC.copy()
feat.ref_uv = uv_curr.copy()
feat.ref_stamp = stamp
```

This resets the Markov chain: the next triangulation uses a fresh
reference pixel with independent noise.

### Lifecycle becomes

1. **Initialize (spawn):** At detection, save current pixel + pose as
   anchor reference.  No update yet (no baseline).

2. **Track & coast (predict):** Each frame, propagate depth uncertainty
   using P_UU.  If `_triangulate()` fails (insufficient parallax or
   gating), do NOT update — just coast.

3. **Gate, update, re-anchor:** When parallax is sufficient and
   triangulation succeeds:
   - Compute z_obs and tau_sq from reference→current baseline
   - Run Vogiatzis mixture update
   - **Re-anchor:** set reference = current frame

Each update now uses an independent observation.  The filter coasts
between updates, accumulating parallax naturally.

### What changes

In `SparseVogiatzisFilter.update()` and `SparseVogiatzisFilter3D.update()`:

- After the three update branches (new feature / reset / regular update),
  before `track_length += 1`, add re-anchoring.
- The existing parallax gating in `_triangulate()` already handles
  coasting — when triangulation returns z_obs <= 0, the filter skips the
  update and does not re-anchor.

### What does NOT change

- **Prediction:** still frame-to-frame (prev→curr), independent of
  reference frame.  This is correct — prediction tracks the physical
  motion of the landmark.
- **baseline_tau_sq:** still computed from reference→current baseline.
  With re-anchoring, dt_total is the time since last successful update,
  which correctly reflects the velocity uncertainty accumulated since
  the reference was set.
- **Convergence rate:** may slow slightly (smaller baselines per
  observation), but each observation now carries its full advertised
  information.  The filter should converge at the statistically correct
  rate.

### Effect on Gaussian-Beta mixture

Re-anchoring should *help* GB discrimination.  With fixed reference, the
correlated noise made inliers and outliers look more similar (both
biased by the same reference error).  With independent observations,
the GB mixture can cleanly separate the two modes.

## Expected outcome

After re-anchoring, NEES should settle near the expected value (1 for
1D filters, 3 for polar3d) within the 95% chi-squared bounds.  If NEES
is still elevated, remaining sources to investigate:

- Linearization error in the prediction Jacobian (especially Euclidean
  at large depth)
- The variance floor (1e-8) preventing variance from tracking true
  uncertainty
- Process noise model mismatch (P_UU may not match the true velocity
  distribution in the simulation)
