# Patch-Grid Direct Depth With Sparse GB Priors

## Motivation

The dense-depth problem here is not simply "compute optical flow, then
triangulate depth." That pipeline is fragile in the exact situations where the
system needs memory:

- Dense optical flow is available at many pixels, but depth from flow is weak or
  undefined when translational parallax is small.
- Sparse Vogiatzis GB landmarks can remember metric depth through weak-parallax
  frames, but they are sparse and do not naturally draw object boundaries.
- FlowDep-style dense bilinear splatting carries dense state, but the
  propagation/rendering path is visually and structurally messy.

The goal of this idea is therefore not to get fresh dense metric depth from
every frame. The goal is:

> Use sparse GB landmarks as persistent metric anchors, and use overlapping
> patch-grid direct alignment as the dense spatial support that turns those
> anchors into a useful obstacle-depth image.

For Phase 1, the densifier itself should not maintain temporal depth memory.
The temporal memory already lives in `SparseVogiatzisFilter`: sparse GB
landmarks can keep metric depth through low-parallax frames. The patch-grid
mapper is a per-frame consumer of those sparse seeds plus the current/reference
image pair.

## Core Idea

Lay down a grid of patch centers. Use a patch stride equal to half the render
cell size, so each render cell is covered by four neighboring patch centers:

```text
patch centers:  x   x   x
                x   x   x
                x   x   x

render cell:       [   ]

Each cell receives votes from the four surrounding patch centers.
```

For each patch center, estimate inverse depth directly by minimizing a joint
cost:

```text
E_i(rho_i) =
    E_photo_i(rho_i)
  + lambda_seed E_seed_i(rho_i)
  + lambda_neighbor E_neighbor_i(rho_i)
```

where `rho_i = 1 / Z_i`.

The Phase 1 implementation should use the previous frame as the reference.
More general co-visible references can be added later, but the previous-frame
case keeps the patch lifecycle simple.

The photometric term uses the current EqVIO pose and a candidate depth to warp
current-frame pixels into the previous/reference frame for comparison. Each
patch center `p` lives in the current image; the inverse depth `rho_i` is
defined at `p`. The caller passes `I_ref` and `T_ref_curr` together:

```text
E_photo_i(rho_i) =
  sum_{p in patch_i} robust(
    I_ref(project(T_ref_curr · unproject(p, rho_i))) - I_curr(p)
  )
```

The warp `project(T_ref_curr · unproject(p, rho_i))` is a full SE(3)
inverse-depth warp: it unprojects `p` at candidate depth in the current frame,
transforms by the full relative pose (rotation + translation), and reprojects
into the reference frame. Both rotation and translation are handled in a single
warp step — do not apply a separate derotation and then warp again, as that
double-counts the rotation.

This convention means the patch grid is rebuilt in the current frame on every
image. There is no patch-depth state transfer in Phase 1. Sparse GB seeds are
the only temporal memory.

The sparse GB term anchors the patch to nearby sparse metric depths, weighted
by inverse seed variance so that converged seeds pull harder than uncertain
ones:

```text
E_seed_i(rho_i) =
  sum_{s in nearby GB seeds} w_is / sigma_s^2 (rho_i - rho_s)^2
```

where `sigma_s^2` is the GB seed's inverse-depth variance and `w_is` is a
spatial proximity weight (e.g. inverse pixel distance, bilinear, or binary
within-patch membership).

If the sparse GB query returns depth variance `sigma_z^2`, convert it before
using it in this term:

```text
sigma_rho^2 ~= sigma_z^2 / Z^4
```

The neighbor term is optional and should be weak unless there is good evidence
that two patches lie on the same surface.

## Patch State

Each patch should carry:

```text
rho              inverse depth
var              inverse-depth variance
status           UNKNOWN / SEED_ONLY / PHOTO_REFINED / REJECTED
last_residual    final robust photometric residual
photo_curvature  Hessian contribution from photometric alignment
seed_support     number/effective weight of nearby sparse GB seeds
```

Use these status definitions:

```text
UNKNOWN
  No nearby sparse GB seed and no reliable photometric depth estimate.

SEED_ONLY
  Initialized mostly from nearby sparse GB depth. Current photometric depth
  curvature is weak or absent, so the patch does not claim a fresh depth
  measurement.

PHOTO_REFINED
  Current photometric curvature is strong enough, the final residual is
  acceptable, and the optimized rho is consistent with nearby sparse GB seeds.

REJECTED
  Photometric residual or seed disagreement is too large. Do not render the
  patch unless it is supported by another seed/observation path.
```

## Optimization

Use inverse depth `rho`, not depth `Z`, because the monocular reprojection
sensitivity is closer to linear in inverse depth.

For a patch, start from one of:

```text
1. weighted median/mean of nearby sparse GB rho
2. optional broad default rho if exploring seedless photometric candidates
3. invalid/unknown if neither is allowed
```

Sparse GB seeds are excellent initial guesses. This is usually better than the
zero-flow initialization used by vanilla DIS, because a correct same-surface GB
seed places the scalar inverse-depth search close to the metric solution before
photometric refinement begins. The danger is not using seeds for
initialization; the danger is allowing a wrong-side seed to act as an
uncapped high-confidence measurement near an occlusion.

Then run a small 1D optimization:

```text
for iteration in 1..N:
    warp patch using rho
    compute photometric residual r_p and scalar Jacobian J_p for each pixel p
    add seed/neighbor residuals
    solve scalar Gauss-Newton step: delta = -(J^T r) / (J^T J)
    apply IRLS weights from the Huber loss (downweight |r_p| > delta_huber)
    clamp rho to valid range
```

For the first implementation, use a finite-difference scalar Jacobian:

```text
J_p ≈ (r_p(rho + eps) - r_p(rho - eps)) / (2 eps)
```

This is slower than an analytic Jacobian but much harder to get wrong, and
`rho` is only one scalar per patch. Once the residual and status logic are
visually sane, replace it with the analytic form below.

The per-pixel analytic photometric Jacobian is also a scalar:

```text
J_p = dI_ref/dp_ref · dp_ref/drho
```

where `dI_ref/dp_ref` is the reference image gradient at the warped location
(2-vector) and `dp_ref/drho` is the derivative of the projected reference
pixel with respect to inverse depth (2-vector), so the dot product yields a
scalar per pixel. Let `q = unproject(p, rho)` in the current frame and
`q_ref = R q + t` where `R, t` are from `T_ref_curr`:

```text
p_ref = project(q_ref)
dp_ref/drho = d[project(q_ref)] / drho
            = (K / q_ref[2]) · (dq_ref/drho)
              - (K q_ref[:2] / q_ref[2]^2) · (dq_ref[2]/drho)
```

Since `rho` only scales the depth of `q` (and `t` is independent of `rho`),
`dq_ref/drho = R · dq/drho` where `dq/drho = -q / rho` for
`q = bearing / rho`.

Because the state is scalar, the Hessian and gradient are both scalars, so
each GN step is a division. The implementation can start with a discrete
search (3–5 samples around the initial `rho`) before switching to GN
refinement.

For Phase 1, use this scalar search instead of an image pyramid:

```text
rho candidates = rho_seed * [0.5, 0.75, 1.0, 1.25, 1.5]
```

or search symmetrically in log-depth:

```text
logZ candidates = logZ_seed + [-a, -a/2, 0, a/2, a]
```

Pick the lowest robust `photo + seed` cost, then run scalar GN. A pyramid is
not required initially because the unknown is one scalar, the camera pose is
known, and sparse GB seeds provide metric initialization. Add a pyramid only if
experiments show repeated convergence failures from large inter-frame motion,
bad seed initialization, older reference frames, or repetitive texture. A
pyramid widens the convergence basin but does not make the photometric cost
convex.

After convergence, the posterior inverse-depth variance is approximately the
inverse of the total Hessian:

```text
var_post ≈ 1 / (H_photo
                 + lambda_seed sum_s w_is/sigma_s^2)
```

This assumes photometric residuals are normalized by intensity noise variance.
If the residuals are in raw intensity units, `H_photo` must be divided by the
assumed intensity noise variance `sigma_I^2` before combining with the seed
Hessian, which is already in inverse-depth units.

This variance is used for same-frame cell fusion only. It is not carried as a
patch prior to the next frame.

## Confidence And Observability

The crucial diagnostic is the photometric curvature:

```text
H_photo = J_photo^T J_photo
```

If `H_photo` is small, current-frame photometry is not informative about depth.
This happens under weak translation, pure rotation, low texture, or fronto-
parallel motion with little depth sensitivity.

In that case, do not pretend the frame measured depth. Instead:

```text
if seed support exists:
    set rho from nearby sparse GB seeds
    mark status = SEED_ONLY
else:
    status = UNKNOWN
```

If `H_photo` is large and the residual is good:

```text
status = PHOTO_REFINED
var = var_post from GN Hessian (see above)
```

If the photometric residual strongly disagrees with the seed term:

```text
status = REJECTED
or reset the patch if disagreement persists
```

## Rendering To Cells

The patch grid is denser than the output cells. Define output cells on a
regular current-image grid with side length `cell_size`. Define patch centers
with `patch_stride = cell_size / 2`, aligned so that each interior output cell
is surrounded by four patch centers. Border cells may have fewer supports and
should be handled by the same weighted fusion rule.

For a cell, fuse the surrounding patch inverse depths:

```text
rho_cell =
  sum_i w_i rho_i / sum_i w_i
```

Suggested weight (Phase 1):

```text
w_i = status_weight_i / max(var_i, var_floor)
```

Status weights:

```text
PHOTO_REFINED : 1.0
SEED_ONLY     : 0.6
UNKNOWN       : 0.0
REJECTED      : 0.0
```

The `1/var` factor already encodes confidence from the GN Hessian, so the
status weight only needs to discount estimates that lack fresh photometric
evidence. Later phases can multiply in a robust-residual downweight or an
edge-aware coverage term, but Phase 1 should start with just these two
factors.

The output should preserve status/confidence, not only depth:

```text
depth_cell
var_cell
status_cell
support_cell
```

This allows the visualizer and obstacle logic to distinguish fresh observations
from seed-only estimates.

## Difference From DIS Then Triangulate

DIS-then-triangulate:

```text
image pair -> unconstrained 2D optical flow -> triangulate depth from flow + pose
```

Patch-grid direct depth:

```text
image pair + pose + sparse GB priors -> optimize depth directly
```

The differences are important:

- DIS estimates 2D correspondence first, without metric-depth knowledge.
- Direct depth constrains the correspondence to the 1D curve induced by camera
  motion and candidate depth.
- Sparse GB seeds can bias the solution toward remembered metric depth when
  the photometric term is weak.
- The optimizer can detect low depth curvature and mark a result as
  seed-only instead of producing a fake fresh measurement.

This does not violate observability. It cannot create new depth without
parallax. It can, however, avoid throwing away useful sparse GB depth just
because the current frame has poor parallax.

## Difference From FlowDep Bilinear Splatting

FlowDep keeps a dense inverse-depth state and forward-warps it using bilinear
splatting. That gives temporal memory, but introduces:

- splat holes,
- many-to-one collisions,
- z-order ambiguity,
- variance propagation over a dense state,
- visually messy rendering artifacts.

Patch-grid direct depth keeps memory in the sparse GB filter and re-renders by
local weighted fusion:

```text
sparse GB metric anchors
+ current photometric support
-> overlapping patch votes
-> cell depth
```

This is less physically complete than dense warping, but it may be cleaner for
obstacle avoidance because it has explicit confidence states and avoids dense
forward-splat artifacts.

## What It Can And Cannot Solve

It can help with:

- weak but nonzero parallax,
- noisy optical flow,
- repetitive texture where unconstrained flow drifts,
- preserving recently observed obstacles,
- making sparse GB depth more spatially useful,
- reducing visual blockiness compared with point splats.

It cannot solve:

- completely unobserved new surfaces,
- pure no-parallax depth discovery,
- textureless regions without nearby priors,
- dynamic-object depth without reset logic,
- occlusion boundaries without edge/status handling.

The defensible claim is:

> The method improves dense obstacle-depth continuity by combining persistent
> sparse metric depth with patch-level direct photometric support. It does not
> claim to recover new monocular depth in unobservable no-parallax conditions.

## Implementation Sketch

Recommended module:

```text
eqvio/patch_depth_mapper.py
```

Recommended settings:

```python
patch_size = 16
patch_stride = 8
cell_size = 16
max_depth = 20.0
min_depth = 0.1
photo_huber_delta = 5.0
lambda_seed = 1.0
lambda_neighbor = 0.05
min_photo_curvature = 1e-6
```

Require:

```text
patch_stride = cell_size / 2
patch_size >= cell_size
```

The first condition gives the intended four-patch overlap per interior output
cell. The second ensures each patch has enough image support to vote into the
cell it influences.

At 752×480 with stride 8, the grid has ~94×60 ≈ 5600 patches. Each carries
only same-frame temporary values, so memory is negligible.

Frame update:

```text
1. undistort current and previous/reference images into the same pinhole frame
   used by sparse GB pixels
2. gather sparse GB seeds and their inverse-depth variances
3. for all patch centers (vectorized across the grid):
     a. find nearby GB seeds (KD-tree or grid lookup)
     b. initialize rho from sparse seeds
     c. solve scalar direct-depth objective (finite-difference GN first)
     d. classify status from curvature, residual, and seed support
4. fuse overlapping patch estimates into output cells
5. expose depth/variance/status for visualization and obstacle logic
```

Step 3 is the hot loop. Because each patch's state is a single scalar, the
warp, residual, Jacobian, and GN solve can all be computed as batched NumPy
operations over the full grid, or JIT-compiled with Numba to avoid Python
per-patch overhead.

For Phase 1, prefer correctness over cleverness:

```text
previous-frame reference only
current-frame patch grid rebuilt every frame
sparse GB seeds as the only temporal memory
finite-difference rho Jacobian
no neighbor term
```

Start simple:

```text
Phase 1: no neighbor term, just seed + photo.
Phase 2: add robust reset on photometric disagreement.
Phase 3: add occlusion-safe seed weighting.
Phase 4: add edge-aware neighbor weighting.
```

Phase 3 should stay lightweight in the core pipeline: cap seed precision, use
spatial gating plus an edge/intensity affinity for weak-curvature `SEED_ONLY`
patches, and optionally re-solve after seed outlier rejection. Details are in
the guardrail note below.

## Guardrail: Seed Bleed

Pure spatial proximity `w_is` causes "seed bleed" at depth discontinuities: a
converged foreground seed with tiny variance can pull nearby background patches
to the wrong depth. This is a guardrail issue, not the core method.

Separate the two roles of sparse seeds:

```text
seed as initialization: usually helpful
seed as strong cost term: dangerous near occlusions
```

When photometric curvature is strong, the image term can correct or reject a
wrong seed:

```text
H_photo >> lambda_seed * precision_s
```

When photometric curvature is weak, the result is seed-driven:

```text
H_photo << lambda_seed * precision_s
```

That is acceptable only if seed association is conservative. A weak-curvature
`SEED_ONLY` patch should not accept an arbitrary nearby seed across a likely
occlusion boundary.

The minimum defensive measure is a variance floor on seed precision:

```text
precision_s = 1 / max(sigma_s^2, sigma_seed_floor^2)
```

This prevents a single hyper-converged seed from dominating the cost.

Then use a cautious rule:

```text
PHOTO_REFINED:
  seeds may be used as initialization and capped regularizers.
  photometric residual and curvature decide the result.

SEED_ONLY:
  require conservative seed association before rendering.
  use spatial gating plus at least one occlusion cue.
```

Useful occlusion cues:

```text
w_intensity    = exp(-(I_patch - I_seed)^2 / 2 sigma_I^2)
w_edge_barrier = exp(-edge_max_on_segment^2 / 2 sigma_edge^2)
```

For `SEED_ONLY`, a hard edge-barrier reject is often cleaner than a soft
weight:

```text
if edge_max_on_segment > edge_threshold:
    reject seed for this patch
```

Post-solve seed outlier rejection can still be used as a cleanup pass, but do
not rely on it as the first line of defense. A wrong low-variance seed can bias
the first solve enough to make itself appear consistent.

## Evaluation

Compare against FlowDep and sparse splatting with metrics that match obstacle
avoidance:

- fill ratio of valid cells,
- nearest-depth stability in obstacle ROIs,
- temporal flicker,
- false-close rate,
- seed-only stale-depth rate after disocclusion,
- visual boundary quality,
- runtime.

The key visual diagnostic is not whether every pixel is filled. It is whether
unknown, seed-only, and photo-refined regions are distinguishable and
behave predictably.
