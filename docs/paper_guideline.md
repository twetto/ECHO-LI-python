# Paper Guideline: High-Altitude Monocular EqVIO with Out-of-State Structure

## One-Sentence Thesis

A monocular VIO system operating at 60-100 m altitude can keep the
mathematical consistency of an in-state Equivariant Filter while surviving
weak-parallax geometry by using an out-of-state Gaussian-Beta depth filter to
discover and initialize large-scale planar structure.

## Primary Motivation

The motivating use case is not generic visual SLAM.  It is monocular
visual-inertial odometry from high altitude, roughly 60 m up to 100 m, where
the camera sees mostly distant ground/structure and the translational parallax
per frame is extremely weak.

At this range, standard monocular VIO has several coupled problems:

- Image motion from real depth variation is tiny, so triangulated depth is
  noisy and slow to converge.
- A small number of in-state landmarks may not cover enough of the scene to
  stabilize geometry.
- Naive Euclidean landmark initialization is poorly conditioned at long range.
- Planar structure, especially ground-like or wall-like surfaces, is a valuable
  source of geometric regularity, but only if the point-plane update is
  parametrized correctly.
- A dense in-state landmark set would be too expensive, but a purely
  out-of-state mapper cannot use the EqF's full consistency machinery.

The paper should be written around this survival problem: how to make
monocular EqVIO remain useful when most visual depth information is weak,
far-range, and structurally planar.

## Target Paper Shape

The strongest version of the paper is not "we added a depth sidecar."  The
paper should be framed as a hybrid estimator architecture:

1. The in-state EqVIO backend handles the trusted, tightly coupled estimation
   problem: IMU state, selected point landmarks, and dual-action planar
   landmarks.
2. The out-of-state Sparse Vogiatzis filter handles scalable discovery:
   hundreds to thousands of tracked features, robust depth hypotheses, and
   candidate plane support.
3. Information crosses the boundary conservatively.  The sparse pool proposes
   or initializes structure; the EqF performs its own statistically principled
   updates.

This gives a journal-style contribution: a complete high-altitude monocular
VIO system design with geometry, consistency reasoning, runtime analysis, and
experimental validation.

## Core Contributions

### 1. Chart-Correct Point-Plane Equivariant Updates

Use the derivations in
`docs/EqVIO_planar_landmarks_parametrisation_euclidean_invdepth_polar.md` as
the mathematical core.

The contribution is that planar incidence constraints are derived in the
correct local point chart:

- Euclidean point landmarks use the direct primal-dual SOT(3) variation.
- Inverse-depth point landmarks use the chain rule through the inverse-depth
  chart.
- Polar/SOT(3) point landmarks use the normal-coordinate injection into the
  4D SOT(3) algebra.

The important claim is not just that the equations are different; it is that
using the wrong lift creates a coordinate mismatch that can push points away
from their planes and destabilize the filter.

### 2. Dual-Action Planar Landmarks In-State

The in-state plane representation should be presented as the trusted structural
part of the estimator.  Planes are not just map decoration; they provide
geometric constraints that couple many point observations and can improve
conditioning in structured scenes.

The paper should emphasize that the planar update is handled inside the EqF
state, where cross-covariances with the sensor and point landmarks are
available.

### 3. Out-of-State Gaussian-Beta Structural Discovery

Use the Sparse Vogiatzis filter as a scalable proposal layer:

- It tracks a large feature pool without increasing EqF covariance dimension.
- It estimates robust per-feature depth and uncertainty under weak parallax.
- It supplies candidate 3D points for plane detection, plane support scoring,
  and landmark/plane initialization.

The robust Gaussian-Beta mixture should be described as a way to survive
triangulation outliers and stale tracks, not as a replacement for the EqF.

### 4. Conservative Interface Between Layers

This is essential for credibility.  Avoid double counting image information.

Safe information paths:

- Sparse GB depths initialize new in-state point landmarks.
- Sparse GB point clouds propose candidate planes.
- Sparse GB support selects which planes are worth adding to the EqF.
- After a plane is accepted, EqF uses its own point-plane/image residuals for
  updates.

Risky information paths:

- Repeatedly feeding sparse-filter depth estimates as measurements into EqF.
- Using the same tracked feature both as an EqF image measurement and as a
  sparse GB depth measurement in the same update without accounting for
  correlation.
- Treating plane proposals from the sparse pool as ground-truth hard
  constraints.

## Claims To Make Carefully

### Safe Claims

- The method targets monocular VIO in weak-parallax, high-altitude regimes
  where depth convergence is slow and sparse in-state landmarks are fragile.
- The architecture preserves a bounded in-state covariance size while allowing
  a much larger structural discovery pool.
- Correct chart-dependent planar lifts prevent the divergence seen with
  mismatched point parametrizations.
- The sparse GB pool improves spatial support for plane discovery.
- The design offers a principled division between statistically coupled
  estimation and scalable out-of-state proposal generation.

### Claims Requiring Experiments

- The system remains stable above 60 m and up to 100 m where point-only
  monocular VIO degrades.
- Trajectory accuracy improves over point-only EqVIO.
- Plane landmarks improve long-term drift.
- Sparse GB plane proposals improve over standard feature selection.
- 3D Sparse Vogiatzis is worth its runtime cost compared with 1D variants.

### Claims To Avoid Unless Proven

- "Dense mapping" unless the output is genuinely dense and evaluated as such.
- "Optimal" unless tied to a precise local Gaussian/EqF statement.
- "No double counting" unless the measurement flow is explicitly separated.
- "Planar landmarks improve accuracy" before numerical evidence exists.

## Minimum Experimental Package

### A. Planar Lift Validation

Goal: prove the math is implemented correctly.

Experiments:

- Synthetic point-on-plane scene.
- Compare Euclidean, inverse-depth, and polar point parametrizations.
- Show that the wrong planar lift diverges or produces biased incidence error.
- Show that the correct lift drives incidence residual `q^T p + 1` to zero.
- Plot NEES or normalized incidence residual if possible.

Required plots:

- Incidence residual over time.
- Point-to-plane distance over time.
- Covariance/NEES consistency, at least in simulation.

### B. Sparse GB Consistency and Runtime

Goal: justify the out-of-state layer.

Experiments:

- 1D Euclidean / inverse-depth / polar Sparse Vogiatzis.
- 3D/polar3d Sparse Vogiatzis.
- No-outlier and outlier cases.
- Long-range case such as `Z_TRUE=100m`.
- Altitude sweep, e.g. 20 m, 40 m, 60 m, 80 m, 100 m.
- Runtime per feature-frame.

Required plots:

- Depth error over time.
- Canonical or depth variance over time.
- Inlier ratio over time.
- NEES over Monte Carlo trials.
- Runtime table: 1D variants vs 3D variant.

### C. Plane Discovery Coverage

Goal: show why the sparse pool is useful.

Experiments:

- EqF-only landmark pool, e.g. 40 in-state features.
- Sparse GB pool, e.g. 300-1500 out-of-state features.
- Plane detection from each pool.

Metrics:

- Number of plane-supporting features.
- Spatial coverage of detected support points.
- Plane detection stability over time.
- Plane fit residual.
- Plane lifetime / persistence.

Visuals:

- Camera overlay with in-state points vs sparse-pool points.
- Plane support coloring.
- Before/after adding sparse-pool proposals.

### D. Full System Evaluation

Goal: show the architecture matters end to end.

Baselines:

- Point-only EqVIO.
- EqVIO with in-state planar landmarks only.
- EqVIO plus sparse GB discovery but no in-state planes.
- Full system: EqVIO + in-state planes + sparse GB proposals.

Metrics:

- ATE/RPE trajectory error.
- Success/failure rate across altitude bands, especially 60-100 m.
- Depth convergence time for far-range features.
- Plane residual / structural map quality.
- Feature/plane support count.
- Runtime breakdown.
- Failure cases: low parallax, outliers, clustered tracking loss.

Datasets:

- EuRoC is a reasonable starting point.
- Add structured indoor sequences if available, since planar landmarks need
  visible planar structure.
- Add or simulate high-altitude monocular sequences with ground/roof/large
  planar structure at 60-100 m.  This is the motivating regime and should not
  be left as a side note.
- Include at least one challenging sequence where feature coverage matters.

## Figure TODOs

Each main figure should answer one reviewer question directly.  Avoid figures
that only look visually better without proving a claim.

### 1. Sparse GB Consistency Under Outliers

Reviewer question: is the robust out-of-state depth filter statistically
calibrated, or does it only look good in selected trials?

Recommended layout:

```text
columns: 0%, 10%, 20% outlier rate
row 1: median NEES over Monte Carlo trials with IQR band
row 2: median relative depth error
row 3: failure / wrong-mode rate
curves: Gaussian vs Gaussian-Beta, preferably for the chosen 1D chart
```

Use median NEES to describe the typical surviving track, but always pair it
with a failure-rate or wrong-mode-rate curve.  This prevents the median plot
from looking like it hides catastrophic early clustered-outlier failures.

Possible failure definitions:

```text
final relative depth error > 20%
query() invalid for more than K consecutive frames after burn-in
inlier probability a/(a+b) below threshold after burn-in
depth estimate locked onto the synthetic outlier mode
```

Caption claim:

> The Gaussian-Beta filter remains calibrated for typical surviving tracks and
> exposes early wrong-mode failures as a separate track-management issue.

### 2. High-Altitude Weak-Parallax Sweep

Reviewer question: why does the 60-100 m regime need this machinery?

Recommended layout:

```text
x-axis: true depth / altitude, e.g. 20, 40, 60, 80, 100 m
y-axis options:
  - final relative depth error
  - convergence time to <10% error
  - valid query rate
  - median NEES after convergence
curves: Gaussian EKF, 1D Sparse GB, optional 3D IEKF
```

This figure should make the weak-parallax problem visible without requiring
readers to infer it from equations.

Caption claim:

> At high altitude, per-frame parallax is too weak for naive depth convergence;
> the robust sparse filter preserves usable metric depth hypotheses over long
> sequences.

### 3. Plane Discovery Coverage

Reviewer question: does the large sparse GB pool actually help plane
detection, or is it just extra machinery?

Recommended layout:

```text
left: image with EqF in-state points only
middle: image with sparse GB pool support
right: detected plane support / fitted plane overlay
bottom or side: support count, spatial coverage, fit residual over time
```

Compare EqF-only feature support against sparse-pool support.  The important
visual is spatial coverage, not only the number of points.

Caption claim:

> The out-of-state pool provides enough spatial support to propose stable
> planes without placing all candidate features in the EqF covariance.

### 4. Full VIO Ablation

Reviewer question: do planes and sparse proposals improve the actual odometry?

Recommended layout:

```text
table:
  methods: point-only EqVIO, EqVIO+planes, EqVIO+sparse proposals, full hybrid
  metrics: ATE, RPE, success rate, runtime, plane support count

plot:
  trajectory overlay for one representative sequence
  altitude/scale error over time for high-altitude sequence
```

This is the figure/table that decides whether the paper can claim odometry
improvement.  If planes do not clearly improve trajectory error, demote the
plane claim and emphasize structural proposal/coverage instead.

### 5. Obstacle-Depth / Patch Mapper Qualitative

Reviewer question: does the sparse pool help obstacle-depth coverage without
hallucinating confidence?

Recommended layout:

```text
left: image frame
middle: sparse GB seeds, colored by depth/status
right: patch-grid obstacle depth map
inset: status map with PHOTO_REFINED / SEED_ONLY / UNKNOWN / REJECTED
```

Use an obstacle-relevant depth cap such as 20-30 m.  Render far or unknown
regions as gray/transparent so the background does not destroy the colormap.

Caption claim:

> The patch mapper is an application-layer rendering of sparse structural
> seeds for obstacle avoidance; it distinguishes fresh photometric depth from
> seed-carried depth instead of claiming dense confidence everywhere.

This figure should remain secondary unless the paper explicitly includes
obstacle avoidance as an evaluated contribution.

## Suggested Paper Outline

### 1. Introduction

Problem: high-altitude monocular VIO has weak parallax and slow depth
convergence.  Point-only VIO is sparse and can miss structural information,
while putting every candidate feature in-state is computationally expensive.

Proposal: a hybrid architecture with exact in-state equivariant structure and
scalable out-of-state structural discovery.

### 2. Related Work

Cover:

- EqF / invariant filtering for VIO.
- MSCEqF / MSCKF-style equivariant filtering.
- Planar landmarks and point-plane constraints.
- OpenVINS / ov_plane and plane-aided VIO.
- Inverse-depth and polar landmark parametrizations.
- Robust depth filters such as SVO/REMODE/Vogiatzis-style filters.
- Feature management and out-of-state landmark handling.

Keep this section factual.  Do not oversell against optimization-based SLAM.

### Positioning Against MSCEqF

This question will come up because the EqVIO authors later developed
MSCEqF-style equivariant filtering.  The answer should be respectful and
specific:

MSCEqF is a natural formulation for compact trajectory-focused VIO.  It uses
feature tracks as multi-view constraints and avoids maintaining a large
persistent landmark map.  That is a strength when the objective is camera/IMU
trajectory estimation.

This paper has a different objective: persistent structural VIO.  Planar
landmarks are not merely temporary constraints; they are geometric objects with
identity, covariance, support, and repeated point-plane incidence relations.
The sparse GB pool also needs somewhere to promote discovered structure.  A
pure MSCKF/MSCEqF design would marginalize away exactly the persistent
structure this paper wants to estimate.

Suggested wording:

> While MSCEqF provides a compact equivariant filtering approach for
> feature-track constraints, our objective requires persistent structural
> states.  We therefore retain an in-state landmark EqF for selected points and
> planes, and use an out-of-state robust depth filter only as a scalable
> proposal layer.

Do not frame this as "old EKF-based VIO is better than MSCEqF."  Frame it as a
different state-management choice for a different objective.

### Positioning Against OpenVINS / ov_plane

OpenVINS and ov_plane are highly relevant baselines.  The distinction should
not be "they use planes, we use planes."  The distinction is the operating
regime and estimator architecture:

- ov_plane uses planar structure primarily to improve trajectory estimation in
  indoor environments where nearby planes such as walls, floors, and tables
  are abundant.
- This paper targets high-altitude monocular VIO at 60-100 m, where parallax
  is weak and the available structure is often far-range ground, roof, or
  large planar surfaces.
- ov_plane is built around the OpenVINS/MSCKF ecosystem.  This paper uses an
  Equivariant Filter with chart-correct point-plane incidence lifts.
- This paper adds a large out-of-state Gaussian-Beta depth pool to discover
  plane support without placing all candidate features in the in-state
  covariance.

Suggested wording:

> Plane-aided VIO systems such as ov_plane demonstrate that planar structure
> can improve trajectory estimation in indoor environments with abundant
> nearby planes.  In contrast, our focus is the weak-parallax high-altitude
> monocular regime, where persistent equivariant plane states are coupled with
> a large out-of-state robust depth pool to discover sufficient structural
> support without placing all candidate features in the filter state.

If feasible, ov_plane should be an experimental baseline.  If not, the paper
must still discuss it directly and explain why the target regime differs.

### 3. EqVIO With Dual-Action Planar Landmarks

Present the state, symmetry, and point-plane incidence model.  Keep the full
derivation concise and move long algebra to an appendix if the venue permits.

Key result: chart-correct `C_t^*` for Euclidean, inverse-depth, and polar
points.

### 4. Out-of-State Sparse Gaussian-Beta Filter

Present the sparse filter lifecycle:

- feature birth,
- reference/re-anchor,
- prediction,
- triangulated depth update,
- Gaussian-Beta inlier/outlier mixture,
- consistency corrections such as reference-noise correlation.

Explain why this layer remains O(N) and does not add dense covariance blocks.

### 5. Hybrid Structural Discovery Interface

This is the architecture section.  Define exactly what crosses from sparse
pool to EqF:

- candidate point depths,
- plane proposals,
- support scores,
- initialization only.

Also define what does not cross:

- no repeated sparse depth measurements as hard EqF observations,
- no unmodeled reuse of identical image residuals.

### 6. Experiments

Organize by claim:

1. chart correctness,
2. sparse filter consistency/runtime,
3. plane discovery coverage,
4. full VIO performance.

### 7. Discussion and Limitations

Be explicit:

- Sparse GB still ignores full EqF cross-covariances.
- Plane proposals can be wrong under poor depth estimates.
- Textureless planes remain hard for sparse point discovery.
- Robustness depends on tracking quality and RANSAC/feature management.

This honesty makes the paper stronger.

## RA-L Version Strategy

RA-L is short.  The RA-L version should focus on one clean contribution stack:

1. High-altitude monocular VIO survival as the motivating problem.
2. Correct chart-dependent planar lifts.
3. Hybrid in-state/out-of-state architecture.
4. Evidence that sparse GB improves plane proposal coverage at practical
   runtime.

Move long derivations and extra ablations to supplementary material or a
technical report.

Avoid trying to include every densification idea.  The RA-L paper should be
about structural VIO, not dense depth rendering.

## T-RO / IJRR Version Strategy

A full journal version can include:

- full derivations,
- expanded consistency analysis,
- 1D vs 3D sparse filter comparison,
- denser structural mapping experiments,
- more datasets,
- failure-mode analysis,
- implementation/runtime details.

This is where the complete hybrid system story belongs.

## Immediate Action List

1. Implement or verify the chart-correct planar incidence Jacobians.
2. Build a synthetic planar test that fails with the wrong lift and passes with
   the correct lift.
3. Build a high-altitude synthetic sweep from 60 m to 100 m and record depth
   convergence, NEES, and failure rate.
4. Disable or fix tracker RANSAC so sparse support does not randomly disappear.
5. Finalize the sparse GB consistency settings, including covariance floors and
   re-anchor correlation handling.
6. Add a plane-proposal pipeline from sparse GB points to in-state plane
   initialization.
7. Run an ablation table with point-only, planes-only, sparse-only, and full
   hybrid modes.
8. Decide whether the first submission is RA-L or a longer journal based on how
   strong the full-system accuracy result becomes.

## Paper Title Candidates

- High-Altitude Monocular Equivariant VIO with Robust Out-of-State Structural
  Discovery
- Scalable Structural Discovery for Equivariant Point-Plane Visual-Inertial
  Odometry
- Equivariant Point-Plane VIO with Robust Out-of-State Depth Filtering
