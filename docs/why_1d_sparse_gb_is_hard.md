# Why The 1D Sparse GB Variants Are Hard To Make Consistent

The 1D sparse Gaussian-Beta filters are attractive because they are cheap, but
their inconsistency is not only an implementation bug.  The scalar depth state
throws away geometric covariance that becomes important at long range and weak
parallax.

## 1D Throws Away Bearing-Depth Coupling

The real landmark uncertainty after repeated bearing observations is 3D.  Even
if the quantity of interest is depth, the depth uncertainty is coupled to image
bearing and camera motion.  A scalar filter stores only one coordinate, such as
`z`, `1/z`, or `-log z`, and therefore cannot represent cross terms between
bearing and depth.

This matters most when parallax is weak.  A small bearing error can look like a
large depth error, and the correct covariance direction is a long, tilted
ellipsoid.  A 1D filter must project that ellipsoid onto one axis every frame.
That projection is easy to make too small.

## Re-Anchoring Creates Hidden Correlations

The 1D update uses triangulation between a reference observation and the current
observation.  The reference pixel is noisy, and after re-anchoring, that noisy
quantity becomes part of the next observation model.

The current 1D implementation can carry a small amount of reference-noise state
with `cov_ce` and `var_eta`, but the exact process is closer to a chain of
correlated observations.  A scalar sequential filter with a tiny augmented state
does not fully represent that Markov correlation.  This is why a batch or
fixed-lag formulation can look sane while the 1D sequential update remains
overconfident.

## Chart Choice Does Not Remove The Structural Issue

`EUCLIDEAN`, `INVDEPTH`, and `POLAR` change the scalar coordinate, but they do
not restore the missing 3D covariance.  They can improve numerical scaling, but
they still rely on a scalar innovation variance that must summarize bearing
noise, reference noise, baseline uncertainty, and chart nonlinearity.

At 60-100 m, small mistakes in this summary dominate the NEES plot.  That is
why all 1D variants can become overconfident under pure Gaussian KLT noise,
even without explicit outliers.

## Why 3D Bearing-Only Is Easier To Trust

The 3D variants keep the full landmark covariance in a local chart.  After
initial triangulation, repeated bearing updates and pose prediction can update
the full 3D covariance directly.  This avoids the extra scalar triangulation
depth update, which otherwise reintroduces reference-noise bookkeeping and
double-counting risk.

For the paper narrative, the clean comparison is therefore:

- 1D sparse GB: cheap, useful as a heuristic, but hard to make rigorously
  consistent at long range.
- 3D sparse chart filter: still sparse and much cheaper than full EqF landmark
  augmentation, but keeps the covariance geometry needed for NEES consistency.
- Dense/patch depth: obstacle-avoidance coverage layer, not the consistency
  proof.

## Practical Direction

Keep the 1D variants as a baseline and speed reference.  Do not spend too much
paper effort claiming the 1D scalar GB filter is fully consistent at high
altitude.  The stronger technical claim should be that preserving 3D landmark
covariance makes the sparse long-range depth filter consistent while remaining
far cheaper than in-state landmark augmentation.
