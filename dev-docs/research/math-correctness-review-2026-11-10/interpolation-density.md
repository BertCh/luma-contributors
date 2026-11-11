# Math correctness review: interpolation and density (2026-11-10)

Scope id: `interpolation-density`. Directories under
`modules/analysis/src/gpu-spatial-analysis/`: `spatial-interpolation`, `point-density`,
`line-density`, `spatial-validation`, `scoring-rules`, `optimal-transport`, `spatial-interaction`,
`geographic-distribution`, `similar-locations`.

Paths below are relative to `modules/analysis/src`. Line numbers are those of the working tree
on 2026-11-10. `gpu-local-kriging.ts` was edited by another session while this review ran (it grew
by about ten lines above the kernel; the kernel text did not change). Its line numbers are the
later ones.

Scratch scripts are in
`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/9972295f-91de-4373-ba95-d2f799611a84/scratchpad/interpolation-density/`
(`grid-coordinate.mjs`, `checks.mjs`, `fold-seed.mjs`). No spec was run.

## 1. Coverage

Read in full (every formula checked):

- `spatial-interpolation`: `kriging-parameters.ts`, `separation-wgsl.ts`, `gpu-kriging.ts`,
  `grid-neighbor-scan.ts`, `gpu-inverse-distance-weighting.ts`,
  `spatial-interpolation-parameters.ts`, `prediction-covariance.ts`, `equilibrated-solve-wgsl.ts`,
  `gpu-local-kriging.ts`, `gpu-local-rbf.ts`, `gpu-natural-neighbor.ts`,
  `gpu-space-time-kriging.ts`, `gpu-focal-statistics.ts`, `vecchia-likelihood-nodes.ts`,
  `gpu-vecchia-log-likelihood.ts`.
- `point-density`: `kernel-density-math.ts`, `kernel-density-wgsl.ts`,
  `space-time-kernel-density-math.ts`, `space-time-kernel-density-wgsl.ts`,
  `point-density-hexagon.ts`, `point-density-d3-hexagon.ts`, `point-density-smoothing.ts`,
  `point-density-kernels.ts`.
- `line-density`: `line-density-kernels.ts`.
- `scoring-rules`: `gpu-gaussian-predictive-scores.ts`, `gpu-ensemble-scores.ts`.
- `optimal-transport`: `sinkhorn-divergence-wgsl.ts`, `sinkhorn-divergence-parameters.ts`.
- `spatial-interaction`: `gpu-spatial-interaction.ts`.
- `spatial-validation`: `conformal-prediction-nodes.ts`.
- `geographic-distribution`: `geographic-distribution-parameters.ts`,
  `geographic-distribution-kernels.ts`.

Read in part (the kernels and the class documentation, not the constructor or the plumbing):

- `spatial-interpolation/gpu-conditional-gaussian-simulation.ts` (every kernel of the cooperative
  path, lines 280 to 609; the retained-preparation control code only skimmed).
- `spatial-interpolation/gpu-vecchia-fisher-scoring.ts` (class doc and the whole site kernel).
- `point-density/gpu-kernel-density.ts` (docs, bandwidth rule, point weights, gather; lines 60 to
  400 and 560 to 1103).
- `point-density/kernel-density-binned.ts` (lines 100 to 541: plan, binning, blur passes, finish).
- `point-density/gpu-adaptive-bandwidth-kernel-density.ts` (docs and every kernel).
- `point-density/gpu-space-time-kernel-density.ts` (docs and the voxel gather, lines 560 to 715;
  the time sort and the cell-ordered table, lines 346 to 560, not read).
- `point-density/gpu-space-time-relative-risk.ts` (all but imports).
- `point-density/gpu-point-density.ts` (class doc, the square-grid path and finalize, lines 236 to
  300 and 560 to 812; the hexagon path and props not read).
- `line-density/gpu-line-density.ts` (docs, packer, radius selection).
- `optimal-transport/gpu-sinkhorn-divergence.ts` (docs and every kernel; constructor skipped).
- `spatial-validation/spatial-validation-design.ts` (groups and NNDM bodies read; the
  environmental k-means and kNNDM bodies only partly, the lines are very long).
- `spatial-validation/gpu-conformal-prediction-intervals.ts` (docs and rank arithmetic; the
  bisection helper only skimmed).
- `spatial-validation/gpu-geo-conformal-prediction-intervals.ts` (docs and the interval kernel).
- `spatial-validation/gpu-area-of-applicability.ts` (docs, moments, standardization, threshold;
  the distance pass only by search).
- `spatial-validation/gpu-nested-interpolation-validation.ts` (selection and metric kernels).
- `spatial-validation/gpu-spatial-cross-validated-ols.ts` (prediction and metric kernels only).
- `spatial-validation/gpu-spatial-cross-validation.ts` (fold, mask and count kernels, props).
- `geographic-distribution/gpu-geographic-distribution.ts` (docs; spread, ellipse, directional
  mean and both median update kernels; the grouped reduction and sort wiring not read).
- `geographic-distribution/gpu-rose-statistic.ts` (docs, angle, bin and Rayleigh lines).
- `similar-locations/gpu-similar-locations.ts` (docs and the standardization and distance lines).

Opened for doc comments only, math not verified:

- `spatial-interpolation/conditional-simulation-blocked.ts` (only the Philox and Box-Muller lines
  were compared with the cooperative path; the blocked Cholesky is not verified).
- `spatial-interpolation/vecchia-conditioning-sets.ts`, `vecchia-fisher-fit.ts`,
  `vecchia-parameters.ts`, `gpu-vecchia-space-time-log-likelihood.ts`.
- `line-density/gpu-line-length-per-polygon.ts` (class doc and function headers; the clip and
  midpoint kernel is not verified).

Not opened: every `index.ts`, `spatial-validation/spatial-validation-descriptors.ts`,
`geographic-distribution/grouped-sum-reduction.ts`,
`similar-locations/similar-locations-parameters.ts`.

Not present in the scope, although named in the brief: empirical variogram binning (it lives in
`dataframe-analysis/pair-statistics`), a radiation model, log score, interval score.

## 2. Findings

| ID | Severity | Status | File:line | Kernel computes | Reference says | Input that goes wrong | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interpolation-density-01 | wrong-edge-case | CONFIRMED (script) | `point-density/point-density-kernels.ts:279-292`, used at `point-density/gpu-point-density.ts:617`; counts at `:590` | Weighted sums use a cell mapping with a "scaled" branch when the bounds straddle zero. Counts come from luma.gl `GPUGridBinning`, which has no such branch. | One point must land in one cell for both count and sum. The doc at `:231` says "same cell arithmetic as `GPUGridBinning`". | Bounds `[-3, 7]`, 10 columns, point at `x = -2`, weight 5. Count goes to cell 1, sum to cell 0. | Use the plain `(value - min) / (max - min) * size` form that `GPUGridBinning` and `GPUGridAggregation` use, or compute counts from the same keys. |
| interpolation-density-02 | wrong-edge-case | CONFIRMED (script) | `spatial-validation/gpu-spatial-cross-validation.ts:154` | `fold = hashUint32(block ^ seed) % foldCount`. | A new seed should give an independent assignment. | Blocks 0..7, 5 folds: fold 4 is empty for every seed 0..7. Blocks 0..15, 10 folds: fold 4 is empty for every seed 0..15. | Mix the seed after the hash, `hashUint32(hashUint32(block) ^ seed)`, or draw from Philox with key `(seed, purpose)` and counter `block`. |
| interpolation-density-03 | wrong-edge-case | CONFIRMED (by hand) | `spatial-interpolation/gpu-local-kriging.ts:500, 606`; `spatial-interpolation/equilibrated-solve-wgsl.ts:38` | Ordinary co-kriging keeps one unbiasedness row per variable even when the neighbourhood holds no sample of that variable. The zero row fails the solve. | With no secondary sample nearby, ordinary co-kriging reduces to ordinary kriging of the target (gstat does this). | Two variables, neighbourhood with 8 target samples and 0 secondary samples: status 2, prediction NaN. | Drop the constraint row of a variable that has no retained sample (keep the failure when the target variable is absent). |
| interpolation-density-04 | wrong-edge-case | CONFIRMED (by hand) | `scoring-rules/gpu-ensemble-scores.ts:304-307` | `powerOfMagnitude` returns `select(0.0, pow(m, p), m > 0.0)`. For `m = NaN` the test is false, so it returns 0. | The comment says "NaN stays NaN". The class doc at `:95` says NaN inputs give NaN scores. | Observations `[NaN, 2]`, one member `[1, 2]`, pair `(0, 1)`, `p = 0.5`: the term is `(0 - 1)^2 = 1`. Expected NaN. | Return `magnitude` itself when it is not finite, or test `magnitude == 0.0` for the zero case. |
| interpolation-density-05 | precision | CONFIRMED (script) | `line-density/line-density-kernels.ts:284-285` | Spherical cell area `R^2 * dLon * (sin(south + h) - sin(south))` in f32. | Same formula in f64; docs call the area "exact". | `h = 0.001` degrees at latitude 60.123: area off by 0.6 percent. `h = 0.0001` degrees at latitude 45: off by 1.4 percent. | Use `2 * cos(south + h / 2) * sin(h / 2)` for the sine difference. |
| interpolation-density-06 | inconsistent | CONFIRMED (by hand) | `spatial-interpolation/vecchia-likelihood-nodes.ts:211-222` with `gpu-local-kriging.ts:606`, against `gpu-vecchia-fisher-scoring.ts:254` | A site whose conditioning rows are all excluded (validity 0) fails in `GPUVecchiaLogLikelihood` (whole log-likelihood NaN). `GPUVecchiaFisherScoring` treats the same site as marginal. | Both evaluate the same likelihood. The class doc says excluded sites "drop out". | Row 0 of the order has validity 0, row 1 conditions on `{0}`: log-likelihood NaN in one class, finite in the other. | In the site-terms kernel, treat "no usable conditioning row" as the marginal term, as the Fisher kernel does. |
| interpolation-density-07 | wrong-edge-case | CONFIRMED (by hand) | `point-density/space-time-kernel-density-math.ts:77-87` | The packer defaults the time window to `[first slice, last slice]` and throws unless `start < end`. | `sliceCount` is documented as "at least 1", and the window as "ignored without temporal edge correction". | `sliceCount = 1`, no `timeWindow`: throws, also with `temporalEdgeCorrection: 'none'`. | Skip the order check when start equals end and document that one slice needs an explicit window for the uniform correction. |
| interpolation-density-08 | inconsistent | arithmetic CONFIRMED (script), impact SUSPECTED | `spatial-interpolation/gpu-kriging.ts:615-633` against `prediction-covariance.ts:122-147` | `GPUKriging` evaluates `1 - (1 + a) exp(-a)` and `1 - exp(-3 s^2)` directly in f32. `GPULocalKriging` switches the Matern forms to a series below `a = 0.1`. | One model family should have one evaluation. | Lag at `1e-3` of the range: Gaussian semivariance off by 0.7 percent, Matern 3/2 by 0.3 percent. At `1e-4` of the range the Gaussian value is 2 times too large. | Share one WGSL shape module (with the series) between `GPUKriging`, `GPULocalKriging` and the variogram fit. |
| interpolation-density-09 | doc-mismatch | CONFIRMED (fixture note) | `docs/api/gpu-spatial-analysis/operators-density.md:446, 473, 524`; code `spatial-interpolation/gpu-kriging.ts:624` | Gaussian model `1 - exp(-3 (h / range)^2)`. | PyKrige's built-in Gaussian is `1 - exp(-(h / (range * 4 / 7))^2)`, exponent 3.0625. The docs name PyKrige as the reference and only say to pass `sill + nugget`. | Same `range` in both: the two surfaces differ. The fixture (`test/.../kriging-reference.ts:6-8`) had to use a custom PyKrige variogram. | State the factor (`range_pykrige = range * 7 / (4 * sqrt(3))`), and the gstat range parameters (`range / 3`, `range / sqrt(3)`). Refresh the stale "within 0.2 on values near 70" tolerance; the spec uses 2e-5. |
| interpolation-density-10 | doc-mismatch | CONFIRMED (by hand) | `line-density/line-density-kernels.ts:57` | A lon/lat segment is cut along a straight line in degrees and each piece is measured as a great circle. | The class doc says the great-circle metric measures lengths. It does not say the total depends on the grid. | Segment (0, 60) to (90, 60): 4,605 km in one cell, about 5,004 km when cut by 1-degree cells. | Say so in the class doc, as `GPULineLengthPerPolygon` already does, and advise densifying long segments. |
| interpolation-density-11 | doc-mismatch | still present (prior F20) | `line-density/gpu-line-density.ts:184` | Length per cell over cell area. | QGIS "Line density" is the circular-search-radius density. `docs/api/.../operators-geometry.md:717` already says this class is not that. | Any input. | Remove `QGIS "Line density"` from the class doc and from `operators-geometry.md:696`. |
| interpolation-density-12 | untested | n/a | see section 6 | | | | |

### interpolation-density-01: counts and weighted sums use two cell mappings

`GPUPointDensity` on a square grid takes counts from the published `GPUGridBinning`
(`gpu-point-density.ts:590`). With `weights` and the default `sumAccumulation` (`'auto'`, the
deterministic `'workgroup'` path) the sums are keyed by `createPointDensityGridKeysNode`
(`gpu-point-density.ts:617`). That node carries its own `getCoordinate`
(`point-density-kernels.ts:279-292`). When `minimum < 0 < maximum` it divides the value and both
bounds by `max(|min|, |max|)` first. `GPUGridBinning` and `GPUGridAggregation` in
`@luma.gl/graph@10.0.0-alpha.10` do not (`dist/gpu-core/gpu-grid-binning.js:193-197`). The scaled
form is the one `GPUGridIndex` uses, so the local copy follows the wrong sibling.

`grid-coordinate.mjs` replicates both in `Math.fround` arithmetic. Results:

- bounds `[-3, 7]`, 10 cells: 4 of the 9 interior cell edges map to different cells (`x = -2`:
  plain 1, scaled 0);
- bounds `[-500, 1500]`, 100 cells: 14 of 99 edges;
- bounds `[-74.3, 40.9]`, 512 cells: 108 of 511 edges;
- random positions: about 1 to 9 in a million.

The plain form is the right one for `x = -2` (`(1 / 10) * 10` is exactly 1).

Effect: a point on such an edge is counted in one cell and its weight is summed in the
neighbour. `statistic: 'sum'` shows weight in a cell whose count is 0. `output.means` loses the
weight (`sum / count` is 0 in the counted cell and forced to 0 in the other). The unsmoothed
extent and histogram mask on `counts`, so they skip the cell that holds the weight. Lattice data
(integer coordinates, integer cell size) hits this on a large share of points. The `'atomic'`
path is consistent because it uses `GPUGridAggregation`.

Caveat: a device may round the division differently from IEEE round-to-nearest, which moves which
points differ. The two formulas stay different.

No spec covers it: every `GPUPointDensity` spec uses bounds that start at 0, and the CPU oracle
(`test/.../point-density-oracle.ts:49`) uses the plain form.

### interpolation-density-02: the seed only relabels blocks

XOR with the seed permutes the hash inputs. For block ids `0..B-1` with `B` a power of two, every
seed below `B` hashes the same set of values. The fold sizes are identical for all those seeds,
and so is the set of empty folds. `fold-seed.mjs` uses the TypeScript twin of the WGSL hash:

- 8 blocks, 5 folds: `[2,2,2,2,0]` blocks per fold for seeds 0..7;
- 16 blocks, 10 folds: fold 4 empty for seeds 0..15;
- 64 blocks, 10 folds: sizes `[4,7,7,8,4,8,5,9,8,4]` for seeds 0..11.

A caller who sees `status` bit 2 (empty test set) and tries seed 1, 2, 3 gets the same empty fold.
Repeated cross-validation over small seeds is not a set of independent designs. The class doc says
the assignment is "not balanced" but not this.

### interpolation-density-03: ordinary co-kriging needs every variable in every neighbourhood

`coKrigingMean: 'ordinary'` sets `driftCount = variableCount`. The drift column of variable `v`
is 1 for samples of `v` and 0 otherwise (`gpu-local-kriging.ts:500`). When no retained sample
belongs to `v`, that constraint row and column are zero. The Ruiz pass finds `rowMaximum == 0`
and returns false (`equilibrated-solve-wgsl.ts:38`), so the query reports status 2.

For the target variable this is right (the constraint `0 = 1` cannot hold). For a secondary
variable the constraint `0 = 0` is vacuous and the prediction exists. Sparse secondary data with
a search radius is an ordinary case. Not documented.

### interpolation-density-05: spherical cell area loses digits

`checks.mjs` output (f32 against f64):

| south, height (degrees) | relative error |
| --- | --- |
| 45, 0.01 | -1.7e-4 |
| 45, 0.001 | -2.5e-4 |
| 60.123, 0.001 | -5.9e-3 |
| 45, 0.0001 | +1.4e-2 |

`densities` inherits the error. `lengths` does not.

### interpolation-density-06: two Vecchia siblings disagree on an all-excluded conditioning set

`GPUVecchiaLogLikelihood` delegates the conditional to `GPULocalKriging` in `simple` mode. That
kernel skips unusable candidates and returns status 1 when none is left
(`gpu-local-kriging.ts:606`). The site-terms kernel only takes the marginal branch when the CSR
row is empty (`vecchia-likelihood-nodes.ts:211`), so the site counts as failed and the total is
NaN. `GPUVecchiaFisherScoring` builds its own list, ends with `count == 0` and uses the marginal
variance. The spec avoids the case by rebuilding the sets with `validity`
(`test/.../vecchia-log-likelihood.spec.ts:166`). A caller who toggles validity per frame without
rebuilding hits it as soon as the first row of the order is excluded.

### interpolation-density-08: three copies of the variogram shapes

The model shapes are written in WGSL three times: `gpu-kriging.ts:615-633` (semivariogram form),
`prediction-covariance.ts:122-147` (covariance form, with the Matern series) and
`dataframe-analysis/pair-statistics/variogram-model-fit-nodes.ts:75-78`. A fourth place holds the
range derivative (`gpu-vecchia-fisher-scoring.ts:209-213`). The constants are shared
(`MATERN_3_2_RANGE_FACTOR`, `MATERN_5_2_RANGE_FACTOR`); the literal `3.0` of the exponential and
Gaussian models is not. The practical effect of the missing series in `GPUKriging` is limited to
zero-nugget models with pairs closer than about a thousandth of the range, where the pivot-ratio
test is likely to reject the system anyway. That last part is not verified.

## 3. Verified correct

Compared by hand against the cited definition unless a script is named.

- `GPUKriging`: ordinary kriging system `[Gamma 1; 1' 0]`, Lagrange row, variance
  `mu + sum(w gamma0)` rescaled by the total sill, prediction relative to the nearest value
  (Cressie 1993). Spherical `1.5 s - 0.5 s^3`; exponential and Gaussian with factor 3 (practical
  range); Matern 3/2 and 5/2 closed forms. Range factors 4.749031 and 5.924463 checked: both give
  `exp(-3)` at `h = range`.
- `grid-neighbor-scan.ts` ring walk: the stopping bound is conservative (distance to the nearest
  block border that still has unvisited cells, less one cell); clipped rows are still covered.
- `GPUInverseDistanceWeighting`: Shepard `1 / d^p`, Franke-Nielson taper
  `((R - d) / (R d))^p`, log-space accumulation, exact-hit rule, row flip of north-up grids.
- `GPULocalKriging`: covariance-form system with observation error on the diagonal, simple
  (known mean), ordinary, universal (centred affine drift, query drift `[1, 0, 0]`), external
  drift, block support by quadrature, variance `C00 - w'c0 - lambda'f0` with the multiplier
  rescaled by the covariance scale. Matern series coefficients re-derived (3/2:
  `1 - a^2/2 + a^3/3 - a^4/8 + a^5/30 - a^6/144`; 5/2: `1 - a^2/6 + a^4/24 - a^5/45 + a^6/144`).
- `equilibrated-solve-wgsl.ts`: `(D A D) y = D b`, `x = D y`.
- `GPULocalRBF`: all eight kernels and their signs equal SciPy `RBFInterpolator`; minimum degrees
  equal SciPy; the scale substitution for polyharmonic kernels leaves the interpolant unchanged,
  including the thin-plate `r^2 log r` case (the extra `r^2` term is absorbed by a degree-1
  polynomial) and the smoothing term. The default degree of -1 for Gaussian and inverse kernels
  differs from SciPy's 0 and is documented.
- `GPUNaturalNeighbor`: Sibson stolen-area weights; the query cell from bisector intersections;
  the in-circle determinant; the argument that natural neighbours alone define the stolen areas;
  linear limit on hull edges.
- `GPUSpaceTimeKriging` and `prediction-covariance.ts` space-time families: separable,
  product-sum (`k Cs Ct + Cs + Ct`), metric, sum-metric (gstat `vgmST`), Gneiting (2002) with
  `tau >= beta` for two spatial dimensions; `getGneitingSpatialScale` and
  `getGneitingTemporalScale` give `exp(-3)` at the stated ranges. Time lags are absolute values
  before `pow`.
- `GPUConditionalGaussianSimulation` (cooperative path): joint Cholesky, conditional mean
  `mu + L21 L11^-1 (z - mu)`, conditional covariance `L22 L22'`, Philox counter
  `(replicate, query)`, Box-Muller with an open interval for the logarithm.
- `GPUVecchiaLogLikelihood` site term `-1/2 (log 2 pi + log d + r^2 / d)`.
- `GPUVecchiaFisherScoring`: derivatives with respect to log sill, log range (exponential
  `a exp(-a)`, Matern 3/2 `a^2 exp(-a)`) and log nugget; score and expected information
  re-derived (Guinness 2019).
- `GPUFocalStatistics`: clipped window, population standard deviation by a centred second pass.
- `GPUKernelDensity`: 2D constants `1 / (2 pi)`, `2 / pi`, `3 / pi`; Gaussian truncation at 5
  bandwidths; canonical bandwidth ratios `192^(1/6)` and `(2304 / 5)^(1/6)` re-derived from
  `R(K)` and the second moment; Scott equals Silverman in 2D; `numpy.cov(aweights)` correction
  and `n_eff`; uniform (`total / e(u)`) and Diggle (`w / e(x)`) corrections; the closed-form
  quadrant mass of the polynomial kernels and the chord integrals.
- `kernel-density-binned.ts`: linear binning, the fractional box (radius, edge weight and
  normalisation re-derived from the variance), the running-sum update, the two-sided exponential
  pole and its anticausal start value `y / (1 + a)`, the `1 / cellArea` scale.
- `GPUAdaptiveBandwidthKernelDensity`: Abramson rule with the weighted geometric mean, clamp,
  pilot including the point itself (documented).
- `GPUSpaceTimeKernelDensity`: 1D constants `1 / sqrt(2 pi)`, 3/4, 15/16; the two cumulative
  forms; lower-bound binary search on time; product kernel and the two edge factors.
- `GPUSpaceTimeRelativeRisk`: ratio and log ratio of two probability densities.
- `point-density-hexagon.ts`: axial conversion, cube rounding, odd-r offset and centre.
  `point-density-d3-hexagon.ts`: line by line equal to d3-hexbin.
- `line-density-kernels.ts`: Liang-Barsky, Amanatides-Woo with integer step counts, planar piece
  length.
- `GPUGaussianPredictiveScores`: `sd (z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi))` and the erf
  series with its recurrence.
- `GPUEnsembleScores`: sample CRPS, energy score, fair divisor `2 m (m - 1)`, variogram score
  (apart from finding 04).
- `GPUSinkhornDivergence`: log-domain dual updates, separable log-sum-exp, averaged symmetric
  update, marginal error `b |exp((g_old - g_new) / eps) - 1|`, divergence
  `<a, f_ab - f_aa> + <b, g_ab - g_bb>` (Feydy et al. 2019). The remark about POT's
  `empirical_sinkhorn_divergence` is right.
- `GPUSpatialInteraction`: Poisson profile of the intercepts for the four constraint modes,
  iterative proportional fitting, Poisson deviance, golden-section bracket.
- `GPUConformalPredictionIntervals`: ranks `ceil((1 - alpha)(n + 1))` and
  `floor(alpha (n + 1))` (Barber et al. 2021); the split interval is symmetric because the two
  ranks are complements.
- `GPUGeoConformalPredictionIntervals`: weighted quantile with the query weight at infinity.
- `GPUAreaOfApplicability`: `n - 1` standardization, mean of mean pairwise distances, type-7
  quartiles, threshold `min(Q3 + 1.5 IQR, max)` (CAST `trainDI`).
- `GPUNestedInterpolationValidation`, `GPUSpatialCrossValidatedOLS`: MSE, MAE, signed bias,
  R squared about the held-out mean.
- `spatial-validation-design.ts` NNDM loop: equal to the published algorithm, default
  `min_train` 0.05 as in blockCV.
- `GPUGeographicDistribution`: mean centre, standard distance with divisor `W` (ArcGIS),
  covariance eigenvalues, ellipse angle `0.5 atan2(Sxy, (Sxx - Syy) / 2) + pi / 2`, the
  `sqrt(2)` convention as an option (ArcGIS) with degrees of freedom `n` stated, Weiszfeld with
  the Vardi-Zhang step, linear directional mean with angle doubling.
- `GPURoseStatistic`: bins, weighted mean angle, Rayleigh `z = n Rbar^2` on unweighted directions.
- `GPUSimilarLocations`: population z-score and average-tie percentile rank, both as documented.

## 4. Standards consistency

- **Cell mapping.** Two local copies of `getCoordinate`:
  `gpu-inverse-distance-weighting.ts:376-390` (exported as `GRID_COORDINATE_WGSL`, reused by
  kriging and all density gathers) and `point-density-kernels.ts:279-292`. Both are the
  `GPUGridIndex` form. The first is used where it must match `GPUGridIndex` and is also widened
  by one cell, so it is safe. The second must match `GPUGridBinning` and does not (finding 01).
- **Variogram and covariance shapes.** See finding 08. The Matern range factors are shared
  constants. The series is private to `prediction-covariance.ts`.
- **Dense solvers.** No file in the scope uses `analysis-core/kernels/dense-solve-wgsl.ts`.
  `gpu-kriging.ts:638-679` has a private Gaussian elimination;
  `equilibrated-solve-wgsl.ts` serves local kriging, local RBF, space-time kriging and the
  Vecchia kernels; the conditional simulation has its own Cholesky. Pivot rules differ in form
  (`1e-6` of `max(1, gamma)` against `1e-6` of the largest equilibrated entry) but share the
  ratio limit `KRIGING_PIVOT_RATIO_TOLERANCE`. Already listed by the library review (G3); still
  present.
- **Random numbers.** The conditional simulation uses Philox with a purpose id and is the same
  in both paths. The Box-Muller transform is written twice
  (`gpu-conditional-gaussian-simulation.ts:625`, `conditional-simulation-blocked.ts:436`) and a
  third variant lives in `scan-statistics`. There is no shared normal draw. The block-fold hash
  uses `hashUint32`, not Philox (finding 02).
- **Uniform to unit float.** `(f32(n) + 0.5) / 2^24` cannot represent the half for
  `n >= 2^23`; it rounds to even, and `n = 2^24 - 1` gives exactly 1. Harmless here
  (`log(1) = 0`), but the "open interval" is not strictly open. The shared `philoxUnitFloat` has
  no half offset.
- **Standard deviation divisor.** `GPUSimilarLocations` divides by `n`,
  `GPUAreaOfApplicability` and the environmental fold design by `n - 1`,
  `GPUGeographicDistribution` and `GPUFocalStatistics` by `n`. Each is documented and follows its
  reference.
- **Nugget.** `GPUKriging`, `GPULocalKriging`, `GPUSpaceTimeKriging` and the simulation all
  treat the nugget as micro-scale variation (exact at data) and take measurement error
  separately. Consistent.
- **Duplicates.** `GPUKriging` coalesces duplicate positions; `GPULocalKriging` and
  `GPULocalRBF` report a singular system. Documented in both.
- **Tie order.** `GPUKriging`, IDW and `GPULocalRBF` break distance ties by row id.
  `GPULocalKriging` and `GPUSpaceTimeKriging` break them by scan order, which equals row order
  for the exhaustive scan and candidate order for supplied neighbourhoods.
- **Earth radius.** Lengths use the mean radius, cell areas the authalic radius
  (`gpu-line-density.ts:230-231`), both from `analysis-core/contracts/earth.ts`. Consistent with
  the contract.
- **Exact-hit offsets.** IDW forms positive offsets from the grid minimum
  (`gpu-inverse-distance-weighting.ts:508-519`), `GPUKriging` signed offsets from the grid start
  (`gpu-kriging.ts:473-481`). `checks.mjs` found no difference in hit rate between the two.
- **NaN helper.** `GPUSpaceTimeRelativeRisk` and `GPUSinkhornDivergence` declare their own
  `getNaN` instead of `GET_NAN_WGSL`.

## 5. luma.gl official implementations

- **Delegated and used as published:** `GPUGridIndex` (IDW, kriging, all exact density gathers),
  `GPUGridBinning` and `GPUGridAggregation` (`GPUPointDensity`), `GPUReduction`,
  `GPUConvolution`, `GPUHistogram`, `GPURasterBufferToTexture`.
- **Delegated, but paired with a local kernel that changes the meaning:** `GPUPointDensity`
  counts with `GPUGridBinning` and keys the sums itself (finding 01). `GPUGridBinning` is right.
  Fix locally by matching it.
- **luma.gl has two cell mappings.** `GPUGridIndex` scales by `max(|min|, |max|)` when the
  bounds straddle zero; `GPUGridBinning` and `GPUGridAggregation` do not. They can put the same
  point in different cells. Propose upstream: one mapping, exported as public WGSL.
- **Local re-implementation forced by a missing public API.** The per-cell gathers read the
  `GPUGridIndex` buffers directly and repeat its private mapping. The public `GPUGridIndexQuery`
  answers one packed query, not one per output row. Propose upstream: a per-row candidate query
  or the exported mapping.
- **`GPUFocalStatistics` against `GPURasterNeighborhood` (`gpu-raster`).** The published class is
  a weighted stencil with border modes and a nodata policy; it cannot give min, max, range or
  standard deviation, so the local class is not a duplicate. One convention differs: the
  published class always keeps an invalid centre invalid, the local default
  (`propagateCenterNoData: false`) fills it from the window.
- **`GPUSimilarLocations` against `GPUSimilaritySearch` (`gpu-vector-search`).** After
  standardization the local ranking is a squared-Euclidean nearest search to one reference
  vector, which the published class does (weights fold in as `sqrt(w)` scales; the order by
  distance and by squared distance is the same). The local class also returns a rank for every
  row and the unsquared distance. Possible delegation of the search step; I did not read the
  published kernel, so this needs a second look.
- **`spatial-validation-design.ts` environmental mode** runs its own serial farthest-first and
  Lloyd k-means; `GPUKMeans` is published. The class doc already names this as not implemented.
- No published entry point offers kriging, IDW, RBF, natural neighbour, kernel density, line
  density, Sinkhorn, scoring rules, conformal intervals or geographic distribution.

## 6. Test oracles

Independent oracle (a reference library produced the fixture):

- `GPUKriging`: PyKrige 1.7.3 (the Gaussian case with a custom variogram), GSTools 1.7.0 for
  Matern.
- `GPULocalKriging`, `GPULocalRBF`, `GPUNaturalNeighbor`, conditional simulation: GSTools,
  PyKrige, SciPy `RBFInterpolator` (always with `degree=2`), CGAL, gstat
  (`interpolation-advanced.py`, `interpolation-cgal.cpp`, `interpolation-gstat.R`).
- `GPUInverseDistanceWeighting`: SciPy `cKDTree` based reference
  (`parity-wp1-inverse-distance-weighting.py`).
- `GPUKernelDensity`: SciPy `gaussian_kde` and normal integrals
  (`geometry-kernel-density.py`, `geometry-kernel-density-binned.py`).
- `GPULineDensity`, `GPULineLengthPerPolygon`: Shapely and pyproj.
- `GPUGeographicDistribution`: SciPy (geometric median by `minimize`, `circmean`, `circvar`).
- `GPUSinkhornDivergence`: POT.
- `GPUSpaceTimeKriging`: GSTools for the metric family only.
- Fold designs: blockCV (`statistics-blockcv.R`, `spatial-validation-advanced.R`).

Second implementation of the same definition in float64 (independent code, not an independent
library):

- `GPUGaussianPredictiveScores`, `GPUEnsembleScores`: `scoring-rules.py` is NumPy written from the
  scoringRules definitions; its header says the scoringRules run is pending.
- `GPUAreaOfApplicability`, both conformal classes: binary64 references inside the specs.
- `GPUSpaceTimeKernelDensity`, `GPUSpaceTimeRelativeRisk`, `GPUAdaptiveBandwidthKernelDensity`:
  binary64 references; the docs say no sparr fixture exists.
- `GPUSpaceTimeKriging` other than metric: binary64 solves; gstat fixture pending.
- `GPUPointDensity`, `GPUSimilarLocations`, `GPUFocalStatistics`, Vecchia likelihood and Fisher
  scoring: CPU oracles.
- `GPURoseStatistic`: hand values.

Gaps (this is finding interpolation-density-12):

- `GPUPointDensity` with weights and bounds that straddle zero (finding 01). The CPU oracle uses
  the plain mapping, so a straddling fixture would fail today.
- Variogram score with a NaN input (finding 04).
- `GPUSpatialCrossValidation` across seeds (finding 02).
- Ordinary co-kriging with a variable missing from a neighbourhood (finding 03).
- `GPUVecchiaLogLikelihood` with excluded rows left in the conditioning sets (finding 06).
- Spherical `densities` at cell sizes below 0.01 degrees (finding 05).
- `GPULocalRBF` at its default degree; every SciPy fixture passes `degree=2`.
- The fair CRPS and energy score have no outside reference (documented).
- I found no spec for `GPUSpatialInteraction` in its own directory; it is exercised from
  `test/gpu-spatial-analysis/spatial-regression/coverage-*.spec.ts`, which I did not read.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis` (`src/...`, `test/...`). "Failed before" means the spec
was run on the test GPU against the code as the review found it.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| interpolation-density-01 (also luma-official-02) | fixed | `src/gpu-spatial-analysis/point-density/point-density-kernels.ts:286`: the key kernel's `getCoordinate` is now the arithmetic of the published `GPUGridBinning` (the "scaled" branch for bounds that straddle zero is gone). The official class publishes counts only, so the key cannot be taken from it. | `test/gpu-spatial-analysis/point-density/gpu-point-density-cell-mapping.spec.ts` (the lattice case failed before: a cell with count 1 held sum 6 and its neighbour count 1, sum 0); `point-density-grid-coordinate-pin.node.spec.ts` pins the text to `GPUGridBinning` and `GPUGridAggregation`; `test/gpu-spatial-analysis/spatial-interpolation/grid-coordinate-pin.node.spec.ts` pins the `GPUGridIndex` copy in `gpu-inverse-distance-weighting.ts:376`, which keeps the index variant | With `weights`, the default sum mode and bounds that straddle zero, the weight of a point on a cell edge now lands in the cell that holds its count. The single point `x = -2` of the report was already consistent on the test GPU (its division does not round as the float32 replication assumed); the lattice shows the defect |
| interpolation-density-02 | fixed | `src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts:146-216`: every block with a usable row draws one Philox word keyed by `(seed, block)`, blocks are ranked by it (ties by block id) and rank `r` goes to fold `r % foldCount`. Two all-pairs passes over the rows. New purpose `spatialCrossValidationBlockOrder` in `src/analysis-core/kernels/philox-random.ts:91` | `test/gpu-spatial-analysis/spatial-validation/spatial-cross-validation-seeds.spec.ts` (all four failed before, with `[2, 2, 2, 2, 0]` blocks per fold) | Every fold assignment changes. Folds now differ by at most one block, no fold is empty when blocks >= folds, and a new seed is a new order. `GPUNestedInterpolationValidation` and `GPUSpatialCrossValidatedOLS` inherit it. Fold numbers of a block are no longer predictable from the block id |
| interpolation-density-03 | fixed | `src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts:611-632` and the system assembly below: ordinary co-kriging keeps the constraint of the target variable and of every secondary variable present in the neighbourhood, and drops the others | `test/gpu-spatial-analysis/spatial-interpolation/local-cokriging-missing-variable.spec.ts` (failed before with status 2; expected values are a float64 ordinary kriging solve) | A neighbourhood without a secondary sample predicts (ordinary kriging of the target) instead of status 2. Without a target sample it is still status 2 |
| interpolation-density-04 | fixed, not observable on the test GPU | `src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.ts:309`: `powerOfMagnitude` returns a NaN argument by a bit test and takes the zero branch only for `magnitude == 0.0` | `test/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.spec.ts` (NaN observation, NaN member, infinite member, `0^p`). It passed before the change too: on this GPU `NaN > 0.0` did not take the zero branch. The defect is real under IEEE comparison rules, which another device may follow | None on the test GPU. On a device with IEEE comparisons, a variogram score with a NaN input is NaN instead of a finite number |
| interpolation-density-05 | fixed | `src/gpu-spatial-analysis/line-density/line-density-kernels.ts:283-287`: `2 cos(south + h/2) sin(h/2)` | `test/gpu-spatial-analysis/line-density/gpu-line-density.spec.ts`, "spherical cell area keeps float32 accuracy" (failed before: 0.59 percent at 0.001 degrees and latitude 60.123, 6.3 percent at 0.0001 degrees and latitude 45, 1.9e-4 at 0.01 degrees) | Spherical `densities` of small cells change by up to several percent, to within 2e-5 of the closed form |
| interpolation-density-06 | fixed (converged on the marginal treatment of `GPUVecchiaFisherScoring`) | New optional prop `emptyNeighborhood: 'insufficient' \| 'prior'` on `GPULocalKriging` (`gpu-local-kriging.ts:210`, kernel at `:782`, simple mode only, default unchanged); `gpu-vecchia-log-likelihood.ts:149` passes `'prior'`; `gpu-vecchia-space-time-log-likelihood.ts:193-202` writes the marginal conditional itself | `test/gpu-spatial-analysis/spatial-interpolation/vecchia-log-likelihood.spec.ts` (also checks the Fisher sibling gives the same value) and `vecchia-space-time-log-likelihood.spec.ts`, both failed before with one failed site; expected values by hand | A site whose conditioning rows are all excluded contributes its marginal term (`conditionalStatus` 0, mean 0, variance `sill + nugget`) instead of failing the whole likelihood. An empty conditioning row now also reports `conditionalStatus` 0 |
| interpolation-density-07 | fixed | `src/gpu-spatial-analysis/point-density/space-time-kernel-density-math.ts:81-97`: the default window of one slice (an instant) is packed; an explicit window still needs `start < end`; `sliceCount` is validated | `test/gpu-spatial-analysis/point-density/space-time-kernel-density.node.spec.ts` and two one-slice cases in `space-time-kernel-density.spec.ts` (both failed before with the packer's throw) | One slice without `timeWindow` no longer throws. With the uniform temporal correction and no window every voxel is NaN (documented); a non-integer or non-positive `sliceCount` now throws by name |
| interpolation-density-08 | fixed | One semivariogram-form definition, `VARIOGRAM_MODEL_SHAPE_WGSL` in `src/dataframe-analysis/pair-statistics/variogram-model.ts:127`, with the small-argument series for the exponential, Gaussian and both Matern shapes; used by `gpu-kriging.ts:612-618` and `variogram-model-fit-nodes.ts:66-69`. The covariance form in `prediction-covariance.ts` keeps its own correlation (it already has the Matern series; `1 - shape` would lose long-lag correlations) and now says so | `test/gpu-spatial-analysis/spatial-interpolation/variogram-model-shape.spec.ts` (all five shapes from 1e-9 of the range, against `Math.expm1`); `kriging-small-lag.spec.ts` (exponential model, measured before: 6e-4 to 3.7e-3 on values spanning 130; after: 2e-5) | `GPUKriging` and `GPUVariogramModelFit` results change in the last digits everywhere and visibly for lags below about 1e-3 of the range. The reviewer's suspicion holds for the smooth models: Gaussian and Matern systems at such lags are rejected by the pivot-ratio test with either form |
| interpolation-density-09 | fixed (words) | `docs/api/gpu-spatial-analysis/operators-density.md:473-481, 531-534`: the PyKrige Gaussian factor `range * 7 / (4 sqrt(3))`, the gstat range parameters, the real tolerances | none (documentation) | none |
| interpolation-density-10 | fixed (words) | Class documentation of `GPULineDensity` (`src/gpu-spatial-analysis/line-density/gpu-line-density.ts:200-208`) and `docs/api/gpu-spatial-analysis/operators-geometry.md:705-708` | `gpu-line-density.spec.ts`, "spherical length of a long segment depends on the grid, as documented" (4,604.5 km in one cell, 5,003.8 km in 1 degree cells) | none |
| interpolation-density-11 | fixed (words) | `gpu-line-density.ts:184-186`, `operators-geometry.md:696` | none | none |
| interpolation-density-12 | fixed for eight of nine gaps | Specs above cover the six gaps tied to findings 01 to 06. `GPULocalRBF` at its default degree: new SciPy fixture `test/oracles/fixtures/interpolation-rbf-default-degree.json` (generator `test/oracles/interpolation-rbf-default-degree.py`, SciPy 1.13.1) and eight cases in `test/gpu-spatial-analysis/spatial-interpolation/interpolation-advanced.spec.ts`, all pass. The fair CRPS and energy score: NumPy 2.0.2 and SciPy 1.13.1 are importable under the system `/usr/bin/python3` (Python 3.9), so they are checked too, see "Review follow-up" (an earlier version of this row said nothing importable gives them, which was wrong; scoringRules itself is an R package and remains unavailable). Left: a spec of `GPUSpatialInteraction` in its own directory (it is covered in `spatial-regression/coverage-*.spec.ts`) | as listed | none |

Extra findings of this pass:

| Item | Outcome | What changed | Spec |
| --- | --- | --- | --- |
| Open-interval uniform of the Box-Muller radius | fixed | `gpu-conditional-gaussian-simulation.ts:627` and `conditional-simulation-blocked.ts:437` use `philoxOpenUnitFloat` | `test/gpu-spatial-analysis/spatial-interpolation/conditional-simulation-open-uniform.spec.ts` (seed 632507 has the largest 24-bit word at replicate 0, component 0; both paths drew exactly 0 before). Every draw changes in its last bits |
| Compensated float sums (`optimal-transport`) | no float compensation in these directories | `GPUSinkhornDivergence` sums through `getRowTileSumNodes` of `spatial-regression/regression-row-tiles.ts`, whose merge is already a pairwise cascade. The stale words "compensated reduction" are corrected in the class documentation and in `docs/api/gpu-spatial-analysis/sinkhorn-divergence.md` | `test/gpu-spatial-analysis/optimal-transport/stm09-sinkhorn-divergence.spec.ts`, "sums a million cells to float32 accuracy": `OT(a, a) = epsilon log(N)` in closed form on 1000 x 1000 equal masses, within 1e-5 (a serial float32 sum is 1 percent off there) |
| `atan2` call sites (`gpu-natural-neighbor.ts:310-311`, `gpu-rose-statistic.ts:323, 377`, `gpu-geographic-distribution.ts:573, 686`) | not reproduced, left as the builtin | The one input that makes a coordinate difference `-0.0` (a segment from `x = +0` to `x = -0`) is binned correctly by `GPURoseStatistic` on the test GPU | `test/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.spec.ts`, last test |
| Blocked Cholesky of `conditional-simulation-blocked.ts` (not verified by the review) | read in full, no defect found | Diagonal block, panel solve, tiled trailing update, forward substitution and moments are a correct right-looking factorization. One suspicion (the absolute tolerance `1e-6 * (sill + nugget)` on the residual pivot of an exact data hit could fail large systems) was probed with 500 to 3,900 observations and four exact hits, exponential and spherical models: status 0 and zero variance at the hits in every run | none added |

Specs of other streams changed because they depended on the old fold rule:
`test/gpu-spatial-analysis/spatial-regression/statistics-upstream.spec.ts` (the blockCV designs
are compared as partitions, fold numbers no longer matter),
`test/gpu-spatial-analysis/spatial-regression/spatial-validation-advanced.spec.ts` (seed 8 gives
the inner partition the nested oracle was computed for, and a new test checks that on the GPU),
`test/oracles/nested-validation-oracle.py` (the partition is now a literal; the fixture is
unchanged), `test/oracles/statistics-blockcv.R` (comment).

Departure from the brief: the shared shape constant of finding 08 lives in
`dataframe-analysis/pair-statistics/variogram-model.ts`, next to the CPU `getVariogramModelShape`
and the Matern range factors, not in `spatial-interpolation`. No file of `dataframe-analysis`
imports from `gpu-spatial-analysis`, and the variogram fit kernel is one of the three copies.

Upstream gaps:

- `@luma.gl/graph/gpu-core` has two cell mappings (`GPUGridIndex` rescales when the bounds
  straddle zero; `GPUGridBinning` and `GPUGridAggregation` do not) and exports neither. Needed:
  one mapping, exported as a public WGSL snippet.
- `GPUGridBinning` publishes counts only and `GPUGridAggregation` sums with float atomics. Needed
  for `GPUPointDensity` to drop its key kernel: a per-row cell key output, or a deterministic
  (fixed-point) sum mode in `GPUGridAggregation`.

Not done here, outside this stream's directories: the lens twin records and website runs of
`spatial-cross-validation`, `spatial-cross-validated-ols`, `nested-interpolation-validation` and
`conditional-gaussian-simulation` hold values recorded with the old fold rule and the old draws.

### Review follow-up

After the gates and the independent review (`fix-review-models.md`, rows fr-models-02, -03, -05,
-07). Paths as above.

| Item | Outcome | What changed | Spec |
| --- | --- | --- | --- |
| `yarn typecheck-tests`, two TS2739 in the Sinkhorn spec | fixed | `test/gpu-spatial-analysis/optimal-transport/stm09-sinkhorn-harness.ts:49-62, 97-98`: `runSinkhornGPU` takes `readonly number[] \| Float32Array` masses and uploads a `Float32Array` as it is (no copy). No assertion changed | A typecheck of the 22 spec and harness files this stream touched (scratch `tsconfig` extending the repository's) reports no error; `stm09-sinkhorn-divergence.spec.ts` 9 passed |
| fr-models-02 | fixed (words) | `docs/api/gpu-spatial-analysis/coverage-statistics.md:369-373`: the designs are compared with blockCV as partitions, blocks are dealt in a seeded Philox order | none |
| fr-models-03 | fixed (words) | `docs/api/gpu-spatial-analysis/interpolation-expansion.md:40` (dropped constraint, the sample-count rule, status 1 or 2 without a target) and `:46` (`emptyNeighborhood`); class documentation `src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts:245-250` now says status 1 or 2 | the status 1 case is asserted in the new co-kriging test below |
| Behaviour change not declared before | declared | Ordinary co-kriging needs as many retained samples as constraints KEPT, not as `variableCount`. Old: a neighbourhood with fewer samples than variables was status 1 even when it held target samples. New: it predicts when it holds at least one target sample and at least as many samples as kept constraints (one target sample alone returns its value) | `local-cokriging-missing-variable.spec.ts:403` (three variables, two target samples: status 0 and the float64 ordinary kriging values; two secondary samples and no target: status 1) |
| fr-models-05 | fixed | New case with three variables, the middle one absent and `targetVariable: 2`, so the kept-constraint table is `[0, 2]` and the target sits in compact column 1. Expected values: a float64 two-variable ordinary co-kriging solve written from the equations, which also differs from the target-only solve by more than 1e-2 | `local-cokriging-missing-variable.spec.ts:344`, passes |
| Folds depend on the mask | kept, documented, pinned | Decision: rank only the blocks that hold a usable row. Ranking every block id whatever its mask would make the fold a function of seed and block set only, but it brings empty folds back: 8 blocks, 4 usable, 4 folds leaves a fold empty for 54 of the 70 equally likely rank subsets. So a mask that removes whole blocks re-deals the rest, a mask that leaves every block one row changes nothing, and nested validation deals its inner folds anew per outer fold (as nested resampling does). Stated in the class documentation (`gpu-spatial-cross-validation.ts:83-91`), on `innerBlocks` (`gpu-nested-interpolation-validation.ts:43-49`) and in `coverage-statistics.md:206-209` | `spatial-cross-validation-seeds.spec.ts:173` (whole blocks masked, 16 seeds, never an empty fold) and `:189` (rows masked without emptying a block: folds unchanged) |
| fr-models-07 | fixed | `guardedAtan2` at `src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts:324` and `spatial-interpolation/gpu-natural-neighbor.ts:315-316`. This supersedes the "left as the builtin" row above for these two sites. `gpu-rose-statistic.ts:378` and `gpu-geographic-distribution.ts:573, 686` take sums or a variance difference, not a coordinate difference, and stay; `urban-morphology/isovists.ts:170` is not in this stream | The signed-zero spec in `gpu-rose-statistic.spec.ts` is kept (it passed before too on this GPU); natural neighbour is covered by the CGAL cases of `interpolation-advanced.spec.ts` |
| Python reference statement | corrected | NumPy 2.0.2 and SciPy 1.13.1 are importable under `/usr/bin/python3` (system Python 3.9); row 12 above is corrected | none |
| Fair CRPS and fair energy score | checked | New generator `test/oracles/scoring-rules-fair.py` and fixture `fixtures/scoring-rules-fair.json`. Two references that do not use the kernel's pairwise sum: the fair CRPS by order statistics (`numpy.sort`) and the fair energy score over distinct pairs (`scipy.spatial.distance.pdist`), per ensemble; and the property that defines fairness, the mean over 512 (CRPS) and 128 (energy) small ensembles of standard normal members against the closed-form score of the distribution (`scipy.stats.norm`, `scipy.special.hyp1f1`), within three standard errors, with the standard estimator outside that band | `test/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-fair.spec.ts`, 2 passed. `docs/api/gpu-spatial-analysis/scoring-rules.md:123-128` names the oracle |
| Function-scope identity table (contract section 11, found by `core-numerics`) | hazard removed, results unchanged | `src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor.ts:170-176, 215`: on the direct path the neighbour slot is the sample row, so `sampleIndices` is no longer declared or filled there and the loop index is used; the triangulation path, which fills the table with real rows, keeps it | `test/utils/function-scope-table-readback.spec.ts`, `interpolation-advanced.spec.ts` (CGAL Sibson cases) and `interpolation-coverage.spec.ts`: 62 passed; four node specs of the kernel text: 50 passed |

Closing pass (2026-11-11): the default-degree RBF and fair-score fixtures were rerun on the pinned set (NumPy 2.5.4, SciPy 1.16.3): the fair scores are equal, the RBF values differ by at most 3.9e-14 relative, both replaced; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).
