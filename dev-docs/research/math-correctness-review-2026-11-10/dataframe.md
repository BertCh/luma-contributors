# Math correctness review: `dataframe-analysis` (2026-11-10)

Scope id `dataframe`. Scope: `modules/analysis/src/dataframe-analysis` (96 files, 29,138 lines).
Paths below are relative to `modules/analysis/src` unless they start with `test/`, `docs/` or
`node_modules/`. Scratch scripts are in the session scratchpad under `dataframe/`
(`pca.mjs`, `group-mean.mjs`, `histogram-edge.mjs`, `histogram-grid.mjs`, `quantile-h.mjs`,
`cressie.py`). No spec file was run. No repository file other than this report was changed.

A note on the brief. The reference list for this scope names Pearson, Spearman, Kendall, partial
correlation, mutual information, distance correlation, pair regression, t and F statistics, Cramér's V,
geometric-interval and pretty breaks, and quantile sketches. None of these exist in
`dataframe-analysis`. `pair-statistics` is spatial pair statistics (variograms, Moran correlogram,
Ripley K, G, F, J, Clark-Evans, quadrats, space-time K, Diggle test), not column correlation.
The review covers what is there.

## 1. Coverage

Read in full (kernel bodies, packers and class documentation; import blocks skipped):

- `pair-statistics`: `pair-statistics-grid.ts`, `pair-histogram.ts`, `pair-statistics-fixed-point-unit.ts`,
  `pair-statistics-cell-neighbors.ts`, `gpu-variogram.ts`, `variogram-parameters.ts`,
  `gpu-spatial-correlogram.ts`, `gpu-ripley.ts`, `gpu-ripley-envelope.ts`,
  `gpu-ripley-distance-functions.ts`, `gpu-point-pattern-indices.ts`, `gpu-space-time-variogram.ts`,
  `gpu-cross-variogram.ts`, `gpu-marked-point-statistics.ts`, `space-time-k-function-nodes.ts`,
  `space-time-k-function-wgsl.ts`, `gpu-space-time-diggle-test.ts`, `variogram-model.ts`,
  `variogram-model-fit-nodes.ts`, `variogram-model-fit-cooperative.ts`, `gpu-variogram-model-fit.ts`,
  `space-time-variogram-model.ts`, and the five parameter packers (`ripley-parameters.ts`,
  `ripley-distance-parameters.ts`, `point-pattern-indices-parameters.ts`,
  `spatial-correlogram-parameters.ts`, `pair-statistics-parameters.ts`).
- `column-classification`: `class-breaks-kernels.ts`, `gpu-class-breaks.ts`,
  `exact-natural-breaks-nodes.ts`, `column-quantiles-nodes.ts`, `order-statistic-wgsl.ts`,
  `color-scale-wgsl.ts`, `scalar-scale-wgsl.ts`.
- `group-statistics`: `group-statistics-moments.ts`, `group-statistics-order.ts`.
- `composite-indicators`: `gpu-composite-score.ts`, `inequality-wgsl.ts`.
- `variable-standardization/gpu-variable-standardization.ts`.
- `column-profile/column-profile-kernels.ts`.
- `calendar-buckets`: `gpu-calendar-buckets.ts`, `calendar-buckets-parameters.ts`,
  `time-zone-transitions.ts`.
- `period-index`: `gpu-period-index.ts`, `period-index-calendar.ts`.
- `time-window-filter/time-words.ts`, `temporal-reduction/temporal-reduction-word-division.ts`,
  `value-cube/cell-period-key-packing.ts`.

Read in part:

- `pair-statistics/gpu-space-time-k-function.ts` (class documentation and constructor, 95-357).
- `group-statistics/gpu-group-statistics.ts` (types and documentation, 30-215).
- `composite-indicators/gpu-inequality.ts` (documentation, 60-200).
- `column-profile/gpu-column-profile.ts` (types and documentation).
- `time-window-filter/time-window-classify-node.ts` (both classify kernels),
  `time-window-sorted-nodes.ts` (range search, 60-230).
- `temporal-reduction/gpu-temporal-reduction.ts` (documentation and the two bucket bodies, 676-727;
  the extreme and first/last tie kernels, 300-670, were not read),
  `temporal-reduction-sorted-range.ts` (40-136).
- `key-join/gpu-key-join.ts` (documentation 40-200, probe and left-row kernels 540-640; the sort,
  segment and aggregate kernels were not read).
- `value-cube/gpu-value-cube.ts` (documentation and kernel bodies), `gpu-cell-period-keys.ts`
  (kernel body).
- `column-classification/gpu-color-scale.ts` (grep of the class and ramp arithmetic only),
  `monotone-natural-breaks-nodes.ts` (header only).

Not opened: `column-classification/natural-breaks-nodes.ts` (1,258 lines, the approximate
Fisher-Jenks), `gpu-bivariate-classification.ts`, `bivariate-classification-parameters.ts`,
`gpu-scalar-scale.ts`, `scalar-scale-parameters.ts`, `color-scale-parameters.ts`,
`class-breaks-parameters.ts`, `column-quantiles-parameters.ts`, `gpu-column-quantiles.ts`,
`column-classification-shared.ts`, `group-statistics-sort.ts`, `group-statistics-common.ts`,
`composite-score-parameters.ts`, `inequality-parameters.ts`, `column-profile-parameters.ts`,
`gpu-time-window-filter.ts`, `time-window-parameters.ts`, `temporal-reduction-parameters.ts`,
`cube-layout/gpu-cube-layout-transpose.ts`, `time-words-relative/gpu-time-words-relative.ts`,
`pair-statistics/space-time-k-function` parameter validation beyond the packer, and every `index.ts`.

Official luma.gl read for comparison: `node_modules/@luma.gl/graph/dist/gpu-core/gpu-histogram.js`
(float and integer bin rule), `gpu-group-aggregation.js` (accumulate and finalize) and the
`gpu-group-aggregation`, `gpu-reduction`, `gpu-dataframe` type declarations. Not read: the official
hash join and hash index, sort, scan, `gpu-timeseries`, `gpu-crossfilter`, and the dataframe query
compilers.

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dataframe-01 | wrong-result | CONFIRMED (script `pca.mjs`, f32 replica against a Jacobi eigen solve) | `composite-indicators/gpu-composite-score.ts:691-713` | First principal component by 64 power-iteration steps started from `1 / sqrt(d)` in every component | The eigenvector of the largest eigenvalue of the correlation matrix | Two indicators with correlation -0.958 and both directions +1: eigenvalue 0.0418, explained ratio 0.021, loadings (0.707, 0.707). Correct: 1.958, 0.979, (0.707, -0.707). Any two indicators with negative correlation give the minor component | Use the shared Jacobi solver in `analysis-core/kernels/symmetric-eigen-wgsl.ts`, or start from a vector that is not an eigenvector of a 2 x 2 correlation matrix and check the residual |
| dataframe-02 | wrong-edge-case | CONFIRMED (script `group-mean.mjs`) | `group-statistics/group-statistics-moments.ts:508` | Group z-score `(v - mean0) / sd`, where `mean0` is the fixed-point mean and `sd` uses the corrected second moment | `(v - mean) / sd` with the same mean in both places | Values 1.0e-4, 1.2e-4, 1.4e-4, default `sumScale`: z = -1.10, -0.10, 0.90 (true -1, 0, 1). Values 3e-6 to 6e-6: z = 2.3 to 4.6 (true -1.16 to 1.16) | Subtract `mean0 + S1 / n` (the `delta` the variance already uses) |
| dataframe-03 | precision | CONFIRMED (script `group-mean.mjs`; the docs state the 2^-17 error) | `group-statistics/group-statistics-moments.ts:218,264,304-306`; `analysis-core/cell-key-wgsl.ts:53-60`; same scheme in `key-join/gpu-key-join.ts` | Sum and mean of `round(v * 65536) / 65536` per value | Sum and mean of the values | Values 1.0e-4, 1.2e-4, 1.4e-4: mean 1.2207e-4 (+1.7 %). Values below 7.6e-6 sum to 0 | Derive the unit from the data as `GPUValueCube` does (96-bit accumulator), or at least derive `sumScale` from the column maximum; state the relative error for small columns in the property documentation |
| dataframe-04 | wrong-edge-case | CONFIRMED (scripts `histogram-edge.mjs`, `histogram-grid.mjs`) | `column-profile/column-profile-kernels.ts:137-147,489` (same rule in `column-classification/class-breaks-kernels.ts:405-415`) | Bin `b` holds `fround(b * w) <= v - lo < fround((b + 1) * w)` with `w = fround((hi - lo) * fround(1 / B))` | Equal-width bins `[lo + k (hi - lo) / B, ...)`, as numpy and the official `GPUHistogram` (`floor((v - lo) / (hi - lo) * B)`) | Domain [-3.5, 12.25], 30 bins, value -0.875 (exactly edge 5): bin 4. Exact arithmetic and the official rule: bin 5. Data on a 0.125 grid over that domain: 10 of 30 counts differ from the official rule | Use the official rule (or delegate to `GPUHistogram`); if adapter-independent rounding is wanted, test `(v - lo) * B` against `k * (hi - lo)` so that no rounded width accumulates |
| dataframe-05 | doc-mismatch | CONFIRMED (script `cressie.py` on the committed scikit-gstat fixture) | `pair-statistics/gpu-variogram.ts:488` (documented at `:181`) | Cressie-Hawkins with denominator `2 (0.457 + 0.494 / N)` | scikit-gstat 1.0.24 (the cited reference and the fixture source) uses `2 (0.457 + 0.494 / N + 0.045 / N^2)` | A bin with N = 1 pair: kernel is 4.7 % above scikit-gstat; N = 2: 1.6 %; N = 5: 0.3 %; N = 10: 0.09 % | Add the `0.045 / N^2` term, or keep the two-term form and say in the documentation that it departs from scikit-gstat and gstat in sparse bins |
| dataframe-06 | inconsistent | CONFIRMED (by reading; the class documentation states it) | `pair-statistics/gpu-marked-point-statistics.ts:249` against `pair-statistics/gpu-ripley.ts:455-457` | Untyped or same-type K normalised by `A / (n n)` | `GPURipley` and spatstat `Kest` use `A / (n (n - 1))` | 4 points, no types, no intensities, `edgeCorrection: 'none'`: marked K = 0.75 x `GPURipley` `'none'` K for the same pattern | Use `n (n - 1)` when focus and neighbour sets are the same; keep `n_i n_j` for cross-type |
| dataframe-07 | doc-mismatch | CONFIRMED (by reading) | `pair-statistics/gpu-point-pattern-indices.ts:110,165-187`, documentation at `:315-317` | Chi-square p-value from the exact incomplete gamma at every degree of freedom | Documentation says Wilson-Hilferty above 100 degrees of freedom, absolute error below 2e-4 | The Wilson-Hilferty branch is dead: with at most 101 quadrats df is at most 100, with more the stable-gamma variant has no such branch | Correct the documentation; delete the dead branch |
| dataframe-08 | precision | CONFIRMED (script `quantile-h.mjs`; the policy text states the f32 product) | `column-classification/column-quantiles-nodes.ts:166-168`; `group-statistics/group-statistics-order.ts` `getQuantile` | Quantile rank `h = fround(f32(n - 1) * p)` | numpy computes `h` in float64 | n = 16,777,216, p = 0.95: kernel fraction 0, numpy fraction 0.25. The interpolation fraction is lost above about 2^23 rows, and `p` itself is f32 | Compute `floor(h)` and the fraction with an integer multiply of `n - 1` by a fixed-point `p`, or accept `p` as a rational |
| dataframe-09 | precision | SUSPECTED (by reading; not replicated) | `pair-statistics/variogram-model-fit-nodes.ts:82-110` | Range-ladder candidates use Cressie weights `N / max(fitted^2, 1e-20)` without normalisation | The cooperative refinement (`variogram-model-fit-cooperative.ts:31-43`) divides by the total weight to avoid overflow | Semivariances near 1e-7 with 1e4 pairs per bin: `weightSum * squareSum` exceeds f32, the determinant is NaN, the joint nugget and sill fit is skipped for every ladder step, so the bracket handed to refinement can be the wrong one | Normalise the weights in `fitLinear` the same way |
| dataframe-10 | untested | SUSPECTED | `pair-statistics/gpu-space-time-diggle-test.ts:612-664` | `U = sum (D - m) / s` with `m`, `s` the permutation mean and sample standard deviation of `D` per class | Diggle et al. (1995) and splancs `stmctest` standardise `D` by an analytic standard error (`stsecal`) and do not centre | Not derived. The class documentation says the splancs definition was not verified. The spec compares with a binary64 copy of the same definition | Generate a splancs fixture, or rename the statistic so that it does not claim the splancs definition |

### dataframe-01. Power iteration returns the minor component

`PRINCIPAL_COMPONENT_BODY` starts at `v = (1, ..., 1) / sqrt(d)`. For two indicators the correlation
matrix is `[[1, r], [r, 1]]`. Its eigenvectors are always `(1, 1)` and `(1, -1)`, with eigenvalues
`1 + r` and `1 - r`. The start vector is exactly the first one. Both rows of the product evaluate
`fround(v + r v)`, so the two components stay bit-identical forever and the iteration never leaves
`(1, 1)`. When `r < 0` that is the smaller eigenvalue. The kernel then reports loadings
`(0.707, 0.707)`, eigenvalue `1 + r` and explained ratio `(1 + r) / 2`, and the
`'principal-component'` score is the sum of the two z-scores, which is close to noise for
anti-correlated indicators. With r = -0.958 the reported explained ratio is 2.1 %, the true one 97.9 %.

For more than two indicators the same thing can happen whenever the start vector is an exact
eigenvector, but f32 rounding usually breaks the symmetry (a 4-indicator block case in `pca.mjs`
converged). Slow convergence when the two largest eigenvalues are close is reported through the
`residual` slot, which is fine.

Tests do not catch it. `test/gpu-dataframe/composite-indicators/gpu-composite-score.node.spec.ts:209-250`
checks the CPU twin against a Jacobi solve on four positively aligned indicators. The parity
fixtures use real indicators that are positively correlated after the direction flags. The CPU twin
(`composite-score-oracle.ts:160-205`) copies the same power iteration.

The documentation (`gpu-composite-score.ts:129-134`) describes the start vector but not this limit.
A user who leaves both directions at +1 for income and poverty gets the wrong component.

### dataframe-02 and dataframe-03. Fixed-point mean in `GPUGroupStatistics`

`cellScaleValue` rounds every value to a multiple of `1 / sumScale` (default 65536) before the
exact integer sum. That is exact for values of magnitude 256 and above, and loses relative precision
below about 1e-2. The user documentation says so in one clause ("the fixed-point mean carries 2^-17
absolute error", `docs/api/gpu-dataframe-analysis.md:712`), but the same paragraph also quotes "about
1e-7 for mean" from a mixed data set, and the property documentation only says "Fixed-point scale of
the sums and means".

The variance, skewness and kurtosis are safe: they use shifted power sums and correct for
`delta = S1 / n`. The z-score is not: it divides the uncorrected `v - mean0` by the corrected
standard deviation, so its error is `delta / sd`, which reaches order one for columns whose
standard deviation is near 1e-5. This is a one-line fix and is separate from the choice of scale.

Three sum schemes now exist for the same job: the 64-bit caller-scale one here and in `GPUKeyJoin`,
the 96-bit data-derived accumulator in `GPUValueCube`, and the pair-histogram unit. The official
`GPUGroupAggregation` uses f32 compare-exchange addition (section 5).

### dataframe-04. Histogram bin edges

The local `getBinIndex` was written so that a CPU oracle with `Math.fround` reproduces it bit for
bit. It does that. But the edges it defines are `fround(k * w)` with a width that has been rounded
twice (`fround(1 / B)`, then the product with the range). For a bin count that is not a power of
two the rounded width is off by up to an ulp and the error grows with `k`. A value that sits exactly
on a mathematical edge, which is common for data on a regular grid, then falls one bin low (or
high). `histogram-grid.mjs` bins the values -3.5, -3.375, ..., 12.25 into 30 bins of their own
extent: the local rule gives `5,4,4,4,5,4,...`, the official rule gives `5,4,4,4,4,5,...`.

The independent check does not see it. `test/oracles/parity-wp6-column-analysis.py:173,275-276`
compares `GPUColumnProfile` histograms with `numpy.histogram` at 64 bins only, where `1 / B` is
exact. Other bin counts are pinned by `column-profile-oracle.ts`, a copy of the same formula.

### dataframe-05. Cressie-Hawkins constant

`cressie.py` recomputes the robust estimator from the positions and values stored in
`test/oracles/fixtures/dataframe-variogram.json` with both denominators. The three-term form
matches the stored scikit-gstat values to 1e-16. The two-term form of the kernel differs by 5.5e-8
at N = 1,341, the smallest bin of the fixture, which is inside the spec tolerance (2^-19 relative).
So the fixture cannot detect the difference, and the CPU oracle (`test/gpu-dataframe/pair-statistics/variogram-oracle.ts:82`)
copies the two-term form. The two-term form is what Cressie (1993) prints, so the kernel is
defensible; the class documentation cites gstat and scikit-gstat without saying it departs from
them. Sparse short-lag bins are where the robust estimator is used.

## 3. Verified correct

Each line names what was compared. "By hand" means the WGSL was checked against the published
formula while reading.

- `GPUVariogram` (Matheron): `sum (z_i - z_j)^2 / (2 N)`, unordered pairs once, mean distance,
  population variance in `statistics`. By hand; also the scikit-gstat fixture. Lag rule
  `min(floor(d / max * L), L - 1)` is documented and differs from scikit-gstat only at `d = max`.
- Pair grid and histogram (`pair-statistics-grid.ts`, `pair-histogram.ts`): cell width at least
  the search radius, forward half stencil visits each unordered pair once, 64-bit carries, inclusive
  `d <= maximumDistance`. By hand.
- `GPUSpatialCorrelogram`: Moran's I, `E[I] = -1 / (n - 1)`, normality and randomisation variances,
  `S1 = 2 S0`, `S2 = 4 sum k_i^2`, kurtosis `n sum c^4 / (sum c^2)^2`, two-sided normal p-value.
  By hand against the esda `Moran` formulas.
- `GPURipley`: `'none'` `A / (n (n - 1))`; border numerator and denominator by difference arrays,
  `lambda = n / A` as spatstat; rectangle isotropic weight (side arcs minus adjacent-corner overlap,
  the overlap condition is equivalent to the corner lying inside the circle); `L`, `L - r`, annulus
  pair correlation; closed radius classes `d <= r`. By hand.
- `GPURipleyEnvelope`: binomial CSR draws on independent Philox words, pointwise rank envelope,
  global envelope on the L scale, `p = (1 + #{T_j >= T_obs}) / (s + 1)`. By hand.
- `GPURipleyDistanceFunctions`: G and F for `'none'`, reduced sample and Kaplan-Meier (events
  before censoring within a slot), `J = (1 - G) / (1 - F)` from survival values. By hand. The
  Hanisch normalisation is a documented approximation (see section 7).
- `GPUPointPatternIndices`: Clark-Evans expectation and standard error (plain and Donnelly, guard
  variant), one-sided and two-sided normal p-values, exact ring nearest-neighbour search, quadrat
  variance-to-mean ratio and chi-square, polygon expected counts by area. By hand.
- `GPUSpaceTimeVariogram`, `GPUCrossVariogram`: signed cross term `(x_i - x_j)(y_i - y_j) / 2`
  split into positive and negative sums, zero-lag classes, complete-case rule. By hand.
- `GPUMarkedPointStatistics`: translation weight `A / ((W - |dx|)(H - |dy|))`, inhomogeneous
  normalisation `sum 1 / (lambda_i lambda_j overlap)`, annular pair correlation, product mark
  correlation. By hand (normalisation convention: dataframe-06).
- `GPUSpaceTimeKFunction`: homogeneous and inhomogeneous K, spatial and temporal marginals, the
  one-dimensional isotropic weight 1 or 2, temporal marginal by binary search
  (`2 N(v) - N(min(v, radius))`), `D = K - K_s K_t`, pair correlation `1 / (4 pi u)` with box and
  Epanechnikov kernels, LISTA scale. By hand against Gabriel and Diggle (2009). Not checked against
  stpp output (none is available; the documentation says so).
- `fitVariogramModel`, `GPUVariogramModelFit`: spherical, exponential, Gaussian and Matern 3/2 and
  5/2 shapes with the practical-range factors (4.7490 and 5.9245 re-derived by hand), non-negative
  two-parameter least squares, Cressie weights `N / gamma^2`, AIC `n ln(RSS / n) + 6`.
- `space-time-variogram-model.ts`: separable, product-sum, metric, sum-metric (gstat `vgmST`
  definitions) and the Gneiting class with both range calibrations; product-sum fit algebra
  (`k = C / (A B)`). By hand.
- `GPUClassBreaks`: equal interval, quantile (R-7), standard deviation (sample, as the mapclassify
  oracle uses `ddof=1`), head/tail with the 40 % rule, box plot, maximum breaks, exact Fisher-Jenks
  dynamic program (duplicates never split, backtrack, class count capped by distinct values),
  the integer class cost `S2 - S1^2 / n` with recentring, Chan moment merge with a common origin.
  By hand. The maximum always lands in the last class.
- `getColumnQuantileNodes` and `ORDER_STATISTIC_WGSL`: lower, higher, nearest (half to even),
  linear and midpoint ranks; radix select; overflow-safe interpolation. By hand (precision limit:
  dataframe-08).
- `GPUGroupStatistics`: central moments from shifted power sums (`m2`, `m3`, `m4` identities
  re-derived), sample or population variance, population `g1` and excess `g2` as documented,
  group quantiles, mode with ties to the smallest value, unique counts. By hand.
- `GPUCompositeScore` scalers and aggregations: min-max, population z-score, average-rank
  percentile `r / (n - 1)`, weighted sum over `sum |w|`, weighted geometric mean with NaN for
  non-positive terms. By hand. (Principal component: dataframe-01.)
- `GPUInequality`: Lorenz-trapezoid Gini (equal to the PySAL formula, shown algebraically), Theil T
  and L, Atkinson including epsilon = 1, Hoover, Palma, Theil between and within decomposition.
  By hand.
- `GPUVariableStandardization`: demean, z-score (sample by default), mean-absolute-deviation
  scaling, range adjust and range standardise; the origin and radius normalisation cancels
  correctly in every method. By hand against the GeoDa definitions.
- `GPUColumnProfile`: Welford and Chan moments with per-lane origins, population and sample
  variance, HyperLogLog (register index, rank, alpha, small-range correction and its series).
  By hand.
- `GPUCalendarBuckets`: floor division of negative Int64 times, unit conversion with floor,
  `civil_from_days`, day of year, ISO week and week-year from the Thursday, weekday with
  `firstDayOfWeek`, transition-table upper bound. By hand against Hinnant's algorithms.
- `GPUPeriodIndex`: `days_from_civil`, step indices, fixed-width periods by exact 64-bit division.
  By hand. A local-calendar `'hour'` step merges the repeated hour and leaves the skipped hour empty
  on daylight-saving days; that follows from the definition and is not stated in the documentation.
- `divideTimeDifference`, `TIME_WORDS_WGSL`: 64 by 64 bit division, signed compare, borrow
  subtraction. By hand.
- `GPUTimeWindowFilter`: closed bounds on both ends, interval overlap rule, sorted lower and upper
  bound consistent with the classify kernel. By hand.
- `GPUTemporalReduction`: bucket `floor((t - origin) / width)` with product correction in f32 and
  exact division in word mode; negative offsets dropped, not truncated to bucket 0. By hand.
- `GPUKeyJoin` probe: exact key equality after a lower-bound search on sorted unique keys (no hash,
  so no collision case), all-ones key never matches, first match is the smallest right row.
- `cell-period-key-packing.ts`: free-bit counts for H3, Quadbin and S2. By hand.
- `GPUValueCube`: bin index, counts, shared 96-bit accumulator, mean as rounded sum over count.

## 4. Standards consistency

- **Class interval closure.** `GPUClassBreaks`, `GPUColorScale` and `GPUBivariateClassification`
  put a value equal to an inner edge in the upper class (d3 `bisectRight`). mapclassify, the cited
  reference for the edges, puts it in the lower class. Values 1, 2, 2, 3 with two quantile classes:
  edge 2, counts here [1, 3], mapclassify [3, 1]. The documentation states the d3 rule
  (`docs/api/gpu-dataframe-analysis.md:270`), so this is a convention, not a defect. Natural breaks
  is consistent with it (edges are the smallest value of the upper class).
- **Lag and radius class closure.** Variograms use left-closed lag bins `floor(d / max * L)`.
  Ripley, the distance functions and the space-time K use right-closed classes `ceil(...) - 1`.
  Both are documented and each is the right choice for its statistic. Zero lag differs across the
  three variograms: `GPUVariogram` and `GPUCrossVariogram` put coincident pairs in bin 0,
  `GPUSpaceTimeVariogram` has a separate zero class.
- **Bin-edge arithmetic.** Two rules: `floor(ratio * B)` (variogram lags, quadrats, correlogram
  bands, the official histogram) and rounded-width products (column profile, class-breaks coarse
  histogram, temporal reduction). See dataframe-04.
- **K normalisation.** `n (n - 1)` in `GPURipley` and the homogeneous space-time K, `n n` in
  `GPUMarkedPointStatistics` and the inhomogeneous space-time K. The space-time case is correct
  (intensity-based). The marked case is dataframe-06.
- **Variance convention.** Group statistics: sample by default, selectable; skewness and kurtosis
  are always the population forms and do not follow the `variance` prop. Class breaks standard
  deviation: sample. Variable standardisation: sample by default. Composite z-score: population.
  Column profile: both. Variogram and correlogram `statistics`: population. Each is documented; the
  library has no single rule. (Also noted in the library review, 4.3.)
- **Chi-square p-value, private copy.** `gpu-point-pattern-indices.ts:82-187` carries its own
  incomplete gamma (series and Lentz fraction, with a centred Stirling prefactor for large shape).
  The shared `analysis-core/kernels/incomplete-gamma-wgsl.ts` switches to Wilson-Hilferty above
  2,000 degrees of freedom and has no lower tail. The two differ numerically above 2,000 degrees of
  freedom (the private one stays exact, the shared one is good to 2e-4). The private one is the
  better kernel; the shared one should take its prefactor and the lower tail.
- **Eigen solve, private copy.** `gpu-composite-score.ts:668-745` has its own power iteration while
  `analysis-core/kernels/symmetric-eigen-wgsl.ts` exists. They do not agree (dataframe-01).
- **Fixed-point sums.** Three schemes: caller `sumScale` (group statistics, key join), 96-bit
  data-derived accumulator (value cube), pair-histogram unit from the robust variance. Already in
  the library review (3.8); dataframe-03 is the numerical consequence.
- **64-bit helpers.** `gpu-ripley.ts:416-427` (`add64`, `subtract64`), `gpu-spatial-correlogram.ts:491-506`,
  `class-breaks-kernels.ts:478-523` and `time-words.ts` each write their own add, subtract and
  to-float. `analysis-core/kernels/uint64-wgsl.ts` is not used in this directory. The copies read
  as numerically the same.
- **Binary search.** Private lower and upper bounds in the composite rank, inequality zone search,
  time-window range, temporal-reduction range, space-time temporal marginal, class counts and key
  join. `analysis-core/kernels/binary-search-wgsl.ts` is not used here. No disagreement found.
- **Isotropic edge weight.** `gpu-ripley.ts:316-336` and `space-time-k-function-wgsl.ts:36-56` are
  two copies of the same arc formula, identical term for term. The weight cap is 100 in both.
- **Shader literals.** `PI` is written as `3.14159265358979` in four files and as
  `3.141592653589793` in `gpu-marked-point-statistics.ts:250`; all round to the same f32.
  `0.26136` (Clark-Evans), `0.0514`, `0.041`, `0.0703`, `0.037` (Donnelly), `0.457`, `0.494`
  (Cressie-Hawkins), `1.3489795` and the HyperLogLog alphas are literals; only the last two have
  named constants.
- **Coordinates.** Pair statistics take f32 planar positions as given and subtract neighbours
  directly. There is no local origin. Positions in projected metres near 1e7 carry 1 m of f32
  rounding before any kernel runs; the class documentation does not warn about it.
- **Still present from the library review (3.2).** `GPUInequality` walks a zone with plain f32
  running sums and has no row cap (`inequality-wgsl.ts:109-124`); past 2^24 unit-weight rows in one
  zone `totalWeight` stops growing. The class documentation describes the precision loss but not
  this hard point.

## 5. luma.gl official implementations

- **Delegates.** `getPairStatisticsInputNodes` builds its cell index with the official
  `GPUGroupAggregation` (count), `GPUScan` and `GPUSort`. `GPUVariableStandardization` reduces with
  the official `GPUReduction` (`sum`, `extent`). Class breaks, quantile ranks, group statistics,
  inequality, composite ranks and the space-time K sort with `GPUSort` and scan with `GPUScan`.
  `GPUTemporalReduction` compacts with `GPUVisibilityWorkflow`. No argument that changes meaning
  was found in these calls.
- **Group aggregates re-implemented.** `GPUGroupStatistics` computes count, sum, mean, minimum and
  maximum itself; the official `GPUGroupAggregation` publishes the same five for dense `uint32`
  keys. Both exclude non-finite values and accept a nonzero mask. They differ in the sum: the
  official one adds f32 values with an atomic compare-exchange loop, so the result depends on thread
  order and loses relative accuracy with length; the local one is exact and reproducible but
  quantises each value to `1 / sumScale` (dataframe-03). For columns of ordinary magnitude the local
  one is the more accurate. For columns below about 1e-2 the official one is. Neither is right for
  both; a data-derived unit would be. Variance, skewness, quantiles, mode and 64-bit keys are not
  published upstream; they are upstream proposals, not copies to remove.
- **Histogram re-implemented.** `GPUColumnProfile` (and the class-breaks coarse histogram) does not
  use the official `GPUHistogram`. Both count finite values inside an inclusive domain and put the
  maximum in the last bin. The edge rule differs (dataframe-04): the official
  `min(u32((v - lo) / (hi - lo) * B), B - 1)` agrees with exact arithmetic on edge values, the local
  rule does not for bin counts that are not powers of two. The official division is not correctly
  rounded on every adapter, which is why the local rule exists; the better fix is a
  multiplication-only test, proposed upstream. Recommendation: delegate to the public
  `GPUHistogram` from `@luma.gl/graph/gpu-core` and raise adapter independence upstream.
- **Join re-implemented.** `GPUKeyJoin` is a sort and binary-search join with exact key compare,
  1:n aggregates and first-match gathers. The official `GPUHashJoin` and `GPUBatchHashJoin` were not
  read, so collision handling and null-key rules were not compared. The local null rule (all-ones
  key never matches) is the SQL rule, not the pandas rule (pandas matches null keys to each other).
- **Quantiles.** No official quantile exists. `getColumnQuantileNodes` is a candidate upstream
  proposal.
- **Time series and crossfilter.** `gpu-timeseries` and `gpu-crossfilter` were not read. The prior
  review says the shapes differ (per-cell irregular buckets, Int64 words, interval rows).
- **WGSL conventions.** The official kernels test finiteness with
  `value == value && abs(value) <= 3.402823466e+38`; the local kernels test the exponent bits.
  The bit test is the safer one under fast-math. No other convention difference was found.

## 6. Test oracles

Independent oracle (a reference library, values stored in `test/oracles/fixtures`):

- `GPUVariogram`: scikit-gstat (`dataframe-variogram.json`). Bins have 1,341 pairs or more, so the
  Cressie-Hawkins constant is not pinned (dataframe-05).
- `GPUClassBreaks`: mapclassify and jenkspy (`dataframe-class-breaks.json`,
  `distribution-mapclassify-fit.json`), including `StdMean` with `ddof=1`.
- `GPUColumnQuantiles`, `GPUColumnProfile`, `GPUGroupStatistics`, `GPUKeyJoin`,
  `GPUTemporalReduction`, `GPUTimeWindowFilter`: numpy, pandas and scipy parity fixtures
  (`parity-wp6-*`). Column-profile histograms only at 64 bins (dataframe-04). No fixture has a
  small-magnitude column (dataframe-02, dataframe-03).
- `GPUCompositeScore`: numpy, scipy and scikit-learn parity fixtures for min-max, z-score, rank and
  the principal component on positively aligned real indicators; a Jacobi check of the CPU twin on
  one four-indicator case. No negatively correlated pair (dataframe-01).
- `GPUInequality`: the PySAL `inequality` package.
- `GPUSpatialCorrelogram`: esda and libpysal.
- `GPURipley`, `GPURipleyDistanceFunctions`, `GPUPointPatternIndices`: pointpats, scipy and a
  spatstat generator (`parity-wp6-ripley.py`, `parity-wp6-point-patterns.py`,
  `parity-wp6-round8-points.py`, `parity-wp6-round8-spatstat.mjs`), and scipy `chi2` for the
  p-values.
- `getGPUTimeZoneTransitionTable`: Python `zoneinfo` (`dataframe-time-zones.json`).

Copy of the same formula only (a TypeScript oracle that restates the kernel's definition):

- `GPUSpaceTimeKFunction`, `GPUSpaceTimeDiggleTest`: binary64 brute force of the documented
  definitions (`space-time-k-function-reference.ts`). stpp and splancs fixtures are pending by the
  code's own statement.
- `GPUVariogramModelFit`: the CPU `fitVariogramModel`, which shares the algorithm.
- `fitSpaceTimeVariogramModel`: self-consistency specs.
- `GPURipleyEnvelope`: CPU replay of the same Philox draws.
- `GPUCalendarBuckets`, `GPUPeriodIndex`, `GPUValueCube`, `GPUCellPeriodKeys`,
  `GPUVariableStandardization`: TypeScript oracles in the spec folders. The calendar one can be
  trusted in practice because it can be compared with `Date`; I did not check that it is.
- Column-profile histograms at bin counts other than 64; the variogram robust estimator in sparse
  bins.

Not located in `test/gpu-dataframe`: specs for `GPUCrossVariogram` and
`GPUMarkedPointStatistics`. They are referenced from
`test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts`, which I did not open.

## 7. Open points for a second opinion

- Hanisch G and F (`gpu-ripley-distance-functions.ts:668-681`): the denominator is the total
  weight of uncensored points plus a term for points with no neighbour inside `maximumDistance`.
  spatstat's `han` normalises by the cumulative value at `rmax` instead. The class documentation
  calls the result "biased slightly high"; the extra term can also bias it low when those points
  are in fact censored. I did not derive a counterexample.
- Diggle test statistic (dataframe-10).
- Directional variogram (`gpu-variogram.ts:414`): `atan2(0, 0)` for coincident points decides the
  sector. I did not check what WGSL guarantees for that call; if it is indeterminate, the sector
  counts of coincident pairs are not reproducible across adapters while the class documentation
  promises bitwise reproducibility.
- The approximate and monotone Fisher-Jenks files were not read. Their correctness rests on the
  mapclassify and jenkspy fixture (34 cases) alone in this review.

## Resolution (2026-11-10)

Paths are relative to `modules/analysis`. "Headless" specs ran on the test GPU. Every `fixed` row
was first reproduced by its spec failing on the old kernel, except where the row says otherwise.

| ID | Outcome | What changed | Spec that pins it | Behaviour change a caller sees |
| --- | --- | --- | --- | --- |
| dataframe-01 | fixed (the multi-start power iteration described here was replaced after review: see fr-statistics-02 under "Review follow-up") | `src/dataframe-analysis/composite-indicators/gpu-composite-score.ts:714` (power iteration from `d + 1` starts, the unit vector and every axis; the largest Rayleigh quotient is kept) and `:769` (sign rule that does not depend on rounding when the loadings sum to zero); three named tolerances in `composite-score-parameters.ts`. The CPU twin `test/gpu-dataframe/composite-indicators/composite-score-oracle.ts` copied the old iteration and was changed with it. | `test/gpu-dataframe/composite-indicators/gpu-composite-score.spec.ts:259` (headless; an exact design with r = -0.1: old eigenvalue 0.9, now 1.1) and `gpu-composite-score.node.spec.ts:253` (twin against closed form and a Jacobi solve) | Two negatively correlated indicators: eigenvalue `1 + r` becomes `1 - r`, loadings `(0.707, 0.707)` become `(0.707, -0.707)`, and the score changes with them. When loadings sum to zero the first of the tied largest loadings is positive. Inputs that already converged to the major component are unchanged to rounding. On the GPU the old kernel was wrong for weak negative correlation and right by accident for strong (fused multiply-add broke the symmetry). |
| dataframe-02 | fixed | `src/dataframe-analysis/group-statistics/group-statistics-moments.ts:634` (z-score centred on the shift mean plus `S1 / n`, the mean the variance uses). The oracle `test/gpu-dataframe/group-statistics/group-statistics-oracle.ts` centred on the quantised mean and was changed. | `test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts:812` (headless) | Values 1.0e-4, 1.2e-4, 1.4e-4: z-scores -1.10, -0.10, 0.90 become -1, 0, 1. |
| dataframe-03 | fixed | `group-statistics-moments.ts:161` and `:425`: `sumValues`, `means` and the shift mean come from the shared 96-bit fixed-point accumulator with a unit set by the largest value in a kept group; the 64-bit `1 / sumScale` words are now accumulated only for `output.sums`. `src/dataframe-analysis/key-join/gpu-key-join.ts:683`, `:697`, `:839`: the same for the `'sum'` and `'mean'` aggregates (words split over two views so that the 2^24-row binding limit is unchanged). This stream made no edit inside `analysis-core/cell-key-wgsl.ts` (another stream added `getGPUCellSumScale` there). The pandas generators `test/oracles/parity-wp6-dataframe-reductions.py` and `parity-wp6-round5.py` computed the float sum and mean from the quantised words; they now use the pandas sum and mean, and `fixtures/parity-wp6-group-statistics.json` was regenerated (8 values changed, all in `unsigned64-carries`; see the note below). | `gpu-group-statistics.spec.ts:812` and `test/gpu-dataframe/key-join/gpu-key-join.spec.ts:412` (headless), the two parity specs, and `test/analysis-core/kernel-audit-dataframe.node.spec.ts` (binding limits) | Mean of 1.0e-4, 1.2e-4, 1.4e-4: 1.2207e-4 becomes 1.2e-4. Values below 7.6e-6 no longer sum to 0. `sumValues`, `means` and key-join `'sum'` and `'mean'` no longer depend on `sumScale`. The `sums` words are unchanged. Recipes that read group means (`fleet-dwell`, `drainage-network`, `drive-time-catchment`, `points-in-polygons-choropleth`) get the unquantised value; their specs pass. |
| dataframe-04 | fixed | `src/dataframe-analysis/column-profile/column-profile-kernels.ts:152` and `:568`: the bin is the `k` with `k * range <= (v - lo) * B < (k + 1) * range`, the convention of the official `GPUHistogram`, decided by float32 products. The mirror `test/gpu-dataframe/column-profile/column-profile-oracle.ts` copied the old rule and was changed. The same rule in `column-classification/class-breaks-kernels.ts` was left: it is the coarse internal discretisation of the approximate Fisher-Jenks, which needs bins consistent with its rounded width, and no output is a bin index. | `test/gpu-dataframe/column-profile/gpu-column-profile.spec.ts:538` (headless: exact integer rule, parity with the official `GPUHistogram` on two inputs, and a domain near the float32 range) | Domain `[-3.5, 12.25]`, 30 bins, value -0.875: bin 4 becomes bin 5. Values on a 0.125 grid over that domain: 10 of 30 counts change. Other values change bin only when they are exactly on an edge. |
| dataframe-05 | fixed (the code changed: the lead's decision was to add the term) | `src/dataframe-analysis/pair-statistics/gpu-variogram.ts:497`; the oracle `variogram-oracle.ts` and `gpu-variogram.node.spec.ts` copied the two-term form and were changed | `test/gpu-dataframe/pair-statistics/gpu-variogram.spec.ts:302` (headless, hand values for bins of 1, 2 and 3 pairs) | Robust semivariance of a bin with N pairs falls by 4.7 % at N = 1, 1.6 % at N = 2, 0.09 % at N = 10. |
| dataframe-06 | fixed, converged on `GPURipley` and spatstat `Kest` | `src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics.ts:266` (`getFinishWGSL`: `n (n - 1)` when focus and neighbour are the same set and no intensities are given; `n_i n_j` for cross-type; unchanged with intensities) | new `test/gpu-dataframe/pair-statistics/gpu-marked-point-statistics.spec.ts:78`, `:151` (headless, hand values and parity with `GPURipley`); `test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts:219` pinned the `n^2` value and was changed | Same-type K and pair correlation without intensities grow by `n / (n - 1)` (4 points: 4 becomes 16 / 3). A same-type set of one point gives NaN and status 1 instead of K = 0. Homogeneous K no longer equals the inhomogeneous K with `lambda = n / A`. |
| dataframe-07 | fixed (documentation) | `src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices.ts:319` and `docs/api/gpu-dataframe-analysis.md`. The dead Wilson-Hilferty branch stays, with a comment at `:110`, so that the shader text of the small-df kernel does not change. | none (words only) | none |
| dataframe-08 | deferred here; fixed in the closing pass of 2026-11-11 with an optional rational probability (see "Closing pass" at the end) | Documentation only: `src/dataframe-analysis/column-classification/order-statistic-wgsl.ts:22`, `gpu-column-quantiles.ts:87`. The suggested exact integer rank was not adopted: measured over 12 decimal fractions and n up to 5,000, `floor(fround((n - 1) p))` agrees with numpy (float64 `p`) in 59,918 of 59,988 cases, the exact product of the float32 `p` in 58,811 (`fround(1000 * fround(0.7))` is 700; the exact product is 699.99999). An exact rank needs the probability as a rational or float64, a new prop (roadmap 12.4). | none | none |
| dataframe-09 | fixed (was SUSPECTED; confirmed as a wrong result on the GPU with status 0) | `src/dataframe-analysis/pair-statistics/variogram-model-fit-nodes.ts:96` (ladder weights divided by their total before the normal equations, as the refinement already did) and `:150` (representability guard, see "Follow-up" below) | `test/gpu-dataframe/pair-statistics/variogram-model-fit-replay.spec.ts:87` (headless: noise-free exponential bins scaled by 1e-8 must return the scaled model; old nugget 0.076e-8 for a true 0.2e-8) | Cressie-weighted fits of semivariances near 1e-8 with about 1e4 pairs per bin now recover nugget, sill and range. Fits at ordinary magnitudes change only in the last float32 bits. |
| dataframe-10 | deferred here (no splancs fixture can be made here: no R); fixed in the closing pass of 2026-11-11 against a port of the splancs source, which also shows that the description of `stmctest` in this row and in finding 10 is wrong (see "Closing pass" at the end) | Documentation reworded to what the kernel computes and no longer claims the splancs definition: `src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test.ts:201`, `docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md`. The formula is unchanged. | none | none |

Items outside the ten findings.

| Item | Outcome | What changed | Spec |
| --- | --- | --- | --- |
| Directional variogram, `atan2(0, 0)` (section 7) | fixed | `gpu-variogram.ts:421`: a coincident pair stays in sector 0; documented. The oracle used `Math.atan2(0, 0) = 0` and was changed. | `gpu-variogram.spec.ts:360`. It also passes on the old kernel on this adapter, whose `atan2(0, 0)` reduced to sector 0; it pins the rule for other adapters. |
| Hanisch G and F denominators (section 7) | documentation corrected, formula unchanged | Counterexample against the Hanisch (1984) definition: two interior points with no neighbour within `maximumDistance` beside one close pair give G = 0.4414 where the estimator is 1, so "biased slightly high" was wrong in that direction. `gpu-ripley-distance-functions.ts:182` and the docs now state both directions. An exact fix needs nearest-neighbour distances beyond `maximumDistance`. | `test/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions.spec.ts:195` (pins the documented value and G = 1 when `maximumDistance` covers every nearest neighbour) |
| `atan2(y, -0.0)` in the isotropic edge weight (extra finding, wrong-edge-case) | fixed | `gpu-ripley.ts:320` and `space-time-k-function-wgsl.ts:41`: a point on a window side returns a quarter turn. A coordinate of `-0.0` against a bound of 0 gave K(3) = 83.3 for a true 150. | `test/gpu-dataframe/pair-statistics/gpu-ripley.spec.ts:277` (headless) |
| Class intervals right-closed in `column-classification` (roadmap 12.3) | deferred, not started | See below. | none |
| luma-official 3.3: `GPUGroupStatistics` and `GPUKeyJoin` re-implement official classes with no reason stated | fixed (documentation) | The reason is now in both class documentations. | none |
| luma-official-05: private 64-bit helpers in `gpu-ripley.ts` | deferred (roadmap 12.4, shared 64-bit helpers) | none | none |
| Compensated float sums (lead's note) | not applicable | `grep -rliE "neumaier|kahan|compensat|twoSum|two-sum"` over `dataframe-analysis` has no hit; no fix here uses an algebraically zero error term. | none |

### Follow-up after the first report (same day)

| Item | Outcome | What changed | Spec |
| --- | --- | --- | --- |
| Regression from dataframe-09: `test/gpu-spatial-analysis/spatial-interpolation/variogram-cooperative-refinement.spec.ts` failed 7 of 15 tests, all in its `overflow` scenario (every semivariance 1e38, expected status 2) | fixed in the kernel; the spec is unchanged | Cause: with unnormalised ladder weights `sum w * gamma` overflowed and every candidate was non-finite, which is where status 2 came from. With normalised weights the nugget-only fit of a constant column is exact, so its residual was 0 when the weighted mean rounded back to 1e38 and infinity when it was one unit in the last place off: statuses 0, 2 and 3 mixed by rounding. The two stages did rank by the same objective (both report the residual in the caller's unnormalised weights); the objective itself was not evaluable. `src/dataframe-analysis/pair-statistics/variogram-model-fit-nodes.ts:150`: the prepare pass now reports status 2 for every model when the float32 resolution of the largest semivariance (`2^-23` of it), squared and summed with the base weights, is not finite, that is above about `1.5e26 / sqrt(sum of weights)`. Documented in `gpu-variogram-model-fit.ts`. | The unchanged cooperative spec (15 of 15 pass); `test/gpu-dataframe/pair-statistics/variogram-model-fit-replay.spec.ts:87` now also fits the bins scaled by 1e12, so the guard is pinned on both sides |
| Private two-sided p-value of `GPUSpatialCorrelogram` without the shared floor | fixed | `src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram.ts:751`: `clamp(erfc, ANALYTIC_P_VALUE_FLOOR, 1)` with the floor `1.17549435e-38`, the rule of the shared `getTwoSidedPValue` | `test/gpu-dataframe/pair-statistics/gpu-spatial-correlogram.spec.ts:222` (headless; a gradient lattice with z above 14: old p-value 0, now `2^-126`) |

Behaviour changes: a `GPUVariogramModelFit` frame whose semivariances exceed about
`1.5e26 / sqrt(sum of weights)` is status 2 for every model, deterministically (before the
dataframe-09 change it was status 2 through overflow for most such inputs; a few with small
residuals could fit). A correlogram p-value that was exactly 0 is now `2^-126`.

### Review follow-up (`fix-review-statistics.md`)

| Row | Outcome | What changed | Spec |
| --- | --- | --- | --- |
| fr-statistics-02 (dataframe-01 fixed for clear negative correlation only) | fixed; reproduced on the GPU for two indicators (r = -0.01: eigenvalue 1.00856 for a true 1.01) and for three (eigenvalues 1.0008 and 1: 1.00055 returned) | `src/dataframe-analysis/composite-indicators/gpu-composite-score.ts:718`: closed form for two non-constant indicators, `(1, sign(r)) / sqrt(2)` with `(1, 1)` at the tie `r = 0`. `:733`: for every other case the last eigenvector of the shared cyclic Jacobi decomposition (`analysis-core/kernels/symmetric-eigen-wgsl.ts`, 24 sweeps at most, tolerance 1e-7), then `:751` four power steps from that vector, which bring the float32 residual back under the 1e-6 of the sklearn and scipy parity fixture. The power iteration, its `d + 1` starts and the Rayleigh margin are gone. `GPU_COMPOSITE_SCORE_POWER_ITERATIONS` stays exported and is marked deprecated. All-constant input now gives zero loadings (it gave `1 / sqrt(d)`; the documentation already said zero). The CPU twin follows; the node spec no longer checks it against its own Jacobi but against an independent long power iteration and closed forms. | `test/gpu-dataframe/composite-indicators/gpu-composite-score.spec.ts:329` (headless: r = -0.01, -0.001, +0.01 as exact float32 designs against the closed form) and `:359` (three indicators, eigenvalues 1.0008, 1, 0.9992); `gpu-composite-score.node.spec.ts:338` (twin, same cases in float64). The scipy `eigh` parity fixture agrees to 1.3e-7 on the loadings. |
| fr-statistics-03 (z-score correction may be reassociated away) | fixed; confirmed on the GPU: values 1e7 .. 1e7 + 3 gave -1.549 for a true -1.162, the 0.39 the reviewer predicted | `src/dataframe-analysis/group-statistics/group-statistics-moments.ts:640`: the deviation `v - mean` passes through a bit operation with a runtime zero (a group index shifted right by 31) before `delta` is subtracted, the device `order-statistic-wgsl.ts` uses against fused multiply-add, so the compiler cannot form `mean + delta`. | `test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts:885` (headless; fails without the bit operation, which the dataframe-02 spec at `:812` did not) |
| Sentence about `analysis-core/cell-key-wgsl.ts` | corrected | The dataframe-03 row now says this stream made no edit there. | none |

Behaviour changes: the first principal component of weakly correlated or nearly tied indicators
is now the eigenvector (two indicators at r = -0.01: loadings (0.871, -0.492) become
(0.707, -0.707)); for well-separated inputs it agrees with the earlier result to float32 rounding.
Group z-scores of columns whose mean is large against their spread are corrected (the error was up
to half a float32 step of the mean divided by the standard deviation).

### Class intervals: why this is deferred

The decided default is right-closed. In `column-classification` that is not an edge-rule change
alone:

- `GPUClassBreaks` natural-breaks edges are lower edges (the smallest value of the upper class) in
  all three variants (`natural-breaks-nodes.ts`, 1,258 lines whose refinement moves each cut to
  "the smallest value on its right"; `exact-natural-breaks-nodes.ts`; `monotone-natural-breaks-nodes.ts`)
  and in their bit-exact mirrors. With a right-closed rule those edges put the smallest value of
  every class in the class below, so the edges must become upper bounds (`exactBreaks` of the
  committed fixture, not `exactLowerEdges`).
- `GPUColorScale`, `GPUBivariateClassification` and `GPUClassBreaks.classCounts` share one rule
  (`<=`, d3 `bisectRight`). Changing only the counts would make the legend disagree with the map.
  Changing the scales departs from d3, which is their named reference, and invalidates
  `color-scale-native-fixtures.json` (d3 output) and `parity-wp6-bivariate.json` (numpy
  `searchsorted(side='right')`).
- `modules/lenses` documents and twins the bisect-right rule (`computation-pack/operators/classification.ts`);
  lenses are out of bounds in this pass.

Risk to flag now: if `distribution-dynamics/gpu-class-assignment.ts` becomes right-closed in this
pass while `GPUClassBreaks` is unchanged, natural-breaks edges fed from `GPUClassBreaks` into
`GPUClassAssignment` place the smallest value of each class one class low. Equal-interval,
quantile, standard-deviation, box-plot, head-tail and maximum-breaks edges are mapclassify bins
and are safe.

### Regenerated fixture

`test/oracles/fixtures/parity-wp6-group-statistics.json` was regenerated with the system Python
(3.9.6, numpy 2.0.2, pandas 2.2.2, scipy 1.13.1; the pinned interpreter with pandas 3.0.6 is not on
this machine) and copied over the committed file. A structural diff against the committed fixture
shows 8 changed values (`sumValues` and `means` of two groups, sample and population) and the four
`versions` entries; every other value is identical. The other fixtures of the same two generators
are identical apart from `versions` and were left as committed.
Closing pass (2026-11-11): rerun on the pinned set (pandas 3.0.6), every value equal, fixture replaced; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).

### Upstream gaps

- `GPUHistogram` (`@luma.gl/graph/gpu-core`) could replace the `GPUColumnProfile` histogram kernel
  if it accepted a domain as two rows at an offset inside a larger view with NaN meaning "the
  finite extent", and an output at an offset of a larger view (several histograms in one view).
- `GPUHistogram` with a GPU-resident `domain` view of `[-1e38, 1e38]` and 16 bins puts the values
  0, 1, -5e37 and 5e37 all in bin 0 on the test GPU; with the same domain as a literal it gives
  bins 8, 8, 4, 12. The runtime division appears to be compiled as a multiplication by a
  reciprocal that is subnormal for a range above 2^126. Its float32 division also makes a value
  within a few units in the last place of an edge adapter dependent; a product test
  (`k * range <= (v - lo) * B`) is adapter independent.
- `GPUGroupAggregation` and `GPUHashJoin`: sparse and 64-bit keys, row masks, order statistics and
  moments, 1:n aggregates, and sums that do not depend on thread order.

### Not verified

- The Cressie and Hawkins three-term constant and the spatstat `han` and splancs `stmctest`
  definitions are from the review and from memory of the packages; none of the three packages
  could be run here. The committed scikit-gstat fixture cannot tell the two Cressie forms apart
  (smallest bin 1,341 pairs).
- `modules/lenses/src/gpu-dataframe/computation-pack/operators/indicators.ts:17` still describes
  the single-start power iteration, and lens twins of the group mean, key-join sum, column-profile
  histogram and same-type marked K may encode the old results. They are out of bounds here.

### Closing pass (2026-11-11)

Stream "dataframe" of roadmap 12.6: the two deferred findings. Paths are relative to
`modules/analysis/`. Reference environment: CPython 3.14.8, NumPy 2.5.4, Shapely 2.2.0 (GEOS
3.14.1). Specs ran on the test GPU with `npx vitest run --project headless <spec>` and
`--project node <spec>`.

| ID | Outcome | Files changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| dataframe-08 | fixed, as a new optional prop; the default path is unchanged | `src/dataframe-analysis/column-classification/order-statistic-wgsl.ts` (new `RATIONAL_ORDER_STATISTIC_RANK_WGSL`), `column-quantiles-nodes.ts` (the rank and fraction branches), `column-quantiles-parameters.ts` (packer), `gpu-column-quantiles.ts` (prop `rationalProbabilities`, limits, documentation), `index.ts` (exports); `docs/api/gpu-dataframe-analysis.md`; generator `test/oracles/dataframe-quantile-rational-rank.py` and its fixture | `test/gpu-dataframe/column-classification/gpu-column-quantiles-rational.spec.ts` (7 tests, headless) and `gpu-column-quantiles-rational.node.spec.ts` (3 tests) | NumPy 2.5.4 `quantile` on binary64 copies of the float32 columns, and the definition on Python integers and `fractions.Fraction` |
| dataframe-10 | fixed: the kernel computed a statistic that neither splancs nor surveillance computes | `src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test.ts` (statistics and summary kernels, documentation); `docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md`; `test/gpu-dataframe/pair-statistics/space-time-diggle-test.spec.ts` (its reference summed standardised residuals); generator `test/oracles/space-time-diggle-splancs-port.py` and its fixture | `test/gpu-dataframe/pair-statistics/space-time-diggle-splancs.spec.ts` (4 tests, headless), `space-time-diggle-splancs.node.spec.ts` (4 tests) with the helper `space-time-diggle-splancs-fixture.ts` | A NumPy port of the splancs 2.01-45 source. It is a port, not a run of R |

**dataframe-08, the quantile rank.**

- Reproduced on the GPU before the change. At 16,777,216 rows and `p = 19/20`, on a step column
  whose value rises by 1 at each exact lower rank, the default path returns 15.0 for `linear`
  where NumPy returns 15.25 (the float32 rank has no fraction). At small `n` the default
  `floor(fround(fround(n - 1) * fround(p)))` differs from NumPy's `floor((n - 1) * q)` in 40 of
  6,912 combinations of 27 row counts and 256 fractions (101 rows, `p = 0.59`: rank 58, NumPy 59).
  Both are kept as tests of the default path, which must not change.
- The prop is `rationalProbabilities`, a uint32 view of `numerator, denominator` pairs written by
  `getGPUColumnQuantilesRationalProbabilityValues`. The kernel forms `(n - 1) * numerator` as a
  64-bit product (`multiplyUint32Wide`) and divides with `divideUint64ByUint32`
  (`analysis-core/kernels/uint64-wgsl.ts`); the lower rank, the upper rank and round-half-even are
  integer comparisons. The `linear` fraction `remainder / denominator` is rounded once to the
  nearest float32 by a second integer division and assembled as a bit pattern, because float
  division is not correctly rounded on the test GPU (contract section 11). No float sum,
  subtraction chain or lookup table was added.
- Evidence. (a) Small `n`: 27 row counts from 1 to 5,000 by 256 fractions by 5 methods, 34,560
  cases. The GPU equals the exact rational value in all of them (bit for bit for `lower`,
  `higher`, `nearest`, `midpoint`; within 4e-4 on values up to 1,000 for `linear`). NumPy equals
  the exact value in 34,538; the generator asserts that the other 22 (`lower` 3, `higher` 8,
  `midpoint` 11) are cases where the exact `(n - 1) p` is an integer and NumPy's float64 product
  is not. It lands just below in 6 of them (101 rows, `q = 0.29`: 28.999999999999996; `lower`
  and `midpoint` are one rank off) and just above in 16 (101 rows, `q = 0.55`:
  55.00000000000001; `higher` and `midpoint` are); all 22 are at 101 rows. So the rational path
  is exact where NumPy itself is one rank off. (b) Large `n` on the GPU: 2^24, 2^24 + 1 and 2^25 rows, 24 fractions by
  5 methods, all equal to NumPy and to the exact value (within 4e-6 for `linear`). (c) The rank
  WGSL alone, with no data: 480 `(count, numerator, denominator)` triples with counts up to
  2^32 - 1; lower rank, remainder and the bits of the float32 fraction equal Python integer
  arithmetic exactly. (d) Two deliberate breaks of the WGSL (a 32-bit product; a wrong rounding
  rule) fail 6 and 1 of the 7 tests.
- The default path: the generated shader text of all 42 kernels (with and without mask and
  filter) is byte for byte the committed text, compared with a scratch script, and the existing
  specs pass unchanged (`gpu-column-quantiles.spec.ts`, `column-classification-composition.spec.ts`,
  `gpu-class-breaks.spec.ts`, the node specs and `kernel-audit-dataframe.node.spec.ts`).
- Limits. With the prop a column may exceed 2^24 rows, up to one storage binding (checked
  against the device at `getCommandNodes`) and never above 2^31 - 1; a filter output above 2^24
  rows is refused, because the percentile filter ranks are still float32.
- Not verified, not done. No column above 2^25 rows was run (the default binding limit), so the
  radix selection is untested there; only the rank arithmetic is covered to 2^32 - 1. The filter
  ranks were not made exact. `GPUClassBreaks` (`'quantile'`, whose probabilities `i / k` are
  rationals already, and `'box-plot'`) and the `GPUGroupStatistics` medians and percentiles still
  use the float32 rank; they are other streams' files.

**dataframe-10, the Diggle statistic.**

- Source. `splancs_2.01-45.tar.gz` from CRAN (SHA-256
  `8bccf1d61d7feaab52da07a9c95aa66bcd3986a6b214f13b232c1e2bea4b76d3`), read in full for
  `stkhat.S`, `stmctest.S`, `stsecal.S`, `stvmat.S`, `stdiagn.S`, `stkhat.f`, `stsecal.f`,
  `cncvwt.f`, `iplace.f`, `plarea.f`. R is not installed; nothing from the package was run.
- What splancs computes. `stmctest` uses `sum(kst - outer(ks, kt))`, the plain sum of `D` over
  the grid, for the data and for each `sample(times)`. It never calls `stsecal`; the standard
  error is used only by `stdiagn` for the plot of `R = D / SE`. surveillance `stKtest` (source
  read online) computes the same sum and `p = mean(c(t0, t) >= t0)`. The statement in this
  review and in the first pass that `stmctest` sums `D / SE` was wrong.
- What the kernel computed. `sum (D - m) / s` with the per-class permutation moments. Reproduced
  on the GPU against the port before the change: `K_s`, `K_t`, `K` and `D` agreed, the statistic
  did not (103.85 against 269.62 with 3 permutations, NaN with 1). It also standardised the
  observed pattern with moments it was not part of and the permutations with moments they were
  part of, so the Monte Carlo ranks were not exactly exchangeable.
- Fix. The statistic is `sum D` for the observed pattern and every permutation. The per-class
  moments stay as the optional outputs `permutationMean` and `permutationStandardDeviation`
  (diagnostics); their kernel is created only when one is requested. `summary.usedClasses`
  counts classes with a finite observed `D`. Behaviour change recorded in `behaviour-changes.md`.
- Evidence for the port (`test/oracles/space-time-diggle-splancs-port.py`). It is written from
  the formulas the source evaluates, with its own code. Checked when it runs: the circle angle of
  the edge weight against Shapely 2.2.0 for every ordered pair of a rectangle scene and an
  L-shaped scene (largest difference of the circle fraction 1.1e-7); the `stsecal` variance
  against all 720 relabellings of six events (relative error 3e-14). The binary64 TypeScript K
  reference, written earlier from the definitions with another circle algorithm, equals the port
  to 1e-9 for `K_s`, `K_t`, `K`, `D` and the statistic on 7 labellings of 72 events.
- Evidence for the kernel. Against the port on the GPU: `K_s`, `K_t`, `K`, `D` and `U` for the
  observed pattern and 6 relabellings, measured within 2e-6 of the largest `K` (bounds 1e-5 to
  2e-5). The kernel's own 199 permutations give a null distribution whose standard deviation is
  within 25% of the port's 199 simulations and put the observed statistic in the same tail (the
  port: third of 200, p = 0.02). `space-time-diggle-test.spec.ts` checks every permuted statistic
  against the K reference on the kernel's own shuffles, and now also the two moments outputs.
- Differences from splancs that remain, documented on the class and pinned: a lag that exactly
  reaches a limit of the time window has temporal weight 1 in the kernels and 2 in splancs (hand
  case: two events at t = 3 and 6 in [0, 10] give `kt` 10 here and 15 in splancs); a pair at
  exactly the largest distance or lag is counted here and dropped by splancs; the spatial weight
  is capped at 100 here. The first two happen only at exact ties, which integer times can hit.
  The weight lives in `space-time-k-function-wgsl.ts` (shared with `GPUSpaceTimeKFunction`, whose
  reference is stpp), so it was left for a decision. Superseded: the review found this to be a
  defect on whole-day times and the two tie rules now follow splancs; see "Review follow-up"
  below.
- Not verified, not done. No run of R: the fixture is a port and says so in its `provenance`
  field, in the generator and in the documentation. The 1995 paper was not available, so the
  statistic is attributed to splancs and surveillance, not to the text of the paper. A splancs
  quirk at `d == s_max` (`stkhat.f` reuses the class and weights of an earlier pair) is not
  reproduced; the scenes have no such pair. `stsecal` is in the fixture but not on the GPU.
  splancs is GPL (>= 2): the generator contains no splancs code and carries the MIT header of
  its neighbours; whether that is acceptable is the maintainers' call. The two generators are not
  yet listed in `test/oracles/GENERATORS.md` (the fixtures stream's file).
- Callers. `website/src/gpu/runs/entries/space-time-events/space-time-diggle-test.ts` draws the
  statistics and its caption describes the standardised sum; `website/src/catalog/entries/space-time-events.ts`
  says "The statistic sums standardized D residuals". Both are out of bounds here.

**Review follow-up (`fix-review-closing-fields.md`, same day).**

| Finding | Outcome | Files changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| frc-fields-01 | fixed: `GPUSpaceTimeDiggleTest` follows splancs at exact ties; `GPUSpaceTimeKFunction` is unchanged and keeps the rules of its own reference, stpp | `src/dataframe-analysis/pair-statistics/space-time-k-function-wgsl.ts` (new `SPACE_TIME_K_SPLANCS_TIE_WGSL`, type `SpaceTimeKTieRule`), `space-time-k-function-nodes.ts` (internal `tieRule` of the pair pass and the temporal marginal), `gpu-space-time-diggle-test.ts` (observed pass and permutation batches), `gpu-space-time-k-function.ts` (documentation only); `docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md`; `test/gpu-dataframe/pair-statistics/space-time-k-function-reference.ts` (option `tieRule`); the generator and fixture of the port (new `integerTimes` section) | `space-time-diggle-splancs.spec.ts` (now 6 tests, headless) and `space-time-diggle-splancs.node.spec.ts` (now 5 tests) | The port of the splancs source on whole-day times; the stpp 2.0-8 source for the other rule |
| frc-fields-04 | fixed (words) | `gpu-column-quantiles.ts`, `docs/api/gpu-dataframe-analysis.md`, `test/oracles/dataframe-quantile-rational-rank.py` (docstring; the fixture is byte for byte the same), this report | none | Recounted from the fixture: 6 of the 22 differences land just below the integer, 16 just above, all at 101 rows |

- What `GPUSpaceTimeKFunction` documents. Its reference is stpp 2.0-8 `STIKhat`, and it states
  its own tie rules (classes closed above, the largest included; weight 1 when both `t_i - t_ij`
  and `t_i + t_ij` lie in the window). `stpp_2.0-8.tar.gz` was downloaded from CRAN (SHA-256
  `3f74251c65db7c27b38b4a1d57ca4079a165b7c82e18ce7e76c19dee960568f5`) and `src/stikfunction.f`
  read: a pair is counted when `(tij.le.t(iv)).and.(hij.le.s(iu))`, and `vij = 1` when
  `(bsup.le.bsupt).and.(binf.ge.binft)`. So the K function's rules are those of its reference,
  and the two packages genuinely disagree at both ties. Following the lead's rule for that case,
  the K function was left alone and the Diggle test was given the splancs rules.
- The change. An internal `tieRule` (`'stpp'` default, `'splancs'`) of `createSpaceTimeKNodes`,
  passed by the Diggle test only. With `'splancs'`: the temporal weight is 2 when
  `radius <= lag`; a pair with `distance >= maximumDistance` or `lag >= maximumTimeLag` is
  dropped; the class is the first break at or above the value, tested with float32 products
  (`value * count <= maximum * (k + 1)`). The temporal marginal uses the same three rules in its
  binary searches, with a guard for an event on a time limit (radius 0: no partner has weight 1,
  and the count may not rely on the event's own lag 0 passing the test). The permutation batches
  of the Diggle test use the same WGSL functions.
- `GPUSpaceTimeKFunction` is unchanged: the shader text of 136 kernels (both edge corrections,
  with and without intensities, every optional output) is byte for byte what it was before this
  change, compared with a scratch script against the committed file plus the first pass's
  rename; its specs pass unchanged. No K function fixture changed (there is none from a
  reference library run).
- Reproduced. With the old rule put back in the observed pass, the whole-day cases fail on the
  GPU (`K_t` of the first class 2.805 against the port's 3.110; the two-event hand case gives
  `kt` 10 against 15). With the old weight put back in the permutation batches only, the
  permutation case fails. Both were reverted.
- Evidence. The fixture has a new scene: 60 events anywhere in the window with whole-day times 0
  to 30 (both limits occupied), limits `[0, 30]`, three lag grids (8.5 in 5 classes, 7 in 7, 13
  in 13), the observed labelling and three relabellings each. The port counts 134, 128 and 201
  ordered pairs whose lag exactly reaches a time limit, 0, 166 and 116 at the largest lag, and
  0, 1,114 and 2,038 on interior class breaks. (a) Node: the binary64 reference with the splancs
  rules equals the port to 1e-9 on all 12 analyses; with the stpp rules `K_t` and `U` differ
  (`K_t(8.5)` 15.67 against 16.81, `K_t(7)` 13.83 against 12.86). (b) GPU: the kernel equals the
  port on all 12 (`K_s`, `K_t`, `K`, `D`, `U`, bounds 1e-5 to 2e-5 of the largest `K`), and its
  own 12 permutations per grid equal the reference on the kernel's shuffles, which exercises
  the batch path on ties. (c) Hand cases on the GPU, the Diggle test against the K function on
  the same input: a lag reaching a time limit (`kt` 15 against 10), a lag at the largest lag
  (`kt` 0 against 10), a distance at the largest distance (`ks` 0 against 80).
- Not reproduced, left alone. In IEEE float32 the scaled class of the stpp path can put a
  whole-number lag on a whole-number break one class up (`7 / 13 * 13` is 7.0000005, so 13
  one-day classes would put lag 7 in the class of day 8; 546 of 14,560 whole-number grids up to
  64 classes have such a lag). On the test GPU the 13-class case bins correctly with the scaled
  form (the whole-day cases passed before the product test was added), so the defect did not
  reproduce and `GPUSpaceTimeKFunction` was not changed. The splancs path does not depend on
  it. Other adapters were not tried.
- Not verified, not done. No run of R, as before. The `stkhat.f` quirk at a distance exactly
  equal to the largest (a stale term added to `kst`) is not reproduced: the pair is dropped
  from both `ks` and `kst`. The weight cap of 100 stays. stpp's `infectious = TRUE` branch and
  its other corrections were not read for ties.
