# Math correctness review: raster imagery (`raster-imagery`)

Date: 2026-11-10. Scope: nine directories of `modules/analysis/src/raster-analysis`:
`raster-time-series`, `remote-sensing`, `change-detection`, `raster-embeddings`, `raster-stretch`,
`raster-algebra`, `raster-zonal-statistics`, `raster-patches`, `focal-pairs` (about 17,700 lines).
Paths below are relative to `modules/analysis/src`. Scratch scripts are in the session scratchpad
under `raster-imagery/` (`whittaker.mjs`, `misc.mjs`, `remap.mjs`, `conditional.mjs`); none are in
the repository. No spec was run; every confirmation is a hand derivation or a `Math.fround` replica
of the kernel arithmetic.

Summary: 12 findings. One wrong edge case, three float32 precision losses, five inconsistencies,
one documentation mismatch, two untested. No kernel was found wrong on ordinary input. Most
estimators agree with their reference and most have an independent fixture.

## 1. Coverage

Read in full (every WGSL body and every parameter packer):

- `raster-time-series/`: all 24 source files. For eleven of them the first 20 to 30 lines (imports)
  were skipped; nothing else.
- `remote-sensing/`: all 11 source files (`raster-masked-registration.ts` and
  `gpu-raster-region-graph.ts` from line 24, imports skipped).
- `change-detection/`: all three source files.
- `raster-stretch/`: both source files (`gpu-raster-stretch.ts` from line 30).
- `raster-algebra/`: all nine source files (imports skipped in five).
- `focal-pairs/`: both source files.
- `raster-patches/`: `gpu-raster-patch-metrics.ts`, `patch-labels.ts`, `gpu-raster-sieve.ts`.

Read in part:

- `raster-zonal-statistics/gpu-raster-zonal-statistics.ts`: lines 40 to 369 and 440 to 1214 (types,
  class documentation, every kernel and the planner). Lines 370 to 439 (constructor validation) not
  read.
- `raster-embeddings/`: every kernel body, class documentation and output type of all 14 source
  files. Constructors (validation only) were skipped in `gpu-raster-embedding-similarity.ts`,
  `-change.ts`, `-linear-probe.ts`, `-knn-classifier.ts` and `-projection.ts`.
  `raster-embedding-common.ts`, `raster-embedding-usable-mask.ts` and
  `gpu-raster-embedding-dequantize.ts` were read through a filter that dropped import lines.

Not opened: the nine `index.ts` barrels and `raster-patches/gpu-raster-sieve-parameters.ts`.

Outside the listed directories, read only as far as needed:

- `raster-analysis/gpu-raster-spectral-indices.ts` (grep of formulas only). It is not in a scope
  directory but the brief names NDVI, EVI, SAVI, NDWI and NBR.
- `analysis-core/kernels`: `wgsl-theil-sen-median.ts` (first 60 lines), `incomplete-gamma-wgsl.ts`
  (`getLogGammaValue` only), `wgsl-common.ts` (finite, ordered key and erfc helpers),
  `fixed-point-accumulator.ts` (exponent rule only). `normal-quantile-wgsl.ts`,
  `sorted-segment-sums.ts` and `dataframe-analysis/time-window-filter/time-words.ts` (the WGSL
  part) were not read; their results are trusted here, not verified.
- `analysis-core/spherical-cell-area.ts` (first 120 lines).
- Published luma.gl: `gpu-raster-ndvi.js`, `gpu-raster-band-math.js` (denominator rule),
  `gpu-raster-global-statistics.js` (percentile kernel), `gpu-raster-contrast.js` (gamma and
  equalization), `gpu-raster-region-measurements.js` (area), `gpu-raster-threshold.js`
  (comparators), `gpu-similarity-search.js` (cosine and validity), `gpu-time-series.js`,
  `gpu-fft1d.js` and `gpu-fft2d.js` (sign and normalisation).
- Tests: file lists, oracle generator headers and a few greps. No spec body was read in full.

The brief also lists tasselled cap, spectral angle mapper and pan-sharpening. The library has no
code for any of them (grep of `modules/analysis/src`).

## 2. Findings

| ID | Severity | Status | File:line | What the kernel computes | What the reference says | Concrete input and wrong output | Suggested fix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raster-imagery-01 | wrong-edge-case | CONFIRMED | `raster-algebra/gpu-raster-conditional.ts:173-177`, `raster-algebra/gpu-raster-arithmetic.ts:131-133` | The no-data sentinel test is applied to the per-frame constants (`constantA`, `constantB`), not only to raster cells. | ArcGIS `Con`, `numpy.where`, `gdal_calc`: a constant operand is never no-data. The prop documentation says the sentinel applies to `conditionValues`, `a` and `b` rasters. | `GPURasterConditional`, `noDataValue: 0`, default constants (1 and 0), `value > 10`, cells `[5, 20, 0]`: output `[NaN, 1, NaN]`, expected `[0, 1, NaN]`. `GPURasterArithmetic` `maximum`, `noDataValue: 0`, default `constantB` 0, cells `[-3, 7]`: output `[NaN, NaN]`, expected `[0, 7]`. | Test the sentinel only on values read from a view. Use `isNaNValue` alone for constants. |
| raster-imagery-02 | precision | CONFIRMED | `raster-time-series/gpu-raster-whittaker-smoother.ts:261-318` (solve), `:37` (cap) | Float32 banded LDL of `W + lambda D'D` on the raw values. The error grows with `lambda` and with the magnitude of the data, not its variation. At the top of the accepted range the factorisation fails. | Binary64 banded solve (the SciPy fixture). The Whittaker solution is shift invariant: smoothing `y + c` gives `z + c`. | 256 slices, `d = 2`, series `300 + 0.3 sin(...)` with one gap in five: `lambda = 1000` gives a maximum error of 0.056, `lambda = 5000` gives 0.23, `lambda = 1e4` gives 0.66 (twice the signal amplitude), status 0. Same series around 0.5: 9e-5, 4e-4, 9e-4. `lambda = 1e7` (`d = 2`) and `lambda = 1e6` (`d = 3`) return status 2, inside the accepted cap of 1e7. | Subtract a per-cell reference (first valid value or weighted mean) before the solve and add it back. In the replica this brings the error to 2e-5 to 3e-4 at every offset. Lower the cap, or document that status 2 is expected near it. |
| raster-imagery-03 | precision | CONFIRMED | `remote-sensing/gpu-raster-conservative-remapping.ts:194-205` | Cell edges are built as `f32(index) * f32(spacing)` in the source frame. Destination and source edges that coincide exactly differ by one float32 step at large indices, which creates sliver overlaps with the next source cell. | xESMF and ESMF conservative weights in binary64. The class documentation says "Exact rectangle overlap integration". | Source spacing 0.1, destination spacing 0.3, shared origin. Destination column 3414 gets an overlap of 1.2e-4 (1.2e-3 of a source cell) with source column 10245, which it does not touch. 912 of the first 4000 destination columns have such a sliver. With `normalization: 'covered-area'` and the three true source cells invalid, the cell publishes the neighbour's value with status 0 instead of NaN with status 1. | Build the weights from integers: compute overlap from `index * numerator` on a common rational grid, or prepare the per-axis edge tables on the host in binary64 and upload them. Treat a covered area below a relative threshold as no overlap. |
| raster-imagery-04 | precision | CONFIRMED | `raster-embeddings/gpu-raster-embedding-change.ts:206-209` | `angle = acos(clamp(dot(a, b) / sqrt(dot(a, a) * dot(b, b))))` in float32. | Exact angle. Near 1 the cosine has a float32 step of 6e-8, so `acos` cannot resolve angles below about 5e-4 rad. | 64-band vector, one component moved by 0.001: exact angle 4.37e-4 rad, kernel angle 0. The `magnitude` output of the same kernel is accurate. | Compute the angle from the chord of the two normalised vectors: `2 * asin(0.5 * length(normalize(a) - normalize(b)))`, or `atan2` of the cross and dot terms. |
| raster-imagery-05 | inconsistent | CONFIRMED | `raster-stretch/gpu-raster-stretch.ts:696-713` | Equalization maps `v` to `(count below the bin + fraction inside the bin * bin count) / total`. No subtraction of the first bin. | Published `GPURasterContrast` (and OpenCV `equalizeHist`): `(cdf[bin] - cdfFirst) / (total - cdfFirst)`. scikit-image interpolates the inclusive CDF at bin centres. | Three equally frequent values `{0, m, 255}`, 1024 bins. Published rule: `m` maps to 0.5 for every `m`. Local rule: `m = 100` gives 0.523, `m = 127.5` gives 0.333, `m = 200` gives 0.379. The result depends on where the value sits inside its bin. | Decide one rule. Either delegate equalization to `GPURasterContrast` semantics, or document the piecewise-linear CDF and state that discrete rasters need `binCount` aligned with the value step. |
| raster-imagery-06 | inconsistent | CONFIRMED | `raster-stretch/gpu-raster-stretch.ts:517-544`, `:555-556` | Percentile rank is `p * N`; the first bin whose cumulative count reaches it; linear position inside the bin. | Published `GPURasterGlobalPercentile`: rank `floor((N - 1) * p)`, centre of the bin. NumPy default: linear between order statistics at `(N - 1) * p`. | Ten values 0, 10, ..., 90, `p = 0.98`, 1024 bins: local 89.98, published kernel 80.02, NumPy 88.2. At `p = 0.5`: 40.08, 40.03, 45. | Use one rank rule in both places and name it in the documentation (the local one is the inverted-CDF rule). Negligible for large rasters; visible for small windows and region masks. |
| raster-imagery-07 | inconsistent | CONFIRMED (by reading) | `raster-algebra/gpu-raster-cell-statistics.ts:204` against `raster-zonal-statistics/gpu-raster-zonal-statistics.ts:973` | `GPURasterCellStatistics.standardDeviation` divides by `n` (population). `GPURasterZonalStatistics.standardDeviations` divides by `n - 1` (sample). | ArcGIS Cell Statistics and Zonal Statistics, rasterstats `std`, exactextract `stdev`, Earth Engine `stdDev`: all population. Each local choice is documented. | Two cells with values 1 and 3: cell statistics gives 1.0, zonal statistics gives 1.414. | Name the divisor in the output name or add a `normalization` prop as `GPURasterFocalPairs` has. Zonal statistics is the one that differs from the GIS convention. |
| raster-imagery-08 | inconsistent | CONFIRMED (by reading) | `raster-embeddings/gpu-raster-embedding-knn-classifier.ts:205-206` against `raster-embeddings/gpu-raster-embedding-similarity.ts:218-219` and `gpu-raster-embedding-change.ts:207` | In the k-nearest classifier, cosine similarity with a zero-length vector is 0 in every case (distance 1). | Published `GPUSimilaritySearch`, followed by the similarity and change contributors: two zero vectors have similarity 1, exactly one zero vector gives 0. | Zero pixel vector and a zero training vector: distance 1 in the classifier, similarity 1 (distance 0) in the other three places. | Use the published rule in the classifier. |
| raster-imagery-09 | inconsistent | CONFIRMED (by reading) | `change-detection/change-detection-wgsl.ts:18-20`; used at `change-detection/gpu-change-detection.ts:361`, `:484`, `:563`, `:676` | A sample is missing only when it is NaN. Infinities are values. There is one mask word per cell, not per observation. | Every contributor of `raster-time-series` and `remote-sensing`: a sample is missing when it is not finite or its per-observation `validity` is zero. Contract section 7. | A stack with one `+Infinity` slice: Theil-Sen and the t-test use it (the t-test returns NaN for the cell), where the temporal contributors drop that slice. A per-date cloud mask cannot be passed. | Take the optional per-observation `validity` view and use the shared finite test. |
| raster-imagery-10 | doc-mismatch | CONFIRMED (by reading) | `raster-time-series/gpu-raster-cold-monitor.ts:144` (class documentation) against `:476-486` (kernel) | The change test uses the mean cosine of consecutive residual vectors, at least `cos 45 deg`. | COLD (Zhu et al. 2020) uses the mean included angle, below 45 degrees. The class documentation says "mean included angle" and does not list the cosine form among its departures. The user page describes the cosine correctly. | Consecutive angles 30, 30, 30, 30, 100 degrees: mean angle 44 (COLD confirms the break), mean cosine 0.658 (this kernel does not). | Say "mean cosine" in the class documentation and add it to the list of departures, or average `acos` of each cosine. |
| raster-imagery-11 | untested | SUSPECTED | `raster-time-series/gpu-raster-cold-monitor.ts` (whole kernel) | A simplified COLD monitor. | pyxccd `cold_detect`. The fixture generator states that pyxccd was not run and the fixture "stays pending"; the reference is a NumPy version written by the same authors from the papers. | None found. The stability test, the coefficient schedule (4, 6, 8 at 12, 18, 24 observations) and the break magnitude were checked by hand against the papers. | Generate the pending pyxccd fixture on a machine that can install it. Done in the closing pass (2026-11-11): one defect found in the stability test and fixed; see the end of the Resolution section. |
| raster-imagery-12 | untested | SUSPECTED | `raster-stretch/gpu-raster-stretch.ts` (whole contributor) | Linear, percentile and equalize stretch, gamma, sigmoid, palette. | The only oracle is `test/gpu-raster/raster-stretch/raster-stretch-oracle.ts`, a CPU copy of the same formulas. No rasterio, rio-color or scikit-image fixture. | None beyond findings 05 and 06. Gamma and sigmoid were checked by hand against rio-color. | Add a NumPy or scikit-image fixture for the percentile bounds and for equalization. |

### raster-imagery-01: a constant equal to the sentinel becomes no-data

`GPURasterConditional` reads `aValue` and `bValue` from a view or from the parameters, picks one,
and then runs `isNoDataValue(chosen)` (line 176). `isNoDataValue` includes `value == noDataValue`.
The default `constantB` is 0 and the default `constantA` is 1. With `noDataValue: 0`, a common
sentinel for integer imagery, every cell where the condition is false becomes NaN. With
`noDataValue: 255` and `constantA: 255` every true cell becomes NaN. `GPURasterArithmetic` does the
same for `bRaw` when no `b` view is bound (line 131 to 133): `maximum(a, 0)` with sentinel 0 returns
NaN everywhere.

Confirmed with `conditional.mjs`, a direct replica of the two kernel expressions. The existing specs
use `noDataValue: -9999` and never a constant equal to it, and the TypeScript oracle
(`test/gpu-raster/raster-algebra/raster-algebra-oracle.ts:336`) copies the same test, so nothing
pins the right behaviour.

### raster-imagery-02: the Whittaker error scales with the size of the values

The kernel factors `W + lambda D'D` and solves for the raw values. In float32 the diagonal holds
`w + 6 lambda` for `d = 2`, so the data weight is a small part of a large number, and the right-hand
side carries the full magnitude of `y`. The rounding error is proportional to that magnitude times
`lambda`.

`whittaker.mjs` replicates the kernel arithmetic with `Math.fround` (regular axis, no reweighting)
and compares with the same code in binary64. Largest absolute difference, 256 slices, `d = 2`:

| `lambda` | values near 0.5 | values near 300 | values near 3000 | near 300, reference subtracted first |
| --- | --- | --- | --- | --- |
| 100 | 1.1e-5 | 5.8e-3 | 5.8e-2 | 1.7e-5 |
| 1000 | 9.1e-5 | 5.6e-2 | 5.6e-1 | 2.2e-5 |
| 5000 | 3.7e-4 | 2.3e-1 | 2.3 | 4.2e-5 |
| 1e4 | 9.5e-4 | 6.6e-1 | 6.6 | 1.7e-4 |
| 1e5 | 3.5e-3 | 2.3 | 23 | 3.0e-4 |

The signal amplitude is 0.3 in every column. The user page already says that `lambda = 5000` agrees
within 2.5e-2 on the fixture and that accuracy degrades with `lambda`. What it does not say: the
error depends on the offset of the data (land surface temperature in kelvin, reflectance scaled by
10,000), the output is still status 0, and the accepted cap of 1e7 is past the point where the
factorisation returns status 2 (`d = 2` at 1e7, `d = 3` at 1e6). A GPU that fuses multiply and add
will give slightly different numbers; the cancellation stays.

Because every difference operator of order one or more annihilates constants, subtracting one
reference value per cell and adding it back is exact. It removes the offset term at no cost.

### raster-imagery-03: sliver overlaps in the conservative weights

`remap.mjs` replicates the prepare kernel on one axis. For exactly nested grids (ratio 3, spacings
0.1 and 0.3) the true overlap with the source cell after the last nested one is zero. The kernel
finds a positive overlap wherever the float32 products `f32(3414) * f32(0.3) + f32(0.3)` and
`f32(10245) * f32(0.1)` round to different values. The size is one float32 step of the coordinate,
so it grows with the index: 1.2e-3 of a source cell near column 10,000, about 3e-3 for a global
0.01 degree grid. No true overlap is ever missed (the candidate window has one spare cell).

Effects: weights are wrong by the same relative amount; and with a validity mask a destination cell
whose real sources are all invalid can still report a covered area and a value. Spacings 0.01 and
0.05 behave the same way (2404 of 4000 columns). Spacings 30 and 90, 0.1 and 0.25, and 1/3 and 1
showed no sliver in 4000 columns, so the effect depends on how the two spacings round.

### raster-imagery-04: angle resolution

The similarity and the threshold mask are fine. Only the `angle` output is affected: it reads 0 for
any pair closer than about 5e-4 rad (0.03 degrees). Decoded AlphaEarth vectors are quantized to 8
bits, so real year-to-year angles are usually larger. Low priority. Confirmed with `misc.mjs`: a
64-band vector with one component moved by 1e-5 to 1e-3 gives angle 0 in the float32 replica, where
the exact angles are 4.4e-6 to 4.4e-4 rad.

### raster-imagery-05 and 06: stretch against the published contrast and percentile kernels

Both rules of `GPURasterStretch` are defensible and documented ("maps through the histogram CDF",
"within one bin width"). They are not the rules of the published `GPURasterContrast` and
`GPURasterGlobalPercentile`, so the same raster stretched through the two paths differs. The
equalization difference is the larger one. For a raster with few distinct values the local mapping
of a middle class moves between 0.33 and 0.67 with its position inside a bin, because a spike of
identical values is treated as spread evenly across the bin. The published rule depends only on
ranks. Confirmed with `misc.mjs`.

## 3. Verified correct

Raster time series:

- `GPURasterWhittakerSmoother`: penalty operator, band storage, LDL factor and both substitutions
  derived by hand; the irregular operator is `d!` times the divided difference scaled by the mean
  spacing and reduces to ordinary differences. Bisquare weights use 6 times the median absolute
  residual (Cleveland). Precision is finding 02.
- `GPURasterHANTS`: design columns (mean, cosine, sine per harmonic), period in slices or years,
  one-at-a-time rejection as documented (not the all-at-once rule of GRASS `r.hants`; stated).
- `GPURasterHarmonicRegression` and `getRasterHarmonicTrendDesignWGSL`: column order, centred years,
  residual standard deviation on `n - k`.
- `getRasterLeastSquaresWGSL` (`remote-sensing/raster-kernel-utils.ts:70-123`): modified
  Gram-Schmidt with a second pass, back substitution, rank test relative to the largest column norm.
- `getGPURasterDateAxisDeclarations`: 64-bit midpoint with carry matches the CPU `BigInt` version;
  365.2425-day year used everywhere.
- `GPURasterBFASTMonitor`: against the `strucchange` OLS-MOSUM definition used by `bfastmonitor`:
  `sigma` on `n - k`, window `floor(h n)` reaching into the history, scale `sigma sqrt(n)`, boundary
  `lambda sqrt(2 log+(t / n))` with `log+ = 1` up to `e`, magnitude as the median monitoring
  residual.
- `GPURasterCOLDMonitor`: initialization score, coefficient schedule, outlier skip and break
  magnitude against Zhu et al. 2020 (see findings 10 and 11).
- `GPURasterPhenologyMetrics`: crossings, interpolation, trapezoid integral, second-season rule as
  documented.
- `GPURasterStreamingAccumulator`, `GPURasterClimateAnomaly`: Welford update, fixed-point exponent
  bounds (`4 M^2` for the moment), sample standard deviation (`n - 1`, stated).
- `GPURasterStandardizedIndexCalibration`: Thom gamma estimator and the method-of-moments
  log-logistic fit match `xclim` `_fit_start` term for term.
- `GPURasterStandardizedIndex` and `RASTER_REGULARIZED_GAMMA_TAILS_WGSL`: series and Lentz continued
  fraction match Numerical Recipes `gser` and `gcf`; the smaller tail is used; zero handling and the
  8.21 clip follow `xclim`.
- `GPURasterClimateRunAccumulator`, `GPURasterRollingMean`: spell, exceedance and degree-day rules;
  any missing sample in the window gives a missing mean.
- `GPURasterLaggedCorrelation`: lag direction as documented, pairwise-complete Pearson.
- `GPURasterTemporalResample`: bracketing merge, exact hit on the last observation, gap rule.
- `GPURasterMedoidComposite`, `GPURasterGeometricMedianComposite`: hdmedians `nanmedoid` definition;
  Weiszfeld recursion with the epsilon floor.
- `GPURasterBestAvailablePixel`: the documented score. Its cloud term is a linear ramp, not the
  sigmoid of Griffiths et al. 2013; the documentation gives the formula used.
- `GPURasterSnapshotCovariance`, `getGPURasterSnapshotEOFDecomposition`, `GPURasterEOFProjection`:
  Gram matrix of weighted anomalies, Jacobi rotation against Numerical Recipes, explained variance
  `s^2 / (T - 1)`, `u = sqrt(w) X v / s`.

Remote sensing:

- `GPURasterTexture`: against scikit-image `graycomatrix` and `graycoprops`: symmetric counting,
  normalisation by the doubled count, contrast, dissimilarity, homogeneity, ASM, energy, correlation
  (1 for a constant window), natural-log entropy.
- `GPURasterTemporal`: running mean, stable sort, median of an even count, earliest tie for the
  extrema; Savitzky-Golay as a least-squares refit evaluated at the centre, equal to SciPy
  `savgol_filter` (`mode='interp'`) on complete data.
- `GPURasterSpectralUnmixing`: QR least squares; projected gradient with step `1 / ||A||_F^2`;
  simplex projection matches the sort-and-threshold algorithm of Duchi et al.
- `GPURasterSeasonalBreak` and `createExtendedSeasonalBreakNodes`: BIC `n ln(RSS / n) + k ln n`
  with two parameters charged for a break, dummy seasons with season zero as reference, cyclic
  coordinate descent, interior-only interpolation.
- `GPURasterRegistration` (periodic): cross-power `R conj(M)`, shift sign ("apply to moving"),
  wrap rule and frequency order equal scikit-image `phase_cross_correlation`; the refinement DFT
  uses the same sign and `1 / N` as the published inverse `GPUFFT2D`.
- Masked registration: the six correlation channels give the masked normalised cross-correlation of
  Padfield; refinement uses bilinear resampling and a Welford covariance.
- `GPURasterRegionGraph`, `addRasterRegionGraphTiles`: four-neighbour shared-edge counts, each edge
  once, count-weighted merged means, tie rule.
- `GPURasterConservativeRemapping`: the three normalisations. Precision is finding 03.

Change detection (`GPUChangeDetection`):

- Welch t and Welch-Satterthwaite degrees of freedom; two-sided p as `I_x(df / 2, 1 / 2)` with
  `betacf` and a Stirling log-gamma (same formula as the shared `getLogGammaValue`).
- Theil-Sen: the heap holds the `P / 2 + 1` smallest slopes; odd and even pair counts checked by
  hand.
- Mann-Kendall: `S`, tie term per first occurrence, variance over 18, continuity correction,
  `erfc(|z| / sqrt 2)`; `u32` range holds for 256 slices.
- Difference, log ratio, percent change, change-vector magnitude and `atan2` direction with the zero
  vector guarded.

Raster embeddings:

- `GPURasterEmbeddingDequantize`, pyramid level: `sign(q) (q / 127.5)^2`, `-128` as NoData,
  unit-vector mean.
- `GPURasterEmbeddingSimilarity`: cosine and zero-vector rule as the published search.
- `GPURasterEmbeddingLinearProbe`: stable softmax, lowest class on ties.
- `GPURasterEmbeddingKNearestClassifier`: insertion keeps the lower sample on equal distance, vote
  tie to the smallest class (scikit-learn). Zero-vector rule is finding 08.
- `GPURasterEmbeddingCovariance`: two-pass centred covariance, divisor `n - 1`.
  `getRasterEmbeddingPrincipalComponents`: Jacobi, sign rule of scikit-learn `svd_flip`.
- `GPURasterEmbeddingProjection`, `GPURasterEmbeddingZoneMean`: projection, coverage-weighted mean,
  fixed-point bounds.
- `GPURasterEmbeddingNearestPixels`, `GPURasterEmbeddingKMeans`: delegate (section 5).

Stretch, algebra, zonal, patches, focal pairs:

- `GPURasterStretch`: exact minimum and maximum through ordered keys, histogram binning with the
  maximum in the last bin, gamma `t^(1 / gamma)`, sigmoid equal to rio-color, palette lookup.
- `GPURasterArithmetic`: all 16 operations and their domain rules; normalized difference is NaN when
  `a + b == 0`. `GPURasterConditional`: comparators, inclusive `between`. Both subject to finding 01.
- `GPURasterReclassify` and `getBreakSearchWGSL`: left-closed class equals `numpy.digitize`;
  right-closed as documented.
- `GPURasterCellStatistics`: centred population standard deviation, majority and minority with the
  smallest value on ties (ArcGIS returns NoData on a tie; stated), variety.
- `GPUWeightedOverlay`: linear and table remap, restricted class, normalisation by the sum of
  absolute weights.
- `GPURasterZonalStatistics`: counts, exact extrema, two-pass variance; the weighted form equals
  `numpy.cov(aweights, ddof=1)`; spherical cell area `2 R^2 dLon cos(mid) sin(dLat / 2)` with the
  authalic radius from `contracts/earth.ts`.
- `GPURasterPatchMetrics`: area as count times `|det|`, perimeter faces weighted by the correct
  affine step, bounding box.
- `GPURasterSieve`: threshold `<`, largest large neighbour, lowest label on ties; the documented
  differences from `gdal_sieve` (single pass, no chains, isolated small patches removed).
- `GPURasterFocalPairs`: reflect without edge duplication, anchored two-pass moments, slope,
  intercept, `r`, `R^2`, sample or population covariance.

## 4. Standards consistency

Two conventions for one thing:

- Standard deviation divisor: `n` in cell statistics, `n - 1` in zonal statistics, climate anomaly
  and embedding covariance, selectable in focal pairs (finding 07).
- Missing-value test: finite plus per-observation `validity` in `raster-time-series`,
  `remote-sensing` and `raster-embeddings`; NaN only plus a per-cell mask in `change-detection`
  (finding 09); NaN or a sentinel, with infinities as values, in `raster-algebra` and
  `raster-stretch`; band validity in zonal statistics and focal pairs. Only the last form is
  contract section 7. The library review already lists this.
- Percentile and equalization rules: local stretch against the published kernels (findings 05, 06).
- Cosine with a zero vector (finding 08).
- Median: four private implementations (insertion sort in `remote-sensing/gpu-raster-temporal.ts:186`,
  `gpu-raster-bfast-monitor.ts:226` and `gpu-raster-cold-monitor.ts:378`; heap sort in
  `gpu-raster-whittaker-smoother.ts:227-247`). All use the mean of the two middle values. Same
  result, four copies.
- Period units: years with a date axis and slices without, stated on every prop. Consistent.
- Year length: 365.2425 days in `raster-irregular-date-axis.ts:20` and again as a separate literal
  in `gpu-raster-best-available-pixel.ts:34`. Same value.

Private copies of something `analysis-core` has:

- `change-detection/gpu-change-detection.ts:570-620`: Theil-Sen heap selection. Same algorithm as
  `analysis-core/kernels/wgsl-theil-sen-median.ts`; the local one skips NaN in place and can divide
  by elapsed years. Numerically the same on a complete regular series.
- `change-detection/change-detection-wgsl.ts:31-44`: `logGammaPositive`. Identical formula to
  `getLogGammaValue` in `incomplete-gamma-wgsl.ts`.
- `change-detection/change-detection-wgsl.ts:12-21`: `quietNan` and `isNanValue`, copies of the
  `wgsl-common.ts` helpers. `raster-algebra/raster-algebra-utils.ts:32` has a third `isNaNValue`.
- `raster-zonal-statistics/gpu-raster-zonal-statistics.ts:1138-1141` and `:1201-1203`:
  `encodeOrderedFloat` and `decodeOrderedKey`, bit for bit the shared `GET_ORDERED_KEY_WGSL` and
  `DECODE_ORDERED_KEY_WGSL` that `raster-stretch` and `gpu-weighted-overlay` import.
- `raster-algebra/raster-algebra-utils.ts:116-136` and `raster-stretch/gpu-raster-stretch.ts:517-531`:
  two private binary searches; `analysis-core/kernels/binary-search-wgsl.ts` exists.
- CPU Jacobi eigensolver twice: `raster-time-series/raster-snapshot-eof-decomposition.ts:118-153`
  and `raster-embeddings/raster-embedding-principal-components.ts:57-97`. Same rotation and the
  same sign rule; stopping thresholds differ (1e-28 against 1e-30, 64 against 60 sweeps).
  `analysis-core` has only the WGSL solver. One CPU helper in `analysis-core` would serve both.
- `getRasterLeastSquaresWGSL` lives in `remote-sensing/raster-kernel-utils.ts` and is imported by
  four files of `raster-time-series` (HANTS, harmonic regression, BFAST, COLD) and four of
  `remote-sensing`. It is a cross-family numeric and belongs in `analysis-core` next to
  `dense-solve-wgsl.ts` (a different method; no numeric conflict).

Used correctly from `analysis-core`: the fixed-point accumulator (streaming sums, snapshot
covariance, zone means, region means), `GET_NORMAL_QUANTILE_WGSL`, `ERFC_VALUE_WGSL`, the shared
log-gamma inside the gamma tails, `GPU_EARTH_AUTHALIC_RADIUS_METERS`, the spherical cell area and
`getSortedSegmentSumNodes`. No kernel in scope draws random numbers.

Shader literals: `remote-sensing` bakes numeric props into WGSL (texture range, merge threshold,
period, tolerances). The library review lists this; no numeric error was found in the generated
literals.

Contract section 6: `GPURasterPatchMetrics` reports area and perimeter in CRS units with no
`cellSizeMode`, so a longitude and latitude grid gives square degrees. Stated in the output
documentation; listed in the library review.

## 5. luma.gl official implementations

Delegates to the published class:

- `GPURasterEmbeddingNearestPixels`: `GPUSimilaritySearch` (`gpu-vector-search`), metric `cosine`.
  The published kernel excludes rows with a non-finite component, as the local documentation says.
- `GPURasterEmbeddingKMeans`: `GPUKMeans` (`gpu-vector-search`). Arguments keep their meaning.
- `GPURasterRegistration`: `GPUFFT2D` (`gpu-core`) for all transforms. Forward sign and inverse
  `1 / N` match the local refinement.
- `GPURasterStretch`: `GPUScan` for the CDF. `GPURasterZonalStatistics`: `GPUGroupAggregation` for
  the atomic path and takes the published `GPURasterBand`.
- `GPURasterPatchMetrics` and `GPURasterSieve` consume labels of the published
  `GPURasterConnectedComponents` and `GPURasterDenseComponents`.
- Outside the scope directories: `GPURasterNDWI`, `GPURasterNBR`, `GPURasterEVI` and `GPURasterSAVI`
  are built from the published `GPURasterBandMath`, with the usual constants (2.5, 6, 7.5, 1;
  `L = 0.5`).

Re-implements something an official entry point publishes:

| Local | Official | Agreement |
| --- | --- | --- |
| `GPURasterArithmetic` (add, subtract, multiply, divide, normalized difference) | `GPURasterBandMath`, `GPURasterNDVI` (`gpu-raster`) | Same zero-denominator rule: invalid when the denominator is 0 (official: `abs(d) <= epsilon`, default 0). The official kernel also rejects a non-finite denominator and writes output validity; the local one writes NaN and has finding 01. The official one is the better base; the local extras are minimum, maximum, power, the unary set and the clamp. |
| `GPURasterConditional` (threshold comparisons) | `GPURasterThreshold` (`gpu-raster`) | Same comparators; `between` equals the official inclusive `range`. The local one selects between two rasters, which the official one does not. |
| `GPURasterStretch` minimum, maximum and count | `GPURasterStatistics` | Same exact extrema. The local comment gives the reason (window, region mask, NaN when empty). |
| `GPURasterStretch` histogram | `GPURasterHistogram` | Same equal-width bins with the maximum in the last bin, by reading the local side only; the official binning kernel was not compared line by line. |
| `GPURasterStretch` percentile bounds | `GPURasterGlobalPercentile` | Different rank rule and different position inside the bin (finding 06). Neither is wrong; the official one returns a bin centre, the local one interpolates. |
| `GPURasterStretch` gamma | `GPURasterContrast` | Same convention, `t^(1 / gamma)`. |
| `GPURasterStretch` equalize | `GPURasterContrast` equalize | Different mapping (finding 05). The official rule is the conventional one. |
| `GPURasterPatchMetrics` pixel count and area | `GPURasterRegionMeasurements` | Same formula (count times the absolute determinant of the affine). The local one adds perimeter and bounding box. |
| `GPURasterEmbeddingSimilarity`, `GPURasterEmbeddingChange` (per-pixel cosine) | `GPUSimilaritySearch` (top-k search) | Same zero-vector rule. The official kernel divides each vector by its largest component before the dot product, so it cannot overflow or underflow; the local kernels form `sqrt(dot(a, a) * dot(b, b))` directly, which is safe only for vectors of moderate length (true for unit embeddings). Different output shape, so not a candidate for delegation. |
| `GPURasterEmbeddingKNearestClassifier` cosine distance | `GPUSimilaritySearch` | Differs for two zero vectors (finding 08). The official rule is the one to keep. |

Nothing to delegate to:

- `gpu-timeseries` publishes only the `GPUTimeSeries` container and `GPUTimeSeriesEnvelope`. No
  reduction, trend or smoother exists upstream. The local temporal contributors take a cell-major
  view with a slice count instead of a `GPUTimeSeries` (uniform `origin` and `interval`); that is a
  shape difference, not a numeric one.
- `GPURasterConvolution`, `GPURasterMorphology` and `GPURasterOtsuThreshold` have no local
  counterpart in these directories. `GPURasterTexture` and `GPURasterFocalPairs` scan their own
  windows; upstream has no co-occurrence or paired-moment kernel.

No import of `_internal` or deep `dist` paths was seen in the scope.

## 6. Test oracles

Independent reference library, run to produce a fixture:

- `GPUChangeDetection`: SciPy `ttest_ind`, `theilslopes`, `kendalltau`, `norm`
  (`parity-wp4-change-detection.py`, `parity-wp4-change-modes.py`).
- `GPURasterStandardizedIndex` and calibration, climate run accumulator, rolling mean, climate
  anomaly: `xclim` 1.0.0 (`rts06-climate-indices-xclim.py`).
- `GPURasterTexture`: scikit-image `graycomatrix` and `graycoprops`. `GPURasterRegistration`:
  scikit-image `phase_cross_correlation` and SciPy `fourier_shift` (`raster-round-two.py`); masked
  mode: SciPy `map_coordinates` (`raster-registration-regions.py`).
- `GPURasterRegionGraph`: NetworkX and scikit-image graph.
- `GPURasterTemporal`: SciPy `savgol_filter`. `GPURasterSpectralUnmixing`: SciPy `nnls` and
  `minimize`, and published Orfeo Toolbox golden data (`raster-otb-unmixing.py`).
- `GPURasterConservativeRemapping`: ESMF 8.9.1 (`raster-esmf.py`). Small grids only; finding 03 is
  outside its range.
- `GPURasterZonalStatistics`: SciPy `ndimage`, NumPy `cov`, `quad` for the spherical area.
- `GPURasterSieve`, `GPURasterPatchMetrics`: scikit-image `regionprops`, `remove_small_objects`,
  `RAG`.
- `GPURasterArithmetic`, `GPURasterCellStatistics`, `GPUWeightedOverlay`, `GPURasterReclassify`:
  NumPy ufuncs, `digitize`, `clip` (`parity-wp4-raster-values.py`, `parity-wp4-raster-final.py`).
  None of the cases uses a constant equal to the sentinel (finding 01).

Independent implementation in NumPy or SciPy, written by the project from the published definition
(no reference package run):

- `GPURasterWhittakerSmoother` (SciPy `solve_banded`; largest `lambda` 5000, tolerance 2.5e-2,
  values of order one), `GPURasterHANTS` (`rts02-whittaker-hants-scipy.py`).
- `GPURasterHarmonicRegression`, `GPURasterBFASTMonitor`, `GPURasterCOLDMonitor`
  (`rts04-change-monitoring-numpy.py`; bfast and pyxccd were not run).
- Medoid, geometric median (also checked against `scipy.optimize.minimize`), best available pixel,
  phenology metrics (`rts05-compositing-numpy.py`).
- Snapshot covariance, EOF projection, lagged correlation (`rts07-snapshot-eof-numpy.py`, NumPy
  SVD).
- All embedding contributors, including the overview rule of `GPURasterEmbeddingPyramidLevel`
  (`rts03-embedding-rasters-numpy.py`).
- `GPURasterSeasonalBreak` (`raster-seasonal-extended.py`, NumPy least squares),
  `GPURasterFocalPairs` (`raster-focal-pairs.py`).

Hand-computed values in the spec:

- `GPURasterTemporalResample` (`rts02-whittaker-hants.spec.ts:172-215`, five points interpolated by
  hand).
- `GPURasterStreamingAccumulator` (`tc07-streaming-accumulator.spec.ts`, binary64 sums formed in the
  test, compared with the fixed-point result and with a naive float32 sum).

Only a CPU copy of the same formula:

- `GPURasterStretch` (`raster-stretch-oracle.ts`), finding 12.
- `GPURasterConditional`: the only expected values found are from `raster-algebra-oracle.ts`, whose
  line 336 repeats the sentinel test on the chosen constant.

Not determined: which spec pins `GPURasterEmbeddingLinearProbe` and the k-nearest classifier beyond
the shared NumPy fixture; spec bodies were not read file by file.

## Resolution (2026-11-10)

Outcome of the 12 findings: 11 fixed (two of them documentation only), 0 not reproduced, 1 deferred
(finding 11). One `luma-official` item is also deferred (below).
Every `fixed` row with a kernel change was first reproduced on the headless GPU with the new spec
(the number seen before the fix is in the "before" column); findings 07 and 10 are words only. Source paths are relative to `modules/analysis/src/raster-analysis`,
spec paths to `modules/analysis/test/gpu-raster`.

| ID | Outcome | What changed | Spec that pins it | Before, on the GPU | Behaviour change a caller sees |
| --- | --- | --- | --- | --- | --- |
| raster-imagery-01 | fixed | `raster-algebra/gpu-raster-conditional.ts:176-183` and `raster-algebra/gpu-raster-arithmetic.ts:136-138`: the sentinel test runs on each operand only when it is read from a view; a constant is tested for NaN only. Prop documentation says so. The TypeScript oracle copied the wrong rule and is fixed (`raster-algebra/raster-algebra-oracle.ts`, `conditionalOnCPU` and `arithmeticOnCPU`). | `raster-algebra/gpu-raster-local-operations.spec.ts` "keep a constant that equals the sentinel" (sentinel 0, hand-written expectations) | `Con(x > 10, 1, 0)`, `noDataValue: 0`, cells `[5, 20, 0, NaN]`: `[NaN, 1, NaN, NaN]` | Now `[0, 1, NaN, NaN]`. `maximum(a, 0)` with `noDataValue: 0` on `[-3, 7]`: was `[NaN, NaN]`, now `[0, 7]`. A raster cell equal to the sentinel is still nodata. |
| raster-imagery-02 | fixed | `raster-time-series/gpu-raster-whittaker-smoother.ts:266-304` (`insertRow`), `:312-330` (per-cell mean subtracted), `:336-389` (solve), `:416` (mean added back). Two changes: the solve runs on values minus their per-cell mean, and the float32 band LDL of `W + lambda D'D` is replaced by a banded Givens QR of `[sqrt(W); sqrt(lambda) D]` (rows inserted in order of first column, so the band is kept and the cost stays `O(slices * order^2)`). The mean alone was not enough: the measured error at offset 0.5 was already 4e-3 (`d = 2`, `lambda = 1e5`) and 5.6e-2 (`d = 3`, `lambda = 1e5`) against an amplitude of 0.3. The cap `MAXIMUM_WHITTAKER_SMOOTHING_PARAMETER` stays `1e7` (`:41`) because it now works for every order; status 2 is now "a diagonal of the triangular factor is zero or not finite, or the solution is not finite" (the first version of this fix also raised it below 1e-6 of the largest diagonal; removed, see Review follow-up). | `raster-time-series/rts02-whittaker-precision.spec.ts` (9 cases: orders 1 to 3, offsets 0.5, 300, 3000, `lambda` 1e2 to 1e7, 256 slices, against a dense binary64 Cholesky of the reference system; a constant series; a singular system). The SciPy fixture tolerances in `rts02-whittaker-hants.spec.ts` are tightened from 1e-3 to 2.5e-2 down to 2e-5. | Series at 300 +- 0.3, `d = 2`: 0.37 off at `lambda = 1e4`, 14 off at 1e6, status 0; `lambda = 1e7` returned values of order 3e9 with status 0. `d = 3`: 2.3 off at 1e4, status 2 from 1e6. | Error is now one float32 step of the published value (1.5e-5 at 300, 1.2e-4 at 3000, below 2.3e-5 around 0.5) for every order up to `lambda = 1e7`. Results change in the last digits at small `lambda` and materially at large `lambda` or large offsets. `lambda = 1e6` and `1e7` with `d = 2` or `3` now return status 0 and a curve where they returned status 2 or garbage. |
| raster-imagery-03 | fixed | `remote-sensing/gpu-raster-conservative-remapping.ts:30-182` (axis plan and WGSL helpers) and `:384-399` (prepare kernel). Edges are placed in source cells from integers. Where the spacing ratio and origin offset are rational with a denominator of at most 4096 the edges are exact integers of `1 / denominator` cell; otherwise whole cells plus a `2^-32` fraction (64-bit integer product), and overlaps of at most `max(1e-6 of the smaller cell, width * 2^-31)` source cells are dropped. Class documentation and the user page now say what "exact" covers. | `remote-sensing/conservative-remapping-nested-grids.spec.ts` (5 cases: 0.1 into 0.3 with a validity mask, 0.3 into 0.1, 0.1 into 0.25 shifted by half a cell, and two incommensurate grids against binary64 interval overlaps). ESMF, `remote-sensing.spec.ts`, `round-two.spec.ts` and the lens spec still pass. | 0.1 into 0.3, 4000 columns, `covered-area`: 456 of the 2000 destination columns whose three source cells are invalid published a neighbour's value with status 0 (the report predicted 912 of 4000 columns with a sliver; half of them are in the masked half). 0.1 into 0.25 shifted by half a cell: 1.8e-3 off on values 2 to 8. Incommensurate, 6000 columns: 3.3e-3 off on values 1 to 5. | Those cells are now NaN with status 1 and `coveredArea` 0. Weights on large grids change by up to about 1e-3 of a cell towards the binary64 value. A candidate slot with no overlap now holds index `0xffffffff` in `preparedWeights.indices` (it held a real index with area 0). New construction error when the destination extent in source cells from the source origin reaches 2^31. |
| raster-imagery-04 | fixed | `raster-embeddings/gpu-raster-embedding-change.ts:217-234`: `angle = 2 * guardedAtan2(|u - v|, |u + v|)` of the two unit vectors; `acos` only when a vector is zero (0 or `pi / 2`, as before). | `raster-embeddings/rts03-embedding-rasters.spec.ts` "resolves angles far below the float32 cosine step" (binary64 chord formula, and closed forms `atan(1e-4)`, `pi / 2`, `pi`); the NumPy fixture tolerance on `angle` is tightened from 5e-4 to 3e-6. | Exact angle 9.92e-4 rad read 1.09e-3 (10 percent off). The 9.9e-5 and 9.9e-6 rad pixels were not read separately on the old kernel (the spec stops at the first failure); their float32 cosine is 1. | `angle` is accurate to about 4e-7 rad absolute. `similarity`, `magnitude` and `changed` are unchanged. |
| raster-imagery-05 | fixed | `raster-stretch/gpu-raster-stretch.ts:724-740` (equalization) and `:578-587` (count of the first occupied bin): `(cdf[bin] - cdfFirst) / (total - cdfFirst)`, the rule of the published `GPURasterContrast`. Converged on published luma.gl. Not delegated (see Upstream gaps). `raster-stretch/raster-stretch-oracle.ts` follows. | `raster-stretch/gpu-raster-stretch.spec.ts` "equalization maps the middle of three equally frequent values to 0.5" (hand-computed) | Three equally frequent values `{0, 100, 255}`: the middle one mapped to 0.5229. | The middle value maps to exactly 0.5 for any position; the smallest counted value maps to 0 and the largest to 1. Equalized output is a step function of the bin (no interpolation inside a bin), so every equalized raster changes. |
| raster-imagery-06 | fixed | `raster-stretch/gpu-raster-stretch.ts:528-563`: rank `floor((N - 1) * p)`, centre of its bin, exact extrema at `p = 0` and `p = 1` with an automatic domain: the rule of the published `GPURasterGlobalPercentile`. Converged on published luma.gl. | `raster-stretch/gpu-raster-stretch.spec.ts` "percentile bounds follow the rank rule of GPURasterGlobalPercentile" (hand-computed) | Ten values 0 to 90: `p = 0.5` gave 40.078 (measured; the report's replica gives 89.98 at `p = 0.98`, not read separately). | Now 40.034 and 80.024. `lo` and `hi` of every percentile stretch move by up to one order statistic (visible on small windows and region masks, within a bin width or so on large rasters). With an explicit domain `p = 0` and `p = 1` are the centre of the first and last occupied bin. |
| raster-imagery-07 | fixed (documentation) | Neither `GPURasterStatistics`, `GPURasterRegionMeasurements` nor `GPURasterGlobalStatistics*` publishes a standard deviation or states a divisor (grep of their `.d.ts` and `.js`), so both local divisors stay. Each class documentation and output now states its divisor, the other class's, and the two-value example: `raster-algebra/gpu-raster-cell-statistics.ts:40-43`, `:83-86`; `raster-zonal-statistics/gpu-raster-zonal-statistics.ts:156-159`, `:313-318`; `docs/api/gpu-raster/operations-analysis.md`. | None (words) | | None. Aligning the two divisors is roadmap 12.4. |
| raster-imagery-08 | fixed | `raster-embeddings/gpu-raster-embedding-knn-classifier.ts:207-209`: two zero vectors have cosine similarity 1 (distance 0), exactly one zero vector 0, as the published `GPUSimilaritySearch` and the sibling contributors. | `raster-embeddings/rts03-embedding-rasters.spec.ts` "k-nearest cosine distance between two zero vectors is zero" | Zero pixel, training set with a zero vector of class 1: label 0. | Label 1. Only pixels and training vectors that are exactly zero are affected. |
| raster-imagery-09 | fixed | `change-detection/change-detection-wgsl.ts:21-25` (`isMissingSample`: not finite) and `change-detection/gpu-change-detection.ts:104-107` (new optional prop `validity`, one row per cell and slice), `:333-340`, and every sample test in the five kernels. The Mann-Kendall tie scan no longer lets a masked-out earlier observation start a tie group (`:714`). The two-band direction uses `guardedAtan2` (`:419`, the site listed for this stream in `parallel-pass-2026-11-10/cells-ecology.md`). The CPU twin `change-detection/change-detection-cpu.ts` follows. Converged on the sibling temporal contributors. | `change-detection/gpu-change-detection.spec.ts` "treats an infinite sample as missing" and "drops observations whose per-observation validity is zero" (hand-computed Welch t, degrees of freedom, Sen slope, Mann-Kendall S and Z) | `[1, 2, Inf, 4, 5, 6]`: Mann-Kendall S 9, t statistic NaN. | S 10, t 4.5826: an infinite sample is dropped like NaN in every output (two-slice outputs against an infinite slice are NaN, where `difference` was infinite). New optional `validity` prop; nothing changes when it is absent and the stack has no infinities. |
| raster-imagery-10 | fixed (documentation) | `raster-time-series/gpu-raster-cold-monitor.ts:143-153`: the class documentation says "mean cosine", gives the 30, 30, 30, 30, 100 degree example and states that no pyxccd fixture exists. `docs/api/gpu-raster/raster-change-monitoring.md` gains the row in the table of departures. | None (words) | | None. |
| raster-imagery-11 | deferred | Untested against pyxccd, recorded in the class documentation. `pyxccd` is not importable here under either interpreter (`ModuleNotFoundError` in Python 3.15 and in the system `/usr/bin/python3` 3.9.6, which does have NumPy 2.0.2 and SciPy 1.13.1), so no pyxccd fixture can be produced on this machine. See Review follow-up. **Closing pass (2026-11-11): fixed.** pyxccd 1.1.0 was run; the stability test of the initialization window differed from pyxccd and is corrected; see "Closing pass" below. | `raster-time-series/rts04-cold-pyxccd.spec.ts` (closing pass) | Closing pass: second segment starting at slice 79 where pyxccd starts it at 70 | Closing pass: segments start earlier; see `behaviour-changes.md`. |
| raster-imagery-12 | fixed | Hand-computed cases for every mode of `GPURasterStretch`: linear, gamma, sigmoid, percentile, equalize and nearest palette (`raster-stretch/gpu-raster-stretch.spec.ts:557-632`). The CPU oracle is kept as a twin for the randomised cases and now follows the published rules. A reference-library fixture (OpenCV, NumPy) was added in the Review follow-up. | The four specs named in rows 05 and 06 plus "linear, gamma and sigmoid match hand-computed values" and "nearest palette lookup" | | None. |

### Extra findings

- Compensated sums. The one file in scope that mentions them,
  `raster-time-series/gpu-raster-streaming-accumulator.ts:152`, says it does not use them: its sums
  are the shared 96-bit fixed-point accumulator. `raster-time-series/tc07-streaming-accumulator.spec.ts`
  already compares it on the GPU with binary64 sums over 512 slices and with a naive float32 sum, and
  passes (3 of 3). Nothing to replace.
- Whittaker (found while fixing 02): subtracting the reference is not enough by itself. The float32
  LDL of the normal equations loses `lambda`-proportional accuracy on centred data too, and with
  status 0. See row 02.
- Mann-Kendall tie scan (found while fixing 09): with a per-observation mask the "seen before" scan
  must skip invalid observations, otherwise a tie group is dropped from the variance. Pinned by the
  second cell of the validity spec (`Z = 1.44463`, 1.3587 without the guard).

### From `luma-official.md`

- Section 3.3 and order-of-work item 15 (`GPURasterArithmetic` delegating five operations to
  `GPURasterBandMath`): deferred, roadmap 12.4. The item itself is conditional on the raw-value
  raster contributors taking a `GPURasterBufferBand`, which is a prop change.
- Section 3.3, `GPURasterStretch` against `GPURasterContrast` and `GPURasterGlobalPercentile`:
  conventions converged (rows 05 and 06), not delegated.

### Upstream gaps

- `GPURasterContrast` would need a statistics window and a region mask for the histogram, a
  per-frame mode, an output in `[0, 1]`, a sigmoid and a palette before `GPURasterStretch` could
  delegate its equalization to it.
- `GPURasterGlobalPercentile` reads a `GPURasterGlobalAccumulator`; `GPURasterStretch` would need it
  to accept a caller histogram and CDF (window, region mask, explicit domain) and a per-frame
  percentile instead of a compile-time literal.
- `GPURasterStatistics` and `GPURasterRegionMeasurements` publish no variance or standard
  deviation, so there is no official divisor to align cell and zonal statistics to.

### Review follow-up

1. Test typecheck. `change-detection/gpu-change-detection.spec.ts`: the two `trend.run({splitSlice: 3})`
   calls now pass complete settings (`beforeSlice: 0, afterSlice: 1, splitSlice: 3`); the trend
   outputs do not read the two slice indices, and no assertion changed. The shared watcher
   (`typecheck-status.mjs`) covers sources only, which is why it reported 0 errors. Checked with a
   scoped `tsc` on the root `tsconfig.json` over the ten spec files of this stream: 0 errors.
2. Whittaker status 2 (`fix-review-fields.md`, "Needs a second look"). The reviewer was right in
   kind. Measured on the GPU against an exact rational solve (new generator
   `test/oracles/rts02-whittaker-sparse-exact.py`, Python `fractions`, because a binary64
   factorisation of the normal equations is itself unusable at this conditioning): 2, 3 and 5 valid
   observations out of 256 slices (the kernel's slice cap; 365 is not accepted), at the start, the
   middle, the end or spread over the series, `lambda` 1, 1e2, 1e4, 1e6, 1e7, every order the
   support allows (120 cases).
   - The reviewer's example (three observations at the start, order 2, `lambda = 1e7`) returned
     status 0, 4.9e-2 off on a curve spread of 95.6.
   - Five cases returned status 2 although the system is non-singular: five adjacent observations
     at the start with order 3 and `lambda` 1e4, 1e6, 1e7, and in the middle with order 3 and
     `lambda` 1e6, 1e7.
   - With the ratio test off those five solve as well as their accepted neighbours (at most 1.24e-2
     of the spread). The ratio is not a usable signal: the penalty rows scale with `sqrt(lambda)`,
     so the largest diagonal grows with `lambda` while the solution does not get worse.
   - Change: the ratio test is removed (`raster-time-series/gpu-raster-whittaker-smoother.ts:369-377`).
     Status 2 is raised for a zero or non-finite diagonal and for a non-finite solution only. Class
     documentation and `docs/api/gpu-raster/raster-time-series.md` say so and give the accuracy on
     sparse clustered support: within one float32 step for order 1, 6e-4 of the spread for order 2
     and 1.3e-2 for order 3 across a 250-slice gap.
   - Pinned by four new cases in `raster-time-series/rts02-whittaker-precision.spec.ts` ("solves 2,
     3 and 5 observations at the ... of 256 slices for every lambda": status 0 and the error bound
     for all 120 fixture cases). The singular case (`lambda = 0` with a gap) still returns status 2.
   - Behaviour change against the first version of this fix: those five inputs return a curve with
     status 0 instead of NaN with status 2. What the code before this pass returned for them was
     not measured.
   - Not covered: a quality weight many orders of magnitude below the others. It is a valid weight
     and the system stays non-singular, but the float32 result was not measured there.
3. Documentation sweep of `docs/api/gpu-raster/*.md` (grep for the old Cholesky and LDL wording,
   "negligible", the percentile interpolation and "one bin width" wording, "histogram CDF", "exact"
   overlap wording, "included angle", "NaN means missing", `acos`, the old 2.5e-2 tolerance and the
   zero-vector, divisor and sentinel statements; the paragraphs around each hit were read). One
   contradiction remained and is fixed: the status 2 sentence of `raster-time-series.md`. Nothing
   else contradicts the fixes.
4. Reference libraries. The first Resolution was wrong to say none is importable: it only tried
   the default `python3` (3.15). Under `/usr/bin/python3` (CPython 3.9.6) NumPy 2.0.2, SciPy 1.13.1,
   scikit-image 0.24.0 and OpenCV 4.12.0 import; rasterio, rio-color, pyxccd, hdmedians and xarray
   do not. These are not the pinned versions of `test/oracles/requirements.txt` (CPython 3.14,
   NumPy 2.5.4); the fixtures record what was run.
   Closing pass (2026-11-11): the stretch and sparse Whittaker fixtures were rerun on the pinned set, every value equal, and replaced; see [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md).
   - New generator `test/oracles/raster-stretch-references.py` and fixture
     `fixtures/raster-stretch-references.json`, three new cases in
     `raster-stretch/gpu-raster-stretch.spec.ts`: equalization against `cv2.equalizeHist` on a
     24 by 16 8-bit raster (every cell within the 8-bit rounding of OpenCV's value); percentile
     bounds against `numpy.percentile(method='lower')` on 600 float values at five percentile pairs
     (each bound within half a bin of the NumPy order statistic, exact at 0 and 100); linear
     stretch, gamma 2.2 and sigmoid against a NumPy evaluation of the rio-color formula (2e-5).
   - scikit-image `exposure.equalize_hist` is importable but is not a reference for this kernel:
     it interpolates the inclusive CDF at bin centres without subtracting the first bin, which is
     the rule finding 05 moved away from.
   - Still out of reach: a rasterio or rio-color fixture (neither is installed, so gamma and
     sigmoid are pinned by the formula, not by rio-color itself), and the pyxccd fixture of
     finding 11 (closing pass, 2026-11-11: the pyxccd fixture now exists, see "Closing pass" below).
   - The hand-computed stretch cases are kept next to the fixture cases.
5. Function-scope identity table (contributor contract section 11). `raster-time-series/gpu-raster-phenology-metrics.ts:224-228`, `:251`, `:256`: without a `dateAxis` the slice time is now `f32(slice)` computed in place and no `times` table is declared; with a `dateAxis` the table filled from the axis is kept. Results are unchanged: `rts05-compositing.spec.ts` (7 of 7) and `test/utils/function-scope-table-readback.spec.ts` (4 of 4) pass.

### Not done

- `docs/api/gpu-raster/operations-analysis.md` and `raster-time-series.md`: the generated
  "Parameters" tables were not regenerated with the repository-wide generator (it rewrites every
  page). The one row whose TSDoc changed (`mode` of `GPURasterStretch`) was edited by hand and
  `doc-parameters.node.spec.ts` reports these pages in sync.
- Nothing under `website/**` or `modules/lenses/**` was touched. The lens scenarios use the linear
  stretch only and `modules/lenses/test/computation-lenses/raster-terrain/remote-sensing.spec.ts`
  still passes; any website run that shows an equalized or percentile stretch, a Whittaker curve at
  large `lambda` or a remapping with a validity mask will draw different numbers.

### Closing pass (2026-11-11)

Stream "raster" of roadmap 12.6. One item belongs to this report; the `GPUHistogram` item is in
[luma-official.md](./luma-official.md).

| Item | Outcome | Files changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| raster-imagery-11: `GPURasterCOLDMonitor` against pyxccd | fixed (one defect in the stability test); break dates, model and magnitudes verified; every other difference is a documented one and is now measured | `raster-time-series/gpu-raster-cold-monitor.ts` (the stability test, one rule, no new prop; class and settings documentation); `docs/api/gpu-raster/raster-change-monitoring.md`; new `test/oracles/rts04-cold-pyxccd.py` and `fixtures/rts04-cold-pyxccd.json`; `test/oracles/rts04-change-monitoring-numpy.py` (the `score` line and the pyxccd sentences) and its regenerated fixture; `GENERATORS.md` and `requirements.txt` (words) | `raster-time-series/rts04-cold-pyxccd.spec.ts`, 13 of 13 on the headless GPU; `rts04-change-monitoring.spec.ts`, 24 of 24 against the regenerated fixture | pyxccd 1.1.0 `cold_detect` (CPython 3.12.13, NumPy 2.5.4, SciPy 1.18.1), run at `lam=20` and `lam=0`; the binary64 NumPy reference regenerated on the pinned set (CPython 3.14.8, NumPy 2.5.4, SciPy 1.16.3) |

What was run. Eight synthetic series in pyxccd's input conventions (ordinal dates, seven integer
bands in 0 to 10000, cfmask QA), 140 to 160 irregular dates each, noise 40 per band: one break, two
breaks, none, a small break, QA-flagged clouds, an unflagged cloud in the first window, four
confirming observations, and a 30-observation ramp. The monitor runs on the five bands pyxccd
scores (green to SWIR2) with the chi-squared quantiles at 0.99 and `1 - 1e-5` for 5 degrees of
freedom, pyxccd's variogram floor (the median absolute difference of successive clear observations,
computed in NumPy) as `minimumRMSE`, and `period` 365.25 / 365.2425. The fixture also holds
pyxccd's coefficients converted to the monitor's convention (a change of origin and unit only).
pyxccd's C source at tag 1.1.0 (`src/cxx/cold.c`, `misc.c`, `defines.h`) was read to name each
difference; what ran is the published wheel.

Results on the test GPU (Apple M3 Pro, headless Chromium):

- Break dates: equal to pyxccd at both penalties and to the known dates on the seven series with
  abrupt breaks (8 breaks). This does not generalize to small breaks; see "After the independent
  review" below.
- The model: where pyxccd's last refit is on exactly the observations the monitor fits (two
  segments), the 8 coefficients of the 5 bands agree with pyxccd at `lam=0` to 0.07 and the RMSE to
  0.02, on bands of 600 to 3200. pyxccd refits only at 3% growth, so its record for the other
  segments is 1 or 2 observations old: gaps up to 2.7 and 0.35.
- Change magnitudes: within 20 of pyxccd at `lam=0` (breaks of up to 900). The gap is the refit
  schedule: the monitor's model at the break is its last 1.33 refit.
- The defect. The stability test of an initialization window added the absolute residuals of the
  first and the last observation; pyxccd takes the larger (`cold.c`: `fabs(v_slope) +
  max(fabs(v_start), fabs(v_end))`). Neither the class documentation nor the difference table named
  this as a difference. Reproduced on the GPU before the fix: segment starts 1 and 91 where pyxccd
  has 0 and 90 (small break), a second segment at 79 where pyxccd has 70 (flagged clouds), at 103
  for 100 (four confirming observations), at 86 for 85 (unflagged cloud). With the larger residual
  every start equals pyxccd's except one first segment (slice 2 for 0), which is the lookback the
  monitor does not have. The kernel now has that one rule and no option for the old one (an
  `initializationEndpointRule` prop existed for part of this pass and was removed; nothing else in
  the repository referenced it). A test of the new spec pins the pyxccd starts on three series and
  records in a comment the starts the sum gave.
- The binary64 NumPy reference follows. `rts04-change-monitoring-numpy.py`: the `score` line of
  `cold_monitor` takes the larger end residual. Before changing it the unmodified generator was run
  on the pinned set and gave the committed fixture exactly, apart from the SciPy version field
  (the committed one recorded SciPy 1.18.1). Regenerated, the fixture carries CPython 3.14.8, NumPy
  2.5.4 and SciPy 1.16.3 (SciPy supplies the chi-squared quantiles only). All inputs and seeds are
  unchanged, and so are the harmonic-regression and BFAST parts. Of the 9 COLD cases the monitor's
  own result changes in 4, as `(start, end, break)` per segment:
  - `one-break`: `(6, 39, 40), (44, 71, none)` becomes `(1, 39, 40), (40, 71, none)`; fits 18 to 10.
  - `two-breaks`: the third segment starts at 70 instead of 71; fits 12 to 11.
  - `masked-one-break`: `(2, 49, 50)` alone becomes `(0, 49, 50), (52, 71, none)`: the stretch
    after the break, which the sum never initialized, is now a segment of 6 coefficients; fits 12 to 10.
  - `lasso-shrinkage`: the segment starts at 0 instead of 1; fits 6 to 5.
  The other five are equal (`tmask-cloud-in-window`, `refit-cadence-drift`,
  `temporal-rmse-season-noise`, both sparse cases). Two toggled runs move with the rule: the Tmask
  toggle starts at 2 instead of 3, the LASSO toggle at 0 instead of 1. Every toggle still differs
  from the monitor's result the way the spec requires.
- Documented differences, now with numbers from pyxccd (the table in `raster-change-monitoring.md`):
  LASSO (pyxccd's default RMSE is 10 to 13 above the monitor's on the one-break series); Tmask (an
  unflagged cloud stays in the monitor's fit: RMSE 274 against 39.5, same break date); the tail
  (pyxccd reports a partial break with probability 16 in the last observations of one series, the
  monitor none; last-segment coefficients differ by up to 20 because the monitor also fits the
  last `consecutiveCount` observations); a gradual change (pyxccd confirms at slice 77, or 80 with
  LASSO, the monitor at 75).
- Words corrected because pyxccd says otherwise: the outlier probability is `1 - 1e-5`
  (`T_MAX_CG` 30.8562), not `1 - 1e-6`; pyxccd refits at every observation up to 24 and then at 3%
  growth, not at every observation throughout; it scores five of its seven bands; its temporally
  adjusted RMSE starts after 24 observations.

Not verified:

- No real Landsat series. pyxccd's snow and insufficient-clear procedures (fewer than 24 clear
  observations) were not run, since the monitor has no counterpart.
- No series was constructed on which the mean cosine and pyxccd's mean included angle disagree; that
  difference stays documented by its arithmetic example only.
- The temporally adjusted RMSE and the refit schedule are measured only through their combined
  effect (magnitudes, the ramp), not one at a time against pyxccd; the one-at-a-time evidence is
  still the NumPy toggles of `rts04-change-monitoring-numpy.py`.
- One adapter. A decision within float32 rounding of a threshold could differ elsewhere; the
  series were not chosen for wide margins beyond passing here.
- The old rule can no longer be run, so the starts it gave (1 and 91, 2 and 79, 0 and 103) are a
  record in a spec comment and in this section, not an assertion.
- The note of the `tmask-cloud-in-window` case in the NumPy generator says the monitor's start
  shifts and the Tmask toggle initializes at once; its numbers say the reverse (monitor 0, toggle
  2, and 0 and 3 before this pass). The spec only asserts that the two differ. Left as found.

Left for others: the website run `raster-time-series/cold-monitor` will draw segments that start
earlier once the analysis package is rebuilt.

After the independent review ([fix-review-closing-fields.md](./fix-review-closing-fields.md)),
same day. No kernel change and nothing a caller sees: words, a fixture and a test.

| Finding | Outcome | Files changed | Spec that pins it | Evidence |
| --- | --- | --- | --- | --- |
| frc-fields-02: break dates do not agree with pyxccd for small breaks | reproduced; documented with measured numbers; the refit schedule left as a documented difference, with what matching it would take | class documentation of `gpu-raster-cold-monitor.ts`; `raster-change-monitoring.md` ("Break size", the refit and variogram rows); `test/oracles/rts04-cold-pyxccd.py` (a break-size study) and new `fixtures/rts04-cold-pyxccd-break-sizes.json` | `rts04-cold-pyxccd.spec.ts`, 14 of 14: the new test runs 160 series on the GPU | pyxccd 1.1.0 at `lam=0` and `lam=20` on each series |
| frc-fields-03: the window advance is missing from the list of differences | added, with its measured size | class documentation; a row in the table of `raster-change-monitoring.md` | none (binary64 reference patched in scratch) | see below |
| frc-fields-07: the fixture runner reported the pyxccd fixture as not written | fixed | `test/oracles/GENERATORS.md` | `node scripts/run-oracle-fixture.mjs rts04-change-monitoring-numpy` and `rts04-cold-pyxccd`: every fixture `IDENTICAL`, none "not written" | a run of each |

- Reproduction of frc-fields-02 with the reviewer's scripts: at the quarter-size disturbance the
  break lists of pyxccd (`lam=0`) and the binary64 reference differ in 18 of 40 and 17 of 40 series
  (35 of 80); at half and full size in 0 of 160.
- An independent draw, committed and run on the GPU. 160 series of 150 dates, one break at a
  random slice, two revisit patterns. The size of a break is given as the change score it would
  give without noise, `sum over bands (shift / max(RMSE, minimumRMSE))^2`, over the threshold.

  | Break | Score over threshold | Shift per band over noise 40 | Series | Break dates equal to pyxccd `lam=0` | `lam=20` | Starts differ, of the equal | pyxccd exactly right | Monitor exactly right |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | Full | 15 to 24 | 6 to 22 | 40 | 40 | 40 | 4 | 40 | 40 |
  | Half | 4 to 6 | 3 to 11 | 40 | 40 | 40 | 10 | 40 | 40 |
  | Quarter | 1.05 to 1.45 | 1.6 to 5.6 | 80 | 49 | 30 | 6 | 16 | 26 |

  Of the 31 quarter-size series that differ: in 23 pyxccd confirms nothing and the monitor
  confirms a break, often late; in 8 both confirm at different slices; in none does only pyxccd
  confirm (the reviewer's draw has one). The GPU counts equal those of the binary64 reference on
  every series. The spec asserts the full and half rows exactly and the quarter row within bands
  of 4, since single borderline series may fall differently on another adapter. A first version of
  the study with 240 series (80 per size) gave 80 of 80 at half and at full size on the GPU; it
  was cut to 160 because one cell takes about 0.1 s and each series needs its own dispatch (the
  floor and the dates are per series).
- Cause, measured with the binary64 reference patched in scratch (not committed): with pyxccd's
  refit schedule (every observation up to 24, then when the count has grown by more than 3%) the
  quarter-size break dates are equal in 64 of 80 instead of 49; adding the temporally adjusted RMSE
  toggle gives 65. The other 15 were not attributed. Mechanism: after a small step pyxccd's model
  is refitted within a few observations and absorbs it; the monitor's model stays at the old
  level until the count has grown by a third, so later windows still exceed the threshold.
- Why the schedule was not changed. It is not small. In the same patched reference the fits per
  series rise from a median of 12 to 14 (at most 17) to a median of 56 to 73 (at most 81). The
  default `maximumFitCount` is 96 and the per-cell work budget allows 163 fits at 160 slices, 5
  bands and 8 coefficients, so a series with two breaks would stop with status 8, and every result
  and fit count of the cases that agree today would move. Matching pyxccd needs a fit updated per
  observation (accumulated normal equations and one 8 by 8 solve per band) or a larger work
  budget. That is stated in `raster-change-monitoring.md`.
- frc-fields-03, measured the same way: with pyxccd's advance (both ends move by one observation)
  3 of the 160 series change, all on the dense-and-sparse revisit: two starts move by 2 and 1
  slices toward pyxccd's, and one quarter-size break is dated 11 slices later, away from pyxccd's.
  Not implemented, for the reason the reviewer gives: it is a behaviour change of its own.
- The class documentation now gives the measured agreement (0.07, 0.02, 20), not the tolerances of
  the spec.
- Not verified: break sizes between 1.45 and 4 times the threshold; more than one break per
  series in the study; real data; a second adapter.
