# Focal cross-band statistics

`GPURasterFocalPairs` computes unweighted covariance, Pearson correlation and simple ordinary
least-squares regression in square raster neighborhoods. The regression is
`response = intercept + slope * predictor`; its fit statistics are descriptive. There are no
p-values, uncertainty estimates, weights, multiple predictors or geographic wraparound.

## Inputs and outputs

Provide `grid`, aligned `predictor` and `response` raster bands, `radius`, and caller-owned
`output` views. Both bands must already occupy the same grid. Buffer and texture inputs use
existing raster canonicalization: native nodata and validity are checked before scale/offset,
and nonfinite calibrated values are invalid. Only observations valid in **both** bands enter
any statistic. Values and masks can change between graph executions without recompilation;
window options and calibration metadata are graph topology.

`radius` is an integer from 0 through 8, bounding the footprint to 17 × 17 samples. Radius zero
is permitted but cannot meet the minimum of two pairs. `minimumCount` defaults to 2 and may
range from 2 through 289. `covarianceNormalization` defaults to `'sample'` (divide by `n - 1`);
`'population'` divides by `n`. The count threshold applies to all float metrics.

Every output view contains at least `grid.width * grid.height` packed rows. `pairedCount` and
`status` are required uint32 outputs. Request at least one float32 metric:

| Output | Meaning |
| --- | --- |
| `covariance` | Centered cross-product sum divided by the selected count denominator |
| `correlation` | Pearson correlation, undefined when either band is constant |
| `slope`, `intercept` | Simple OLS coefficients, undefined when the predictor is constant |
| `rSquared` | `1 - residualSumSquares / responseCenteredSumSquares`, undefined for a constant response |
| `residualSumSquares` | Sum of squared centered fitted residuals |

An undefined float result is NaN. No output buffer may alias an input or another output.
Destroying the compiled graph releases its scratch buffers and leaves borrowed inputs and
outputs owned by the caller.

## Missing pairs and borders

`noDataPolicy: 'ignore'` (default) drops a pair when either band is invalid.
`'propagate'` makes every float result undefined if any footprint position is invalid.
`propagateCenterNoData: true` additionally invalidates results when either center observation is
invalid; the default is false. `pairedCount` is always the number of participating valid pairs,
even when propagation or insufficient support makes all metrics undefined.

`borderMode` defaults to `'nodata'`: positions outside the grid are invalid, so `'ignore'`
clips the window and `'propagate'` invalidates edge windows. `'clamp'` repeats the nearest edge
sample. `'reflect'` reflects without duplicating the edge pixel, including repeated reflection
for tiny rasters; a one-cell axis maps every offset to that cell. Repeated clamp/reflect samples
count as repeated observations.

The contributor advertises `requiredHalo = radius`. For tiles, load both bands and masks over
the full halo-expanded footprint, execute there, and retain only each tile's core outputs.
Apply border conventions at the true domain boundary. Applying them at an interior tile edge
changes the neighborhood and its observation count.

## Diagnostic bits and degeneracy

`GPU_RASTER_FOCAL_PAIRS_STATUS` defines combinable per-cell bits (documented on `GPURasterFocalPairsOutput.status`; the radius limit is `MAXIMUM_FOCAL_PAIRS_RADIUS`, 8, and `minimumCount` is at most `MAXIMUM_FOCAL_PAIRS_WINDOW_CELLS`, 289):

| Bit | Name | Consequence |
| --- | --- | --- |
| 1 | `insufficientPairs` | Fewer than `minimumCount` pairs; all metrics NaN |
| 2 | `missingPair` | Required neighborhood or center invalidity; all metrics NaN |
| 4 | `constantPredictor` | Zero predictor centered sum of squares; covariance remains zero, regression/correlation/R² are NaN |
| 8 | `constantResponse` | Zero response centered sum of squares; covariance remains zero and correlation/R² are NaN |
| 16 | `nonFiniteResult` | Moment/fit overflow or nonfinite arithmetic; all metrics NaN |

With a varying predictor and constant response, the fit is valid: slope zero, intercept equal
to the response, and residual sum of squares zero. Status 8 records the undefined correlation
and R². When both bands are constant, bits 4 and 8 are set. Constant flags are evaluated only
once count and missing-pair checks permit a fit. Zero status means every metric is defined.

## Numerical method and limits

The GPU scans each footprint in fixed row-major order. A first pass accumulates differences
from the first valid pair, a second accumulates centered moments, and a third evaluates fitted
residuals directly. Each reduction corrects the remaining residual mean using only the small
centered values. This preserves centering when a shader backend reassociates subtraction into
a rounded large centroid. It avoids subtracting raw squared moments and avoids deriving
residual squares by subtracting explained variance from total variance. Pearson correlation and R² are clamped to their
mathematical ranges to absorb float32 rounding.

Arithmetic and calibration are float32. Input differences lost to float32 rounding cannot be
recovered; squared sums can underflow to zero or overflow. Zero centered predictor variance is
treated as singular, with no implicit ridge or pseudoinverse. No near-singularity threshold or
cross-device error guarantee beyond the verified fixtures is claimed.

Canonicalization schedules one pass per band; moments and output projection add two passes.
Moment scratch uses six float32 values per pixel (24 bytes), plus canonical values and validity
for each band (16 bytes per pixel). The moments pass uses five storage bindings and maximal
output projection uses seven, within the default eight-binding limit. Work is bounded by three
17 × 17 scans per cell at the maximum radius. Large-domain performance is not yet benchmarked.

## Reference and verification

The generator at `modules/analysis/test/oracles/raster-focal-pairs.py` pins **NumPy 2.5.4**
and records its SHA-256 in `fixtures/raster-focal-pairs.json`. It uses independent `numpy.cov`,
`numpy.corrcoef` and centered `numpy.linalg.lstsq` calls after explicit window extraction and
paired mask/calibration adaptation. Reference inputs are float32-rounded and the library
computations use float64. This original fixture is a composed NumPy reference.

The additional `modules/analysis/test/oracles/raster-terra.R` generator directly executes **terra 1.9.50** with **R 4.4.3** and jsonlite 2.0.0. Its `raster-terra.json` fixture compares GPU covariance, Pearson correlation, intercept, slope and paired counts for six nodata/clamp/reflect × ignore/propagate cases. Clamp/reflect use explicit padded input rasters before calling terra, then crop the core; nodata uses native terra fillvalue=NA. The generator also checks temporal stack composites. All six focal cases passed native GPU comparison.

The supported unweighted subset corresponds to
[terra focalPairs](https://rspatial.github.io/terra/reference/focalPairs.html) with Pearson
correlation and paired NA removal, and
[terra focalReg](https://rspatial.github.io/terra/reference/focalReg.html) with an intercept and
one predictor. In terra's layer order, response precedes predictor. Clamp/reflect and the
explicit singular-fit flags are this API's documented conventions.

Fixtures cover positive/negative and noisy fits, population/sample covariance, separate band
masks, calibration, native nodata, insufficient pairs, constant bands, large offsets, tiny
rasters, large-offset fractional means and maximum radius. GPU tests also cover changing inputs/masks, nonzero buffer offsets,
overflow, integer inputs, ownership and whole-domain versus halo-expanded tile results.

## Executable examples

These examples use explicit synthetic inputs and expose real GPU graph captures.

### GPURasterFocalPairs

{/* lens:GPURasterFocalPairs */}

::lens{scenario=gpu-raster.raster-focal-pairs rung=graph}
