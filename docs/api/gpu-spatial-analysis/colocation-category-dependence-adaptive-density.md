# Colocation quotient, local category dependence and adaptive-bandwidth density

STX-08, second group. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. Weights are the published `GPUSpatialWeights` CSR; the density reuses the kernels, parameter packer and grid-index gather of `GPUKernelDensity`.

## `GPUColocationQuotient`

Both categorical-association contributors capture their declared settings, category/CSR views,
CSR status/information/provenance metadata and output maps before validation. Inherited getters
are read once; later edits to caller maps cannot redirect a graph or change its category count.
The captured GPU views retain their identities and their buffer contents remain per-frame inputs.
Neither class exposes Cramer's V or a general contingency-table chi-square result; local
composition and co-location use the specific chance/tail conventions documented here.

Retain the producer's `status.overflow` and `status.candidateOverflow` views when supplying
weights to either categorical contributor. These uint32 scalar flags remain live per frame:
either nonzero flag suppresses the dependent results to NaN, and clearing both flags restores
ordinary computation on the next execution. Category row counts remain available during loss.
`status.requiredCount` is admitted only alongside `status.overflow`, which certifies whether
the final capacity was sufficient; `candidateOverflow` alone cannot certify that relationship.
A required-count-only input is rejected at construction. Without status views, callers must
supply a complete CSR.

When the producer supplies `status.complete`, retain that uint32 scalar certificate too.
A zero certificate suppresses the same dependent outputs even when both overflow flags are zero
(for example, incomplete network costs whose retained edges fit). A nonzero certificate permits
ordinary computation only while both overflow flags are zero. Recovery occurs on the next
execution when the certificate becomes nonzero and both loss flags are clear. Omitted status
provides no evidence: the caller remains responsible for supplying a complete CSR.
If the producer supplies `converged`, `iterationLimitReached` or `unresolvedCount`, verify
that the CSR is complete and omit those status views.

Co-location uses finite positive weights, including weights above 3e38. If the largest eligible
weight in a neighborhood is at least 2^64, every eligible weight is scaled by 2^-64 before summing.
This common power-of-two factor preserves the ratios and bounds the sum below 2^96 for uint32
slot counts, avoiding overflow and a subnormal reciprocal on Metal. Normal scaled weights use
an exponent-bit adjustment; extremely small relative weights can underflow in float32.
Neighborhoods whose largest eligible weight is between 2^-64 (inclusive) and 2^64 (exclusive)
retain their original accumulation order and arithmetic.

When the largest eligible weight is below 2^-64, co-location instead scales every eligible
weight up by 2^64 before summing, keeping a sole subnormal weight's share at one without an
overflowing reciprocal. Positive finite weights are recognized from their bits; subnormal
weights are raised using their integer significand, avoiding arithmetic on subnormal inputs.

Reference: Leslie and Kronenfeld (2011), "The colocation quotient: a new measure of spatial association between categorical subsets of points", Geographical Analysis 43(3). No pinned package implements it, so the specs compare with an independent binary64 evaluation of the definition.

For a row `i` of category `a`, the neighbour share `f(i, b)` is the weight on neighbours of category `b` divided by the weight on all neighbours. The chance share is `ratio(a, b) = (N_b - [a = b]) / (N - 1)`. The local quotient is `f(i, b) / ratio(a, b)`; the global quotient `CLQ(a -> b)` is the mean of `f(i, b)` over the rows of `a` that have a neighbour, divided by the same ratio. With one neighbour per row it is Leslie and Kronenfeld's `(N_{a->b} / N_a) / (N_b / (N - 1))`; Gaussian-kernel weights give the adaptive form of Wang et al. (2017). 1 is independence, above 1 attraction, below 1 avoidance.

- Neighbours are listed slots with a finite positive weight, to another row with a valid category. Rows with category `0xffffffff` (or at or above `categoryCount`) are left out. Results are NaN where a row has no neighbour or the chance share is 0.
- Outputs: `globalQuotients` (`categoryCount^2`, entry `a * categoryCount + b`), optional `localQuotients` (`rows * categoryCount`) and `categoryRowCounts`.
- At most 32 categories. The shares are summed in the shared fixed-point accumulator, so the global values do not depend on thread order.
- Incomplete weights (nonzero `weights.status.overflow` or `weights.status.candidateOverflow`) make `globalQuotients` and `localQuotients` NaN; `categoryRowCounts` does not depend on the weights and is kept.
- Suppression writes exactly `categoryCount^2` global entries and `rows * categoryCount` local entries. Excess caller storage in these outputs and in `categoryRowCounts` is preserved.
- Not implemented: the permutation test of the paper.

{/* lens:GPUColocationQuotient */}

## `GPULocalCategoryDependence`

Accepts 1–32 input categories.

Reference: Boots (2003), "Developing local measures of spatial association for categorical data", Journal of Geographical Systems 5, which defines the LICD family. Implemented: a local composition measure. Per row (`GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS`), with the row itself never counted: own-category neighbours (`likeCount`), neighbours with a category (`neighborCount`, `k`), the local share `likeCount / k` (`proportion`), the global share `N_c / N` (`expectedProportion`), the binomial upper tail `P(X >= likeCount)` for `Binomial(k, N_c / N)` and the z score of the same binomial. A neighbour is an active slot of the shared rule (positive weight, another row); the size of a positive weight is not used.

This is not the output of `licd_multi` of the R package spdep, and no parity with it is claimed (R is not installed, so it cannot be checked here). spdep counts the focal row among its own neighbours (`k + 1` trials, the focal row always a like), reports the global share as `prop_i`, and its `pbinom_like_BW` is the lower tail `P(X <= like)`. Example: 4 neighbours, 3 of them like, share 0.5. This class gives `P(X >= 3) = 0.3125` on `Binomial(4, 0.5)`; spdep's `pbinom_like_BW` is `P(X <= 4) = 0.96875` on `Binomial(5, 0.5)`.

Incomplete weights (nonzero `weights.status.overflow` or `weights.status.candidateOverflow`) make every column of `statistics` NaN, including both count columns. Exactly `rows * GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS.length` entries (six per row) are suppressed. Independent `categoryRowCounts` and excess caller storage are preserved.

Not implemented: the local configuration part (local join counts and the chi-square), the permutation p-values and the unlike-neighbour tails. The binomial test is checked against hand-computed tails (0.3125, 0.648) and a binary64 evaluation.

The inclusive upper tail uses probabilities relative to the binomial mode and normalizes their
sum, avoiding cancellation from a log-binomial coefficient accumulated from zero. A SciPy
regression with 2,190 neighbors, 1,134 like and global share 0.5 gives 0.049933762130512475;
the GPU tolerance is 2e-7 absolute, small enough to preserve its side of the 0.05 boundary.

{/* lens:GPULocalCategoryDependence */}

## `GPUAdaptiveBandwidthKernelDensity`

Reference: Abramson (1982), "On bandwidth variation in kernel estimates: a square root law", Annals of Statistics 10(4), with the two-stage form of Silverman (1986), Density Estimation, section 5.3.1. A fixed-bandwidth pilot `f` at every point (including itself) sets `h_i = h * clamp((f(x_i) / g)^(-alpha), 1/L, L)`, where `g` is the weighted geometric mean of `f` and `alpha` the sensitivity (default 1/2; 0 is the fixed bandwidth). The density is `sum_i w_i K_{h_i}(|u - x_i|)` with the gaussian (truncated at 5 bandwidths), quartic or epanechnikov kernel of `GPUKernelDensity`.

- Per-frame: `parameters` from `getGPUAdaptiveBandwidthKernelDensityParameterValues` (the layout of `getGPUKernelDensityParameterValues`, plus the sensitivity in slot 9). Compile-time: grid, index, kernel, normalization, `bandwidthFactorLimit` (1 to 16, default 4; it bounds the gather radius, so it is a deliberate approximation) and the optional `pointBandwidths` and `pilotDensities` outputs.
- `GPUKernelDensity` is not modified, so its results are unchanged. The new class is separate because the existing gather, edge-correction mass and binned path all assume one bandwidth.
- With `alpha = 0` it matches `GPUKernelDensity` with `edgeCorrection: 'none'` to 1e-4 of the peak (tested).
- Not implemented: edge correction, the binned method, polygon windows, bandwidth selection.
- Accuracy against binary64 all-pairs (weighted clustered data, 24 by 20 grid): point bandwidths and pilots within 1e-3 relative, cell values within 2e-3 of the peak, for the three kernels. The pilot is evaluated in float32.

{/* lens:GPUAdaptiveBandwidthKernelDensity */}

{/* parameters: GPUAdaptiveBandwidthKernelDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUAdaptiveBandwidthKernelDensity` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUAdaptiveBandwidthKernelDensity`.

- Packer: `getGPUAdaptiveBandwidthKernelDensityParameterValues(settings, grid)`, writes `GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH` `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid: axis-aligned with a nonzero finite cell size.

Settings of `getGPUAdaptiveBandwidthKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Pilot and global bandwidth `h` in position units. Positive and finite. |
| `sensitivity` (optional) | `number` | Abramson exponent `alpha` in `[0, 1]`; 0 is the fixed bandwidth. Defaults to 1/2. |

{/* /parameters */}
