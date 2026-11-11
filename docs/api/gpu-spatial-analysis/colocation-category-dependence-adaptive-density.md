# Colocation quotient, local category dependence and adaptive-bandwidth density

STX-08, second group. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. Weights are the published `GPUSpatialWeights` CSR; the density reuses the kernels, parameter packer and grid-index gather of `GPUKernelDensity`.

## `GPUColocationQuotient`

Reference: Leslie and Kronenfeld (2011), "The colocation quotient: a new measure of spatial association between categorical subsets of points", Geographical Analysis 43(3). No pinned package implements it, so the specs compare with an independent binary64 evaluation of the definition.

For a row `i` of category `a`, the neighbour share `f(i, b)` is the weight on neighbours of category `b` divided by the weight on all neighbours. The chance share is `ratio(a, b) = (N_b - [a = b]) / (N - 1)`. The local quotient is `f(i, b) / ratio(a, b)`; the global quotient `CLQ(a -> b)` is the mean of `f(i, b)` over the rows of `a` that have a neighbour, divided by the same ratio. With one neighbour per row it is Leslie and Kronenfeld's `(N_{a->b} / N_a) / (N_b / (N - 1))`; Gaussian-kernel weights give the adaptive form of Wang et al. (2017). 1 is independence, above 1 attraction, below 1 avoidance.

- Neighbours are listed slots with a finite positive weight, to another row with a valid category. Rows with category `0xffffffff` (or at or above `categoryCount`) are left out. Results are NaN where a row has no neighbour or the chance share is 0.
- Outputs: `globalQuotients` (`categoryCount^2`, entry `a * categoryCount + b`), optional `localQuotients` (`rows * categoryCount`) and `categoryRowCounts`.
- At most 32 categories. The shares are summed in the shared fixed-point accumulator, so the global values do not depend on thread order.
- Not implemented: the permutation test of the paper.

{/* lens:GPUColocationQuotient */}

## `GPULocalCategoryDependence`

Reference: Boots (2003), "Developing local measures of spatial association for categorical data", Journal of Geographical Systems 5, which defines the LICD family. Implemented: a local composition measure. Per row (`GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS`), with the row itself never counted: own-category neighbours (`likeCount`), neighbours with a category (`neighborCount`, `k`), the local share `likeCount / k` (`proportion`), the global share `N_c / N` (`expectedProportion`), the binomial upper tail `P(X >= likeCount)` for `Binomial(k, N_c / N)` and the z score of the same binomial. A neighbour is an active slot of the shared rule (positive weight, another row); the size of a positive weight is not used.

This is not the output of `licd_multi` of the R package spdep, and no parity with it is claimed (R is not installed, so it cannot be checked here). spdep counts the focal row among its own neighbours (`k + 1` trials, the focal row always a like), reports the global share as `prop_i`, and its `pbinom_like_BW` is the lower tail `P(X <= like)`. Example: 4 neighbours, 3 of them like, share 0.5. This class gives `P(X >= 3) = 0.3125` on `Binomial(4, 0.5)`; spdep's `pbinom_like_BW` is `P(X <= 4) = 0.96875` on `Binomial(5, 0.5)`.

Not implemented: the local configuration part (local join counts and the chi-square), the permutation p-values and the unlike-neighbour tails. The binomial test is checked against hand-computed tails (0.3125, 0.648) and a binary64 evaluation.

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

- Packer: `getGPUAdaptiveBandwidthKernelDensityParameterValues(settings, grid)`, writes `GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH` `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid: axis-aligned with a nonzero finite cell size.
- Packer: `getGPUKernelDensityParameterValues(settings, grid)`, writes `GPU_KERNEL_DENSITY_PARAMETER_LENGTH` (9) `float32` values.
  - `grid` (`GPURasterMetadata`): The output grid: axis-aligned, with a nonzero finite cell size; its width and height must match the contributor's.

Settings of `getGPUAdaptiveBandwidthKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Pilot and global bandwidth `h` in position units. Positive and finite. |
| `sensitivity` (optional) | `number` | Abramson exponent `alpha` in `[0, 1]`; 0 is the fixed bandwidth. Defaults to 1/2. |

Settings of `getGPUKernelDensityParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU computes it) when `GPUKernelDensityProps.bandwidthRule` is set; pass any positive value. |
| `window` (optional) | `GPUKernelDensityWindow \| 'study-window'` | Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells. |

{/* /parameters */}
