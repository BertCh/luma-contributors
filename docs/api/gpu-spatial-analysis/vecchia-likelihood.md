# Vecchia likelihood (STM-07)

Nearest-neighbour Gaussian process likelihood, fitting and prediction. The likelihood of ordered points factors into one conditional per point, `log L = -1/2 Σ [log 2π + log d_i + r_i² / d_i]`, and each conditional is a small kriging solve on earlier points, which is what `GPULocalKriging` already does.

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

```ts
import {
  GPUVecchiaLogLikelihood,
  GPUVecchiaFisherScoring,
  GPUVecchiaSpaceTimeLogLikelihood,
  buildVecchiaConditioningSets,
  getVecchiaMaxMinOrdering,
  buildVecchiaPredictionNeighborhoods,
  fitVecchiaCovarianceFisherScoring,
  maximizeVecchiaLogLikelihood,
  getGPUVecchiaLogLikelihoodParameterValues,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues,
  getVecchiaLogLikelihoodResult,
  getVecchiaFisherEvaluation,
  getVecchiaNuggetFloor
} from '@luma-contributors/analysis/gpu-spatial-analysis';
```

## Conditioning sets (CPU setup)

`buildVecchiaConditioningSets({positions, neighborCount, order, validity, times, timeScale})` returns CSR `{offsets, neighbors}` over the input row order: for each row, the `neighborCount` (at most 32) rows earlier in the Vecchia order that are nearest. `order` is any permutation; `getVecchiaMaxMinOrdering(positions)` gives the exact max-min order (Guinness 2018). Pass `times` and `timeScale` (spatial units per unit of time) to use the space-time metric `sqrt(h² + (timeScale·u)²)`. Build once per ordering; the sets do not depend on the covariance. `buildVecchiaPredictionNeighborhoods({positions, queries, neighborCount})` is the same search from arbitrary query points over the data.

## `GPUVecchiaLogLikelihood`

`neighborhoods` and `neighborhoodOffsets` are the CSR above; `positions` and `observations` have one row per site; `parameters` is written by `getGPUVecchiaLogLikelihoodParameterValues({model, range, sill, nugget})`, per frame, with no recompile.

Each site's solve is the `simple` mode of `GPULocalKriging` over its conditioning set: `d_i` is `values.y`, and the observation error variance of the site is added to it. Observations are residuals from any known mean; subtract an estimated trend first.

**Outputs** (`output`):

| Field | Rows | Content |
| --- | --- | --- |
| `conditionals` | one per site | `[mean, variance d_i, pivot ratio, retained]` from the kriging solve |
| `conditionalStatus` | one per site | the solve status |
| `siteTerms` | one per site | `-1/2 (log 2π + log d_i + r_i² / d_i)`; NaN if excluded or failed |
| `siteStatus` | one per site | 0 used, 1 excluded, 2 solve failed, 3 variance non-positive |
| `logLikelihood` | 2 | `[high, low]` double-single total: `high + low` |
| `status` | 3 words | `[failedSiteCount, excludedSiteCount, usedSiteCount]` |

`getVecchiaLogLikelihoodResult(total, words)` reads these as a number. The total is a fixed-point sum of the float32 terms, so it does not depend on the order or on a float32 accumulation.

**Failure.** Any site that does not solve, or has a non-positive conditional variance, is counted in `status[0]` and the whole likelihood is NaN. It is a failed likelihood, not a partial one.

**Limits and floors.** Exponential and Matérn 3/2 only. `getGPUVecchiaLogLikelihoodParameterValues` throws if a Matérn 3/2 nugget is below `getVecchiaNuggetFloor(covariance)`, which is `1e-3` of the sill (0 for the exponential). Prediction does not use this floor. Row and conditioning-set sizes are compile-time; the conditioning set is capped at 32 by the kriging private arrays.

{/* lens:GPUVecchiaLogLikelihood */}

## `GPUVecchiaFisherScoring`

One evaluation of the likelihood, the score and the expected information in `(log sill, log range, log nugget)`, per Guinness 2019. Fixed-point totals, one `[high, low]` pair per cell: read them with `getVecchiaFisherEvaluation(totals, status)`. Rows of its `siteTerms` are `[logLik, score(3), information(6)]`.

`fitVecchiaCovarianceFisherScoring({initial, evaluate, ...})` runs the CPU outer loop: Fisher steps in log parameters, halving line search, at most 30 iterations by default, with `converged` and `iterationLimitReached` in the result. A nugget below the floor is pinned at the floor; while it is pinned the nugget moves with the sill.

{/* lens:GPUVecchiaFisherScoring */}

## `GPUVecchiaSpaceTimeLogLikelihood`

Same outputs as the spatial form, over the per-frame `SpaceTimeVariogramModel` covariance of `GPUSpaceTimeKriging` (separable, product-sum, metric, sum-metric, Gneiting). `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target, searchAnisotropy)` packs the layout of `getGPUSpaceTimeKrigingParameterValues` and enforces the floor for smooth components (the variogram at a tiny lag must be at least `1e-3` of `C(0, 0)`). Conditioning sets come from the CSR; `GPUSpaceTimeKriging` itself has no CSR path and is unchanged, so that contributor stays within the core eight storage buffers.

There is no space-time Fisher kernel: families have different parameter sets. Use `maximizeVecchiaLogLikelihood({initial, evaluate})`, a CPU Nelder-Mead over the free per-frame parameters (usually log-transformed).

{/* lens:GPUVecchiaSpaceTimeLogLikelihood */}

{/* parameters: GPUVecchiaSpaceTimeLogLikelihood generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUVecchiaSpaceTimeLogLikelihood` parameters

- Packer: `getGPUSpaceTimeKrigingParameterValues(settings)`, writes `GPU_SPACE_TIME_KRIGING_PARAMETER_LENGTH` (28) `float32` values.
- Packer: `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, searchAnisotropy?)`, writes the `float32` parameter view.
  - `model` (`SpaceTimeVariogramModel`): Space-time covariance.
  - `searchAnisotropy` (`number`): Space units per unit of time, needed when the model has none.

Settings of `getGPUSpaceTimeKrigingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `covariance` | `SpaceTimeVariogramModel` | Space-time covariance, for example the result of `fitSpaceTimeVariogramModel`. |
| `searchAnisotropy` (optional) | `number` | Spatial units per unit of time of the neighbor metric `sqrt(h^2 + (anisotropy u)^2)`. Defaults to the `anisotropy` of a metric or sum-metric covariance; required otherwise. |
| `searchRadius` (optional) | `number` | Largest spatial lag of a neighbor. Defaults to infinity. |
| `maximumTimeLag` (optional) | `number` | Largest absolute time lag of a neighbor. Defaults to infinity. |
| `mean` (optional) | `number` | Known mean of `simple` mode. Defaults to 0. |

{/* /parameters */}

## Prediction

Prediction does not get a new contributor: use `GPULocalKriging` in `simple` mode with `buildVecchiaPredictionNeighborhoods` and the same parameters. The existing kriging rules apply, including no nugget floor, so a smooth covariance with no nugget can report status 2 at some queries. For space-time, use `GPUSpaceTimeKriging` with its own neighbour search.

## Float32 on the GPU

Repeat of the float32 experiment in [the spatiotemporal review](../../../dev-docs/research/spatiotemporal-review-2026-11-09/README.md), same points, conditioning sets, ordering (`random`), grid and kernels, with `GPUVecchiaLogLikelihood` on an Apple GPU (Metal, headless Chrome) compared with a binary64 CPU reference of the same sets. Run: `VITE_VECCHIA_FLOAT32_STUDY=1 VITE_VECCHIA_FLOAT32_STUDY_SIZES=10000 npx vitest run --project headless --silent=false modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-float32-gpu-study.spec.ts`. The binary64 reference reproduces the emulated experiment's `ll64 at best` and gaps to the printed digits. n = 10,000, m = 30.

| Kernel | Nugget / sill | Failed sites | Error | Gap | Same best range | ms per evaluation |
| --- | --- | --- | --- | --- | --- | --- |
| Exponential | 0 | 0 | 0.0021 | 4.2 | yes | 15 |
| Exponential | 1e-3 | 0 | 0.0095 | 3.1 | yes | 15 |
| Exponential | 0.1 | 0 | 0.00054 | 0.07 | yes | 16 |
| Matérn 3/2 | 0 | 6,578 | n/a | 43 | n/a | 15 |
| Matérn 3/2 | 1e-4 | 48 | n/a | 25 | n/a | 15 |
| Matérn 3/2 | 1e-3 | 0 | 0.21 | 16 | yes | 16 |
| Matérn 5/2 | 0 | 9,411 | n/a | 136 | n/a | 16 |
| Matérn 5/2 | 1e-4 | 1,211 | n/a | 24 | n/a | 16 |
| Matérn 5/2 | 1e-3 | 0 | 0.092 | 12 | yes | 16 |
| Matérn 5/2 | 1e-2 | 0 | 0.016 | 5.9 | yes | 15 |
| Gaussian | 0 | 9,046 | n/a | n/a | n/a | 15 |
| Gaussian | 1e-4 | 3,055 | n/a | 40 | n/a | 15 |
| Gaussian | 1e-3 | 0 | 0.46 | 16 | yes | 17 |
| Gaussian | 1e-2 | 0 | 0.0050 | 7.1 | yes | 15 |

"Error" is the largest absolute difference from the binary64 log-likelihood over the ten-point range grid and "gap" is how far the best grid point sits above the next. Where sites failed the likelihood is NaN by design, so no error is defined. The failures are stricter than the emulated Cholesky failures because `GPULocalKriging` also reports status 2 at a pivot ratio below `1e-4`.

- At n = 100,000, m = 30 (about 100 ms per evaluation on the same GPU): exponential 1e-3 error 0.26 against a gap of 45, Matérn 3/2 1e-3 error 0.51 against a gap of 42, Matérn 3/2 1e-2 error 0.016 against 13. Exponential with **no** nugget now reports 18 failed sites, from the kriging pivot-ratio check on closely spaced neighbours, so the conclusion that the exponential is safe at any nugget does not extend past n = 10,000 under this solver. Use a small nugget.
- At n = 500,000 one evaluation takes about 570 ms; the single-threaded setup (conditioning sets by grid) about 2 s.
- With max-min ordering at n = 10,000 the same pattern holds: exponential 1e-3 error 0.0096, Matérn 3/2 1e-3 error 0.19, Gaussian 1e-3 error 0.73, same best range.

Fisher scoring in float32 was within binary64 precision for six datasets at n = 2,500 and two at n = 10,000 (exponential and Matérn 3/2, four to six iterations, parameters agreeing to about 5 digits, log-likelihood within 0.001). When the data has no nugget and Matérn 3/2 is held at the floor, Fisher scoring zigzags in binary64 as well as float32, so `converged` is not reached in 30 iterations: that limit is not an arithmetic one.

GpGp, the intended external oracle, is R-only; the reference here is a binary64 CPU implementation of the same conditioning sets and the same ordering.

{/* parameters: GPUVecchiaLogLikelihood generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUVecchiaLogLikelihood` parameters

- Packer: `getGPUVecchiaLogLikelihoodParameterValues(covariance)`, writes the `float32` parameter view.

Settings of `getGPUVecchiaLogLikelihoodParameterValues`:

| Field | Type | Description |
|---|---|---|
| `model` | `GPUVecchiaCovarianceModel` | - |
| `range` | `number` | Effective range, positive. |
| `sill` | `number` | Partial sill, at least 0. |
| `nugget` (optional) | `number` | Process microscale variance. Matern 3/2 needs at least `VECCHIA_NUGGET_FLOOR_FRACTION * sill`. |

{/* /parameters */}
