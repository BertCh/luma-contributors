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

`buildVecchiaConditioningSets({positions, neighborCount, order, validity, times, timeScale})` returns CSR `{offsets, neighbors, counts, formats}` over the input row order: for each row, the `neighborCount` (at most 32) rows earlier in the Vecchia order that are nearest. `order` is any permutation; `getVecchiaMaxMinOrdering(positions)` gives the exact max-min order (Guinness 2018). Pass `times` and `timeScale` (spatial units per unit of time) to use the space-time metric `sqrt(h² + (timeScale·u)²)`. Build once per ordering; the sets do not depend on the covariance. `buildVecchiaPredictionNeighborhoods({positions, queries, neighborCount})` is the same search from arbitrary finite query points over the data, including points outside its extent. Empty, singleton, coincident and all-masked data are supported. Both builders use only finite XY rows with a nonzero setup mask (and finite times when supplied), preserving original source-row IDs. Setup masks retain their nonzero convention: negative integer values such as `Int8Array.of(-1)` include a row; zero or NaN excludes it. This differs from the packed observation's float validity lane, which includes only strictly positive values. Inputs require complete XY pairs and mask/time columns covering all selected rows. An explicit `rowCount` selects a prefix, so longer full-source mask/time columns remain valid. `order` must be an exact in-range permutation of the selected rows without duplicates; supplying `times` requires an explicit finite `timeScale`.

`counts` contains `{rowCount, neighborCount, entryCount}`: the result row domain, the requested maximum width even when no span fills it, and the readable number of CSR entries. `formats` declares both arrays as `uint32`. `uploadVecchiaNeighborhoods(sets, uploadUint32)` calls your `(array, 'uint32')` uploader and returns `{neighborhoodOffsets, neighborhoods, neighborCount}` ready for a consumer; prediction counts refer to query rows, while its neighbor IDs refer to data rows. Builders return `PreparedVecchiaConditioningSets` with guaranteed metadata; the legacy `VecchiaConditioningSets` type still accepts caller-constructed `{offsets, neighbors}` values. The uploader takes the prepared form. CPU evaluators similarly return prepared result types with guaranteed evidence, while their existing result types allow callers to omit the new evidence field.

The conditioning builder uses the GPU Vecchia width limit of 32. The prediction builder accepts positive integer widths beyond 32, including geographic conformal calibration's default 64 and supported widths through 256; downstream GPU solve classes enforce their own width limits. Both setup routes reject allocation products above `MAXIMUM_VECCHIA_ENTRY_COUNT = 2,147,483,647` entries and row counts above 2,147,483,646; this is a representability ceiling for signed grid links and uint32 CSR offsets, not a claim that such allocations fit available memory.

## `GPUVecchiaLogLikelihood`

`neighborhoods` and `neighborhoodOffsets` are the CSR above; `positions` and `observations` have one row per site; `parameters` is written by `getGPUVecchiaLogLikelihoodParameterValues({model, range, sill, nugget})`, per frame, with no recompile.

Each site's solve is the `simple` mode of `GPULocalKriging` over its conditioning set: `d_i` is `values.y`, and the observation error variance of the site is added to it. A site with an empty conditioning set, or whose conditioning rows are all unusable (validity not strictly positive, non-finite value, error variance or position), contributes its marginal term with variance `sill + nugget`, in this class, in the space-time class and in `GPUVecchiaFisherScoring` alike; rebuilding the sets after changing `validity` is therefore not required for a finite likelihood. Observations are residuals from any known mean; subtract an estimated trend first.

**Outputs** (`output`):

| Field | Rows | Content |
| --- | --- | --- |
| `conditionals` | one per site | `[mean, variance d_i, pivot ratio, retained]` from the kriging solve |
| `conditionalStatus` | one per site | the solve status |
| `siteTerms` | one per site | `-1/2 (log 2π + log d_i + r_i² / d_i)`; NaN if excluded or failed |
| `siteStatus` | one per site | 0 used, 1 excluded, 2 solve failed, 3 variance non-positive |
| `logLikelihood` | 2 | `[high, low]` double-single total: `high + low` |
| `status` | 3 words | `[failedSiteCount, excludedSiteCount, usedSiteCount]` |

`getVecchiaLogLikelihoodResult(total, words)` returns an object containing `logLikelihood`, `failedSiteCount`, `excludedSiteCount` and `usedSiteCount`. Read the total with `const {logLikelihood} = getVecchiaLogLikelihoodResult(total, words)`. Both retained numeric readers validate the full layout before interpreting it and accept `Uint8Array` or `ArrayBuffer` readback as bytes: likelihood needs two float32 components, Fisher needs twenty, and each needs three uint32 status words. Partial four-byte elements and short ports throw. `decodeGPUVecchiaLogLikelihoodResult({logLikelihood, status, neighborhoodStatus?})` and `decodeGPUVecchiaFisherScoringResult({totals, status, neighborhoodStatus?})` provide named port reads, respecting each byte view's offset and length. The space-time counterpart is `decodeGPUVecchiaSpaceTimeLogLikelihoodResult`. Named decoders reject typed arrays whose element type differs from the port (`Float32Array` totals, `Uint32Array` counts/evidence); retained readers continue to accept numeric arrays. Byte detection works across realms. Each class also has a `decode<Class>Status` helper returning named counts, `complete`, `overflow` and `candidateOverflow`. The total is a fixed-point sum of the float32 terms, so it does not depend on the order or on a float32 accumulation.

**Failure.** Any site that does not solve, or has a non-positive conditional variance, is counted in `status[0]` and the whole likelihood is NaN. It is a failed likelihood, not a partial one.
This includes a marginal site with no conditioning rows: a nonzero `conditionalStatus` from an
invalid site position (or space-time timestamp) gives `siteStatus = 2`, a NaN site term and NaN
total. Excluded observations are handled first and retain `siteStatus = 1`; a valid empty
conditioning set still contributes its ordinary marginal Gaussian density.
All three contributors and operation forms accept optional `neighborhoodStatus: {overflow?, candidateOverflow?}` uint32 scalar input ports. A nonzero supplied loss word suppresses every likelihood, score and information total with NaN. Exact CPU-built CSR needs no certificate. Solve counts keep their existing meaning; optional `output.neighborhoodStatus` is a separate three-word uint32 port `[complete, overflow, candidateOverflow]` describing the input neighborhoods. The CPU evaluators always return that evidence array. Its `complete` describes the CSR only; the named result decoders combine it with solve counts and finite totals to determine overall `status.complete`. Either loss word suppresses decoded results even if the certificate's `complete` word is inconsistent. Site terms and conditionals remain provisional when input loss is present. Clearing the input loss words restores totals on the next evaluation, including reuse of a compiled graph.

Input `neighborhoodStatus.complete` is unsupported: every Vecchia constructor and CPU evaluator rejects it, even alongside a separate zero loss flag. These routes consume negative `overflow`/`candidateOverflow` flags, and do not assume that either flag is equivalent to a producer's independent positive completeness certificate. Use a completeness-aware consumer when the producer supplies `complete`.

CPU input loss flags must be nonempty `Uint32Array` word views; byte views and other array-like objects are rejected. GPU loss flags and optional evidence destinations are captured once and validated as packed uint32 graph views before use.

The spatial contributor captures its output views at construction so its solve and likelihood
stages use the same conditional-status destination. Replacing the caller's output map afterward
does not redirect either stage; write buffer contents to change per-frame inputs.

**Limits and floors.** Exponential and Matérn 3/2 only. `getGPUVecchiaLogLikelihoodParameterValues` throws if a Matérn 3/2 nugget is below `getVecchiaNuggetFloor(covariance)`, which is `1e-3` of the sill (0 for the exponential). Prediction does not use this floor. Row and conditioning-set sizes are compile-time. All three GPU routes accept `neighborCount` in 1..32 (`MAXIMUM_VECCHIA_NEIGHBOR_COUNT = 32`), enforced by their private solve arrays; the CPU evaluators solve all usable CSR slots. Over-width CSR has route-specific selection: GPU spatial likelihood retains the nearest `neighborCount` usable rows, whereas GPU Fisher and space-time likelihood retain the first `neighborCount` usable CSR entries in visit order. Unusable entries do not consume that width. CPU twins evaluate all usable slots, so an over-width input can yield different likelihoods. Build CSR with the intended width to compare forms.

{/* lens:GPUVecchiaLogLikelihood */}

{/* parameters: GPUVecchiaLogLikelihood generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUVecchiaLogLikelihood` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUVecchiaLogLikelihood`.

- Packer: `getGPUVecchiaLogLikelihoodParameterValues(covariance)`, writes the `float32` parameter view.

Settings of `getGPUVecchiaLogLikelihoodParameterValues`:

| Field | Type | Description |
|---|---|---|
| `model` | `GPUVecchiaCovarianceModel` | - |
| `range` | `number` | Effective range, positive. |
| `sill` | `number` | Partial sill, at least 0. |
| `nugget` (optional) | `number` | Process microscale variance. Matern 3/2 needs at least `VECCHIA_NUGGET_FLOOR_FRACTION * sill`. |

{/* /parameters */}

## `GPUVecchiaFisherScoring`

One evaluation of the likelihood, the score and the expected information in `(log sill, log range, log nugget)`, per Guinness 2019. Fixed-point totals, one `[high, low]` pair per cell: read them with `getVecchiaFisherEvaluation(totals, status)`. Rows of its `siteTerms` are `[logLik, score(3), information(6)]`.

`fitVecchiaCovarianceFisherScoring({initial, evaluate, ...})` runs the CPU outer loop: Fisher steps in log parameters, halving line search, at most 30 iterations by default. `stopReason` is `converged` when the decrement meets tolerance, `iteration-limit` when the budget is exhausted, `initial-failure` when the initial evaluation is unusable, or the legacy `line-search-stall` when no usable acceptable step can be found. For that legacy reason, the optional `stallReason` distinguishes `direction-failure` (information/score could not resolve all required branches; no line search occurred) from `line-search-rejected` (the line search found no acceptable step). At a fixed positive nugget equal to its floor, the fitter conservatively requires both feasible one-sided sill branches to be resolved; failure of either branch stops the fit even when the other provides a usable direction. `converged`, `iterationLimitReached` and `failed` accompany the stop reason. A stall preserves the last usable finite fit, including the initial fit if no step was accepted. `history` records accepted steps only: each row contains the accepted post-step likelihood and the pre-step decrement that proposed it, with no initial or rejected trial row.

With `fitNugget: false`, the requested nugget is the initial nugget (zero if omitted). The effective nugget is at least the sill-dependent model floor; while the floor is active it moves with the sill even though the requested nugget is fixed. An estimated nugget pinned to the floor also follows the trial sill in either direction. `atNuggetFloor` reports that constraint.

{/* lens:GPUVecchiaFisherScoring */}

## `GPUVecchiaSpaceTimeLogLikelihood`

Same outputs as the spatial form, over the per-frame `SpaceTimeVariogramModel` covariance of `GPUSpaceTimeKriging` (separable, product-sum, metric, sum-metric, Gneiting). `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target, searchAnisotropy)` packs the layout of `getGPUSpaceTimeKrigingParameterValues` and enforces the floor for smooth components (the exact coincident discontinuity from the model parameters must be at least `1e-3` of `C(0, 0)`; finite probe lags are not used). Conditioning sets come from the CSR; `GPUSpaceTimeKriging` itself has no CSR path and is unchanged, so that contributor stays within the core eight storage buffers.

There is no space-time Fisher kernel: families have different parameter sets. Use `maximizeVecchiaLogLikelihood({initial, evaluate})`, a CPU Nelder-Mead over the free per-frame parameters (usually log-transformed).

{/* lens:GPUVecchiaSpaceTimeLogLikelihood */}

{/* parameters: GPUVecchiaSpaceTimeLogLikelihood generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUVecchiaSpaceTimeLogLikelihood` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUVecchiaSpaceTimeLogLikelihood`.

- Packer: `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)`, writes the `float32` parameter view.
  - `target` (`Float32Array`): Optional destination of at least 28 elements.
  - `searchAnisotropy` (`number`): Space units per unit of time, needed when the model has none.
- Packer: `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)`, writes the `float32` parameter view.
  - `target` (`Float32Array`): Optional destination of at least 28 elements.
  - `searchAnisotropy` (`number`): Space units per unit of time, needed when the model has none.
- Packer: `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)`, writes the `float32` parameter view.
  - `target` (`Float32Array`): Optional destination of at least 28 elements.
  - `searchAnisotropy` (`number`): Space units per unit of time, needed when the model has none.
- Packer: `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)`, writes the `float32` parameter view.
  - `target` (`Float32Array`): Optional destination of at least 28 elements.
  - `searchAnisotropy` (`number`): Space units per unit of time, needed when the model has none.
- Packer: `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)`, writes the `float32` parameter view.
  - `target` (`Float32Array`): Optional destination of at least 28 elements.
  - `searchAnisotropy` (`number`): Space units per unit of time, needed when the model has none.

Settings of `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)` for `{ type: "separable"; sill: number; spatial: SpaceTimeVariogramStructure; temporal: SpaceTimeVariogramStructure; }`:

| Field | Type | Description |
|---|---|---|
| `type` | `'separable'` | - |
| `sill` | `number` | - |
| `spatial` | `SpaceTimeVariogramStructure` | - |
| `temporal` | `SpaceTimeVariogramStructure` | - |

Settings of `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)` for `{ type: "product-sum"; k: number; spatial: SpaceTimeVariogramStructure; temporal: SpaceTimeVariogramStructure; }`:

| Field | Type | Description |
|---|---|---|
| `type` | `'product-sum'` | - |
| `k` | `number` | - |
| `spatial` | `SpaceTimeVariogramStructure` | - |
| `temporal` | `SpaceTimeVariogramStructure` | - |

Settings of `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)` for `{ type: "metric"; joint: SpaceTimeVariogramStructure; anisotropy: number; }`:

| Field | Type | Description |
|---|---|---|
| `type` | `'metric'` | - |
| `joint` | `SpaceTimeVariogramStructure` | - |
| `anisotropy` | `number` | - |

Settings of `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)` for `{ type: "sum-metric"; spatial: SpaceTimeVariogramStructure; temporal: SpaceTimeVariogramStructure; joint: SpaceTimeVariogramStructure; anisotropy: number; }`:

| Field | Type | Description |
|---|---|---|
| `type` | `'sum-metric'` | - |
| `spatial` | `SpaceTimeVariogramStructure` | - |
| `temporal` | `SpaceTimeVariogramStructure` | - |
| `joint` | `SpaceTimeVariogramStructure` | - |
| `anisotropy` | `number` | - |

Settings of `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target?, searchAnisotropy?)` for `{ type: "gneiting"; nugget: number; sill: number; spatialRange: number; temporalRange: number; alpha: number; gamma: number; beta: number; tau: number; }`:

| Field | Type | Description |
|---|---|---|
| `type` | `'gneiting'` | - |
| `nugget` | `number` | - |
| `sill` | `number` | - |
| `spatialRange` | `number` | - |
| `temporalRange` | `number` | - |
| `alpha` | `number` | Temporal smoothness, in (0, 1]. |
| `gamma` | `number` | Spatial smoothness, in (0, 1]. |
| `beta` | `number` | Space-time interaction, in [0, 1]; 0 is separable. |
| `tau` | `number` | Temporal exponent, at least `beta` in two spatial dimensions. |

{/* /parameters */}

## Operation forms and CPU twins

`GPUVecchiaLogLikelihoodOperation`, `GPUVecchiaSpaceTimeLogLikelihoodOperation` and `GPUVecchiaFisherScoringOperation` use graph-value names for the same ports, including optional loss inputs and evidence output. Their CPU twins are `CPUVecchiaLogLikelihoodOperation`, `CPUVecchiaSpaceTimeLogLikelihoodOperation` and `CPUVecchiaFisherScoringOperation`. The pure evaluators are `computeVecchiaLogLikelihoodOnCPU`, `computeVecchiaSpaceTimeLogLikelihoodOnCPU` and `computeVecchiaFisherScoringOnCPU`; each accepts an optional `id` for validation errors.

Coordinates and effective ranges use the native planar distance unit. Times are relative ticks in the temporal context's unit, and `timeScale`/`searchAnisotropy` use planar distance units per tick. Observations use the modeled variable's unit; observation error, sill, nugget and conditional variance use its square. Log likelihood and scores in log parameters are dimensionless. The observation fourth component is a float mask: strictly positive includes; zero, negative and NaN exclude. CSR offsets count entries; neighborhoods are source row indices; retained counts are live usable-row counts. Packed outputs use the exact row and component counts above; any allocated tail beyond those counts is unspecified.

## Prediction

Prediction does not get a new contributor: use `GPULocalKriging` in `simple` mode with `buildVecchiaPredictionNeighborhoods` and the same parameters. The existing kriging rules apply, including no nugget floor, so a smooth covariance with no nugget can report status 2 at some queries. For space-time, use `GPUSpaceTimeKriging` with its own neighbour search.

## Float32 on the GPU

Repeat of the float32 experiment in [the spatiotemporal review](https://github.com/visgl/luma-contributors/blob/master/dev-docs/research/spatiotemporal-review-2026-11-09/README.md), same points, conditioning sets, ordering (`random`), grid and kernels, with `GPUVecchiaLogLikelihood` on an Apple GPU (Metal, headless Chrome) compared with a binary64 CPU reference of the same sets. Run: `VITE_VECCHIA_FLOAT32_STUDY=1 VITE_VECCHIA_FLOAT32_STUDY_SIZES=10000 npx vitest run --project headless --silent=false modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-float32-gpu-study.spec.ts`. The binary64 reference reproduces the emulated experiment's `ll64 at best` and gaps to the printed digits. n = 10,000, m = 30.

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
