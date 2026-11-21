# Interpolation and simulation

<a id="bounded-gpu-interpolation-and-simulation"></a>

These contributors keep observations, fitting, prediction and simulation on the GPU. All graph views are packed, caller-owned, and belong to the target graph. Capacities specialize a graph. `GPULocalKriging` covariance models and parameters always come from a runtime buffer; input buffer contents can change between encodings. Outputs must have distinct buffers and must not alias inputs. Coordinates are native planar coordinates; project longitude/latitude before using these APIs.

Constructors capture compile-time options, including nested covariance, loading arrays, contexts and output routing. Mutating the original options after construction does not change graph declaration. Graph views retain their identity and buffer contents remain live; use runtime parameter buffers or the documented setters for per-frame updates. The graph operation forms also capture options before deferred lowering.

| Contributor | Contract | Explicit capacity |
| --- | --- | --- |
| `GPUVariogramModelFit` | Spherical, exponential, Gaussian and Matérn 3/2 and 5/2 empirical-bin fitting, pair/unweighted/Cressie weights | 5 model candidates, 3–1024 ladder points, 0–64 refinement iterations |
| `GPULocalKriging` | Simple, ordinary, universal/custom drift, external drift and multivariate LMC co-kriging; spherical, exponential, Gaussian and Matérn (smoothness 1/2, 3/2, 5/2) covariances in a runtime parameter buffer | 32 nearest observations, 8 drift columns, 64 weighted quadrature points; 4 variables and 4 PSD structures |
| `GPUSpaceTimeKriging` | Local simple or ordinary space-time kriging, five covariance families, metric/octant/staged neighbor search, per-frame parameters | 32 retained neighbors, 8 per octant, 64 staged candidates; O(queries · samples) scan |
| `GPULocalRBF` | Eight SciPy kernels, scalar/per-sample smoothing, resident fixed-width or CSR neighborhoods | 32 nearest observations, polynomial degree −1 through 2 subject to kernel constraints |
| `GPUConditionalGaussianSimulation` | Exact joint conditional Gaussian realizations with known mean | 4096 observation and query rows combined; dense factor and realization batch must fit device storage bindings |
| `GPUNaturalNeighbor` | Sibson stolen-Voronoi-area interpolation | 64 local sites and 64 polygon vertices; indexed Delaunay cavities over up to 2²⁴ sites |

`GPUVariogramModelFit` is exported from `@luma-contributors/analysis/dataframe-analysis`; the prediction contributors are exported from `@luma-contributors/analysis/gpu-spatial-analysis`. Capacities are explicit and audited against core WebGPU binding and shader-memory limits. Indexed natural-neighbor interpolation consumes resident Delaunay topology and containing-triangle seeds; it does not silently construct a CPU index.

## Resident variogram fitting

Provide equal-length `distances: float32`, `semivariances: float32`, and `pairCounts: uint32` graph views. Outputs contain one row per requested model, in the same order:

- `parameters: float32x4`: nugget, partial sill, effective range, weighted squared residual.
- `diagnostics: float32x4`: usable bin count, unweighted squared residual, Gaussian residual AIC with three parameters, final log-range bracket width.
- `status: uint32`: 0 interior converged fit, 1 insufficient usable bins, 2 nonfinite fit, 3 boundary optimum or exhausted refinement.

The range search uses `[minimumLag / 10, maximumLag * 3]`. Every fixed range solves nonnegative nugget/sill least squares, comparing both boundary fits. Cressie weighting starts from pair weights and refits three times with `pairCount / fittedSemivariance²`; the f32 denominator floor is `1e-20`. Invalid/nonpositive lag or empty bins are skipped. At least two distinct positive lags are needed. AIC compares the same empirical bins and is an unweighted residual diagnostic, not a spatial likelihood. Status 3 returns a candidate without claiming an identified interior range. The existing `fitVariogramModel` remains an explicitly CPU-side helper. Request final-pass diagnostics explicitly:

```ts
import {fitVariogramModelWithDiagnostics} from '@luma-contributors/analysis/dataframe-analysis';

const fit = fitVariogramModelWithDiagnostics({
  distances: [1, 2, 3], semivariances: [0.2, 0.5, 0.7], pairCounts: [10, 10, 10]
}, {model: 'exponential'});
// fit.model is the candidate; convergence requires an interior, sufficiently narrow bracket.
const converged = !fit.atBoundary && fit.logRangeBracket <= 1e-5;
```

`fitVariogramModelWithDiagnostics` returns `model`, `logRangeBracket`, `atBoundary` and
`converged`. These diagnostics describe the final weighting pass.

Explicit `rangeSteps` (3–1024) and `maximumIterations` (0–64) bind the GPU contributor, both
operation forms and the CPU helpers for every weighting pass. The ladder defaults to 200 points;
omitted refinement budgets retain GPU 32 and CPU 60 steps. Zero performs no refinement steps,
while still evaluating the ladder and the initial two interior candidates. Both operation forms
publish the actual final log-range bracket and status 3 for an endpoint ladder optimum or a bracket
wider than `1e-5`; finite candidates remain available. GPU can stop once the bracket is at most
`1e-5`, while CPU uses float64 refinements up to its budget.

Range candidates execute in parallel. Refinement uses one 64-lane workgroup per model to reduce bin statistics and residuals, with ordered golden-section steps and Cressie rounds. Floating accumulation order differs from the CPU helper and the earlier serial refinement; compare numerical tolerances and statuses rather than bit identity. The refinement adds fixed workgroup storage without increasing the candidate-search buffer or combining input batches.

## Kriging modes and support

Supply sample `positions: float32x2` and `observations: float32x4` rows `[value, measurementErrorVariance, variableIndex, validity]`. Float validity includes an observation exactly when `validity > 0`: zero, negative values and NaN omit it. This is distinct from integer masks used by raster IDW/kriging, where any nonzero integer includes a sample. Nonfinite positions/value/error, negative errors, or missing sample drift are omitted. A query with no usable neighbors is invalid. `supports` stores query-major quadrature points. `float32x2` uses equal weights; `float32x4` stores `[x, y, weight, reserved]`. Nonnegative weights are normalized per query; negative, nonfinite, or zero-total weights invalidate that query. `supportCount` defaults to one and is at most 64. Neighborhoods use distance from the weighted support centroid, breaking equal-distance ties by original row order.

Candidates come from caller `neighborhoods` (fixed rows or CSR) when supplied. Otherwise `sampleIndex` (compile-time) chooses: `'brute-force'` scans every sample from each query (rejected above `GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK`, 2³² sample-query tests); `'grid'` writes the support centroids and a mask of the usable samples, then a `GPUNeighborSearch` kNN grid lists the `neighborCount + max(neighborCount, 8)` nearest usable samples of each centroid by float32 `dx² + dy²`. The kernel ranks the candidates exactly as the scan does, by `distance()` with ties to the lower row; when a query's candidate list is full and its farthest candidate is within 1e-6 (relative) of the retained boundary, a tie may continue past the list and that query rescans every sample, so the grid keeps exactly the scan's neighbors even on float32 near-ties. Queries run in chunks whose candidate rows, together with a copy of the sample positions, fit one storage binding; the grid kernel therefore binds as many storage buffers as the scan. With the default `'auto'`, the constructor first checks whether there are at most `GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT` (4,096) samples and at most 2³² sample-query tests. When both conditions hold it scans. Otherwise it selects the grid if there is at least one sample and one query, the sorted targets fit the device’s storage binding limit at 12 bytes per sample, and a separate binding can hold the packed sample positions plus at least one query’s candidate rows. If the grid fails those checks, `'auto'` falls back to a scan and rejects total sample-query work above 2³². An explicit `'grid'` requires at least one sample and one query and checks the grid’s storage bindings when its nodes are built. Every choice retains the same neighbors in the same order, so results do not depend on it. Optional `spatialContext` declares the coordinate space; only native planar coordinates are accepted.

| Mode | Mean/design contract |
| --- | --- |
| `simple` | Known scalar `means[0]`, default zero; solves the covariance system without an unbiasedness constraint. |
| `ordinary` | Unknown constant mean with one unbiasedness constraint. |
| `universal` | Defaults to an unknown affine trend `[1, x, y]` with locally centered/scaled coordinates. Supplying `sampleDrift` and `queryDrift` selects an arbitrary caller-defined design with 1–8 columns; rank-deficient designs fail explicitly. |
| `external-drift` | Caller supplies 1–8 columns in `sampleDrift` and support-averaged `queryDrift`. Flat `float32` stores row-major `driftCount` columns; `float32x4` remains available for up to 4 columns. Include an intercept explicitly. The caller computes support-averaged drift using the same quadrature weights; missing query drift invalidates prediction. |
| `co-kriging` | Set `variableCount` to 2–4 and select `targetVariable`. `coKrigingMean: "known"` uses supplied means; `"ordinary"` imposes a separate unknown intercept per variable. `coregionalizationModels` supplies 1–4 structures, each with its own covariance and square row-major loading factors, so `C_ab(h)=Σ_s (L_s L_sᵀ)_ab C_s(h)` is PSD. The original 2×2 `coregionalization` shortcut remains supported. With `"ordinary"`, a secondary variable that has no sample in a neighbourhood drops out of that neighbourhood's system (its constraint is vacuous), so with every secondary variable absent the prediction is ordinary kriging of the target. The system needs at least as many retained samples as kept constraints (one for the target plus one per secondary variable present), so a neighbourhood with fewer samples than `variableCount` predicts as long as that holds: one target sample alone predicts its value. A neighbourhood without a sample of the target variable has no unbiased predictor: it reports status 1 when it retains fewer samples than kept constraints and status 2 otherwise. |

Covariance values always use the layout from `getGPULocalKrigingParameterValues`. Pass a caller-owned `parameters` view and write its `GPUParameterBuffer` between frames, or keep the convenience `covariance`, `coregionalization` and `coregionalizationModels` props as initial values and call `setCovariance(settings)` to change them. Model family, nugget, sill, range and loadings change without recompiling; variable and structure counts stay fixed. Submit preceding encodings before updating parameters. Each compiled graph owns its convenience parameter upload buffer and releases it on destruction; the contributor needs no separate cleanup. On devices where adding the runtime buffer would exceed the storage-binding limit, the convenience path appends parameters to a transient copy of the observation table, preserving the existing drift/weights configurations. That copy must fit one storage binding. Initialized zero-loading co-kriging models use normalization scale 1, keeping zero process variance distinct from an unwritten parameter buffer. With known means and positive observation error, such a model predicts the target mean with latent variance zero; an unwritten all-zero buffer still reports status 1.

The local kriging, RBF, space-time kriging and Vecchia kernels use the analysis-core LU factorization and substitution. The prediction adapter retains four sequential symmetric Ruiz passes and the existing pivot thresholds, so conditioning remains comparable to earlier results within float32 tolerance.

Each covariance structure has a supplied spherical/exponential/Gaussian/Matérn effective range and partial sill. `model: 'matern'` needs `smoothness` 1/2, 3/2 or 5/2 and uses the closed form with the argument `a = factor * h / range`, where `factor` is 3 at 1/2 (the exponential model), 4.749031386012701 at 3/2 and 5.924462618951797 at 5/2, so the correlation at `h = range` is `exp(-3)`, like the exponential and Gaussian models. For `a < 0.1` the correlation is evaluated by a Taylor series, because the f32 product `(1 + a) exp(-a)` can round above 1 and blurs the small `1 - correlation` that separates near-duplicate rows. To use GSTools `Matern`, set `len_scale = sqrt(nu) * range / factor`. Process nugget applies at coincident positions. Measurement error is added only to observation covariance diagonals; it is never added to latent prediction variance. Positive errors smooth sample hits. Duplicate observations with zero error can make the conditioning matrix singular and are reported; this API never silently adds jitter or merges them.

For block prediction, sample-to-block covariance sums weighted support covariances, and block variance uses the double weighted sum over every support pair. This includes the process-nugget contribution of repeated quadrature sites. The quadrature count is explicit provenance; it is a discrete weighted approximation to the intended support, not an exact polygon integral.

`output.values: float32x4` is `[prediction, latent variance, pivot ratio, retained neighbor count]`. `status` is 0 on success, 1 for missing/invalid data and 2 for deficient rank or unreliable variance. A query with no usable sample in its neighbourhood is status 1 with NaN by default (`emptyNeighborhood: 'insufficient'`). With `emptyNeighborhood: 'prior'` (compile-time, mode `simple` only, since only a known mean has a prior prediction) it reports status 0 with the known mean, the prior point or block variance, pivot ratio 1 and retained count 0, the simple kriging predictor of an empty neighbourhood; `GPUVecchiaLogLikelihood` uses it for sites whose conditioning rows are all excluded. Optional `weights` contains query-major neighbor weights in `(distance, row)` order, padded with zeros to `neighborCount`. Partial-pivot elimination of the equilibrated system rejects pivots below `1e-6` of the initial matrix scale and pivot ratios below `KRIGING_PIVOT_RATIO_TOLERANCE` (`1e-4`, the limit `GPUKriging` uses, measured at 5% weight error for a ratio of 3e-5; `GPULocalKriging` and `GPULocalRBF` used `1e-5` before and accepted such systems with status 0, now status 2); normalized negative variance within `1e-4` of the prior variance is clamped to zero. Outputs at failed queries contain NaN predictions. Weights are only valid with successful status. These f32 conditioning guards deliberately reject some solvable f64 systems.

### Reusing resident search neighborhoods

Both `GPULocalKriging` and `GPULocalRBF` accept `neighborhoodWeights: {offsets, neighbors, status?}`.
Pass the resident search output intact so producer `overflow` and `candidateOverflow` remain live:

```ts
import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULocalKriging, type GPULocalKrigingProps, type GPUNeighborSearch
} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const graph: GPUCommandGraph<void>;
declare const device: Device;
declare const positions: NonNullable<NonNullable<GPULocalKrigingProps>['positions']>;
declare const observations: NonNullable<NonNullable<GPULocalKrigingProps>['observations']>;
declare const queries: NonNullable<NonNullable<GPULocalKrigingProps>['supports']>;
declare const predictions: NonNullable<NonNullable<NonNullable<GPULocalKrigingProps>['output']>['values']>;
declare const predictionStatus: NonNullable<NonNullable<NonNullable<GPULocalKrigingProps>['output']>['status']>;
declare const search: GPUNeighborSearch;

const predictor = new GPULocalKriging({
  positions, observations, supports: queries, mode: 'ordinary',
  covariance: {model: 'exponential', range: 100, sill: 1},
  neighborCount: 16,
  neighborhoodWeights: search.outputWeights,
  output: {values: predictions, status: predictionStatus}
});
```

`offsets` is packed uint32 with query count + 1 rows; `neighbors` contains uint32 sample-row indices.
Optional `status.overflow` and `status.candidateOverflow` are uint32 scalars. `GPULocalKriging`
and `GPULocalRBF` both reject a supplied `complete` view at construction in
`neighborhoodWeights.status` or `neighborhoodStatus`, even when `candidateOverflow` is also supplied.
Use a neighborhood producer that reports loss through the supported flags. A nonzero producer
loss suppresses predictions to NaN, clears optional weights and writes query status **3**. This is
separate from status 1 (invalid/insufficient data) and status 2 (singular/unreliable solve).
Choosing a smaller complete neighborhood intentionally still reports numerical success. When the
producer recovers on a later frame, the same graph writes ordinary predictions again.
The legacy `neighborhoods`/`neighborhoodOffsets` spelling remains supported; attach
`neighborhoodStatus: search.outputWeights.status` to preserve loss on that route. If the legacy
and wrapper views are both supplied, their identities must agree. Tail rows beyond the query
count are unspecified. `decodeGPULocalKrigingStatus` and `decodeGPULocalRBFStatus` decode one query
word from uint32 or byte readback, including an optional row argument.

Covariance packers publish only their selected record prefix, preserving following arena records.
A rejected update leaves the entire destination unchanged. Co-kriging requires each float32
`L Lᵀ` coefficient and normalization scale to be finite; zero-loading covariances remain legal.

### Conditioning of smooth covariances

The pivot-ratio limit above is the only conditioning rule. No nugget floor is applied; a Matérn 3/2, 5/2 or Gaussian system with a small or zero nugget either passes the limit or reports status 2 with its pivot ratio in `values.z`. A floor was not added because it would silently change the requested model at every query, while this API never adds jitter, and it would change existing Gaussian results; the 1e-3-of-sill floor that the float32 Vecchia experiment recommends concerns the likelihood, not the prediction solve. Measured once on headless Chromium WebGPU over random 16- and 32-point neighborhoods (sill 1, range 0.6 of the scene), every system accepted at nugget 0 predicted within 1.6e-3 of the double-precision solve and within 2.5e-5 in variance; Matérn 3/2 and 5/2 systems had prediction errors of at most 1.4e-5 and 1.5e-4, and status 2 occurred for Gaussian and for one Matérn 5/2 query. Treat status 2 as a failed site in a likelihood or cross-validation sweep; do not skip it.

## Space-time kriging

`GPUSpaceTimeKriging` takes sample `positions` with relative float32 `times`, the observation rows of `GPULocalKriging`, query `queries` with `queryTimes`, and a per-frame `parameters` buffer from `getGPUSpaceTimeKrigingParameterValues`. `temporalContext` declares the time columns and is resolved with `float32-relative` (the output context of `GPUTimeWordsRelative`); `queryTemporalContext` defaults to it and must have the same unit and epoch, checked by `assertCompatibleGPUTemporalContexts`. Compile-time: the sample and query counts, the mode (`simple` or `ordinary`), `neighborCount`, and `neighborSearch` with its capacities. Everything in the buffer, including the model, changes without a recompile.

Covariance models (`SpaceTimeVariogramModel`) use the gstat 2.1-6 `vgmST` definitions, with `h` the spatial lag and `u` the time lag:

| Model | Covariance |
| --- | --- |
| `separable` | `sill · cs(h) · ct(u)`, on structures standardized to nugget + sill = 1 |
| `product-sum` | `k Cs(h) Ct(u) + Cs(h) + Ct(u)` |
| `metric` | `Cj(sqrt(h² + (anisotropy · u)²))` |
| `sum-metric` | `Cs(h) + Ct(u) + Cj(sqrt(h² + (anisotropy · u)²))` |
| `gneiting` | `sill ψ^-τ exp(-c h^(2γ) ψ^(-βγ)) + nugget at (0,0)`, `ψ = a u^(2α) + 1`, `c = 3 / spatialRange^(2γ)`, `a = (exp(3/τ) - 1) / temporalRange^(2α)`, with `β ∈ [0,1]`, `τ ≥ β` |

Each structure has the families and effective-range convention of the spatial model, and a nugget that applies at its own zero lag: `Cs` at `h = 0`, `Ct` at `u = 0`, `Cj` at zero joint lag. `anisotropy` is spatial units per unit of time (gstat `stAni`). Gneiting is not a gstat model; the definition is Gneiting (2002).

Neighbor search semantics (gstat documents staged and octant search as not yet available, so these are this library's):

| `neighborSearch` | Semantics |
| --- | --- |
| `'metric'` | The `neighborCount` samples nearest in the metric `sqrt(h² + (searchAnisotropy · u)²)`. |
| `'octant'` | Per quadrant of the sample offset from the query and per side in time (sample time before the query is the past, otherwise the future), the `neighborsPerOctant` nearest in the metric; then the `neighborCount` nearest of those. A sample exactly at the query goes to the positive-offset quadrant. |
| `'staged'` | The `stagedCandidateCount` samples nearest in the metric, then the `neighborCount` of those with the largest model covariance to the query, ties going to the nearer candidate. |

Exact metric ties resolve to the lower sample row. `searchRadius` and `maximumTimeLag` exclude samples before the search. `searchAnisotropy` defaults to the anisotropy of a metric or sum-metric model and is required otherwise. The kernel keeps the conditioning rule of `GPULocalKriging`: status 2 below `KRIGING_PIVOT_RATIO_TOLERANCE`, no nugget floor. `output.values` and `status` match `GPULocalKriging`. `sampleIndex` (compile time, default `'auto'`) scans every sample for every query (cost queries × samples) up to 4,096 samples and then builds a per-frame metric grid with identical neighbors, including distance ties; `'brute-force'` always scans and is limited to 1,048,576 samples, `'grid'` always indexes (up to 1,073,741,808 samples, `GPU_SPACE_TIME_KRIGING_INDEX_MAXIMUM_SAMPLE_COUNT`, subject to device bindings and at least 8 storage bindings per stage). Coincident samples, wide ties, restrictive filters or unfilled octant quotas can make a grid query visit the whole index. There is no caller-resident neighbor index input.

The indexed route's algorithmic sample ceiling does not guarantee that the packed sample bindings
fit a device. For one query, one retained neighbor and the ordinary indexed solve, the 16-byte
observation plane fits **8,388,608 samples** in a 128 MiB storage binding at zero offset, or
**8,388,607 samples** after a 16-byte prefix. Other bindings and scratch must also fit. Query
chunking splits queries and their outputs; it does not split the packed samples. The 4,096-sample
auto threshold, 1,048,576-sample forced-scan ceiling and eight-storage-binding indexed requirement
still apply.

`fitSpaceTimeVariogramModel` fits a model to the bins of `GPUSpaceTimeVariogram` on the CPU, reading back at most 409 bins, like `fitVariogramModel`. Use `getGPUSpaceTimeVariogramBinLags` for the representative lags (zero for class 0 of an axis, class midpoints otherwise). Nonlinear parameters (ranges, anisotropy, nugget fractions of the separable and product-sum structures, Gneiting `β`) use the ladder and golden-section search of `fitVariogramModel` in coordinate cycles, and the remaining sills and nuggets are exact non-negative weighted least squares. The spatial and temporal margins start the ranges. `maximumCycles` is an integer in `[1, 24]` (default 24), applied separately to each weighting pass.
The returned `cycleCount` is the actual number of completed coordinate cycles in the final pass,
not the sum over passes. `converged` reports that the final pass met the relative residual-improvement
criterion (`1e-9`); it can be true even at a boundary. `atBoundary` independently reports a nonlinear
parameter at the edge of its ladder. Gneiting `α`, `γ` and `τ` are fixed options, defaulting to 0.5, 0.5 and 1. gstat `variogramST` and `krigeST` fixtures are pending because R is not installed in this environment; covariances and predictions are verified against independent binary64 solves, and GSTools (metric models only, its one space-time family) covers the metric family.

{/* lens:GPUSpaceTimeKriging */}

## GPULocalRBF: local radial-basis interpolation

`GPULocalRBF` accepts scalar values, sample positions and query positions. Nonfinite samples are omitted. `epsilon` multiplies radial distance; `smoothing` adds a nonnegative kernel diagonal; optional `sampleSmoothing` supplies one diagonal per sample and replaces the scalar. Kernels are thin-plate-spline, cubic, Gaussian, linear, quintic, multiquadric, inverse-multiquadric and inverse-quadratic. Polynomial degree is −1, 0, 1, 2: thin-plate/cubic require at least 1; quintic requires 2; linear/multiquadric require at least 0. Gaussian and inverse kernels default to −1; set degree 0 to match SciPy's default constant polynomial.

Use `neighborhoodWeights: search.outputWeights` as above to retain search completeness. Optional legacy `neighborhoods: uint32` restricts candidate indices. Fixed rows use `neighborhoodCapacity` slots and `0xffffffff` padding; `neighborhoodOffsets` instead accepts resident CSR directly from `GPUNeighborSearch`. Each row contains distinct sample indices. The kernel reorders candidates by distance and original row index and retains at most 32. Without candidate inputs it scans all samples. The augmented indefinite system uses partial-pivot elimination, with centered/scaled polynomial coordinates. Zero-smoothing duplicate data and rank-deficient polynomial designs fail explicitly. Outside-hull extrapolation is allowed; `searchRadius` and `neighborCount` delimit the local model. `output.values` is `[prediction, pivot ratio, retained count, 0]`; statuses are 0 success, 1 insufficient neighborhood, 2 singular/numerically unreliable solve, 3 incomplete supplied candidates. Per-query scratch is bounded and there is no global quadratic matrix.

## GPUConditionalGaussianSimulation: conditional Gaussian simulation

`GPUConditionalGaussianSimulation` uses the same covariance and observation row format, ignoring the variable-index lane, and rejects `model: 'matern'`: its dense Cholesky has no pivot ratio check, and Matérn is validated only for local kriging. It conditions a joint Gaussian field with a supplied known `mean`. Up to 128 observations and 128 query sites retain the four-node cooperative path. Larger domains use 16-row blocked Cholesky: a small diagonal factor, parallel panel solves, and 16×16 tiled trailing updates synchronize through GPU dispatch boundaries. Blocked forward solves and parallel query moments prepare a shared conditional factor, then tiled multiplication generates all realizations. There is no CPU solve or readback. By default each encoding repeats preparation. The supported capacity is 4096 observation and query rows combined, counting masked input slots. The factor requires `4 * (2 + N² + 5N)` bytes for combined capacity `N` (about 64 MiB at 4096), and blocked draws add `4 * queryCount * replicateCount` bytes of temporary normal samples. Both must fit an individual device storage binding. Use smaller explicit replicate batches to reduce draw storage; the joint site domain must remain intact. This is still a dense cubic-work, quadratic-storage method, not a large-raster approximation. Query covariance may be semidefinite, so exact observed query sites can have zero variance. Singular observation covariance fails; masks may omit all observations, producing an unconditional joint field.

`values` is replicate-major. `moments: float32x2` holds each query's conditional mean and variance. `status` has one row per replicate: 0 success, 1 invalid input, 2 numerical failure. `seed` and `replicateOffset` identify independent Philox streams, so explicit batches preserve replicate identities. Box–Muller uses f32 transcendental functions: compare reproducibility on the same backend and distributional moments across implementations. Blocking preserves the covariance model and observation-error handling, without silently adding jitter; changed floating-point evaluation order need not reproduce older draw bits. Exact-hit and repeated query sites retain semidefinite support. Splitting query sites into separate calls creates separate joint fields and does not preserve cross-call covariance. Conditioning, loading factors and random normals remain GPU-resident.

### GPUConditionalGaussianSimulationOperation and CPUConditionalGaussianSimulationOperation

`GPUConditionalGaussianSimulationOperation` names graph values in place of the producer's views;
`observationCount` and `queryCount` declare their exact lengths. Both CPU execution and GPU lowering reject resolved input lengths that disagree with these counts, preserving the advertised replicate stride. Inputs are `positions: float32x2`,
`observations: float32x4`, and `queries: float32x2`. Outputs name `values: float32`
(query count times replicate count, replicate-major), `moments: float32x2` (one per query), and
`status: uint32` (one per replicate), with the conventions above. Compile the GPU form with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`. `CPUConditionalGaussianSimulationOperation` uses the same
ports; `computeConditionalGaussianSimulationOnCPU` accepts typed arrays directly. The CPU twin's
binary64 conditioning and random-number arithmetic can give different draw bits; compare moments
and statuses. See [operation forms and CPU twins](../operation-forms-and-cpu-twins.md).

### Reusing preparation

For repeated draw batches, construct `GPUConditionalGaussianSimulation` with `reusePreparation: true`. Add the producer to one graph, compile it, and reuse the compiled graph across submissions. After submitting a batch, call `simulation.setDrawParameters({replicateOffset: nextOffset, replicateCount: nextCount})` before encoding the next one. Both operation forms prepare the simulation on each run.

The constructor's `replicateCount` becomes the maximum mutable batch size. The producer owns persistent factor, moment, control and validity buffers: 17,980 bytes at 32+32 or 268,348 bytes at 128+128, excluding caller inputs and outputs. The blocked path uses GPU-written indirect dispatch gates to skip factor and moment workgroups when their retained generations are current. It can be added to one graph once. Destroy compiled graphs before calling the producer's `destroy()`; input and output buffers remain caller-owned.

| Update | Required operation |
| --- | --- |
| Seed or replicate range | `setDrawParameters({seed?, replicateOffset?, replicateCount?})`; factor and moments are reused |
| Observation values | Write the observation buffer, then call `invalidateObservations()`; validation and conditional moments refresh |
| Known mean | `setMean(mean)`; the mean must be finite in f32 and conditional moments refresh |
| Positions, queries, validity flags or measurement errors | Write inputs, then call `invalidatePreparation()` |
| Compile-time covariance (`covariance`) or view shapes | Create a new producer and graph specialization |

Simulation covariance is captured at construction: changing it requires a new simulation and graph. `invalidatePreparation()` refreshes geometry and errors using that captured covariance; mutable covariance buffers belong to local kriging, described at that operation above.

Input writes do not trigger automatic invalidation. Observation invalidation also checks compacted geometry and error variances and rebuilds if they differ, but callers should use full invalidation for geometry changes. Failed validation never draws from stale moments; invalid inputs and recovery retain the default status contract. Reuse validity is committed by GPU execution, so discarding an encoded command buffer does not mark unbuilt state valid.

The setters enqueue buffer writes. Submit preceding encodings before changing their controls, just as with other mutable input buffers; encoding alone does not snapshot parameter values. Application code still owns command encoding, submission and completion. `replicateCount` must be at least one and at most the constructor capacity, and the last replicate identity must fit uint32. Rows beyond the active batch prefix are cleared to NaN with status 1. Only the active prefix belongs to the requested batch; moments remain one row per query.

If node generation fails (for example, a retained allocation fails), newly acquired buffers are released and preparation statistics remain unpublished. After node generation returns, the public graph API can still reject insertion (for example, a duplicate node ID) and does not notify the producer. In that case its retained buffers and statistics remain owned until you call `producer.destroy()`. On **any** failed `graph.add(producer)`, call `producer.destroy()` and discard the failed graph before rebuilding the graph, imports and producer. Imported IDs and partially inserted nodes cannot be rolled back by this contributor.

After node generation, `preparationStatistics` exposes a `uint32x2` view containing actual GPU factorization and moment-solve attempt counts. It is undefined before node generation and after destruction; its presence does not certify that graph insertion succeeded. Counts are diagnostic, accumulate across executions and use uint32 arithmetic; they are not timing measurements or cache-validity tokens.

## GPUNaturalNeighbor: natural neighbors

`GPUNaturalNeighbor` computes actual Sibson coordinates: it builds the inserted query's Voronoi cell, intersects it with each original sample Voronoi cell, and normalizes the stolen areas. A nearest-triangle method does not provide the same interpolation. Without a topology input, the direct path scans a bounded local site set. With `triangles: uint32`, each six-word triangle stores `[a,b,c,neighborAB,neighborBC,neighborCA]`, using `0xffffffff` at hull edges. `queryTriangles` supplies a containing triangle per query, or `0xffffffff` for outside-hull invalidity. The GPU traverses intersected circumcircles and gathers a bounded cavity (`maximumNeighborCount`, default 32, maximum 64; at most 4× that many inspected triangles), then computes the same stolen areas. Input triangles must be a valid Delaunay triangulation with unique sites and reciprocal adjacency. Site count may reach 2²⁴, subject to GPU buffer limits; no global pair scan occurs in indexed mode. Topology construction and seed location remain explicit caller operations.

Successful `output.values` is `[prediction, inserted cell area, weight sum, contributing count]`. Optional query-major `weights` is scalar in original sample order for the direct path. Indexed mode uses `float32x2` rows `[sampleIndex, weight]` in cavity discovery order, with `maximumNeighborCount` slots per query; unused rows are `[-1,0]`. The 2²⁴ site cap preserves exact integer indices in these f32 diagnostic rows. Weights are nonnegative, sum to one and reproduce affine fields within f32 geometry tolerances. Sample hits reproduce the sample; multiple samples exactly at the query average. The direct path reports duplicate sites elsewhere or globally collinear samples as status 2; indexed mode requires duplicate-free Delaunay topology. Hull-edge queries use the limiting interpolation between the closest bracketing edge samples; outside-hull queries return status 1. Exhausted cavity, traversal, or polygon capacity returns status 3 without a successful partial estimate. Geometry uses query-centered coordinates scaled by the maximum sample distance, with explicit f32 tolerances; extremely thin configurations may be rejected rather than assigned unreliable weights.

## Vecchia likelihood and Fisher operation forms

`GPUVecchiaLogLikelihoodOperation`, `GPUVecchiaSpaceTimeLogLikelihoodOperation`, and
`GPUVecchiaFisherScoringOperation` are the graph operation forms of the correspondingly named
producers. Each names `positions: float32x2`, `observations: float32x4`, CSR
`neighborhoodOffsets: uint32` / `neighborhoods: uint32`, and runtime `parameters: float32` inputs;
`rowCount` and `neighborCount` specialize the graph. The space-time form also names `times: float32`
and declares `temporalContext`. Float observation validity uses the strictly positive comparison
above. Compile these forms with `GPU_CONTRIBUTOR_OPERATION_LOWERING`.

The spatial and space-time likelihood forms name `conditionals: float32x4`
(prediction, latent variance, pivot ratio, retained count), `conditionalStatus: uint32`,
`siteTerms: float32`, `siteStatus: uint32`, `logLikelihood: float32` (high/low pair), and
`status: uint32` (failed, excluded, used site counts). The Fisher form names `siteTerms: float32`
(ten cells per site: likelihood, three log-parameter scores, six upper-triangular information cells),
`siteStatus: uint32`, `totals: float32` (twenty high/low words), and the same three count words in
`status`. Decode totals with `getVecchiaLogLikelihoodResult` or `getVecchiaFisherEvaluation`.
`CPUVecchiaLogLikelihoodOperation`, `CPUVecchiaSpaceTimeLogLikelihoodOperation`, and
`CPUVecchiaFisherScoringOperation` expose the same named ports on CPU. Their direct helpers are
`computeVecchiaLogLikelihoodOnCPU`, `computeVecchiaSpaceTimeLogLikelihoodOnCPU`, and
`computeVecchiaFisherScoringOnCPU`. CPU twins currently consume all usable CSR slots; GPU Fisher and GPU space-time likelihood
retain the first `neighborCount` usable CSR entries, skipping padding, out-of-range IDs and unusable observations without consuming capacity. GPU spatial likelihood retains the nearest
`neighborCount` usable entries. Keep supplied rows within capacity for parity. See the
[Vecchia reference](./vecchia-likelihood.md) for covariance packing, conditioning and status details,
and [operation forms and CPU twins](../operation-forms-and-cpu-twins.md) for graph lowering.

`fitVecchiaCovarianceFisherScoring` returns `stopReason`: `converged`, `iteration-limit`,
`initial-failure`, or `line-search-stall`. The existing booleans keep their meanings. `iterationCount`
and `history.length` count accepted steps; immediate convergence and a first-step stall have empty
history. A stall preserves the last accepted covariance and likelihood. Disabling `fitNugget`, or
starting at zero, fixes the requested nugget, while the effective nugget still respects the model's
floor. At a sill-dependent Matérn floor its score and information carry into the sill direction. An estimated nugget pinned to that floor follows it for both upward and downward sill proposals. At a fixed positive nugget equal to the floor, convergence checks the feasible one-sided directions separately. Zero-score, zero-information coordinates such as range in a marginal-only fit stay fixed; an unusable information solve retains `stopReason: 'line-search-stall'` and reports
`stallReason: 'direction-failure'` without attempting a line search or evaluating a nonfinite
proposal. An attempted line search with no acceptable step instead reports
`stallReason: 'line-search-rejected'`. Both retain the last usable covariance and likelihood,
including the initial fit when no step was accepted.

## Verification and limits

The original numerical suite `interpolation-coverage.spec.ts` uses pinned NumPy covariance-system fixtures, SciPy `RBFInterpolator`, Shapely/GEOS Voronoi intersection areas, analytic conditional moments and a 4096-replicate ensemble covariance check. `test/oracles/interpolation-coverage.py` regenerates independent fixtures with NumPy 2.2.6, SciPy 1.15.3 and Shapely 2.1.2. Tests also cover rank failure, hull edges, exact hits, outside-hull invalidity, forced geometric capacity, and all existing variogram weighting conventions. Node tests parse all mode kernels and enforce the default eight-storage-binding budget.

A native Apple WebGPU/Metal 3 run on October 23, 2026 validated ten small/bounded larger benchmark cases with unchanged and changed input buffers. At the larger shapes, warm encode/submit/queue-completion medians were 0.9 ms for 16-neighbor kriging over 128 queries with 16 support points, 0.7 ms for 16-neighbor RBF over 128 queries, 2.9 ms for 32-sample Sibson over 128 queries, and 3.9 ms for 128 conditional realizations of 16 sites given 16 observations. The original serial three-model variogram fit used 52.1 ms at 256 bins (51.8–117.4 ms over five samples). These historical results predate cooperative fitting and simulation factorization.

The [benchmark procedure and measured phases](https://github.com/visgl/luma.gl/blob/master/modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md) and raw measurement artifact separate host compile, upload, first execution, warm samples, result readback, updates, caller storage and graph transient bytes. These are device-specific observations, not latency guarantees or GPU-only timestamps. The device exposed expanded limits; a separate static audit passed the default eight-binding and shader-memory budgets at maximum supported options. Original run source identifiers were omitted; extraction-time hashes carry an explicit provenance limitation.

These earlier measurements cover the first bounded implementation. The expanded indexed and shared-factor paths are measured below. No general throughput advantage is asserted. Applications should measure their own graph preparation, upload, execution and optional readback before choosing these contributors for a workload.


## Direct reference parity and expanded scaling

`interpolation-advanced.spec.ts` adds pinned GSTools 1.7.0 and PyKrige 1.7.3 mode-by-mode prediction/variance fixtures, SciPy 1.15.3 eight-kernel quadratic RBF fixtures, CGAL 6.0.1 natural-neighbor coordinates and linear interpolation, and gstat 2.1-6 weighted block and three-variable/two-structure LMC predictions. The committed `test/oracles/interpolation-advanced.py`, `interpolation-cgal.cpp`, and `interpolation-gstat.R` regenerate these references. CGAL and R dependencies are offline reference tools, not bundled runtime dependencies. GSTools effective ranges are converted explicitly (`len_scale = range/3` for exponential); gstat Gaussian uses `range/√3`. Arbitrary eight-column designs also use independent NumPy constrained solves.

The simulation comparison checks GSTools conditional means/variances and marginal conditioned-field moments, and gstat sequential conditional simulation for full joint covariance. GSTools 1.7 `CondSRF` pointwise rescales an unconditional random field by kriging variance, which generally gives a different off-diagonal covariance from exact joint Gaussian conditioning; this implementation does not claim joint distribution equality with that approximation. Seeded sequences are checked for this backend, while independent-library ensembles are compared statistically.

Expanded benchmarks include resident grid/CSR RBF at 1,024 and 16,384 samples, indexed Sibson at 1,024/16,384/65,536 sites, and shared-factor simulation at 32+32 and 128+128 observations/sites. They measure compile, upload, first/warm/updated execution, readback and transient storage separately. A repeated-fit comparison measures resident fitting against actual empirical-bin readback, CPU fitting, parameter upload and queue completion at 20/256 bins. These are observed workload comparisons, not a general GPU speedup claim.


On the same native Apple WebGPU/Metal device, the October 23 expanded run recorded these historical warm encode/submit/queue-completion medians (five samples, output readback excluded):

| Workload | Shape | Median | Graph transient bytes |
| --- | --- | ---: | ---: |
| Indexed Sibson | 1,024 / 16,384 / 65,536 sites, 128 queries | 0.3 / 0.3 / 0.4 ms | 0 |
| Resident grid + CSR RBF | 1,024 / 16,384 sites, 128 queries, 32 neighbors | 5.6 / 6.9 ms | 76,356 / 420,424 |
| Joint simulation | 32+32 / 128+128 observations+queries, 128 realizations | 5.1 / 146.2 ms | 17,672 / 267,272 |
| Three-model resident fit | 20 / 256 bins | 4.3 / 51.3 ms | 0 |
| Actual CPU fit round trip | Same bins, including readback and parameter reupload | 1.3 / 3.2 ms | Caller buffers only |

At that checkpoint, the CPU fit round trip was faster than the serial GPU fit. Later cooperative refinement records 0.7 ms warm at 256 bins in three sessions and 0.7 ms for changed-input GPU execution versus 2.3–2.5 ms CPU round trips; small fits have no consistent comparator winner. Cooperative simulation factorization records 8.1 ms at 128+128 across three sessions. See the refinement report (`dev-docs/research/gpu-geospatial-performance-2026-10-23/integration-notes.md` in the repository) and simulation report (`dev-docs/research/gpu-geospatial-performance-2026-10-23/integration-notes.md` in the repository) for source hashes, first-use variability and validation. These results precede preparation reuse. A later reuse comparison (`dev-docs/research/gpu-geospatial-performance-2026-10-23/integration-notes.md` in the repository) measures complete sequences: 100 large draw batches take 92.55 ms retained versus 816.70 ms forced, including first preparation and per-batch completion. Each batch has 128 replicates and overwrites the same output buffer; per-batch readback and output retention are excluded. Value updates improve 2.66 times, geometry updates show no gain, and single-batch reuse has no demonstrated benefit. Indexed Sibson timings exclude topology construction and seed location; RBF timings include GPU grid construction and neighbor search. The full expanded record includes first execution, phase timings, every warm sample, changed-input validation, device metadata and capture-time source hashes. GPU timestamps were not enabled.


The blocked dense path adds full conditional-covariance checks at 129+129 and 128+384 rows, including repeated query sites, noiseless observation hits, masked counts crossing block boundaries, and an unconditional field. Retained lifecycle tests cover both algorithms, including discarded encodings and smaller active replicate batches. The optional `conditional-simulation-reuse-benchmark.spec.ts` also includes 256+256 and 512+512 domains; enable it with `VITE_SIMULATION_REUSE_BENCHMARK=true` and select workloads using the shared benchmark filter. These added sizes have no published timing claim; the historical measurements above cover the cooperative path.

## Great-circle raster neighborhoods

`GPUInverseDistanceWeighting` and `GPUKriging` accept `longitudeSearch: 'wrapped'` with a
great-circle spatial context and index-bound longitudes within `[-180, 180]` (checked at
construction). Wrapped searches reduce per-frame query longitudes to a canonical turn before
constructing candidate boxes, including multi-turn placements and inclusive ±180° endpoints.
This construction option searches both intersecting longitude
intervals across ±180° and all longitudes for a cap reaching a pole. Each represented index cell
is visited once, even when the widened intervals overlap in a coarse index. The default
`longitudeSearch: 'clipped'` preserves the clipped candidate-box route; bounded searches on that
route can omit samples across the seam. Distances and radius tests remain spherical on both routes.

## Raster IDW settings and focal outputs

Use `idw.getParameterValues(settings, grid)` before uploading IDW settings. This checks grid
width/height against construction and rejects positive `neighborCount` on a radius-only instance
(`maximumNeighborCount: 0`) or a request above the allocated capacity. The free packer remains
available for callers managing the encoded layout. Negative/NaN radii and negative power retain
the documented GPU nodata behavior (NaN cells, zero counts); valid finite settings that overflow
or lose their usable positive domain in float32 are rejected. For values 10 and 100 at distances 1 and 2,
nearest-one gives 10, while radius-only power-two IDW gives `(10 + 100/4) / (1 + 1/4) = 28`.
`searchRadius` uses position units (meters for a great-circle context); counts are sample counts,
and `power` is dimensionless. The clipped and wrapped longitude routes keep their existing rules.

Corner-derived IDW, raster-kriging and density packers reject float32 corners whose direction or finite nonzero per-cell step is lost.
For projected coordinates such as x = 5,000,000 with a 0.125-unit cell, subtract a common tile
origin from grid, positions and index bounds before packing. Positive covariance ranges must
remain positive and finite in float32; modified-Shepard radii must also remain finite.

`GPUFocalStatistics` requires one terminal output at minimum. Minimal selections are
`output: {mean}`, `{sum}`, `{min}`, `{max}`, `{range}`, `{standardDeviation}`, or `{counts}`.
Each float output is in the input value unit; counts is a uint32 valid-cell count. Radius and
`maximumRadius` use cells. Windows clip at physical edges and skip invalid neighbors by default:
for `[1, NaN, 3]` and radius 1, means are `[1, 2, 3]` and counts `[1, 2, 1]`. The missing center
is filled unless `propagateCenterNoData` is enabled.
