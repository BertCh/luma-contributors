# From observations to a defensible spatial result

## Summary

The most consequential defect is in the space-time Hawkes fit. The CPU twin reports convergence at parameters where changing just the decay rate improves the library’s own likelihood. I recommend fixing the estimator before the alpha.

Two ordinary GIS workflows can produce plausible numbers for a different question. IDW accepts a nearest-neighbor request that its default constructor cannot honor. SKATER can produce two regions whose labels are `0` and `2`, so sizing a downstream evaluation for two labels silently drops a region.

The clustering algorithms themselves agree on a useful convention. On a core–border–noise fixture, the library and scikit-learn agreed on every assignment after translating `0xffffffff` to `-1`. The Gi*, Mann–Kendall, ordinary kriging and isotropic Ripley checks also agreed with independent calculations.

The highest-value ergonomic change is a contributor-bound parameter packer. It can check frame settings against the actual grid, neighborhood capacity and presence of a time column before the caller uploads them. Emerging hot spots already provides this pattern.

## Census

`S` below means `modules/analysis/src/gpu-spatial-analysis`, `D` means `docs/api/gpu-spatial-analysis`, and `T` means `modules/analysis/test`. **KI** identifies a file read in the newer `luma-contributors-ki` checkout after checking `.typecheck/r19/staging-owned.txt`. Other paths refer to `luma-contributors-r19`.

I counted declarations matched by `export class (\w+)` in the 19 requested directories and retained names exported by the corresponding `index.ts`. I counted public `export function …ParameterValues` and `compute…OnCPU` declarations the same way. The inventory used the KI file for a listed path. `space-time-*` matched `space-time-hawkes`, `space-time-statistics` and `space-time-tests`.

| Public shape | Count | Distribution |
|---|---:|---|
| GPU contributors | 55 | 12 interpolation, 6 regionalization, 5 point density, 5 space-time tests, 4 clustering, 4 point process, 19 elsewhere |
| GPU operation classes | 19 | Interpolation and regionalization 4 each, point process 3, change points 2, six other families 1 each |
| CPU operation classes / direct `compute…OnCPU` functions | 19 / 19 | The distribution matches the GPU operation forms |
| Named parameter-packing functions | 28 | 19 return `Float32Array`, 9 return `Uint32Array` |
| Named packers accepting a caller destination | 18 | All 18 return `Float32Array`. Six clear elements beyond their declared layout |
| Contributors declaring `getParameterValues(settings)` | 2 | `GPUEmergingHotSpots`, `GPULocationChangePoints` |
| Clustering contributors with compact labels and unsigned noise | 4 | `GPUSpatialClustering`, `GPUOPTICS`, `GPUHierarchicalDensityClustering`, `GPUSpatialKMeans` |
| Regionalizers exposing row identifiers as region labels | 3 | `GPUSkaterRegions`, `GPUWardRegions`, `GPURedcapSingleLinkage` |

Two small checks are particularly easy to repeat:

```sh
rg -n '^getGPU\w+ParameterValues\(\{' docs/api/gpu-spatial-analysis
rg -n '(target|destination)\.fill\(0\)' modules/analysis/src/gpu-spatial-analysis/spatial-interpolation
```

The first found two examples that discard their freshly packed settings. The second found four unbounded clears. Following the two public Vecchia forwarding functions brings the affected packer count to six.

### Fifteen calls from the reference pages

These are the expressions I checked. Identifiers such as `output`, `parameters` and `treeEdgeFlags` stand for caller-owned ports of the documented format. `raster` below means `{positions, grid, indexBounds, indexGridSize, parameters}`. The table distinguishes host execution, test-only CPU oracles and shader source. A test-only oracle is not a public CPU twin.

| Operation | Call or setting a reader would write | Check |
|---|---|---|
| Point density | `new GPUPointDensity({positions, bounds, gridSize, statistic: 'count', output})` | `D/operators-density.md:11` correctly distinguishes count per cell from intensity per unit area. |
| Line density | `getGPULineDensityParameterValues({minX, minY, cellWidth: 1, cellHeight: grid.affine[4]})` | EXECUTED. A north-up step of `-1` throws. Making it positive reverses the physical row order relative to a north-up KDE. |
| Fixed KDE | `new GPUKernelDensity({...raster, edgeCorrection: 'uniform', normalization: 'probability', output})` | EXECUTED public kernel and window-mass functions. The documented normalization can give integrated mass `0.48043`. |
| Space-time KDE | `getGPUSpaceTimeKernelDensityParameterValues({bandwidth: 1, temporalBandwidth: 1, firstSliceTime: 7, sliceInterval: 1}, grid, 1)` | EXECUTED. The default time window packs as `[7,7]`. For a single corrected slice the reader needs an explicit observation window. |
| ST-DBSCAN | `new GPUSpatialClustering({positions, times, temporalContext, parameters, gridSize, output: {labels}})` followed by `parameters.write(words)` | EXECUTED host constructor with graph views. The same `parameters` object cannot serve both roles in the example. |
| OPTICS | `computeOPTICSOnCPU(positions, {minimumSamples: 3, maximumDistance: 1, extractionMethod: 'dbscan', extractionDistance: 1})` | The CPU twin’s default DBSCAN extraction gave `[0,0,0,0,4294967295]` on the fixture below. Use the visit-completion status before consuming labels. |
| IDW | `new GPUInverseDistanceWeighting({...raster, values, output})` and `getGPUInverseDistanceWeightingParameterValues({searchRadius: 3, neighborCount: 1}, grid)` | EXECUTED constructor and test oracle. The constructor selects capacity `0`; the two possible answers are `10` and `28`. |
| Raster kriging | `getGPUKrigingParameterValues({searchRadius: Infinity, neighborCount: 8, variogram: {model: 'exponential', nugget: 2, sill: 40, range: 450}}, grid)` | An independent SciPy ordinary-kriging solve reproduced the fixture’s first prediction and variance. `sill` is the partial sill. |
| Emerging hot spots | `hotSpots.getParameterValues({radius: 0, temporalWindow: 1, confidenceLevel: 0.95})` | The CPU test oracle agreed with independent Gi* and tied-series Mann–Kendall calculations. The instance method carries the contributor’s multiple-testing setting. |
| Inhomogeneous point pattern | `new GPUInhomogeneousPointPattern({positions, studyWindow, suppliedIntensity, radii, edgeCorrection: 'translation', output})` | SOURCE. The page distinguishes polygon translation overlap from border erosion and the pcf half-width from the intensity bandwidth. I checked homogeneous Ripley separately below. |
| SKATER | `new GPUSkaterRegions({weights, treeEdgeFlags, componentLabels, values, maximumRegionCount: 2, parameters, output})` | EXECUTED direct CPU twin. A four-row path yields two regions labeled `[0,0,2,2]`. |
| Expectation-based scan | `getGPUExpectationBasedScanParameterValues({seed: 2**32 + 7, replicateCount: 9})` | EXECUTED KI packer, yielding `[7,1,9,0]`. The spatial scan preserves the same two seed words. |
| Point-pattern envelope | `computePointPatternEnvelopeOnCPU(input, {simulationCount: 2, statistic: 'extreme-rank', nullModel: 'CSR', fittedParameters: 'none'})` | EXECUTED KI twin. Two arrays unused by the rank statistic still have mandatory shapes. |
| Sinkhorn divergence | `getGPUSinkhornDivergenceParameterValues({cellWidth: 1, cellHeight: 1, epsilon: 1, epsilonStart: 2})` | EXECUTED packer and CPU twin. The default schedule spends the whole budget at `2`. |
| Space-time Hawkes | `computeSpaceTimeHawkesOnCPU({positions, times, parameters}, {maximumIterations: 1000})` | EXECUTED KI twin. Its convergence flag and the independent likelihood disagree about whether the fit is stationary. |

For a sense of first-use cost, the Sinkhorn test harness goes from graph creation at line 59 to the first scalar `divergence` read at line 138, an 80-line span including allocation, options and diagnostic bindings. That is an actual fully instrumented path, not a claimed minimum. The public example can save a reader most of that work only if it shows which object receives the packed parameters. `T/gpu-spatial-analysis/optimal-transport/stm09-sinkhorn-harness.ts:59`

### Independent numerical checks

I ran the TypeScript probes through esbuild to Node over a pipe. The separate `/usr/bin/python3` process used NumPy 2.0.2, SciPy 1.13.1 and scikit-learn 1.6.1.

| Definition | Input | TypeScript result | Independent result |
|---|---|---|---|
| Gaussian KDE bandwidth | `h=2`, distance `0` | Public evaluator `0.039788735772973836` | SciPy bivariate normal with covariance `4I` gives `0.03978873577297383` |
| DBSCAN cores, border and noise | x coordinates `[0,.25,.5,1.5,6]`, all y `0`, `eps=1`, minimum `3` | Test oracle cores `[1,1,1,0,0]`, labels `[0,0,0,0,4294967295]` | scikit-learn cores `[0,1,2]` by index, labels `[0,0,0,0,-1]`. Row 3 is border at exactly the radius |
| IDW | Distances `[1,2]`, values `[10,100]`, power `2`, radius `3` | Test oracle `28` with both neighbors, `10` with one | SciPy `cKDTree` and NumPy weighted mean `28`; nearest value `10` |
| Ordinary kriging | First cell of `kriging-reference.ts`, eight nearest samples, exponential partial sill `40`, nugget `2`, range `450` | Stored external fixture `59.985624`, variance `28.026750` | Independent SciPy augmented-system solve `59.98562429`, variance `28.02674954` |
| Ripley K | Window `[0,0,5,5]`, points `(0,1),(1,1),(4,4)`, `r=1.25` | CPU test oracle uncorrected `8.33333333`, isotropic `12.5`, border `NaN` | `25×2/6` uncorrected. The two directed contributions have isotropic weights `2` and `1`, giving `25×3/6`. No eligible border focus |
| Getis–Ord Gi* | Cell-major `[1,2,3,4,5,6]`, two cells, three slices, spatial radius `0`, one preceding slice | Test oracle first cell `[-1.46385011,-1.85164020,-0.92582010]` | Same three values using global population SD and binary window weights |
| Mann–Kendall | `[1,1,2,3]` | Test oracle `S=5`, variance `7.66666667`, corrected `z=1.44463024`, `p=.14856178` | Independent tie count and SciPy normal tail `p=.14856177` |

## Proposed convention

Every public contributor should have an executable example that takes named settings through the actual parameter write to one decoded result, with the grid, neighborhood, label domain and completion condition used by that result. A host checker should verify encoded values, layout bounds and capacity compatibility; a small independent numerical fixture should verify the statistic or estimator.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UPT-01 | W | ORDINARY | BREAKING | M | Space-time Hawkes | A converged fit can improve its own likelihood by changing only the decay rate. |
| UPT-02 | W | UNUSUAL | ADDITIVE | S | Six interpolation packers | Packing one layout clears subsequent caller-owned words. |
| UPT-03 | W | ORDINARY | ADDITIVE | S | KDE, IDW, kriging packers | A valid projected grid can become zero-width during packing. |
| UPT-04 | W | UNUSUAL | ADDITIVE | S | Spatial scan, similar locations, DBSCAN | Invalid external settings can become a different valid analysis. |
| UPT-05 | W | UNUSUAL | ADDITIVE | S | Two public KDE mass helpers | One returns negative Gaussian mass; the other loses a representable tail. |
| UPT-06 | C | ORDINARY | ADDITIVE | S | IDW and ST-DBSCAN | A valid free-standing frame packer cannot tell whether its contributor can honor the request. |
| UPT-07 | C | ORDINARY | ADDITIVE | M | SKATER, Ward, REDCAP | Region count does not bound region labels consumed by the next operation. |
| UPT-08 | D | ORDINARY | ADDITIVE | S | KDE normalization | The default edge correction can leave a “probability” surface with less than half the stated unit mass. |
| UPT-09 | S | ORDINARY | ADDITIVE | S | Three first-use examples | One confuses a graph view with its owner; two discard the packed settings. |
| UPT-10 | C | ORDINARY | ADDITIVE | M | Automatic KDE bandwidth | Excluded events can determine the entire surface’s smoothing scale. |
| UPT-11 | S | EXTREME | ADDITIVE | M | Automatic KDE bandwidth | Rescaling all weights can destroy an otherwise well-defined Scott bandwidth. |
| UPT-12 | C | ORDINARY | ADDITIVE | M | Line density | A density raster cannot share the north-up placement accepted by its siblings. |
| UPT-13 | E | ORDINARY | ADDITIVE | S | Sinkhorn | The packer accepts a schedule that cannot reach the requested regularization. |
| UPT-14 | E | ORDINARY | ADDITIVE | M | Point-pattern envelopes | Two unused input arrays stand between an observed curve and its rank test. |
| UPT-15 | D | ORDINARY | ADDITIVE | S | KDE and raster kriging reference | The bandwidth formula and advertised variogram choices omit implemented behavior. |
| UPT-16 | D | UNUSUAL | ADDITIVE | S | Geographic DBSCAN | The suggested centroid workaround can invalidate every point. |

### UPT-01: The Hawkes decay update omits the finite-window compensator derivative

**Where.** `S/space-time-hawkes/cpu-space-time-hawkes-operation.ts:194,200,205` (KI), `gpu-space-time-hawkes.ts:667,680`, `D/space-time-hawkes.md:31`. This builds on KNOWN(PTS-11) and KNOWN(PTS-140). Both likelihood outputs now make it possible to demonstrate a distinct estimator defect.

**EXECUTED.** Positions `(0,0),(1,0),(0,1),(1,1)`, times `[.55,.70,.85,1]`, area `16`, duration `1`, maximum lag `2`, maximum distance `4`. Start `(background, branchingRatio, decayRate, spatialSD)` at `(.1,.5,.1,.8)`. All earlier pairs are eligible. One CPU iteration moved log likelihood from **−10.5944624 to −10.6645994**. Independent evaluation of the documented point-process likelihood gave **−10.5944629 to −10.6645992**. With tolerance `1e-6`, the CPU reported `converged=1` at iteration 25. Reducing only its fitted decay rate by 10% improved log likelihood from **−8.7896780 to −8.7031654**.

**Now.** `if (result.converged[0]) use(result.estimates)` accepts a nonstationary fit. The update uses `triggeredSum / lagSum`. With branching ratio `ω`, the decay score also contains `−ω Σᵢ (T−tᵢ) exp(−β(T−tᵢ))`.

**Wanted.** Optimize decay against the finite-window objective, refitting the branching ratio for the candidate decay. Safeguard the update against a likelihood decrease. Test both the first step and the final numerical score with the fixture above. **SOURCE GPU probe:** submit these four events with `maximumIterations: 1`, read `logLikelihood` and `fittedLogLikelihood`, then repeat to convergence.

**In-repository callers that change.** No signature changes. The numerical contract changes at **50** direct constructor/helper occurrences from `rg` over `modules`, `examples` and `website`, excluding definitions and `dist`. Of these, 45 are in analysis, 3 in kepler and 2 in the website. Recompute the Hawkes fixtures and inspect catalog/lens outputs that display estimates. **Value and cost.** Users fitting triggering rates can interpret convergence against the published likelihood. M.

### UPT-02: Six packers overwrite the next parameter block

**Where.** `S/spatial-interpolation/kriging-parameters.ts:127`, `gaussian-random-field-parameters.ts:62`, `prediction-covariance.ts:288,526`, `vecchia-parameters.ts:78`, `gpu-vecchia-space-time-log-likelihood.ts:341`.

**EXECUTED.** I supplied `new Float32Array(layoutLength + 2).fill(77)` as the target. Raster kriging, local kriging, space-time kriging, Gaussian random field and both Vecchia likelihood packers each changed the final two words to **`[0,0]`**. The KDE control preserved `[77,77]`. Space-time kriging also cleared those words before throwing on an invalid separable sill.

**Now.** `pack(settings, frameWords.subarray(offset))` corrupts a later frame field even though the destination meets the documented “at least” length. **Wanted.** Clear only `[0, layoutLength)`, including for variable-length local kriging. Validate before modifying the destination. Add the two-word canary to the packer tests. **Callers that change:** none. **Value and cost:** callers can safely reuse a staging array across contributors. S.

### UPT-03: Projected grid corners can collapse after validation

**Where.** `S/spatial-interpolation/spatial-interpolation-parameters.ts:40,129`, `kriging-parameters.ts:128`, `S/point-density/gpu-kernel-density.ts:174,195`. Adaptive and space-time KDE forward through the KDE packer at `gpu-adaptive-bandwidth-kernel-density.ts:111` and `space-time-kernel-density-math.ts:97`. This extends KNOWN(DPC-108) to raster placement and supplies a concrete case for KNOWN(PTS-139).

**EXECUTED.** A one-cell area grid with `affine: [.125,0,5000000,0,-1,1]` has x corners `5000000` and `5000000.125`. KDE, IDW and kriging all returned **`[5000000,1,5000000,0]`**. The KDE nonzero-width check runs before narrowing. In separate probes a positive KDE bandwidth and kriging range of `1e-50` both packed as `0`; finite IDW `searchRadius: 1e50` with modified Shepard taper packed as `Infinity`, which violates that taper’s finite-radius requirement.

**Wanted.** Compare the encoded corners and validate encoded field domains before writing. A useful error is `grid x corners coincide in float32; subtract a local origin from grid and positions before upload`. **SOURCE GPU probe:** one IDW cell as above, samples at `(5000000,.5)` and `(5000000.5,.5)` with values `0,100`, power `0`, both within radius. The intended center is a non-hit with answer `50`; check whether the collapsed center takes the exact-hit branch. **Callers that change:** none for representable grids. **Value and cost:** projected-coordinate users get an actionable error before constructing a misleading raster. S.

### UPT-04: Misspelled directions and short bounds turn into valid control words

**Where.** `S/scan-statistics/scan-statistic-parameters.ts:229` (KI), `S/similar-locations/similar-locations-parameters.ts:73` (KI), `S/spatial-clustering/spatial-clustering-parameters.ts:88,129`.

**EXECUTED.** From untyped configuration, `rateDirection: 'low-rate'` packed `0` (**high**), while `'low'` packed `1`. `windowShape: 'knn'` packed `0` (**circle**). Similar locations accepted `direction: 'least-similar'` and packed `0` (**most**). DBSCAN accepted `bounds: [-1,-1,1]`. With `epsilon: 8`, `minimumPoints: 3` and an eight-word target filled with `77`, it returned `[-1,-1,1,8,3,0,0,77]`. The intended radius became the fourth bound.

**Wanted.** Reject unknown enum values and require `bounds.length === 4` before writing. Name the field, received value and allowed alternatives. Test plain JavaScript and typed-array bounds. **Callers that change:** none with valid settings. **Value and cost:** JSON-configured applications cannot accidentally search for the opposite anomaly or ranking. S.

### UPT-05: Gaussian window-mass helpers can return a negative number or zero

**Where.** `S/point-density/kernel-density-math.ts:86,167`, `space-time-kernel-density-math.ts:144,159`, `kernel-density-wgsl.ts:62`, `space-time-kernel-density-wgsl.ts:48`. The two public helpers are `getGPUKernelDensityWindowMass` and `getGPUSpaceTimeTemporalWindowMass`.

**EXECUTED.** Gaussian `h=1`, point `(0,0)`, rectangle `[-1e-8,-1,1e-8,1]` returned **`−1.5033576e−8`**. SciPy gives **`5.4470741e−9`**. The erfc approximation error dominates subtraction near zero. The temporal helper at `t=0`, window `[9,10]`, returned **`0`**; `norm.sf(9)-norm.sf(10)` gives **`1.1285122e−19`**. The temporal shader already uses a tail-difference branch that its stated float64 reference lacks.

**Wanted.** Share a stable interval-mass implementation with a narrow-interval branch and tail subtraction. Test sign and relative error against SciPy. **SOURCE GPU probe:** expose the two WGSL mass functions for the exact intervals above and read one float each. The public host defect is independent of the estimator’s small-mass guard. **Callers that change:** none. **Value and cost:** edge-correction references return valid probability measures. S.

### UPT-06: Nearest-neighbor and time constraints need the contributor’s settings

**Where.** `S/spatial-interpolation/gpu-inverse-distance-weighting.ts:264,481,717`, `spatial-interpolation-parameters.ts:105`; `S/spatial-clustering/spatial-clustering-parameters.ts:52,129`, `spatial-clustering-kernels.ts:143`. Compare `GPUEmergingHotSpots.getParameterValues`.

**EXECUTED + SOURCE.** Constructing IDW without `maximumNeighborCount` produced `maximumNeighborCount === 0`. Its free packer accepted `neighborCount: 1`. The compiled radius-only branch ignores that request. The test oracle gave **`28`** using the radius and **`10`** using the nearest sample. Similarly, the DBSCAN packer accepts omitted `timeEpsilon` and writes `0`; the temporal lattice requires a positive word. The constructor knows whether `times` exists, while the packer does not.

**Wanted.** Add `idw.getParameterValues(settings, grid)` and `clustering.getParameterValues(settings)`. Reject a requested positive `k` on a radius-only instance, a `k` above capacity, and missing `timeEpsilon` on a temporal instance. Keep the raw packers for code that owns the encoded layout. **SOURCE GPU probes:** run the two-sample IDW fixture in the census with default capacity and requested `k=1`; run two coincident timed DBSCAN events with `minimumPoints: 1` and omitted time epsilon. **Callers that change:** none for the additive methods. **Value and cost:** users get the neighborhood they requested or a diagnosis at upload. S.

### UPT-07: Region count cannot be used as label capacity

**Where.** `S/spatial-regionalization/cpu-skater-regions-operation.ts:174`, `gpu-skater-regions.ts:179` (KI), `gpu-ward-regions.ts:167` (KI), `S/redcap-regions/gpu-redcap-single-linkage.ts:113` (KI), `gpu-region-partition-evaluation.ts:112,147,607` (KI). `D/operators-statistics.md:2037` (KI) explicitly recommends evaluating SKATER labels.

**EXECUTED.** On a path `0–1–2–3` with attributes `[0,0,10,10]`, `computeSkaterRegionsOnCPU(..., {maximumRegionCount: 2})` returned **`regionCount=[2]`, `labels=[0,0,2,2]`**. The evaluator interprets `labelCapacity: 2` as accepting labels `0` and `1`. Its default of row count is safe here. Passing the known number of regions as an optimization excludes half the dataset.

**Wanted.** Show this four-row example next to the composition. Offer optional `compactLabels` or one reusable compaction operation returning `{labels, sourceLabels, count}`. Keep the existing row identifiers for callers joining back to root features. **SOURCE GPU probe:** evaluate `[0,0,2,2]` twice with capacities `2` and `4` and read `ignoredRows`, region sizes and `pseudoF`. **Callers that change:** none. **Value and cost:** spatial analysts can compare partitions on the same population. Documentation S, compaction M.

### UPT-08: Uniform edge correction does not preserve probability mass

**Where.** `D/operators-density.md:241` (KI), `S/point-density/gpu-kernel-density.ts:213,337`, `kernel-density-wgsl.ts:143`. The analogous claim appears in `gpu-space-time-kernel-density.ts:92`.

**EXECUTED.** For one unit-weight event at the corner of `[0,10]²`, `h=1`, I integrated the public truncated Gaussian kernel divided by the public window mass on a `400×400` midpoint grid. The result was **`0.48042896`**. Independent SciPy quadrature for the untruncated Gaussian gave **`0.48045301`**, close to `(ln 2)²`. The docs describe probability normalization as integrating to “about one”; elsewhere the same page correctly distinguishes the mass-preserving event-wise correction.

**Wanted page text.** “`probability` divides by total included event weight. With `uniform` edge correction, the integral over the observation window depends on event locations. Use event-wise `diggle` correction when window mass must be conserved, allowing for truncation and raster quadrature.” **SOURCE GPU probe:** run the corner scene in exact mode; sum `values × cellArea` for both corrections. **Callers that change:** none. **Value and cost:** users integrating a density surface can choose the right estimator. S.

### UPT-09: Three examples fail at the parameter write

**Where.** `D/space-time-clustering.md:55,59,63`, `D/space-time-hawkes.md:47,63`, `D/sinkhorn-divergence.md:28,43`, `modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts:103,112`. This adds concrete failures to KNOWN(DOC-110) and KNOWN(ERG-06).

**EXECUTED.** I passed a real `GPUParameterBuffer` backed by a host stub into the DBSCAN constructor. It threw `spatial-clustering parameters must be packed, uint32-aligned float32 GPU data`. A valid graph view was accepted, but `typeof view.write` was **`"undefined"`**. The Hawkes and Sinkhorn examples each call their allocating packer as a bare expression. Neither sends the returned words to the bound input.

**Wanted.** Use `parameters: parameterBuffer.importToGraph(graph)` at construction and `parameterBuffer.write(get…ParameterValues(settings))` at update, or initialize with `values` before import. Add a snippet check that fails when an allocating parameter packer’s return value is discarded. **Callers that change:** three documentation snippets, no application source. **Value and cost:** all three pages lead to initialized inputs on the first attempt. S.

### UPT-10: An excluded point can set the bandwidth for included points

**Where.** `S/point-density/gpu-kernel-density.ts:309,317`, `kernel-density-bandwidth.ts:115,156,318`, `D/operators-density.md:269` (KI).

**Evidence.** The page documents that automatic bandwidth estimation ignores `indexBounds` and the study window. The estimator can therefore use different populations for smoothing, normalization and `invalidCount`. Independent SciPy calculation for `(0,0),(1,0),(0,1)` gives pooled Scott `h=0.48074986`. Appending `(1000,1000)` gives **`396.718156`**, even if that fourth event is deliberately outside the observation domain and contributes no density.

**Wanted.** Add `bandwidthPopulation: 'included' | 'all-finite'`, make the population explicit in the automatic-bandwidth example, and return its row count alongside the optional bandwidth. **SOURCE GPU probe:** use a fixed index domain `[-1,-1,2,2]`, exact Gaussian KDE and unit weights; compare `bandwidth`, `weightSum` and `invalidCount` before and after appending the fourth point. **Callers that change:** none for the additive option. **Value and cost:** changing the study population has predictable statistical consequences. M.

### UPT-11: Scott bandwidth is not invariant to weight units across the float32 domain

**Where.** `S/point-density/kernel-density-bandwidth.ts:218,315`. Both `bandwidthRule: 'scott'` and `'silverman'` take this route.

**SOURCE.** The first pass stores `weight * weight` in float32. The final pass forms `total * total / totalSquare`. Three finite weights of `1e20` overflow the former product; three weights of `1e-24` round the squared terms to zero. The statistical bandwidth is well-defined in both cases. **EXECUTED reference:** SciPy gave **`0.4807498568`** on the triangle in UPT-10 at weight scales `1`, `1e20` and `1e-24`.

**Wanted.** Normalize weights by a shared power of two before computing the bandwidth moments. Sinkhorn already uses this strategy for mass fields. **Smallest GPU probe:** three triangle vertices, three equal weights, three submissions changing only the common scale; compare `output.bandwidth`. **Callers that change:** none. **Value and cost:** a change of weight units preserves the smoothing scale. M.

### UPT-12: Line-density row zero is physically opposite the usual raster row zero

**Where.** `S/line-density/gpu-line-density.ts:105,125,186`, `line-density-kernels.ts:277,282`, `D/operators-geometry.md:881` (KI). Compare `S/spatial-interpolation/spatial-interpolation-parameters.ts:40`.

**EXECUTED.** For a one-column, two-row north-up raster, taking `cellHeight` from `affine[4]` produced `line-density field cellHeight is below its minimum`. Taking its absolute value packed `[0,0,1,1]`, a lower-left origin. KDE, adaptive KDE, space-time KDE, IDW and raster kriging accept raster metadata and preserve its row order. The primary hazard is an overlay or cell-by-cell comparison that looks correctly shaped.

**Wanted.** Supply a `GPURasterMetadata` overload with a defined output row mapping. In the meantime, show the explicit row reversal when combining a south-first line-density result with a north-up raster. **SOURCE GPU probe:** put one short horizontal line entirely in the northern cell of a `1×2` grid, then overlay the two returned values on the original grid. **Callers that change:** none. **Value and cost:** GIS users can join rasters by cell without reflecting the data. Documentation S, native row-order support M.

### UPT-13: A legal Sinkhorn schedule can never reach its target

**Where.** `S/optimal-transport/sinkhorn-divergence-parameters.ts:29,48`, `cpu-sinkhorn-divergence-operation.ts:244`, `D/sinkhorn-divergence.md:55,60`. The docs acknowledge the behavior. The new issue is that the packer can diagnose it exactly before any solver work.

**EXECUTED.** Source `[9,1]`, target `[1,9]`, unit cells, `epsilon: 1`, `epsilonStart: 2`. With omitted `scalingFactor`, the CPU returned divergence **`0.64337438`**, residual `0`, `converged=0`, iterations `40/40`. Setting `scalingFactor: .5` returned **`0.65258425`**, `converged=1`, iterations `9`. Starting directly at `1` gave the same latter answer.

**Wanted.** Add a schedule check to the contributor-bound packer that diagnoses `epsilonStart > epsilon` and encoded `scalingFactor === 1`. Preserve the raw packer for solver-status tests. **Callers that change:** none with this additive check. **Value and cost:** analysts avoid spending the entire iteration budget on a different regularization. S.

### UPT-14: Extreme-rank inference requires two dummy arrays

**Where.** `S/point-process/gpu-point-pattern-envelope.ts:35,128` (KI), `cpu-point-pattern-envelope-operation.ts:98,109,134,165` (KI), `D/point-process.md:80` (KI).

**EXECUTED.** Observed `[3]`, simulations `[1,2]`, extreme-rank test. Empty `integrationWeights` threw `integrationWeights length must equal observed.length`; `[0]` and `[1]` both produced status `0` and conservative p-value `2/3`. The reference is also shape-checked and unused in this branch. For maximum deviation, changing the scientifically irrelevant integration weight from `1` to `0` changed a valid p-value of `1/3` to `NaN`, status `1`.

**Wanted.** Use mode-specific props. Extreme rank needs observed and simulated curves. Maximum deviation additionally needs a reference. Integrated squared deviation additionally needs integration weights. Continue accepting existing complete input bags. **Callers that change:** none. **Value and cost:** a caller can construct the statistic from precisely the data its definition requires. M.

### UPT-15: Two model descriptions understate what the packers implement

**Where.** `D/operators-density.md:269,500` (KI), `S/point-density/kernel-density-math.ts:34`, `kernel-density-bandwidth.ts:318`, `S/spatial-interpolation/kriging-parameters.ts:55`.

**EXECUTED.** The automatic KDE section gives the pooled Gaussian-scale formula alongside an example using `kernel: 'quartic'`. The kernel-specific multipliers are **`1`, `2.40187391`, `2.77919324`** for Gaussian, Epanechnikov and quartic. Separately, the raster kriging paragraph lists three variograms; the packer accepted all five and encoded spherical/exponential/Gaussian/Matérn-3/2/Matérn-5/2 as `0/1/2/3/4`. With nugget `.5`, partial sill `2` and range `3`, evaluating either Matérn at lag `3` gave **`2.40042586`**.

**Wanted.** Give the multiplier next to the Scott/Silverman formula; distinguish Gaussian standard deviation from compact support radius. List both Matérn models in the `GPUKriging` section and generate that list from the accepted model codes. **SOURCE GPU probe:** three triangle points with automatic bandwidth, switching only between Gaussian and quartic; compare returned bandwidth ratio to `2.77919324`. **Callers that change:** none. **Value and cost:** SciPy and geostatistics users can transfer the parameter they actually mean. S.

### UPT-16: The geographic centroid workaround needs a reversible rotation

**Where.** `D/operators-statistics.md:1734` (KI), `S/spatial-clustering/gpu-spatial-clustering.ts:142` (KI), `spatial-clustering-kernels.ts:135`.

**SOURCE.** The page correctly warns that arithmetic longitude means place a seam-straddling cluster far from its observations. It then suggests shifting longitudes into a continuous range. The obvious unwrap `[179.9,-179.9] → [179.9,180.1]` requires bounds outside `[-180,180]`; the geographic lattice rejects them. This is separate from the IDW search-box issue in KNOWN(PTS-117).

**Wanted.** Give the complete transformation: rotate longitudes around a chosen central meridian, wrap into the supported range, rotate the bounds consistently and invert the rotation on the centroid. For these two events a `180°` rotation makes the working longitudes approximately `−.1,+.1`. **Smallest GPU probe:** the two equatorial events, `epsilon: 30000` metres, `minimumPoints: 2`; compare original, naïvely unwrapped and rotated inputs, reading labels and centroid. **Callers that change:** none. **Value and cost:** a valid cluster can be drawn on the correct side of the globe. S.

## Departures by name

| Departure | Public contributors or functions |
|---|---|
| Whole-target clear | `getGPUKrigingParameterValues`, `getGPULocalKrigingParameterValues`, `getGPUSpaceTimeKrigingParameterValues`, `getGPUGaussianRandomFieldParameterValues`, `getGPUVecchiaLogLikelihoodParameterValues`, `getGPUVecchiaSpaceTimeLogLikelihoodParameterValues` |
| Raster corners narrowed after the useful width check, or without one | `GPUKernelDensity`, `GPUAdaptiveBandwidthKernelDensity`, `GPUSpaceTimeKernelDensity`, `GPUInverseDistanceWeighting`, `GPUKriging`; `GPUSpaceTimeRelativeRisk` composes the space-time KDE |
| Row identifiers presented as region labels | `GPUSkaterRegions`, `GPUWardRegions`, `GPURedcapSingleLinkage`; the affected composition is `GPURegionPartitionEvaluation` |
| Checked constructor and free packer can disagree about the requested operation | `GPUInverseDistanceWeighting`, temporal `GPUSpatialClustering`, `GPUSinkhornDivergence` |
| Unrecognized external selectors or malformed tuples pack successfully in the demonstrated cases | `getGPUSpatialScanStatisticParameterValues`, `getGPUSimilarLocationsParameterValues`, `getGPUSpatialClusteringParameterValues` |
| Wrong first-use parameter ownership or discarded settings | `GPUSpatialClustering`, `GPUSpaceTimeHawkes`, `GPUSinkhornDivergence` |
| Mathematical reference or description differs from the demonstrated behavior | `GPUSpaceTimeHawkes`, `GPUKernelDensity`, `GPUSpaceTimeKernelDensity`, `GPUKriging`, `getGPUKernelDensityWindowMass`, `getGPUSpaceTimeTemporalWindowMass` |
| Unused required inference inputs | `GPUPointPatternEnvelope`, `GPUPointPatternEnvelopeOperation`, `CPUPointPatternEnvelopeOperation`, `computePointPatternEnvelopeOnCPU` |
| Lower-left-only density raster | `GPULineDensity` |

I would keep `epsilon` for DBSCAN, `searchRadius` for an interpolation cutoff, and `bandwidth` for a smoothing parameter. They describe different controls. Likewise, both scan packers preserved a 53-bit-safe seed as low/high words in the executed probe. The observed interoperability failures concern what a parameter selects, the units it carries, and whether the selected neighborhood can be represented.

## Quick wins

- [ ] Fix the three parameter-write examples and run them with typed owners and graph views.
- [ ] Limit each of the six packer writes to its advertised layout; include success and failure canaries.
- [ ] Reject collapsed float32 grid extents, unrepresentable positive bandwidths/ranges and modified-Shepard radii that become infinite.
- [ ] Reject unknown scan/ranking selectors and DBSCAN bounds with a length other than four.
- [ ] Add instance-level neighborhood checks to IDW and ST-DBSCAN.
- [ ] Add the corner-event mass example to KDE and the two-label path example to regionalization.
- [ ] Correct the compact-kernel bandwidth multipliers and raster kriging model list.
- [ ] Show the north-up line-density row conversion and reversible antimeridian rotation.
- [ ] Diagnose an unreachable Sinkhorn target when packing against a contributor.
- [ ] Add independent narrow-interval and tail fixtures for the two Gaussian mass helpers.

## Larger proposals

**Fix the finite-window Hawkes estimator before alpha.** This is the breaking numerical change I recommend. Renaming the convergence flag would leave users with the same nonstationary estimates. A safeguarded finite-window update fixes the statistical result and gives the flag a defensible interpretation.

**Make region labels composable while preserving feature identity.** A reusable compaction result is preferable to changing every regionalizer’s existing label output. It provides dense IDs for grouped statistics and a mapping back to the source rows.

**Make raster placement and sampled populations explicit.** Add raster-metadata input to line density, an included-event population option for automatic KDE bandwidth, and mode-specific inputs for point-pattern envelopes. Each can ship independently without forcing migrations.

## Not read

I read the parameter paths and CPU calculations underlying the reported probes, including both Gaussian mass helpers, the Hawkes update and likelihood, the Sinkhorn iteration, DBSCAN packing, and the KI point-pattern envelope twin. I sampled WGSL at the relevant eligibility, accumulation, labeling, parameter-read and publication points. I checked candidates against the synthesis, spatial-points report, API/docs and direction reports, and the decision list.

For location-allocation, REDCAP, Dorling, catchment accessibility, neighborhood summary, change points, space-time tests and the remaining interpolation solvers, coverage was the export census plus selected props, docs or composition points. I did not audit all kernels, benchmarks, recipes, consumers or lens records. I ran no GPU, browser, build, typecheck or test suite. Shader predictions above are marked `SOURCE`; the CPU and Python results are executed observations.