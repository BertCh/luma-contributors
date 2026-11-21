# Scale variants and limits in `@luma-contributors/analysis`

## Summary

Two raster streaming contributors can lose a measurable variance on just two valid observations. The host arithmetic probe gave squared deviations of `1` and `0` for the two input orders. The correct answer is `0.5` in both orders. GPU confirmation is still needed.

Three streaming parameter packers also let the caller change the unit of already-written state between chunks. The public decoder demonstrably reads the same three words as either `1` or `2`.

Three independent planners reject work that fits an existing route. Network visibility chooses an unusable pivot count, terrain auto budgets a tiled route it would never execute, and class breaks enforces the row limit of a method the caller disabled.

The docs understate working capacity for fractional categorical statistics, rank Markov, tiled terrain, and two dataframe contributors. In the other direction, the island-attachment catalog advertises a grid maximum that exceeds its own index arithmetic.

The most valuable API change is a side-effect-free `getExecutionPlan(shape, deviceLimits)` backed by the same route rules as construction. It should report the selected algorithm, numerical guarantees, limiting port or product, required bytes, and total commands. The streaming state issues need separate numerical fixes.

## Census

`S` means `modules/analysis/src`, `D` means `docs/api`, and `T` means `modules/analysis/test`. Paths are relative to the review checkout unless marked `KI`, which identifies the newer copy under `/Users/robertchristie/Documents/GitHub/luma-contributors-ki`. I checked `.typecheck/r19/staging-owned.txt` before using a finding about an owned file. Counts are observations of the live checkout.

**COUNT.** I used Node with the installed TypeScript API, without emitting files or invoking `tsc`. I mapped each package export's `types` target from `dist/*.d.ts` to `src/*.ts`, called `getExportsOfModule`, resolved aliases, and deduplicated by public export name. This yielded **761 class names, 631 beginning with `GPU`, and 498 classes with `getCommandNodes`**. For the literal census I inspected union members on the first constructor parameter's top-level properties with `strict: true`, including mapped properties. Counts overlap.

| Public selection mechanism | Classes | Examples or qualification |
|---|---:|---|
| Literal `'auto'` | 28 | 29 properties. `GPUShapeDescriptors` has two. Additional classes select a route when a prop is omitted. |
| Literal `'exact'` | 16 | 19 properties, including `sampling`, `traceMode`, `strategy`, and `logDeterminant` |
| Literal `'approximate'` | 1 | `GPUClassBreaks.naturalBreaksMode`. Other approximate algorithms use names such as `'stochastic'` or a pivot count. |
| Literal `'reference'` | 8 | Across `mode`, `fitMode`, and `solver` |
| Literal `'tiled'` | 9 | Across `mode`, `execution`, `strategy`, `fitMode`, `momentMethod`, and `searchStructure` |
| Literal `'chunked'` | 1 | `GPUGeographicDistribution.reduction`. Its positions must still fit one packed view. |
| Named `GPUChunked…` classes | 4 | Spatial weights algebra, transform, transpose, validation |
| Named tiled / prepared / preparation / streaming classes | 1 / 1 / 2 / 1 | `GPUTerrainTiledFlow`, `GPUSpatialJoinPrepared`, two preparation classes, `GPURasterStreamingAccumulator` |
| Separate `GPUExact…` / `GPUApproximate…` classes | 0 / 0 | Exactness is generally selected inside an operation. |
| `prepared` / `studyWindow` props | 4 / 8 | Shared join indexes and study-window consumers respectively |
| `parameterLayoutVersion` | 1 | `GPURasterZonalStatistics` |
| Public classes exposing static `limits` | **0** | Across all 761 class names |

**COUNT.** There are **437** uppercase public names containing `MAXIMUM`. **64** start with `MAXIMUM_` and **373** place it elsewhere. There are also **77** public `get…Maximum…` functions. A name containing `MAXIMUM` can designate an automatic crossover, a conservative default, an address ceiling, or a hard limit. These counts therefore cannot be used as a count of supported input sizes.

The following are reproducible details of the census and probes:

```js
// Starting with the public source entry points described above.
const program = ts.createProgram(entries, {strict: true, noEmit: true,
  module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler})
const checker = program.getTypeChecker()
const producers = classes.filter(([, symbol]) =>
  checker.getDeclaredTypeOfSymbol(symbol).getProperty('getCommandNodes'))
// For each producer's first construct signature, enumerate props.getProperties().
// Read each property at property.valueDeclaration ?? constructorParameter.valueDeclaration.
```

- **COUNT.** An AST walk over reachable first-party source counted **1,131** `validateCapacity` calls with an explicit maximum, **893** using the default, and **350** `validateMaximumCount` calls. Of the 1,481 explicit maxima, 965 were identifiers, 109 numeric literals and 407 expressions. There were **177** `validateCommandNodeBudget` calls, 173 using the default budget. These are enforcement sites, including shared helpers, not distinct public contracts.
- **EXECUTED.** `getGPUAnalysisOperatorCatalog()` returned **557 entries**, of which **179** had numerical `limits`, with **359** entries in those arrays. Helpers and recipes are included in this denominator. **Nine of the 28** explicit-auto classes had no numerical catalog limits. Their names appear below.
- **COUNT.** `rg -n 'const [A-Z][A-Z_0-9]*(ROWS|ROW_COUNT)\s*=\s*(2 \*\* 24( - 1)?|0x[fF]{6}|16_777_21[56]);' modules/analysis/src` found **18 declarations**. Following their uses matters. Several are reference-only, one caller subtracts one from the constant, and the rational-quantile route crosses the border.
- **EXECUTED** below means a host packer, decoder, constructor, planner, or NullDevice graph declaration ran through `node_modules/.bin/esbuild --bundle --platform=node --format=esm --loader=ts` piped into `node --input-type=module -`. Node counts are declaration results. **SOURCE** identifies conclusions about WGSL execution.

### Hard limits checked against routes

Values are inclusive unless specified. A logical maximum still requires every binding, buffer, and command budget to fit. The table covers the route limits traced in this review. The 1,481 explicit enforcement sites above have not each been independently matched to prose.

| Operation and quantity | Source constant or calculation | Enforcing site under `S` | Documented number or discrepancy |
|---|---|---|---|
| Raster stream slices per dispatch / slots | `MAXIMUM_RASTER_STREAM_CHUNK_SLICE_COUNT = 256`, `MAXIMUM_RASTER_STREAM_SLOT_COUNT = 366` | `raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts:175–189`, calibration sibling `:134–136` | KI `D/gpu-raster/raster-time-series.md:56–58` gives 256. Its whole-stream claim omits the count bound. |
| Raster stream fixed-point headroom / observation count | `DEFAULT_MAXIMUM_RASTER_STREAM_SLICE_COUNT = 2^24`; term-count argument `0…2^32−1`; each slot count is one `uint32` | Streaming accumulator `:73,291–292`, `analysis-core/kernels/fixed-point-accumulator.ts:49` | KI time-series page `:27,56,81` gives both “no slice limit” and a default longest stream of `2^24`. See SCL-06. |
| Class breaks, natural breaks only | `NATURAL_BREAKS_MAXIMUM_{MATERIALIZED,EXACT,MONOTONE}_ROWS = 1,024 / 8,192 / 262,144` | `dataframe-analysis/column-classification/gpu-class-breaks.ts:304–344`; `natural-breaks-limits.ts:6–15` | KI `D/gpu-dataframe-analysis.md:417–449` gives the right numbers. Enforcement reaches unrelated methods. |
| Class breaks, whole input | `MAXIMUM_ROW_COUNT = 2^24`; 64 classes | `gpu-class-breaks.ts:271,343–344` | Separate from the three natural-breaks limits. The three named constants above are module exports but absent from public package barrels. |
| Fractional categorical, full-sort `'exact'` | `MAXIMUM_REFERENCE_ZONE_CATEGORY_GROUPS = 65,535`, `MAXIMUM_REFERENCE_TUPLES = 1,048,576` | `raster-analysis/gpu-raster-fractional-categorical-statistics.ts:202–218` | `D/gpu-raster/fractional-categorical-statistics.md:89–97` presents both as universal. |
| Fractional categorical, default tiled | `MAXIMUM_ZONE_CATEGORY_GROUPS` and `MAXIMUM_TUPLES` both `2^32−2`; `MAXIMUM_TUPLE_TILE_SIZE = 1,048,576`; all routes `MAXIMUM_CATEGORY_CAPACITY = 65,534` | Same file `:194–231,258–270` | Same page omits the different domain. The estimate `8 + 256 × tiles` and 32,768-command limit allow at most **127** tiles, before binding checks. |
| Rank Markov | `GPU_RANK_MARKOV_PAIRWISE_MAXIMUM_ROWS = 64`; sorted `GPU_TRANSITION_MATRIX_MAXIMUM_CLASS_COUNT = 65,535`; periods 65,536; panel values `2^31` | KI `gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts:90–123` | KI `D/gpu-spatial-analysis/coverage-statistics.md:22` says above 64 unsupported. `:816` correctly gives 65,535 for sorted ranks. |
| Terrain flow, resident / tiled D8 | `TERRAIN_TILED_MAXIMUM_CELL_COUNT = floor((2^32−1)/8) = 536,870,911`; `TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT = 2^32−1` | `gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts:156–170` | `D/gpu-terrain/terrain-processing.md:202–205` and catalog `operator-annotations.ts:2912` give 536,870,911 for the grid without the route qualification. |
| Terrain D8 sublimits | Axis `TERRAIN_TILED_MAXIMUM_GRID_EXTENT = 2^31−2`; forest and boundary domains each checked against 536,870,911; index words `≤ 2^32−1`; estimate `512 + 5 × tiles` | `gpu-terrain/terrain-processing/terrain-tiled-plan.ts:23–33,54–96` | The documented 32,768-command cap is real. With the present estimate, 6,451 tiles fit and 6,452 fail. |
| Region graph | `MAXIMUM_REGION_GRAPH_REGION_COUNT = 256`, Borůvka 65,535, iterations 256, cells `0x3fffffff` | `raster-analysis/remote-sensing/gpu-raster-region-graph.ts:46–52,127–148` | KI `D/gpu-raster/remote-sensing.md:49,157` contains both the reference and larger route. Dense public matrices still require `4R²` bytes each, as already recorded in `KNOWN(GPU-13)`. |
| Island attachment | Auto scan crossover 4,096; forced scan 65,535; grid `rows × 16 ≤ 2^32−1` plus schedule budget | KI `gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts:65,82,207–224` | Catalog `operator-annotations.ts:2348–2352` gives **4,294,967,294 grid rows**. Even the arithmetic alone allows at most **268,435,455**. |
| Spatial weights traces | `GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT = 4,096`; `GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT = 16,384` | `gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters.ts:24–27`; KI `gpu-spatial-weights-traces.ts:279–285` | First is a default crossover. Only the second is the forced-exact maximum. |
| Raster embedding nearest classifier | `MAXIMUM_RASTER_EMBEDDING_TRAINING_COUNT = 4,096` brute force; tiled count and `count × dimensions ≤ 2^32−1`; neighbors 16, classes 32 | `raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts:153–177` | KI `D/gpu-raster/raster-embeddings.md:138–146` correctly qualifies 4,096. The public constant's name omits the route. |
| Point-cloud grid | `GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS = 2^24`; `GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS = 2^32−2` | `pointcloud-analysis/operations/gpu-pointcloud-surfaces.ts:74–81`, `pointcloud-grid-index.ts:59–65` | KI `D/pointcloud-analysis.md:156,246` says “below 2^32,” which includes one rejected value. |
| Prepared spatial join, default internally built index | `leafCapacity = nextPowerOfTwo(featureCount)`; bounds/children binding `8 × (2 × leafCapacity − 1)` bytes | `gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts:269,545–552` | KI `D/gpu-spatial-analysis/concepts.md:162–167` describes partition metadata but gives no capacity calculation for the prepared index. |
| Raster harmonic regression | `MAXIMUM_HARMONIC_REGRESSION_SLICE_COUNT = 256` reference; `MAXIMUM_STORAGE_HARMONIC_REGRESSION_SLICE_COUNT = 4,096` | `raster-analysis/raster-time-series/gpu-raster-harmonic-regression.ts:141–151` | `D/gpu-raster/raster-change-monitoring.md:19–27` correctly gives both and identifies the default. Good model for other pages. |
| Gaussian random field | `GPU_GAUSSIAN_RANDOM_FIELD_MAXIMUM_*`: FFT side 1,024, rows `2^24`, modes 4,096, spectral work `2^30`, conditioning neighbors 32 | `gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field.ts:178,198–235,292–299` | KI `D/gpu-spatial-analysis/gaussian-random-field.md:32,70–93` distinguishes grid, work, and conditioning bounds. |
| Network flow | `NETWORK_FLOW_REFERENCE_MAXIMUM_*`: 128 nodes, 1,024 edges, `2^24−1` total capacity; parallel 30,000,000 nodes or edges, `2^32−1` total capacity | `gpu-network/network-optimization/network-flow-limits.ts:5–15`, `gpu-network-flow.ts:278–297` | `D/gpu-network/network-movement-coverage.md:12` gives the route-specific numbers. |
| Network structure | `GPU_NETWORK_STRUCTURE_EXACT_MAXIMUM_*`: 1,024 nodes / 65,536 edges; general `2^31−2` each | `gpu-network/network-structure/gpu-network-structure.ts:135,150,159` | The general/reference docs mismatch is `KNOWN(DOC-104)`. |
| Panel lag ML | Reference 32 entities / 32 periods; tiled 1,024 entities / `2^24` rows, using `GPU_PANEL_LAG_*` constants | KI `gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood.ts:343–383` | Catalog lists all four with route qualifications. |
| Multivariate local Geary permutation test | `EXACT_MULTIVARIATE_GEARY_MAXIMUM_ROWS = 256`, literal 4,096 permutations; tiled `2^24−1` rows and 9,999 permutations | KI `gpu-spatial-analysis/spatial-autocorrelation/gpu-multivariate-local-geary-permutation-test.ts:215–235` | Raw simulation output adds a separate route restriction. A table that shows only the largest row number would lose that distinction. |
| Zonal statistics parameter buffer | Default version 2 requires exactly 7 words; explicit 1 accepts ≥4, explicit 2 accepts ≥7 | `raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics.ts:456–488` | KI `D/gpu-raster/operations-analysis.md:355` still describes the former default. `KNOWN(RT-101, RT-138)`. The constructor already names both remedies. |

### Where larger inputs actually stop

The earlier cross-package inventory found **408 packed, 78 chunk-capable, 12 tiled, and no host-paged producers** among 498. Its named family inventory is in `dev-docs/research/library-review-2026-11-19/area-luma-upstream-scale.md:148–160`. I checked the following distinctions that matter when choosing a route.

| Largest logical input in the traced class | Producers and enforcement | Practical consequence |
|---|---|---|
| **16,777,215 rows** | `GPUGroupGeometry` and `GPUGroupConvexHull`, `S/gpu-spatial-analysis/group-geometry/gpu-group-geometry.ts:197` and `gpu-group-convex-hull.ts:270`; `GPUMinimumBounds`, `minimum-bounds/gpu-minimum-bounds.ts:190`; KI `GPUGeographicDistribution`, `geographic-distribution/gpu-geographic-distribution.ts:320–326`; KI `GPUQuantileLISA`, `spatial-autocorrelation/gpu-quantile-lisa.ts:124`; KI `GPURateStandardization`, sibling `gpu-rate-standardization.ts:286`; KI `GPUSpatialPanelRegression`, `spatial-regression/gpu-spatial-panel-regression.ts:110` | No prop on these classes raises the checked row bound. In particular, geographic distribution's `'chunked'` reduction still accepts one packed positions view. Chunked weights do not lift Quantile LISA's values limit. |
| **16,777,216 rows** | `GPUClassBreaks`, `S/dataframe-analysis/column-classification/gpu-class-breaks.ts:343`; `GPUGaussianRandomField`, `S/gpu-spatial-analysis/spatial-interpolation/gpu-gaussian-random-field.ts:178`; KI `GPUSpatialPanelLagMaximumLikelihood`, `gpu-spatial-panel-lag-maximum-likelihood.ts:378–383` | Splitting the input or selecting the larger algorithm still leaves a whole-domain limit. |
| **Depends on requested outputs** | `GPUColumnQuantiles`, `S/dataframe-analysis/column-classification/gpu-column-quantiles.ts:198–204` | Rational probabilities without filter outputs allow up to `2^31−1`. Requesting the filter mask or bounds restores the `2^24` limit. `KNOWN(DOC-104)` covers the earlier documentation issue. |
| **One binding survives tiling or preparation** | `GPUSpatialJoinPrepared` global BVH; `GPURasterRegionGraph` dense adjacency and weights; `GPURankMarkov` dense transition counts; `addRasterRegionGraphTiles` assembled labels and values | On a 128 MiB binding, a `4R²` output fits **5,792** states or regions at zero offset. Tiling the source cannot make that output fit at 5,793. Prepared joins have a different power-of-two boundary, measured in SCL-14. |

## Proposed convention

Use `execution` for equivalent ways to execute the same operation, a separately named `method` or `accuracy` choice when the mathematical result or guarantees change, and a separate preparation class when callers can own and reuse its state. Expose `Contributor.limits` and a side-effect-free `getExecutionPlan(shape, deviceLimits)` that identifies the selected route, input and output layout, inclusive logical bounds, per-binding bytes, command count, and any valid alternative.

Name hard constants `<OPERATION>_<ROUTE>_MAXIMUM_<QUANTITY>`, auto crossovers `<OPERATION>_AUTO_<ROUTE>_THRESHOLD_<QUANTITY>`, and defaults `DEFAULT_<QUANTITY>`. Generate the docs and catalog from the same descriptors, and test each route with inputs immediately below, at, and above its limit. A routing error should name the requested value, the constraint that failed, and a valid next choice, including any loss of accuracy.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| SCL-01 | W | ORDINARY | BREAKING numerical correction | M | Raster streaming and index calibration | Two exactly representable inputs lose their variance in an order-dependent way. |
| SCL-02 | W | ORDINARY | ADDITIVE | M | Three raster streaming packers | Changing a per-frame setting reinterprets the preceding chunks. |
| SCL-03 | W | UNUSUAL | ADDITIVE | S | Raster stream packers | Two packers accept nonfinite or unrepresentable bounds, and three accept impossible active counts. |
| SCL-04 | S | ORDINARY | ADDITIVE | S | Network visibility recipe | Auto fails command or divisibility checks where an explicit sampled plan succeeds. |
| SCL-05 | S | ORDINARY | ADDITIVE | S | `GPUClassBreaks` | Disabling natural breaks leaves its row limits active. |
| SCL-06 | D/W | EXTREME | ADDITIVE | M | Raster streaming counts | “No slice limit” hides a `uint32` count and an unenforced headroom declaration. |
| SCL-07 | S | UNUSUAL | ADDITIVE | S | `GPUTerrainTiledFlow` | Auto rejects resident work on the budget for an unused tiled schedule. |
| SCL-08 | S | ORDINARY | ADDITIVE | M | `addRasterRegionGraphTiles` | A failed route leaves nodes and IDs behind, preventing a corrected retry. |
| SCL-09 | D | ORDINARY | ADDITIVE | S | Fractional categorical statistics | The page publishes full-sort limits for the default tiled algorithm. |
| SCL-10 | D | ORDINARY | ADDITIVE | S | Two dataframe contributors | The page tells users to reject 30-million-row inputs that both classes declare successfully. |
| SCL-11 | C | UNUSUAL | ADDITIVE | S | `GPURasterRegionGraph` | The public route can say Borůvka for an entirely dense graph. |
| SCL-12 | C/D | ORDINARY | ADDITIVE | S | Island attachment and spatial weights traces | Two public “maximum” constants are auto crossovers. An island grid catalog ceiling is impossible. |
| SCL-13 | D | ORDINARY | ADDITIVE | S | `GPURankMarkov` | The same page says both 64 and 65,535 rank states. |
| SCL-14 | E | ORDINARY | ADDITIVE | S | `GPUSpatialJoinPrepared` | Partition memory estimates cannot size the global prepared index. |
| SCL-15 | D | UNUSUAL | ADDITIVE | S | Terrain and point-cloud grids | Published grid bounds disagree with constructor checks in opposite directions. |
| SCL-16 | E | ORDINARY | ADDITIVE | M | Catalog and public classes | Route constraints cannot be queried in a form that reproduces construction. |

### SCL-01: Two observations give order-dependent squared deviations

**Where.** `S/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts:291–301`, `gpu-raster-standardized-index-calibration.ts:250–259`, and `S/analysis-core/kernels/fixed-point-accumulator.ts:124–127,152–182`.

**Evidence.** **EXECUTED** the production parameter packer and a `Math.fround` model of the shader's mean and deviation expressions. For one slot, `[10000000, 10000001]` produced a modeled moment increment of `1`. Reversing the values produced `-0`. The expected sum of squared deviations is `0.5` both ways. The packed moment-unit exponent was `−21`, so the output format can represent `0.5` exactly. **SOURCE:** both shaders round the updated sum to `f32` before division. For `[f32(1e20), -f32(1e20)]`, the modeled product was `Infinity`; the quantizer maps a nonfinite term to zero.

**Now.** A caller requesting `state.moments` can get a different answer after reordering two valid observations. The 96-bit storage does not recover a deviation lost before quantization.

**Wanted.** Compute the deviation from a centered or wider residual with scaling before multiplication. Pin the two adjacent values in each order, both inside one chunk and across two one-slice submissions. The smallest GPU probe uses one cell and one slot, reads the three moment words, and decodes them against `0.5`. For standardized calibration, read the third three-word moment at offset 6.

**In-repository callers that change.** Zero call expressions require edits. `rg 'new (GPURasterStreamingAccumulator|GPURasterStandardizedIndexCalibration)\b' modules examples website -g '*.ts' -g '*.tsx'` found **26** construction sites, including Kepler raster time series, tests and website runners. Numerical expectations change where they capture the old result.

**Value and cost.** Raster and climate users get a reproducible dispersion for low-variance series with large baselines. **M.**

### SCL-02: The state encoding can change halfway through a stream

**Where.** `S/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator.ts:42–78,271,291–301`, `gpu-raster-standardized-index-calibration.ts:47–66,232–259`, `gpu-raster-climate-run-accumulator.ts:44–66,230,253–254`. KI `D/gpu-raster/raster-time-series.md:56–81`.

**Evidence.** **EXECUTED** `getGPURasterStreamingAccumulatorParameterValues({reset:false, maximumValueMagnitude:m}, 1)`. For `m=1`, the result was `[0,1,-69,-67,1]`. For `m=2`, it was `[0,1,-68,-65,2]`. The production decoder read the same persisted words `[0,0,32]` as **1** and **2** respectively. **SOURCE:** the next dispatch loads the prior words and adds using the new unit. Calibration does the same for moments, and climate runs do it for degree days.

**Now.** A caller who packs `maximumValueMagnitude: maxForThisChunk` on each frame can silently reinterpret all prior data. Changing `maximumStreamSliceCount` can also change the unit. Both settings are supplied alongside `reset` and `activeSliceCount` in the per-frame packer.

**Wanted.** Add an immutable stream encoding plan containing the magnitude bound, headroom and exponents. Let the frame packer take that plan plus `{reset, activeSliceCount}`. Document that changing the encoding requires resetting or converting the state. A GPU regression needs just two one-slice chunks, value `1` in each, with a magnitude bound changing from `1` to `2`. The accepted outcome should be a sum of `2` or an explicit incompatible-state error.

**In-repository callers that change.** None for an additive plan overload and documentation. **Value and cost.** Streaming callers can safely resume an accumulation and decode its state. **M.**

### SCL-03: Stream packers accept bounds that cannot reach the shader intact

**Where.** Streaming accumulator `:62–78`, climate run accumulator `:44–66`, and standardized index calibration `:47–66`, under `S/raster-analysis/raster-time-series`.

**Evidence.** **EXECUTED.** The climate packer accepted `maximumIncrement: NaN, threshold: 0`, returning `[0,1,0,59]`. Calibration accepted `maximumValueMagnitude: 1e39`, returning `[0,1,0,59,-63,59,Infinity]`. It packed `1e-46` as a zero magnitude and accepted `-1`. The streaming accumulator rejected both `1e39` and `1e-46`. With `chunkSliceCount=1`, its active-count packer returned `-1`, `0.5`, and `2` unchanged. All three packers write the active count without enforcing its integer range.

**Now.** `pack({activeSliceCount: 2, …}, 1)` succeeds. **SOURCE:** the shader clamps it to one, and a fractional count is converted to an integer. A caller counting submitted observations gets no indication that its frame description was wrong.

**Wanted.** Share validation of positive finite magnitudes **after conversion to float32**. Validate `activeSliceCount` as an integer in `[0, chunkSliceCount]`. Name the contributor, field, supplied value, and bound before returning the parameter words.

**In-repository callers that change.** None for valid settings. **Value and cost.** Callers catch unit and batch-size mistakes at the packer that introduced them. **S.**

### SCL-04: Automatic visibility sampling can choose an unusable pivot count

**Where.** The **network** `addVisibilityGraphRecipe`, `S/gpu-network/visibility-graph/visibility-graph.ts:158–186,199–205,277–283`; `D/gpu-network/visibility-graph.md:5–8`. This is separate from the qualified-catalog issue in `KNOWN(ERG-103)`.

**Evidence.** **EXECUTED** with CSR, one segment, `edgeCapacity=4*n`, and `maximumDistance=1.1`.

| Positions | Lanes | Maximum iterations | Automatic pivots | Same input with explicit pivots |
|---:|---:|---:|---|---|
| 256 | 1 | 256 | Fails the 32,768-command limit | `pivotCount: 2` succeeds, estimate **3,746** |
| 10,000 | 3 | 8 | Fails “laneCount must divide pivotCount” | `pivotCount: 30` succeeds, estimate **2,098** |
| 257 | 32 | 16 | Fails divisibility | Control at 256 positions succeeds with 256 pivots |

**Now.** The documented `metricsMode: 'auto'` tries all sources and then a fixed sample size. A caller can receive a divisibility error for a `pivotCount` they never supplied.

**Wanted.** On an otherwise failing auto request, choose a sampled count that fits the actual lane, byte, and command constraints. If no legal count fits, report the smallest feasible plan and its failed constraint. Preserve explicit-exact semantics and report the selected pivots in provenance.

**In-repository callers that change.** None if successful auto selections stay stable. **Value and cost.** A GIS developer can use the documented auto mode without reverse-engineering the batch scheduler. **S.**

### SCL-05: Disabled natural breaks still limits equal-interval classification

**Where.** `S/dataframe-analysis/column-classification/gpu-class-breaks.ts:304–344,444–451`; `natural-breaks-limits.ts:6–15`. KI `D/gpu-dataframe-analysis.md:417–449,538`.

**Evidence.** **EXECUTED** with `methods: ['equal-interval']` only. At 1,025 rows, `{naturalBreaksMode:'exact', naturalBreaksExactStorage:'materialized'}` failed the 1,024-row limit. Switching storage to `'auto'` declared **4 nodes**. At 8,193 rows, explicit `'exact'` failed and omitting it declared **4 nodes**. At 262,145 rows, `'monotone'` failed and omitting it again declared **4 nodes**.

**Now.** `new GPUClassBreaks({...savedSettings, methods:['equal-interval']})` can fail on the row limit for a dynamic program absent from the resulting graph.

**Wanted.** Apply natural-breaks **capacity** checks only when the enabled methods include `'natural-breaks'`. Keep validation of contradictory explicit settings. Expose the three existing route maxima through the public limits description.

**In-repository callers that change.** None. **Value and cost.** Classification UIs can reuse settings when users change methods. **S.**

### SCL-06: Frame count, stream headroom, and per-slot count are different limits

**Where.** Streaming accumulator `:39–54,73,135,291–292`; `S/analysis-core/kernels/fixed-point-accumulator.ts:49`; `S/gpu-spatial-analysis/ontology/all-entry-points/operator-annotations.ts:654–664`; KI `D/gpu-raster/raster-time-series.md:27,56,81`.

**Evidence.** **EXECUTED.** The packer accepted `maximumStreamSliceCount` of both `0` and `4,294,967,295`. At `4,294,967,296` it threw `fixed-point maximumTermCount must be an integer in [0, 2^32 - 1]`. The default `2^24` determines headroom. **SOURCE:** each valid sample executes `priorCount + 1u` with no overflow check.

**Now.** The docs say a stack has “no slice limit.” The caller can keep submitting frames, but the count cannot represent a 4,294,967,296th observation in the same cell and slot. The supplied headroom can be exhausted earlier.

**Wanted.** State the three quantities independently: at most 256 slices per submission, a stream-wide declared headroom defaulting to `2^24`, and at most `2^32−1` representable valid observations per cell and slot. Reject zero headroom for a nonempty intended stream. Add an overflow status or checked extended count. A one-dispatch GPU probe can seed the count to `0xffffffff`, use a zero sum and one valid zero, and read count and status.

**In-repository callers that change.** None for documentation and an optional status. **Value and cost.** Long-running services gain a measurable rollover condition. Documentation **S**, runtime protection **M**.

### SCL-07: Terrain auto validates the command cost of an unused route

**Where.** `S/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts:58–70,189–202,290–324`; `terrain-tiled-plan.ts:26–27`.

**Evidence.** **EXECUTED** on a 128 MiB binding device with one one-cell D8 tile. `(estimated, declared nodes)` was `auto (517,11)`, `resident (17,11)`, and `tiled (517,14)`. With 6,452 disjoint one-cell tiles, auto failed the 32,768-command budget. The tiled estimate is **32,772**. The whole resident fraction plane is just **206,464 bytes** and meets the actual resident-selection predicate. I did not declare the full 6,452-tile resident graph.

**Now.** `execution:'auto'` can throw before it reaches the same device-dependent predicate the constructor already uses for stream reads.

**Wanted.** Resolve the route once against the views' graph device. Apply the command budget to that route and publish its estimate. If callers need the worst case across devices, expose it as a separately named estimate.

**In-repository callers that change.** None. **Value and cost.** Many-small-tile inputs gain the resident path already available to them. **S.**

### SCL-08: Correcting a tiled region-graph route cannot be retried on the same graph

**Where.** `S/raster-analysis/remote-sensing/raster-region-tiles.ts:84–95,114–116`; `gpu-raster-region-graph.ts:133–148`.

**Evidence.** **EXECUTED.** One valid tile and 257 regions with `searchStructure:'dense'` failed the `[1,256]` check **after one assembly node had been added**. Retrying with the same graph and ID, changing only to `'boruvka'`, failed with `GPUCommandGraph resource id "tile-test-labels" is already in use`. In a successful four-tile, two-region case, the graph contained **15** nodes and the inner contributor's estimate was **11**. The four omitted nodes were the four tile assemblies.

**Now.** The normal response to the first exception, retrying with the larger route, requires discarding the graph. A caller using the returned contributor estimate as the recipe cost misses one command per tile.

**Wanted.** Preflight the full recipe, including tile views, chosen route, assembled allocations and total budget, before reserving IDs or adding nodes. Return a recipe-level estimate equal to assembly plus inner work. Pin both the unchanged graph after failure and the successful retry.

**In-repository callers that change.** None for a preflight and additive recipe estimate. **Value and cost.** A failed scale choice cannot invalidate other work already present in the caller's graph. **M.**

### SCL-09: The fractional categorical page makes users reject the default route

**Where.** `D/gpu-raster/fractional-categorical-statistics.md:89–97`; `S/raster-analysis/gpu-raster-fractional-categorical-statistics.ts:71–95,194–231,258–270`.

**Evidence.** **EXECUTED.** With 30,000 zones, category capacity 2, and 100 tuples, the default accepted **90,000 groups**, estimate **264**. Explicit `'exact'` rejected the input at 65,535 groups. With two zones and **1,048,577 tuples**, default accepted, estimate **520**; `'exact'` rejected at 1,048,576.

**Now.** A newcomer implementing the documented preflight writes:

```ts
if (zoneCount * (categoryCapacity + 1) > 65535 ||
    tupleCapacity > 1048576) throw new RangeError('Raster aggregation is too large')
```

Both tested default workloads are then lost before construction.

**Wanted.** Put the reference and default limits in separate rows using the figures in the census. Show that `tupleTileSize` limits one sort allocation and that caller-owned output bindings and the total command budget still apply.

**In-repository callers that change.** None. **Value and cost.** Users can process larger rasters with the existing default. **S.**

### SCL-10: Two dataframe pages confuse the size of a chunk with the size of a column

**Where.** `D/gpu-dataframe-analysis.md:1778,1854`; KI `:1793,1869`. `S/dataframe-analysis/time-words-relative/gpu-time-words-relative.ts:93–95,141–160`; `S/dataframe-analysis/value-cube/gpu-cell-period-keys.ts:159,200–217`, also checked in KI. Extends `KNOWN(DOC-104)` to two different operations.

**Evidence.** **EXECUTED independently in both checkouts.** Three aligned chunks of 10,000,000, 9,000,000 and 11,000,000 rows on a 128 MiB binding device produced `GPUTimeWordsRelative 3` and `GPUCellPeriodKeys 4` command nodes, with `invalidCount` requested for keys. An unchunked 30-million-row `uint32x2` input would require 240,000,000 bytes.

**Now.** From the page alone, a caller would write `if (cells.length > 16_777_216) reject()`. The documented per-binding number is correct for a single zero-offset chunk, but it is attached to the whole column.

**Wanted.** Say that corresponding row fields must have aligned chunk lengths, each individual binding must fit, and the small status port stays packed. Show `[10_000_000, 9_000_000, 11_000_000]` as a working declaration. Keep the downstream distinct-group bound separate from the input row count.

**In-repository callers that change.** None. **Value and cost.** Time-series and value-cube callers can use their existing Arrow batches. **S.**

### SCL-11: Region-graph provenance can report a route that did not run

**Where.** `S/raster-analysis/remote-sensing/gpu-raster-region-graph.ts:101–123,236–246,420`; `T/gpu-raster/remote-sensing/region-graph-boruvka.node.spec.ts:199`.

**Evidence.** **EXECUTED** with a synthetic 192-byte binding limit, 25 cells, four regions and three rounds. Auto exposed `searchStructure === 'boruvka'`. Graph declaration returned **13 nodes, zero Borůvka nodes**, ending in `raster-region-graph-finish` on the dense path. The existing test covers the fallback. It does not establish that the public field reports it.

**Now.** `record({algorithm: contributor.searchStructure})` can produce inaccurate provenance.

**Wanted.** Return `requestedSearchStructure` and device-resolved `selectedSearchStructure` from the execution plan. Use the selected value when recording work. Avoid mutable “last graph” state, since one contributor can be queried for different device limits.

**In-repository callers that change.** None for the additive selected field. **Value and cost.** Benchmarks and computation records can identify the algorithm actually declared. **S.**

### SCL-12: An automatic crossover is exported as a maximum, while a catalog maximum is unreachable

**Where.** KI `S/gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts:65,82,207–224`, checkout `:59,76,162–179`; `S/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters.ts:24–27`, KI `gpu-spatial-weights-traces.ts:279–285`; `S/gpu-spatial-analysis/ontology/all-entry-points/operator-annotations.ts:2348–2352`.

**Evidence.** **EXECUTED in both trees.** Forced island scan accepted **4,097** and **65,535** rows, then rejected **65,536**. Its public scan “maximum” is **4,096**. For traces, 4,097 rows defaulted to stochastic, explicit exact succeeded, and 16,385 exact rows gave the good routing error `exact mode supports at most 16384 rows; use mode 'stochastic'`. For island grid, **268,435,456** rows failed `candidate slots (rows * 16) must be at most 4294967295`. At **268,435,455**, the command budget already failed. The catalog gives **4,294,967,294** grid rows.

**Now.** A caller disabling the “exact” button above `GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT` withholds a working choice. A caller enabling grid from the catalog maximum admits impossible choices.

**Wanted.** Export distinct auto-threshold and forced-route maximum names. Add `GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROWS = 65535`. Retain old constants as deprecated aliases with their existing values. Represent island grid as a product limit **and** a schedule limit. Give the scan error a grid hint only when the grid plan fits.

**In-repository callers that change.** None with aliases. **Value and cost.** UIs can enable exactly the routes the constructor supports. **S.**

### SCL-13: Rank Markov's overview rules out its working sorted route

**Where.** KI `D/gpu-spatial-analysis/coverage-statistics.md:22,816`; checkout second occurrence `:450`; KI `S/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts:103–123` and `cpu-rank-markov-operation.ts:76`.

**Evidence.** **EXECUTED in both trees.** A complete panel of **65 rows × 2 periods**, 130 rank outputs and 4,225 transition counts selected `'sorted'` automatically. Explicit `'pairwise'` failed `rowCount must be an integer in [1, 64]`. The overview lists “More than 64 rank states” under unsupported behavior.

**Now.** A reader deciding whether the operation fits their panel can stop at the overview and never reach the correct explanation later on the same page.

**Wanted.** Replace that overview cell with “pairwise ranking and its bounded diagnostics support at most 64 states; sorted ranking supports up to 65,535 logical states subject to dense output bindings.” Add `use rankingMethod: 'sorted'` to the forced-pairwise size error where its requested outputs are supported. Correct the CPU-side description of the GPU bound.

**In-repository callers that change.** None. **Value and cost.** Panel users discover the existing larger route at the decision point. **S.**

### SCL-14: Prepared join partition metadata does not size the global BVH

**Where.** KI `D/gpu-spatial-analysis/concepts.md:162–167`; `S/analysis-core/contracts/partitioning.ts:231–251`; `S/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts:94,258–269,545–552`.

**Evidence.** **EXECUTED.** I supplied 8,388,609 points in partitions of 4,194,304 and 4,194,305 rows, with zero halo. `getGPUPartitionMemoryPlan(..., 8)` returned `packedBytes: 67108872` and `peakPartitionBytes: 33554440`. The prepared index chose `leafCapacity: 16777216`, then graph declaration failed: `BVH node bounds/children needs 268435448 bytes in one binding (limit 134217728)`. The partition planner correctly described one-partition-at-a-time input consumption. The prepared contributor built a whole-domain index.

**Now.** `new GPUSpatialJoinPrepared({geometry, partitioning})` accepts the metadata, so checking `peakPartitionBytes` before calling it appears reasonable. Just one feature beyond the power-of-two boundary nearly doubles index capacity.

**Wanted.** Expose `getGPUSpatialJoinPreparedMemoryPlan` or the equivalent operation plan. Report the rounded leaf count and largest index binding before graph mutation. Put the calculation and an explicit statement about whole-domain index residency beside the `partitioning` prop. At 128 MiB, **8,388,608** features pass this particular bounds/children check with default leaf sizing; other constraints still apply.

**In-repository callers that change.** None. **Value and cost.** Join callers can size the expensive allocation before preparing an index. **S.**

### SCL-15: Terrain and point-cloud grid limits disagree with their pages in opposite directions

**Where.** `D/gpu-terrain/terrain-processing.md:202–205`; `S/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts:156–170`; catalog `operator-annotations.ts:2912`. KI `D/pointcloud-analysis.md:156,246`; `S/pointcloud-analysis/operations/pointcloud-limits.ts:24–27`, `gpu-pointcloud-surfaces.ts:74–81`, `pointcloud-grid-index.ts:59–65`.

**Evidence.** **EXECUTED.** A `32768 × 16384` D8 grid, **536,870,912 cells**, in 512 disjoint `1024 × 1024` tiles passed tiled construction with an estimate of **3,072**. Explicit resident rejected it at **536,870,911**. Separately, `validatePointCloudGrid` rejected `65535 × 65537 = 4,294,967,295` with `grid cell count must be at most 4294967294`. Control `65535 × 65536` passed. These were host checks, without large allocations.

**Now.** The terrain page excludes a valid tiled domain. The point-cloud page includes one impossible product even before device binding constraints.

**Wanted.** Publish the two terrain maxima by route, along with per-tile and boundary limits. For point-cloud grids, say `width × height ≤ 4,294,967,294` wherever the shared grid validator is used. Generate both descriptions from the checked quantities.

**In-repository callers that change.** None. **Value and cost.** Grid planners can use the published inequalities verbatim. **S.**

### SCL-16: Catalog limits cannot reproduce route selection

**Where.** `S/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:317–325,360` and `ontology/all-entry-points/operator-annotations.ts`. The affected explicit-auto classes are named below.

**Evidence.** **COUNT, EXECUTED.** Zero of 761 public classes expose static `limits`. Nine explicit-auto classes have no numeric catalog entries. The available `GPUAnalysisLimit` fields are `name`, inclusive `maximum`, `reason`, and optional `source`. Even a populated entry cannot encode `rows × 16`, power-of-two padding, a runtime binding limit, or a condition such as “only when the output mask is requested” except as prose. Among six exercised rejection paths—forced island scan, forced rank pairwise, exact natural breaks, resident terrain, fractional full-sort, and forced exact traces—only the traces error explicitly named an alternate mode.

**Now.** A caller making a method picker needs independent knowledge of property spelling, route conditions, output-dependent constraints, and whether an alternate changes the answer. The net result is visible in the incorrect rejection code in SCL-09 and SCL-10.

**Wanted.** Add a route descriptor with `quantity`, `unit`, `appliesWhen`, hard bound or byte expression, `scope` such as chunk/domain/binding, and alternate route with its accuracy implications. Use an executable planner for actual device and shape checks. Start with the nine missing auto entries and the reproduced disagreements. This adds a concrete validation target to the earlier catalog concerns.

**In-repository callers that change.** None for additive metadata and planning APIs. **Value and cost.** GIS frontends, recipes and docs can all answer “will this request fit, and what will run?” from the same rules. Initial schema and first families **M**; library-wide coverage **L**.

## Departures by name

These lists identify the producers counted above. A different property name can be justified when it selects a different mathematical quantity. The departure that needs fixing across these groups is the absence of a common way to discover the choice and its constraints.

| Selection axis | Public classes, with the common `GPU` prefix omitted |
|---|---|
| Auto scratch, reduction or accumulation | `CellAggregation`, `CellPyramid`, `GroupStatistics` (`scratchMode`); `GeographicDistribution` (`reduction`); `PointDensity` (`sumAccumulation`) |
| Auto candidate search, indexes or geometry | `CellCover` (`candidateEnumeration`); `NeighborSearch`, `RasterRegionGraph` (`searchStructure`); `ConstrainedDelaunayTriangulation` (`flipMethod`); `SpatialScanStatistic`, `ExpectationBasedSpaceTimeScan` (`nearestZoneSearch`); `ShapeDescriptors` (`convexityMethod`, `momentMethod`); `OPTICS` (`neighborIndex`); `LocalKriging`, `SpaceTimeKriging` (`sampleIndex`); `SpatialPredicateJoin` (`engine`); `IslandAttachment` (`nearestSearch`); `DorlingCartogram` (`neighborGrid`) |
| Other explicit-auto fields | `RankMarkov` (`rankingMethod`); `GlobalSpatialStatistics` (`symmetryStrategy`); `KernelDensity`, `GaussianRandomField`, `SpatialInteraction` (`method`); `SpatialPanelLagMaximumLikelihood` (`fitMode`); `NetworkFlow` (`solver`); `NetworkStructure` (`mode`); `TerrainTiledFlow` (`execution`); `ClassBreaks` (`naturalBreaksExactStorage`) |
| Explicit `'exact'` | `KernelDensity`, `MultivariateLocalGearyPermutationTest`, `GeographicallyWeightedRegression`, `GeographicallyWeightedRegressionNonstationarityTest`, `SpatialErrorGM`, `SpatialWeightsTraces`, `SpatialLagMaximumLikelihood`, `SpatialErrorMaximumLikelihood`, `SpatialPanelErrorMaximumLikelihood`, `NetworkCentrality`, `NetworkCentralityScheduleStep`, `NetworkStructure`, `ClassBreaks`, `RipleyDistanceFunctions`, `DistanceField`, `RasterFractionalCategoricalStatistics` |
| Explicit `'reference'` | `GlobalSpatialStatistics`, `MultivariateLocalGeary`, `SpatialPanelLagMaximumLikelihood`, `NestedInterpolationValidation`, `NetworkFlow`, `NetworkPointRouting`, `Inequality`, `RasterHarmonicRegression` |
| Explicit `'tiled'` | `GlobalSpatialStatistics`, `ShapeDescriptors`, `MultivariateLocalGeary`, `MultivariateLocalGearyPermutationTest`, `SpatialPanelLagMaximumLikelihood`, `NestedInterpolationValidation`, `TerrainTiledFlow`, `RasterFractionalCategoricalStatistics`, `RasterEmbeddingKNearestClassifier` |
| Preparation and reuse | `SpatialJoinPrepared`, `StudyWindowPreparation`, `LonLatPolygonPreparation`; `prepared` is accepted by `PointInPolygonJoin`, `NearestFeatureJoin`, `SpatialPredicateJoin`, `SpatialJoinCandidates`. `studyWindow` appears on `KernelDensity`, `InhomogeneousPointPattern`, `Variogram`, `SpatialCorrelogram`, `Ripley`, `RipleyEnvelope`, `PointPatternIndices`, `RipleyDistanceFunctions`. |
| Named chunked variants | `ChunkedSpatialWeightsAlgebra`, `ChunkedSpatialWeightsTransform`, `ChunkedSpatialWeightsTranspose`, `ChunkedSpatialWeightsValidation`. Validation forwards to the ordinary class. Algebra has a smaller operation set, so the same consolidation argument does not automatically apply. |
| Streaming state | `RasterStreamingAccumulator`, `RasterStandardizedIndexCalibration`, `RasterClimateRunAccumulator`. Only the first name signals streaming. All three need an explicit state-encoding contract. |
| Explicit-auto classes with no numeric catalog limits | `SpatialInteraction`, `GeographicDistribution`, `SpatialPredicateJoin`, `NeighborSearch`, `CellPyramid`, `PointDensity`, `KernelDensity`, `ClassBreaks`, `ExpectationBasedSpaceTimeScan` |

`addVisibilityGraphRecipe` and `addRasterRegionGraphTiles` are functions and sit outside the 498-class population. Both need recipe-level plans because the cost includes work beyond one underlying contributor.

## Quick wins

- [ ] Apply the natural-breaks row checks only when `methods` includes `'natural-breaks'`. Add the 1,025-row equal-interval case.
- [ ] Fix visibility auto's fallback count using the two successful explicit plans in SCL-04 as regression cases.
- [ ] Budget the terrain route selected for the supplied device. Test estimate and actual route separately.
- [ ] Validate `activeSliceCount` and packed float32 magnitudes consistently in all three stream packers. Document which fields remain constant across frames.
- [ ] Correct the fractional categorical, time-word, cell-period-key, rank Markov, terrain, and point-cloud limit tables with the executed boundaries above.
- [ ] Add distinct public threshold and forced-maximum names for island scan and exact traces. Fix the island grid catalog product.
- [ ] Publish the resolved region-graph route and the prepared-BVH allocation calculation. Both can be additive.

## Larger proposals

1. **Correct streaming moments before alpha.** This changes observable numerical output. Twenty-six in-repository constructions are in reach, with zero required signature changes. Pin the two-value result on a real GPU before accepting the fix. Then add a reusable stream encoding plan so a mathematically correct update cannot subsequently be decoded in the wrong units.
2. **Introduce the common planner incrementally.** Start with visibility, class breaks, terrain flow, raster region graph, and prepared joins. These give five concrete acceptance cases: find a feasible sampled route, ignore an inactive algorithm, budget the selected route, preflight before graph mutation, and account for padded global scratch. Generate route tables only after the planner and constructor agree on those cases.
3. **Consolidate the redundant chunked validation name before publication.** `S/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-validation.ts:41–46` forwards to `GPUSpatialWeightsValidation`. Deprecating the alias is additive. Removing it is **BREAKING** and changes **five `new GPUChunkedSpatialWeightsValidation` occurrences**: two spatial-weights tests, one kernel-audit case, one website runner and one website catalog example string, plus corresponding exports, imports and catalog identity. I recommend one canonical validation class. Keep separately shaped algebra, transform and transpose APIs until their operation and output contracts have been reconciled.

## Not read

I read the tiled region-graph recipe, natural-breaks limit definitions, class-break input helper, and terrain flow budget helpers in full. For the larger contributors I traced constructor validation, route selection, command estimation, relevant parameter packers, and the specific WGSL expressions cited. I sampled the API pages from the caller's decision point and then checked them against source, the catalog, and host probes. I checked candidate findings against the synthesis, area reports, API/catalog review, direction review, and decision queue. Known layout-version, general network-structure, rational-quantile, and dense region-matrix issues are marked as controls above.

The exported-surface and validator counts cover the package. The route-by-route numerical audit does not cover every private kernel, CPU twin, recipe, or all 359 catalog bounds. No GPU or browser ran. The two small GPU probes in SCL-01 and SCL-02 and the seeded rollover probe in SCL-06 are the outstanding numerical confirmations.