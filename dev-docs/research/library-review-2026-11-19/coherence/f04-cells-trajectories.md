# From cell IDs and GPS fixes to maps, stops, and landscape metrics

## Summary

The GPU cell operators mostly agree on a 64-bit representation. The harder problems arise when one operator’s valid output becomes another operator’s valid input with a different meaning.  
The highest-value fix is to let the landscape recipes consume the raster package’s patch-label port directly. Today, passing those labels unchanged can count background as a patch and discard the last real patch.  
A truncated cell table can likewise produce a plausible outline with `overflow = 0`. The documented composition passes the count and loses the evidence that cells were dropped.  
For urban morphology, the isovist recipe computes accurate local ray distances, then loses them again when it measures rounded map coordinates.  
For movement, I reproduced a ten-minute stay spanning a missing GPS fix. I also found that the default stop example reports zero stops for an entirely stationary track, and that the MovingPandas cross-reference conflates three different definitions of a stop.  
Two published graph examples fail at declaration. Both failures can be caught without compiling a shader or having a GPU.  
I would keep the caller-owned graph model and fix these handoffs before publishing the alpha.

## Census

`S/` below means `modules/analysis/src/gpu-spatial-analysis/`, `G/` means `modules/analysis/src/geospatial-analysis/`, and `D/` means `docs/api/gpu-spatial-analysis/`. `KI` marks evidence read in the newer `luma-contributors-ki` copy. I checked `.typecheck/r19/staging-owned.txt` for the cited paths.

I enumerated the scoped directories with `rg --files`, then used Node and the TypeScript syntax parser to follow named and wildcard exports from each family’s `index.ts`. I counted distinct class or recipe names, collapsing function overloads. There were 248 TypeScript files present in r19 and four additional staging-owned files. Of the 36 scoped source paths on the staging list, those four were present only in KI. The staged choropleth module adds a public pure CPU function, not a public operation class.

| Family | GPU classes excluding operation forms | GPU operation classes | CPU operation classes | Public `add…Recipe` functions |
|---|---:|---:|---:|---:|
| `cell-*` | 18 | 4 | 4 | 0 |
| `healpix-cells`, `hilbert-keys` | 3 | 1 | 1 | 0 |
| `global-coordinates`, `geospatial-analysis` | 7 | 1 | 1 | 0 |
| `trajectory-*` | 21 | 11 | 11 | 0 |
| `track-similarity` | 2 | 1 | 1 | 0 |
| `urban-morphology` | 0 | 0 | 0 | 7 |
| `landscape-ecology` | 1 | 1 | 1 | 8 |
| `recipes` | 0 | 0 | 0 | 24 |
| **Total** | **52** | **19** | **19** | **39** |

Two smaller counts explain much of the first-use experience.

| Convention measured | Count | Interpretation |
|---|---:|---|
| Cell-family GPU producer source files mentioning `wordOrder` | 7 | `GPUCellAggregation`, `GPUCellPyramid`, `GPUCellGeometry`, `GPUCellMeasures`, `GPUCellCompaction`, `GPUCellGridPath`, `GPUCellTopology`. Little-endian `(low, high)` is the interchange layout. Alternate input order is a convenience on selected readers. |
| Recipe destination supplied as an `outputs` object | 19 | All in `recipes/`. |
| Recipe destination supplied as an `output` object | 15 | All seven urban and eight landscape functions, including the named tessellation routes. Raster resistance requires its output. |
| Recipe destination supplied as one `output` view | 2 | `addExcessRiskRecipe`, `addSpatialRateRecipe`. |
| No top-level `output` or `outputs` override | 3 | `addDrainageNetworkRecipe`, `addExposureByZoneRecipe`, `addRegionalRasterChangeRecipe`. |

For the first count I matched `wordOrder` in the defining files of barrel-exported classes, excluding `*Operation`. For the second I followed each exported function’s props, including `GPURecipeOverrides` and inherited props. For migration reach I used `rg -l` over `modules examples website test scripts`, then counted call expressions and literal destination properties with the syntax parser.

### Fifteen calls from the docs

`T` below stands for `{positions, timestamps, trackOffsets}`. I supplied packed graph views of the documented formats. “Declared” means the call succeeded on a CPU-only null device. It does not establish a shader result.

| Operation and call attempted | Result |
|---|---|
| `new GPUPointToCell({family:'h3', resolution:8, positions, output:{cells}})` | Declared. For `[-122,37]`, the CPU twin returned `[2933915647,142771009]`. Reassembly gives `8828341aedfffff`, exactly `h3-js`’s `latLngToCell(37,-122,8)`. |
| `new GPUCellGeometry({family:'h3', cells, maximumVertexCount:10, output:{centers,boundaries,vertexCounts}})` | Declared with 20 boundary rows for two cells. The H3 example gives enough information to allocate this correctly. |
| `new GPUCellTopology({family:'h3', operation:{type:'disk',k:2}, cells, output:{cells:diskCells,distances,counts}})` | Declared with 38 slots for two input rows. The caller must use the per-row counts when reading those slots. |
| `new GPUCellCompaction({family:'h3', operation:{type:'compact',resolution:9}, cells, output:{cells:compacted,status:{count,overflow}}})` | Declared. Here `resolution` describes the input cells. In `uncompact` it describes the target. The operation discriminant matters. |
| `new GPUHealpixPointToCell({order:1, scheme:'nested', positions, output:{cells,validity}})` | Declared. CPU `[0°,90°]` gave nested words `[3,0]` and UNIQ words `[19,0]`. Both had validity 1. |
| `computeS2CapCoveringOnCPU(Float32Array.of(1,0,0,.01), {level:0,capacity:6})` | KI twin returned key `[0,268435456]`, counts `[1,1]`, status `[0]`. This API takes unit XYZ and radians. A shared geographic point-to-cell call takes longitude and latitude in degrees. |
| `selectionGraph.add(new GPUCellLevelSelection({levelCounts,activeLevel,output:{status:{count},drawArguments}}))` | The documented reuse of the pyramid graph’s `levelCounts` failed graph ownership validation. UCT-09. |
| `new GPUTrajectoryMetrics({...T,parameters,output:{trackLengths,averageSpeeds,maximumSpeeds,trackStopCounts,stops}})` | Declared with the documented compact stop output. The threshold is distance per timestamp unit. Epoch milliseconds do not make this a metres-per-second threshold. |
| `new GPUTrajectoryStaypoints({...T,parameters,output:{ids,status:{count,overflow},startRows,finishRows,centroids,durations,staypointIds}})` | Declared. This example needs **12 views** including the three trajectory columns and parameters. The CPU twin demonstrated both the final-stop default and the missing-fix problem below. |
| `new GPUTrajectoryResample({...T,sampleCount:32,spacing:'arc-length',output:{samples,sampleElevations,sampleTimes}})` | The published example threw because it requests sampled elevations without supplying elevations. UCT-09. |
| `computeModalSplitOnCPU({...T,triplegs,modes:Uint32Array.of(7)}, {metric:'count',modeCount:2})` | One tripleg produced `[0,1]`. The CPU table wants `triplegs.count`; the corresponding GPU table wants `triplegs.status.count`. UCT-08. |
| `new GPUTrackSimilarity({linesA,pairA,pairB,editDistanceTolerance:25,editDistanceGapPoint:[0,0],output:{editDistanceOnRealSequence:edr,editDistanceWithRealPenalty:erp,symmetricSegmentPathDistance:sspd}})` | KI graph declared. The KI CPU twin also supplies a counterexample to the ERP interpretation on the page. UCT-16. |
| `addIsovistsRecipe(graph,{positions,segments,rayCount:4,maximumDistance:.1})` | Declared. The result provides both distances and endpoints. Using the latter to calculate the former’s maximum introduces a projected-coordinate error. UCT-03. |
| `addLandscapeMetricsRecipe(graph,{width,height,patches,patchClasses,cellSize:[1,1],classCount:1,connectivity:4})` | Declared. Both the patch-ID convention and the units of the packed result matter. UCT-01 and UCT-14. |
| `new GPUPointBuffer({points,parameters,offsets,verticesPerRing:32,coordinateSystem:'geographic'})` | Declared with 32 offsets per point. Its radius packer accepted a positive input that became zero. UCT-07. |

I would retain the domain terms `resolution` for H3 and shared indexing, `level` for the S2 compatibility API, and `order` for HEALPix and Hilbert. A Hilbert order of 16 produces a **32-bit** sorting key. It is not a 64-bit DGGS cell ID. A conversion table is more helpful here than a rename.

## Proposed convention

A documented producer-to-consumer example must carry the producer’s encoding, units, active-row count, and available loss evidence through to the answer. Check each example by declaring it on the null device and running the smallest distinguishing fixture on the CPU twin or GPU oracle. For recipe destinations, also assert that the returned view is the exact view the caller supplied.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UCT-01 | W/C | ORDINARY | ADDITIVE | M | Raster patches → landscape | Background and the last patch exchange roles when native raster labels are passed through. |
| UCT-02 | W/C | ORDINARY | ADDITIVE | M | Cell aggregation → outline | A partial input can yield a complete-looking boundary, including a false interior edge. |
| UCT-03 | W | ORDINARY | BREAKING numeric correction | S | Isovists | Valid ray distances can produce zero area, perimeter, and maximum distance. |
| UCT-04 | W | ORDINARY | ADDITIVE | M | Staypoints | A missing GPS fix can contribute an entire interval of reported dwell. |
| UCT-05 | W/C | ORDINARY | ADDITIVE | S | Geographic resampling → cells | Three of five indexing families change the cell when fed an equivalent unwrapped longitude. |
| UCT-06 | W | ORDINARY for JS/config inputs | ADDITIVE | S | HEALPix CPU and export | The unsupported `ring` scheme silently produces nested IDs. |
| UCT-07 | W | UNUSUAL | ADDITIVE | S | Brownian bridge, point buffer | Host packers accept a collapsed grid, a zero encoded radius, or an empty destination. |
| UCT-08 | W/C | ORDINARY | BREAKING in part | S | Modal split | An unknown category becomes the last mode, and the CPU twin accepts breaks the GPU rejects. |
| UCT-09 | S/D | ORDINARY | ADDITIVE | S | Cell and trajectory examples | Two literal documentation examples fail before GPU execution. |
| UCT-10 | C/E | ORDINARY | ADDITIVE | M | 39 recipe builders | A plausible destination property can be silently ignored. |
| UCT-11 | E/S | ORDINARY | ADDITIVE | S | DGGS host interchange | The documented readback consumer accepts neither of the two obvious typed-array representations. |
| UCT-12 | D/E | ORDINARY | ADDITIVE | S | Stop detection | The cross-reference conflates diameter, speed, and anchor distance, and the example omits its last stay. |
| UCT-13 | C/E | UNUSUAL | ADDITIVE | M | Radius of gyration | Two nearby dateline visits produce a radius of 19,904 km under the compatibility convention. |
| UCT-14 | D/E | ORDINARY | ADDITIVE | S | Landscape metrics | A FRAGSTATS user must find a conversion near the end of the page to interpret MESH. |
| UCT-15 | D | ORDINARY | ADDITIVE | S | Brownian bridge | The declared bound on raster probability mass is false even for two stationary fixes. |
| UCT-16 | D | ORDINARY | ADDITIVE | S | Track similarity | ERP distance zero does not establish that two input tracks are identical. |

### UCT-01: Accept the raster package’s patch labels directly

**Where.** `modules/analysis/src/raster-analysis/raster-patches/patch-labels.ts:41`, `S/landscape-ecology/landscape-patch-metrics.ts:49`, `patch-proximity.ts:51`, `patch-proximity-boundary.ts:384`, KI `D/landscape-ecology.md:47,99`.

**Evidence and current call.** `addLandscapeMetricsRecipe(graph,{...settings,patches:dense.labels,patchClasses})` is type-correct. The raster contract uses zero for background and `1..N` for patches. Landscape accepts `0..N-1` as patch rows and treats larger values as background. **SOURCE.** The adapter literally adds one to every owner below `patchClasses.length`. Smallest GPU probe: one class, one patch, unit cell area, three cells, dense labels `[0,1,1]`. Read `patchMetrics[0].area`. The traced result is **1**, representing the background cell. The intended area is **2**. The reference and boundary proximity routes also interpret the input as zero-based patch rows.

**Wanted.** Accept the existing `GPURasterPatchLabels` port through an explicit `denseLabels` alternative to `patches`. Carry its `converged`, `overflow`, and `componentCount` through the adapter. Show the three-cell example on both recipe pages. **Callers:** none must change. **Value and cost:** raster users can compose two public families without silently reversing foreground and background, M.

### UCT-02: Carry input completeness into cell outlines

**Where.** KI `D/operators-cells.md:619,646,650`, KI `S/cell-set-outline/gpu-cell-set-outline.ts:100,418,525`, `S/cell-aggregation/gpu-cell-rollup.ts:93`.

**Evidence and current call.** `new GPUCellSetOutline({cells:table.cells,count:table.status.count,...})` is the documented composition. I declared it. Its input has no place for the table’s `overflow` or `candidateOverflow`. **SOURCE.** A missing neighbor is a boundary; nested ring assembly checks only the outline’s own destination overflow. Smallest GPU probe: aggregate two adjacent Quadbin cells into capacity one, then outline the retained cell into four segment slots. Expect four emitted edges and zero outline overflow, including the edge shared by the original pair. The complete two-cell outline has six exterior edges and omits that shared edge.

**Wanted.** Accept `source:{cells,status}` alongside the current bare-cell input. Publish source loss as `candidateOverflow`, retain destination loss as `overflow`, and suppress nested rings when either is set. `GPUCellRollup` already illustrates the distinction. **Callers:** none must change; update the documented table handoff. **Value and cost:** map authors can distinguish a partial cell set from an actual exterior boundary, M.

### UCT-03: Measure isovists before rounding the endpoints into map coordinates

**Where.** `S/urban-morphology/isovists.ts:332,358,389`. This concerns the measured geometry, separately from the allocation issue in `KNOWN(URB-01)`.

**Evidence and current call.** `addIsovistsRecipe(graph,{positions,segments,rayCount:4,maximumDistance:.1})` declared successfully. **SOURCE.** The cast writes a distance, then writes `origin + direction * distance` as float32. The measure pass subtracts the origin from those rounded endpoints. **EXECUTED arithmetic probe:** at `origin = 5_000_000`, the four x-components of `fround(origin + directionX * .1) - origin` were `[0,0,0,0]`; y behaves identically. Smallest GPU probe: one free viewpoint at `(5_000_000,5_000_000)`, one distant segment, four uniform rays. Read distances and measures. The four range-limited rays should give area `.02`, perimeter `4√2 × .1`, and maximum `.1`. The traced measure pass gives `[0,0,0,1]`.

**Wanted.** Form local vertices from the existing `angles` and `distances` when reducing measures. Keep absolute endpoints for drawing. **Callers:** zero source edits across 32 direct `addIsovistsRecipe` call expressions; numerical expectations change for affected fixtures. **Value and cost:** projected-coordinate users recover valid local measurements, S.

### UCT-04: Give missing fixes a defined staypoint policy

**Where.** `S/trajectory-staypoints/cpu-trajectory-staypoints-operation.ts:150,185`, `gpu-trajectory-staypoints.ts:392,657`; `dev-docs/contributor-contract.md:104`.

**Evidence and current call.** **EXECUTED.** I passed `computeTrajectoryStaypointsOnCPU` three positions `[(0,0),(NaN,NaN),(1000,0)]`, times `[0,300000,600000]`, offsets `[0,3]`, distance threshold `100`, time threshold `300000`, and gap limit `900000`. It returned `count=1`, `duration=600000`, `centroid=[NaN,NaN]`, `overflow=0`, `candidateOverflow=0`. A consumer requesting only counts and durations sees a ten-minute stay. The NaN fails the distance comparison and the search continues. **SOURCE GPU probe:** use the same three rows and read count, duration and centroid. This is independent of the search cap in `KNOWN(TRJ-102)`.

**Wanted.** Add a row `mask` and an explicit `missingFixPolicy:'break'` that closes the contiguous run of observed fixes and restarts after the missing row. Count missing fixes separately from capacity loss. For this fixture the break policy should yield zero qualifying stays. **Callers:** none for the additive policy. **Value and cost:** GPS dropouts cease to count as evidence that someone remained stationary, M.

### UCT-05: Specify longitude handling at the trajectory-to-cell handoff

**Where.** `S/trajectory-interpolation/gpu-trajectory-resample.ts:138`, `gpu-trajectory-playhead.ts:172`, `S/cell-indexing/gpu-point-to-cell.ts:119`, `cpu-point-to-cell-operation.ts:82,178,272`.

**Evidence and current call.** `new GPUPointToCell({positions:resampled.samples,family,resolution:5,output:{cells}})` accepts positions on a continuous longitude branch. **EXECUTED CPU**, at `(210°,0°)` and its equivalent `(-150°,0°)`:

| Family | Words at 210° | Words at −150° |
|---|---|---|
| Quadbin | `[4294967295,1213727743]` | `[4294967295,1213731839]` |
| Quadkey | `[512,335544320]` | `[516,335544320]` |
| Geohash | `[30758229,1342177280]` | `[8668228,1342177280]` |
| H3, S2 | Equal within each family | Equal within each family |

All ten rows had validity 1. This affects ordinary dateline-crossing output, independently of extreme-coordinate precision in `KNOWN(DGGS-106)`. **SOURCE GPU probe:** submit these two rows at resolution 5 for each family and compare both words.

**Wanted.** Document the handoff on both pages and offer `longitudeHandling:'wrap'` for periodic input. Preserve existing tile-edge clipping for callers that request it. **Callers:** none. **Value and cost:** a route crossing the date line bins into the intended cells across all five families, S.

### UCT-06: Refuse unsupported HEALPix schemes on every backend

**Where.** `S/healpix-cells/cpu-healpix-point-to-cell-operation.ts:37,105`, `gpu-healpix-point-to-cell.ts:138`, `healpix-dggs-json.ts:105`.

**Evidence and current call.** **EXECUTED.** At order 1 and `[0°,90°]`, the JS call `computeHealpixPointToCellOnCPU(p,{order:1,scheme:'ring'})` returned `[3,0]`, validity 1, exactly the nested answer. The CPU operation constructed successfully. The GPU contributor rejected the same scheme with `scheme must be 'nested' or 'uniq'`. The JSON builder accepted `scheme:'ring', cells:[3n]` and placed the sample in nested slot 3. TypeScript rejects the literal, but a JSON configuration can reach all three APIs.

**Wanted.** Share the two-value scheme validator across the contributor, operation constructors, pure CPU function and JSON builder. Keep `undefined` as nested. Mention RING explicitly in the error because it is a real HEALPix numbering scheme. **Callers:** none using a supported scheme. **Value and cost:** a HEALPix user cannot accidentally label nested results as RING, S.

### UCT-07: Validate the values the GPU will actually receive

**Where.** `S/trajectory-home-range/gpu-brownian-bridge-utilization.ts:86`, `G/point-buffer/gpu-point-buffer.ts:60`. New packers in the defect class of `KNOWN(CORE-101)` and `KNOWN(CORE-107)`.

**Evidence and current call.** **EXECUTED.** Packing bridge extent `[5000000,0,5000000.1,1]` returned `[5000000,0,5000000,1]`. The CPU twin consuming those parameters returned identical densities `[353.67767,353.67767]` for a two-column grid. `getGPUPointBufferParameterValues(1e-50)` returned `[0,0]`, although the radius must be positive. `getGPUPointBufferParameterValues(1,new Float32Array(0))` returned `[]`.

**Wanted.** Validate strictly increasing encoded extent endpoints, a strictly positive encoded radius, and a destination of at least two floats before writing. Give the bridge error a remedy: subtract a local origin before uploading the positions and extent. **Callers:** none with representable settings and correctly sized targets. **Value and cost:** a construction error replaces duplicate grid columns or collapsed rings, S.

### UCT-08: Make modal classification agree across CPU, GPU, and supplied categories

**Where.** `S/trajectory-mobility/cpu-modal-split-operation.ts:59,149,166,176`, `gpu-modal-split.ts:84,211,341,373,402`; `modules/analysis/test/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation.node.spec.ts:86`.

**Evidence and current call.** **EXECUTED.** For one tripleg, `modeCount:2, modes:[7], metric:'count'` returned counts `[0,1]` and `predictedModes:[0]`. The supplied mode was neither 1 nor 0. The last-bin convention is documented in source; the first-use page explains it for *speeds*, where it makes sense. Also, the pure CPU function accepted `modeSpeedBreaks:[-Infinity,Infinity]` and put speed 1 in class 0; the GPU constructor rejected the settings. For `[1.00000001,Infinity]` and exactly unit speed, the CPU returned class 0. **SOURCE GPU probe:** the first constant in `array<f32>` rounds to 1, so the strict comparison selects class 1. Both constructors also accept distinct double breakpoints that round to the same float.

**Wanted.** Validate one encoded array of speed breaks, require strict ascent after float32 conversion, and use those breaks in both implementations. Expose an explicit unknown-category policy for supplied IDs. Reserve `predictedModes` for runs that performed prediction, or return missing keys there. **Callers:** changing CPU cutoff semantics requires zero call-syntax changes across 17 direct pure-function calls. One supplied-mode test explicitly expects six zero “predictions” and would change if that result is corrected. Kepler’s CPU adapter uses actual prediction. **Value and cost:** transport categories have the same meaning on both backends, S.

### UCT-09: Repair the two graph examples that fail at declaration

**Where.** KI `D/operators-cells.md:156–165` and `D/operators-lines.md:1197`; KI `S/cell-aggregation/gpu-cell-level-selection.ts:130`, r19 `S/trajectory-interpolation/gpu-trajectory-resample.ts:225`. These are concrete failures beyond the snippet-testing gap in `KNOWN(DOC-110)`.

**Evidence and current call.** **EXECUTED on a null device.**

| Documented handoff | Actual error | Correction |
|---|---|---|
| `selectionGraph.add(new GPUCellLevelSelection({levelCounts,...}))` using the pyramid graph’s view | `cell-level-selection views must belong to the target graph` | Allocate one caller-owned counts buffer, import it separately into both graphs, and show the two distinct views. |
| `new GPUTrajectoryResample({...T,sampleCount:32,spacing:'arc-length',output:{samples,sampleElevations,sampleTimes}})` | `trajectory-resample sampleElevations requires elevations` | Include the `elevations` input or omit `sampleElevations` from the basic example. |

**Wanted.** Run both corrected snippets through `graph.add` with real views. **Callers:** two examples; zero implementation migrations. **Value and cost:** the documented first attempt works without reverse-engineering graph ownership or optional ports, S.

### UCT-10: Give recipe destinations one spelling and observable ownership

**Where.** `S/recipes/recipe-utils.ts:57`, `rate-mapping-recipes.ts:53,65`, KI `S/landscape-ecology/landscape-metrics.ts:85,225`. The 39 entry points are grouped below. This concerns the *input* destination independently of the result and rollback differences in `KNOWN(RCP-102)`.

**Evidence and current call.** **EXECUTED in r19 and KI.** Given a variable with otherwise valid settings plus `outputs:{landscape:myView}`, `addLandscapeMetricsRecipe(graph,settings).outputs.landscape === myView` was `false`. The call declared successfully and allocated a transient. A fresh TypeScript object literal helps catch this; an options variable can pass structural typing with the extra field. Nineteen sibling recipes teach precisely that spelling.

**Wanted.** Standardize on `output` for supplied destinations and `result.outputs` for returned views. Accept `outputs` as an alias during migration, reject conflicting assignments, and test view identity for each family. The three builders with no top-level override should list which terminal results are transient and how to retain them. **Callers:** additive alias, zero required changes. An immediate removal of the plural spelling has substantial reach, quantified under Larger proposals. **Value and cost:** a caller can trust the buffer they allocated to receive the answer, M.

### UCT-11: Accept typed cell IDs at the readback boundary

**Where.** `S/healpix-cells/healpix-dggs-json.ts:42,106`, `D/healpix-cells-dggs-json.md:69`, `examples/deck/kepler-compute/src/dggs-source.ts:82`, `modules/kepler/src/analysis/catalog-platform4-dggs-cells.ts:329`.

**Evidence and current call.** **EXECUTED.** For order 0 and value 3, `createHealpixDGGSJSON` accepted `cells:[0n]`. `BigUint64Array.of(0n)` threw `Cannot convert 0 to a BigInt`, because the typed-array `.map` tries to store a numeric slot back into a bigint array. Raw readback `Uint32Array.of(0,0)` threw `Cannot mix BigInt and other types`. The declared type is `readonly bigint[]`. The docs invite the caller to use aggregation output. The H3 result from the fifteen-call table also lost bits on a round trip through `Number`.

**Wanted.** Accept `ArrayLike<bigint>` using indexed iteration, and provide a word-based overload or helper taking `(words, count, wordOrder?)`. Show H3 hex, S2 token, and HEALPix nested/UNIQ in one host-interchange table. For example, S2 token `"1"` denotes `0x1000000000000000`, not `1n`; the repository already implements right-padding in the example reader. **Callers:** none. **Value and cost:** exact IDs survive ingest and readback without a per-application unpacker, S.

### UCT-12: Put three stop definitions and their endpoint rules side by side

**Where.** `docs/api/gpu-spatial-analysis-cross-reference.md:246`, `D/README.md:237`, KI `D/movement-semantics.md:23,94`, `S/trajectory-analysis/trajectory-metrics-kernels.ts:199`, `S/trajectory-staypoints/gpu-trajectory-staypoints.ts:117`.

**Evidence and current call.** The cross-reference specifically names `movingpandas.TrajectoryStopDetector` beside `GPUTrajectoryMetrics`. The former’s user-facing threshold is a maximum stop diameter; metrics groups consecutive slow steps; staypoints measures distance from a window anchor. **EXECUTED.** Using the staypoint example’s settings, three coincident fixes at `[0,300000,600000]` produced **zero** stays. Adding `includeLast:true` produced **one** with duration `600000`. In another fixture an anchor-to-next distance exactly equal to `distanceThreshold` closed the window.

**Wanted.** Give each definition a row with its spatial test, time unit, equality rule, end-row convention, and treatment of an open final window. Label the MovingPandas diameter detector as having no exact counterpart here. Show `includeLast:true` in the parked-vehicle example. **Callers:** documentation only. **Value and cost:** GIS users can choose the intended statistic before fitting thresholds to the wrong algorithm, S.

### UCT-13: Name the geographic center used by radius of gyration

**Where.** `S/trajectory-mobility/cpu-mobility-indicators-operation.ts:222,242`, `gpu-mobility-indicators.ts:155,461`, KI `D/movement-semantics.md:209`.

**Evidence and current call.** **EXECUTED.** Two equatorial staypoints at `179°` and `−179°` under `metric:'great-circle'` yielded `jumpLengths[1]=222390.15625 m` and `radiusOfGyration[0]=19903920 m`. The latter uses the coordinate mean at Greenwich. A center on the short arc gives approximately `111195 m`. The current behavior deliberately follows the scikit-mobility convention; the staypoint/location family already uses anchor-relative longitudes in geographic centroids. **SOURCE GPU probe:** request jump length and radius for these same two stays.

**Wanted.** Put this numerical example next to the metric and offer `gyrationCenter:'coordinate-mean'|'spherical-mean'`, retaining the coordinate mean for reproducibility. Specify the undefined-center result for antipodal data. **Callers:** none. **Value and cost:** global mobility users can select geographic behavior while reproducing existing studies, M.

### UCT-14: State FRAGSTATS units at the output slot

**Where.** KI `D/landscape-ecology.md:66,499`, KI `S/landscape-ecology/landscape-metrics.ts:323,328`, `modules/analysis/test/oracles/urban-landscape.R:23,27,31`.

**Evidence and current call.** `const mesh = landscape[1]` returns effective mesh in square coordinate units. The output table names the metric; the conversion from upstream hectares appears roughly 430 lines later. **SOURCE.** The R oracle multiplies patch area, class MESH, and landscape MESH by 10,000. Smallest GPU probe: one 100 m × 100 m patch, one class. Expect patch area and landscape MESH of `10000`, corresponding to `1 ha` in FRAGSTATS-style reporting. Percentages need no such conversion.

**Wanted.** Put `AREA`, `MESH`, `COHESION`, `AI`, and `CONTAG` alongside the actual result slot, unit, and undefined-value rule. For metre grids, print `MESH (ha) = landscape[1] / 10_000`. Link the landscape and urban contract pages directly from the spatial family index; their existing links through `coverage-expansion.md:19` are easy to miss. **Callers:** none. **Value and cost:** a landscape ecologist can compare reported numbers without a factor-of-10,000 error, S.

### UCT-15: Distinguish sampled Brownian-bridge density from cell probability

**Where.** `S/trajectory-home-range/gpu-brownian-bridge-utilization.ts:125,288,323`, `cpu-brownian-bridge-utilization-operation.ts:102,117`, KI `D/home-range-and-recursion-mob07.md:79,103`.

**Evidence and current call.** The public output comment says `sum(density * cellArea)` is below 1 by the mass outside the grid. **EXECUTED.** Two stationary fixes at `(0,0)`, times `[0,1]`, location errors `[1,1]`, zero motion variance, one midpoint, extent `[-5,-5,5,5]`, grid `[1,1]`: the CPU twin returned center density about `.31830987`, hence estimated mass **31.8309873**. The analytic cell-center value is `1/π`. **SOURCE GPU probe:** run that one cell and compare its density to `1/π`. The discrepancy comes from spatial quadrature, not truncation or GPU precision.

**Wanted.** Write “cell-center density; the area-weighted raster sum approximates captured mass and may exceed one on coarse grids.” State that the isopleth stage normalizes over the supplied raster. **Callers:** none. **Value and cost:** home-range users can distinguish resolution error from probability outside their study area, S.

### UCT-16: Qualify the ERP metric claim

**Where.** KI `D/track-similarity-measures.md:60,111`, KI `S/track-similarity/cpu-track-similarity-operation.ts:265,299`.

**Evidence and current call.** **EXECUTED KI CPU twin.** With default gap point `[0,0]`, `measures:['editDistanceWithRealPenalty']` returned **0** for `[(1,0),(2,0)]` versus `[(1,0),(0,0),(2,0)]`. Inserting the gap point itself costs zero. Distinct input tracks can therefore compare equal. **SOURCE GPU probe:** one pair with these two lines, request ERP only, expect zero.

**Wanted.** Describe ERP on these raw sequences as a pseudometric when the gap point is a permitted vertex. The reverse-triangle lower bound still applies. Add the counterexample next to the gap-point option so users do not deduplicate tracks by `erp === 0`. **Callers:** none. **Value and cost:** correct interpretation of a valid computed distance, S.

## Departures by name

For recipe destinations, these are the complete groups in scope. Names below omit the common `add` prefix and `Recipe` suffix.

- **Plural `outputs`, 19.** `HotSpotAnalysis`, `RateClusterMap`, `SpatialRegression`, `SpaceTimeHotSpots`, `PeriodComparison`, `PointsInPolygonsChoropleth`, `ClusterAndOutline`, `ChangeOfSupport`, `StraightLineCatchments`, `FleetDwell`, `FleetDwellZoneEvents`, `ZonalDistribution`, `InterpolationDiagnostics`, `RasterZonalDistribution`, `SurfacePreparation`, `LocalVariability`, `CellFlowTopology`, `DasymetricRedistribution`, `ContiguityOrderCorrelogram`.
- **Singular output object, 15.** `StreetProfiles`, `BuildingAdjacency`, `Isovists`, `VisibilityGraph`, `BuildingTessellation`, `BuildingTessellationReference`, `LocalBuildingTessellation`, `LandscapeMetrics`, `PatchProximity`, `ResistanceConnectivity`, `SparseResistanceConnectivity`, `RasterResistanceConnectivity`, `SegregationDecomposition`, `SegregationCounterfactual`, `PopulationSupportAlignment`.
- **Singular bare view, 2.** `ExcessRisk`, `SpatialRate`.
- **No top-level destination override, 3.** `DrainageNetwork`, `ExposureByZone`, `RegionalRasterChange`.

For cell depth vocabulary, `GPUPointToCell`, `GPUCellAggregation`, `GPUCellCover`, `GPUCellRollup`, `GPUCellPyramid`, `GPUCellTopology`, and `GPUCellCompaction` use resolution terminology at their respective input, output, level-list, or operation boundaries. `GPUS2CellHierarchy` and `GPUS2CapCovering` use `level`; `GPUS2PolygonCovering` has `minimumLevel` and `maximumLevel`. `GPUHealpixPointToCell`, `GPUHealpixCellAggregation`, and `GPUHilbertKeys` use `order`. `GPUCellLevelSelection.activeLevel` is an **index into the supplied level list**. Treating it as an H3 resolution or map zoom selects the wrong row.

For trajectories, the ordinary input is packed XY, timestamps, and a `trackCount + 1` offset array, with observations grouped by object and ordered in time inside each range. Repeated consecutive offsets can represent an empty track. A timestamp unit must travel with the thresholds: `0.5 m/s` is `0.0005 m/ms`. `GPUTrajectoryKalman` and restriction retain `times`/`offsets` aliases. `GPUTrackSimilarity` consumes `lineOffsets` and no timestamp column. Its distance says nothing about whether the two objects visited a location at the same time.

## Quick wins

The ten changes I would put first on an implementer’s brief are:

- [ ] Document native-to-landscape patch conversion with the `[0,1,1]` fixture, including source completeness. UCT-01.
- [ ] Repair the cell-outline example now to inspect the source table’s two loss flags as well as destination overflow. UCT-02.
- [ ] Add the translated four-ray isovist oracle and calculate measures from local rays. UCT-03.
- [ ] Show the five-family longitude comparison at the resample-to-cell handoff. UCT-05.
- [ ] Reuse one HEALPix scheme validator in the CPU functions, operation constructors, and JSON export. UCT-06.
- [ ] Reject a collapsed encoded bridge extent, zero encoded radius, and short point-buffer target before writing. UCT-07.
- [ ] Fix and null-device-declare the pyramid-selection and resampling snippets. UCT-09.
- [ ] Accept `BigUint64Array` in DGGS-JSON and show one exact H3/S2/HEALPix word round trip. UCT-11.
- [ ] Put speed-based, anchor-based, and diameter-based stops in one comparison with a stationary final-window example. UCT-12.
- [ ] Put FRAGSTATS identifiers and units in the landscape result table. UCT-14.

## Larger proposals

**Adopt the native patch-label port and carry completeness through geometric outputs.** Both changes can be additive. I would do these before a vocabulary rename. They prevent maps and statistics that look entirely credible while describing different input data.

**Standardize recipe destinations before publication, with a compatibility alias during the repository migration.** The 19 plural-destination functions had **347 direct calls**, of which **146 explicitly supplied an inline `outputs` property**. Those 146 comprise 75 analysis tests, one analysis implementation, 13 Kepler implementation sites, one Kepler test, 11 lenses, 19 website sites, and 26 example sites. Another 105 calls passed a nonliteral second argument; a property-name search alone cannot give an exact migration count for those configurations. Add the singular spelling and identity assertions first. A later removal of the plural spelling is breaking.

**Give movement users explicit policies where reference-package compatibility changes the scientific interpretation.** Preserve the current scikit-mobility gyration definition by name, expose a geographic center option, and distinguish missing observations, unknown categories, and unfinished final stops. These choices are part of the answer a GIS developer reports.

## Not read

I read the scoped family barrels, HEALPix JSON builder and display-boundary helper, trajectory sort check, parameter packers cited above, geographic viewport helper, and the relevant CPU classification, staypoint, bridge-density and track-distance paths. I sampled the shared cell operators, S2 covering, graph declarations, recipe status propagation, isovist reduction, landscape label adapter, and R oracle by tracing the fifteen calls and their handoffs. I checked candidates against the earlier synthesis, area reports, API/catalog and direction reports, and the decision file.

I did not audit every H3 lookup table, S2 refinement kernel, encounter or Kalman pass, visibility-graph traversal, resistance solver, or recipe stage. No browser, network, GPU, build, typecheck, or test runner was used. The reported executions are in-memory host probes and graph declarations; each shader-result finding includes its outstanding device probe.