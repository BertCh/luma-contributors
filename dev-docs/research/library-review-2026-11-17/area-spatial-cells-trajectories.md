# Cells/DGGS, trajectories, urban and landscape, and recipes

## What I checked

I found five public-API cases where the implementation can change a result or lose data. The highest priorities are a signed cell-table delta that can reverse direction, a hot-spot recipe that reports a different multiple-testing family from the one it uses, and a track-distance calculation that can place the nearest point farther away than either endpoint.

I used these path prefixes throughout the report.

| Prefix | Repository path |
|---|---|
| `S/` | `modules/analysis/src/gpu-spatial-analysis/` |
| `T/` | `modules/analysis/test/gpu-spatial-analysis/` |
| `C/` | `modules/analysis/src/analysis-core/` |

- **Inventory.** The 27 scoped directories plus `S/index.ts` contain 250 TypeScript files and 73,525 lines. I located 224 specs under matching directory names and another 39 under `T/urban-ecology/` and `T/rts08-healpix-a5/`. Those are inventory counts, not executed tests.
- **Contributor and recipe census.** I counted 51 exported classes explicitly implementing `GPUCommandNodeProducer`, including 21 in `trajectory-*`, plus 19 `GPUContributorOperation` subclasses and 19 `GPUCPUOperation` subclasses. There are **37 public `add…Recipe` factories** in the scoped families: 24 implemented under `recipes/`, five exported by urban morphology, and eight by landscape ecology. This excludes three internal recipe helpers and counts overloaded names once. The barrels include both named and wildcard exports. `S/recipes/index.ts:96`, `S/recipes/index.ts:108`, `S/urban-morphology/index.ts:5`, `S/landscape-ecology/index.ts:19`
- **Full reads included** `AGENTS.md`, `dev-docs/contributor-contract.md`, `S/index.ts`, `C/contracts/temporal-context.ts`, `cell-index-families.ts`, `healpix-dggs-json.ts`, `brownian-bridge-variance-estimation.ts`, `prepare-trajectory-observations.ts`, `trajectory-sort-check.ts`, `trajectory-binding-validation.ts`, `trajectory-counts-wgsl.ts`, and `contiguity-order-correlogram-recipe.ts`.
- **Deep samples** included the 1,452-line hot-spot recipe, space-time hot spots, both fleet-dwell variants, cell-table compare and its oracle, CPU/GPU track similarity, zone events, HEALPix aggregation, segregation counterfactuals, isovists, and zonal statistics. I sampled S2 compaction, building tessellation, hot-spot inference, fleet dwell, home-range and trajectory-order specs. I used scoped `git log`, the earlier research reports and targeted roadmap searches to check prior coverage.
- **Limits.** This was read-only. I ran small extracted host-function and arithmetic probes. I did not compile, run a repository test, dispatch a GPU kernel, benchmark a device, or independently validate every DGGS reference table. Global coordinates, grid generators, and the remaining trajectory families received lighter sampling. `UNVERIFIED` below identifies the device outcome separately from what the source or host probe established.

The principal census commands are reproducible from the repository root. For shared ownership I also counted matching imports by source file and containing family.

```zsh
S=modules/analysis/src/gpu-spatial-analysis
T=modules/analysis/test/gpu-spatial-analysis
D=("$S"/{cell-*,trajectory-*,healpix-cells,hilbert-keys,grid-generators,global-coordinates,track-similarity,urban-morphology,landscape-ecology,zonal-statistics,recipes,shared})

rg --files "${D[@]}" -g '*.ts' | wc -l    # 249, plus S/index.ts
rg -Ul 'export\s+class\s+\w+[^{]{0,120}?implements\s+GPUCommandNodeProducer' "${D[@]}"
rg -n 'extends (GPUContributorOperation|GPUCPUOperation)\b' "${D[@]}"
rg --no-filename -o 'export function add\w+Recipe\b' "$S/recipes" | sort -u | wc -l
rg -n '\bgraph\.add\(' "$S/recipes" -g '*recipe*.ts'
rg -U 'export type \w*RecipeResult\s*=\s*GPURecipeResult\s*[<&]' "$S/recipes"
rg -n "from ['\"][^'\"]*/shared/" modules -g '*.ts' -g '**/src/**'
```

## Findings

### DGGS-01 Cell-table comparison can reverse the sign of a change [P0] [correctness]

- **Evidence.** The API promises that `after - before` is “computed exactly in integers” before rounding. The kernel uses `cellAddI64(afterValue, cellNegateI64(beforeValue))`, then interprets bit 63 as the sign. The test oracle explicitly repeats `BigInt.asIntN(64, afterMeasure - beforeMeasure)`. `S/cell-table-compare/gpu-cell-table-compare.ts:84`, `S/cell-table-compare/gpu-cell-table-compare.ts:548`, `C/cell-key-wgsl.ts:106`, `C/cell-key-wgsl.ts:126`, `T/cell-table-compare/cell-table-compare-oracle.ts:114`
- **Why it matters.** At `sumScale: 1`, two rows of `−2^61` produce a valid before sum of `−2^62`. Two rows of `+2^61` produce a valid after sum of `+2^62`. The correct delta is `+2^63`, exactly representable in float32. The implemented integer operation decodes it as `−2^63`. Neither input table overflowed. The oracle currently endorses the reversal.
- **Proposed change.** Compare the signed inputs, compute the unsigned magnitude of their difference in two words, then apply the sign when converting to float. Change the oracle to subtract unbounded BigInts. Add both directions of this case and retain a near-cancellation case. **M, nonbreaking.**
- **Confidence.** High. BigInt arithmetic checked. GPU readback `UNVERIFIED`.

### RCP-01 Hot-spot permutations test rows the recipe says it excluded [P0] [correctness]

- **Evidence.** The recipe promises that rows below `minimumNeighbors`, default 8, are outside the multiple-testing family. It passes the threshold to analytic Gi*, then constructs the permutation test with the original `mask` and only a *maximum* neighbor count. The permutation contributor describes its own eligibility as “1 to … positive-weight neighbors.” The recipe overwrites that description with the analytic one. The paired statistic spec sets `minimumNeighbors: 1`. `S/recipes/hot-spot-analysis-recipe.ts:513`, `:1224`, `:1338`, `:1387`; `S/permutation-inference/gpu-local-permutation-test.ts:792`, `:1083`; `T/recipes/hot-spot-permutation-statistic.spec.ts:107`
- **Why it matters.** On an unmasked 5×5 queen lattice, nine interior rows have eight neighbors. All 25 have at least three. With finite nonconstant values and sufficient capacities, the two paths can use different families for correction while the returned metadata says they used the same family. This affects interpretation of interior rows as well as boundary rows.
- **Proposed change.** Give the permutation stage a focal-row eligibility threshold separate from donor eligibility. Apply it before multiple-testing correction. Derive `testedRowsDefinition` from that stage. Add a 5×5 fixture that checks the actual eligible rows, corrected output and metadata together. **M, nonbreaking.**
- **Confidence.** High for the divergent predicates and incorrect host metadata. Device p-values `UNVERIFIED`.

### TRJ-01 A geographic track’s reported nearest point can be farther away than both endpoints [P0] [correctness]

- **Evidence.** The CPU projects the start and end longitudes independently relative to the query, but evaluates the resulting fraction using the wrapped **start-to-end** longitude. The two calculations can select opposite arcs. The GPU helper has the same pair of expressions. The existing antimeridian fixture puts both tracks at the seam. `S/track-similarity/cpu-track-similarity-operation.ts:165`, `:187`, `:390`; `S/track-similarity/gpu-track-similarity.ts:709`; `S/geometry-measures/geodesic-wgsl.ts:296`, `:316`; `T/track-similarity/cpu-track-similarity-operation.node.spec.ts:287`
- **Why it matters.** In an extracted CPU probe, the equatorial point `(0°, 0°)` and segment `(100°, 0°) → (−100°, 0°)` produced a nearest angular distance of **180°**. Each endpoint is only **100°** away. The segment follows the antimeridian side, so 100° is the correct minimum. This reaches Hausdorff and symmetric segment-path distance. The documented local approximation still has both endpoints available as candidates.
- **Proposed change.** Use one consistently unwrapped segment for projection and evaluation, and include both endpoint distances in the minimum. Put the remote-query case in both backends and assert the endpoint upper bound. Coordinate the shared WGSL change with the geometry-measures reviewer. **M, nonbreaking.**
- **Confidence.** High for CPU, reproduced from the implementation. GPU outcome `UNVERIFIED`.

### TRJ-02 Brownian motion variance estimation excludes a valid maximum at zero [P0] [correctness]

- **Evidence.** For any positive residual-derived scale, the optimizer starts at `Math.log(scale) - 12`, evaluates strictly positive variances and returns `Math.exp(...)`. The likelihood itself accepts zero motion variance when location error is positive. The test compares a positive estimate with `0.98 × estimate` and `1.02 × estimate`. `S/trajectory-home-range/brownian-bridge-variance-estimation.ts:70`, `:86`, `:110`, `:135`; `T/trajectory-home-range/home-range-recursion-mob07.spec.ts:89`
- **Why it matters.** For `x=[0,1,0]`, `y=[0,0,0]`, times `[0,1,2]` and location error `1`, the model variance is `1.5 + 0.5 × motionVariance` and the squared residual is `1`. The likelihood is maximized at **zero**. The extracted optimizer returned `0.000006144212353786916`, with lower likelihood than zero. The windowed estimator uses the same optimizer.
- **Proposed change.** Evaluate zero explicitly, bracket toward zero when the first grid point wins, and compare the endpoint and interior likelihoods. Add the three-fix case and a positive interior optimum. **S, nonbreaking.**
- **Confidence.** High. Reproduced on the host.

### DGGS-02 HEALPix DGGS-JSON silently drops a named property [P0] [correctness]

- **Evidence.** The public builder accepts `Record<string, HealpixDGGSJSONProperty>`, initializes `schema.properties` and `values` as `{}`, then assigns `map[name] = …` to each. An extracted builder call with two own property names, `__proto__` and `temperature`, serialized only `temperature` in both places. `S/healpix-cells/healpix-dggs-json.ts:44`, `:121`, `:125`, `:134`, `:139`
- **Why it matters.** A valid input column and its schema disappear on export without an error. This occurs within the writer’s existing format, independently of OGC conformance work.
- **Proposed change.** Build the two dictionaries with `Object.fromEntries` or `Object.create(null)`. Round-trip both names through `JSON.stringify` and `JSON.parse` in a fixture. **S, nonbreaking.**
- **Confidence.** High. Reproduced on the host.

### RCP-02 Fleet dwell drops the trajectory’s declared time encoding [P1] [correctness]

- **Evidence.** Both variants accept a trajectory port carrying `temporalContext`. The stops variant forwards `timestampsLow` but omits `temporalContext`. The zone-events variant reconstructs the port without either field. Its docs say the low part is unused, but its argument type still accepts it. The underlying zone contributor validates the encoding when it receives a context. `C/contracts/ports.ts:57`, `:66`; `S/recipes/fleet-dwell-recipe.ts:172`, `:324`, `:458`, `:623`; `S/trajectory-zones/gpu-zone-events.ts:380`; `S/trajectory-analysis/trajectory-common.ts:35`
- **Why it matters.** A caller can declare double-single timestamps and pass highs `[2^24, 2^24]` and lows `[0,1]`. The zone stage receives two equal timestamps and loses the declaration that would have exposed the mismatch. More generally, callers get different validation through a recipe and its underlying contributor.
- **Proposed change.** Forward the complete supported time fields into each stage. Give the zone variant a type for its supported encodings and reject a supplied low part or incompatible declaration before graph mutation. Add a one-tick fixture and a mismatched-declaration fixture for each variant. **S, stricter for previously accepted unsupported inputs.**
- **Confidence.** High for field loss. Device dwell output `UNVERIFIED`.

### RCP-03 Unused stop slots can manufacture a fleet-dwell overflow [P1] [correctness]

- **Evidence.** The metrics kernel writes `(0,0)` for unused centroid slots. Fleet dwell sends the entire centroid view into `GPUPointInPolygonJoin`, then masks rows past `stops.status.count` **after** the join. Candidate loss is computed in point-row order. The existing fixture supplies 16 spare stop slots and candidate capacity 4096. `S/trajectory-analysis/trajectory-metrics-kernels.ts:845`; `S/recipes/fleet-dwell-recipe.ts:366`, `:380`, `:397`; `S/spatial-join/gpu-point-in-polygon-join.ts:178`; `T/recipes/fleet-dwell-recipe.spec.ts:45`, `:96`
- **Why it matters.** With zero real stops, 64 spare slots, one zone containing the origin and candidate capacity one, the join can report overflow entirely from padding. A consumer using completeness to accept or reject the result can discard a complete, empty answer.
- **Proposed change.** Feed the GPU-resident stop count into candidate generation as an active prefix, or pass a row mask consumed before candidates are counted. Test zero stops with an origin-containing zone and a tight capacity. **M, additive.**
- **Confidence.** High for the dataflow. Device status `UNVERIFIED`.

### LAND-01 Overlapping counterfactual outputs can silently turn a decomposition into a 50/50 split [P1] [correctness]

- **Evidence.** `compositionOnly` and `spatialOnly` are caller-overridable. Two separate stages write them, then the recipe passes both into the decomposition. Each dispatch can be individually legal when the caller supplies the same view twice. In the Shapley formula, equal counterfactual inputs make both components `0.5 × (final − baseline)`. `S/landscape-ecology/segregation-counterfactual.ts:75`, `:89`, `:134`; `S/landscape-ecology/segregation-decomposition.ts:106`, `:196`
- **Why it matters.** The two components still add up to the observed change. A conservation check passes even though the attribution is wrong.
- **Proposed change.** Preflight the named input and output ranges with the existing physical-buffer-aware `validateLandscapeViews` before adding either scenario. Test identical views, partially overlapping views and disjoint subviews of one allocation. **S, stricter for overlapping outputs.** `S/landscape-ecology/landscape-validation.ts:17`, `:24`
- **Confidence.** High for the alias and algebra. Device output `UNVERIFIED`.

### DGGS-03 The public sum-scale picker can loop forever on an accepted argument [P1] [correctness]

- **Evidence.** Cell aggregation publicly re-exports and recommends `getGPUCellSumScale`. It accepts every positive finite JavaScript number. For `getGPUCellSumScale(Number.MIN_VALUE, 1)`, `limit` and `exponent` become `Infinity`. The condition `2 ** (exponent + 1) <= limit` remains true after `exponent++`. The cap of 127 is applied only after the loop. `S/cell-aggregation/index.ts:20`; `S/cell-aggregation/gpu-cell-aggregation.ts:119`; `C/cell-key-wgsl.ts:40`, `:56`, `:62`, `:66`
- **Why it matters.** An accepted host input can hang graph setup before the caller gets a scale or an error.
- **Proposed change.** Return `2 ** 127` as soon as `limit >= 2 ** 127`, then run the correction loops only within the representable range. Cover `Number.MIN_VALUE` and values adjacent to the upper boundary. **S, nonbreaking.**
- **Confidence.** High. I checked the arithmetic and loop condition without invoking the unbounded call.

### TRJ-03 The trajectory sort diagnostic can certify non-finite timestamps as ordered [P1] [correctness]

- **Evidence.** The status contract says “non-finite float times also count.” The loop starts at `begin + 1`, and the float32 test is `current >= previous`. Double-single checks only the high and low ordering comparisons. The existing tests cover ordinary reversals, equality, reversed offsets and one-tick precision. `S/trajectory-analysis/trajectory-sort-check.ts:34`, `:102`; `S/trajectory-analysis/trajectory-metrics-kernels.ts:97`, `:110`; `T/trajectory-analysis/trajectory-sort-check.spec.ts:92`
- **Why it matters.** A singleton `[NaN]` executes no comparison. `[0,+Infinity]` passes the literal ordering comparison. Both are inputs for which the advertised diagnostic should count an invalid row.
- **Proposed change.** Check each float timestamp’s finiteness, including the first row and both double-single components. Count a non-finite row once and define how the next finite row is treated. Retain the exact Int64 comparison. **S, nonbreaking.**
- **Confidence.** High for the missing first-row check. Device non-finite behavior `UNVERIFIED`.

### RCP-04 Space-time hot spots can leave a partial graph and omit a contributor from its result [P1] [architecture]

- **Evidence.** All 20 recipe implementation files under `recipes/` use `RecipeBuilder`. Only `space-time-hot-spots-recipe.ts` also calls `graph.add` directly, seven times. One call adds a full `GPUPeriodIndex`. The recipe validates `props.parameters.length` after those additions and returns only `builder.contributors`. The result contract says the list contains the contributors added in declaration order. `S/recipes/space-time-hot-spots-recipe.ts:479`, `:517`, `:526`, `:546`, `:580`, `:601`, `:659`, `:692`, `:769`; `S/recipes/recipe-utils.ts:58`, `:80`, `:95`
- **Why it matters.** A short parameter buffer leaves commands in the caller’s graph after the factory throws. On success, code inspecting the returned list misses the period-index contributor as well as glue. These affect retry and inspection separately.
- **Proposed change.** Move parameter and caller-view preflight before the first graph addition. Use `builder.add` for `GPUPeriodIndex` and `builder.addGlue` for adapters. Test that invalid parameters leave the original node count intact and that the successful list contains the period-index stage. **S, nonbreaking.**
- **Confidence.** High. `KNOWN(STD-11)` is incomplete here. The row describes eleven kernel-only recipes, while this defect is in a composed recipe that already uses the builder. `dev-docs/roadmap.md:423`

### TRJ-04 Requesting only maximum distance can still allocate billions of unused points [P1] [performance]

- **Evidence.** The CPU accepts `densify` when its reciprocal rounds to a safe integer. It builds both dense point arrays before checking the requested measures. `maximumDistance` and symmetric segment-path distance subsequently use the original tracks. The densifier allocates a point array for every subdivision. `S/track-similarity/cpu-track-similarity-operation.ts:73`, `:131`, `:140`, `:382`, `:395`, `:407`
- **Why it matters.** Two two-point tracks, `measures: ['maximumDistance']` and the accepted `densify: 1e-9` request roughly two billion intermediate points for a result requiring four endpoint comparisons. I did not allocate them.
- **Proposed change.** Construct dense points lazily only for measures that consume them. For those measures, calculate the expanded length before allocation and apply the relevant algorithm limit first. Add a maximum-distance-only regression with a tiny densification fraction. **S for lazy allocation, nonbreaking.**
- **Confidence.** High from host control flow.

### DGGS-04 `GPUCellFamily` names two different capability sets, and the S2 roadmap is stale [P2] [ergonomics]

- **Evidence.** The spatial entry exports aggregation’s `GPUCellFamily = 'quadbin' | 'h3'`. Indexing has another `GPUCellFamily` that includes `'s2'`, but omits that name from its barrel. S2 already has a compaction layout, center covering, and same-level outlines. A compaction oracle covers all six faces through level 30. `S/index.ts:14`, `:17`; `S/cell-aggregation/cell-keys.ts:13`; `S/cell-indexing/cell-index-families.ts:20`; `S/cell-indexing/index.ts:11`; `S/cell-topology/gpu-cell-compaction.ts:236`; `S/cell-cover/gpu-cell-cover.ts:153`; `S/cell-set-outline/gpu-cell-set-outline.ts:106`; `T/cell-topology/gpu-cell-compaction.node.spec.ts:171`
- **Why it matters.** An application using the public umbrella name cannot type an S2 operation that the library already accepts. A maintainer reading the roadmap could plan to implement working code.
- **Proposed change.** Export explicitly named capability types such as `GPUCellAggregationFamily` and `GPUCellTopologyFamily`, retain the old public type as a compatibility alias, and publish a family-by-operation matrix including resolution and containment limits. Update the S2 row to distinguish shipped subsets from remaining parity work. **S–M, additive.**
- **Confidence.** High. `KNOWN(DUP-03)` is stale on compaction and understates cover and outline support. `dev-docs/roadmap.md:436`

### URB-01 Isovist callers lack a reliable output-size rule for vertex sampling [P2] [ergonomics]

- **Evidence.** Five per-ray output descriptions say `positions.length * rayCount`. Vertex sampling actually adds six rays per obstacle segment. Validation checks the expanded count, while shader generation and `angularSpacing` still use the original `props.rayCount`. With one segment and a base count of `−1`, the expanded count is five and passes the count check. `S/urban-morphology/isovists.ts:38`, `:48`, `:90`, `:114`, `:165`, `:176`, `:308`
- **Why it matters.** A caller following the output docs can underallocate every per-ray view. A malformed base count can also survive construction-time count validation and reach shader generation.
- **Proposed change.** Export `getGPUIsovistRayCount(baseRayCount, segmentCount, sampling)`. Validate the base count before expansion, use the helper for sizing and provenance, and document both counts. Preflight all override lengths before viewpoint classification adds graph work. **S, additive with stricter invalid-input handling.**
- **Confidence.** High for the sizing and validation mismatch. Shader behavior for a negative base count `UNVERIFIED`.

### RCP-05 The standard recipe output path loses useful TypeScript types [P2] [ergonomics]

- **Evidence.** Among declarations matching `export type …RecipeResult = GPURecipeResult…` in `recipes/`, 11 use a bare intersection and three supply generic arguments. The base defaults `outputs` and `intermediates` to `Record<string, unknown>`. Space-time hot spots types `result.cube`, but the same view at `result.outputs.cube` inherits `unknown`. The correlogram already supplies concrete types for both maps. `S/recipes/recipe-utils.ts:53`; `S/recipes/space-time-hot-spots-recipe.ts:285`, `:290`, `:776`; `S/recipes/contiguity-order-correlogram-recipe.ts:92`
- **Why it matters.** Callers following the common recipe contract lose autocomplete and type checking at the point where recipes are supposed to compose uniformly.
- **Proposed change.** Supply named output and intermediate types to `GPURecipeResult` for the bare intersections. Preserve existing top-level convenience fields. Start with fleet dwell and space-time hot spots, then add a type fixture that passes `result.outputs` directly to a downstream stage. **M, source-compatible for ordinary consumers.**
- **Confidence.** High. The 11-of-14 count describes declaration style. The loss of `outputs.cube` typing was checked directly.

## Cross-cutting patterns

**Cell keys have a common physical layout and several distinct validity contracts.** Point indexing documents `(low, high)` for Quadbin, H3, S2, quadkey and geohash. Generic cell tables use all-ones padding and a counted prefix. HEALPix aggregation uses zero padding, and NESTED zero is a real cell. The HEALPix point writer also emits zero for an invalid UNIQ row, although valid UNIQ keys are never zero. The aggregator’s keep predicate checks only the optional mask and value finiteness. A caller forwarding just `output.cells` can therefore group an invalid UNIQ row as cell zero. Hilbert keys serve a different purpose as scalar sort keys, with an explicitly documented collision at order 16. Preserve those distinctions in typed ports and carry validity with the key-producing output. `S/cell-indexing/cell-index-families.ts:8`, `:30`; `S/cell-aggregation/cell-table.ts:29`; `S/healpix-cells/gpu-healpix-point-to-cell.ts:265`; `S/healpix-cells/gpu-healpix-cell-aggregation.ts:45`, `:63`, `:280`; `S/healpix-cells/healpix-index-wgsl.ts:86`; `S/hilbert-keys/gpu-hilbert-keys.ts:47`

**Trajectories have a common logical port, with gaps at composition boundaries.** The port already captures ordered positions, terminal offsets, timestamps, optional source IDs and a temporal context. The context defines units and epoch explicitly. Kalman and restriction accept the canonical names alongside mutually exclusive legacy aliases. Two places where callers still have to translate are fleet dwell’s encoding handling and windowed Brownian variance: the estimator returns `n−1` values, while the advertised consumer requires `n` rows and ignores the last row **of each track**. A documented packing helper would prevent multi-track misalignment. Track similarity can reasonably continue to accept geometric lines for its time-independent measures. `C/contracts/ports.ts:56`; `C/contracts/temporal-context.ts:41`; `S/trajectory-analysis/trajectory-common.ts:50`; `S/trajectory-home-range/brownian-bridge-variance-estimation.ts:178`; `S/trajectory-home-range/gpu-brownian-bridge-utilization.ts:109`, `:187`; `S/track-similarity/cpu-track-similarity-operation.ts:355`

**`shared/` has eight modules with three different ownership outcomes.** Counts below are direct source-file imports through the `shared/` path.

| Modules | Direct users | Proposed ownership |
|---|---:|---|
| `hash-uint32`, `point-neighbor-grid-index` | 3 and 4 | Already forwarding to core. Migrate the remaining imports and remove the internal shims. |
| `bin-index-wgsl`, `complementary-error-function` | 2 each | All users are in spatial regression and point density respectively. Move each to its consuming family. |
| `cpu-polygon-rows`, `grid-cell-row-order`, `spatial-wgsl-point-helpers`, `triangle-edge-adjacency` | 3, 5, 16, 3 | Used by 3, 3, 6 and 3 families respectively. Move to internal geometry, grid or kernel areas in `analysis-core`. |

This narrows `KNOWN(STD-09)` to four actual moves into core. The forwarding destinations and the domain-specific helpers are visible at `S/shared/hash-uint32.ts:6`, `S/shared/point-neighbor-grid-index.ts:7`, `S/shared/bin-index-wgsl.ts:5`, `S/shared/complementary-error-function.ts:11`, `S/shared/cpu-polygon-rows.ts:24`, `S/shared/grid-cell-row-order.ts:38`, `S/shared/spatial-wgsl-point-helpers.ts:19`, `S/shared/triangle-edge-adjacency.ts:41`. `dev-docs/roadmap.md:421`

## Keep

- **Keep recipes as thin graph construction functions.** `RecipeBuilder.add` already delegates to the graph and records declaration order. `addGlue` gives adapter kernels names and includes them in the returned stages. The correlogram combines that with typed outputs and per-order overflow. It is a good model for new composed recipes. `S/recipes/recipe-utils.ts:73`, `:89`; `S/recipes/contiguity-order-correlogram-recipe.ts:92`, `:215`, `:241`
- **Keep the urban graph-shape tests.** The building tessellation spec checks a 100,000-building declaration produces a single clipping dispatch, bounded storage and a bounded command count. It also checks rejection before graph mutation. These assertions catch growth and lifecycle regressions that small numerical fixtures miss. `T/urban-ecology/building-tessellation-local.node.spec.ts:46`, `:54`, `:109`, `:115`
- **Keep the specific numerical promises.** Zonal statistics documents ordering, same-device reproducibility, CPU tolerance and the packed-input default separately. The trajectory reservation counter similarly reserves a sentinel strictly above every accepted capacity. Both give consumers information they can act on. `S/zonal-statistics/gpu-zonal-statistics.ts:133`, `:314`; `S/trajectory-analysis/trajectory-counts-wgsl.ts:5`, `:17`

## Future

1. **Add an optional cell descriptor to table and key ports.** Carry family, encoding or scheme, resolution, validity representation and, for fixed-point sums, `sumScale`. Start with point-to-cell → aggregation → compare. Compare currently requires identical family and resolution and explicitly says it cannot check them. An optional descriptor permits immediate validation for new pipelines while retaining existing raw-view callers. The cost is maintaining descriptors through rollup and reprojection. `S/cell-table-compare/gpu-cell-table-compare.ts:111`; `S/cell-aggregation/cell-table.ts:41`, `:49`
2. **Use one trajectory time-port adapter for recipe boundaries.** Forward the declared encoding, units, epoch and optional low words together. Let each contributor declare accepted encodings. Migrate the recipes first, where callers can already supply the richer type. This adds some type complexity but removes repeated field reconstruction and makes unsupported precision visible at construction. `C/contracts/temporal-context.ts:41`, `:57`; `S/trajectory-analysis/trajectory-common.ts:32`; `S/recipes/fleet-dwell-recipe.ts:623`
3. **Give recipes one composition contract and keep execution in published luma.gl.** Use `RecipeBuilder` for resident GPU contributor chains. Use the published `GPUGraph` operation forms when a workflow needs CPU/GPU substitution. Pilot one mixed workflow with an existing CPU/GPU pair before promoting more recipes to operation form. A separate recipe scheduler would duplicate execution responsibilities without solving the observed typing, status or preflight defects. `dev-docs/contributor-contract.md:10`, `:91`; `S/recipes/recipe-utils.ts:80`, `:95`
4. **Add boundary invariants to the oracle suite.** Start with four cheap properties: the sign of an exact sum delta, nearest-segment distance bounded by endpoint distance, recorded tested-row eligibility matching the rows corrected, and rejected recipes preserving the prior command count. These complement reference-output fixtures and avoid copying implementation-specific behavior into the expected result. `T/cell-table-compare/cell-table-compare-oracle.ts:44`, `:114`; `T/track-similarity/cpu-track-similarity-operation.node.spec.ts:287`; `T/recipes/hot-spot-permutation-statistic.spec.ts:107`; `T/urban-ecology/building-tessellation-local.node.spec.ts:109`

## Top 5

1. **DGGS-01** — fix signed cell-table differences and replace the wrapping oracle.
2. **RCP-01** — align hot-spot permutation eligibility, correction and reported metadata.
3. **TRJ-01** — correct antimeridian point-to-segment projection in the CPU and shared WGSL paths.
4. **TRJ-02** — include zero in Brownian variance optimization.
5. **DGGS-02** — preserve arbitrary DGGS-JSON property names with a small serializer fix.

## Questions for the lead

- Can the geometry-measures stream own the shared WGSL half of **TRJ-01** and check its other callers? Track similarity supplies the counterexample and CPU regression.
- Can the analysis-core stream take **DGGS-03** and the mixed-graph review take **RCP-04**? Both have small, clearly located fixes that cross stream ownership.
- For cell API evolution, is an additive compatibility period acceptable for capability-specific family types and optional descriptors? That would let the roadmap distinguish missing operations from missing metadata without widening aggregation to families it cannot execute.