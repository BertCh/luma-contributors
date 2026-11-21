# Status and completeness across analysis families

## Summary

The most valuable pre-alpha change is to make completeness travel with a result. A caller should be able to pass a producer’s output directly to a consumer and retain the evidence that some input, candidate, or iteration was lost.

I found an executable counterexample. A CPU contiguity producer reports candidate loss on two adjacent squares. Passing that very object to weights algebra clears the flag, passing it to a traces calculation changes the second trace from `2` to `0`, and spreading it into a neighborhood summary produces ordinary-looking counts.

Two other failures warrant early fixes. A Fréchet threshold decision can return `false` with clean pair status when the decision was undefined. The documented raster-embedding composition drops precisely the fields that the repaired consumer needs to reject a partial polygon mean.

Of 498 public graph producers, 197 accept a named `output.status` object, 94 accept a raw status view, and 207 use another arrangement or have no destination with that spelling. Of the 291 conventional destinations, 99 are optional. Optionality is consequential when the ordinary output can also represent a valid answer.

A single physical status layout would buy little. A positive, scoped `complete` predicate, typed readback layouts, and direct-composition tests would address the failures while preserving existing buffers, shader codes, and useful partial results.

## Census

`S` below means `modules/analysis/src`, `P` means `S/gpu-spatial-analysis`, `N` means `S/gpu-network`, and `D` means `docs/api`. Paths and counts refer to the live `r19` checkout, sampled at `85a29fcc`. `KI` identifies a check of the newer, staging-owned copy in `luma-contributors-ki`. Executed graph probes constructed real command graphs with the null device. They establish host wiring and accepted inputs. Shader outcomes are marked **SOURCE**.

**COUNT.** `rg -il 'status' modules/analysis/src -g '*.ts' | wc -l` returned **843 files**. `rg -il 'overflow|truncat|converg|incomplete|exhaust' modules/analysis/src -g '*.ts' | wc -l` returned **848**. To count public classes, I followed the 11 `package.json` export entries to their source barrels with the installed TypeScript API, resolved aliases, deduplicated declaration symbols, and selected class instance types with `getCommandNodes`, including inherited methods. I inspected constructor or instance props and fell back to instance `output`. This recovered 498 producers and 488 selectable output properties. The top-level census is exhaustive for those declarations.

| Family | Named `output.status` | Raw `output.status` view | Other or absent | Producers |
|---|---:|---:|---:|---:|
| Analysis core | 2 | 0 | 4 | 6 |
| Computation record | 0 | 0 | 2 | 2 |
| Spatial analysis | 132 | 55 | 83 | 270 |
| Network | 23 | 4 | 14 | 41 |
| Terrain | 12 | 2 | 32 | 46 |
| Dataframe | 8 | 6 | 20 | 34 |
| Raster | 20 | 17 | 49 | 86 |
| Point cloud | 0 | 9 | 2 | 11 |
| Geospatial | 0 | 1 | 1 | 2 |
| Testing | 0 | 0 | 0 | 0 |
| **Total** | **197** | **94** | **207** | **498** |
| **Destination optional** | **93** | **6** | — | **99** |

**COUNT.** I also walked first-party nested output properties and unions, stopping at a view or array. The counts below are distinct public producers accepting a spelling. “Optional only” means every observed route to that spelling crossed an optional property. This measures the public type surface. Some composed ports accept more fields than an individual operation writes.

| Output spelling | Producers | Optional only | Meaning that needs to survive composition |
|---|---:|---:|---|
| `overflow` | 123 | 38 | Final output capacity |
| `requiredCount` | 94 | 82 | Required cardinality, with varying certainty |
| `candidateOverflow` | 79 | 65 | Intermediate search or candidate loss |
| `converged` | 79 | 66 | Iterative stopping criterion |
| `iterationLimitReached` | 65 | 60 | Budget exhausted |
| `invalidCount` | 65 | 61 | Rejected inputs or remaining invalid outputs, depending on producer |
| `iterationCount` | 54 | 48 | Work actually executed |
| `approximation` | 42 | 40 | Approximation used |
| `uncertainCount` | 31 | 31 | Includes both resolved and unresolved cases in recipe adapters |
| `complete` | 4 | 3 | Positive predicate |
| `incomplete` | 2 | 2 | Negative predicate |
| `candidateTruncated` | 5 | 5 | Candidate loss, alongside the more common `candidateOverflow` |
| `unresolvedCount` | 4 | — | Work still unresolved |
| `issueCode` | 2 | 1 | Operation-specific code |
| `iterationCounts`, `medianConverged`, `reason` | 1 each | — | Specialized iteration or decision evidence |
| Plain `truncated`, plain `iterations` | 0 | — | Neither is a conventional output field |

There were **304** producers with a property named `status` somewhere in the traversed output, versus **291** at its top level. Other examples outside this traversal include `GPUNetworkAccessibility`’s top-level destination, packed summary words, and `GPUNetworkAnalyticsColumns`’ component status. The 207 in the first table must not be counted as 207 silent failures.

**Words, bits, and decoders.** Filtering unique public symbols by `/^[A-Z0-9_]+$/` and `/STATUS|FLAG|ISSUE/` returned **114** names, including layout constants. Exactly **20** objects end in `_STATUS`. Five are **word-index maps**, six are **bit-valued reason maps**, and nine are **code maps**. The five index maps are named in STA-13. The six bit-valued maps belong to track similarity, trajectory Kalman, trajectory restriction, raster focal pairs, raster fractional zonal distribution, and datum-grid transform. The nine code maps belong to line locate, local variability, instrumental-variables covariance, GWR, iterative regression, MGWR, trajectory playhead, maximum inscribed circle, and terrain peak snap. The suffix alone does not say how to read the number.

There are **21** exported `decode…` names. Six explicitly end in `Status`—the four spherical operations, GWR, and MGWR. Three decode geometry-validity, segment-snap, or population-alignment issues. Ten decode results or counts that can carry evidence—spacing, contiguity, region statistics, space-time inference metadata, K-shortest paths, network coarsening, network statistics, subgraph filter counts, and two spherical count layouts. The other two decode horizon values and narrowing error. There is no shared status decoder. Eight of the nine point-cloud status publishers export named status values or bits. `GPUPointCloudNormalScale` is the exception. Even within point clouds, bit `4` can mean unsupported neighborhood or failed feature eigensolve. SMRF’s inpainted bit records provenance. Spherical containment code `2` is a valid boundary result. A universal `word === 0` test would misclassify real data.

**CPU, recipes, catalog, and docs.**

| Surface and census command or selection | Count and result |
|---|---|
| Unique exported functions matching `/^compute.*OnCPU$/`, classified by return type | 131. Flat status terms only 51, raw `.status` only 42, raw plus flat 4, named `.status` only 5, named plus flat 1, other 28. Flat terms included overflow, convergence, limits, truncation, completeness, approximation, invalid count, and model-specific status names. |
| Qualified exports matching `/^add.*Recipe$/`, first call signature | 44. Staged status 30, custom bag 7, raw view 2, no `.status` 5. Overloads can also differ. |
| `getGPUAnalysisOperatorCatalog()` | 557 records. Status unaudited/unknown 222, audited empty scalar list 136, one or more catalog kinds 199. `overflow` appears in 131 records, `candidateOverflow` in 72, `converged` in 39, `iterationLimitReached` in 19, `iterationCount` in 11. |
| Search `D` for the 291 classes with top-level status, then inspect exact-name heading sections | All 291 names occur somewhere. 274 had an exact-name heading token. For 234 of the 291, its section mentioned status, overflow, convergence, truncation, incompleteness, or issue code. A word match does not establish scope, polarity, or a usable readback example. |

## Proposed convention

Every public producer that can return a plausible partial answer must expose retrievable `status.complete` evidence for the promised result at a declared scope—row, query, stage, or batch—with distinct causes for final `overflow`, `candidateOverflow`, invalid input, and an iteration limit. Authors may make the caller’s destination optional when the result still exposes equivalent evidence, and consumers must forward supplied upstream loss, suppress the dependent result, or reject the input. Check this with a typed `decodeGPUAnalysisStatus(layout, readback, row?)` returning `state: 'complete' | 'incomplete' | 'unknown'` and scoped causes, plus paired clean/loss fixtures that pass the producer result directly to each consumer.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| STA-01 | W | ORDINARY | ADDITIVE | M | CPU spatial weights | Direct composition erases a reported loss and changes downstream numbers. |
| STA-02 | W | ORDINARY | ADDITIVE | S | Track similarity | An undefined Fréchet threshold decision looks like a clean `false`. |
| STA-03 | W | ORDINARY | ADDITIVE | S | Raster embedding docs | Following the page bypasses the repaired partial-coverage guard. |
| STA-04 | W | ORDINARY | ADDITIVE | M | Point-cloud neighborhoods → features | A feature result cannot carry its upstream halo failure. |
| STA-05 | W | ORDINARY | ADDITIVE | S | Dasymetric recipe | A valid source with no eligible recipients is reported as invalid. |
| STA-06 | W | ORDINARY | ADDITIVE | S | Spacing statistics | The recommended one-read decoder discards the invalid-input count. |
| STA-07 | W | UNUSUAL | ADDITIVE | S | Spherical, contiguity, network decoders | Unknown words or empty evidence can decode as successful. |
| STA-08 | C | ORDINARY | ADDITIVE | S | Drainage recipe | The same field describes both resolved and unresolved work. |
| STA-09 | E | ORDINARY | ADDITIVE | S | Visibility CSR → capacity planner | An explicitly unknown count yields a 32–40 GiB retry plan. |
| STA-10 | C | ORDINARY | ADDITIVE | S | Track-quality recipe | Enabling similarity leaves its loss evidence out of the stage list. |
| STA-11 | E | ORDINARY | ADDITIVE | M | K-means and Sinkhorn | A budgeted, usable iterate can have no publicly retrievable stopping flag. |
| STA-12 | C | ORDINARY | ADDITIVE | S | Core port and catalog | The contributor contract names a predicate the port and catalog cannot express. |
| STA-13 | C | ORDINARY | BREAKING | S | Five public layout constants | `_STATUS` denotes a word address in five APIs and a word value in fifteen. |
| STA-14 | C | EXTREME | ADDITIVE | M | Network travel-time percentiles | A reached, unrepresentable finite-path cost shares the unreachable sentinel. |

### STA-01: A CPU weights result loses its warning when passed intact

**Where.** `P/spatial-weights/cpu-contiguity-weights-operation.ts:184`, `P/spatial-weights/cpu-spatial-weights-algebra-operation.ts:34,295`, `P/spatial-regression/cpu-spatial-weights-traces-operation.ts:96`, `P/spatial-regression/cpu-regression-weights.ts:12`, `P/neighborhood-summary/cpu-neighborhood-summary-operation.ts:114`. Checked the staging-owned algebra, traces, regression reader, and summary in KI too. Its nested loss check and renamed summary input retain the mismatch.

**Evidence — EXECUTED.** Two adjacent unit squares, rook contiguity, ring offsets `[0,4,8]`, final capacity `4`, pair capacity `1`: the producer returned offsets `[0,1,1]`, `overflow=0`, `candidateOverflow=1`. Raising pair capacity to `32` supplied the complete control.

| Consumer invocation | Direct bounded result | Control |
|---|---|---|
| `computeSpatialWeightsAlgebraOnCPU({weights: w}, {operation: 'selfWeight', selfWeight: 1, capacity: 6})` | Output loss flags `[0,0]` | `{weights: {...w, status: w}}` preserved candidate loss `1` |
| `computeSpatialWeightsTracesOnCPU({weights: w}, {mode: 'exact', maximumPower: 2})` | Power traces `[0,0]` | Complete weights gave `[0,2]` |
| `computeNeighborhoodSummaryOnCPU({...w, values: [10,20]}, {statistics: ['count']})` | `[1,0]` | Explicit `weightsCandidateOverflow` gave `[NaN,NaN]`. Complete weights gave `[1,1]`. |

**Now.** `weights: w` preserves the CSR arrays but loses the meaning of the result. This adds a direct CPU producer-to-consumer reproducer to `KNOWN(PTS-101)` and `KNOWN(SSR-102)`. The producer did report loss, and the caller passed the object through.

**Wanted.** Publish a nested `status` alias on CPU weights results. Have the shared CPU reader accept the existing flat fields, the nested fields, and legacy prefixed fields where applicable. Preserve either positive loss signal when aliases coexist. Consume clean concrete CPU flags normally. Test algebra-to-algebra as well as the three paths above.

**In-repository callers that change.** None required with aliases. **Value and cost.** GIS callers can compose published operations without silently analyzing a different graph. M.

### STA-02: An undefined Fréchet decision is indistinguishable from a valid rejection

**Where.** `P/track-similarity/cpu-track-similarity-operation.ts:382,465`, `P/track-similarity/gpu-track-similarity.ts:146,214,587`. Checked both KI copies, including the decision at CPU line 488 and graph status allocation at line 620.

**Evidence — EXECUTED**, both copies. I compared a two-vertex line `[(0,0),(1,y)]` with itself. Tuples below are `[frechetWithin, pairStatus, frechet]`.

| Input | Result |
|---|---|
| `y=0`, epsilon `1`, cap `2` | `[1,0,0]` |
| `y=NaN`, epsilon `1`, cap `2` | `[0,0,NaN]` |
| `y=0`, epsilon `NaN`, cap `2` | `[0,0,0]` |
| `y=0`, epsilon `1`, cap `1` | `[0,2,NaN]` |

**Now.** `if (!frechetWithin[i]) rejectPair(i)` rejects a pair even when its distance is zero and the threshold was unusable. Requesting pair status still reports clean in the second and third cases. On the graph route `pairStatus` itself is optional.

**Wanted.** Expose a scoped `decisionValidity` or equivalent decoded cause. Name `invalidDecisionThreshold` separately from `invalidGeometry` and `frechetCapExceeded`. A bad threshold affects the decision even when the distance remains usable. **SOURCE GPU probe:** repeat the finite self-line with cap `1`, cap `2`, and `NaN` epsilon under cap `2`, reading decision and pair evidence together. Nonfinite-coordinate results above are CPU observations.

**In-repository callers that change.** None for an added validity destination and decoder. **Value and cost.** A threshold search can distinguish “too far” from “comparison unavailable.” S.

### STA-03: The embedding page teaches a composition that drops the coverage guard

**Where.** `D/gpu-raster/raster-embeddings.md:185–193`, KI `:196–204`; `S/raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean.ts:48,69,296,398`. This is a documentation departure remaining after `KNOWN(RT-126)`.

**Now.** The page names four fields. Written from that description, the call is:

```ts
tuples: {
  zoneIds: coverage.zoneIds,
  cellIds: coverage.cellIds,
  fractions: coverage.fractions,
  status: {count: coverage.status.count}
}
```

**Evidence — EXECUTED.** This projection and `tuples: coverage` both constructed six-node zone-mean graphs. The first omits `requiredCount`, `overflow`, `candidateOverflow`, and `zoneValidity`. The consumer can use each of those when supplied. The newer docs still give the four-field list.

**Wanted.** Show `tuples: coverage` on the page. Explain that `status.count` alone is sufficient for externally certified complete tuples. **SOURCE GPU probe:** one zone, two covered pixels with different unit vectors, capacity for one tuple. Compare the projected input with the whole output. The former can publish the retained pixel’s direction. The latter should suppress the incomplete mean.

**In-repository callers that change.** None. **Value and cost.** A user gets the repaired behavior with one line instead of reconstructing four fields. S.

### STA-04: A point-cloud feature cannot carry its neighborhood’s halo warning

**Where.** `S/pointcloud-analysis/operations/gpu-pointcloud-neighborhoods.ts:99,279,565` and KI `:564`, `gpu-pointcloud-features.ts:67,76,261`; `D/pointcloud-analysis.md:98–102,843–845`, with the separate-read instruction also present in KI at `:851`.

**Now.** The documented pipeline passes `positions, indices, counts, capacity` into `GPUPointCloudFeatures`. It retains the neighborhood and feature statuses as separate outputs. Counts can reveal slot truncation. They cannot certify that a query sphere stayed inside the populated halo. The page tells an expert to read both. A downstream recipient of the features has no input or output field carrying that relationship.

**Evidence — SOURCE.** Neighborhood bit `4` means unsupported query extent. Feature bit `4` means eigensolve did not converge. OR-ing the words would create a false diagnosis. **Smallest GPU probe:** four coplanar points on a unit square, radius `2`, capacity `4`. Run once with generous certified bounds and once with bounds just enclosing the square. Read neighborhood and feature status for the same query; the local point population and slot counts can stay identical while only the halo certification changes.

**Wanted.** Add optional `neighborhoodStatus` to feature inputs and a distinct `INCOMPLETE_SUPPORT` feature cause. Accept it on the direct neighborhoods-to-features path and retain its row alignment through chunking.

**In-repository callers that change.** None required. Update the point-cloud example to use it. **Value and cost.** Tiled LiDAR consumers can distinguish a well-conditioned fit on incomplete support from a certified feature. M.

### STA-05: The dasymetric stage labels retained mass as invalid input

**Where.** `P/recipes/dasymetric-redistribution-recipe.ts:454,477,583,593`, `D/gpu-spatial-analysis/dasymetric-redistribution.md:58,76`, `modules/kepler/src/curated/bindings/curated-dasymetric-redistribution-binding.ts:416,509`.

**Evidence — EXECUTED.** Constructing the graph with two cells and one source zone gave both identities: `conservationStage.status.invalidCount === result.status.zeroEligibleCount` and `conservationStage.status.converged === result.status.complete`. Coverage validation likewise exposes `coverageComplete` as `converged`. The Kepler binding publishes the metric names to users.

**Now.** A generic consumer reading `stage.status.invalidCount` treats “no eligible recipient” as rejected input. **SOURCE GPU probe:** give a valid covered source total `7` and zero ancillary weights. Check retained mass `7`, conservation complete `1`, zero-eligible count `1`, then read that same `1` under the stage label `invalidCount`.

**Wanted.** Expose `complete` on the validation and conservation stages and `zeroEligibleCount` on conservation. Switch the Kepler evidence table to those fields. Preserve existing aliases for compatibility while consumers migrate.

**In-repository callers that change.** No caller is forced to change by the addition. Two explicit test expectations identify the migration points: `modules/analysis/test/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe.node.spec.ts:128` and `modules/kepler/test/curated/curated-dasymetric-redistribution-binding.spec.ts:278`. **Value and cost.** A valid decision to retain unallocated mass stops showing up as a data-quality error. S.

### STA-06: The spacing decoder erases evidence already in the readback

**Where.** `P/analysis-spacing/gpu-analysis-spacing.ts:47,177,229,247`, `analysis-spacing-statistics-nodes.ts:130,173`, `P/ontology/analysis-experience/analysis-spacing-statistics.ts:249,367`, `modules/kepler/src/profile/dataset-profile-gpu.ts:317`, `D/gpu-spatial-analysis/analysis-experience.md:39`.

**Evidence — EXECUTED.** I decoded two 544-word version-one summaries with words `0`, `5`, and `7` set to `1`. Changing only word `1`, the documented `invalidCount`, from `0` to `1` gave identical results: `pointCount: 1`, `sampleSize: 1`, `sampled: false`. The CPU statistics function likewise returned identical JSON for planar `[0,0]` and `[0,0,NaN,0]`. A zero accepted count produced `undefined`.

**Now.** `decodeGPUAnalysisSpacingStatistics(words)` is the advertised one-read route. Its consumer cannot distinguish all input rows accepted from half the input skipped. The packed bytes still contain the answer.

**Wanted.** Expose `invalidCount` in decoded statistics and in the CPU result. Add a detailed decode result that can distinguish empty input, all-invalid input, and loss even when there are no statistics to return. Retain the convenience decoder.

**In-repository callers that change.** None required. The Kepler profile already spreads the decoded object. **Value and cost.** Dataset profiling can disclose its effective population with no additional GPU read. S.

### STA-07: Public readback decoders can certify unknown or missing evidence

**Where.** `P/global-coordinates/spherical-status.ts:5,14,51,56`, `P/spatial-weights/gpu-analysis-contiguity-statistics.ts:103–113`, `D/gpu-spatial-analysis/gpu-analysis-contiguity-statistics.md:18`, `N/network-statistics/gpu-network-statistics.ts:77,152,169,175,312`.

**Evidence — EXECUTED.**

| Decoder input | Actual result | Expected |
|---|---|---|
| Both spherical containment decoders, word `4` or `NaN` | `{valid:true, invalidInput:false, boundary:false}` | Unknown or malformed; documented values are `0`, `1`, `2` |
| `decodeGPUSphericalArcOverlayCounts(new Uint32Array(0))` | Three undefined counts and `referenceInside:true` | Short-read error |
| Contiguity counts `[1,1,0,1,1,0]`, median `[1]`, validity `[0xffffffff]`, both overflow flags zero | `exact:true` | The page specifies validity `1` for this certification |
| `decodeGPUNetworkStatistics(new Uint32Array(), {degreeBinCount: NaN})` | `componentsConverged:true`, `modularityValid:true`, `modularity:0` | Layout error; graph constructor accepts integer `1…4096` |
| Same network input with `degreeBinCount: -6` | Same two success flags | Layout error |

**Now.** `if (decoded.componentsConverged)` can succeed without reading a single word. Unknown enum values also become ordinary valid rows.

**Wanted.** Validate layout parameters before computing readback length, enforce minimum word counts, and validate enum membership or return `unknown` with the raw word. Enforce producer-specific validity semantics. Keep valid spherical boundary code `2` distinct from invalid input.

**In-repository callers that change.** None for documented valid inputs. **Value and cost.** Readback mistakes fail at the decoder before they become plausible geometry or network statistics. S.

### STA-08: Drainage uses a resolved-work counter for unresolved work

**Where.** `S/analysis-core/contracts/status.ts:25–28`, `P/recipes/drainage-network-recipe.ts:113,190,196`.

**Evidence — EXECUTED.** In a two-cell, one-step graph the `downstream-distance` stage’s `uncertainCount` was the `distance-unresolved` view. The `stream-network` stage’s `uncertainCount` was its `unresolved` view. The core port defines `uncertainCount` as the exact number resolved through fallback.

**Now.** `stage.status.uncertainCount` can count a successful recovery in one recipe and remaining uncertainty in another. The appropriate action differs.

**Wanted.** Add `unresolvedCount` to the stage port and publish it for these two stages. Reserve `uncertainCount` for its documented fallback meaning, with a clearer `resolvedByFallbackCount` alias if needed. Four producers already use `unresolvedCount` by name.

**In-repository callers that change.** None for the additional port. **Value and cost.** A drainage report can distinguish “used a fallback” from “increase the budget or mark these rows unknown.” S.

### STA-09: Unknown visibility-graph cardinality becomes an enormous allocation plan

**Where.** `P/urban-morphology/visibility-graph-csr.ts:78,564`, `P/spatial-weights/spatial-weights.ts:348,367,374`; KI planner `:377,394`.

**Evidence — EXECUTED.** For `{rowCount:2, capacity:4, requiredCount:0xffffffff}`, the R19 capacity helper proposed **5,368,709,119 slots, 42,949,672,964 bytes, `limited:false`**. KI caps the slots at uint32 maximum but still proposed **4,294,967,295 slots, 34,359,738,372 bytes, `limited:false`**. No allocation was attempted.

**Now.** The generic retry instructions pass the read-back count to `getGPUSpatialWeightsCapacityPlan`. Visibility CSR documents that very value as **unknown** when `candidateOverflow` is set. It cannot tell the caller how many final slots to buy. **SOURCE GPU probe:** use a visibility query that exhausts its candidate traversal budget, then read `candidateOverflow` and `requiredCount` together. Compare with the same query at a sufficient budget.

**Wanted.** Accept a decoded requirement such as `{kind:'unknown'}`, `{kind:'lower-bound', value}`, `{kind:'exact', value}`, or `{kind:'saturated', value}`. On an unknown visibility requirement, report that candidate coverage must be retried before sizing final storage. Preserve the numeric overload for callers with a known count. The decoder has to inspect the producer and candidate flag; the numeric maximum alone is insufficient.

**In-repository callers that change.** None for the overload. **Value and cost.** A caller following the retry loop avoids an unusable multi-gigabyte request. S.

### STA-10: Track quality omits enabled similarity from its stage evidence

**Where.** `N/recipes/track-quality-recipe.ts:202,227,235,272,290`, `P/track-similarity/gpu-track-similarity.ts:146` and its KI copy.

**Evidence — EXECUTED.** Enabling similarity for one pair, without map matching, returned `status.stages.length === 0` and `outputs.similarity.pairStatus.length === 1`. The recipe has already allocated the evidence.

**Now.** `result.status.stages.every(isAcceptable)` is vacuously true. The reader needs to know to leave the recipe status API and inspect individual output columns. **SOURCE GPU probe:** compare two three-vertex tracks at a two-vertex Fréchet cap. Read the pair’s cap flag and confirm that the stage list still has no similarity entry.

**Wanted.** Add a `similarity` stage with a per-pair layout pointing at the existing view. Identify which measures each cause affects; a Fréchet cap need not invalidate Hausdorff distance.

**In-repository callers that change.** None must supply a new buffer. Stage consumers will see an additional stage that actually ran. **Value and cost.** A recipe inspector can discover every enabled stage with no extra dispatch. S.

### STA-11: A finite budgeted iterate can outlive its only stopping evidence

**Where.** `P/spatial-clustering/gpu-spatial-kmeans.ts:174,1435`, `S/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts:29,93`, `P/optimal-transport/gpu-sinkhorn-divergence.ts:87,100` and `cpu-sinkhorn-divergence-operation.ts:248,258`. This adds three concrete first-use paths to `KNOWN(CORE-110)`.

**Evidence — EXECUTED.** At budget one, the three graphs accepted no public status destination and built **18, 12, and 16 nodes** respectively. For CPU Sinkhorn, source masses `[9,1]`, target masses `[1,9]`, two cells and one iteration gave finite divergence `0.6839309930801392`, marginal error `0.7090250849723816`, `converged=0`, and `iterationLimitReached=1`.

**Now.** Reading just the divergence or clusters can produce the ordinary result shape. Iteration count alone cannot reconstruct a solver’s stopping decision: an algorithm can first meet tolerance on its last permitted iteration.

**Wanted.** When a caller omits stopping destinations, expose graph-owned stopping evidence through the result. Give each algorithm a descriptor for its own stopping rule and keep the current iterate available. **SOURCE GPU probe:** repeat the two-cell Sinkhorn fixture with and without explicit destinations. Compare the divergence and read the stopping flags on the explicit route. For each Lloyd implementation, use one round on a fixture whose assignments or centers change.

**In-repository callers that change.** None for exposing the fallback views. **Value and cost.** A first-time user can distinguish “usable estimate” from “met tolerance” after constructing the graph. M.

### STA-12: The contributor contract’s `complete` predicate is unrepresentable in the shared catalog

**Where.** `dev-docs/contributor-contract.md:117`, `S/analysis-core/contracts/status.ts:69,106`, `P/ontology/spatial-analysis-ontology.ts:270`, `P/ontology/all-entry-points/operator-status-audits.ts:79`, `P/urban-morphology/building-adjacency.ts:324`.

**Evidence — EXECUTED and COUNT.** `queryGPUAnalysisOperators({status:'complete'})` returned zero records in JavaScript. The literal is outside the TypeScript status-kind union. The audit says “status.complete only, not a catalog kind” for a publisher whose audited scalar list is empty. The shared bounded-result and recipe-stage types lack `complete`. This is an expressiveness gap beyond the incorrect registrations in `KNOWN(CAT-101)`.

**Now.** An author following “use `status.complete`” cannot register that capability, and a user searching for the canonical predicate cannot find it.

**Wanted.** Add `GPUCompletenessStatusPort` with `complete`, include it in the stage and bounded-result types, add the catalog kind and scope, and update known publishers. Check that an audited `complete`-only publisher is discoverable and distinct from an audited publisher with no scalar evidence.

**In-repository callers that change.** None for existing queries. **Value and cost.** The contributor contract, public types, and operator finder acquire the same meaning. S.

### STA-13: Five `_STATUS` constants are indexes in a buffer

**Where.** `P/location-allocation/gpu-location-allocation.ts:60`; under `P/spatial-regionalization`, `gpu-spatial-weights-minimum-spanning-tree.ts:56`, `gpu-azp-regions.ts:139`, `gpu-ward-regions.ts:86`, `gpu-max-p-regions.ts:176`. Also checked the three staging-owned regionalization constants in KI at `:143`, `:88`, and `:189`.

**Evidence — COUNT and SOURCE.** `GPU_LOCATION_ALLOCATION_STATUS.converged === 5` means **read word five**. `GPU_MULTISCALE_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.CONVERGED === 0` means **compare a word with zero**. Both are public numeric members. This is independent of which regression reason code wins the `KNOWN(SSR-112)` decision.

**Now.** Both `words[STATUS.converged]` and `word === STATUS.CONVERGED` look reasonable to a newcomer; TypeScript represents both operands as numbers.

**Wanted.** Rename the five index maps to `GPU_LOCATION_ALLOCATION_STATUS_LAYOUT`, `GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT`, `GPU_AZP_STATUS_LAYOUT`, `GPU_WARD_STATUS_LAYOUT`, and `GPU_MAX_P_STATUS_LAYOUT`. Keep storage order fixed. Offer aliases during migration, then retire the old names before alpha.

**In-repository callers that change.** `rg -o '\bGPU_(LOCATION_ALLOCATION|MINIMUM_SPANNING_TREE|AZP|WARD|MAX_P)_STATUS\s*\.' modules examples website scripts` over JS/TS sources found **265 explicit member references**: 56 location allocation, 23 MST, 50 AZP, 49 Ward, 87 Max-P. **47 files** mention the identifiers, including their declarations, tests, and site uses. **Value and cost.** An author can distinguish a layout from a value at the call site. S, mechanical breaking rename.

### STA-14: A reached travel time can be infinity with clean departure status

**Where.** `N/network-optimization/cpu-network-time-routing-operation.ts:218,231`, `N/network-time-geography/cpu-network-travel-time-percentiles-operation.ts:74,130–135`, `D/gpu-network/network-time-geography-mob09.md:32,45`, also checked in KI.

**Evidence — EXECUTED.** On `0→1→2`, both edge base times `3e38`, zero slopes, departure zero, percentile 50, I got `travelTimes=[0,3.0000000054977558e38,Infinity]`, `reachedCounts=[1,1,1]`, departure status `[1,1]`. Replacing the edge costs with `7` gave final cost `14` with the same flags. The routing summary for the large-cost run was `[1,1,2,2139095040]`, `invalidCount=0`, `converged=1`.

**Now.** `travelTimes[i] === Infinity` commonly means unreachable, yet the reached count says this destination was reached. The CPU source explicitly documents the float32 behavior. The gap is the absence of a representability cause in the public result.

**Wanted.** Add a per-result `unrepresentable` cause or a scoped count with an optional row mask. Preserve reachability, convergence, and the existing float32 payload as separate facts.

**In-repository callers that change.** None required. **Value and cost.** A network consumer can distinguish disconnected destinations from reached destinations whose costs exceed the output format. M.

## Departures by name

The following names omit the common `GPU` prefix. Optionality here describes declarations. It does not prove that every optional status is the sole evidence of loss.

**All six optional raw `output.status` views:** `SpatialWeightsMinimumSpanningTree`, `NetworkTravelTimePercentiles`, `SpaceTimeKFunction`, `SpaceTimeDiggleTest`, `RasterEmbeddingKMeans`, `PointCloudNeighborhoods`. Point-cloud bounds can make evidence required on a particular route. Travel-time percentiles also offers reached counts. These are different guarantees behind the same `?`.

**All 93 optional named `output.status` destinations:**

| Family | Public producers |
|---|---|
| Spatial | Pycnophylactic, LocationChangePoints, RankMarkov, TransitionMatrix, TransitionDiagnostics, SpatialMarkov, LISAMarkov, DotDensity, RandomPointsInPolygon, GeometryMeasures, GeodesicPairs, GeodesicDestination, GeometryPredicates, GeometryValidity, GlobalSpatialStatistics, GroupGeometry, LineClipByPolygon, LineSimplification, LineSplit, MapColoring, MinimumClearance, NeighborhoodSummary, GlobalPermutationTest, PointDensity, KernelDensity, SpaceTimeKernelDensity, SpaceTimeRelativeRisk, CoverageValidity, EmpiricalBayesRates, RegionStatistics, SpatialScanStatistic, SinkhornDivergence, SpaceTimeHawkes, SegmentIntersection, HotSpotAnalysis, LocalMoran, LocalGeary, AssuncaoReisStandardization, SpatialKMeans, HierarchicalDensityClustering, Kriging, BufferSelection, SpatialPredicateJoin, RedcapSingleLinkage, GeographicallyWeightedRegression, GeographicallyWeightedRegressionNonstationarityTest, SpatialWeightsTraces, GeographicallyTemporallyWeightedRegression, MultiscaleGeographicallyTemporallyWeightedRegression, SpatialWeightsTranspose, MobilityIndicators, TrajectoryRecursions, ZoneEvents, ZonalStatistics, SegmentSnap |
| Network | NetworkFlow, NetworkLouvain, NetworkTimeRouting, NetworkFacilityOptimization, NetworkRationalAccess, MapMatching, NetworkSnapping, NetworkCostMatrix, NetworkServiceAreas, NetworkCentrality, NetworkCentralityScheduleStep, NetworkCostTable, NetworkCostWeights, NetworkIsochrones, NetworkKFunction, NetworkNoding, NetworkReachability |
| Terrain | TerrainCatchmentSummary, TerrainFlow, TerrainHeightAboveDrainage, TerrainWatersheds, TerrainStreamOrder, TerrainSpikeRepair, TerrainSummits, TerrainPeakSnap, ProfilePeaks |
| Dataframe | Variogram, SpatialCorrelogram, PeriodIndex, CellPeriodKeys, ValueCube |
| Raster | CostDistance, CostDistancePath, Streamlines, Isobands, RasterJoin, RasterZonalStatistics, RasterSpectralFilter |

**Distinct status spellings that affect how a user reads a result:**

| Spelling | Public producers |
|---|---|
| `complete` | AlternativeRoutes requires it. NetworkCostTable, NetworkCostWeights, NetworkKernelDensity have an optional route to it. |
| `incomplete` | RasterJoin, RasterZonalStatistics |
| `candidateTruncated` | PointInPolygonJoin, NearestFeatureJoin, BufferSelection, SpatialPredicateJoin, ZonalStatistics. SpatialPredicateJoin also has `candidateTruncatedCount`. |
| `unresolvedCount` | NetworkReachability, TerrainBranchingFlow, TerrainBreach, TerrainStreamNetwork |
| `statuses`, `reason`, `droppedCount`, `issueCode` | LineLocate; OPTICS; CellCompaction; GeometryValidity and SegmentSnap, respectively |
| `conditionalStatus`, `siteStatus` | VecchiaLogLikelihood and VecchiaSpaceTimeLogLikelihood expose both. VecchiaFisherScoring exposes site status. |
| `localStatus`, `componentSearchIterationLimitReached` | GeographicallyWeightedRegression, GeographicallyTemporallyWeightedRegression, MultiscaleGeographicallyTemporallyWeightedRegression use `localStatus`. MultiscaleGeographicallyWeightedRegression uses the component-search limit. |
| Other per-result status names | FuzzyContiguity `featureStatus`, TrackSimilarity `pairStatus`, TrajectoryPlayhead `trackStatus`, TrajectoryRestriction `compactStatus`, SpatialCrossValidatedOLS `fitStatus` and `metricStatus`, TerrainBreach `pitStatus`, TerrainCatchmentSummary `profileStatus`, TerrainPeakSnap `snapStatus` |
| Scoped overflow names | CellAggregation, CellRollup, CellTableCompare `sumOverflow`; GroupConvexHull and MinimumBounds `groupOverflow`; GroupConvexHull `vertexOverflow`; TrajectoryClosestApproach `entryOverflow` and `hitOverflow`; ZoneEvents `trackOverflow` and `eventOverflow`; FlowAggregation and MapMatchedEdgeVolumes `pairOverflow`; MapMatching `routeOverflow`; Isolines `polylineOverflow`; Isobands and IsobandRings `edgeOverflow` |
| Scoped convergence names | GeographicDistribution `medianConverged`; TerrainFlow `fillConverged`, `flatsConverged`, `accumulationConverged` and matching iteration counts; RasterSpectralUnmixing `iterationCounts` |

**Recipes outside the prevailing stage-list shape.** Five first signatures lack `.status`: StreetProfiles, Isovists, LandscapeMetrics, SegregationDecomposition, SegregationCounterfactual. Two return a raw view: ResistanceConnectivity and SparseResistanceConnectivity. Seven use custom bags: BuildingAdjacency, BuildingTessellation, BuildingTessellationReference, LocalBuildingTessellation, PatchProximity, and the spatial and network VisibilityGraph recipes. Both visibility recipes also have a packed route. PatchProximity has another overload with no status. These are the places to register layouts first; differences in result scope can justify differences in physical storage.

## Quick wins

- [ ] Change the zone-mean page to `tuples: coverage`. Show a full two-operation example that preserves coverage loss and zone validity.
- [ ] Expose spacing `invalidCount` in both CPU and decoded GPU results. Test empty input, one invalid point, and mixed valid/invalid points.
- [ ] Add malformed-word and short-buffer fixtures for the decoders in STA-07. Keep valid boundary and non-error provenance values in the positive fixtures.
- [ ] Add `complete` and `unresolvedCount` to core status ports and the catalog. Correct the drainage and dasymetric stage labels with aliases.
- [ ] Include the already allocated pair-status view in the track-quality stage list. Document its per-measure scope.
- [ ] Give `GPUPointCloudNormalScale` named status bits alongside the other eight point-cloud publishers. Show why neighborhoods and features require different layouts.
- [ ] Teach the visibility-CSR retry example to check candidate loss before asking for a final capacity.

## Larger proposals

1. **Fix the direct result handoffs first.** Add compatible CPU weights aliases and readers, then the point-cloud neighborhood-status input. A regression fixture should pass the producer object unchanged. Assert the downstream numerical result **and** the surviving cause. STA-01 demonstrates why tests that hand-build a correctly named status bag miss the failure.
2. **Add a shared decoder interface while preserving operation-specific layouts.** A descriptor needs physical encoding (`named`, `words`, `enum`, or `bits`), result scope, units for counts, recognized values or bits, and the outputs each cause affects. Preserve raw words for inspection. Missing evidence yields `unknown`. Keep valid boundary, approximation, fallback use, and provenance independently readable. This is additive and compatible with per-model decoders under `KNOWN(SSR-112)`.
3. **Make retrievable evidence the requirement.** For a producer with a plausible bounded answer, reuse an existing status view or expose a graph-owned one when destinations are omitted. Keep data and status attached through recipes. An unconditional requirement for caller-allocated status on all 99 optional destinations would add unnecessary allocations and boilerplate, including on routes with equivalent evidence.
4. **Take the five layout renames before alpha.** The 265 member references make this a bounded mechanical change. I recommend the breaking rename in STA-13. The evidence does not justify renumbering every family’s shader codes.

## Not read

I traversed public declarations across all families and read the shared status contract, spherical decoder file, and selected result/decoder implementations in full. I traced the relevant producer, consumer, CPU, recipe, documentation, and adapter paths for the findings above. I checked the prior synthesis, relevant area reports, API/catalog and direction reports, decision register, and the newer copies of staging-owned finding paths.

The declaration census covers all 498 public graph producers. I did not establish the runtime meaning of every shader literal, every overload, or every optional binding on all 498, and the docs-section count is a search measure rather than a page-by-page correctness audit. In particular, the 222 unknown catalog records cannot support a claim that their operators have no loss semantics. I did not run a GPU, browser, build, or test suite. The CPU values and host graph identities are executed evidence; the identified shader cases still need the small GPU probes given above.