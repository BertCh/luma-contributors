# CPL — Loss that does not travel

## Summary

A valid row, a converged solver and a complete answer are three different claims. Several public compositions preserve the first two while losing the third.

I reproduced a change in Vecchia log likelihood from **−2.555136 to −2.842278** by dropping one conditioning edge. The spatial likelihood, space-time likelihood and Fisher scorer all reported two usable sites and zero failures.

On a three-cell raster, an exhausted watershed produced an unresolved middle cell. Region analysis turned that into **zero shared boundary** and finished successfully. On a three-node road, a short routing budget changed **one served demand into one unserved demand**, with the same successful equilibrium summary.

The most valuable change is to let callers pass a **producer result with its evidence** wherever today they extract `offsets`, `count`, `costs` or `labels`. Start with the named `outputWeights` handle and the documented producer-to-consumer examples. Both are places where a careful caller can follow the API and still lose evidence.

## Census

I scanned `modules/analysis/src/**/*.ts` in r19, then repeated the scan substituting the KI file for each path in `.typecheck/r19/staging-owned.txt`. KI below means `/Users/robertchristie/Documents/GitHub/luma-contributors-ki`. Paths in findings are relative to `modules/analysis/src` unless they start with `docs/`, `modules/` or `website/`. Line numbers are r19 unless marked KI.

| Mechanical measure | r19 | With listed KI files |
|---|---:|---:|
| TypeScript files scanned | 1,817 | 1,817 |
| `^export (?:abstract )?class GPU\w+` declarations | 633 | 633 |
| Files containing `getGPUSpatialWeightsLossViews` | 9 | 12 |
| Files containing `candidateOverflow` | 222 | 259 |
| Files containing `costsComplete` | 8 | 8 |
| Exported-class files with `weights?:` or `weights:` typed directly as `GPUSpatialWeights` or `GPUChunkedSpatialWeights` | 53 | 52 |

The reproducible file-count command for an individual token is `rg -l 'getGPUSpatialWeightsLossViews' modules/analysis/src -g '*.ts' | wc -l`. For the class counts I applied the declaration regex above per file. For the weights row I intersected files containing `^export class ` with `\bweights\??:\s*(?:GPUSpatialWeights|GPUChunkedSpatialWeights)\b`. These are discovery counts. A delegated check can be correct without mentioning the helper.

| Consumer convention inspected | Public classes | Names |
|---|---:|---|
| Can consume matrix evidence and suppress dependent numbers without a separate `complete` destination | 2 | `GPUNetworkAccessibility`, `GPUCatchmentAccessibility` |
| Read matrix evidence only when a separate completeness destination is requested | 2 | `GPUNetworkKernelDensity`, `GPUNetworkCostWeights` |
| Has a mandatory result-status array, but reads the supplied horizon only inside the `costsComplete` branch | 1 | `GPULocationAllocation` |
| Optimizes over supplied travel costs without an input certificate | 3 | `GPUNetworkRationalAccess`, `GPUNetworkFacilityOptimization`, `GPUBoundedFacilityAllocation` |
| Takes resident interpolation or conditioning CSR as `neighborhoodOffsets` and `neighborhoods`, with no input loss word | 5 direct classes, 3 GPU operation forms | `GPULocalKriging`, `GPULocalRBF`, `GPUVecchiaLogLikelihood`, `GPUVecchiaSpaceTimeLogLikelihood`, `GPUVecchiaFisherScoring`, and the three Vecchia `GPU…Operation` forms |
| CPU twins sharing those Vecchia inputs | 3 | `CPUVecchiaLogLikelihoodOperation`, `CPUVecchiaSpaceTimeLogLikelihoodOperation`, `CPUVecchiaFisherScoringOperation` |
| Point-cloud consumers taking neighborhood `counts` and slots directly | 2 | `GPUPointCloudFeatures`, `GPUPointCloudStatisticalFilter` |

The interpolation census used `rg -n '\bneighborhoodOffsets\??:\s*(GraphDataView|string|Uint32Array)' modules/analysis/src`. Following the shared CPU input type adds the three CPU twins. GeoConformal uses different field names and is counted separately in CPL-04.

## Proposed convention

**A consumer of a bounded producer result must accept its evidence alongside its data and, on every execution, forward that evidence, suppress the dependent result, or refuse the composition.** Preserve the distinction between input completeness, local solver convergence and final output capacity. A finite routing horizon and a per-row unsupported-neighborhood bit are evidence too.

Check this with paired graphs that hold the retained data constant and change only a producer status word. Run each pair with and without the consumer’s optional status destination, through direct construction and any GPU/CPU operation forms. For chunked inputs, set the word on one chunk only. For costs, also hold convergence at one and vary the horizon across the consumer’s threshold.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| CPL-01 | W | UNUSUAL | ADDITIVE | M | Vecchia likelihood and Fisher scoring | A lost predecessor becomes a legitimate marginal term. |
| CPL-02 | W | ORDINARY | ADDITIVE | S | Network disruption recipe | Two converged searches can publish the wrong accessibility difference. |
| CPL-03 | W | ORDINARY | ADDITIVE | M | `GPUNetworkCostWeights.outputWeights` | The advertised downstream handle drops the requested matrix certificate. |
| CPL-04 | W | ORDINARY | ADDITIVE | S | GeoConformal intervals | A producer-truncated CSR can yield a narrower interval with status zero. |
| CPL-05 | W | UNUSUAL | ADDITIVE | S | Raster region graph and tile recipe | Unresolved watershed cells become ordinary background and erase a boundary. |
| CPL-06 | W | ORDINARY | ADDITIVE | S | Trajectory locations and twins | A clipped staypoint table produces a smaller, converged location count. |
| CPL-07 | W | UNUSUAL | ADDITIVE | M | Rational access and facility optimizers | A missing or provisional route changes service, feasibility and optimality conclusions. |
| CPL-08 | W | ORDINARY | ADDITIVE | M | Local kriging and RBF | Both accept the search producer’s CSR without its status. |
| CPL-09 | W | ORDINARY | BREAKING recommended | S | Kernel flow smoothing | A dropped OD pair becomes a measured zero flow. |
| CPL-10 | W | UNUSUAL | ADDITIVE | S | Location allocation | A supplied, insufficient horizon can coexist with `complete=1`. |
| CPL-11 | W | UNUSUAL | ADDITIVE | S | Network kernel density | Supplied evidence does nothing if the caller requests only ordinary outputs. |
| CPL-12 | W | ORDINARY | ADDITIVE | M | Cell compaction, outline and direct ring assembly | Capacity loss disappears at a counted-cell or counted-segment handoff. |
| CPL-13 | E | ORDINARY | ADDITIVE | M | Point-cloud features and statistical filter | Tiled support failure cannot accompany the slots into either analysis. |
| CPL-14 | E | UNUSUAL | ADDITIVE | M | Consumers of cluster and region labels | Valid summaries of provisional labels lose their partition’s convergence state. |
| CPL-15 | C | UNUSUAL | ADDITIVE | S | Spatial rate recipe | The returned weights stage can be empty when a chunk supplies loss evidence. |
| CPL-16 | D | ORDINARY | ADDITIVE | S | Spatial weights capacity planning | The generic retry example can stop with a missing directed edge. |

For each **ADDITIVE** finding, existing call signatures can stay in place. No in-repository caller is required to migrate. The named examples and in-repository compositions should adopt the new wiring. CPL-09 gives the call-site count for the recommended breaking change.

### CPL-01 — One missing conditioning edge changes three successful likelihood calculations

**Where.** `gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood.ts:68`, `gpu-vecchia-space-time-log-likelihood.ts:72`, `gpu-vecchia-fisher-scoring.ts:85`. Their GPU operation forms expose only the two CSR graph names at `gpu-vecchia-log-likelihood-operation.ts:22`, `gpu-vecchia-space-time-log-likelihood-operation.ts:23` and `gpu-vecchia-fisher-scoring-operation.ts:23`. All three CPU twins share `CPUVecchiaRows`, `cpu-vecchia-evaluation.ts:31`. I checked the KI evaluator at lines 408–416.

**EXECUTED.** I ran `computeContiguityWeightsOnCPU` on two triangles sharing one rook edge, with capacity two. `pairCapacity: 8` produced offsets `[0,1,2]`. `pairCapacity: 1` produced `[0,1,1]`, `overflow=0`, `candidateOverflow=1`. Keeping only earlier neighbors for the Vecchia ordering leaves site 1 with `[0]` or `[]`. The observation values were `[1,1]`, sites were one unit apart, and the exponential covariance had range 3, sill 1 and nugget 0.1.

| CPU consumer | Full conditioning | Truncated conditioning | Status in both runs |
|---|---:|---:|---|
| `computeVecchiaLogLikelihoodOnCPU`, high word | −2.5551362 | −2.8422782 | `[0,0,2]` |
| `computeVecchiaSpaceTimeLogLikelihoodOnCPU`, equal times, matching metric covariance | −2.5551362 | −2.8422782 | `[0,0,2]` |
| `computeVecchiaFisherScoringOnCPU`, `totals[0]` | −2.5551362 | −2.8422782 | `[0,0,2]` |

**Now.** `neighborhoodOffsets[i] === neighborhoodOffsets[i + 1]` is sufficient to select a marginal term. That is correct for an intentionally empty conditioning set. Here it converted an identified producer failure into a different fitted model. This input is *under* the requested width. KNOWN(PTS-120) concerns over-width sets.

**Wanted.** Accept optional `neighborhoodStatus: {overflow?, candidateOverflow?}` in all three direct classes, operation forms and CPU functions. Suppress likelihood, score and information totals on supplied loss and expose the cause separately from failed site solves. **GPU probe:** run the two-site example with an empty second span and toggle only `candidateOverflow`. Exact CPU-built conditioning sets should retain their current results.

**Value and cost.** Model fitters can distinguish a different likelihood from a successful evaluation of their requested model. M.

### CPL-02 — A converged disruption scenario can report an accessibility loss that never occurred

**Where.** `gpu-network/recipes/network-scenarios-recipe.ts:39,197,226,253`. The two accessibility reductions take the matrices and scoring parameters. The delta checks both routing convergence words. The search also supplies `costLimit`, which the delta does not check. The retained `baseline` handle at line 47 contains costs and convergence. The guide promises cumulative accessibility within the supplied threshold at `docs/api/gpu-spatial-analysis/expansion-recipes.md:84`; I also checked the KI guide.

**EXECUTED producer, SOURCE recipe.** One origin, one target, one directed edge and one unit opportunity suffice. I ran `computeNetworkCostMatrixOnCPU` with edge cost 1 before and 2 after.

| Search horizon | Before cost | After cost | Convergence |
|---|---:|---:|---|
| 1 | 1 | `Infinity` | 1 for both |
| `Infinity` | 1 | 2 | 1 for both |

With scoring threshold 2, accessibility is one in both scenarios. **GPU probe:** pass the first pair through `addNetworkDisruptionRecipe` with `replacementCosts=[2]`. The source predicts accessibility `1 → 0`, delta `−1`. The complete-domain delta is `0`. The guide already defines `newlyUnreachable` relative to the search horizon. The defect is the *accessibility difference*, which has a separate scoring threshold. KNOWN(NET-101, NET-124) supplies the underlying matrix semantics.

**Wanted.** Forward each search’s `converged` and `costLimit` into its `GPUNetworkAccessibility`; add an `accessibility` stage with `complete`; gate `accessibilityDifferences` on both accessibility certificates. A retained baseline should carry its effective horizon or explicitly inherit the same search configuration. Independent finite cost differences can still be published.

**Value and cost.** Scenario maps avoid reporting a loss of access caused solely by a shorter routing search. S.

### CPL-03 — `outputWeights` drops a certificate the caller explicitly requested

**Where.** `gpu-network/network-cost-weights/gpu-network-cost-weights.ts:228,318,332,509,542`; `gpu-spatial-analysis/spatial-weights/spatial-weights-loss.ts:20,26`; `gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility.ts:103,451`. The documentation returns `contributor.outputWeights` at `docs/api/gpu-network/operators.md:806`, KI line 822.

**EXECUTED.** On a CPU-only declaration graph I constructed a one-row, two-column matrix conversion with distinct `costsComplete`, `costLimit`, final `overflow` and output `complete` views. The nonempty fields in `contributor.outputWeights.status` were **`['overflow']`**. The identity test for the requested `complete` view was **false**. Omitting that destination also yielded `['overflow']`.

**Now.** The obvious code is `const weights = converter.outputWeights;`. A separate caller holding the original output object can wire `costsComplete` into catchment accessibility. A function returning the documented handle has already lost that option. Even a consumer correctly using `getGPUSpatialWeightsLossViews` cannot recover it.

**Wanted.** Preserve the computed positive `complete` view on the described weights and teach the shared weights check its polarity, or publish the equivalent inherited loss view on the handle. When matrix evidence is supplied without a destination, provide a graph-owned certificate or refuse that combination. Keep final CSR capacity overflow separate. **GPU probe:** one facility, two unit demands, one provisional infinite cost, adequate CSR capacity. Pass only `outputWeights` downstream. Both orientations must carry the failed certificate.

**Value and cost.** Every downstream weights consumer benefits from fixing the conversion once. M.

### CPL-04 — GeoConformal can report a narrower interval with no truncation bit

**Where.** `gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals.ts:52,229,232,272`, also checked in KI. Compare `gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts:347`, including KI. The localized prediction guide describes CSR from any neighbor search and says a truncated list sets the bit at `docs/api/gpu-spatial-analysis/spatial-validation-advanced.md:90`, KI line 172.

**SOURCE.** The contributor sets `GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED` when the *supplied span* exceeds its internal maximum. An upstream search clamps its offsets to storage capacity. Its truncated span looks small and valid.

**Now.** From the page, `neighbors: {offsets: w.offsets, indices: w.neighbors, weights: w.weights}` is sufficient to run. **GPU probe:** one query, prediction zero, unit scale, scores `[1,9]`, equal weights and self-weight one, `alpha=.5`. The complete finite-sample rank is two, giving `[-9,9]`. Keep only the score `1` upstream and set search overflow. The retained rank is one, giving `[-1,1]` with no local truncation bit.

**Wanted.** Accept `neighbors.status` including `overflow` and `candidateOverflow`. On supplied loss, set a named input-incomplete bit and suppress the interval, or return the conservative unbounded interval. State exactly which truncation the existing bit measures.

**Value and cost.** Prediction users avoid interpreting losses in calibration support as increased certainty. S.

### CPL-05 — An unresolved watershed cell becomes a confirmed absence of adjacency

**Where.** KI `raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation.ts:99,144` and `raster-analysis/gpu-raster-watershed-segmentation.ts:338`; `raster-analysis/remote-sensing/gpu-raster-region-graph.ts:57,332`; KI `cpu-raster-region-graph-operation.ts:80,89` and `gpu-raster-region-graph-operation.ts`. The tiled adapter at `raster-region-tiles.ts:83,109,114` verifies rectangular coverage and maps unknown local labels to background.

**EXECUTED, end to end in KI.** Flat `3×1` raster, markers `[1,0,2]`, FIFO watershed, region count 3 and merge threshold zero:

| FIFO steps | Watershed labels | Watershed converged | Region boundary `adjacency[1,2]` | Region mapping | Region limit reached |
|---:|---|---:|---:|---|---:|
| 1 | `[1,0xfffffffe,2]` | 0 | 0 | `[0,1,2]` | 0 |
| 3 | `[1,1,2]` | 1 | 1 | `[0,1,1]` | 0 |

The r19 region twin also reproduced both downstream results when given those labels. No actual background cell is present.

**Now.** `computeRasterRegionGraphOnCPU(watershed.labels, values, options)` makes the unresolved cell indistinguishable from an intentional exclusion. `iterationLimitReached=0` correctly describes the region merge in isolation.

**Wanted.** Add `labelsComplete` to the direct and named graph forms and CPU twin. For `addRasterRegionGraphTiles`, accept a completeness word per tile and combine them before assembly. Expose `status.complete` for the mapping and adjacency. **GPU probe:** run the same three-cell case directly and split across two tile cores.

**Value and cost.** Basin boundaries and region mergers retain the distinction between unknown contact and no contact. S for the direct path, M including tiles.

### CPL-06 — Location clustering forgets that staypoints were dropped

**Where.** `gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations.ts:109–119`, `gpu-trajectory-locations-operation.ts:32`, `cpu-trajectory-locations-operation.ts:74,146`. KI still takes only the input `count` at direct line 119 and operation line 24. The staged CPU iteration handling changes the solver, but preserves this input boundary.

**EXECUTED in r19 and against the KI location twin.** Two tracks, one stationary pair of fixes per track at `(0,0)` and `(10,0)`, timestamps `[0,1,0,1]`, offsets `[0,2,4]`. I passed the public staypoint outputs straight into `computeTrajectoryLocationsOnCPU`.

| Stay capacity | Stay `count, requiredCount, overflow` | Location count | Location converged |
|---:|---|---:|---:|
| 1 | `1,2,1` | 1 | 1 |
| 2 | `2,2,0` | 2 | 1 |

**Now.** `locations({centroids: s.centroids, tracks: s.ids, count: s.count, parameters})` cannot convey `s.overflow`. KNOWN(TRJ-101) addresses a different loss, the location solver’s own unfinished iterations and their effect on mobility indicators. Here the solver finishes on both runs.

**Wanted.** Add `staypointStatus: {overflow?, candidateOverflow?}` to the direct class, `GPUTrajectoryLocationsOperation`, `CPUTrajectoryLocationsOperation` and CPU function. Publish location-result `complete` independently of location-solver `converged`. **GPU probe:** repeat the two-track CPU case and compare both status words.

**Value and cost.** Location catalogs can distinguish “one place found in a prefix” from “one place in the dataset.” S.

### CPL-07 — Incomplete travel costs can certify service, infeasibility or an optimum

**Where.** `gpu-network/network-optimization/gpu-network-rational-access.ts:65,270`, `gpu-network-rational-access-operation.ts:16`, `cpu-network-rational-access-operation.ts:129,203`; `gpu-network/network-optimization/gpu-network-facility-optimization.ts:340,391,466`; `gpu-spatial-analysis/location-allocation/gpu-bounded-facility-allocation.ts:42,322`.

**EXECUTED.** I routed the unit chain `0→1→2` with one local hop per round. One round produced costs `[0,1,Infinity]`, `converged=0`; four rounds produced `[0,1,2]`, `converged=1`. With the last node as the sole facility, unit supply and demand, `computeNetworkRationalAccessOnCPU` returned:

| Input cost | Assignment | Unserved demand | `[valid, gapWithinTolerance, completedCycles, maxGapBits]` |
|---|---:|---:|---|
| `Infinity` | 0 | 1 | `[1,1,0,0]` |
| 2 | 1 | 0 | `[1,1,0,0]` |

**SOURCE sibling probes.** Give `GPUNetworkFacilityOptimization` the same one-entry matrix in backup-coverage mode, threshold 2. It can enumerate every subset and find no feasible one on the short matrix; the singleton facility covers demand on the full matrix. For `GPUBoundedFacilityAllocation`, use p-center, one demand and one selected facility. The short matrix yields a missing assignment and infinite objective; the global-optimum word is calculated from mode and input validity. Enumeration and equilibrium can genuinely finish on the wrong costs.

**Wanted.** Accept `costsComplete` and `costLimit` in all three classes and the rational-access twins. Attach `status.complete` to the decision. Backup coverage needs a horizon reaching its threshold. Unbounded objectives need complete costs or a separate proof that omitted costs cannot affect the winner. Preserve the local iteration/enumeration words.

**Value and cost.** Facility planners avoid treating an unfinished search as evidence that an area cannot be served. M.

### CPL-08 — The documented search-to-prediction CSR loses its status

**Where.** `gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf.ts:101,401,458,496` and `gpu-local-kriging.ts:222,959,1075,1200`. The RBF guide explicitly recommends resident CSR from `GPUNeighborSearch` at `docs/api/gpu-spatial-analysis/interpolation-expansion.md:100`. The fixed-width conditioning adapter in `gpu-gaussian-random-field.ts:611,619` has the related distinction between an intentional local model and missing support.

**Now.** `neighborhoods: search.outputWeights.neighbors, neighborhoodOffsets: search.outputWeights.offsets` strips the other fields of `outputWeights`. Both predictors can solve the retained system and publish ordinary numerical-success status.

**SOURCE / GPU probe.** Put samples with values `[0,10]` on either side of their midpoint query. Search for both, then run with storage for just one. Use ordinary kriging with a nonsingular symmetric covariance, and Gaussian RBF with a constant polynomial. Full support predicts `5` by symmetry. Retaining only the first sample predicts `0` in both models. The observed search overflow is the evidence that distinguishes this from deliberately fitting one neighbor.

**Wanted.** Add a common CSR input accepting offsets, indices and loss status; bind it in both classes. Publish input-incomplete separately from singular or insufficient local systems. Forward the same evidence on the random-field conditioning route when supplied. Its existing approximation flag is also set on successful, intentionally local conditioning.

**Value and cost.** Users can reuse one device-resident index across predictors without implementing their own synchronization or readback. M.

### CPL-09 — Flow smoothing converts omitted pairs into measured zeros

**Where.** `gpu-network/flow-aggregation/gpu-flow-aggregation.ts:297–306`; `gpu-kernel-flow-smoothing.ts:102,304,312,410,438`; `docs/api/gpu-network/flow-models-mob10.md:10,70`.

**Now.** The example calls for `flows: {kind: 'pairs', originZoneIds, destinationZoneIds, weights, validPairCount: aggregation.status.count}`. The producer distinguishes top-K capacity loss from pair-table loss. The consumer accepts neither word. It zero-initializes the dense matrix and scatters only retained pairs.

**SOURCE / GPU probe.** Use two distinct zones with flows `0→1 = 2` and `1→0 = 1`, ample pair-table capacity, K=1, and `neighborhoodSize=0`. This makes smoothing the identity. The result for `1→0` is predicted to be `0`, against a full-table value of `1`. Repeat with enough final capacity and pair-table loss only. Kernel bandwidth arithmetic, KNOWN(NET-112), plays no role.

**Wanted.** On the pair branch, replace `validPairCount` with `status: {count, overflow, candidateOverflow?}` for producer results. Allow a separate `{count, completeness: 'known-complete'}` shape for fixed imported rows, following `GPUMapMatchedEdgeVolumes`. Expose `output.status.complete`. An additive overload is possible; I recommend removing the count-only producer shape before alpha.

**In-repository callers that change.** `rg -n '\bvalidPairCount\b'` found **three executable smoothing call sites**: the website GPU runner `flows-cubes/kernel-flow-smoothing.ts:58`, `cases-mob10-flow-models.ts:65`, and `flow-models-mob10.spec.ts:325`. Also update the website catalog’s displayed example and the API guide. Kepler’s same-named cell-flow readout is unrelated.

**Value and cost.** Flow analysts can distinguish zero trips from a flow excluded by storage. S.

### CPL-10 — Location allocation can ignore a horizon supplied by its caller

**Where.** `gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation.ts:329–338`, `gpu-location-allocation.ts:1325–1365`, `gpu-location-allocation-operation.ts:27,123`.

**EXECUTED.** For one facility, two unit demands, costs `[0,Infinity]`, `coverageDistance=[2]`, `costLimit=[1]`, maximum coverage and facility budget one:

| Supplied matrix evidence | Objective | Allocation `complete` |
|---|---:|---:|
| `costLimit=[1]` | 1 | **1** |
| `costLimit=[1], costsComplete=[1]` | 1 | **0** |

A true second cost of 2 would make the objective 2. Adding evidence that the search *succeeded* makes the allocation retract its earlier certificate. The named graph form accepts the two inputs independently.

**Wanted.** Check a supplied horizon even when the convergence view is absent, as `GPUNetworkAccessibility` already does at `gpu-network-accessibility.ts:339`. Alternatively reject a horizon-only configuration at construction and in the CPU function. The latter is an additive rejection of an input combination that can certify an unsupported claim. **GPU probe:** toggle the presence of the convergence view on the two-demand fixture; neither run should certify the score.

**In-repository callers.** An AST check found 91 direct named constructor/function calls, five object literals explicitly supplying both fields, and zero explicitly supplying `costLimit` alone. Forwarded props still need the runtime check.

**Value and cost.** A known limit can no longer accidentally strengthen an allocation certificate. S.

### CPL-11 — Network KDE accepts an input loss word that no numerical node reads

**Where.** `gpu-network/gpu-network-kernel-density.ts:49,209,247,261,328`. This concerns behavior *after* the KNOWN(NET-101) port addition.

**EXECUTED host.** With both `costsComplete` and `costLimit` supplied, the declared node suffixes were `['mass','intensity']`. Adding only the output `ports.status.complete` view produced `['mass','intensity','complete']`. The first two node bindings do not include either input word.

**SOURCE / GPU probe.** One event, two quadrature points with lengths `[1,1]`, bandwidth 2. Distances `[0,Infinity]` from an incomplete search give intensities `[1,0]`; correct distances `[0,1]` give `[2/3,1/3]`. Both rows satisfy the present numerical validity check. Omit only the optional scalar destination, leaving `costsComplete=0` connected.

**Wanted.** When evidence is present and there is no certificate destination, suppress affected mass and intensity and mark the per-query status. When the destination is requested, the documented incumbent-plus-certificate behavior can remain. This matches the existing accessibility convention.

**Value and cost.** Wiring an input diagnostic protects a density layer even if its caller consumes only ordinary output columns. S.

### CPL-12 — A cell prefix can grow into a clean but wrong polygon

**Where.** `gpu-spatial-analysis/cell-topology/gpu-cell-compaction.ts:109,116,698`; `gpu-spatial-analysis/cell-set-outline/gpu-cell-set-outline.ts:115,367` and KI lines 114, 312, 515; `analysis-core/ring-assembly/gpu-segment-ring-assembly.ts`, KI line 372. There is a real handoff at `gpu-network/network-isochrones/gpu-network-isochrones.ts:1034`, KI lines 1041–1045, which passes `table.cells` and `table.status.count`.

**SOURCE / GPU probes.**

- Feed two adjacent Quadbin cells from a bounded table into outline, retaining one. Give the outline ample edge capacity. The retained square has four boundary edges; the two-cell union has six. The retained square’s edge against its missing neighbor is an invented exterior boundary.
- Uncompact the same one-cell prefix by one Quadbin level. Four children fit and final overflow can be zero; the complete source yields eight. `GPUCellCompaction` accepts the active count but no evidence about its source.
- Feed direct `GPUSegmentRingAssembly` one complete four-edge square from what was an eight-edge, two-component outline. The square closes, so open-edge detection cannot reveal the lost component.

**Now.** `new GPUCellSetOutline({cells: table.cells, count: table.status.count, …})` loses source overflow. The KI nested ring adapter **does** suppress assembly when the outline’s *own* edge buffer overflows. That fix is a useful control. It cannot detect loss that occurred before the outline was built, and the direct ring API lacks that adapter.

**Wanted.** Accept counted inputs with `{count, overflow, candidateOverflow?}` on all three boundaries. Propagate input loss to `candidateOverflow` or suppress geometry. Wire the whole table status in the isochrone composition. Keep the local edge, ring and cell capacity flags available for sizing.

**Value and cost.** GIS users avoid exporting closed, plausible polygons with false exterior boundaries. M.

### CPL-13 — An unsupported tile neighborhood can influence globally classified rows

**Where.** `pointcloud-analysis/operations/gpu-pointcloud-neighborhoods.ts:99,144`, KI lines 145, 564; `gpu-pointcloud-features.ts:82,240,251,271`; `gpu-pointcloud-statistical-filter.ts:67,182,250,258`. The point-cloud guide explicitly asks users to inspect neighborhood halo status separately at `docs/api/pointcloud-analysis.md:845`, KI line 851.

**Now.** `new GPUPointCloudFeatures({positions, indices, counts, capacity, output})` can recover slot truncation from counts but cannot recover producer bit 4, unsupported spatial extent. The statistical filter uses retained distances to compute a shared threshold. Reading the unsupported row’s bit after classification does not repair other rows’ classifications.

**SOURCE / GPU probes.** For features, use three noncollinear neighbors in the plane `z=0`, capacity four, and a query whose radius crosses a tile boundary. Including a fourth neighbor at `z=1` changes population covariance `zz` from `0` to `3/16`. For filtering, use two query rows with retained mean distances `[1,3]` and `standardDeviations=0`. A missing nearer neighbor that changes the second mean to `.25` changes the **first**, fully supported row from inlier to outlier.

**Wanted.** Accept the producer’s per-row `neighborhoodStatus`. Give features an explicit unsupported-support bit. For filtering, expose whether the global threshold is complete and document the policy for unsupported rows before calculating moments. Preserve the intentional nearest-K interpretation; exceeding a radius population and lacking the certified halo are different conditions.

**Value and cost.** Tiled point-cloud users can distinguish a valid computation on retained neighbors from a complete spatial result. M.

### CPL-14 — Group summaries lose the provenance of provisional labels

**Where.** `gpu-network/network-coarsening/gpu-network-coarsening.ts:164`; `gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds.ts:119`, `cpu-minimum-bounds-operation.ts:29,225`, `gpu-minimum-bounds-operation.ts:13`; `group-geometry/gpu-group-geometry.ts:118`, `gpu-group-convex-hull.ts:154`, `concave-hull/gpu-concave-hull.ts:332`, `polygon-coverage-topology/gpu-coverage-dissolve.ts:119`; KI `spatial-regionalization/gpu-region-partition-evaluation.ts:151` and `geographical-detector/gpu-geographical-detector.ts:92`. These last five directory paths are under `gpu-spatial-analysis`.

**EXECUTED.** I ran `computeNetworkLouvainOnCPU` on four vertices, symmetric weights 2 inside pairs `(0,1)` and `(2,3)`, 1.25 across pairs, resolution 1. One level gave labels `[1,1,3,3]`, `converged=0`. Eight allowed levels gave `[3,3,3,3]`, `converged=1`. On square-corner positions, `computeMinimumBoundsOnCPU` returned group 3 bounds `[0,1,1,1]` versus `[0,0,1,1]`; both `overflow` and `groupOverflow` were zero in both runs.

**Now.** These are correct bounds for the supplied labels. They become misleading when published as the bounds of the finished communities. The same handoff affects counts, hulls, dissolves and partition scores. KNOWN(PTS-103) concerns missing *adjacency* in partition evaluation; the input here is a complete graph with provisional *labels*.

**Wanted.** For solver-derived groups, accept optional `labelsComplete` and carry it as partition provenance. Keep incumbents available to users deliberately inspecting an intermediate iteration. Ordinary administrative or categorical labels need no solver word. **GPU probe:** reuse the two published label vectors with the completion flag toggled; coarsening should expose that the two-group quotient used provisional labels.

**Value and cost.** Dashboards can label intermediate communities accurately without discarding useful geometry. M.

### CPL-15 — A chunk-local loss word disappears from recipe status

**Where.** `gpu-spatial-analysis/recipes/rate-mapping-recipes.ts:59,169,221`; `docs/api/gpu-spatial-analysis/rate-mapping.md:5–9`. KI `rate-smoothing/gpu-spatial-empirical-bayes-rates.ts:302` already suppresses numeric outputs for outer and chunk-local flags.

**EXECUTED.** I declared `addSpatialRateRecipe` over two rows in one CSR chunk with `chunks[0].weights.status.candidateOverflow` present and no outer status. Reading the actual recipe result on the host gave **`[['weights', []]]`** for stage names and status keys.

**Now.** `recipe.status.stages.find(s => s.stage === 'weights').status` contains no diagnostic. With the KI numeric fix, a consumer can receive suppressed rates without being able to distinguish a missing neighbor table from an island through the documented recipe status. This finding is about the recipe result beyond KNOWN(SSR-102).

**Wanted.** Fold outer and chunk-local flags into named `overflow` and `candidateOverflow` views on the `weights` stage, preserving the two causes. **GPU probe:** run with only the chunk flag set, clear it, and resubmit the same graph. The recipe diagnostic should clear and the numeric result should recover.

**Value and cost.** Recipe users can tell whether to grow the producer or investigate their rate data. S.

### CPL-16 — The weights grow-loop example can finish one edge short

**Where.** `docs/api/gpu-spatial-analysis/operators-weights.md:54–66`, KI line 60; `gpu-spatial-analysis/spatial-weights/spatial-weights.ts:343,374`, KI lines 345, 377; `gpu-contiguity-weights.ts:321` provides the producer-specific planner.

**EXECUTED.** On the two-triangle rook fixture from CPL-01, with final slot capacity two:

| Pair capacity | Final `requiredCount` | Final `overflow` | `candidateOverflow` | Generic plan `capacity, limited` |
|---:|---:|---:|---:|---|
| 1 | 1 | 0 | 1 | `2, false` |
| 8 | 2 | 0 | 0 | `2, false` |

**Now.** The guide first says to grow when `requiredCount > capacity` and later says to avoid lossy weights. A loop implementing the first instruction terminates on the first row of the table. The generic helper correctly prices final slots; it cannot tell the caller whether candidate enumeration was complete.

**Wanted.** Put the test in execution order in the example: inspect `candidateOverflow` first, resize the producer’s candidate stage with its own planner, rerun, then size the final CSR from `requiredCount`. State inline that `limited=false` refers to the final allocation. Use this two-triangle case as the documentation test.

**Value and cost.** A first-time user gets a complete, symmetric rook graph without having to infer which allocation to grow. S.

## Departures by name

| Boundary | Public departures established in this review |
|---|---|
| A cost certificate is detached, conditional on a second destination, or insufficiently checked | `GPUNetworkCostWeights`, `GPUNetworkKernelDensity`, `GPULocationAllocation`, `GPULocationAllocationGraphOperation`, `CPULocationAllocationOperation`, `computeLocationAllocationOnCPU`, `addNetworkDisruptionRecipe` |
| Travel-cost consumers lacking a route-completeness input | `GPUNetworkRationalAccess`, `GPUNetworkRationalAccessOperation`, `CPUNetworkRationalAccessOperation`, `computeNetworkRationalAccessOnCPU`, `GPUNetworkFacilityOptimization`, `GPUBoundedFacilityAllocation` |
| Search or conditioning support detached from its evidence | `GPUGeoConformalPredictionIntervals`, `GPULocalKriging`, `GPULocalRBF`, `GPUGaussianRandomField` conditioning, `GPUPointCloudFeatures`, `GPUPointCloudStatisticalFilter` |
| Conditioning loss indistinguishable from an intentionally empty set | `GPUVecchiaLogLikelihood`, `GPUVecchiaSpaceTimeLogLikelihood`, `GPUVecchiaFisherScoring`, their three GPU operation forms, three CPU operation forms and three `compute…OnCPU` functions |
| Counted product stripped to count and columns | `GPUTrajectoryLocations`, `GPUTrajectoryLocationsOperation`, `CPUTrajectoryLocationsOperation`, `computeTrajectoryLocationsOnCPU`, `GPUKernelFlowSmoothing`, `GPUCellCompaction`, `GPUCellSetOutline`, direct `GPUSegmentRingAssembly`, and the `GPUNetworkIsochrones` cell-table-to-outline handoff |
| Unresolved or provisional partition | `GPURasterRegionGraph`, `GPURasterRegionGraphOperation`, `CPURasterRegionGraphOperation`, `computeRasterRegionGraphOnCPU`, `addRasterRegionGraphTiles`, `GPUNetworkCoarsening`, `GPUMinimumBounds` and its GPU/CPU operation forms and CPU function, `GPUGroupGeometry`, `GPUGroupConvexHull`, `GPUGroupConcaveHull`, `GPUCoverageDissolve`, `GPURegionPartitionEvaluation`, `GPUGeographicalDetector` |
| Loss omitted from a recipe diagnostic or retry example | `addSpatialRateRecipe`, documented use of `getGPUSpatialWeightsCapacityPlan` |

## Quick wins

- [ ] Fix the weights grow-loop example and pin the one-pair/two-slot case.
- [ ] Return chunk-local causes from the spatial rate recipe’s `weights` stage.
- [ ] Compare the disruption scoring threshold with the search horizon before publishing accessibility deltas.
- [ ] Make a supplied location-allocation horizon effective independently, or reject an unpaired horizon.
- [ ] Close the output-destination omission path in network KDE.
- [ ] Add input status to GeoConformal and the three trajectory-location entry points. Test an input flag in isolation from the consumer’s own overflow or iteration limit.

## Larger proposals

1. **Preserve evidence on reusable results.** Start with `GPUNetworkCostWeights.outputWeights`. A caller should be able to return the documented handle from a function and retain everything the next consumer needs to decide whether its score is complete. Extend the same design to neighborhood CSR and solver-derived labels.
2. **Change the pair-mode smoothing API before alpha.** Three executable call sites make this a small migration. Use a status-bearing producer form and an explicit complete-external-data form. The repository already uses this distinction for map-matched traversals.
3. **Test handoffs across families.** The highest-value fixtures here are contiguity → conditioning → likelihood, reachability → rational access, and watershed → region graph. In all three, both contributors can pass their own isolated tests while the final number is wrong for the requested domain.

## Not read

I read the shared r19 weights-loss collector/folder and the r19 trajectory-location GPU operation adapter in full. I traced the relevant producer, constructor, lowering, finalization and returned-handle paths for the findings above, checked listed KI counterparts, and ran the stated CPU and null-device probes. I searched the prior synthesis, area reports, API/catalog and direction reports, and the decision list for overlap.

The census covers the source tree. The semantic trace is a sample selected by product type and actual producer-to-consumer wiring. I did not inspect every numerical kernel, every prepared join implementation, every generic dataframe grouping caller, or every downstream lens, Kepler and website readout. GPU numerical outcomes above are marked **SOURCE** and have device probes; none was executed on a GPU.