# Point patterns, clustering, interpolation and space-time (2026-11-19)

- Scope: 20 directories of `gpu-spatial-analysis` (about 70,000 lines) plus the variogram, K-function and Ripley files of `dataframe-analysis/pair-statistics`. 75 public `GPU*` classes in the directories, plus the pair-statistics classes.
- All of the 2026-11-17 findings that fall here (PTS-01 to PTS-12, RCP-01 hot-spot metadata, DOC-01 variogram fit) are FIXED except PTS-06 (PARTLY), DOC-01 (PARTLY) and the Vecchia over-width conditioning-set item (OPEN). No fix came undone, and each fix has a test that would fail without it.
- Twelve new wrong-result or visible-failure findings (PTS-101 to PTS-112), all UNUSUAL or ORDINARY-budget reach, none EXECUTED on a GPU. Seven were reproduced on a CPU path or a transcription of the shader; five are source traces with a GPU probe listed.
- The dominant new class is the one the previous review named: completeness does not travel. Weights loss (`overflow`, `candidateOverflow`) is now handled by MST, SKATER, REDCAP, Ward, Knox, Mantel, Jacquez, Huff and catchment, and still ignored by neighbourhood summary, AZP, Max-p, partition evaluation, local Knox, near-repeat Knox and emerging hot spots (PTS-101, 103, 104).
- Iterative solvers publish status words in most places. The gaps are: the public CPU variogram fit (no boundary or budget signal), the Vecchia Fisher stall (a fourth, undocumented stopping state), the Max-p construction budget (indistinguishable from infeasibility), and OPTICS grid overflow without a bound port.
- Docs are the weakest part: stale limits for OPTICS and HDBSCAN, a wrong Dorling cross-reference, no heading for four regionalizers and a location-allocation solver, and 20 operation-form or CPU-twin classes not named in `docs/api`.
- Tests: external oracles are strong for kriging, KDE, similar locations, Ripley envelope, Sinkhorn and OPTICS. They are weak where an oracle was written from the same reading as the code (Hawkes, space-time K, quadrats, Ripley G/F, Max-p contiguity) and absent at projected-metre magnitude in density, space-time kriging and Vecchia.

## Method

- Lead: roadmap rows, census, diff statistics (62 files, +3,136/-574 lines since `bdc715ea` in this scope), Sinkhorn CPU twin read in full and checked against a NumPy dual-Sinkhorn oracle, `gpu-ripley.ts`, `gpu-point-pattern-rank-envelope.ts`, `gpu-inhomogeneous-point-pattern.ts`, `gpu-ripley-envelope.ts` (summary and simulation nodes), GPU Sinkhorn update kernel; verification of every W and S finding below by opening the cited lines or re-running the subagent probe.
- Six Opus slices. Density and neighbourhood (point-density, line-density, neighbourhood-summary, similar-locations, catchment-accessibility incl. Huff, analysis-spacing). Clustering and change points. Regionalization, REDCAP, location-allocation, Dorling. Interpolation and the variogram side of pair-statistics (the slice lead did it without sub-subagents). Scan statistics, space-time tests, Hawkes, emerging hot spots, space-time K and Diggle. Point process, point-pattern indices, marked statistics, Ripley distance functions.
- Read in full by slices: emerging hot spots including all WGSL (previous review's least-read part), `cpu-optics-operation.ts`, `gpu-optics.ts`, `gpu-spatial-clustering.ts`, the CPU Max-p, SKATER and location-allocation twins, REDCAP, `gpu-kriging.ts`, `gpu-local-kriging.ts` (most), the Vecchia Fisher fit, space-time kriging, the balanced and continuous samples and their twins, translation K, `gpu-point-pattern-indices.ts`, marked statistics. Per-slice lists are in the last section.
- Executed (CPU only): Sinkhorn twin against NumPy (eps 0.1, 0.5, 1; match to 1e-7); Max-p budget reproduction; `fitVariogramModel` bins reproduction; Fisher stall and nugget probes; Mann-Kendall and classifier NumPy oracle on 80,000 series (matched the TS oracle); scikit-learn OPTICS on 60 cases; Pettitt brute force on 2,000 series; change-point enumeration on 400 series; balanced-sample inclusion over 20,000 replicates; isotropic rectangle weight against quadrature; Python transcriptions of three WGSL expressions (marked TRANSCRIPT where used). No test, gate, build or browser ran.

## Census

Commands run from `modules/analysis/src/gpu-spatial-analysis`, zsh array of the 20 directories.

| Measure | Result | Command |
|---|---|---|
| GPU classes in the 20 directories | 135 matches for `export class GPU*` (includes operation forms); 75 unique names | `rg -o 'export class GPU\w+' -- $D -g '*.ts'` |
| `getCommandNodes(` definitions | 135 | `rg -o 'getCommandNodes\(' -- $D` |
| Bare `assert(` | 0 | `rg -n "\bassert\(" -- $D` |
| Named-limit helper calls | 135 matches of `validate(Capacity|MaximumCount|Requirement|Count)`; `validateCapacity` alone 359 mentions | `rg -o` |
| Files with `status` | 99 | `rg -l status -- $D` |
| Files over 1,500 lines | `scan-statistics/gpu-spatial-scan-statistic.ts` 1,973; `location-allocation/gpu-location-allocation.ts` 1,801; `emerging-hot-spots/gpu-emerging-hot-spots.ts` 1,731 | `wc -l` |
| Spec files for the area | 366 (name match over `modules/analysis/test`) | `find ... \| rg` |
| Most used `...Ids` | frameIds 37, rowIds 32, sourceIds 28, sortedIds 25 | `rg -o '\w+Ids'` |
| `maximum<Entity>Count` | maximumNeighborCount 39, maximumRegionCount 33, maximumFacilityCount 26 | `rg -o` |
| Not named in `docs/api` (16 operation-form classes) | GPUAZPRegionsOperation, GPUConditionalGaussianSimulationOperation, GPUContinuousSpatialSampleOperation, GPULocationChangePointsOperation, GPUMantelTestOperation, GPUMaxPRegionsOperation, GPUNeighborhoodSummaryOperation, GPUOPTICSOperation, GPUPettittTestOperation, GPUPointPatternEnvelopeOperation, GPUSpaceTimeHawkesOperation, GPUSpatialBalancedSampleOperation, GPUVecchiaFisherScoringOperation, GPUVecchiaLogLikelihoodOperation, GPUVecchiaSpaceTimeLogLikelihoodOperation, GPUWardRegionsOperation | `rg -F` per class over `docs/api` |
| Not named in the ontology (19) | the 16 above minus GPUMantelTestOperation..., plus GPULocationAllocationGraphOperation, GPUSinkhornDivergenceOperation, GPUSkaterRegionsOperation | `rg -F` over `ontology/` |
| Own docs heading missing (contributors) | GPUAZPRegions, GPUMaxPRegions, GPUWardRegions, GPULocationAllocation, GPUConditionalGaussianSimulation, GPULocalRBF, GPUNaturalNeighbor | per-slice `rg` |
| Contract checker, baselined sites | 7 regions, 8 space-time, 4 interpolation, 1 clustering (all `single-invocation-data-loop`); no new findings | `node scripts/check-contributor-contract.mjs --all` |
| Upstream-first | no violations in regions slice | `node scripts/check-upstream-first.mjs` |
| Unprefixed `throw new Error` | regions 33 (CPU twins), interpolation 39 plus 14 prefixed without prop or limit, space-time 34, clustering 35 bare, indices 10, marked 6 | per-slice `rg` |
| Props kept by reference (no snapshot) | space-time 9 of 13 contributors; interpolation 16 of 18; regions: AZP shallow, Max-p, partition evaluation, both facility solvers, Dorling | `rg snapshotConstructionFields` |

Status shapes: named scalar ports (REDCAP, Max-p partly, k-means, HDBSCAN, change points, Hawkes), packed words (OPTICS `uint32[3]`, AZP 4, Max-p 6, Ward 4-5, MST 2, location-allocation 7, bounded facility 3, Dorling 4, Vecchia 3), per-row code views called `status` (local kriging, RBF, natural neighbour). No status: IDW, focal statistics, SKATER (publishes `regionCount` and the cut log), Pettitt. Mask polarity: nonzero includes everywhere in these directories; the float validity weight in interpolation is `> 0` (PTS-119).

## Follow-through on the 2026-11-17 findings

| ID | 2026-11-17 claim (short) | Verdict | Evidence |
|---|---|---|---|
| PTS-01 | Sinkhorn divergence converged on the cross term only | FIXED | CPU twin now needs the cross plan and both symmetric self plans (`cpu-sinkhorn-divergence-operation.ts:224-252`); GPU update kernel does the same (`gpu-sinkhorn-divergence.ts:713`). Lead probe: source `[90,10]`, target `[10,90]`, 2x1 grid, epsilon 0.1/0.5/1 gives 0.76860785, 0.67936295, 0.65258425; NumPy dual Sinkhorn gives 0.76860784, 0.67936293, 0.65258426. Pinned by `cpu-sinkhorn-divergence-operation.node.spec.ts:90-130` (two-cell case, budget 2 must report not converged; would catch a cross-only test). |
| PTS-02 | Finite 1e30 distance treated as unreachable | FIXED | Bit-pattern reachability (`gpu-location-allocation.ts:1212`), shared tie rule (:1674, :1718). Pinned by `cpu-location-allocation-operation.node.spec.ts:50,79,108`, `gpu-location-allocation.spec.ts:302,770`; would catch a reinstated sentinel. |
| PTS-03 | Huff keeps old edges after CSR shrinks | FIXED | `clear-probabilities` pass (`gpu-huff-trade-areas.ts:422`). Pinned by `huff-resubmission.node.spec.ts:131` (offsets [0,2] to [0,1] to [0,0]) and the GPU twin `huff-resubmission.spec.ts:144`. |
| PTS-04 | Optional outputs without a writer | FIXED (both halves) | Space-time cross-correlation counts finite p-values without `adjustedPValues` (`gpu-space-time-cross-correlation.ts:327-378`), pinned by `cross-correlation-optional-count.spec.ts`; REDCAP `iterationCount` fill (`gpu-redcap-single-linkage.ts:473`), pinned by `redcap-iteration-count.spec.ts:101,129` with a prefilled view of 99. |
| PTS-05 | Regionalizers unroll budgets without a node estimate | FIXED | `regionalization-command-budget.ts`, SKATER clamp (`gpu-skater-regions.ts:280-296`). Pinned by `gpu-spatial-regionalization.node.spec.ts:77,158,174,204` comparing emitted node counts with the estimate. |
| PTS-06 | One-zone scan triggers all-pairs nearest-zone search | PARTLY | `auto` uses the indexed route for lists of at most 65 rows (`scan-statistic-nearest-index.ts:40-45`); pinned by `scan-nearest-index.node.spec.ts:24,289`. Remaining: geographic route prunes on latitude only (`:195-203`), indexed route has no work estimate; not timed. |
| PTS-07 | Poisson scan depends on baseline units | FIXED | `scan-baseline-scaling.ts`, `gpu-spatial-scan-statistic.ts:807-810`. Pinned by `scan-overflow-review.spec.ts:131-150` (scales 1e-30 to 3e38, a subnormal); would catch the old `w*C/W` order. New regression in the fix: PTS-107. |
| PTS-08 | Early-stopped permutation tests describe 10 draws as 150 | FIXED | `requestedPermutationCount`, `stoppingRule`, `decodeGPUSpaceTimeInferenceMetadata` (`space-time-parameters.ts:172-249`). Pinned by `gpu-space-time-stopping.spec.ts:269`, `space-time-parameters.node.spec.ts:61-87` (150 requested, 10 run). |
| PTS-09 | Huff powers before normalising | FIXED | Log-ratio scores with row-maximum subtraction (`gpu-huff-trade-areas.ts:287-335`). Pinned by `huff-resubmission.node.spec.ts:165` (scale 1e4, exponent 10; 1e-20; exponent 3e38); a `pow`-first version gives inf/inf. |
| PTS-10 | Max-p calls improvement rounds a move count | FIXED | Renamed `improvementIterationCount` (`gpu-max-p-regions.ts:186`), CPU twin honours the budget. Pinned by `cpu-max-p-regions-operation.node.spec.ts:24,32,45`. No reader of the old name remains. |
| PTS-11 | Hawkes likelihood from a different EM iterate | FIXED (additive) | `fittedLogLikelihood` added; `logLikelihood` kept. Pinned by `cpu-space-time-hawkes-operation.node.spec.ts:63-65` (closed form), GPU `space-time-hawkes-operation.spec.ts:204`. |
| PTS-12 | Two roadmap descriptions misdescribed work | FIXED | STX-08 row and `geographical-detector-redcap-entropy.md:55` separate tree and cut; IN05/IN02 row says degree -1 to 2. |
| RCP-01 | Hot-spot permutation `testedRowsDefinition` describes another family | FIXED | `hot-spot-analysis-recipe.ts:364-373,1405-1421` returns the contributor's own metadata; pinned by `hot-spot-permutation-statistic.spec.ts:229`. Read only those lines. |
| DOC-01 | Variogram fit silently shortens unequal bin arrays | PARTLY | CPU twin and CPU operation throw and are pinned (`cpu-variogram-model-fit-operation.node.spec.ts:395-427`). `fitVariogramModel` and `fitSpaceTimeVariogramModel` still cut to the shortest array (PTS-105). |
| Vecchia over-width conditioning sets | Three different answers above the neighbour capacity | OPEN | No change in these files since `bdc715ea`; see PTS-120. |
| LR fix X7 | CPU OPTICS ignored `maximumIterations` | FIXED | `cpu-optics-operation.ts:223-279`; pinned by `cpu-optics-operation.node.spec.ts:49-76` and `optics-operation.spec.ts:16-74`. |
| LR fix | DBSCAN empty route excluded inputs from the disjointness check | FIXED | `gpu-spatial-clustering.ts:456`, `f-cluster-integrity.node.spec.ts:870-900`. |

The diffs of the second tranche for MST, SKATER, REDCAP and Ward were read hunk by hunk by the regions slice: no new defect there beyond PTS-103 (AZP and Max-p were not in that tranche). Hot-spot, Ripley, K and Diggle files are unchanged since `bdc715ea`.

## Findings

| ID | Kind | Reach | Evidence | Where | What goes wrong | Proposed change |
|---|---|---|---|---|---|---|
| PTS-101 | W | UNUSUAL | EXECUTED-CPU | `neighborhood-summary/gpu-neighborhood-summary.ts:122,200`; `gpu-neighborhood-summary-operation.ts:59,76`; `cpu-neighborhood-summary-operation.ts:28,121,270` | Only `weights.status.overflow` is read. `candidateOverflow=1` with `overflow=0` (pairs dropped by `pairCapacity`) publishes ordinary numbers. | Fold both flags through `spatial-weights-loss.ts` in the GPU class, the operation form, the CPU twin and the doc line. |
| PTS-102 | W | UNUSUAL | EXECUTED-CPU, GPU traced | `spatial-regionalization/cpu-max-p-regions-operation.ts:205-227`; `gpu-max-p-regions.ts:1036,1073` | Enclave rounds move rows out of a below-threshold region one round at a time. When `maximumGrowthSteps` runs out mid-way the remaining parts keep one label but are not adjacent. | Release the whole below-threshold region before the rounds, or split by connected component at finalize; test steps 1 to 8 in both forms. |
| PTS-103 | W | UNUSUAL | SOURCE | `gpu-azp-regions.ts`, `gpu-max-p-regions.ts`, `cpu-region-inputs.ts`; `regionalization-inputs.ts:281`; `gpu-region-partition-evaluation.ts:152,549` | AZP, Max-p and partition evaluation accept weights carrying loss flags and build or score regions on truncated adjacency without a status change. | Use the same suppression nodes as Ward and SKATER, or refuse status-bearing weights. |
| PTS-104 | W | UNUSUAL | SOURCE (`rg -c` finds 0 matches for status, overflow, suppression in the three files) | `space-time-tests/gpu-local-knox.ts`, `gpu-near-repeat-knox.ts`; `emerging-hot-spots/gpu-emerging-hot-spots.ts` (weights mode) | Siblings Knox, Mantel and Jacquez suppress on incomplete pairs; these three compute local counts, near-repeat cells and Gi* on truncated rows with no flag. | Add `createSpatialWeightsSuppressionNodes`. |
| PTS-105 | W | UNUSUAL | EXECUTED-CPU | `pair-statistics/variogram-model.ts:262-266`; `space-time-variogram-model.ts:489-494` | Public fits cut bin arrays to the shortest length. | Move `validateVariogramModelFitBins` into both functions; name the argument and both lengths. |
| PTS-106 | W | UNUSUAL | TRANSCRIPT of WGSL | `space-time-k-function-nodes.ts:337,416,427`; `space-time-k-function-wgsl.ts:12`; `pair-histogram.ts:270-283` | Inhomogeneous space-time K accumulates each pair weight in units of 2^-16. Relative intensity 100 gives +6.8 %, 300 gives +37 %, 363 and above gives -100 %. `pairCounts` stays exact. | Per-frame power-of-two scale or a finer channel; fixture with intensities to 1,000. |
| PTS-107 | W | UNUSUAL (outside stated contract) | TRANSCRIPT of WGSL | `gpu-spatial-scan-statistic.ts:898,914`; `scan-baseline-scaling.ts:12-23` | `[1,1,NaN,1]` normalises to `[0,0,1.5,0]`; `Inf` to `[0,0,1,0]`; `-1` to `1`; `invalidCount` stays 0. | Flag non-finite or sign-set words in the block-maximum pass and set the invalid word. |
| PTS-108 | W | UNUSUAL | SOURCE (`gpu-point-pattern-indices.ts:981-1014`) plus CPU replay of the formula | `pair-statistics/gpu-point-pattern-indices.ts:827-840,981-1014` | Polygon quadrats with zero samples are skipped, but expected counts use `n` of all points. Window [0,1]^2 plus a thin spike, counts [45,10,45,0]: chi-square 1.0, df 1, p 0.317; with true areas 183.8, df 3. The test oracle copies the rule (`point-pattern-indices-oracle.ts:331-345`). | Report the dropped points or use the active total; independent area oracle. GPU probe wanted: that window on a device. |
| PTS-109 | W | UNUSUAL | SOURCE plus float emulation | `gpu-point-pattern-indices.ts:835,838` | `floor((x - min) / width * columns)`: with `a/b` compiled as `a*(1/b)`, 6 of 9 interior integer edges on [0,100] with 10 columns go to the lower quadrat (IEEE division: 0). GPU probe wanted: window [0,0,100,100], grid [10,10], point (30,5) on Metal; read `quadratCounts[3]` (expected 1). | `(x - min) * columns / width` plus an edge correction. |
| PTS-110 | S | UNUSUAL | SOURCE; CPU interpreter | `catchment-accessibility/gpu-huff-trade-areas.ts:106,292,315` | Infinite attractiveness or weight passes the `> 0` gate and reaches `frexp(inf)` (undefined). CPU interpreter: `[Inf,1]` gave `p=[1, 2.9e-39]`. GPU probe wanted: attractiveness `[Inf,1]`, weights `[1,1]`, exponents 1 and 2. | Treat non-finite as excluded or NaN; say "finite" in the doc. |
| PTS-111 | S | UNUSUAL | SOURCE | `point-process/gpu-continuous-spatial-sample.ts:248,282-293,331-343` | Each wave is one serial invocation; validated maxima allow about 1.7e9 inner iterations (default about 9e7). GPU probe wanted: 1,365 triangles, 1,048,576 candidates, one wave. | One invocation per candidate with order-preserving compaction, or cap candidates times primitives. |
| PTS-112 | S | ORDINARY | SOURCE arithmetic | `gpu-continuous-spatial-sample.ts:140-146,174-180,333` | Area lattice size is known at construction (bounds [0,0,1,1], spacing 0.003 gives 335^2 = 112,225 > default 65,536) but only a runtime flag 4 reports overflow; the CPU twin ignores the budget and completes. | Default `maximumCandidates` to the lattice size and reject at construction. |
| PTS-113 | C | ORDINARY | EXECUTED-CPU | `variogram-model.ts:519` (`fitVariogramModelWithDiagnostics` is `@internal`) | `fitVariogramModel` gives no boundary or budget signal: gamma = h, spherical, returns range 30.000 (3 x maxLag edge); the twin returns status 3. | Export diagnostics or add a status. |
| PTS-114 | C | UNUSUAL | EXECUTED-CPU | `spatial-interpolation/vecchia-fisher-fit.ts:207-224,42-51` | Line-search stall gives `converged`, `iterationLimitReached` and `failed` all false; the rejected iterate is pushed to `history` though the doc says accepted only. | Add `stopReason`; fix history; test. |
| PTS-115 | C | UNUSUAL | EXECUTED-CPU | `vecchia-fisher-fit.ts:24,129-134,161,170-174` | "Zero nugget stays zero" is false for Matern 3/2 (floor 0.002, `atNuggetFloor`); sill-carry terms apply only when the nugget is estimated. | Fix doc; apply the carry whenever the nugget is pinned. |
| PTS-116 | C | UNUSUAL | SOURCE | `space-time-variogram-model.ts:211-217,222,393-447` | 24-cycle budget, result carries only `atBoundary`. | Add `converged` and `cycleCount`. |
| PTS-117 | C | UNUSUAL | SOURCE | `spatial-interpolation/separation-wgsl.ts:99-111` | Great-circle kriging and IDW cut the query box at +-180 degrees; samples across the antimeridian are dropped silently. Documented only in TSDoc. GPU probe wanted: samples (179.9,0)=1 and (-179.9,0)=3, one cell at 179.95, radius 50 km, k 2; blended value expected. | Docs note and roadmap limit now; two-box query later. |
| PTS-118 | C | UNUSUAL | SOURCE | `line-density/gpu-line-density.ts:207-214`; `line-density-kernels.ts:76-80` | In longitude-latitude a segment (179,0) to (-179,0) walks about 358 degrees of cells; no status (the polygon sibling has `antimeridianUncertainCount`). | Document; add a count. |
| PTS-119 | C | UNUSUAL | SOURCE | `gpu-local-kriging.ts:766`, `space-time-kriging-kernel.ts:63`, `space-time-kriging-index.ts:37`, Fisher kernel | Float validity is `w > 0`; docs say "0 skips" and the contract says nonzero usable. A negative or NaN validity is skipped silently. | Document `> 0`. |
| PTS-120 | C | ORDINARY | COUNT; KNOWN(2026-11-17 Future 4) | `gpu-vecchia-log-likelihood.ts:69`, `gpu-vecchia-fisher-scoring.ts`, `cpu-vecchia-fisher-scoring-operation.ts:44` | Neighbour capacity is `neighborCount` in 9 classes and `maximumNeighborCount` in GPUKriging. A conditioning set wider than the capacity gets nearest-k (likelihood), first-k slots (Fisher kernel) or all slots (CPU twin). | Rename; reject or diagnose over-width rows. |
| PTS-121 | C | ORDINARY | SOURCE | `gpu-max-p-regions.ts:177-189` | No "budget reached" word for construction or improvement; a truncated construction reads as infeasible; inner AZP convergence flags are not forwarded. | Add `constructionLimitReached`, `improvementConverged`. |
| PTS-122 | C | ORDINARY | SOURCE/COUNT | `docs/api/gpu-spatial-analysis/point-process.md:13,117,119`; `space-time-clustering.md:67` | OPTICS docs say 65,536 grid rows and batch default 64; source is 2^20 (`gpu-optics.ts:66`) and 256 (`:88`). Xi stays 65,536. | Rewrite three passages. |
| PTS-123 | C | ORDINARY | SOURCE | `operators-statistics.md:1656,1660`; `space-time-clustering.md:122` | HDBSCAN docs say 2^20 rows and a `GPUNeighborSearch` core-distance route; source is 2^22 and the BVH route (`gpu-hierarchical-density-clustering.ts:102,154-158`). | Correct both. |
| PTS-124 | C | ORDINARY | EXECUTED (scikit-learn 1.6.1) plus SOURCE; KNOWN(DEC-HDBSCAN) | `gpu-hierarchical-density-clustering.ts:164-166,288-290` | Omitted `minimumSamples` defaults to `min(minimumClusterSize, 32)` while the doc says scikit-learn defaults. Three blobs, `min_cluster_size` 100: 243 noise rows at 100, 167 at 32; ARI 0.794. | Reject the omitted value above 32 or state the divergence. |
| PTS-125 | C | ORDINARY | EXECUTED-CPU | `gpu-location-change-points.ts:146-148,498` | "Partition does not depend on unit" is false for mean and slope costs (penalty is in squared data units): series `[0,.1,-.1,0,1,1.1,.9,1]`, penalty 2: unit x1 gives 0 changes, x10 gives 1. | Limit the sentence to variance, mean-variance, Poisson. |
| PTS-126 | C | UNUSUAL | SOURCE | `gpu-optics.ts:913-915,810` | Grid-route candidate overflow goes to a transient unless `candidateOverflow` is bound; named ports then read zeros and the labels are all noise. | Require the port when any status port is bound, or add a reason word. |
| PTS-127 | C | ORDINARY | COUNT | `docs/api/gpu-spatial-analysis-cross-reference.md:99` | Dorling "at most 1024 circles/256 passes"; code admits 0x3ffffffe rows on the grid route and 65,536 all-pairs (`gpu-dorling-cartogram.ts:58,60`). | Copy `coverage-matrix.md:231`. |
| PTS-128 | C | ORDINARY | COUNT | `operators-statistics.md:1420,1434,1448,1194`; Census table | AZP, Max-p, Ward and location-allocation have no own heading; 16 operation forms and 4 point-process CPU twins are not named in `docs/api`. | Add sections. |
| PTS-129 | C | UNUSUAL | EXECUTED-CPU | `gpu-huff-trade-areas.ts` row loop | Weights `[-1,1]` give `p=[0,1]`: slots with weight <= 0 are silently non-candidates (fix behaviour change). | Document. |
| PTS-130 | C | ORDINARY | COUNT | `gpu-catchment-accessibility.ts:55`; consumers `examples/deck/spatial-analysis-showcase/.../trade-areas.compute.ts:392`, `.../trade-areas-mode.ts:387`, `modules/lenses/.../catchment-accessibility.lens.ts:127` | `reachableFacilities` can hold sentinel `0xffffffff` (unknown) and consumers sum it. | Separate validity flag or handle the sentinel in consumers. |
| PTS-131 | C | UNUSUAL | SOURCE | `kernel-density-math.ts:250`; `kernel-density-binned.ts:296` | Binned KDE error bound holds only for bandwidth >= 3 cells; no status when below. | Add a status word. |
| PTS-132 | C | ORDINARY | SOURCE | `cpu-neighborhood-summary-operation.ts:79` | CPU twin returns a median above `maximumNeighbors`; GPU returns NaN and sets overflow. | Follow the cap or document. |
| PTS-133 | C | EXTREME | COUNT | see Census (props not snapshotted) | Props mutated after construction are re-read by 9 of 13 space-time classes, 16 of 18 interpolation classes, Max-p and others. | `snapshotConstructionFields` everywhere. |
| PTS-134 | C | ORDINARY | SOURCE plus external doc; KNOWN(CUBE-OCCUPANCY) | `gpu-emerging-hot-spots.ts:628-635`; `space-time-hot-spots-recipe.ts:791-802` | Lattice mode treats every bin as valid; the Esri cube page says only locations with data in at least one step are included. Never-occupied cells lower the mean and enlarge the FDR family. Not in `recipes.md`. | Ever-occupied mask, default on. |
| PTS-135 | C | ORDINARY | SOURCE | `gpu-emerging-hot-spots.ts:222` vs `:66` | Comment says 1024-bin blocks; `BIN_BLOCK = 4096`. | Fix comment. |
| PTS-136 | C | ORDINARY | SOURCE | `operators-statistics.md:1578`; `gpu-spatial-kmeans.ts:256-262` | Docs say `kmeans++` matches scikit-learn; scikit-learn uses greedy k-means++, the GPU one D^2 draw per round. | Reword. |
| PTS-137 | C | ORDINARY | COUNT | Census | 35 clustering, 34 space-time, 39 interpolation and 33 regions throws without contributor ID or a named limit; ripley/indices/marked 19 throws bypass the helpers. | Use named-limit helpers in the constructors; CPU-twin messages need IDs. |
| PTS-138 | C | UNUSUAL | SOURCE; LIMIT | `gpu-ripley-distance-functions.ts:174-189,443-450,469` | Hanisch G for points with no neighbour within `maximumDistance` is biased (own doc example 0.44 vs 1); root cause is the capped 3x3 nearest search. | Exact ring search (as in the indices). |
| PTS-139 | T | ORDINARY | COUNT | density, space-time kriging, Vecchia specs | No spec at projected-metre magnitude (`rg '1e6|1e7|utm|translat'` finds Earth radii only; space-time kriging oracle uses +-100). | Translate one scene by (5e5, 5e6) per family and compare. |
| PTS-140 | T | ORDINARY | SOURCE | see Tests | Oracles written from the same reading: Hawkes, space-time K, quadrats (PTS-108), Ripley G/F, Max-p contiguity only at default budget, space-time KDE imports the kernel under test. | Independent fixtures. |
| PTS-141 | T | ORDINARY | COUNT | `gpu-spatial-kmeans.node.spec.ts:217-218`, `kmeans-oracle.ts:199-200`, `gpu-spatial-clustering.node.spec.ts:416-417` | Oracle generators cited (`fix/work/...`) are not in the repository. | Move scripts into `test/oracles`. |
| PTS-142 | T | ORDINARY | COUNT | `parity-wp1-emerging-hot-spots.parity.spec.ts:106` | All four external hot-spot parity specs run with `sporadicRequiresFinalSlice: false`; the ArcGIS default is pinned only by the in-repo oracle. A NumPy oracle written from the Esri page matched it on 80,000 series (S, z, p to 9.4e-8). | Commit that oracle as a fixture. |
| PTS-143 | T | UNUSUAL | SOURCE | `vecchia-fisher-scoring.spec.ts:139-212`, `vecchia-fisher-reference.node.spec.ts:151` | Fisher tests assert only `converged`; no budget, stall, failed or over-width case (PTS-114, 120). | Add. |
| PTS-144 | T | ORDINARY | SOURCE | `stx07-expectation-based-scan.spec.ts:531,551`; `marked-point` oracle `geospatial-expansion-statistics.py:177` | Guard that never fires; LLR compared to one decimal; duplicate-pair shell rule differs between GPU (shell 0) and oracle (`> lower`) and is never exercised. | Fix both. |
| PTS-145 | M | - | COUNT | `gpu-spatial-scan-statistic.ts` 1,973, `gpu-location-allocation.ts` 1,801, `gpu-emerging-hot-spots.ts` 1,731, `gpu-local-kriging.ts` 1,329, `gpu-optics.ts` 1,192 | Large mixed-concern files; local kriging also holds a test factory (`createLocalKrigingForTesting`, lines 345-349, 1306-1328). | Split as listed under Maintainability. |

### W and S findings: reproductions and probes

**PTS-101.** Probe `scratchpad/review19/spatial-points/density/nsum-probe.ts`. Input: CSR offsets `[0,1,2,2]`, neighbors `[1,0]`, weights `[1,1]`, values `[10,20,30]`, statistics count and mean. `overflow=0, candidateOverflow=0` gives table `[1,20, 1,10, 0,NaN]`; `candidateOverflow=1` gives the same table unchanged; `overflow=1` gives all NaN. Expected for the second: all NaN, as for the documented siblings. The lead confirmed the class declares `status?: Pick<..., 'overflow'>` only (`gpu-neighborhood-summary.ts:122`).

**PTS-102.** Probe `regions/maxp-repro.ts` (re-run by the lead). Input: adjacency `[[1],[0,2,6,7],[1,3],[2,4,5],[3],[3],[1],[1]]`, unit weights, values `[9,2,0,2,2,1,1,9]`, extensive `[1,1,1,3,2,3,3,3]`, threshold 8, seed 619. `maximumGrowthSteps: 2`, iterations 0: labels `0,1,1,1,1,1,2,2`, status `[3,0,2,0,11,0]`; region 2 is rows 6 and 7, both adjacent only to row 1 (region 1). With `maximumIterations: 4` the AZP check reports `invalidCount 1`, `regionCount 0`. Steps 3, 8 and 256 give contiguous output. At steps 1 the labels `0,4,1,3,2,3,4,5` also split a label (rows 1 and 6). Expected: contiguous labels and `invalidCount 0` on valid input.

**PTS-103, 104.** SOURCE: no read of loss flags in the cited files. GPU probe wanted: a 4-row path with `weights.status.candidateOverflow = 1` through AZP; 3 events within a radius from a `GPUNeighborSearch` with capacity 1 through `GPULocalKnox` against `GPUKnoxTest` (NaN expected for both).

**PTS-105.** Probe `interp/probe-variogram.ts`, re-run by the lead. Input: distances `[1..5]`, semivariances `[0.2,0.4,0.6,0.8,5.0]`, pairCounts of length 4, exponential, pairs weighting. Output: `{sill:1.161, range:12}` (first four bins); five counts give `{sill:3.855, range:15}`; the CPU twin throws `variogram-model-fit invalid model or bin topology`. Expected: a throw.

**PTS-106.** Probe `spacetime/stk-quantize.mjs` (re-run). Interior pair, equal relative intensity: 50 gives -0.8 %, 100 gives +6.8 %, 200 +22.1 %, 300 +37.3 %, 362 +100 %, 363 and above -100 %. TRANSCRIPT: GPU probe wanted (two events in a window of mean intensity 1, relative intensity 400; read K at the largest radius; expected nonzero).

**PTS-107.** Probe `spacetime/baseline-scale.mjs` (re-run): `["1","1","NaN","1"] -> [0,0,1.5,0]`, Infinity `[0,0,1,0]`, a negative `[1,1,1,1]`. TRANSCRIPT; GPU probe wanted with a NaN baseline cell; expect the invalid word set.

**PTS-108 and PTS-109** are as in the table; probes `process/polygon_quadrat.py`, `process/quadrat_boundary.py`.

**PTS-110 to PTS-112** are SOURCE with the GPU probes listed in the table.

## Refuted or downgraded

- DBSCAN grid-cell seam misses: only at epsilon 0.0014 with |x| about 9.2e5, in the regime the doc calls unreliable; 16M UTM-like pairs gave 0.
- CPU OPTICS against scikit-learn: 59 of 60 exact, the 60th differs at a reachability near-tie of 6e-15.
- Pettitt (2,000 series) and change-point CPU (400 series, 12 ruptures cases): 0 mismatches.
- Clip-slot overflow in translation overlap: at most 6 of 8 slots over about 2,400 shifts, areas match the closed form.
- Balanced-sample inclusion and Philox stream collision: frequencies match n/N within 0.01; counters disjoint.
- Rectangle isotropic weight: matches quadrature to 1e-6 in six cases.
- f32 support centre in local kriging: 0.5 to 1 m at 5e6 to 9e6 equals one ulp of the input positions.
- Space-time kriging indexed against full scan: same total order and metric; no disagreement found.
- Variogram and coregionalization fit budgets: CPU and GPU consistent (best-of, status 3 or limit flag).
- Huff attractiveness `[3.4e38, 3.4e38]`, weights `[3e38, 1e38]`: `p=[0.75,0.25]`, correct.
- Mann-Kendall, tie term, Theil-Sen, 17 categories, barrier uniformity and private-array size (about 4.7 KB) in emerging hot spots: clean. Monte Carlo p-values `(1 + #{>=})/(P+1)` in scan, Knox, Mantel, Jacquez, Diggle and the Ripley envelope: consistent.
- Kriging compatibility table: every row matches the code. Two differences are missing from the table: GPUKriging's default minimum of 3 distinct neighbours (`kriging-parameters.ts:102`) against none in local kriging, and unequilibrated against equilibrated pivot tests.
- Border K, isotropic K, the rank-envelope midrank test and the Ripley envelope `max |L - r|` statistic: formula checks by the lead found no error. Note: the envelope's maximum deviation ranges over finite radii only (`gpu-ripley-envelope.ts` deviations node), so replicates with undefined border K at large radii compare over a shorter range; documented as NaN handling, not graded.

## Tests

Independent oracles exist for: fixed KDE (scipy, sklearn), similar locations, neighbourhood summary (libpysal fixture, sha-checked), analysis spacing, Huff (hand-computed), kriging (gstat JSON, SciPy, GSTools), OPTICS (scikit-learn generators in repository), HDBSCAN (scikit-learn, regenerated bit-identical), SKATER (spopt), MST and location-allocation (SciPy MILP, exhaustive), Ward (scipy), Sinkhorn (the new two-cell case and the reference), translation pair correlation (spatstat, Shapely), indices on rectangles (pointpats on 105,808 projected points), CPU Ripley envelope (closed form).

Discriminating-test rule in this scope: PTS-01, 02, 03, 04, 08, 09, 10 and X7 fixes each have a test whose wrong version fails. Counter-examples: PTS-107 was introduced by a fix whose test used finite values only; DOC-01's test pinned the twin but not the public function.

Five weakest-tested contributors (merged across slices):
1. `GPUSpaceTimeHawkes`, `GPUSpaceTimeKFunction`, `GPUExpectationBasedSpaceTimeScan`: in-repo oracles from the same model; K-function intensities drawn only in 0.5 to 2 (hides PTS-106).
2. `GPUPointPatternIndices` polygon quadrats and `GPURipleyDistanceFunctions` KM and Hanisch: the oracle mirrors the drop rule (PTS-108); no spatstat Gest/Fest/Jest fixture; no edge-point case (PTS-109).
3. `GPUMaxPRegions` and `GPUAZPRegions`: contiguity only at the default budget; no weights-loss spec; quality only as a ratio to spopt.
4. `GPUVecchiaFisherScoring` and `fitVariogramModel`/`fitSpaceTimeVariogramModel`: converged path only, no input-contract tests, no gstat fixture for the space-time fit.
5. `GPUSpatialKMeans`, `GPUSpatialClustering`, `GPUHierarchicalDensityClustering`: in-house or missing generators (PTS-141), no CPU twin, no test of the default above 32. `GPUContinuousSpatialSample` has no work-ceiling or large-coordinate test either.

Also weak: `GPUAdaptiveBandwidthKernelDensity` (one binary64 reference from the doc), `GPUSpaceTimeKernelDensity` (reference imports kernel under test), `GPUBoundedFacilityAllocation` (feasibility only), `GPULineDensity` (self-written oracle, no antimeridian case), `GPULocalKnox` and near-repeat (no incomplete-pairs case).

## Maintainability

- Duplicates: Huff and catchment each carry a loss-flag collector (`gpu-huff-trade-areas.ts:586`, `gpu-catchment-accessibility.ts:489`) instead of `spatial-weights-loss.ts`; `gpu-point-pattern-indices.ts:82-191` has its own incomplete gamma and normal p-value beside `analysis-core/kernels/incomplete-gamma-wgsl.ts` (a third log-gamma in `change-detection-wgsl.ts:36`); `gpu-space-time-cross-correlation.ts:247-253` has its own erfc; adaptive KDE copies fixed KDE's bounds block (`gpu-adaptive-bandwidth-kernel-density.ts:420`); `polygon-overlap-metric.ts:13-91` rebuilds `orientExact` terms.
- Kriging pair: GPUKriging and GPULocalKriging (DUP-01) differ in neighbour pipeline and pivot policy; blocked Cholesky in `conditional-simulation-blocked.ts` belongs in analysis-core. Three k-means (`GPUSpatialKMeans`, `GPUKMeans`, `GPURasterEmbeddingKMeans`, DUP-02). Three facility solvers (DUP-01 rest). Two snapshot helpers in regions.
- Deprecated or dead: `GPUCrossVariogram` plus `legacy-variogram-fallback.ts` (212 lines) and the `legacyBins` branch (`empirical-variogram-nodes.ts:56`), five consumers in lenses, two in examples, five in website; test-only mirrors `azp-exact-reachability-legacy.ts` (913) and `legacy-pair-swap.ts` (284); `GPU_OPTICS_DEFAULT_FRONTIER_BATCH_SIZE`, `GPU_NO_NEAREST_NEIGHBOR` and `CPU_POINT_PATTERN_ENVELOPE_MAXIMUM_VALUE_COUNT` unreferenced outside barrels; an unreachable Wilson-Hilferty branch kept for byte-identical shader text (`gpu-point-pattern-indices.ts:108-113,180-185`); five exports used only in their own file (`cpu-vecchia-evaluation.ts`, `prediction-covariance.ts`).
- Shims: four density importers still use `shared/point-neighbor-grid-index` while others import the core file; four density files get `GRID_COORDINATE_WGSL` from `gpu-inverse-distance-weighting`, which only re-exports the luma.gl constant.
- Splits proposed: swap phase of location-allocation to its own file; WGSL builders of emerging hot spots (`:1440-1731`); bandwidth rule out of `gpu-kernel-density.ts` (1,395 lines, about 200 lines); grid frontier out of `gpu-optics.ts`; test factory out of `gpu-local-kriging.ts`.
- Internal switch in public props: `orderingMethod: 'parallel'` in OPTICS (opt-in, no measured advantage), `legacyIndex: true` in indices.

## Roadmap rows

| Row ID | Verdict | Evidence | Replacement text if NARROW |
|---|---|---|---|
| PP01 / PP02 | NARROW | Border, isotropic (rectangle and polygon Ripley), exact polygon translation K and pair correlation exist; extreme-rank envelope uses midranks and reports a [strict, conservative] p interval (`gpu-point-pattern-rank-envelope.ts:46,76-79`). | "Border and translation K, smoothed pair correlation and the extreme-rank envelope (midrank, p interval) exist to 4,096 events. Open: inhomogeneous isotropic correction and an extreme-rank-length tie breaker." |
| PP05 | NARROW | Balanced sample and continuous sample sources. | "`GPUSpatialBalancedSample` (finite frame) has no replacement sites or design-variance estimator. `GPUContinuousSpatialSample` provides complete independent replacement waves only; no response-dependent substitution or variance. Neither is GRTS." |
| IN01 | NARROW | Timings predate cooperative refit. | "Resident variogram fitting is kept; its 2026-10-23 timings predate the parallel candidate and cooperative refinement and must be re-measured before any speed claim." |
| IN02 / IN05 | KEEP | Degree -1 to 2; 33 rejected (`gpu-local-kriging.ts:375`). | |
| IN04 | KEEP | Caps of 4 (`coregionalization-parameters.ts:30,32`). | |
| IN06 | KEEP | | |
| KG02 | NARROW | Dorling status words and the 256-pass budget. | "Heuristic by contract, no GeoDa coordinate parity. Budget 0 to 256 passes (default 100). `output.status` reports passes used, converged, budget exhausted, excluded rows. Open: a GeoDa parity fixture; no convergence guarantee planned." |
| STX-08 | KEEP | REDCAP still cuts the k-c heaviest edges (`gpu-redcap-single-linkage.ts:137-161`); CPU SKATER matched brute-force single-cut SSD on 3,000 trees. | |
| STM-07 | KEEP, add | GpGp still pending. | Add: "Fisher fit has no stall or budget test (PTS-114, 143)." |
| PLAT-03 | NARROW | Slices agree. | "Fixed, adaptive and space-time KDE, DBSCAN, marked point statistics and cross variograms use `analysis-core/kernels/point-neighbor-grid-index.ts`. OPTICS uses `GPUNeighborSearch`, HDBSCAN its own BVH walk, k-means scans at most 256 centres, `GPUPointPatternIndices` still uses `legacyIndex`. Share only where eligibility, distance and tie order and overflow agree." |
| GPU-05 | KEEP | KDE exact gather has no work estimate; adaptive KDE searches up to 16 times the radius; geographic scan prunes latitude only (PTS-06 remainder). | |
| FX-03 / FX-05 | NARROW | `sporadicRequiresFinalSlice` (`emerging-hot-spot-parameters.ts:162,239`). | "FX-05 only: local G/G* permutation on negative data publishes GPU invalid status; decide host rejection or a kepler diagnostic." (FX-05 not re-checked.) |
| CUBE-OCCUPANCY | REPRIORITISE | PTS-134, Esri cube page. | To an S item: "Lattice space-time hot spots count never-occupied cells as zero-valued locations; ArcGIS includes only locations with data in at least one step. Add an ever-occupied `cellMask`, default on, with an opt-out." |
| DEC-HDBSCAN | KEEP | PTS-124; a 256-entry private top-k is about 2 KiB, below the 8,192-byte limit. | |
| dataframe-10 | NARROW | Diggle statistic is splancs' `sum D` (`gpu-space-time-diggle-test.ts:206-257`). | "Accept the licence of the MIT-headed NumPy splancs port; decide optional splancs tie rules for `GPUSpaceTimeKFunction` (measured U 43.95 vs 52.26)." |
| DUP-01 | NARROW | | "Merging GPUKriging into GPULocalKriging needs a compatibility design (duplicates, exact hits, minimum distinct neighbours, pivot policy, distance models). `GPUCrossVariogram` is deprecated; remove it and `legacy-variogram-fallback.ts` after the alpha. Remaining: three facility solvers." |
| DUP-02 | KEEP | Add `GPURasterEmbeddingKMeans`. | |
| STD-10 (interpolation) | CLOSE | GPUKriging uses `getDenseLUWGSL`; local kriging, RBF, space-time kriging and Vecchia use `equilibrated-solve-wgsl.ts`. | |
| STD-04 / STD-09 (OPTICS) | NARROW | OPTICS ships `output.ports.status` (`gpu-optics.ts:175-182`). | "Decide `output.ports.status` for depression hierarchy and network KDE; for OPTICS only whether the packed [visited, complete, invalid] status stays; whether `validateSinglePackedView` is public." |
| STD-14 | NARROW | | "Kernel density accepts chunked input only with `method: 'binned'`; exact, adaptive and space-time KDE, relative risk, line density, neighbourhood summary, Huff, catchment and similar locations are packed-only." |
| STD-07 / STD-15 | KEEP | Iteration-budget names: `componentIterations`, `maximumSwapIterations`, `maximumGrowthSteps`, `maximumIterations`. | |
| Known limit: space-time variogram | NARROW | | "`GPUSpaceTimeVariogram` and `GPUVariogram` require `gridSize` (GPUCrossVariogram defaults it); parameters are eight floats; the CPU fit has no convergence flag (PTS-116)." |
| Known limit: space-time kriging | KEEP | 16-byte observations at 128 MiB about 8.4M; scan cap 1,048,576. | |
| Known limit: OPTICS xi | KEEP | `gpu-optics.ts:69,421`; pinned by `optics-parallel-frontier.node.spec.ts:67`. | |
| Known limit: space-time K / Diggle | KEEP, add | | Add: "Inhomogeneous weights quantise at 2^-16 (PTS-106) until fixed." |
| SIM-10, LRN-03 | KEEP | Classes absent from the tree. | |
| LUM-30, AX-40, PLAT-06 | KEEP | `gpu-analysis-spacing.spec.ts:155,220`; ETAS still has no fixture. | |
| New row | ADD | PTS-117 | "Great-circle kriging and IDW drop samples across the antimeridian (query box cut at +-180)." |
| STD-13 | KEEP | Dorling part is the stale cross-reference (PTS-127). | |
| GPU-10 | KEEP | Three facility solvers remain. | |
| GAP-04 | KEEP | All four clustering classes take `float32x2` positions only. | |

## Next steps

### Before the alpha

1. Weights-loss propagation in neighbourhood summary, AZP, Max-p, partition evaluation, local Knox, near-repeat Knox and emerging hot spots, using the shared helper, with a two-submission set-and-clear test each (PTS-101, 103, 104). S, needed by every kepler and lens consumer of those operators.
2. Max-p contiguity at small budgets and status words (PTS-102, 121); `fitVariogramModel` input check and status (PTS-105, 113, 116); Fisher stop reason (PTS-114). S.
3. Quadrat fixes with an independent-area oracle and a lattice-edge test (PTS-108, 109); scan baseline validity (PTS-107); inhomogeneous space-time K scale (PTS-106, M); continuous-sample budgets (PTS-111, 112); Huff non-finite (PTS-110). S to M.
4. Documentation: OPTICS, HDBSCAN, Dorling, regionalizer headings, the 20 operation forms and twins, nugget, validity, change-point units, the antimeridian notes, kriging compatibility rows (PTS-115, 117 to 119, 122 to 125, 127, 128, 136). S.
5. Translated-scene (5e5, 5e6) tests for density, space-time kriging and Vecchia, and committing the missing oracle generators and the hot-spot NumPy oracle (PTS-139, 141, 142). S.
6. CUBE-OCCUPANCY mask default on (PTS-134). S.

### After

- Finish: one-invocation-per-candidate continuous sample (M); HDBSCAN cap to 256 if DEC-HDBSCAN agrees (M); geographic scan pruning on both axes (M); antimeridian query wrap for kriging and line density (M).
- Connect: promote DBSCAN and k-means oracles to public CPU twins (M); a CPU twin for bounded facility allocation (S-M); external fixtures for ETAS, SaTScan prospective, stpp, GpGp, spatstat Gest/Fest (M each); sparr for space-time KDE and relative risk (M).
- Consolidate: shared incomplete-gamma and grid builder in the indices (M); move the blocked Cholesky to analysis-core (M); merge the facility solvers (L, DUP-01); settle the Vecchia conditioning-set contract and rename capacity props (M).
- Remove or freeze: remove `GPUCrossVariogram` and the legacy fallback; freeze OPTICS `orderingMethod: 'parallel'`, the legacy Sporadic hot-spot rule, relative risk (until it has an oracle), the balanced sample (reference tier, correct), Dorling (until a GeoDa fixture), and the GPUKriging to GPULocalKriging merge. Deprecate `logLikelihood` in favour of `fittedLogLikelihood`.

## Limits of this report

- No GPU result was observed. PTS-103, 104, 108, 109, 110, 111, 112, 117 and the PTS-106/107 transcriptions are source traces or Python/JavaScript transcriptions of shader arithmetic; each lists a GPU probe.
- Not opened: HDBSCAN WGSL and tree modules; k-means seeding and relocation kernel bodies; Ward WGSL, `ward-active-graph.ts`, `ward-mutual-selection.ts`; `gpu-region-partition-evaluation.ts` kernels; Dorling relax and grid kernels; `location-allocation-sparse.ts`; `gpu-natural-neighbor.ts` body; `gpu-focal-statistics.ts`; `gpu-gaussian-random-field.ts`; conditional-simulation internals; `vecchia-conditioning-sets.ts`, `vecchia-likelihood-nodes.ts`, the Vecchia space-time contributor; `gpu-point-density.ts` and tiles; `analysis-spacing-statistics-nodes.ts` rest; `scan-statistic-nearest-zones.ts`, the expectation-based scan WGSL beyond its diff; the Diggle WGSL kernels (`:330-768`); the Hawkes parameter and operation files; Pettitt and Ripley spec bodies; most spec bodies in every slice; the study-window WGSL; ontology record contents.
- Sampled only: `gpu-spatial-scan-statistic.ts` (about 40 percent), `gpu-local-kriging.ts` lines 760-860, `gpu-kernel-density.ts` (about 95 percent), adaptive and space-time KDE, `gpu-ripley-envelope.ts` (summary and simulation nodes only), `gpu-huff-trade-areas.ts` (265 onward sampled).
- Counts in the census depend on name matching and `rg` word boundaries; the 75-class figure is for unique GPU class names excluding the pair-statistics classes; the "not in ontology" list was not re-derived by the lead.
- Probe scripts are in `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/56cea564-9b4b-472f-a337-15a6b37b7a0b/scratchpad/review19/spatial-points/` (subfolders density, clustering, regions, interp, spacetime, process); they are not part of the repository.
