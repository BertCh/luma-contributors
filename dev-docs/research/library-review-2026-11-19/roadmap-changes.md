# Roadmap changes applied from review reports

## area-dataframe-pointcloud-record.md

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| KI-01, KI-03 | none (known-issues rows) | dev-docs/roadmap.md:80,82 | KI rows are not edited. Report: KI-01 KEEP; KI-03 contribution is a source re-review of named join/forest fixes and fourteen color graph declarations, device qualification open. |
| KG03, KG06, LN-DQ, LR-22, GPU-13, LUM-31, PC01, PC05, GPU-05, PLAT-03, IN04, IN01 limit, space-time variogram/K limits, CUBE-OCCUPANCY, LN-ENC, INT-01, CPU-05, UPS-02 | none (KEEP) | not re-checked | report verdict KEEP |
| DF-DESC | narrowed | node_modules/@luma.gl/graph/src/gpu-dataframe/gpu-data-frame-query.ts:77-85, gpu-expression.ts:66 | |
| ARROW-01 | narrowed | analysis-core/graph-support/arrow-analytics-normalization.ts:126-175, arrow-analytics-wide-numeric.ts:12 | Dropped the report's "track option-map correctness separately" and "name the next type" (instructions) |
| DEC-ARROW-CONSUMERS | narrowed | dev-docs/upstream-workarounds.md (Arrow row), arrow-analytics-time-upload.ts:40 | |
| LR-04 Arrow requirement | none | modules/analysis/package.json:95 | Row already says the required Arrow peer exists |
| GPU-01 inequality | none | gpu-inequality.ts | Row already says default inequality statistics/Lorenz are parallel |
| KJ-01 | none | roadmap.md:534 | No row exists; closure already logged in the LR-16 line |
| GAP-04 dataframe/window clauses | not applied | gpu-key-join.ts:170-190 | Roadmap row has only "windows, group regression and as-of joins"; report text names SQL windows, ranking, lag/lead, which the row does not; kept as is (key join is equality join confirmed) |
| PC02 | narrowed | gpu-pointcloud-statistical-filter.ts:76-84 | |
| PC03 | narrowed | gpu-pointcloud-height.ts:60-70,111-125 (no nearest or Delaunay) | |
| GAP-05 point-cloud clause | narrowed | pointcloud-analysis/voxel-merge/voxel-means.ts:6; docs/api/pointcloud-analysis.md:19 | |
| STD-04 / STD-09 | not applied | gpu-pointcloud-neighborhoods.ts:279 | Status-view counts (nine, three, six, four) not confirmed by a bounded check; the report's text addresses a different decision than the roadmap row (depression hierarchy, network KDE, OPTICS) |
| STD-05 | narrowed (stronger than report) | gpu-pointcloud-neighborhoods.ts:52,108-110; gpu-pointcloud-height.ts:49,111-113; grep of getGPUPointCloud*ParameterValues across operations | Row wrongly said point-cloud neighbourhood/height/surface settings stay baked |
| STD-07 / STD-15 | not applied | gpu-key-join.ts:184 | "All 47 producers use getCommandNodes" and the naming list not confirmed; decision row |
| STD-10 | none | gpu-pointcloud-features.ts:239 | Row already says features and ICP use the shared Jacobi |
| STD-14 / Chunked input | not applied | gpu-group-statistics.ts:348 | Eighteen/thirteen/four producer census not reproducible in a bounded check |
| DUP-01 cross-variogram | none | gpu-cross-variogram.ts:247 | Row already says it forwards to shared kernels |
| Temporal reduction (limit) | narrowed | temporal-reduction/gpu-temporal-reduction.ts:60,67,218,257,402-406 | 16,777,216 assumes a 128 MiB binding limit (getStorageBindingElementLimit(device, 8)) |
| CPU arithmetic (limit) | not applied | cpu-inequality-operation.ts:159 | Row already states the limit; report text is a measurement instruction; DPC-109/124 counterexamples not examined |
| CPU budgets (limit) | narrowed (part) | pair-statistics/variogram-model.ts:87-100 | Coregionalization sweep-budget clause not applied: cpu-coregionalization-fit-operation.ts:49 says sweep caps are dropped |
| LN-REC | not applied | computation-record/describe-command-graph.ts:82,457; create-kernel-annotation.ts:77 | Invocation-specialization and mixed-pass identity defects not confirmed from these lines; report text is a fix instruction |
| LUM-30 | none | submit-timed-command-graph.ts:55 | REPRIORITISE |

### Decisions and proposed rows

- LUM-30: REPRIORITISE to before alpha for the scoped case: an allocated query survives injected encoder failure (`submit-timed-command-graph.ts:55`), small reproducer.
- LR-04: report proposes closing the Arrow-requirement clause; the roadmap row already lists the required peer as done, so nothing to delete.
- STD-04/STD-09 and STD-07/STD-15: report proposes status-inventory and naming decisions (see report table); needs the maintainer.
- KI-03: report suggests recording a source re-review of named join/forest fixes and fourteen color graph declarations (not edited).

## Fixes made on 2026-11-19

| Row ID | Fix | Roadmap action | Source checked |
|---|---|---|---|
| STD-06 | Terrain classes reject a sheared grid at construction (`validateRasterGridOrthogonal`); orthogonal grids unchanged on the GPU; walking cost not yet covered | narrowed the shear clause | `rg -l validateRasterGridOrthogonal modules/analysis/src/gpu-terrain modules/analysis/src/raster-analysis` lists 23 files (17 `gpu-terrain` class files, helpers, cost distance and a raster ridge class); spec modules/analysis/test/gpu-terrain/terrain-grid-shear.node.spec.ts exists. The lead's count of 15 classes was not reproduced exactly, so the row names no number. |

## area-tooling-tests-release.md

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| LR-04 | sentence added (TOOL-102 stated as open) | modules/analysis/dist/**/*.d.ts scan: 20 extensionless relative specifiers in 10 files; modules/analysis/package.json:95 | Report says 22 in 12; the fix is in progress, so the row states the defect as open and gives no closure |
| UF-10 | narrowed | modules/analysis/package.json (publishConfig, private flags in lenses/kepler); dev-docs/contributor-contract.md, release-package.md exist | |
| UF-12 | narrowed | .github/workflows/test.yml:59-65; examples/deck/spatial-analysis-showcase/scripts/check-sprint-scenes.mjs:12; scripts/playwright/ holds only get-playwright-launch-options.mjs | |
| PLAT-07 | sentence added | modules/analysis/test/gpu-spatial-analysis/spatial-join/spatialbench-scale-benchmark.spec.ts:28-40 | |
| LUM-30 | narrowed (text only) | modules/analysis/test/analysis-core/gpu-resource-counts.spec.ts:26-52 | REPRIORITISE is a decision, see below |
| Published fp64 | sentence added | modules/analysis/test/analysis-core/shader-fp64-module.spec.ts:21-33 | |
| INT-04 | closed | test/kepler-compute-example-specs.json: no "alpha.10", "physical" or "transient" text; no alpha.10 in modules/kepler/test | |
| LR-19 | not applied | dev-docs/release-package.md:179-208 | Table shows peak RSS unavailable; the report's 2.15/3.52 GiB peaks not found there |
| UF-03 | not applied | none | Claim that no alpha.16 baseline exists is not checkable in bounded time; replacement is an instruction |
| PLAT-06 | not applied | scripts/oracle-fixtures.json (290 generators) | Counts 337/790/188/453 not reproduced |
| KE-TESTS / KE-PORT-6 | none | test/kepler-compute-example-specs.json: 1 runtime and 58 typecheck exclusions confirmed | Row already derives counts from the manifest |
| KE-PORT-0, STD-12, LR-22 / KI-04, PLAT-04/05 | none | | Row already accurate or replacement is an instruction or decision; LR-22 already holds the round-count finding |
| LR-02, INT-08, LR-17, LR-20, LR-26, KI-01/03, SH-DATA, dataframe-10, COLD, Lag ML, GPU triangulation, urban tessellation, IN01 | none | | KEEP |
| UF-01 | not applied | | REPRIORITISE |

### Decisions and proposed rows (tooling report)

- UF-01: raise to H for publication; the maintainer decides canonical public identity (package.json and README say visgl, checkout remote and website header say BertCh).
- LUM-30: also REPRIORITISE (from the dataframe report).
- PLAT-04/PLAT-05: link the existing contributor-contract rules (compensated sums, pinned copies) from AGENTS.md; keep upstream filings and device qualification as separate actions.
- LR-04: report headline is to hold the alpha until TOOL-102 is fixed and a native (`.cts`) consumer check is added; also TOOL-104, 105 and hash drift.

## area-roadmap-devdocs.md (first pass; rows contested by earlier reports or by work in progress are left)

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| NW05 | narrowed (report right; roadmap wrong) | gpu-network-facility-optimization.ts:43-51 and 95-155: `validateMaximumCount(... 'demand rows' ..., MAXIMUM_DEMAND_COUNT)` runs for every operation, then the candidate count (2^F or F^D) is limited to 2^20 | The earlier edit said the 64-row ceiling applies only to the other operations |
| STD-09 | narrowed | analysis-core/index.ts:46; modules/analysis/package.json:69 (`./analysis-core` export) | `validateSinglePackedView` is public |
| STD-04 (domain row) | narrowed | gpu-terrain-depression-hierarchy.ts:85-90, gpu-network-kernel-density.ts:50-56 (`ports?`) | Decision row text asked to add `ports.status` that exists |
| STD-04 / STD-09 (decision row) | narrowed (factual part only) | same | Decision about legacy views kept for the maintainer |
| STD-13 | narrowed; dead Source path expanded | ontology/kepler-analysis-catalog.ts:48-52 (Dorling limit text correct); `modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/` exists | |
| ROSE-01 | narrowed | docs/api/gpu-spatial-analysis/operators-statistics.md:1001 | Docs describe both origins; file is clean in git |
| SH-NAMES | narrowed, not closed | examples/deck/*/package.json:2-3 | Names fixed; two versions still 10.0.0-alpha.10, so the row keeps that |
| INT-06 | closed (lens row; lead may reverse) | modules/lenses/src/.../local-permutation-test.lens.ts:22,24 import `@luma-contributors/analysis/gpu-spatial-analysis` | |
| LR-10 | not applied | | Report says randomization validation moved in the working source; fix in progress, not told it is done |
| KE-PORT-6, INT-01 / LN-REC2, LN-ST, INT-09, INT-07, INT-08, UF-03 and remaining rows | not applied | | Kepler/lens/website rows (work descriptions left alone), merges into UF-01 are structural, or not yet checked |

Remaining NARROW/CLOSE rows in this report (ST10, ST11, R6.4, RS11, A-18, LUM-24, EMB-01, SIM-06, CART-07, MOB-10, GAP-02, FP64-02, EC04, CPU-04, CPU-03, GE01, COLD, Sky view, Lag ML, Panel RE, Panel error ML, CPU output tails, Network flow, FP64-04, Space-time variogram, UPS-CPU-1, ...) are still to be processed in a second pass.

### Structure proposals (for the lead; not applied)

- Proposed four-section table of contents and split files (limits.md, kepler/lenses/website roadmaps): area-roadmap-devdocs.md:359.
- Eight component IDs denote both implementation and decision records; final two KE-WF workflow rows sit outside the rendered table: ROAD-110, findings table at line ~120.
- 31 cells of 50 or more words, consumer stages and pass histories competing with release work: ROAD-114.
- Four committed rows no longer present (NW02, GEOM-01, GEOM-02, LUM-23 closed): line ~660.
- Reconciliation test pins phrases (ROAD-112): roadmap-reconciliation.node.spec.ts needs reviewing if rows are removed.
- Dead Source links: only `ontology/all-entry-points/` (fixed). KE-PORT-1/5/SAMPLES paths are example-relative and exist.

### Fixes made on 2026-11-19 (running list, continued)

| Row ID | Fix | Roadmap action | Source checked |
|---|---|---|---|
| LR-10, STD-11 | 13 urban/landscape recipes validate before first `graph.add`; segregation decomposition validates `randomization` first; not on `addPreparedRecipe`; overlay device-limit residual | narrowed, kept visible as "fixed in the working tree, GPU specs not yet run" | spec urban-landscape-recipe-preflight.node.spec.ts exists; segregation-decomposition.ts modified; fixes/fix-urban-recipes.md |
| Visibility graph (limit) | spatial CSR publishes an empty row for an invalid viewpoint; gpu-network pivots still withhold the graph | sentence replaced | urban-morphology/visibility-graph-csr.ts:476-493 (invalid pivots via `additionalInvalidCount` gate) |
| STD-04 | weights-loss consumers updated; open list kept; old counts removed | sentence rewritten | fixes/fix-points-weights-loss.md, fix-stats-weights.md, fix-geometry.md (status files, taken from the coordinator's message; consumers' code not re-read) |
| Network cost completeness | `GPUNetworkCostWeights` and `GPUNetworkAccessibility` take `costsComplete` | none: no row or limit says cost-matrix consumers lack completeness (grep) | gpu-network-cost-weights.ts:128-157 |
| CR04 / CR05, FP64-03 | spherical containment ring orientation; point-to-cell longitude modulo | none: no row states the opposite | roadmap.md rows 180, 424 |

## area-roadmap-devdocs.md, second pass

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| ST10 | narrowed | spatial-validation/gpu-spatial-cross-validation.ts:51-100 (`design`, groups/NNDM/clustering) | |
| ST11 | narrowed | spatial-interaction/gpu-spatial-interaction.ts:70,103-108 | |
| R6.4 | sentence added | raster-analysis/gpu-raster-fractional-coverage.ts:158-164 | |
| RS11 | narrowed | raster-time-series/gpu-raster-bfast-monitor.ts:46,101,127 | |
| A-18 | narrowed | gpu-terrain-spike-repair.ts:234,963 | |
| GAP-02 | narrowed (breach clause) | gpu-terrain-breach.ts:56 (explicit candidate pits) | |
| MOB-10 | narrowed | gpu-radiation-flow-model.ts:42; gpu-kernel-flow-smoothing.ts:42,185 and doc comment | |
| SIM-06 | sentence added | raster-analysis/remote-sensing/gpu-raster-spectral-unmixing.ts:61 | |
| Network flow | sentence added | gpu-network-flow.ts:110-125 (`maximumCostPhases` selects primal-dual) | |
| CPU output tails | narrowed | cpu-network-path-extraction-operation.ts:70-80 | |
| CPU arithmetic / CPU budgets | sentences added | cpu-network-reachability-operation.ts, cpu-alternative-routes-operation.ts doc comments | |
| COLD | sentence added | rts04-cold-pyxccd.spec.ts:122-135 | LASSO/Tmask/partial-tail differences not added (not verified) |
| raster-fields-05 / STD-06 | sentence added | | count dated 2026-11-11 |
| FP64-04 | closed (accepted state, in the workaround register) | dev-docs/upstream-workarounds.md:234,308 | |
| Space-time K / Diggle | rewritten, marked fixed in working tree | space-time-k-function-nodes.ts:237,461 (`getIntensityWeightScale`); fix-points-process.md PTS-106 | |
| fitVariogramModel / fitSpaceTimeVariogramModel | none | variogram-model.ts:264, space-time-variogram-model.ts:496 reject unequal lengths; no row says they truncate | |
| LUM-24, EC04, CPU-03, CPU-04, GE01 / UR, NW-GA, LR-24, LR-26, PLAT-05, PLAT-04/05, Lag ML, Panel RE, Sky view | none | | Row already states the report's facts, or replacement is an instruction |
| Panel error ML, Space-time variogram | not applied (report: CLOSE → research/API) | | Both rows carry current limits; moving them to docs is a structural decision for the lead |
| LN-REC, LN-ST, INT-01 / LN-REC2, UPS-CPU-1, WEB-PIC-12, KE-PORT-6, INT-07/08/09, UF-01/03 | not applied | | Lens/Kepler/website rows or recounts not reproducible; INT-09 closure is a merge into UF-01 |

## area-network.md (deepened version)

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| NW05 | none | gpu-network-facility-optimization.ts:43-155 | Today's rewrite matches the new verdict; code confirms |
| NW02 / LR-25, NW03, NW06 / X04, GPU-10, UPS-02, PLAT-05, GPU-01, GPU-11, STD-10, STD-14, KI-01/03, WEB-PIC-12 | none | | KEEP |
| LR-10 network-barrel clause | none | | Row already says the spatial and network barrels are atomic |
| LR-22 network grid adoption | none | | LR-22 row already says shared grid WGSL is adopted |
| NW-GA | not applied | | Report's "individual adoption tests" text is an instruction; installed gpu-graph index not inspected |
| GAP-03 | narrowed (edge volumes, nearest facility only) | flow-aggregation/gpu-map-matched-edge-volumes.ts; network-analysis/gpu-network-service-areas.ts | Angular turn penalties in LineGraph and fractional routing not verified, so left in the open list |
| CPU arithmetic, CPU budgets, CPU output tails, Network flow | narrowed (see above) | | |
| STD-12, LR-23 / UPS-01, DUP-01, STD-04 / STD-09, STD-07 / STD-15, STD-11, PLAT-04, MOB-10 | not applied except MOB-10 (done above) | | Replacement texts are instructions or decision text, or rows already hold them |
| LR-D4 | not applied | | REPRIORITISE |

### Decisions and proposed rows (network)

- LR-D4 (line 532 decision): REPRIORITISE before alpha; dense propagation still has NET-101 and NET-124.

## area-raster-terrain.md (deepened version)

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| GPU-13 (region graph) | narrowed | raster-remote-sensing/gpu-raster-region-graph.ts:46-48 (256 and 65,535 caps); 4R² = 128 MiB at R = 5,792 is arithmetic | |
| EMB-03 | sentence added | raster-embeddings/gpu-raster-embedding-change.ts:107-110 | EMB-04/05 not changed (Covariance and similarity classes exist but row claims not contradicted) |
| Sky view | sentence added | gpu-terrain-horizon-rvt.spec.ts:94-96 | |
| CPU budgets | sentence added | raster-watershed/cpu-raster-watershed-segmentation-operation.ts:36-46; region graph from the report | |
| R6.4 | punctuation fix only | | REPRIORITISE H is a decision |
| R7.2, R7.3, RS02, RS03, RS10, RS11, R-REPROJ, TE05, TE06, RELIEF-01, LUM-09, DEC-VARIANCE, DEC-WALK-ROUNDING, GAP-02, EMB-01/06, SIM-02/03, CART-07, PLAT-06, LR-02, COLD | none | | KEEP |
| RS06, A-18, LUM-24, STD-06, STD-05, GAP-02 | none | | Rows already edited today carry the new verdicts' content |
| STD-04 / STD-09, STD-07 / STD-15, raster-fields-05, UPS-03, LR-17, LR-22 / KI-04 | not applied | | Replacement texts are instructions or decisions; the STD-07 census (61 of 132, 52 textures, nine buffers) differs from the row (61 of 129, 53) and was not recounted |

### Decisions and proposed rows (raster-terrain)

- R6.4: REPRIORITISE to H: scalar-output guard exists at gpu-raster-fractional-coverage.ts:653, but embedding zone mean (RT-126) and rasterization-to-join/zonal (RT-137) lack it; prioritize producer-to-consumer completeness before broader tiled acceptance.

## area-api-docs-catalog.md

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| LR-06, EMB-02 | none | roadmap.md | No such rows exist in the roadmap now; the delivered clauses are already gone |
| UF-10, LR-04, LR-10, STD-04, STD-05, STD-09, STD-13 | none | | Already edited today to the same content (UF-10, LR-04 with TOOL-102 open, LR-10 with the preflight fix, STD-04 ports, STD-05 runtime point-cloud settings, STD-09 validator public, STD-13 Dorling clause removed) |
| UF-01 | not applied | | Report text is an instruction (confirm URL, update 26 README links); decision |
| LR-17 / STD-12 | not applied | scripts/contributor-contract-baseline.json | Replacement is an instruction about docs anchors for 289 classes; count not reproduced |
| STD-11 | not applied | segregation-counterfactual.ts:250-268 | The cited lines show `RecipeBuilder` wrapping population alignment inside the counterfactual; whether the recipe as a whole still falls outside the prepared matrix was not established, so the 13-recipe sentence stays |
| STD-15, decision rows STD-04 / STD-09 and STD-07 / STD-15 | not applied beyond today's STD-04/STD-09 edit | gpu-terrain/terrain-illumination/*.ts still mention `addToGraph` | Report says addToGraph migration is complete; source still contains `addToGraph` in ten files, so the decision row keeps its text |
| Space-time variogram, CPU budgets, IN01, Known visibility graph, space-time kriging | none | | Row already states the facts, or a coordinator instruction says to leave it (space-time variogram) |
| KEEP rows (LR-08 / LR-24, LR-22 / KI-03, KI-01, CPU-05, STD-07, STD-14, DUP-01, DUP-02, AX) | none | | KEEP |

### Decisions and proposed rows (api-docs-catalog)

- GPU-13: REPRIORITISE the documentation correction (class-breaks 2^24, exact natural breaks 1,024/8,192 routes) to before alpha; code limits stay.
- NW-SCC matrix/API correspondence: REPRIORITISE before alpha (`dev-docs/coverage-matrix.md:255` is route-specific; `docs/api/gpu-network/expansion.md:44` is stale).
- GPU-11 / GE01-GE08 / UR01-UR05 urban tessellation: REPRIORITISE the reference-route correction before alpha (dissolved rings remain on the 64-building reference route).
- Known space-time kriging: keep row, correct the API page (automatic index above 4,096 samples versus explicit scan cap).

## area-core-contract.md

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| FP64-02 / LUM-20 | sentence added | dev-docs/upstream-workarounds.md:209; research/extended-precision-2026-11-11/README.md:31 | |
| UPS-CPU-1 | sentence added | dev-docs/upstream-drafts/01-packed-cpu-gpu-transfer.md exists | Filing status "not filed" from the report |
| STD-04, STD-05, STD-09, STD-10, STD-14, Chunked input, CPU output tails, CPU budgets, LR-04 Arrow, Published fp64, INT-06, STD-04 / STD-09 decision | none | | Already edited today to the same content |
| FP64-04 known limit | none; disagreement | | This report says KEEP (local residual helpers preserve caller-rounded operands and documented cutoffs); the roadmap-devdocs report said CLOSE and it was deleted earlier. Lead to decide whether to restore it as a known limit |
| STD-06, STD-12, STD-15, GPU-01, GPU-05 / PLAT-03, GPU-09, LUM-21, LUM-30, LR-17, LR-22, LR-23, LR-26, KI-01/03, FP64-03, UPS-CPU-2..4, CPU-08, CPU arithmetic, PLAT-04/05, KE-PORT-3, visibility graph, ARROW-01 | none | | KEEP, or row already holds the content, or replacement is an instruction; Kepler row left alone; fixes for findings are in progress so no closures |
| STD-13 | not applied | | REPRIORITISE |
| FP64-05 | not applied | | REPRIORITISE |

### Decisions and proposed rows (core-contract)

- STD-13: REPRIORITISE to high before alpha for coverage (538 qualified identities collapse to 537 audit names, two producers outside discovery, 31 missing named headings); capability-ID renaming can follow.
- FP64-05: REPRIORITISE before portability claims: fix the backend-dependent assertion, then measure on a second GPU.

### FP64-04 restored

| Row ID | Action | Source checked |
|---|---|---|
| FP64-04 (known limit) | restored from `git show HEAD:dev-docs/roadmap.md` line 501, before `Published fp64`, limit cell extended | analysis-core/kernels/addition-rounding-error-wgsl.ts:17-35 (caller-supplied `sum`, zero return below `2^-103`, non-finite); multiplication-rounding-error-wgsl.ts:20-35 (correctly rounded product exact, otherwise rounded; zero below `2^-80`) |

Verdicts: area-roadmap-devdocs.md said CLOSE to the workaround register; area-core-contract.md said KEEP (local residual helpers preserve caller-rounded operands and documented cutoffs). Restored per the coordinator's instruction.

## area-luma-upstream-scale.md

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| A-18 | narrowed (stronger than the earlier report: no upstream blocker) | gpu-terrain-spike-repair.ts:490-500; analysis-core/kernels/masked-weak-components.ts:28-40 | "initializes output" not read |
| Network flow | sentence added | network-flow-parallel.ts:158-163 | 8,388,608 = 128 MiB / 16 bytes; assumes a 128 MiB binding limit |
| LUM-23 | none | | Row is already absent |
| R-RELAX | none | | Row is already absent |
| LR-22, LR-23, UPS-01..03, UPS-CPU-1..4, PLAT-04, PLAT-05, LUM-09, LUM-13/32, LUM-14/33, LUM-21, LUM-24, LUM-28, LUM-30, LUM-31, LUM-34, FP64-02, STD-14, GPU-05, GPU-13, UF-03, DF-DESC, KG06, P.7a, DEC-VARIANCE, OPTICS xi, visibility graph, known chunked/space-time kriging | none | | KEEP, already edited today, replacement is an instruction, or counts (498 producers, 5,792/32,767) not reproduced |
| LUM-27, LUM-35 | none | | Kepler/website audit rows: left alone |
| FP64-05 | not applied | | REPRIORITISE |
| PLAT-04/05 maintainer decision | not applied | | Decision: file concrete batch iteration, dry-run planning and public-only defects upstream; keep geospatial semantics here |

## area-consumer-seams-lenses.md

| Row ID | Action | Source checked | Reason if not applied |
|---|---|---|---|
| LR-20 | narrowed (chunked-algebra sentence) | gpu-chunked-spatial-weights-algebra.ts:95; modules/lenses spatial-weights-algebra.lens.ts:36 uses `higherOrder` | Report's 384/43 archive comparison counts not reproduced |
| LN-ST | narrowed (85 to 89) | lens-coverage.node.spec.ts: 89 entries assigned `SPATIOTEMPORAL_REVIEW_PENDING` | |
| LN-SEL | sentence added | ladder-coordinator.ts:350-358 | |
| INT-01 / LN-REC2 | closed | commit 4e124a61; docs/lens-records has 428 files | Remaining oracle work stays under LN-REC |
| INT-06 | none | | Already closed |
| LN-REC, LN-ENC, LN-DQ, CPU-05, CPU-06, LUM-31, DEC-ARROW-CONSUMERS | none | | KEEP, counts not reproduced, or Kepler/website rows |
| LN-STAB, LN-V5 | not applied | | REPRIORITISE |

### Decisions and proposed rows (these two reports)

- LN-STAB: REPRIORITISE (defer StabilityMap/Purpose until result identity, missingness and archive coverage are fixed).
- LN-V5: REPRIORITISE (defer `GPURecipePlan`, `ComputationDrawer` and solver-wavefront visualization; no current reader regression depends on them).
- FP64-05: REPRIORITISE (local factory forces integer fp64 more broadly than the installed assembler; qualify non-Apple first).
- PLAT-04/05 maintainer decision: narrow to filing concrete batch iteration, dry-run planning and public-only defects upstream.
