# Roadmap

Status date: 2026-11-11. Baseline: `@luma-contributors/analysis`, `/lenses` and `/kepler` on published `@luma.gl/*@10.0.0-alpha.12`. The work of sections 10 to 14 was reviewed, committed and pushed to `origin` (`master` and `upstream-first`) on 2026-11-11, and the website is published on GitHub Pages (section 15). Outcome paragraphs in sections 10 to 13 that say "in the working tree, uncommitted" describe the state before that commit.

This is the single current roadmap. It replaces about 19 planning documents from the luma.gl `map-contributors` branch. The superseded documents are kept verbatim in [archive/](./archive/README.md) as the evidence trail.

Companion documents:

- [coverage-matrix.md](./coverage-matrix.md): function-by-function coverage and where each symbol lives.
- [upstream-workarounds.md](./upstream-workarounds.md): every dropped capability, workaround and accepted regression caused by depending on published luma.gl.
- [upstream-first-plan.md](./upstream-first-plan.md), [upstream-first-pending.md](./upstream-first-pending.md): the decisions behind the split.
- [research/card-interaction-audit-2026-11-09](./research/card-interaction-audit-2026-11-09/README.md): the audit of the card header pictures before the work recorded in section 7.

## Website demo data and interactive cards (2026-11-08)

The website has 439 catalog entries across twelve operation groups. The media plan follows the data through the scene, its static capture and the interactive card.

1. Trace the inputs used by every flagship demo and operation run. Reuse small, redistributable observations where they fit the operation. Keep the exact geometries, perturbations and counterfactuals required to demonstrate algorithm behavior, and identify them as scenarios. Record source, date, units, sampling and terms with each extract.
2. Keep the data in the private website workspace. Share each extract between related operations. Preserve coordinate aspect, topology, row order and field units through the typed-array conversion used by the GPU. Include attribution next to the operation where the data is used.
3. Render the entry's own live scene with its catalog seed to capture both theme posters. Use the still in static HTML and during adapter initialization. The same scene should take over when a reader hovers or focuses a card. Pointer input changes an operation parameter or a spatial input. Release the live slot when the interaction ends.
4. Exercise both themes, real pointer changes, keyboard navigation, reduced motion, touch navigation and missing GPU support. Check every poster against the registered scene and run the catalog and package checks. The separate deck.gl showcase already tracks its captured story previews against the source datasets in its capture manifest.

The 439 catalog pages share 438 scene implementations. The website's CPU provenance audit resolves a source for each scene. Both themes have a captured poster for each page. Source terms, transformations and modeled inputs are indexed in the [data notice](../THIRD-PARTY-DATA.md#website-demo-extracts).

## How to use and maintain this file

- Every open item has an ID. Reuse the ID in commits, issues and test names. IDs from the old documents (ST01, RS02, TE06, CR01, KG06 and so on) are kept. New IDs use the prefixes `UF` (upstream-first completion), `AUD`, `NW`, `LN`, `KE`, `SH`, `WEB`, and, from the 2026-11-09 spatiotemporal review, `FX` (defects), `TC` (time contract), `STX`, `STM`, `MOB`, `RTS`, `PLAT`, and, from the 2026-11-09 library review, `STD` (contract and standards), `GPU` (rework to a GPU-scale design), `DUP` (merges), `UPS` (moves to luma.gl), `GAP` (missing capabilities), and, from the 2026-11-11 research on new contributors (section 14), `LRN` (models the analyst brings or fits), `EMB` (embeddings), `SIM` (simulation, uncertainty and factorization), `CART` (presentation), `ENC` (pretrained encoders), and, from the integration pass of 2026-11-11 (section 15), `INT`.
- Priority: **H** = clear scope and real analysis or correctness value, do next. **M** = useful, needs a consumer or a design step. **L** = research, belongs in luma.gl, or an explicit non-goal.
- "Current limit" cites `file:line` in this repository as of the status date. Source files are still moving, so re-grep before you rely on a line number.
- "Source" points at the archived document (path relative to this directory) or, for documents that were not archived, the luma.gl branch path.
- When you finish an item, delete its row and, if it changes coverage, update [coverage-matrix.md](./coverage-matrix.md). When you accept a new regression or add a workaround, add it to [upstream-workarounds.md](./upstream-workarounds.md) in the same change.
- Review this file at each `@luma.gl` alpha bump (see the upgrade procedure in the workarounds file) and at each release.
- Nothing in the coverage inventory is "Planned". Every row is Implemented, Partial or Excluded. This roadmap lists what is left inside and around those rows.

## High-priority summary

| ID | Item | Why |
| --- | --- | --- |
| MOB-03 | The FX-08 point-centred default is undecided | Small, and it changes what callers see |
| NW-CENT | `GPUNetworkCentrality` still unrolls its rounds at compile time | The cliff is now a constructor error above 32,768 command nodes; large graphs need sampled pivots until a GPU-gated continuation exists upstream |
| WEB-PIC-12 | 83 of the weakest pictures were reworked on 2026-11-11 and reviewed; the rest of the 149 rest pictures and 120 gestures that scored 1 or 2 are untouched | The pictures are the first thing a reader of the site sees; no re-score of the whole site exists after the rework |
| KE-TESTS | 131 of 368 kepler spec files are still excluded from every run | 140 were re-enabled on 2026-11-10; the rest need deck.gl, undeclared packages or unpublished luma.gl APIs |
| STD-04, STD-06, STD-07, STD-10 | The breaking half of the contract alignment landed 2026-11-11 (aliases removed, one geometry output layout, the core surface) | The raster family follows the `mask` rule since wave 3. Remaining: raster inputs through the published band validity, per-cell status code words, status in most statistics classes, the solver sites whose fixtures would move |
| GPU-05, FX-18 | Brute force remains in MGWR; kernels that read padding rows were fixed where found, and the other examined sites are safe | One padding-row site is reachable when offsets do not start at 0 (`getSnapOrigin`), and triangulation was not examined |

## 1. Now: finish upstream-first

Plan: [upstream-first-plan.md](./upstream-first-plan.md). Phases 1 to 6 are done. This block is what is left of the baseline before new feature work. Done and deleted on 2026-11-07: UF-02 (CI gates), UF-04 (test and example typechecks), UF-05 (stale subpath imports), UF-07 (`graph-support` is reached through the `./analysis-core` export), UF-08 (dead test globs).

| ID | Item | Current state | Pri |
| --- | --- | --- | --- |
| UF-01 | Repository metadata | The working tree was committed by area and pushed on 2026-11-11 (section 15). Left: the `repository` URL in `modules/*/package.json` is `https://github.com/visgl/luma-contributors`, not the remote `https://github.com/BertCh/luma-contributors` (private), and the website's GitHub link points at the private repository | M |
| UF-03 | Re-measure performance after the split | The branch measurements used forked primitives. Run `yarn test-benchmarks` and record numbers next to the accepted regressions in [upstream-workarounds.md](./upstream-workarounds.md) section 3. The bounded-dispatch row there is no longer a regression (PLAT-01). Scale factors for the join and aggregation benchmarks: PLAT-07 | M |
| UF-12 | Example checks | The example typecheck exists only in `.github/workflows/test.yml` (`npx tsc --noEmit -p examples/deck/<name>`), not in a root script, so `yarn typecheck` never covers the examples. 36 showcase check scripts import `scripts/playwright/*.mjs`, which is not in the repository or its history, so about 43 of them cannot run; the explorer has no check script. No example picture was looked at after the changes of 2026-11-10 and 2026-11-11 | M |
| UF-10 | Release housekeeping | `CHANGELOG.md` exists. No `CONTRIBUTING.md` or publish guide. `modules/analysis` is public at `0.1.0-alpha.0`; lenses and kepler are private | L |

## 2. Backlog by domain

### 2.1 Spatial statistics, regression and local statistics

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| ST01 | HC2/HC3 and clustered covariance, positive-semidefinite certification for spatial 2SLS | Not implemented. `GPUSpatialTwoStageLeastSquares` and `GPUInstrumentalVariablesCovariance` give the base covariance | M | `archive/geospatial-coverage-parallel-implementation.md` (ST01) |
| ST02 | Spatially robust restriction inference | Not implemented (`GPURestrictedRegimeRegression` exists) | L | same (ST02) |
| ST03 | Time effects, clustered covariance, dynamic panels | Fixed and random effects panel error and lag ML exist (`gpu-spatial-panel-*-maximum-likelihood.ts`). `gpu-spatial-panel-regression.ts:59` rejects other effects | L | same (ST03); `archive/geospatial-toolkit-expansion-follow-up.md` (item 4) |
| ST04 / ST05 | Iterated SUR; latent spatial probit | Single-pass `GPUSeeminglyUnrelatedRegression` and `GPUProbitRegression` only | L | same |
| ST06 | MGWR: AICc and continuous bandwidth search, full backfit selection uncertainty | Finite-grid LOO selection only. `gpu-multiscale-geographically-weighted-regression.ts:174` rejects other variance conventions | M | same (ST06); `archive/gpu-analysis-algorithms-roadmap.md` (line 29) |
| ST06b | Adaptive or selected bandwidth and separation handling for Poisson and logistic GWR | Fixed bandwidth. `gpu-generalized-geographically-weighted-regression.ts:170` (bandwidth mode), `:175` (family), `:639` (kernel) | M | `archive/geospatial-toolkit-expansion-follow-up.md` (item 4) |
| ST07 | LOSH and multivariate Geary: joint randomization, more correction families | Descriptive output plus separate inference only | M | same; parallel record (ST07) |
| ST08 | Rank Markov: fractional ties, missing-panel interpolation, more than 64 states | Not implemented (`GPURankMarkov`) | L | parallel record (ST08) |
| ST09 | Geographic (non-planar) geometry and topology repair for polygon statistics | Planar only. `gpu-island-attachment.ts:70` rejects distance output | L | parallel record (ST09) |
| ST10 | blockCV: hierarchical and merge blocks, response balancing, continuous variogram optimizer | Not implemented (`GPUSpatialCrossValidation`) | L | parallel record (ST10) |
| ST11 | Spatial interaction: multiple covariates, other count distributions, coefficient uncertainty in calibrated flow | Not implemented (`GPUSpatialInteraction`) | L | parallel record (ST11) |
| KG01f | Multivariate quantile LISA | Single-variable only (`GPUQuantileLISA`) | M | `archive/kepler-analysis-implementation.md` (line 63) |
| WEIGHTS7 | Chunked output for spatial-weights transpose | Not implemented: `gpu-chunked-spatial-weights-transpose.ts:48` | M | code stub |
| PERM-01 | Permutation batching with early stopping | `gpu-global-permutation-test.ts:368` rejects `permutationBatchSize` with `stopAfterExceedances` | L | code stub |
| JOIN-01 | Distance output from predicate join | `gpu-spatial-predicate-join.ts:738` rejects `output.weights.distances` | L | code stub |
| ROSE-01 | Circular angle conventions for the rose statistic | `gpu-rose-statistic.ts:133` | L | code stub |
| KJ-01 | Other operations in the key join | `dataframe-analysis/key-join/gpu-key-join.ts:200` | L | code stub |
| KG02 | Dorling cartogram: GeoDa coordinate parity, convergence guarantee | Heuristic by contract; grid path added 2026-11-07; relaxation limited to 256 passes (`gpu-dorling-cartogram.ts:127`) | L | `archive/kepler-analysis-implementation.md` |
| KG03 | Exact natural breaks beyond 8,192 rows or 64 classes; float64 objective | Exact mode materializes costs to 1,024 rows and recomputes to 8,192 (`gpu-class-breaks.ts:127-129`). Kepler uses a worker Jenks instead | L | same |

### 2.2 Point processes, sampling and clustering

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| PP01 / PP02 | Intensity estimation, more study-window corrections, kernel-smoothed pair correlation, tie breaking for extreme-rank-length envelopes | Inhomogeneous K, kernel intensity and the smoothed pair correlation are implemented in `GPUInhomogeneousPointPattern` (to 4,096 events); the comment at `gpu-ripley.ts:183` is stale. Open: more window corrections, envelope tie breaking | L | `archive/geospatial-toolkit-expansion-follow-up.md` (item 4); parallel record (PP02) |
| PP04 | Parallel OPTICS ordering | Indexed core distances and an indexed seed-list frontier landed 2026-11-10 (`neighborIndex`, exact, at most 65,536 rows). The ordering is still sequential in one workgroup (reference tier) and xi extraction stays at 4,096 rows | M | parallel record (PP04)  |
| PP05 | Response-dependent replacement and design variance in GRTS sampling | Not implemented | L | parallel record (PP05) |

### 2.3 Interpolation and simulation

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| IN04 | Fitted linear model of coregionalization (more than 4 variables or structures) | `GPULocalKriging` takes a supplied LMC; `GPUVariogramModelFit` fits one model | M | parallel record (IN04) |
| IN01 | Resident variogram fitting is slower than CPU on small inputs | Implemented. Branch measurement: CPU 1.3 and 3.2 ms against GPU 4.3 and 51 ms. Keep and document | L | parallel record (IN01) |
| IN05 / IN02 | Degree above 1 or global dense RBF; global indexed kriging | Not implemented (`GPULocalRBF`, `GPULocalKriging` are local) | L | parallel record |
| IN06 | Joint covariance across separate conditional simulations | Not implemented, by design | L | parallel record (IN06) |

### 2.4 Network

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| NW-CENT | Centrality without compile-time unrolled rounds | Reachability, sigma and delta rounds are still unrolled per batch (`maximumIterations` 1 to 1,024, default 64). Interim guard done: `GPUNetworkCentrality.estimatedCommandNodeCount` is exact and the constructor rejects more than 32,768 command nodes (`gpu-network/network-centrality/gpu-network-centrality.ts:70`, `:418-423`), documented in the class and in `docs/api/gpu-network/operators.md`; the kernel audit has an over-cap rejection (`kernel-audit/rejections-network.ts:39-43`). Exact all-pivot betweenness above roughly 15,000 to 20,000 nodes now needs sampled pivots, fewer rounds or a larger `laneCount`. Structural fix: a GPU-gated continuation, which the published compiler lacks | H | `archive/gpu-analysis-algorithms-roadmap.md` (line 161); `archive/geospatial-toolkit-expansion-follow-up.md` (line 142) |
| NW-SCC | Large-graph strongly connected components | Dropped. Only exact reachability SCC, 1,024 vertices and 65,536 CSR slots (`gpu-network-structure.ts:45-67`) | M | [upstream-first-pending.md](./upstream-first-pending.md) |
| NW-KSP | Scalable k-shortest and simple route enumeration | `GPUAlternativeRoutes` is bounded to 256 nodes (`gpu-alternative-routes.ts:68`) | M | `archive/geospatial-toolkit-expansion-follow-up.md` (item 4) |
| NW04 | Sparse scalable Louvain | At most 32 vertices, dense (`gpu-network-louvain.ts:54`). Published `@luma.gl/graph` has modularity optimization; evaluate it first | M | parallel record (NW04) |
| NW02 | Sparse push-relabel and path decomposition for max flow | At most 128 nodes and 1,024 edges (`gpu-network-flow.ts:69-70`) | L | parallel record (NW02) |
| NW03 | Negative-cost circulation and lower-bound constraints | Not implemented | L | parallel record (NW03) |
| NW05 | Mixed-integer or fractional allocation; certified capacitated p-median | Exhaustive and bounded: demand at most 64 (`gpu-network-facility-optimization.ts:69`) | L | parallel record (NW05) |
| NW06 / X04 | Non-FIFO schedules, time windows, transit | FIFO only (`gpu-network-time-routing.ts:73`); transit is excluded | L | parallel record (NW06) |
| NW-CH | Contraction hierarchies (PHAST) | Rejected by policy | L | `archive/gpu-analysis-algorithms-roadmap.md` (line 264) |
| NW-GA | Generic graph adapters for dense traversals | Not implemented | L | follow-up (item 4) |

### 2.5 Raster

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| R6.3 | Seam-safe contour ownership across tiles; external vector overlays | Isolines handle one raster | M | `archive/gpu-raster-roadmap.md` (Tranche 6.3) |
| R6.4 | Broader fractional and zonal composition envelopes | Partial: `GPURasterFractionalCoverage` and zonal statistics exist; whole-domain and tiled acceptance cases are not all closed | M | same (Tranche 6.4) |
| R7.2 | Watershed at larger scale; other segmentation methods | Complete domain of at most 16,384 cells (`gpu-raster-watershed-segmentation.ts:48,89`) | M | same (Tranche 7.2) |
| R7.4 / RS07 | Spectral (FFT) raster filtering; registration peak refinement and ambiguity | Not implemented. FFT convolution is deferred | M | same (Tranche 7.4); `archive/gpu-analysis-algorithms-roadmap.md` (line 31) |
| R-FLOW | Cross-tile D-infinity and MFD network stitching | Partial: `GPUTerrainTiledFlow`, `GPUTerrainBranchingFlow` exist | M | follow-up (item 4) |
| R-RELAX | `settledAt` output with continuation in cost distance | `raster-relaxation.ts:799` | L | code stub |
| R7.1 / R7.3 | Adaptive local analysis (CLAHE); volumetric microscopy | Gate-deferred, no code | L | `archive/gpu-raster-roadmap.md` |
| RS02 | Sparse or string zone IDs | Dense numeric IDs only, up to 65,534 classes | L | parallel record (RS02) |
| RS03 | Curvilinear grids for conservative remapping | Not implemented | L | parallel record (RS03) |
| RS06 | Irregular dates in temporal composites and filters | Not implemented | L | parallel record (RS06) |
| RS10 | Endmember discovery, ISRA and NMF | Not implemented | L | parallel record (RS10) |
| RS11 | Multiple break search, calibrated p-values, full BFAST | Single-break seasonal model | L | parallel record (RS11) |
| X03 | Atmospheric correction and RPC orthorectification | Excluded | L | [coverage-matrix.md](./coverage-matrix.md) |
| R-REPROJ | Explicit source-tile routing for reprojection; extra precision and filtering | Not implemented | L | follow-up (item 4) |

Raster base classes, exact percentile, wide counters, 4K+ partitioning and FFT convolution belong to luma.gl (see section 5).

### 2.6 Terrain and hydrology

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| TE06 | Out-of-core and independently filled tiles | Resident full-domain composition only (`GPUTerrainTiledFlow`) | M | parallel record (TE06) |
| HAND-01 | Height above drainage over D-infinity flow | `gpu-terrain-height-above-drainage.ts:138` | M | code stub |
| RELIEF-01 | Percentile stretch in relief blending | `gpu-relief-blend.ts:205` (needs a histogram) | L | code stub |
| TE03 | Parallel depression-hierarchy activation pass | Serial (partial) | L | parallel record (TE03) |
| A-18 | `GPUTerrainSpikeRepair` persistent buffer | Upstream gap, unchanged | L | `archive/analysis-contributors-quality-roadmap.md` (line 63) |
| TE05 | Horizontal, Nuth-Kaab and tilt coregistration models | Not implemented (`GPUTerrainDEMCoregistration`) | L | parallel record (TE05) |
| X05 | Flood simulation | Excluded | L | [coverage-matrix.md](./coverage-matrix.md) |

### 2.7 Dataframe analytics

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| KG06 | SQL: DOUBLE, WKB, arbitrary strings, right-column projections, UNION | Partial. The engine is published `@luma.gl/graph/gpu-sql`; nothing in this repository. Needs upstream work | L | `archive/geospatial-function-coverage.md` (KG06) |
| DF-DESC | Dataframe query description (intent, shader plan, formulas) | Dropped with the describe feature; needed only for lens scenarios | M | [upstream-first-foundation.md](./upstream-first-foundation.md) section 2 |
| ARROW-01 | More Arrow analytics types | `analysis-core/graph-support/arrow-analytics-normalization.ts:173` rejects unsupported types | L | code stub |
| GEO-DEPR | Remove deprecated `spherical-cell-area.ts:48` (use `getGPUAuthalicSphereRadius`) | Deprecated shim | L | code |

### 2.8 Point cloud

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| PC06 | Automatic normal-scale selection, fitted correlation model; M3C2 binary has never been run | Not implemented | M | parallel record (PC06) |
| PC01 | Globally propagated normal orientation; unlimited support | Viewpoint or up orientation per query | L | parallel record (PC01) |
| PC02 | Distributed global threshold reconciliation | Not implemented. Voxel downsample and radius filter are published | L | parallel record (PC02) |
| PC03 | Ground Delaunay or nearest interpolation; vertical datum | Not implemented | L | parallel record (PC03) |
| PC05 | Global alignment, trimmed objective for ICP | Not implemented | L | parallel record (PC05) |
| X01 / X06 | File readers; solid booleans | Excluded | L | [coverage-matrix.md](./coverage-matrix.md) |

### 2.9 Projection and datum

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| CR02 / CR03 | Epoch-aware grid resources, bounded residency | `GPUDatumGridTransform` (`geospatial-analysis/gpu-datum-grid-transform.ts`) applies supplied grids; no epoch handling | M | luma.gl branch `dev-docs/roadmaps/gpu-project-roadmap.md` (CR02, CR03) |
| CR01 | Difficult geodesic cases: poles, antipodes, singular inverse | Float32 envelopes are not certified | M | parallel record (CR01) |
| CR04 / CR05 | Exact S2 predicates; S2 covering minimization | Not implemented | L | parallel record |
| P.7a | 3D contract, ECEF, static Helmert | In neither this repository nor published luma.gl; belongs in luma.gl `gpu-project` | L | luma.gl branch `gpu-project-roadmap.md:237-238` |
| DATUM-01 | Vertical-datum conversion and geoid input | Declared and checked, never converted. Non-goal | L | `archive/gpu-spatial-analysis-roadmap.md` |

### 2.10 Geometry, urban morphology and ecology

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| GE01 / GE08 / UR01-UR05 | Capacity limits: nesting above 64; straight skeleton above 1,024 vertices; urban recipe capacities | Bounded by contract | L | parallel record |
| EC02 | High patch-count behaviour is unmeasured | Not measured | L | parallel record (EC02) |
| EC04 | Other segregation statistics and nulls | Not implemented | L | parallel record (EC04) |
| GEOM-01 | Rectangle clip outputs for polygons | `gpu-rectangle-clip.ts:199,205` | L | code stub |
| GEOM-02 | Line smoothing source rows and measures | `gpu-line-smooth.ts:164` | L | code stub |

### 2.11 Kepler

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| KE-TESTS | Run the remaining kepler tests | 140 of the 271 parked specs were re-enabled on 2026-11-10 against the example sources in place (858 tests pass, 9 skipped), so 236 of 368 spec files run. 131 are still excluded (`test/kepler-compute-example-specs.json`, `excludedFromRuns`, one reason per spec): 49 need deck.gl, about 57 need other undeclared or uninstalled packages, about 25 need luma.gl APIs that are not published. 59 more run but are left out of the test typecheck (`excludedFromTypecheck`). 140 specs now depend on the example sources, so deleting the example would remove their coverage (KE-EX). Open: `duckdb-host.spec.ts` is excluded because Vite re-bundles `@duckdb/duckdb-wasm` mid-test on a cold cache, which leaves two copies of Arrow on the page ("Table and inner RecordBatch schemas must be equivalent"); add it to the headless project's `optimizeDeps.include` in `vitest.config.ts` and re-enable the spec. To decide: `prepareArrowTableForGPUAnalytics` and its types are exported from `analysis-core` (through `graph-support`) but are a copy of luma.gl's Arrow preparation code, registered as a workaround; repointing the parked example at that copy would let at most 4 more headless specs load and 3 join the typecheck, and adds consumers of a copy that the upstream-first rules want gone. Detail: [research/parallel-pass-2026-11-10/kepler.md](./research/parallel-pass-2026-11-10/kepler.md) | H | `archive/kepler-compute-integration-plan.md`  |
| KE-EX | Fate of `examples/deck/kepler-compute` (about 750 files) | Parked: not in `workspaces`; imports `@deck.gl-community/gpu-layers` and `@deck.gl/core`, which are not dependencies | M | `archive/kepler-compute-integration-implementation.md` |
| KE-UI | Remaining host work: native Group By panel binding, non-point altitude layers, richer labels, micro and nanosecond time admission, multipart timed trips, acceptance on other hardware tiers | Open (mostly example code). Acceptance was measured on one Apple M-class machine | M | `archive/kepler-compute-integration-implementation.md` (Remaining implementation) |
| KE-GUARD | About 14 "Unsupported ..." validation guards | By design, not stubs. Examples: `kepler-group-by.ts:111`, `style-plan.ts:104,147`, `analysis-recipe.ts:357,1367`, `prepare-spatial-selection.ts:51,77` | L | code |
| KE-DECK | Layers that depended on deck GPU layers removed on the branch (cluster, flow, globe) | External; track with the deck.gl-community package | L | luma.gl branch `dev-docs/reviews/comprehensive-gpu-review-2026-11-07.md` |

### 2.12 Lenses and computation visualization

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| LN-REC | Recording, remaining parts | Done 2026-11-10 without a per-contributor prop: for a graph that opts in (`recordWGSLKernelNodes(graph)` or `recorder.recordKernels(graph)`), `createWGSLKernelNode` captures source text, entry point, bindings and dispatch; `describeGPUCommandGraph` reads them; `submitGPUCommandGraphWithTimings` adds timings where the device has timestamp queries (decision in [upstream-first-recording.md](./upstream-first-recording.md)). Records, twins and website view models regenerated 2026-11-11: 412 records, about 49 MB of a 64 MB limit, none over 512 KB (two terrain scenarios carry no source text; the raster resistance scenario's iteration bound is 64). Remaining: timings in the dataframe and S2 harnesses; kernels of published luma.gl primitives (`GPUScan`, `GPUSort`, `GPUReduction`, `GPUGroupAggregation`) carry no facts until luma.gl publishes a per-node shader description; records are not byte-reproducible, because 182 store GPU timings (stated in `docs/api/computation-record.md`); 135 numeric captions in the docs for scenarios whose values changed were not checked one by one (list in `w2-lenses-records.md`); nine records moved in the last digit with no stated reason (`review-consumers.md`); formulas, loop provenance and `derivedFrom` stay pack-level | M | `archive/computation-visualization-roadmap.md` (V0, V3) |
| LN-SEL | GPU selection backend | `GPUCrossfilterSelectionBackend` exists (`lenses/src/computation-views/coordinator/gpu-crossfilter-selection-backend.ts`). Verify it against published `GPUCrossfilter`, which lacks external masks (workarounds, section 2) | M | same |
| LN-DQ | Rebuild dataframe-query and formula scenarios on public query metadata | 8 scenarios dropped | M | [upstream-first-foundation.md](./upstream-first-foundation.md) section 2 |
| LN-STAB | StabilityMap and Purpose rung | Not built (also not on the branch) | L | `archive/computation-visualization-roadmap.md` |
| LN-V5 | `GPURecipePlan`, `ComputationDrawer`, V6 to V8 phases, space-time cube view of solver wavefronts (a visualization of the archive, not a data cube) | Never built | L | same |
| LN-ENC | `CommandEncoder` pass and slot labels; rendering and trace packs | Belongs in luma.gl core | L | same |
| LN-ST | Lenses for the contributors added by the spatiotemporal review | 85 contributors have a documentation slot but no pack operator, formula, scenario or record. They are the `SPATIOTEMPORAL_REVIEW_PENDING` entries in `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts` | M | [research/spatiotemporal-review-2026-11-09](./research/spatiotemporal-review-2026-11-09/README.md) |

### 2.13 Showcase and examples

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| SH-DATA | Large binary data tracked in git | `examples/deck/spatial-analysis-showcase/public/data` has 3 to 5 MB PNG and bin files. Consider git LFS or a download script; provenance is in `THIRD-PARTY-DATA.md` | L | `archive/showcase-sprint-datasets-2026-10-24.md` |
| SH-NAMES | Example package names and versions | Examples still carry `luma.gl-examples-*` names and `10.0.0-alpha.10` versions | L | health audit |
| SH-SPRINT | 2026-10-24 sprint scenes | Finished; outcomes in [research/showcase-overhaul-2026-10-25](./research/showcase-overhaul-2026-10-25) | - | `archive/showcase-sprint-2026-10-24.md` |

### 2.14 Website (Gridlight)

Visualization-specific pointer gestures for all 438 scenes, the picture fixes from the audit and regenerated posters were implemented on 2026-11-09; the status, measures and what was decided are in [section 7](#7-plan-card-header-pictures-web-int-web-pic). The rows below are what is left.

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| WEB-INT-08 | Touch steering on the function-page hero | Not built; a tap on a card navigates by design | M | section 7.5 |
| WEB-INT-09 | Shared GPU device spike | One device per live card; median start 668 ms in the harness | M | section 7.5 |
| WEB-INT-14 | Keyboard path to the hero's gesture | Arrow keys steer cards only; on the hero the sliders are the keyboard path | M | section 7.5 |
| WEB-INT-16 | Label the remaining runs | Every run of the ranked list carries labels since 2026-11-11 except `study-window-preparation`; the atlas lacks `^`, `'` and a superscript two | L | section 7.5 |
| WEB-INT-13, 15, 17, 18 | Handle after a restart, pointer-to-frame timing, highlight as a layer feature, `record-demo.ts` | See section 7.5 | L | section 7.5 |
| WEB-PIC-09 | Data that fits | No open water or post-fire scene in the imagery extract; a nearly bimodal lidar tile | M | section 7.6 |
| WEB-PIC-12 | Act on the independent re-score | 83 runs reworked and reviewed 2026-11-11; about 20 gestures among them still score 1 or 2, and the weak runs outside the 83 are untouched | H | section 7.6 |
| WEB-PIC-13, 14 | Inert sliders and weak gestures after the 2026-11-10 pass and its review | Five sliders were removed because the checked-in lidar and elevation tiles cannot make them act; about 40 gestures stay under 3% | M | section 7.6 |
| WEB-PIC-15 to 18 | Siblings still alike, entries left partly done, suspected defects to confirm, six constructed scenarios | See section 7.6 | L | section 7.6 |

### 2.15 Spatial and spatiotemporal directions (2026-11-09)

Source for every row: [research/spatiotemporal-review-2026-11-09](./research/spatiotemporal-review-2026-11-09/README.md), which has the file and line evidence, the reference implementations and the items judged not worth doing. `FX` rows are defects in existing code, re-read in source on 2026-11-09. The other rows are new work. Effort: S is days, M is one to two weeks, L is more.

The rows below are what remains. Everything else in the review was implemented on 2026-11-10; section 8.5 has the outcome. Lenses for the new contributors are row LN-ST.

Exploratory space-time statistics (STX):

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| STX-03 | ST-DBSCAN, ST-OPTICS, ST-HDBSCAN | ST-HDBSCAN is not delivered; ST-DBSCAN and ST-OPTICS are. | M |
| STX-08 | Geographical detector q, REDCAP, contiguity-order correlogram, spatial entropy, colocation quotient | REDCAP is first-order single linkage only; LICD is the local composition part only; the colocation quotient has no permutation test. | L |

Space-time models and inference (STM):

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| STM-06 | GTWR and multiscale GTWR (oracle to be pinned) | GTWR and MGTWR are done without a golden-section or adaptive bandwidth search. | M |
| STM-07 | Vecchia likelihood, estimation and prediction. The review's CPU emulation found float32 adequate when the nugget is at least 1e-3 of the sill (any nugget for the exponential kernel); next step is a GPU comparison against GpGp | Spatial Fisher scoring is done; the space-time fit uses Nelder-Mead and the existing neighbour search; GpGp fixtures are pending (R). | M |

Movement (MOB):

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| MOB-03 | Cleaning and splitting primitives; word timestamps in time-aware simplification | TD-TR word timestamps go through `GPUTrajectoryTrackRelativeTimes`, not in place. The FX-08 point-centred default is an open decision (section 8.4). | H |
| MOB-06 | Time-varying vector fields, FTLE, connectivity, residence time | Utilization volumes are done without ridge extraction or longitude/latitude input. | M |
| MOB-07 | Brownian bridge home ranges; revisits and first passage time | Brownian bridge and recursions are done without the dBBMM change-point step. | L |
| MOB-10 | Radiation model; kernel flow smoothing | The scalable radiation variant is absent; the kernel smoothing definition is unverified against the paper; 1,024 zones at most. | L |

Raster time series and grids (RTS):

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| RTS-08 | HEALPix and A5 cells; DGGS-JSON output | HEALPix is done. A5 is not started: about 3,100 lines of reference, its boundary handling needs a redesign for a fixed kernel. DGGS-JSON zone IDs are not OGC-conformant. | L |

Platform (PLAT):

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| PLAT-03 | One point neighbour index for clustering, LOF and KDE, k above 32, optional time axis | `k` reaches 256 and `GPUNeighborSearch` has a time axis. "One index" is a shared builder, not one walk: DBSCAN and KDE keep their gathers; the correlogram pair grid, k-means, island attachment and two newer KDEs are not migrated. HDBSCAN `minimumSamples` stays at 32 (section 8.4). | M |
| PLAT-04 | luma.gl: GPU-resident iteration by batched indirect dispatch (the structural fix for NW-CENT) | Analysis half done. The luma.gl half is a proposal only: `upstream-proposal-gpu-resident-iteration.md`, no PR. | M |
| PLAT-05 | luma.gl or host: request adapter limits; subgroups with a fallback; `buffer_view` | Analysis half done. Checked 2026-11-10: adapter-limit requests (`requiredLimits`) and the `subgroups` feature are already public in alpha.10 and alpha.12, so only the adapter-maximum shorthand and `buffer_view` remain to propose (`upstream-proposal-adapter-limits-subgroups.md`, not sent). | L |
| PLAT-06 | Partly done 2026-11-09: `yarn oracle-fixture <generator>` (`scripts/run-oracle-fixture.mjs`) regenerates a family in a sandbox and diffs it, writing only with `--write`, and reports a missing interpreter, Python package or R with exit 2; the dead `modules/graph` paths in four generators are fixed. Open: a new external fixture for the trajectory, track-similarity and Markov families (they already have giddy, shapely, similaritymeasures, tslearn, dtaidistance and MovingPandas fixtures; the only installed environment holds numpy, scipy and gstools, so a new one would be a binary64 reference only); MobilityDuck | The runner exists. R-only fixtures pending: gstat, sparr, stpp, stopp, changepoint, scanstatistics, ETAS, GpGp, CAST, phenofit/GRASS, bfast. Also pending: pyxccd, xeofs and scikit-learn for RTS-03, and the authors' GeoConformal code. | M |
| PLAT-07 | Partly done 2026-11-09: `spatialbench-scale-benchmark.spec.ts` (opt-in, in `yarn test-benchmarks`, `VITE_SPATIALBENCH_SF`) runs a point-in-zone join with per-zone counts at trips = 6M x SF against 156,095 synthetic zones. Apple headless WebGPU, median per encode: SF0.1 7.2 ms, SF1 42.6 ms, SF2 81.3 ms, SF5 216.3 ms; SF10 (60M trips) crashes the page because the runner builds JS point arrays. SF100 and above (600M trips, 9.6 GB of positions) do not fit a default device or one buffer. Open: real Overture zone geometry, the aggregation-only and KNN queries, a typed-array runner for SF10 | Measured to SF5 only, on synthetic zones and one query shape. | L |

### 2.16 Library convergence: one contract, GPU-scale designs (2026-11-09)

Source for every row: [research/library-review-2026-11-09](./research/library-review-2026-11-09/README.md). The contract is [contributor-contract.md](./contributor-contract.md). A first implementation pass ran on 2026-11-09; rows that say "Done 2026-11-09" list what landed and keep only what remains, and finished rows (STD-01, GPU-02, GPU-03, GPU-06) are deleted. Nine reviewers gave a verdict to 429 contributors or sibling groups: 182 conform, 133 need contract alignment only, 64 need the algorithm reworked, 30 duplicate another contributor, 20 belong upstream. The per-contributor tables, with `file:line`, are in the `code-*.md` reports named in each row. Five decisions the rows depend on are listed in the review README ("Decisions for the maintainer"). Effort: S is days, M is one to two weeks, L is more.

Contract and standards (STD). No behaviour change intended; the existing tests are the safety net.

| ID | Item | Current state | Effort | Pri |
| --- | --- | --- | --- | --- |
| STD-02 | Contract checker: turn the remaining rules from baseline to failing as each count reaches zero | `yarn check-contributor-contract` runs in CI and fails on departures not in `scripts/contributor-contract-baseline.json`. Counted 2026-11-11: 0 bare `assert` calls, 0 unregistered contributors and 0 `@internal` exports, so those three rules can fail outright; the baseline was rewritten on 2026-11-11 and holds 64 findings (112 before, none added); 46 single-invocation loops (heuristic; reference-tier kernels need a `contract-allow` comment); 21 multi-class files; 21 root files. The loop rule cannot see a kernel body passed as a function parameter, the producer-declaration rule misreads doc comments, and nothing checks for a WGSL helper declared twice in one kernel (one such duplicate broke 12 specs on 2026-11-11 and only the headless specs showed it) | S | M |
| STD-04 | Named status ports everywhere; one bit vocabulary for per-row flags | Done 2026-11-09 and 2026-11-10: `candidateOverflow` and per-row `candidateTruncated` on the joins in every nearest mode, passed through by zonal statistics, buffer selection and line-length-per-polygon; named ports beside the region statistics header words; `candidateOverflow` on line clip, minimum clearance, shared paths and linear referencing; `approximation` on segmentize and great-circle arcs; `iterationLimitReached` on reachability, the cost matrix, cost distance, watershed, height above drainage, watersheds, stream order, profile peaks and spike repair; `requiredCount` and `overflow` on depression hierarchy; documented status bits in point cloud, the spherical family, focal pairs, the Kalman filter and restriction. Wave 3 (2026-11-11): `status.iterationLimitReached` on the raster region graph, `status.invalidCount` (cells still invalid after the fill) on raster fill no-data, `status.uncertainCount` on island attachment. Remaining: per-row flags have per-family bit meanings; registration, temporal, unmixing, seasonal break, remapping and the time-series classes keep documented per-cell code words (naming them means new outputs; region statistics is the precedent for doing so); the status word of `GPURasterFractionalZonalDistribution`; tiled flow and catchment summary; network inhomogeneous K, isochrones, map matching and flow aggregation; hull and minimum-bounds overflow bit sets; closest approach keeps `entryOverflow` and `hitOverflow`; most statistics classes. To decide: three classes whose `output.status` is a packed view carry their named ports under `output.ports.status` (depression hierarchy, OPTICS, network kernel density), a shape nothing else uses | M | H |
| STD-05 | Numeric per-frame props read from a parameters view, not written into WGSL | Walking cost and fractional coverage require `parameters` since 2026-11-11 (the shader-literal path is deleted). Baked: point-cloud thresholds and viewpoint, about 20 statistics props, the other raster affines. `GPULocalKriging` bakes its covariance model unless `parameters` is supplied; making the per-frame path the default changes results and remains. Three parameter mechanisms coexist | M | M |
| STD-06 | Declared context on every contributor | Done 2026-11-09 to 2026-11-11: optional `spatialContext` on the spherical classes, both S2 coverings, the point-process and validation classes, local kriging, local RBF, natural neighbour, conditional simulation, both Vecchia classes, the Kalman filter and restriction; `cellSizeMode` on six terrain-processing classes and walking cost; walking cost and fractional coverage read their affine from a required parameters view. Terrain needs nothing further (section 12 also made seven terrain classes and the point-horizon classes take `rowDirection` from the grid affine). `cellSizeMode` on the categorical and distribution fractional aggregators and on patch metrics since 2026-11-11: the row edges are read from the `parameters` view when one is given and written from `grid.affine` otherwise (requiring `parameters` for a non-uniform mode would remove the written path). Remaining, all raster: the reprojection affines; patch metrics column step, row step and pixel area are compile-time in every mode; a packer `extent` independent of `grid.affine` in isolines, isobands and the flow family, which also ignore the affine's row direction (section 12.4; a flip plan is in the math review folder); pixel-only classes do not declare their coordinate model; a shared raster-context resolver in `analysis-core/raster-grid.ts`. Per class: [research/parallel-pass-2026-11-10/raster.md](./research/parallel-pass-2026-11-10/raster.md) | M | H |
| STD-07 | One no-data rule: input `mask`, output `validity`, rasters through the published band validity | Done 2026-11-09 to 2026-11-11, with no alias left: `mask` is the only input filter on `GPUScalarScale`, the datum grid transform, encounters and closest approach, isolines, isobands, isoband rings, sampling, profile, stretch and reclassify; `validity` is the only output flag on the statistical filter and thinning; `validity` input on six terrain-processing classes. Point-cloud rasters default to `'south'` rows, the raster convention; `PointCloudRasterGrid` has no affine or signed cell height, so an explicit direction that contradicts the data cannot be detected. Terrain needs nothing further. Wave 3 (2026-11-11), no alias left: input `mask` on texture, remapping, temporal, unmixing, seasonal break, 17 time-series classes, 9 embedding classes, patch metrics and sieve, and new on arithmetic, cell statistics and the weighted overlay; registration takes `referenceMask` and `movingMask`, embedding change `beforeMask` and `afterMask`, change detection `observationMask`; the overlay's output flag is `validity`; the conditional's `outputMask` is `conditionResult` (a condition flag, not a no-data flag; the name is to confirm). Remaining: these classes take bare value views, not the published raster band validity; the spectral index `outputValidity`; `GPUDistanceField.seedMask`; the library-wide naming decisions of section 12.4. The Hampel and outlier `flags` mean 1 = bad and stay | M | H |
| STD-08 | One layout for variable-length geometry output and one provenance rule | Done 2026-11-11: `count` and `requiredCount` count features in the segmentize family and rectangle clip (vertices are in `vertexCount` and `requiredVertexCount`); overlay and buffer report rings in `ringCount` and `requiredRingCount` and lost the ring-based `requiredCount` and `geometry.sourceIds`; linear referencing reports `candidateOverflow` and an optional `requiredCandidateCount`; shared paths carry `leftSourceIds` and `rightSourceIds`; line merge carries `sourceIds`; the layout types are in `analysis-core/contracts` and `ports.ts` states the provenance rule. Remaining: offset curve and outline leave the fixed-stride layout; per-feature source rows for overlay and buffer (L, changes results); line chunk's `requiredCount` is a lower bound when one line needs more pieces than the capacity; hull and minimum-bounds overflow bit sets. Inventory: [research/parallel-pass-2026-11-10/geometry-output.md](./research/parallel-pass-2026-11-10/geometry-output.md) | M | M |
| STD-09 | `analysis-core` becomes the only home of the contract | Done 2026-11-11: the core `validatePackedView`, which shadowed a published function, is `validateSinglePackedView` (380 files; the 28 files that import the published one are unchanged); 11 `@internal` symbols are no longer exported from entry barrels and `SpatialSortCurve` is public; the 183 contract names re-exported from `gpu-spatial-analysis`, `gpu-network` and `gpu-terrain` are removed and every consumer imports them from `analysis-core`. Remaining: fold `gpu-spatial-analysis/shared` and the raster iteration gate (`cost-distance/raster-relaxation.ts`, used by line simplification) into core; move `regression-row-tiles.ts` and `line-merge/list-ranking.ts` to core when a second family needs them. To decide: export `validateSinglePackedView` from `analysis-core` (it is in no barrel) | M | L |
| STD-10 | Shared WGSL modules in core: adoption | Modules landed 2026-11-10 in `analysis-core/kernels` with headless specs (`dense-solve-wgsl.ts`, `symmetric-eigen-wgsl.ts`, `binary-search-wgsl.ts`, `uint64-wgsl.ts`, `sphere-distance-wgsl.ts`, `row-access-wgsl.ts`, `isNanValue` and `guardedAtan2` in `wgsl-common.ts`). Adopted 2026-11-11 with identical results and no fixture change: the regression Cholesky family (about 15 users), the equilibrated Cholesky, the global kriging solver (pivoting LU), about 17 float-test copies; `GPUCompositeScore` uses the shared eigen solver (section 12). Remaining: sites whose fixtures would move (local kriging, RBF, space-time kriging, Vecchia, the panel-lag determinant and inverse, the eigenvalue range, the point-cloud Jacobi and Kalman solvers); about 140 binary-search copies (the network-noding ordered key treats -0 as +0 and is not the core function); the 64-bit copies, which use (low, high) words where the module uses (high, low); mask tests and row accessors; a QR module for the three non-square solver sites; four deprecated float-test constants still imported by about 110 files; a composer that removes duplicate declarations; four wrappers and about 100 local lambdas around `createWGSLKernelNode`. Site table: [research/parallel-pass-2026-11-10/w2-adoption.md](./research/parallel-pass-2026-11-10/w2-adoption.md) | M | M |
| STD-11 | One rule for recipes | Done 2026-11-09: `RecipeBuilder.addGlue` registers raw glue kernels as named stages (11 recipe files), every recipe reports per-stage status, the 12 urban and landscape recipes accept caller-owned `output` views, and five recipes' status counts are reductions. Remaining: those 12 are still kernel-only functions, not classes; `recipe-utils.ts` and `expansion-recipe-utils.ts` are two helper modules; `GPUQuantileLISA` is a class in `recipes/`; `interpolation-diagnostics` and `network-scenarios-recipe.ts:383` still summarize in one invocation | M | M |
| STD-12 | Layout: one class per file, package roots hold only `index.ts`, test directories mirror source, docs grouped by domain | 16 loose root files; 10 multi-class files; test directories named for the luma.gl branch; docs split into `operators-*` and `*-expansion` pages; `geospatial-analysis` is not a coherent entry | M | L |
| STD-13 | One registry | Done 2026-11-09: the operator catalog covers every exported contributor of all nine entry points (`getGPUAnalysisOperatorCatalog()`; 175 added entries in 40 capability families under `ontology/all-entry-points/`), with `tier` (`scale` or `reference`) and confirmed `limits`; a test fails when a contributor has no entry. Remaining: rename the 14 wave-named capability IDs (`expansion-*`, `coverage-*`; the website generator and kepler enumerate them); check the coverage matrix, docs index and website catalog against the registry; `kepler-analysis-catalog.ts` still states the old Dorling limits; regroup `docs/api` by domain | M | M |
| STD-14 | State packed-only input as policy and guard the binding size everywhere (generalizes PLAT-02), or plan chunked input for joins, aggregation and density | 294 class files require one packed view; 11 accept chunks; only some check the binding size | S | M |

Rework to a GPU-scale design (GPU). Cheapest first.

| ID | Item | Where (report) | Effort | Pri |
| --- | --- | --- | --- | --- |
| GPU-01 | Replace the remaining single-invocation loops over data | Done 2026-11-09 for the raster convergence scans, registration peaks, fractional-coverage compaction, stream network, coregistration sums, SMRF and statistical-filter counts, five recipes, urban and landscape totals, the border K, the balanced-sample rank and the network K radius finish. Remaining (each has a comment stating the size at which it hurts): masked registration running means and refinement scoring, zonal statistics over fractional tuples, the adjacency union-find, structures pack, patch reconcile, counterfactual quantile mapping, the Ang shell scan, `spatial-validation-design.ts` fold designs, `GPUInequality` | raster, recipes, network reports | M | M |
| GPU-05 | Index by default (widens PLAT-03) | Done 2026-11-09 and 2026-11-10: M3C2 takes an optional grid; unindexed point-cloud work above 2^27 pairs is rejected; isovists, street profiles, the visibility graph and the viewpoint classifier switch to the segment BVH above 4,096 segments; fuzzy contiguity uses a Morton-sorted BVH (65,536 features; 9 ms for 65,536 shuffled rows) with a query box that is conservative at the tolerance; local kriging takes a neighbour grid that ranks exactly like the scan, and `'auto'` scans only up to 4,096 samples; OPTICS takes a neighbour grid; cross-variogram and marked statistics take an opt-in `gridSize` whose finish and moments passes are parallel. Network snapping refuses unindexed work above 2^27 pairs for direct callers (`maximumUnindexedWork`); the K function and the drive-time catchment recipe raise that limit on purpose, because they have no snap radius for the indexed path, and say so in their documentation. Done 2026-11-11: island attachment takes `nearestSearch` (`'auto'` uses the neighbour grid above 4,096 rows; 7 ms against 52 ms at 60,000 rows) and equals the scan, ties included; planar cross-join queries of `GPUNeighborSearch` outside `bounds` return what a scan returns (a query far outside walks the whole lattice, about three times the cost at 200,000 targets). Remaining: MGWR local fits (route: `gwr-grid-index.ts`); the island-attachment rescan, for an island with 15 or more rows tied at the nearest distance, is a one-invocation loop over all rows, counted in `status.uncertainCount` but not bounded; great-circle cross-join queries outside `bounds` stay empty and are counted in `invalidCount`; the dense default of cross-variogram and marked statistics, and their pair pass when the radius reaches the data extent; callers of snapping in kepler (`catalog-gpu-snapping.ts`, which falls back to the CPU above the cap), two lens examples and the two deck examples do not pass `candidateCapacity`. The point-cloud private grid cannot move to the published `GPUGridIndex` (cell size derived from bounds, closed upper bound, unordered points within a cell, no 2-D raster mode); the upstream request is written in a comment on `createPointCloudGridIndex` | terrain and point-cloud, recipes, statistics, network, core reports | L | H |
| GPU-07 | Indexed hole-to-shell assignment; group-ranged face scan in make-valid; group-aware noding | Done 2026-11-09: the next-edge choice in ring assembly is exact for planar input with shared vertices (the suspected wrong ring was reproduced: two edges 2^-43 radians apart tied in float32). Remaining: hole-to-shell loops every hole over every ring (a sort by minimum x does not bound it; use a BVH over shell boxes per group); `gpu-make-valid.ts:349` scans every noded segment per face (sort segments by group and loop the group's range); `orientSign` is now called from two sites in the match kernel, against the one-call-site rule in `exact-geometry-wgsl.ts:34` | `analysis-core/ring-assembly`, `geometry-topology` (polygon report) | M | M |
| GPU-08 | Terrain-processing, remaining serial designs | Done 2026-11-09: flow distance and stream network use pointer jumping; compaction and sums use published primitives. Remaining: depression hierarchy (reuse `analysis-core/kernels/parallel-merge-tree.ts` on the negated surface with ascending spill edges, a lowest-cell elder rule and segmented stage curves; TE03), the per-pit breach search, per-basin catchment statistics, the O(n²) correlated coregistration rows | terrain report | L | M |
| GPU-09 | Remaining regression fits that run in one invocation | Done 2026-11-09: probit, both regime regressions and the instrumental-variable covariance are row-tiled or sorted per regime, with the cap raised from 65,536 to 2^24 rows. Remaining: panel regression, the MGWR global fit, SUR (`spatial-sur-nodes.ts:97`), panel lag ML's dense determinant; `regression-row-tiles.ts` is the helper to reuse | statistics report | M | M |
| GPU-10 | Network-optimization | Done 2026-11-09: reference tier and real limits documented and registered. Remaining: drop Louvain for the published modularity optimization (NW04), delete point routing (needs a same-edge kernel and predecessor edges from the cost matrix), move facility optimization into location-allocation, rework flow (NW02), rational access and time routing | network report | L | M |
| GPU-11 | Urban and landscape: landscape metrics on `raster-patches`, patch proximity without patches × cells buffers, tessellation and visibility past 64 buildings and 256 points | recipes report | L | M |
| GPU-12 | Geometry reworks | Done 2026-11-09: level-synchronous S2 polygon and cap coverings as an opt-in (`refinement: 'level-synchronous'`; caps reach level 15), shared paths on pointer jumping, shape-descriptor hull strategy from a declared maximum. Remaining: make the level-synchronous coverings the default (the serial path's partial output at budget exhaustion is pinned by fixtures) and gate their rounds (needs a GPU-sized `GPUScan`); spherical arc overlay, constrained Delaunay flips, offset-curve components, ratio concave hull, segment snap, shape-descriptor moments | polygon, line, cells reports | L | M |
| GPU-13 | Remote sensing and dataframe | Done 2026-11-09: parallel registration argmax; group moments dispatched only for occupied groups. Remaining: region graph (Borůvka rounds over a sorted edge list), sparse fractional-coverage candidates, one 256-lane workgroup per group in group moments (hurts near 2^20 rows in one group), counts held in float32 (2^24-row cap on profile, quantiles, class breaks), `GPUInequality` | raster, core reports | M | M |

Merges, upstream moves and missing capabilities.

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| DUP-01 | The 30 merge verdicts. Largest: the two kriging classes; cross-variogram into the space-time variogram; three facility solvers; walking cost into cost distance; S2 measures and hierarchy into the cell classes; great-circle arcs into segmentize; label point with maximum inscribed circle; canonical equality into the predicates; the dense resistance recipe into the sparse one | M | M |
| DUP-03 | Operation parity across H3, S2 and Quadbin (matrix in the cells report); one `GPUCellFamily` type | M | M |
| UPS-01 | luma.gl pull request: graph algorithms accept command-graph views, then delete `createNetworkAnalyticsTopology` and the local components and degree | M | M |
| UPS-02 | Offer upstream: generic image operators (rank filter, reconstruction, skeleton, medial axis, phase correlation, GLCM), generic graph algorithms (reachability engine, Brandes, SCC, flow), exact quantiles | L | L |
| UPS-03 | Use the published raster tile-halo and cross-tile components instead of per-contributor halos; build arithmetic, stretch and focal statistics on the published band math, contrast and neighbourhood operators | M | M |
| GAP-01 | Geometry: pairwise overlay that keeps both inputs' attributes, per-feature buffer, attribute dissolve of overlapping polygons, topology-preserving simplification, Voronoi polygons, split by a second geometry, antimeridian cutting, points along lines | M each | M |
| GAP-02 | Raster and terrain: QA bits to masks, clip by polygon, rasterize lines and points, polygonize, zonal histogram, median and majority; pour-point snapping, stream orders, whole-DEM breaching | S to M each | M |
| GAP-03 | Network: demand loaded onto shortest paths (edge volumes; pairs with MOB-05), nearest points of interest, street-network statistics, graph simplification, angular (space-syntax) measures | M each | M |
| GAP-04 | Statistics and dataframe: spatial Durbin, SLX and SDEM, GWR prediction, multi-attribute clustering, SARAR; window functions, rolling windows, per-group regression, as-of joins | S to M each | L |
| GAP-05 | Point cloud and applied: voxel centroid downsample (the voxel merge has no producer for its input), k-nearest above 64, cloth-simulation ground filter; FRAGSTATS core set at raster scale, Circuitscape pairwise modes (with EC03-R), least-cost corridors | M each | L |

Found while implementing (2026-11-09).

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| FX-17 | `atan2(y, -0.0)` returns `-sign(y)·π/2` on the test GPU, and `atan2(0, 0)` returns NaN. Fixed 2026-11-10 with one helper, `guardedAtan2` (`GUARDED_ATAN2_WGSL` in `analysis-core/kernels/wgsl-common.ts`, contract section 11), wherever a spec reproduced it: the S2 cap and polygon covering, the H3 boundary longitude, the terrain aspect with north-up rows, the Imhof swing in relief shading, the planar trajectory heading; section 12 then adopted it in about twenty more kernels (trajectory headings, linear referencing, the shape generator, rose, natural neighbour, isovists, spherical overlays, the point buffer). Pinned by a spec and left unchanged because the second argument cannot be `-0.0` on this GPU: spherical polygon containment, `geodesic-kernels.ts:212`, the point-cloud covariance rotation, the relief main kernel. Remaining: check the table in [research/parallel-pass-2026-11-10/fix-atan2.md](./research/parallel-pass-2026-11-10/fix-atan2.md) against what section 12 converted (variogram, change detection and rhumb were triaged by reading only) | S | L |
| FX-18 | Kernels that read rows belonging to no ring or line. Padding rows are reused scratch storage, so a stale NaN raised a false overflow and results could depend on which other nodes a graph held (found when removing an unrelated overlay output made a tessellation recipe report a building incomplete). Fixed 2026-11-11, each with a spec that poisons the padding rows: the overlay's `combine-positions`, the snap-rounded noder, `GPULineSplit` (and with it line noding, polygonize, network noding, offset-curve components and loop removal), `GPUMakeValid`, the coverage `snap-range`, line clip by polygon (which emitted a phantom segment from leading padding rows). Not fixed, reachable only when the first offset is above 0: `getSnapOrigin` reading row 0 in `coverage-topology-kernels.ts`. Examined 2026-11-11 and safe, each with a spec that poisons the padding rows: line clip's polygon side, Voronoi, the ungrouped concave hull, the join bounds and edge layout, loop removal and offset-curve components. The grouped concave hull and Delaunay specs cover only stale output words, because the kernels that would read padding cannot be reached from a spec. Not examined: triangulation; `GPUSegmentIntersection` alone with padded polygon positions. Reports: `w2-ring-assembly.md`, `review-kernels.md`, `w2-fix-kernels.md` in [research/parallel-pass-2026-11-10](./research/parallel-pass-2026-11-10) | S | M |
| STD-15 | Decide the public vocabulary for statistic outputs (`minimums` against `minimum`, `counts` against `cellCounts`, `validCount` against `count`); the table of current names is in the dataframe implementation notes | S | L |

## 3. Dropped by upstream-first

Not backlog items. They were removed on purpose; see [upstream-first-pending.md](./upstream-first-pending.md) for the replacement of each symbol and [upstream-workarounds.md](./upstream-workarounds.md) for the full register. In short:

- Large-graph SCC, the iteration gate, direct local clustering node, command-graph-transient CSR views.
- Histogram and skip-cube crossfilter (use published `GPUCrossfilter`).
- Residency arena, resident row selection, tile LOD.
- Onesweep sort, look-back scan, double-single helpers, benchmark harness.
- Arrow batch stream, partition descriptors, `GPUCPUOperation`, `GPUKernelNode`.
- Command-graph recording hooks and `captureGPUCommandGraph` (replaced by `computation-record`).
- Dataframe query description and the `gpu-core`, dataframe-query, SCC and SQL lens packs.
- Compensated sums and block compaction in published primitives (accepted regressions). Bounded dispatch above about 4.2M rows was verified obsolete (PLAT-01).

## 4. Corrections to the branch documents

The archived documents say the following. These statements are out of date or wrong for this repository:

| Archived claim | Current fact |
| --- | --- |
| Quality roadmap "still open": AZP articulation test | Done: variant `local-articulation-candidates` in `gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions.ts:801` |
| Quality roadmap "still open": D-06 crossing points to snap rounding | Done: `gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts` |
| Algorithms roadmap line 29: variogram fitting is a CPU helper | `GPUVariogramModelFit` exists (`dataframe-analysis/pair-statistics/gpu-variogram-model-fit.ts`); IN01 is implemented |
| Spatial roadmap phase 9: no likelihood, panel, SUR, probit or MGWR admitted | ST03 to ST06 are implemented (panel ML, SUR, probit, MGWR) |
| Final review: no planned or partial rows | KG06 is Partial in this repository. EC03 is Implemented again (sparse recipe restored 2026-11-07); its raster adapter is EC03-R |
| Kepler plan: residency arena and tile LOD | Dropped; kepler uses its own `resident-*` code |
| SCC and centrality notes in the follow-up document | SCC is capped at 1,024 vertices here; centrality still unrolls rounds and rejects more than 32,768 command nodes (NW-CENT) |
| KG03 exact natural breaks limited to 1,024 rows | Recompute mode reaches 8,192 rows (`gpu-class-breaks.ts:127-129`); materialized storage is 1,024 |
| Computation-visualization roadmap: GPU selection backend is an interface only | `GPUCrossfilterSelectionBackend` exists in lenses |
| Roadmap and matrix paths `modules/graph/src/...`, `docs/api/*` | Dead. Code is in `modules/analysis/src/<directory>`; entry points are `@luma-contributors/analysis/<directory>` |

## 5. Out of scope: belongs in luma.gl

Do not implement these here. If a consumer needs one, open a luma.gl pull request.

- Raster base classes and operations: exact or standalone percentile, wide counters, 4K+ transparent partitioning, `rgba8unorm` contrast, FFT convolution (`archive/gpu-raster-roadmap.md` tranches 0 to 6.2 and 8.x).
- 3D datum contract, ECEF, static Helmert (`gpu-project`, P.7a).
- Compute primitives and their fixes: bounded dispatch, compensated sums, block compaction, faster sort and scan.
- Public `GPUJacobiPCG`, `GPUSegmentedReduction`, `getSpatialOperationNodes`, `makeGeometryGPUVectors`, command-graph annotation hooks, GPU-gated continuation.
- SQL engine gaps (KG06), `CommandEncoder` labels, rendering and trace lens packs.
- Physics, splats, glTF, shaders, effects, ANARI, area light, gpu-trace, gpu-batch and the deck.gl core patches.

## 6. Order of work

1. The full gates of 11.4 for sections 11, 12.6 and 13 together, with lens records and posters regenerated after section 12.6 ends; then the decisions listed in 11.6 and 11.8. UF-01 is done locally for the work up to 2026-11-10 morning; committing the rest and pushing are the maintainer's decisions.
2. The ten findings that are wrong on ordinary input in [research/math-correctness-review-2026-11-10](./research/math-correctness-review-2026-11-10/README.md), which has no roadmap rows yet; then the rest of KE-TESTS. Done 2026-11-07: CI gates (UF-02), test and example typechecks (UF-04), AUD-01, the NW-CENT interim guard, the EC03 sparse recipe.
3. UF-06, then M items by consumer demand: ST06, ST06b, ST07, PP04, NW04, NW-KSP, TE06, R6.3, R7.2.
4. NW-CENT structural fix when luma.gl ships a GPU-gated continuation.
5. At each `@luma.gl` bump, run the upgrade procedure in [upstream-workarounds.md](./upstream-workarounds.md).
6. Website, independent of the items above: the open items of sections 7.5 and 7.6, starting with WEB-PIC-12 (an independent re-score of the pictures) and WEB-INT-16 (text in pictures). How-it-works pages: the WEB-HIW rows of section 9.6, starting with WEB-HIW-01 and WEB-HIW-02.
7. Spatiotemporal, section 2.15: what remains after the 2026-11-10 implementation (section 8.5): LN-ST, the R-only fixtures of PLAT-06, A5 in RTS-08, then the M rows.
8. Library convergence, section 2.16. Landed 2026-11-09: the contract and its checker, the registry, and the cheap reworks. Next: FX-17 and LN-REC2; STD-10 once the solver files are quiet; the rest of STD-03, STD-06 and STD-07 by directory; the breaking half of STD-09 in one pass with lenses, kepler and the website; then GPU-05, GPU-07 and the larger reworks by consumer demand. New contributors follow `contributor-contract.md` and add their catalog entry and kernel-audit case in the same change. GAP rows wait for a consumer.
9. CPU twins and mixed CPU/GPU graphs, section 13. Landed 2026-11-11: the operation form, contract section 14 and 125 twins (sections 13.5 and 13.8). Next: FP64-01 and FP64-02 (native fp64 in kernels, section 13.7), CPU-07 (a `docs/api` page), CPU-05 (the twin in the catalog), the leads in the research folder's two `existing-contributor-findings` files, and the upstream pull request UPS-CPU-1.
10. New contributors, section 14. Nothing is implemented; the research and three GPU probes are in [research/new-contributors-2026-11-11](./research/new-contributors-2026-11-11/README.md). First the decisions of 14.5, then wave 0 (the shared fitting kernels, the forest layout, the probes on a second device), then LRN-01, EMB-01 to EMB-05, SIM-01 and CART-03.

## 7. Plan: card header pictures (WEB-INT, WEB-PIC)

Status date: 2026-11-09. Scope: the live scene behind every catalog card and function-page hero on the Gridlight site: 428 operation runs (`website/src/gpu/runs/entries/**`) and 10 flagship demos (`website/src/gpu/demos/**`). The audit that motivated the work, with the state before it and the per-entry proposals, is in [research/card-interaction-audit-2026-11-09](./research/card-interaction-audit-2026-11-09/README.md). The contract that run authors follow is in `website/src/gpu/runs/README.md`, "Pointer interaction".

Two goals. The pointer does what a reader expects for that operation, in the picture's own space (WEB-INT). The data and configuration of each picture explain the operation before and while the pointer moves (WEB-PIC).

### 7.1 Status

Implemented on 2026-11-09 in the working tree (not committed). What is left is in 7.5 and 7.6.

- **Framework.** `defineOperationRun` takes an `interaction` (`website/src/gpu/runs/interaction.ts`): the gesture's kind, hint, rest position and a `resolve` that returns parameter values, debounced compile-time values (`settle`), glyphs, a readout, a swipe curtain or a hovered element, and can read the operation's output through `peek`. `pick.ts` has the CPU lookups, `pointer-overlay.ts` draws glyphs and the probe spotlight, and `pointer-driver.ts` gives the hand-written demos the same behaviour. `canvas-pointer.ts`, `gpu-canvas.tsx` and `use-live-slot.ts` carry the runtime rules of 7.4.
- **Scenes.** All 428 runs and all 10 demos declare a gesture. The generic horizontal scrub remains only as a fallback for a scene without one.
- **Pictures.** Each group applied the audit's fixes through its shared helpers: fixed ranges, the subject fitted to the frame, input against output, zero and reference lines, a grey ramp for illumination, distinct pictures for siblings.
- **Posters.** All 878 posters (439 entries, two themes) were regenerated at rest, with the gesture's handle drawn.
- **Gates passed.** `verify-runs.mjs` 428 of 428; `verify-demos.mjs` 10 of 10; `audit-card-interactions.mjs`; `check-card-interactions.mjs` on one card per gesture; `check-catalog.ts`; `audit-demo-data.ts`; `yarn website-build`; `check-all-pages.mjs`; website typecheck and lint. An independent review of the framework found 15 issues; 14 are fixed and one is open (WEB-INT-13), as is one of its minor notes (WEB-INT-14). The fixes were verified by the gates above but not reviewed a second time.

### 7.2 Measures

| Measure | Before | Now | Target | Checked by |
| --- | --- | --- | --- | --- |
| Scenes that declare a visualization-specific gesture | 3 of 438 | 438 of 438 | 438 | `audit-card-interactions.mjs`, `verify-demos.mjs` |
| Cards whose gesture drives an input-data knob | 160 | 0 | 0 | verifier |
| Cards that recompile while the pointer moves | 111 | 0 (about 25 runs debounce a compile-time prop with `settle` and `buildCache`) | 0 | verifier |
| At least three distinct pictures at five pointer positions | not measured | 424 of 428 | every card | verifier |
| Largest pixel change across the gesture | scrub median 10.5%; 67 cards under 2% | median 9.5%; 85 cards under 3% (mostly probes and single-vertex drags) | no card under 3% | verifier |
| Pointer move to presented frame | not measured | not measured | 95th percentile within two frames | WEB-INT-15 |
| Start to first frame in the harness | median 565 ms | median 668 ms, measured with posters rendering alongside | under 250 ms warm | WEB-INT-09 |
| Reviewed picture quality 1 or 2 | 177 | not re-scored; each group reviewed its own contact sheets | 0 | WEB-PIC-12 |
| `verify-runs.mjs` | 425 of 428 pass, 26 inert sliders | 428 pass, 11 inert sliders | 428 pass, 0 inert | `verify-runs.mjs` |
| Keyboard, reduced motion, touch, both themes, no GPU | one card | twelve cards, one per gesture and one demo | one per gesture plus every high-priority entry | `check-card-interactions.mjs` |

### 7.3 Interaction grammar

Every scene declares one primary gesture. The pointer always drives a real prop or a real input of the operation, or reveals structure the operation computed.

| Gesture | The pointer is | What happens | Glyph at the cursor | Runs |
| --- | --- | --- | --- | --- |
| `probe` | hovering one element | its linked items light up: neighbours, join partners, parent cell, downstream path, class members, the features behind a chart bar. Everything outside the hovered neighbourhood dims (the spotlight) | outline of the element, links to its partners | 152 |
| `drag` | carrying one site, vertex or feature of the input | the output re-forms around it | the carried piece, emphasized | 58 |
| `place` | a spatial input: observer, source, target, pour point, query point | the operation re-runs for that location, snapped to a cell or node | pin or crosshair, with a range ring when the operation has one | 44 |
| `reach` | a distance: how far it is from an anchor or from the nearest input geometry | the buffer edge, search radius, tolerance or cost limit follows | ring or corridor of that distance, tick to the nearest geometry | 43 |
| `level` | a value: the cell value or axis position under it | the threshold, break or contour level becomes that value | the contour through the cursor, or a rule on the histogram | 38 |
| `swipe` | a curtain | input left, output right; no re-run | curtain line | 27 |
| `pad` | two real props, one per axis | used only when the operation has no spatial handle | marks for the two values | 21 |
| `time` | a playhead | time, iteration or sequence position follows x | a tick on a time strip | 19 |
| `window` | the centre of a box, mask, kernel footprint or region | the selection or footprint follows | the outline itself | 17 |
| `aim` | a direction: its angle around an anchor, and optionally its distance | azimuth and altitude, wind, rotation and scale follow | arrow from the anchor to the cursor | 9 |

Choosing: prefer `place`, `aim`, `reach`, `level`, `window` and `drag` over `probe`; `probe` over `pad`; `pad` over a one-axis scrub. One gesture per scene, no modes. A `probe` draws the linked structure and reads out a computed value, not only a highlight, because it carries a third of the catalog (most statistics, regression and weights entries have no spatial input).

Examples as built:

- `viewshed`: the pointer is the observer; a dashed `maximumDistance` ring follows.
- `slope-aspect-hillshade`: the angle around the picture centre is `azimuthDegrees`, the distance from it is `altitudeDegrees` (far is a low sun); a sun arrow is drawn.
- `fill-nodata`: a curtain, the raster with its gaps left, the filled result right.
- `neighbor-search`: the hovered point's links, read back from the operation's output, are drawn with a ring at the farthest neighbour.
- `focal-correlation-regression`: the pointer moves the regression window; an inset scatter shows the pairs and the fitted line.
- `point-to-cell` and the cell runs: the pointer is one extra row through the real `GPUPointToCell`, so no CPU H3 or S2 library answers for the cursor.
- `knox-test`, `mantel-test`: the hovered event's neighbours in space and time, over the shuffled distribution with the observed value marked.

### 7.4 Feel

These rules apply to every gesture and live in the runtime, not in each run.

1. **Rest is the poster.** Each interaction declares a rest position that reproduces the default parameters. The poster is captured there with its glyph drawn, so the idle picture already shows the handle and the first live frame matches it. Keyboard focus starts a card at rest.
2. **Glide in.** When a scene becomes live under a pointer that is already somewhere, the gesture glides from rest to the pointer over about 180 ms. With reduced motion it jumps. Home and Reset jump straight back to rest.
3. **Latest wins.** Pointer events are coalesced per animation frame. The graph runs at most once per frame and never queues.
4. **The handle stays under the cursor.** After the glide in, the gesture is resolved at the pointer itself, once per frame; glyphs and placed positions never trail the hand.
5. **Sticky snapping.** A snapped node, cell or tract changes only when the new candidate is clearly nearer (about 15%). This stops flicker on boundaries.
6. **Reachable ends.** The inner 90% of the picture spans the whole range. Props that span more than a decade map logarithmically.
7. **Hold on the card.** While the pointer is on the card's text the last state holds. Leaving the card releases the scene.
8. **Leave softly.** The canvas stays for a 200 ms grace period, so crossing a gap or re-entering does not restart the device, then the poster returns.
9. **Never recompile under a moving pointer.** A compile-time prop is returned as `settle`, snapped to at most five values, and applied 140 ms after the pointer rests; `buildCache` keeps visited builds. The gesture is paused while a rebuild is pending, so it never sees new values with old buffers.
10. **Say what the hand is doing.** Before the first move the chip shows the gesture (`move the observer`). After it, a readout the run builds from real names, values and units.
11. **Keyboard is the same gesture.** Arrow keys move a virtual pointer from the rest position. Home returns to rest. The readout is in the canvas `aria-label`.
12. **A slider owns what it changed.** When the reader moves a slider, the pointer stops writing values until it moves again; glyphs and readout that would contradict the slider are hidden; everything the gesture derives from the values is resolved again.
13. **Function page.** The hero uses the same gesture. Values the pointer writes move the sliders. A click pins the state so the reader can go to the sliders; a second click releases it. Reset returns to rest.

Not built: touch steering on the function page (WEB-INT-08) and a keyboard path to the hero's gesture (WEB-INT-14).

### 7.5 Open framework items (WEB-INT)

| ID | Item | Current limit | Pri |
| --- | --- | --- | --- |
| WEB-INT-08 | Touch steering on the function-page hero | Cards ignore touch by design (a tap navigates). On the hero a horizontal drag should steer one-axis gestures and a short press should capture two-axis ones. Not built | M |
| WEB-INT-09 | Hover latency spike | Each live card still creates its own GPU device. Published `@luma.gl/core` has `device.createPresentationContext`; measure one shared device presenting to each live card. Adopt only if hover-to-live drops below about 250 ms with no device-loss or validation regressions | M |
| WEB-INT-13 | Restart with pointer-written values hides the handle | When the hero restarts (theme change, scrolled out and back) with values the pointer wrote, the gesture is at rest and its glyphs and readout stay hidden until the pointer moves (`define-run.ts`, the `isUntouched` branch). Remember the pointer position across restarts in `gpu-canvas.tsx` | L |
| WEB-INT-14 | Keyboard path to the hero's gesture | `canvas-pointer.ts` binds arrow keys only on cards. The hero canvas is not focusable, so the gesture there is pointer-only; the sliders remain the keyboard path | M |
| WEB-INT-15 | Pointer-to-frame timing | The verifier checks that the picture follows and never recompiles, not how fast. Add a timing check and set the target from it | L |
| WEB-INT-16 | Text in pictures | Built, wired and first used 2026-11-10: text is drawn on the GPU from a signed-distance glyph atlas (posters are canvas screenshots, so DOM text would never reach them) in `website/src/gpu/runs/text/`, with label helpers for axes, bar labels, legends and matrix ticks; a run adds `{layer: 'text', ...}` to `draw`. The atlas loads lazily and only for runs with text; it is a Modified Version of two KaTeX fonts under `OFL-1.1` (notice in `THIRD-PARTY-DATA.md`; that reading has not had a legal review). Labelled: `spatial-regression-diagnostics`, `network-statistics`, `spatial-weights-summary`, `class-breaks`, `color-scale`, `knox-test`, `mantel-test`, `global-permutation-test`, `network-cost-matrix` and the nine runs of the shared regression coefficient panel; all pass the verifier in both themes. `verify-runs.mjs` no longer counts a text layer as drawn output (the failing case was not exercised). Labelled 2026-11-11 (section 11.8): the rest of the ranked list and the runs reworked that day, about 60 more. Remaining: `study-window-preparation`; the atlas has no `^`, `'` or superscript two; glyphs cannot be coloured and the chart layer has no log axis (both asked for by the picture streams); on a card the hover hint chip covers the bottom 30 pixels until the pointer moves; the function-page hero's poster text is about three times the live size until the canvas fades in; phone width (111 px) was not checked; the standard-error bars, probit marginal-effect bars and the SUR residual scatter are unlabelled | M |
| WEB-INT-17 | Highlight as a layer feature | Probes draw highlights as glyphs or as render-only mask buffers written from `resolve`. A `highlight` option on the dots, polygons and raster layers (by id, value or mask) would remove about a dozen local helpers | L |
| WEB-INT-18 | `record-demo.ts` for the six computation runs | The graph layout from `describeGPUCommandGraph` differs after a rebuild, so five runs warn that the picture differs after parameters return to defaults; readbacks are not cached, so three runs rebuild to redraw; `extra` is not passed to `resolve` | L |

### 7.6 Open picture items (WEB-PIC)

| ID | Item | Detail | Pri |
| --- | --- | --- | --- |
| WEB-PIC-09 | Data that fits | No new extract was added. `normalized-difference-water-index` and `make-raster-band` still have no open water; `normalized-burn-ratio` shows a post-fire scenario derived from the measured July to September 2021 NDVI drop and says so in its caption. A Sentinel-2 scene with water and a post-fire date would replace both. The lidar tile classifies almost bimodally, so four ground-filter sliders barely act; a tile with intermediate relief would fix it. `pair-gather` needs two related real point sets. Register each new asset in `website/build/demo-data-sources.ts` and `THIRD-PARTY-DATA.md` | M |
| WEB-PIC-12 | Act on the independent re-score | Re-scored 2026-11-10 from the pictures of 2026-11-09 ([research/picture-rescore-2026-11-10](./research/picture-rescore-2026-11-10/README.md), `scores.json`): rest pictures 6, 143, 239, 40 and 0 at scores 1 to 5; gestures 5, 115, 244, 64 and 0. 149 rest pictures score 1 or 2 (the audit had 177; 81 of those improved and 53 others fell to 2). Weakest: regression-weights and spatial-statistics. Suspected operation defects to confirm: `cell-grid-path` (18 cells against a maximum of 14), `network-point-routing`, `terrain-dem-coregistration`, `multiscale-gwr`, `spatial-weights-traces`. Reworked 2026-11-11 (section 11.8): the 40 worst and 43 more runs, in six streams, then two independent reviews. The reviews scored 30 of 38 and 35 of 45 runs higher than before and found seven pictures that misrepresented their operation although the verifier passed them (a false matrix note, chords labelled as a matched route, an invented glacier, bars drawn from the bottom of a signed axis, titles that named the wrong side of a curtain, a stale caption, a zero fill hiding negative coefficients); all seven are fixed. Still weak after the fixes: `spatial-weights-traces` (needs a log axis), `chunked-weights-transpose`, `clock-encounters`, `pointcloud-neighborhoods`, `geometry-orientation` (the readout never changes), `network-snapping` (1.5% on drag), and about 20 gestures that change nothing the operation computes. The lag scatter plots proposed for `spatial-pearson` and `global-spatial-statistics` were not built, because the lag is not an output of those operations. Open: the weak runs outside these 83 (list per group in the `w3-web-*` reports), then a re-score of the whole site from new posters. Limits: three of five pointer positions per run were visible; the light theme was not scored | H |
| WEB-PIC-13 | Inert sliders | Reworked and reviewed 2026-11-10. Now acting: `maximumIterations` (`datum-grid-transform`, which runs the inverse direction), `significanceLevel`, `interceptBandwidth`, `stages`, `targetHeight`, and faintly `joinStyle` (0.43%) and `reciprocal` (0.5%). Removed from their cards because they cannot act on the checked-in tiles, with the fixed value in the caption: `window` (`pointcloud-smrf`), `slope` and `maximumThreshold` (`pointcloud-ground`), `jump` and `tolerance` (`spike-repair`); a first attempt that tilted the lidar tile and smoothed the elevation was reverted by the review as a misrepresentation. A lidar tile with intermediate relief (WEB-PIC-09) would bring them back. `pair-gather` `how` is inert again (0.46%). In `spike-repair` one damaged block stays visible in the repaired half: an operation question, not investigated | M |
| WEB-PIC-14 | Weak gestures | Reworked and reviewed 2026-11-10 ([research/parallel-pass-2026-11-10/review-website.md](./research/parallel-pass-2026-11-10/review-website.md)): about 20 runs have a real gesture change (for example `probit-regression` 1.3% to 14.8%, `constrained-delaunay-triangulation` 1.7% to 13.3%, `terrain-critical-points` 0.5% to 7.6%, `point-in-polygon-join` 2 to 5 distinct pictures). Of 27 changes that only thickened strokes or enlarged dots, the review accepted 18 and 9 were reverted. About 40 gestures stay under 3% (single-vertex drags, probes that already draw their structure); `pointcloud-statistical-filter` is back to 2 distinct pictures. The verifier compares against the rest picture, so structure shown at rest cannot raise a drag's score. Not done: a whole-group verification pass of the twelve groups, the light theme, and new posters for every run whose rest picture changed (lists in `web-g1.md`, `web-g2.md`, `web-g3.md` and `fix-website.md`) | M |
| WEB-PIC-15 | Siblings still alike | `variogram`, `cross-variogram` and `variogram-model-fit` (a rising curve on one layout); the encounter trio and the zone-events trio (one fleet, one bar chart); the two measures cards share a ramp family | L |
| WEB-PIC-16 | Left partly done | `resistance-connectivity-recipe` shows current magnitude, the potential is still nearly flat; `terrain-branching-flow` fans out modestly; `point-pattern-envelope` has bare curves; `max-p-regions` bars are small against their axis; `rank-markov` maps barely change between periods; `spherical-arc-overlay` is outlines only | L |
| WEB-PIC-17 | Suspected defects to confirm | Resolved 2026-11-10: the cross-map links of `contiguity-weights` and `fuzzy-contiguity` came from the runs' tolerance (new defaults link only touching tracts; the GPU links were not counted against the measured distances); `normalizedChoice` is NaN by documented contract when any viewpoint is invalid, so the `visibility-graph-recipe` run should drop points near walls. Open: several quad zones receive no points, so their quantile is NaN. Confirmed as documented contracts, not defects: `GPUNetworkFacilityOptimization` returns an empty selection when any demand is uncovered; `GPUStudyWindowPreparation` skips later encodings until invalidated; `GPURasterRegistration` is precise to 1 / `upsampleFactor`; the Mantel operation correlates listed pairs only | M |
| WEB-PIC-18 | Constructed scenarios | Six scenes now use constructed inputs because the operation needs a shape no extract has: `hilbert-keys` (an 8 by 8 grid), `coverage-simplification` (a jagged coverage), `fill-holes`, `medial-axis` and `skeleton` (a centred binary shape), `terrain-depression-hierarchy` (nested craters). Their cards read "Modeled scenario". The three rate-smoothing runs add simulated sampling noise to real counts and say so. Review whether real extracts could serve | L |

### 7.7 Flagship demos

| Demo | Gesture as built |
| --- | --- |
| `kernel-density` | `reach`: x sizes the kernel on a log scale; a ring at the cursor is the bandwidth. The home page hero uses this scene |
| `idw` | `place`: the pointer is the query location; the gauges that carry at least 4% of its estimate are linked to it, inside a ring at the search radius, with the estimate in millimetres |
| `hot-spots` | `probe`: the hovered gauge, its neighbourhood ring and neighbours, and its Gi* z-score |
| `local-morans-i` | `probe`: the hovered gauge, its neighbours and its cluster class |
| `dbscan` | `probe`: an epsilon ring on the nearest epicentre, the neighbours inside it, and whether it is core, border or noise |
| `k-means` | `probe`: the hovered epicentre's cluster and its distance to the centre. A dragged centre is not possible: `GPUKMeans` seeds internally |
| `terrain-hydrology` | `aim`: a sun arrow; angle is azimuth, distance is altitude |
| `delaunay-voronoi` | `drag`: one site follows the pointer |
| `network-isochrones` | `place`: the origin snaps to the nearest node |
| `shortest-path` | `place`: the destination snaps to the nearest node |

### 7.8 Decisions taken

| Decision | Taken | Still open |
| --- | --- | --- |
| Where glyphs are drawn | On the GPU in the scene, so the poster and the live picture match | Text needs a decision (WEB-INT-16) |
| Posters show the handle at rest | Yes | |
| `probe` as the primary gesture | Accepted for 152 runs, with a framework spotlight so the hovered neighbourhood reads at card size | |
| New imagery extract | Not added | WEB-PIC-09 |
| Shared GPU device across cards | Not tried | WEB-INT-09 |
| Click to pin on the function page | Built | |

## 8. Plan: implementing the spatiotemporal review (section 2.15)

Evidence and item definitions are in [research/spatiotemporal-review-2026-11-09](./research/spatiotemporal-review-2026-11-09/README.md). Each wave ends with lint, typecheck and the affected specs. A row in 2.15 is marked done only when its specs pass.

### 8.1 Waves

| Wave | Items | Exit condition |
| --- | --- | --- |
| 0 | Baseline gates | Lint, typecheck, test typecheck and `yarn test-node` recorded before any edit |
| 1 | FX-01 to FX-16 and the documentation and register corrections | Each defect has a spec that fails on the old behaviour |
| 2 | TC-01 to TC-08, STM-01, PLAT-02 | Filter, cube, Gi\*, trend, Markov, Knox and panels compose on one pipeline with no readback, shown by one spec |
| 3 | S-effort items: STX-01, 02, 03, 08; STM-03, 04, 05; MOB-02, 03, 08; RTS-01, 02, 03, 05, 06; PLAT-01, 06 | Specs against a pinned fixture or an in-repo reference |
| 4 | M-effort items: STX-04 to 07; STM-02, 06, 08, 09; MOB-01, 04, 05, 06, 07, 09, 10; RTS-04, 07, 08; PLAT-03, 07 | As wave 3 |
| 5 | STM-07 (Vecchia) | GPU log-likelihood compared with a binary64 reference, with the nugget floor and failed-site status from the experiment |
| 6 | Whole-diff review, then all gates including `yarn build` and `yarn verify-package-exports` | Gates recorded in this section |

### 8.2 Rules for the work

- Agents own disjoint directories within a wave. Sonnet implements; Opus designs the shared contracts (TC-01, TC-06, STM-01) and reviews.
- Nothing is committed without the maintainer's instruction.
- `website/**` and `modules/*/dist` are out of bounds while another session verifies the website runs against that build. `yarn build` waits for that session to finish.
- The "Not recommended" list in the review stays out of scope.

### 8.3 Limits known at the start

- PLAT-04 and the luma.gl half of PLAT-05 belong upstream under AGENTS.md. This repository gets a written proposal only, and no pull request is opened without the maintainer's instruction.
- R is not installed on the development machine, so fixtures from R-only oracles (gstat, stpp, stopp, sparr, changepoint, scanstatistics, ETAS, GpGp, move) cannot be generated here. Those items use in-repo CPU references and stay marked "external fixture pending" in 2.15.
- Python oracles are pinned to CPython 3.14 and the machine has 3.15, so Python fixtures depend on a pinned environment installing cleanly.

### 8.4 Decisions for the maintainer

Each has a default the work proceeds on.

| Decision | Default | Alternative |
| --- | --- | --- |
| FX-03 sporadic hot spot rule | Follow current ArcGIS by default, with an option that keeps the older rule | Keep the old default and make the ArcGIS rule opt-in |
| FX-05 negative values in G | GPU validity status and an invalid result, since values are not known at construction | Host-side rejection before upload |
| Never-occupied lattice cells in the cube (open question in the review) | Unchanged until the ArcGIS behaviour is confirmed | Exclude them by default |
| FX-08 point-centred speed outliers | Opt-in (`speedOutlierMode`, default `'step'`) | Make it the default at MOB-03 |
| PLAT-03 HDBSCAN `minimumSamples` cap | Stays at 32 | Raise it; the search allows 256, and one rejection spec and one audit case change |
| Kepler diagnostic for global G NaN on negative data | None | Add one |

### 8.5 Outcome (2026-11-10)

Implemented in the working tree, uncommitted. Open rows in section 2.15 state what remains of partial items; the R-only fixtures are in PLAT-06 and the lenses in LN-ST.

Gates after integration: `yarn lint` 0 errors, 64 warnings; `yarn typecheck`, `yarn typecheck-tests`, `yarn build`, `yarn verify-package-exports` (17 entry points), `check-upstream-first` and `check-contributor-contract` pass; node tests 8,497 passed, 375 skipped; headless tests 9,822 run, with the one failure a regression test run before its fix, then the specs affected by the final fixes rerun (1,527 passed, 5 skipped).

Behaviour changes for callers: the FX-03 sporadic rule moves the 12-month showcase from 35 sporadic hot and 189 no pattern to 0 and 224; `GPUSpaceTimeVariogram` requires `gridSize` and has 8 parameters; temporal-reduction word mode allows at most 16,777,216 slots; `GPUTrajectoryRestriction` counts each invalid segment once in `unknown`, plus gap-overlap pairs; `GPUSpaceTimeKriging` is reference tier with at most 1,048,576 samples.

## 9. Plan: how-it-works pages (WEB-HIW)

Status date: 2026-11-10. Scope: `website/src/routes/how-it-works*`, `website/src/components/meta/**`, `website/src/lens/**`. Evidence: a read-through of `modules/analysis/src`, the 410 lens records and the eight existing pages, plus a review of explainer practice (Victor's ladder of abstraction, Hohman et al. on interactive articles, Red Blob Games' figure notes, Diátaxis on explanation pages, The Pudding on scrollytelling, W3C on complex images).

### 9.1 What was wrong

| Finding | Evidence |
| --- | --- |
| No page shows the code a caller writes, and no page shows a workflow of several operations | all eight pages |
| The pipeline says "Table, GPU vectors", but contributors take `GraphDataView`s over imported buffers; `GPUTable` appears nowhere in `modules/analysis/src` | `analysis-core/kernels/gpu-contributor-utils.ts:109` |
| Captions describe aliasing, timings and source text that no record holds: every resource is `imported`, no record has timings or WGSL | `describe-command-graph.ts:191`, all 410 records |
| Three figures repeat on two pages each (hazards, one dispatch, largest buffers) | `topic-pages.tsx` |
| Colour means two things: teal is "built on the CPU" in the pipeline and "compute node" in the graph | `meta.css` |
| The main point of several figures sits behind hover; the dispatch fallback is a grid of empty frames | anatomy, kernels |
| The distinctive parts of the design have no page: parallel primitives, GPU-gated unrolled iteration, the contributor contract, recipes | `contributor-contract.md`, `network-frontier.ts:20` |
| No reading order, no previous or next, no glossary; "409 of 439" is a typed string | `topics.ts:59` |
| The upstream-first example is pseudocode | `topic-pages.tsx:362` |

### 9.2 Rules for every page

1. A concrete run before the abstraction: real code, a recorded graph, the picture it made.
2. Every figure makes its point in its default, prerendered state. Interaction deepens it; hover never carries it.
3. One idea per figure. The caption says what to notice, in at most two sentences.
4. One colour per concept on every page: accent = a kernel (GPU compute), signal = data leaving or needing attention (copy, readback, overflow, a write-after-read hazard), muted outline = CPU-side setup.
5. The reader controls time: steppers and sliders with standard controls, never autoplay that hides content, reduced motion honoured.
6. Explanation only (Diátaxis): why and how, with links to function pages and the reference for instructions and props.
7. Only what is recorded or in source. Numbers are computed from the records at build time. Code examples are files typechecked against the public API.

### 9.3 Pages, in reading order

| Part | Slug | Page | State before |
| --- | --- | --- | --- |
| One run | `anatomy` | The kernel-density call, line by line, linked to its recorded graph and its output | rewrite |
| One run | `data-on-the-gpu` | Buffers and views; points, rasters, polygon offsets, CSR; capacity, count and overflow; mask and validity | rewrite |
| One run | `command-graph` | Read and write sets become hazards and an order, on a real nine-kernel record | rework |
| One run | `scheduling` | Compile once, encode every frame; step a recorded order; buffer lifetimes | rework |
| One run | `kernels` | One shader, a grid of invocations, the index guard | keep, fix |
| Parallel analysis | `building-blocks` | Map, reduce, scan, scatter, sort and the grid index, stepped on small data beside the recorded kernels that do it | new |
| Parallel analysis | `iteration` | Unrolled rounds gated on the GPU: the recorded frontier of a reachability run | new |
| Parallel analysis | `contract` | The contributor contract; the real validators run in the page | new |
| Workflows | `workflows` | Five realistic chains with code, data types between steps and each step's picture | new |
| The system | `lenses` | What a record holds, measured over all records, and how one is made | rework |
| The system | `this-page` | The page measuring itself | keep |
| The system | `upstream-first` | The stack, the rule ladder, real imports | rework |
| The system | `glossary` | Every term, who owns it, where it is explained | new |

### 9.4 Shared work

- View models gain per-node read and write sets (uncollapsed graphs), GPU-gated counts and invocations, the recorded round series, and a `summary.json` over all records. Formulas of network records are filtered to the operation.
- One page kit: shell with part, position and previous or next, figure, code figure with highlighted line ranges, "try it" links.
- One folder per page (`components/meta/pages/<slug>/`) holding the page, its figures, its data and its typechecked examples.

### 9.5 Exit

`npx tsc -p website/tsconfig.json`, `yarn lint`, `yarn website-build` with `check-all-pages.mjs`, and a screenshot pass of all pages in both themes at 1280 and 390 pixels.

### 9.6 Outcome (2026-11-10)

Implemented in the working tree, uncommitted. All 13 pages of 9.3 exist, in one folder each under `website/src/components/meta/pages/`, on the page kit of 9.4. Gates: `npx tsc -p website/tsconfig.json` shows only the existing error in `website/src/gpu/runs/entries/dataframe-pointcloud-records/space-time-variogram.ts`; `biome check` is clean on `website/src/components/meta`, `website/src/routes/how-it-works` and `website/src/lens`; `yarn website-build` prerenders 590 pages; `check-all-pages.mjs --only how` passes 14 of 14; every route was screenshotted in dark at 1,280 px and light at 390 px with no page error and no sideways page scroll. The function-page half of `check-all-pages.mjs` was not rerun; three function pages and the home page were loaded without errors.

What changed outside the page folders: `website/vite.config.ts` reads slugs from `topic-slugs.ts` (a config cannot load the `?raw` examples that `topics.ts` imports); the 410 view models were regenerated with read and write sets, invocations, gated counts, round series and per-edge hazard kinds, plus `website/src/lens/summary.json`; `LensGraphPanel` takes `highlightNodeIds`; `pipeline-diagram.tsx`, `columns-demo.tsx` and `scenes/columns-scene.ts` were deleted with their pages. The committed view models had no per-edge hazard kinds, so the hazard colours on the old scheduling page drew every edge as read-after-write; they are correct now.

Facts the read-through corrected, now stated on the pages: `GPUNetworkReachability` publishes `converged` and `iterationCount`, not `iterationLimitReached`, and needs no gate node because queue pushes size the next round's dispatch; the kernel-density gather sets its workgroup size to 64 itself; network lens records carry all 18 formulas of their pack.

| ID | Item | Current limit | Pri |
| --- | --- | --- | --- |
| WEB-HIW-01 | Timings, WGSL and transient ownership on the pages | The records regenerated on 2026-11-11 hold durations and shader source (dataframe and S2 records are untimed; two terrain scenarios carry no source text). The how-it-works pages do not show them yet. Which transients share one buffer needs luma.gl to publish its allocation plan | M |
| WEB-HIW-02 | Raster snapshots in view models | 20 of 410 view models have a raster map although 322 raster captures exist: `extract-lens-view-model.ts` needs `origin` and `cellSize` | M |
| WEB-HIW-03 | Workflow pictures | Each step shows that operation's own recorded picture, not the chain's intermediate; `GPUPeriodIndex` has no catalog entry; the drainage record ran on a 3 by 3 grid | M |
| WEB-HIW-04 | Block usage counts | `GPUSort`, `GPUReduction` and `GPUGridIndex` kernels carry no operator name in records, so the building-blocks table matches node ids; reductions are probably undercounted (10 records, 30 source files) | L |
| WEB-HIW-05 | Small text on phones | Some SVG labels are 8 to 9.5 px at 390 px; wide figures scroll inside their frame | L |
| WEB-HIW-06 | Index cards of the five new pages | Static thumbnails only; no live scene on hover | L |

## 10. Plan: catalog entries and live runs for the 92 uncovered exports (WEB-COV)

`website/scripts/check-catalog.ts` reported 92 public symbols of `@luma-contributors/analysis` with no
catalog entry (2026-11-10): they are described in `docs/api/**` but have no `/fn/` page, run or card.
Goal: every one gets a catalog entry and a live run whose pointer gesture teaches the concept.

### 10.1 Groups (one new catalog file and one run folder each, one owner each)

| Group (`entries/<group>.ts`, `runs/entries/<group>/`) | Symbols |
| --- | --- |
| `raster-time-series` | BFASTMonitor, COLDMonitor, HANTS, WhittakerSmoother, HarmonicRegression, PhenologyMetrics, RollingMean, TemporalResample, StreamingAccumulator, LaggedCorrelation |
| `raster-composites-climate` | BestAvailablePixel, MedoidComposite, GeometricMedianComposite, ClimateAnomaly, ClimateRunAccumulator, StandardizedIndex, StandardizedIndexCalibration, EOFProjection, SnapshotCovariance |
| `raster-embeddings-flow` | the 11 GPURasterEmbedding* operations, FiniteTimeLyapunovExponent, LagrangianFlowMap |
| `trajectory-mobility` | TrajectoryHampel, OutlierFlags, Split, Thinning, TrackRelativeTimes, Staypoints, Locations, Segments, Recursions, BrownianBridgeUtilization, UtilizationVolumeIsopleths, MobilityIndicators, ModalSplit |
| `space-time-events` | Knox (local, near-repeat), Jacquez, Hawkes, space-time KDE and relative risk, adaptive-bandwidth KDE, space-time K function, Diggle test, expectation-based space-time scan |
| `space-time-models` | space-time local Moran and cross-correlation, space-time kriging, the three Vecchia operations, GTWR, MGTWR, (geo)conformal intervals, area of applicability, ensemble and Gaussian scores |
| `categories-cells-tests` | colocation quotient, local category dependence, spatial entropy, category pair counts, geographical detector, Pettitt and location change points, REDCAP, raster FDR mask, Sinkhorn divergence, contiguity-order correlogram recipe, HEALPix (3) |
| `flows-cubes` | kernel flow smoothing, map-matched edge volumes, radiation flow model, network potential path area, travel-time percentiles, value cube, cell-period keys, period index, cube layout transpose, relative time words |

### 10.2 Rules

- Follow `website/src/gpu/runs/README.md`; reuse the checked-in extracts in `website/src/gpu/data`
  (Dixie NDVI stack, spectral bands, AIS tracks, earthquakes, flashes, commute flows, SF streets). A
  scenario modification (for example spectral bands standing in for learned embeddings) is said in
  the caption.
- Each gesture drives a real prop or input and draws what the hand controls; a time-series run shows
  the pixel series under the pointer next to the map.
- Frozen: shared framework files, other groups, `src/gpu/types.ts`, `schema.ts`, `taxonomy.ts`.
  Framework gaps are reported to the lead. No state-changing git.
- Gates per group: `check-catalog.ts <group>.ts` with 0 errors, website typecheck, `verify-runs.mjs
  --group <group>` with screenshots read. Lead then runs `snapshot-runs.mjs` for new ids,
  `yarn website-build`, `check-all-pages.mjs`, biome, and an Opus review of the whole diff.

### 10.3 Outcome (2026-11-10)

Implemented in the working tree, uncommitted. All 92 exports have a catalog entry and a live run in the
eight new groups of 10.1 (92 runs, 238 files). The catalog covers 531 of 532 exports; the remaining one,
`addRasterResistanceConnectivityRecipe`, was added to source after this pass and is not yet in `dist`.
Process: eight implementors, a curator pass that reviewed each group and set shared conventions (probed
series inset, permutation histogram, timeline strip, grid panel, fixed ranges, theme colors, real-unit
readouts), then an independent Opus review of every group (about 48 defects, verified and fixed by the
owners, including three catalog examples that would throw and several mislabelled statistics).

Known weak points: the Brownian bridge picture is dominated by a moored vessel; trajectory recursions look
sparse because most fixes are visited once; `best-available-pixel` shows few winning scenes; modal split
gives two distinct pointer pictures because its breaks settle only at rest; `gaussian-predictive-scores`
drives two `Data:` sliders because the operation has no scalar props.

| ID | Item | Current limit | Pri |
| --- | --- | --- | --- |
| WEB-COV-01 | Positioned raster and chart panels | The raster layer always fills the canvas, so each group drew tables and side charts with its own quad mesh; the four `_lib` panel helpers should become one shared layer option | M |
| WEB-COV-02 | Per-segment values on `lines` | Ten or more runs read segment values back on the CPU to colour them | M |
| WEB-COV-03 | Shared helper copies | `permutation-chart.ts` and `kernel.ts` are copied into `space-time-events/_lib`; the embeddings group imports `space-time-models/_lib/inset.ts` | L |
| WEB-COV-04 | Text and axis labels in run pictures | No group could label axes; panels rely on captions and readouts | M |

## 11. Plan: parallel completion pass (2026-11-10)

Scope: every item in the "outstanding" list of the 2026-11-10 roadmap review, run as parallel agents with disjoint ownership. Section 10 (WEB-COV) runs at the same time in another session and owns the eight new website groups; this pass keeps away from them and from the shared run framework until that session finishes.

### 11.1 Wave 1 (parallel)

| Stream | Items | Owns |
| --- | --- | --- |
| Commit | UF-01 | Local commits in chunks before any agent starts; no push |
| Bump | UF-11 | An isolated git worktree; produces a patch and a report, applied in wave 2 |
| Kepler | KE-TESTS | `modules/kepler/test/**`, `test/kepler-compute-example-specs.json`, the kepler exclusions in `vitest.config.ts` and `scripts/typecheck-tests.mjs` |
| Records | LN-REC | `modules/analysis/src/computation-record/**`, `modules/lenses/**` |
| Core WGSL | STD-10 (modules and specs; adoption is wave 2) | `modules/analysis/src/analysis-core/**` |
| Trajectories | STD-03, STD-06, STD-07 | `trajectory-*`, `track-similarity` |
| Interpolation and clustering | STD-06, GPU-05 (local kriging, OPTICS) | `spatial-interpolation`, `spatial-clustering` |
| Weights, regression, pair statistics | GPU-05 (MGWR, fuzzy contiguity, island attachment, cross-variogram, marked statistics) | the directories of those classes |
| Geometry output | STD-08, STD-04 for the line family | line, overlay, buffer and hull directories |
| Joins and zonal | STD-04, GPU-04 | `spatial-join` and the zonal, selection and region statistics directories |
| Raster | STD-04, STD-06, STD-07 | `raster-analysis/**` |
| Terrain | STD-04, STD-07 | `gpu-terrain/**` |
| Point cloud | GPU-05 (published grid index), STD-07, STD-03 | `pointcloud-analysis/**`, `geospatial-analysis/**` |
| Network and urban | GPU-05 (snapping, visibility graph, viewpoint classifier), STD-04 | `gpu-network/**`, `urban-morphology` |
| Cells and ecology | FX-17, EC03-R | `cell-topology`, `landscape-ecology`, the EC03 lens pack files |
| Sweep | The bare asserts and registry gaps in directories no other stream owns | those files only |
| Website sliders | WEB-PIC-13, the four weak gestures of WEB-PIC-14 | the named run files |
| Website gestures | The 85 gestures under 3% (WEB-PIC-14) | run files of the twelve existing groups |
| Website text | WEB-INT-16 | New files only; wiring into the shared framework waits for section 10 |
| Website re-score | WEB-PIC-12 | Read-only; a report under `dev-docs/research` |

### 11.2 Wave 2 (serial, after wave 1 and after section 10 finishes)

DUP-02 and the breaking half of STD-09 in one pass with lenses, kepler, the website and the examples; adoption of the STD-10 modules in the solver files; the UF-11 patch; `yarn build`; LN-REC2 (regenerate lens records); a tightened contract baseline; all gates; a whole-diff review.

### 11.3 Rules

- Wave 1 is additive: a new prop is optional, a renamed prop or output keeps the old name as a deprecated alias, and each stream lists the renames it wants for wave 2. The website and section 10 run against `modules/analysis/dist`, which is not rebuilt during wave 1.
- No agent runs `yarn build`, `yarn install`, a full `yarn typecheck`, a full test suite or `yarn lint-fix` over the repository, and none runs `git checkout`, `restore`, `stash`, `reset`, `clean`, `add` or `commit`. One shared `yarn typecheck-watch` serves everyone through `scripts/typecheck-status.mjs`.
- Shared files (barrels, the operator catalog, kernel-audit case files, `lens-coverage.node.spec.ts`) get small in-place edits only.
- Defaults taken for open decisions: KE-TESTS moves the example code the specs need under `modules/kepler/test`; LN-REC adds recording through the shared kernel-node helper and the recorder, with no per-contributor prop unless the design shows it is needed; the library review's recommendations 1 to 3 stand (named-limit validation, a labelled reference tier, break in one pass); packed-only input stays the policy.
- Not done without the maintainer's word: pushing, and sending the PLAT-04 and PLAT-05 proposals to luma.gl.

### 11.4 Exit

`yarn lint`, `yarn typecheck`, `yarn typecheck-tests`, `yarn build`, `yarn verify-package-exports`, `check-upstream-first`, `check-contributor-contract`, `yarn test-node`, `yarn test-headless`, the website typecheck, `verify-runs.mjs`, `yarn website-build` and `check-all-pages.mjs`. Finished rows are deleted from sections 1 and 2 and the outcome is recorded here.

### 11.5 Outcome of wave 1 (2026-11-10)

In the working tree on top of `74dddcf5`, uncommitted. Twenty agents ran wave 1; four independent reviewers then read it and six agents fixed what they confirmed. Reports, reviews and fix reports are in [research/parallel-pass-2026-11-10](./research/parallel-pass-2026-11-10/README.md); rows in sections 1, 2, 7 and 9 state what remains.

What the reviews changed. Fuzzy contiguity could list a pair at the tolerance from one side only (reproduced on the GPU, fixed). Local kriging's grid and scan disagreed on near-ties and `'auto'` could keep an unbounded scan (both fixed). The new snapping cap broke a K function and a catchment recipe that built before (fixed by a documented raised limit). Line merge bound ten storage buffers against a default of eight (fixed; every other class that gained outputs binds at most seven). The eigen solver returned unrotated input for badly scaled matrices (fixed). Two website runs misrepresented their operation with constructed input (reverted; five sliders removed). The text atlas carried the wrong licence (now `OFL-1.1` with the font's notices).

Open from the reviews:

- Walking cost without `parameters` is not bit-identical to the kernel of `74dddcf5` for the Tobler function: 8 of 35 cells differ by one unit in the last place. The spec records it with a fixture taken from the old kernel. Accept it or restore exact output.
- Three classes carry named status under `output.ports.status` (STD-04).
- The font notice for the text atlas is drafted in `web-text.md` and not yet in `THIRD-PARTY-DATA.md`; `katex` is not yet a declared development dependency of the website.

Gates on the combined tree, run at 19:15 to 19:55 while the review fixes and section 12 were already editing sources: `yarn typecheck-tests` 0 errors; `yarn test-node` 917 of 917 files, 9,383 tests passed, 386 skipped; `yarn test-headless` 9,946 passed, 14 failed in 9 files, 806 skipped. The 14 are specs written to fail before a review fix that had not landed (line merge, fuzzy contiguity, stream order) and specs of section 12's fixes in progress. `check-upstream-first` passes. `check-contributor-contract` reports 0 bare asserts and 0 unregistered contributors. Not run: `yarn build`, `yarn verify-package-exports`, the website gates. The gates must run again when sections 10, 11 and 12 are all quiet.

### 11.6 Outcome of wave 2 (2026-11-11)

In the working tree on top of `74dddcf5`, uncommitted. Six agents made the breaking source changes, then the `@luma.gl` pins moved to `10.0.0-alpha.12`, the packages were rebuilt, and agents updated lenses, kepler, the website and the examples. Three independent reviewers read the result and three agents fixed what they confirmed. Reports are in [research/parallel-pass-2026-11-10](./research/parallel-pass-2026-11-10/README.md) (the `w2-*` files and the three wave 2 reviews); the rows in sections 1 and 2 state what remains per item.

Landed: `GPUSpatialKMeans` (DUP-02); `validateSinglePackedView` and the contract names out of the domain barrels (STD-09); the deprecated aliases removed and `parameters` required on walking cost and fractional coverage (STD-05, STD-07); point-cloud rasters default to `'south'` rows; one variable-length geometry layout (STD-08); the shared WGSL modules adopted where results stay identical (STD-10); the padding-row fixes (FX-18); alpha.12 with one new workaround row; 412 lens records regenerated through the kernel recorder; the text layer wired in and 18 runs labelled; the raster resistance recipe run; a snippet import check for the examples, which found 144 imported names in 49 scene snippets that no longer resolve (fixed).

Gates on the combined tree, 01:25 to 02:03, before the review fixes: `yarn lint`, `yarn build`, `yarn verify-package-exports` (17 entry points), `yarn typecheck`, `yarn typecheck-tests`, both example typechecks, the website typecheck, `check-upstream-first` and `check-contributor-contract` (0 bare asserts, 0 `@internal` exports, 0 unregistered contributors) pass. `yarn test-node`: 929 of 930 files; the one failure pinned shader text that a rename changed, and its fingerprints were updated after showing that undoing the rename reproduces the old ones. `yarn test-headless`: 729 files passed and none failed, then the browser connection closed and about 160 files did not run; run it in two halves. `check-catalog` 532 of 532; 1,064 posters rendered; `verify-demos` 10 of 10; `yarn website-build`; `check-all-pages` 547 of 547.

Not gated yet: the wave 2 review fixes (kernels, lens and website texts, example snippets), the spec edits that followed, section 12.6 and section 13. The lens texts fixed after the last capture need a recapture, and posters need regenerating for every run changed since the snapshot. The maintainer asked on 2026-11-11 to continue with the remaining work first and run the full gates afterwards. A staged gate script exists for that run; headless is split in two.

To decide:

- Walking cost: with the required `parameters` view the GRASS function differs from the kernel of `74dddcf5` by one unit in the last place in 8 of 35 cells (Tobler is exact). Accept or restore.
- The `output.ports.status` shape on three classes (STD-04); exporting `validateSinglePackedView` (STD-09); the name `GPUSpatialKMeans`.
- Whether kepler and the examples may keep using the analysis copy of luma.gl's Arrow preparation (KE-TESTS).
- The row-direction flip of isolines, isobands and the flow family (plan in the math review folder).
- The `OFL-1.1` reading for the text atlas has had no legal review.
- Commit chunks and pushing (UF-01); sending the PLAT-04 and PLAT-05 proposals.

### 11.7 Wave 3 (2026-11-11)

What remains of the items of this section that needs no decision. Sections 12.6 and 13 run at the same time in other sessions: this wave keeps out of `spatial-regression`, coverage validity, ring assembly, polygon triangulation and the new `*-operation.ts` files. The rules of 11.3 apply, with breaking changes allowed as in wave 2 (consumers in `modules/` updated in the same change, a work list for the website and examples).

| Stream | Items | Owns |
| --- | --- | --- |
| Pictures, regression and weights | WEB-PIC-12, WEB-INT-16 | `website/src/gpu/runs/entries/regression-weights/**` |
| Pictures, statistics and density | WEB-PIC-12, WEB-INT-16 | `entries/spatial-statistics/**`, `entries/density-interpolation-regions/**` |
| Pictures, geometry construction | WEB-PIC-12, WEB-INT-16 | `entries/geometry-construction-overlay/**` |
| Pictures, geometry processing and trajectories | WEB-PIC-12, WEB-INT-16 | `entries/geometry-processing/**`, `entries/recipes-trajectories/**` |
| Pictures, network, cells and terrain | WEB-PIC-12, WEB-INT-16 | `entries/network-routing-access/**`, `entries/cells-index-join/**`, `entries/terrain-hydrology-flow/**`, `entries/terrain-relief-light/**` |
| Pictures, point cloud and raster | WEB-PIC-12, WEB-INT-16 | `entries/dataframe-pointcloud-records/**`, `entries/raster-bands-indices/**` |
| Padding rows | FX-18: line clip's polygon side, Delaunay, concave hull, the join BVH builders; padding specs for loop removal and offset-curve components | those directories |
| Indexes | GPU-05: the island-attachment search; `GPUNeighborSearch` cross-join queries outside its bounds | `spatial-weights/gpu-island-attachment.ts`, `neighbor-search` |
| Raster, imagery | STD-04, STD-07 for remote sensing, time series, embeddings and change detection | those `raster-analysis` directories |
| Raster, algebra | STD-07 for raster algebra, STD-04 for fill no-data, STD-06 `cellSizeMode` for two fractional aggregators and patch metrics | `raster-algebra`, `raster-patches` and six root files of `raster-analysis` |

Not in this wave: MGWR (GPU-05) and the STD-10 sites in the regression family, which wait for section 12.6; the isoline, isoband and flow families, which wait for the row-direction decision; the 131 kepler specs that need packages the repository does not declare.

Close: an independent review of the website streams and of the library streams, fixes, then `yarn build`, the website and example consumers, lens records, posters, a tightened contract baseline and the gates of 11.4 for sections 11, 12.6 and 13 together.

### 11.8 Outcome of wave 3 (2026-11-11)

In the working tree, uncommitted. Ten agents ran the streams of 11.7; three independent reviewers read the result; three agents fixed what the reviews confirmed. Reports, reviews and fix reports are the `w3-*` files, `review-pictures-a.md`, `review-pictures-b.md` and `review-library-w3.md` in [research/parallel-pass-2026-11-10](./research/parallel-pass-2026-11-10/README.md). The rows of sections 1, 2 and 7 state what remains per item.

Website: 83 runs reworked or labelled. Every stream reported a passing verifier, and the reviews still found seven pictures that misrepresented their operation and about 25 major problems (text unreadable in the light theme, legends that did not match the marks, gestures that produced designs the operation rejects). The seven and nearly all the major ones are fixed; what is still weak is listed in WEB-PIC-12. Two suspected operation defects from the re-score were checked and are not defects (`cell-grid-path` reports the true length of a truncated path; `spatial-weights-traces` returns the traces of the scaled binary weights).

Library: the raster family follows the `mask` rule (about 40 classes renamed with their consumers, results unchanged); `cellSizeMode` on two fractional aggregators and patch metrics; island attachment through the neighbour grid; `GPUNeighborSearch` cross-join queries outside its bounds; the padding-row sites examined. The review found one spec that still passed the old key (its mask was silently dropped), three example sites that pass renamed keys through object spreads where the compiler does not check them, row edges written into the shader against contract section 6, a silent full-scan fallback, and padding specs that could not fail. All fixed or stated in the rows, except the unbounded rescan (GPU-05).

Built and applied after the fixes: `yarn build` and `yarn verify-package-exports` pass with section 13's exports included; the raster renames are applied to 8 website runs, 36 catalog lines and 17 example sites; the website typecheck, both example typechecks, the snippet import check and `check-catalog` (532 of 532, 0 errors) pass; the catalog inventory skips the operation forms of section 13.

Not done here: the full gates of 11.4 on the combined tree (sections 11, 12.6 and 13); lens records and posters must be regenerated once more after section 12.6 ends, for polygon triangulation, ring assembly and the regression family. The decisions of 11.6 are still open, with one more: the name `conditionResult`.

## 12. Plan: fixes from the math correctness review (2026-11-10)

Evidence and finding IDs are in [research/math-correctness-review-2026-11-10](./research/math-correctness-review-2026-11-10/README.md) (166 findings in seventeen domain reports, 13 in the comparison with published luma.gl). This pass fixes what a contributor computes and what its documentation claims. It runs while section 11 wave 2 is open in another session and keeps out of that wave's work.

### 12.1 Waves

| Wave | Work | Exit condition |
| --- | --- | --- |
| 0 | Baseline: `check-upstream-first`, `check-contributor-contract`, shared typecheck at 0 errors, `yarn test-node` (9,368 passed; the 13 failures under load pass on rerun), a snapshot of the tree to diff against | Recorded in the review README |
| 1 | Eighteen streams, one per review report, each owning that report's directories, their specs and their documentation headings. For each finding: reproduce it (a spec that fails on the old behaviour where a GPU run can show it), fix it, and record the outcome in a "Resolution" section at the end of the report | Every finding is `fixed`, `not reproduced`, or `deferred` with a reason; the stream's specs pass; no new typecheck error |
| 2 | Integration by the lead: typecheck, lint on changed files, `check-upstream-first`, `check-contributor-contract`, `yarn test-node`, the affected headless specs, then a review of the whole diff by reviewers that did not write it | Gates recorded in 12.5 |

### 12.2 Rules

- A finding is not fixed on the report's word. The stream re-reads the code and confirms the defect first; a finding that does not reproduce is recorded as such and the code is left alone.
- A fix changes the smallest thing that makes the result right. No exported symbol is renamed, no prop is removed, and a new prop is optional.
- Where published luma.gl has an official implementation of the same operation, the fix converges on it: delegate through the public entry point if that is possible today, otherwise match its convention and note the upstream gap. Nothing is copied from `dist` and no private path is imported.
- A test oracle that is a copy of the kernel's formula does not count as a check. A fixed formula gets a hand-computed value, a committed fixture from the reference library, or an independent closed form.
- Section 11.3's operating rules apply unchanged: no build, install, repository-wide lint-fix, full typecheck or full test run by a stream; no state-changing git command; shared files get small in-place edits; `website/**`, `modules/*/dist`, `modules/lenses` and `modules/kepler` are out of bounds.

### 12.3 Defaults taken for the review's open decisions

| Decision | Default taken | Why |
| --- | --- | --- |
| Contour saddle rule | The bilinear decider of official `GPURasterContours` | luma.gl's implementation |
| Region mask boundaries | Inclusive for rectangle and lasso | Official `GPUPointSpatialQuery` |
| Class intervals | Left-closed stays the default; `GPUClassAssignment` gets an explicit option for the right-closed rule of mapclassify and giddy | Closure depends on the kind of edge: natural-breaks output and d3 scales are lower edges (left-closed), mapclassify bins are upper bounds (right-closed). A default flip would move the smallest value of each natural-breaks class one class low. First default taken was right-closed; reversed during the pass |
| Cost bands | `(a, b]` in reachability and isochrones | A band limit means "within" |
| Undirected self-loop in noding output | Listed once | Official `GPUGraphAlgorithmsTopology`, coarsening and statistics |
| Terrain `rowDirection` | Defaults from the grid affine; an explicit value that contradicts the affine throws | One source of truth without breaking north-up callers |
| `rhumbMidpoint`, trajectory acceleration bit, isochrone `'max'`, LIC step | The code is changed to do what the name and documentation say | The name is the contract |
| Resistance conductance | Kept (divided by ground edge length); the Circuitscape parity claim is removed | The formula is documented and defensible |
| Reference-parity claims that cannot be checked here (R packages, MovingPandas) | The claim is reworded to what the kernel does; the formula is left alone | No way to verify on this machine |

### 12.4 Deferred, with the reason

| Item | Reason |
| --- | --- |
| Rename of the local `GPUKMeans` and `validatePackedView` | Exported names used by lenses, kepler, the website and 376 files; belongs in the breaking pass of section 11.2 |
| Shared solvers, binary search, sphere distance and 64-bit helpers adopted by contributors (contract section 11) | STD-10 adoption, section 11.2 |
| `addToGraph` against `getCommandNodes`; `outputValidity` against `validity`; one no-data policy vocabulary; one iteration-cap name | Library-wide API conventions; a maintainer decision, then the breaking pass |
| Row direction of isolines, isobands and the flow family (they ignore the grid affine) | Following the affine flips every existing picture and caller that compensates; needs one coordinated change with lenses and the website. The current rule is documented in this pass. A caller-by-caller plan was written on 2026-11-11: [row-direction-flip-plan.md](./research/math-correctness-review-2026-11-10/row-direction-flip-plan.md) |
| Variance divisor of raster cell against zonal statistics | Both documented; aligned to the official raster statistics only if they state one |

### 12.5 Outcome (2026-11-10)

Implemented in the working tree, uncommitted. Each report in the review folder ends with a "Resolution" section (one row per finding: what changed, the spec that pins it, the behaviour change), and most have a "Review follow-up" subsection.

- **Findings.** Of the 166, 161 are fixed (some by rewording a claim, as 12.3 says) and 5 are deferred: isoline, isoband and flow-family row direction (12.4); the anisotropic sky-view angle convention (no RVT here); the quantile rank above 2^23 rows (needs a rational probability prop); the Diggle statistic (no R); the COLD monitor fixture (no pyxccd). The streams fixed about 30 further defects they found on the way.
- **Review of the fixes.** Five reviewers that did not write them read the whole diff ([fix-review-statistics](./research/math-correctness-review-2026-11-10/fix-review-statistics.md), [-models](./research/math-correctness-review-2026-11-10/fix-review-models.md), [-geometry](./research/math-correctness-review-2026-11-10/fix-review-geometry.md), [-fields](./research/math-correctness-review-2026-11-10/fix-review-fields.md), [-callers](./research/math-correctness-review-2026-11-10/fix-review-callers.md)): no blocker, five major findings. Four are fixed (restricted regimes with a common intercept and a varying slope; betweenness treating an absorbed positive weight as a zero-weight cycle; the point-horizon classes outside the row-direction rule; a parity spec left on the old lasso rule). The fifth is a lens formula that binds a removed kernel identifier, with the lens work below. The minor findings are fixed or answered in the follow-up subsections.
- **What the pass established about the test GPU's shader compiler** (contract section 11, register section 6, proposal section 13): a float Kahan, Neumaier or two-sum correction is removed; a chain `(value - shift) - small terms` is regrouped; a function-scope array written only by an identity loop can read back as zeros; division is not correctly rounded and a loop sum is not the IEEE serial sum. The cures are `analysis-core/kernels/addition-rounding-error-wgsl.ts`, the fixed-point accumulator, a pairwise sum, an integer-pinned subtraction, and module-scope tables. Every "compensated" sum found in the library was dead on the GPU and is replaced.
- **Gates after integration.** `yarn lint` 0 errors, 72 warnings; shared source typecheck 0 errors; `yarn typecheck-tests` passes; `check-upstream-first` and `check-contributor-contract` pass; node tests 9,420 passed, 6 failed, 386 skipped; headless tests 10,377 passed, 4 failed, 806 skipped (893 files; one earlier run lost its browser connection after 623 files and was rerun). All ten failures are lens specs that compare recorded graphs or formula bindings with kernels this pass changed (`record-structure` for four network contributors in both projects, `s3-geometry-pack`, `statistics-regions-pack`); no analysis spec fails. `yarn build` and `yarn verify-package-exports` were not run: section 11 wave 2 rebuilds.
- **Size.** 291 source files and 186 test files changed against the snapshot, 142 new files under `modules/analysis`.

Left for section 11 wave 2, which was waiting on this pass:

- Callers. [behaviour-changes.md](./research/math-correctness-review-2026-11-10/behaviour-changes.md) lists every change; [fix-review-callers.md](./research/math-correctness-review-2026-11-10/fix-review-callers.md) lists the call sites. No call site in examples, lenses, kepler or the website breaks; 87 draw or assert different numbers on their default path; 181 of 410 lens records belong to a contributor whose code changed; seven texts restate an old rule.
- The six lens specs above, with the record regeneration.
- Everything in 12.4.

Open after this pass (not started on 2026-11-10; closed by the closing pass, see 12.7):

- A headless spec that runs published `GPUHistogram` with a literal domain and with a domain view (proposal section 6).
- The lag-ML, GM and 2SLS residual passes were not tested for the regrouped subtraction that affected the regime and error-ML residuals. `GPUOrdinaryLeastSquares` keeps a float32 response mean that can inflate the residual sum of squares by a few percent when residuals are tiny against the mean (documented and pinned).
- Fixtures regenerated or patched on system Python 3.9 (NumPy 2.0.2, SciPy 1.13.1) instead of the pinned set: group statistics, focal statistics, stretch references, sparse Whittaker, class assignment, fair scores, trajectory checks. Two region fixtures, one noding fixture and the rhumb midpoints were recomputed without the reference library (Shapely, PyGeodesy) and verified by the fix review with exact arithmetic.
  - Closed 2026-11-11: all rerun on the pinned set with the libraries, no value was wrong; see [fixtures-pinned-rerun.md](./research/math-correctness-review-2026-11-10/fixtures-pinned-rerun.md).
- Ring assembly: the float32 innermost-shell choice and the tolerance-match turn key (no failing input built). `polygon-triangulation`: two suspected deviations from Earcut (no failing input built), and the monotone path's combinatorics unread.
- The staggered-gap rule of coverage validity against GEOS, which one Shapely run settles.
- The cause of the identity-table fault inside the compiler, and whether to report it to the browser.

### 12.6 Plan: closing pass (2026-11-11)

Scope: the "Open after this pass" list of 12.5 and the five deferred findings. Section 11 wave 2 (another session) is applying its own review fixes at the same time and owns `modules/lenses`, `modules/kepler`, `website/**`, `examples/**`, `yarn build`, record regeneration and the final gates of 11.4. This pass edits `modules/analysis/src`, `modules/analysis/test` and the documentation of the classes it touches, and hands its behaviour changes to that session through `behaviour-changes.md` and `fix-review-callers.md`.

Reference libraries are now available in a scratch environment on CPython 3.14 with the pinned set of `test/oracles/requirements.txt` (NumPy 2.5.4, SciPy 1.16.3, Shapely 2.2.0, scikit-image 0.26.0, mapclassify 2.11.0, PyGeodesy 26.9.9, rvt-py 2.2.1). R is still not installed.

| Stream | Work | Owns |
| --- | --- | --- |
| Regression | The regrouped-subtraction check for the lag-ML, GM and 2SLS residual passes; the response-mean term of `GPUOrdinaryLeastSquares`; the probit bound case | `spatial-regression` and its specs |
| Coverage and rings | The staggered-gap rule of coverage validity against GEOS; failing inputs for the innermost-shell choice and the turn key of ring assembly | `polygon-coverage-topology`, `analysis-core/ring-assembly` and their specs |
| Triangulation | The two suspected deviations from Earcut; a read of the monotone path | `analysis-core/polygon-triangulation` and its specs |
| Sky view | The anisotropic sky-view angle convention against rvt-py (terrain-surface-10) | `gpu-terrain-horizon.ts`, its spec and documentation |
| Dataframe | An exact quantile rank above 2^23 rows through an optional rational probability (dataframe-08); the Diggle statistic against an independent port of the splancs source (dataframe-10) | `column-classification` order statistics and quantiles, `gpu-space-time-diggle-test.ts`, their specs |
| Raster | The COLD monitor against pyxccd if it builds here (raster-imagery-11); a headless spec of published `GPUHistogram` with a literal domain and a domain view | `raster-change-monitoring` COLD files and specs; one new spec |
| Fixtures | Rerun on the pinned set every generator that 12.5 lists as run on system Python or recomputed by hand; a fixture changes only if the pinned run disagrees | `modules/analysis/test/oracles/**` |
| Compiler | A minimal reproduction of the identity-table fault and the generated shader the browser hands to the driver; a draft report, not sent | New specs and a report only |
| Row direction | Read-only: what following the grid affine in isolines, isobands and the flow family changes, caller by caller (raster-fields-05) | A report only |

Rules: 12.2 and 11.3 apply. A new prop is optional. A stream that changes a number a caller can see appends a row to `behaviour-changes.md`. Each stream appends a "Closing pass" subsection to its report's Resolution section. Exit: every item is fixed, verified as already right, or left with a reason that is not "no reference library"; an independent review of the diff; then the gates, run by the wave 2 session on the combined tree.

### 12.7 Outcome of the closing pass (2026-11-11)

In the working tree, uncommitted. Evidence is in the "Closing pass (2026-11-11)" subsections of the domain reports in [research/math-correctness-review-2026-11-10](./research/math-correctness-review-2026-11-10/README.md); every caller-visible change is in the last section of [behaviour-changes.md](./research/math-correctness-review-2026-11-10/behaviour-changes.md).

How it ran. Nine streams, then independent reviewers that wrote none of the code, then fix rounds until the reviews closed: the fields review once, geometry twice, regression four times (three GPU reviews by Claude agents and one float64 review by a Codex `gpt-6.1-sol` agent, which cannot run a GPU spec inside its sandbox). The reviews found more than the streams had: most of the regression and triangulation defects below were found by a reviewer, not by the stream that owned the file.

| Item of 12.5 or 12.4 | Outcome |
| --- | --- |
| Lag-ML, GM and 2SLS residual passes | All three had the regrouped-subtraction fault (variance 6% to 20% high, Anselin-Kelejian 16.25 for 1.98). Fixed with centred, guarded sums. The reviews then found, and the later rounds fixed: the lag-ML `rho` search differencing three float32 sums (variance up to 2,801 times spreg's with status 0), the float32 lag of the raw response (4.2% in the variance), the same two faults in panel lag ML, a 2SLS solve error (3%), and the panel random-effects ratio clamped at 1e-5. Pinned against spreg 1.9.1 |
| `GPUOrdinaryLeastSquares` response mean | Fixed (residual sum of squares 3.7e-6 from spreg; was 4% to 19.6% high) |
| Probit bound case | Redesigned. Status 0 only at a genuine maximum, status 4 only with a certified separating direction, status 3 otherwise with the iterate. Predictions and marginal effects come from the centred design. Pinned against statsmodels 0.15.0; the last review found no false status in 846 fits |
| Staggered-gap rule of coverage validity | A defect: GEOS flags both edges, the kernel flagged none. Now the GEOS near-parallel rule; it matches Shapely 2.2.0 on 17,524 reviewer cases, with differences only within 1e-4 relative of a GEOS threshold |
| Ring assembly | Both suspected faults were real (wrong shell for a hole on thin wedges; a ring lost at a tolerance match). Fixed with exact tests, and the turn order is now the same for a scene and its mirror image |
| Polygon triangulation | Ten defects fixed in the hole bridge, the diagonal test, touching rings and the monotone path. On 126,382 triangulations per method no result flagged valid is wrong (before: 32 to 106); one polygon is flagged; the `earcut` 3.2.4 package gets 641 wrong on the same set |
| Published `GPUHistogram` with a domain view | The defect is confirmed and pinned without failing the suite; proposal section 6 has the reproduction |
| Fixtures made on system Python or by hand | All twelve reproduce on the pinned set and against the real libraries; eleven now record the pinned versions ([fixtures-pinned-rerun.md](./research/math-correctness-review-2026-11-10/fixtures-pinned-rerun.md)) |
| Identity-table fault | Reduced to the Apple GPU back end: Tint's MSL and the Metal front end are correct, the native compile is wrong with fast math on and off. `var<private>` is not a cure (the first pass said it was); a literal initializer is. No library kernel loses a table today (330 declarations screened). The other three compiler behaviours are Metal fast math. Bug report drafts for Chromium and Apple are written, not filed ([shader-compiler-findings.md](./research/math-correctness-review-2026-11-10/shader-compiler-findings.md)) |
| Sky-view angle convention (terrain-surface-10) | Verified against rvt-py 2.2.1, no kernel change; fixture and spec added |
| Quantile rank above 2^23 rows (dataframe-08) | Fixed by an optional `rationalProbabilities` prop with a 64-bit integer rank; the default path is byte-identical |
| Diggle statistic (dataframe-10) | A defect: the kernel standardised `D`, splancs sums it (103.85 for 269.62), and tied times were weighted differently. Fixed; pinned against a NumPy port of the splancs 2.01-45 source, not a run of R |
| COLD monitor (raster-imagery-11) | One defect fixed (the stability test summed the end residuals; pyxccd 1.1.0 takes the larger). Break dates equal pyxccd on full and half-size breaks (80 of 80) and on 49 of 80 quarter-size breaks; the difference is the refit schedule, documented with what matching it costs |
| Row direction of isolines, isobands and the flow family (raster-fields-05) | Still deferred, now with a plan: [row-direction-flip-plan.md](./research/math-correctness-review-2026-11-10/row-direction-flip-plan.md) (108 call sites, about 57 files, a maintainer decision) |

Also changed: three flow packers reject a zero or non-finite cell size; the `GPUFiniteTimeLyapunovExponent` documentation no longer asks for a reordering it does not need; `AGENTS.md` lists `yarn check-upstream-first` and states the `_internal` rule as the guard script enforces it.

Known limits left, each documented and pinned:

- Lag ML on a near-exact fit can publish a slope one float32 step off, which inflates a variance near 1e-12 of the response scale (33 times on the pinned scene) with status 0.
- Panel random effects: 1.1% in the variance and 0.2 to 0.3 standard errors on the intercept at a variance ratio near 2e-7, from float32 entity means. A ratio below 1e-8 is status 3.
- Probit: a separated dataset that the search does not certify is status 3, not 4 (two-column quasi-separation with ties). A loose `tolerance` (1e-2) is not guarded.
- Triangulation: the diagonal midpoint test is float32 (one polygon in 126,382 flagged that the package tiles).
- `GPUSpaceTimeKFunction` keeps the stpp tie rules, which differ from splancs; the Diggle test alone follows splancs. Not matched to splancs: its weight cap of 100 and one stale term at the largest distance.
- COLD: the refit schedule and the window advance differ from pyxccd.
- Sky view against rvt on steep terrain and at grid edges: sampling differences of up to 0.11 and 0.57 in sky-view factor, measured and not changed.
- Error ML was probed and is right; panel error ML was not read.

Gates run by this pass on the combined tree (other sessions were adding files): see the next paragraph. `yarn build`, record regeneration, the full headless suite and the website gates are run once by the section 11 session on the combined tree.

The combined gate run is recorded in the tables of 13.8 (source last changed at 14:20 on 2026-11-11; the runs finished at 16:10) and in section 15, which lists what was rerun before the commit.

Decisions for the maintainer:

1. The row-direction flip (plan above).
2. Whether to file the two compiler bug reports and send the `GPUHistogram` proposal section to luma.gl.
3. Licence: the Diggle reference is an MIT-headed NumPy port written from a reading of the GPL splancs source; no splancs code is copied.
4. `AGENTS.md`: two proposed rule changes were not applied (a registered, pinned private copy of an official WGSL function as an allowed workaround; a code-style line against float compensated sums).
5. Whether `GPUSpaceTimeKFunction` should offer the splancs tie rules.

## 13. Plan: CPU twins and mixed CPU/GPU graphs (2026-11-11)

Evidence, measurements and the survey are in [research/cpu-suitability-2026-11-11](./research/cpu-suitability-2026-11-11/README.md); the rules are section 14 of the [contributor contract](./contributor-contract.md). Some contributors are better served by the CPU: of 512 surveyed rows, 60 are CPU-preferred (serial in one kernel invocation, capped at a few thousand rows, rounds unrolled into command nodes, or short of float32 precision), 160 would gain from a CPU twin, and 292 should stay GPU-only. Published luma.gl already runs CPU stages beside GPU stages in a `GPUGraph` (`GPUCPUOperation`); the contributors could not use it because a command-node producer has no contract. This pass adds the bridge and the first twins. The GPU contributors stay as they are.

### 13.1 Decisions taken

| Decision | Taken | Why |
| --- | --- | --- |
| Where mixed work is composed | A published luma.gl `GPUGraph`. Contributors stay `GPUCommandNodeProducer`s and are lowered into it by `GPUContributorOperation` and one lowering | It is luma.gl's own mechanism; verified in headless WebGPU on `10.0.0-alpha.12` with no private import |
| Shape of a CPU twin | `CPU<Name>Operation extends GPUCPUOperation` over the same named graph values as `GPU<Name>Operation`, with the computation in an exported `compute<Name>OnCPU` | The pattern of luma.gl's `CPUProjectionOperation`; the function is usable without a graph |
| What is not changed | Existing contributors, recipes (`RecipeBuilder` on a caller's `GPUCommandGraph`), kepler and website hosts | A twin is additive; hosts move to `GPUGraph` only where they want CPU stages |
| When a twin is written | Survey verdict CPU-preferred or twin-useful, plain-vector inputs and outputs | A CPU stage costs a readback and an upload: 0.7 ms at 1,000 rows, 90 ms at one million |
| Choosing CPU or GPU | The caller picks the class. No automatic choice | The crossover depends on the boundary cost, which is an upstream fix (UPS-CPU-1) |

### 13.2 Waves

| Wave | Work | Exit condition |
| --- | --- | --- |
| 0 | Read the published mixed executor; probe a real contributor in a `GPUGraph` beside CPU stages; measure the boundary; survey every contributor; write `analysis-core/operations`, the contract section and the pilot (`GPUVariogramModelFitOperation`, `CPUVariogramModelFitOperation`) | Done 2026-11-11: shared typecheck clean, both checkers clean, four headless tests pass |
| 1 | Seven streams, one per family, 36 contributors: every CPU-preferred row that has a CPU reference and the 19 small twin-useful promotions, less three in directories another session is editing (CPU-02). Each stream writes, per contributor, the operation form, the twin, a node spec against an independent oracle and a headless spec comparing the two | Each contributor is `done` or reported `not done` with the reason; scoped typecheck, lint, node specs and both checkers pass in the stream |
| 2 | Integration by the lead: the headless comparison specs (the streams cannot start a browser), failures sent back to the stream; a review of every twin by reviewers that did not write it; files copied into `modules/analysis/src` only when they compile; barrel lines last | Gates recorded in 13.5 |

### 13.3 Rules

- Streams work in a staging copy outside the checkout. A source file enters `modules/analysis/src` only when it compiles, so the shared typecheck and another session's build never see a half-written file.
- A stream creates files and edits none: no existing contributor, spec, barrel or document. What looks wrong in an existing contributor is reported, not fixed.
- Section 12.2's oracle rule applies: a copy of the same formula is not a check, and a test reference promoted into a twin stops being that twin's oracle.
- Section 11.3's operating rules apply: no build, install, repository-wide lint-fix, full typecheck or full test run by a stream; no state-changing git command.

### 13.4 Open items

| ID | Item | Priority |
| --- | --- | --- |
| CPU-03 | The 75 twin-useful rows not selected in 13.6 (`survey.json`; per-cell raster time-series fits, per-focus local regressions, point-cloud work), only on consumer demand | L |
| CPU-04 | Recipes with a natural CPU stage as `GPUGraph` programs: `addSegregationCounterfactualRecipe`, `addPopulationSupportAlignmentRecipe`, `addBuildingTessellationRecipe`; retire the deprecated `addResistanceConnectivityRecipe` in favour of the float64 solve | M |
| CPU-05 | Record the twin in the operator catalog (a `cpuTwin` field) and show it on the website function page. Until then `website/scripts/extract-inventory.ts` leaves operation forms and twins out of the inventory (files named `gpu-*-operation.ts` or `cpu-*-operation.ts`, and `analysis-core/operations`), so the catalog stays at 532 of 532 | M |
| CPU-06 | Kepler: replace the hand-written CPU fallbacks (`catalog-gpu-snapping.ts` and the `catalog-platform*` bindings) with twins where one exists | M |
| CPU-07 | A `docs/api` page for operation forms and twins, with the variogram example | M |
| CPU-08 | A measured crossover per twin and an optional automatic choice, after UPS-CPU-1 | L |
| UPS-CPU-1 | luma.gl: `GPUCPUExecution` copies row by row on readback and on upload (`readCPUVector`, `getCPUView`), about 11 ns per byte. A packed fast path when `byteStride` equals the row size | H (upstream pull request) |
| UPS-CPU-2 | luma.gl: textures and state vectors in a graph that has a CPU stage. Raster and terrain contributors cannot have an operation form until then | M (upstream) |
| UPS-CPU-3 | luma.gl: consecutive GPU operations between CPU stages run as one submission | L (upstream) |
| UPS-CPU-4 | luma.gl: a published operation that gives a command-node producer a contract, so `GPUContributorOperation` can be deleted (registered in `upstream-workarounds.md`) | M (upstream) |

### 13.5 Outcome (2026-11-11)

Thirty-five contributors have an operation form and a CPU twin: the pilot and 34 of the 36 assigned to wave 1. Two were left undone as too large to finish properly in the pass: `GPUTrajectoryRestriction` and `GPUS2PolygonCovering` (CPU-01).

| Family | Twins |
| --- | --- |
| Dataframe | `GPUVariogramModelFit` (pilot), `GPUInequality` |
| Trajectories and cells | `GPUTrajectoryStaypoints`, `GPUBrownianBridgeUtilization`, `GPUTrajectoryRecursions`, `GPUHealpixPointToCell` |
| Geometry | `GPULabelPoint`, `GPUMaximumInscribedCircle`, `GPUShapeDescriptors`, `GPULocalFrameProjection` |
| Network | `GPURadiationFlowModel`, `GPUNetworkStructure`, `GPUNetworkTimeRouting`, `GPUNetworkTravelTimePercentiles`, `GPUNetworkPathExtraction` |
| Raster | `GPUCostDistancePath`, `GPURasterRegionGraph`, `GPURasterWatershedSegmentation`, `GPURasterFractionalZonalDistribution`, `GPURasterMedoidComposite`, `GPURasterGeometricMedianComposite` |
| Statistics | `GPUMultipleTestingAdjustment`, `GPUContiguityWeights`, `GPUSpatialWeightsValidation`, `GPUPettittTest`, `GPUMapColoring`, `GPUTransitionDiagnostics`, `GPUWardRegions`, `GPUOPTICS` |
| Terrain | `GPUProfilePeaks`, `GPUSolarPosition`, `GPUTerrainWatersheds`, `GPUTerrainStreamOrder`, `GPUTerrainFlowDistance`, `GPUTerrainPeakProminence` |

What changed: 158 new files (79 in `modules/analysis/src`, 79 in `modules/analysis/test`), export lines appended to 31 barrels, contract section 14, this section, and the research folder. No existing contributor, spec or kernel was edited.

Gates, in this checkout on 2026-11-11:

| Gate | Result |
| --- | --- |
| Shared typecheck of `modules/*/src` | 0 errors |
| Scoped typecheck of the 156 new TypeScript files, sources and specs | 0 errors |
| Lint (`biome check`) on the new files and the 31 barrels | clean |
| `check-upstream-first`, `check-contributor-contract` | no violations, no new findings |
| Node specs of the pass (37 files) | 195 tests pass |
| Structure specs that scan the source tree (`test/analysis-core`, 75 files; the lens coverage and record specs, 14 files) | pass |
| Headless specs of the pass (35 files), each running the GPU contributor and the twin on the same inputs | 202 tests pass |
| `yarn build`, `verify-package-exports`, the full `test-node` and `test-headless` | not run by this pass; they run in the combined gate run of the session that owns section 11 |

What the review found. The first run of the comparison specs passed for every twin (157 of 157), and the reviewers, who had not written the code, still found a wrong result in six of the seven streams:

- Staypoints treated longitude -180 and 180, and two spellings of a pole, as different places.
- The radiation model's longitude wrapping cancelled small differences and created false distance ties.
- A float32 no-data sentinel was compared unrounded, so no-data cells stayed in the watershed and zonal distribution twins.
- A wrapped profile-peak index could equal the profile length; a solar azimuth could round to exactly 360; a zero-area ring changed a shape centroid; unscaled stationary weights overflowed.

All were fixed, and each disputed input was added to the comparison spec and run on the GPU. Two reviewer statements about the GPU were wrong on the test device (the sign of a zero, twice); the specs now compare those values numerically and the twins document it. The reviewers also found nine comparisons too weak to catch a wrong twin; those were strengthened.

Limits to know:

- Every twin documents how it differs from its contributor ("What differs from the GPU contributor"). The common ones: float64 arithmetic with one rounding, so decisions at a float32 tie or threshold can differ; no round, candidate or size cap, so the twin finishes where the GPU reports a limit; rows past `count` are zero-filled where the GPU leaves them unwritten.
- The `cellSizeMode` support of the fractional zonal distribution twin was written after its review, when the contributor gained the prop; it is covered by an analytic node test and two comparison cases, not by a second review.
- Thirty things noticed in existing contributors along the way are listed in [existing-contributor-findings.md](./research/cpu-suitability-2026-11-11/existing-contributor-findings.md). Three were measured on the GPU; the rest are leads.
- Implementors were Codex CLI agents (`gpt-6.1-sol`): seven streams and seven reviewers, working in a staging copy; the lead ran every headless spec.
- The website inventory treated the 36 new `GPU*Operation` classes as operations without a catalog entry. Its extraction rule now skips operation forms (CPU-05); `check-catalog` reports 532 of 532 and 0 errors.
- Two follow-ups the same day, after another session's review fixes: the fractional zonal distribution twin now reads its row edges from `parameters` when given, as the contributor does since that fix; and `raster-watershed` got its own `index.ts`, because an entry barrel re-exports subdirectory indexes whole (`analysis-layering.node.spec.ts`).

### 13.6 Plan: second wave, the twins that are better or clearly help (2026-11-11)

After 13.5, 165 surveyed contributors had no twin. The maintainer's direction: build the ones where the CPU is actually better or a twin clearly helps, not all of them. Selected: 90 contributors.

- All 39 remaining CPU-preferred rows (research README, section 4.1), including the three deferred in wave 1 and the two left undone.
- 51 twin-useful rows, chosen by one of these tests: the GPU version approximates the method or stops at a budget (a capped search, unrolled rounds that can run out, a parallel schedule that differs from the sequential definition); float32 precision is a documented problem for it (small dense fits, likelihoods that feed an optimizer, high-resolution cell keys, geodesics); or the natural algorithm is a CPU one (Dijkstra, flood fill, dynamic programming, union-find, a sort).
- Not selected: 75 twin-useful rows whose only case is "small inputs are common" (per-cell raster time-series fits, per-focus local regressions, point-cloud work, summaries that are already cheap), and the 20 recipe rows, which stay under CPU-04.

| Stream | Contributors |
| --- | --- |
| trajectory-serial | `GPUTrajectorySplit`, `GPUTrajectorySegments`, `GPUTrajectoryThinning`, `GPUTrajectoryLocations`, `GPUModalSplit`, `GPUMobilityIndicators` |
| trajectory-large | `GPUTrackSimilarity`, `GPUTrajectoryRestriction`, `GPUTrajectoryKalman` |
| spherical-cells | `GPUPointToCell`, `GPUCellMeasures`, `GPUS2PolygonCovering`, `GPUS2CapCovering`, `GPUSphericalArcOverlay` |
| geometry-skeleton | `GPUConvexStraightSkeleton`, `GPUVectorStraightSkeleton`, `GPUOffsetCurveComponents`, `GPULineLoopRemoval` |
| geometry-normalize | `GPUMinimumBounds`, `GPUGeodesicPairs`, `GPULineSimplification`, `GPUSegmentSnap`, `GPUNormalizePolygons`, `GPUNormalizeGeometry` |
| geometry-triangulation | `GPUPolygonTriangulation`, `GPUDelaunayTessellation`, `GPUConstrainedDelaunayTriangulation`, `GPURatioConcaveHull` |
| network-routing | `GPUNetworkReachability`, `GPUNetworkCostMatrix`, `GPUNetworkCostTable`, `GPUNetworkServiceAreas`, `GPUNetworkPointRouting`, `GPUAlternativeRoutes` |
| network-optimization | `GPUNetworkCentrality`, `GPUNetworkRationalAccess`, `GPUNetworkLouvain`, `GPUNetworkFlow`, `GPUMapMatching` |
| regression-linear | `GPUSpatialLagImpacts`, `GPUOrdinaryLeastSquares`, `GPURestrictedRegimeRegression`, `GPUSeeminglyUnrelatedRegression`, `GPUSpatialWeightsTraces`, `GPUSpatialRegressionDiagnostics` |
| regression-spatial | `GPUSpatialErrorGM`, `GPUSpatialTwoStageLeastSquares`, `GPUSpatialLagMaximumLikelihood`, `GPUSpatialErrorMaximumLikelihood`, `GPUSpatialPanelLagMaximumLikelihood` |
| regression-local | `GPUProbitRegression`, `GPUSpatialInteraction`, `GPUMultiscaleGeographicallyWeightedRegression` |
| validation-sampling | `GPUSpatialBalancedSample`, `GPUContinuousSpatialSample`, `GPUSpatialCrossValidation`, `GPUNestedInterpolationValidation` |
| inference | `GPURankMarkov`, `GPUPointPatternEnvelope`, `GPUMantelTest`, `GPURipleyEnvelope`, `GPUSpaceTimeDiggleTest` |
| likelihoods | `GPUVecchiaLogLikelihood`, `GPUVecchiaSpaceTimeLogLikelihood`, `GPUVecchiaFisherScoring`, `GPUSpaceTimeHawkes`, `GPUConditionalGaussianSimulation` |
| regions-allocation | `GPUSpatialWeightsAlgebra`, `GPULocationAllocation`, `GPUSkaterRegions`, `GPUMaxPRegions`, `GPUAZPRegions` |
| statistics-small | `GPUColumnQuantiles`, `GPUEnsembleScores`, `GPUNeighborhoodSummary`, `GPULocationChangePoints`, `GPUSinkhornDivergence`, `GPUClassBreaks` |
| terrain | `GPUTerrainBranchingFlow`, `GPUTerrainFlow`, `GPUTerrainCatchmentSummary`, `GPUTerrainDepressionHierarchy`, `GPUTerrainBreach` |
| raster | `GPURasterFractionalZonalStatistics`, `GPURasterFillHoles`, `GPURasterReconstruction`, `GPURasterWalkingCost`, `GPUCostDistance`, `GPUStreamlines`, `GPUIsolines` |

Method, changed from wave 1 by what it taught (13.5): each stream is one scripted pipeline in a staging copy: implement; scoped typecheck, lint, node specs, checkers and the headless comparison specs run by the lead's script, with failures returned to the implementor; an independent review by a session that did not write the code; findings returned to the implementor; the same checks again. The brief lists the first wave's mistakes. A stream lands only when it passes, one at a time: files are copied, re-verified in the checkout (removed again if that fails), exports appended, then the layering and export specs and both checkers run. Up to eight Codex `gpt-6.1-sol` agents at once; one headless run at a time.

### 13.7 Native 64-bit arithmetic on the GPU, beside the twins (2026-11-11)

Published `@luma.gl/shaders` (`10.0.0-alpha.12`) has double-single arithmetic in WGSL: `fp64arithmetic` (add, subtract, multiply, divide, square root, compare, and exact subtraction of raw binary64 words) and `fp64` (exp, log, sin, cos, tan). luma.gl's own geospatial and projection kernels use them by assembling the shader with the public `WGSLShaderAssembler`, which turns on an integer-controlled implementation for Apple WebGPU because Metal fast math removes the classic float transforms. This repository does not use them: its kernels are raw WGSL handed to `Kernel`, which does not assemble modules, and `GPUPointToCell` carries a private copy (register row "Double-float WGSL").

Measured on the test GPU (`test/analysis-core/shader-fp64-module.spec.ts`): add 7e-16, multiply 3e-15, divide 3e-15, square root 3e-15, log 5e-13 relative error against float64; a million-term sum of 0.1 is 2e-15 off where float32 is 1e-2 off. The cost on Apple is high: about 4 microseconds per addition in one invocation, several hundred times a float32 addition. It is about 48 bits, not 53, with the float32 exponent range.

What follows from the numbers:

| Kind of contributor | Better tool |
| --- | --- |
| Data-parallel, large, precision needed in a few operations per row: cell keys at high resolution, geodesic distances, coordinates against a far origin, a final difference of large values | Native fp64 on the GPU. The data stays on the GPU and there is no stage boundary |
| Small dense fits, likelihoods that feed an optimizer, sequential or queue-driven algorithms, anything with thousands of 64-bit operations per item | A CPU twin (true float64, ordinary control flow) |
| Long sums | The fixed-point accumulator the library already has |

| ID | Item | Priority |
| --- | --- | --- |
| FP64-01 | Let `createWGSLKernelNode` take shader modules and defines and assemble them with `WGSLShaderAssembler` before it builds the `Kernel`; `@luma.gl/shaders` is already a dependency | H |
| FP64-02 | Replace the private double-float code of `GPUPointToCell` (`h3-index-extended-wgsl.ts`) with the published modules and delete the register row | H |
| FP64-03 | An fp64 mode for the precision-limited data-parallel contributors: `GPUCellMeasures` (unreliable beyond resolution 10), S2 keys at levels 26 to 30 in `GPUPointToCell`, `GPUGeodesicPairs` (near-antipodal fallback), `GPUSphericalPolygonContains` (tolerance band). Their CPU twins from 13.6 serve as the oracle | M |
| FP64-04 | Review the integer rounding-error helpers and "rounded difference" guards of contract section 11 against `fp64arithmetic`: where one published call replaces a local helper, use it | M |
| FP64-05 | Measure the classic double-single path on a non-Apple GPU (it should be far cheaper than the integer path) before deciding how widely to use fp64 | M |

### 13.8 Outcome of the second wave (2026-11-11)

All 90 contributors selected in 13.6 have an operation form and a CPU twin. With the 35 of 13.5 that makes 125 twins. The pipeline ran as 29 streams: the 18 planned ones, and 11 follow-up streams of one to three contributors for the 15 that a larger stream had left undone as too big to finish beside its others. Every one of those was finished once it had a stream to itself.

Gates, in this checkout at 14:30:

| Gate | Result |
| --- | --- |
| Shared typecheck of `modules/*/src` | 0 errors |
| Scoped typecheck of the 561 TypeScript files of both waves | 0 errors |
| Lint (`biome check`) on those files | clean |
| `check-upstream-first`, `check-contributor-contract` | no violations, no new findings |
| Full `yarn test-node` | 1,062 files, 10,407 tests pass, 0 fail |
| Headless specs of both waves (126 files), each running the GPU contributor and the twin on the same inputs | 846 tests pass |
| Website `check-catalog` | 532 of 532, 0 errors |
| `yarn build`, `verify-package-exports`, `typecheck`, `typecheck-tests`, `lint` (run by another session on the same tree, 14:28 to 16:10) | pass; lint has 72 warnings and no errors |
| Headless, all 1,055 spec files, run per directory by that session (`modules/analysis` in three invocations, 985 files and 11,241 tests; `modules/kepler` 52 files; `modules/lenses` 17 files) | 0 failures |
| The website build and website typecheck | not run after the second wave. The only website change of this section is the inventory rule of CPU-05 (`check-catalog`: 532 of 532, 0 errors) |
| `yarn test-headless` as one command | did not finish: the browser connection closed after about 27 minutes and 904 files, in a kepler spec that passes alone. The single-command gate is not reliable on the development machine at this suite size; run it per directory |

What the independent reviews found, before anything landed: 71 findings across the 29 streams. Seventeen were wrong results, in 14 streams; 22 were contract departures (mostly a twin rejecting input its contributor accepts, or the reverse); 16 were undocumented differences; 16 were tests too weak to catch a wrong twin. All were fixed or documented, and each disputed input was run on the GPU. Examples of the wrong results: a basin mean lost to cancellation where the GPU's fixed-point sum keeps it; Quadbin and quadkey rows in the point-to-cell twin; a Poisson change-point cost that lost the optimal partition; a restricted-regime fit that dropped one regime's response variation; a Hawkes fit that could cycle for billions of iterations; streamline points rounded to just outside the grid; AZP and Max-P using unrestricted connectivity where the method defines a two-hop rule.

Changes outside the twins:

- `GPUContributorOperation` has an `emptyOutputPorts` option: outputs must have at least one row except ports the operation names. A stream had worked around the missing option with an adapter class; the adapter is gone.
- `GPULocationAllocationGraphOperation` is named that way because `GPULocationAllocationOperation` is an existing public type (contract section 14 has the rule).
- `raster-analysis/raster-watershed` and `raster-analysis/raster-morphology` are new directories with their own `index.ts`, for twins of contributors that still sit in the package root.
- `spatial-analysis-ontology.node.spec.ts` leaves operation forms out of "public contributors": an operation form is the GPUGraph form of a contributor that has its own catalogue entry.
- The polygon triangulation twin is exported from `gpu-spatial-analysis`, where its contributor is exported, not from `analysis-core`.

Limits to know:

- The polygon triangulation and constrained Delaunay twins always use Earcut; a polygon that only the GPU's monotone method can finish can fail in the twin. Their node specs use the exact tiling checker, since two correct triangulations need not match.
- The lag maximum-likelihood twin publishes the rejected iterate with status 3 when the stationarity check fails; the contributor clears its table and summary there. Whether the contributor should publish it too is open with the session that owns that file.
- Things noticed in existing contributors during this wave are in [existing-contributor-findings-wave-2.md](./research/cpu-suitability-2026-11-11/existing-contributor-findings-wave-2.md): leads, almost all unverified on a GPU.
- `GPUPointToCell`, `GPUCellMeasures` and `GPUGeodesicPairs` have twins, and are also the first candidates for native fp64 on the GPU (13.7, FP64-03); their twins are the oracle for that work.

## 14. Plan: new contributors (2026-11-11)

Source for every row: [research/new-contributors-2026-11-11](./research/new-contributors-2026-11-11/README.md), which has the census, five research reports with sources, three probes on the test GPU and the list of what is not verified. Nothing in this section is implemented. The five directions are options: the maintainer picks which to take and in what order; 14.3 is the order recommended by the session that wrote the research. Effort: S is days, M is one to two weeks, L is more.

### 14.1 What the probes measured

Apple M3 Pro, headless Chromium, plain WGSL through raw WebGPU, float32, each result checked against a CPU reference. Numbers and the probe sources are in the research folder.

- A dense per-point network of a few thousand weights evaluates one million points in about 10 ms (0.55 to 0.61 trillion multiply-accumulates per second). A naive 3 by 3 convolution with 32 channels on 512 by 512 takes 25 ms (0.1 trillion per second).
- The same kind of network trains by hand-written backpropagation and Adam at 4 ms per step on 65,536 samples, with each weight's gradient summed in two fixed-order levels (the `regression-row-tiles.ts` pattern), so no kernel loops over the data. Held-out error on a synthetic surface fell to 10% of its spread in 4 s and 6% in 8 s.
- A forest of 500 trees of depth 8 scores a 1024 by 1024 stack of 16 features in 69 ms.
- At 262,144 samples the training probe kept 281 MB of per-row state in one buffer, over the 128 MiB a default device allows for one binding. Contributors need minibatches or a check against the device limit.
- Not measured: a second machine, real unbalanced forests, a hash-grid encoder, attention, float16, the cost of a luma.gl command graph around the kernels.

### 14.2 The five directions

Models the analyst brings or fits (LRN). No shipped weights.

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| LRN-01 | `GPUForestInference`: predictions (and leaf indices) of an imported random forest or boosted model over rows or a raster stack, with a CPU function that reads the scikit-learn, XGBoost and LightGBM model formats into a node array. Oracle: scikit-learn | M | H |
| LRN-02 | Shared fitting kernels in `analysis-core/kernels`: row-tiled gradient sums, an Adam step with a budget and a convergence status, a seeded minibatch order, coordinate encoders (Fourier, spherical harmonics) | M | H |
| LRN-03 | `GPUNeuralField`: a continuous surface through points or a raster for interpolation at scale, gap filling and compression; Fourier features first, a hash grid with fixed-point scatter-add second; two dimensions and space-time. Documented as not more accurate than kriging and without uncertainty. Oracle: a PyTorch script | L | M |
| LRN-04 | `GPUGaussianMixture`: expectation-maximization with full covariances and BIC. Oracle: scikit-learn | M | M |
| LRN-05 | `GPUSelfOrganizingMap` with a GeoSOM mode | S to M | M |
| LRN-06 | `GPUPermutationImportance`, optionally in spatial blocks, for any contributor with a predict kernel | S | M |
| LRN-07 | `GPUGradientBoostedTrees`: histogram boosting trained in the browser (squared, logistic and quantile loss) | L | L |
| LRN-08 | `GPUTreeShap`: contribution maps per covariate | M | L |
| LRN-09 | An ensemble option on the fitted networks whose spread feeds the existing conformal classes | S | L |

Embeddings, deeper (EMB). Extensions of the eleven `RasterEmbedding*` classes.

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| EMB-01 | Decoders for TESSERA (int8 times a per-pixel scale), Presto (uint16) and float16, and the int8 encoder the docs say is missing. Verify each formula against its source first | S | H |
| EMB-02 | More than 128 dimensions (192, 256, 384; 1,024 where the binding allows) | S | M |
| EMB-03 | `GPURasterEmbeddingTrajectory`: cosine to a base year, year of largest step, cumulative angle across all years | S | M |
| EMB-04 | `GPURasterEmbeddingAnomaly`: Mahalanobis distance from `GPURasterEmbeddingCovariance`, or distance to the nearest label | S | M |
| EMB-05 | A regression output on the linear probe; mean-of-references similarity | S | M |
| EMB-06 | Edge strength between neighbouring embeddings, as input to watershed and sieve (check the overlap with `GPURasterTexture` first) | S | L |

Simulation, uncertainty and factorization (SIM).

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| SIM-01 | `GPUGaussianRandomField`: fields from a variogram model by `GPUFFT2D` (to 1024 by 1024, since the padded grid is capped at 2048), by random spectral modes beyond that and for points, and conditioned by kriged residuals. Lifts the 4,096-row cap of `GPUConditionalGaussianSimulation`. Oracle: GSTools | S to M | H |
| SIM-02 | Terrain error ensemble recipe: probable viewshed, depression probability and flow-path probability from DEM error realizations | M | M |
| SIM-03 | `GPUSensitivityIndices`: Sobol and Morris indices for `GPUWeightedOverlay` and other contributors with per-frame parameters. Oracle: SALib | M | M |
| SIM-04 | `GPURandomizedSVD` as a shared primitive; moves the EOF eigen step off the CPU. Decide first whether it belongs in luma.gl | M | M |
| SIM-05 | `GPUGeographicallyWeightedPCA` on the GWR neighbour grid | M | M |
| SIM-06 | `GPUNonnegativeMatrixFactorization`; unblocks RS10 | S to M | M |
| SIM-07 | `GPUZoneRaking`: iterative proportional fitting across many zones, with integerisation | S to M | L |
| SIM-08 | `GPUMovementStates`: hidden Markov model states for many tracks. The published `GPUSegmentedScan` is a uint32 prefix sum, so the scan over matrix products would be local code | M | L |
| SIM-09 | `GPULocationObfuscation`: donut, Gaussian and planar-Laplace displacement with its effect on density and hot spots | S | L |
| SIM-10 | `GPUStructuredKernelInterpolation`: the kriging mean of 100,000 to 10 million points on a grid by FFT and `GPUJacobiPCG`, with the model from the Vecchia fit; variance later. Oracle: GPyTorch | M | M |

Presentation (CART). Computations that produce data for a map.

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| CART-01 | `GPUFlowCartogram`: contiguous density-equalising cartogram (Gastner, Seguy and More 2018) on `GPUFFT2D`. Reference programs are AGPL or modified MIT: fixtures only | M to L | M |
| CART-02 | `GPUPointLabelPlacement`. No public reference code; a CPU twin checked by invariants is the oracle | M | M |
| CART-03 | `GPURasterContrastEqualization`: CLAHE and histogram matching. Closes R7.1. Oracle: scikit-image | M | H |
| CART-04 | `GPUDisplacementInterpolation`: a raster between two densities at a time fraction, on the Sinkhorn kernel. Oracle: POT | M | M |
| CART-05 | `GPUPointClusterHierarchy`: cluster parents, counts and centroids per zoom, and a blue-noise display sample. The parallel algorithm is a design from the research, not a published one, and needs its own review | M | M |
| CART-06 | `GPUTravelTimeMap`: map vertices rescaled radially by travel time from an origin. Also a design from the research | S | M |
| CART-07 | Small set: illuminated (Tanaka) contours, value-suppressing uncertainty classes and exceedance probability, colour-vision simulation with a palette check, guided filter, parallel-coordinate density rasters, pansharpening (needs a scene with a panchromatic band, WEB-PIC-09) | S each | M |
| CART-08 | `GPUFieldTSNE` linked to the map | L | L |
| CART-09 | `GPURoadStrokes` for network thinning | M | L |

Pretrained encoders (ENC).

| ID | Item | Effort | Pri |
| --- | --- | --- | --- |
| ENC-01 | `GPULocationEncoder`: the SatCLIP location encoder (spherical harmonics, dense layers, a sine; about 1.2 million weights, MIT) as covariates per coordinate. Needs a one-off extraction of the weights and decision 3 of 14.5 | S to M | M |
| ENC-02 | Transformer kernels (layer norm, attention, GELU) and the Galileo-Nano encoder (0.8 million parameters, MIT) | L | L |
| ENC-03 | Presto on sample pixels only (about 89 million operations per pixel); weight licence not stated | M after ENC-02 | L |
| ENC-04 | Spike: ONNX Runtime Web on a device luma.gl created, tensors kept on the GPU, in an example. Decides whether a separate optional package for larger encoders is worth having | S | L |

Not recommended, with the reason in the research folder: GNNWR, geographical random forest, quantile regression forests, UMAP, sparse variational Gaussian processes, many-chain MCMC and redistricting ensembles (until PLAT-04), multiple-point and sequential indicator simulation, promptable segmentation, OmniCloudMask, Dynamic World, OlmoEarth weights, Major TOM embeddings, neural relief shading, tile-grid assignment, necklace maps.

### 14.3 Recommended order

1. Wave 0, foundations: LRN-02, the forest node layout of LRN-01, and the probes rerun on a second, weaker device (pairs with FP64-05). Fix the limits of 14.4 from that.
2. Wave 1, independent and each S or M, one owner each: LRN-01, EMB-01 to EMB-05, SIM-01, CART-03, CART-06, the CART-07 set.
3. Wave 2: LRN-03, ENC-01, SIM-02, SIM-03, CART-01, LRN-04, LRN-05.
4. By demand: the rest. ENC-02 and ENC-03 wait for a consumer; ENC-04 is a spike that can run at any time.

### 14.4 Rules for the work

- Every new contributor follows [contributor-contract.md](./contributor-contract.md) and adds its catalog entry, docs heading and kernel-audit case in the same change; the website inventory is regenerated after each wave.
- Size tiers, at a planning speed of 0.1 trillion multiply-accumulates per second: a network trained or applied per point or per pixel has at most 100,000 parameters and 5 billion multiply-accumulates per run; a pretrained network written as WGSL has at most 2 million parameters (8 MB float32); anything larger is outside the analysis package. Limits are named constants enforced at construction.
- One storage binding stays under 128 MiB unless the contributor checks the device limit at the start of `getCommandNodes`. A run stays in the tens of dispatches.
- A fitted model reports its loss history, `converged` and `iterationLimitReached`, and publishes what it has at the limit. Sums over data use the fixed-order row tiles or the fixed-point accumulator, so a fit repeats bit for bit on one device.
- Weights and imported models are caller-supplied views. No weights or datasets are bundled in a package; anything an example or the website fetches is recorded in `THIRD-PARTY-DATA.md` with its licence.
- No layer library here. A network is a fused kernel named for its analysis purpose; generic layers are proposed to luma.gl.
- Oracles on this machine (`/usr/bin/python3`): scikit-learn, PyTorch, scikit-image and OpenCV are installed; XGBoost, LightGBM, GSTools, POT and SALib are not (PLAT-06).

### 14.5 Decisions for the maintainer

1. Which directions to take, and whether 14.3 is the order.
2. Where generic layers live. Recommended: fused kernels here, shared fitting kernels in `analysis-core`, and a luma.gl proposal for channel convolution, a dense layer with bias and activation, softmax and normalisation.
3. Pretrained weights in examples and on the website: whether SatCLIP's model card (research use, no defence or surveillance use) is acceptable for ENC-01.
4. Where the families live. Recommended: no new source directory; forest, mixture and map models in `dataframe-analysis`, the neural and random fields beside interpolation, the cartogram beside the Dorling cartogram, CLAHE beside raster stretch.
5. ONNX Runtime Web: recommended as a spike in an example only (ENC-04), never a dependency of the analysis package.
6. Whether the website and examples create devices with default limits or the adapter's, which sets the capacity of LRN-03.

## 15. Integration, commit and deployment (2026-11-11)

The work of sections 10 to 14 sat uncommitted in one working tree (about 5,800 changed or new files from several sessions). This pass reviewed what had not been read by anyone but its author, applied the fixes, committed by area, pushed, and published the website.

### 15.1 What was relied on, and what was rerun

No source file under `modules`, `examples`, `website`, `docs`, `scripts` or `test` had changed after the gate runs recorded in 13.8 (checked by modification time), so the long suites were not rerun: `yarn typecheck`, `yarn typecheck-tests`, the full `yarn test-node` (1,062 files, 10,407 tests) and the headless suite per directory (1,055 files) stand as recorded there.

Rerun on the unchanged tree, all passing: `check-upstream-first`, `check-contributor-contract` (no new findings), `yarn lint` (72 warnings, no errors), `yarn verify-package-exports` (17 entry points), both example typechecks, the website typecheck, `check-catalog` (532 of 532), `yarn install --immutable`, and `yarn website-build`. The website gates had not been run after section 13's second wave; `check-all-pages.mjs` now passes on all 532 function pages, the home page and the how-it-works pages, on a build for the GitHub Pages sub-path.

Posters: section 11.8 said they must be regenerated after section 12.6 ended. All 1,064 were rendered again from the final build; 1,062 are byte-identical to the set captured at 10:30, and the two that differ (`computation-metrics`, which pictures run timings) were kept as captured. The posters are current.

After the review fixes of 15.2: `yarn lint-fix` (nothing to fix), `yarn build`, `yarn verify-package-exports` (17 entry points), `yarn typecheck-tests` (0 errors), both checkers, the website typecheck, `check-catalog`, the sub-path website build, and `check-all-pages.mjs` on the home page, the 14 how-it-works pages and five function pages pass. The specs of the files the fixes touched were run one by one (node: the mixed-execution folder, the four routing twins, track similarity, layering; headless: the fractional zonal distribution, variogram and four routing comparison specs). The full node and headless suites were not rerun after the fixes.

### 15.2 Reviews

Two independent reviewers read what the sessions had named as unreviewed: the mixed-execution foundation (`analysis-core/operations`, its test helper, the late `cellSizeMode` support of the fractional zonal distribution twin), and the landing edits of section 13 (about 77 barrels, the rule changes to checks and specs, leftovers, `docs/api` links). Neither found a wrong result or anything that breaks a build, a test or an entry point. The fractional zonal distribution twin agrees with its contributor in 24 configurations run on the GPU (worst difference 3.2e-7 relative).

Fixed from the reviews:

- `GPUContributorOperation` looked port names up through the object prototype (a port named `constructor` was refused; `getView('valueOf')` gave a luma.gl error). Now `Object.hasOwn`, with a test.
- Tests that could not catch a wrong twin: the fractional zonal distribution comparison used one cell and always passed `parameters` (now three rows far from the equator, with and without `parameters`, per `cellSizeMode`); the variogram comparison left out `diagnostics` and allowed 5% on a nugget (now every output at 3e-4, measured at most 1.4e-4; the exponential row's AIC is the logarithm of a rounding-noise residual and is bounded, not compared); the export guard counted 35 twins of 125 and did not check that an operation form is one.
- The four routing twins reported another twin's name in their messages. A forwarding file added and used once was removed. The fractional zonal distribution twin documents that it accepts a determinant the contributor refuses.
- Ten broken relative links in `docs/api` (one `../` too many), a usage snippet importing a luma.gl type from this package, and catalog text naming a p-center objective that does not exist.
- The how-it-works page data was regenerated (`generate-data.ts` of five pages): the upstream-first page showed `10.0.0-alpha.10` and the contract page 118 bare asserts where the baseline now has none.

Also recorded: the read-only comparison of the luma.gl snapshot with `edf9ca10` made by another session on 2026-11-08, in [research/split-review-2026-11-08](./research/split-review-2026-11-08/README.md).

### 15.3 Deployment

The website is served by GitHub Pages at `https://bertch.github.io/luma-contributors/` from the `gh-pages` branch. `yarn website-deploy` (`website/scripts/deploy-pages.mjs`) builds with `SITE_BASE_PATH=/luma-contributors/` and force-pushes the build as one commit; `website/README.md` has the steps. The site is built on the development machine, not in CI: it reads `modules/*/dist`, the page checks need a real GPU, and no GitHub Actions run has ever been recorded for this repository, so the CI build is untested. The site is public while the repository is private.

### 15.4 Open items from this pass

| ID | Item | Priority |
| --- | --- | --- |
| INT-01 | Lens records were captured at about 10:20 on 2026-11-11, before the last fixes to lag maximum likelihood (12:49) and ring assembly (11:39). The record structure specs passed on the final sources; captured values of lenses in those families may be stale. Recapture with `VITE_CAPTURE_LENS_RECORDS=1` and the steps of `research/parallel-pass-2026-11-10/w2-lenses-records.md` | M |
| INT-02 | `GPUContributorOperation`: a zero length that comes from a prop is reported against the internal port name, not the prop and its limit; output formats and input value names are not validated at construction (they fail later inside luma.gl) | M |
| INT-03 | `website/scripts/extract-inventory.ts` leaves operation forms out of the inventory by file name. Decide by class (extends `GPUContributorOperation` or the published `GPUCPUOperation`), as the ontology spec does. Part of CPU-05 | M |
| INT-04 | KE-TESTS: the exclusion reasons in `test/kepler-compute-example-specs.json` and six skip comments cite `10.0.0-alpha.10`. Nine specs are excluded for `stats.physicalTransientResourceBytes`, which alpha.12 publishes (`compilationStats` is still missing): classify them again | M |
| INT-05 | `analysis-core/kernels/row-access-wgsl.ts` and `sphere-distance-wgsl.ts` are used by one spec and no contributor. Use them or remove them | L |
| INT-06 | `modules/lenses/.../local-permutation-test.lens.ts` imports an analysis subpath that is not in the package `exports` map; it resolves only through the tsconfig paths | L |
| INT-07 | The open findings of the split review (research folder above): the gpu-graph lens pack and four showcase files dropped without a record, five kepler specs in no exclusion list, lost tests, the lower EC03 limits and weaker validation not in the register, four errors in `upstream-workarounds.md`. Checked against `edf9ca10` only; re-check each before acting | M |
| INT-08 | CI: `.github/workflows/test.yml` has never produced a run. Find out why before relying on it, and decide whether the Pages build should move into a workflow | M |
| INT-09 | The website's GitHub link (`site-header.tsx`) points at the private repository, which a visitor cannot open | L |
