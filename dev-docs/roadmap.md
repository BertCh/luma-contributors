# Roadmap

Status date: 2026-11-10. Baseline: `@luma-contributors/analysis`, `/lenses` and `/kepler` on published `@luma.gl/*@10.0.0-alpha.10`, HEAD `edf9ca10` (pushed to `origin` on `master` and `upstream-first`) plus local commits and working-tree changes from the parallel pass of [section 11](#11-plan-parallel-completion-pass-2026-11-10). npm's `beta` tag for `@luma.gl/*` is `10.0.0-alpha.12` (UF-11).

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

- Every open item has an ID. Reuse the ID in commits, issues and test names. IDs from the old documents (ST01, RS02, TE06, CR01, KG06 and so on) are kept. New IDs use the prefixes `UF` (upstream-first completion), `AUD`, `NW`, `LN`, `KE`, `SH`, `WEB`, and, from the 2026-11-09 spatiotemporal review, `FX` (defects), `TC` (time contract), `STX`, `STM`, `MOB`, `RTS`, `PLAT`, and, from the 2026-11-09 library review, `STD` (contract and standards), `GPU` (rework to a GPU-scale design), `DUP` (merges), `UPS` (moves to luma.gl), `GAP` (missing capabilities).
- Priority: **H** = clear scope and real analysis or correctness value, do next. **M** = useful, needs a consumer or a design step. **L** = research, belongs in luma.gl, or an explicit non-goal.
- "Current limit" cites `file:line` in this repository as of the status date. Source files are still moving, so re-grep before you rely on a line number.
- "Source" points at the archived document (path relative to this directory) or, for documents that were not archived, the luma.gl branch path.
- When you finish an item, delete its row and, if it changes coverage, update [coverage-matrix.md](./coverage-matrix.md). When you accept a new regression or add a workaround, add it to [upstream-workarounds.md](./upstream-workarounds.md) in the same change.
- Review this file at each `@luma.gl` alpha bump (see the upgrade procedure in the workarounds file) and at each release.
- Nothing in the coverage inventory is "Planned". Every row is Implemented, Partial or Excluded. This roadmap lists what is left inside and around those rows.

## High-priority summary

| ID | Item | Why |
| --- | --- | --- |
| UF-11 | Move from `@luma.gl/*@10.0.0-alpha.10` to `alpha.12` | npm's `beta` tag moved; the upgrade procedure has not been run, and it should run before the breaking half of STD-09 |
| EC03-R | Raster resistance adapter and lens scenario for `addSparseResistanceConnectivityRecipe` | Sparse recipe is implemented; the raster conductance adapter (`addRasterResistanceConnectivityRecipe`) and the lens pack operator and scenario are pending |
| MOB-03, DUP-02 | The FX-08 point-centred default is undecided; the local `GPUKMeans` collides with the published class of the same name | Both are small and both change what callers see |
| NW-CENT | `GPUNetworkCentrality` still unrolls its rounds at compile time | The cliff is now a constructor error above 32,768 command nodes; large graphs need sampled pivots until a GPU-gated continuation exists upstream |
| LN-REC | No analysis contributor accepts a `GPUComputationRecorder` | Lens records lose source text, bindings, formulas and provenance |
| KE-TESTS | 271 of 367 kepler spec files are excluded from every run | Most kepler code is untested |
| STD-04, STD-06 to STD-10 | The contributor contract is written down and checked in CI, and about half of the alignment has landed | Remaining: 89 bare asserts, context and no-data on the rest of the library, the shared WGSL modules, the breaking half of the `analysis-core` surface; section 2.16 |
| GPU-05, FX-17 | Brute force is still the only path in ten contributors; one S2 angle defect is half fixed | The cheap reworks of section 2.16 landed on 2026-11-09; the larger ones are ranked M there |

## 1. Now: finish upstream-first

Plan: [upstream-first-plan.md](./upstream-first-plan.md). Phases 1 to 6 are done. This block is what is left of the baseline before new feature work. Done and deleted on 2026-11-07: UF-02 (CI gates), UF-04 (test and example typechecks), UF-05 (stale subpath imports), UF-07 (`graph-support` is reached through the `./analysis-core` export), UF-08 (dead test globs).

| ID | Item | Current state | Pri |
| --- | --- | --- | --- |
| UF-01 | Commit the working tree in coherent chunks | Remote `origin` is `https://github.com/BertCh/luma-contributors` (private); HEAD `edf9ca10` is pushed on `master` and `upstream-first`, and it includes the 2026-11-07 gate, audit, centrality and EC03 work (`794e2f4a`). The work of 2026-11-09 and 2026-11-10 (sections 2.16, 7, 8 and 9) is committed locally in chunks on 2026-11-10 as the first step of section 11; pushing is the maintainer's decision. The `repository` URL in `modules/*/package.json` is `https://github.com/visgl/luma-contributors`, not the remote | H |
| UF-03 | Re-measure performance after the split | The branch measurements used forked primitives. Run `yarn test-benchmarks` and record numbers next to the accepted regressions in [upstream-workarounds.md](./upstream-workarounds.md) section 3. The bounded-dispatch row there is no longer a regression (PLAT-01). Scale factors for the join and aggregation benchmarks: PLAT-07 | M |
| UF-06 | Docs baseline | Done: the `gpu-core` mentions left in `docs/` are public `@luma.gl/graph/gpu-core` imports or luma.gl documentation links; no removed-feature docs remain; `AGENTS.md` no longer mentions `_pending`; the doc-snippet and doc-parameter specs run and pass. Open: `docs/api/computation-views.md:32` gives `gpu-core.scan` as an example scenario ID, but the `gpu-core` lens scenarios were dropped | L |
| UF-09 | SPDX headers | Three tracked example files have no header: `examples/deck/spatial-analysis-showcase/vite.config.ts`, `examples/deck/spatial-analysis-showcase/vite-env.d.ts`, `examples/deck/spatial-analysis-explorer/vite-env.d.ts`. `website/src/routeTree.gen.ts` is generated by TanStack Router and is left as is | L |
| UF-11 | `@luma.gl` alpha.12 | `package.json` pins `~10.0.0-alpha.10`; npm's `beta` tag is `10.0.0-alpha.12` (seen 2026-11-10). Run the upgrade procedure in [upstream-workarounds.md](./upstream-workarounds.md) section 4, and check whether the release exports `GPUJacobiPCG` or a GPU-gated continuation (NW-CENT, PLAT-04) | H |
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
| PP04 | Indexed large-cloud OPTICS | Brute force, at most 4,096 rows (`gpu-optics.ts:103`) | M | parallel record (PP04) |
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
| AUD-02 | Rejection cases for point-cloud capacity limits | Point-cloud contributors check maxima with bare `assert` (for example `gpu-pointcloud-features.ts:33`, `gpu-pointcloud-neighborhoods.ts:52`, `gpu-pointcloud-icp.ts:65`, `gpu-pointcloud-change.ts:124`), so the kernel audit cannot pin a rejection message. Convert them to `validateCapacity` / `validateMaximumCount` (`analysis-core/kernels/gpu-contributor-utils.ts:225,247`) and add rejection cases. The 12 contributors already have 23 build cases (`kernel-audit/cases-pointcloud-and-datum.ts`) | L | AUD-01 review, 2026-11-07 |
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
| EC03-R | Raster resistance adapter | `addSparseResistanceConnectivityRecipe` is implemented (at most 1,048,576 vertices, 2,097,152 edges). Pending: `addRasterResistanceConnectivityRecipe` (resistance raster to edges with conductance `2 / (R(a) + R(b)) / edgeLength`) and the lens pack operator, formula and scenario (pending entry in `lens-coverage.node.spec.ts`) | H | `archive/geospatial-coverage-parallel-implementation.md` (EC03) |
| GE01 / GE08 / UR01-UR05 | Capacity limits: nesting above 64; straight skeleton above 1,024 vertices; urban recipe capacities | Bounded by contract | L | parallel record |
| EC02 | High patch-count behaviour is unmeasured | Not measured | L | parallel record (EC02) |
| EC04 | Other segregation statistics and nulls | Not implemented | L | parallel record (EC04) |
| GEOM-01 | Rectangle clip outputs for polygons | `gpu-rectangle-clip.ts:199,205` | L | code stub |
| GEOM-02 | Line smoothing source rows and measures | `gpu-line-smooth.ts:164` | L | code stub |

### 2.11 Kepler

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| KE-TESTS | Run the kepler tests | `test/kepler-compute-example-specs.json` lists 271 of the 368 spec files in `modules/kepler/test`; `vitest.config.ts:10-19` excludes them all from test runs and `scripts/typecheck-tests.mjs:22` excludes them from `yarn typecheck-tests`, because they need the parked kepler-compute example. Decide: move the needed example code under `modules/kepler/test`, or keep the example and make it a workspace | H | `archive/kepler-compute-integration-plan.md` |
| KE-EX | Fate of `examples/deck/kepler-compute` (about 750 files) | Parked: not in `workspaces`; imports `@deck.gl-community/gpu-layers` and `@deck.gl/core`, which are not dependencies | M | `archive/kepler-compute-integration-implementation.md` |
| KE-UI | Remaining host work: native Group By panel binding, non-point altitude layers, richer labels, micro and nanosecond time admission, multipart timed trips, acceptance on other hardware tiers | Open (mostly example code). Acceptance was measured on one Apple M-class machine | M | `archive/kepler-compute-integration-implementation.md` (Remaining implementation) |
| KE-GUARD | About 14 "Unsupported ..." validation guards | By design, not stubs. Examples: `kepler-group-by.ts:111`, `style-plan.ts:104,147`, `analysis-recipe.ts:357,1367`, `prepare-spatial-selection.ts:51,77` | L | code |
| KE-DECK | Layers that depended on deck GPU layers removed on the branch (cluster, flow, globe) | External; track with the deck.gl-community package | L | luma.gl branch `dev-docs/reviews/comprehensive-gpu-review-2026-11-07.md` |

### 2.12 Lenses and computation visualization

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| LN-REC | Wire contributors to the recorder | The dead `record:` metadata is gone: `createWGSLKernelNode` has no `record` option and `WGSLKernelRecordMetadata` no longer exists (no `record: {` sites in `modules/analysis/src`). `GPUComputationRecorder` (`computation-record/computation-recorder.ts`) is used by the `computation-record` specs and three lens specs (`modules/lenses/test/computation-lenses/gpu-network/{structure,construction,search}-scenarios.node.spec.ts`); no analysis contributor accepts a `recorder` prop. Lens records therefore carry structure and timings but no source text, bindings, formulas, loop provenance or `derivedFrom`. Open design decision: add an optional `recorder?` prop per contributor family (pattern in [upstream-first-recording.md](./upstream-first-recording.md)) or accept structure-only records | H | `archive/computation-visualization-roadmap.md` (V0, V3) |
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
| WEB-INT-16 | Text in pictures | Layers draw no text: no axis, bar, legend or matrix labels | M | section 7.5 |
| WEB-INT-13, 15, 17, 18 | Handle after a restart, pointer-to-frame timing, highlight as a layer feature, `record-demo.ts` | See section 7.5 | L | section 7.5 |
| WEB-PIC-09 | Data that fits | No open water or post-fire scene in the imagery extract; a nearly bimodal lidar tile | M | section 7.6 |
| WEB-PIC-12 | Independent re-score of all pictures | Each group reviewed only its own contact sheets | M | section 7.6 |
| WEB-PIC-13, 14 | 11 inert sliders; 4 gestures with fewer than three distinct pictures; 85 gestures that change under 3% of pixels | Listed by run in section 7.6 | M | `website/.reports/final` |
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
| PLAT-05 | luma.gl or host: request adapter limits; subgroups with a fallback; `buffer_view` | Analysis half done. The luma.gl half is a proposal only: `upstream-proposal-adapter-limits-subgroups.md`, no PR. | L |
| PLAT-06 | Partly done 2026-11-09: `yarn oracle-fixture <generator>` (`scripts/run-oracle-fixture.mjs`) regenerates a family in a sandbox and diffs it, writing only with `--write`, and reports a missing interpreter, Python package or R with exit 2; the dead `modules/graph` paths in four generators are fixed. Open: a new external fixture for the trajectory, track-similarity and Markov families (they already have giddy, shapely, similaritymeasures, tslearn, dtaidistance and MovingPandas fixtures; the only installed environment holds numpy, scipy and gstools, so a new one would be a binary64 reference only); MobilityDuck | The runner exists. R-only fixtures pending: gstat, sparr, stpp, stopp, changepoint, scanstatistics, ETAS, GpGp, CAST, phenofit/GRASS, bfast. Also pending: pyxccd, xeofs and scikit-learn for RTS-03, and the authors' GeoConformal code. | M |
| PLAT-07 | Partly done 2026-11-09: `spatialbench-scale-benchmark.spec.ts` (opt-in, in `yarn test-benchmarks`, `VITE_SPATIALBENCH_SF`) runs a point-in-zone join with per-zone counts at trips = 6M x SF against 156,095 synthetic zones. Apple headless WebGPU, median per encode: SF0.1 7.2 ms, SF1 42.6 ms, SF2 81.3 ms, SF5 216.3 ms; SF10 (60M trips) crashes the page because the runner builds JS point arrays. SF100 and above (600M trips, 9.6 GB of positions) do not fit a default device or one buffer. Open: real Overture zone geometry, the aggregation-only and KNN queries, a typed-array runner for SF10 | Measured to SF5 only, on synthetic zones and one query shape. | L |

### 2.16 Library convergence: one contract, GPU-scale designs (2026-11-09)

Source for every row: [research/library-review-2026-11-09](./research/library-review-2026-11-09/README.md). The contract is [contributor-contract.md](./contributor-contract.md). A first implementation pass ran on 2026-11-09; rows that say "Done 2026-11-09" list what landed and keep only what remains, and finished rows (STD-01, GPU-02, GPU-03, GPU-06) are deleted. Nine reviewers gave a verdict to 429 contributors or sibling groups: 182 conform, 133 need contract alignment only, 64 need the algorithm reworked, 30 duplicate another contributor, 20 belong upstream. The per-contributor tables, with `file:line`, are in the `code-*.md` reports named in each row. Five decisions the rows depend on are listed in the review README ("Decisions for the maintainer"). Effort: S is days, M is one to two weeks, L is more.

Contract and standards (STD). No behaviour change intended; the existing tests are the safety net.

| ID | Item | Current state | Effort | Pri |
| --- | --- | --- | --- | --- |
| STD-02 | Contract checker: turn the remaining rules from baseline to failing as each count reaches zero | `yarn check-contributor-contract` runs in CI and fails on departures not in `scripts/contributor-contract-baseline.json`. Counted 2026-11-10: 89 bare `assert` calls (the baseline still allows 118, so it can be tightened), 46 single-invocation loops (heuristic; reference-tier kernels need a `contract-allow` comment), 13 `@internal` exports, 21 multi-class files, 21 root files. The loop rule cannot see a kernel body passed as a function parameter | S | M |
| STD-03 | Named-limit validation in place of the remaining bare `assert` calls | Done 2026-11-09 for point cloud, remote sensing, focal pairs, terrain-processing, network-optimization, urban morphology, landscape ecology, point process, global coordinates, cell topology, recipes, the loose raster files and `GPUScalarScale` (593 calls down to 122), and the spatiotemporal work of 2026-11-10 brought it to 89 in 32 files. Remaining: `yarn check-contributor-contract --all` lists them; no directory has more than four files | S | M |
| STD-04 | Named status ports everywhere; one bit vocabulary for per-row flags | Done 2026-11-09: `candidateOverflow` on line clip, minimum clearance and the three joins (with per-row `candidateTruncated`), `approximation` on segmentize and great-circle arcs, `requiredCount` on linear referencing, named and documented status bits in point cloud, the spherical family and focal pairs, `status.count` meaning documented across the line family. Remaining: zonal statistics, buffer selection and line-length-per-polygon do not pass the join flags through; a BVH overflow in the joins still marks no rows; region statistics packs status in header words; per-row flags have per-family bit meanings; terrain, raster and most statistics classes use ad hoc status | M | H |
| STD-05 | Numeric per-frame props read from a parameters view, not written into WGSL | Baked: the covariance model in `GPULocalKriging`, point-cloud thresholds and viewpoint, about 20 statistics props, the raster affine. Three parameter mechanisms coexist | M | M |
| STD-06 | Declared context on every contributor | Done 2026-11-09: optional `spatialContext` on the five spherical classes, both S2 coverings and four point-process and validation classes; optional `cellSizeMode` on six terrain-processing classes (the geographic modes bake the georeference into the shader). Remaining: local kriging, local RBF, natural neighbour, conditional simulation, Kalman, restriction and similarity search take no `spatialContext`; georeferencing reaches raster contributors six ways; the walking-cost affine is still a shader literal | M | H |
| STD-07 | One no-data rule: input `mask`, output `validity`, rasters through the published band validity | Done 2026-11-09: optional `validity` input on six terrain-processing classes; `GPUScalarScale` takes `mask` (`validity` deprecated). Remaining: about 60 other mask and validity names; four raster no-data conventions; point-cloud rasters flip the row direction | M | H |
| STD-08 | One layout for variable-length geometry output and one provenance rule | Seven layouts in the line family; `status.count` means five different things; overlay and buffer output carry no source rows (`gpu-polygon-overlay.ts:1266`) | M | H |
| STD-09 | `analysis-core` becomes the only home of the contract | Done 2026-11-09 (additive): `analysis-core` exports the contracts, validators and kernel helpers, and they are no longer tagged `@internal`. Remaining (breaking): remove the copies re-exported from `gpu-spatial-analysis`, `gpu-network` and `gpu-terrain`; narrow the `export *` barrels that leak 13 `@internal` builders; rename the core `validatePackedView`, which shadows a published function with different behaviour; fold `gpu-spatial-analysis/shared` and the raster iteration gate (`cost-distance/raster-relaxation.ts`, used by line simplification) into core; move `regression-row-tiles.ts` and `line-merge/list-ranking.ts` to core when a second family needs them | M | H |
| STD-10 | Shared WGSL modules in core: dense solve, float tests, ordered keys, binary search, 64-bit words, geodesic distance, mask test, row accessors | `getWGSLKernelBinding` added 2026-11-09; nothing adopted yet. 17 dense solvers (closes FX-09 and FX-10; the solver files were being edited for those items, so this waited); seven names for two float predicates, inlined in 43 more files; four wrappers and about 100 local lambdas around `createWGSLKernelNode` | M | H |
| STD-11 | One rule for recipes | Done 2026-11-09: `RecipeBuilder.addGlue` registers raw glue kernels as named stages (11 recipe files), every recipe reports per-stage status, the 12 urban and landscape recipes accept caller-owned `output` views, and five recipes' status counts are reductions. Remaining: those 12 are still kernel-only functions, not classes; `recipe-utils.ts` and `expansion-recipe-utils.ts` are two helper modules; `GPUQuantileLISA` is a class in `recipes/`; `interpolation-diagnostics` and `network-scenarios-recipe.ts:383` still summarize in one invocation | M | M |
| STD-12 | Layout: one class per file, package roots hold only `index.ts`, test directories mirror source, docs grouped by domain | 16 loose root files; 10 multi-class files; test directories named for the luma.gl branch; docs split into `operators-*` and `*-expansion` pages; `geospatial-analysis` is not a coherent entry | M | L |
| STD-13 | One registry | Done 2026-11-09: the operator catalog covers every exported contributor of all nine entry points (`getGPUAnalysisOperatorCatalog()`; 175 added entries in 40 capability families under `ontology/all-entry-points/`), with `tier` (`scale` or `reference`) and confirmed `limits`; a test fails when a contributor has no entry. Remaining: rename the 14 wave-named capability IDs (`expansion-*`, `coverage-*`; the website generator and kepler enumerate them); check the coverage matrix, docs index and website catalog against the registry; `kepler-analysis-catalog.ts` still states the old Dorling limits; regroup `docs/api` by domain | M | M |
| STD-14 | State packed-only input as policy and guard the binding size everywhere (generalizes PLAT-02), or plan chunked input for joins, aggregation and density | 294 class files require one packed view; 11 accept chunks; only some check the binding size | S | M |

Rework to a GPU-scale design (GPU). Cheapest first.

| ID | Item | Where (report) | Effort | Pri |
| --- | --- | --- | --- | --- |
| GPU-01 | Replace the remaining single-invocation loops over data | Done 2026-11-09 for the raster convergence scans, registration peaks, fractional-coverage compaction, stream network, coregistration sums, SMRF and statistical-filter counts, five recipes, urban and landscape totals, the border K, the balanced-sample rank and the network K radius finish. Remaining (each has a comment stating the size at which it hurts): masked registration running means and refinement scoring, zonal statistics over fractional tuples, the adjacency union-find, structures pack, patch reconcile, counterfactual quantile mapping, the Ang shell scan, `spatial-validation-design.ts` fold designs, `GPUInequality` | raster, recipes, network reports | M | M |
| GPU-04 | Join overflow, remaining part | Done 2026-11-09: `candidateOverflow`, per-row `candidateTruncated`, and an anti-join that withholds truncated rows. Remaining: a BVH build overflow still drops leaves without marking rows, so the anti-join can list matched rows in that case; the nearest-join flags work in planar nearest-feature mode only | `spatial-join-passes.ts` (cells report) | S | M |
| GPU-05 | Index by default (widens PLAT-03) | Done 2026-11-09: M3C2 takes an optional grid (with an incomplete-grid status bit), unindexed point-cloud work above 2^27 pairs is rejected, isovists and street profiles switch to the segment BVH above 4,096 segments. Remaining: the point-cloud private grid index is not the published `GPUGridIndex` (five callers); visibility graph and viewpoint classifier are brute force; MGWR, local kriging, OPTICS, fuzzy contiguity, island attachment, network snapping, cross-variogram and marked statistics are brute force | terrain and point-cloud, recipes, statistics, network, core reports | L | H |
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
| DUP-02 | Rename or upstream the local `GPUKMeans`, which collides with `@luma.gl/graph/gpu-vector-search` | S | H |
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
| FX-17 | `s2AnalysisAngle` returned -π/2 for `atan2(y, 0)` on the test GPU, so a zero-radius cap on a face axis covered four spurious cells. Fixed in the cap path; `gpu-s2-polygon-covering.ts` calls `atan2(numerator, denominator)` the same way and has not been checked | S | H |
| LN-REC2 | Lens records (`docs/lens-records`, including `twins/*~graph.json`) and the website lens JSON are stale for the reworked classes: the four iterative raster classes, fractional coverage, terrain-processing, probit and the regime regressions, nine urban and landscape recipes. The twins specs still pass | S | M |
| DOC-01 | Stale text: `examples/deck/spatial-analysis-showcase/src/scenes/raster/missing-observations.compute.ts:1097` and three comments under `website/src/gpu/runs/entries/raster-bands-indices/` still say an unconverged run publishes nothing; `docs/api/gpu-network/network-movement-coverage.md:15` still gives 64 demand rows for NW05; `docs/api/variable-standardization.md` says inputs may have different chunk boundaries; `operators-validity.md` lacks the minimum-clearance `candidateOverflow` and `leafCapacity` note | S | L |
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

1. UF-01: commit the working tree (maintainer's decision on chunks).
2. KE-TESTS, EC03-R, then the LN-REC decision. Done 2026-11-07: CI gates (UF-02), test and example typechecks (UF-04), AUD-01, the NW-CENT interim guard, the EC03 sparse recipe.
3. UF-06, then M items by consumer demand: ST06, ST06b, ST07, PP04, NW04, NW-KSP, TE06, R6.3, R7.2.
4. NW-CENT structural fix when luma.gl ships a GPU-gated continuation.
5. At each `@luma.gl` bump, run the upgrade procedure in [upstream-workarounds.md](./upstream-workarounds.md).
6. Website, independent of the items above: the open items of sections 7.5 and 7.6, starting with WEB-PIC-12 (an independent re-score of the pictures) and WEB-INT-16 (text in pictures). How-it-works pages: the WEB-HIW rows of section 9.6, starting with WEB-HIW-01 and WEB-HIW-02.
7. Spatiotemporal, section 2.15: what remains after the 2026-11-10 implementation (section 8.5): LN-ST, the R-only fixtures of PLAT-06, A5 in RTS-08, then the M rows.
8. Library convergence, section 2.16. Landed 2026-11-09: the contract and its checker, the registry, and the cheap reworks. Next: FX-17 and LN-REC2; STD-10 once the solver files are quiet; the rest of STD-03, STD-06 and STD-07 by directory; the breaking half of STD-09 in one pass with lenses, kepler and the website; then GPU-05, GPU-07 and the larger reworks by consumer demand. New contributors follow `contributor-contract.md` and add their catalog entry and kernel-audit case in the same change. GAP rows wait for a consumer.

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
| WEB-INT-16 | Text in pictures | The layers draw no text: chart axes, bar labels, class legends and matrix ticks cannot be labelled. Several groups asked for it. Decide between a DOM overlay and a glyph atlas layer | M |
| WEB-INT-17 | Highlight as a layer feature | Probes draw highlights as glyphs or as render-only mask buffers written from `resolve`. A `highlight` option on the dots, polygons and raster layers (by id, value or mask) would remove about a dozen local helpers | L |
| WEB-INT-18 | `record-demo.ts` for the six computation runs | The graph layout from `describeGPUCommandGraph` differs after a rebuild, so five runs warn that the picture differs after parameters return to defaults; readbacks are not cached, so three runs rebuild to redraw; `extra` is not passed to `resolve` | L |

### 7.6 Open picture items (WEB-PIC)

| ID | Item | Detail | Pri |
| --- | --- | --- | --- |
| WEB-PIC-09 | Data that fits | No new extract was added. `normalized-difference-water-index` and `make-raster-band` still have no open water; `normalized-burn-ratio` shows a post-fire scenario derived from the measured July to September 2021 NDVI drop and says so in its caption. A Sentinel-2 scene with water and a post-fire date would replace both. The lidar tile classifies almost bimodally, so four ground-filter sliders barely act; a tile with intermediate relief would fix it. `pair-gather` needs two related real point sets. Register each new asset in `website/build/demo-data-sources.ts` and `THIRD-PARTY-DATA.md` | M |
| WEB-PIC-12 | Independent re-score | Each group read its own contact sheets. Nobody has re-scored all 428 pictures against the audit's 1 to 5 scale. Do it from `.reports/final/<group>-pointer-<page>.png` and the posters | M |
| WEB-PIC-13 | Inert sliders | 11 remain: `maximumIterations` (`datum-grid-transform`: only the inverse direction iterates), `slope` and `maximumThreshold` (`pointcloud-ground`) and `window` (`pointcloud-smrf`: the lidar tile), `joinStyle` (`offset-curve-components`), `significanceLevel` (`quantile-lisa`), `interceptBandwidth` (`multiscale-gwr`: no intercept surface is drawn), `stages` (`terrain-depression-hierarchy`), `targetHeight` (`point-horizon-visibility`), `jump` and `tolerance` (`spike-repair`). Also `instrumentOrder` (`spatial-two-stage-least-squares`) and `reciprocal` (`island-attachment`), whose effect is below the verifier's threshold | M |
| WEB-PIC-14 | Weak gestures | Fewer than three distinct pictures at the five test positions: `point-in-polygon-join` (the pointer is outside every lake at four of them), `pointcloud-statistical-filter` (its threshold is compile-time and settles), `group-concave-hull`, `multiscale-gwr`. 85 gestures change under 3% of pixels | M |
| WEB-PIC-15 | Siblings still alike | `variogram`, `cross-variogram` and `variogram-model-fit` (a rising curve on one layout); the encounter trio and the zone-events trio (one fleet, one bar chart); the two measures cards share a ramp family | L |
| WEB-PIC-16 | Left partly done | `resistance-connectivity-recipe` shows current magnitude, the potential is still nearly flat; `terrain-branching-flow` fans out modestly; `point-pattern-envelope` has bare curves; `max-p-regions` bars are small against their axis; `rank-markov` maps barely change between periods; `spherical-arc-overlay` is outlines only | L |
| WEB-PIC-17 | Suspected defects to confirm | `contiguity-weights` and `fuzzy-contiguity` draw cross-map links: the default `snapTolerance` of 0.03 world units looks too generous for the Chicago tracts. `normalizedChoice` is NaN for some `visibility-graph-recipe` nodes. Several quad zones receive no points, so their quantile is NaN. Confirmed as documented contracts, not defects: `GPUNetworkFacilityOptimization` returns an empty selection when any demand is uncovered; `GPUStudyWindowPreparation` skips later encodings until invalidated; `GPURasterRegistration` is precise to 1 / `upsampleFactor`; the Mantel operation correlates listed pairs only | M |
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
| WEB-HIW-01 | Timings, WGSL and transient ownership in records | No scenario passes a `GPUComputationRecorder` or reads timings, so no page can show durations, shader source from a record, or shared allocations (`describe-command-graph.ts:191`) | M |
| WEB-HIW-02 | Raster snapshots in view models | 20 of 410 view models have a raster map although 322 raster captures exist: `extract-lens-view-model.ts` needs `origin` and `cellSize` | M |
| WEB-HIW-03 | Workflow pictures | Each step shows that operation's own recorded picture, not the chain's intermediate; `GPUPeriodIndex` has no catalog entry; the drainage record ran on a 3 by 3 grid | M |
| WEB-HIW-04 | Block usage counts | `GPUSort`, `GPUReduction` and `GPUGridIndex` kernels carry no operator name in records, so the building-blocks table matches node ids; reductions are probably undercounted (10 records, 30 source files) | L |
| WEB-HIW-05 | Small text on phones | Some SVG labels are 8 to 9.5 px at 390 px; wide figures scroll inside their frame | L |
| WEB-HIW-06 | Index cards of the five new pages | Static thumbnails only; no live scene on hover | L |
| WEB-HIW-07 | Reference text that disagrees with source | `docs/api/gpu-spatial-analysis/concepts.md:276` cites the removed `compiled.describe()`; `recipes.md:13` gives a different recipe return shape from `recipe-utils.ts:51` | L |

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
