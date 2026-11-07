# Roadmap

Status date: 2026-11-07. Baseline: `@luma-contributors/analysis`, `/lenses` and `/kepler` on published `@luma.gl/*@10.0.0-alpha.10`, HEAD `9a66f08` plus an uncommitted working tree.

This is the single current roadmap. It replaces about 19 planning documents from the luma.gl `map-contributors` branch. The superseded documents are kept verbatim in [archive/](./archive/README.md) as the evidence trail.

Companion documents:

- [coverage-matrix.md](./coverage-matrix.md): function-by-function coverage and where each symbol lives.
- [upstream-workarounds.md](./upstream-workarounds.md): every dropped capability, workaround and accepted regression caused by depending on published luma.gl.
- [upstream-first-plan.md](./upstream-first-plan.md), [upstream-first-pending.md](./upstream-first-pending.md): the decisions behind the split.

## How to use and maintain this file

- Every open item has an ID. Reuse the ID in commits, issues and test names. IDs from the old documents (ST01, RS02, TE06, CR01, KG06 and so on) are kept. New IDs use the prefixes `UF` (upstream-first completion), `AUD`, `NW`, `LN`, `KE`, `SH`.
- Priority: **H** = clear scope and real analysis or correctness value, do next. **M** = useful, needs a consumer or a design step. **L** = research, belongs in luma.gl, or an explicit non-goal.
- "Current limit" cites `file:line` in this repository as of the status date. Source files are still moving, so re-grep before you rely on a line number.
- "Source" points at the archived document (path relative to this directory) or, for documents that were not archived, the luma.gl branch path.
- When you finish an item, delete its row and, if it changes coverage, update [coverage-matrix.md](./coverage-matrix.md). When you accept a new regression or add a workaround, add it to [upstream-workarounds.md](./upstream-workarounds.md) in the same change.
- Review this file at each `@luma.gl` alpha bump (see the upgrade procedure in the workarounds file) and at each release.
- Nothing in the coverage inventory is "Planned". Every row is Implemented, Partial or Excluded. This roadmap lists what is left inside and around those rows.

## High-priority summary

| ID | Item | Why |
| --- | --- | --- |
| EC03 | Sparse resistance connectivity was lost in the split | Silent capability loss; the 256-vertex recipe comment still tells users to use a solver that no longer exists |
| NW-CENT | `GPUNetworkCentrality` unrolls up to 1,024 rounds at compile time | Compile-cost cliff on large graphs |
| LN-REC | Kernel `record:` annotations are accepted and ignored | Lens records lose source text, bindings, formulas and provenance |
| AUD-01 | 12 point-cloud and datum-grid contributors have no kernel-audit case | Finishes the migration; cheap |
| KE-TESTS | 271 of 367 kepler spec files are excluded from every run | Most kepler code is untested |

## 1. Now: finish upstream-first

Plan: [upstream-first-plan.md](./upstream-first-plan.md). Phases 1 to 6 are mostly done in the working tree. This block is the baseline to reach before any new feature work.

| ID | Item | Current state | Pri |
| --- | --- | --- | --- |
| UF-01 | Commit the working tree in coherent chunks and add a git remote | About 550 uncommitted entries on top of `9a66f08`; no remote configured; `package.json` `repository` points at a repository that is not set up | H |
| UF-02 | CI gates | `.github/workflows/test.yml` runs typecheck at line 39 but not `node scripts/check-upstream-first.mjs`, test or example typechecks, or a website build | H |
| UF-03 | Re-measure performance after the split | The branch measurements used forked primitives. Run `yarn test-benchmarks` and record numbers next to the accepted regressions in [upstream-workarounds.md](./upstream-workarounds.md) section 3 | M |
| UF-04 | Typecheck tests and examples | `tsconfig.check-src.json` covers `src` only. Tests and examples are not covered | H |
| UF-05 | Remove stale subpath imports | `@luma-contributors/analysis/gpu-core`, `/gpu-dataframe`, `/gpu-raster`, `/gpu-graph-algorithms` still appear in `examples/deck/spatial-analysis-showcase/src/scenes/flows/airline-network*.ts` (3 files), `examples/deck/kepler-compute/src/time-contract-interval.ts`, and a comment at `modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag.ts:85` | M |
| UF-06 | Docs baseline | `docs/` mentions `@luma.gl/graph/experimental` or `gpu-core` internals in 8 files and the removed-feature docs (fluid, fire, ocean, fft, sort) are still tracked. `modules/lenses/README.md` still uses `@luma-contributors/graph/*` aliases. `AGENTS.md:27` still describes `src/_pending/`, which no longer exists. Re-confirm the doc-snippet and doc-parameter specs (`modules/analysis/test/analysis-core/doc-*.node.spec.ts`) run and pass: they are no longer `it.skip` | M |
| UF-07 | Package exports | `@luma-contributors/analysis` has no `./analysis-core/graph-support` export, which kepler and examples need. Run `node scripts/verify-package-exports.mjs` after a build | M |
| UF-08 | Prune dead test globs | `vitest.config.ts` `nodeOnlyTestPatterns` and `nodeWrapperSourcePatterns` (lines 24 to 38) list tests of deleted directories (`gpu-tables`, `arrow`) | L |
| UF-09 | SPDX headers | About 14 test, script and example files lack `SPDX-License-Identifier` and `SPDX-FileCopyrightText` | L |
| UF-10 | Release housekeeping | No `CHANGELOG.md`, `CONTRIBUTING.md` or publish guide. `modules/analysis` is public at `0.1.0-alpha.0`; lenses and kepler are private | L |

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
| PP01 / PP02 | Intensity estimation, more study-window corrections, kernel-smoothed pair correlation, tie breaking for extreme-rank-length envelopes | Not implemented. `gpu-ripley.ts:183` rejects inhomogeneous K | L | `archive/geospatial-toolkit-expansion-follow-up.md` (item 4); parallel record (PP02) |
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
| NW-CENT | Centrality without compile-time unrolled rounds | Reachability, sigma and delta rounds are unrolled per batch: default 64, at most 1,024 (`gpu-network/network-centrality/gpu-network-centrality.ts:149`). About 197,000 command nodes at the maximum. Needs a GPU-gated continuation that the published compiler lacks. Interim: reject or warn on large round counts, and document the cliff | H | `archive/gpu-analysis-algorithms-roadmap.md` (line 161); `archive/geospatial-toolkit-expansion-follow-up.md` (line 142) |
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
| AUD-01 | Kernel-audit cases for 12 contributors: `GPUDatumGridTransform`, `GPUPointBuffer`, `GPUPointCloudChange`, `Features`, `Ground`, `ICP`, `Neighborhoods`, `SMRF`, `StatisticalFilter`, `Surfaces`, `Height`, `VoxelMerge` | Skipped as `NOT_YET_AUDITED` (`modules/analysis/test/analysis-core/kernel-audit.node.spec.ts:35-49`). Four chunked-weights contributors are also skipped (`:50-53`) with a documented reason | H | [upstream-first-foundation.md](./upstream-first-foundation.md) section 5 |
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
| EC03 | Sparse resistance connectivity | **Lost in the split.** `addResistanceConnectivityRecipe` solves at most 256 vertices and 65,536 edges (`landscape-ecology/resistance-connectivity.ts:34,37`; iterations at most 4,096). The branch had `addSparseResistanceConnectivityRecipe`, `addSparseResistanceComponents`, `addSerialSparseResistanceComponents` and `GPURasterResistanceProps` on the non-public `GPUJacobiPCG`; none of them exist here. The comment at `:33` still says "use the general sparse solver for larger systems". User docs (`docs/api/gpu-spatial-analysis/landscape-ecology.md`, `urban-ecology-lenses.md`) still describe the sparse path. Steps: (1) fix the comment and docs now; (2) restore on public APIs (own CG or Jacobi-PCG in `landscape-ecology/`) or ask luma.gl to export `GPUJacobiPCG`; (3) re-run the Circuitscape golden comparisons | H | `archive/geospatial-coverage-parallel-implementation.md` (EC03) |
| GE01 / GE08 / UR01-UR05 | Capacity limits: nesting above 64; straight skeleton above 1,024 vertices; urban recipe capacities | Bounded by contract | L | parallel record |
| EC02 | High patch-count behaviour is unmeasured | Not measured | L | parallel record (EC02) |
| EC04 | Other segregation statistics and nulls | Not implemented | L | parallel record (EC04) |
| GEOM-01 | Rectangle clip outputs for polygons | `gpu-rectangle-clip.ts:199,205` | L | code stub |
| GEOM-02 | Line smoothing source rows and measures | `gpu-line-smooth.ts:164` | L | code stub |

### 2.11 Kepler

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| KE-TESTS | Run the kepler tests | `test/kepler-compute-example-specs.json` lists 271 of the 367 spec files in `modules/kepler/test`; `vitest.config.ts:14-21` excludes them all because they need the parked kepler-compute example. Decide: move the needed example code under `modules/kepler/test`, or keep the example and make it a workspace | H | `archive/kepler-compute-integration-plan.md` |
| KE-EX | Fate of `examples/deck/kepler-compute` (about 750 files) | Parked: not in `workspaces`; imports `@deck.gl-community/gpu-layers` and `@deck.gl/core`, which are not dependencies | M | `archive/kepler-compute-integration-implementation.md` |
| KE-UI | Remaining host work: native Group By panel binding, non-point altitude layers, richer labels, micro and nanosecond time admission, multipart timed trips, acceptance on other hardware tiers | Open (mostly example code). Acceptance was measured on one Apple M-class machine | M | `archive/kepler-compute-integration-implementation.md` (Remaining implementation) |
| KE-GUARD | About 14 "Unsupported ..." validation guards | By design, not stubs. Examples: `kepler-group-by.ts:111`, `style-plan.ts:104,147`, `analysis-recipe.ts:357,1367`, `prepare-spatial-selection.ts:51,77` | L | code |
| KE-DECK | Layers that depended on deck GPU layers removed on the branch (cluster, flow, globe) | External; track with the deck.gl-community package | L | luma.gl branch `dev-docs/reviews/comprehensive-gpu-review-2026-11-07.md` |

### 2.12 Lenses and computation visualization

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| LN-REC | Wire kernel annotations to the recorder | `record:` is accepted and ignored. There are 32 `record: {` sites in 23 files under `modules/analysis/src`, plus the option plumbing in `analysis-core/kernels/wgsl-kernel-nodes.ts` (lines 73, 111, 296, 618, 709). `GPUComputationRecorder` (`computation-record/computation-recorder.ts`) is used by three lens test specs only; no analysis contributor accepts a `recorder` prop. Lens records therefore carry structure and timings but no source text, bindings, formulas, loop provenance or `derivedFrom`. Decide per contributor family: add an optional `recorder?` prop (pattern in [upstream-first-recording.md](./upstream-first-recording.md)), or delete the dead `record:` metadata | H | `archive/computation-visualization-roadmap.md` (V0, V3) |
| LN-SEL | GPU selection backend | `GPUCrossfilterSelectionBackend` exists (`lenses/src/computation-views/coordinator/gpu-crossfilter-selection-backend.ts`). Verify it against published `GPUCrossfilter`, which lacks external masks (workarounds, section 2) | M | same |
| LN-DQ | Rebuild dataframe-query and formula scenarios on public query metadata | 8 scenarios dropped | M | [upstream-first-foundation.md](./upstream-first-foundation.md) section 2 |
| LN-STAB | StabilityMap and Purpose rung | Not built (also not on the branch) | L | `archive/computation-visualization-roadmap.md` |
| LN-V5 | `GPURecipePlan`, `ComputationDrawer`, V6 to V8 phases, space-time cube | Never built | L | same |
| LN-ENC | `CommandEncoder` pass and slot labels; rendering and trace packs | Belongs in luma.gl core | L | same |
| LN-PKG | Lenses package hygiene | `modules/lenses/README.md` uses stale names and refers to a missing script; `package.json` exports pattern `./*/computation-pack/index` does not match the source directories; dependency ranges differ from the analysis peer range | M | health audit |

### 2.13 Showcase and examples

| ID | Item | Current limit | Pri | Source |
| --- | --- | --- | --- | --- |
| SH-IMPORT | Showcase scenes import removed subpaths | See UF-05 | M | health audit |
| SH-DATA | Large binary data tracked in git | `examples/deck/spatial-analysis-showcase/public/data` has 3 to 5 MB PNG and bin files. Consider git LFS or a download script; provenance is in `THIRD-PARTY-DATA.md` | L | `archive/showcase-sprint-datasets-2026-10-24.md` |
| SH-NAMES | Example package names and versions | Examples still carry `luma.gl-examples-*` names and `10.0.0-alpha.10` versions | L | health audit |
| SH-SPRINT | 2026-10-24 sprint scenes | Finished; outcomes in [research/showcase-overhaul-2026-10-25](./research/showcase-overhaul-2026-10-25) | - | `archive/showcase-sprint-2026-10-24.md` |

## 3. Dropped by upstream-first

Not backlog items. They were removed on purpose; see [upstream-first-pending.md](./upstream-first-pending.md) for the replacement of each symbol and [upstream-workarounds.md](./upstream-workarounds.md) for the full register. In short:

- Sparse resistance connectivity (not intentional; see EC03).
- Large-graph SCC, the iteration gate, direct local clustering node, command-graph-transient CSR views.
- Histogram and skip-cube crossfilter (use published `GPUCrossfilter`).
- Residency arena, resident row selection, tile LOD.
- Onesweep sort, look-back scan, double-single helpers, benchmark harness.
- Arrow batch stream, partition descriptors, `GPUCPUOperation`, `GPUKernelNode`.
- Command-graph recording hooks and `captureGPUCommandGraph` (replaced by `computation-record`).
- Dataframe query description and the `gpu-core`, dataframe-query, SCC and SQL lens packs.
- Bounded dispatch above about 4.2M rows, compensated sums and block compaction in published primitives (accepted regressions).

## 4. Corrections to the branch documents

The archived documents say the following. These statements are out of date or wrong for this repository:

| Archived claim | Current fact |
| --- | --- |
| Quality roadmap "still open": AZP articulation test | Done: variant `local-articulation-candidates` in `gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions.ts:801` |
| Quality roadmap "still open": D-06 crossing points to snap rounding | Done: `gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts` |
| Algorithms roadmap line 29: variogram fitting is a CPU helper | `GPUVariogramModelFit` exists (`dataframe-analysis/pair-statistics/gpu-variogram-model-fit.ts`); IN01 is implemented |
| Spatial roadmap phase 9: no likelihood, panel, SUR, probit or MGWR admitted | ST03 to ST06 are implemented (panel ML, SUR, probit, MGWR) |
| Final review: no planned or partial rows | KG06 is Partial, and EC03 is now Partial in this repository |
| Kepler plan: residency arena and tile LOD | Dropped; kepler uses its own `resident-*` code |
| SCC and centrality notes in the follow-up document | SCC is capped at 1,024 vertices here; centrality still unrolls rounds (NW-CENT) |
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

1. UF-01, UF-02, UF-04: commit, CI gates, test and example typechecks.
2. EC03 (fix comment and docs first), AUD-01, KE-TESTS.
3. NW-CENT interim guard, then LN-REC decision.
4. UF-05 to UF-07, then M items by consumer demand: ST06, ST06b, ST07, PP04, NW04, NW-KSP, TE06, R6.3, R7.2.
5. At each `@luma.gl` bump, run the upgrade procedure in [upstream-workarounds.md](./upstream-workarounds.md).
