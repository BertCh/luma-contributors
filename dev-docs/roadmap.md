# Roadmap

Status date: 2026-11-18. Baseline: `@luma-contributors/analysis`, `@luma-contributors/lenses` and `@luma-contributors/kepler` use published luma.gl. The [alpha package contract](./release-package.md) records the current integration target, accepted peer range and unreleased status separately.

Companions: [coverage-matrix.md](./coverage-matrix.md) records coverage; [contributor-contract.md](./contributor-contract.md) defines contributor rules; [upstream-first-plan.md](./upstream-first-plan.md) defines the package boundary; [upstream-first-recording.md](./upstream-first-recording.md) defines recording; [upstream-workarounds.md](./upstream-workarounds.md) records accepted limits; [upstream-proposals.md](./upstream-proposals.md) holds upstream work.

Reuse item IDs in issues and changes. H means clear scope and analysis or correctness value; M needs a consumer or design step; L is research or deferred work. Source paths are repository-relative; analysis paths below omit `modules/analysis/src/`. Research sources hold detail, not an instruction to repeat a finished pass.

Delete completed rows. Update coverage when coverage changes, and the workaround register when accepting a regression. Review the backlog at each luma.gl bump and release. Validate retained code references before changing them. New contributors include their catalog entry, documentation and kernel-audit case in the same change.

[coverage]: ./coverage-matrix.md
[workarounds]: ./upstream-workarounds.md
[proposals]: ./upstream-proposals.md
[library]: ./research/library-review-2026-11-09/README.md
[spatiotemporal]: ./research/spatiotemporal-review-2026-11-09/README.md
[math]: ./research/math-correctness-review-2026-11-10/README.md
[cpu]: ./research/cpu-suitability-2026-11-11/README.md
[new]: ./research/new-contributors-2026-11-11/README.md
[usage]: ./research/luma-usage-review-2026-11-11/README.md
[pictures]: ./research/picture-rescore-2026-11-10/README.md
[split]: ./research/split-review-2026-11-08/README.md
[fork]: ./research/fork-comparison-2026-11-13/README.md
[review17]: ./research/library-review-2026-11-17/README.md

## High-priority summary

This index points to the open records below; it does not add separate tasks. Generic device-side iteration is PLAT-04: host-controlled centrality continuation already ships.

| ID | Still open |
| --- | --- |
| MOB-03 | Decide whether point-centred speed-outlier detection becomes the default. |
| WEB-PIC-12 | Rework the remaining weak pictures and gestures, then re-score the whole site in both themes. |
| KE-TESTS | Resolve runtime and typecheck exclusions from the per-file manifest and the cold-cache DuckDB failure. |
| UPS-CPU-1 | File and qualify the upstream packed CPU-stage transfer path; CPU-08 depends on it. |
| AX-01, AX-20, AX-40, AX-70 | Qualify the implemented host journeys, profiling latency and live map-change latency; rerun release journeys. See the open AX gates below. |
| STD-04, STD-06, STD-07 | Finish status ports, raster context and published raster-band validity. STD-10 solver adoption remains M priority. |
| GPU-05 | Index large-radius work and snapping candidates; the point-cloud grid stays local by contract mismatch. |
| LUM-09 | Decide the fill-holes status contract before adopting published connected components. |
| LR-04, LR-22, LR-23 | The library-review fixes and the luma.gl 10.0.0-alpha.16 bump landed on 2026-11-18. Before publishing: settle the canonical repository URL and publish from the checked tarball; file the four drafted upstream issues. See "Library review 2026-11-17 (LR)". |

## Earlier implementation passes

BL-PASS, SC and IMP have landed; their completion records are below. SC/IMP implementation and documentation are committed in `69305eb4` and `bdc715ea`. The domain tables and known-limits register retain their outstanding acceptance work; the dated research directories retain the audits and measurements. Historical gate results are recorded as reported results, not reruns on the current tree.

## Library review 2026-11-17 (LR)

A review of `@luma-contributors/analysis` and `@luma-contributors/lenses` from the outside: ergonomics, consistency, release readiness and direction. Evidence, the verified defect tables and the reasoning are in [research/library-review-2026-11-17][review17]; finding IDs below (ERG, CORE, SSR, GEO, PTS, DGGS, TRJ, RCP, NET, RT, DPC, CAT, LENS, TOOL, DOC) are defined in its area files.

The implementation pass of 2026-11-18 landed LR-01 to LR-21 in two tranches and drafted LR-23; what landed, how it was verified and the gate results are in [implementation.md](./research/library-review-2026-11-17/implementation.md), and caller-facing changes are in `CHANGELOG.md`. The rows below are what remains. New operator families wait for the alpha unless a named consumer needs one.

Every new or changed public result contract lands with one discriminating test: a fixture on which a plausible wrong implementation (ignored candidate flag, row and ID swapped, tail not written, raw intercept read back) gives a different consumer-visible answer. A fix records the test failing on the parent revision.

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| LR-02 | Visibility oracle regeneration | The visibility oracle fixtures were not regenerated numerically: GRASS, R and the GDAL Python bindings are absent on the development machine (the runner, its inventory and its failure paths are tested). | M | TOOL-06 |
| LR-04 | Publish the alpha | The packed-consumer gate, the checked release tarball, the required Arrow peer and the licence inventory exist. Not done: package metadata and READMEs name `visgl/luma-contributors` while the configured remote is a private repository, so the canonical URL needs the maintainer; nothing has been published. | H | TOOL-02; [release-package.md](./release-package.md) |
| LR-08 | Multipart feature-level joins | Multipart point and line `featureOffsets` work in `GPUSpatialPredicateJoin` for `intersects`, `within` and `coveredBy` (engine `auto` or `fast`, no `relate` output), in `GPUNearestFeatureJoin` neighbors mode, join candidates and prepared joins. Still thrown: other predicates, DE-9IM output and `engine: 'relate'`, nearest-feature weights, buffer selection, point-in-polygon join, line length per polygon and `GPUGeometryPredicates`; polygon/polygon containment in the CPU and graph-operation join forms. Not built: feature-level bounds and per-predicate reductions beyond the three predicates. | M | GEO-12 |
| LR-10 | Recipe scheduling rollback | A recipe's declaration, shape, alias, device-limit and contributor-expansion failures are atomic: they leave the caller graph and its resources unchanged (all `add...Recipe` functions in the spatial and network recipe indexes). A failure during the final caller-graph mutation (an ID collision with an unrelated caller node or resource, an already compiled graph, an overridden scheduling method, allocation or device failure) is not rolled back, because `GPUCommandGraph` has no transaction, enumeration or removal API. Atomic preparation declares each recipe twice (once on a scratch graph, once on the caller graph), so recipe declaration costs about double. | L | RCP-04 |
| LR-17 | Loose root files | Sixteen `raster-analysis` and five `gpu-network` family-root files stay at their roots (`root-file` sites in `scripts/contributor-contract-baseline.json`). | L | CORE-14; `scripts/contributor-contract-baseline.json` |
| LR-19 | Typecheck and test memory | Test typechecks run per family with separate caches and node tests use one worker under `yarn verify`. `scripts/measure-typecheck.mjs` exists; no peak RSS was observed (`ps` is refused in the measuring environment), so the per-family project-reference decision is open. Adopt only with at least 20% lower sampled peak RSS and no more than 10% cold or warm time regression at the same heap on an idle machine. One sample: source 4343 files 12.5 s cold, references 8953 files 19.5 s cold (55.8% slower), warm 14.0 s vs 0.12 s. | M | TOOL-04, TOOL-11 |
| LR-20 | Lenses | All 427 scenarios capture and have a committed record. Every scenario has a declaration, choice and record-signature gate (node IDs, kinds, read and write resources, capture names); numeric capture values change only on recapture. The chunked-algebra weights scenario waits on `higherOrder` support in `GPUChunkedSpatialWeightsAlgebra`. Dataset licence clearance and workspace separation come before the package can be published. | M | LENS-06, LENS-08, LENS-09 |
| LR-22 | Qualify alpha.16; keep segmented sums/PCG adoption separate | Shared grid WGSL and compile-option adaptations are done and gated on the bumped tree: Node suite 1,855 files, GPU suites pass, the 30-case grid regression passes 30 of 30 (reader and published builder agree on the device). Open: the cost-distance continuation round count failed 2 of 10 runs on alpha.16 (0 of 9 on alpha.14), then passed 20 of 20; cause not found, strict equality and raw-state diagnostics kept, no production fix claimed. The bump landed on master on 2026-11-18; every `yarn verify` step passed on the bumped tree as separate runs, not as one uninterrupted command. Remaining: segmented sums and PCG adoption as follow-ups. | M | [workarounds] |
| LR-23 | Upstream filings | Four issue bodies with public-only reproductions are in [upstream-drafts](./upstream-drafts/) (packed CPU/GPU transfer with its benchmark, histogram edges and domain, graph algorithms on caller CSR with masks, typed raw-buffer import). None is filed. | M | [upstream-first-plan.md](./upstream-first-plan.md) |
| LR-24 | Multipart nearest modes | `GPUNearestFeatureJoin` multipart input requires neighbors mode (`neighborIds` and `neighborCounts`). | L | GEO-12 |
| LR-25 | Flow path decomposition size | `paths` accepts at most 128 expanded vertices, 1024 expanded edges and `pathCapacity` 1 to 1024 (`NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT`, `NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT`); the decomposition has no separate iteration budget. | L | NET-10 |
| LR-26 | Regression tile ceiling | Error GM, 2SLS, weights traces and the likelihood contributors need at most 4096 row tiles; wider inputs need a larger `tileRowCount` or fewer chunks; no hierarchical merge exists. | L | CORE-14 |

## Open backlog by domain

### Repository, examples and integration

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| UF-01 | Repository metadata | Package repository URLs name visgl; the website links to the private BertCh repository. | M | `modules/*/package.json`; INT-09 |
| UF-03 | Re-measure performance on published primitives | Record benchmark results beside accepted regressions; use PLAT-07 for join scale factors. The fork recorded numbers for its own versions (compaction, sort, point-in-polygon, projection, residency); none compares against alpha.13. | M | [workarounds]; [fork] |
| UF-12 | Example checks | Example typechecks are CI-only. Showcase checks import missing `scripts/playwright/*.mjs`; the explorer has no check script. Visual acceptance remains open. | M | `.github/workflows/test.yml`; `examples/deck/` |
| UF-10 | Release housekeeping | No contributor or publish guide; analysis is public at 0.1.0-alpha.0, lenses and kepler are private. | L | `modules/*/package.json` |
| SH-DATA | Large example assets | Consider LFS or a download script for 3–5 MB PNG/bin files; preserve provenance. | L | `examples/deck/spatial-analysis-showcase/public/data`; `THIRD-PARTY-DATA.md` |
| SH-NAMES | Example package metadata | Names still use `luma.gl-examples-*`; versions are alpha.10. | L | `examples/deck/*/package.json` |
| INT-07 | Resolve split-review leads | Re-check dropped lens/showcase files, unlisted specs, lost tests, EC03 limits/validation and register errors against current code before acting. The fork changes to luma.gl-owned code are mapped separately. | M | [split]; [fork] |
| INT-08 | Establish CI and Pages build policy | CI execution is unverified; decide whether Pages builds belong in a workflow. | M | `.github/workflows/test.yml`; `website/scripts/deploy-pages.mjs` |
| INT-09 | Public website repository link | The header points to a private repository. | L | `website/src/components/brand/site-header.tsx` |

### Spatial statistics, regression and local statistics

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| ST02 | Spatially robust restriction inference | Not implemented (`GPURestrictedRegimeRegression` exists) | L | [coverage]; `gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts`: `GPURestrictedRegimeRegression` |
| ST03 | Panel-error random effects, time effects, clustered covariance, dynamic panels | Panel error ML supports unit fixed effects only; panel lag ML supports fixed and random effects. Other effects and covariance extensions remain open; see `gpu-spatial-panel-error-maximum-likelihood.ts`, `gpu-spatial-panel-lag-maximum-likelihood.ts` and `gpu-spatial-panel-regression.ts`. | L | [coverage] |
| ST04 / ST05 | Iterated SUR; latent spatial probit | Single-pass `GPUSeeminglyUnrelatedRegression` and `GPUProbitRegression` only | L | [coverage]; `gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression.ts`: `GPUSeeminglyUnrelatedRegression` |
| ST06 | MGWR: bandwidth confidence intervals (full backfit selection uncertainty) | AICc and golden-section bandwidth search exist (`bandwidthSelection`); selection uncertainty is not reported. `gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts` rejects other variance conventions | M | [coverage] |
| ST08 | Rank Markov: fractional ties, missing-panel interpolation, diagnostics above 64 states | Sorted ranking supports more than 64 states, bounded by the dense transition-count binding; optional reference diagnostics remain capped at 64 states. Fractional ties and missing-panel interpolation remain open (`GPURankMarkov`). | L | [coverage]; `gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts`: `GPURankMarkov` |
| ST09 | Geographic (non-planar) geometry and topology repair for polygon statistics | Planar only. `gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts` rejects distance output | L | [coverage] |
| ST10 | blockCV: hierarchical and merge blocks, response balancing, continuous variogram optimizer | Not implemented (`GPUSpatialCrossValidation`) | L | [coverage]; `gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts`: `GPUSpatialCrossValidation` |
| ST11 | Spatial interaction: multiple covariates, other count distributions, coefficient uncertainty in calibrated flow | Not implemented (`GPUSpatialInteraction`) | L | [coverage]; `gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction.ts`: `GPUSpatialInteraction` |
| ROSE-01 | Other circular angle conventions | Radians with positive-x counter-clockwise origin only. | L | [coverage]; `gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts`: `GPURoseStatistic` |
| KG02 | Dorling cartogram: GeoDa coordinate parity, convergence guarantee | Heuristic by contract; relaxation limited to 256 passes (`gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts`) | L | [coverage] |
| KG03 | Exact natural breaks beyond 8,192 rows or 64 classes; float64 objective | Exact mode materializes costs to 1,024 rows and recomputes to 8,192 (`dataframe-analysis/column-classification/gpu-class-breaks.ts`). Kepler uses a worker Jenks instead | L | [coverage] |

### Point processes, sampling and clustering

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| PP01 / PP02 | Study-window corrections and envelope tie breaking | Inhomogeneous K, intensity and smoothed pair correlation exist to 4,096 events; more corrections and extreme-rank-length tie breaking remain. | L | `gpu-spatial-analysis/point-process/`: `GPUInhomogeneousPointPattern` |
| PP05 | Response-dependent replacement and design variance in GRTS sampling | Not implemented | L | [coverage] |

### Interpolation and simulation

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| IN04 | Linear model of coregionalization beyond 4 variables or structures | `GPUCoregionalizationFit`, `CPUCoregionalizationFitOperation` and `computeCoregionalizationFitOnCPU` fit up to 4 variables and 4 structures (Goulard and Voltz); `GPULocalKriging` takes the same layout. Larger model layouts remain open. | L | [coverage]; `dataframe-analysis/pair-statistics/coregionalization-parameters.ts` |
| IN05 / IN02 | Larger local kriging systems, automatic RBF indexing and global solvers | `GPULocalRBF` supports polynomial degrees -1 through 2; global-system RBF and indexed global kriging remain open. Larger local kriging systems must lift `GPULocalKriging`'s 32-neighbor private-system cap with bounded storage/cooperative solves (33 neighbors is the first rejected case). For automatic RBF indexing, filter usable samples before top-k and preserve distance/source-row ties across query chunks: a nonfinite nearest value must not exclude the next usable sample. Supplied candidate rows alone do not provide this index. | L | [coverage]; `gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf.ts`; `gpu-spatial-analysis/spatial-interpolation/kriging-compatibility.md` |
| IN06 | Joint covariance across separate conditional simulations | Not implemented, by design | L | [coverage] |

### Network

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| NW02 | Max-flow path decomposition | `GPUNetworkFlow` already selects parallel preflow push-relabel above the 128-node/1,024-edge reference limits, or when a CSR topology is supplied. Decomposing the resulting edge flow into source-to-sink paths remains open. | L | [coverage]; `gpu-network/network-optimization/gpu-network-flow.ts` |
| NW03 | Negative-cost circulation and lower-bound constraints | Not implemented | L | [coverage] |
| NW05 | Mixed-integer or fractional allocation; certified capacitated p-median | Exhaustive and bounded: demand at most 64 (`gpu-network/network-optimization/gpu-network-facility-optimization.ts`) | L | [coverage] |
| NW06 / X04 | Non-FIFO schedules, time windows, transit | FIFO only (`gpu-network/network-optimization/gpu-network-time-routing.ts`); transit is excluded | L | [coverage] |
| NW-GA | Generic graph adapters for dense traversals | Not implemented | L | [coverage] |

### Raster, terrain and hydrology

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| R6.4 | Broader fractional and zonal composition envelopes | Partial: `GPURasterFractionalCoverage` and zonal statistics exist; whole-domain and tiled acceptance cases are not all closed | M | [coverage]; `raster-analysis/gpu-raster-fractional-coverage.ts`: `GPURasterFractionalCoverage` |
| R7.2 | Scikit-image watershed plateau-order parity | `GPURasterWatershedSegmentation` implements marker-controlled priority flood (FIFO discovery age, up to 16,384 cells) and a rooted minimum-spanning forest (device-bound, fewer than 2^30 cells). Neither promises scikit-image plateau ordering; qualify that distinct ordering only for a consumer requiring parity. | L | [coverage]; `raster-analysis/gpu-raster-watershed-segmentation.ts` |
| R7.3 | Volumetric microscopy | Gate-deferred, no code. CLAHE (R7.1) exists as `GPURasterContrastEqualization` | L | [coverage] |
| RS02 | Sparse or string zone IDs | Dense numeric IDs only, up to 65,534 classes | L | [coverage] |
| RS03 | Curvilinear grids for conservative remapping | Not implemented | L | [coverage] |
| RS06 | Irregular dates in temporal composites and filters | Not implemented | L | [coverage] |
| RS10 | Endmember discovery, ISRA and NMF | Not implemented | L | [coverage] |
| RS11 | Multiple break search, calibrated p-values, full BFAST | Single-break seasonal model | L | [coverage] |
| R-REPROJ | Automatic source-window selection and additional resampling filters | Ordered source-tile mosaic assembly and a double-single inverse-projection path exist. `GPURasterTileReprojection` packs the entire bounded source window into scratch; callers still select/fetch windows and supply interpolation halos. Automatic window selection and filters beyond nearest/bilinear remain open. | L | [coverage]; `raster-analysis/gpu-raster-tile-reprojection.ts`; `raster-analysis/gpu-raster-reprojection.ts` |
| TE06 | Host paging, independent depression filling and tiled branching transport | D8 has bounded per-tile scratch and a coarse boundary forest, including a structural test above the old resident ceiling. Inputs/outputs remain caller-owned with no host paging or filling; routing must be globally prepared. Branching transport remains resident. Qualify each of these three extensions separately. | M | [coverage]; `gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts`: `GPUTerrainTiledFlow` |
| RELIEF-01 | Percentile stretch in relief blending | `gpu-terrain/relief-visualization/gpu-relief-blend.ts` (needs a histogram) | L | [coverage] |
| A-18 | `GPUTerrainSpikeRepair` persistent buffer | Persistent-buffer support needs an upstream change | L | [workarounds]; `gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts`: `GPUTerrainSpikeRepair` |
| TE05 | Horizontal, Nuth-Kaab and tilt coregistration models | Not implemented (`GPUTerrainDEMCoregistration`) | L | [coverage]; `gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts`: `GPUTerrainDEMCoregistration` |
| LUM-24 | Embedding nearest-classifier on similarity search | 4,096 training-vector cap; migration adds `pixelCount × k × 8` transient bytes and a pass. | M | [usage]; `raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts`: `GPURasterEmbeddingKNearestClassifier` |

### Dataframe, point cloud, projection and geometry

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| KG06 | SQL: DOUBLE, WKB, arbitrary strings, right-column projections, UNION | Partial. The engine is published `@luma.gl/graph/gpu-sql`; nothing in this repository. Needs upstream work | L | [coverage] |
| DF-DESC | Public dataframe query description (intent, shader plan, formulas) | Unavailable; needed to rebuild lens scenarios. The fork implementation exists and is a candidate upstream pull request. | M | [coverage]; [fork] |
| ARROW-01 | More Arrow analytics types | `analysis-core/graph-support/arrow-analytics-normalization.ts` rejects unsupported types | L | [coverage] |
| PC01 | Globally propagated normal orientation; unlimited support | Viewpoint or up orientation per query | L | [coverage] |
| PC02 | Distributed global threshold reconciliation | Not implemented. Voxel downsample and radius filter are published | L | [coverage] |
| PC03 | Ground Delaunay or nearest interpolation; vertical datum | Not implemented | L | [coverage] |
| PC05 | Global alignment, trimmed objective for ICP | Not implemented | L | [coverage] |
| CR02 / CR03 | Epoch-aware grid resources, bounded residency | `GPUDatumGridTransform` (`geospatial-analysis/datum-grid-transform/gpu-datum-grid-transform.ts`) applies supplied grids; no epoch handling | M | [coverage]; `geospatial-analysis/datum-grid-transform/gpu-datum-grid-transform.ts`: `GPUDatumGridTransform` |
| CR04 / CR05 | Exact S2 predicates; S2 covering minimization | Not implemented | L | [coverage] |
| P.7a | 3D contract, ECEF, static Helmert | In neither this repository nor published luma.gl; belongs in luma.gl `gpu-project` | L | [coverage] |
| GE01 / GE08 / UR01-UR05 | Capacity limits: nesting above 64; straight skeleton above 1,024 vertices; urban recipe capacities | Bounded by contract | L | [coverage] |
| EC02 | High patch-count behaviour is unmeasured | Not measured | L | [coverage] |
| EC04 | Other segregation statistics and nulls | Not implemented | L | [coverage] |
| GEOM-01 | Rectangle clip outputs for polygons | `gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip.ts` | L | [coverage] |
| GEOM-02 | Line smoothing source rows and measures | `gpu-spatial-analysis/line-segmentize/gpu-line-smooth.ts` | L | [coverage] |
| LUM-23 | Dot-density triangle CDF on segmented scan | Prove triangles of different features never interleave before migration. | L | [usage] |

### Kepler

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| KE-TESTS | Run excluded specs and expand typecheck coverage | Resolve the per-file reasons in `excludedFromRuns` and `excludedFromTypecheck`; runtime and typecheck membership differ. Derive counts from the manifest with the command in [release-package.md](./release-package.md). Resolve remaining package/API dependencies and qualify cold-cache DuckDB/Arrow-schema behavior. KE-EX controls example-backed coverage. | H | `test/kepler-compute-example-specs.json`; `vitest.config.ts` |
| KE-EX | Qualify the compute example’s port | The example is registered in root workspaces and declares its deck.gl dependencies. Complete the KE-PORT acceptance gates below; installed dependencies alone do not qualify the fork-only capabilities recorded in [fork]. | M | `examples/deck/kepler-compute/package.json`; [fork] |
| KE-UI | Remaining host work: corrected multipart XYZ-pick and expanded polygon column-altitude qualification; grouped vertex-table polygon assembly, richer labels and acceptance on other hardware tiers | Open (mostly example code). Acceptance was measured on one Apple M-class machine | M | [coverage] |
| KE-DECK | Cluster, flow and globe layer integration | External; track with the deck.gl-community package | L | [coverage] |
| INT-04 | Reclassify test exclusions | Reasons and skip comments cite alpha.10; nine specs cite missing physical-transient statistics that alpha.12 publishes, while compilationStats remains unavailable. | M | `test/kepler-compute-example-specs.json` |
| CPU-06 | Replace hand-written CPU fallbacks with twins | Use twins where available in snapping and platform bindings. | M | [cpu]; `modules/kepler/src/analysis/catalog-gpu-snapping.ts` |
| LUM-31 | Group-by on published dataframe aggregation | Migration loses the null-key group and nonfinite counts; preserve those contracts. | L | [usage] |
| LUM-13 / LUM-32 | Predicate encode gate | Mask-only filters re-encode each frame; safe gating requires submit confirmation in explorer and scalar-selection contracts. | M | [usage] |
| LUM-14 / LUM-33 | GPU permutation feeds sorted-page gather | Needs a `KeplerSessionExplorer` pass-through and changes two pinned readback byte lengths. | M | [usage] |

#### Kepler port onto published luma.gl (KE-PORT, in progress 2026-11-14)

Decided 2026-11-14: run `examples/deck/kepler-compute` on published deck.gl 10.0.0-alpha.1, which pins luma.gl 10.0.0-alpha.14 and math.gl/loaders.gl 5.0.0-beta.8, so the whole repository moves to those versions. The 30-file closure of the unpublished `@deck.gl-community/gpu-layers` is copied from the old fork (`e0ed33791:modules/deck-gpu-layers`) into the example. The fork's geometry, polygon-topology and trip preparation moves into the example as Kepler-named domain helpers built on public luma.gl APIs; it is registered in [workarounds] and proposed upstream. Kepler stays on 3.3.0-alpha.18 until the acceptance baseline holds. The evaluation found that `@luma-contributors/*` imports all resolve, that alpha.14 adds nothing to the example's luma.gl surface (31 fork-only symbols), and that only deck.gl patch hunks 4, 5 and 6 are still missing from deck 10 alpha.1.

| ID | Stage | Gate |
| --- | --- | --- |
| KE-PORT-0 | Validate dependency and workspace integration: luma.gl/math.gl pins and example workspace registration are already present. Qualify the `@math.gl/arrow` import moves, deck/editable-layer dependencies, Vite aliases and launch options against the gates; current release targets are in [release-package.md](./release-package.md). | `yarn typecheck`, `yarn build`, `yarn test-node`, `yarn check-upstream-first` |
| KE-PORT-1 | Example-local gpu-layers closure ported to deck 10 (`COORDINATE_SYSTEM` removal, `coordinateFrame`, globe meter offsets) | Example tsc clean for `gpu-layers/`; its node specs |
| KE-PORT-2 | Geometry, polygon-contiguity and trip preparation re-homed in the example (22 symbols, 17 files) | Example tsc clean for those files; ported specs |
| KE-PORT-3 | Small luma gaps: Arrow normalisation and `GPUParameterBuffer` from `@luma-contributors/analysis`; domain out-of-memory error; local glyph walker and graph-vector view helpers; `planResources`/`compilationStats` replacements; kriging model type | Example tsc clean |
| KE-PORT-4 | Deck 10 host: `COORDINATE_SYSTEM` shim for Kepler's deck 9 code, Kepler luma 9 import aliases, a three-hunk deck 10 yarn patch (depth attachment, canvas-leave hover), removal of obsolete deck 9 WGSL patches | `vite build` of app and viewer |
| KE-PORT-5 | Bring-up in headless WebGPU Chromium | `scripts/smoke.mjs`, then native-layer-management, native-table, native-filter-lifecycle, native-csv, native-sql-gpu, check-native-join, check-native-group-by, layers, native-charts |
| KE-PORT-6 | Re-enable excluded kepler specs by tier (43 package-only, then gpu-layers and luma-symbol families); typecheck the example-backed specs | `yarn test`, `yarn typecheck-tests`; shrink `test/kepler-compute-example-specs.json` |
| KE-PORT-7 | Independent review of every stream, fixes, docs (README, workarounds, proposals for deck hunks and the unexported luma helpers) | Full gates |
| KE-PORT-SAMPLES | Runtime public Kepler Sample maps tab, configs and attribution; earthquakes startup/offline fallback; explicit sparse-vs-dense tree aggregation and wide-integer parcel style limitations with location fallbacks; remote completion awaits native registration, single-descriptor saved-ID binding and streaming render admission; independently probed HTTP service outages have a visible named state; cancellable sample transactions await publication and roll back data/presentation; per-dataset attribution persists; remote recovery explicitly defers without invalidating the healthy checkpoint | `sample-maps.node.spec.ts`, `resident-layers.node.spec.ts`, scoped example tsc, checkers; driver runs `scripts/kepler-demo-data.mjs` for all entries and default/offline startup, plus smoke/acceptance/gallery |

Wave 3 flow/trips/memory parity audit (2026-11-14): all 21 scoped browser scripts passed the driver baseline. Twenty scripts match the old fork exactly; `flow-field.mjs` replaces a stale unsupported-style assertion with successful rendering (the old runtime already implemented speed color/debug grid, and the separate style script checks screenshots). Published alpha.14 `FlowParticleSimulation` has no spawn-region control. The port now crops velocity/altitude textures through public core render passes and calls `setField` with a padded, source-clipped viewport domain, preserving the full zoomed particle budget on CPU and GPU grid paths. Settling takes 150 ms with 2% extent hysteresis. Compared with the fork this resamples the field and reseeds all particles after a domain change; pitched/globe views use bounding rectangles and split maps share a runtime. Node coverage and scoped typecheck pass; the new crop rendering and all 21 scripts need the driver rerun at `http://127.0.0.1:5328/?rows=256`. Trip membership and flow chunk alignment now reuse analysis's `getGraphViewChunks`. The other near matches in the reuse audit have different contracts (flow pair ordering/capacity, sampled UV/elevation fields versus terrain deflection, trajectory ingestion versus analysis, H3/heatmap rendering versus analysis); no equivalent contributor substitution was found. Baseline resident performance (100k rows, 10 iterations): submitted-frame p50 about 36 ms, p95 74 ms; these are reported numbers, not a new measurement.
Wave-3 reuse: exact-source polygon equality-token preparation and an optional CPU contiguity route
now live in `analysis/gpu-spatial-analysis`; exact Int64/Decimal128 summaries, CPU references and
Float32 narrowing diagnostics live in `analysis/dataframe-analysis/column-profile`. Integer maximum
narrowing error is an exact bigint. Kepler owns the device-wide OOM scope and CPU rate diagnostics.
The example uses `getGraphViewChunks` and `MeteredCommandGraph` from the packages. Re-enabled mock
specs reset shared module caches before imports and clean up afterward; H3/icon cold imports run
outside timed hooks. See [the extracted contracts](../docs/api/reused-analysis-preparation.md).


Wave-3 contrib-spacetime adds fourteen GPU-only catalog bindings for the existing density, Hawkes, scan, space-time inference and change-point contributors. Resident Point or numeric coordinate columns and epoch time columns map to bounded tables with source identities; summaries and UI controls expose inference metadata and limits (512 rows, 64 locations, 128 periods, 16,384 density voxels). Node mapping checks and scoped typechecks run in the implementor sandbox; the driver passed all 14 direct-contributor GPU parity specs and catalog discovery. Round 2 connects resident inputs to the existing bounded original-column reader (preserving epoch precision and guarding source revisions), while keeping non-resident inputs; the driver subsequently passed the native-dataset browser workflow. Round 3 reproduces and fixes subsecond density placement, Float32 radius-boundary inclusion and colliding resident derived aliases with node regressions; new subsecond and boundary GPU fixtures await a driver rerun.

Wave 3 raster time series: raster-native bindings in `modules/kepler/src/analysis/catalog-raster-timeseries-bindings.ts` wrap the existing time-series family, with explicit acquisition/band mapping and zonal publication in the raster tools panel. Node stack/refusal specs and driver-run headless specs plus `scripts/raster-timeseries.mjs` qualify the integration; the existing raster-tiles, raster-zonal-panel and raster-terrain-gpu scripts remain regression gates. GPU/browser execution and full-example typechecking are driver gates. Cloud-scored BAP and historical streaming beyond a loaded stack require additional input metadata/routes; see the example README for scope and evidence.

#### Kepler analysis workbench (KE-WF, in progress 2026-11-14)

A node-graph analysis workbench in the Kepler host, over one serialisable workflow document. The design study is [research/kepler-workflows-2026-11-14](./research/kepler-workflows-2026-11-14/README.md); section 9 of that file records the experiments and the decisions below. Built in a staging copy and landed after KE-PORT wave 3.

Provisional decisions taken for this build, not yet confirmed by the maintainer (each is the study's recommendation unless noted; the canvas dependency and the host-only persistence are the two that most need confirming):

- **Canvas and list over one document.** A bottom dock holds a React Flow canvas (`@xyflow/react` 12.12.1, MIT; verified under React 19 in the host), an inspector and a run log, with the conventions of noodles.gl and Houdini (Tab or N adds a step, B disables, D displays, F frames, undo by snapshot). A Workflows tab in the Kepler side panel (`CustomPanelsFactory`) is the narrow step-list view. No Kepler fork.
- **Every node is a step.** Data enters through source steps (a host dataset, all rows or the current selection, or embedded GeoJSON), not through per-input bindings as in the draft spec. Steps connect output port to input port.
- **Hand-off by table** on the CPU, within the catalog capacities, checked before a result is passed on. `WorkflowData` is the seam where a GPU-resident form can be added (KE-WF-08, not built).
- **Catalog bindings are not changed.** Each becomes a workflow operation through an adapter that derives port kinds; glue operations (join attributes, filter, calculate, select, sort, summarise, concatenate) are workflow-only because a catalog input cannot be a table.
- **Status is derived from content keys**, not stored: a step is done when it holds a result for its current key, stale when it holds one for an older key. Editing a step or a change in a host dataset, filter or selection re-keys exactly the affected steps.
- **Display publishes to the map.** A displayed output becomes a resident dataset with a layer and is replaced in place on re-run; from there the existing resident analysis panels, filters and charts apply.
- **Workflow inputs** (parameters promoted to the workflow) are in; step-output parameter binding, expressions on parameters, iterators and branches are out.
- **Persistence** is the host sandbox extension `settings.extensions.workflows` only. Kepler's own Save Map drops it.
- **Assistant** tools list and describe operations and propose a draft; nothing runs before the user approves the whole workflow.
- **Templates** read embedded synthetic geometry only, so no third-party licence applies.

| ID | Work | Where | Gate |
| --- | --- | --- | --- |
| KE-WF-A | Contract: types, port kinds, host interface, UI vocabulary | `modules/kepler/src/workflows/workflow-types.ts`, `examples/deck/kepler-compute/src/workflow/workbench-contract.ts` | Scoped typecheck |
| KE-WF-B | Data hand-off, content hashes, expression language, strict document parser | `workflow-data.ts`, `workflow-expression.ts`, `workflow-document.ts` | Node specs |
| KE-WF-C | Operations: catalog adapter, registry, search, sources, table glue | `workflow-operations.ts`, `workflow-source-operations.ts`, `workflow-table-operations.ts` | Node specs; every registered binding adapts and its sample chain runs on the CPU engine |
| KE-WF-D | Plan (order, issues, keys), edit operations, layout, undo history | `workflow-plan.ts`, `workflow-edit.ts`, `workflow-history.ts` | Node specs |
| KE-WF-E | Runner: derived status, node cache, capacity gate, cancel, publish, run record | `workflow-runner.ts` | Node specs: an upstream edit marks dependents stale and reuses unaffected results |
| KE-WF-F | Templates, JSON and TypeScript export, history promotion, assistant tools | `workflow-templates.ts`, `workflow-export.ts`, `workflow-assistant.ts` | Every template runs to done on the CPU engine in a node spec |
| KE-WF-G | Host: store, Kepler host adapter (native and resident datasets in, published datasets out), sandbox extension | `examples/deck/kepler-compute/src/workflow/workbench-store.ts`, `workbench-host.ts`, `workbench-extension.ts` | Node specs with a fake host; headless save and replay round trip |
| KE-WF-H | Workbench UI: dock, toolbar, canvas, node, palette, inspector, parameter form, result view, run log, side panel, dialogs | `examples/deck/kepler-compute/src/workflow/workbench-*.tsx` | Static-render node specs; headless acceptance scripts |
| KE-WF-I | Acceptance in headless WebGPU Chromium: build a chain from the palette, run, display on the map, edit and see stale, undo, template, save and replay, keyboard only | `examples/deck/kepler-compute/scripts/workflow-*.mjs` | Scripts pass; screenshots reviewed |
| KE-WF-J | Independent review of every stream, fixes, docs | all of the above | Full gates |

| KE-WF-K | Second wave: colour-by display style, SQL step on a workbench-owned DuckDB instance, drawn-shapes source, user templates, inlining a workflow into another, copy and paste, suggested fixes, run report and DOT export, saved workflows as assistant tools | `workflow-sql-operation.ts`, `workflow-suggestions.ts`, `workflow-report.ts`, additions across both directories | Node specs; browser probes of each feature |
| KE-WF-L | Third wave: dataset-aware recommendations, swapping a step's source without rewiring, zoom to a displayed output | `workflow-recommendations.ts`, `replaceWorkflowStepOperation` | Node specs |

Not built, in order of value: a GPU-resident hand-off between adjacent steps with a fusion planner (KE-WF-08 of the study); resident recipes (hot spots, kriging, OLS on resident columns) as step kinds; composition by reference (a workflow used as an operation of another, so edits propagate); share links and the standalone viewer; a map preview of a step that is not displayed.

#### Analysis experience (AX, implementation pass landed 2026-11-15)

The workbench question gallery consumes the AX library and maps curated templates to user datasets. All 28 curated records (the original 27 plus drive-time bands) have registered bindings, descriptors and licensed synthetic samples. Registry and CPU checks do not close the headless WebGPU, browser journey or real-sample gates. The review of the existing state, the research on similar tools, the design, the gate for each row, the measured baseline and eleven maintainer decisions are in [research/analysis-workflow-ux-2026-11-14](./research/analysis-workflow-ux-2026-11-14/README.md).

The shared metadata is plain CPU TypeScript; execution bindings select CPU or WebGPU:

- `modules/analysis/src/gpu-spatial-analysis/ontology/analysis-experience/`: twelve questions and operator-to-question rules, the data-kind vocabulary with supplements for operators whose kinds are empty, measurement-level guesses and field-role matching, the requirement evaluator (passed, failed or unknown, with its evidence), spacing statistics and default rules, class schemes, result sentences and cautions. Documented in `docs/api/gpu-spatial-analysis/analysis-experience.md`.
- `modules/kepler/src/profile/`: dataset profile per revision with declarations that replace guesses.
- `modules/kepler/src/descriptors/`: hand-written descriptors for all 217 catalog bindings, resolved defaults, applicability, `suggestKeplerAnalyses`, and adapters in the shapes the KE-WF workbench reads. The registry spec permits no derived-descriptor allowance.
- `modules/kepler/src/presentation/`: scale, classes, legend, sentence, chosen-parameter report and cautions for a catalog result.
- `modules/kepler/src/curated/`: 28 question-named analyses (the original 27 plus drive-time bands), with registered bindings, descriptors, licensed synthetic samples and AX-30 templates. Sixteen have CPU paths; twelve require WebGPU.
- `examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs` and its recorded baseline.

Checked by three independent review rounds with probes (eight reviewers, then seven, then three verifiers) and four fix rounds; 30 spec files, about 2,490 tests. Defects the reviews found in existing catalog bindings are in [binding-defects.md](./research/analysis-workflow-ux-2026-11-14/binding-defects.md); the bindings were fixed on 2026-11-15 and the workarounds that had become wrong were removed with them (its status section lists both).

Known limits of the layer: the rate cluster map recovered a lone planted outlier in 12 of 20 runs; the default clustering radius fragments irregular concentrations and merges uniformly repeated locations; a requirement that depends on an estimated distinct count is always `unknown`; edge and node capacity cannot be exact from a profile; the measurements behind these were made before the binding fixes of 2026-11-15 and have not been repeated.

The KE-WF workbench (workflow document and parser, runner, palette, assistant, result view; `modules/kepler/src/workflows/` and `examples/deck/kepler-compute/src/workflow/`) landed on 2026-11-15. It consumes AX descriptors, templates and native sample sources. The remaining GPU-resident hand-off and browser qualification are AX-50, AX-60 and AX-70 below.

The design, research, gates and maintainer decisions are in [research/analysis-workflow-ux-2026-11-14](./research/analysis-workflow-ux-2026-11-14/README.md). The library layer landed on 2026-11-14; the implementation pass (AX-PASS) of 2026-11-15 wired it into the Kepler host and the KE-WF workbench and closed most rows.

How AX-PASS was run: three read-only Codex `gpt-6.1-sol` audits of the tree first (they found AX-02's resident catalog input already wired and AX-30, AX-50, AX-51, AX-60 and AX-61 partly built in the workbench), then 58 planned streams plus 6 integration streams (curated registration, four whole-area review fixes, browser fixes), each in its own clone of an integration copy, verified by scoped typecheck, biome, its node specs, structure specs and both checkers, reviewed once or twice by an independent read-only Codex reviewer, and landed in dependency order. Headless WebGPU specs and host browser scripts were run by the lead on the integration copy. A whole-diff review in four areas (host, Kepler AX library, workflows, analysis and website) found about 20 cross-stream defects; all but the website pictures were fixed.

Provisional decisions taken (the research README's recommended defaults unless noted, not yet confirmed by the maintainer): vocabulary and rules in analysis, adapters in kepler, UI in the host; converge onto the generated sheet one operation at a time, keeping bespoke result views; correctable guesses that warn and never block (an unknown row meaning is a warning with a one-click answer); twelve wave-1 analyses plus a drive-time bands analysis, with the reachable-demand analysis kept under its truthful name; synthetic licence-clean samples only; the existing host and workbench assistant route; no significance classes from sampled previews; the twelfth question `what-is-at-each-place` kept; descriptor-level `apply: live | rebuild`.

Only unmet AX acceptance gates remain below. Implemented defaults, applicability, suggestions, presentation, templates, sensitivity and named weights are covered by the existing descriptor, curated-template and workflow specs; the completion record is below. These source/spec checks do not replace browser or GPU qualification.

| ID | Still open | Evidence / acceptance |
| --- | --- | --- |
| AX-01 | Qualify the two-action form journey in the host. | Shared dataset-revision seeding exists in `analysis-form-initialization.ts`; rerun `ax-form-seeding.mjs` and review the interaction. |
| AX-04 | Independently review result screenshots. | Catalog and workbench use `encodeKeplerResultPresentation`; review table, planar and geographic outputs. |
| AX-20 | Qualify staged-profile latency on 1,000,000 points and 8 columns. | Use `modules/kepler/test/profile/dataset-profile-gpu.spec.ts` on AX-70's reference configuration: metadata suggestions in the same frame as the dataset card, column statistics within 1 s, spacing within 10 s with progress. The small `ax-profile-qualification.mjs` resident-CSV/declaration journey does not close this timing gate. |
| AX-31 | Qualify licensed real-world samples. | All 28 curated analyses have registered bindings and synthetic samples (`curated-template-gates.node.spec.ts`); real-data admission needs the licence decision. |
| AX-33 | Capture chain-intermediate website pictures. | Catalog questions/kinds and curated workflow pages exist; the capture script supports drainage only and the broader capture gate still fails. |
| AX-40 | Qualify the 500 ms live map-change gate for every wave-one template, including kernel density. | A live parameter change must produce the corresponding map update within 500 ms on each template's sample and AX-70's reference configuration. A fast table preview alone leaves the gate open: `ax-analysis-sheet-qualification.mjs` records null map latency and `qualified: false` for all candidates, including kernel density. |
| AX-50 | GPU-resident hand-off between workflow steps. | Lifecycle, estimates and records exist; KE-WF-08 is not built. |
| AX-60 | Complete workbench browser qualification. | Palette, defaults and templates are wired; rerun `ax-workbench-qualification.mjs` and resolve its null publication-state observation described below. |
| AX-70 | Rerun release journeys and resolve catalog/workbench browser failures. | `ax-journey-baseline.mjs` was not rerun on this tree; preserve fixture hashes and review the failures below. |

Host browser scripts on the final tree (headless WebGPU Chromium, Vite dev server, 2026-11-15): 20 of 23 pass, among them the curated gallery, form seeding, profile declarations, result publication, assistant approval, keyboard and palette, save extensions and the existing kriging, cartogram, polygon, projected and published analysis journeys. Still failing: `ax-analysis-sheet-qualification.mjs` (deliberately, until the AX-40 map-latency gate is qualified; only kernel density has a retained preview executor), `analysis-catalog.mjs` (Run disabled at a later step of the catalog journey) and `ax-workbench-qualification.mjs` (the script reads a null publication state; node regressions show AX class scales survive publication). `ax-journey-baseline.mjs` (AX-70) was not rerun.

Source-stream first-round audit: [upload/source parity, script fixes and pending browser qualification](./research/kepler-source-parity-2026-11-14.md). All eight fixture module graphs build; source/resume/rollback Node checks pass. The 17 Playwright workflows and six native demo uploads still need the driver run; this is not an all-Kepler parity claim. Round 2: the driver subsequently confirmed all 17 scripts pass (native-csv retried once after a cold Vite start). Independent review found a globe-label hook ordering defect; the follow-up preserves inherited injection order and runs culling last, verified by final GLSL/WGSL assembly tests in the same audit.

### Lenses, recording and CPU operation forms

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| LN-REC | Finish recording coverage and verify captions | Dataframe/S2 harness timings, published primitive shader facts, 135 numeric captions and nine last-digit changes remain unchecked; formulas, loop provenance and derivedFrom remain pack-level. Timing-bearing records are not byte-reproducible. | M | [workarounds]; `modules/analysis/src/computation-record/` |
| LN-SEL | GPU selection backend | `GPUCrossfilterSelectionBackend` exists (`modules/lenses/src/computation-views/coordinator/gpu-crossfilter-selection-backend.ts`). Verify it against published `GPUCrossfilter`, which lacks external masks (see the workaround register) | M | [workarounds] |
| LN-DQ | Rebuild dataframe-query and formula scenarios on public query metadata | 8 scenarios dropped | M | [workarounds] |
| LN-STAB | StabilityMap and Purpose rung | Not built (also not on the branch) | L | [coverage] |
| LN-V5 | `GPURecipePlan`, `ComputationDrawer`, V6 to V8 phases, space-time cube view of solver wavefronts (a visualization of the archive, not a data cube) | Never built | L | [coverage] |
| LN-ENC | Encoder labels and rendering/trace packs | Requires published luma.gl support. | L | [proposals] |
| LN-ST | Spatiotemporal lenses | Spatiotemporal entries have documentation slots but no pack operator, formula, scenario or record (85 in the recorded inventory). | M | [spatiotemporal]; `modules/lenses/test/computation-docs/lens-coverage.node.spec.ts`: `SPATIOTEMPORAL_REVIEW_PENDING` |
| INT-01 / LN-REC2 | Recapture lens records | Lag-ML and ring-assembly values may predate final fixes; published-gather adoption removes local kernel text. Capture with `VITE_CAPTURE_LENS_RECORDS=1`. | M | `modules/lenses/`; [usage] |
| INT-06 | Fix unpublished analysis subpath import | The local-permutation lens resolves through tsconfig paths rather than package exports. | L | `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/local-permutation-test.lens.ts` |
| CPU-03 | Remaining twin-useful contributors and findings | 75 unselected survey rows; additional twins only on consumer demand. Triage the existing-contributor leads before acting; most lack GPU verification. | L | [cpu]; [existing findings](./research/cpu-suitability-2026-11-11/existing-contributor-findings.md), [wave 2](./research/cpu-suitability-2026-11-11/existing-contributor-findings-wave-2.md) |
| CPU-04 | Recipes as mixed GPUGraph programs | Population support alignment is done. Segregation counterfactual and building tessellation remain; retire the deprecated dense resistance recipe for the float64 solve. | M | [cpu] |
| CPU-05 / INT-03 | Expose twins in catalog and website | Add cpuTwin metadata; identify operation forms by class rather than filename in inventory extraction. | M | [cpu]; `website/scripts/extract-inventory.ts` |
| CPU-08 | Measured crossover and optional automatic choice | Wait for the packed CPU/GPU boundary fast path, UPS-CPU-1. | L | [cpu] |
| UPS-CPU-1 | Upstream packed CPU-stage transfer | Row-by-row read/upload costs about 11 ns/byte; propose a packed fast path when stride equals row size. | H (upstream) | [proposals]; [cpu] |
| UPS-CPU-2 | Upstream textures and state vectors across CPU stages | Needed for raster/terrain operation forms that use those resources. | M (upstream) | [proposals] |
| UPS-CPU-3 | Upstream GPU submission coalescing between CPU stages | Consecutive GPU operations should share a submission. | L (upstream) | [proposals] |
| UPS-CPU-4 | Upstream operation contract for command-node producers | Would retire the local GPUContributorOperation bridge. | M (upstream) | [proposals]; [workarounds] |

### Website pictures and interaction

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| WEB-INT-08 | Touch steering on function hero | Cards navigate on tap; hero steering is unbuilt. | M | `website/src/gpu/canvas-pointer.ts` |
| WEB-INT-09 | Shared-device latency experiment | One device per live card; measured median start 668 ms. Adopt sharing only below about 250 ms without device-loss/validation regressions. | M | [usage] |
| WEB-INT-13 | Restore pointer state after hero restart | Restarts hide pointer-written handles/readouts until movement; remember pointer position. | L | `website/src/gpu/runs/define-run.ts`: `isUntouched`; `website/src/gpu/gpu-canvas.tsx` |
| WEB-INT-14 | Keyboard path to hero gesture | Arrows steer cards; hero canvas is not focusable, leaving sliders as its keyboard path. | M | `website/src/gpu/canvas-pointer.ts` |
| WEB-INT-15 | Measure pointer-to-frame latency | Verifier checks change and compilation, not latency; set the target from measurements. | L | `website/scripts/verify-runs.mjs` |
| WEB-INT-16 / WEB-COV-04 | Finish labels and text support | Label study-window, standard errors, probit marginal effects and SUR residuals; add ^, apostrophe, superscript two, glyph colour and log axes. Check hint overlap, hero poster/live text scale and 111 px cards. Font licensing needs a decision. | M | `website/src/gpu/runs/text/`; `THIRD-PARTY-DATA.md` |
| WEB-INT-17 | Layer highlight option | Probes use local glyph/mask helpers; shared id/value/mask highlighting could replace about twelve helpers. | L | `website/src/gpu/runs/` |
| WEB-INT-18 | Computation-run helper | Rebuild layout is unstable, readbacks are uncached and resolve lacks extra data. | L | `website/src/gpu/runs/record-demo.ts` |
| WEB-PIC-09 | Suitable real extracts | Need water/post-fire imagery, lidar with intermediate relief and related real point sets for pair-gather. Register source/licence for each asset. | M | `website/build/demo-data-sources.ts`; `THIRD-PARTY-DATA.md` |
| WEB-PIC-12 | Rework weak runs; re-score the whole site | Baseline: 149 rest pictures and 120 gestures score 1–2. 83 runs have rework; remaining weak runs and about 20 weak gestures within that set need attention. Confirm suspected operation defects in `network-point-routing`, `terrain-dem-coregistration` and `multiscale-gwr`. Score both themes and all pointer positions. | H | [pictures] |
| WEB-PIC-13 | Inert sliders and spike-repair question | pair-gather how changes only 0.46%; lidar/elevation tiles cannot exercise five removed sliders. Investigate the damaged block visible after spike repair. | M | [pictures]; WEB-PIC-09 |
| WEB-PIC-14 | Weak gestures | About 40 stay below 3%; statistical-filter gives two pictures. Whole-group and light-theme verification remain; regenerate any changed rest poster. | M | [pictures] |
| WEB-PIC-15 | Differentiate siblings | Variogram trio, encounter/zone-event trios and measures cards still share layouts. | L | [pictures] |
| WEB-PIC-16 | Finish partly effective pictures | Flat resistance potential, modest branching flow, bare envelope curves, small Max-P bars, weak Markov changes and outline-only arc overlay. | L | [pictures] |
| WEB-PIC-17 | Confirm remaining picture/input question | Some quad zones receive no points and produce NaN quantiles; avoid invalid wall-adjacent viewpoints in visibility pictures. | M | [pictures] |
| WEB-PIC-18 | Review constructed scenarios | Six modeled shapes and simulated rate noise are labelled; assess whether real extracts can serve. | L | [pictures] |
| WEB-COV-01 | Position raster/chart panels | Four local panel helpers should share one layer option. | M | `website/src/gpu/runs/entries/` |
| WEB-COV-02 | Per-segment line values | At least ten runs read values to CPU only to colour segments. | M | `website/src/gpu/runs/entries/` |
| WEB-COV-03 | Shared helper copies | Consolidate permutation-chart/kernel copies and the cross-group inset dependency. | L | `website/src/gpu/runs/entries/space-time-events/_lib/` |
| LUM-27 | Keep display reshaping on GPU; assess shared devices | 85 afterRun paths read, reshape and re-upload only to draw; shared-device loss affects all cards. Regenerate posters after changes. | M | [usage] |
| LUM-34 | Capture settlement policy | settle warns after 400 frames with pending work; decide whether capture should fail. | L | [usage] |
| LUM-35 | Batch remaining readbacks | 48 website entries and six kepler files still allocate/submit per readAsync call. | M | [usage] |

### Website explanation pages

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| WEB-HIW-01 | Show timings, WGSL and ownership | Dataframe/S2 records lack timings; two terrain records lack source. Pages omit captured timings/source; transient aliasing needs an upstream allocation plan. | M | `website/src/components/meta/pages/`; [workarounds] |
| WEB-HIW-02 | Raster snapshots in view models | 20/410 models have a raster map although 322 captures exist; supply origin and cellSize. | M | `website/src/lens/extract-lens-view-model.ts` |
| WEB-HIW-03 | Workflow intermediates | Pictures are independent operation runs; the drainage record is 3×3. Use actual chain outputs. | M | `website/src/components/meta/pages/workflows/` |
| WEB-HIW-04 | Reliable building-block usage counts | Primitive kernels lack operator names; node-id matching can undercount reductions. | L | `website/src/components/meta/pages/building-blocks/` |
| WEB-HIW-05 | Phone text sizes | Some SVG labels are 8–9.5 px at 390 px; wide figures scroll internally. | L | `website/src/components/meta/` |
| WEB-HIW-06 | Live index cards | Five new-page thumbnails are static. | L | `website/src/components/meta/` |

### Spatiotemporal and platform

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| STX-08 | Additional REDCAP linkage/constraints, LICD and colocation inference | `GPURedcapSingleLinkage` builds the first-order single-linkage tree but cuts its heaviest edges, not the SSD partition (six-node example: SSD 13.26 versus 5.85). For SSD cuts compose `GPUSpatialWeightsMinimumSpanningTree` with `GPUSkaterRegions`, whose `minimumRegionSize` row-count floor is already enforced on both sides of a cut (including the size-two fixture). The missing size floor applies to the direct contributor `GPURedcapSingleLinkage`; full-order/average/complete linkage also remains open. LICD is local composition only and colocation quotient lacks permutation inference. | L | [spatiotemporal]; `gpu-spatial-analysis/redcap-regions/gpu-redcap-single-linkage.ts`; `gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions.ts` |
| STM-07 | External Vecchia validation | GpGp GPU comparison/fixtures pending; spatial Fisher scoring exists, space-time fit uses Nelder–Mead. Float32 evidence assumes nugget ≥1e-3 of sill except exponential. | M | [spatiotemporal] |
| MOB-03 / FX-08 | Point-centred cleaning default | Current speedOutlierMode default is step; point-centred is opt-in. TD-TR word times require the track-relative conversion. | H | [spatiotemporal] |
| MOB-07 | Dynamic Brownian bridge change points | dBBMM change-point step absent. | L | [spatiotemporal] |
| MOB-10 | Radiation scale and flow-smoothing definition | No scalable radiation variant; smoothing formula needs paper verification; 1,024-zone cap. | L | [spatiotemporal] |
| RTS-08 | A5 and conformant DGGS-JSON | A5 not started; ~3,100 reference lines need fixed-kernel boundary redesign. Zone IDs are not OGC-conformant. | L | [spatiotemporal] |
| PLAT-03 | Shared neighbour traversal and candidate semantics | Fixed, adaptive and space-time KDE, DBSCAN, marked point statistics and cross variograms already use the shared point-neighbour grid builder. Share the remaining walks only where eligibility, distance/tie order and overflow agree; builder adoption alone does not close traversal work. K-means scans at most 256 centres. HDBSCAN's sample-cap decision is DEC-HDBSCAN. | M | [spatiotemporal]; GPU-05; `analysis-core/kernels/point-neighbor-grid-index.ts` |
| PLAT-04 | Reusable GPU-gated iteration batches upstream | `createNetworkCentralityContinuation` already runs bounded graphs with persistent frontier storage across host-controlled submissions, beyond the direct contributor's 1,024-round limit. The open proposal is reusable GPU-gated batches that avoid unrolling the entire worst-case budget, with one convergence readback per batch; the host chooses the next batch size. Convergence in the second batch must stop further work while preserving the first batch's state. | M | [proposals]; `gpu-network/network-centrality/network-centrality-continuation.ts` |
| PLAT-05 | Adapter-maximum shorthand and buffer_view | Explicit requiredLimits and subgroups are already public; only remaining gaps belong upstream. | L | [proposals] |
| PLAT-06 | Qualify remaining external oracle comparisons | Inventory each missing comparison by generator, reference version and consuming spec before assigning it. GeoConformal (STM-03), CAST (STM-05), ETAS (STM-08) and GpGp (STM-07) remain explicit gaps; phenofit/GRASS, bfast and xeofs are not qualified by the NumPy definition fixtures. pyxccd 1.1.0, track-similarity and Markov fixtures already exist. Fixture presence alone does not prove every requested external comparison or GPU case ran. | M | [spatiotemporal]; `modules/analysis/test/oracles/GENERATORS.md`; `scripts/oracle-fixtures.json` |
| PLAT-07 | SpatialBench extensions | One synthetic-zone query measured to SF5: 216.3 ms. SF10 JS arrays crash; add typed arrays, real Overture zones, aggregation-only and KNN. SF100 needs 9.6 GB positions. | L | [spatiotemporal]; `modules/analysis/test/gpu-spatial-analysis/spatial-join/spatialbench-scale-benchmark.spec.ts` |

### Contributor contracts and GPU-scale designs

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| STD-04 | Named status ports and flag vocabulary | Catchment, tiled flow, network facility/flow/time-routing/rational-access/cost-table/service-area, local Moran, hot spot and Local Geary ports exist. Per-cell codes, fractional-zonal distribution, geometry overflow beyond hull/bounds, remaining network and most other statistics remain. Decide output.ports.status on three packed-status classes. | H | [library] |
| STD-05 | Runtime numeric parameters | Point-cloud ICP thresholds/viewpoint, hot-spot self weights, lattice cell sizes, kernel bandwidths, LOSH powers, spatial-weights summary and transpose tolerances, multivariate Geary seeds/significance and local-kriging covariance are runtime. Point-cloud neighbourhood/height/surface settings, remaining statistics props and raster affines stay baked. Three parameter mechanisms coexist. | M | [library] |
| STD-06 | Raster context | A shared context resolver, per-frame reprojection affines and patch area/perimeter exist. Contour/flow extent and row direction, and pixel coordinate declarations remain. | H | [library]; `analysis-core/contracts/raster-grid.ts` |
| STD-07 | Published band validity and final vocabulary | Reclassify, conditional, stretch, arithmetic, sampling, profiles, cell statistics, weighted overlay, isolines and isobands accept published bands. Other bare raster value views remain; spectral outputValidity and distance-field seedMask need decisions. Preserve Hampel/outlier flags meaning 1=bad. | H | [library] |
| STD-08 | Remaining geometry layout/provenance | Line-chunk requiredCount is exact and saturating; hull/bounds overflow is boolean with named diagnostics. Offset curve and outline use fixed stride; overlay/buffer lack per-feature source rows. | M | [library] |
| STD-09 | Finish core ownership | Move shared utilities/raster iteration gate to core; row tiles and list ranking only when another family needs them. Decide whether validateSinglePackedView is public. | L | [library]; `analysis-core/`; `gpu-spatial-analysis/shared/` |
| STD-10 | Adopt shared WGSL without moving fixtures | Kriging, RBF, space-time kriging and Vecchia use the shared LU solve; point-cloud features and ICP use the shared Jacobi. Remaining: panel-lag determinant/inverse, eigen range, Kalman, ~140 binary searches, word-order-dependent uint64, masks/accessors, QR, deprecated float tests and kernel wrappers. | M | [library]; `analysis-core/kernels/` |
| STD-11 | Recipe consistency: implementation form and result compatibility | Utility modules are merged and interpolation-diagnostic/network-scenario summaries are parallel; kernel-only migrations remain a separate implementation task. Population support alignment has a contributor and mixed-graph operations, plus canonical outputs/intermediates/status.stages and explicit packedStatus. Generic result acceptance must independently verify every public overload and optional-stage branch; a contributor migration alone does not close it. | M | [library]; [recipe contract](./contributor-contract.md#1-shape) |
| STD-12 | Source/test/docs layout | Loose roots, multi-class files, old test names and split API pages remain; geospatial entry needs a coherent scope. | L | [library] |
| STD-13 | Registry consistency | Rename 14 wave-named capability IDs; check coverage/docs/site against registry, correct kepler Dorling limits and group API docs by domain. | M | [library]; `ontology/all-entry-points/`; `gpu-spatial-analysis/ontology/kepler-analysis-catalog.ts` |
| STD-14 | Remaining packed-input and scratch limits | Point-in-polygon joins, cell aggregation and point/kernel density already accept chunked input; cross-chunk identity and several paged scratch routes exist. Re-census the remaining packed-only contributors and row-proportional scratch by route, recording binding preflight and a failing over-binding acceptance case for each gap; the old 294/11 census predates SC/IMP. | M | [library]; `analysis-core/kernels/graph-chunk-dispatch.ts`; `gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join.ts` |
| STD-15 | Statistic output vocabulary | Decide minimums/minimum, counts/cellCounts and validCount/count. | L | [library] |
| GPU-01 | Remaining serial data loops | LOSH means and the default inequality statistics/Lorenz route are parallel. Masked registration means/refinement, fractional tuples, adjacency union-find, structures pack, quantile mapping, Ang shell and fold designs remain to audit by route. | M | [library] |
| GPU-05 | Index by default | Marked point statistics and cross variograms index by default and great-circle queries outside bounds are handled. Large-radius work remains; snapping callers lack candidateCapacity. The point-cloud grid stays local: cell-size arithmetic, upper-bound inclusion, ordering and raster mode differ from published `GPUGridIndex` (`pointcloud-analysis/operations/pointcloud-grid-index.ts`). | H | [library] |
| GPU-09 | Serial regression fits | MGWR global fit, panel means and SUR weight validation are tiled. The panel-lag dense determinant remains. | M | [library]; `gpu-spatial-analysis/spatial-regression/regression-row-tiles.ts` |
| GPU-10 | Facility-solver consolidation and time-routing scale | Point routing defaults to scalable CSR with same-edge costs and reconstructable predecessors; parallel max flow and opt-in primal-dual min-cost flow exist. Consolidate the bounded facility solvers and qualify larger time-routing problems; retain exhaustion and predecessor semantics. Max-flow path decomposition is NW02. | M | [library]; `gpu-network/network-optimization/gpu-network-point-routing.ts`; `gpu-network/network-optimization/gpu-network-flow.ts` |
| GPU-11 | Remaining urban/landscape route and budget restrictions | Landscape metrics use raster patches with parallel seam reconciliation; avoid patches×cells proximity buffers. Local tessellation already supports convex and simple holed enclosures up to its 32-vertex/32-ring bounds; dissolved-ring overlay remains on the 64-building reference route. Network CSR visibility already publishes integration and normalized choice in all-source or sampled modes. Qualify larger enclosure/clipping capacities and visibility command/frontier budgets; sampling remains an explicit approximation, not a missing metric. | M | [library]; `gpu-spatial-analysis/urban-morphology/building-tessellation-local.ts`; `gpu-network/visibility-graph/visibility-graph.ts` |
| GPU-12 | Geometry algorithms | Level-synchronous S2 cannot be the default until budget-exhaustion behaviour matches; rework arc overlay, Delaunay flips, offset components, ratio hull, segment snap and shape moments (parallel enumeration must keep the tested descending-offset order). | M | [library] |
| GPU-13 | Remaining raster/dataframe output and count limits | Fixed-linkage region graphs already use Borůvka; dense public matrices remain device-binding bounded and adaptive linkage is capped at 256 regions. `GPUClassBreaks` still rejects more than 2^24 rows, even for quantile classification (16,777,216 accepted; 16,777,217 rejected): lift float32 moment-count/rank limits with exact boundary acceptance cases. `GPUColumnQuantiles`' default float32-probability route is also capped at 2^24; `rationalProbabilities` already supports 2^31 - 1 rows across chunks when no percentile filter output is requested. Filter ranks remain float32 on both routes. Sparse fractional bounds candidates, exact skewed group moments and parallel inequality already ship. | M | [library]; `raster-analysis/remote-sensing/gpu-raster-region-graph.ts`; `dataframe-analysis/column-classification/gpu-class-breaks.ts`; `dataframe-analysis/column-classification/gpu-column-quantiles.ts` |
| DUP-01 | Consolidate overlapping contributors | GPUCrossVariogram forwards to the shared indexed variogram kernels. Merging GPUKriging into GPULocalKriging needs a compatibility design (duplicates, exact hits, distance models); facility solvers, walking/cost distance, cells, arcs/segmentize, label/MIC, equality/predicates and resistance recipes remain. | M | [library] |
| DUP-03 | S2 aggregation and broader containment/outline support | S2 compaction and exact equal-level grid paths/distances support levels through 30; polygon cover supports center containment through level 15; outlines support same-level S2 cells. Capability types now distinguish aggregation (`GPUCellAggregationFamily`: H3/Quadbin), topology (`GPUCellTopologyFamily`: adds S2) and indexing (`GPUCellIndexFamily`). S2 aggregation, rollup and pyramid, full/intersects cover and mixed-level outline reconciliation remain to qualify independently. | M | [library]; `gpu-spatial-analysis/cell-topology/gpu-cell-compaction.ts`; `gpu-spatial-analysis/cell-cover/gpu-cell-cover.ts`; `gpu-spatial-analysis/cell-set-outline/gpu-cell-set-outline.ts`; `gpu-spatial-analysis/cell-topology/gpu-cell-grid-path.ts` |
| UPS-01 | Upstream graph algorithms on command-graph views | Then retire local topology, components and degree. | M | [proposals]; [library] |
| UPS-02 | Offer generic algorithms upstream | Image morphology/phase correlation/GLCM, graph reachability/Brandes/SCC/flow and exact quantiles. | L | [proposals]; [library] |
| UPS-03 | Adopt published raster operators where contracts match | Band math, contrast and neighbourhood operators still have registered gaps; state which global operators cannot tile with a halo. | M | [workarounds] |
| GAP-01 | Geometry capabilities on consumer demand | Attribute-preserving overlay, per-feature buffer, overlapping dissolve, topology-preserving simplification, Voronoi polygons, splitting, antimeridian cutting and points along lines. | M | [library] |
| GAP-02 | Raster/terrain capabilities on consumer demand | QA masks, polygon clip, line/point rasterization, polygonize, zonal histogram/median/majority, pour-point snapping, stream orders and whole-DEM breach. | M | [library] |
| GAP-03 | Network capabilities on consumer demand | Edge demand, nearest POIs, street statistics, simplification and angular measures. | M | [library] |
| GAP-04 | Statistics/dataframe capabilities on consumer demand | Durbin/SLX/SDEM, GWR prediction, multi-attribute clustering, SARAR, windows, group regression and as-of joins. | L | [library] |
| GAP-05 | Point-cloud/applied capabilities on consumer demand | Voxel-centroid producer (the fork voxel downsample wrote centroids and attribute means), k-nearest >64, cloth ground filter, raster-scale FRAGSTATS, pairwise Circuitscape and least-cost corridors. | L | [library] |
| LUM-21 | Dependency-carrying helpers as shader modules | About 30 kernels and 16 concatenation-order rules; assembly costs 5–17 ms/kernel. Do not convert 140 single-function snippets indiscriminately. | M | [usage] |
| LUM-28 | Budgeted execution for long host graphs | createExecution in kepler/lenses is an unmeasured lead. | L | [usage] |
| LUM-30 | Broader resource leak coverage | Analysis spacing already snapshots active buffers and checks the count after graph and caller-buffer teardown. Extend that lifecycle fixture to other resource-owning contributors, repeated execution and textures. Read the device's `GPU Resource Counts` (`Buffers Active`/`Textures Active`); the published helper's `Texture2Ds Active` counter is insufficient. | L | [usage]; `modules/analysis/test/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing.spec.ts` |

### Precision

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| FP64-02 / LUM-20 | Published fp64 in point-to-cell | Integer variant has equal measured accuracy but is about 3× slower than local double-float; wait for FP64-05 or public vendor helpers. | M | [usage]; `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts` |
| FP64-03 | Optional fp64 for precision-limited parallel work | Cell measures accept `precision: 'fp64'`. S2 levels 26–30 outside measures, antipodal geodesics and spherical containment tolerance remain; CPU twins are oracles. | M | [cpu] |
| FP64-05 | Measure on a non-Apple GPU | Compare classic double-single with integer-controlled arithmetic before broader adoption. | M | [cpu]; [usage] |

### New contributors: unexecuted options

The maintainer chooses the directions and order. Proposed sequence: foundations LRN-02, the LRN-01 forest layout and second-device probes; then LRN-01, EMB-01–05, SIM-01, CART-03, CART-06 and CART-07; then LRN-03, ENC-01, SIM-02/03, CART-01 and LRN-04/05; the rest by demand. ENC-02/03 require a consumer; ENC-04 is an example-only spike.

Planning limits: at 0.1 trillion multiply-accumulates/s, fitted/per-point networks use ≤100,000 parameters and ≤5 billion multiply-accumulates/run; WGSL pretrained models use ≤2 million parameters (8 MB float32). Enforce named limits at construction, keep each binding ≤128 MiB unless checked against the device, and keep runs to tens of dispatches. Use minibatches: the 262,144-row probe needed a 281 MB binding. Second-device, real unbalanced forest, hash-grid, attention, float16 and command-graph overhead measurements remain open.

Fits publish loss history, converged and iterationLimitReached, including the iterate at the limit; use fixed-order tiles or fixed-point sums for repeatability. Models and weights are caller-supplied; no bundled weights/datasets. Record example data licences. Use fused analysis kernels, shared fitting helpers in core, and propose generic layers upstream. Proposed locations: models in dataframe analysis, fields beside interpolation, cartograms beside Dorling and CLAHE beside stretch. Verify each decoder formula and independent oracle before implementation; check required oracle availability rather than relying on a machine inventory.

Keep the research’s not-recommended candidates out of implementation; its reasons and the measurements supporting these options remain in [the research](./research/new-contributors-2026-11-11/README.md).

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| LRN-03 | `GPUNeuralField`: a continuous surface through points or a raster for interpolation at scale, gap filling and compression; Fourier features first, a hash grid with fixed-point scatter-add second; two dimensions and space-time. Documented as not more accurate than kriging and without uncertainty. Oracle: a PyTorch script | Not implemented; effort L. | M | [new] |
| LRN-04 | `GPUGaussianMixture`: expectation-maximization with full covariances and BIC. Oracle: scikit-learn | Not implemented; effort M. | M | [new] |
| LRN-05 | `GPUSelfOrganizingMap` with a GeoSOM mode | Not implemented; effort S to M. | M | [new] |
| LRN-06 | `GPUPermutationImportance`, optionally in spatial blocks, for any contributor with a predict kernel | Not implemented; effort S. | M | [new] |
| LRN-07 | `GPUGradientBoostedTrees`: histogram boosting trained in the browser (squared, logistic and quantile loss) | Not implemented; effort L. | L | [new] |
| LRN-08 | `GPUTreeShap`: contribution maps per covariate | Not implemented; effort M. | L | [new] |
| LRN-09 | An ensemble option on the fitted networks whose spread feeds the existing conformal classes | Not implemented; effort S. | L | [new] |
| EMB-01 | Presto (uint16) embedding decoder | TESSERA int8 and float16 decoders and the int8 encoder exist (`GPURasterEmbeddingDequantize`, `GPURasterEmbeddingQuantize`); Presto needs a verified scaling formula (the published documentation states uint16 storage only) | M | [new] |
| EMB-03 | `GPURasterEmbeddingTrajectory`: cosine to a base year, year of largest step, cumulative angle across all years | Not implemented; effort S. | M | [new] |
| EMB-04 | `GPURasterEmbeddingAnomaly`: Mahalanobis distance from `GPURasterEmbeddingCovariance`, or distance to the nearest label | Not implemented; effort S. | M | [new] |
| EMB-05 | A regression output on the linear probe; mean-of-references similarity | Not implemented; effort S. | M | [new] |
| EMB-06 | Edge strength between neighbouring embeddings, as input to watershed and sieve (check the overlap with `GPURasterTexture` first) | Not implemented; effort S. | L | [new] |
| SIM-02 | Terrain error ensemble recipe: probable viewshed, depression probability and flow-path probability from DEM error realizations | Not implemented; effort M. | M | [new] |
| SIM-03 | `GPUSensitivityIndices`: Sobol and Morris indices for `GPUWeightedOverlay` and other contributors with per-frame parameters. Oracle: SALib | Not implemented; effort M. | M | [new] |
| SIM-04 | `GPURandomizedSVD` as a shared primitive; moves the EOF eigen step off the CPU. Decide first whether it belongs in luma.gl | Not implemented; effort M. | M | [new] |
| SIM-05 | `GPUGeographicallyWeightedPCA` on the GWR neighbour grid | Not implemented; effort M. | M | [new] |
| SIM-06 | `GPUNonnegativeMatrixFactorization`; unblocks RS10 | Not implemented; effort S to M. | M | [new] |
| SIM-07 | `GPUZoneRaking`: iterative proportional fitting across many zones, with integerisation | Not implemented; effort S to M. | L | [new] |
| SIM-08 | `GPUMovementStates`: hidden Markov model states for many tracks. The published `GPUSegmentedScan` is a uint32 prefix sum, so the scan over matrix products would be local code | Not implemented; effort M. | L | [new] |
| SIM-09 | `GPULocationObfuscation`: donut, Gaussian and planar-Laplace displacement with its effect on density and hot spots | Not implemented; effort S. | L | [new] |
| SIM-10 | `GPUStructuredKernelInterpolation`: the kriging mean of 100,000 to 10 million points on a grid by FFT and `GPUJacobiPCG`, with the model from the Vecchia fit; variance later. Oracle: GPyTorch | Not implemented; effort M. | M | [new] |
| CART-01 | `GPUFlowCartogram`: contiguous density-equalising cartogram (Gastner, Seguy and More 2018) on `GPUFFT2D`. Reference programs are AGPL or modified MIT: fixtures only | Not implemented; effort M to L. | M | [new] |
| CART-02 | `GPUPointLabelPlacement`. No public reference code; a CPU twin checked by invariants is the oracle | Not implemented; effort M. | M | [new] |
| CART-04 | `GPUDisplacementInterpolation`: a raster between two densities at a time fraction, on the Sinkhorn kernel. Oracle: POT | Not implemented; effort M. | M | [new] |
| CART-05 | `GPUPointClusterHierarchy`: cluster parents, counts and centroids per zoom, and a blue-noise display sample. The parallel algorithm is a design from the research, not a published one, and needs its own review | Not implemented; effort M. | M | [new] |
| CART-06 | `GPUTravelTimeMap`: map vertices rescaled radially by travel time from an origin. Also a design from the research | Not implemented; effort S. | M | [new] |
| CART-07 | Small set: illuminated (Tanaka) contours, value-suppressing uncertainty classes and exceedance probability, colour-vision simulation with a palette check, guided filter, parallel-coordinate density rasters, pansharpening (needs a scene with a panchromatic band, WEB-PIC-09) | Not implemented; effort S each. | M | [new] |
| CART-08 | `GPUFieldTSNE` linked to the map | Not implemented; effort L. | L | [new] |
| CART-09 | `GPURoadStrokes` for network thinning | Not implemented; effort M. | L | [new] |
| ENC-01 | `GPULocationEncoder`: the SatCLIP location encoder (spherical harmonics, dense layers, a sine; about 1.2 million weights, MIT) as covariates per coordinate. Needs a one-off extraction of the weights and the SatCLIP model-card decision below | Not implemented; effort S to M. | M | [new] |
| ENC-02 | Transformer kernels (layer norm, attention, GELU) and the Galileo-Nano encoder (0.8 million parameters, MIT) | Not implemented; effort L. | L | [new] |
| ENC-03 | Presto on sample pixels only (about 89 million operations per pixel); weight licence not stated | Not implemented; effort M after ENC-02. | L | [new] |
| ENC-04 | Spike: ONNX Runtime Web on a device luma.gl created, tensors kept on the GPU, in an example. Decides whether a separate optional package for larger encoders is worth having | Not implemented; effort S. | L | [new] |

### Decisions for the maintainer

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| FX-03 / FX-05 | Confirm hotspot/global-G defaults | Current sporadic rule follows ArcGIS (older rule opt-in); negative G data produce GPU invalid status. Decide whether host rejection or a kepler diagnostic is wanted. | Decision | [spatiotemporal] |
| CUBE-OCCUPANCY | Never-occupied lattice cells | Keep current treatment until ArcGIS behaviour is confirmed; exclusion by default remains undecided. | Decision | [spatiotemporal] |
| DEC-HDBSCAN | HDBSCAN minimumSamples | Keep 32 or raise to 256 with rejection/audit updates; separate from PLAT-03 traversal adoption. | Decision | [spatiotemporal]; `gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts` |
| STD-04 / STD-09 | Public contract shape | Decide output.ports.status for depression hierarchy, OPTICS and network KDE; whether validateSinglePackedView is public. | Decision | [library] |
| STD-07 / STD-15 | Library API vocabulary | Decide addToGraph/getCommandNodes, outputValidity/validity, conditionResult, iteration-cap names and statistics naming before a coordinated API change. | Decision | [math]; [library] |
| DUP-02 | Confirm spatial-k-means name | GPUSpatialKMeans is implemented; the public name remains a maintainer decision. | Decision | [library] |
| raster-fields-05 / STD-06 | Contour/flow row-direction flip | Follow affine in one coordinated caller change: 108 sites across about 57 files. | Decision | [row-direction plan](./research/math-correctness-review-2026-11-10/row-direction-flip-plan.md) |
| DEC-WALK-ROUNDING | Walking-cost rounding acceptance | Required-parameters GRASS path differs by one ULP in 8/35 cells from the earlier kernel; Tobler is exact. Accept or restore; STD-05 tracks parameter adoption. | Decision | [library] |
| DEC-VARIANCE | Raster cell/zonal variance divisor | Both documented; align only if published raster statistics specify a convention. | Decision | [math] |
| DEC-ARROW-CONSUMERS | Arrow-preparation copy consumers | Decide whether to add consumers of the registered upstream copy when re-enabling parked examples. KE-TESTS tracks per-file qualification; derive current opportunities from its manifest rather than the old four/three estimate. | Decision | [workarounds]; KE-TESTS |
| WEB-INT-16 | Font licence review | Text atlas is a modified KaTeX font under OFL-1.1; legal reading remains unchecked. | Decision | `THIRD-PARTY-DATA.md` |
| LUM-09 | Fill-holes connected-components adoption | Published labelling uses 3–5 rounds and 59 nodes versus 89 flood nodes on measured ≤512² cases. Decide whether iterationCount changes meaning or is explicitly incomparable with the CPU twin. | H (decision) | [usage] |
| DEC-LAG | Lag-ML rejected iterate | Twin publishes status-3 iterate; GPU clears table/summary. Decide whether GPU should expose it too. | Decision | [cpu] |
| dataframe-10 | Splancs parity and reference licence | Decide optional splancs K-function tie rules and the MIT-headed NumPy Diggle port written from GPL-source reading; no copied splancs code is claimed. | Decision | [math] |
| PLAT-04 / PLAT-05 | Send upstream proposals and compiler reports | Decide filing Chromium/Apple drafts and GPUHistogram proposal; proposed AGENTS rules for pinned private WGSL copies and float compensated sums remain unapplied. | Decision | [proposals]; [compiler findings](./research/math-correctness-review-2026-11-10/shader-compiler-findings.md) |
| LRN / EMB / SIM / CART / ENC | Choose new-contributor scope | Confirm order, family locations, generic-layer boundary, SatCLIP model-card restrictions, example-only ONNX spike and default versus adapter device limits. SIM-04 randomized SVD needs an upstream ownership decision. | Decision | [new] |

## Known limits recorded by finished passes

| Area | Limit | Where documented or pinned |
| --- | --- | --- |
| Lag ML | Near-exact fits can inflate variance 33× with status 0: 7.8e-13 against 2.4e-14. | `gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood.ts`; `math-review-lag-conditioning.spec.ts` |
| Panel random effects | Float32 entity means leave variance errors of 1.1% and intercept errors of 0.2–0.3 standard errors near variance ratio 2e-7; below 1e-8, status 3, no covariance. | [Math review](./research/math-correctness-review-2026-11-10/regression.md); `math-review-panel-lag-conditioning.spec.ts` |
| Probit | Uncertified separation can end as status 3 or 2 rather than 4; loose `tolerance` (1e-2) has no upper guard. | `gpu-spatial-analysis/spatial-regression/gpu-probit-regression.ts`; `math-review-probit-bound.spec.ts` |
| GPU triangulation | Float32 diagonal midpoints can fail: 1 of 126,382 polygons flagged although package Earcut tiles it. | [Core numerics](./research/math-correctness-review-2026-11-10/core-numerics.md); `gpu-polygon-triangulation-search.spec.ts` |
| Space-time K / Diggle | K keeps stpp ties; only Diggle follows splancs. Splancs parity still excludes the weight cap of 100 and a stale term at the largest distance. | `dataframe-analysis/pair-statistics/gpu-space-time-k-function.ts`; `gpu-space-time-diggle-test.ts`; `space-time-diggle-splancs.spec.ts` |
| COLD | Refit schedule and window advance differ from pyxccd. | `raster-analysis/raster-time-series/gpu-raster-cold-monitor.ts`; `rts04-cold-pyxccd.spec.ts` |
| Sky view | Sampling differs from rvt: sky-view-factor discrepancies up to 0.11 on steep terrain and 0.57 at grid edges. | [Terrain review](./research/math-correctness-review-2026-11-10/terrain-surface.md); `gpu-terrain-horizon-rvt.spec.ts` |
| Panel error ML | Not read for correctness. | [math] |
| Triangulation twins | Polygon and constrained Delaunay twins always use Earcut; polygons only GPU monotone can finish may fail. | `analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation.ts`; `gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation.ts` |
| CPU arithmetic | Float64 arithmetic rounded once can change float32 ties and threshold decisions. | [cpu]; twins' “What differs from the GPU contributor” comments |
| CPU budgets | GPU-only round, candidate and size caps are removed; twins can finish when GPU reports a limit. | [cpu]; twins' contract comments |
| CPU output tails | Rows past `count` are zero-filled where GPU leaves them unwritten. | `gpu-network/network-analysis/cpu-network-path-extraction-operation.ts` |
| IN01 | Keep and document resident variogram fitting: slower on small inputs, CPU 1.3/3.2 ms versus GPU 4.3/51 ms. | [Interpolation benchmarks](../modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md) |
| Space-time variogram | `GPUSpaceTimeVariogram` requires `gridSize` and eight parameters. | `dataframe-analysis/pair-statistics/gpu-space-time-variogram.ts` |
| Temporal reduction | Word mode permits at most 4,294,967,294 slots (`cellCount * bucketCount`). | `dataframe-analysis/temporal-reduction/gpu-temporal-reduction.ts`: `MAXIMUM_SLOT_COUNT` |
| Trajectory restriction | `unknown` counts each invalid segment once plus gap-overlap pairs. | `gpu-spatial-analysis/trajectory-restriction/trajectory-restriction-pair-merge-kernels.ts` |
| Space-time kriging | Indexed by default above 4,096 samples (device-binding bound, about 8.4M packed samples at 128 MiB); the explicit full scan stays capped at 1,048,576. | `gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts`: `GPU_SPACE_TIME_KRIGING_INDEX_MAXIMUM_SAMPLE_COUNT` |
| OPTICS xi | Xi extraction stays at 65,536 rows: float32 rounding of `1 - xi` makes intervals cross, and a parallel conflict selection changes results; the grid DBSCAN route goes to 1,048,576. | `gpu-spatial-analysis/spatial-clustering/gpu-optics.ts`: `GPU_OPTICS_XI_MAXIMUM_ROWS`; `optics-xi-dependencies.spec.ts` |
| Urban tessellation | The local route supports up to 1,000,000 buildings with convex or simple, possibly holed enclosures (at most 32 enclosure vertices and 32 rings), bounded per-building sampling/candidate/clipping work and radius certification. Dissolved shell/hole overlay remains the 64-building reference route. Candidate-search or certification exhaustion reports `candidateOverflow` instead of a certified polygon. | `gpu-spatial-analysis/urban-morphology/building-tessellation-local.ts`; `modules/analysis/test/gpu-spatial-analysis/urban-ecology/building-tessellation-improvements.spec.ts` |
| Visibility graph | The network CSR recipe supports integration and normalized choice with all-source pivots when the command budget fits. Auto uses sampled pivots when all-source work exceeds that budget; `metricsMode: 'exact'` rejects over-budget runs and sampling sets `status.approximation`. Frontier scratch is laneCount × nodeCount; round-budget exhaustion publishes partial metrics with status. The spatial CSR recipe returns adjacency/degree only; the 256-point dense reference is a separate route. | `gpu-network/visibility-graph/visibility-graph.ts`; `gpu-spatial-analysis/urban-morphology/visibility-graph-csr.ts`; `modules/analysis/test/gpu-spatial-analysis/urban-ecology/visibility-graph-metrics.spec.ts` |
| Network flow | Parallel max flow uses preflow push-relabel. Costed parallel flow defaults to bounded successive shortest paths; explicit primal-dual mode batches shortest admissible paths with its own phase budget. Exhaustion publishes a partial result with `iterationLimitReached`. | `gpu-network/network-optimization/gpu-network-flow.ts`; `gpu-network/network-optimization/network-flow-parallel.ts` |
| Chunked input | Most chunked contributors still keep row-proportional scratch in one binding (aggregation, quantile moments, order statistics, join compaction), so totals are bounded by the device binding size; the kepler host now requests the adapter maximum. | contributor class docs; `analysis-core/graph-support/device-limits.ts` |
| FP64-04 | Keep local addition/multiplication rounding-error helpers. | [workarounds]; `analysis-core/kernels/addition-rounding-error-wgsl.ts`; `multiplication-rounding-error-wgsl.ts` |
| Published fp64 | About 48 bits, not 53, with float32 exponent range. | `modules/analysis/test/analysis-core/shader-fp64-module.spec.ts`; [cpu] |

## Belongs in luma.gl

Use [upstream-proposals.md](./upstream-proposals.md) for proposed changes and [upstream-workarounds.md](./upstream-workarounds.md) for current gaps and accepted regressions. Do not implement general compute, rendering or platform APIs here.

## Completed passes

- 2026-11-18 — LR-22 bump adaptations: luma.gl target/peers moved to alpha.16; installed public APIs audited and website inventory regenerated. Replaced five grid WGSL copies to preserve builder/reader agreement, updated float32 CPU grid oracles, retained generated-source pins, added a 30-case headless regression (30 of 30 on the GPU), and forwarded Kepler compile options. Typechecks, build, export and packed-consumer checks, the Node suite (1,855 files) and the GPU suites passed on the bumped tree, apart from the intermittent continuation spec recorded in LR-22; all 427 lens scenarios were recaptured. The gate steps ran separately, not as one `yarn verify` command. Separate segmented-sum/PCG adoption remains in LR-22.
- 2026-11-18 — LR implementation: LR-01 to LR-21 landed (the 36 verified wrong results, the gates that passed when nothing ran, `yarn verify`, the packed-consumer gate, first-run docs, completeness ports, one meaning per port, twin parity, record identity, lens tables); LR-23 drafted. Decisions taken: no new operator families before the alpha except for a named consumer (LR-D1); `apache-arrow` is a required peer (LR-D2); wrong names break now, without aliases (LR-D3); a consumer of possibly incomplete input propagates, suppresses or refuses (LR-D4); an explicit iteration budget binds a CPU twin (LR-D5); lens coverage counts a named family scenario (LR-D6); the gate is local `yarn verify`, sized to one machine (LR-D7). Evidence and gate results: [research/library-review-2026-11-17/implementation.md](./research/library-review-2026-11-17/implementation.md). LR-07 and LR-09 closed in the second tranche; LR-02, LR-08, LR-10, LR-17, LR-19 and LR-20 narrowed.
- 2026-11-18 — LR second tranche: multipart point and line feature joins (`intersects`, `within`, `coveredBy`, nearest neighbors) on shared multipart ports; SKATER, REDCAP and minimum spanning forest consume weights loss; network flow path decomposition; atomic recipe preparation for every `add...Recipe` function; tile-count ceiling 4096 for error GM, 2SLS and weights traces; lens public surface reduced (views 461 → 8 names) and a registry-wide declaration and record-signature gate over 427 scenarios; point-cloud and geospatial root files moved into families; shader memory census (3,095 tests); kepler WKB empty-chunk import; app typechecks clean after a package build (kepler-compute 6 → 0, website 12 → 0, showcase and explorer 0 → 0). Remaining: the LR rows above.
- 2026-11-18 — LR-16 roadmap reconciliation: removed completed TE03 activation, R-RELAX settlement/continuation, EMB-02 streaming widths and NW-CENT host continuation; closed unspecified KJ-01 (count, sum, mean, minimum, maximum and count-distinct ship; reopen only for a named operation/consumer). Narrowed source-verified route limits and retained only unmet AX gates. Decision IDs DEC-HDBSCAN, DEC-WALK-ROUNDING and DEC-ARROW-CONSUMERS replace the ambiguous decision uses of PLAT-03, STD-05 and KE-TESTS.
- 2026-11-14 — BL-PASS: waves A/C and B1/B2 landed; remaining work is in the domain tables. Migration contracts retained for the changelog hand-off: `GPUNetworkFacilityOptimization` and `GPUNetworkRationalAccess` renamed packed `output.status` to `output.summary`; hull/minimum-bounds `output.status.overflow` is boolean, with separate hull `groupOverflow`/`vertexOverflow` reasons and minimum-bounds `groupOverflow`; `GPURasterReprojection` needs `destroy()` even for same-CRS use; raster patch metrics parameters are eight words; `GPUMarkedPointStatistics` indexes by default (`searchStructure: 'brute-force'` keeps the reference route). The dated research retains definitions and test evidence.
- 2026-11-15–17 — SC: scalability implementation and docs committed in `69305eb4` and `bdc715ea`; chunked inputs, adapter-limit requests and scalable network routes ship. Reported integration gates passed, including the 69 new/changed headless specs; no rerun is implied here.
- 2026-11-16–17 — IMP: family improvements committed in `69305eb4` and `bdc715ea`, including tiled D8, Borůvka region graphs, streaming embeddings through 1,024 dimensions, centrality continuation and S2 subsets. Reported integration gates and 899-file Apple Metal headless pass were clean; retained limitations are tracked by domain and execution route.
- 2026-11-15 — AX-PASS: dataset-aware defaults, applicability, suggestions, result presentation, curated templates, sensitivity, named weights and workbench adapters landed. AX-02 resident-catalog input/declaration behavior has a 24-row CSV journey in `ax-profile-qualification.mjs`; it does not close AX-20's million-row timing gate. Registry, descriptor and template specs cover the implementation; open latency, screenshot, real-sample and browser gates remain in the AX table.
- 2026-11-18 — LR-06 documentation facts: integration-version and mode-specific limits, mixed-executor/local-adapter boundaries, current citations, catalog summaries and maintained upgrade inventory tools. Scoped node and source checks pass; GPU and full release gates remain with the lead.
- 2026-10-25 — Showcase scene/data overhaul; [research/showcase-overhaul-2026-10-25](./research/showcase-overhaul-2026-10-25/README.md) holds the reference JSON its check scripts read.
- 2026-11-07 — Upstream-first package split, public-entry checks, test/example typechecks, centrality guard and sparse resistance recipe; [upstream-first-plan.md](./upstream-first-plan.md) holds the boundary.
- 2026-11-09 — Card gestures, matching posters and initial picture fixes; remaining interaction and picture work is in the website tables.
- 2026-11-09–11 — Contract checker, registry, shared kernels, geometry layout, mask renames and indexed paths; [research/library-review-2026-11-09](./research/library-review-2026-11-09/README.md) holds the contract/design evidence.
- 2026-11-10 — Spatiotemporal implementation; [research/spatiotemporal-review-2026-11-09](./research/spatiotemporal-review-2026-11-09/README.md) holds definitions and partial limits.
- 2026-11-10 — Thirteen explanation pages and 92 missing catalog/run entries delivered; follow-ups remain in the website tables.
- 2026-11-10–11 — Math fixes, oracle checks and closing pass; [research/math-correctness-review-2026-11-10](./research/math-correctness-review-2026-11-10/README.md) holds resolutions and documented numeric limits.
- 2026-11-11 — Mixed execution and 125 CPU twins; [research/cpu-suitability-2026-11-11](./research/cpu-suitability-2026-11-11/README.md) holds survey, comparisons and leads. New-contributor research is an unexecuted plan.
- 2026-11-11 — Integration fixes, consumer refresh and GitHub Pages deployment; open integration items are in the domain tables.
- 2026-11-13 — Comparison of the luma.gl fork with published luma.gl: what the fork changed in luma.gl-owned code and where each change is tracked; [research/fork-comparison-2026-11-13](./research/fork-comparison-2026-11-13/README.md).
- 2026-11-12 — Alpha.13 adoption, published gathers/scans, shader assembly, halo corrections and readback/lifetime fixes; [research/luma-usage-review-2026-11-11](./research/luma-usage-review-2026-11-11/README.md) holds adoption costs and remaining leads.
