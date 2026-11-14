# Roadmap

Status date: 2026-11-13. Baseline: `@luma-contributors/analysis`, `@luma-contributors/lenses` and `@luma-contributors/kepler` use published `@luma.gl/*@10.0.0-alpha.13`. The pins have not moved to alpha.14; its colour-conversion, expression-operation and SQL changes do not change this backlog.

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

## High-priority summary

| ID | Still open |
| --- | --- |
| MOB-03 | Decide whether point-centred speed-outlier detection becomes the default. |
| NW-CENT | Centrality unrolls rounds and rejects graphs above 32,768 command nodes; the structural fix needs published GPU-gated continuation. |
| WEB-PIC-12 | Rework the remaining weak pictures and gestures, then re-score the whole site in both themes. |
| KE-TESTS | Resolve the 131 excluded kepler spec files and the cold-cache DuckDB failure. |
| STD-04, STD-06, STD-07 | Finish status ports, raster context and published raster-band validity. STD-10 solver adoption remains M priority. |
| GPU-05, FX-18 | Index MGWR fits; bound island-attachment rescans; fix the leading-padding read in `getSnapOrigin` and finish the padding checks. |
| LUM-09 | Decide the fill-holes status contract before adopting published connected components. |

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
| INT-02 | Validate operation contracts at construction | Prop-derived zero lengths name internal ports; output formats and input names fail later in luma.gl. | M | `analysis-core/operations/gpu-contributor-operation.ts`: `GPUContributorOperation` |
| INT-05 | Use or remove unused WGSL helpers | Row-access and sphere-distance modules have specs but no contributor consumers. | L | `analysis-core/kernels/row-access-wgsl.ts`; `analysis-core/kernels/sphere-distance-wgsl.ts` |
| INT-07 | Resolve split-review leads | Re-check dropped lens/showcase files, unlisted specs, lost tests, EC03 limits/validation and register errors against current code before acting. The fork changes to luma.gl-owned code are mapped separately. | M | [split]; [fork] |
| INT-08 | Establish CI and Pages build policy | CI execution is unverified; decide whether Pages builds belong in a workflow. | M | `.github/workflows/test.yml`; `website/scripts/deploy-pages.mjs` |
| INT-09 | Public website repository link | The header points to a private repository. | L | `website/src/components/brand/site-header.tsx` |

### Spatial statistics, regression and local statistics

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| ST01 | HC2/HC3 and clustered covariance, positive-semidefinite certification for spatial 2SLS | Not implemented. `GPUSpatialTwoStageLeastSquares` and `GPUInstrumentalVariablesCovariance` give the base covariance | M | [coverage]; `gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts`: `GPUSpatialTwoStageLeastSquares` |
| ST02 | Spatially robust restriction inference | Not implemented (`GPURestrictedRegimeRegression` exists) | L | [coverage]; `gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts`: `GPURestrictedRegimeRegression` |
| ST03 | Time effects, clustered covariance, dynamic panels | Fixed and random effects panel error and lag ML exist (`gpu-spatial-panel-*-maximum-likelihood.ts`). `gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression.ts` rejects other effects | L | [coverage] |
| ST04 / ST05 | Iterated SUR; latent spatial probit | Single-pass `GPUSeeminglyUnrelatedRegression` and `GPUProbitRegression` only | L | [coverage]; `gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression.ts`: `GPUSeeminglyUnrelatedRegression` |
| ST06 | MGWR: AICc and continuous bandwidth search, full backfit selection uncertainty | Finite-grid LOO selection only. `gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression.ts` rejects other variance conventions | M | [coverage] |
| ST06b | Adaptive or selected bandwidth and separation handling for Poisson and logistic GWR | Fixed bandwidth. `gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts` (bandwidth mode, family and kernel) | M | [coverage] |
| ST07 | LOSH and multivariate Geary: joint randomization, more correction families | Descriptive output plus separate inference only | M | [coverage] |
| ST08 | Rank Markov: fractional ties, missing-panel interpolation, more than 64 states | Not implemented (`GPURankMarkov`) | L | [coverage]; `gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov.ts`: `GPURankMarkov` |
| ST09 | Geographic (non-planar) geometry and topology repair for polygon statistics | Planar only. `gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts` rejects distance output | L | [coverage] |
| ST10 | blockCV: hierarchical and merge blocks, response balancing, continuous variogram optimizer | Not implemented (`GPUSpatialCrossValidation`) | L | [coverage]; `gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts`: `GPUSpatialCrossValidation` |
| ST11 | Spatial interaction: multiple covariates, other count distributions, coefficient uncertainty in calibrated flow | Not implemented (`GPUSpatialInteraction`) | L | [coverage]; `gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction.ts`: `GPUSpatialInteraction` |
| KG01f | Multivariate quantile LISA | Single-variable only (`GPUQuantileLISA`) | M | [coverage]; `gpu-spatial-analysis/recipes/quantile-lisa-recipe.ts`: `GPUQuantileLISA` |
| WEIGHTS7 | Chunked spatial-weights transpose output | Inputs may be chunked; output remains packed. | M | [coverage]; `gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose.ts`: `GPUChunkedSpatialWeightsTranspose` |
| PERM-01 | Permutation batching with early stopping | `gpu-spatial-analysis/permutation-inference/gpu-global-permutation-test.ts` rejects `permutationBatchSize` with `stopAfterExceedances` | L | [coverage] |
| JOIN-01 | Distance output from predicate join | `gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts` rejects `output.weights.distances` | L | [coverage] |
| ROSE-01 | Other circular angle conventions | Radians with positive-x counter-clockwise origin only. | L | [coverage]; `gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts`: `GPURoseStatistic` |
| KJ-01 | Other key-join operations | Aggregates support count, sum, mean, minimum and maximum. | L | [coverage]; `dataframe-analysis/key-join/gpu-key-join.ts`: `GPUKeyJoin` |
| KG02 | Dorling cartogram: GeoDa coordinate parity, convergence guarantee | Heuristic by contract; relaxation limited to 256 passes (`gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts`) | L | [coverage] |
| KG03 | Exact natural breaks beyond 8,192 rows or 64 classes; float64 objective | Exact mode materializes costs to 1,024 rows and recomputes to 8,192 (`dataframe-analysis/column-classification/gpu-class-breaks.ts`). Kepler uses a worker Jenks instead | L | [coverage] |

### Point processes, sampling and clustering

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| PP01 / PP02 | Study-window corrections and envelope tie breaking | Inhomogeneous K, intensity and smoothed pair correlation exist to 4,096 events; more corrections and extreme-rank-length tie breaking remain. | L | `gpu-spatial-analysis/point-process/`: `GPUInhomogeneousPointPattern` |
| PP04 | Parallel OPTICS ordering | Indexed frontier supports 65,536 rows; ordering is sequential in one workgroup; xi extraction stops at 4,096. | M | `gpu-spatial-analysis/spatial-clustering/gpu-optics.ts`: `GPUOPTICS` |
| PP05 | Response-dependent replacement and design variance in GRTS sampling | Not implemented | L | [coverage] |

### Interpolation and simulation

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| IN04 | Fitted linear model of coregionalization (more than 4 variables or structures) | `GPULocalKriging` takes a supplied LMC; `GPUVariogramModelFit` fits one model | M | [coverage]; `gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts`: `GPULocalKriging` |
| IN05 / IN02 | Degree above 1 or global dense RBF; global indexed kriging | Not implemented (`GPULocalRBF`, `GPULocalKriging` are local) | L | [coverage]; `gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf.ts`: `GPULocalRBF` |
| IN06 | Joint covariance across separate conditional simulations | Not implemented, by design | L | [coverage] |

### Network

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| NW-CENT | Centrality without compile-time unrolled rounds | `maximumIterations` is 1–1,024 (default 64); `estimatedCommandNodeCount` is capped at 32,768. Use sampled pivots, fewer rounds or more lanes until GPU-gated continuation is public. | H | `gpu-network/network-centrality/gpu-network-centrality.ts`: `GPUNetworkCentrality` |
| NW-SCC | Large-graph strongly connected components | Exact reachability SCC only: 1,024 vertices, 65,536 CSR slots. The fork has a trim, colour and backward-mark implementation with O(V) scratch, checked against Tarjan to 5,000 vertices; port it as a network contributor. | M | `gpu-network/network-structure/gpu-network-structure.ts`: `GPUNetworkStructure`; [fork] |
| NW-KSP | Scalable k-shortest and simple route enumeration | `GPUAlternativeRoutes` is bounded to 256 nodes (`gpu-network/alternative-routes/gpu-alternative-routes.ts`) | M | [coverage]; `gpu-network/alternative-routes/gpu-alternative-routes.ts`: `GPUAlternativeRoutes` |
| NW04 | Sparse scalable Louvain | At most 32 vertices, dense (`gpu-network/network-optimization/gpu-network-louvain.ts`). Published `@luma.gl/graph` has modularity optimization; evaluate it first | M | [coverage] |
| NW02 | Sparse push-relabel and path decomposition for max flow | At most 128 nodes and 1,024 edges (`gpu-network/network-optimization/gpu-network-flow.ts`) | L | [coverage] |
| NW03 | Negative-cost circulation and lower-bound constraints | Not implemented | L | [coverage] |
| NW05 | Mixed-integer or fractional allocation; certified capacitated p-median | Exhaustive and bounded: demand at most 64 (`gpu-network/network-optimization/gpu-network-facility-optimization.ts`) | L | [coverage] |
| NW06 / X04 | Non-FIFO schedules, time windows, transit | FIFO only (`gpu-network/network-optimization/gpu-network-time-routing.ts`); transit is excluded | L | [coverage] |
| NW-GA | Generic graph adapters for dense traversals | Not implemented | L | [coverage] |

### Raster, terrain and hydrology

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| R6.3 | Seam-safe contour ownership across tiles; external vector overlays | Isolines handle one raster | M | [coverage] |
| R6.4 | Broader fractional and zonal composition envelopes | Partial: `GPURasterFractionalCoverage` and zonal statistics exist; whole-domain and tiled acceptance cases are not all closed | M | [coverage]; `raster-analysis/gpu-raster-fractional-coverage.ts`: `GPURasterFractionalCoverage` |
| R7.2 | Watershed at larger scale; other segmentation methods | Complete domain of at most 16,384 cells (`raster-analysis/gpu-raster-watershed-segmentation.ts`) | M | [coverage] |
| R7.4 / RS07 | Spectral (FFT) raster filtering; registration peak refinement and ambiguity | Not implemented. FFT convolution is deferred | M | [coverage] |
| R-FLOW | Cross-tile D-infinity and MFD network stitching | Partial: `GPUTerrainTiledFlow`, `GPUTerrainBranchingFlow` exist | M | [coverage]; `gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts`: `GPUTerrainTiledFlow` |
| R-RELAX | `settledAt` output with continuation in cost distance | `raster-analysis/cost-distance/raster-relaxation.ts` | L | [coverage] |
| R7.1 / R7.3 | Adaptive local analysis (CLAHE); volumetric microscopy | Gate-deferred, no code | L | [coverage] |
| RS02 | Sparse or string zone IDs | Dense numeric IDs only, up to 65,534 classes | L | [coverage] |
| RS03 | Curvilinear grids for conservative remapping | Not implemented | L | [coverage] |
| RS06 | Irregular dates in temporal composites and filters | Not implemented | L | [coverage] |
| RS10 | Endmember discovery, ISRA and NMF | Not implemented | L | [coverage] |
| RS11 | Multiple break search, calibrated p-values, full BFAST | Single-break seasonal model | L | [coverage] |
| R-REPROJ | Explicit source-tile routing for reprojection; extra precision and filtering | Not implemented | L | [coverage] |
| TE06 | Out-of-core and independently filled tiles | Resident full-domain composition only (`GPUTerrainTiledFlow`) | M | [coverage]; `gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts`: `GPUTerrainTiledFlow` |
| HAND-01 | Height above drainage over D-infinity flow | `gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts` | M | [coverage] |
| RELIEF-01 | Percentile stretch in relief blending | `gpu-terrain/relief-visualization/gpu-relief-blend.ts` (needs a histogram) | L | [coverage] |
| TE03 | Parallel depression-hierarchy activation pass | Serial (partial) | L | [coverage] |
| A-18 | `GPUTerrainSpikeRepair` persistent buffer | Persistent-buffer support needs an upstream change | L | [workarounds]; `gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts`: `GPUTerrainSpikeRepair` |
| TE05 | Horizontal, Nuth-Kaab and tilt coregistration models | Not implemented (`GPUTerrainDEMCoregistration`) | L | [coverage]; `gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration.ts`: `GPUTerrainDEMCoregistration` |
| LUM-24 | Embedding nearest-classifier on similarity search | 4,096 training-vector cap; migration adds `pixelCount × k × 8` transient bytes and a pass. Fix the empty-slot vote defect independently. | M | [usage]; `raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier.ts`: `GPURasterEmbeddingKNearestClassifier` |
| LUM-25 | Reclassify, conditional and stretch on raster-band inputs | Bare value views bypass published band validity; caller prop change overlaps STD-07. | M | [usage] |
| LUM-26 | Project inside the reprojection sampler | Use `CompiledProjection.getShader()` to remove a pass and two full-raster transients; requires a kernel rewrite. | M | [usage]; `raster-analysis/gpu-raster-reprojection.ts`: `GPURasterReprojection` |

### Dataframe, point cloud, projection and geometry

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| KG06 | SQL: DOUBLE, WKB, arbitrary strings, right-column projections, UNION | Partial. The engine is published `@luma.gl/graph/gpu-sql`; nothing in this repository. Needs upstream work | L | [coverage] |
| DF-DESC | Public dataframe query description (intent, shader plan, formulas) | Unavailable; needed to rebuild lens scenarios. The fork implementation exists and is a candidate upstream pull request. | M | [coverage]; [fork] |
| ARROW-01 | More Arrow analytics types | `analysis-core/graph-support/arrow-analytics-normalization.ts` rejects unsupported types | L | [coverage] |
| GEO-DEPR | Remove deprecated `getRasterSphereRadius` alias | Use `getGPUAuthalicSphereRadius`; retain the other cell-area helpers. | L | [coverage] |
| PC06 | Automatic normal-scale selection, fitted correlation model; M3C2 binary has never been run | Not implemented | M | [coverage] |
| PC01 | Globally propagated normal orientation; unlimited support | Viewpoint or up orientation per query | L | [coverage] |
| PC02 | Distributed global threshold reconciliation | Not implemented. Voxel downsample and radius filter are published | L | [coverage] |
| PC03 | Ground Delaunay or nearest interpolation; vertical datum | Not implemented | L | [coverage] |
| PC05 | Global alignment, trimmed objective for ICP | Not implemented | L | [coverage] |
| CR02 / CR03 | Epoch-aware grid resources, bounded residency | `GPUDatumGridTransform` (`geospatial-analysis/gpu-datum-grid-transform.ts`) applies supplied grids; no epoch handling | M | [coverage]; `geospatial-analysis/gpu-datum-grid-transform.ts`: `GPUDatumGridTransform` |
| CR01 | Difficult geodesic cases: poles, antipodes, singular inverse | Float32 envelopes are not certified | M | [coverage] |
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
| KE-TESTS | Run excluded specs and expand typecheck coverage | 131/368 spec files excluded; 59 more excluded from typecheck. Resolve package/API dependencies and prebundle DuckDB for the cold-cache Arrow-schema failure. KE-EX controls example-backed coverage. | H | `test/kepler-compute-example-specs.json`: `excludedFromRuns`, `excludedFromTypecheck`; `vitest.config.ts` |
| KE-EX | Decide the parked compute example’s future | About 750 files, outside workspaces; deck.gl and community GPU-layer imports are undeclared. A published GPU-layers package alone would not restore it: it also imports `OutOfMemoryError` from `@luma.gl/core` and `walkTextGlyphs` from `@luma.gl/engine/text`, neither of which alpha.13 exports. | M | `examples/deck/kepler-compute`; [fork] |
| KE-UI | Remaining host work: native Group By panel binding, non-point altitude layers, richer labels, micro and nanosecond time admission, multipart timed trips, acceptance on other hardware tiers | Open (mostly example code). Acceptance was measured on one Apple M-class machine | M | [coverage] |
| KE-DECK | Cluster, flow and globe layer integration | External; track with the deck.gl-community package | L | [coverage] |
| INT-04 | Reclassify test exclusions | Reasons and skip comments cite alpha.10; nine specs cite missing physical-transient statistics that alpha.12 publishes, while compilationStats remains unavailable. | M | `test/kepler-compute-example-specs.json` |
| CPU-06 | Replace hand-written CPU fallbacks with twins | Use twins where available in snapping and platform bindings. | M | [cpu]; `modules/kepler/src/analysis/catalog-gpu-snapping.ts` |
| LUM-31 | Group-by on published dataframe aggregation | Migration loses the null-key group and nonfinite counts; preserve those contracts. | L | [usage] |
| LUM-13 / LUM-32 | Predicate encode gate | Mask-only filters re-encode each frame; safe gating requires submit confirmation in explorer and scalar-selection contracts. | M | [usage] |
| LUM-14 / LUM-33 | GPU permutation feeds sorted-page gather | Needs a `KeplerSessionExplorer` pass-through and changes two pinned readback byte lengths. | M | [usage] |

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
| CPU-04 | Recipes as mixed GPUGraph programs | Segregation counterfactual, population support and building tessellation; retire the deprecated dense resistance recipe for the float64 solve. | M | [cpu] |
| CPU-05 / INT-03 | Expose twins in catalog and website | Add cpuTwin metadata; identify operation forms by class rather than filename in inventory extraction. | M | [cpu]; `website/scripts/extract-inventory.ts` |
| CPU-07 | Document operation forms and twins | Add an API page with the variogram example. | M | [cpu] |
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
| STX-03 | Space-time HDBSCAN | ST-DBSCAN and ST-OPTICS exist; ST-HDBSCAN does not. | M | [spatiotemporal] |
| STX-08 | Complete exploratory statistics | REDCAP is first-order single linkage; LICD is local composition only; colocation quotient lacks permutation inference. | L | [spatiotemporal] |
| STM-06 | GTWR/MGTWR bandwidth selection | No golden-section or adaptive bandwidth search. | M | [spatiotemporal] |
| STM-07 | External Vecchia validation | GpGp GPU comparison/fixtures pending; spatial Fisher scoring exists, space-time fit uses Nelder–Mead. Float32 evidence assumes nugget ≥1e-3 of sill except exponential. | M | [spatiotemporal] |
| MOB-03 / FX-08 | Point-centred cleaning default | Current speedOutlierMode default is step; point-centred is opt-in. TD-TR word times require the track-relative conversion. | H | [spatiotemporal] |
| MOB-06 | Flow-field extensions | Utilization volumes lack ridge extraction and longitude/latitude inputs. | M | [spatiotemporal] |
| MOB-07 | Dynamic Brownian bridge change points | dBBMM change-point step absent. | L | [spatiotemporal] |
| MOB-10 | Radiation scale and flow-smoothing definition | No scalable radiation variant; smoothing formula needs paper verification; 1,024-zone cap. | L | [spatiotemporal] |
| RTS-08 | A5 and conformant DGGS-JSON | A5 not started; ~3,100 reference lines need fixed-kernel boundary redesign. Zone IDs are not OGC-conformant. | L | [spatiotemporal] |
| PLAT-03 | Shared neighbour-index adoption | k reaches 256, HDBSCAN minimumSamples stays 32. Pair grid, k-means and two KDEs still need adoption; DBSCAN/KDE retain separate walks. | M | [spatiotemporal]; GPU-05 |
| PLAT-04 | Upstream resident iteration | Analysis support exists; luma.gl GPU-gated continuation remains a proposal. | M | [proposals] |
| PLAT-05 | Adapter-maximum shorthand and buffer_view | Explicit requiredLimits and subgroups are already public; only remaining gaps belong upstream. | L | [proposals] |
| PLAT-06 | External oracle fixtures | Pending R: gstat, sparr, stpp, stopp, changepoint, scanstatistics, ETAS, GpGp, CAST, phenofit/GRASS, bfast. Also pyxccd, xeofs, scikit-learn, GeoConformal, MobilityDuck and new trajectory/track/Markov fixtures. | M | [spatiotemporal]; `scripts/run-oracle-fixture.mjs` |
| PLAT-07 | SpatialBench extensions | One synthetic-zone query measured to SF5: 216.3 ms. SF10 JS arrays crash; add typed arrays, real Overture zones, aggregation-only and KNN. SF100 needs 9.6 GB positions. | L | [spatiotemporal]; `modules/analysis/test/gpu-spatial-analysis/spatial-join/spatialbench-scale-benchmark.spec.ts` |

### Contributor contracts and GPU-scale designs

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| STD-02 | Tighten the checker as baselines reach zero | Fail zero-count rules outright; improve loop/producer detection and catch duplicate WGSL declarations. Baseline has 64 findings. | M | [library]; `scripts/contributor-contract-baseline.json` |
| STD-04 | Named status ports and flag vocabulary | Per-cell codes, fractional-zonal distribution, tiled-flow/catchment, network, geometry overflow and most statistics remain. Decide output.ports.status on three packed-status classes. | H | [library] |
| STD-05 | Runtime numeric parameters | Point-cloud thresholds/viewpoint, ~20 statistics props and raster affines remain baked; local kriging defaults to baked covariance. Three parameter mechanisms coexist. | M | [library] |
| STD-06 | Raster context | Reprojection affines, patch-metric steps/area, contour/flow extent and row direction, pixel coordinate declarations and a shared context resolver remain. | H | [library]; `analysis-core/raster-grid.ts` |
| STD-07 | Published band validity and final vocabulary | Bare raster value views remain; spectral outputValidity and distance-field seedMask need decisions. Preserve Hampel/outlier flags meaning 1=bad. | H | [library] |
| STD-08 | Remaining geometry layout/provenance | Offset curve and outline use fixed stride; overlay/buffer lack per-feature source rows; line-chunk requiredCount can be a lower bound; hull/bounds overflow needs named bits. | M | [library] |
| STD-09 | Finish core ownership | Move shared utilities/raster iteration gate to core; row tiles and list ranking only when another family needs them. Decide whether validateSinglePackedView is public. | L | [library]; `analysis-core/`; `gpu-spatial-analysis/shared/` |
| STD-10 | Adopt shared WGSL without moving fixtures | Remaining solvers: local/RBF/space-time kriging, Vecchia, panel-lag determinant/inverse, eigen range, point-cloud Jacobi, Kalman. Also ~140 binary searches, word-order-dependent uint64, masks/accessors, QR, deprecated float tests and kernel wrappers. | M | [library]; `analysis-core/kernels/` |
| STD-11 | Recipe consistency | Twelve recipes remain kernel-only functions; merge two utility modules, move quantile LISA out of recipes, parallelize interpolation-diagnostic/network-scenario summaries. | M | [library] |
| STD-12 | Source/test/docs layout | Loose roots, multi-class files, old test names and split API pages remain; geospatial entry needs a coherent scope. | L | [library] |
| STD-13 | Registry consistency | Rename 14 wave-named capability IDs; check coverage/docs/site against registry, correct kepler Dorling limits and group API docs by domain. | M | [library]; `ontology/all-entry-points/`; `gpu-spatial-analysis/ontology/kepler-analysis-catalog.ts` |
| STD-14 | Packed-input policy and binding limits | 294 class files require packed input, 11 accept chunks; binding-size checks are incomplete. Decide whether joins/aggregation/density need chunking. | M | [library] |
| STD-15 | Statistic output vocabulary | Decide minimums/minimum, counts/cellCounts and validCount/count. | L | [library] |
| GPU-01 | Remaining serial data loops | Masked registration means/refinement, fractional tuples, adjacency union-find, structures pack, patch reconciliation, quantile mapping, Ang shell, fold designs and inequality. | M | [library] |
| GPU-05 | Index by default | MGWR fits need gwr-grid-index; island ties ≥15 trigger an unbounded rescan. Great-circle queries outside bounds return empty. Dense pair-statistics defaults and large-radius work remain; snapping callers lack candidateCapacity. Point-cloud grid migration to published `GPUGridIndex` needs matching cell-size, upper-bound, ordering and raster-mode contracts. | H | [library]; `gpu-spatial-analysis/spatial-regression/gwr-grid-index.ts`; `gpu-spatial-analysis/spatial-weights/gpu-island-attachment.ts`; `pointcloud-analysis/pointcloud-grid-index.ts`:76–86 |
| GPU-07 | Indexed ring/face operations | Hole-to-shell scans all rings; make-valid scans all noded segments per face. Two orientSign call sites violate the exact-helper rule. | M | [library]; `analysis-core/ring-assembly/`; `gpu-spatial-analysis/geometry-topology/gpu-make-valid.ts` |
| GPU-08 | Terrain serial designs | Depression activation (TE03), per-pit breach search, per-basin catchment statistics and quadratic coregistration rows remain. | M | [library]; `analysis-core/kernels/parallel-merge-tree.ts` |
| GPU-09 | Serial regression fits | Panel regression, MGWR global fit, SUR and panel-lag dense determinant remain. Reuse regression-row-tiles. | M | [library]; `gpu-spatial-analysis/spatial-regression/regression-row-tiles.ts` |
| GPU-10 | Network optimization | Evaluate published Louvain; replace point routing only with same-edge/predecessor support; consolidate facility solvers; rework flow, rational access and time routing. | M | [library] |
| GPU-11 | Urban/landscape scale | Use raster patches for metrics, avoid patches×cells proximity buffers, lift 64-building/256-point tessellation and visibility limits. | M | [library] |
| GPU-12 | Geometry algorithms | Default/gate level-synchronous S2 only with budget-exhaustion parity; rework arc overlay, Delaunay flips, offset components, ratio hull, segment snap and shape moments. | M | [library] |
| GPU-13 | Raster/dataframe scale | Region graph Borůvka rounds, sparse fractional candidates, skewed group moments, float32 counts (2^24 cap) and inequality remain. | M | [library] |
| DUP-01 | Consolidate overlapping contributors | Kriging, variograms, facility solvers, walking/cost distance, cells, arcs/segmentize, label/MIC, equality/predicates and resistance recipes. | M | [library] |
| DUP-03 | Cell-family parity | Complete H3/S2/Quadbin parity and one GPUCellFamily type. | M | [library] |
| UPS-01 | Upstream graph algorithms on command-graph views | Then retire local topology, components and degree. | M | [proposals]; [library] |
| UPS-02 | Offer generic algorithms upstream | Image morphology/phase correlation/GLCM, graph reachability/Brandes/SCC/flow and exact quantiles. | L | [proposals]; [library] |
| UPS-03 | Adopt published raster operators where contracts match | Band math, contrast and neighbourhood operators still have registered gaps; state which global operators cannot tile with a halo. | M | [workarounds] |
| GAP-01 | Geometry capabilities on consumer demand | Attribute-preserving overlay, per-feature buffer, overlapping dissolve, topology-preserving simplification, Voronoi polygons, splitting, antimeridian cutting and points along lines. | M | [library] |
| GAP-02 | Raster/terrain capabilities on consumer demand | QA masks, polygon clip, line/point rasterization, polygonize, zonal histogram/median/majority, pour-point snapping, stream orders and whole-DEM breach. | M | [library] |
| GAP-03 | Network capabilities on consumer demand | Edge demand, nearest POIs, street statistics, simplification and angular measures. | M | [library] |
| GAP-04 | Statistics/dataframe capabilities on consumer demand | Durbin/SLX/SDEM, GWR prediction, multi-attribute clustering, SARAR, windows, group regression and as-of joins. | L | [library] |
| GAP-05 | Point-cloud/applied capabilities on consumer demand | Voxel-centroid producer (the fork voxel downsample wrote centroids and attribute means), k-nearest >64, cloth ground filter, raster-scale FRAGSTATS, pairwise Circuitscape and least-cost corridors. | L | [library] |
| FX-17 | Finish atan2 triage | Reconcile remaining variogram, change-detection and rhumb sites with guardedAtan2 adoption. | L | [math]; `analysis-core/kernels/wgsl-common.ts`: `GUARDED_ATAN2_WGSL` |
| FX-18 | Padding rows outside rings/lines | getSnapOrigin reads row 0 when the first offset is positive. Finish triangulation/standalone segment-intersection checks and reachable grouped-hull/Delaunay padding checks. | M | `gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels.ts`: `getSnapOrigin`; [math] |
| LUM-21 | Dependency-carrying helpers as shader modules | About 30 kernels and 16 concatenation-order rules; assembly costs 5–17 ms/kernel. Do not convert 140 single-function snippets indiscriminately. | M | [usage] |
| LUM-22 | Shared stable multi-key sorts | Eight local sorts; some require a gather of the sorted low word. | M | [usage]; `analysis-core/kernels/stable-lexicographic-index-sort.ts`: `createStableLexicographicIndexSortNodes` |
| LUM-28 | Budgeted execution for long host graphs | createExecution in kepler/lenses is an unmeasured lead. | L | [usage] |
| LUM-29 | Guard scan totals | About 20 inline totals need a bound or saturated total nodes. | M | [usage]; `analysis-core/kernels/` |
| LUM-30 | Resource leak checks | No spec checks for leaks. Published `getResourceCounts` (`@luma.gl/core/test-utils`) reads `'Texture2Ds Active'`, which luma.gl 10 never increments, so a check must read `luma.stats.get('GPU Resource Counts')` (`'Textures Active'`, `'Buffers Active'`) directly. | L | [usage]; [proposals] |

### Precision

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| FP64-02 / LUM-20 | Published fp64 in point-to-cell | Integer variant has equal measured accuracy but is about 3× slower than local double-float; wait for FP64-05 or public vendor helpers. | M | [usage]; `gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl.ts` |
| FP64-03 | Optional fp64 for precision-limited parallel work | Cell measures beyond resolution 10, S2 levels 26–30, antipodal geodesics and spherical containment tolerance; CPU twins are oracles. | M | [cpu] |
| FP64-05 | Measure on a non-Apple GPU | Compare classic double-single with integer-controlled arithmetic before broader adoption. | M | [cpu]; [usage] |

### New contributors: unexecuted options

The maintainer chooses the directions and order. Proposed sequence: foundations LRN-02, the LRN-01 forest layout and second-device probes; then LRN-01, EMB-01–05, SIM-01, CART-03, CART-06 and CART-07; then LRN-03, ENC-01, SIM-02/03, CART-01 and LRN-04/05; the rest by demand. ENC-02/03 require a consumer; ENC-04 is an example-only spike.

Planning limits: at 0.1 trillion multiply-accumulates/s, fitted/per-point networks use ≤100,000 parameters and ≤5 billion multiply-accumulates/run; WGSL pretrained models use ≤2 million parameters (8 MB float32). Enforce named limits at construction, keep each binding ≤128 MiB unless checked against the device, and keep runs to tens of dispatches. Use minibatches: the 262,144-row probe needed a 281 MB binding. Second-device, real unbalanced forest, hash-grid, attention, float16 and command-graph overhead measurements remain open.

Fits publish loss history, converged and iterationLimitReached, including the iterate at the limit; use fixed-order tiles or fixed-point sums for repeatability. Models and weights are caller-supplied; no bundled weights/datasets. Record example data licences. Use fused analysis kernels, shared fitting helpers in core, and propose generic layers upstream. Proposed locations: models in dataframe analysis, fields beside interpolation, cartograms beside Dorling and CLAHE beside stretch. Verify each decoder formula and independent oracle before implementation; check required oracle availability rather than relying on a machine inventory.

Keep the research’s not-recommended candidates out of implementation; its reasons and the measurements supporting these options remain in [the research](./research/new-contributors-2026-11-11/README.md).

| ID | Item | Current limit | Priority | Source |
| --- | --- | --- | --- | --- |
| LRN-01 | `GPUForestInference`: predictions (and leaf indices) of an imported random forest or boosted model over rows or a raster stack, with a CPU function that reads the scikit-learn, XGBoost and LightGBM model formats into a node array. Oracle: scikit-learn | Not implemented; effort M. | H | [new] |
| LRN-02 | Shared fitting kernels in `analysis-core/kernels`: row-tiled gradient sums, an Adam step with a budget and a convergence status, a seeded minibatch order, coordinate encoders (Fourier, spherical harmonics) | Not implemented; effort M. | H | [new] |
| LRN-03 | `GPUNeuralField`: a continuous surface through points or a raster for interpolation at scale, gap filling and compression; Fourier features first, a hash grid with fixed-point scatter-add second; two dimensions and space-time. Documented as not more accurate than kriging and without uncertainty. Oracle: a PyTorch script | Not implemented; effort L. | M | [new] |
| LRN-04 | `GPUGaussianMixture`: expectation-maximization with full covariances and BIC. Oracle: scikit-learn | Not implemented; effort M. | M | [new] |
| LRN-05 | `GPUSelfOrganizingMap` with a GeoSOM mode | Not implemented; effort S to M. | M | [new] |
| LRN-06 | `GPUPermutationImportance`, optionally in spatial blocks, for any contributor with a predict kernel | Not implemented; effort S. | M | [new] |
| LRN-07 | `GPUGradientBoostedTrees`: histogram boosting trained in the browser (squared, logistic and quantile loss) | Not implemented; effort L. | L | [new] |
| LRN-08 | `GPUTreeShap`: contribution maps per covariate | Not implemented; effort M. | L | [new] |
| LRN-09 | An ensemble option on the fitted networks whose spread feeds the existing conformal classes | Not implemented; effort S. | L | [new] |
| EMB-01 | Decoders for TESSERA (int8 times a per-pixel scale), Presto (uint16) and float16, and the int8 encoder the docs say is missing. Verify each formula against its source first | Not implemented; effort S. | H | [new] |
| EMB-02 | More than 128 dimensions (192, 256, 384; 1,024 where the binding allows) | Not implemented; effort S. | M | [new] |
| EMB-03 | `GPURasterEmbeddingTrajectory`: cosine to a base year, year of largest step, cumulative angle across all years | Not implemented; effort S. | M | [new] |
| EMB-04 | `GPURasterEmbeddingAnomaly`: Mahalanobis distance from `GPURasterEmbeddingCovariance`, or distance to the nearest label | Not implemented; effort S. | M | [new] |
| EMB-05 | A regression output on the linear probe; mean-of-references similarity | Not implemented; effort S. | M | [new] |
| EMB-06 | Edge strength between neighbouring embeddings, as input to watershed and sieve (check the overlap with `GPURasterTexture` first) | Not implemented; effort S. | L | [new] |
| SIM-01 | `GPUGaussianRandomField`: fields from a variogram model by `GPUFFT2D` (to 1024 by 1024, since the padded grid is capped at 2048), by random spectral modes beyond that and for points, and conditioned by kriged residuals. Lifts the 4,096-row cap of `GPUConditionalGaussianSimulation`. Oracle: GSTools | Not implemented; effort S to M. | H | [new] |
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
| CART-03 | `GPURasterContrastEqualization`: CLAHE and histogram matching. Closes R7.1. Oracle: scikit-image | Not implemented; effort M. | H | [new] |
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
| PLAT-03 | HDBSCAN minimumSamples | Keep 32 or raise to 256 with rejection/audit updates. | Decision | [spatiotemporal] |
| STD-04 / STD-09 | Public contract shape | Decide output.ports.status for depression hierarchy, OPTICS and network KDE; whether validateSinglePackedView is public. | Decision | [library] |
| STD-07 / STD-15 | Library API vocabulary | Decide addToGraph/getCommandNodes, outputValidity/validity, conditionResult, iteration-cap names and statistics naming before a coordinated API change. | Decision | [math]; [library] |
| DUP-02 | Confirm spatial-k-means name | GPUSpatialKMeans is implemented; the public name remains a maintainer decision. | Decision | [library] |
| raster-fields-05 / STD-06 | Contour/flow row-direction flip | Follow affine in one coordinated caller change: 108 sites across about 57 files. | Decision | [row-direction plan](./research/math-correctness-review-2026-11-10/row-direction-flip-plan.md) |
| STD-05 | Walking-cost rounding acceptance | Required-parameters GRASS path differs by one ULP in 8/35 cells from the earlier kernel; Tobler is exact. Accept or restore. | Decision | [library] |
| DEC-VARIANCE | Raster cell/zonal variance divisor | Both documented; align only if published raster statistics specify a convention. | Decision | [math] |
| KE-TESTS | Arrow-preparation copy consumers | Repointing parked examples can enable at most four headless specs and three typechecks, but adds consumers of a registered upstream copy. | Decision | [workarounds] |
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
| Space-time kriging | `GPUSpaceTimeKriging` is reference tier, scans samples per query, maximum 1,048,576 samples. | `gpu-spatial-analysis/spatial-interpolation/gpu-space-time-kriging.ts`; [spatiotemporal] |
| FP64-04 | Keep local addition/multiplication rounding-error helpers. | [workarounds]; `analysis-core/kernels/addition-rounding-error-wgsl.ts`; `multiplication-rounding-error-wgsl.ts` |
| Published fp64 | About 48 bits, not 53, with float32 exponent range. | `modules/analysis/test/analysis-core/shader-fp64-module.spec.ts`; [cpu] |

## Belongs in luma.gl

Use [upstream-proposals.md](./upstream-proposals.md) for proposed changes and [upstream-workarounds.md](./upstream-workarounds.md) for current gaps and accepted regressions. Do not implement general compute, rendering or platform APIs here.

## Completed passes

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
