# Library review: comprehensiveness, GPU quality and standards (2026-11-09)

A review of all of `@luma-contributors/analysis` (`modules/analysis/src`: 974 files, about 317,000 lines, 398 exported classes, 39 `add*Recipe` functions) at commit `edf9ca10` plus the working tree of 2026-11-09. The question: the library was assembled in waves from many reference libraries; what has to change so that every part is a GPU-scale implementation that applies one set of standards. The work items are in [../../roadmap.md](../../roadmap.md) section 2.16. This directory is the evidence.

It complements the [spatiotemporal review](../spatiotemporal-review-2026-11-09/README.md) of the same day, which hunted defects and planned the time contract. This review does not repeat that; it measures every contributor on one scale.

## Summary

1. **Breadth is not the problem.** 437 exported contributors; 434 are named in a test, 432 in the docs and 432 on the website. Each domain has a short list of things a practitioner would miss (section "Comprehensiveness"), but no domain is thin.
2. **About one contributor in seven is not a GPU-scale design.** Reviewers gave a verdict to 429 contributors or sibling groups: 182 conform, 133 work at scale and need only contract alignment, 64 need the algorithm reworked, 30 duplicate another contributor, 20 belong upstream or elsewhere. The 64 are mostly kernels that do the real work in one GPU invocation, or search every item for every item, with a size limit that follows from that choice (32 vertices, 64 buildings, 256 rows, 4,096 events, 65,536 rows).
3. **The reworks cluster in the later "coverage expansion" wave, but not only there.** Point cloud, `terrain-processing`, `network-optimization`, `urban-morphology`, `landscape-ecology`, `point-process`, `spatial-validation`, `global-coordinates`, the loose `gpu-raster-*.ts` files and part of `remote-sensing` were written to match a reference library's fixtures at small size. Counter-examples exist both ways: older contributors with quadratic or serial steps (make-valid, ring assembly, shared paths, shape descriptors, focal statistics, six regression classes) and later ones that scale (branching flow, the watershed forest, the medial axis, the spherical polygon overlay).
4. **The library has a contract, and most of it is convention.** `analysis-core` defines status ports, capacity helpers, a spatial context and shared numerics. Adoption is uneven (table below), several contract types have fewer than five users, and `analysis-core` itself exports 16 symbols, so the contract is reached through `gpu-spatial-analysis`.
5. **The same thing is built several times.** 17 small dense solvers on the GPU, five private ways to find nearby points, seven layouts for variable-length geometry output, five raster input shapes with four no-data conventions, two kriging classes, a `GPUKMeans` that shares its name with a published luma.gl class, and four domain wrappers around the one kernel builder.
6. **There are four inventories and they disagree.** The coverage matrix lists 137 of the 437 symbols, the ontology 259 (one package only), the website catalog 432, the docs 432. The docs are organized by the wave in which code was written (`operators-*.md` against `*-expansion.md` and `coverage-*.md`), not by what it does.

The order of work that follows: write the contract down and enforce it with a script; align the 133; merge the 30; rework the 64 in order of how cheaply the serial step can be removed; make one registry the source of the inventories; then add the missing capabilities.

## Method and what was checked

- **Gates, run before the review.** `node scripts/check-upstream-first.mjs` passes. `yarn typecheck` fails with one error and `yarn lint` with two; both come from another session's uncommitted spatiotemporal fixes (`GPUSpaceTimeVariogram` now requires `gridSize`; `modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/expansion.run.ts:79` is not updated yet). `yarn test-node` was not run for this review: the morning run (7,906 passed, 1 timeout under load) is recorded in the spatiotemporal review, and the source this review reads was being edited by that other session while it ran.
- **Census.** Two scripts in [census/](./census) read every source file: `census.mjs` counts contract markers per contributor file (validation style, status types, spatial context, parameter buffers, published primitives, WGSL function names defined in more than one file, declared maxima); `crossref.mjs` checks every exported class and recipe against docs, tests, the coverage matrix, the ontology, the website, lenses, kepler and examples. The counts are regular-expression matches, not type analysis. One was wrong and is corrected below: the first status count matched type names only; reviewers found the shared field names in use where the types are not imported.
- **Code review.** Nine reviewers read the code in disjoint scopes and wrote the `code-*.md` files against one brief (contract, table columns, verdict vocabulary). They ran nothing. Each report lists the files read in full, read in part and not opened, and tags findings CONFIRMED or SUSPECTED. No reviewer read all of its scope; verdicts for unread files rest on class documentation, dispatch counts and grep, and say so.
- **Re-checked by the lead reviewer.** These were re-read in source: the float32 turn ordering in ring assembly and the absent overlay provenance; the walking-cost relaxation; the point-cloud default search path and the M3C2 scan; the single-invocation depression hierarchy; the single-invocation probit fit; the shader-baked covariance in `GPULocalKriging`; the group-moments dispatch; the single-invocation S2 polygon covering; the tessellation caps; the absence of a command-node guard outside centrality; the join candidate cut-off. Everything else is as the reviewers reported it.
- **Not done.** No benchmarks, no runtime confirmation of the suspected watchdog and compile-time effects, no web research (the reference-library comparisons are from the reviewers' knowledge of those libraries).

## Files

| File | Scope | Symbols | Conform | Align | Rework | Merge | Move |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [code-geometry-polygons.md](./code-geometry-polygons.md) | Overlay, noding, validity, buffer, triangulation, Delaunay, hulls, spherical geometry, ring assembly, exact predicates | 49 | 17 | 21 | 8 | 3 | 0 |
| [code-geometry-lines.md](./code-geometry-lines.md) | Line clip, merge, split, simplify, segmentize, linear referencing, bounds, hull groups, grids, shape descriptors | 23 | 8 | 8 | 2 | 3 | 2 |
| [code-cells-joins-density.md](./code-cells-joins-density.md) | H3, S2, Quadbin, spatial joins, neighbour search, zonal and region statistics, areal interpolation, density | 45 | 24 | 14 | 2 | 4 | 1 |
| [code-recipes-applied.md](./code-recipes-applied.md) | All 39 recipes, urban morphology, landscape ecology, point process, validation, allocation, the ontology | 61 | 30 | 13 | 14 | 3 | 1 |
| [code-statistics-models-movement.md](./code-statistics-models-movement.md) | Weights, autocorrelation, inference, regression, clustering, regionalization, interpolation, trajectories | 91 | 35 | 32 | 13 | 6 | 5 |
| [code-raster.md](./code-raster.md) | All of `raster-analysis`, focal statistics | 40 rows (about 55) | 12 | 14 | 8 | 4 | 2 |
| [code-terrain-pointcloud.md](./code-terrain-pointcloud.md) | All of `gpu-terrain` and `pointcloud-analysis` | 56 | 29 | 12 | 9 | 1 | 5 |
| [code-network.md](./code-network.md) | All of `gpu-network`, `network-topology.ts` | 38 | 15 | 10 | 6 | 3 | 4 |
| [code-core-dataframe.md](./code-core-dataframe.md) | `dataframe-analysis`, `analysis-core` contracts and kernels, entry barrels, naming | 26 | 12 | 9 | 2 | 3 | 0 |
| Total | | 429 | 182 | 133 | 64 | 30 | 20 |

Verdicts: **conform** follows the contract and scales; **align** scales, but the contract or style differs; **rework** the algorithm is not a GPU-scale design; **merge** overlaps another contributor; **move** belongs upstream in luma.gl or elsewhere. Each report's section 2 has one row per contributor with its parallelism, work, iteration scheme, limits, parameters, inputs, context, status and validation.

## How evenly the contract is applied

Counts of source files that export a contributor class (379 files in the seven analysis directories). "Shared status names" means the file uses `requiredCount`, `candidateOverflow`, `iterationLimitReached`, `invalidCount` or `uncertainCount`. The two validation columns come from the first census pass, which also counted recipe files (251, 36, 46, 44, 23, 9 and 2 files).

| Directory | Class files | Declares `implements GPUCommandNodeProducer` | `spatialContext` prop | Imports published `gpu-raster` | Shared status names | Has any `status` prop | Files with bare `assert` | Files with `validateCapacity` | Accepts chunked input | Per-frame `parameters` view |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gpu-spatial-analysis` | 221 | 216 | 107 | 7 | 77 | 136 | 40 | 133 | 7 | 69 |
| `gpu-network` | 33 | 33 | 7 | 2 | 8 | 20 | 8 | 22 | 2 | 12 |
| `gpu-terrain` | 46 | 46 | 0 | 44 | 3 | 13 | 11 | 5 | 0 | 28 |
| `raster-analysis` | 44 | 43 | 2 | 30 | 8 | 22 | 19 | 7 | 0 | 21 |
| `dataframe-analysis` | 24 | 24 | 9 | 0 | 4 | 7 | 1 | 17 | 2 | 20 |
| `pointcloud-analysis` | 9 | 0 | 0 | 0 | 0 | 8 | 9 | 0 | 0 | 0 |
| `geospatial-analysis` | 2 | 2 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 1 |

What the table and the reports show:

- **Validation.** 516 bare `assert(` calls with no message sit in the expansion families (remote sensing 57, point process 44, terrain-processing 38, network-optimization 32, landscape ecology 29, urban morphology 23, point cloud 9 of 9 files). The rest of the library uses named-limit helpers and messages (2,311 `throw`, 3,436 `validate*` calls), and the kernel audit pins those messages. AGENTS.md says "Keep shipped runtime validation minimal. Put explanatory assertion messages in comments", which describes the minority style. The rule and the practice need to be made one (decision 1).
- **Status.** 207 of 379 class files have a `status` prop; 100 use the shared field names; about 60 import one of the shared status or output types. The rest use per-row bit flags with private meanings ("bit 1 insufficient, bit 2 truncated"), packed header words, or nothing. Siblings disagree: line clip folds candidate overflow into `overflow`, line split reports `candidateOverflow`; the joins fold candidate, BVH and output overflow into one flag. `GPUCandidateStatusPort` and `GPUApproximationStatusPort` have no user.
- **Space.** In `gpu-spatial-analysis`, 34 files that take positions or geometry have no `spatialContext`, among them seven that measure distance (local kriging, local RBF, natural neighbour, conditional simulation, Kalman, restriction, similarity search) and the whole spherical family. Terrain and raster use the raster grid instead, which is right, but through six different routes: a per-frame packer, the affine baked into the shader, a separate extent that can disagree with the grid, pixel units only, a private rectilinear grid, or nothing. Nine newer terrain classes measure in grid units and ignore `cellSizeMode`, so they return degrees on a geographic DEM.
- **No-data.** Masks are `mask` in 97 files and `validity` in 43, with about 60 other names. Rasters carry no-data four ways (validity view with sentinel, NaN plus sentinel, NaN plus validity, label sentinels). Twenty-nine older terrain classes take a `GPURasterBand`; nine newer ones take a raw float32 view in which NaN is the only no-data marker, so a `-9999` cell is terrain.
- **Parameters.** Per-frame values reach kernels three ways (`parameters` view plus a packer, `GPUParameterBuffer`, `defineGPUSpatialParameterSchema`), and many numeric props are written into the shader as literals, so changing them recompiles: the covariance model in `GPULocalKriging`, the point-cloud viewpoint and thresholds, about twenty statistics props, the raster affine.
- **Input size.** 294 class files require one packed view; 11 accept chunked vectors. The row ceiling of almost every contributor is therefore the storage binding size, and only some check it at construction. That is a defensible policy, but it is not stated anywhere.
- **Output.** Variable-length geometry output has seven layouts in the line family alone, and `status.count` counts vertices in one and pieces, runs, chains or rows in the others. Polygon overlay and buffer output carry no source-row provenance (`gpu-polygon-overlay.ts:1266` writes the output index as the source ID).
- **Recipes.** Three shapes share the name: 27 builder recipes with per-stage status, classes that only compose other classes, and 12 urban and landscape `add*Recipe` functions that are raw kernels whose outputs are graph-owned transient views the caller cannot read.
- **Package surface.** The root entry exports only `computation-record`. `analysis-core` exports 16 symbols, three of them marked `@internal`; the contracts are re-exported from `gpu-spatial-analysis`, and `GPUParameterBuffer` and `GPUCompactOutput` from three domain entries. A third party cannot write a contributor from `analysis-core`. Five entry barrels leak `@internal` builders. The core `validatePackedView` shadows a published luma.gl function of the same name with different behaviour (315 imports against 14).
- **Layout and names.** Three directory naming schemes (`gpu-x`, `x-analysis`, `gpu-x-analysis` in the docs); 11 loose `gpu-raster-*.ts` files and 5 loose network files at package roots; 10 files with more than one class; test directories named after the luma.gl branch (`gpu-dataframe`, `gpu-raster`, `gpu-project`); `gpu-spatial-analysis/shared` is a second core. Class names, `Props` suffixes and verb-first functions are consistent (392 of 392 classes carry the `GPU` prefix).

## GPU quality: the recurring patterns

Every item below is in a domain report with `file:line`. Paths are relative to `modules/analysis/src`.

**1. The work is done in one invocation.** A kernel dispatched once loops over every row, cell or pair.

- Whole algorithms: depression hierarchy (`gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts:130-185`), Louvain, maximum flow, time routing, rational access, S2 polygon covering (`gpu-s2-polygon-covering.ts:107`), spherical arc overlay, offset-curve components, the cross-validation fold designs, the probit, regime, panel and instrumental-variable fits (`gpu-probit-regression.ts:177`), region-graph merging, bridges and articulation points.
- Finishing steps inside otherwise parallel contributors: convergence tests that scan the grid (`gpu-raster-reconstruction.ts:125`, `gpu-raster-skeleton.ts:101`, `gpu-raster-walking-cost.ts:166`), registration peak searches over up to 4.2 million cells, compaction and counting in stream network, coregistration, SMRF and the statistical filter, status counting in seven recipes.

The second group is the cheapest work in this review: each is a `GPUReduction`, `GPUCompaction` or scan-total call, primitives that about 90 contributor files already use.

**2. Every item is tested against every item, next to an index that exists.** Point-cloud neighbourhoods and ICP default to brute force and the M3C2 change contributor has no index at all (`gpu-pointcloud-change.ts:165`); the one-million-point figures in the coverage matrix were measured with an optional grid passed in, and SMRF at that size on a grid capped at 256². Also brute force: urban ray casting (a segment BVH exists in `segment-intersection/segment-bvh.ts`), MGWR, `GPULocalKriging`, conditional simulation, OPTICS, fuzzy contiguity, island attachment, network snapping, cross-variogram and marked point statistics (16,384 rows), make-valid's face scan, ring assembly's hole-to-shell step, landscape metrics (patches × cells), inhomogeneous K on the plane and on networks. Five private point-search paths exist beside `GPUNeighborSearch`, which none of them uses.

**3. Iteration is unrolled with no budget, or relaxes one cell per round.** Only centrality estimates and caps its command-node count. Cost table, cost matrix, network K, location allocation (up to 1,000 swap rounds), Delaunay (up to 4,096 steps of about 9 nodes), Douglas-Peucker (1,024 rounds of 5) and the sparse resistance solver (4,096 iterations of 5) grow the same way with no guard; a 100,000-node cost table works out to about 220,000 nodes (computed from the code, not run). Walking cost, reconstruction, fill holes and skeleton run one kernel per round, advance one cell per round, and publish an all-no-data result unless they converged, which a realistic grid will not do in the default 64 or 128 rounds. The gated, tiled relaxation in `raster-analysis/cost-distance/raster-relaxation.ts` already solves this for cost distance.

**4. Order is restored quadratically after indexing.** KDE, IDW and the GWR grid restore point order inside each grid cell with an O(m²) kernel (`gpu-kernel-density.ts:758`, `gpu-inverse-distance-weighting.ts:290`, `gwr-grid-index.ts`); on clustered data that dominates. A stable sort removes it.

**5. Dispatch ignores occupancy.** Group moments dispatch `capacity × 256` invocations whatever the number of groups, one workgroup per group (`group-statistics-moments.ts:354-372`); patch proximity allocates two patches × cells buffers; fractional coverage builds a dense zone × cell candidate grid capped at 4,194,304.

**6. Exactness stops one step short.** Planar overlay, buffer and precision reduction run on an exact integer lattice with exact orientation tests, but ring assembly, which they all use, chooses the next edge at a shared vertex by a float32 turn key (`analysis-core/ring-assembly/gpu-segment-ring-assembly.ts:804`, `:823-829`). Whether nearly parallel long edges can produce a wrong ring is suspected, not shown. Line noding and polygonize without `snapRounding` are float32 with tolerance matching; the spherical family is float32 throughout.

**7. Published luma.gl pieces are bypassed.** Nothing imports `@luma.gl/graph/gpu-pointcloud`, `gpu-dggs`, `gpu-h3`, `gpu-vector-search`, `gpu-dataframe`, `gpu-crossfilter` or `gpu-timeseries`; `gpu-graph-algorithms` is used in two files through a cast (`analysis-core/network-topology.ts:231-244`). The raster tile-halo, tile-cache and cross-tile components are used nowhere although tiling is handled per contributor. Some of this is justified (the published graph algorithms need physical buffers, not graph views), and that is a luma.gl pull request, not a local copy.

## Duplication

| What | How many | Where | Converge on |
| --- | --- | --- | --- |
| Small dense solvers in WGSL | 17 (3 Cholesky, 6 elimination or LU, 5 Jacobi, 1 QR, 1 power iteration, 1 more on the CPU) | listed in `code-statistics-models-movement.md` section 5.1 | one `dense-solve` module in `analysis-core` with a relative pivot tolerance; this also closes FX-09 and FX-10 |
| Nearby-point search | 5 private paths plus 3 k-nearest front ends | DBSCAN grid walk, HDBSCAN join, `grid-neighbor-scan.ts`, `gwr-grid-index.ts`, trajectory grids; point-cloud grid index duplicates published `GPUGridIndex` | one candidate interface (roadmap PLAT-03, widened) |
| Kriging | 2 classes | `GPUKriging` (indexed, per-frame model, tolerance 1e-4), `GPULocalKriging` (brute force by default, baked model, 1e-5, no context) | one class |
| Variable-length geometry output | 7 layouts | line family | one layout, one meaning of `status.count`, one provenance rule |
| Raster input and no-data | 5 shapes, 4 conventions | `raster-analysis`, `terrain-processing`, point-cloud rasters (row 0 at minimum y, the opposite of terrain) | published `GPURasterBand` with validity |
| Raster iteration | 5 schemes | cost distance, distance field, fill no-data, reconstruction and skeleton, watershed | the gated relaxation and jump flooding |
| Noding | 2 planar paths plus network noding | `GPULineNoding` (float), `GPUSnapRoundedNoding` (exact, `@internal`) | snap-rounded as the default, float as an option |
| Kernel-builder wrappers | 4 wrappers and about 100 local binding lambdas over `createWGSLKernelNode` | point cloud, raster, terrain, recipes | binding and accessor helpers in core |
| Float tests in WGSL | 7 names for 2 predicates; inlined in 43 more files; NaN bit pattern in 104 | `wgsl-common.ts:17-60` and callers | one module |
| Facility location | 3 solvers over one cost matrix | `location-allocation`, `network-optimization` | `location-allocation` |
| Hilbert keys, connected components, k-means, variograms, envelopes, standardization, scales | 2 each | see reports | one each; `GPUKMeans` also collides with published `@luma.gl/graph/gpu-vector-search` |

Three cell systems offer uneven operations: H3 has nearly everything but disk and ring stop at k ≤ 8; Quadbin has no cell area or grid distance; S2 has parent, one level of children, edge neighbours and area, no k-ring, compaction or aggregation, and a cover that does not scale. The matrix is in `code-cells-joins-density.md` section 4.1.

## Inventories, documentation and packaging

- **Ontology.** 261 operators in 49 capabilities, all from `gpu-spatial-analysis`. 168 classes and 4 recipes in the other entry points have no entry. 13 capability IDs are named after the wave (`expansion-*`, `coverage-*`); 200 operators have `semantics: 'unspecified'`; 195 have `dimensionality: 'unspecified'`; `GPUParameterBuffer` is listed as an operator.
- **Coverage matrix.** 142 rows naming 137 symbols. 300 exported contributors are "unassessed" by its own definition. Several rows overstate: PC02.voxel names a voxel merge whose input no contributor produces; the point-cloud scale claims are from a non-default path; NW05 documents 64 demand rows where the real limit is facilities^demand ≤ 2^20.
- **Docs.** Every contributor is mentioned; 388 of 437 have a heading. Point cloud has none of its 10. Pages are split by wave: `operators-*.md` beside `coverage-expansion.md`, `statistical-expansion.md`, `interpolation-expansion.md`, `expansion-recipes.md`, `expansion-lenses.md`.
- **Tests.** Near-complete by name. Directory names and seven empty directories are left from the luma.gl branch (already noted in the spatiotemporal review).

## Comprehensiveness

What a practitioner coming from the reference libraries would look for and not find. Items already on the roadmap are not repeated. Detail, reuse and effort are in each report's section 6.

| Domain | Missing, most used first |
| --- | --- |
| Polygons | Pairwise overlay that keeps attributes from both inputs (GeoPandas `overlay`); per-feature buffer (today all features merge into one surface); dissolve by attribute for overlapping polygons; topology-preserving simplification of single polygons; Voronoi polygons per site |
| Lines | Split by a second geometry; topology-preserving simplification of line networks; antimeridian cutting; points at an interval along lines; line substring and measured geometry |
| Cells, joins, density | Zonal statistics with fractional coverage on vector zones, and zonal median, percentile, majority; exact polygon areal interpolation; S2 at parity with H3; disk and ring beyond k = 8; line density with a search radius |
| Raster | QA bit bands to masks; clip by polygon; rasterize lines and points with burn values; polygonize a categorical raster; zonal histogram and tabulate area |
| Terrain | Pour-point snapping; automatic whole-DEM least-cost breaching; stream orders (Shreve, Horton); D-infinity angle output |
| Point cloud | Voxel centroid and attribute-mean downsample; true k-nearest above 64 with no radius cap; cloth-simulation ground filter |
| Network | Origin-destination demand loaded onto shortest paths (edge volumes); Pandana-style aggregation and nearest points of interest; OSMnx street statistics; graph simplification and intersection consolidation; space-syntax angular measures |
| Statistics and models | Spatial Durbin, SLX and SDEM; GWR prediction at unsampled locations; clustering on many attribute columns (k-means, k-medoids, Gaussian mixture); SARAR and regimes for lag and error models; more rate smoothers |
| Applied | The core FRAGSTATS set at raster scale; Circuitscape pairwise and cumulative modes; least-cost corridors; segment angular analysis; null point-pattern simulation and `ppm` |
| Dataframe | Window functions per group; rolling and time windows; per-group correlation and regression; multi-column and weighted group-by; as-of and interval joins |

Candidates to leave the library (20 "move" verdicts): generic image operators that belong in luma.gl `gpu-raster` (rank filter, reconstruction, fill holes, skeleton, medial axis, phase correlation, the GLCM core); generic graph algorithms (the reachability engine, Brandes centrality, SCC, Louvain, flow, components, coarsening); graph and outline visualization helpers (`GPUAdjacencyMatrix`, `GPUOutlineGeometry`, `GPUReliefBlend` and the colour stage of relief shading); `GPURegionStatisticsReadback`; the CPU `fitVariogramModel` in the public API; the Arrow `graph-support` files that only kepler uses. Terrain illumination (horizon, sky view, openness, shadows, insolation) is analysis and stays.

## The standard to converge on

Proposed as `dev-docs/contributor-contract.md` (roadmap STD-01), enforced by a script (STD-02) grown from `census/census.mjs`.

1. **Shape.** A contributor is a class that declares `implements GPUCommandNodeProducer`, one per file, named after the file. A recipe is a function that composes contributors through `RecipeBuilder`, writes caller-owned outputs and reports per-stage status. Raw kernels do not live in recipes.
2. **Parallel by construction.** No kernel loops over a data-sized range in one invocation. Reductions, counts, argmax, convergence tests and compaction use the published primitives. A contributor that cannot meet this is registered with `tier: 'reference'` and its maximum size, and says so in its documentation.
3. **Candidates through an index.** Pair and neighbour candidates come from the grid index, the BVH or a sort. Brute force is an explicit option with a stated cap, never the default.
4. **Iteration with a budget.** Rounds are gated; any contributor whose node count depends on a prop exposes `estimatedCommandNodeCount` and uses one shared guard. An unconverged result is published with `converged = 0`, not withheld.
5. **Per-frame parameters.** Numeric props that do not change buffer shapes are read from a parameters view with a packer. Only topology is compiled in.
6. **Declared context.** Vector contributors take `spatialContext`. Raster and terrain contributors take the published raster band and grid, honour `cellSizeMode`, and never bake the affine. Time follows TC-01.
7. **One no-data rule.** Input `mask` (nonzero keeps), output `validity`; rasters use band validity. NaN is a value-level null, not the only marker.
8. **One output layout** for variable-length geometry, with `status.count` in output rows, `requiredCount`, `overflow`, and source rows carried through.
9. **Named status.** Scalar status uses the port types from `analysis-core`; candidate overflow is reported separately from output overflow; per-row flags come from one shared bit vocabulary.
10. **Named limits.** Construction-time checks use the helpers with an id-prefixed message and a named limit, including the storage binding size. AGENTS.md is changed to say this.
11. **Shared numerics.** One dense-solve module, one float-test module, one geodesic module, Philox for every random number, the fixed-point accumulator for sums that must be order-independent.
12. **One registry.** Every exported contributor has a registry entry (the ontology, extended to all entry points) with tier, limits, context and reference. The coverage matrix, the docs index and the website catalog are checked against it.
13. **`analysis-core` is the public home of the contract.** Domain entries do not re-export it; `@internal` symbols are not exported.

## Plan

Roadmap section 2.16 has the rows. In order:

| Phase | What | Size |
| --- | --- | --- |
| 0. Decide | The five decisions below; write the contract; land the checker in report-only mode | days |
| 1. Align | The 133 "align" verdicts: validation, status names, mask names, declared context, `analysis-core` surface, barrel leaks, shared WGSL modules, recipe shape, file layout. No behaviour change; the existing 7,906 tests are the safety net | weeks, parallel by directory |
| 2. Merge | The 30 "merge" verdicts and the duplicate table above | weeks |
| 3. Rework | The 64 "rework" verdicts, cheapest first: (a) single-invocation finishing steps; (b) quadratic order restore; (c) default to an index; (d) raster iteration onto the shared relaxation and node-budget guards; (e) algorithm redesigns (terrain-processing, network-optimization, urban and landscape, serial regression fits, S2 coverings, make-valid) | months; (a) to (d) are days to two weeks each |
| 4. Inventory | Registry over all entry points; docs regrouped by domain; coverage matrix checked against the registry; corrected scale claims | weeks |
| 5. Extend | The comprehensiveness table, by consumer demand | open |
| Upstream | The "move" candidates and the graph-view inputs for `gpu-graph-algorithms`, as luma.gl pull requests | independent |

Phase 3(e) is where judgement is needed: some reference-sized contributors (exhaustive facility selection, exact conditional simulation, the visibility graph) are honest about being small. Labelling them `reference` in the registry is a legitimate end state; presenting them beside million-row contributors with no label is not.

## Decisions for the maintainer

1. **Validation rule.** Change AGENTS.md to match the majority practice (named-limit helpers with messages), or strip messages library-wide? Recommended: change AGENTS.md; the kernel audit already depends on the messages.
2. **Reference tier.** Accept a labelled `reference` tier for small exact algorithms, or require every public contributor to scale? Recommended: accept the tier, rework wherever a known parallel formulation exists.
3. **Breaking changes.** Phases 1 and 2 rename props, outputs and entry points. The package is `0.1.0-alpha.0` with lenses, kepler and the website as its consumers. Recommended: break now, in one pass per directory.
4. **Upstream moves.** Which "move" candidates go to luma.gl now, and which stay until luma.gl accepts graph views in its graph algorithms?
5. **Packed-only input.** State it as the policy (with binding-size guards everywhere), or plan chunked input for the high-volume contributors (joins, aggregation, density)?

## Limits of this review

- Counts in the tables are pattern matches. They locate where to look; the per-contributor tables in the reports are the judgement.
- No reviewer read every file in scope. Each report says what it did not open; the largest unread areas are WGSL kernel bodies in the older terrain, network and join families, which the reviewers judged from documentation and dispatch structure.
- Nothing was run. The suspected effects (device timeouts in single-invocation kernels, compile time of large unrolled graphs, wrong rings from float32 turn ordering, non-convergence of label propagation on long road networks) need a test each before they are treated as defects.
- The working tree was being edited by another session during the review, so line numbers in the calendar, temporal-reduction, variogram, permutation, kriging and map-matching files may have moved.
