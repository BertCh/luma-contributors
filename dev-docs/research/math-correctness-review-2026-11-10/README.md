# Math correctness review (2026-11-10)

A review of the math in `@luma-contributors/analysis` (`modules/analysis/src`, about 380,000 lines) at commit `74dddcf5` plus the working tree of 2026-11-10 (336 uncommitted files from other sessions). Three questions: is the math right, are standards applied consistently, and does the library do things the luma.gl way, with luma.gl's official implementation where one exists. The review itself changed no source file.

**The findings were then fixed the same day.** The plan and the outcome are [../../roadmap.md](../../roadmap.md) section 12: 161 of the 166 findings fixed and 5 deferred, an independent review of the fixes ([fix-review-statistics.md](./fix-review-statistics.md), [fix-review-models.md](./fix-review-models.md), [fix-review-geometry.md](./fix-review-geometry.md), [fix-review-fields.md](./fix-review-fields.md), [fix-review-callers.md](./fix-review-callers.md)), and one list of every behaviour change for callers ([behaviour-changes.md](./behaviour-changes.md)). Each report below ends with a "Resolution" section saying what was done about each finding. The text of this README and of the reports above their Resolution sections describes the code as it was reviewed, before the fixes; line numbers there are of that state.

**A closing pass followed on 2026-11-11** (roadmap section 12.6 and 12.7), with the reference libraries the first pass lacked. It closed the open list of roadmap 12.5 and four of the five deferred findings, and found further defects on the way. Each domain report has a "Closing pass (2026-11-11)" subsection in its Resolution section. New files from that pass:

- [fixtures-pinned-rerun.md](./fixtures-pinned-rerun.md): every fixture the first pass made on system Python or by hand, rerun on the pinned set.
- [shader-compiler-findings.md](./shader-compiler-findings.md): the lost-table fault reduced to the Apple GPU back end, the fast-math behaviours, a library audit, and bug report drafts (not filed).
- [row-direction-flip-plan.md](./row-direction-flip-plan.md): the plan for the one deferred finding that remains (raster-fields-05).
- Independent reviews of the closing fixes: [fix-review-closing-regression.md](./fix-review-closing-regression.md) (with its re-review), [fix-review-closing-regression-codex.md](./fix-review-closing-regression-codex.md), [fix-review-closing-geometry.md](./fix-review-closing-geometry.md) (with its re-review), [fix-review-closing-fields.md](./fix-review-closing-fields.md).
- [behaviour-changes.md](./behaviour-changes.md) has a final section "Closing pass (2026-11-11)".

It complements the [library review](../library-review-2026-11-09/README.md) (GPU scale and contract shape) and the [spatiotemporal review](../spatiotemporal-review-2026-11-09/README.md) (defects in statistics, space-time and movement) of the day before. It does not repeat them.

## Summary

1. **The math is mostly right.** The seventeen math reports list several hundred contributors and helpers whose formulas were traced against the reference method and found correct (each report's section 3). No shared numeric in `analysis-core` is wrong on ordinary input.
2. **Ten findings are wrong on ordinary input, and 36 more on a plausible boundary input.** 166 findings in all across the math scopes, a few of them the same defect seen from two scopes. The ten are in the first table below.
3. **The most common cause is float32, not a wrong formula.** Raw (uncentred) normal equations, variance by a difference of sums, products that overflow near 3.4e38, absolute coordinates without a local origin, a `floor(v + 0.5)` grid key, and exact equality between two rounded distances account for most of the precision and edge-case rows.
4. **The tests cannot see most of it.** Many formulas are pinned only by a CPU copy of the same formula, and the independent fixtures use integer coordinates, order-1 predictors, power-of-two bin counts and a `-9999` sentinel. Each report's section 6 says which contributors have an independent oracle.
5. **Standards are uneven in a small number of repeated ways** (row direction of a raster, interval closure, the no-data test, self-loops, the shared numerics that nobody imports). They are listed under "Standards" below, each as one decision.
6. **luma.gl: 36 official symbols are delegated to, 27 operations are re-implemented, and two exported names collide with official ones.** Where both were compared, the math agrees except for the grid cell mapping, the contour saddle rule, histogram equalization and the percentile rank. No private import exists; `node scripts/check-upstream-first.mjs` passes.

## Method and what was checked

- **Eighteen reviewers** (Claude Fable 5.2), one per scope, read-only, each with the same brief: read the WGSL kernel bodies and parameter packers, compare with the cited reference, and mark a finding CONFIRMED only with a hand derivation on a concrete input, a pure-Node float32 replica of the kernel arithmetic, or a spec that pins the wrong value. Everything else is SUSPECTED. Scratch scripts were kept outside the repository and are not preserved here.
- **No reviewer ran a kernel on a GPU or ran a spec.** Confirmed means confirmed by arithmetic. A float32 replica assumes correctly rounded operations; a real device may fuse multiply-adds or round division differently, so the exact failing values can move while the defect stays.
- **Reference libraries were mostly not installed** (spreg, mgwr, spatstat, spdep, Shapely, scikit-image, R packages). Statements about their behaviour come from committed fixtures or from the reviewer's knowledge of the source. The reports say which.
- **Re-read by the lead reviewer in source:** the rows marked "yes" in the tables below. For those the cited lines do what the report says. The lead did not re-run the replicas.
- **Coverage is partial.** Every report's section 1 lists files read in full, in part and not opened. The gaps that matter are under "Not covered".

## Reports

| Report | Scope | Findings | Wrong on ordinary input | Wrong edge case | Precision | Inconsistent | Doc mismatch | Untested |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [regression.md](./regression.md) | `spatial-regression` | 14 | 0 | 3 | 3 | 5 | 3 | 0 |
| [spatial-statistics.md](./spatial-statistics.md) | Weights, autocorrelation, inference, scan statistics, segregation | 9 | 0 | 1 | 1 | 3 | 4 | 0 |
| [interpolation-density.md](./interpolation-density.md) | Kriging, IDW, RBF, KDE, validation, scoring, transport | 12 | 0 | 5 | 1 | 2 | 3 | 1 |
| [point-process-clustering.md](./point-process-clustering.md) | Point process, space-time tests, change points, clustering, regionalization | 8 | 0 | 1 | 1 | 3 | 3 | 0 |
| [trajectories-lines.md](./trajectories-lines.md) | Trajectories, track similarity, the line family | 9 | 1 | 2 | 1 | 1 | 4 | 0 |
| [applied-recipes.md](./applied-recipes.md) | Allocation, accessibility, landscape, urban morphology, recipes | 6 | 1 | 1 | 1 | 1 | 2 | 0 |
| [polygon-topology.md](./polygon-topology.md) | Overlay, noding, validity, Delaunay, coverage, buffer | 12 | 0 | 5 | 2 | 2 | 3 | 0 |
| [geometry-measures.md](./geometry-measures.md) | Area, geodesics, hulls, bounds, grids, datum grid | 11 | 3 | 2 | 3 | 1 | 2 | 0 |
| [cells-joins.md](./cells-joins.md) | H3, S2, Quadbin, HEALPix, joins, zonal, areal interpolation | 5 | 1 | 1 | 0 | 2 | 1 | 0 |
| [network-paths.md](./network-paths.md) | Shortest paths, isochrones, centrality, network K | 10 | 1 | 1 | 2 | 4 | 2 | 0 |
| [network-flows.md](./network-flows.md) | Flow aggregation, map matching, bundling, noding, network recipes | 12 | 1 | 2 | 2 | 3 | 3 | 1 |
| [terrain-hydrology.md](./terrain-hydrology.md) | Flow, fill, streams, watersheds, features, geomorphons | 7 | 0 | 1 | 2 | 1 | 3 | 0 |
| [terrain-surface.md](./terrain-surface.md) | Slope, curvature, illumination, horizon, relief, decode | 10 | 1 | 3 | 0 | 2 | 3 | 1 |
| [raster-imagery.md](./raster-imagery.md) | Time series, remote sensing, stretch, algebra, zonal, embeddings | 12 | 0 | 1 | 3 | 5 | 1 | 2 |
| [raster-fields.md](./raster-fields.md) | Cost distance, isolines, sampling, distance field, flow texture, rasterization | 8 | 0 | 3 | 1 | 3 | 1 | 0 |
| [dataframe.md](./dataframe.md) | `dataframe-analysis` | 10 | 1 | 2 | 3 | 1 | 2 | 1 |
| [core-numerics.md](./core-numerics.md) | `analysis-core`, `pointcloud-analysis` | 11 | 0 | 2 | 2 | 2 | 4 | 1 |
| Total | | 166 | 10 | 36 | 28 | 41 | 44 | 7 |
| [luma-official.md](./luma-official.md) | Library-wide comparison with published luma.gl | 13 | | | | | | |

Paths in the tables below are relative to `modules/analysis/src`.

## Wrong on ordinary input

| ID | Where | What happens | Re-read by lead |
| --- | --- | --- | --- |
| applied-recipes-01 | `gpu-spatial-analysis/urban-morphology/street-profiles.ts:413` | A facade is visible only if two float32 distances computed by different routes agree within an absolute `1e-6`. The replica rejects 59% of unobstructed facades at coordinates of 1 to 50 and nearly all at projected-metre scale, so facade lengths read 0. The fixture uses integer coordinates. | yes |
| geometry-measures-06 | `gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:173-177` | The authalic-to-geodetic series has `17 e^4/360` where Snyder has `23 e^4/360`. Ellipsoidal polygon centroid latitudes are up to 5 m off. The test twin copies the constant. | yes |
| geometry-measures-02 | `gpu-spatial-analysis/geometry-measures/rhumb-wgsl.ts:113-117` | The rhumb midpoint halves longitude and latitude. Longitude is linear in Mercator latitude, not latitude, so the point is off the rhumb line (1.16 km London to Paris, 140 km New York to London). The oracle pins this on purpose: change the name or the formula. | yes |
| geometry-measures-03 | `gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull.ts:237-285` | Parallel erosion rounds do not give the documented GEOS order: 351 of 900 random 28-point runs end with a different (valid) hull. Rests on a CPU replica of the round rule. | no |
| cells-joins-01 | `gpu-spatial-analysis/spatial-join/spatial-predicate-wgsl.ts:586-603` | Polygon `contains` and `within` return true for a polygon that exactly fills a hole of the other. The hole check runs only when a float32 edge midpoint lands exactly on the boundary. Wrong for 220 of 400 random triangular holes in the replica. `engine: 'auto'` picks this kernel for small polygons. | yes |
| dataframe-01 | `dataframe-analysis/composite-indicators/gpu-composite-score.ts:691-713` | The PCA power iteration starts at `(1,...,1)/sqrt(d)`, an exact eigenvector of every 2 x 2 correlation matrix. Two negatively correlated indicators return the minor component (2% explained instead of 98%). The shared eigen solver is not used. | yes |
| network-flows-01 | `gpu-network/gpu-network-inhomogeneous-k.ts:247-262` | The Ang shell count uses exact float32 comparisons, so the edge into the destination is counted as an interior crossing on 32 to 44% of pairs; K is 1 to 3% low on an irregular grid. Every fixture has integer edge lengths. | yes |
| network-paths-01 | `gpu-network/network-isochrones/network-isochrones-passes.ts:186-210` | In `'max'` mode a two-way road keeps the worse direction, so the start node is the most expensive pixel (20 for a true 0). The doc sentence is literally true; the surface is not an isochrone. | yes |
| terrain-surface-01 | `gpu-terrain/terrain-illumination/gpu-solar-irradiance.ts:536` | Diffuse insolation is scaled by the sky-view factor `GPUTerrainHorizon` produces, `1 - mean(sin h)`. An isotropic sky needs `mean(cos^2 h)`. A uniform 30 degree horizon gives 33% too little diffuse energy. | arithmetic only |
| trajectories-lines-01 | `gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-outlier-flags.ts:241-243` | The acceleration bit is documented as marking a displaced fix. It needs excess acceleration into and out of a row, which a single displaced fix does not have; it fires on the first good row after a sustained jump. Either the rule or the label is wrong. | yes |

## Edge cases and precision worth fixing first

Chosen from the 64 edge-case and precision rows because the input is ordinary data at a realistic size or location and the status says success.

| ID | Where | What happens | Re-read by lead |
| --- | --- | --- | --- |
| spatial-statistics-01, -02 | `gpu-spatial-analysis/global-spatial-statistics/gpu-global-spatial-statistics.ts:1517-1522`, `:1476` | General G, Moran and Geary variances overflow float32 (`denominator * denominator`, `sumSquares * sumSquares`). 40,000 values near 300,000 give NaN z and p. Pre-scaling the values fixes all three. | yes |
| regression-02, -03, -04 | `gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression.ts:176-197`, `gpu-spatial-error-maximum-likelihood.ts:676-688`, `gpu-spatial-panel-lag-maximum-likelihood.ts:220-227` | Raw, uncentred normal equations solved in float32. Slope error 0.8% for a "year" predictor, 11% for elevation 1500 +- 3, 31% for a UTM northing, with status 0. | first file |
| regression-05 | `gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression.ts:369-377` | GTWR publishes AICc when `n - tr(S) - 1 <= 0`, where it is hugely negative, so a minimum-AICc bandwidth search picks the degenerate bandwidth. GWR returns NaN there. | yes |
| interpolation-density-01 | `gpu-spatial-analysis/point-density/point-density-kernels.ts:279-292` | Weighted sums use the `GPUGridIndex` cell mapping; counts use `GPUGridBinning`. When bounds straddle zero a point can put its count in one cell and its weight in the next. | yes |
| interpolation-density-02 | `gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts:154` | `hash(block ^ seed) % foldCount`: a small seed only relabels blocks, so the same fold is empty for every seed 0 to 7 with 8 blocks and 5 folds. | yes |
| point-process-clustering-01 | `gpu-spatial-analysis/change-points/gpu-location-change-points.ts:409`, `:414` | The variance floor is `1e-11` in data units, though the comment says relative. The same series finds its change point in units of 1 and none in units of 1e-6. | yes |
| raster-imagery-01 | `raster-analysis/raster-algebra/gpu-raster-conditional.ts:173-177`, `gpu-raster-arithmetic.ts:131-133` | The no-data sentinel test is applied to constants. With `noDataValue: 0`, `Con(x > 10, 1, 0)` is NaN for every false cell. | yes |
| raster-imagery-02 | `raster-analysis/raster-time-series/gpu-raster-whittaker-smoother.ts:261-318` | The float32 band solve runs on raw values: a series at 300 +- 0.3 is 0.66 off at `lambda` 1e4 with status 0. Subtract a per-cell reference first. | no |
| polygon-topology-03 | `gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-dissolve.ts:308-314`, `polygon-coverage-simplification/gpu-coverage-simplification.ts:408-414` | The snap key is taken from the absolute coordinate and clamps near 2^31. At `y = 3,000,000` with `snapTolerance = 0.001`, vertices 100 m apart merge. Also reported in `spatial-weights/gpu-contiguity-weights.ts`. | yes |
| polygon-topology-01, -04 | `gpu-spatial-analysis/geometry-topology/gpu-make-valid.ts:346`, `:250-255` | Make-valid drops a valid ring whose edge is one float32 step tall, and loses the closing edge of an implicitly closed ring (count 0). | wiring only |
| polygon-topology-02 | `gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification.ts:1519-1576` | Topology repair detects only segment crossings. A shortcut that sweeps over an island moves it into the neighbouring polygon. | no |
| geometry-measures-01 | `gpu-spatial-analysis/geometry-measures/geometry-measures-kernels.ts:411-417` | Great-circle polygon area by a spherical-excess fan on float32 unit vectors: a 100 m square reads 60,920 m^2 for 10,008 m^2, a 1 km square is 9 to 32% off. A cancellation-safe triangle function exists and is not used here. | site only |
| geometry-measures-05 | `geospatial-analysis/gpu-datum-grid-transform.ts:271-274` | The inverse tests convergence before the first update with a default tolerance of 1e-5 degrees (1.1 m), so a smaller grid shift is returned unchanged with status success. | site only |
| geometry-measures-04 | `gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction.ts:303` | `floor(v + 0.5)` in float32 moves odd on-grid values in [2^23, 2^24). | no |
| dataframe-02 | `dataframe-analysis/group-statistics/group-statistics-moments.ts:508` | The group z-score centres on the fixed-point mean and divides by a separately corrected deviation: -1.10, -0.10, 0.90 for a true -1, 0, 1 at values near 1e-4. | site only |
| dataframe-03, core-numerics-03 | `analysis-core/cell-key-wgsl.ts:9`, `:53-60` | Fixed-point sums round each row to 1/65536 by default. A constant column of 1e-4 has a mean 6.8% high; values below 7.6e-6 sum to 0. The docs call the sums exact. `GPUValueCube` already derives the unit from the data. | yes |
| dataframe-04 | `dataframe-analysis/column-profile/column-profile-kernels.ts:137-147` | A value exactly on a histogram bin edge falls one bin low when the bin count is not a power of two. The official `GPUHistogram` gives the right bin. | no |
| network-flows-02, -03 | `gpu-network/recipes/drive-time-catchment-recipe.ts:635-654` | Demand on the same edge as its facility goes around through an end node (100 for a true 0), and one-way edges are treated as two-way. | yes |
| network-paths-02, -03 | `gpu-network/network-centrality/network-centrality-passes.ts:102-105`, `:478-482` | Betweenness loses dependency across zero-weight edges, and float32 path counts overflow on a 67 x 67 unit lattice (NaN at 68 x 68), with `converged` 1 in both cases. | arithmetic only |
| terrain-hydrology-01 | `gpu-terrain/terrain-processing/gpu-terrain-branching-flow.ts:187-188` | A no-data cell publishes NaN fractions, which branching transport reads as "not ready": every cell touching no-data and everything downstream stays unresolved. | no |
| terrain-hydrology-05 | `gpu-terrain/hydrology/terrain-flow-tracing.ts:44-48`, `docs/api/gpu-terrain/operators.md:1705` | The docs say TauDEM and Whitebox direction grids work. The decoder is ESRI-only: a Whitebox grid is read rotated 45 degrees with no error. | yes |
| terrain-hydrology-02 | `gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts:220-243` | Per-basin mean and area are serial float32 sums: a 1,000,000-cell basin at a constant 2000 m reports 2015.58. | yes |
| trajectories-lines-02 | `gpu-spatial-analysis/trajectory-mobility/gpu-modal-split.ts:264-272` | A speed at or above a finite last break falls into mode 0, the slowest class. | yes |
| core-numerics-04 | `analysis-core/kernels/normal-quantile-wgsl.ts:21`, `:29`, `:34` | Three AS 241 coefficients are mistyped. Effect: 2e-7 to 7e-7 relative for p below 1.4e-11, nothing visible elsewhere at float32. | yes |
| core-numerics-01 | `pointcloud-analysis/pointcloud-plane-icp.ts:13-16`, `:39-45` | Point-to-plane ICP reports `DEGENERATE` for a well-posed scene far from the origin. Loud, not silent. | no |

## Recurring causes

- **Float32 conditioning.** Raw normal equations (regimes, panel lag, error ML), variance as a difference of sums (SKATER gain, group z-score), overflowing products (global statistics, path counts), serial sums over a million cells (catchment summary, trajectory arc length). The library already has the cure in each case (centring, the direct form Ward uses, the fixed-point accumulator, the Neumaier sum of the line family); it is not applied everywhere.
- **Absolute coordinates.** Snap keys, lattice steps sized from the extent only, reprojection origins, stream-network vertices and the street-profile test fail at UTM magnitudes. Contract section 6 asks for a spatial context; these kernels do not use its local origin.
- **Exact comparison of rounded values.** Street profiles, the Ang shell count, the polygon `contains` midpoint, shared-edge clip points.
- **Fixtures that cannot fail.** Integer coordinates and edge lengths, order-1 predictors with moderate R^2, power-of-two bin counts, `-9999` as the only sentinel, positive correlations only, and oracles that are a TypeScript copy of the kernel formula.

## Standards: one decision each

Each row is one convention applied two or more ways. The reports give every site.

| Topic | State today | Reports |
| --- | --- | --- |
| Raster row direction | Terrain derivatives and five siblings take a `rowDirection` prop (default `'south'`) that is not checked against the grid affine. Isolines, isobands and the flow family put row 0 at the smallest y and ignore the affine. Sampling, distance field, cost distance and rasterization follow the affine. | terrain-surface-02, raster-fields-05 |
| Border cells of a stencil | Local `'clamp'` halves slope and invents curvature at the border. Official `GPUFiniteDifference2D` uses one-sided differences. | terrain-surface-03, terrain-hydrology |
| Class and band intervals | Class assignment and `GPUClassBreaks` are left-closed; mapclassify and giddy are right-closed. Reachability bands are `(a, b]`, isochrone bands `[a, b)`. Region masks: rectangles closed, lasso polygons half-open; official `GPUPointSpatialQuery` is inclusive. | point-process-clustering-03, network-paths-07, cells-joins-03 |
| Missing values | NaN only (change detection), finite test (most), infinities kept as data (sampling, focal statistics), sentinel applied to constants (raster algebra), NaN propagated or skipped by sibling (Huff against catchment, Vecchia). Four missing-float markers in the network package. | raster-imagery-09, raster-fields-07, applied-recipes-05, interpolation-density-06 |
| Neighbour loop and self weights | Four contributors have their own neighbour rule, so a self-slot weight is counted differently. | spatial-statistics-05 |
| Self-loops in a graph | Noding lists an undirected loop twice, coarsening and statistics once. Degree counts a self-loop once in one class and twice in another. Official luma.gl also has both rules (degree one, modularity two). | network-flows-04, network-paths-06 |
| Stream cell | `streams == 1` in two contributors, `!= 0` in four. | terrain-hydrology-04 |
| Variance divisor | Cell statistics `n`, zonal statistics `n - 1`. Same-type K by `A/n^2` in marked statistics, `A/(n(n-1))` in Ripley. | raster-imagery-07, dataframe-06 |
| Information criteria | GWR uses the mgwr AICc, GTWR the mgtwr one, in the same summary slot. AIC counts parameters four ways. | regression-06, -08 |
| Analytic p-value floor and centring | Floored in two contributors, not in two; three mean-centring schemes. | spatial-statistics-07, -08 |
| Antimeridian | Great-circle DBSCAN does not wrap, OPTICS does. Vincenty and rhumb use a lossy seam difference. | point-process-clustering-05, geometry-measures-08 |
| Earth radius | Constants come from `contracts/earth.ts` almost everywhere. Exceptions: point horizon uses the mean sphere where the raster grid uses WGS84 radii; official `GPUHaversineDistance` defaults to 6371 (km). | terrain-surface-09, geometry-measures, core-numerics |
| Shared numerics (contract section 11) | Six shared kernels have no importer outside `analysis-core`: both dense solvers, the symmetric eigen solver, binary search, sphere distance, the 64-bit helpers. In their place: a private Cholesky in 20 files with two pivot policies, an equilibrated solve in 6, about 43 binary searches, six inline haversines, four spherical-destination functions, three variogram shape copies. | core-numerics, regression-10 |
| Random numbers | Three generators are not Philox (`gpu-kmeans.ts:728`, `gpu-minimum-bounds.ts:576`, `gpu-network-louvain.ts:171`, the last stuck at 0 for one seed). Four kernels use an "open interval" uniform that returns 1.0 once in 2^24 draws. | core-numerics-05, network-paths-08 |
| Fixed-point sums | Shared 96-bit accumulator, a 64-bit one at 1/65536 (cell tables, group statistics, key join), a private 64-bit one in network coarsening, a data-derived unit in `GPUValueCube`. | core-numerics-03, network-flows-05 |
| Reference claims in docs | Several class docs name a library whose convention the kernel does not follow: Circuitscape (conductance divided by edge length), tobler (denominator default), spdep `licd_multi`, PyKrige (Gaussian range factor), MovingPandas `StopSplitter`, `move::getVolumeUD`, QGIS line density, REDCAP. The code is usually defensible; the claim is not. | applied-recipes-02, cells-joins-02, spatial-statistics-03, interpolation-density-09, trajectories-lines-04 to -07, point-process-clustering-06 |

## The luma.gl way

Detail, the full match table and the upstream proposals are in [luma-official.md](./luma-official.md).

- **Rules hold.** No private import, no re-export, no copied source file. One gap: the guard script accepts `@luma.gl/graph/_internal/arrow` because it is an explicit exports key; no file imports it.
- **Delegated (36 official symbols).** Reductions, scans, sorts, compaction, COO to CSR, the grid index, point in polygon, raster gradient, contours for terrain, band math for spectral indices, similarity search, the official k-means for raster embeddings, FFT. Arguments keep their meaning wherever a reviewer checked.
- **Official and local disagree on numbers.**
  - Grid cell mapping: `point-density-kernels.ts:279` carries the `GPUGridIndex` variant where it should match `GPUGridBinning`. Four more files hold private copies of the same official WGSL, identical today and pinned by nothing. Official luma.gl itself has two mappings that disagree (x = -177 over `[-180, 180]`, 360 columns: column 2 against column 3); that is an upstream fix.
  - Contour saddles: local isolines and isobands use the corner mean, official `GPURasterContours` the bilinear decider.
  - Histogram equalization and percentile rank in `GPURasterStretch` against `GPURasterContrast` and `GPURasterGlobalPercentile`.
  - Zero-vector cosine: 0 in the embedding kNN classifier, 1 in official `GPUSimilaritySearch`.
  - Morphological reconstruction defaults to 4-connectivity, official dilation to 8.
- **Name collisions.** Local `GPUKMeans` (`spatial-clustering/gpu-kmeans.ts:273`) is a different algorithm from the official class of the same name, which this package also uses. The Lloyd step agrees; seeding, empty clusters, the stop test and the final assignment differ, so it cannot delegate today without changing results. Local `validatePackedView` (376 importing files) shadows the official one (14).
- **Re-implemented with no stated reason (16 rows).** Among them key join against `GPUHashJoin`, group statistics and cell aggregation against `GPUGroupAggregation`, raster arithmetic against `GPURasterBandMath`, stretch against `GPURasterContrast`, focal statistics against `GPURasterNeighborhood`, the SMRF windows against `GPURasterErosion` and `GPURasterDilation`. Fourteen of the 27 re-implementation rows were not compared numerically.
- **Replaceable today after a parity spec.** `GPUCellGeometry` centres by `GPUDGGSCellProjection`; `GPUGeodesicPairs` sphere distance by `GPUHaversineDistance` with the radius passed from the spatial context.
- **Blocked upstream.** Graph algorithms need physical `GPUVector` topology (already in the register), so local shortest paths, components, degree and modularity stay; their math agrees with the official classes. `analysis-core/network-topology.ts:193-245` builds an object cast to the official topology type, bypassing its constructor.
- **Fragile coupling.** `cell-indexing/h3-index-extended-wgsl.ts:11` parses constant tables out of the official shader source text; a renamed upstream function yields zero rows and no error.
- **Conventions that differ from official code.** 475 of 485 local classes attach with `getCommandNodes`, the official raster, geospatial and graph entries with `addToGraph`. The contract says `validity`, the official raster entry `outputValidity`. Three no-data policy vocabularies, three names for the iteration cap, `(low, high)` 64-bit words in one unused local helper against `(high, low)` officially.
- **Register.** No row of [../../upstream-workarounds.md](../../upstream-workarounds.md) can be retired at alpha.10. Five rows are stale and nine are missing (report section 6).

## Decisions for the maintainer

1. Row direction: one source of truth (the grid affine) for every raster and terrain contributor, or a checked prop.
2. Interval closure for classes, bands and region boundaries. The luma.gl convention for region queries is inclusive.
3. The saddle rule for contours: the official bilinear decider, or the corner mean that GDAL and Matplotlib use.
4. Self-loop convention, given that official luma.gl has both.
5. Variance divisor for raster statistics.
6. `rhumbMidpoint`, the trajectory acceleration bit, isochrone `'max'` mode and the LIC step on non-square cells: in each the code does something definite and the name or doc promises something else. Fix the code or the words.
7. Resistance connectivity: keep the division by ground edge length (then drop the Circuitscape claim) or remove it.
8. Local `GPUKMeans`: rename now and propose k-means++ seeding and a tolerance stop upstream, or keep the collision.
9. `addToGraph` against `getCommandNodes`, and `outputValidity` against `validity`: the contract and the official raster entry disagree and the code follows both.
10. Whether section 11 of the contract is enforced: either contributors move to the shared solvers, searches and distance functions, or the unused shared kernels are deleted.

## Not covered

- `analysis-core/ring-assembly/` (1,759 lines) and `analysis-core/polygon-triangulation/` (about 3,300 lines of ear clipping, hole bridging and the monotone path). Overlay, polygonize, make-valid and dissolve depend on them. The side probe behind polygon-topology-01 is also built in ring assembly.
- About 40% of the cells-and-joins scope: the DE-9IM relate engine (`spatial-relate-wgsl.ts`), `cell-set-outline`, the exact tests of `cell-cover`, most of `region-statistics`, join candidate plumbing.
- `dataframe-analysis/column-classification/natural-breaks-nodes.ts` (1,258 lines), the tiled relaxation engine `raster-analysis/cost-distance/raster-relaxation.ts:596-1077` (the epsilon fill depends on it), the Euler-tour part of SKATER, the Ward merge kernels, the blocked Cholesky of conditional simulation, seven recipes, the weights algebra kernels and every chunked weights variant.
- `modules/lenses`, `modules/kepler` and the examples.
- Any run on a GPU. A headless spec per finding in the first table would turn "confirmed by arithmetic" into "confirmed".

## Gates on the day

- `node scripts/check-upstream-first.mjs`: no violations.
- `node scripts/check-contributor-contract.mjs`: one new finding at the start of the review (`addRasterResistanceConnectivityRecipe` not in the operator catalog); none at the end, after another session registered it.
- `yarn test-node`, run while the reviewers shared the machine (19 minutes for a usual 4): 9,368 passed, 13 failed, 384 skipped. The 11 failing files were rerun on a quiet machine after the review: all 519 tests in them pass. The failures were load timeouts and the catalog entry above, not assertion failures in the math.
- `yarn typecheck`, `yarn lint`, `yarn build` and `yarn test-headless` were not run; the review changed no source.
