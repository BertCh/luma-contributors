# Spatial and spatiotemporal review, and directions (2026-11-09)

A review of the spatial and spatiotemporal parts of `@luma-contributors/analysis` at commit `edf9ca10` plus the uncommitted working tree of 2026-11-09, with web research on where to take them. The work items are in [../../roadmap.md](../../roadmap.md#215-spatial-and-spatiotemporal-directions-2026-11-09) (section 2.15). This directory is the evidence.

## Summary

The spatial side is broad and careful. The spatiotemporal side exists as separate islands: each family (dataframe time filters, calendar buckets, the hot spot cube, Knox and Mantel, Markov chains, panels, trajectories, raster stacks) defines its own time encoding, unit and layout, and nothing checks that two of them agree. That is the main finding, and it sets the order of work:

1. **Fix what is wrong today.** About fifteen confirmed defects, most of them small. The ones a user would hit: nanosecond or second timestamps decode to wrong calendar dates without an error; clock resampling draws a straight line through a 12-hour gap, so encounters are reported for time nobody observed; the "sporadic hot spot" class does not match current ArcGIS; local G flips its one-sided tail on negative data; GWR reports nearly collinear local fits as valid.
2. **Give time a contract**, the way `resolveGPUSpatialContext` gives space one, and add four small glue nodes. After that the existing pieces compose: filter, cube, Gi\*, trend, Markov, Knox and panels on one pipeline with no readback.
3. **Add the missing space-time methods**, most of which are then small: the Knox near-repeat table and the Jacquez test, space-time weights, ST-DBSCAN, space-time kernel density and K function, change points, staypoints and trips, space-time kriging, conformal intervals.
4. **Two research bets**: a Vecchia (nearest-neighbour Gaussian process) likelihood, which fits the platform's "many small solves" strength, and GPU-resident iteration through batched indirect dispatch, which is luma.gl work.

## Method and what was checked

- **Gates, run before the review.** `yarn typecheck` passes. `yarn test-node`: 716 files, 7,906 tests pass, 375 skipped, 1 failed. The failure was a 30 s timeout in `modules/analysis/test/utils/polygon-edge-slabs-dedupe.node.spec.ts` while ten review agents loaded the machine (the run took 649 s against the usual 240 s). The spec passes alone in 13 s.
- **Code review.** Five reviewers read the code in disjoint scopes and wrote the `code-*.md` files. They did not run anything. Each finding is tagged CONFIRMED (the reviewer traced the code path) or SUSPECTED.
- **Web research.** Five researchers wrote the `research-*.md` files. Each cites pages it opened and marks the rest UNVERIFIED. The GPU landscape report ran out of search budget, so part of its ecosystem section is marked NOT RESEARCHED.
- **Re-checked by the lead reviewer.** Every finding in the table in "Defects in existing code" below was re-read in source. These external claims were re-fetched: the current ArcGIS definition of a sporadic hot spot, the WGLog paper (arXiv 2607.17571) and its figures, the archival of cuSpatial, Chrome's `linear_indexing` extension, and kriging-rs 0.4.0. Two errors found this way are corrected in `research-gpu-landscape.md`. The other external claims are as the researchers reported them.
- **Experiment.** One numerical experiment was run for this review: [experiments/vecchia-float32.mjs](./experiments/vecchia-float32.mjs), described under "Experiment" below.

## Files

| File | What |
| --- | --- |
| [code-spacetime-statistics.md](./code-spacetime-statistics.md) | Emerging hot spots, Knox, Mantel, scan statistic, Markov chains, time filters, calendar buckets, temporal reduction, raster change detection and seasonal break: time models, validity against references, scaling walls, composability |
| [code-movement.md](./code-movement.md) | Trajectory contributors, track similarity, map matching, time routing, flow aggregation, particle advection: data model, twelve correctness findings, absent operations |
| [code-spatial-statistics.md](./code-spatial-statistics.md) | Weights, neighbour search, autocorrelation, permutation inference, multiple testing, clustering, regionalization, density: shared infrastructure, 22 findings, API inconsistencies, which blocks can take a time dimension |
| [code-spatial-models.md](./code-spatial-models.md) | Regression, GWR, kriging, RBF, simulation, point processes, pair statistics: solvers and float32 behaviour, sizes the coverage matrix does not state, time readiness, feasibility of Vecchia, conformal, Hawkes |
| [code-architecture.md](./code-architecture.md) | Contributor contract, the missing temporal context, index structures, luma.gl constraints checked against the workarounds register, ontology, test fixtures, docs |
| [research-spacetime-statistics.md](./research-spacetime-statistics.md) | Exploratory space-time statistics 2022 to 2026: twelve ranked candidates, reference software, what not to build |
| [research-spacetime-models.md](./research-spacetime-models.md) | Space-time geostatistics, Vecchia and other scalable Gaussian processes, GTWR, Hawkes and ETAS, conformal prediction, scoring rules, optimal transport |
| [research-movement.md](./research-movement.md) | Staypoints and trips, mobility indicators, cleaning, similarity at scale, home ranges, time geography, Lagrangian diagnostics, interchange formats |
| [research-raster-time-series.md](./research-raster-time-series.md) | CCDC and COLD, BFAST, Whittaker, phenology, compositing, climate indices, EOF, embedding rasters, discrete global grids, data cube conventions |
| [research-gpu-landscape.md](./research-gpu-landscape.md) | WebGPU status by browser and feature, GPU spatial index and join literature, the native GPU ecosystem, precision without f64, benchmarks. Partial |
| [experiments/vecchia-float32.mjs](./experiments/vecchia-float32.mjs) | The float32 Vecchia likelihood experiment |

## Where the library stands

**Spatial.** One sparse weights format with a chunked variant, one definition of a neighbour, fixed-order sums with no float atomics, two-pass centred moments, a replayable Philox permutation stream, and multiple-testing adjustment that matches statsmodels. Gi\*, Mann-Kendall with tie correction, Theil-Sen and the FDR family trace correctly against their references. The weak areas are clustering, regionalization and density, which do not use the shared weights or neighbour search: there are three private point indices (the DBSCAN grid, the HDBSCAN nearest-feature join with its own BVH, the correlogram pair grid), and OPTICS, k-means and island attachment are brute force.

**Spatiotemporal.** What exists, by family:

| Family | Contributors | Time model |
| --- | --- | --- |
| Dataframe time | `GPUTimeWindowFilter`, `GPUTemporalReduction`, `GPUCalendarBuckets` | Int64 words in any unit (filter, reduction); Int64 words that must be epoch milliseconds (calendar); or float32 relative |
| Cube statistics | `GPUEmergingHotSpots`, `addSpaceTimeHotSpotsRecipe`, `GPUSpatialScanStatistic` | Integer slice index, cell-major |
| Pair tests | `GPUKnoxTest`, `GPUMantelTest`, `GPUSpaceTimeVariogram` | Float32, origin subtracted on the CPU (Knox, Mantel); raw float32 (variogram) |
| Dynamics and panels | `GPUTransitionMatrix`, `GPUSpatialMarkov`, `GPULISAMarkov`, `GPURankMarkov`, spatial panel regressions | Integer period index, period-major, balanced |
| Trajectories | Metrics, resample, playhead, zone events, encounters, closest approach share `GPUTrajectoryGeometryPort` (float32 or Int64-word times); Kalman and restriction take float32 seconds only; map matching and track similarity have no time | Mixed |
| Raster stacks | `GPUChangeDetection`, `GPURasterTemporal`, `GPURasterSeasonalBreak` | Integer slice index, regular spacing only, at most 256 slices |

One pipeline already runs on the GPU end to end: time-window filter, count cube, Gi\*, Mann-Kendall, Theil-Sen. Everything else needs CPU glue.

## Defects in existing code

Re-read in source by the lead reviewer. Paths are relative to `modules/analysis/src`.

| ID | Finding | Where | Effect |
| --- | --- | --- | --- |
| FX-01 | `GPUCalendarBuckets` assumes epoch milliseconds and has no unit prop, while the Arrow upload keeps each column's own unit | `dataframe-analysis/calendar-buckets/gpu-calendar-buckets.ts:372`; `analysis-core/graph-support/arrow-analytics-normalization.ts:321-379` | A `timestamp[ns]` column (the pandas default) decodes to dates tens of thousands of years out, with no error |
| FX-02 | Word-mode bucket width is capped at 2^32 − 1 units | `dataframe-analysis/temporal-reduction/temporal-reduction-parameters.ts:68` | 49.7 days in milliseconds, 71.6 minutes in microseconds, 4.29 seconds in nanoseconds: hourly buckets of nanosecond data cannot be made |
| FX-03 | Sporadic hot and cold spots are assigned without a significant final slice; the comment says this is ArcGIS | `gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots.ts:216-217`, `:1273` | Current ArcGIS Pro requires "a statistically significant hot spot for the final time-step interval" (page re-fetched 2026-11-09). The showcase headline ("35 are sporadic hot") depends on the older rule |
| FX-04 | Mantel correlates raw distances and time differences; pointpats correlates reciprocal transforms by default | `gpu-spatial-analysis/space-time-tests/gpu-mantel-test.ts:142` | The pointpats default cannot be reproduced; the parity tests use scikit-bio without transforms |
| FX-05 | Local G and G\* rank permutations on the unscaled lag sum; the dropped denominator is called a positive constant, and negative values are not rejected | `gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test.ts:159-162`, `:1108-1109` | With negative data and `alternative: 'greater'` the p-value is the lower tail. G is defined for non-negative data, so the fix is a validity bit or a rejection |
| FX-06 | Clock resampling marks a sample absent only outside the whole track's span | `gpu-spatial-analysis/trajectory-interpolation/trajectory-interpolation-kernels.ts:694-697` | Gaps are interpolated, so clock encounters, closest approach and zone dwell report events inside gaps. Playhead, Kalman and restriction have a `maximumGap`; the others do not |
| FX-07 | Kalman and restriction store relative times as float32 seconds from one epoch | `gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts:64`, `:85-89` | Resolution is 0.125 s after three weeks and 2 s after a year, so 10 Hz and then 1 Hz samples collapse. The MV01 coverage row does not state this |
| FX-08 | Track quality flags the row that ends a fast step | `gpu-network/recipes/track-quality-recipe.ts:98-102` | The valid fix after an outlier is flagged along with the outlier |
| FX-09 | The GWR family's Cholesky fails only on a non-positive pivot; the documented "ill-conditioned" status has no test behind it | `gpu-spatial-analysis/spatial-regression/spatial-regression-solve.ts:40-42`; `local-weighted-least-squares-wgsl.ts:70-84` | Nearly collinear local designs return coefficients with status 0. OLS uses a pivot tolerance of 1e-3 for the same data |
| FX-10 | Two kriging solvers disagree on the conditioning limit: 1e-4 (measured 5% weight error at 3e-5) against 1e-5 | `gpu-spatial-analysis/spatial-interpolation/gpu-kriging.ts:51-59`; `equilibrated-solve-wgsl.ts:20` | `GPULocalKriging` and `GPULocalRBF` accept systems `GPUKriging` rejects |
| FX-11 | `GPUSpaceTimeVariogram` is brute force to 16,384 rows, takes raw float32 times with no origin, has no zero-lag classes and no gstat fixture | `dataframe-analysis/pair-statistics/gpu-cross-variogram.ts:180-205` | Epoch seconds round to 128 s; the purely temporal variogram cannot be isolated. It has no coverage-matrix row |
| FX-12 | The minimum spanning tree standardizes columns without a finite check | `gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree.ts:265-285` | One NaN makes every edge cost NaN, and SKATER with it. AZP, Ward and max-p skip non-finite values |
| FX-13 | The uncommitted edit re-exports luma.gl's `GPUFloat32Positions` from the public entry | `analysis-core/contracts/ports.ts:16` | Breaks the AGENTS.md rule against re-exporting luma.gl APIs. `scripts/check-upstream-first.mjs` checks imports only |
| FX-14 | At the default 999 permutations the smallest two-sided p is 0.002 | `permutation-inference`, `multiple-testing` | With 10,000 rows Benjamini-Hochberg needs about 400 rows at the floor before it can reject, and Bonferroni cannot reject above 25 rows. Nothing warns |
| FX-15 | Map matching: no drop of fixes within 2σ of the previous one, no per-step score renormalization, unchecked scratch size, and it emits the matched edge per point, not the traversed route | `gpu-network/map-matching/gpu-map-matching.ts` | Breaks on backward jitter on one-way edges; cannot feed edge volumes |

Suspected and not settled: the lattice cube counts never-occupied cells as valid zeros, which ArcGIS is believed to exclude; SKATER's cut gain is computed as `Σx² − (Σx)²/n` in float32; the conditional simulation Cholesky uses an absolute tolerance that smooth covariances will hit early.

Documentation and register corrections (confirmed):

- `gpu-ripley.ts:183` and roadmap PP01 say inhomogeneous K is not implemented. `GPUInhomogeneousPointPattern` implements it.
- Coverage matrix: the ST03.random row names `GPUSpatialPanelErrorMaximumLikelihood`; the class is `GPUSpatialPanelLagMaximumLikelihood`, limited to 32 entities × 32 periods × 7 predictors. ST06 does not state that full MGWR inference runs only for n ≤ 128. ST09.fuzzy does not state the 512-feature limit of `GPUFuzzyContiguity`. Adaptive GWR bandwidth is capped at 128 neighbours.
- `upstream-workarounds.md`: the "bounded dispatch above about 4.2M rows" regression looks obsolete, because published `GPUScan`, `GPUSort`, `GPUCompaction` and `GPUReduction` apply bounded dispatch layouts. One runtime test above 4.2M rows would settle it. The segmented-sum row names two options that no longer exist.
- Roadmap LN-V5 "space-time cube" means a visualization of solver wavefronts in the archive, not a data cube.
- Defaults that break a quick parity check, each documented but worth a table in the docs: islands excluded (esda keeps them), the two-sided p-value convention (`'folded'` matches esda), k-means on raw float32 coordinates unless `centerCoordinates` is set, `'uniform'` KDE edge correction, and `GPULineDensity` as length per cell.

## Structural gaps

1. **No temporal context.** Unit exists only on the CPU side (`ArrowAnalyticsTimeUnit`) and is dropped at upload; kepler declares the same type again. Three encodings (float32 relative, high/low float32 pairs, Int64 words) and two layouts (cell-major cubes, period-major panels) coexist with no converter. Only 1 of 261 catalogued operators is tagged `2d+time`, and the dataframe and raster temporal contributors are outside the catalog.
2. **Missing glue.** The absolute period index is inline WGSL inside the hot spot recipe. Cubes hold counts, min, max, first and last, but no sum or mean. There is no cell-major to period-major transpose and no GPU node from Int64 words to relative float32, although the kernel (`timeWordsDifferenceToF32`) exists. `GPUGroupStatistics` already takes 64-bit keys, so a DGGS × period cube is one adapter kernel away.
3. **Two trajectory models and four gap behaviours.** See FX-06 and FX-07. The sort-by-track-then-time precondition is never validated.
4. **Point search is 2D and private.** Neighbour search, DBSCAN, HDBSCAN, OPTICS, LOF and KDE hard-code two dimensions. The 3D `GPUGridIndex` use in trajectory encounters shows the pattern for (x, y, t).
5. **Covariance parameters are compiled into the shader** in `GPULocalKriging`, so each optimizer step recompiles. This blocks likelihood fitting, Vecchia and space-time kriging with fitted models. Only spherical, exponential and Gaussian models exist: no Matérn, no nesting, no anisotropy.
6. **Iteration.** Loops are unrolled at compile time and later rounds are gated to zero-workgroup dispatches. That is fine for hundreds of rounds. Only centrality has a node budget.
7. **Single-thread quadratic kernels.** Trajectory restriction compiles up to 4,096 pairs into the shader and loops over both tracks on one thread; similarity top-k is O(tracks × pairs × K²) per thread; time routing is one thread with 256 states.
8. **Fixtures.** All generators are in the repository, but there is no runner, 204 of 746 fixture-to-generator mappings are inferred, and 64 generators write a dead `modules/graph/...` path. Trajectory analysis, trajectory interpolation, track similarity, distribution dynamics and neighbour search rely on in-repo CPU references only.

## Directions

Effort: S is days, M is one to two weeks, L is more. "Oracle" is the reference implementation for fixtures; GPL packages are test oracles only.

### A. Time contract and glue (do first)

| ID | Item | Effort |
| --- | --- | --- |
| TC-01 | Temporal context in `analysis-core/contracts`: unit, epoch, encoding, optional time zone, regular or irregular axis; carried from `PreparedArrowAnalyticsTimeColumn` to the uploaded vector and checked like the spatial context. Kepler uses it in place of its own type | M |
| TC-02 | Unit-aware calendar (FX-01) and 64-bit bucket width (FX-02) | S |
| TC-03 | Period-index contributor extracted from `addSpaceTimeHotSpotsRecipe`: calendar step or fixed width, origin, count; `uint32` period per row | S |
| TC-04 | Value cubes: sum and mean per (cell, period); DGGS (H3, S2, Quadbin) × period through `GPUGroupStatistics` | S |
| TC-05 | Layout transpose between cell-major and period-major; Int64 words to relative float32 node | S |
| TC-06 | One trajectory port with word timestamps in Kalman, restriction and time-aware simplification; one `maximumGap` rule everywhere (FX-06, FX-07); optional GPU check of sort order | M |
| TC-07 | Irregular-date axis for raster stacks (centred fractional years in float32) and streaming accumulators past 256 slices | M |
| TC-08 | Time in the ontology: tag temporal operators, add an events or time-series data kind, include dataframe and raster packages | S |

### B. Exploratory space-time statistics

Detail and sources: [research-spacetime-statistics.md](./research-spacetime-statistics.md).

| ID | Item | Reuses | Oracle | Effort |
| --- | --- | --- | --- | --- |
| STX-01 | Near-repeat Knox table (pair counts by distance band × time band with a permutation p per cell), local Knox, Jacquez k-nearest-neighbour test, pointpats Mantel transforms (FX-04) | Knox pair list, Philox, Besag-Clifford stopping | pointpats 2.6.0 (BSD-3); NearRepeat (GPL) | S |
| STX-02 | Space-time weights: spatial W ⊗ temporal lag in the weights algebra; lagged and differential local Moran as recipes; cross-correlation per location over lags with FDR across locations and lags | Weights algebra, local permutation test, multiple testing | esda, GeoDa definitions | S to M |
| STX-03 | ST-DBSCAN, then ST-OPTICS and ST-HDBSCAN: a time filter on the neighbour graph | DBSCAN grid, clustering kernels | st-dbscan 0.2.3 (MIT) | S |
| STX-04 | Space-time kernel density and space-time relative risk, one thread per voxel over the grid index with a time-sorted run per cell | Grid index, KDE kernels | sparr 2.3-16 (GPL) | S to M |
| STX-05 | Space-time K and pair correlation, the Diggle D(s, t) test, LISTA | Indexed pair histogram, study-window corrections | stpp 2.0-8, stopp 1.0.0 (GPL) | M |
| STX-06 | Per-location change points by unpruned optimal partitioning (the same answer as PELT with bounded O(T²) work); Pettitt test; autocorrelation-corrected Mann-Kendall for the hot spot trend | Prefix sums, cube layout | changepoint (R, GPL); ArcGIS Change Point Detection as a specification | M |
| STX-07 | Expectation-based space-time scan with Fast Subset Scan, plus a prospective mode for the existing scan. The existing space-time permutation scan conditions on marginal counts and cannot see a uniform rise across all regions | Scan kNN zones, sort, prefix sums | scanstatistics 1.1.2 (GPL) | M |
| STX-08 | Missing spatial pieces that are small: geographical detector q, REDCAP single linkage, correlogram over contiguity orders, spatial entropy; medium: colocation quotient, LICD, adaptive-bandwidth KDE | Region partition evaluation, MST, weights | Second group built (`docs/api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density.md`): colocation quotient, Leslie and Kronenfeld 2011; LICD local composition, Boots 2003 as in spdep `licd_multi` (configuration part and permutations not built); adaptive-bandwidth KDE, Abramson 1982 / Silverman 1986 5.3.1 (separate class, no edge correction). Binary64 references; no package oracle | S each |

### C. Space-time models and inference

Detail and sources: [research-spacetime-models.md](./research-spacetime-models.md).

| ID | Item | Reuses | Oracle | Effort |
| --- | --- | --- | --- | --- |
| STM-01 | Covariance parameters in a per-frame buffer; Matérn; one conditioning rule (FX-10) | `GPULocalKriging` | GSTools, gstat | M |
| STM-02 | Space-time variogram on the indexed pair histogram with zero-lag classes (FX-11); separable, product-sum, metric, sum-metric and Gneiting models; fitting; local space-time kriging with octant and staged neighbour search (gstat documents these as not yet available) | Pair histogram, variogram fit ladder, local kriging solve | gstat 2.1-6 (GPL) | M |
| STM-03 | Conformal prediction intervals from the existing cross-validation outputs: split and CV+, then spatially weighted (GeoConformal) | `GPUSpatialCrossValidation`, sort | Lou, Luo and Meng 2024 (authors' code); coverage checked by simulation | S, then M |
| STM-04 | Proper scoring rules: CRPS, energy score, variogram score, for the conditional simulations and for kriging | Reductions | scoringRules 1.1.3 | S |
| STM-05 | Area of applicability and dissimilarity index | Neighbour search in predictor space; kNNDM exists | CAST | S to M |
| STM-06 | GTWR and multiscale GTWR: a space-time kernel in the GWR fit, with the conditioning status of FX-09 | `fitLocation`, golden-section bandwidth | To be pinned (GWmodel not yet checked) | S to M |
| STM-07 | Vecchia likelihood, Fisher scoring and prediction, spatial and space-time | Local kriging neighbour solves, sort, STM-01 | GpGp 1.0.0 (MIT) | M to L |
| STM-08 | Space-time Hawkes by EM with bounded parent windows; ETAS later. Single precision is reported to work at one million events with log-sum-exp and a floor on tiny sums (Ko, Suchard and Holbrook 2024) | Ordered pair enumeration, Knox time sort | ETAS 0.7.2 (R) | M |
| STM-09 | Sinkhorn divergence between spatial distributions, log domain, separable blur on grids | Raster convolution | POT (MIT) | M |

### D. Movement

Detail and sources: [research-movement.md](./research-movement.md).

| ID | Item | Reuses | Oracle | Effort |
| --- | --- | --- | --- | --- |
| MOB-01 | Staypoints, locations, triplegs and trips with trackintel semantics. The greedy sliding-window stop rule becomes a bounded forward search plus pointer doubling. Today's "stops" are a per-step speed test, so parked GPS jitter splits one stop into many | Segmented scans, DBSCAN, connected components | trackintel (MIT); MovingPandas 0.23 (BSD-3) | M |
| MOB-02 | Indicators on staypoints: radius of gyration, jump lengths, waiting times, entropies, home and work, tracking quality, modal split | Segmented reductions | scikit-mobility 1.3.1 (BSD) | S |
| MOB-03 | Cleaning: point-centred speed and acceleration outliers (FX-08), Hampel filter, gap, stop and value-change splitters that rewrite offsets, minimum-distance and minimum-time thinning, word timestamps in TD-TR | Track quality recipe, `GPULineSimplification` | MovingPandas 0.23, PTRAIL (BSD-3) | S to M |
| MOB-04 | Remove the single-thread quadratic kernels: two-pointer time merge for restriction with pairs from a GPU view; segmented select for similarity top-k; spatial context in similarity search | Encounters output | Existing fixtures | M |
| MOB-05 | Map matching: traversed route output, 2σ preprocessing, score renormalization, scratch checks (FX-15); then edge volumes into flow aggregation | Map matching, flow aggregation | Valhalla or pgRouting traces | M |
| MOB-06 | Time-varying vector fields (two-slice interpolation), then FTLE, connectivity matrices and residence time | Particle advection | lcs-parcels (MIT) | S, then M |
| MOB-07 | Brownian bridge and dynamic Brownian bridge utilization distributions with isopleths; revisits, residence time and first passage time | Grid index, sort, prefix sums | move (R, licence unverified); recurse 1.4.0 (MIT) | M; S |
| MOB-08 | EDR, ERP, Hausdorff and SSPD on the existing dynamic programs, with lower bounds for top-k. A 2024 study measured batched GPU DTW at only about 4.5× a CPU, so pruning matters more than raw throughput | Track similarity kernels | MovingPandas 0.23 distances | S each |
| MOB-09 | Time geography: network potential path area from a forward and a reverse shortest-path tree; travel-time percentiles over a departure window | FIFO time routing, isochrones | NetworkX; r5r semantics | S to M |
| MOB-10 | Radiation model and kernel flow smoothing on OD aggregation | Flow aggregation | scikit-mobility | S; M |

### E. Raster time series and grids

Detail and sources: [research-raster-time-series.md](./research-raster-time-series.md).

| ID | Item | Reuses | Oracle | Effort |
| --- | --- | --- | --- | --- |
| RTS-01 | FDR significance masks over any p-value raster (the vector node exists in `multiple-testing`) | Multiple testing, histogram | statsmodels | S |
| RTS-02 | Weighted Whittaker smoother and HANTS with fixed reweighting passes; gap filling onto a regular grid; needs TC-07 | Per-pixel least squares | phenofit; GRASS `r.hants` | S |
| RTS-03 | Embedding rasters: dequantize, cosine similarity search, change between years, k-means, kNN and linear-probe inference, PCA to RGB, per-polygon mean vectors, and a unit-vector-mean pyramid mode. AlphaEarth stores `sign(q)·(q/127.5)²` in int8, and its readme says overviews must average decoded vectors and renormalize, so the generic pyramid average is wrong for it | k-means, similar locations, zonal statistics, pyramid | NumPy, scikit-learn | S to M |
| RTS-04 | Irregular-date harmonic regression and a simplified COLD monitor (CCDC family); BFAST Monitor. No GPU or browser port was found | `getRasterLeastSquaresWGSL`, seasonal break | pyxccd (Apache-2.0); bfast (GPL) | M to L |
| RTS-05 | Compositing: medoid, geometric median with fixed iterations, best-available-pixel scoring; phenology metrics on the smoothed curve | `GPURasterTemporal` | hdmedians (Apache-2.0) | S; S to M |
| RTS-06 | Climate indices: SPI and SPEI, spell lengths, degree days, climatology and anomaly; needs the streaming accumulators of TC-07 | Raster algebra | xclim 1.0.0 (Apache-2.0) | S to M |
| RTS-07 | EOF by the method of snapshots (T × T covariance on the GPU, eigenproblem on the CPU), lagged correlation maps | Focal pairs, reductions | xeofs (MIT) | M |
| RTS-08 | HEALPix nested and A5 point-to-cell and aggregation; DGGS-JSON output. OGC API DGGS Part 1 was approved 2025-10-24 | Cell indexing, cell aggregation | healpix-geo; A5 (Apache-2.0) | M |

### F. Platform

Detail and sources: [research-gpu-landscape.md](./research-gpu-landscape.md). Items marked luma.gl go upstream as ordinary pull requests, per AGENTS.md.

| ID | Item | Effort |
| --- | --- | --- |
| PLAT-01 | Settle the 4.2M-row register entry with one runtime test per primitive. If a limit remains it is the one-dimensional dispatch product (65,535 workgroups × 64), which a 2D dispatch removes | S |
| PLAT-02 | Guard storage-binding size at construction in the single-view temporal and cell contributors (about 16.7M Int64 timestamps on a default device) | S |
| PLAT-03 | Unify point neighbour search (one grid index for DBSCAN, HDBSCAN, OPTICS, LOF, KDE), lift k above 32, and give it an optional time axis | M |
| PLAT-04 | luma.gl: GPU-resident iteration. GPU-written indirect dispatch arguments with several iterations recorded per submit and one convergence readback per batch. WGLog (arXiv 2607.17571, July 2026) measured a 4-byte readback at 2.5 ms, 4.7× from indirect dispatch alone and 15.4× with batching, on WebGPU. This is the structural answer to NW-CENT | L |
| PLAT-05 | luma.gl or host: request the adapter's real limits. A public survey reports 98% of devices allow 10 or more storage buffers per stage, 80% allow 16, and 77% allow 2 GiB buffers. Subgroups (Chrome only, about 68% of devices) with a fallback for scan and sort; `buffer_view` (Chrome 153) to pack arrays into one binding | M |
| PLAT-06 | Fixture runner that regenerates one family and diffs; fix the dead generator paths; pin external fixtures for the trajectory, track-similarity and Markov families. MobilityDuck ships DuckDB-Wasm builds, which would put MEOS semantics in a browser test harness | M |
| PLAT-07 | Adopt SpatialBench (Apache Sedona, December 2025) scale factors for join and aggregation benchmarks (UF-03) | M |

### Not recommended

- Exact flocks and convoys (NP-hard maximal flocks, no open oracle), TRACLUS for now (the only Python oracle warns of bugs), T-LoCoH, continuous-time movement model fitting and autocorrelated KDE.
- Full MCMC, full INLA, dense Gaussian-process conjugate gradients at a million points, training neural point processes, GNNWR or geographically weighted random forests.
- Spatiotemporal co-location pattern mining, flexible scan (exponential; Fast Subset Scan covers irregular clusters exactly), ad hoc "spatiotemporal Moran" indices, e-values.
- BEAST, TIMESAT parity, double-logistic fits in float32, running foundation-model encoders in the library, and the simulations already excluded (flood, wildfire, urban growth).
- Ray-tracing-core spatial joins: there is no ray query proposal for WebGPU.

## Context from outside

- **Browser GPU is no longer niche.** WebGPU ships by default in Chrome, in Safari 26 (September 2025) and in Firefox on Windows and macOS. Firefox on Linux and Android, and parts of Android, are the gaps. There is no proposal for f64 or i64 in WGSL, so the float32 techniques stay.
- **The native GPU vector stack thinned out.** The cuSpatial repository was archived on 2025-07-28 with v25.04 as its final release. SedonaDB reportedly ships GPU spatial joins on ray-tracing cores, which WebGPU cannot use.
- **Space-time kriging in the browser is not new.** kriging-rs 0.4.0 (MIT, 2026-04-26) offers WebGPU batch kriging with separable and product-sum space-time variograms. STM-02 would differ by the sum-metric and Gneiting models, the neighbour searches, weighted fitting and gstat parity.
- **Nothing was found on the GPU** for space-time K, Knox, Jacquez, the Diggle test or space-time scans, or for CCDC and COLD, or for any trajectory method on WebGPU. This rests on searches coming up empty.

## Experiment: Vecchia log-likelihood in float32

The top open question for STM-07 was whether a Vecchia log-likelihood survives float32, since no source the researchers opened reports it. [experiments/vecchia-float32.mjs](./experiments/vecchia-float32.mjs) answers it on the CPU by rounding every operation to binary32 with `Math.fround` and comparing against binary64.

Setup: 10,000 uniform points in a 100 km square (coordinates centred on the origin), random ordering, 30 nearest earlier neighbours, unit sill, true range 10 km, data simulated from the Vecchia model in binary64. The log-likelihood is evaluated on a grid of ten range values from 0.5 to 2 times the truth. "Error" is the largest absolute difference between the float32 and float64 log-likelihood over the grid. "Gap" is how far the best grid point sits above the next best in float64, which is what an optimizer has to resolve. Run with `node vecchia-float32.mjs 10000 30` (21 minutes on a shared machine).

| Kernel | Nugget / sill | Sites where the float32 Cholesky fails | Error | Gap | Same best range |
| --- | --- | --- | --- | --- | --- |
| Exponential | 0 | 0 | 0.0006 | 4.2 | yes |
| Exponential | 1e-3 | 0 | 0.005 | 3.1 | yes |
| Exponential | 0.1 | 0 | 0.0003 | 0.07 | yes |
| Matérn 3/2 | 0 | 86 | 82 | 43 | yes |
| Matérn 3/2 | 1e-4 | 0 | 0.27 | 25 | yes |
| Matérn 3/2 | 1e-3 | 0 | 0.17 | 16 | yes |
| Matérn 5/2 | 0 | 6,353 | 48,900 | 136 | no |
| Matérn 5/2 | 1e-4 | 0 | 3.7 | 24 | yes |
| Matérn 5/2 | 1e-3 | 0 | 0.25 | 12 | yes |
| Matérn 5/2 | 1e-2 | 0 | 0.011 | 5.9 | yes |
| Gaussian | 0 | 9,732 | not finite | n/a | no |
| Gaussian | 1e-4 | 0 | 8.2 | 40 | yes |
| Gaussian | 1e-3 | 0 | 0.48 | 16 | yes |
| Gaussian | 1e-2 | 0 | 0.008 | 7.1 | yes |

All twenty rows are in the script's output; the ones left out here (nugget 0.01 and 0.1 for the smooth kernels, 1e-4 and 0.01 for the exponential) have errors of 0.012 or less.

What it shows:

- **Float32 is enough when the nugget is at least 1e-3 of the sill.** The error is under 0.5 log-likelihood units at n = 10,000 for every kernel, against gaps of 12 or more, and the best range is the same as in float64.
- **The exponential kernel is safe at any nugget**, including zero (error under 0.005).
- **Smooth kernels with a nugget of 1e-4 are marginal.** Errors of 0.3 to 8 units do not move the maximum on this grid but would distort a likelihood-ratio test, where 2 units matter.
- **Smooth kernels with no nugget are not usable.** The float32 factorization fails at 86 (Matérn 3/2), 6,353 (Matérn 5/2) and 9,732 (Gaussian) of 10,000 sites. The Gaussian kernel with no nugget is degenerate in float64 too.
- **Accumulating the 10,000 terms in float32** adds at most 0.03 units, so the existing fixed-point accumulator is not needed at this size.

Design consequences for STM-07: enforce a nugget floor of 1e-3 of the sill for Matérn 3/2 and smoother (or reject below it), report the count of failed sites as a status and treat any failure as a failed likelihood, and offer the exponential and Matérn 3/2 kernels first.

Limits of the experiment: one realization per row; a random ordering, not max-min; `exp`, `log` and `sqrt` are computed in float64 and then rounded, while WGSL builtins may be a few units in the last place worse and a shader compiler may contract multiply-adds; the grid is coarse. The next step is the same comparison on the GPU against GpGp.

## Open questions

- Does ArcGIS exclude never-occupied lattice cells from the cube before Gi\*? Decides a default in the hot spot recipe.
- Which package pins GTWR (STM-06)?
- Are 64-bit atomic min and max (`atomic-vec2u-min-max`, merged in the spec) enabled in release browsers? They would give a single-atomic argmin for nearest neighbour and shortest paths.
- Should FX-03 change the default, or add an option and keep the old rule for existing results?
- The R-INLA, move and MobilityDuck licences were not confirmed.
