> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/geospatial-coverage-tranche-one.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Tranche completed; historical record only.

# Geospatial coverage tranche one

Implementation date: October 23, 2026. This tranche extends the contributor expansion and is now
integrated into the `map-contributors` checkout. Its graph contributors, fixtures and scenarios
follow the upstream `modules/graph` package layout. It covers the bounded scope below within
the 78-item [coverage backlog](./geospatial-coverage-backlog.md).

## Scope and independent references

| ID | Implemented bounded contract | Reference evidence | Remaining boundary |
| --- | --- | --- | --- |
| RS01 | [Fractional zonal distributions](../../docs/api-reference/experimental/gpu-raster/fractional-zonal-distribution.md): exactextract histogram quantiles, centered population and reliability-weighted sample variance, Kish effective support; 65,536 unique zone/cell tuples and 32 probabilities | Executed exactextract 0.3.0 and NumPy 2.3.4 fixture generator; direct histogram quantiles/population moments, composed sample variance/effective support | Planar constant-area cells, no separate observation-weight raster; never average tile quantiles. |
| RS02 | [Fractional categories](../../docs/api-reference/experimental/gpu-raster/fractional-categorical-statistics.md): class coverage, area, proportions, majority/minority, separate missing area | Executed exactextract 0.3.0 / NumPy 2.2.6 / Shapely 2.1.2 generator; direct upstream statistics and repeated mask/category frames | Dense category IDs, at most 1,024 categories and 1,048,576 tuples; unique zone/cell tuples and disjoint tile cores are caller obligations. |
| RS05 | [Focal pairs](../../docs/api-reference/experimental/gpu-raster/focal-pairs.md): paired-valid covariance, correlation and simple OLS; radius 0–8; nodata/clamp/reflect boundaries | Executed NumPy 2.5.4 cov/corrcoef/lstsq references with independently composed extraction/calibration | Descriptive unweighted fits; no inference or claim of direct terra execution. Exact-zero singularity policy is explicit. |
| RS09 | [Spectral indices](../../docs/api-reference/experimental/gpu-raster/spectral-indices.md): explicit Gao/McFeeters NDWI, NBR, fixed-coefficient EVI and SAVI, supplied calibration/masks | Executed Spyndex 0.12.0 / NumPy 2.2.6 direct formulas; Gao compared with NDMI algebra | No sensor calibration inference, spectral response model or claim of OTB execution. |
| UR04 | [Dasymetric redistribution](../../docs/api-reference/experimental/gpu-spatial-analysis/dasymetric-redistribution.md): source fractional support, ancillary weights, whole-cell target ownership, source mass ledger | Executed Tobler 0.14.0 / Shapely 2.1.2 / GeoPandas 1.1.1 / Rasterio 1.4.3 / NumPy 2.2.6; aligned binary cases direct, continuous/partial support composed | Arbitrary fractional target polygons remain outside this recipe. Zero support leaves mass unallocated; incomplete or invalid arithmetic remains visible. |

## Integration and validation

All eight public entry points have computation-pack metadata and runnable computation-lens examples.
The maximal-options kernel audit covers each branch, texture/calibration paths and binding-scale
shapes. Construction checks are separate from numerical GPU execution.

Final targeted hardware execution passes **100 tests across six files** in 17.96 seconds on native
Apple WebGPU (`fallback: false`, Metal 3 architecture, Chromium). This includes 15 distribution,
5 categorical, 34 focal-pair, 14 spectral, 16 redistribution, and 16 executable numerical/record
capture tests. A separate opt-in benchmark test passes all six workloads and every alternating update.

The hardware run exposed and fixed two large-offset float32 centroid problems. Fractional moments
now materialize centered inputs across a dispatch boundary; focal moments and residual squares
correct residual mean error. Original numerical tolerances were retained. A non-dyadic singleton
fixture also guards sample-support detection. Polygon-composed categorical winners are explicitly
based on represented coverage weights; ideal geometric ties can be disturbed by float32 clipping.
Direct tuple-fed exactextract tie assertions remain exact. Real lens capture tests also exposed and
fixed retention of transient captures, without retaining imported caller-owned buffers.

Repository gates: `nvm use` (22.22.1), `yarn install`, `yarn lint fix`, both website build commands,
and raster/spatial/computation-view ESM/CJS/declaration import checks have passed. The final `yarn build` after numerical fixes and formatting passed. The final focused node run
passed **161 tests across 19 files**, with 2 skipped. This includes documentation/layering,
ontology, maximal kernel audits, boundary checks, fixture provenance and real-example construction.
The broad node run completed with **12,061 passed, 355 skipped and 3 failed**; those three
integration failures (canonical scenario IDs and the focal subdirectory barrel) were corrected,
and their complete files pass in the final focused run. The entire broad node suite was not rerun
after those fixes. Both website builds passed before the last numerical and capture-retention
repairs; the final build covers those TypeScript changes.
The full `yarn test` command was not run in this isolated worktree; targeted native GPU execution
is reported explicitly, separate from the original checkout's full-suite evidence.
The original expansion's two known physical-rendering pixel failures are separate baseline evidence;
this tranche must record its own outcomes rather than inheriting a blanket green result.

## Bounded measurements

Raw samples, device metadata and source hashes (local research log, not checked in)
retain setup, upload, host compilation, first execution, five repetitions, readback, three alternating
updates and graph-accounted transient bytes. Execution here means host encode + submit + queue
completion, **not** GPU timestamp duration. First execution can include deferred driver compilation.
No other verification browser was active, but node/build jobs were active on the host.

| Workload | Cells | Host compile, ms | First execution, ms | Repeated median, ms | Graph transient bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Focal pairs, radius 2 | 64 | 10.3 | 82.4 | 0.4 | 2,048 |
| Focal pairs, radius 2 | 1,024 | 1.7 | 64.2 | 0.4 | 32,768 |
| Fractional distributions, 8 zones | 64 | 2.2 | 138.5 | 0.7 | 2,924 |
| Fractional distributions, 8 zones | 1,024 | 2.1 | 88.4 | 3.2 | 49,228 |
| Dasymetric, 8 sources / 4 targets | 64 | 2.2 | 312.5 | 1.4 | 3,880 |
| Dasymetric, 8 sources / 4 targets | 1,024 | 2.0 | 299.3 | 2.4 | 69,668 |

These are small bounded observations, not maximum-capacity or skewed-domain scaling evidence.
Categorical/spectral throughput, GPU timestamps, driver peak memory and optimized CPU-library
comparisons remain unmeasured. No speedup claim or performance threshold follows from these samples.

## Integration status

All 71 tranche files were restored into the shared checkout during the GitHub master integration.
The original isolated worktree and preservation stash remain available for comparison.
On the combined tree, the six targeted native GPU files passed all **100 tests**, and the seven
focused node files passed all **61 tests**. RS02 was regenerated with the pinned exactextract,
NumPy and Shapely environment after its reproduction-command path changed; only the generator
hash changed, with every numerical fixture value unchanged.

These results cover this tranche. Full repository, package and website verification belongs to
the encompassing upstream integration and is reported separately from the isolated results above.
