# Direction (2026-11-19)

Scope: where `@luma-contributors/analysis` should go in the two quarters after the correctness and release work. This report supplies options and evidence; the other area reports size the correctness work. Finding IDs are `DIR-101` to `DIR-116`; here a finding is an option (101 to 109) or a risk (110 to 116).

Summary:

- The library has 633 exported `GPU*` classes (504 contributors and 129 operation forms), 548 catalogued operators, 92 capabilities and twelve user questions. Breadth is the strongest property; no public tool found in this review covers the same range in a browser on WebGPU (see Comparables for what was and was not searched).
- Depth is uneven. By script: 130 of 633 classes have a CPU twin, 361 appear in a test that loads a checked-in reference fixture, 496 appear in a kernel-audit case, 385 appear in a lens scenario, 240 have a Kepler catalog binding. In the catalog, 370 of 548 operators have contract `unaudited`, 201 have no tier and 288 carry no scale evidence.
- The front door is still missing. The documented first GPU result is about 55 lines for five points in a 2 by 2 grid (`docs/api/operation-forms-and-cpu-twins.md`). There is no file reader, no GeoArrow adapter, no worker story, no Node GPU path, and no published number that compares a contributor with a CPU library on any machine other than one Apple GPU.
- Runtime reach is wide because the library asks for almost nothing: no `requiredFeatures`, no `enable` directive, no subgroups or `shader-f16`, only optional `timestamp-query` and raised buffer limits. WebGPU is on by default in Chrome, Edge, Safari 26 and Firefox on Windows and Apple Silicon; it is not in Firefox on Linux or Android.
- Recommended three bets: (1) five flagship journeys finished end to end, with a thin facade, GeoArrow in and out, a published benchmark on two GPU vendors and a documented limit stated in device terms; (2) a trust layer: a stable core and an experimental tier, with the catalog filled in so that it is a machine-readable contract; (3) the agent and host contract built on that catalog. Three stops: new operator families before the alpha is used, a generic input/output/run runner, and retrospective lens and operation-form coverage.
- Decisions the maintainer must take first: the canonical repository and package identity, the stability tiers, which five journeys, and whether the luma.gl dependency follows the `beta` tag (alpha.17 appeared one day after the pin).

## Method

- Lead read: `dev-docs/roadmap.md` (all sections named in the task), `dev-docs/research/library-review-2026-11-17/README.md` and `thinking-partner.md` (round one, in full), the new-contributors README, the first pages of the scalability audit and the workflow-UX study, `modules/analysis/README.md`, `dev-docs/release-package.md`, `docs/api/operation-forms-and-cpu-twins.md`, `.github/workflows/test.yml`, `modules/analysis/src/analysis-core/graph-support/device-limits.ts` and parts of `modules/kepler/src/workflows` and `descriptors`. Not read in full: the five new-contributors research files, `scalability-limits-2026-11-15/README.md` (331 KB), the cpu-suitability README, the gpu-geospatial implementation and performance folders (only the performance integration notes), `coverage-matrix.md` beyond its header.
- Scripts run (all read-only, in `/private/tmp/claude-501/.../scratchpad/review19/direction/`): `census2.py` (class list from `^export class GPU\w+` outside `src/testing`, then identifier-token membership in docs, lenses, Kepler, examples, website, ontology and tests); `oracle.py` (test files that contain a path to a `fixtures/*.json` file, then class membership; kernel-audit membership from files under `test/analysis-core/kernel-audit/`); `cat2.ts`, `cat3.ts`, `cat4.ts` (esbuild bundles that call `queryGPUAnalysisOperators({})`, `getGPUAnalysisCapabilityCatalog()` and `getGPUAnalysisQuestionCoverage()` from the source and print counts). No probe was run against a GPU, no test, build or typecheck was run.
- Subagents (three Opus slices, web only, rules from the brief): runtime availability; nearest comparable tools; large-library practice and agent-facing toolboxes. Their claims that I could not re-open are marked UNVERIFIED. I re-checked the facts that matter for the plan (luma.gl publication dates with `npm view @luma.gl/graph time`, `dist-tags`).
- The identifier-token census measures mention, not exercise. "Lens" counts classes named in `modules/lenses/src/computation-lenses/scenarios`; "Kepler" counts classes named in `modules/kepler/src/analysis/catalog-*.ts`; "fixture" counts classes named in a test file that also names a checked-in `fixtures/*.json` file, which includes tests where the fixture is for a different class in the same file. Treat each as an upper bound on independent verification.

## Census

| Count | Value | Command or source |
| --- | --- | --- |
| Source files, lines | 1,770 files, 565,068 lines | `find modules/analysis/src -name '*.ts' \| xargs wc -l` |
| Exported `GPU*` classes | 633 (504 not named `*Operation`, 129 named `*Operation`) | `census2.py` |
| Classes with a CPU twin (`compute<Name>OnCPU` or `CPU<Name>Operation` exists) | 130 | `census2.py` |
| Recipes (`export function add*Recipe`) | 54, in 26 files in `gpu-spatial-analysis/recipes` | `rg` |
| Catalogued operators; capabilities; questions | 548; 92; 12 | `cat2.ts` (`queryGPUAnalysisOperators({})`) |
| Catalog tier | `scale` 317, `reference` 30, no tier (`unknown`) 201 | `cat2.ts` |
| Catalog contract evidence | `unaudited` 370 of 548; `status: proven` 22, `partial` 526 | `cat3.ts`, `cat4.ts` |
| Catalog scale evidence (`representativeSourceRows` set) | 260 of 548; 288 have none | `cat4.ts` |
| Catalog `semantics` | `unspecified` 433, approximate 61, exact 28, model-based 16, stochastic 5, robust-exact 5 | `cat2.ts` |
| Catalog `readback` | none 315, explicit-adapter 8, unknown 225 | `cat2.ts` |
| Operators per primary question | where-concentrated 21, where-significant 35, how-much-per-area 44, near-or-reachable 47, how-related 30, value-where-unmeasured 23, what-changed 24, how-it-moves 37, shape-of-surface 38, which-groups 22, what-is-at-each-place 13, prepare-data 214 | `cat3.ts` |
| Fixtures | 790 entries in `modules/analysis/test/oracles/fixtures`; pinned reference libraries include libpysal 4.15.0, esda 2.10.0, pysheds 0.5, scikit-gstat 1.0.24, scikit-learn 1.9.1, networkx 3.7, gstools, pykrige, whitebox | `ls`, `modules/analysis/test/oracles/requirements.txt` |
| Test files; test files that load a fixture path | 3,261; 395 | `oracle.py` |
| `new GPUCommandGraph(` | tests 3,594 in 1,421 files; examples 130 in 93 files; website 154 in 14 files; Kepler 13 in 10 files; analysis src 2 | `rg -o` |
| Website runs | 667 run files in 20 groups under `website/src/gpu/runs/entries` | `find` |
| Lens scenario files | 300 (427 scenarios by the roadmap) | `find`, `LR-20` |
| Kepler | 295 source files; 217 hand-written descriptors; 28 curated analyses, 16 with CPU paths and 12 requiring WebGPU | roadmap AX section |

Per family (all columns are class counts):

| Family | Classes | Operation forms | CPU twin | Docs mention | Lens | Kepler binding | Fixture test | Kernel audit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpu-spatial-analysis` | 352 | 79 | 81 | 278 | 220 | 156 | 186 | 270 |
| `raster-analysis` | 99 | 13 | 13 | 88 | 54 | 37 | 78 | 86 |
| `gpu-network` | 58 | 17 | 17 | 43 | 36 | 27 | 20 | 40 |
| `gpu-terrain` | 57 | 11 | 11 | 46 | 46 | 0 | 33 | 46 |
| `dataframe-analysis` | 41 | 7 | 7 | 37 | 26 | 5 | 23 | 34 |
| `pointcloud-analysis` | 11 | 0 | 0 | 11 | 0 | 11 | 11 | 11 |
| `analysis-core` | 9 | 2 | 1 | 9 | 3 | 2 | 9 | 7 |
| `computation-record` | 4 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| `geospatial-analysis` | 2 | 0 | 0 | 2 | 0 | 2 | 1 | 2 |
| All | 633 | 129 | 130 | 518 | 385 | 240 | 361 | 496 |

Gaps worth naming: no terrain class has a Kepler binding (57 classes, 46 in lens scenarios); no point-cloud class is in a lens scenario or has a CPU twin; the three families with the weakest fixture coverage are `gpu-network` (20 of 58), `gpu-spatial-analysis` (186 of 352) and `dataframe-analysis` (23 of 41).

Seven non-operation classes have no named consumer in lenses, Kepler, a website run or the Kepler curated set: `GPUAdamStep`, `GPUFittingCoordinateEncoder`, `GPUFittingGradientSums`, `GPUFittingMinibatchOrder`, `GPULocalPermutationReplay`, `GPUSnapRoundedNoding`, `GPUWideNumericScan`. The website token search is unusable for this purpose because the run files reference the catalog.

## Position

### What the library is

A set of 504 contributor classes (plus 129 operation forms) that add WGSL compute kernels to a caller-owned `GPUCommandGraph` from published luma.gl. It is a kernel library, not a runtime: the caller allocates every input and output buffer, compiles, submits, reads back and destroys. Metadata lives beside the code: a catalog (548 operators with inputs, outputs, coordinate model, status words, tier, limits, evidence), twelve analysis questions with operator-to-question rules (`gpu-spatial-analysis/ontology/analysis-experience`), a descriptor layer in the Kepler module and a lens layer that records executions.

Capability map by family (classes; the largest sub-areas by class count):

| Family | Classes | Largest sub-areas |
| --- | ---: | --- |
| `gpu-spatial-analysis` | 352 | spatial regression 36, geometry refinement 22, spatial weights 18, interpolation 16, regionalization 10, cell topology 10, distribution dynamics 9, autocorrelation 8, validation 8, trajectories (cleaning 7, staypoints 6, mobility 4, others), point process 7, joins 7, space-time tests 6, global coordinates 6 |
| `raster-analysis` | 99 | time series 21, embeddings 12, remote sensing 8, algebra 5, cost distance 5, spectral indices 4, isolines 4 |
| `gpu-network` | 58 | optimization 11, analysis 6, flow aggregation 5, accessibility 4, time geography 3, centrality 3, structure 3 |
| `gpu-terrain` | 57 | processing 11, hydrology 11, illumination 8, features 7, analysis 5, topographic position 4, relief 4 |
| `dataframe-analysis` | 41 | pair statistics 17, column classification 7, composite indicators 3 |
| `pointcloud-analysis` | 11 | ground filtering, ICP, features, change detection |

By user question (primary assignment), the catalog is dominated by data preparation: 214 of 548 operators (39%) have `prepare-data` as their only question. The substantive analysis questions have 13 to 47 operators each. That shape is a property of the catalog rules, not a measurement of use; it does mean that "633 operators" overstates the number of end-user analyses: about 334 are assigned to the eleven analysis questions.

### Depth against breadth

| Property | Classes (of 633) | Share |
| --- | ---: | ---: |
| Docs page mentions the class | 518 | 82% |
| In a kernel-audit case | 496 | 78% |
| In a lens scenario | 385 | 61% |
| Test file names it and loads a fixture JSON | 361 | 57% |
| Kepler catalog binding | 240 | 38% |
| CPU twin | 130 | 21% |
| Catalog `contract` audited (of 548 operators) | 178 | 32% |
| Catalog `correctnessOracle` path recorded (of 548) | 260 | 47% |
| Catalog `tier` recorded (of 548) | 347 | 63% |

Reading: the library has documentation and structural audits for most operators and an independent numerical reference for roughly half. A CPU twin exists for one in five; for the rest, the only CPU path is the user's own. The catalog records its own incompleteness (`unaudited`, `unknown`), which is the honest position, but a consumer cannot yet filter on "verified against an external library on a GPU".

Evidence that is single-source: every timing in the repository comes from one Apple M-class machine under headless Chromium (`dev-docs/research/new-contributors-2026-11-11/README.md` states "any other machine" as not shown; `dev-docs/upstream-proposals.md` says historical measurements cover one adapter; the roadmap's SC and AX acceptance "was measured on one Apple M-class machine"). The only recorded non-Apple case in the roadmap is an open row (`FP64-05`, "Measure on a non-Apple GPU"). The GPU-only defect classes that the review found (Metal division, barrier uniformity, aliasing) were found on Metal; the other vendors' compilers have not been exercised in this repository.

No comparison with a CPU geospatial library has a published number. `UF-03` and `PLAT-07` say so: one synthetic-zone query measured to scale factor 5 in 216.3 ms; "none compares against alpha.13". `docs/api/gpu-spatial-analysis/statistical-expansion.md:303` states that its checks "are correctness checks, not evidence of a speedup". A user cannot learn from the repository when to prefer a contributor over esda, GeoPandas or DuckDB.

### Who can use it today

Install and import: the package is unpublished at `0.1.0-alpha.0`. Its peers are `@luma.gl/core`, `@luma.gl/shaders`, `@luma.gl/graph` at `~10.0.0-alpha.16` and `apache-arrow >=17.0.0`. luma.gl's `latest` dist-tag still points to `10.0.0-alpha.5`; `beta` points to `10.0.0-alpha.17` (published 2026-11-18T14:52Z, 37 hours after alpha.16 at 2026-11-17T01:44Z). A new user must install the `beta` tag explicitly. The accepted range `~10.0.0-alpha.16` admits alpha.17 by semver; the roadmap records alpha.16 as the only tested version.

Runtimes, as of 2026-11-19 (sources in the table; accessed 2026-11-19):

| Runtime | State | Source (date) |
| --- | --- | --- |
| Chrome desktop (Windows x64, macOS, ChromeOS) | Default on since 113 | gpuweb wiki "Implementation Status" (edited 2026-10-02), https://github.com/gpuweb/gpuweb/wiki/implementation-status |
| Chrome Linux | Intel Gen12+ from 144; NVIDIA on Wayland from 147; others need flags | same; https://developer.chrome.com/docs/web-platform/webgpu/news |
| Chrome Android | Android 12+ on ARM, Qualcomm, Intel GPUs from 121; Imagination on Android 16+ from 139 | gpuweb wiki |
| Edge, Opera, Samsung Internet | Chromium parity (Edge 113+; Samsung Internet 24+; Xclipse GPUs TBD) | https://caniuse.com/webgpu |
| Safari macOS, iOS, iPadOS, visionOS | Safari 26.0, default on; desktop default only on macOS Tahoe, flag on older macOS | https://webkit.org/blog/17333/webkit-features-in-safari-26-0/ (2025-09-15); caniuse |
| Firefox | Windows 141; Apple Silicon macOS 147 (2026-01-13); Intel Mac and Linux Nightly only; Android off by default | gpuweb wiki; https://www.theregister.com/2026/01/14/firefox_and_tbird_147/ |
| Node.js | `webgpu` package (Dawn bindings, `dawn-gpu/node-webgpu`) with d3d12, metal, vulkan backends; software Vulkan supported; no canvas | https://github.com/dawn-gpu/node-webgpu |
| Deno | Built-in WebGPU on wgpu; whether the flag is still required in 2026 is UNVERIFIED | https://docs.deno.com/api/deno/~/Deno.webgpu |
| Bun | No native `navigator.gpu` found; third-party Dawn FFI (`bun-webgpu`) and wgpu bindings | https://github.com/kommander/bun-webgpu |
| Headless Chromium on Linux CI | Works with ANGLE Vulkan flags and a real driver; without a driver it silently falls back to SwiftShader | https://developer.chrome.com/blog/supercharge-web-ai-testing |
| Share of global page views | 88.63% (86.83% full plus 1.8% partial), StatCounter October 2026 | https://caniuse.com/webgpu |
| Share with a working adapter | 84.24% (undated, no sample size) | https://web3dsurvey.com/webgpu |
| Standard status | W3C Candidate Recommendation Draft (dated 2026-06-23 CRD found); web-features rates it "limited availability" because Firefox is incomplete | https://www.w3.org/TR/2026/CRD-webgpu-20260623/; https://web-platform-dx.github.io/web-features-explorer/features/webgpu |

What the library requires of the device (read from source):

- Features: none required. `rg "requiredFeatures"` finds no use in `modules/*/src`. The only feature probe is `device.features.has('timestamp-query')` in `computation-record/submit-timed-command-graph.ts:44`, optional. `shader-f16` appears only in a comment (`raster-analysis/raster-embeddings/gpu-raster-embedding-dequantize.ts:94`: decoding "without requiring `shader-f16`"). `subgroups` appears only in a comment (`gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores.ts:130`). No `enable` directive exists in any WGSL string (`rg "enable [a-z_0-9]+;"` finds none).
- Limits: `getAnalysisRequiredLimits` (`analysis-core/graph-support/device-limits.ts:49`) raises `maxStorageBufferBindingSize` and `maxBufferSize` to the adapter's maxima, requests up to 16 storage buffers per stage (floor 8, the default) and optionally raises workgroup storage and invocations. Without that request a device is bound to the 128 MiB binding and 256 MiB buffer defaults, and contributors compute their limits from `graph.device.limits` (31 files read `maxStorageBuffersPerShaderStage`; kernels needing more bindings are split to fit 8).
- Arithmetic: float32 and `u32`/`i32` atomics (240 files use `atomic<`; 186 use `atomicAdd`, `atomicMax` or `atomicMin`); fp64 is emulated (double-single and integer paths).

Consequence: the code does not depend on any feature that only Chrome ships, so Safari 26 and Firefox on Windows and Apple Silicon are plausible targets on paper. None has been run. The binding that will bite on other devices is capacity: Web3DSurvey reports every adapter at least 128 MiB for `maxStorageBufferBindingSize`, only about 16% reach roughly 2 GiB; Apple adapters 91% at least 768 MiB and 47% at about 2 GiB; Safari 100% at least about 429 MB; iOS at least about 307 MiB; Android 75% at least 256 MiB; `maxComputeWorkgroupStorageSize` 32 KiB for 99% of adapters and 64 KiB for about 5% (https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize and sibling pages, undated, skewed to visitors of graphics sites). The repository's documented limits, derived from "binding size", are therefore device-dependent by design (`scalability-limits-2026-11-15`: 85 of 373 audited limits are binding-size limits).

Who has actually used it: Kepler (the `modules/kepler` host: 240 class bindings, 217 descriptors, 28 curated analyses, a workflow workbench), the lens recorder (private), the examples and the website. All are first-party. There is no external consumer, no published package, no issue tracker traffic and no CI result in the repository record. `.github/workflows/test.yml` runs checkers, lint, build, typechecks and Node tests (the visible portion; the file continues) but the roadmap row `INT-08` says "CI execution is unverified" and the remote is private (`LR-04`).

## Comparables

Facts below were collected by a web-research subagent from registries, release pages and project documentation, accessed 2026-11-19; I did not re-open each page (I did re-check the luma.gl dates with `npm view`). Items it could not find on a primary page are marked UNVERIFIED. "This library" means the repository's analysis package as it is on 2026-11-19, including its limits.

| Tool | Latest (date) | Does that this library does not | This library does that it does not | Link |
| --- | --- | --- | --- | --- |
| RAPIDS cuSpatial | 25.04.0 (2025-04-10), final; dropped from RAPIDS at 25.06; repo archived | Nothing current. Historically CUDA point-in-polygon, quadtree joins, trajectories | Everything: it is unmaintained and NVIDIA-only | https://docs.rapids.ai/notices/rsn0045/ |
| RAPIDS cuDF, cuGraph, cuML | 26.10 (cuDF 26.10.2, 2026-10-22) | Multi-GPU and Dask scale; `cudf.pandas` zero-change acceleration; mature graph algorithms; broad ML | No CUDA, no server; any vendor's GPU; geospatial operators at all (RAPIDS has none after cuSpatial) | https://docs.rapids.ai/notices/rsn/ |
| GeoPandas, Shapely | GeoPandas 1.2.0 (2026-09-28); Shapely 2.2.0 (2026-10-07) | Exact GEOS overlay and robust predicates; GDAL/pyogrio I/O for every format; pyproj CRS handling; the interchange standard | GPU parallelism; runs in the page with nothing installed | https://docs.geopandas.org/en/latest/docs/changelog.html |
| PySAL | v26.07 (2026-07-31); libpysal 4.15.1, esda 2.10.0, spreg 1.9.2 (2026-11-09), mgwr 2.2.1 (2024-01-06, stale), spopt 0.8.0, momepy 1.0.0, tobler 0.14.1 | The reference for ESDA and spatial econometrics; peer-reviewed, citable; panels in spreg; MGWR; regionalisation; areal interpolation; format and CRS I/O via GeoPandas | GPU permutation tests and large-N weights; interactive recompute; no Python stack | https://zenodo.org/records/21726526 |
| WhiteboxTools, Whitebox Workflows | WhiteboxTools 2.4.0 (2024-05-22); Workflows 2.0.6 (2026-06-14); "Next Gen" Rust rewrite announced 2026-06-27 (about 750 tools) | A deep hydrology, terrain and LiDAR catalogue on native multicore CPU | GPU kernels; browser execution; slider-rate re-run | https://lists.osgeo.org/pipermail/discuss/2026-June/040372.html |
| GRASS | 8.5.0 (2026-05-08); GPL-2.0+ | Decades-validated raster, vector and imagery modules; topological vector model; disk streaming of large rasters; r.watershed, r.sun, viewshed | GPU compute; zero-install client deployment | https://github.com/OSGeo/grass/releases |
| Google Earth Engine | earthengine-api 1.7.53 (2026-11-16); noncommercial EECU-hour quotas since 2026-04-27 | Petabyte catalogue next to the compute; planetary-scale map and reduce; built-in classifiers | Offline and private on local data; no quota or account | https://developers.google.com/earth-engine/guides/noncommercial_tiers |
| DuckDB spatial, DuckDB-Wasm | DuckDB 2.0.0 LTS (2026-10-21); GEOMETRY core type in 1.5.0 (2026-03-09); about 160 `ST_` functions on the 2.0 docs page (subagent count); duckdb-wasm `next` dev119 (2026-11-19) | SQL joins and aggregates over GeoParquet, out-of-core, GDAL reading, R-tree index; in the browser reads GeoParquet with no server | Raster analysis, spatial statistics, kriging, routing, terrain (DuckDB spatial has none of these); GPU | https://duckdb.org/2026/10/21/announcing-duckdb-200 |
| Turf.js | 7.4.0 (2026-08-03) | Exact GeoJSON vector operations; tiny; ubiquitous | Scale beyond about 1e5 features; statistics, rasters, routing | https://github.com/Turfjs/turf/releases |
| deck.gl, luma.gl | deck.gl 9.4.1 (2026-10-28), 10.0.0-alpha.1 (2026-11-12); luma.gl 9.4.2 stable (2026-09-19), 10.0.0-alpha.17 (2026-11-18) | Production rendering, picking, tiling, 3D Tiles. Aggregation layers (Screen grid, Hexagon, Contour, Grid, Heatmap) are ported to WebGPU in 9.4 but the guide still says WebGPU is not production ready | Statistics, weights, regression, hydrology; typed results usable outside a render pass | https://deck.gl/docs/developer-guide/webgpu |
| loaders.gl, GeoArrow, Parquet, COG, PMTiles | loaders.gl 4.5.5 (2026-11-17), 5.0.0-beta.12; geoarrow-js 0.3.3 (2025-10-27); @geoarrow/deck.gl-layers 0.3.2 (2026-04-15); geoarrow-rs 0.9.0 (2026-09-11); lonboard 0.17.0; hyparquet 1.33.1; parquet-wasm 0.9.0; geotiff.js 3.0.5 (3.1.0-beta.0); PMTiles 4.5.0 | Format coverage and columnar I/O: the feeders this library lacks | Analysis. Of these only geoarrow-rs has CPU WASM geometry operations | https://loaders.gl |
| Apache Sedona, SedonaDB | Sedona 1.9.1 (2026-08-05); SedonaDB 0.5.0 (tag 2026-10-02); 0.4.0 had 176 `ST_`/`RS_` functions and an opt-in NVIDIA GPU spatial join (Docker only) | Distributed spatial SQL over billions of rows; KNN joins; raster map algebra and zonal statistics in SQL; Zarr | Browser and client execution; spatial statistics, kriging, routing, hydrology; vendor-neutral GPU | https://sedona.apache.org/latest/blog/archive/2026/ |
| xarray-spatial | 0.10.20 (2026-10-27); README claims 150+ raster functions on NumPy, Dask, CuPy | Closest Python raster analogue: hydrology (D8, D-inf, MFD), fire and flood models, proximity, pathfinding; Dask out-of-core | Vendor-neutral GPU (its GPU path is CUDA-only); browser; vector and network statistics | https://github.com/xarray-contrib/xarray-spatial |
| Datashader, TorchGeo | Datashader 0.19.1 (2026-05-19); TorchGeo 0.10.1 (2026-09-29) | Billion-point server rasterisation; geospatial deep-learning datasets and pretrained models | Interactive client GPU statistics | https://pypi.org/project/torchgeo/ |
| kepler.gl | 3.3.0-alpha.24 (2026-11-18) | No-code exploration UI with DuckDB SQL; its AI assistant runs LISA, regression, buffer and isochrone tools on WebAssembly/CPU (subagent's source: https://geodaai.github.io/docs-kepler-ai/guide.html) | The GPU implementations of the same statistics at larger N | https://github.com/keplergl/kepler.gl |
| CARTO Analytics Toolbox | BigQuery 1.26.0 (2026-09-29); Snowflake 1.24.0 (2026-04-27, space-time hot-spot classification) | Warehouse-scale hot spots, GWR, k-ring smoothing on H3 and Quadbin without moving data | No warehouse cost; offline; interactive recompute | https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/release-notes |
| Esri ArcGIS Pro, GeoAnalytics Engine | Pro 3.6 (2025-11-13); GeoAnalytics Engine 2.1 (June 2026); tool count UNVERIFIED | Broadest commercial catalogue; enterprise data; network datasets; some NVIDIA GPU tools | Open licence; zero install | https://pro.arcgis.com/en/pro-app/get-started/whats-new-in-arcgis-pro.htm |
| QGIS Processing | 4.0 (2026-03-06, Qt6); algorithm count UNVERIFIED | Aggregates GDAL, GRASS, SAGA, Whitebox providers; graphical modeller | GPU compute; browser embedding | https://qgis.org/project/visual-changelogs/visualchangelog40/ |
| DGGS libraries | h3-js 4.5.0 (2026-07-01); h3 (Python) 4.5.1 (2026-11-18); a5-js 1.0.0 (2026-10-15); s2js 1.44.0 | Canonical CPU cell mathematics; warehouse interoperability | GPU bulk indexing and aggregation | https://github.com/felixpalmer/a5 |

What changed in 2026, relative to the repository's own research (the 2026-10-23 and 2026-11-11 folders predate these):

- cuSpatial is end of life. The closest NVIDIA-side successor found is SedonaDB 0.4's opt-in GPU spatial join, which is NVIDIA-only and Docker-only. The "RAPIDS as the GPU geospatial reference" framing in older research no longer applies.
- DuckDB spatial is the real competitor for data jobs: GEOMETRY is a core type since 1.5.0, DuckDB 2.0.0 LTS shipped on 2026-10-21, and DuckDB-Wasm reads GeoParquet in the browser. It has no raster analysis, spatial statistics or GPU path. The Kepler host already embeds a DuckDB instance for its SQL step (`KE-WF-K`), so DuckDB is a partner for I/O and joins rather than something to compete with.
- deck.gl 9.4 ported its aggregation layers to WebGPU; the guide still says WebGPU is not production ready. luma.gl 10 is still on alphas (alpha.17 on 2026-11-18; `latest` is still alpha.5).
- Whitebox announced a Rust rewrite (June). The classic tool has had no release since 2024-05. Earth Engine introduced noncommercial quotas on 2026-04-27. PySAL shipped v26.01 and v26.07 with no GPU backend.
- A5 reached 1.0 (2026-10-15). The repository's roadmap row `RTS-08` says A5 is not started.

WebGPU geospatial analysis libraries found by the subagent's GitHub and npm search (no WebGPU library with a statistics, regression or kriging catalogue appeared; searches for kriging, Moran, flow accumulation, spatial statistics and IDW returned nothing):

| Project | Scope | Maturity | Link |
| --- | --- | --- | --- |
| mapgpu | WebGPU map and globe renderer with a small analysis module (line of sight, buffer, elevation query, route sampling, hillshade) | MIT, 114 stars, created 2026-03-17, last push 2026-04-22 | https://github.com/mgurbuz-tr/mapgpu |
| tgpu-htj2k | WebGPU HTJ2K codec with an operation-graph runtime and a spatial-analysis toolbox for imaging | 1 star, no licence, last push 2026-08-01 | https://github.com/xinaesthete/tgpu-htj2k |
| Browzarr | Browser visualisation and analysis of Zarr and NetCDF; GPU-compute scope not verified | Apache-2.0, 79 stars, last push 2026-11-19 | https://github.com/EarthyScience/Browzarr |
| wasm-spatial-core | CPU WebAssembly spatial engine | MIT, 4 stars | https://github.com/reed-soul/wasm-spatial-core |

Reading: the position "a browser GPU library with a spatial-statistics, regression, interpolation, terrain and network catalogue" is unoccupied as far as this search shows. The neighbours that users will actually compare against are PySAL (correctness and citations), DuckDB spatial (data and SQL) and Whitebox or GRASS (terrain and hydrology depth). Against each, the library's argument must be interactivity, privacy and zero install, supported by a published number; today the repository has none for any of the three.

## Users and jobs

Who would adopt a browser GPU spatial analysis library, in order of how directly the repository serves them:

| User | Job | Why browser GPU | Alternative they use today |
| --- | --- | --- | --- |
| Host application developer (Kepler, deck.gl or MapLibre apps) | Interactive analysis on data already resident in the browser: hot spots, density, joins, filters on 1e5 to 1e7 rows, recomputed on slider or brush | Data is already in the page; no server round trip; resident results can drive layers | Server tile or SQL services; DuckDB-Wasm; Turf.js on small data; kepler.gl's own WebAssembly analytics |
| Data journalist or public-health analyst without a Python stack | Rate maps, cluster maps, spatial regression, interpolation on one dataset, shared as a link | No install, no upload (privacy), shareable state | GeoDa (desktop), QGIS, hosted notebooks |
| Educator or workshop author | Show what a spatial statistic does with sliders | Immediate recompute | Static tutorials, GeoDa |
| Agent or assistant author (Kepler assistant, WebMCP-style page tools) | A set of operators an LLM can select, parameterise and explain, run on the user's data in the page | Data does not leave the page; fast enough to iterate | SQL or PyQGIS generation on a server |
| Server-side pipeline author | Batch analysis in Node on a GPU box | Same code as the browser | cuSpatial, PySAL, GRASS, Whitebox |
| Library author (other visgl packages) | Reuse kernels | Published, typed contributors | none |

What each needs before a 634th operator, against what the repository has (verified by reading):

| Need | Exists today | Missing |
| --- | --- | --- |
| Installable package | `0.1.0-alpha.0` metadata, packed-consumer gate (`scripts/check-packed-consumer.mjs`), required Arrow peer, licence inventory | Nothing published; repository URL names `visgl/luma-contributors` while the remote is private (`LR-04`, `UF-01`); luma.gl `latest` tag is alpha.5, so the install instruction needs `@beta`; `apache-arrow` peer is required even for a validator |
| First result in a few lines | CPU variogram example with expected values; a 55-line GPU point-density example | No short GPU path; Local Moran in the test harness hides weight-overflow status (11-17 thinking partner; still no facade); tests contain 3,594 `new GPUCommandGraph(` calls and ten `Rig`/`Harness` classes |
| Data in | Caller supplies typed arrays or `GraphDataView`s. GeoArrow offsets are the geometry convention (`docs/api/gpu-spatial-analysis/operators-joins.md:27`, `operators-geometry.md:326`); `analysis-core/graph-support/arrow-analytics-normalization.ts` reads Arrow columns | No GeoArrow reader or `WKB`/GeoParquet/COG/PMTiles adapter; loaders.gl appears only as a root dev dependency (`package.json:43`); no ring-closure or CRS handling on ingest |
| Results out | Caller-owned output views; `readAsync` in examples; Kepler publishes datasets | No documented Arrow table output; no layer hand-off without readback documented in the analysis package (whether a deck.gl 10 layer can bind a caller-owned luma buffer directly is UNVERIFIED here; the Kepler example has resident-input code but no verified direct binding) |
| Scale statements to trust | 373 audited limits (`scalability-limits-2026-11-15`), catalog `limits` on 172 operators, `getAnalysisMaximumRowCount` | Limits are mostly constants, not "device memory"; 288 operators carry no `representativeSourceRows`; one-GPU timings; no numbers against a CPU library |
| CPU fallback | 130 twins (`compute<Name>OnCPU`), plus `GPUCPUOperation` bridge with measured transfer costs | 503 classes without a twin; CPU twins use float64 and have recorded behavioural differences (roadmap "CPU arithmetic", "CPU budgets", "CPU output tails"); no automatic route (`CPU-08`, waits on upstream `UPS-CPU-1`) |
| Worker story | None in `modules/analysis` (the only worker mention is Kepler's cluster engine, `modules/kepler/src/cluster/kepler-cluster-engine.ts`) | Whether luma.gl devices run in a worker is an upstream question; no guidance, test or example here |
| Server-side Node | 128 CPU twins and `*.node.spec.ts` specs run in Node | No GPU execution in Node: no `webgpu` (Dawn) dependency in `package.json`, no Node GPU test project; the GPU suite is headless Chromium on Apple Metal only |

The 2026-11-17 review asked for one front door. Nothing in the repository since then adds one: the roadmap rows `LR-04`, `LR-08` to `LR-26` do not name a facade, and the AX/KE-WF work built its runners inside Kepler (`modules/kepler/src/workflows/workflow-runner.ts`) rather than in the library.

## Options

| ID | Option | For whom | Builds on | Cost | Risk | Evidence it worked | Displaces |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIR-101 | Five flagship journeys, finished end to end | Host developers, analysts, anyone judging the alpha | recipes in `gpu-spatial-analysis/recipes`, 28 curated analyses, oracle fixtures, `docs/api` quick starts | L | Scope creep; each journey exposes defects in the lower layers | A stranger completes each journey from a clean install in under 30 minutes; each has a CPU reference check and a two-GPU benchmark | New families; retrospective lenses |
| DIR-102 | GeoArrow in and out; layer hand-off without readback; loaders.gl sources | Everyone with real data | `arrow-analytics-normalization.ts`, `GPUCommandGraph.importGPUData`, GeoArrow offset layouts in the join and geometry docs | M | Interop surface drifts with GeoArrow, loaders.gl and deck.gl alphas; luma ownership rules | Round trip: GeoArrow table to result Arrow table without a manual layout step, on three public datasets | Hand-rolled packers in examples |
| DIR-103 | Thin per-journey facades; no generic runner | Newcomers, Kepler, lenses, website | `RecipeBuilder`, `GPUContributorOperation`, `createNetworkCentralityContinuation` pattern, `device-limits.ts` | M | A second graph vocabulary; resource ownership mistakes | Hello-world GPU journey in at most 15 lines; one facade replaces at least three copies of runner code | The generic `input/output/run` idea |
| DIR-104 | Device-memory scale and a published benchmark suite on two vendors | Anyone choosing between GPU and CPU | `scalability-limits-2026-11-15`, `test-benchmarks`, SpatialBench scale factors | L | Measurement noise; vendor drivers expose defects; time sinks | A `benchmarks/` report with raw numbers on at least Apple, NVIDIA or AMD, and an Intel iGPU; documented limits phrased as bytes | New-limit raising for its own sake |
| DIR-105 | Stable core and experimental tier; delete what has no consumer or oracle | Maintainer, users | catalog `tier`, `evidence.status`, `contract`, public-surface checker | M | Argument over membership; breaking imports | Package exports split; deprecation policy exercised once; catalog has no `unknown` tier | Operator count as the headline |
| DIR-106 | Upstream generic pieces to luma.gl | Maintainer, luma.gl users | `dev-docs/upstream-drafts` (four drafts), `upstream-proposals.md` | S per filing, M to follow | Maintainers of luma.gl may decline; pins drift | Four filings opened; two answered; local copies deleted | Local workarounds |
| DIR-107 | New operator families: at most three, chosen by a named consumer | Whoever names the consumer | `GPUForestInference`, `fitting-kernels`, `GPUGaussianRandomField` | M each | Adds surface that the alpha has not absorbed | Each has an external oracle, a journey and a consumer | The other 40 LRN/EMB/SIM/CART/ENC rows |
| DIR-108 | Sustainability: tiers of gates, ownership, reference-test discipline | The one maintainer | `scripts/verify.mjs`, `check-contributor-contract`, oracle manifest, kernel-audit cases | M | Process overhead; false confidence from agent-generated tests | A cheap pre-merge gate under 15 minutes; a weekly full gate; no release without the nightly record | Per-pass 60-stream fan-outs |
| DIR-109 | Agent-facing contract: catalog as the machine-readable interface | Kepler assistant, other agent hosts | `ontology/`, `analysis-experience`, `kepler/descriptors`, `workflow-assistant.ts` | M to L | Descriptors drift from code; an unfinished contract is worse than none | An agent run proposes a valid parameterised call from the catalog alone on a benchmark of 30 questions; failures are typed | Hand-written descriptors |

### DIR-101 Flagship journeys

What. Choose five journeys and finish each completely, across the layers: installable, a first result in at most 15 lines, data in from a file or Arrow table, results out as Arrow and as a layer input, a CPU reference or twin the user can run, a statement of limits in device terms, a benchmark and an independent oracle. The five, with the reason each qualifies:

1. Hot spots from points to polygons with false-discovery control (`GPUHotSpotAnalysis`, `hot-spot-analysis-recipe.ts`; curated `hot-spot-map`). Oracle: esda 2.10.0 and libpysal 4.15.0 are pinned in `modules/analysis/test/oracles/requirements.txt`. Most requested spatial statistic; the recipe exists.
2. Rate maps with empirical Bayes and a cluster and outlier map (`rate-cluster-map-recipe.ts`; curated `rate-cluster-map`, `excess-risk`). Oracle: esda. Public-health and journalism users; the recipe already returns status per stage, which the 11-17 partner noted as the model.
3. A surface from samples: IDW then kriging with diagnostics (`GPUKriging`, `GPUInverseDistanceWeighting`, `interpolation-diagnostics-recipe.ts`; curated `kriging-with-diagnostics`, `idw-surface`). Oracles: gstools, pykrige, scikit-gstat. Large unmet demand in browsers; the GPU case is strongest above about 1e5 sites.
4. DEM to hydrology: fill, flow direction, accumulation, height above drainage, stream network (`gpu-terrain/hydrology`, `drainage-network-recipe.ts`; 57 terrain classes, 46 in lens scenarios, 0 Kepler bindings). Oracles: pysheds 0.5 and whitebox. A genuine browser niche (WhiteboxTools and GRASS are not browser tools), and tile-parallel by nature.
5. Catchments on a road network: isochrones and drive-time bands (`GPUNetworkIsochrones`, curated `drive-time-catchments`, `straight-line-catchments-recipe.ts`). Oracle: networkx 3.7. Needs a network ingest path, so it exercises DIR-102.

Why these and not others: each has a recipe or curated analysis, an external library already pinned as oracle, a data size where a GPU is plausible, a comparison a user will make unprompted, and a place in the Kepler or website surface. What is not on the list: regression (36 classes; high correctness risk and the fixtures are internal), trajectories (large but no external oracle for most), raster time series (21 classes, special data), embeddings.

Cost L in total; each journey is M, so the first two together are one quarter's work. Risk: finishing a journey will surface defects in lower layers; that is intended. Evidence it worked: a user who has never seen the repository follows the journey page on Chrome and Safari; the journey has a pinned expected output; the benchmark page names hardware and compares with the oracle library on the same data. Displaces: the new-family backlog, operation-form generation, further lens recapture.

### DIR-102 I/O and interop

What. (1) A small `fromGeoArrow` reader for point, line and polygon columns that yields the packed positions plus offsets views the contributors take (the layout is already the contract in `operators-joins.md:27`), with ring-closure and winding stated. (2) Result tables as Arrow: `toArrowTable(readback)` for the common outputs. (3) A documented hand-off from a caller-owned luma buffer to a deck.gl 10 layer with no readback; the Kepler example keeps resident GPU data in layers (`examples/deck/kepler-compute/src/analysis-catalog-resident-input.ts`), but I did not verify that any path binds an analysis output buffer to a layer attribute without readback, so the first step is a spike that proves or refutes it. (4) A loaders.gl example for GeoParquet or GeoArrow, COG and PMTiles; the library should not depend on loaders.gl, an example should.

What it builds on: `analysis-core/graph-support/arrow-analytics-normalization.ts`, `docs/api/gpu-spatial-analysis/operators-geometry.md` layouts, `examples/deck/kepler-compute`.

Cost M. Risk: GeoArrow, loaders.gl (root pin 5.0.0-beta.8) and deck.gl (10.0.0-alpha.1) are all prereleases; a reader tied to one alpha breaks as the others move. Keep the reader to the Arrow data model (public `apache-arrow` types) and test it with generated tables. Evidence it worked: the journeys in DIR-101 use the reader with no per-journey packer. Displaces: the per-example packers.

Upstream-first check: a GeoArrow-to-offset packer is geospatial, named for the domain and does not duplicate luma.gl, so it belongs here. A generic Arrow-to-GPU upload belongs upstream (`dev-docs/upstream-drafts/04-typed-buffer-and-cpu-vector.md`).

### DIR-103 Convenience layer or caller-owned graphs

The 2026-11-17 argument (`thinking-partner.md`, H1 and H4): a generic `input/output/run` session would saddle the project with a second graph vocabulary; `importGraphBuffer` is two public calls; contributors, operation forms and recipes each serve a different job; start with output layouts (`getKShortestPathsOutputLayout`) and facades only where the domain supplies behaviour (continuation, completeness).

Evidence today for a convenience layer:

- The count of hand-written graph set-ups has grown, not shrunk: `new GPUCommandGraph(` is 3,594 in 1,421 test files (11-17: 3,203 in 1,303), 130 in examples, 154 in website files, 13 in Kepler source.
- Every consumer has built its own runner: Kepler's workflow runner and catalog bindings (`modules/kepler/src/workflows/workflow-runner.ts`, `modules/kepler/src/analysis/catalog-*.ts`), the lens scenario runner (`modules/lenses/src/computation-lenses/run-lens-scenario.ts`), the website's `defineOperationRun` (667 run files), and ten test `Rig`/`Harness` classes. That is at least four independent implementations of allocate, import, compile, submit, read, destroy.
- The documented GPU hello world is 55 lines.

Evidence against:

- The runners differ for good reasons. Kepler's keeps resident datasets and cancels; the website's draws; lens recorders capture; the test rigs poison outputs. A shared runner would have to be a library of hooks.
- Completeness is method-specific (Moran's neighbour overflow, cost-distance convergence, Yen's certified prefix). A generic runner cannot decide what "usable" means; a per-method facade can.
- luma.gl owns `GPUGraph` and the operation bridge. A runner here would be the re-implementation AGENTS.md forbids unless it is named for the analysis domain and small.

Position: re-argue to a middle: build no generic runner; build one `prepare<Journey>(device, props)` facade for each of the five flagship journeys, each returning an object with `run(inputs)`, typed results with completeness status, `destroy()`. Extract the allocation arithmetic as layout functions (the partner's d) so that the facade, the operation form and the docs use one definition. Revisit a shared helper only if the second journey needs code that the first already has. Cost M; evidence it worked is that the website, lens and Kepler copies for those journeys shrink. If a generic helper is ever built, it belongs in `analysis-core` and should be proposed to luma.gl first (`UPS-CPU-4`).

### DIR-104 Scale and benchmarks

What. (a) Make documented limits state a device quantity where the limit is a binding size, not a constant (85 binding-size entries and 70 device helpers exist; 169 implementation-choice constants and 38 reference-algorithm limits remain; the audit's fifteen "no size-extending route" rows are the starting list). (b) Host paging and tiled routes for the flagship journeys only: terrain has a tiled D8 route with bounded scratch (`GPUTerrainTiledFlow`) but no host paging (`TE06`); chunked contributors keep row-proportional scratch in one binding. (c) A published benchmark suite on public data (Overture or OSM extracts for points and network, a public DEM, a public polygon set), run on at least two vendors, with the oracle library on the same machine. `scripts/test-benchmarks.mjs` already selects the 45 benchmark specs; the missing part is the public data, the CPU baseline and the second machine.

Cost L. Risk: a vendor run will find GPU-only defects (the repository's own experience on Metal); that is cost and benefit. The roadmap's `FP64-05` is the same step. Evidence it worked: a table in the README with hardware, data size, GPU time, esda or GeoPandas time and result agreement; limits pages in bytes. Displaces: raising a single limit for its own sake.

Honest expectation: for several operators the GPU will lose on small inputs (`IN01` records CPU 1.3 and 3.2 ms against GPU 4.3 and 51 ms for variogram fitting); the benchmark should publish those too and the facade should choose the CPU twin below a crossover.

### DIR-105 Reduce the surface

What. (1) Mark each operator `stable` or `experimental` in the catalog and split exports (`@luma-contributors/analysis/experimental` or an `experimental` flag in the catalog plus a naming rule) so that the deprecation policy covers only stable. Admission rule: independent oracle (a fixture produced by a library other than this one, or a closed form), a docs page, a catalog record with `contract` audited, a CPU twin or a stated reason, a named consumer. On the census above, an applied rule would put roughly 130 to 200 classes in stable and the rest in experimental; the exact number is the output of the work, not an input. (2) Merge or demote overlaps: the 129 operation forms are wrappers over contributors; they are exports, not additional methods, and could be generated or moved behind a single entry. (3) Delete what has no consumer and no oracle: the seven classes listed in the census have no named consumer outside tests; most are shared fitting helpers (`GPUAdamStep`, `GPUFittingGradientSums`, `GPUFittingMinibatchOrder`, `GPUFittingCoordinateEncoder`) that exist for unbuilt LRN-03 and ENC-01; freeze them rather than delete if LRN-03 stays, delete if it is dropped.

Evidence for the approach: scikit-learn's `sklearn.experimental` opt-in import with "no deprecation cycle" and Julia's `Base.Experimental` (sources in the practice list under DIR-108). Cost M. Risk: argument over where the line falls; avoid by publishing the rule and letting the catalog fields decide. Evidence it worked: the catalog has no operator with `tier: unknown`; the README states the stable count. Displaces: "633 operators" as the headline.

### DIR-106 Upstream to luma.gl

Nothing has been filed (`upstream-proposals.md`: "no issue and no pull request"). Four drafts exist: packed CPU/GPU transfer, histogram edges and domains, caller CSR with masks, typed raw-buffer import. The 2026-11-18 audit found that grid WGSL was adopted from alpha.16 and several other requests are "not provided". The rule in AGENTS.md ("luma.gl improvements go upstream as ordinary pull requests") has not been exercised once in the history of this repository.

Recommendation: file the four drafts now (S each; the maintainer of luma.gl is the one who knows whether to accept), with the `UPS-CPU-1` benchmark first because every mixed CPU and GPU path pays about 11 ns per byte (`dev-docs/research/cpu-suitability-2026-11-11`). Then propose the generic items the roadmap already names: graph algorithms on caller CSR (`UPS-01`, `UPS-02`), image morphology and phase correlation and exact quantiles, a randomized SVD (`SIM-04`). Do not file more than the maintainer will track; luma.gl alphas arrive every one to three days (alpha.7 to alpha.17 from 2026-11-01 to 2026-11-18), so a filed issue that waits for two weeks is stale against three bumps. Cost S to file, M to follow. Evidence it worked: two responses, and the local copy deleted in the next bump. Displaces: local workarounds and the register `upstream-workarounds.md`.

### DIR-107 New operator families

The roadmap's "New contributors: unexecuted options" lists forty-odd unbuilt rows. Several already shipped (`GPUForestInference` with readers for scikit-learn, XGBoost and LightGBM models; the shared fitting kernels; `GPUGaussianRandomField`). Verdicts per row are in "Roadmap rows". Three that earn a place, each tied to a journey and an oracle:

1. `LRN-06` `GPUPermutationImportance` (S): works with any contributor that has a predict kernel, so it attaches to forest inference and to regression; oracle scikit-learn `permutation_importance`; it also gives agents an explanation primitive.
2. `EMB-03` and `EMB-04` embedding trajectory and anomaly (S each): small, built on the existing eleven `RasterEmbedding*` classes and `GPURasterEmbeddingCovariance`; the oracle is NumPy by definition; they serve the AlphaEarth-style change workflow without new model code.
3. `SIM-03` `GPUSensitivityIndices` (M): Sobol and Morris for `GPUWeightedOverlay` and any parameterised contributor; oracle SALib; the host can use it to show which parameter matters, and it is a sound complement to the agent contract.

Everything that needs weights or new layer types (`ENC-01` to `ENC-03`, `LRN-07`, `LRN-08`), anything whose reference code is AGPL (`CART-01`), and anything with no consumer should leave the roadmap. Cost M in total. Risk: each addition costs the same obligations as any operator (catalog, docs, kernel audit, lens). Evidence it worked: each is used by a journey page. Displaces: the rest of the family list.

### DIR-108 Sustainability

Facts: source grew from about 317,000 to 565,000 lines in about three weeks (brief); the gate is a two-hour sequential run on one laptop (`AGENTS.md`, `yarn verify`); the CI workflow exists but its execution is unverified (`INT-08`); 66 implementation streams were followed by four review rounds that found 147, 102, 52 and 1 further defects (brief); roadmap rows lag the code (11-17: 12 of 15 sampled rows stale; this review found GAP-01 and GAP-02 largely stale, below). One maintainer owns all of it.

Practices from comparable libraries, with sources (accessed 2026-11-19) and the mapping to one maintainer:

| Practice | Source | Mapping |
| --- | --- | --- |
| Opt-in experimental import and "no deprecation cycle" | scikit-learn `IterativeImputer` docs, https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html; Julia `Base.Experimental`, https://docs.julialang.org/en/v1.9/base/base/ | DIR-105 tiers |
| Two-release deprecation window for public API, private API unprotected | scikit-learn 1.5 release notes, https://scikit-learn.org/1.5/whats_new/v0.22.html | A script that deletes deprecated code at the stated version |
| One shared check suite every estimator must pass | scikit-learn `check_estimator`, https://scikit-learn.org/1.5/developers/develop.html | The contributor-contract checker and kernel-audit cases play this role; make them a gate for stable |
| Reference values from an independent higher-precision tool with the generating command in the test | scipy.stats tests (mpmath, R); Boost.Math `test_data.qbk`, https://sources.debian.org/src/boost1.90/1.90.0-5/libs/math/doc/internals/test_data.qbk | The oracle generators and `GENERATORS.md` exist; require them for stable |
| Per-function backend capability table; skips declared | SciPy array API, https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html | Catalog fields `tier`, `readback`, `limits` already hold this; fill them |
| Public implementation status matrix and gold files | Apache Arrow integration docs, https://arrow.apache.org/docs/format/Integration.html | `coverage-matrix.md` (status date 2026-11-07, 144 rows) and the fixtures |
| Calibration tests for randomized methods | Stan simulation-based calibration, https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html | Permutation, sampling and kriging simulation operators |
| Numbered decision records instead of re-arguing | GDAL RFC 72, https://gdal.org/development/rfc/rfc72_pytest.html | `dev-docs/` decision rows |
| On-demand GPU CI with a spend cap | scikit-learn GPU CI, https://blog.scientific-python.org/scikit-learn/gpu-ci/ (label-triggered, about $50 a month in 2024); RAPIDS self-hosted GPU runners, https://docs.rapids.ai/resources/github-actions/ | A label-triggered or nightly GPU job on hosted runners or a second machine; GitHub hosted GPU price page: https://docs.github.com/en/billing/reference/actions-runner-pricing (figures differ across pages; check before budgeting) |
| Software GPU only for API conformance | Dawn WebGPU CTS README, https://dawn.googlesource.com/dawn/+show/main/webgpu-cts/README.md | SwiftShader or lavapipe in CI catches validation errors; numbers still need real hardware |

The sources above were reported by a research subagent; I did not re-open each page. Items the subagent could not source (a PySAL tier system, GeoPandas and Shapely practice, Eigen test tiers, Kokkos CI) are not used.

Recommended design for this repository:

1. Three gate levels. (a) Pre-merge, under 15 minutes: lint, the two checkers, the Node specs for the touched family, source typecheck for the touched family (the runner already groups by family), kernel-audit cases for touched contributors. (b) Nightly or on label: full Node suite, headless GPU suite for stable classes only, oracle fixture verification. (c) Release: the existing full `yarn verify`, plus a recorded run on the second GPU (DIR-104). The roadmap's `LR-19` already measures the typecheck memory question.
2. Ownership by family, written down. One maintainer cannot review 565,000 lines; the assignable unit is the stability tier. For stable operators the maintainer reviews every change; for experimental ones an agent pass with an independent reviewer is acceptable, and the catalog says so.
3. Admission and reference-test discipline: a "discriminating test" rule exists for new result contracts (roadmap LR section). Extend it: a stable operator has an oracle produced outside this repository, and a test that fails if the oracle is replaced by the CPU twin's output.
4. A cap on fan-out. The record shows that 66 parallel streams produced 300 further findings across four rounds; a pass of that size should be followed by a settling period before the next feature work.

Cost M. Risk: process that nobody runs; keep the gates executable by one command. Evidence it worked: pre-merge under 15 minutes; a nightly record in the repository; no release with a stale roadmap row. Displaces: wide agent passes without a settling period.

### DIR-109 The agent-facing side

Is "the spatial analysis toolbox for agents in the browser" a stronger position than "a GPU GIS library"?

For: the existing assets fit. The catalog (548 operators with inputs, outputs, coordinate models, limits), twelve questions, a requirement evaluator that returns passed, failed or unknown with evidence, hand-written descriptors for 217 bindings with resolved defaults and applicability, a dataset profile, an assistant that proposes a draft workflow and executes nothing until approved (`modules/kepler/src/workflows/workflow-assistant.ts`, `WorkflowError` typed errors). Competing agent toolboxes expose bare `inputSchema` and free-text errors: the MCP tool definition has no cost, unit, CRS or typed-error field (https://modelcontextprotocol.io/specification/2025-06-18/server/tools, via subagent). Published agent benchmarks report low strict-scoring success (GISAgentBench best agent 32.7% under tolerance-aware scoring, https://arxiv.org/abs/2608.01645; GeoBenchX and GeoAnalystBench cited by the subagent; I did not re-open them). The only browser agent toolbox found that runs spatial statistics (kepler.gl's AI assistant with LISA, regression, buffer) runs on WebAssembly or the CPU. WebMCP (page-exposed tools; Draft CG Report; Chrome 146 Canary behind a flag per the subagent's source, https://w3c-cg.github.io/aikr/webMCP/webmcp-technical-notes.html) is the plausible transport and is pre-standard.

Against: the catalog today is not a contract. Of 548 operators, 433 have `semantics: unspecified`, 225 have unknown readback, 201 no tier, 370 an unaudited contract, 288 no scale evidence; the descriptors live in the Kepler module, not in the analysis package; an agent that trusts a bare `unknown` will be wrong without a message. An agent product also depends on the host (Kepler's assistant is not installed in the host per the AX study) and on a transport that is pre-standard.

Position: the stronger claim is "an auditable operator library whose contracts a machine can read", of which the agent use is the second consumer after the Kepler workbench. Do not rebrand; fill the contract first. Needs, from the subagent's survey mapped to repository fields:

| Need | Where it lives today | Gap |
| --- | --- | --- |
| Units and coordinate model per input | catalog `coordinateModels` (148 of 548 `unspecified`) | Fill; add distance and area unit; reject mismatched CRS class at the host |
| Checkable preconditions | `analysis-requirements.ts` (requirement evaluator), construction-time validators with ID-prefixed messages | Expose as data in the analysis package, not only Kepler |
| Cost and size estimate | catalog `limits`, `representativeSourceRows`, `transientMemory`, `dispatch`; `getAnalysisMaximumRowCount` | Make a `estimateCost(props, deviceLimits)` function per stable operator returning bytes, dispatches and a device-limit verdict |
| Typed failure reasons | status words (`overflow`, `invalidCount`), construction messages | A small closed set: precondition, device-limit, precision, non-convergence, incomplete; return it from the facade (DIR-103) |
| Determinism and provenance | catalog `deterministic` (548 true), `semantics`, computation records | Add seed, precision path, adapter, version and tier to the result record |

Cost M to L; the first three rows are S each per stable operator and can be done journey by journey. Evidence it worked: a fixed set of 30 natural-language questions on three datasets; an agent (Kepler's assistant or a harness) picks a catalog operator, fills parameters from descriptors, receives a typed error when it should, and the result matches the oracle. Displaces: hand-written per-binding descriptors in Kepler (derive them from the catalog once the fields are complete).

## Risks

| ID | Risk | Evidence | Mitigation |
| --- | --- | --- | --- |
| DIR-110 | Dependency churn: luma.gl alphas appear every one to three days (alpha.16 on 2026-11-17, alpha.17 on 2026-11-18) and the library's peer range `~10.0.0-alpha.16` accepts all later prereleases without test | `npm view @luma.gl/graph time` and `dist-tags` (`latest` alpha.5, `beta` alpha.17) | Pin the tested version in the README; bump on a fixed cadence (for example once per week or at each release candidate), not per alpha; add a nightly check that runs the gate against `beta` and records the result |
| DIR-111 | One-device evidence: all timing and all GPU suites ran on Apple Metal; Metal-only defects were found but other compilers are untested; Firefox and Safari 26 are claimed only by absence of feature use | Roadmap `FP64-05`, `upstream-proposals.md` | DIR-104 second vendor; one Safari and one Firefox run of the headless suite |
| DIR-112 | Packaging identity: the package, README and website name `visgl/luma-contributors` while the remote is the private `BertCh` repository; licence `MIT AND Apache-2.0 AND BSD-3-Clause` | `LR-04`, `UF-01`, `INT-09` | Decide the owner and name before publishing; a rename after publication costs every consumer |
| DIR-113 | The catalog's self-description is too thin to carry an agent or a stable tier (see DIR-109) | `cat2.ts` and `cat4.ts` counts | Fill fields for the stable set first |
| DIR-114 | The roadmap row set is stale against the code and a new reader cannot trust it | GAP-01, GAP-02 closed in part (below); 11-17 found 12 of 15 sampled rows stale | Delete closed rows in the same change that closes them; the review's roadmap update covers this |
| DIR-115 | Unreviewed fix rounds from 2026-11-18 (the brief lists five unread rounds and the alpha.16 bump) sit under any release | Brief | Other area reports; do not announce until they are settled |
| DIR-116 | Browser memory ceilings decide who can run a documented size: Apple at least 768 MiB for 91%, Safari about 429 MB, iOS about 307 MiB, Android 256 MiB for 75% | https://web3dsurvey.com/webgpu/limits/maxStorageBufferBindingSize (undated) | State sizes per device class in the benchmark; make facades report a device-limit verdict before allocating |

## Recommended sequence

Assumptions: the correctness and release work of the other area reports completes first (weeks 0 to 3); then two quarters, called Q1 and Q2.

Decisions the maintainer must take first (before week 4):

1. Package identity: owner, repository URL, package name, and whether the first public install is `0.1.0-alpha.0` or later (`LR-04`). Without it nothing else can be shared.
2. Stability tiers: adopt the stable and experimental split and the admission rule (DIR-105); decide that the headline is a verified core.
3. The five journeys (DIR-101) or replacements; each needs a named person who will use it.
4. luma.gl cadence: pin to a tested version and bump on a schedule, or follow `beta` nightly (DIR-110).
5. Whether to buy or borrow a second GPU machine and a Safari and Firefox run (DIR-104, DIR-111).
6. The `LRN`/`EMB`/`SIM`/`CART`/`ENC` decision in the roadmap's "Decisions for the maintainer": confirm three (DIR-107) and drop the rest.

Three bets:

1. Depth over breadth on five journeys, with the facade, GeoArrow in and out, and a published two-vendor benchmark (DIR-101, 102, 103, 104). Q1: journeys 1 and 2 and the GeoArrow reader; Q2: journeys 3, 4 and 5 and the second-vendor benchmark.
2. A stable core with a tier, a reference-test admission rule and three gate levels (DIR-105, 108). Q1: catalog fields filled for the stable set, tier split, pre-merge gate; Q2: nightly GPU record and deprecation policy exercised once.
3. A machine-readable contract on that core, used by Kepler's assistant and a benchmark of 30 questions (DIR-109). Q2, after the tier split, because the contract is only worth publishing for operators that are stable.

Three stops:

1. New operator families before the alpha has external users, except the three in DIR-107 attached to journeys. The roadmap's own LR section already states the rule; the "New contributors" section contradicts it by carrying a sequence.
2. A generic `input/output/run` runner and blanket operation-form generation (DIR-103). Facades per journey only.
3. Retrospective lens and website coverage for operators outside the journeys (385 of 633 classes have lens scenarios; recapturing and reviewing them each time the substrate moves costs more than any user receives). Keep the 28 curated analyses and the journeys current; freeze the rest.

Uncertainty:

- I did not run any GPU code; every statement about browsers other than Chrome on Metal comes from sources, not from execution.
- The census measures mention, not exercise; the true fixture and consumer coverage is probably lower than the numbers.
- Whether a deck.gl layer can bind a caller-owned luma buffer without readback is UNVERIFIED in this review.
- The effort figures are my estimates in the roadmap's own S, M, L scale; the roadmap's own effort figures for LRN rows were measured only on one machine.
- The selection of five journeys rests on oracle availability and recipe existence, not on demand data; nobody outside the repository has been asked.

## Roadmap rows

Verdicts for the rows named in the task and the neighbours that change with them. Evidence is by source search (class existence) unless stated.

| Row ID | Verdict | Evidence | Replacement text if NARROW |
| --- | --- | --- | --- |
| LRN-01 (forest inference) | CLOSED (already removed from the table) | `GPUForestInference` in `dataframe-analysis`; `docs/api/forest-inference.md` documents readers for scikit-learn, XGBoost and LightGBM | none; update the "Proposed sequence" paragraph, which still lists LRN-01, LRN-02, SIM-01 and CART-03 as pending |
| LRN-02 | CLOSED | `docs/api/analysis-core/fitting-kernels.md`; classes `GPUAdamStep`, `GPUFittingGradientSums`, `GPUFittingMinibatchOrder`, `GPUFittingCoordinateEncoder` exist with no consumer | remove from the sequence |
| LRN-03 `GPUNeuralField` | DROP | No class (`rg "class GPUNeuralField"` returns nothing); the research says it is not more accurate than kriging and has no uncertainty; effort L; fixtures need a PyTorch oracle that is not pinned | none |
| LRN-04 `GPUGaussianMixture` | DROP | Effort M; scikit-learn oracle exists but no named consumer; clustering already has 5 contributors | none |
| LRN-05 `GPUSelfOrganizingMap` | DROP | No consumer; overlaps k-means and regionalization | none |
| LRN-06 `GPUPermutationImportance` | KEEP, REPRIORITISE to first | S effort; works with the forest and regression operators; oracle scikit-learn; supports agent explanations | |
| LRN-07 `GPUGradientBoostedTrees` | DROP | Effort L; training in a browser has no named consumer; inference already exists | none |
| LRN-08 `GPUTreeShap` | DROP | Effort M; only meaningful after LRN-07 | none |
| LRN-09 ensemble option for conformal | DROP | S effort but depends on fitted networks that are not built | none |
| EMB-01 Presto decoder | KEEP, REPRIORITISE after verification | Blocked on a verified scaling formula (the row says the published documentation states storage only); do not build until the formula is verified | Not implemented; needs a verified uint16 scaling formula and an independent oracle. |
| EMB-03, EMB-04 | KEEP | S each; classes absent; built on existing embedding classes and `GPURasterEmbeddingCovariance` | |
| EMB-05, EMB-06 | DROP | Row says check overlap first; no consumer; low priority | none |
| SIM-01 `GPUGaussianRandomField` | CLOSED | `GPUGaussianRandomField` exists | remove from the sequence |
| SIM-02 terrain error ensemble | KEEP as part of the DEM journey | Needs DEM error realizations; build only with DIR-101 journey 4 | |
| SIM-03 `GPUSensitivityIndices` | KEEP | M; oracle SALib; attaches to any parameterised contributor | |
| SIM-04 `GPURandomizedSVD` | DROP here, file upstream first | Row asks whether it belongs in luma.gl; a generic linear algebra primitive is upstream's business | none |
| SIM-05 geographically weighted PCA | DROP | No consumer; GWR family already large | none |
| SIM-06 NMF | DROP | Unblocks RS10 only; RS10 is itself L-priority | none |
| SIM-07, SIM-08, SIM-09 | DROP | Low priority rows; no consumers | none |
| SIM-10 structured kernel interpolation | REPRIORITISE after the kriging journey | M; extends kriging beyond 1e5 points; GPyTorch oracle; worth building only if journey 3 shows the limit binds | |
| CART-01 `GPUFlowCartogram` | DROP | Reference programs are AGPL or modified MIT (fixtures only); effort M to L | none |
| CART-02, CART-04, CART-05, CART-08, CART-09 | DROP | No public reference, or designs "from the research, not a published one", or L effort; no consumer | none |
| CART-06 `GPUTravelTimeMap` | KEEP, tie to journey 5 | S; design from the research, so needs its own review | |
| CART-07 small presentation set | NARROW | Pick the items a journey needs (uncertainty classes, exceedance probability); the rest remain optional | Not implemented; build only the items a flagship journey uses, effort S each. |
| ENC-01 `GPULocationEncoder` | DROP | Needs weights extraction and a model-card decision; no consumer; weight licence | none |
| ENC-02, ENC-03 | DROP | L effort, weight licence unstated | none |
| ENC-04 ONNX Runtime Web spike | KEEP as an example only | S; ONNX Runtime Web 1.30.0 README still labels the WebGPU backend experimental (https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/README.md); WebNN origin trial disabled per the subagent's sources | |
| `LRN / EMB / SIM / CART / ENC` decision | RESOLVE | Replace with the three DIR-107 rows | Confirm: LRN-06, EMB-03/04, SIM-03; others removed. |
| GAP-01 | NARROW | Voronoi (`GPUVoronoiDiagram`), splitting (`GPULineSplit`), simplification (`GPULineSimplification`, `GPUCoverageSimplification`), dissolve (`GPUCoverageDissolve`) exist. Not found by class search: antimeridian cutting as a contributor, points along lines (`GPULineLocate` and `GPULinearReferencing` exist), per-feature buffer (`GPUBufferSelection`, `GPUOffsetCurve` exist) | Attribute-preserving overlay, overlapping dissolve and antimeridian cutting; on consumer demand. |
| GAP-02 | NARROW | `GPUPolygonize`, `GPUPolygonRasterization`, `GPURasterSieve`, `GPUTerrainStreamOrder`, `GPUTerrainBreach` exist | QA masks, line and point rasterization, zonal histogram, median and majority, pour-point snapping; on consumer demand. |
| GAP-03 | KEEP | No class named for nearest POIs, street statistics or angular measures was found (`GPUNetworkIsochrones`, `GPUNetworkCentrality` exist) | |
| GAP-04 | KEEP, demand only | No class for Durbin, SLX, SDEM, SARAR, as-of join or window functions was found in the census list | |
| GAP-05 | KEEP, demand only | No voxel-centroid producer, cloth filter, Circuitscape-style class found | |
| PLAT-07 SpatialBench extensions | REPRIORITISE | Fold into DIR-104's public benchmark; its SF5 result (216.3 ms) is one machine | Becomes part of the published benchmark suite. |
| UF-03 re-measure on published primitives | REPRIORITISE | Same as PLAT-07; the repository has no CPU-library comparison | Becomes the benchmark in DIR-104. |
| FP64-05 non-Apple GPU | REPRIORITISE to H | It is the same work as the second-vendor run and gates every performance claim | |
| UPS-CPU-1 | KEEP, file first | Nothing filed; four drafts exist | |

## Limits of this report

- No GPU, browser, build or test was run. All runtime and comparable facts come from web sources read by subagents; the lead re-checked only the luma.gl dates.
- The comparables table is as complete as the subagent search; a search that returned nothing is not proof that no such library exists.
- Counts come from identifier-token scripts; a class can be named in a file without being exercised. The fixture column may count a class whose test does not compare against the fixture.
- The GAP verdicts rest on class names; a capability can exist under a name my patterns did not match, and a class can exist with narrower behaviour than the row.
- Roadmap rows for the new contributors were not checked against the five research files beyond their summaries.
- The cost scale is S days, M one to two weeks, L more, as defined by the roadmap; the numbers for a one-maintainer team with agents are guesses.
