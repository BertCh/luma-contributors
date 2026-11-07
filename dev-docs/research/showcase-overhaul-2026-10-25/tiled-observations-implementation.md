# Tiled observations implementation ledger

## Experiment scope

- Source: bundled Dixie Sentinel-2 calibrated NIR reflectance on 2021-07-13 (ignition day, not a
  pre-fire baseline) and 2021-09-21, with the corresponding strict native-pixel QA masks. The full
  250 × 250 arrays are decoded once per date. Missing source cells stay missing.
- Crop: source rows `[50, 178)` and columns `[60, 188)`, 128 × 128 cells at 60 m. Projected UTM
  bounds are `[666400, 4445360, 674080, 4453040]` in EPSG:32610. The opening camera is centered at
  `frame.cellToLngLat(64, 64)`.
- Transport: a decoded, in-memory reader backed by bundled arrays. This does not implement or
  benchmark remote fetching, HTTP range requests, worker decoding, or Cloud Optimized GeoTIFF I/O.
- Ownership: four row-major 64 × 64 cores, NW/NE/SW/SE. Halo cells are borrowed context. Core
  extraction owns each output cell once; only source cores enter global statistics and histograms.

## Implemented compute paths

- `GPURasterTileReader`, `GPURasterTileCache`, and `GPURasterTileHaloAssembler` read the selected
  crop windows and selected `nir` band, keep real hit/miss/pin statistics, and hold leases through
  graph use. Replacement, discard, compile failure, and scene destruction release leases before
  cache resources. `GraphSwap` keeps the last committed graph until its replacement commits.
- Four halo-filled cores run `GPURasterGaussianBlur` and `GPURasterTileCoreExtract`; GPU passes
  stitch the disjoint cores. A second GPU blur runs over the whole crop using the same radius, crop
  edge, and nodata controls. The map can compare halo output to intentionally isolated-core output
  or the untiled whole-crop reference. Bounded output-difference readbacks are reduced on CPU only
  for max/RMS diagnostic readouts; analytical maps are GPU outputs.
- One `GPURasterGlobalInitialize` precedes four sequential owned-core statistics merges. Histogram
  replay follows the completed global extent pass. The percentile control rebuilds the actual GPU
  query, including validity; endpoints 0 and 1 use exact extrema, interior values are histogram-bin
  estimates. Readouts expose extent, bin width, binned population, overflow flag, and estimate
  validity. A CPU sum from the original crop is shown only as an independent reference.
- Weighted overview outputs carry sum and exact valid count. A direct factor-4 overview is compared
  on GPU with two factor-2 stages; the readout reports sum and mean deltas and exact count agreement.
  A separate categorical path maps the halo-smoothed NIR reflectance to integer bins `0: <0.1`, `1: [0.1,0.2)`,
  `2: [0.2,0.3)`, and `3: ≥0.3`, then runs selectable nearest-center or mode reduction. Mode ties
  choose the lower integer bin. These labels are NIR reflectance bins, not land cover.
- `GPURasterTileReprojection` samples four ordered owned cores on a half-cell-shifted identity-CRS
  lattice, then is compared with `GPURasterReprojection` over the same full crop. Nearest, bilinear
  strict, and bilinear renormalize modes rebuild the graph. The omit-east option removes eastern
  source tiles and masks the same area in the untiled reference; support and shared-support errors
  are read from the GPU outputs. The display frame follows the shifted destination lattice.
- Source settings, radius, edge/nodata policy, bin count, percentile, overview scale and policy,
  resampling method, and omitted-tile state participate in the graph key. Pending rebuilds clear
  old readouts and hide mismatched maps. Summary callbacks check the committed graph generation
  and option revision before updating values.
- The map has a distinct NW core outline, tile boundaries, reflectance/difference/support/category
  legends, source credits, and metric scale bar.

## Controls and defaults

- Date: July 13 or September 21, 2021; initial July.
- Map output: source, halo blur, isolated-core blur, halo-minus-isolated difference, tiled-minus-
  whole-crop GPU blur difference, weighted overview, support, NIR-bin categorical overview,
  half-cell-shift tiled resampling, or tiled-minus-untiled resampling difference. Initial halo-blur map.
- Gaussian radius: 1–6 source pixels; initial 3 (180 m). Crop-edge border: outside nodata, clamp,
  or reflect; initial outside nodata. Missing neighbors: ignore and renormalize or propagate; initial
  ignore and renormalize.
- Histogram: 32 or 128 bins; percentile 0, 0.1, 0.5, 0.9, or 1; initial 32 and 0.5.
- Weighted overview: factor 2 or 4; initial factor 2. Categorical overview: nearest-center or mode;
  initial mode.
- Tiled reprojection: nearest, bilinear strict, or bilinear renormalize; optional east-tile omission;
  initial bilinear renormalize with all four tiles.

## Verification and limits

- Showcase TypeScript check was run with a temporary `.scene.ts` copy so the `.wip` metadata and
  compute were both included. It reported no errors in either tiled-observations file, but the
  workspace command remains failing on unrelated dirty scenes including `flows/bixi-demand-model`,
  `geometry/footprint-shape-lab`, `hydrology/terrain-boundaries`, and `points/sampling-frames`.
- Scoped Biome check passed for the compute file and for the metadata through the temporary scene
  copy. `check-story-steps.mjs raster` passed. Strict raster cartography marked tiled-observations
  `ok` and failed only on the existing eight-step `spectral-texture` story.
- No GPU/browser run was authorized or performed. Graph pipeline compilation on a device, numerical
  parity, cache counters at runtime, map rendering, and all control changes remain unverified. The
  outputs are source-ready, not GPU-verified.

## References

- GDAL overview generation documents average resampling with nodata exclusion and mode selection
  for categorical data: <https://gdal.org/en/stable/programs/gdaladdo.html>.
- OGC Cloud Optimized GeoTIFF 1.0 distinguishes tiled access, range requests, and overviews:
  <https://docs.ogc.org/is/21-026/21-026.html>. This decoded in-memory experiment is not a COG
  transport benchmark.

## Planner review before wave 12

Replaced 256 row-copy kernels with four owned-core assembly kernels, made runtime NaN bit patterns valid WGSL, added explicit tiled/reference resampling support counts and mismatches, and prevented empty overview blocks from poisoning diagnostic maximums. Both nearest and mode labels identify their actual selection rule. Raster story and strict cartography checks pass after consolidating the separate spectral-texture story to six steps. Independent NumPy source reductions, histogram quantiles, separable Gaussian support and categorical overview references are in `earth-source-oracles.json`; `check-tiled-observations-browser.mjs` is prepared but has not run.
