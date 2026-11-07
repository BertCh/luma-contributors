# Terrain boundaries implementation ledger

This ledger defines the small deterministic inputs and readouts for the `terrain-boundaries` lab. GPU results should be compared with the contributor contracts and this input definition; no numeric output is claimed as independently validated here.

## Source and grid

- Dataset: `grand-canyon-dem`, shipped Terrarium elevation raster attributed to AWS Terrain Tiles (Mapzen/Tilezen), built from USGS 3DEP/NED. Elevation quantization is 0.5 m.
- The source image is 2048 × 2048 at 19.109257 m EPSG:3857 pixel spacing. `createCanyonGrid(dataset, 32)` averages each 32 × 32 source block into a 64 × 64 north-first raster. Missing samples are not present in this source.
- The contributor grid remains EPSG:3857 with `cellSizeMode: 'web-mercator'`. Display coordinates and markers are converted into local ground meters around the grid's center. Nominal ground spacing at the center is 494.083 m per cell; the contributor evaluates row-dependent ground distances from Web Mercator. The rendered bounding rectangle is a local projected approximation.
- Input elevations use the block means, not invented or interpolated heights. The measured full-source manifest range is 688–2626 m; the block-mean range may be narrower.
- Cell numbering is zero-based row-major, with row 0 at the north edge. Selection defaults to cell 2080 (row 32, column 32); clicking a finite positive prominence peak selects that cell for the peak chart.

## Peak prominence

- `GPUTerrainPeakProminence` receives the actual 64 × 64 block-mean elevations and writes per-cell prominence, key saddle, and elder-parent cell. The finite peak summary scans these GPU readbacks: finite peak count is the number of finite prominence values greater than zero; component elder count is the number of `+Infinity` values; the selected finite peak is the first row-major cell achieving the largest positive finite value.
- The peak chart reads the selected GPU output and measured elevation and displays peak elevation, key saddle, and `key saddle + prominence`. The elder link is the GPU parent index or `none` for `0xffffffff`.
- Infinite elders are drawn separately as diamond markers; they are finite-window components that never merge with a higher elder within this closed raster. They are not claimed as globally prominent peaks or referenced to sea level.

## Drainage recipe

- One D8 `addDrainageNetworkRecipe` uses the same 64 × 64 raster grid and elevation, `fillDepressions: true`, `resolveFlats: true`, 256 fill iterations, 256 accumulation iterations, cell-count runoff units, and stream threshold 12. Flow parameter values use `fillEpsilon: 0.01`.
- The map displays recipe stream order and one generated segment from each resolved stream cell to its D8 receiver. Segments are emitted from the actual recipe direction/stream views; the extracted `GPUTerrainStreamNetwork` reach/vertex outputs are additionally read for reach/vertex count, overflow, and unresolved-walk status.
- The drainage status text reports depression fill, flat resolution, accumulation, stream-order and catchment convergence; downstream-distance uncertainty count; catchment-summary overflow; network reach/vertex count, network overflow, and unresolved walks. It does not claim complete results when one of these statuses reports a limit or overflow.
- The selected-cell readout and tooltip use actual recipe summary arrays: catchment ID, cell count and group mean downstream distance. The summary uses the recipe's distance-validity mask; an unresolved or non-summarized group remains explicitly unavailable.

## Four owned tiles

- The 64 × 64 raster is split into four 32 × 32 non-overlapping owned rectangles. Each tile input includes the one-cell neighboring halo where available. Halo cells provide precomputed global D8 directions and stream flags for routing across seams; only cells inside each tile's owned rectangle receive runoff 1, while halo runoff is 0.
- `GPUTerrainTiledFlow` runs with the selected budget of 4, 32, or 128 DAG sweeps. In D8 mode, tile inputs use D8 directions from the full-domain recipe. In branching mode, `GPUTerrainFlowFractions` recomputes D-infinity or Freeman-MFD slopes from the original block-mean elevation and uses the filled/resolved D8 direction raster for fallback directions; the tile graph and whole-window `GPUTerrainBranchingFlow` reference consume those identical fractions and one unit of runoff per cell. Branching mode is an alternative routing comparison, not the depression-filled D8 recipe. This surface/fallback difference is explicit and should not be read as a branching version of the exact D8 fill surface.
- The GPU readback summary counts owned cells with status 1 (resolved), 2 (unresolved), or other (invalid), and counts D8 edges or positive branching links crossing either ownership seam. Separate per-tile assembly passes keep each shader within the common storage-buffer binding limit.
- The GPU reduction compares finite resolved tiled accumulation with the matching full-window reference and reports maximum absolute difference in cell units. D8 comparison is gated by the recipe's accumulation-converged scalar; branching comparisons include only cells whose whole-window branching status is resolved. If there are no jointly resolved reference cells, the readout says the difference is unavailable instead of reporting zero. The difference is a comparison between outputs for the same grid and unit runoff per cell, not a physical discharge error.
- The four-cell-quadrant ownership view is a resident global routing-DAG demonstration. It is not out-of-core processing or distributed hydrology. An unresolved tiled status invalidates that cell's accumulation claim.

## Verification notes

- The independent CPU oracle in `earth-terrain-boundaries-oracle.json` reconstructs the exact block-mean float32 grid and computes finite-window peak prominence using a stable descending-saddle 8-neighbor union-find. `scripts/check-terrain-boundaries-browser.mjs` is the browser-side verifier for the GPU peak selection and the three tiled routing/reference pairings. Neither check asserts that the simulated tile results represent measured flow or a regional drainage system.
- Biome and showcase typecheck were run; this implementation emitted no TypeScript diagnostics. The package-wide typecheck currently reports errors in the concurrently modified raster `habitat-connectivity` scene.
