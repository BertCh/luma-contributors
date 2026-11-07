# Tiled Sentinel topology oracle

This ledger checks the first step of `tiled-observations`, which now runs a second, bounded connected-region analysis alongside the existing tile-reader/filter/overview/resampling graphs. It is a deterministic numerical reference for CPU review and the browser verifier; it does not replace the GPU path.

## Source window and grid

- Dataset: bundled Dixie Fire Sentinel-2 L2A calibrated NIR (`nir_before.bin`, 2021-07-13; `nir_after.bin`, 2021-09-21) and strict per-native-pixel validity masks.
- Source raster: 250 × 250 at 60 m, EPSG:32610, UTM zone 10N.
- Parent tile experiment: crop source rows `[50,178)`, columns `[60,188)`, 128 × 128.
- Topology focus: crop rows `[32,96)`, columns `[32,96)`, equivalently source rows `[82,146)`, columns `[92,156)`. It is 64 × 64 and is split into four disjoint 32 × 32 ownership cores at origins `(0,0)`, `(32,0)`, `(0,32)`, `(32,32)`.
- The focus's projected top-left is UTM easting 668,320 m, northing 4,451,120 m; cells are 60 m square. The contributor metadata uses a translated local metric grid for these cells, while the visible outline is drawn from the full crop's projected frame.

## Processing and validity

For each finite, strict-QA-valid source cell, the GPU classifies NIR `< 0.2` as low and NIR `>= 0.2` as high. This threshold is an analyst-defined reflectance split, not a land-cover or burn class. Missing or non-finite cells are excluded from both masks.

Each 32 × 32 core runs local 8-connected labeling, dense remapping, and region measurements. `GPURasterCrossTileComponents` reconciles each class independently using 8-connectivity, and the caller assigns stable IDs 0–127 to low regions and 128–255 to high regions. A GPU region graph then receives those IDs plus the unmodified measured NIR values for the four cores. Its adjacency counts distinct four-neighbor shared edges, and its fixed zero mean-difference threshold permits only exact equal-mean merges. The browser readouts and topology map stay withheld unless all eight local chains, both cross-tile stages, capacity checks, and the bounded region-graph propagation finish without overflow.

The visible topology raster displays those stable component IDs. Its 15-color categorical palette repeats modulo 15; color is an identity cue, not an ordered quantity. Invalid cells and the rest of the 128 × 128 parent crop remain transparent. The crop-frame outline places the exact focus in the surrounding Dixie geography, while the inner two lines mark the four 32 × 32 owners.

## Independent oracle

`scripts/make-tiled-observation-topology-oracle.py` uses only the bundled binaries and Python's standard library. It independently flood-fills each whole 64 × 64 class mask with eight-neighbor connectivity, repeats the flood fill inside each owned 32 × 32 core, then counts four-neighbor adjacency by canonical whole-focus component IDs. It modifies only the `tiledObservationTopology` key in `earth-source-oracles.json`.

| Acquisition | Valid focus cells | Low population; local → global components | High population; local → global components | Local reductions (seam joins) | Low/high region pairs | Shared four-edge contacts | Contacts crossing focus core seams |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021-07-13 | 4,057 | 272; 103 → 96 | 3,785; 6 → 2 | 11 | 97 | 763 | 16 |
| 2021-09-21 | 1,935 | 1,934; 78 → 65 | 1; 1 → 1 | 13 | 1 | 1 | 0 |

The readout population totals must equal the two class populations. The sum of four local component counts minus the whole-focus count is the number of component joins caused by reconciliation. The contact totals include only observed four-neighbor edges between low and high regions; they do not count diagonal contacts or invalid cells. Both local component populations and the independent oracle arrays are retained in the JSON fixture for further comparisons.

The browser verifier is `scripts/check-tiled-observation-topology-browser.mjs`. It captures the served build evidence before opening Chromium, checks both acquisitions against the CPU oracle in the Controls panel, writes screenshots and a report, closes the browser, and verifies the build evidence is unchanged. It is intended for the parent GPU verification slot.
