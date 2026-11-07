# Missing observations implementation ledger

This note records the source window, holdout geometry and output semantics for
`examples/deck/spatial-analysis-showcase/src/scenes/raster/missing-observations.scene.ts`.
Numerical browser results remain pending GPU verification; no CPU reference values below are
presented as GPU results.

## Source and probes

- Dataset: pinned Dixie Sentinel-2 L2A NIR reflectance; `nir_before` (13 July 2021) and
  `nir_after` (21 September 2021), each with its supplied strict validity mask.
- Crop: source rows 50–113 and columns 101–164, row-major 64 × 64. The crop preserves the
  60 m EPSG:32610 grid and covers 3,840 m × 3,840 m. July support is 4,096/4,096 cells;
  September support is 3,049/4,096 cells, with 1,047 source gaps.
- The selected probe starts at local row 31, column 31 (source row 81, column 132), inside the
  central block holdout. It is not treated as a representative sample.
- Artificial isolated holdouts are the valid July cells whose row-major crop index `i` satisfies
  `(17*i + 13) mod 31 = 0` (132 cells in this window). The contiguous experiment removes local
  rows 27–35 and columns 27–35 inclusive (81 cells). Both masks are applied to July observations,
  while a separate copy preserves their measured truth.

## GPU output semantics

- `GPURasterFillNoData` is built for three real inputs (isolated July, block-held-out July and
  September source mask) × three donor rules (nearest, all-donor inverse distance and
  four-quadrant inverse distance). Search radius is 2, 4, 8 or 12 pixels (120–720 m). Original
  valid samples are unchanged, and filled values are never donors. The GPU writes value,
  validity, filled provenance and unresolved count for each variant.
- The error map contains `filled − measured July` only at deliberately withheld July cells;
  every other output is NaN. RMSE/MAE are computed from GPU-readback predictions and the preserved
  July truth only over valid holdout outputs. September has no known truth, so the error map and
  scores are unavailable in that mode.
- `GPURasterFillHoles` receives a binary observation mask (valid=1, missing=0). It classifies
  enclosed missing-mask components, not reflectance. Its 4/8 connectivity and fixed 128-round
  budget are explicit; its map and count are accepted only when the GPU convergence flag is set.
- Grayscale dilation, erosion, opening and closing use the measured July reflectance with the
  selected square/cross structuring element and radius. NoData policy is `ignore`, with July
  fully valid in this crop.
- Grayscale opening reconstruction uses a radius-2 erosion marker and dilation beneath the
  measured July mask. Closing uses a radius-2 dilation marker and erosion above that mask. The
  GPU checks an additional round after the selected budget (8, 32 or 128); a non-converged
  result remains invalid and is not rendered as a partial reconstruction.
- Contour classification covers the 63 × 63 interior cells; geometry capacity is two segments per
  cell (7,938). Level, required/emitted counts, overflow, per-cell case and segment count are
  read back. Exact-level samples count as high; ambiguous cases use the bilinear determinant and
  the contributor's deterministic high-diagonal policy on exact ties. Pixel-center coordinates
  are transformed to the crop's local UTM frame before map rendering.

## Interpretation references

The fill choices and zero-smoothing behavior are documented against
[GDAL FillNodata](https://gdal.org/en/stable/api/python/utilities.html). Marker ordering and
constrained grayscale propagation follow the definitions in
[scikit-image morphology](https://scikit-image.org/docs/stable/api/skimage.morphology). The local
GPU contributor contract remains authoritative for implementation bounds and status behavior.
