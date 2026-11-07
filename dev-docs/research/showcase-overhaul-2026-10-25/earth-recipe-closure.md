# Earth recipe closure — 2026-10-25

## Scope and source contract

This wave changes the raster missing-observation and spectral-evidence stories and adds the two
scene-local recipe helpers `missing-surface-preparation.ts` and `spectral-regions.ts`. Source work
is after the wave 11 snapshot. Production rebuild and GPU/browser validation remain with the parent.
No dataset acquisition, shared script, oracle file, install, root build, test, or commit was made.

## Missing-observation preparation

- Source: `dixie-spectral`, July 13 and September 21, 2021; 64 × 64 crop at source row 50, column
  101, north row first, 60 m cells, EPSG:32610. The existing July known-truth tests remain 132
  isolated holdouts and 81 cells in the 9 × 9 block. The real September mask contains 1,047 gaps
  (3,049 valid cells). September gaps have no truth scores.
- The original `fill` output layer and the tested baseline option defaults remain in place. A
  separately labelled `prepared` map uses `addSurfacePreparationRecipe` through
  `addMissingSurfacePreparation`; its inputs retain the selected fill method and search radius.
- Preparation defaults to no rank filter. The selectable median filter uses radius 1 or 2, clamped
  raster edges, and `ignore` or `propagate` nodata policy. These options rebuild the fill/preparation
  graph. `filtered` records changed values, including changes at original-valid pixels;
  `originalValidity`, `filled`, and final `validity` remain separate. Raw fill unresolved count and
  post-filter unresolved count are reported separately. Selected-cell text reports all three
  provenance flags.
- The GPU readback reports fill-only and prepared holdout RMSE/MAE on their respective valid
  withheld cells. Prepared evaluated count is explicit. No donor rule is described as recovering
  hidden truth. Grayscale morphology and structure stay in the existing story step; no seventh step
  was added.

## Dixie regional distribution and paired change

- Source and crop: the same pinned `dixie-spectral` assets described in
  [`scripts/data/dixie-spectral/README.md`](../../../examples/deck/spatial-analysis-showcase/scripts/data/dixie-spectral/README.md):
  2021-07-13 ignition-day and 2021-09-21 observations; rows 50–113 and columns 101–164 of the
  north-first 250 × 250 grid; 60 m EPSG:32610 pixels. Attribution remains “Contains modified
  Copernicus Sentinel data 2021”; the README records the Copernicus open-use notice, pinned COG
  inputs, calibration and strict SCL support.
- Four whole-cell quadrants have stable label order: north-west 0, north-east 1, south-west 2,
  south-east 3. Each has 1,024 assigned cells. Geometry support is computed independently as
  1,024 × 3,600 m² = 3.6864 km² per quadrant; no fractional cell assignment or administrative
  boundaries are used. The graph grid records its EPSG:32610 affine frame.
- Both dates use calibrated NIR reflectance and their strict all-band/SCL validity mask. The
  `addRasterZonalDistributionRecipe` wrapper composes `addZonalDistributionRecipe`, whose
  `GPUGroupStatistics` output `rowCounts` counts masked-in, valid-key rows regardless of value
  finiteness; `validCounts` counts finite NIR values. Median, numeric sample mode, unique count and
  linear quantiles are GPU outputs. The scene’s direct `addDixieZonalDistribution` wrapper is
  available but is not called; the composed raster path is what the scene uses. Numeric modes are
  described as sample modes, never as land-cover labels.
- Date-specific median maps, quantile curves, support readouts and paired-difference map come from
  GPU recipe outputs/readbacks. The selected quadrant is outlined in the original projected frame.
  The default remains the existing selected-pixel probe and spectral index map.
- `addRegionalRasterChangeRecipe` is invoked by `addDixieRegionalChange` in a separate compiled
  graph. The input is a cell-major two-slice, one-band stack `[July NIR, September NIR]` with NaN
  missing values; its parameters select before slice 0 and after slice 1. It is isolated from the
  index epsilon/clamp graph, so those index sensitivity controls do not rebuild or alter regional
  results. The live threshold defaults to 0.05 reflectance and offers 0.02, 0.10 and 0.20 without
  graph replacement. Exact uniform cell areas are 3,600 m². The difference map uses a fixed
  symmetric −0.4 to 0.4 reflectance domain. Regional total/valid/changed area, coverage, valid
  count, common-support mean change, threshold status and recipe stage overflow flags are surfaced.
- The paired mean is the recipe mean of `after − before` on common finite support, not the
  difference of independently supported means. Unknown support stays missing and is never counted
  as unchanged. The cutoff is descriptive, not severity, significance, confidence or causality.
  July 13 is ignition day, and date, season and QA support differ; two dates cannot identify a
  causal fire effect. No t-test is used.

## Verification and remaining evidence

- Scoped Biome check/format on the six owned source/helper files: passed.
- Raster story-step check: passed for all raster stories, including both edited scenes.
- Raster WGSL identifier scan: passed (696 templates scanned).
- Strict raster cartography check: passed for the raster chapter, including both edited scenes.
- Showcase package TypeScript check: run. The latest run reports an unrelated missing-field error
  for `BufferGeometryResult` in `geometry/block-relationships.compute.ts`; preceding runs also
  overlapped active edits in other tracks. No owned file appears in the TypeScript errors.
  This is not a clean package typecheck.
- No GPU/browser execution or numerical oracle comparison was run in this wave. The parent’s
  independent NumPy quantile/mask and native GDAL checks, followed by serialized GPU validation,
  remain pending. This ledger records source implementation, not GPU acceptance.
