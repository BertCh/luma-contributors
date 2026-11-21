---
title: Raster patch metrics and sieving
description: Per-patch shape metrics and small-patch removal over dense label rasters.
---

# Raster patch metrics and sieving

Import from `@luma-contributors/analysis/raster-analysis`.

`GPURasterPatchMetrics` and `GPURasterSieve` are companions to upstream `GPURasterConnectedComponents` and
`GPURasterDenseComponents` (`@luma.gl/graph/gpu-raster`). Both read dense labels (`0` is background) and take the same optional
`mask` (an input row filter: zero turns the pixel into background), `converged`, `componentCount`,
and `overflow` guards, the last three as in `GPURasterRegionMeasurements`; the sparse labels of `GPURasterConnectedComponents` must go through
`GPURasterDenseComponents` first.

- `GPURasterPatchMetrics` writes per-patch columns, where row `r` is label `r + 1`: `pixelCounts`,
  `areas`, `perimeterFaceCounts`, `perimeters` (length in CRS units through `grid.affine`), and the
  inclusive bounding box `minColumns`, `minRows`, `maxColumns`, `maxRows`. Faces on the raster
  edge count as perimeter unless `countRasterBorder` is `false`. `cellSizeMode: 'geographic'` (degrees)
  or `'web-mercator'` (EPSG:3857 meters) turns `areas` and `perimeters` into WGS84 ground square
  meters and meters, with the scale taken at each row center (axis-aligned grids only; the
  default `'uniform'` keeps CRS units). The optional `parameters` view (layout of
  `getGPURasterPatchMetricsParameterValues`, shared with fractional coverage) supplies affine
  steps and area in every mode, plus row latitude in ground modes; without it the initial affine
  stays fixed. It
  also repeats the pixel count and area so shape ratios (FRAGSTATS patch metrics, GRASS
  `r.object.geometry`) need one node set. Counts, extents and perimeter faces are integer atomics, so
  they are exact. The ground sums are 64-bit fixed-point atomics, deterministic and independent of
  pixel order, but **not exact**: each row's weight is rounded to a multiple of 2^-26 of the largest
  scale the model can reach (equator for x, pole for y), a relative error of at most 2^-27 over the
  row's weight, about 1e-7 at 60 degrees latitude and about 1e-5 at 89.95 degrees. A perimeter face
  along a parallel is counted at its pixel's row center, so it differs from the geodesic length by a
  second-order term (below 1e-4 for one-degree cells).
- `GPURasterSieve` (GDAL `gdal_sieve`, GRASS `r.reclass.area`) removes patches smaller than
  `minimumPixels`. `mode: 'remove'` (default) turns them into background. `mode: 'merge'` gives each
  small patch the label of its largest neighbor that is not small (lowest label on ties; `connectivity`
  4 or 8) and removes it when it has none, where `gdal_sieve` leaves it unaltered. Background (label 0)
  is never a merge target, as `gdal_sieve` never merges into masked pixels. `minimumPixels` is per-frame, written with
  `getGPURasterSieveParameterValues` into `parameters`. Outputs are `labels`, optional `patchTargets`
  (final label per patch) and `sievedCount`. Merging is a single pass. A clump labeling never has
  touching labels, so `'merge'` is meant for segmentation label rasters.

```ts
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPURasterPatchMetrics, GPURasterSieve, getGPURasterSieveParameterValues} from '@luma-contributors/analysis/raster-analysis';

declare const rasterMetadata: GPURasterMetadata;
// Dense labels and mask cover the grid; the three status inputs each have one row.
declare const denseLabels: GraphDataView<'uint32'>;
declare const mask: GraphDataView<'uint32'>;
declare const converged: GraphDataView<'uint32'>;
declare const componentCount: GraphDataView<'uint32'>;
declare const overflow: GraphDataView<'uint32'>;
// Per-patch columns have capacity 256. sievedLabels covers the grid; sievedCount is scalar.
declare const pixelCounts: GraphDataView<'uint32'>;
declare const minColumns: GraphDataView<'uint32'>;
declare const minRows: GraphDataView<'uint32'>;
declare const maxColumns: GraphDataView<'uint32'>;
declare const maxRows: GraphDataView<'uint32'>;
declare const sievedLabels: GraphDataView<'uint32'>;
declare const sievedCount: GraphDataView<'uint32'>;
declare const areas: GraphDataView<'float32'>;
declare const perimeters: GraphDataView<'float32'>;
const sieveParameters = new GPUParameterBuffer(device, {
  id: 'sieve-settings', format: 'uint32', length: 4,
  values: getGPURasterSieveParameterValues({minimumPixels: 8})
});
graph.add(new GPURasterPatchMetrics({
  grid: rasterMetadata, labels: denseLabels, mask,
  status: {converged, componentCount, overflow}, // the labeling passes' status scalars
  output: {pixelCounts, areas, perimeters, minColumns, minRows, maxColumns, maxRows}
}));
graph.add(new GPURasterSieve({
  grid: rasterMetadata, labels: denseLabels, mask, patchCapacity: 256,
  status: {converged, componentCount, overflow},
  parameters: sieveParameters.importToGraph(graph),
  mode: 'merge', connectivity: 8, output: {labels: sievedLabels, sievedCount}
}));
// Between encodings; destroy the compiled graph before sieveParameters.destroy().
sieveParameters.write(getGPURasterSieveParameterValues({minimumPixels: 16}));
```

## Handoff to landscape metrics

Pass the complete `GPURasterPatchLabels` port as `denseLabels` to the landscape adapter.
Raster labels use zero for background and IDs `1..N`; the legacy `patches` prop instead uses
zero-based patch rows, so passing the raw raster labels under that name changes membership.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterPatchLabels} from '@luma-contributors/analysis/raster-analysis';
import {addLandscapeMetricsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
// Three unit cells, labels [0,1,1], no mask, converged [1], overflow [0], componentCount [1].
declare const patchLabels: GPURasterPatchLabels;
// One row, class [0], for dense label 1.
declare const patchClasses: GraphDataView<'uint32'>;
declare const patchMetrics: GraphDataView<'float32x4'>;
addLandscapeMetricsRecipe(graph, {
  width: 3, height: 1, cellSize: [1,1], classCount: 1, connectivity: 4,
  denseLabels: patchLabels, patchClasses, output: {patchMetrics}
});
```

`patchLabels` includes `grid`, `labels`, optional `mask`, and
`status: {converged, overflow, componentCount}` from the producer. Carry all available guards:
`converged = 0` or `overflow != 0` suppresses downstream metrics; `componentCount` bounds accepted
IDs. The complete fixture `[0,1,1]` yields area **2** in `patchMetrics` row 0, first component,
because the background cell contributes nothing. With unit cells and the default boundary count,
the row is `[2,6,2,2]` (area, perimeter, cell count, directed like-adjacent sides).
See [landscape metrics and proximity](../gpu-spatial-analysis/landscape-ecology.md).

#### `GPURasterPatchMetrics`

Measures pixel count, area and perimeter of each labelled patch of the burn mask.

Pass `parameters` packed by `getGPURasterPatchMetricsParameterValues(grid)` to update the affine
between encodings. The eight-word layout matches fractional coverage. Uniform mode reads the
absolute determinant for area and both affine axis lengths for perimeter, including rotated and
sheared cells. Ground modes also read each row's latitude; they require axis-aligned grids on every
encoding. Their fixed-point weights and perimeter normalization follow the new resolution.
The ten-word layout from
`getGPURasterFractionalCoverageParameterValues(absoluteGrid, {localOrigin})` is also accepted:
word 7 stays ignored legacy padding, word 8 is `2`, and word 9 restores absolute Y for ground
areas/perimeters. Subtract the same local origin in binary64 before uploading polygon vertices;
reuse those ten words for clipping and area consumers, as in the
[shared-frame example](./expansion.md#geometry-and-coverage). Legacy padded packets retain their
absolute origin in word 1; nonzero padding never shifts their row latitude.
Dimensions, pixel interpretation and `cellSizeMode` stay fixed. Without `parameters`, the initial
affine remains fixed for compatibility.
The packer requires a nonsingular packed affine and a physical area determinant that remains
finite and nonzero after float32 rounding. It validates before writing any target word. A legacy
write into a reused target of at least ten words clears words 8 and 9, deactivating a previous
version-2 absolute-Y extension. The parameter-driven path requires a nonsingular affine representable in float32.
Fixed-affine pixel counts do not depend on that packing, so an underflowing determinant does
not prevent counting patches.

::lens{scenario=gpu-raster.raster-patch-metrics rung=result caption="Burn-mask patches; one patch holds most of the 2473 burned cells."}

{/* parameters: GPURasterPatchMetrics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

##### `GPURasterPatchMetrics` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterPatchMetrics`.

- Packer: `getGPURasterPatchMetricsParameterValues(grid)`, writes `GPU_RASTER_PATCH_METRICS_PARAMETER_LENGTH` (8) `float32` values.
  - `grid` (`GPURasterMetadata`): Raster grid whose affine will be measured. The physical determinant is rounded from the declared affine, separately from packed-axis coordinate metadata. It must be finite and nonzero in float32; validation precedes all writes. Reusing a ten-word version-2 target clears its discriminator and absolute-Y offset.

{/* /parameters */}

#### How it computes

{/* lens:GPURasterPatchMetrics */}

::lens{scenario=gpu-raster.raster-patch-metrics rung=datum datum=0 caption="Patch 0, the largest: its pixel count, exposed faces, area and perimeter."}
::lens{scenario=gpu-raster.raster-patch-metrics rung=formula caption="Area is pixels times cell area; perimeter sums exposed faces times their face length."}
::lens{scenario=gpu-raster.raster-patch-metrics rung=kernel caption="Run segments accumulate counts and exposed faces with integer atomics, then columns are finalized."}
::lens{scenario=gpu-raster.raster-patch-metrics rung=graph caption="Connected components and dense labels feed the key, count, face and area nodes."}

#### `GPURasterSieve`

Removes patches smaller than a pixel threshold from a patch label raster.

::lens{scenario=gpu-raster.raster-sieve rung=result caption="Burn patches of 12 cells or more survive; smaller patches become background."}

{/* parameters: GPURasterSieve generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

##### `GPURasterSieve` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterSieve`.

- Packer: `getGPURasterSieveParameterValues(settings)`, writes `GPU_RASTER_SIEVE_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPURasterSieveParameterValues`:

| Field | Type | Description |
|---|---|---|
| `minimumPixels` | `number` | Patches with fewer pixels than this are sieved. Integer from 1 through 2^32 - 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPURasterSieve */}

::lens{scenario=gpu-raster.raster-sieve rung=datum datum=32 caption="Patch 32, 11 cells: smaller than the threshold of 12, so its target is background."}
::lens{scenario=gpu-raster.raster-sieve rung=formula caption="A patch keeps its label when its size reaches the minimum; otherwise it becomes background."}
::lens{scenario=gpu-raster.raster-sieve rung=kernel caption="Counts per patch decide each target, then one pass relabels every pixel."}
::lens{scenario=gpu-raster.raster-sieve rung=graph caption="Labelling feeds counts, per-patch targets and the relabel node."}

## Related pages

- [GPURaster analysis contributors](/docs/api/gpu-raster/operations-analysis)
- [luma.gl GPU Raster](https://luma.gl/docs/api-reference/experimental/gpu-raster)
