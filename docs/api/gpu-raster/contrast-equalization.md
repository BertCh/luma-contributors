# Raster contrast equalization

## `GPURasterContrastEqualization`

{/* lens:GPURasterContrastEqualization */}

::lens{scenario=gpu-raster.raster-contrast-equalization rung=graph}

The synthetic lens records reflected two-cell CLAHE tiles over a gradient with one missing source cell. Its graph shows histogram clipping, CDF scans and interpolation.

`GPURasterContrastEqualization` computes contrast limited adaptive histogram equalization (CLAHE)
or matches a raster's histogram to a caller-supplied reference. Import it from
`@luma-contributors/analysis/raster-analysis`. It complements `GPURasterStretch` with local
contrast control and reference distribution matching. It ships no imagery, datasets or weights.

Pass a published `GPURasterBand`, its `grid`, a `parameters` view and caller-owned `output`
views. Buffer and texture bands use the existing raster canonicalization helpers: raw nodata
and validity are applied before scale/offset; nonfinite calibrated samples are omitted. The
reference in `'match-histograms'` mode supplies its own band and grid and may have a different
shape. Neither method reads world coordinates or computes ground distances, slopes or areas.

| Output | Layout |
| --- | --- |
| `values` | float32, at least `grid.width * grid.height` rows; CLAHE in [0, 1], matching in the reference domain |
| `validity` | uint32 per source cell; nonzero only when the source and mapping are valid |
| `status.approximation` | one uint32, always 1 because distributions are quantized |

Invalid source cells remain invalid and have NaN output values. An empty reference distribution
invalidates every matching output. In CLAHE, empty tiles contribute no interpolation weight;
weights of nonempty neighbors are renormalized. A valid center with no weighted nonempty tile
is invalid. This mask behavior extends the unmasked scikit-image algorithm.

### CLAHE

Use `mode: 'clahe'` (the default), `tileWidth` and `tileHeight` (default 32 each), and `binCount`
(default 256). Tile dimensions and bins are compile-time integers; clip limit and domains are
per-frame parameters written with `getGPURasterContrastEqualizationParameterValues` into a
`GPUParameterBuffer`. After float32 rounding, each domain endpoint must be zero or a finite
normal float32 value, and the increasing domain width must be finite and at least 2^-126.
Subnormal endpoints and widths are rejected: [WGSL permits denormal flushing](https://www.w3.org/TR/WGSL/#floating-point-evaluation),
which could otherwise erase the normalization divisor. Zero endpoints and exactly 2^-126 widths
are supported. Band calibration, view topology and grids are compile-time; band and
validity contents may change between executions.

For a finite sample `v`, first clamp to `inputDomain = [a, b]` and normalize to
`t = (clamp(v, a, b) - a) / (b - a)`. Quantize to a 14-bit integer
`g = round(16383 * t)`, then bin with `floor(g / (1 + floor(16384 / binCount)))`.
Half-integer rounding follows WGSL `round`; exact half-integers can choose either nearest
integer on a GPU. This is scikit-image 0.24's unusual bin width, including its occasionally
unoccupied final bins, rather than a uniform `floor(t * binCount)` histogram.

Each tile has a histogram of its valid samples. The per-bin clip threshold is
`max(1, floor(clipLimit * validCount))`; `clipLimit: 0` disables clipping. The packer raises
positive fractions below 2^-126 to the smallest normal float32, preventing conversion underflow
or GPU denormal flushing from selecting the zero sentinel. This preserves threshold one for
every supported tile (at most 65,536 samples). Clipped excess is
redistributed using scikit-image's quotient, threshold and strided residual procedure, with
integer histogram counts. Each productive residual sweep removes at least one excess count,
so the fixed 65,536-sweep budget covers the maximum tile area; skewed three-bin histograms
can require more sweeps than the bin count. If `threshold * binCount < validCount`, the capped histogram cannot
retain every count; saturated excess is discarded as in scikit-image. The denominator stays
`validCount`. An inclusive published `GPUScan` computes CDFs; subtracting the previous tile's
final prefix yields the tile CDF. The tile map is
`min(16383, floor(16383 * CDF[bin] / validCount)) / 16383`.

Tiles cover the image from its upper-left corner. Incomplete trailing tiles reflect the image
without repeating the edge sample, including its validity; singleton dimensions repeat their
only sample. For interpolation, a cell's tile coordinates are
`((column + floor(tileWidth / 2)) / tileWidth - 1,
(row + floor(tileHeight / 2)) / tileHeight - 1)`.
Clamp the four neighboring tile indices at the tile-grid boundary, then bilinearly interpolate
their mappings at the cell's bin. Odd tile dimensions follow scikit-image's integer half-tile
convention. The four weights sum to one before empty-tile handling.

The formula, clipping sequence and boundary conventions were checked against
[scikit-image 0.24 `_adapthist.py`](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/exposure/_adapthist.py),
which implements Zuiderveld's CLAHE (Graphics Gems IV, 1994). The clipping helper retains its
BSD-3-Clause attribution alongside this project's MIT attribution.

### Histogram matching

Use `mode: 'match-histograms'` and `reference: {input, grid}`. Quantize each normalized domain
with `bin = floor(t * (binCount - 1) + 0.5)`, so half-bin ties round upward. A bin represents
`domainMinimum + bin * (domainMaximum - domainMinimum) / (binCount - 1)`.
Build source and reference histograms using only their valid cells. At a source bin's inclusive
quantile `q = sourceCDF[bin] / sourceCount`, find adjacent **occupied** reference bins with
CDF quantiles `q0, q1` and representatives `v0, v1`. The mapping is
`v0 + (v1 - v0) * (q - q0) / (q1 - q0)`, clamped to the first/last occupied representative
outside the reference quantile interval. Empty reference bins are skipped, so gaps and tied
values follow `numpy.interp` at the actual CDF knots.

Knot ordering uses exact integer cross-products: compare `sourceCDF * referenceCount` with
`referenceCDF * sourceCount`, rather than comparing rounded float32 quantiles. Once the bracketing
reference counts `lowerCount, upperCount` are known, compute the interpolation fraction as
`(sourceCDF * referenceCount - lowerCount * sourceCount) /
((upperCount - lowerCount) * sourceCount)`. The shared uint64 WGSL helpers keep products (at most
2^48) and their difference exact; only this final ratio and value interpolation use float32.
This preserves one-count CDF gaps even when both normalized knots would round to the same float.
For source counts `[16777215, 1]` and reference counts `[12582911, 1]`, the first source bin maps
to 0.25; public scikit-image gives 0.25000000034924597 using float64 quantiles.

This is the quantile rule in
[scikit-image 0.24 `histogram_matching.py`](https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/exposure/histogram_matching.py).
Matching is exact for the quantized distributions, subject to float32 arithmetic. It does not
claim exact matching of all distinct float inputs. Values sharing a bin share a source quantile;
this can change the output rank mapping substantially for densely clustered inputs. Increase
bins or choose tighter domains when that distinction matters.

### Limits and execution

| Limit | Reason |
| --- | --- |
| Source/reference cells, each at most 16,777,216 | 64 MiB packed scalar bindings; exact float32 counts |
| Reflected CLAHE tile cells, at most 16,777,216 | Bounds virtual padding work and histogram totals |
| Total histogram bins across tiles, at most 16,777,216 | Each histogram, scan and map binding is at most 64 MiB |
| `binCount` from 2 through 256 | Bounds per-tile redistribution and inverse-CDF lookup |
| Tile width/height from 1 through 256 | At most 65,536 virtual samples per tile; safe integer map products |

Limits and shapes are validated at construction. Input and reference buffer values and validity
must hold exactly one row per grid cell; declared formats, native nodata, texture channels and
extents are checked through the published, non-owning `GPURaster` API. Output views may have
extra rows, which are left untouched. Device storage-binding limits include the aligned binding
prefix for nonzero view byte offsets and are checked at
the start of `getCommandNodes`, before scratch or node declarations. Histogram clearing,
integer atomic binning, clipping, scan, mapping and apply are separate GPU passes. No invocation
walks over a data-sized range of pixels; clipping and inverse lookup walk only bounded histogram
bins. `estimatedCommandNodeCount` reserves 16 nodes for the published scan and eight per band
canonicalization, plus the contributor's fixed passes. Kernel audits check the actual count at
maximum cells, histogram length and bin count. Counts and CDFs are deterministic integers;
float32 interpolation can vary by a few ulps across devices.

Tile larger images explicitly. CLAHE depends on the contextual tile lattice and neighboring
mappings: independently processing arbitrary image chunks can produce seams. Preserve tile
alignment and provide the neighboring contextual tiles when composing chunked processing.

### Oracle mapping and differences

The checked-in generator is
`modules/analysis/test/gpu-raster/raster-contrast-equalization/generate-reference.py`, run with
`/usr/bin/python3`. It records NumPy, SciPy and scikit-image versions in a small JSON fixture.

- CLAHE: map `(tileHeight, tileWidth)` to `equalize_adapthist(kernel_size=...)`,
  `clipLimit` to `clip_limit`, and `binCount` to `nbins`. For ordinary fixtures, inputs span [0, 1]
  and remain away from bin boundaries. scikit-image first converts through uint16 and globally
  rescales the input to 14 bits; here the caller's explicit domain replaces that conversion and
  rescale. Its `_clahe` supplies the raw output oracle; its public `equalize_adapthist` supplies
  the independently rescaled comparison. The GPU retains fractional interpolation instead of
  truncating it to a 14-bit integer, so the raw tolerance is 1.1/16383. It also omits the public
  API's final global output min/max rescale; rescale GPU results explicitly for that display
  convention. Constant inputs use the fixed domain rather than a data-derived range.
- Masked CLAHE: scikit-image has no mask argument. Fixtures use its `clip_histogram` and
  `map_histogram` and SciPy's linear interpolation, with the documented empty-tile extension.
- Matching: `match_histograms` runs on independently quantized valid source/reference samples,
  with `channel_axis=None`. Fixtures include ties, missing histogram bins, unequal source/reference
  sizes, empty distributions and a constant reference. The large-count fixture stores only the
  two histograms and output values from public `match_histograms` on the full synthetic rasters.
  Its headless regression builds 4096×4096 and 4096×3072 raster bands, then samples four outputs
  to avoid a full raster readback. Matching tolerance is 3e-6 on [0, 1].

Headless specs also cover parameter updates without recompilation, histogram clearing between
runs, calibrated bands, no-data/nonfinite inputs, and unwritten output padding rows. Raster work
has no CPU twin under the contributor contract; fixture generation is an offline oracle only.

### Example

```ts
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPURasterContrastEqualization,
  getGPURasterContrastEqualizationParameterValues,
  GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/raster-analysis';

const contrastParameters = new GPUParameterBuffer(device, {
  id: 'contrast-parameters', format: 'float32',
  length: GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH,
  values: getGPURasterContrastEqualizationParameterValues({
    inputDomain: [0, 1], clipLimit: 0.02
  })
});
graph.add(new GPURasterContrastEqualization({
  grid, input: rasterBand, tileWidth: 32, tileHeight: 32, binCount: 256,
  parameters: contrastParameters.importToGraph(graph),
  output: {values: equalizedValues, validity: equalizedValidity, status: {approximation}}
}));
// Between encodings of the same compiled graph:
contrastParameters.write(getGPURasterContrastEqualizationParameterValues({clipLimit: 0.05}));
// Destroy the compiled graph before contrastParameters.destroy().
```

{/* parameters: GPURasterContrastEqualization generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPURasterContrastEqualization` parameters

- Packer: `getGPURasterContrastEqualizationParameterValues(settings?)`, writes `GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPURasterContrastEqualizationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `inputDomain` (optional) | `readonly [number, number]` | Fixed input domain, clamped before binning. Defaults to [0, 1]. Endpoints must round to zero or normal finite float32 values, with an increasing finite normal width (at least 2^-126). |
| `referenceDomain` (optional) | `readonly [number, number]` | Reference domain for histogram matching. Defaults to [0, 1]. Endpoints must round to zero or normal finite float32 values, with an increasing finite normal width (at least 2^-126). |
| `clipLimit` (optional) | `number` | Fraction of valid tile samples per bin, in [0, 1]. Default 0.01; zero disables clipping. Tiny positive values pack to at least 2^-126, preserving threshold one. |

{/* /parameters */}
