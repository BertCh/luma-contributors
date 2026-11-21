# GPURasterTileReprojection

Import from `@luma-contributors/analysis/raster-analysis`.

`GPURasterTileReprojection` explicitly assembles resident source tiles on the GPU, then delegates
projection and sampling to `GPURasterReprojection`. It implements `GPUCommandNodeProducer` and can
be added to a `GPUCommandGraph`. It does not fetch tiles or read source data back to the CPU.

### `GPURasterTileReprojection`

Pass `parameters` packed by `getGPURasterReprojectionParameterValues` to change source and
destination affines between encodings. Tile bounds remain fixed in source pixel coordinates;
resolution, rotation and origin can change while both grid shapes and pixel interpretations stay
fixed. The sampler composes same-CRS destination-to-source sample transforms in float64 before
packing, preserving identical grids on wide sheared lattices. It also supports double-single
projection output. Manual parameter buffers must use [reprojection layout version 3](./expansion.md#gpurasterreprojection),
which anchors same-CRS evaluation near source support to preserve large scale/translation cancellation.

## Source contract

Supply `sourceGrid`, `destinationGrid`, `sources`, `output`, and the existing reprojection options
`inverseProjection`, `method` and `noDataPolicy`. Each source has a borrowed `input: GPURasterBand` and
half-open integer `pixelBounds: [left, top, right, bottom]` relative to the top-left sourceGrid pixel.
Its packed row-major storage covers exactly that rectangle, including any supplied halo.
Bounds must be nonempty and contained in sourceGrid. Empty source lists are supported.
Construction captures both grids' dimensions and affine arrays as well as the tile routing plan.
Declared options, grid fields, tile descriptors and output bindings are read once, including
inherited fields and non-enumerable getters, before validation.
Each grid's CRS authority, well-known text and nested projection JSON are copied as JSON values
at construction. Later edits to either caller-owned CRS cannot change the sampler's same-CRS
comparison. CRS object members are compared by value regardless of member order, recursively;
array order remains significant (for example, axis arrays `[x, y]` and `[y, x]` differ).
Construct a new contributor to change CRS metadata. Incompatible projection input formats are
rejected before mosaic allocation. The affine packer compares source CRS against destination CRS
when `inverseProjection` is absent; it does not compare either grid with the compiled contributor.
Supplying an inverse projection permits different source and destination CRS. Changing both CRS
together to another matching pair also passes the unprojected packer's check. Between submissions,
the caller must keep both grids and the inverse projection consistent with the compiled contributor.
A rejected affine or unprojected source/destination CRS mismatch leaves the previous packet intact.
Changing the original grid objects afterward cannot shrink the mosaic, move validated bounds
outside its allocation, or change the sampler's initial geometry. Construct a new contributor
for a different grid shape; supported affine updates still go through the explicit parameter
buffer described above. GPU views and their contents remain borrowed.

Each input may be a buffer band or a selected channel of a texture band; the two forms may be
mixed in one source list. Texture extents equal the tile rectangle's width and height, rather than
the whole sourceGrid. Textures select one 2D mip/layer with `r32float`/`rgba32float`,
`r32uint`/`rgba32uint` or `r32sint`/`rgba32sint` matching the scalar band format. RGBA channels are
zero-based (default zero); single-channel textures require zero. Construction validates texture
shape, format, channel and the optional band validity mask through the published raster contract.
Texture handles must include `Texture.SAMPLE` usage. Supplied tile `input.scale` and `input.offset`
must remain finite after float32 conversion; invalid usage or calibration is rejected at construction.
Every source's graph ownership is checked before the mosaic clear or any scratch allocation.

The sourceGrid declares one shared CRS, pixel interpretation, affine lattice and resolution for all
source tiles. Bounds select cells on that lattice; mixed overviews, resolutions or CRS require
separate explicit alignment passes. Rotated and sheared affine lattices are supported. The caller
selects the finite source window; tile discovery and automatic window planning are outside this API.
When adapting `GPURasterTile` or `GPURasterTileHaloSource`, subtract the selected source window's
pixel origin from their absolute available bounds, and use the corresponding source window affine.
Do not silently mix level-zero bounds and overview-level pixel bounds.

## Priority, coverage and interpolation

Sources are ordered from low to high priority. Later sources overwrite earlier samples, including
invalid samples. This rule is stable across overlapping cores and halos; put authoritative cores
after lower-priority halo sources. Each band resolves its own raw nodata, validity, finite-value
checks, scale and offset before copying. Missing coverage and invalid samples produce output
validity zero and NaN values. Invalid higher-priority samples do not fall back to older tiles.

Bilinear interpolation resolves all four taps from the assembled mosaic, so taps can cross both
horizontal and vertical tile boundaries. Only positive-weight taps matter. `strict` requires every
such tap; `renormalize` divides by the sum of available valid weights.
Both direct and tiled reprojection accept only `method: 'nearest' | 'bilinear'` (default
`'bilinear'`) and `noDataPolicy: 'propagate' | 'strict' | 'renormalize'` (default `'propagate'`).
`'strict'` is the deprecated spelling of `'propagate'`; both require all positive-weight taps.
Unknown method or policy values reject at construction. Missing outer support remains
nodata under strict sampling; there is no implicit edge clamp. Include an interpolation halo of one
source pixel beyond the required sample centers when dividing work into independent windows.
Nearest sampling compares the fractional coordinate to 0.5 before incrementing the integer part.
Exact half-cell ties select the higher index; representable coordinates just below a tie select
the lower index without rounding an added 0.5.

Projection remains destination driven. An explicit inverse projection is required when destination
and source CRS differ. Projection-domain invalidity propagates to output validity. Coordinates and
samples are float32; double-single inverse projections are supported by the sampler. The adapter adds no longitude wrapping.
Use local coordinates and bounded windows when large world origins would lose pixel precision.

## Continuous values and native integer labels

`method: 'nearest'` alone still returns calibrated **continuous float32** `output.values`.
Integers beyond 2^24 may round, so do not use that output to recover categorical IDs.
For exact labels, additionally request `output.labels` with format `uint32` or `sint32`, matching
every source band. This route requires explicit `method: 'nearest'`, integer inputs of one common
format, and identity calibration (`scale: 1`, `offset: 0`). Native nodata is compared before any
float conversion, and mosaic assembly preserves label bits.

Always request/read `output.validity` alongside labels: zero means the label word is unspecified;
nonzero makes that destination label usable. For uint32 input `[16777216, 16777217]` and identity geometry,
`method: 'nearest', output: {values, validity, labels}` preserves both exact label words with
validity `[1, 1]`, while continuous float32 values are `[16777216, 16777216]`.
With unit cells, shifting the destination origin by half a source cell in the positive X
direction selects exact ID `16777217` then outside support: validity `[1, 0]`, and the second
label word is unspecified. Nearest ties select the higher source index; no extrapolation occurs.
`output.values` remains required alongside optional native labels. A missing tile or native nodata
makes validity zero. A supplied native nodata sentinel must be an integer in the selected uint32
or sint32 domain. This is the same
[integer-label route as single-grid reprojection](./expansion.md#gpurasterreprojection).

## Resources and reuse

Packing is explicit: dense mosaic scratch uses eight bytes per sourceGrid pixel (float32 values and
uint32 validity), plus the existing per-tile calibration and projection intermediates. This is intended
for bounded local source windows, not a sparse whole-world atlas. Each buffer must fit the device's
storage binding and allocation limits. Sequential copy passes use four storage bindings, independent
of source tile count. `estimatedCommandNodeCount` is `3 + 2 * sources.length`: one mosaic clear,
one allowance for final mosaic canonicalization, one sampler, and one canonicalization allowance
plus one route per tile. The published buffer canonicalizer and texture-to-buffer converter each
currently use one command node;
tests check the actual graph does not exceed this estimate. Construction calls
`validateCommandNodeBudget` against 32,768 nodes, accepting at most **16,382 source tiles** and
rejecting 16,383, including overlapping one-cell tiles. The error identifies `sources.length`.
This check uses only list length and runs before copying or reading any tile metadata.
Compile size and dispatch overhead grow with tile count. There is no measured throughput or
large-window performance guarantee.

Changing borrowed input buffer or texture contents and executing the same compiled graph recomputes the mosaic
and output. Tile metadata and source order are graph topology: rebuild the contributor and graph when
they change. Routes use a private construction-time snapshot of the list, bounds, band calibration
and input bindings, retaining GPU view identities. Band and storage fields supplied through inherited
or non-enumerable properties are captured with the same calibration and channel meanings as own
enumerable properties. Replacing `contributor.props.sources`, changing
borrowed tile bounds or replacing borrowed bindings does not change an existing plan; updates to
the contents of the captured GPU buffers remain visible. After destroying the compiled graph,
call `contributor.destroy()` to release optional
projection resources. Borrowed input buffers remain caller-owned.

## Reference evidence

`test/oracles/raster-tile-reprojection.py` generates the checked-in JSON using NumPy 2.2.6 and
Rasterio 1.4.3 / GDAL 3.9.3. Rasterio directly checks interior horizontal and corner-seam bilinear
samples. Strict missing-support edges, renormalization, and nodata-overwriting overlap priority are
explicit NumPy compositions; they are not claims of matching GDAL's edge-clamping behavior.

The focused tests cover shader parsing and eight-storage-binding limits with 32 source tiles, real
WebGPU core-device seam and corner-seam sampling, missing tiles, overlap priority, nodata, empty
coverage, and changed borrowed inputs on a reused compiled graph.

{/* parameters: GPURasterTileReprojection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

## `GPURasterTileReprojection` parameters

Import from `@luma-contributors/analysis/raster-analysis`: `GPURasterTileReprojection`.

- Packer: `getGPURasterReprojectionParameterValues(settings)`, writes `GPU_RASTER_REPROJECTION_PARAMETER_LENGTH` (20) `float32` values.
  - Input port: `parameters`.

Settings of `getGPURasterReprojectionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `sourceGrid` | `GPURasterMetadata` | - |
| `destinationGrid` | `GPURasterMetadata` | - |
| `inverseProjection` (optional) | `CompiledProjection` | Same compiled inverse projection used by the contributor. |

{/* /parameters */}
