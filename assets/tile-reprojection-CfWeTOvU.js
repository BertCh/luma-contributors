var e={slug:`api/gpu-raster/tile-reprojection`,title:`GPURasterTileReprojection`,html:`<h1 id="gpurastertilereprojection">GPURasterTileReprojection</h1>
<p><code>GPURasterTileReprojection</code> explicitly assembles resident source tiles on the GPU, then delegates
projection and sampling to <code>GPURasterReprojection</code>. It implements <code>GPUCommandNodeProducer</code> and can
be added to a <code>GPUCommandGraph</code>. It does not fetch tiles or read source data back to the CPU.</p>
<h3 id="gpurastertilereprojection-1"><code>GPURasterTileReprojection</code></h3>
<p>Pass <code>parameters</code> packed by <code>getGPURasterReprojectionParameterValues</code> to change source and
destination affines between encodings. Tile bounds remain fixed in source pixel coordinates;
resolution, rotation and origin can change while both grid shapes and pixel interpretations stay
fixed. The sampler composes same-CRS destination-to-source sample transforms in float64 before
packing, preserving identical grids on wide sheared lattices. It also supports double-single
projection output. Manual parameter buffers must use <a href="/luma-contributors/docs/api/gpu-raster/expansion#gpurasterreprojection">reprojection layout version 3</a>,
which anchors same-CRS evaluation near source support to preserve large scale/translation cancellation.</p>
<h2 id="source-contract">Source contract</h2>
<p>Supply <code>sourceGrid</code>, <code>destinationGrid</code>, <code>sources</code>, <code>output</code>, and the existing reprojection options
<code>inverseProjection</code>, <code>method</code> and <code>noDataPolicy</code>. Each source has a borrowed <code>input: GPURasterBand</code> and
half-open integer <code>pixelBounds: [left, top, right, bottom]</code> relative to the top-left sourceGrid pixel.
Its packed row-major storage covers exactly that rectangle, including any supplied halo.
Bounds must be nonempty and contained in sourceGrid. Empty source lists are supported.
Construction captures both grids' dimensions and affine arrays as well as the tile routing plan.
Declared options, grid fields, tile descriptors and output bindings are read once, including
inherited fields and non-enumerable getters, before validation.
Each grid's CRS authority, well-known text and nested projection JSON are copied as JSON values
at construction. Later edits to either caller-owned CRS cannot change the sampler's same-CRS
comparison. Construct a new contributor to change CRS metadata.
Changing the original grid objects afterward cannot shrink the mosaic, move validated bounds
outside its allocation, or change the sampler's initial geometry. Construct a new contributor
for a different grid shape; supported affine updates still go through the explicit parameter
buffer described above. GPU views and their contents remain borrowed.</p>
<p>Each input may be a buffer band or a selected channel of a texture band; the two forms may be
mixed in one source list. Texture extents equal the tile rectangle's width and height, rather than
the whole sourceGrid. Textures select one 2D mip/layer with <code>r32float</code>/<code>rgba32float</code>,
<code>r32uint</code>/<code>rgba32uint</code> or <code>r32sint</code>/<code>rgba32sint</code> matching the scalar band format. RGBA channels are
zero-based (default zero); single-channel textures require zero. Construction validates texture
shape, format, channel and the optional band validity mask through the published raster contract.
Texture handles must include <code>Texture.SAMPLE</code> usage. Supplied tile <code>input.scale</code> and <code>input.offset</code>
must remain finite after float32 conversion; invalid usage or calibration is rejected at construction.
Every source's graph ownership is checked before the mosaic clear or any scratch allocation.</p>
<p>The sourceGrid declares one shared CRS, pixel interpretation, affine lattice and resolution for all
source tiles. Bounds select cells on that lattice; mixed overviews, resolutions or CRS require
separate explicit alignment passes. Rotated and sheared affine lattices are supported. The caller
selects the finite source window; tile discovery and automatic window planning are outside this API.
When adapting <code>GPURasterTile</code> or <code>GPURasterTileHaloSource</code>, subtract the selected source window's
pixel origin from their absolute available bounds, and use the corresponding source window affine.
Do not silently mix level-zero bounds and overview-level pixel bounds.</p>
<h2 id="priority-coverage-and-interpolation">Priority, coverage and interpolation</h2>
<p>Sources are ordered from low to high priority. Later sources overwrite earlier samples, including
invalid samples. This rule is stable across overlapping cores and halos; put authoritative cores
after lower-priority halo sources. Each band resolves its own raw nodata, validity, finite-value
checks, scale and offset before copying. Missing coverage and invalid samples produce output
validity zero and NaN values. Invalid higher-priority samples do not fall back to older tiles.</p>
<p>Bilinear interpolation resolves all four taps from the assembled mosaic, so taps can cross both
horizontal and vertical tile boundaries. Only positive-weight taps matter. <code>strict</code> requires every
such tap; <code>renormalize</code> divides by the sum of available valid weights. Missing outer support remains
nodata under strict sampling; there is no implicit edge clamp. Include an interpolation halo of one
source pixel beyond the required sample centers when dividing work into independent windows.
Nearest sampling compares the fractional coordinate to 0.5 before incrementing the integer part.
Exact half-cell ties select the higher index; representable coordinates just below a tie select
the lower index without rounding an added 0.5.</p>
<p>Projection remains destination driven. An explicit inverse projection is required when destination
and source CRS differ. Projection-domain invalidity propagates to output validity. Coordinates and
samples are float32; double-single inverse projections are supported by the sampler. The adapter adds no longitude wrapping.
Use local coordinates and bounded windows when large world origins would lose pixel precision.</p>
<h2 id="resources-and-reuse">Resources and reuse</h2>
<p>Packing is explicit: dense mosaic scratch uses eight bytes per sourceGrid pixel (float32 values and
uint32 validity), plus the existing per-tile calibration and projection intermediates. This is intended
for bounded local source windows, not a sparse whole-world atlas. Each buffer must fit the device's
storage binding and allocation limits. Sequential copy passes use four storage bindings, independent
of source tile count. <code>estimatedCommandNodeCount</code> is <code>3 + 2 * sources.length</code>: one mosaic clear,
one allowance for final mosaic canonicalization, one sampler, and one canonicalization allowance
plus one route per tile. The published buffer canonicalizer and texture-to-buffer converter each
currently use one command node;
tests check the actual graph does not exceed this estimate. Construction calls
<code>validateCommandNodeBudget</code> against 32,768 nodes, accepting at most <strong>16,382 source tiles</strong> and
rejecting 16,383, including overlapping one-cell tiles. The error identifies <code>sources.length</code>.
This check uses only list length and runs before copying or reading any tile metadata.
Compile size and dispatch overhead grow with tile count. There is no measured throughput or
large-window performance guarantee.</p>
<p>Changing borrowed input buffer or texture contents and executing the same compiled graph recomputes the mosaic
and output. Tile metadata and source order are graph topology: rebuild the contributor and graph when
they change. Routes use a private construction-time snapshot of the list, bounds, band calibration
and input bindings, retaining GPU view identities. Band and storage fields supplied through inherited
or non-enumerable properties are captured with the same calibration and channel meanings as own
enumerable properties. Replacing <code>contributor.props.sources</code>, changing
borrowed tile bounds or replacing borrowed bindings does not change an existing plan; updates to
the contents of the captured GPU buffers remain visible. After destroying the compiled graph,
call <code>contributor.destroy()</code> to release optional
projection resources. Borrowed input buffers remain caller-owned.</p>
<h2 id="reference-evidence">Reference evidence</h2>
<p><code>test/oracles/raster-tile-reprojection.py</code> generates the checked-in JSON using NumPy 2.2.6 and
Rasterio 1.4.3 / GDAL 3.9.3. Rasterio directly checks interior horizontal and corner-seam bilinear
samples. Strict missing-support edges, renormalization, and nodata-overwriting overlap priority are
explicit NumPy compositions; they are not claims of matching GDAL's edge-clamping behavior.</p>
<p>The focused tests cover shader parsing and eight-storage-binding limits with 32 source tiles, real
WebGPU core-device seam and corner-seam sampling, missing tiles, overlap priority, nodata, empty
coverage, and changed borrowed inputs on a reused compiled graph.</p>
<p>{/* parameters: GPURasterTileReprojection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h2 id="gpurastertilereprojection-parameters"><code>GPURasterTileReprojection</code> parameters</h2>
<ul>
<li>Packer: <code>getGPURasterReprojectionParameterValues(settings)</code>, writes <code>GPU_RASTER_REPROJECTION_PARAMETER_LENGTH</code> (20) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterReprojectionParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>sourceGrid</code></td>
<td><code>GPURasterMetadata</code></td>
<td>-</td>
</tr>
<tr>
<td><code>destinationGrid</code></td>
<td><code>GPURasterMetadata</code></td>
<td>-</td>
</tr>
<tr>
<td><code>inverseProjection</code> (optional)</td>
<td><code>CompiledProjection</code></td>
<td>Same compiled inverse projection used by the contributor.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};