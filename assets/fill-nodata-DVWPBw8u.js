var e={slug:`api/gpu-raster/fill-nodata`,title:`Nodata filling`,html:`<h1 id="nodata-filling">Nodata filling</h1>
<p><code>GPURasterFillNoData</code> prepares a continuous raster from original valid donors, entirely on the GPU.
The default <code>inverse-distance</code> method weights every original donor in the bounded pixel disk.
<code>nearest</code> selects one donor; equal distances retain row-major search order.</p>
<p>Set <code>method: 'gdal-four-quadrant'</code> for the separately implemented GDAL inverse-distance selection.
It selects at most one nearest original donor per quadrant, with weight <code>1 / distance</code>, and uses
an inclusive Euclidean pixel distance limit. <code>maximumDistance</code> must be an integer from 1 through
256. Distances ignore the grid affine. No smoothing is performed (GDAL <code>smoothing_iterations=0</code>).
This initial method does not expose GDAL's unlimited-distance or smoothing options.</p>
<p>The selection follows <a href="https://github.com/OSGeo/gdal/blob/v3.10.3/alg/rasterfill.cpp">GDAL's rasterfill implementation</a>:
upper quadrants include the target row; lower quadrants start on the next row. The target column
belongs to the left quadrants. Horizontal search visits increasing offsets, clamps columns at
image borders, and retains the first donor on equal distances. In particular, right-border
clamping may also select a target-column donor in a right quadrant. These details matter for
parity; dividing a disk into four symmetric sectors gives different answers.</p>
<p>Input validity, raw nodata, non-finite values and calibration use the standard raster-band
canonicalization. Original valid calibrated values are preserved. Filled values never become
donors, including on repeated graph submissions. An unresolved cell returns NaN, validity 0 and
filled 0; a newly filled cell returns validity 1 and filled 1. <code>output.status</code> is required (an object);
its optional <code>invalidCount</code> (one <code>uint32</code> row) counts the cells of the result that are still invalid
(<code>validity</code> 0: no valid source within the search distance). It is the status vocabulary's
<code>invalidCount</code> over output cells, not over input rows rejected by validation. GDAL preserves unresolved input values instead, so reference tests compare masks
and normalize that sentinel difference. GPU float32 arithmetic can differ from GDAL's double
accumulation; parity uses numerical tolerances.</p>
<p><code>input</code> accepts a published <code>GPURasterBand</code> backed by a packed buffer or a selected texture
channel. Texture dimensions must equal <code>grid.width</code> and <code>grid.height</code>; use a single 2D mip/layer
of <code>r32float</code>, <code>rgba32float</code>, <code>r32uint</code>, <code>rgba32uint</code>, <code>r32sint</code> or <code>rgba32sint</code> matching the band
format. Set <code>input.storage.channel</code> (default zero) for RGBA inputs. The published texture-to-buffer
conversion evaluates raw nodata and the optional band <code>validity</code> mask before calibration. Invalid
texture shape/channel/format is rejected at construction, and foreign graph ownership is rejected
before scratch allocation. The texture handle must include <code>Texture.SAMPLE</code> usage. Supplied
<code>input.scale</code> and <code>input.offset</code> must remain finite after float32 conversion. These checks run
before output allocation in the surface-preparation recipe as well, allowing a corrected same-ID
retry after rejection. Band metadata, nested storage and mask/view identities are captured at
construction; texture and mask contents remain per-submission inputs.</p>
<p>The direct Rasterio oracle generator is
<code>modules/analysis/test/oracles/raster-fill-four-quadrant.py</code>; its JSON records Rasterio and
GDAL versions. Cases cover asymmetric holes, borders and ties, bounded original donors and no
donors. GPU tests resubmit the same compiled graph after changing input values. The implementation
uses bounded local search rather than GDAL's scanline work files: work grows with pixel count and
the squared search radius. Throughput at the maximum bound is not established.</p>
<h2 id="work-limit">Work limit</h2>
<p>Every missing pixel scans its full <code>(2 * maximumDistance + 1)^2</code> window, and the unresolved count is
a parallel atomic reduction. Work is <code>missingPixels * (2 * maximumDistance + 1)^2</code> in one dispatch, so
on large, mostly missing grids a large distance can approach GPU watchdog limits; lower
<code>maximumDistance</code> or fill tiles with a halo. A jump-flood approximation is not used because it would not reproduce the
exact nearest, all-donor IDW or GDAL four-quadrant results.</p>`};export{e as default};