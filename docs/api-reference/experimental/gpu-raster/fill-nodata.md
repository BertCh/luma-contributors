# Nodata filling

`GPURasterFillNoData` prepares a continuous raster from original valid donors, entirely on the GPU.
The default `inverse-distance` method weights every original donor in the bounded pixel disk.
`nearest` selects one donor; equal distances retain row-major search order.

Set `method: 'gdal-four-quadrant'` for the separately implemented GDAL inverse-distance selection.
It selects at most one nearest original donor per quadrant, with weight `1 / distance`, and uses
an inclusive Euclidean pixel distance limit. `maximumDistance` must be an integer from 1 through
256. Distances ignore the grid affine. No smoothing is performed (GDAL `smoothing_iterations=0`).
This initial method does not expose GDAL's unlimited-distance or smoothing options.

The selection follows [GDAL's rasterfill implementation](https://github.com/OSGeo/gdal/blob/v3.10.3/alg/rasterfill.cpp):
upper quadrants include the target row; lower quadrants start on the next row. The target column
belongs to the left quadrants. Horizontal search visits increasing offsets, clamps columns at
image borders, and retains the first donor on equal distances. In particular, right-border
clamping may also select a target-column donor in a right quadrant. These details matter for
parity; dividing a disk into four symmetric sectors gives different answers.

Input validity, raw nodata, non-finite values and calibration use the standard raster-band
canonicalization. Original valid calibrated values are preserved. Filled values never become
donors, including on repeated graph submissions. An unresolved cell returns NaN, validity 0 and
filled 0; a newly filled cell returns validity 1 and filled 1. `unresolvedCount` counts all remaining
invalid cells. GDAL preserves unresolved input values instead, so reference tests compare masks
and normalize that sentinel difference. GPU float32 arithmetic can differ from GDAL's double
accumulation; parity uses numerical tolerances.

The direct Rasterio oracle generator is
`modules/graph/test/oracles/raster-fill-four-quadrant.py`; its JSON records Rasterio and
GDAL versions. Cases cover asymmetric holes, borders and ties, bounded original donors and no
donors. GPU tests resubmit the same compiled graph after changing input values. The implementation
uses bounded local search rather than GDAL's scanline work files: work grows with pixel count and
the squared search radius. Throughput at the maximum bound is not established.

## Work limit

Every missing pixel scans its full `(2 * maximumDistance + 1)^2` window, and the unresolved count is
a parallel atomic reduction. Work is `missingPixels * (2 * maximumDistance + 1)^2` in one dispatch, so
on large, mostly missing grids a large distance can approach GPU watchdog limits; lower
`maximumDistance` or fill tiles with a halo. A jump-flood approximation is not used because it would not reproduce the
exact nearest, all-donor IDW or GDAL four-quadrant results.
