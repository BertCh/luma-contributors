# GPUIsolines: seam-safe tiles and vector overlays

`GPUIsolines`, `GPUIsolinesOperation` and `CPUIsolinesOperation` accept the same optional
`tile: GPUIsolinesTile`. Import these from `@luma-contributors/analysis/raster-analysis`.
Omitting `tile` retains the single-raster behavior described in
[the raster API](operations-analysis.md#gpuisolines-and-gpuisobands).

## Cell ownership and halo samples

Marching cell `(column, row)` reads samples `(column, row)`, `(column + 1, row)`,
`(column + 1, row + 1)` and `(column, row + 1)`. Own the cell by its lower-left sample,
using the **half-open** local rectangle `ownedCells = [minColumn, minRow, maxColumn, maxRow]`.
The upper bounds are excluded. A cell on a core's right/top boundary belongs to the next
core. The owning core must include the right/top samples in its halo. Extra halo cells
emit nothing. Partition the domain's `(domainWidth - 1) × (domainHeight - 1)` marching cells
exactly once. Overlap or gaps between ownership rectangles are the caller's responsibility;
contributors process tiles independently and do not inspect other tiles on the CPU.

For a domain of 9 × 7 samples, four tiles without lower/left padding can be configured as:

| Tile grid | Domain sample offset | Owned local cells |
| --- | --- | --- |
| 5 × 4 | (0, 0) | [0, 0, 4, 3) |
| 5 × 4 | (4, 0) | [0, 0, 4, 3) |
| 5 × 4 | (0, 3) | [0, 0, 4, 3) |
| 5 × 4 | (4, 3) | [0, 0, 4, 3) |

For example the top-right tile has:

```ts
const tile = {
  domainWidth: 9,
  domainHeight: 7,
  columnOffset: 4,
  rowOffset: 3,
  ownedCells: [0, 0, 4, 3] as const
};
```

Lower/left padding is also allowed: move `columnOffset`/`rowOffset` to the first halo sample
and increase the ownership lower bounds accordingly. Do not clamp or duplicate out-of-domain
samples as if they were real samples. No-data and `mask` apply to all four samples of an owned
cell. An empty ownership rectangle is valid and publishes zero counts and zero overflow.

`grid` is the **tile** grid. Build `getGPUIsolinesParameterValues(settings, domainGrid)` using
the **domain** dimensions and extent, then reuse identical parameters, active levels and
`localOrigin` across every tile. The packer chooses shared float32-exact integer sample anchors
near an explicit origin. Tile offsets are added and these anchors are subtracted in integer
arithmetic **before** conversion to float32 and interpolation. This preserves small relative
coordinates even when a domain sample offset exceeds 2^24. Seam crossings therefore use
identical samples and the same expression, including at a four-tile corner. Packing a separate
extent or origin for each tile loses this guarantee.

The eight-word parameter layout is now `[levelCount, anchorX, anchorY, cellWidth, cellHeight,
columnAnchor, rowAnchor, 0]`. The previously reserved words 5 and 6 hold the sample anchors;
`anchorX` and `anchorY` are their world positions relative to `localOrigin`, calculated in
float64 by the packer. Without an explicit origin the anchors remain zero, so legacy manually
packed parameters with zero reserved words retain their interpretation. Anchors are rounded
from in-domain coordinates to float32-exact integers; at very large indices the anchor may be
several samples from the requested origin. Rounding follows the small distances from that
anchor and its world residual, rather than the absolute domain index. For example a 2×2 tile
at column offset 16,777,217 in a 16,777,220×2 domain, with unit samples and shared origin
`[16,777,218, 0]`, places the equality corner at `(0.5, 0.5)` on both incident edges.

Only owned cells contribute to `status.requiredCount`, `count` and `overflow`. A small segment
capacity publishes a deterministic prefix and reports overflow; it suppresses stitched output.
Polyline vertex overflow keeps complete fragments and sets `polylineOverflow`, as in the
single-raster contract. Raw equality-at-level zero-length bridges are retained for topology.

## Joining fragments

With `tile`, `segmentEdges` contains **domain** raster edge IDs, independent of halo layout:

- Horizontal edge `(column, row) → (column + 1, row)`:
  `row * (domainWidth - 1) + column`.
- Vertical edge `(column, row) → (column, row + 1)`:
  `domainHeight * (domainWidth - 1) + row * domainWidth + column`.

Match an end edge to a start edge with the same `segmentLevels` index to join fragments. The
oriented segments keep high values to their left. No coordinate snapping or geometric clipping
at tile boundaries is needed. Optional `polylines` stitch each owned rectangle independently;
this API does not assemble one cross-tile polyline buffer. Keep zero-length raw bridges when
joining by edges: although they have no geometric length, they connect different raster edges.

Tile dimensions, offsets, ownership, capacities and optional views are compile-time. Sample
values, masks, levels and placement parameters are per-frame. Construction validates integer
bounds, all four samples of every owned cell, and domain samples ≤ 2,147,483,647 so two edge
families fit in uint32. The maximum possible segment requirement
`ownedCellCount * levels.length * 2` must fit uint32. GPU scratch bindings are checked against
the device's storage-binding limit before command nodes are created. CPU arithmetic remains
float64 with one output rounding; its documented near-tie differences still apply.

## External vector overlays

The output is in the domain extent's coordinate frame, relative to the shared `localOrigin`.
Subtract that origin from external polygon/line vertices before upload. Use a common planar
projection; this is not a geographic intersection or reprojection API. Row zero is at `minY`
and rows advance toward larger y. The existing grid-affine/north-up limitation remains: reverse
north-up sample rows (and masks) before contouring, rather than mirroring only the contours.

Use existing `GPUSegmentIntersection` for line/polygon-boundary intersections, or existing
line noding/clipping contributors for further overlay steps. Contour segments are packed
`float32x4`; their buffer can also be viewed as `float32x2` positions of two-vertex lines.
Write `lineOffsets[index] = 2 * min(index, count)` for all `segmentCapacity + 1` offsets on
the GPU. This makes every unused line empty and avoids exposing dirty output padding. Then
pass the positions and offsets as `left: {kind: 'lines', positions, lineOffsets}` and an external
polygon geometry as `right`. Overlay `leftRows` are start-vertex rows, so `leftRows / 2` recovers
the owning contour segment and its level; `rightRows` identify external boundary segments.
Check contour overflow and overlay overflow separately. Boundary intersections do not themselves
classify or clip the contour portions inside a polygon.

## Oracle and acceptance

`test/oracles/isolines-tiles.py` generates the committed fixture from whole-domain
`skimage.measure.find_contours` and Shapely line/polygon-boundary intersection. The fixture
covers an affine plane, a masked hill and exact bilinear saddle ties, with a nonzero projected
origin and non-square samples. Node specs compare the CPU twin's tiled union to the whole
domain and the independent fixture, and check equality bridges, empty ownership, capacity and
invalid bounds. Headless specs check the GPU union, bit-identical seam endpoints, every defined
CPU/GPU operation port and resident composition with an external polygon, including dirty padding.
Coordinate comparisons use an absolute 3e-5 budget in a local domain smaller than 25 units;
edge IDs, counts and levels are exact. Large-offset regressions additionally execute the
scatter WGSL in the node interpreter and check the corresponding headless GPU and CPU
operations against hand-derived equality bridges, on both axes and with even/odd shared
origins near sample 2^24. These binary-exact cases require exact coordinate equality.
Headless execution must be run by the landing lead.

## Reading stitched geometry on the host

`decodeGPUIsolinesPolylines({polylines, status}, {levels, localOrigin})` reads retained
polylines from typed arrays, `Uint8Array` buffer readbacks (including offset byte views), or
`ArrayBuffer` readbacks. The stitched decoder accepts Float32Array coordinate/level columns and
Uint32Array count/index/status columns, or Uint8Array/Buffer/ArrayBuffer byte readbacks. Other
typed-array brands are rejected rather than reinterpreted as successful geometry; cross-realm
arrays work. It validates whole elements, scalar status words, live counts, increasing
vertex offsets and level indices before returning geometry. Only `polylineCount` lines and
`vertexCount` XY rows are readable; allocated tails are ignored. `levelIndex` indexes the supplied
`levels`; `levelValue` is that level's raster-value unit. `closed` identifies a ring, whose final
vertex repeats its first. Coordinates include `localOrigin` in the extent's world units and are
not reprojected.

`segmentComplete` means no raw segments were dropped. `polylineComplete` reflects the combined
`polylineOverflow` word: it is false if segments or complete polylines were dropped. Check
`complete` before treating the returned lines as the whole contour dataset. Even without
stitching, a requested `status.polylineOverflow` always receives segment loss (1) or completeness
(0), including on repeated submissions with empty output.

The CPU twin has the same column layout. Supply a positive `vertexCapacity` to request stitching;
the decoder rejects its segment-only result, which has no allocated polyline vertex rows.
This self-contained example uses EPSG:3857 XY in meters. It packs and decodes with the same
`localOrigin`, then explicitly transforms projected coordinates to longitude/latitude degrees.
For another projection, replace the typed transform with the appropriate projection transform.
A degenerate one-vertex chain can arise at an exact-level corner and is omitted from LineString
export. For GPU output, pass the corresponding readbacks of every polylines/status view.

```ts
import {
  computeIsolinesOnCPU,
  decodeGPUIsolinesPolylines,
  getGPUIsolinesParameterValues,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';

const localOrigin = [500000, 4600000] as const;
const grid = makeGPURasterMetadata({
  width: 3, height: 3, origin: localOrigin, cellSize: 1, rowDirection: 'north'
});
const input = {
  values: Float32Array.of(0, 1, 2, 0, 1, 2, 0, 1, 2),
  levels: Float32Array.of(0.5, 1.5),
  parameters: getGPUIsolinesParameterValues({
    levelCount: 2, extent: [500000, 4600000, 500003, 4600003], localOrigin
  }, grid)
};
const result = computeIsolinesOnCPU(input, {grid, segmentCapacity: 8, vertexCapacity: 12});
const decoded = decodeGPUIsolinesPolylines(
  {polylines: result, status: result}, {levels: input.levels, localOrigin}
);
const toLongitudeLatitude = (x: number, y: number): [number, number] => {
  const radiusMeters = 6378137;
  return [
    (x / radiusMeters) * 180 / Math.PI,
    (2 * Math.atan(Math.exp(y / radiusMeters)) - Math.PI / 2) * 180 / Math.PI
  ];
};
if (!decoded.complete) throw new Error('Contour output is incomplete; increase capacity');
const geojson = {
  type: 'FeatureCollection',
  features: decoded.polylines.filter(line => line.coordinates.length >= 2).map(line => ({
    type: 'Feature',
    properties: {levelIndex: line.levelIndex, level: line.levelValue, closed: line.closed},
    geometry: {
      type: 'LineString',
      coordinates: line.coordinates.map(([x, y]) => toLongitudeLatitude(x, y))
    }
  }))
};
console.log(geojson);
```
