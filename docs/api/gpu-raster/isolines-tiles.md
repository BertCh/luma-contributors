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
