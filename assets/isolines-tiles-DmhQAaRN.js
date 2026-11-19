var e={slug:`api/gpu-raster/isolines-tiles`,title:`GPUIsolines: seam-safe tiles and vector overlays`,html:`<h1 id="gpuisolines-seam-safe-tiles-and-vector-overlays">GPUIsolines: seam-safe tiles and vector overlays</h1>
<p><code>GPUIsolines</code>, <code>GPUIsolinesOperation</code> and <code>CPUIsolinesOperation</code> accept the same optional
<code>tile: GPUIsolinesTile</code>. Import these from <code>@luma-contributors/analysis/raster-analysis</code>.
Omitting <code>tile</code> retains the single-raster behavior described in
<a href="/luma-contributors/docs/api/gpu-raster/operations-analysis#gpuisolines-and-gpuisobands">the raster API</a>.</p>
<h2 id="cell-ownership-and-halo-samples">Cell ownership and halo samples</h2>
<p>Marching cell <code>(column, row)</code> reads samples <code>(column, row)</code>, <code>(column + 1, row)</code>,
<code>(column + 1, row + 1)</code> and <code>(column, row + 1)</code>. Own the cell by its lower-left sample,
using the <strong>half-open</strong> local rectangle <code>ownedCells = [minColumn, minRow, maxColumn, maxRow]</code>.
The upper bounds are excluded. A cell on a core's right/top boundary belongs to the next
core. The owning core must include the right/top samples in its halo. Extra halo cells
emit nothing. Partition the domain's <code>(domainWidth - 1) × (domainHeight - 1)</code> marching cells
exactly once. Overlap or gaps between ownership rectangles are the caller's responsibility;
contributors process tiles independently and do not inspect other tiles on the CPU.</p>
<p>For a domain of 9 × 7 samples, four tiles without lower/left padding can be configured as:</p>
<table>
<thead>
<tr>
<th>Tile grid</th>
<th>Domain sample offset</th>
<th>Owned local cells</th>
</tr>
</thead>
<tbody>
<tr>
<td>5 × 4</td>
<td>(0, 0)</td>
<td>[0, 0, 4, 3)</td>
</tr>
<tr>
<td>5 × 4</td>
<td>(4, 0)</td>
<td>[0, 0, 4, 3)</td>
</tr>
<tr>
<td>5 × 4</td>
<td>(0, 3)</td>
<td>[0, 0, 4, 3)</td>
</tr>
<tr>
<td>5 × 4</td>
<td>(4, 3)</td>
<td>[0, 0, 4, 3)</td>
</tr>
</tbody>
</table>
<p>For example the top-right tile has:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> tile</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  domainWidth: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  domainHeight: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">7</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  columnOffset: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  rowOffset: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ownedCells: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">as</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span></code></pre>
<p>Lower/left padding is also allowed: move <code>columnOffset</code>/<code>rowOffset</code> to the first halo sample
and increase the ownership lower bounds accordingly. Do not clamp or duplicate out-of-domain
samples as if they were real samples. No-data and <code>mask</code> apply to all four samples of an owned
cell. An empty ownership rectangle is valid and publishes zero counts and zero overflow.</p>
<p><code>grid</code> is the <strong>tile</strong> grid. Build <code>getGPUIsolinesParameterValues(settings, domainGrid)</code> using
the <strong>domain</strong> dimensions and extent, then reuse identical parameters, active levels and
<code>localOrigin</code> across every tile. The packer chooses shared float32-exact integer sample anchors
near an explicit origin. Tile offsets are added and these anchors are subtracted in integer
arithmetic <strong>before</strong> conversion to float32 and interpolation. This preserves small relative
coordinates even when a domain sample offset exceeds 2^24. Seam crossings therefore use
identical samples and the same expression, including at a four-tile corner. Packing a separate
extent or origin for each tile loses this guarantee.</p>
<p>The eight-word parameter layout is now <code>[levelCount, anchorX, anchorY, cellWidth, cellHeight, columnAnchor, rowAnchor, 0]</code>. The previously reserved words 5 and 6 hold the sample anchors;
<code>anchorX</code> and <code>anchorY</code> are their world positions relative to <code>localOrigin</code>, calculated in
float64 by the packer. Without an explicit origin the anchors remain zero, so legacy manually
packed parameters with zero reserved words retain their interpretation. Anchors are rounded
from in-domain coordinates to float32-exact integers; at very large indices the anchor may be
several samples from the requested origin. Rounding follows the small distances from that
anchor and its world residual, rather than the absolute domain index. For example a 2×2 tile
at column offset 16,777,217 in a 16,777,220×2 domain, with unit samples and shared origin
<code>[16,777,218, 0]</code>, places the equality corner at <code>(0.5, 0.5)</code> on both incident edges.</p>
<p>Only owned cells contribute to <code>status.requiredCount</code>, <code>count</code> and <code>overflow</code>. A small segment
capacity publishes a deterministic prefix and reports overflow; it suppresses stitched output.
Polyline vertex overflow keeps complete fragments and sets <code>polylineOverflow</code>, as in the
single-raster contract. Raw equality-at-level zero-length bridges are retained for topology.</p>
<h2 id="joining-fragments">Joining fragments</h2>
<p>With <code>tile</code>, <code>segmentEdges</code> contains <strong>domain</strong> raster edge IDs, independent of halo layout:</p>
<ul>
<li>Horizontal edge <code>(column, row) → (column + 1, row)</code>:
<code>row * (domainWidth - 1) + column</code>.</li>
<li>Vertical edge <code>(column, row) → (column, row + 1)</code>:
<code>domainHeight * (domainWidth - 1) + row * domainWidth + column</code>.</li>
</ul>
<p>Match an end edge to a start edge with the same <code>segmentLevels</code> index to join fragments. The
oriented segments keep high values to their left. No coordinate snapping or geometric clipping
at tile boundaries is needed. Optional <code>polylines</code> stitch each owned rectangle independently;
this API does not assemble one cross-tile polyline buffer. Keep zero-length raw bridges when
joining by edges: although they have no geometric length, they connect different raster edges.</p>
<p>Tile dimensions, offsets, ownership, capacities and optional views are compile-time. Sample
values, masks, levels and placement parameters are per-frame. Construction validates integer
bounds, all four samples of every owned cell, and domain samples ≤ 2,147,483,647 so two edge
families fit in uint32. The maximum possible segment requirement
<code>ownedCellCount * levels.length * 2</code> must fit uint32. GPU scratch bindings are checked against
the device's storage-binding limit before command nodes are created. CPU arithmetic remains
float64 with one output rounding; its documented near-tie differences still apply.</p>
<h2 id="external-vector-overlays">External vector overlays</h2>
<p>The output is in the domain extent's coordinate frame, relative to the shared <code>localOrigin</code>.
Subtract that origin from external polygon/line vertices before upload. Use a common planar
projection; this is not a geographic intersection or reprojection API. Row zero is at <code>minY</code>
and rows advance toward larger y. The existing grid-affine/north-up limitation remains: reverse
north-up sample rows (and masks) before contouring, rather than mirroring only the contours.</p>
<p>Use existing <code>GPUSegmentIntersection</code> for line/polygon-boundary intersections, or existing
line noding/clipping contributors for further overlay steps. Contour segments are packed
<code>float32x4</code>; their buffer can also be viewed as <code>float32x2</code> positions of two-vertex lines.
Write <code>lineOffsets[index] = 2 * min(index, count)</code> for all <code>segmentCapacity + 1</code> offsets on
the GPU. This makes every unused line empty and avoids exposing dirty output padding. Then
pass the positions and offsets as <code>left: {kind: 'lines', positions, lineOffsets}</code> and an external
polygon geometry as <code>right</code>. Overlay <code>leftRows</code> are start-vertex rows, so <code>leftRows / 2</code> recovers
the owning contour segment and its level; <code>rightRows</code> identify external boundary segments.
Check contour overflow and overlay overflow separately. Boundary intersections do not themselves
classify or clip the contour portions inside a polygon.</p>
<h2 id="oracle-and-acceptance">Oracle and acceptance</h2>
<p><code>test/oracles/isolines-tiles.py</code> generates the committed fixture from whole-domain
<code>skimage.measure.find_contours</code> and Shapely line/polygon-boundary intersection. The fixture
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
Headless execution must be run by the landing lead.</p>`};export{e as default};