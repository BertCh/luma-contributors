var e={slug:`api/gpu-raster/expansion`,title:`Raster, terrain and geometry expansion`,html:`<h1 id="raster-terrain-and-geometry-expansion">Raster, terrain and geometry expansion</h1>
<p>These experimental command-graph contributors keep inputs and results on the GPU. They add
bounded analytical operations and explicit completeness evidence. They have small independent
reference fixtures; they do not yet establish a speed advantage or large-domain production parity.
All outputs are caller-owned, packed graph views. Input/output buffers must be distinct. Import
raster contributors from <code>@luma-contributors/analysis/raster-analysis</code>, terrain contributors from
<code>@luma-contributors/analysis/gpu-terrain</code>, and hull/mesh contributors from
<code>@luma-contributors/analysis/gpu-spatial-analysis</code>.</p>
<p><code>GPURasterRankFilter</code>, <code>GPURasterFillNoData</code> and <code>GPURasterReprojection</code> accept the published
<code>GPURasterBand</code>: either packed buffer storage or a selected channel of a single-mip, single-layer
2D <code>r32float</code>/<code>rgba32float</code>, <code>r32uint</code>/<code>rgba32uint</code>, or <code>r32sint</code>/<code>rgba32sint</code> texture matching
the band's scalar format. Texture extents must equal the input grid (the source grid for
reprojection). Channels are zero-based; single-channel textures require channel zero. Texture
handles must include <code>Texture.SAMPLE</code> usage; <code>Texture.COPY_DST</code> alone is insufficient. Supplied
<code>input.scale</code> and <code>input.offset</code> must remain finite when converted to float32.
The published <code>GPURasterTextureToBuffer</code> adds one graph conversion pass, compares nodata in the
raw selected channel, combines it with the band's optional <code>validity</code> mask, and applies scale
and offset once. These continuous-value operators produce calibrated float32 values and uint32
validity; integer category/label operators retain their separate exact-integer input contracts.
Texture shape, format, channel and mask are validated at construction; graph ownership is checked
before any conversion scratch or nodes are created. Updating texture contents or mask flags and
resubmitting the compiled graph recomputes the result.
Input band metadata and its storage descriptor are captured at construction, including channel,
calibration, nodata and view/mask identities. Declared fields are read by property access, including
inherited and non-enumerable values on both the band and its storage descriptor. Mutating the caller's
descriptors does not change the measurement read by a wired graph; update the captured GPU contents
or rebuild the contributor.
The surface-preparation recipe validates the texture, target graph and caller-provided output
boundary, including each supplied view's format, packing and exact row count, before allocating
default outputs. A rejected texture or invalid output can be corrected and retried on the same
graph with the same recipe ID.</p>
<h2 id="operations-and-contracts">Operations and contracts</h2>
<table>
<thead>
<tr>
<th>Proposal</th>
<th>Public API</th>
<th>Contract and current boundary</th>
</tr>
</thead>
<tbody>
<tr>
<td>C04</td>
<td><code>GPURasterFractionalCoverage</code>, <code>GPURasterFractionalZonalStatistics</code></td>
<td>Triangulates polygons including holes, clips every triangle to affine raster cells, and emits compact <code>(zoneId, cellId, fraction)</code> tuples. Overlapping features retain separate tuples. Coverage candidates are explicitly capped at 4,194,304 zone/cell pairs; positive pairs are compacted in parallel (flag, scan, scatter) in zone-major, cell-ascending order. Weighted aggregation distinguishes extensive cell totals from density/intensive area integrals.</td>
</tr>
<tr>
<td>C05</td>
<td><code>GPURasterFillNoData</code></td>
<td>Disk search of original valid donors, with nearest or all-donor inverse-distance interpolation. Radius 1–256 pixels. Preserves observed calibrated values; returns filled mask, validity and the remaining invalid count in <code>status.invalidCount</code>. The IDW method is <strong>not</strong> GDAL's four-quadrant method.</td>
</tr>
<tr>
<td>C06</td>
<td><code>GPURasterRankFilter</code></td>
<td>Square footprints with radius 0–8. Quantile selects <code>min(floor(q*n),n-1)</code> (upper middle for an even median). Clamp or nodata borders; strict or ignored missing neighbors. Invalid centers stay invalid.</td>
</tr>
<tr>
<td>C07</td>
<td><code>GPUTerrainFlowDistance</code></td>
<td>D8 paths stop at streams or terminal cells. Defaults to planar affine distance; explicit nonnegative transition costs support other units. Cycles, unresolved directions/stream masks, numerical overflow and budget exhaustion remain unresolved.</td>
</tr>
<tr>
<td>C08</td>
<td><code>GPUTerrainStreamNetwork</code></td>
<td>Compact polylines between headwaters, junctions and outlets, with source/target raster cell IDs, vertex offsets, affine cell-center positions and lengths. Valid D8 stream paths only. Reach order is source-cell order. Stream order can be gathered from the source-cell IDs.</td>
</tr>
<tr>
<td>C10</td>
<td><code>GPUConcaveHull</code>, <code>GPUGroupConcaveHull</code></td>
<td>Alpha-complex area hulls retain Delaunay triangles by maximum circumradius. Directed boundary edges preserve holes and disconnected components. This is radius-based alpha geometry, not GEOS ratio-based concave hull. Collinear/isolated points emit no area.</td>
</tr>
<tr>
<td>C21</td>
<td><code>GPURasterWalkingCost</code></td>
<td>Directed eight-neighbor movement from elevation and friction. Tobler time or explicitly supplied GRASS-style energy coefficients; nonnegative finite edges only. Whole-grid Jacobi rounds are gated on the GPU and stop once a round changes nothing; the current costs are always published and <code>status.converged</code> tells whether they are final. Required <code>parameters</code> (packed by <code>getGPURasterWalkingCostParameterValues</code>, 16 float32 values) carry the affine, row edges, <code>frictionWeight</code> and GRASS coefficients per frame, so a changed georeference does not recompile. Optional <code>cellSizeMode</code> (<code>'uniform'</code> default, <code>'web-mercator'</code>, <code>'geographic'</code>) converts steps and slopes to WGS84 ground meters per row.</td>
</tr>
<tr>
<td>C22</td>
<td><code>GPURasterReconstruction</code>, <code>GPURasterFillHoles</code></td>
<td>Grayscale geodesic dilation/erosion under a mask; hole filling reconstructs background from the boundary. <code>connectivity</code> is 4 or 8: reconstruction defaults to 8 (the 3 x 3 square, the default footprint of <code>skimage.morphology.reconstruction</code> and of the published <code>GPURasterDilation</code>; it was 4 before 2026-11-10), hole filling defaults to a 4-connected background (the rule of <code>scipy.ndimage.binary_fill_holes</code>). The current values are always published; <code>status.converged=0</code> marks an unfinished run, whose hole mask reports background the boundary flood has not reached as holes.</td>
</tr>
<tr>
<td>C23</td>
<td><code>GPURasterReprojection</code></td>
<td>Generates destination centers, applies an explicit inverse <code>CompiledProjection</code>, inverts a nonsingular source affine and samples nearest/bilinear values. Supports rotated grids. One caller-selected source tile; no tile fetching. Float32 projection path only. Grids in the same CRS are differenced at their origins in float64, so large projected origins (UTM northings) keep sub-cell accuracy; with an inverse projection, keep <code>destinationGrid</code> in a local frame and set the projection's <code>destinationOrigin</code> near the source grid origin.</td>
</tr>
<tr>
<td>C24</td>
<td><code>GPUConstrainedDelaunayTriangulation</code></td>
<td>Existing shell/hole triangulation followed by bounded constrained Lawson flips. Boundary edges remain fixed. Per-polygon triangulation validity and Delaunay convergence are separate; exhaustion retains a valid initial/refined mesh with <code>output.status.converged=0</code>.</td>
</tr>
<tr>
<td>C25</td>
<td><code>GPURasterWatershedSegmentation</code></td>
<td>Default <code>method: 'fifo'</code> preserves marker priority flooding by path maximum elevation and discovery age, within 16,384 cells. Explicit <code>'minimum-spanning-forest'</code> uses parallel rooted Boruvka contraction with linear scratch and device-sized domains. Both support four/eight connectivity and fixed markers; their plateau tie policies differ. The FIFO order is by path maximum, not by the cell's own value as in <code>skimage.segmentation.watershed</code>, so the two can differ where fronts from higher markers meet on lower ground.</td>
</tr>
<tr>
<td>C26</td>
<td><code>GPURasterSkeleton</code>, <code>GPURasterMedialAxis</code></td>
<td>Zhang–Suen thinning is separate from maximal Euclidean disk centers. The thinning is the 1984 algorithm unchanged, so an isolated 2 x 2 block is deleted completely and two-pixel-thick strokes can vanish. Medial axis uses an exact distance transform and an integer lattice-step containment test, with linear storage and dimensions up to 32,768 subject to device storage limits. Exterior pixels are background; radii remain in pixel units. Even-width plateaus can retain multiple centers.</td>
</tr>
</tbody>
</table>
<h3 id="watershed-methods">Watershed methods</h3>
<p>The default FIFO method orders pending cells by the maximum elevation along their flood path,
then by discovery age. Seeds enter in row-major order. <code>maximumSteps</code> bounds the number of popped
cells; it applies only to this method. Its binary heap uses O(N log N) queue work and O(N) storage.</p>
<p>For larger complete domains, select <code>method: 'minimum-spanning-forest'</code>. Grid edges are ordered by
maximum endpoint elevation, then by canonical edge ID: four times the row-major source cell plus
the direction index (right, down-left, down, down-right). Four-connectivity omits diagonal edges.
Signed zero elevations compare equally. Parallel contraction keeps each marker fixed and excludes
invalid pixels and reserved marker words. Unseeded connected components retain label zero.
This static edge order can assign plateaus differently from FIFO; it is an explicit alternative,
not FIFO or scikit-image output parity.</p>
<p>Neither method reproduces <code>skimage.segmentation.watershed</code> in general. The FIFO method orders by
the maximum elevation along the flood path; scikit-image orders its queue by the cell's own value.
For the row <code>5, 2, 2, 2, 2, 2, 5</code> with markers on the two ends the FIFO method labels
<code>1 1 1 1 2 2 2</code> (both fronts hold priority 5 and advance in turn) and scikit-image labels
<code>1 1 1 1 1 1 2</code>. The committed scikit-image watershed fixture is a case where the two orders
agree; it checks the labelling of that case, not parity of the queue order.</p>
<p>The forest method accepts <code>maximumIterations</code> from 1 through 32, defaulting to
<code>ceil(log2(width * height)) + 1</code>. It requires fewer than 2^30 cells and scratch buffers that fit
the device's storage binding limits. Supplying <code>maximumSteps</code> with this method, or
<code>maximumIterations</code> with FIFO, is rejected. Exhaustion reports <code>converged = 0</code> and labels valid
non-marker cells unresolved (<code>0xfffffffe</code>), rather than publishing a partial partition as complete.
Neither method assembles tiles; independent tile segmentation does not preserve global ownership.</p>
<h3 id="exact-medial-disk-containment">Exact medial disk containment</h3>
<p><code>GPURasterMedialAxis</code> composes the exact banded distance transform with one candidate check per
pixel. Given a nearest background pixel, any disk containing the current pixel's disk must lie on
the ray directly away from that background pixel. The first lattice point along that ray suffices
to decide containment using integer squared distances. This removes both dense all-pairs searches.</p>
<p>Containment follows the mathematical Euclidean-disk definition, correcting previous float32
comparisons at tangent disks. Output radii are correctly rounded float32 values. The grid affine
does not rescale pixel-unit distances. This operation does not perform thinning and does not
promise a one-pixel-wide or topology-preserving skeleton. Storage is linear in cell count, but the
distance-transform scratch still needs to fit device limits; the dimension bound alone is not a
memory or throughput guarantee.</p>
<h3 id="fractional-quantities">Fractional quantities</h3>
<p><code>GPURasterFractionalCoverage</code> accepts <code>grid</code> and <code>zones</code> using the shared four-level polygon
layout. Allocate <code>zoneIds</code>, <code>cellIds</code>, and <code>fractions</code> with a common tuple capacity, plus scalar
<code>status.count</code>, <code>status.requiredCount</code>, <code>status.overflow</code>, optional <code>status.candidateOverflow</code>, and
per-feature <code>zoneValidity</code>. A failed polygon triangulation invalidates its feature; that feature
contributes no tuples. On final tuple-capacity overflow, <code>requiredCount</code> counts positive tuples:
increase the tuple capacity to that count and rerun. On indexed candidate exhaustion,
<code>candidateOverflow = 1</code>, <code>requiredCount = 0xffffffff</code>, and <code>overflow</code> may remain zero. Increase
the candidate budget and rerun <strong>before interpreting requiredCount as a tuple allocation size</strong>.
The sentinel is not a request for a 0xffffffff-row tuple buffer. Keep the two overflow flags separate.
A vertex whose pixel coordinate is
within <code>min(1e-5, 4 * 2^-23 * max(|k|, 1))</code> pixels of a cell boundary is placed on that boundary:
integers for area grids, half-integers for point grids. This removes boundary slivers only when
the inverse-affine error is within that cap. Decimal cells can exceed the cap even with zero
origin: at width 0.3, <code>[39.3, 39.6]</code> can leave about 1.53e-5 coverage in column 130 beside the
intended column 131. Every positive sliver occupies a tuple and can exhaust an otherwise exact
tuple capacity; allocate spare tuple capacity, checking candidateOverflow before requiredCount/overflow. Device division can
change these tiny supports. The exact identity
affine <code>[1, 0, 0, 0, 1, 0]</code> bypasses snapping, preserving all representable subpixel coordinates.
For other affines, snapping compares displacement from the nearest lattice point and moves a
vertex by at most 1e-5 pixel per axis, independently of pixel index. Clipping uses cell-local
coordinates. The snap cannot
absorb the rounding of <code>position - origin</code>: keep the grid origin near the geometry. The required <code>parameters</code> view
(<code>getGPURasterFractionalCoverageParameterValues(grid)</code>, 8 float32 values) carries the affine, so a
moved georeference of the same size does not recompile.
The website coverage runs distinguish pending readback, incomplete polygon coverage and a valid
empty intersection. Pending/incomplete readouts show that state and clear the highlighted support;
only complete coverage may display zero cells and zero area.</p>
<p><code>GPURasterFractionalZonalStatistics</code> consumes those views, a calibrated band, <code>zoneCount</code>, and
<code>quantity</code>. It returns <code>totals</code>, coverage-weighted raw-value <code>means</code>, <code>coveredArea</code>, <code>validArea</code>,
and <code>validity</code>. Extensive cell totals use <code>value * fraction</code>; density and intensive integrals use
<code>value * fraction * cellArea</code>. Missing values reduce observed area and remain unknown.
Means divide the value/fraction sum by observed fractions. Empty or wholly unobserved zones have
invalid totals/means; covered/observed areas remain available. Incomplete coverage invalidates all
aggregates. The default <code>cellSizeMode: 'uniform'</code> uses source-coordinate units squared.
Scalar, categorical and distribution consumers require <code>count &#x3C;= tuple capacity</code>,
<code>requiredCount == count</code>, zero <code>overflow</code>, zero optional <code>candidateOverflow</code>, and nonzero
zone validity. Prepared-coverage operation ports preserve optional candidateOverflow. Missing
candidate evidence is accepted only when that optional input is not supplied. A clean subsequent
submission restores usable results. Means divide by the actual positive fraction sum, including
normal float32 sums below 1e-30; only zero support uses a safe denominator.
<code>'geographic'</code> (longitude/latitude degrees) and <code>'web-mercator'</code> (EPSG:3857 meters) use WGS84
ground square meters at each row center, with the same approximation as the fractional categorical
and distribution contributors. Both modes require an axis-aligned grid. Alternatively, supply
positive finite <code>cellAreas</code>, exactly one per raster cell, in caller-declared area units; this requires
uniform mode. An invalid explicit area suppresses the affected zone's result (NaN areas and
totals/means, validity zero). <code>coveredArea</code> and <code>validArea</code> use the selected area units; density
and intensive <code>totals</code> have input units times area, while extensive totals retain input units.
Means remain coverage-fraction weighted in input units for every area model.
The default planar determinant is that of <code>grid.affine</code>, fixed at compile time, unless
the optional <code>parameters</code> prop is given: pass the same per-frame georeference view as the coverage
class, and the cell area is read from it on every encoding. Do so whenever the coverage
<code>parameters</code> are rewritten for another cell size; otherwise the fractions follow the new
georeference while <code>coveredArea</code>, <code>validArea</code> and density <code>totals</code> keep the compiled cell area.
<code>GPURasterFractionalCategoricalStatistics</code> and <code>GPURasterFractionalZonalDistribution</code> take the
same prop.
Ground-mode row centers also follow the origin y and signed row step in <code>parameters</code>.
Explicit <code>cellAreas</code> override the affine area and must be updated by the caller when needed.
With explicit areas the unused affine determinant need not fit float32. Ground modes also check
the per-frame cross-axis words: a repacked non-axis-aligned affine suppresses all scalar results
(NaN aggregates and validity zero). Each zone's published areas and, when observed, total/mean
must fit finite float32; an unrepresentable aggregate suppresses that zone, preserving usable
zones. Empty or wholly missing zones retain their finite area outputs and validity zero.</p>
<p>{/* parameters: GPURasterFractionalCoverage generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurasterfractionalcoverage-parameters"><code>GPURasterFractionalCoverage</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterFractionalCoverageParameterValues(grid)</code>, writes <code>GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): Grid whose affine is packed; it must be invertible.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h3 id="preparing-continuous-surfaces">Preparing continuous surfaces</h3>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> preparation</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterFillNoData</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  input: elevationBand,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumDistance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  method: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'nearest'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {values: preparedValues, validity: preparedValidity, filled, status: {invalidCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(preparation);</span></span></code></pre>
<p>All-donor interpolation never uses a newly filled cell as a donor. A tile requires a halo at least
<code>maximumDistance</code> wide to match the full-domain result on its core. Rank filtering requires its
radius as a halo. These contributors do not assemble tiles or suppress halo counts automatically.
Filling nodata and hydrological depression filling remain different operations. The surface
preparation recipe exposes provenance through both stages.</p>
<h3 id="drainage-outputs">Drainage outputs</h3>
<p><code>GPUTerrainFlowDistance</code> returns dense <code>distance</code>, <code>validity</code>, and <code>terminalCells</code>.
<code>0xffffffff</code> is invalid; <code>0xfffffffe</code> is unresolved. <code>maximumSteps</code> defaults to the cell count.
A resolved stream cell has distance zero. If no streams are supplied, outlets/pits stop the path.
Distances use the full affine's planar lengths; geographic and Mercator physical corrections are
not implicit.</p>
<p><code>GPUTerrainStreamNetwork</code> takes <code>flowDirections</code>, <code>streams</code> and the same grid. Allocate
<code>sourceCells</code>, <code>targetCells</code>, <code>lengths</code> to a reach capacity, <code>vertexOffsets</code> to capacity plus one,
and <code>positions</code> to a separate vertex capacity. It returns scalar reach/vertex counts, <code>status.overflow</code> and
unresolved-cell count. Junction cells appear in every incident polyline. Isolated stream cells have
no line reach. Cycles and paths crossing unknown directions or stream status are omitted with
unresolved evidence. Cross-tile drainage and branching D-infinity/MFD are not implied.</p>
<h3 id="group-outlines-and-recipes">Group outlines and recipes</h3>
<p><code>GPUGroupConcaveHull</code> uses one group-aware Delaunay build and preserves original point rows without
readback. Its default fixed layout reserves <code>T = max(0, 2*pointCount-5)</code> triangle slots and <code>3*T</code>
boundary slots per group. With <code>layout: 'packed'</code>, those reservations cover all groups together;
retained triangles and boundary edges form separate compact group-major lists. The four views in
<code>output.groups</code> (<code>triangleOffsets</code>, <code>triangleCounts</code>, <code>edgeOffsets</code>, <code>edgeCounts</code>) each have one row
per group. Packed <code>status.count</code> and <code>requiredCount</code> count boundary edges and <code>overflow</code> is zero
with the required full reservation. <code>status.converged</code> has one row per group; incomplete groups
emit no geometry. Validity marks the used compact prefix, with zero padding. The triangle
reservation is bounded by 715,827,882 rows for uint32 addressing and by device binding sizes.</p>
<p><code>addClusterAndOutlineRecipe</code> accepts <code>outline: {kind: 'alpha', maximumCircumradius}</code>. The result's
<code>outlineType</code> identifies the selected mode. Alpha geometry lives in <code>concaveHull</code>; <code>areas</code>,
<code>perimeters</code>, and <code>centroids</code> describe its retained area. In alpha mode the legacy single-ring
<code>hullOffsets</code> and <code>hullCounts</code> are empty, because one ring cannot represent holes and disconnected
components. Default convex behavior is unchanged. Empty area has zero area/perimeter and undefined
centroid. Nonconverged groups report invalid measures.</p>
<h3 id="gpurasterwalkingcost-georeference"><code>GPURasterWalkingCost</code> georeference</h3>
<p>Walking cost reads the grid's linear affine terms, its row edges, <code>frictionWeight</code> and the GRASS
coefficients from <code>parameters</code>, a <code>GPUParameterBuffer</code> view written with
<code>getGPURasterWalkingCostParameterValues(settings, grid)</code>. Rewriting it between encodings moves the
same compiled graph to a new georeference, cell size or friction weight without recompiling. Only
<code>model.kind</code> (Tobler or GRASS) and <code>cellSizeMode</code> change the kernels. <code>cellSizeMode: 'uniform'</code> (the
default) treats affine units as meters; <code>'geographic'</code> (degrees) and <code>'web-mercator'</code> (EPSG:3857
meters) scale each step on the WGS84 ellipsoid at the latitude between the two cell centres, so
slopes divide meters of rise by meters of run.</p>
<p>Pack <code>parameters</code> for the same <code>model.kind</code> the contributor was compiled with. The packed slots
8 to 12 identify the model (all zero for Tobler, a negative threshold for GRASS), and the kernel
checks them on the GPU: parameters packed for the other model move nothing (sources cost 0, every
other cell stays <code>+Infinity</code>) and set the optional <code>status.invalidCount</code> to 1. Before this check, a
GRASS kernel given Tobler's zero coefficients charged only friction, silently.</p>
<p>{/* parameters: GPURasterWalkingCost generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurasterwalkingcost-parameters"><code>GPURasterWalkingCost</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterWalkingCostParameterValues(settings, grid)</code>, writes <code>GPU_RASTER_WALKING_COST_PARAMETER_LENGTH</code> (16) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): Grid whose georeference is packed. Its width and height must match the compiled one.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterWalkingCostParameterValues</code>:</p>
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
<td><code>frictionWeight</code> (optional)</td>
<td><code>number</code></td>
<td>Coefficient multiplying additive average-friction * ground-distance cost. Default 1.</td>
</tr>
<tr>
<td><code>model</code> (optional)</td>
<td><code>GPURasterWalkingCostModel</code></td>
<td>Movement model; only the GRASS coefficients and threshold are packed. Default Tobler.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpurasterreprojection"><code>GPURasterReprojection</code></h3>
<p>The sampling transform is read from a float32 parameter view packed by
<code>getGPURasterReprojectionParameterValues({sourceGrid, destinationGrid, inverseProjection})</code>.
Write new words into a <code>GPUParameterBuffer</code> between encodings to change origins, rotation, shear
or resolution. Shapes, CRS, pixel interpretation and the inverse projection stay fixed. Affines
whose packed float32 axes are singular are rejected before encoding. The packer
composes the entire destination-to-source transform in float64 for same-CRS sampling, including
origin differences and area/point sample shifts. Evaluation is centered on a destination pixel
near source support before packing, preserving cancellation between large scales and translations.
Identical grids therefore
preserve sample coordinates on wide sheared strips within float32 sampling-index precision.
With an explicit projection, the source inverse linear affine is computed in float64 before packing;
the packer rejects a source inverse whose packed coefficients form a singular matrix.
Double-single outputs subtract a split
source origin. Projection inputs should still use a local frame; ill-conditioned source affines can
amplify float32 projection error, so choose a source lattice that resolves the required cells.
Without <code>parameters</code> the contributor owns a buffer containing the initial affines. Call <code>destroy()</code>
after destroying its compiled graph, including for same-CRS sampling. Tile reprojection forwards
this same parameter view to its sampler; source tile bounds stay fixed in source pixel coordinates.</p>
<p>The 20-word packed layout is now <strong>version 3</strong>, exposed as
<code>GPU_RASTER_REPROJECTION_PARAMETER_VERSION</code>, with the version stored in word 17. Repack manual
parameter buffers using the current packer. A different version produces NaN values and validity
zero. Versions 1 (separate axes/determinant) and 2 (unanchored composition) must be repacked.</p>
<table>
<thead>
<tr>
<th>Words</th>
<th>Same CRS (no inverse projection)</th>
<th>Explicit inverse projection</th>
</tr>
</thead>
<tbody>
<tr>
<td>0–3</td>
<td>Composed sample matrix <code>[a, b, d, e]</code></td>
<td>Destination world matrix <code>[a, b, d, e]</code></td>
</tr>
<tr>
<td>4–5</td>
<td>Source sample position at the destination anchor, including pixel shifts</td>
<td>Destination world origin</td>
</tr>
<tr>
<td>6–9</td>
<td>Reserved zeros</td>
<td>Source inverse matrix <code>[a, b, d, e]</code></td>
</tr>
<tr>
<td>10</td>
<td>Reserved zero</td>
<td>Reserved zero</td>
</tr>
<tr>
<td>11–12</td>
<td>Zeros</td>
<td>Local-f32 projection-origin minus source-origin shift; zeros for double-single</td>
</tr>
<tr>
<td>13–16</td>
<td>Zeros</td>
<td>Source origin <code>[xHigh, xLow, yHigh, yLow]</code> for double-single; zeros for local-f32</td>
</tr>
<tr>
<td>17</td>
<td>Version 3</td>
<td>Version 3</td>
</tr>
<tr>
<td>18–19</td>
<td>Destination pixel anchor <code>[column, row]</code></td>
<td>Reserved zeros</td>
</tr>
</tbody>
</table>
<p>Without an inverse projection, subtract words 18–19 from integer destination column/row,
multiply by words 0–3, and add words 4–5 to obtain source sample column/row directly.
Area centers and the source half-cell shift
are already included. With a projection, generate destination centers as before, project, subtract
the source origin, apply words 6–9, and subtract 0.5 for an area-interpreted source grid.</p>
<p>Nearest sampling compares the coordinate's fractional part to 0.5 before incrementing its floor,
on both axes. Exact ties select the higher index. This avoids rounding a coordinate just below
the half-cell boundary up when adding 0.5 in float32.</p>
<p>{/* parameters: GPURasterReprojection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurasterreprojection-parameters"><code>GPURasterReprojection</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterReprojectionParameterValues(settings)</code>, writes <code>GPU_RASTER_REPROJECTION_PARAMETER_LENGTH</code> (20) <code>float32</code> values.</li>
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
<p>{/* /parameters */}</p>
<h3 id="projection-ownership-and-incomplete-iteration">Projection ownership and incomplete iteration</h3>
<p><code>GPURasterReprojection</code> accepts an inverse projection already compiled for destination-to-source
coordinates. Without one, CRS metadata must match. Source pixels outside the provided support and
invalid projection domains remain nodata; strict bilinear sampling requires every participating
sample, while renormalization uses surviving support. A projection-bearing contributor registers
once and owns its projection parameter buffer: destroy it after destroying its compiled graph.
Renormalization divides by the actual positive weight, so a lone value of 7 remains 7 even at
normal float32 weight 1e-32. Zero support uses a safe denominator and remains invalid.</p>
<p>Reconstruction, fill holes, thinning and walking perform an additional check round to establish the
final fixed point. The rounds run as GPU-gated iterations: once a round changes nothing, later rounds
dispatch no work, so a generous <code>maximumIterations</code> costs only command nodes (see the contributor's
<code>estimatedCommandNodeCount</code>, checked against the 32,768-node budget). The node budget fixes the largest accepted <code>maximumIterations</code>: 26,196 for walking cost, 26,209 for reconstruction, 26,207 for fill holes and 10,920 for skeletonization; one more is rejected at construction. The current values are
<strong>always published</strong>, whether or not the run converged. <code>output.status.converged</code> is 1 only when a
round changed nothing before the limit. Optional <code>output.status.iterationLimitReached</code> (1 when the
limit stopped an unconverged run) and <code>output.status.iterationCount</code> (rounds that ran, including the
verifying round; thinning counts full two-phase iterations) are written when supplied. An unconverged
walking cost holds <code>+Infinity</code> for cells the front has not reached and upper bounds elsewhere, an
unconverged reconstruction holds the partially spread marker, and an unconverged thinning holds a
partially thinned mask. Earlier releases published all-NaN or <code>0xfffffffe</code> output instead; callers
that relied on that must now test <code>converged</code>. Watershed segmentation leaves original
markers identifiable but publishes unresolved non-marker pixels when its step budget is exhausted.</p>
<p>For reconstruction and fill holes, <code>maximumIterations</code> bounds the published propagation distance
in neighbor steps. The extra verification pass only checks whether the budgeted result is a fixed
point; it never advances that result. <code>iterationCount</code> includes this check and can therefore be
<code>maximumIterations + 1</code>. With marker <code>[1, 0, 0, 0]</code>, mask <code>[1, 1, 1, 1]</code> and a one-round dilation
budget, reconstruction publishes <code>[1, 1, 0, 0]</code>, count 2, <code>converged=0</code> and
<code>iterationLimitReached=1</code>. The CPU reconstruction and hole-fill twins obey the same budget and
default of 64 rounds; they drop only the GPU command-node cap. For an unfinished hole fill,
background farther than the budget from the boundary is still marked as a hole.
The CPU hole-fill queue stops at that frontier and only inspects immediate neighbors there to
verify completeness; it does not traverse the remaining connected component. Construction captures
the grid, iteration budget and nested port names in both operation forms and contributors.
The website's fill-hole and reconstruction previews return convergence, label unfinished propagation,
and withhold final fill/cut counts and their derived overlays until convergence.
The showcase's missing-observations scene likewise omits the reconstruction bar from its selected-cell
chart until convergence, matching its map and readout even when the partial value is finite.</p>
<p>Reconstruction band inputs in both operation forms use the first <code>grid.width * grid.height</code> samples
and validity flags. Each supplied values/validity view must contain at least that many entries; spare
capacity is ignored. CPU calibration borrows that logical prefix at the supplied typed-array offset,
so a nonzero offset in either values or validity remains meaningful. Undersized bands are rejected.</p>
<h2 id="reference-evidence">Reference evidence</h2>
<p><a href="../../../modules/analysis/test/oracles/raster-terrain-expansion.py"><code>raster-terrain-expansion.py</code></a>
records exact package versions and produces committed fixtures using SciPy median/hole filling,
scikit-image reconstruction/watershed, NetworkX path lengths, Shapely intersections/CDT, and a
SciPy-Delaunay/Shapely alpha-complex composition. The matching browser tests execute actual GPU
kernels. Other cases test boundary masks, directed costs, capacities, provenance, cyclic receivers
and forced nonconvergence. The maximal-disk medial axis has a direct SciPy radius reference and
structural center tests; it does not claim direct scikit-image medial-axis parity.</p>
<p>Performance gates remain open: measure compilation, upload, execution, readback and repeated runs
on representative large domains. Bounded priority queues, disk searches, grouped tessellation and
per-path walks are useful complete reference implementations; their existence is not a performance
claim. No hidden CPU analysis, GPU readback or implicit command submission occurs in these APIs.</p>
<p>For explicitly supplied tiles on one shared source lattice, <a href="/luma-contributors/docs/api/gpu-raster/tile-reprojection">GPURasterTileReprojection</a> assembles a bounded mosaic on GPU before sampling. <a href="/luma-contributors/docs/api/gpu-raster/fill-nodata">Nodata filling</a> documents the optional original-donor four-quadrant method.</p>`};export{e as default};