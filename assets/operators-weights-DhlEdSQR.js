var e={slug:`api/gpu-spatial-analysis/operators-weights`,title:`GPU Spatial Analysis: Spatial weights`,html:`<h1 id="spatial-weights">Spatial weights</h1>
<p>A <code>GPUSpatialWeights</code> is the sparse <code>W</code> that spatial statistics are built on, in CSR form: row <code>i</code>
lists the neighbors <code>j</code> of focus row <code>i</code> and the weight <code>w_ij</code> of each. Producers write it on the
GPU, transforms rewrite it, and the statistics below read it, so one statistic works with any
neighborhood definition. Applications can also upload weights built elsewhere.</p>
<p>Every producer guarantees: <code>offsets</code> holds <code>rows + 1</code> non-decreasing entries starting at 0; row <code>i</code>
occupies slots <code>[offsets[i], offsets[i + 1])</code>; neighbor IDs are strictly ascending within a row; a
self join never lists a row as its own neighbor (<code>w_ii = 0</code>); weights are finite and non-negative.
Capacity overflow clamps offsets to <code>neighbors.length</code> and sets an <code>overflow</code> flag on the GPU.</p>
<p>A <code>GPUSpatialWeights</code> is a first-class object, not only buffers. Every producer in this section
(<code>GPUContiguityWeights</code>, <code>GPULatticeWeights</code>, <code>GPUSpatialWeightsTransform</code>,
<code>GPUSpatialWeightsAlgebra</code>, <code>GPUSpatialWeightsTranspose</code>) exposes <code>outputWeights</code>: the caller-owned
views with <code>information</code> (<code>rowCount</code>, <code>transform</code> state <code>'raw' | 'binary' | 'row' | 'double' | 'variance' | 'kernel' | 'symmetric' | 'unknown'</code>, <code>symmetric</code> as <code>true | false | 'unknown'</code> only where
guaranteed, <code>hasSelfWeights</code>, <code>provenance</code> with source, criterion, <code>k</code>, <code>radius</code> and <code>order</code>) and
<code>status</code> (<code>overflow</code> and, where the producer computes it exactly, <code>requiredCount</code>) attached. Pass
<code>outputWeights</code> to consumers: statistics read <code>information.transform</code> (a <code>'row'</code> W is
re-standardized over the neighbors that remain after masking) and gate on both loss flags.
Uploaded weights may omit <code>information</code>, which reads as <code>'unknown'</code>. <code>validateGPUSpatialWeights</code>
checks on the CPU what it can: view formats and lengths, <code>information.rowCount</code>, the declared
transform and symmetry, and the status formats.</p>
<p>The optional <code>spatialContext</code> describes the coordinates and metric used to build the weights;
<code>distances</code> and <code>information.provenance.radius</code> use its length unit. <code>GPUNeighborSearch</code> sets it;
<code>GPUNearestFeatureWeights</code> sets it from the producing join's context passed to the adapter.
Transforms, transpose and algebra propagate it; algebra checks declared input contexts with
<code>assertCompatibleGPUSpatialContexts</code>. Topology-only producers and uploaded weights may omit it.</p>
<p>For weighted statistics, a slot is an active neighbor when its weight is
positive, its neighbor ID is below the row count and it is not the row itself. A zero-weight slot is
listed but is not a neighbor. An island is a row with no active slot, whatever slots it lists.</p>
<p><code>GPUSpatialWeightsMinimumSpanningTree</code> uses topology instead: every listed slot is a link,
including reciprocal zero-weight slots. It ignores the weight value and uses attribute
dissimilarity as edge cost. SKATER and REDCAP forest compositions keep that topology-only rule;
zero-weight links must be removed from the CSR if they should not connect forest components.</p>
<p>Uploaded CSR must satisfy the invariants above; consumers do not re-check them (a binary search
over unsorted neighbors silently gives wrong <code>S1</code>, <code>S2</code> and symmetry counts). Run
<code>GPUSpatialWeightsValidation</code> once after upload, and sort, merge duplicates and drop self slots on
the CPU if it reports violations.</p>
<p>Capacity grow loop: <code>GPUContiguityWeights</code>, <code>GPULatticeWeights</code> and <code>GPUSpatialWeightsAlgebra</code>
write into one caller-owned <code>output: GPUSpatialWeightsOutput</code>, the CSR views plus
<code>status: {overflow, requiredCount?, candidateOverflow?}</code> (<code>overflow</code> required). Read <code>overflow</code> and
<code>requiredCount</code> after a frame; when <code>requiredCount</code> exceeds the capacity, allocate
<code>getGPUSpatialWeightsCapacityPlan({rowCount, capacity, requiredCount}).capacity</code> slots (the required
count plus 25 percent headroom, <code>GPU_SPATIAL_WEIGHTS_CAPACITY_HEADROOM</code>) and rebuild. The plan's
<code>estimatedOutputBytes</code> is the bytes of the CSR views at that capacity
(<code>GPU_SPATIAL_WEIGHTS_SLOT_BYTE_LENGTH</code> per slot, plus <code>GPU_SPATIAL_WEIGHTS_DISTANCE_BYTE_LENGTH</code>
with distances, plus the offsets). The contiguity and lattice producers also expose
<code>getCapacityPlan(observation)</code>, which adds their own transient bytes (from the same constants their
<code>getCommandNodes</code> allocates with; the device-dependent gpgpu sort and scan scratch is not included)
and, for contiguity, the intermediate pair stage. Rows past the point of truncation are empty, so
never use weights whose <code>overflow</code> or <code>candidateOverflow</code> is nonzero.</p>
<p>Keep the complete producer port (for example, <code>contiguity.outputWeights</code>) when wiring weights
into a minimum spanning forest, SKATER, REDCAP or Ward. These consumers accept supplied loss
words even when both are zero, read their contents on the GPU on every submission, and suppress
partitions on either loss cause. Forest component labels and region labels become <code>0xffffffff</code>;
forest edge counts and SKATER/REDCAP region counts become zero/unavailable. Forest compact edge status reports
either upstream cause as <code>candidateOverflow</code>, while its <code>overflow</code> describes final edge capacity
only. SKATER cut gains become NaN. Clearing the flags restores ordinary results without rebuilding
the graph. See the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-statistics#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation">regionalization contract</a>
for the operation and CPU-twin ports.
Forest status is available without compact edges: its packed <code>candidateOverflow</code> word reports
either upstream cause, separately from the attribute <code>invalidCount</code> word. Forest and SKATER lens
captures retain both producer loss flags and refuse a record if either is nonzero. Forest figures
size their edge key spaces and segment captures from the published compact count, including
disconnected graphs, rather than assuming <code>rowCount - 1</code> edges.</p>
<table>
<thead>
<tr>
<th>Producer</th>
<th>Neighborhood</th>
<th>PySAL equivalent</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUNeighborSearch</code></td>
<td>k nearest neighbors or distance band between points</td>
<td><code>KNN</code>, <code>DistanceBand</code>, <code>Kernel</code></td>
</tr>
<tr>
<td><code>GPUContiguityWeights</code></td>
<td>polygons sharing a vertex (queen) or an edge (rook)</td>
<td><code>Queen</code>, <code>Rook</code></td>
</tr>
<tr>
<td><code>GPULatticeWeights</code></td>
<td>regular grid cells</td>
<td><code>lat2W</code></td>
</tr>
<tr>
<td><code>GPUSpatialPredicateJoin</code> with <code>weights</code></td>
<td>any predicate join (intersects, within, dwithin, ...)</td>
<td>-</td>
</tr>
<tr>
<td><code>GPUSpatialWeightsAlgebra</code></td>
<td>union, intersection, difference, higher order, self weights, subgraph, block</td>
<td><code>w_union</code>, <code>higher_order</code>, <code>w_subset</code>, <code>block_weights</code></td>
</tr>
</tbody>
</table>
<h3 id="gpucontiguityweights"><code>GPUContiguityWeights</code></h3>
<p>Join Chicago tract polygons that share a vertex (queen) or an edge (rook) into binary neighbour weights.</p>
<p>::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=result caption="Queen links join 791 Chicago tracts that share a vertex; tracts with no link are islands."}</p>
<p>Polygon contiguity as binary weights. <code>criterion: 'queen'</code> joins features that share at least one
vertex; <code>'rook'</code> joins features that share a ring edge (both endpoints, either direction). Input is
<code>polygons: GPUPolygonGeometryPort</code> (the four-level layout <code>positions</code>, <code>ringOffsets</code>,
<code>polygonOffsets</code>, <code>featureOffsets</code>), with one output row per FEATURE: every ring of every part
counts, holes included, so a multipart feature is the neighbor of anything any of its parts touches,
and parts of one feature are never self neighbors. This is the row space of
<code>GPUZonalStatistics</code> and the points-in-polygons recipe, so points -> counts -> W -> Moran chain on
one feature table without a cast.</p>
<ul>
<li><code>snapTolerance</code> (default 0, exact f32 equality) snaps vertices with
<code>floor((x - origin) / tolerance + 0.5)</code> per axis (from 2^23 steps the float32 quotient is an
integer and is used as it is), where the origin is the first vertex of the data, so the grid
depends on the extent of the data and not on its distance from the coordinate origin. It is the
snap key of <code>GPUCoverageDissolve</code> and <code>GPUCoverageSimplification</code>. This is a grid snap, not a distance test: two vertices closer than the tolerance can
fall in different cells. The extent divided by the tolerance must stay at or below
<code>GPU_CONTIGUITY_WEIGHTS_MAXIMUM_SNAP_CELLS</code> (2^24); pass <code>coordinateExtent</code> to have that checked
at construction.</li>
<li>Exact 64-bit vertex keys, stable radix sorts and run detection; no hashing, so no collisions, and
the output is deterministic and symmetric.</li>
<li>A point shared by <code>m</code> features emits <code>m^2</code> candidate pairs, bounded by <code>pairCapacity</code> (default
<code>GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR</code> = four times the neighbor capacity); exceeding it
sets <code>candidateOverflow</code> only. Overflow reports final neighbor-slot loss only. The producer's
outputWeights always exposes both flags, allocating candidate status in the graph if omitted.</li>
<li><code>output.status.requiredCount</code> is exact unless the pair list overflowed (then a lower bound).
<code>getCapacityPlan({requiredCount, candidateOverflow})</code> returns the next <code>slots</code> and <code>pairs</code>
capacities (pairs keep the default ratio and double after a candidate overflow), so the rebuild
loop converges. <code>outputWeights.information</code> is binary, symmetric, with provenance
<code>{source: 'GPUContiguityWeights', criterion}</code>.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUContiguityWeights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  criterion: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'rook'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  polygons: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, ringOffsets, polygonOffsets, featureOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUContiguityWeights */}</p>
<p>::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen neighbours, and how many of the 791 tracts are islands."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=formula caption="Follow the rule: tracts are neighbours when their snapped vertex keys intersect."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=kernel caption="Inspect the key sort, pair emission and CSR compaction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=graph caption="Trace the command graph from tract vertices to the CSR neighbour lists."}</p>
<h3 id="gpulatticeweights"><code>GPULatticeWeights</code></h3>
<p>Connect the cells of a width by height grid to their rook or queen neighbours as binary CSR weights.</p>
<p>::lens{scenario=gpu-spatial-analysis.lattice-weights rung=result caption="Queen neighbours on a 32 by 32 terrain lattice: 8 inside, 5 along edges, 3 in corners."}</p>
<p>Weights over a <code>width x height</code> grid with row-major IDs <code>y * width + x</code>. <code>criterion: 'rook'</code> uses
Manhattan distance <code>&#x3C;= radius</code>, <code>'queen'</code> Chebyshev distance <code>&#x3C;= radius</code> (<code>radius</code> defaults to 1, at
most 32). A zero in the optional <code>mask</code> removes the cell as a row and as a neighbor. With <code>cellSize</code>
it also writes Euclidean <code>distances</code>, so kernel transforms apply. <code>output.status.requiredCount</code> is
the exact unclamped neighbor total; <code>getCapacityPlan({requiredCount})</code> plans the next slots.
<code>outputWeights.information</code> is binary and symmetric, with provenance <code>{criterion, order: radius}</code>.</p>
<p><code>cellSize</code> accepts either a numeric <code>[width, height]</code> default or a packed float32 view from
<code>getGPULatticeWeightsParameterValues</code>. Host that view in a <code>GPUParameterBuffer</code> and write new
values between encodings to update distances on the same compiled graph. Width, height,
criterion, radius, capacities and optional views remain compile-time; changing the distance
scale keeps the same neighbors and capacity status.</p>
<p>The capacity plan includes 8 bytes for uploaded numeric cell sizes when <code>output.distances</code> is
present. Borrowed parameter views and configurations without distance output add no parameter
scratch; the per-cell count and offset buffers use 8 bytes per cell.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPULatticeWeights */}</p>
<p>::lens{scenario=gpu-spatial-analysis.lattice-weights rung=datum bind=page datum=528 view=neighbourhood-view caption="Inspect cell 528: its 8 queen neighbours, with the degree histogram of all 1,024 cells."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=formula caption="Follow the rule: two cells are neighbours within the rook or queen radius."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=kernel caption="Inspect the neighbour count and fill kernels over 1,024 cells."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=graph caption="Trace the command graph from lattice size to CSR neighbours."}</p>
<p>{/* parameters: GPULatticeWeights generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulatticeweights-parameters"><code>GPULatticeWeights</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULatticeWeightsParameterValues(settings?)</code>, writes <code>GPU_LATTICE_WEIGHTS_PARAMETER_LENGTH</code> (2) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULatticeWeightsParameterValues</code>:</p>
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
<td><code>cellSize</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Positive cell width and height. Defaults to <code>[1, 1]</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialweightstransform"><code>GPUSpatialWeightsTransform</code></h3>
<p>Rewrite CSR weights by row, binary, kernel, symmetric, double or variance standardisation into a new weights graph.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=result caption="Row standardisation keeps the queen links of 791 tracts and rescales each row to sum to one."}</p>
<p>Rewrites weights in place or into a separate <code>output</code>. Chain several instances, for example a kernel
followed by row standardization. Numeric <code>bandwidth</code> uses the length unit of
<code>weights.spatialContext</code>, matching <code>weights.distances</code>. The optional <code>spatialContext</code> declares the
bandwidth's context and is checked against <code>weights.spatialContext</code> with
<code>assertCompatibleGPUSpatialContexts</code> when both are supplied.</p>
<ul>
<li><code>'row'</code>: <code>w_ij / sum_j w_ij</code>. A row whose weights sum to zero (or to a non-finite value) stays all
zero: it has no active neighbor, so it is an island for every statistic (no uniform fallback).
To restore row standardization after masking or <code>'subgraph'</code> removed neighbors, apply <code>'row'</code> again
to the result.</li>
<li><code>'binary'</code>: <code>w_ij > 0 ? 1 : 0</code>, keeping every slot.</li>
<li><code>'kernel'</code>: <code>K(d_ij / h_i)</code> from <code>weights.distances</code>, with <code>kernel</code> one of <code>gaussian</code>
(<code>exp(-z^2 / 2) / sqrt(2 pi)</code>), <code>triangular</code> (<code>max(1 - z, 0)</code>), <code>epanechnikov</code> (<code>3/4 max(1 - z^2, 0)</code>,
PySAL quadratic), <code>bisquare</code> (<code>15/16 max(1 - z^2, 0)^2</code>, PySAL quartic) or <code>uniform</code> (<code>1/2</code> for
<code>z &#x3C;= 1</code>). <code>bandwidth</code> is required (no default) and is either a positive number, the same <code>h</code> for
every row (PySAL fixed bandwidth; for distance-band weights pass the band radius), or <code>'adaptive'</code>:
PySAL's adaptive bandwidth, <code>h_i</code> = the distance to the k-th nearest neighbor times
<code>eps = 1.0000001</code> (<code>GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON</code>). On the CSR that is the row's
largest distance, but only for kNN rows, so <code>'adaptive'</code> requires
<code>weights.information.provenance.criterion === 'knn'</code> and throws otherwise. Because of <code>eps</code> the k-th
neighbor keeps a tiny positive weight, as in libpysal (in float32 <code>eps</code> rounds to <code>1 + 2^-23</code>, so
the tail weight is about 1e-7, never exactly 0). Only the stored neighbors take part, so the
Gaussian is truncated at the neighbor set. Weights never hold self slots, so <code>K(0)</code> is not
written: compute it with <code>getGPUSpatialWeightsKernelValue(kernel, 0)</code> and pass it as <code>selfWeight</code>
to <code>GPUHotSpotAnalysis</code> for a kernel Gi*. <code>getGPUSpatialWeightsKernelValue(kernel, z)</code> is the CPU
value of every profile.</li>
<li><code>'double'</code>: <code>w_ij / S0</code>, with <code>S0</code> the sum of all weights, so the weights sum to 1 (PySAL <code>'D'</code>).
<code>doubleSum: 'rows'</code> scales by <code>n / S0</code> instead, so the weights sum to <code>n</code>.</li>
<li><code>'variance'</code>: <code>s_ij = w_ij / sqrt(sum_j w_ij^2)</code>, then <code>w'_ij = s_ij n / sum s</code> over the whole
matrix, with <code>n</code> the row count (PySAL <code>'V'</code>).</li>
<li><code>'symmetrize'</code>: <code>(w_ij + w_ji) / 2</code>, treating a missing reverse slot as 0. It keeps the sparsity
pattern, so it is only symmetric for symmetric patterns (contiguity, lattice, distance band), and
needs a separate <code>output</code>. For a directed pattern such as kNN, take the union of the weights and
their <code>GPUSpatialWeightsTranspose</code> with <code>GPUSpatialWeightsAlgebra</code>.</li>
</ul>
<p><code>'double'</code> and <code>'variance'</code> compute a per-row partial, reduce it with <code>GPUReduction</code>, and apply it in
a second pass, in place or into <code>output</code>. A zero or non-finite total gives zero weights. The
operation, kernel profile and adaptive-versus-fixed choice are compile-time options. A fixed
<code>bandwidth</code> also accepts a packed float32 view from <code>getGPUSpatialWeightsTransformParameterValues</code>;
write new values through its <code>GPUParameterBuffer</code> between encodings. Numeric bandwidths remain
supported as uploaded defaults. Updating bandwidth changes only existing slots; it does not
rebuild the neighbor set or update a previously supplied numeric provenance radius. <code>output</code>, when given, must have exactly
<code>weights.neighbors.length</code> values.</p>
<p><code>outputWeights.information.transform</code> is the operation (<code>'symmetrize'</code> gives <code>'symmetric'</code>). <code>symmetric</code>
is kept only where it is guaranteed: <code>'binary'</code>, <code>'double'</code>, <code>'symmetrize'</code> and a fixed-bandwidth
<code>'kernel'</code> keep a <code>true</code> input flag; <code>'row'</code>, <code>'variance'</code> and an adaptive kernel give <code>'unknown'</code>
(row standardization makes a symmetric pattern asymmetric unless all row sums are equal).</p>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsTransform */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen weights after row standardisation, and the islands among 791 tracts."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=formula caption="Follow the row rule: each weight is divided by its row sum; empty rows stay zero."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=kernel caption="Inspect the row-sum and rescale kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=graph caption="Trace queen contiguity into row standardisation on the command graph."}</p>
<p>{/* parameters: GPUSpatialWeightsTransform generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialweightstransform-parameters"><code>GPUSpatialWeightsTransform</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialWeightsTransformParameterValues(settings)</code>, writes <code>GPU_SPATIAL_WEIGHTS_TRANSFORM_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialWeightsTransformParameterValues</code>:</p>
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
<td><code>bandwidth</code></td>
<td><code>number</code></td>
<td>Positive fixed bandwidth, in the units of the resident neighbor distances.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatiallag"><code>GPUSpatialLag</code></h3>
<p>If either supplied <code>weights.status.overflow</code> or <code>weights.status.candidateOverflow</code> is nonzero,
every lag component is NaN; a retained prefix is never presented as a complete weighted sum.
For chunked weights, loss on the outer status or any chunk suppresses the whole result.
Clearing the loss words and submitting complete CSR data restores the lag on the same graph.</p>
<p>Average neighbouring values through row-standardised weights, turning a tract variable into its spatial lag.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-lag rung=result caption="Poverty percentage beside its neighbour-weighted lag across 791 tracts; the lag smooths local extremes."}</p>
<p><code>lag_i = sum_j w_ij y_j</code> (PySAL <code>lag_spatial</code>). An optional <code>mask</code> skips masked neighbors and writes
0 for masked rows (square weights only); <code>normalize: true</code> divides by the included weight sum, which
gives the row-standardized lag without a transformed copy of the weights. Rows are summed by a
load-balanced (merge-path) CSR loop, so a hub row with 1e5 neighbors spreads over many workgroups
instead of one thread. The association depends only on the offsets, so the result is bitwise
reproducible; it matches a sequential sum within f32 rounding. A weights matrix with a self term (<code>GPUSpatialWeightsAlgebra</code>
<code>'selfWeight'</code>) includes it in the sum.</p>
<p>Weights need not be square. The value row count is <code>values.length / columnCount</code>: the rows of
<code>weights</code> are targets and the neighbor IDs index those value rows; neighbors at or above it are
skipped. <code>columnCount</code> (default 1) lags several columns in one pass: <code>values</code> is row-major,
<code>output</code> is <code>rows * columnCount</code>, and <code>normalize</code> shares one weight sum per row. This is how the area
shares of <code>GPUArealInterpolation</code> transfer values between zone systems. <code>values</code> may be <code>float32</code> or
<code>uint32</code> (counts, exact below 2^24; one conversion node is added).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> contiguity</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUContiguityWeights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({criterion: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'queen'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, polygons, output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">weights, status: {overflow}}});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(contiguity);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsTransform</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({operation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'row'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, weights: contiguity.outputWeights}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialLag</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights: contiguity.outputWeights, values, output: lag}));</span></span></code></pre>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPUSpatialLag */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-lag rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its neighbours and weights, and the islands that receive a zero lag."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=formula caption="Follow the lag: the sum of each neighbour weight times its value."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=kernel caption="Inspect the load-balanced segment sum over the CSR slots."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=graph caption="Trace queen weights, row standardisation and the lag on the command graph."}</p>
<h3 id="gpuneighborsearch"><code>GPUNeighborSearch</code></h3>
<p>Find the k nearest tract centroids by great-circle distance and write them as CSR neighbour weights.</p>
<p>::lens{scenario=gpu-spatial-analysis.neighbor-search rung=result caption="Each of 791 tract centroids links to its 4 nearest others, giving 3,164 directed links."}</p>
<p>Maximum: <code>gridSize[0] * gridSize[1]</code> is at most <code>getGPUGridIndexMaximumCellCount(device)</code> (2^25 - 1 on a default device), checked when the contributor is added to a graph.</p>
<p><code>GPUNeighborSearch</code> finds exact k nearest neighbors (<code>mode: 'knn'</code>, <code>k</code> up to <code>GPU_NEIGHBOR_SEARCH_MAXIMUM_K</code>, 256) or every point
within a distance band (<code>mode: 'radius'</code>) between planar points, or between longitude/latitude points
with great-circle distances (see below). A self join never lists a row as its
own neighbor; pass <code>queryPositions</code> for a cross join. It writes a <code>GPUSpatialWeights</code> CSR:
<code>offsets</code> (<code>queryRows + 1</code>), <code>neighbors</code> (the slot capacity), <code>weights</code>, and optional <code>distances</code>.</p>
<ul>
<li>Rows list neighbor IDs in strictly ascending order with distances and weights aligned, so
consumers can binary-search <code>w_ji</code>.</li>
<li>kNN keeps the <code>k</code> smallest <code>(d^2, id)</code> pairs, so ties go to the lowest ID. An optional
<code>radius</code> bounds the search. Fewer valid targets give a shorter row.</li>
<li>Weights: <code>binary</code>, <code>inverseDistance</code> (<code>max(d, distanceFloor)^-power</code>, default floor <code>1e-6</code>,
the same default as <code>GPUNearestFeatureWeights</code>; two rows at the same position then weigh
<code>1e6^power</code> to each other and, with <code>rowStandardize</code>, take nearly all of each other's row, so pass
a floor on the scale of the data when positions repeat), or <code>kernel</code> (gaussian,
triangular, epanechnikov, bisquare, uniform). The bandwidth is <code>radius</code> in radius mode and the
row's k-th neighbor distance times 1.0000001 in kNN mode, which is libpysal <code>Kernel(fixed=False)</code>
with the self slot removed (a row with fewer than <code>k</code> neighbors uses its farthest listed
neighbor). <code>rowStandardize</code> divides each row by its sum. Non-finite weights are written as 0.
Rows never hold a self slot: the kernel's self weight <code>K(0)</code> is the consumer's <code>selfWeight</code>
(<code>getGPUSpatialWeightsKernelValue(kernel, 0)</code>). A zero-weight slot (for example a triangular
kernel at <code>d = radius</code>, or coincident points under inverse distance with <code>distanceFloor: 0</code>) is not a
neighbor: consumers skip slots whose weight is not positive.</li>
<li>Capacity: offsets are clamped to <code>neighbors.length</code>, so readers always see consistent, possibly
truncated rows. <code>output.status.overflow</code> is set on the GPU, and <code>output.status.requiredCount</code> and
<code>output.neighborCounts</code> stay unclamped. The total <code>requiredCount</code> saturates at <code>0xffffffff</code>;
offsets remain monotonic after a uint32 carry, preserving the output prefix and setting overflow.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> searchParameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {bounds, rowStandardize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'knn'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUNeighborSearchParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(searchParameters)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUNeighborSearch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'knn'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, k: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], positions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {offsets, neighbors, weights, distances, status: {overflow}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weightTransform: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUNeighborSearchWeightTransform</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(searchParameters)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><strong>Spatial context.</strong> <code>spatialContext</code> (compile-time) defaults to native planar coordinates. Planar
contexts require metric <code>'native'</code>; local frames with <code>units: 'meters'</code> are accepted. This operator
measures distance, so <code>local-lambert-equal-area</code> frames throw. The resolved context is attached to
<code>outputWeights.spatialContext</code>. For
longitude/latitude pass <code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units?: 'meters' | 'kilometers', sphereRadius?}</code> (<code>sphereRadius</code> in the context's length unit, default the mean Earth radius).
Positions are <code>[longitude, latitude]</code> degrees, <code>bounds</code> are <code>[minLongitude, minLatitude, maxLongitude, maxLatitude]</code> degrees (longitude span at most 360, for example the world), and
<code>radius</code>, <code>distanceFloor</code> and the output <code>distances</code> are great-circle distances in <code>units</code>
(meters by default). Any other space or metric throws at construction (project to a planar frame
first). Distances are f32 central angles on a sphere. The lattice stays in degrees: radius mode
visits every cell that can hold a target in the cap (latitude span <code>radius / R</code>, longitude span the
exact cap extent <code>asin(sin(a) / cos(latitude))</code>, the whole circle when the cap reaches a pole, and
bands wrap across +-180), and kNN stops only when a conservative spherical lower bound of the
distance to every unvisited cell exceeds the k-th distance, so results never depend on <code>gridSize</code>.
The bound is deliberately loose near the poles, which only costs speed. Points with latitude beyond
+-90 are invalid. Whole longitude turns do not change query distances or neighbor ordering:
distance evaluation reduces each stored float32 longitude exactly modulo 360 using integer
significand arithmetic, including very large finite values. Candidate lookup shifts that canonical
angle by whole turns toward the lattice midpoint; the shift does not affect distance evaluation.</p>
<p><strong>Planar precision.</strong> Planar f32 coordinates keep about seven digits: UTM northings (~4e6 m) and
Web Mercator (~1e7) quantize to 0.5 to 1 m and make band edges and ties noisy. Project to a local
frame with <code>createGPULocalFrame</code> (or subtract an origin) before upload; the node cannot recover
digits lost in its inputs.</p>
<p>Pass optional <code>spatialContext</code> to <code>getGPUNeighborSearchParameterValues(settings)</code> to warn on short <code>radius</code> values; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p><strong>Row standardization.</strong> The optional compile-time <code>rowStandardize</code> prop is the single source of truth
when given: the weights kernel uses it as a constant and ignores the per-frame <code>rowStandardize</code>
parameter. Consumers re-standardize rows over the neighbors left after masking only when
<code>info.transform</code> is <code>'row'</code>, so set it for any row-standardized band. <code>rowStandardize: true</code> with a
<code>weightTransform</code> other than <code>'row'</code>, or <code>false</code> with <code>'row'</code>, throws. When it is omitted, the per-frame
field and the <code>weightTransform</code> declaration apply as before.</p>
<p><strong>Output.</strong> <code>outputWeights</code> is <code>output</code> with <code>information</code> and <code>status</code> attached, for consumers:
<code>rowCount</code>, <code>transform</code> (<code>'row'</code> when the compile-time <code>rowStandardize</code> prop is true, otherwise the
optional compile-time <code>weightTransform</code>, else <code>'unknown'</code>, because weight kind and row standardization
are per-frame parameters; derive it from the same parameters with
<code>getGPUNeighborSearchWeightTransform</code>), <code>symmetric</code> (true only for a planar radius self join
without <code>queryMask</code> and a declared non-row transform; otherwise <code>'unknown'</code>), <code>provenance</code>
(<code>criterion</code> <code>'knn'</code> or <code>'distance-band'</code>, and <code>k</code>) and <code>status</code> (<code>overflow</code>, <code>requiredCount</code>,
<code>invalidCount</code>, and optional <code>candidateOverflow</code>). A requested candidateOverflow is overwritten
with 0 on every submission, since neighbor search has no bounded intermediate candidate list.
kNN output has criterion <code>'knn'</code>, which <code>bandwidth: 'adaptive'</code>
kernel transforms require. PySAL's <code>min_threshold_distance</code> (the smallest band that leaves no
island) is the maximum of the returned <code>distances</code> of a <code>k = 1</code> kNN run.</p>
<p><strong>Dropped rows.</strong> Targets and self-join rows count only when their mask is nonzero, their coordinates are finite, and
they lie inside the per-frame <code>bounds</code>; the others silently become isolates and never a neighbor. A
cross-join query (<code>queryPositions</code>) needs only a nonzero <code>queryMask</code> and finite coordinates: a query outside <code>bounds</code>
returns exactly the brute-force neighbors, so <code>bounds</code> need only cover the targets. Great-circle queries also require latitude within ±90 degrees;
longitude wrapping includes targets across the seam, even with regional bounds. A planar query far outside <code>bounds</code> (beyond a side or corner by many lattice
widths) cannot stop its kNN ring walk early and visits every lattice cell in one invocation: with 200,000 targets in
a 512 x 512 lattice, k = 8 and 2,000 queries the dispatch took about 7.5 ms with the queries inside and about 25 ms
with them 1e6 beyond a side or corner, same results. When every squared distance of a query overflows float32
(offsets beyond about 1.8e19), all targets tie at infinity and kNN returns the lowest target IDs, as the scan does,
on the grid and on the BVH; the written distances of those rows are infinite.
Pass the optional <code>output.status.invalidCount</code> (one uint32) to learn how many masked rows were dropped for a
non-finite coordinate or for lying outside <code>bounds</code> (target rows, plus the query rows of a cross
join). A self join applies <code>mask</code> to the query rows too. Bounds,
radius, the weight function and row standardization are per-frame parameters, so changing them
never rebuilds the graph. Targets are bucketed with <code>GPUGridIndex</code>, whose <code>boundsBuffer</code> carries the
per-frame lattice (cells at least <code>radius</code> wide) without recompiling. kNN
searches expanding cell rings with a private sorted top-k list and stops once the k-th distance is
strictly inside the visited box, with a slack for f32 rounding. Radius mode counts, scans, emits, and
insertion-sorts each row by ID. No float atomics are used, so outputs are bitwise reproducible.</p>
<p><strong>Search structure.</strong> The optional compile-time <code>searchStructure</code> (<code>'auto'</code> by default, <code>'grid'</code> or
<code>'bvh'</code>) only changes speed: all three give the same IDs, offsets, distances and weights (bitwise),
except which neighbors the one row cut by an exceeded capacity keeps. A grid cell that holds most of
the points (clustered data) makes every query in it scan that cell, so <code>'bvh'</code> instead walks a
Hilbert-packed point BVH in near-first depth-first order with the same <code>(d^2, id)</code> top-k
list and radius limit, then shares the grid path's count, scan, emit and sort-by-ID stages. Invalid
targets (mask, non-finite, outside the per-frame bounds) get empty boxes, so per-frame inputs still
never recompile. <code>'auto'</code> now uses exactly the grid graph, without allocating or building a BVH or
computing an occupancy gate. All four measured 200k-point cases (clustered/uniform, kNN/radius)
favored the grid; occupancy alone did not predict traversal cost. <code>'bvh'</code> remains opt-in.
<code>GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD</code> is retained for compatibility but no longer used.
The BVH path is planar
only: <code>'bvh'</code> with a longitude-latitude <code>spatialContext</code> throws, and great-circle search always uses
the grid. The new prop is compile-time and not part of the per-frame parameter
packer, so the Parameters section below is unchanged.</p>
<h4 id="space-time-search">Space-time search</h4>
<p>Pass <code>times</code> (one relative <code>float32</code> per target, with a <code>float32-relative</code> <code>temporalContext</code>) to
search a space-time cylinder: a target is a neighbor only when it also satisfies
<code>|t_target - t_query| &#x3C;= timeRadius</code>, and kNN means the <code>k</code> nearest targets inside the window. A
cross join also needs <code>queryTimes</code>. The parameter view grows to
<code>GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH</code> (15) words holding <code>timeRadius</code> and <code>timeBounds</code>,
packed by the same <code>getGPUNeighborSearchParameterValues</code>. A target or self-join row whose time is non-finite or
outside <code>timeBounds</code> is invalid like a row outside <code>bounds</code>, and <code>invalidCount</code> counts it; a cross-join query
needs only a finite time.</p>
<p><code>timeBucketCount</code> (compile-time, default 1, at most <code>GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS</code>) also
indexes the targets by time, so a window much shorter than <code>timeBounds</code> skips most of them. It
never changes a result. On 100,000 clustered points with a window of 5% of the time span, 8 buckets
took 29.7 ms against 40.4 ms for one (radius) and 13.1 ms against 25.2 ms (kNN, <code>k = 16</code>); 64
buckets gave no further gain. A kNN query whose window holds fewer than <code>k</code> targets finishes every
ring, so set a <code>radius</code> when windows may be sparse. <code>searchStructure: 'bvh'</code> does not take times.</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPUNeighborSearch */}</p>
<p>::lens{scenario=gpu-spatial-analysis.neighbor-search rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its four nearest centroids, and the degree histogram of all 791 tracts."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=formula caption="Follow the rule: keep the k smallest distance and ID pairs, excluding the tract itself."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=kernel caption="Inspect the grid search kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=graph caption="Trace centroids into the grid and k-nearest search on the command graph."}</p>
<p>{/* parameters: GPUNeighborSearch generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuneighborsearch-parameters"><code>GPUNeighborSearch</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUNeighborSearchParameterValues(settings)</code>, writes <code>GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH</code> (12) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUNeighborSearchParameterValues</code>:</p>
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
<td><code>spatialContext</code> (optional)</td>
<td><code>GPUSpatialContext</code></td>
<td>Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values.</td>
</tr>
<tr>
<td><code>bounds</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td><code>[minX, minY, maxX, maxY]</code> extent of the search lattice. Targets and self-join rows outside it (or with non-finite coordinates) are excluded: they become isolates and never a neighbor, so it must cover every target of interest (count the dropped rows with <code>invalidCount</code>). The queries of a cross join are not clipped by it: a query outside the bounds still finds its exact neighbors (great-circle latitude must be within +-90). For a great-circle search these are degrees <code>[minLongitude, minLatitude, maxLongitude, maxLatitude]</code> with a longitude span of at most 360 (<code>[-180, -90, 180, 90]</code> for the world).</td>
</tr>
<tr>
<td><code>radius</code> (optional)</td>
<td><code>number</code></td>
<td>In the units of the spatial context (position units, or meters/kilometers of great-circle distance). Radius mode: the distance band (neighbors satisfy <code>d &#x3C;= radius</code>); a non-positive or non-finite radius yields empty rows. kNN mode: an optional maximum distance; <code>Infinity</code> (the default) or any non-positive value means unbounded.</td>
</tr>
<tr>
<td><code>weightKind</code> (optional)</td>
<td><code>GPUNeighborSearchWeightKind</code></td>
<td>Weight function. Defaults to <code>'binary'</code>.</td>
</tr>
<tr>
<td><code>power</code> (optional)</td>
<td><code>number</code></td>
<td>Inverse-distance exponent. Defaults to 1.</td>
</tr>
<tr>
<td><code>distanceFloor</code> (optional)</td>
<td><code>number</code></td>
<td>Inverse distance uses <code>max(d, distanceFloor)</code>, in the length unit of the distances. Defaults to <code>1e-6</code>, the default of <code>GPUNearestFeatureWeights</code> too, so coincident points (<code>d = 0</code>) get the finite weight <code>distanceFloor^-power</code> (<code>1e6</code> at power 1). An explicit 0 removes the floor: coincident points then get weight 0, because their weight is not finite, and a zero-weight slot is not a neighbor. Under the default, a coincident neighbor outweighs every ordinary neighbor by many orders of magnitude and takes nearly all of a row-standardized row; pass a floor on the scale of the data when positions repeat (see the class documentation).</td>
</tr>
<tr>
<td><code>kernel</code> (optional)</td>
<td><code>GPUNeighborSearchKernel</code></td>
<td>Kernel profile for <code>weightKind: 'kernel'</code>. Defaults to <code>'triangular'</code>.</td>
</tr>
<tr>
<td><code>rowStandardize</code> (optional)</td>
<td><code>boolean</code></td>
<td>Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false.</td>
</tr>
<tr>
<td><code>timeRadius</code> (optional)</td>
<td><code>number</code></td>
<td>Time window of a space-time search, in the <code>temporalContext</code> unit: a neighbor satisfies <code>|t_target - t_query| &#x3C;= timeRadius</code>. Non-positive or non-finite makes every row empty. Required with <code>times</code>, and selects the <code>GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH</code>-word layout.</td>
</tr>
<tr>
<td><code>timeBounds</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td><code>[minimumTime, maximumTime]</code> covered by the index. Rows outside it are invalid, like rows outside <code>bounds</code>. Required with <code>timeRadius</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialweightsalgebra"><code>GPUSpatialWeightsAlgebra</code></h3>
<p>Combine or expand CSR weights by union, intersection, higher order, self weights, subgraph or block into a new graph.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=result caption="Second-order queen neighbours of 791 tracts: each tract links to tracts exactly two steps away."}</p>
<p>Builds a new <code>GPUSpatialWeights</code> from existing ones with a capacity-bounded <code>output</code> (rows + 1
offsets; it must not alias an input). <code>operation</code> is one of:</p>
<ul>
<li><code>'union'</code>, <code>'intersection'</code>, <code>'difference'</code>, <code>'symmetricDifference'</code> with <code>left</code> and <code>right</code> (same
row count) and an optional <code>weightRule</code>: <code>'left'</code> (default), <code>'right'</code>, <code>'sum'</code>, <code>'min'</code>, <code>'max'</code>,
<code>'product'</code> or <code>'binary'</code>. The rule applies where both operands list a link, and one-sided links keep
their weight. <code>'binary'</code> gives 1 everywhere, which reproduces libpysal 4.15 <code>Graph</code> set operations.
Distances come from <code>left</code> where it lists the link, else from <code>right</code>.</li>
<li><code>'higherOrder'</code> with <code>weights</code>, <code>order</code> (1 to 32), <code>cumulative</code> (default false) and <code>workCapacity</code>
(intermediate capacity, default the output capacity). Writes the neighbors exactly <code>order</code> steps away
by shortest path, excluding the diagonal, with weight 1; <code>cumulative</code> matches <code>lower_order=True</code>.
Cost grows with the squared frontier degree per row, so it suits sparse contiguity and kNN weights.</li>
<li><code>'selfWeight'</code> with <code>weights</code> and <code>selfWeight</code> (a number at least 0, or a per-row <code>float32</code> view).
Inserts or replaces <code>w_ii</code> with distance 0. It relaxes the <code>w_ii = 0</code> invariant. <code>GPUSpatialLag</code>,
<code>GPUSpatialWeightsTransform</code>, <code>GPUSpatialWeightsAlgebra</code> and <code>GPUSpatialWeightsSummary</code> accept it;
<code>GPUGlobalSpatialStatistics</code>, the permutation tests and the local autocorrelation kernels ignore the
self slot.</li>
<li><code>'subgraph'</code> with <code>weights</code> and a per-row <code>mask</code>. Rows keep their IDs: masked rows become empty and
slots that point at masked rows are dropped.</li>
<li><code>'block'</code> with <code>groupIds</code> and <code>groupCount</code>. Row <code>i</code> lists every other row of the same group with
weight 1; IDs at or above <code>groupCount</code> have no group. The output holds the sum of squared group
sizes minus the row count.</li>
</ul>
<p><code>output.status.overflow</code> (one <code>uint32</code>, required) is 1 when the capacity was too small, and the
optional <code>output.status.requiredCount</code> holds the slots needed (published as
<code>outputWeights.status.requiredCount</code>; omitted there for <code>'higherOrder'</code> with
<code>order > 1</code>, where an overflowing intermediate set would make it a lower bound, so raise
<code>workCapacity</code> instead). On overflow the output is still a valid CSR: the leading rows are complete,
one row is a prefix and the rest are empty. Output rows are sorted and unique with finite
non-negative weights. <code>outputWeights.information</code> carries the transform state (<code>'binary'</code> for
<code>weightRule: 'binary'</code>, <code>'higherOrder'</code> and <code>'block'</code>; an unscaled <code>'raw'</code> or <code>'binary'</code> input
stays so; a row-standardized input becomes <code>'unknown'</code> because slots changed) and <code>symmetric: true</code>
only when it follows from symmetric operands.</p>
<p>Algebra carries either operand loss cause into candidateOverflow, independently of whether
the retained input fits the new destination. Higher-order work scratch loss also sets
candidateOverflow; overflow is only destination slot loss. Increasing destination capacity
cannot repair inherited loss. The named operation and CPU twin require a candidate output name
when operand loss ports or higher-order intermediate work are supplied. Named contiguity operations
always require both output loss names; their candidate scratch is bounded. Correlogram incompleteOrders includes either upstream cause
as well as loss introduced by that order, and incomplete orders suppress global Moran inference.
The CPU twin honors an explicit higher-order workCapacity at every intermediate frontier and
visited union. When omitted, its frontiers are unbounded; candidate loss makes requiredCount a
lower bound in either form.</p>
<p>Algebra, contiguity and neighbor-search constructors snapshot their declared props and nested
port/status maps once, including inherited getters. Their GPU buffer contents remain per-frame
inputs. Rebuild the contributor to change a view, status destination, operation or capacity.
Both named contiguity backends validate scalar options before output-port requirements and report
the same scalar error even when a candidate destination is missing.
Both forms also require distinct named output values: reusing a name for final and candidate loss
is rejected before either diagnostic can be overwritten.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsAlgebra</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  operation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'union'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, left: contiguity, right: knn, weightRule: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'binary'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsAlgebra */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its second-order neighbours, and the degree histogram of all 791 tracts."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=kernel caption="Inspect the frontier expansion, sort and compaction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=graph caption="Trace first-order queen weights into second-order expansion on the command graph."}</p>
<h3 id="gpuspatialweightssummary"><code>GPUSpatialWeightsSummary</code></h3>
<p>Summarise CSR weights into S0, S1, S2, slot counts, asymmetry, isolates and degree range.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=result caption="Read S0, S1, S2 and the isolate count of the queen weights of 791 tracts."}</p>
<p>Summary statistics of a weights matrix, written into <code>output: {statistics, counts, cardinality?}</code>.
<code>statistics</code> (<code>float32</code>, at least 3) receives <code>S0</code>,
<code>S1 = 1/2 sum (w_ij + w_ji)^2</code> and <code>S2 = sum_i (row_i + col_i)^2</code>, as in libpysal. <code>counts</code> (<code>uint32</code>,
at least 6, see <code>GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT</code>) receives the used slots, the asymmetric slots,
the isolates, the minimum and maximum cardinality and the empty rows. The optional <code>cardinality</code> is the
per-row active neighbor count. <code>symmetryTolerance</code> accepts a numeric default (0 for exact comparison)
or a packed one-float view from <code>getGPUSpatialWeightsSummaryParameterValues</code> hosted in a
<code>GPUParameterBuffer</code>. Updating that buffer changes the reverse-weight comparison on the same compiled
graph, for both whole and chunked weights. View shapes and chunk layout stay compile-time. Sums
run in a fixed order. Isolates are rows with no active slot, the one island definition of every
statistic (libpysal <code>islands</code> for a W built without zero-weight entries); <code>emptyRows</code> counts rows with
no slot at all. Cardinality counts active slots, so a zero-weight slot is not a neighbor: this deviates
from libpysal <code>cardinalities</code>, which counts every listed neighbor, and agrees with it for a W built
without zero-weight entries (pinned against libpysal in the specs). The used slots count stays the
listed total. Connected components are not computed. A pair of unequal weights counts as two
asymmetric slots, where libpysal <code>asymmetry()</code> returns pairs.</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsSummary */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=datum bind=page datum=420 caption="Inspect tract 420: its degree among the 791 tracts, beside the global moments."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=formula caption="Follow three moments: S0, S1 and S2 from row sums, column sums and reverse weights."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=kernel caption="Inspect the row, column and blocked reduction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=graph caption="Trace the row, column and global reduction passes on the command graph."}</p>
<p>{/* parameters: GPUSpatialWeightsSummary generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialweightssummary-parameters"><code>GPUSpatialWeightsSummary</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialWeightsSummaryParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialWeightsSummaryParameterValues</code>:</p>
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
<td><code>symmetryTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Non-negative absolute tolerance for counting unequal reverse weights. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialweightstranspose"><code>GPUSpatialWeightsTranspose</code></h3>
<p>Reverse every link of CSR weights, turning outgoing weights into incoming ones.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=result caption="Transposed row-standardised queen weights: rows no longer sum to one, since they hold incoming weights."}</p>
<p>The CSR transpose <code>W'</code> of any <code>GPUSpatialWeights</code>, square or rectangular (libpysal <code>Graph</code> with focal and
neighbor swapped). <code>columnCount</code> (default: the row count) sets the number of rows of the result for
cross weights whose neighbor IDs index target rows; slots with a neighbor ID at or above it are dropped,
and columns that nothing reaches give empty rows. <code>output</code> is a caller-owned CSR with <code>columnCount + 1</code>
offsets and at least the input capacity in <code>neighbors</code> and <code>weights</code> (and <code>distances</code> when the input has
them); slots past <code>offsets[columnCount]</code> are zeroed. Slots are keyed by column, stable-sorted, counted, scanned and gathered, so every output
row is ascending and the result is bitwise reproducible with no atomics. The optional
<code>output.asymmetricSlots</code>
(one <code>uint32</code>, square weights only, with <code>symmetryTolerance</code>, default 0) counts the positions <code>(i, j)</code> of
the union of both patterns whose weights differ or that are present in one only, so a one-way link counts
twice and 0 means <code>W = W'</code>. Reference: libpysal 4.15 (no deviations; the check is exact by default).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsTranspose</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">transposed, asymmetricSlots}}));</span></span></code></pre>
<p><code>symmetryTolerance</code> accepts a numeric default or a packed one-float view from
<code>getGPUSpatialWeightsTransposeParameterValues</code> hosted in a <code>GPUParameterBuffer</code>. Updating that
buffer changes the symmetry comparison on the same compiled graph. Equality at the tolerance is
symmetric; a missing reverse slot always differs, including when the present slot's weight is zero.
CSR values, distances and tolerance are per-frame; view shapes, column count and optional outputs
stay compile-time. The tolerance is unused when <code>asymmetricSlots</code> is omitted.</p>
<p><code>outputWeights</code> carries the input <code>symmetric</code> flag (a transpose is symmetric exactly when <code>W</code> is) and
the input <code>status</code> (a truncated <code>W</code> gives a truncated <code>W'</code>); a <code>'row'</code> or <code>'variance'</code> state becomes
<code>'unknown'</code>.</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsTranspose */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its incoming weights, which were outgoing weights of its neighbours."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=kernel caption="Inspect the column sort and row boundary search kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=graph caption="Trace row standardisation into transposition on the command graph."}</p>
<p>{/* parameters: GPUSpatialWeightsTranspose generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuspatialweightstranspose-parameters"><code>GPUSpatialWeightsTranspose</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUSpatialWeightsTransposeParameterValues(settings?)</code>, writes <code>GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUSpatialWeightsTransposeParameterValues</code>:</p>
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
<td><code>symmetryTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Non-negative absolute reverse-weight tolerance. Missing reverse slots always differ. Defaults to 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuspatialweightsvalidation"><code>GPUSpatialWeightsValidation</code></h3>
<p>Flag rows of uploaded CSR weights whose offsets, neighbour order, range, diagonal or weights break the invariants.</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=result caption="Validation flags per tract for queen weights; expect zero of 791 rows flagged."}</p>
<p>Checks the <code>GPUSpatialWeights</code> invariants of weights that were built elsewhere and uploaded, on the GPU:
<code>offsets[0] = 0</code>, non-decreasing offsets with <code>offsets[rows] &#x3C;= neighbors.length</code>, every neighbor ID below
the row count, strictly ascending neighbor IDs within each row (sorted and unique), no self slot, and
finite non-negative weights. <code>output.status.invalidCount</code> (one <code>uint32</code>) receives the number of rows with at least one
violation; the optional <code>output.flags</code> (one <code>uint32</code> per row) receives the bitmask of
<code>GPU_SPATIAL_WEIGHTS_VIOLATION</code> (<code>offsets</code> 1, <code>neighborRange</code> 2, <code>order</code> 4, <code>self</code> 8, <code>weight</code> 16). Run it
once after upload and before using the weights; there is no GPU sort or dedupe, so fix a failing CSR on the
CPU. <code>outputWeights</code> is the checked weights with <code>status.invalidCount</code> attached.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialWeightsValidation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  weights: uploaded,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {flags, status: {invalidCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUSpatialWeightsValidation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=datum bind=page datum=420 caption="Inspect tract 420: its violation flags, and the total number of invalid rows."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=kernel caption="Inspect the per-row invariant check kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=graph caption="Trace queen contiguity into validation on the command graph."}</p>
<h3 id="gpuchunkedspatialweightsvalidation"><code>GPUChunkedSpatialWeightsValidation</code></h3>
<p>Validate chunked CSR weights with one rows dispatch per chunk.</p>
<p>Runs <code>GPUSpatialWeightsValidation</code> over a <code>GPUChunkedSpatialWeights</code> whose chunks are consecutive row ranges, each with its own CSR buffers. Chunk <code>c</code> checks its own offsets and capacity, and its neighbor IDs against the total row count. <code>output.flags</code> and <code>output.status.invalidCount</code> follow the whole-CSR contract in global row order. Dispatch count is linear in the chunk count and results do not depend on the chunking.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChunkedSpatialWeightsValidation</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights: chunked, output: {flags, status: {invalidCount}}}));</span></span></code></pre>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUChunkedSpatialWeightsValidation */}</p>
<h3 id="gpuchunkedspatialweightstranspose"><code>GPUChunkedSpatialWeightsTranspose</code></h3>
<p>Reverse every link of chunked CSR weights into packed or chunked CSR.</p>
<p>Input chunk-local overflow/candidateOverflow ports are refused at construction. Supply a matrix
port with aggregated status and no chunk-local loss ports; a larger transpose destination cannot
recover omitted source neighbors. Matrix-level input status remains attached to packed output.</p>
<p>The <a href="/luma-contributors/docs/api/gpu-spatial-analysis/chunked-spatial-weights-transpose">chunked transpose guide</a>
details output layouts, capacity status and binding limits.</p>
<p>The packed-output form below remains supported. For independently partitioned output chunks,
use the caller-owned <code>output.chunks</code> layout in that guide, including a status port per chunk.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChunkedSpatialWeightsTranspose</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights: chunked, output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">transposed, asymmetricSlots}}));</span></span></code></pre>
<p>The per-frame <code>symmetryTolerance</code> view uses <code>getGPUSpatialWeightsTransposeParameterValues</code> in
both output forms. See the guide for compilation-time shapes and output-metadata semantics.</p>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPUChunkedSpatialWeightsTranspose */}</p>
<h3 id="gpuchunkedspatialweightstransform"><code>GPUChunkedSpatialWeightsTransform</code></h3>
<p>Standardise, kernel-weight or symmetrize chunked CSR weights.</p>
<p>Takes the operations of <code>GPUSpatialWeightsTransform</code>. <code>row</code>, <code>binary</code> and <code>kernel</code> are row-local and run once per chunk. <code>double</code> and <code>variance</code> write per-row partials into one global array, take one fixed-order sum and apply per chunk. <code>symmetrize</code> transposes the chunks into one whole scratch CSR and looks the reverse weight up there, so the summed chunk capacity must fit one binding. Results equal the whole transform at any chunking, up to float32 association of the partial sums. Optional <code>output</code> entries give a destination per chunk; omitted entries write in place.
A fixed <code>bandwidth</code> parameter view is shared by every chunk and uses
<code>getGPUSpatialWeightsTransformParameterValues</code>, exactly as for the whole-CSR transform.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChunkedSpatialWeightsTransform</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({operation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'row'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, weights: chunked}));</span></span></code></pre>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPUChunkedSpatialWeightsTransform */}</p>
<h3 id="gpuchunkedspatialweightsalgebra"><code>GPUChunkedSpatialWeightsAlgebra</code></h3>
<p>Combine chunked CSR weights with row-local set algebra.</p>
<p>Supports <code>union</code>, <code>intersection</code>, <code>difference</code>, <code>symmetricDifference</code>, <code>selfWeight</code> and <code>subgraph</code> of <code>GPUSpatialWeightsAlgebra</code>. One count, scan and emit pipeline runs per chunk and writes the matching chunk of a <code>GPUChunkedSpatialWeightsOutput</code>, so output rows do not depend on the chunking, there is no repacking, and overflow is reported per chunk. Binary operands and the output must be chunked identically. <code>higherOrder</code> and <code>block</code> are not chunked; use <code>GPUSpatialWeightsAlgebra</code>.</p>
<p>The producer's outputWeights adds aggregate matrix overflow for destination truncation across output
chunks and candidateOverflow for either operand loss cause or intermediate loss. Writable output
ranges, including diagnostics, must be disjoint across chunks. Its chunks omit local loss fields so it can feed statistics directly; callers
retain per-chunk diagnostics in their original output allocation. Aggregate words are overwritten
on every submission, including recovery. Lattice and fuzzy-contiguity producers, which have no
bounded intermediate stage, overwrite a requested candidateOverflow with zero each submission.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUChunkedSpatialWeightsAlgebra</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({operation: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'union'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, left, right, output}));</span></span></code></pre>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPUChunkedSpatialWeightsAlgebra */}</p>
<p>Lens coverage is pending for a supported row-local algebra scenario. The named whole-CSR
<code>spatial-weights-algebra</code> scenario demonstrates <code>higherOrder</code> with order 2, which this class cannot
run, so that scenario does not establish coverage for this chunked variant.</p>
<h3 id="gpumapcoloring"><code>GPUMapColoring</code></h3>
<p>Assign each tract a colour index so that no two neighbouring tracts share one.</p>
<p>::lens{scenario=gpu-spatial-analysis.map-coloring rung=result caption="At most five colours separate 791 rook-adjacent tracts; neighbouring tracts never match."}</p>
<p>Consumes an adjacency pattern only, so it takes no <code>spatialContext</code>. Weights may carry their own
context; no compatibility check is needed because coloring uses only adjacency.</p>
<p>Greedy parallel graph coloring (Jones-Plassmann, largest degree first by default) over a
symmetric <code>GPUSpatialWeights</code> such as contiguity, for choropleth maps where neighbors need distinct
colors (QGIS topological coloring). <code>output.colors</code> holds one <code>uint32</code> per row,
<code>GPU_MAP_COLORING_UNCOLORED</code> for rows left unfinished. Optional outputs are <code>colorCount</code>, <code>conflictCount</code>
(pairs sharing a color, 0 for a proper coloring) and <code>status: {converged, iterationCount}</code>. The per-frame
<code>parameters</code> view (<code>getGPUMapColoringParameterValues({seed})</code>, <code>GPU_MAP_COLORING_PARAMETER_LENGTH</code> <code>uint32</code>
words) selects the priorities without recompiling, and <code>maximumIterations</code> (default 64, at most 1024)
bounds the rounds. The compile-time <code>priority</code> prop selects the row order:
<code>'largest-degree-first'</code> (default; Welsh-Powell order, degree is the CSR row length, ties broken by the seeded
hash and then the lowest ID) or <code>'random'</code> (seeded hash, then lowest ID). The result equals sequential greedy
coloring in that order, a proper and seed-deterministic coloring that uses at most the maximum degree plus one
colors, not the minimum number. Largest-degree-first typically saves a color on planar maps (6 versus 6 to 7
on Delaunay graphs, matching networkx <code>largest_first</code>) at about the same number of rounds. Different seeds
give different valid colorings.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUContiguityWeights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({criterion: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'rook'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, polygons, output: {</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">weights, status: {overflow}}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUMapColoring</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({weights, parameters, output: {colors, conflictCount, status: {converged}}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUMapColoringParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({seed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUMapColoring */}</p>
<p>::lens{scenario=gpu-spatial-analysis.map-coloring rung=datum bind=page datum=420 caption="Inspect tract 420: its colour and its neighbours, which all differ."}
::lens{scenario=gpu-spatial-analysis.map-coloring rung=kernel caption="Inspect the priority round kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.map-coloring rung=graph caption="Trace rook contiguity into rounds of greedy colouring on the command graph."}</p>
<p>{/* parameters: GPUMapColoring generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpumapcoloring-parameters"><code>GPUMapColoring</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUMapColoringParameterValues(settings?)</code>, writes <code>GPU_MAP_COLORING_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUMapColoringParameterValues</code>:</p>
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
<td><code>seed</code> (optional)</td>
<td><code>number</code></td>
<td>Seed of the hashed priorities, a uint32. Default 0. Different seeds give different valid colorings (with <code>'largest-degree-first'</code> the seed only orders rows of equal degree).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};