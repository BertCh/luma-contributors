var e={slug:`api/pointcloud-analysis`,title:`Point cloud analysis`,html:`<h1 id="point-cloud-analysis">Point cloud analysis</h1>
<p><code>@luma-contributors/analysis/pointcloud-analysis</code> provides experimental WebGPU point-cloud
neighborhoods, local shape, terrain surfaces, ground classification, rigid registration, and
surface change as operations in a caller-owned <code>GPUCommandGraph</code>. Voxel downsampling, radius
filtering and octree construction (<code>GPUPointCloudVoxelDownsample</code>, <code>GPUPointCloudRadiusFilter</code>,
<code>GPUPointCloudOctree</code>) are owned by luma.gl: see
<a href="https://luma.gl/docs/api-reference/experimental/gpu-pointcloud"><code>@luma.gl/graph/gpu-pointcloud</code></a>.</p>
<table>
<thead>
<tr>
<th>Operation</th>
<th>Result</th>
<th>Cost</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUPointCloudNeighborhoods</code></td>
<td>Bounded nearest 3D neighbors and actual radius support</td>
<td>With <code>grid</code>: sort + visited candidates × capacity. Without: O(queries × points × capacity), refused above 2^27 query-point pairs</td>
</tr>
<tr>
<td><code>GPUPointCloudFeatures</code></td>
<td>Oriented normals, covariance eigenvalues, shape measures</td>
<td>O(points × capacity)</td>
</tr>
<tr>
<td><code>GPUPointCloudStatisticalFilter</code></td>
<td>Mean neighbor distance and outlier mask</td>
<td>O(points × capacity) + three published reductions</td>
</tr>
<tr>
<td><code>GPUPointCloudSurfaces</code> / <code>GPUPointCloudHeight</code></td>
<td>Indexed DTM/DSM/CHM and cell, bilinear or bounded-IDW HAG</td>
<td>Sort + O(points + cells); HAG O(points × sampling window)</td>
</tr>
<tr>
<td><code>GPUPointCloudGround</code> / <code>GPUPointCloudSMRF</code></td>
<td>Progressive morphology / SMRF ground mask/classification</td>
<td>O(cells × sum(windowArea) + points × stages)</td>
</tr>
<tr>
<td><code>GPUPointCloudICP</code></td>
<td>Point-to-point / point-to-plane rigid transform, robust weights, status/history</td>
<td>With <code>targetGrid</code>: O(iterations × visited candidates + sourcePoints). Without: O(iterations × source × target), refused above 2^27 pairs</td>
</tr>
<tr>
<td><code>GPUPointCloudChange</code></td>
<td>Signed cylinder change, confidence interval and support</td>
<td>With <code>grid</code>: sort + O(cores × points in the cylinder's cells). Without: O(cores × totalPoints), refused above 2^27 core-point pairs</td>
</tr>
<tr>
<td><code>GPUPointCloudVoxelMerge</code></td>
<td>Associative merge of per-part voxel counts, centroids, representatives and attribute means</td>
<td>O(voxels × parts)</td>
</tr>
</tbody>
</table>
<h2 id="runtime-parameter-updates">Runtime parameter updates</h2>
<p>For features, statistical filtering, progressive ground filtering, SMRF, ICP and M3C2 change, import a caller-owned
<code>GPUParameterBuffer</code> from <code>@luma-contributors/analysis/analysis-core</code>, initialize it with the
contributor's <code>getGPUPointCloud*ParameterValues</code> packer, and pass its graph view as <code>parameters</code>.
Writing new packed values before the next submission changes that encoding without rebuilding the
graph or shaders. Parameter views support nonzero byte offsets. Values must satisfy the packer's
limits; do not write unchecked raw settings. Destroy the compiled graph before destroying the buffer.
Constructor threshold/orientation props remain available for existing callers; a supplied parameter
view takes precedence. This does not change output layouts or status values.</p>
<p>Constructors capture declared props and caller-owned <code>output</code> fields, including inherited and
non-enumerable getters. The captured bindings remain fixed after construction; buffer contents
remain per-frame data. This includes voxel-merge output attribute fields and a statistical-filter
<code>standardDeviations</code> value of zero, which retains the global-mean threshold instead of the default
two-deviation threshold.</p>
<h2 id="grid-index-compatibility">Grid-index compatibility</h2>
<p>The local point-cloud index remains necessary with published luma.gl <code>10.0.0-alpha.16</code>:</p>
<ul>
<li>Point-cloud cells use float32 <code>(position - origin) / cellSize</code>; <code>GPUGridIndex</code> normalizes through
inclusive bounds and gridSize, which need not produce the same float32 cell edges.</li>
<li>Point-cloud domains are half-open and exclude the upper edge; the published index includes it.</li>
<li>Point-cloud IDs are ascending within each cell for reproducible sums and tie handling; published
IDs within a cell have unspecified order after atomic scatter.</li>
<li>Raster indexing ignores Z while retaining <code>float32x3</code> positions and supports north/south row
order. The published index requires position and grid dimensions to match and has no raster mode.</li>
</ul>
<p>Published caller-owned <code>cellOffsets</code> and <code>objectIds</code> already provide the needed output views;
output access is not a blocker. Replacing the local helper would require the four semantic
contracts above to match. Existing half-open, ordering and raster-direction specs pin them.</p>
<h2 id="quick-start">Quick start</h2>
<p>Follow the <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">complete GPU walkthrough</a>
for device creation, allocation, imports, submission, typed readback and cleanup. Statistical
filtering consumes neighborhoods rather than raw positions: first run <code>GPUPointCloudNeighborhoods</code>
for your <code>float32x3</code> points, or supply equivalent packed squared distances and counts.</p>
<table>
<thead>
<tr>
<th>Input</th>
<th>Format and rows</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>squaredDistances</code></td>
<td><code>float32</code>, pointCount × capacity</td>
<td>Row-major retained neighbor distances squared, with <code>capacity</code> slots per point.</td>
</tr>
<tr>
<td><code>counts</code></td>
<td><code>uint32</code>, pointCount</td>
<td>Radius support per point; may exceed <code>capacity</code>. The filter uses <code>min(count, capacity)</code> retained slots.</td>
</tr>
<tr>
<td><code>capacity</code></td>
<td>constructor number, 1–64</td>
<td>Slots per point, matching the neighborhood producer.</td>
</tr>
</tbody>
</table>
<p>Add <code>GPUPointCloudStatisticalFilter</code> with these inputs, <code>standardDeviations: 2</code>, and separate
output buffers passed as <code>output: {meanDistances, validity}</code>:</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Format and rows</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>meanDistances</code></td>
<td><code>float32</code>, pointCount</td>
<td>Mean retained neighbor distance; -1 when no neighbors exist.</td>
</tr>
<tr>
<td><code>validity</code></td>
<td><code>uint32</code>, pointCount</td>
<td>1 for inliers, 0 for outliers or rows without neighbors.</td>
</tr>
</tbody>
</table>
<p>For a small filter-only fixture, set <code>capacity: 1</code>, squared distances <code>[1, 1, 100, 0]</code>, and
counts <code>[1, 1, 1, 0]</code>. Mean distances are <code>[1, 1, 10, -1]</code>; the last row is excluded from
the global mean and population standard deviation and has validity 0. With multiplier 2,
the first three rows are inliers, giving validity <code>[1, 1, 1, 0]</code>. Read both outputs using the
walkthrough's lifecycle, then destroy the compiled graph before the imported buffers. A
neighborhood producer's candidate/overflow conditions must be handled before interpreting the
filter as a complete spatial-neighborhood analysis; the filter operates on the supplied slots.</p>
<h2 id="ownership-and-batching">Ownership and batching</h2>
<p>Every contributor takes its caller-owned result views in <code>output</code>, typed as
<code>GPUPointCloud&#x3C;Name>Output</code>. Inputs stay at the top level. For example, neighborhood <code>counts</code>
is <code>output.counts</code> on the producer and becomes the top-level <code>counts</code> input of features:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUPointCloudNeighborhoods,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUPointCloudFeatures</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/pointcloud-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// All views are allocated by the caller on graph.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointCloudNeighborhoods</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, radius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, capacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {indices, squaredDistances, counts, status: neighborhoodStatus}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointCloudFeatures</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, indices, counts, capacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {normals, eigenvalues, shapes, status: featureStatus}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>The <code>Outputs</code> lists below name fields inside <code>output</code>; formats, lengths and status values are
unchanged. Update an output override as <code>{...props, output: {...props.output, normals}}</code> so the
other result views remain present. Ground filtering still receives <code>minimumSurface</code> and <code>coverage</code>
as inputs, and height sampling still receives <code>terrain</code> and <code>coverage</code> as inputs. Voxel merge
keeps each part's statistics in <code>parts</code>, and writes merged statistics to <code>output</code>, including
<code>output.attribute</code> when requested. Statistical filtering publishes flags at <code>output.validity</code>
(also available through <code>contributor.props.output.validity</code>); the former top-level props and
<code>contributor.validity</code> access have been removed.
Authored catalog and showcase examples use the same nested <code>output</code> contract; the catalog examples
are typechecked against the public point-cloud package entry point, separately from its generated
inventory. Their parameter lists expose the output object instead of repeating writable fields as inputs.
Status views keep their producer's row domain: SMRF's <code>smrfStatus</code> has one word per point, while
voxel merge's separate <code>mergedStatus</code> has one word per voxel, even when both run on the same graph.</p>
<p>Inputs and outputs are non-owning <code>GraphDataView</code>s with support for nonzero byte offsets. Output
buffers must be separate from input buffers. Writable output storage bindings must remain disjoint
after their start offsets are rounded down to 256-byte alignment; disjoint logical views alone are
not sufficient. Use separate buffers or leave sufficient aligned space between output views. The caller owns
allocation, graph compilation, encoding, submission, rendering and optional readback. Destroying
compiled kernels does not destroy imported buffers.</p>
<p>Initial operations accept one resident packed view per attribute, fitting device storage-binding
limits. They never concatenate streaming chunks. <code>GPUPointCloudVoxelMerge</code> explicitly merges disjoint owned-row reductions on an identical grid; it never infers overlap or deduplicates halo copies.</p>
<h2 id="contributor-reference">Contributor reference</h2>
<p>One heading per class. All eleven implement <code>GPUCommandNodeProducer</code>, take non-owning packed
<code>GraphDataView</code>s, and refuse bad props at construction with a message that starts with the
contributor ID (default <code>gpu-pointcloud-&#x3C;name></code>) and names the prop and its accepted range. Numeric
props use the per-frame parameter views described below where available. Constructor numeric
props remain compatible defaults when no parameter view is supplied; those defaults are compiled
into the shader. Capacities, view shapes, grid geometry and morphology windows are compile-time.
The limits are exported from <code>pointcloud-analysis/operations/pointcloud-limits.ts</code> as
<code>GPU_POINT_CLOUD_MAXIMUM_*</code>. Status bit and code values are exported as
<code>GPU_POINT_CLOUD_&#x3C;NAME>_STATUS_*</code>.</p>
<h3 id="gpupointcloudneighborhoods">GPUPointCloudNeighborhoods</h3>
<ul>
<li>Props: <code>positions</code> (<code>float32x3</code>), <code>radius</code> (inclusive, positive), <code>capacity</code> (1 to 64,
<code>GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY</code>), optional <code>queryRows</code>, <code>strategy</code> (<code>'serial'</code> default or
<code>'cooperative'</code>), <code>supportBounds</code> (requires <code>status</code>) and <code>grid</code>.</li>
<li>Outputs: <code>indices</code> and <code>squaredDistances</code> (<code>capacity</code> slots per query), <code>counts</code> (full radius
support), optional <code>status</code>.</li>
<li>Status bits: 1 <code>..._NEIGHBORHOODS_STATUS_INVALID_QUERY</code> (query row out of range, not finite or outside
<code>grid</code>; the list is empty), 2 <code>..._TRUNCATED</code> (support above <code>capacity</code>), 4 <code>..._UNSUPPORTED</code>
(the search sphere crosses <code>supportBounds</code>). Bits combine; higher bits are zero.</li>
<li>Limits: <code>grid</code> axes at most 2^24 cells (<code>GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS</code>, exact float32 cell
coordinates), total cells below 2^32, offsets within one storage binding. Without <code>grid</code>,
queries × positions at most 2^27 (<code>GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK</code>).</li>
</ul>
<p>{/* parameters: GPUPointCloudNeighborhoods generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudneighborhoods-parameters"><code>GPUPointCloudNeighborhoods</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudGridParameterValues(settings)</code>, writes <code>GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>gridParameters</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUPointCloudNeighborhoodsParameterValues(settings)</code>, writes <code>GPU_POINT_CLOUD_NEIGHBORHOODS_PARAMETER_LENGTH</code> (9) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointCloudGridParameterValues</code>:</p>
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
<td><code>origin</code></td>
<td><code>readonly [number, number] | readonly [number, number, number]</code></td>
<td>-</td>
</tr>
<tr>
<td><code>cellSize</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUPointCloudNeighborhoodsParameterValues</code>:</p>
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
<td><code>radius</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>supportBounds</code> (optional)</td>
<td><code>GPUPointCloudNeighborhoodsProps['supportBounds']</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudfeatures"><code>GPUPointCloudFeatures</code></h3>
<ul>
<li>Props: <code>positions</code>, <code>indices</code>, <code>counts</code>, <code>capacity</code> (1 to 64), optional <code>viewpoint</code> (finite).</li>
<li>Outputs: <code>normals</code>, <code>eigenvalues</code>, <code>shapes</code> (<code>float32x3</code>), <code>status</code>, optional packed <code>covariance</code>
(6 values per row).</li>
<li>Status bits: 1 <code>..._FEATURES_STATUS_DEGENERATE</code> (fewer than three neighbors, or collinear or
coincident neighbors: normal and shape are zero), 2 <code>..._TRUNCATED</code> (counts above <code>capacity</code>),
4 <code>..._NOT_CONVERGED</code> (eight cyclic Jacobi sweeps, at most 24 pair rotations, were not enough).</li>
<li>Per-frame: optional <code>parameters: float32[4]</code> from <code>getGPUPointCloudFeaturesParameterValues</code>
overrides <code>viewpoint</code>, including enabling/disabling it. Other input contents are per-frame.</li>
<li>Compile-time: capacity, view shapes and optional covariance output. Runtime orientation uses a
second bounded-neighborhood pass so covariance output still fits eight storage bindings.</li>
<li>Limits: capacity 64, per-lane arrays and the shared eigensolve's eight-sweep budget.</li>
</ul>
<p>{/* parameters: GPUPointCloudFeatures generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudfeatures-parameters"><code>GPUPointCloudFeatures</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudFeaturesParameterValues(settings?)</code>, writes <code>GPU_POINT_CLOUD_FEATURES_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointCloudFeaturesParameterValues</code>:</p>
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
<td><code>viewpoint</code> (optional)</td>
<td><code>readonly [number, number, number]</code></td>
<td>Orient normals toward this point; omitted chooses nonnegative z, then y, then x.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudstatisticalfilter"><code>GPUPointCloudStatisticalFilter</code></h3>
<ul>
<li>Props: <code>squaredDistances</code>, <code>counts</code>, <code>capacity</code> (1 to 64), <code>standardDeviations</code> (default 2).</li>
<li>Outputs: <code>meanDistances</code> (<code>-1</code> without support), <code>validity</code> (1 inlier, 0 outlier).</li>
<li>Per-frame: optional <code>parameters: float32[1]</code> from
<code>getGPUPointCloudStatisticalFilterParameterValues</code> overrides <code>standardDeviations</code>.</li>
<li>Compile-time: capacity and view shapes.</li>
<li>Status: none per row. A row without neighbors is an outlier by definition.</li>
<li>Limits: capacity 64. The global statistics are three <code>GPUReduction</code> sums plus one single-row
threshold kernel; no kernel loops over the rows.</li>
</ul>
<p>{/* parameters: GPUPointCloudStatisticalFilter generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudstatisticalfilter-parameters"><code>GPUPointCloudStatisticalFilter</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudStatisticalFilterParameterValues(settings?)</code>, writes <code>GPU_POINT_CLOUD_STATISTICAL_FILTER_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointCloudStatisticalFilterParameterValues</code>:</p>
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
<td><code>standardDeviations</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative multiplier of the population standard deviation; default 2.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudsurfaces">GPUPointCloudSurfaces</h3>
<ul>
<li>Props: <code>positions</code>, <code>classifications</code>, <code>grid</code> (<code>PointCloudRasterGrid</code>), <code>groundClass</code> (default 2).</li>
<li>Outputs: <code>terrain</code>, <code>surface</code>, <code>canopy</code>, <code>coverage</code> (one per cell).</li>
<li>Coverage bits: 1 <code>GPU_POINT_CLOUD_COVERAGE_ANY_POINT</code>, 2 <code>GPU_POINT_CLOUD_COVERAGE_GROUND</code>.</li>
<li>Limits: grid cells below 2^32 (<code>GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS</code>),</li>
<li>Row order: <code>grid.rowDirection</code> is <code>'south'</code> by default (row 0 at the maximum y, rows run toward decreasing y), the order of the raster and terrain families (GDAL and image order, <code>GPURasterMetadata</code>, D8 flow codes), so these rasters feed <code>GPUTerrainFlow</code>, the derivatives and the illumination contributors without flipping rows. Pass <code>'north'</code> for row 0 at the minimum y. Unlike the terrain classes, <code>PointCloudRasterGrid</code> has no affine or signed cell height, so the direction cannot be derived from the grid and a contradiction cannot be detected: give every point-cloud contributor that shares the grid the same direction. <code>GPUPointCloudSurfaces</code>, <code>GPUPointCloudHeight</code>, <code>GPUPointCloudGround</code> and <code>GPUPointCloudSMRF</code> honour it; a point on a cell boundary belongs to the cell above in either order. Membership is the same half-open test in both orders, on <code>o = (y - origin.y) / cellSize</code>: the point is inside when <code>0 &#x3C;= o &#x3C; height</code>, in north row <code>floor(o)</code> or south row <code>height - 1 - floor(o)</code>. Before this change the default was <code>'north'</code>, so a caller that left the prop out now gets the rows mirrored.</li>
</ul>
<p>{/* parameters: GPUPointCloudSurfaces generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudsurfaces-parameters"><code>GPUPointCloudSurfaces</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudGridParameterValues(settings)</code>, writes <code>GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>gridParameters</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUPointCloudSurfacesParameterValues(settings?)</code>, writes <code>GPU_POINT_CLOUD_SURFACES_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointCloudGridParameterValues</code>:</p>
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
<td><code>origin</code></td>
<td><code>readonly [number, number] | readonly [number, number, number]</code></td>
<td>-</td>
</tr>
<tr>
<td><code>cellSize</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUPointCloudSurfacesParameterValues</code>:</p>
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
<td><code>groundClass</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudheight">GPUPointCloudHeight</h3>
<p>Height queries and aligned outputs accept chunked GraphVectorView input. Height sampling settings
are runtime inputs; grid axes and float32 support bounds are validated at construction.</p>
<ul>
<li>Props: <code>positions</code>, <code>grid</code>, <code>terrain</code>, <code>coverage</code>, <code>coverageBit</code>, <code>interpolation</code> (<code>bilinear</code> or
<code>idw</code> with <code>radius</code>, <code>power</code>, <code>minimumSupport</code>).</li>
<li>Outputs: <code>heights</code>, <code>status</code>.</li>
<li>Status codes (values, not bits): 0 <code>GPU_POINT_CLOUD_HEIGHT_STATUS_VALID</code>, 1 <code>..._INVALID_POINT</code>,
2 <code>..._UNSUPPORTED_TERRAIN</code>, 3 <code>..._NONFINITE</code>.</li>
<li>Limits: IDW radius at most 64 cells (<code>GPU_POINT_CLOUD_MAXIMUM_IDW_REACH_CELLS</code>, each point visits that many cells squared),
power in (0, 8] (<code>GPU_POINT_CLOUD_MAXIMUM_IDW_POWER</code>, higher powers underflow the normalized weights).</li>
</ul>
<p>{/* parameters: GPUPointCloudHeight generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudheight-parameters"><code>GPUPointCloudHeight</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudGridParameterValues(settings)</code>, writes <code>GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>gridParameters</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUPointCloudHeightParameterValues(settings, grid)</code>, writes <code>GPU_POINT_CLOUD_HEIGHT_PARAMETER_LENGTH</code> (5) <code>uint32</code> values.
<ul>
<li><code>grid</code> (<code>PointCloudRasterGrid</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUPointCloudGridParameterValues</code>:</p>
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
<td><code>origin</code></td>
<td><code>readonly [number, number] | readonly [number, number, number]</code></td>
<td>-</td>
</tr>
<tr>
<td><code>cellSize</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUPointCloudHeightParameterValues</code>:</p>
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
<td><code>coverageBit</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>interpolation</code> (optional)</td>
<td><code>GPUPointCloudHeightProps['interpolation']</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudground"><code>GPUPointCloudGround</code></h3>
<ul>
<li>Props: <code>positions</code>, <code>classifications</code>, <code>grid</code>, <code>minimumSurface</code>, <code>coverage</code>, <code>radii</code>,
<code>initialThreshold</code>, <code>slope</code>, <code>maximumThreshold</code>.</li>
<li>Per-frame: optional <code>parameters: float32[radii.length]</code> from
<code>getGPUPointCloudGroundParameterValues(settings, grid, radii)</code> overrides all three threshold props.
The packer computes each stage threshold in float64 and rounds once, retaining the existing
float32 boundary behavior. Use the same grid and radii passed to the contributor.</li>
<li>Compile-time: grid, radii and view shapes.</li>
<li>Outputs: <code>groundMask</code>, <code>outputClassifications</code>, <code>status</code>.</li>
<li>Status bits: 1 <code>..._GROUND_STATUS_UNSUPPORTED</code> (outside the grid, not finite or uncovered cell),
2 <code>..._INCOMPLETE_WINDOW</code> (the final window reaches past covered cells). Bit 1 excludes bit 2.</li>
<li>Limits: 1 to 32 radii (<code>GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_COUNT</code>, three kernels each), each radius 1 to 32
cells (<code>GPU_POINT_CLOUD_MAXIMUM_GROUND_RADIUS_CELLS</code>, each cell scans (2r+1)² neighbors).</li>
</ul>
<p>{/* parameters: GPUPointCloudGround generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudground-parameters"><code>GPUPointCloudGround</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudGridParameterValues(settings)</code>, writes <code>GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>gridParameters</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUPointCloudGroundParameterValues(settings, grid, radii)</code>, writes <code>getGPUPointCloudGroundParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>PointCloudRasterGrid</code>): The contributor's compile-time grid.</li>
<li><code>radii</code> (<code>readonly number[]</code>): The contributor's compile-time radii, in cells.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUPointCloudGridParameterValues</code>:</p>
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
<td><code>origin</code></td>
<td><code>readonly [number, number] | readonly [number, number, number]</code></td>
<td>-</td>
</tr>
<tr>
<td><code>cellSize</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUPointCloudGroundParameterValues</code>:</p>
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
<td><code>initialThreshold</code></td>
<td><code>number</code></td>
<td>Nonnegative elevation threshold added at every stage, in z units.</td>
</tr>
<tr>
<td><code>slope</code></td>
<td><code>number</code></td>
<td>Nonnegative terrain slope per cell of radius increment.</td>
</tr>
<tr>
<td><code>maximumThreshold</code></td>
<td><code>number</code></td>
<td>Nonnegative upper threshold, at least initialThreshold.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudsmrf"><code>GPUPointCloudSMRF</code></h3>
<ul>
<li>Props: <code>positions</code>, <code>classifications</code>, <code>grid</code> (at least 2 by 2), <code>selection</code>, <code>slope</code>, <code>scalar</code>,
<code>threshold</code>, <code>window</code>, <code>cut</code>, <code>groundClass</code>, <code>otherClass</code>, <code>onlyGround</code>.</li>
<li>Per-frame: optional <code>parameters: float32[2 + maximumRadius]</code> from
<code>getGPUPointCloudSMRFParameterValues(settings, grid, maximumRadius)</code> overrides <code>slope</code>, <code>scalar</code>
and <code>threshold</code>. The packer rounds per-radius slope products once, matching constructor thresholds.
Pass the contributor's <code>maximumRadius</code> and grid.</li>
<li>Compile-time: grid, window, cut, class labels, onlyGround, optional selection and view shapes.</li>
<li>Outputs: <code>groundSurface</code> (per cell), <code>groundMask</code>, <code>outputClassifications</code>, <code>status</code> (per point).</li>
<li>Status bits: 1 <code>..._SMRF_STATUS_INVALID</code>, 2 <code>..._NO_TERRAIN</code>, 4 <code>..._EXCLUDED</code>, 8 <code>..._INPAINTED</code>
(inpainted terrain, or within the filter margin of the grid edge). Bits 1, 2 and 4 are exclusive
and exclude bit 8.</li>
<li>Limits: window at most 64 cells (<code>GPU_POINT_CLOUD_MAXIMUM_SMRF_WINDOW_CELLS</code>; the openings are unrolled
into published <code>GPURasterErosion</code> and <code>GPURasterDilation</code> passes of radius at most 8, about 50 passes at
the default 18 and 350 at 64), cut at most 32 cells
(<code>GPU_POINT_CLOUD_MAXIMUM_SMRF_CUT_CELLS</code>). The inpainting valid counts are published <code>GPUReduction</code> sums. The
inpainting ring search is still a per-cell loop that can reach <code>max(width, height)</code> rings on sparse coverage.</li>
</ul>
<p>{/* parameters: GPUPointCloudSMRF generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudsmrf-parameters"><code>GPUPointCloudSMRF</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudSMRFParameterValues(settings, grid, maximumRadius)</code>, writes <code>getGPUPointCloudSMRFParameterLength(...)</code> <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>PointCloudRasterGrid</code>): The contributor's compile-time grid.</li>
<li><code>maximumRadius</code> (<code>number</code>): The contributor's compile-time maximumRadius, in cells.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUPointCloudSMRFParameterValues</code>:</p>
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
<td><code>slope</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative terrain slope; default 0.15.</td>
</tr>
<tr>
<td><code>scalar</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative multiplier of the terrain gradient; default 1.25.</td>
</tr>
<tr>
<td><code>threshold</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative elevation threshold in z units; default 0.5.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudicp"><code>GPUPointCloudICP</code></h3>
<ul>
<li>Props: <code>source</code>, <code>target</code>, <code>maximumDistance</code>, <code>iterations</code> (1 to 100, default 20,
<code>GPU_POINT_CLOUD_MAXIMUM_ICP_ITERATIONS</code>), <code>targetGrid</code>, <code>metric</code>, <code>targetNormals</code>, <code>robust</code>,
<code>initialTransform</code>, <code>origin</code>, tolerances.</li>
<li>Per-frame: optional <code>parameters: float32[22]</code> from <code>getGPUPointCloudICPParameterValues(settings, origin)</code>
overrides maximum distance, robust scale, both tolerances and the initial transform. Pass the
same origin as the contributor so the packer converts absolute transforms to its local frame.
Set the packer's <code>robustMethod</code> to the contributor's compile-time <code>robust.method</code> (the packer
defaults to <code>'tukey'</code> to preserve earlier validation). Huber accepts every positive finite
float32 scale; only Tukey requires a finite, nonzero float32 square. When a Huber square is out
of range, the unused square slot is 1; the unsquared scale retains its requested float32 value.</li>
<li>Compile-time: iterations, metric, robust method (including whether it is enabled), targetGrid,
origin and view shapes.</li>
<li>Outputs: <code>transform</code> (16), <code>history</code> (one <code>float32x3</code> per iteration), <code>inlierMask</code>, <code>status</code> (1 row).</li>
<li>Status codes (values, not bits): 0 <code>..._ICP_STATUS_CONVERGED</code>, 1 <code>..._ITERATION_LIMIT</code>,
2 <code>..._INSUFFICIENT_MATCHES</code>, 3 <code>..._DEGENERATE</code>, 4 <code>..._EIGENSOLVE_EXHAUSTED</code>.</li>
<li>Limits: iterations are unrolled graph nodes. Without <code>targetGrid</code>, source × target at most 2^27.</li>
</ul>
<p>{/* parameters: GPUPointCloudICP generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudicp-parameters"><code>GPUPointCloudICP</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudICPParameterValues(settings, origin?)</code>, writes <code>GPU_POINT_CLOUD_ICP_PARAMETER_LENGTH</code> (22) <code>float32</code> values.
<ul>
<li><code>origin</code> (<code>readonly [number, number, number]</code>): The contributor's compile-time local origin, when present.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUPointCloudICPParameterValues</code>:</p>
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
<td><code>maximumDistance</code></td>
<td><code>number</code></td>
<td>Positive inclusive correspondence distance with a finite, nonzero float32 square.</td>
</tr>
<tr>
<td><code>translationTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative translation-step tolerance; default 1e-5.</td>
</tr>
<tr>
<td><code>rotationTolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative rotation-column-change tolerance; default 1e-5.</td>
</tr>
<tr>
<td><code>robustScale</code> (optional)</td>
<td><code>number</code></td>
<td>Positive finite float32 scale; default 1. Only Tukey also requires a finite nonzero square.</td>
</tr>
<tr>
<td><code>robustMethod</code> (optional)</td>
<td><code>'huber' | 'tukey'</code></td>
<td>Match the contributor's compile-time robust.method; default tukey preserves earlier validation.</td>
</tr>
<tr>
<td><code>initialTransform</code> (optional)</td>
<td><code>readonly number[]</code></td>
<td>Absolute column-major rigid initial transform; default identity.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudchange"><code>GPUPointCloudChange</code></h3>
<ul>
<li>Props: <code>reference</code>, <code>comparison</code>, <code>cores</code>, <code>normals</code>, <code>alignment</code>, <code>radius</code>, <code>halfDepth</code>,
<code>minimumSupport</code>, uncertainty options, optional <code>grid</code>.</li>
<li>Per-frame: optional <code>parameters: uint32[12]</code> from <code>getGPUPointCloudChangeParameterValues</code>
overrides minimum support, registration error/covariance, confidence multiplier and the supplied
sampling covariance values. Word 0 holds exact uint32 support; other words carry float32 bit
patterns. Covariance model presence and uncertainty combination stay compile-time.</li>
<li>Compile-time: radius, halfDepth (bounded cylinder candidate loops), grid, optional covariance
model, uncertainty combination, alignment-view presence and view shapes. Alignment transform
contents are already per-frame. With parameters, one extra per-core pass projects registration
covariance and grid support into a <code>float32x3</code> scratch view, keeping every stage at eight or fewer
storage bindings.</li>
<li>Outputs: <code>changes</code>, <code>roughness</code> (<code>float32x3</code> per core), <code>counts</code> (two per core), <code>status</code>.</li>
<li>Status codes (values, not bits): 0 <code>..._CHANGE_STATUS_VALID</code>, 1 <code>..._INVALID_CORE</code>,
2 <code>..._INSUFFICIENT_SUPPORT</code>, 3 <code>..._NONFINITE_UNCERTAINTY</code>. In grid mode the flag
<code>GPU_POINT_CLOUD_CHANGE_STATUS_GRID_INCOMPLETE</code> (0x100) is OR-ed onto any code except
<code>..._INVALID_CORE</code> when a cell the core's cylinder needs lies outside the grid, so its counts may
be incomplete; mask it off to read the code.</li>
<li>Grid limit: each core visits <code>(ceil(2 * sqrt(halfDepth^2 + radius^2) / cellSize) + 3)^3</code> cells, at
most <code>GPU_POINT_CLOUD_MAXIMUM_CHANGE_CELLS_PER_CORE</code> (2^20, about 4,800 times the largest use in
the tests); the constructor asks for a larger <code>cellSize</code> above it.</li>
<li>Limits: without <code>grid</code>, cores × (reference + comparison) at most 2^27; with <code>samplingCovariance</code>,
at most 32768 points per cloud (<code>GPU_POINT_CLOUD_MAXIMUM_SAMPLING_COVARIANCE_POINTS</code>).</li>
<li>Grid path: pass a <code>grid</code> of the same kind <code>GPUPointCloudNeighborhoods</code> takes. The reference and
the aligned comparison are indexed once per encoding, and each core visits only the cells that the
bounding box of its cylinder touches (half-depth along the normal, radius across it, plus one cell of margin).
Counts and statuses equal the unindexed scan exactly; changes and roughness equal it to float32
summation order (the tests compare to four decimals on the PDAL golden fixture and on a random cloud
with tilted normals, alignment and sampling covariance). Points outside the half-open grid do not
participate, so the grid must cover both clouds after alignment. The work is one kernel per cloud and a final
combining kernel in both modes.</li>
</ul>
<p>{/* parameters: GPUPointCloudChange generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpupointcloudchange-parameters"><code>GPUPointCloudChange</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUPointCloudChangeParameterValues(settings?)</code>, writes <code>GPU_POINT_CLOUD_CHANGE_PARAMETER_LENGTH</code> (12) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointCloudChangeParameterValues</code>:</p>
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
<td><code>minimumSupport</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>registrationError</code> (optional)</td>
<td><code>number</code></td>
<td>Nonnegative registration uncertainty in coordinate units, added to the sampling standard error.</td>
</tr>
<tr>
<td><code>registrationCovariance</code> (optional)</td>
<td><code>readonly [number, number, number, number, number, number]</code></td>
<td>Optional aligned-frame translation covariance [xx,xy,xz,yy,yz,zz]; projected along each normal.</td>
</tr>
<tr>
<td><code>confidenceMultiplier</code> (optional)</td>
<td><code>number</code></td>
<td>Two-sided normal quantile, default1.96 (95%); caller specifies their confidence convention.</td>
</tr>
<tr>
<td><code>samplingCovariance</code> (optional)</td>
<td><code>{ referenceVariance: number; comparisonVariance: number; correlationLength: number; }</code></td>
<td>Optional known covariance of normal-direction measurement error: sigma² exp(-distance²/(2 length²)). Uses all within-cylinder pairs exactly; variances are supplied separately from empirical roughness.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpupointcloudvoxelmerge">GPUPointCloudVoxelMerge</h3>
<ul>
<li>Props: <code>parts</code> (counts, centroids, representatives, <code>sourceRowOffset</code>, optional attribute),
<code>partitionPolicy: 'disjoint-owned-rows'</code>, merged outputs and optional <code>attribute</code>.</li>
<li>Outputs: <code>counts</code>, <code>centroids</code>, <code>representatives</code>, <code>status</code>, optional attribute means and counts.</li>
<li>Status bits: 1 <code>..._VOXEL_MERGE_STATUS_COUNT_OVERFLOW</code> (merged count above uint32),
2 <code>..._REPRESENTATIVE_OVERFLOW</code> (representative plus offset above the largest row ID). Reject flagged voxels.</li>
<li>Limits: part <code>sourceRowOffset</code> below 2^32 - 1. No published or local contributor produces the
<code>centroids</code> input yet; the tests build the parts by hand.</li>
</ul>
<h2 id="3d-neighborhoods-normals-and-shape">3D neighborhoods, normals and shape</h2>
<p><code>GPUPointCloudNeighborhoods</code> accepts packed Cartesian <code>float32x3</code> positions. Distances use all
three axes. <code>radius</code> is inclusive; each row excludes itself, while separate duplicate rows count.
<code>capacity</code> is 1–64. Allocate <code>indices: uint32[N * capacity]</code>,
<code>squaredDistances: float32[N * capacity]</code>, and <code>counts: uint32[N]</code>. Neighbors sort by squared
distance, breaking ties by original row ID. Missing slots have ID <code>0xffffffff</code> and maximum finite
float32 distance. <code>counts</code> contains the <strong>full radius support</strong>, so <code>counts > capacity</code> explicitly
identifies truncation. A finite radius covering the cloud gives bounded k-nearest neighbors.
Nonfinite points do not participate. This fixed-stride representation is not CSR.</p>
<p><code>strategy: 'serial'</code> is the default. For dense candidate sets, <code>strategy: 'cooperative'</code> assigns
one workgroup to each query, scans candidates across its lanes and merges local nearest-neighbor
lists. Both strategies preserve full support counts, distance/ID ordering, radius boundaries and
status bits, with or without a grid. Cooperative selection uses at most 8,704 bytes of workgroup
storage and adds no global partial-list buffer. It can cost more on sparse queries; compare the
complete grid-and-query workflow before choosing it. Strategy selection is explicit.</p>
<p><code>GPUPointCloudFeatures</code> consumes those three neighborhood fields and the original positions.
Allocate <code>normals</code>, <code>eigenvalues</code>, and <code>shapes</code> as <code>float32x3[N]</code>, and <code>status</code> as <code>uint32[N]</code>.
Optional <code>covariance: float32[N * 6]</code> exposes population moments in xx, xy, xz, yy, yz, zz order.
It centers coordinates relative to each query before accumulating population covariance, then
uses the shared symmetric eigensolver with at most eight cyclic sweeps (24 pair rotations) and
relative tolerance 1e-6. The solver returns ascending eigenpairs internally; this contributor keeps
its descending output order and the existing degeneracy thresholds. Eigenvalues are descending. Shape components are
<code>(largest-middle)/largest</code>, <code>(middle-smallest)/largest</code>, and <code>smallest/largest</code>.
Normals are unit vectors toward the supplied <code>viewpoint</code>; without one, z is nonnegative, then y,
then x at exact ties. This is a local orientation convention, not globally consistent orientation.</p>
<p>Feature status is a bitmask: 1 means fewer than three retained neighbors or a rank-deficient
neighborhood; 2 means radius support exceeded capacity; 4 means the eigensolver exhausted its
bound. Degenerate neighborhoods produce a zero normal and zero shape. Check status before use.
Without a <code>grid</code> the search costs O(N² × capacity) and uses O(N × capacity) output storage; the
constructor refuses queries × positions above <code>GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK</code> (2^27) and
asks for a <code>grid</code>. The optional
<code>grid: {origin: [x,y,z], cellSize, dimensions: [width,height,depth]}</code> adds a GPU sorted 3D grid.
The grid bounds are half-open: points outside them do not participate. Grid key sorting, cell offset
construction, candidate search and features stay in the graph and rebuild on each encoding.
The exact distance test and deterministic nearest-ID order are identical to the reference path.
A conservative extra cell around the query protects inclusive boundaries from float32 bin rounding.
Optional neighborhood <code>status: uint32[N]</code> reports bit 1 for invalid/outside-grid queries and bit 2
for truncation. Choose grid bounds that cover the source and any required tile halos.</p>
<p>Indexed work is the GPU sort plus O(cells × log(points)) offset construction and the visited
candidate pairs × capacity; dense/skewed cells can still approach quadratic work. Scratch is
O(points + cells). The octree does not participate in this grid. The opt-in scaled benchmark
measures the complete indexed-neighborhood-plus-features graph rather than the eigensolve alone.</p>
<h2 id="statistical-filtering-and-voxel-averaging">Statistical filtering and voxel averaging</h2>
<p><code>GPUPointCloudStatisticalFilter</code> consumes neighborhood <code>squaredDistances</code>, <code>counts</code>, and <code>capacity</code>.
It writes the mean distance to retained neighbors (<code>-1</code> without support), and a row-preserving
<code>uint32</code> mask. Its threshold is the population mean of supported rows' mean distances plus
<code>standardDeviations</code> (default 2) times their population standard deviation. Zero-support rows are
rejected. Radius truncation remains observable through the input counts. The implementation is
O(N × capacity) for neighbor means; the valid count, mean and variance are three published
<code>GPUReduction</code> sums, each term divided by the valid count before it is summed so that a finite
distance cannot overflow the accumulated moment. It performs no compaction or attribute rewriting.
A point exactly at the threshold can classify differently across devices, because the sums are
tree-ordered and the reduction strategy depends on the device.</p>
<p>The published <code>GPUPointCloudVoxelDownsample</code> (<code>@luma.gl/graph/gpu-pointcloud</code>, 10.0.0-alpha.10)
takes <code>positions</code>, <code>origin</code>, <code>voxelSize</code> and <code>gridSize</code> and writes <code>representatives</code> (the smallest
source row ID per voxel), <code>counts</code> and <code>output</code> (the representative's position). It has no
<code>centroids</code> and no <code>attribute</code> option: an earlier version of this page described both, and they
are not in the published release. Gather sampled attributes using the representative IDs.</p>
<p><code>GPUPointCloudVoxelMerge</code> needs a per-voxel <code>centroids</code> input (the arithmetic mean of the member
positions) and, optionally, per-voxel attribute means with their own counts. Nothing published or
local produces them, so today the caller computes them per part: on the CPU while the part is
built, or on the GPU as a keyed mean, with the voxel index
<code>(z * gridSize[1] + y) * gridSize[0] + x</code> of <code>floor((position - origin) / voxelSize)</code> as a dense
<code>uint32</code> key and the coordinates (and attribute components) as columns, for example with
<code>GPUGroupStatistics</code> and <code>keyCount = voxelCount</code>. Empty voxels must hold count 0. Do not pass the
published <code>output</code> as <code>centroids</code>: the merge would then return a count-weighted mean of
representative points, which is not the centroid. Voxel centroids and attribute means on
<code>GPUPointCloudVoxelDownsample</code> are an upstream gap. <code>GPUPointCloudVoxelMerge</code> combines disjoint partition reductions as described below. Applications must supply complete
neighborhood halos for tiled filtering; per-tile statistical thresholds are different from a
global-cloud threshold.</p>
<p>The website voxel-merge example and lidar support showcase compute each tile's arithmetic means
with <code>computePointCloudVoxelMeans(ownedPositions, voxelSize, gridSize, origin)</code> from the public
point-cloud entry point. This helper accepts CPU-owned float32 XYZ rows and returns dense float32
voxel means in the same coordinate frame; empty voxels contain zero. The helper rejects trailing
partial XYZ rows and voxel sizes that are not positive and finite after float32 rounding
(including zero, underflow and overflow). The helper also requires three positive safe-integer
grid dimensions, a safe-integer total cell
count below 4,294,967,295, and three origin coordinates finite after float32 rounding, matching
<code>GPUPointCloudVoxelDownsample</code>. Invalid grids and origins are rejected before allocation.
Upload those means separately from the downsampler's selected positions. The merge
consumes GPU counts and representative rows. Feature and change displays retain their status,
suppress unusable measurements, and explicitly label truncated or insufficient support.</p>
<p>Height displays accept status 0 only: unsupported terrain (status 2) is shown as unavailable,
distinct from a supported height of zero. ICP displays show aligned geometry only for status 0
(converged); an exhausted budget or failed fit is labelled and the alignment is hidden. Ground
and SMRF displays exclude every nonzero-status classification from ground counts and colored
point geometry, and report the number of incomplete or unsupported points. The SMRF raster is
explicitly provisional, including its inpainted terrain.</p>
<h2 id="terrain-surfaces-and-height-above-ground">Terrain surfaces and height above ground</h2>
<p><code>GPUPointCloudSurfaces</code> uses a caller-supplied planar grid (<code>origin: [x, y]</code>, <code>cellSize</code>, <code>width</code>,
<code>height</code>) and one <code>uint32</code> classification per point. Cells are row-major with x varying fastest;
bounds include the minimum and exclude the maximum. Each cell produces:</p>
<ul>
<li><code>terrain</code>: minimum elevation among <code>groundClass</code> points (default LAS class 2).</li>
<li><code>surface</code>: maximum elevation among all finite points.</li>
<li><code>canopy</code>: surface minus terrain where both exist.</li>
<li><code>coverage</code>: bit 1 has any point; bit 2 has classified ground.</li>
</ul>
<p>Outputs are zero where unsupported, so always consume coverage. A sorted cell index makes the reduction O(points + cells) after sorting.</p>
<p>When using classified-ground Surfaces terrain to seed Ground, first derive a separate nonzero
seed mask from <code>(coverage &#x26; GPU_POINT_CLOUD_COVERAGE_GROUND) != 0</code>; occupancy bit 1 alone is
insufficient. The lidar-ground showcase uses this mask and suppresses class-2 classifications
with any nonzero Ground status before the second Surfaces pass. Thus absent seeds and incomplete
morphology support cannot become recovered terrain. A filter deliberately initialized from all
returns can instead prepare an unclassified minimum surface with every point eligible.</p>
<p>The showcase uses a radius-one window on its sparse class-2 seed grid so that some recovered
terrain remains fully supported. Its ground colors and legend distinguish supported acceptance,
supported rejection, provisional incomplete windows, and unsupported/invalid rows. In particular,
<code>groundMask = 1</code> with status 2 is provisional, never accepted ground for the height pipeline.
Surfaces, Ground and Height capture nested grid settings (including a copy of the origin) at construction;
Ground also copies the radii and Height captures interpolation settings. Mutating caller settings
later does not change the declared passes or sampling frame. Buffer contents and explicit runtime
parameter buffers remain the per-frame inputs.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUPointCloudSurfaces, GPUPointCloudHeight, GPU_POINT_CLOUD_COVERAGE_GROUND}</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/pointcloud-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointCloudSurfaces</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, classifications, grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {terrain, surface, canopy, coverage}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUPointCloudHeight</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, grid, terrain, coverage,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  coverageBit: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">GPU_POINT_CLOUD_COVERAGE_GROUND</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {heights, status: heightStatus}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>GPUPointCloudHeight</code> samples the containing terrain cell and subtracts its elevation from each
point's z. Pass the same caller-owned <code>coverage</code> view as Surfaces' <code>output.coverage</code> and Height's
<code>coverage</code> input, with <code>coverageBit: 2</code> to require classified-ground support; otherwise any
nonzero bit is accepted. Heights remain signed. Status is 0 for a supported sample, 1 for an invalid
or outside point, 2 for unsupported terrain, and 3 for a nonfinite interpolated elevation or height
difference. The displayed lidar pipeline uses ground-only <code>groundSeedCoverage</code> for classification
and <code>coverageBit: 2</code> for height sampling, matching the live scene: coverage word 1 alone cannot
establish ground support. Every invalid result is zero and must be masked by status. The default is <strong>cell-constant raster sampling</strong>; optional bilinear and bounded IDW sampling are described below. Direct nearest-ground and Delaunay interpolation are not provided. Inputs and
outputs retain the caller's units and vertical datum; no datum transform is performed.</p>
<h2 id="progressive-morphological-ground-classification">Progressive morphological ground classification</h2>
<p><code>GPUPointCloudGround</code> consumes an <strong>unclassified cell-minimum surface</strong>, its nonzero coverage,
original point positions and original classifications. Use <code>GPUPointCloudSurfaces</code> with an
all-equal classification input and that value as <code>groundClass</code> to prepare such a minimum surface.
<code>radii</code> is a strictly increasing array of 1–32 cell radii. Each stage opens the original minimum
surface using square erosion then dilation, and rejects points whose elevation above the opening
exceeds <code>min(maximumThreshold, initialThreshold + slope * cellSize * radiusIncrement)</code>, where
<code>radiusIncrement</code> is <code>radii[k] - radii[k - 1]</code> and the first stage uses <code>radii[0]</code>.
A point rejected at an earlier stage stays rejected. Negative residuals remain ground candidates.</p>
<p>This threshold is a convention of this class, not equation 7 of Zhang et al. (2003). Zhang uses
<code>initialThreshold</code> alone while the window is at most 3 cells wide, and
<code>initialThreshold + slope * (w[k] - w[k - 1]) * cellSize</code> afterwards, with the full window width
<code>w = 2 * radius + 1</code>. So the first stage here already adds <code>slope * cellSize * radii[0]</code>, and at
later stages the same <code>slope</code> is half as permissive, because a radius increment is half a
window-width increment: pass twice the slope to match Zhang's later stages. With <code>radii: [1, 2]</code>,
<code>cellSize: 1</code>, <code>slope: 0.1</code> and <code>initialThreshold: 0.2</code> the thresholds here are 0.3 and 0.3;
Zhang's are 0.2 and 0.4. No PCL or PDAL <code>filters.pmf</code> fixture pins this class.</p>
<p>The output <code>groundMask</code> identifies survivors. <code>outputClassifications</code> sets survivors to LAS ground
class 2, changes rejected former ground points to unclassified class 1, and retains other rejected
classes. The original classification input is never changed. Status bit 1 means outside/unsupported;
bit 2 means incomplete morphological support, including holes or grid edges. Incomplete windows
still use available samples; treat the flag according to the application's edge policy. The
algorithm is a documented progressive square-window morphological filter, <strong>not PDAL SMRF parity</strong>.
SMRF slope/scalar/cut semantics, hole filling, automatic cell sizing, and authoritative ground truth
are outside this implementation. Work is O(cells × sum(windowArea) + points × stages).</p>
<h2 id="simple-morphological-filtering-smrf">Simple morphological filtering (SMRF)</h2>
<p><code>GPUPointCloudSMRF</code> separately implements the SMRF stages described by
<a href="https://github.com/PDAL/PDAL/blob/2.8.4/filters/SMRFilter.cpp">PDAL 2.8.4 SMRFilter</a>, using
<a href="https://github.com/PDAL/PDAL/blob/2.8.4/pdal/private/MathUtils.hpp">PDAL's diamond morphology and finite differences</a>.
It accepts <code>positions</code>, original <code>classifications</code>, optional <code>selection</code>, and a caller-owned grid.
For the PDAL grid alignment, use the selected cloud's minimum XY as origin, <code>cellSize</code> as PDAL's
<code>cell</code>, and <code>floor(extent / cellSize) + 1</code> dimensions. Dimensions must be at least 2 × 2.</p>
<p>The parameters preserve SMRF meanings: <code>slope</code> defaults to 0.15; <code>scalar</code> to 1.25; <code>threshold</code> to
0.5; <code>window</code> to 18 times <code>cellSize</code>; and <code>cut</code> to zero. <code>window</code> is a physical radius, converted
with <code>ceil(window / cellSize)</code>, bounded to 64 cells. Nonzero <code>cut</code> is physical net spacing, rounded
up to at most 32 cells. Unlike the square-window PMF above, this operation:</p>
<ol>
<li>Rasterizes selected points to cell minima and fills empty cells from eight nearest valid cells.</li>
<li>Detects low outlier cells by opening the inverted surface with a one-cell diamond and a 5 × cell
elevation threshold.</li>
<li>Optionally cuts a grid net using a diamond opening of radius twice the net spacing.</li>
<li>Erodes progressively by one cell and dilates to each radius. It flags differences from the
<strong>previous opening</strong> above <code>slope * cellSize * radius</code>.</li>
<li>Removes object, low-outlier and net cells, then fills the provisional DEM from eight nearest
remaining cells.</li>
<li>Computes central interior and one-sided edge gradients, and accepts absolute point-to-DEM
residuals up to <code>threshold + scalar * gradientMagnitude</code>.</li>
</ol>
<p>Allocate <code>groundSurface: float32[width * height]</code>, <code>groundMask: uint32[N]</code>,
<code>outputClassifications: uint32[N]</code>, and <code>status: uint32[N]</code>. <code>groundClass</code> defaults to 2 and
<code>otherClass</code> to 1. <code>onlyGround: true</code> preserves rejected original classes. Original input classes
are always retained separately. Zero <code>selection</code> rows preserve their original class and never
contribute to the terrain; construct this mask explicitly for return-number, class-bit and ignored
range policies. No LAS return inference or implicit point deletion occurs.</p>
<p>Status is 0 for supported interior samples, 1 for invalid/outside points, 2 when no terrain anchors
survive, 4 for excluded rows, and 8 for inpainted/edge-supported results. Code 8 still has a valid
classification under clipped-window semantics; it exposes the support limitation. Unsupported
results retain original classifications and have zero ground masks and zero terrain. This is a
classification estimate, not a claim of universal ground truth.</p>
<p>Eight-neighbor fill ties use column-major source-cell order deterministically. PDAL's KD-tree tie
ordering is not a portable public contract; equally distant cutoff ties may select different
anchors. The independent SciPy oracle covers holes, low outliers, slope scaling, scalar zero and
net cutting. Its checked-in generator optionally invokes the actual PDAL CLI for differential
classifications. Arithmetic is float32, so threshold-boundary decisions can differ from PDAL's
double-precision implementation. Inpainting expands square rings and stops only after every cell capable of beating or tying the eighth distance is visited. It is exact and remains quadratic for large unsupported regions; completely anchorless grids are detected once and bypass searching. Diamond morphology delegates to the published <code>GPURasterErosion</code> and <code>GPURasterDilation</code> (<code>@luma.gl/graph/gpu-raster</code>) with the <code>'cross'</code> structuring element, a <code>'nodata'</code> border and the <code>'ignore'</code> policy, so cells outside the grid do not take part. One pass covers a radius of at most 8 and a larger diamond is a sequence of passes, which preserves clipped diamond semantics exactly (a diamond is a Minkowski sum of smaller diamonds). A headless spec compares the windows with a full diamond scan for radii 1 to 37. Cell minima use a sorted index rather than scanning every point per cell. Provide full
halos for tiled use: silently classifying independently cropped tiles is not partition-equivalent.</p>
<h2 id="bounded-rigid-registration">Bounded rigid registration</h2>
<p><code>GPUPointCloudICP</code> estimates a source-to-target rigid transform using point-to-point ICP. A caller
may supply a column-major rigid <code>initialTransform</code>; identity is the default. Every iteration finds
exact nearest targets within inclusive <code>maximumDistance</code>, accumulates centered correspondences,
and solves Horn's quaternion system using the shared symmetric eigensolver with at most eight
cyclic sweeps (48 pair rotations) and relative tolerance 1e-6. The largest ascending eigenpair
supplies the quaternion; the existing eigenvalue-gap degeneracy check and status codes remain. Optional <code>robust: {method: 'huber' | 'tukey', scale}</code> applies distance-based iteratively reweighted least squares. Tukey rejects residuals at or above its scale; Huber downweights them. Reflection and scale are not estimated.</p>
<p>Allocate <code>transform: float32[16]</code>, <code>history: float32x3[iterations]</code>, <code>inlierMask: uint32[sourceCount]</code>,
and <code>status: uint32[1]</code>. Iterations are bounded to 1–100 (default 20) and emitted as graph passes;
there is no intermediate host readback. The GPU freezes updates after convergence or failure.
History records pre-update support, RMSE, and translation step length. Final inlier flags are
recomputed using the returned transform. A zero-support RMSE of zero is <strong>not</strong> success: inspect
status (0 converged, 1 budget exhausted, 2 fewer than three matches, 3 underdetermined rotation,
4 eigensolve exhaustion). Convergence requires both translation and rotation-column change below
their respective tolerances (default <code>1e-5</code>). Frozen history entries repeat the current residual
with zero step.</p>
<p>Without <code>targetGrid</code>, correspondence work is O(iterations × sourceCount × targetCount). With a supplied exact 3D grid, the target is indexed once per encoding and every iteration searches intersecting cells. Target rows outside the half-open grid are excluded; supply a grid covering the full intended target. Ties retain the lowest source target-row ID. Stable weighted moments are computed in 256-row blocks and merged before the small transform solve; the solve no longer scans every source point serially. Independent scans can be represented by independent operations. Poor initialization,
partial overlap, repeated structures, or high outlier rates can lead to local minima. No global registration guarantee or automatic initialization is claimed. Robust reweighting is not a trimmed objective.</p>
<h3 id="local-origin-and-work-limits">Local origin and work limits</h3>
<p>Positions are float32. At projected magnitudes near 1e6 m the float32 spacing is about 0.06 m, which
destroys millimetre registration. Recenter on the CPU with a shared float64 origin: call
<code>getPointCloudLocalOrigin(positions)</code>, upload <code>makePointCloudLocalPositions(positions, origin)</code> for the source,
target, cores and any <code>targetGrid</code>, and pass the same <code>origin</code> to <code>GPUPointCloudICP</code>. The GPU runs in the
local frame; <code>initialTransform</code> stays absolute and is converted for you. The published <code>transform</code> is in the
local frame: convert a readback with <code>icp.getAbsoluteTransform(transform)</code>
(<code>getPointCloudAbsoluteTransform</code>, or <code>getPointCloudLocalTransform</code> for the inverse). Every cloud that is
registered or compared together must use one origin; <code>GPUPointCloudChange</code> needs no origin option because it
consumes local-frame clouds and an ICP transform in that same frame.</p>
<p>The origin bounds use only points whose three coordinates are finite. For an audit count, call
<code>getPointCloudLocalOriginSummary(positions)</code> to obtain <code>{origin, skippedPointCount}</code>; a point with
any non-finite coordinate counts once and contributes none of its axes to the bounds. Empty or
all-invalid clouds return <code>[0, 0, 0]</code>. Both origin helpers and <code>makePointCloudLocalPositions</code> reject
a trailing partial XYZ triple. Recentering preserves invalid rows and row order; only originally
invalid points remain invalid when finite local coordinates fit float32.</p>
<p>Without <code>targetGrid</code>, <code>GPUPointCloudICP</code> refuses <code>sourceCount * targetCount</code> above
<code>GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK</code> (2^27, about 11,585 by 11,585) and asks for <code>targetGrid</code>.
<code>GPUPointCloudChange</code> does the same for <code>coreCount * (referenceCount + comparisonCount)</code> and asks for
<code>grid</code>; <code>GPUPointCloudNeighborhoods</code> does the same for <code>queryCount * positionCount</code>. With a grid there
is no cap on the pair count, because the work follows the points near each query. With
<code>samplingCovariance</code> each cloud is also limited to 32768 points because the in-cylinder correlation
sum is quadratic. Subsample or tile larger inputs.</p>
<p>How the limit was chosen: the largest unindexed run in the tests and the website is 5,000 by 5,000
points (25 million pairs, the feature-extraction page), and the largest kernel-audit case is 4,096
squared (16.8 million). 2^27 is about five to eight times those, so no existing run changes. The older
ceiling <code>GPU_POINT_CLOUD_MAXIMUM_BRUTE_FORCE_WORK</code> (2^32) is still exported.</p>
<h2 id="surface-change-and-confidence">Surface change and confidence</h2>
<p><code>GPUPointCloudChange</code> requires explicit alignment. With <code>alignment: 'already-aligned'</code>, the caller
certifies that reference, comparison, core points and normals share a coordinate frame. Alternatively
<code>alignment: {transform}</code> applies a resident column-major comparison-to-reference rigid transform
in a GPU graph pass. An ICP operation with comparison as source and reference as target can supply
this view directly; the caller must inspect/accept its convergence status. No alignment is guessed. Each core
uses its normalized oriented normal and a cylinder with inclusive <code>radius</code> and <code>halfDepth</code>.
All finite points in the cylinder contribute, without truncation.</p>
<p><code>counts: uint32[coreCount * 2]</code> contains paired reference/comparison sample support. <code>changes</code> and
<code>roughness</code> are <code>float32x3[coreCount]</code>. Changes contain signed comparison-minus-reference mean
normal distance and its lower/upper 95% confidence bounds. Roughness contains the two sample
standard deviations and the detection limit:</p>
<p><code>1.96 * (sqrt(referenceVariance/referenceCount + comparisonVariance/comparisonCount) + registrationError)</code>.</p>
<p>Registration uncertainty is nonnegative, defaults to zero, and has coordinate units. Reversing
normal orientation reverses the signed change. <code>minimumSupport</code> defaults to 3 and must be at least
2. Status is 0 for valid results, 1 for an invalid core/normal, 2 for insufficient support, and
3 for nonfinite accumulated uncertainty or confidence endpoints. Invalid result values stay zero
and must be masked; support counts remain available. This M3C2-style estimator uses caller-selected
projection scales and independent-sample Gaussian intervals by default. Compose <code>GPUPointCloudNormalScale</code> below for automatic normal-scale selection. Optional known spatial covariance is described below. Work is O(coreCount × totalCloudPoints) without <code>grid</code> and O(coreCount × points in the cells the cylinder touches) with it.</p>
<h2 id="numerical-evidence-and-current-limits">Numerical evidence and current limits</h2>
<p>The numerical browser suite checks translated tilted planes against NumPy 2.2.6 <code>eigh</code> fixtures,
rigid registration against independent NumPy SVD Kabsch, analytic signed surface change/roughness,
3D-only neighbor separation, truncation and ties, collinear degeneracy, distinct samples/centroids,
statistical outliers, signed HAG, missing coverage, morphological edge support and no-match failure.
The fixture generator is checked in beside the tests. All kernels use baseline float32 arithmetic:
choose a local coordinate origin and scale that avoids distance/covariance overflow. No CPU fallback,
implicit concatenation, or hidden data-dependent readback is used. The opt-in benchmark records full indexed-pipeline timing phases, storage limits and scratch memory.
PDAL differential fixture coverage is recorded explicitly by the SMRF generator; these contracts do not establish PDAL
algorithm equivalence or global streaming/partition equivalence.</p>
<h3 id="measured-indexed-neighborhood-and-feature-scope">Measured indexed-neighborhood and feature scope</h3>
<p>The October 23, 2026 native Apple WebGPU run completed the entire sorted 3D index rebuild,
nearest-eight radius query and feature graph at all four tested sizes. Radius was 1.5 on a unit
Cartesian lattice, with cell size 1. Each warm number below is the median of three full
encode/submit/queue-fence measurements; it includes host overhead and is not an isolated GPU timestamp.</p>
<table>
<thead>
<tr>
<th>Points</th>
<th>Allocate/upload</th>
<th>Graph compile</th>
<th>Cold encode/execute</th>
<th>Warm resident median</th>
<th>Physical scratch</th>
<th>Input + output + scratch</th>
</tr>
</thead>
<tbody>
<tr>
<td>1,000</td>
<td>1.7 ms</td>
<td>1.6 ms</td>
<td>153.7 ms</td>
<td>5.1 ms</td>
<td>20,068 bytes</td>
<td>140,068 bytes</td>
</tr>
<tr>
<td>10,000</td>
<td>1.4 ms</td>
<td>0.9 ms</td>
<td>151.8 ms</td>
<td>3.2 ms</td>
<td>202,916 bytes</td>
<td>1,402,916 bytes</td>
</tr>
<tr>
<td>100,000</td>
<td>7.0 ms</td>
<td>1.1 ms</td>
<td>151.9 ms</td>
<td>3.2 ms</td>
<td>2,018,448 bytes</td>
<td>14,018,448 bytes</td>
</tr>
<tr>
<td>1,000,000</td>
<td>65.3 ms</td>
<td>1.1 ms</td>
<td>167.9 ms</td>
<td>19.1 ms</td>
<td>20,031,424 bytes</td>
<td>140,031,424 bytes</td>
</tr>
</tbody>
</table>
<p>Cold execution includes lazy pipeline creation. Two four-byte result reads cost 0.7–4.1 ms after
the resident samples. Graphs used 28 nodes for the two smaller sizes and 44 for the two larger sizes;
each measurement made four submissions. The million-point case used 108 MB of output and 12 MB of
input; its largest output binding was 32 MB. The native device advertised approximately 4 GiB
storage-binding/buffer limits, though these individual outputs fit the baseline 128 MiB binding limit.</p>
<p><strong>Which path these numbers measured.</strong> The neighborhood run used the indexed path: the benchmark
passes <code>grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [side, side, side]}</code>
(<code>gpu-pointcloud-benchmark.spec.ts</code>). It did not measure the default path without a grid, which tests
every point for every query and is now refused above 2^27 pairs. The advanced table below used
<code>targetGrid</code> for ICP, and for SMRF a grid capped at 256 by 256 cells with <code>window: 4</code> (the default
window is 18 cells), so its cost follows the raster and the window, not the point count. No
one-million-point measurement of the unindexed path, of <code>GPUPointCloudChange</code> or of <code>grid</code>-indexed
change exists.</p>
<p>These numbers establish this lattice workload and device only. Dense coincident points, highly
uneven cells, larger capacities, wide radii, and other GPUs can behave differently. The scaled
canary checks known first-point support and numerical status; the separate numerical suites compare
full small-case outputs to independent NumPy/SciPy and all-pairs references. Raw timing samples,
device metadata, phase definitions and invocation are recorded in
<code>dev-docs/research/evidence/pointcloud-coverage-2026-10-23.json</code> in the repository.</p>
<h2 id="owned-rows-halos-and-explicit-partition-merges">Owned rows, halos and explicit partition merges</h2>
<p><code>GPUPointCloudNeighborhoods.queryRows</code> selects owned query rows while <code>positions</code> contains both owned and halo points. Output row order follows <code>queryRows</code>; neighbor IDs always index the supplied local position view, so callers retain the local-to-global mapping. Invalid query IDs set status bit 1. <code>supportBounds</code> certifies the extent whose point population is complete; status bit4 marks a radius sphere that crosses that extent. Truncation remains bit 2 and full support counts remain available. Status bits combine. Supply halos at least as wide as the query radius, keep grid and support extents consistent, and compare only owned outputs when stitching tiles. The operator cannot detect an omitted point inside a falsely certified extent.</p>
<p><code>GPUPointCloudFeatures</code> accepts a neighborhood row count independent of the number of candidate positions, so the owned-row neighborhoods compose directly. Covariance is centered on each neighborhood; viewpoint orientation uses its centroid. Read neighborhood halo-status separately from feature degeneracy/truncation status.</p>
<p>The <code>attribute</code> of <code>GPUPointCloudVoxelMerge</code> accepts matching float32 scalar or x2/x3/x4 means for the parts and the merged output. Its finite-support count is separate from the geometric population, so the caller that builds a part's means decides which vectors contribute (count only vectors whose components are all finite). The merge itself is linear in voxels times parts.</p>
<p><code>GPUPointCloudVoxelMerge</code> combines counts, centroids, minimum global representative IDs and optional vector means from aligned per-grid reductions. Every part declares <code>sourceRowOffset</code>; the caller certifies <code>partitionPolicy: 'disjoint-owned-rows'</code>. Exclude halo replicas before voxel reduction. Geometric and attribute means use their own population weights. Status bit 1 reports uint32 population overflow and bit 2 reports representative-ID overflow; reject flagged cells. Floating-point merge order can affect the last bits. Sampled positions can be gathered from the returned global representative IDs through the caller's global source mapping.</p>
<h2 id="interpolated-height-sampling">Interpolated height sampling</h2>
<p><code>GPUPointCloudHeight.interpolation</code> defaults to the containing cell. <code>{method: 'bilinear'}</code> interpolates cell-centre samples and requires every positive-weight corner to be finite and covered; it does not extrapolate across the grid edge. <code>{method: 'idw', radius, power, minimumSupport}</code> samples supported cell centres inside an inclusive physical radius, with power default 2, minimumSupport default 1 and radius at most 64 cells. An exact supported centre is reproduced even when its other neighbors are missing. IDW fills a query's missing terrain support, not the input raster. Both modes preserve negative heights and return status 2 if their declared support is unavailable. Status 3 reports a nonfinite interpolated elevation or height difference and leaves the output zero, including overflow from finite input values. Their parameters are the interpolation provenance; all heights retain the input vertical datum. Datum conversion, direct Delaunay ground interpolation and automatic extrapolation are separate operations.</p>
<h2 id="point-to-plane-registration">Point-to-plane registration</h2>
<p>Set <code>metric: 'point-to-plane'</code> and supply target-row <code>targetNormals</code>. Nearest correspondence still uses Euclidean distance; the update minimizes weighted normal residuals through a six-parameter Gauss–Newton system. Normals are normalized, zero/nonfinite normals do not contribute, and the final mask includes normal validity. Six independent constraints and a full-rank system are required. The moments are taken about the centroid of the matched source points (256-row blocks, each about its own centre, moved to the common centroid before the solve), and the normal matrix is scaled to a unit diagonal before elimination, so the rank test does not depend on the coordinate unit or on how far the cloud lies from the origin: a room corner 500 m from the origin, a scene 10 km across and a scene in millimetres all solve. A pivot at or below 1e-5 of that scaled matrix is rank loss. Rank-deficient configurations, including an isolated flat plane, return status 3 rather than an arbitrary translation/rotation. The rotational increment uses Rodrigues' formula about the centroid and is composed back into the transform about the origin; the history's translation step is the change of that origin-frame translation, as in point-to-point mode. Float32 coordinates still limit the result to their own spacing, so use <code>origin</code> for large projected coordinates. History RMSE is the normal residual for this mode, versus Euclidean residual for point-to-point. Both remain local algorithms whose initialization matters.</p>
<h2 id="declared-uncertainty-models">Declared uncertainty models</h2>
<p><code>registrationCovariance: [xx,xy,xz,yy,yz,zz]</code> supplies positive-semidefinite translation covariance in the aligned frame. Each core projects it along its oriented normal. Its projected variance is added to <code>registrationError²</code>. The default <code>uncertaintyCombination: 'linear'</code> retains the M3C2-style addition of sampling and registration standard errors. <code>'quadrature'</code> combines their independent variances. <code>confidenceMultiplier</code> defaults to 1.96 and explicitly sets the two-sided Gaussian quantile.</p>
<p>Optional <code>samplingCovariance: {referenceVariance, comparisonVariance, correlationLength}</code> uses a supplied Gaussian measurement-error model, <code>Cov(i,j) = variance × exp(-distance(i,j)² / (2 × correlationLength²))</code>. The mean variance is the exact within-cylinder double covariance sum divided by support squared. These supplied normal-direction variances are distinct from observed roughness; the GPU contributor does not estimate a covariance model from the sample. The explicit CPU helper <code>fitPointCloudSamplingCovariance</code> below fits detrended residuals. The two clouds are assumed independent. This mode costs O(cores × cloudPoints²), targets bounded supports, and is tested against independent SciPy distance-matrix integration. It is not a correction for an unknown correlation process.</p>
<h2 id="actual-external-references-and-advanced-scaling">Actual external references and advanced scaling</h2>
<p>The pinned generators now execute PDAL 2.8.4 directly. Four SMRF fixtures (holes, low outliers, slope sensitivity, scalar zero and net cutting) agree on every classification. Planar normals agree with PDAL <code>filters.normal</code>; that fixture uses PDAL's self-inclusive support and compares orientation only, not covariance eigenvalues across differing supports. Point-to-point ICP agrees with PDAL's composed source-to-target matrix, including its centroid adjustment. Outlier classifications agree on a separated-cluster fixture; PDAL uses sample standard deviation while this API explicitly uses population standard deviation, so exact threshold parity is not claimed. Raster HAG agrees with <code>hag_nn(count=1)</code> where queries share ground samples' cell-centre XY, including negative heights; this is not general nearest-ground interpolation parity. <code>pdal-pointcloud.json</code> and <code>smrf.json</code> retain the version and actual reference outputs. Independent NumPy/SciPy fixtures still check covariance, interpolation and uncertainty conventions.</p>
<p>The advanced native Apple WebGPU measurements rebuild their indexes every encoding. ICP ran with <code>targetGrid</code> and SMRF with a 256 by 256 grid and <code>window: 4</code> (see the note above); neither measures the default window or an unindexed search. Warm values are medians of three full encode/submit/queue-fence wall measurements, including host overhead. No individual binding exceeds 128 MiB.</p>
<table>
<thead>
<tr>
<th>Workload</th>
<th>10,000 points</th>
<th>100,000 points</th>
<th>1,000,000 points</th>
</tr>
</thead>
<tbody>
<tr>
<td>SMRF, window4, small raised objects, maximum256×256 cells</td>
<td>3.1ms</td>
<td>4.6ms</td>
<td>12.8ms</td>
</tr>
<tr>
<td>Indexed point-to-point ICP, four iterations, Huber weights</td>
<td>4.7ms</td>
<td>6.4ms</td>
<td>27.6ms</td>
</tr>
<tr>
<td>Coincident cloud,64 owned exact queries, nearest4 plus full counts</td>
<td>5.0ms</td>
<td>43.9ms</td>
<td>439.4ms</td>
</tr>
</tbody>
</table>
<p>The skew test deliberately exposes occupancy-dependent cost: it does not establish that querying every point in a million-point coincident cloud is practical. SMRF evidence covers the stated grid/window/object pattern, not every sparse-hole morphology. ICP evidence checks known rigid translation and status; the small numerical cases separately cover rotation, outliers, point-to-plane rank loss and invalid support. Raw records, native device limits, cold timings, readback timing, scratch memory and exact workload definitions are in <code>dev-docs/research/evidence/pointcloud-advanced-2026-10-23.json</code>.</p>
<p>The executed PDAL 2.8.4 and isolated 2.9.0/2.9.2 cores did not include <code>filters.m3c2</code>. The attempted 2.10.2 core required a newer GDAL ABI unavailable in the isolated reference environment. Temporary reference installations were removed after fixture generation to reclaim disk space; the pinned generators and outputs remain in the repository. M3C2 known correlated-noise fixtures use independent NumPy/SciPy references. A separate executed
GPU comparison now covers the actual upstream PDAL M3C2 golden fixture, without claiming a local
PDAL binary execution: tree <code>0e64af327cba0a4c54e501db0d61b62390b69a79</code>,
<code>M3C2FilterTest.cpp</code> / <code>verifyPoints</code>, two Autzen BMX LAS epochs (829 and 687 points), normal radius
1.390432, cylinder radius 2.890432 and half-depth 5.5. All 91 golden rows having at least two
samples in both epochs pass signed-distance error below 0.01 coordinate units and exact support
counts. The remaining 12 rows have no reported reference solution and are explicitly listed;
they do not establish GPU behavior outside its minimum sample-variance support contract.</p>
<p>The caller-supplied normals are independently prepared with NumPy 2.2.6 eigendecomposition of
reference neighbors at the upstream radius and oriented upward. Coordinates are recentered before
f32 upload. Core row zero runs separately with the first source point omitted, reproducing PDAL's
explicit first-point exclusion. PDAL reports population standard deviations; the GPU reports
sample standard deviations. Converting the GPU values back to the population convention agrees
with the rounded golden roughness and detection limits within 0.002 coordinate units, and all
significance classifications agree. This is explicit convention-aware comparison, not identical
uncertainty APIs or automatic normal-scale selection. The repository fixture record at <code>modules/analysis/test/oracles/upstream-golden/README.md</code>
retains unmodified source data, expected outputs, filter/test sources, copyright/license and SHA-256
hashes. Two GPU batches and three provenance-integrity node tests passed.</p>
<h2 id="gpupointcloudnormalscale"><code>GPUPointCloudNormalScale</code></h2>
<p>Minimum population plane roughness over supplied radii, excluding the query itself. Each scale uses
four compute stages plus parameter uploads, with at most 32 scales. Truncated scales are rejected;
grid coverage must include intended support.</p>
<p><code>GPUPointCloudNormalScale</code> selects the minimum-roughness plane over 1–32 supplied spherical
<code>radii</code>. Radii are strictly increasing <strong>radii</strong>, in local Cartesian coordinate units;
CloudCompare's UI normal scale uses diameters. The selection rule is the smallest population
covariance eigenvalue (squared orthogonal roughness), with exact ties resolved by the smaller radius.
This implements the minimum-roughness option, not Brodu–Lague dimensionality classification.
See the <a href="https://www.cloudcompare.org/doc/wiki/index.php/PluginM3C2">CloudCompare M3C2 multiscale description</a>.</p>
<p>Supply <code>positions: float32x3</code>, optional owned <code>queryRows: uint32</code>, <code>capacity</code> (3–64), and
<code>parameters: float32[5]</code> packed by <code>getGPUPointCloudNormalScaleParameterValues</code>. The settings
<code>minimumSupport</code> (3–64, default 3) and <code>viewpoint</code> are per-frame; support above capacity gives
no usable scale. Each query excludes its own source row; other coincident
rows remain separate samples. <code>viewpoint</code> orients from each core toward a sensor, otherwise normals face positive
z (then y, then x). <code>spatialContext</code> defaults to native planar Cartesian coordinates; geographic
coordinates are refused. Recenter large absolute coordinates before upload.</p>
<p>Caller-owned outputs have one row per query: <code>normals: float32x3</code>, <code>normalScales: float32</code>
(selected radius), <code>normalVariances: float32</code> (squared population roughness), <code>status: uint32</code>.
Status bits are 1 no usable scale, 2 <code>candidateOverflow</code> at any scale, and 4 eigensolve exhaustion
at any scale. Only complete, converged, nondegenerate supports participate. If bit 1 is clear, a
normal is published even when another scale overflowed; inspect the other bits to see rejected
candidates. When bit 1 is set every other output is zero. Invalid IDs, empty supports and
collinear supports cannot produce a normal. Exact flat-plane ties select the smallest eligible radius.</p>
<p>The largest-radius nearest list is reused. If it overflows, a smaller radius is still complete
when its boundary precedes the last retained neighbor; a boundary tied with that neighbor is
conservatively rejected. Missing slots never contribute. The optional <code>grid</code> is built once per
encoding and must cover the full intended support; outside-grid points are excluded, and missing
halos cannot be inferred. Without a grid, query-count times position-count is limited to 2^27.
Scratch is checked against the storage binding limit before nodes are constructed. The exposed
<code>estimatedCommandNodeCount</code> includes a conservative 512-node allowance for the published index.</p>
<p>Radii, capacity, grid and view shapes are compile-time. Position, query-row and parameter contents
can change per encoding. Write packed settings into a caller-owned <code>GPUParameterBuffer</code> to change
support eligibility and orientation without recompiling. Each scale emits three kernels with no
readback. Pass <code>normals</code> directly to <code>GPUPointCloudChange</code>; its <code>cores</code> must be the same owned
positions, in the same row order. Inspect selector status before accepting change results.
<code>gpu-pointcloud-normal-scale.spec.ts</code> exercises this full GPU path against NumPy eigendecomposition
and cylinder means, sample variances and confidence intervals on planes with known offsets.</p>
<p>{/* parameters: GPUPointCloudNormalScale generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpupointcloudnormalscale-parameters"><code>GPUPointCloudNormalScale</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUPointCloudGridParameterValues(settings)</code>, writes <code>GPU_POINT_CLOUD_GRID_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>gridParameters</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUPointCloudNormalScaleParameterValues(settings?)</code>, writes <code>GPU_POINT_CLOUD_NORMAL_SCALE_PARAMETER_LENGTH</code> (5) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUPointCloudGridParameterValues</code>:</p>
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
<td><code>origin</code></td>
<td><code>readonly [number, number] | readonly [number, number, number]</code></td>
<td>-</td>
</tr>
<tr>
<td><code>cellSize</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>Settings of <code>getGPUPointCloudNormalScaleParameterValues</code>:</p>
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
<td><code>minimumSupport</code> (optional)</td>
<td><code>number</code></td>
<td>Minimum other points for a candidate plane, 3–64, default 3. Above capacity no scale qualifies.</td>
</tr>
<tr>
<td><code>viewpoint</code> (optional)</td>
<td><code>readonly [number, number, number]</code></td>
<td>Orient toward this viewpoint; omitted uses positive z, then y, then x.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="fitting-sampling-covariance">Fitting sampling covariance</h2>
<p><code>fitPointCloudSamplingCovariance(positions, residuals, correlationLengths)</code> is an explicit CPU
fit of the Gaussian model already accepted by <code>GPUPointCloudChange.samplingCovariance</code>.
Inputs are packed XYZ positions and detrended normal-direction residuals. For every unordered
pair, its observed semivariance is half the squared residual difference. At each supplied length,
the helper fits a nonnegative variance to <code>variance * (1 - exp(-distance² / (2 * length²)))</code>
by ordinary least squares. It returns <code>variance</code>, <code>correlationLength</code> and <code>meanSquaredError</code>;
exact ties choose the smaller length. With two distinct points, every candidate fits the single
semivariance exactly: the error is analytically zero and the smallest length wins. Fits with more
pairs retain direct residual errors without an error tolerance. This is a finite candidate search,
not a continuous optimizer.</p>
<p>The float64 reference fit is bounded to 4,096 points and 32 candidate lengths because its work
is quadratic in point count. Fit epochs separately with a common fixed length to populate
<code>referenceVariance</code>, <code>comparisonVariance</code>, and <code>correlationLength</code>. The helper does not infer
surface detrending, fit a nugget, or combine registration uncertainty. Constant residuals give
zero variance; coincident-only positions cannot identify a length and are refused. The NumPy
fixture uses an independent vectorized least-squares solve.</p>
<h3 id="executable-backed-m3c2-check">Executable-backed M3C2 check</h3>
<p><code>gpu-pointcloud-m3c2-binary.node.spec.ts</code> runs a supplied PDAL executable through CSV readers,
<code>filters.m3c2</code> and a CSV writer, comparing distance, population roughness, support, detection limit
and significance against independent NumPy synthetic planes. Run with:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">PDAL_M3C2_BINARY</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">/path/to/pdal</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> yarn</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> vitest</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> run</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> --project</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> node</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> --maxWorkers=2</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> modules/analysis/test/gpu-pointcloud/gpu-pointcloud-m3c2-binary.node.spec.ts</span></span></code></pre>
<p>The test is explicitly skipped when no executable is configured; an explicitly configured binary
that lacks the stage fails. This staging environment has no PDAL/CloudCompare executable, so the
new binary test has <strong>not been executed</strong> here. The existing upstream golden evidence above remains
distinct from local binary execution. GPU confidence uses sample variance; PDAL reports population
variance, and the executable test checks PDAL's convention directly.</p>
<h2 id="kepler-catalog-bindings">Kepler catalog bindings</h2>
<p>The Kepler contributor catalog binds all eleven classes on this page through
<code>modules/kepler/src/analysis/catalog-pointcloud-bindings.ts</code>. Operation IDs use the <code>pointcloud-</code>
prefix: <code>ground</code>, <code>smrf</code>, <code>surfaces</code>, <code>height</code>, <code>neighborhoods</code>, <code>features</code>, <code>statistical-filter</code>,
<code>normal-scale</code>, <code>icp</code>, <code>change</code> and <code>voxel-merge</code>. Import <code>GPUPointCloudHeight</code> from the public
<code>@luma-contributors/analysis/pointcloud-analysis</code> entry point; it computes height above explicit
class-2 terrain support. The ground
binding first runs GPUPointCloudSurfaces with every point eligible to prepare the unclassified
cell minima and coverage consumed by GPUPointCloudGround; original LAS classes are kept separate
for the final classification pass. The catalog preserves finite covariance and converged
eigenvalues for adequately supported collinear neighborhoods while publishing null normals and
shape descriptors. Voxel-merge centering ignores zero-count entries, retaining their slots with
local zero coordinates so arbitrary empty-cell placeholders cannot reduce occupied-centroid precision.</p>
<p>These GPU-only host adapters consume bounded XYZ Point inputs, preserve source row identity and
altitude, and publish Arrow columns through native Sources for point color/size/altitude styling.
Geographic inputs use the catalog local metric frame and a shared binary64 origin; planar inputs
keep their Cartesian unit. The host form exposes a bounded subset of each contributor's options,
not a new numerical implementation. Voxel merge requires sufficient statistics, identical voxel
IDs/grid and explicit disjoint ownership; it is not raw-point voxel downsampling. See the
<a href="/luma-contributors/docs/../examples/deck/kepler-compute#point-cloud-contributor-catalog">Kepler example contract and output table</a>
for input aliases, capacities, output columns, publication steps and remaining form limitations.</p>`};export{e as default};