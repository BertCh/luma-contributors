var e={slug:`api`,title:`Analysis modules`,html:`<h1 id="analysis-modules">Analysis modules</h1>
<p><code>@luma-contributors/analysis</code> is an extension of <a href="https://luma.gl">luma.gl</a>. It holds only the
geospatial and analysis operations; the GPU infrastructure they run on (<code>GPUCommandGraph</code>, GPU Core
operations, <code>GPUData</code>/<code>GPUVector</code>/<code>GPUTable</code>, GPU Dataframe, GPU Graph Algorithms, GPU Project, GPU
DGGS, GPU Trace, physics and simulation, renderers) is owned and documented by luma.gl. Operations
here are built from the public <code>@luma.gl/core</code>, <code>@luma.gl/shaders</code> and <code>@luma.gl/graph</code> APIs.</p>
<h2 id="run-your-first-computation">Run your first computation</h2>
<p>The <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#variogram-fitting-without-a-graph">CPU variogram walkthrough</a>
fits generated data without a GPU. Follow it with the complete
<a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins#point-density-from-device-to-readback">point-density GPU walkthrough</a>
to create a device, allocate views, submit work, read typed results and release resources.</p>
<h2 id="entries">Entries</h2>
<table>
<thead>
<tr>
<th>Entry</th>
<th>Use it for</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis"><code>@luma-contributors/analysis/gpu-spatial-analysis</code></a></td>
<td>Spatial weights and statistics, joins, density, cells, lines, geometry, trajectories, and regression.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-network"><code>@luma-contributors/analysis/gpu-network</code></a></td>
<td>CSR network analysis: reachability, service areas, accessibility, flow aggregation, and edge bundling.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-terrain"><code>@luma-contributors/analysis/gpu-terrain</code></a></td>
<td>Elevation-tile analysis: slope, contours, viewshed, hydrology, solar shadows, and relief shading.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster"><code>@luma-contributors/analysis/raster-analysis</code></a></td>
<td>Raster analysis: fill nodata, tile reprojection, fractional coverage, focal pairs, spectral indices, regions, contours, and statistics.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-dataframe-analysis"><code>@luma-contributors/analysis/dataframe-analysis</code></a></td>
<td>Analysis contributors over luma.gl GPU Dataframe columns: profiles, group statistics, joins, and standardization.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/geospatial"><code>@luma-contributors/analysis/geospatial-analysis</code></a></td>
<td>Geospatial contexts, datum grid transforms and point buffers layered on luma.gl GPU Project.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/pointcloud-analysis"><code>@luma-contributors/analysis/pointcloud-analysis</code></a></td>
<td>Point cloud neighborhoods, features, ground classification, registration, and change.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/computation-record"><code>@luma-contributors/analysis/computation-record</code></a></td>
<td>Recorded executions of a computation as plain data, for the lens views.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts"><code>@luma-contributors/analysis/analysis-core</code></a></td>
<td>Shared kernels and contracts used by the analysis entries.</td>
</tr>
</tbody>
</table>
<p>See the <a href="/luma-contributors/docs/api/analysis-catalog">analysis catalog</a> for every contributor
at a glance and the <a href="/luma-contributors/docs/api/gpu-spatial-analysis-cross-reference">cross-reference</a>
for a map from familiar tool names to contributors.</p>
<p>Read <a href="/luma-contributors/docs/api/operation-forms-and-cpu-twins">operation forms and CPU twins</a> to choose between
caller-owned GPU views, named graph operations and CPU execution, with a complete variogram fit.</p>
<h2 id="what-lumagl-owns">What luma.gl owns</h2>
<p>Link to the luma.gl documentation rather than to copies here:</p>
<ul>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-core">GPU Core and <code>GPUCommandGraph</code></a></li>
<li><a href="https://luma.gl/docs/api-reference/graph/gpu-data">GPU Data and GPU Tables</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-dataframe">GPU Dataframe and GPU SQL</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-graph">GPU Graph Algorithms</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-project">GPU Project and GPU DGGS</a></li>
<li><a href="https://luma.gl/docs/glossary">Glossary</a></li>
</ul>
<h2 id="conventions">Conventions</h2>
<p>Contributors declare work on a caller-owned <code>GPUCommandGraph</code> and return command nodes; callers own
the frame loop, command submission and readback. Entries are WebGPU-first; individual pages state
backend, capacity and precision limits.</p>`};export{e as default};