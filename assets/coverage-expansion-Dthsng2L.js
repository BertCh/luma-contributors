var e={slug:`api/gpu-spatial-analysis/coverage-expansion`,title:`Geospatial coverage expansion`,html:`<h1 id="geospatial-coverage-expansion">Geospatial coverage expansion</h1>
<p>The coverage expansion adds graph contributors with explicit bounded contracts. Import spatial
operations from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>; raster, terrain, network, dataframe,
projection and point-cloud operations retain their own package subpaths.</p>
<p>These APIs declare GPU work in the caller's graph. Inputs and outputs are GPU views; callers
control submission and readback. Capacity, convergence, missing-data and unsupported-case results
must be inspected before interpreting numerical outputs. An available constructor does not imply
complete compatibility with a reference library or every mode in the maintainer roadmap.</p>
<table>
<thead>
<tr>
<th>Area</th>
<th>APIs and contracts</th>
</tr>
</thead>
<tbody>
<tr>
<td>Geometry</td>
<td><a href="/luma-contributors/docs/api/gpu-geometry-refinement">Polygon normalization, precision reduction, subdivision, reference-vertex insertion, ratio hulls and bounded maximum-inscribed-circle search</a>. Each operation retains its own geometry and numerical limits.</td>
</tr>
<tr>
<td>Global coordinates</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/global-coordinates">S2 hierarchy, neighbors, measures and cap coverings; convex spherical predicates and overlap</a>. Datum-grid transforms live in <code>gpu-project</code>.</td>
</tr>
<tr>
<td>Prediction</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/interpolation-expansion">Local kriging, RBF, natural neighbors and conditional Gaussian simulation</a>. Resident variogram fitting lives in <code>gpu-dataframe</code>.</td>
</tr>
<tr>
<td>Statistical contracts</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/coverage-statistics">Robust IV inference, restricted regimes, panel error likelihood, SUR, probit, local diagnostics, permutation inference, rank transitions, fuzzy weights, held-out regression and spatial interaction</a>. Training masks must also govern preprocessing and parameter selection.</td>
</tr>
<tr>
<td>Point processes</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/point-process">Inhomogeneous window statistics, simulation-curve envelopes, OPTICS and probability sampling</a>. Network kernel density lives in <code>gpu-network</code>.</td>
</tr>
<tr>
<td>Urban analysis</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/urban-morphology">Building tessellation, street profiles, shared walls, isovists and visibility graphs</a>.</td>
</tr>
<tr>
<td>Ecology</td>
<td><a href="/luma-contributors/docs/api/gpu-spatial-analysis/landscape-ecology">Landscape summaries, patch proximity, resistance connectivity and segregation decomposition</a>.</td>
</tr>
<tr>
<td>Raster</td>
<td><a href="/luma-contributors/docs/api/gpu-raster/remote-sensing">Conservative remapping, texture, temporal processing, translation registration, region graphs, spectral unmixing and seasonal breaks</a>.</td>
</tr>
<tr>
<td>Network and movement</td>
<td><a href="/luma-contributors/docs/api/gpu-network/network-movement-coverage">Routing, flow, communities, facilities, rational-agent access, trajectory smoothing, restriction and similarity search</a>.</td>
</tr>
<tr>
<td>Point clouds</td>
<td><a href="/luma-contributors/docs/api/pointcloud-analysis">Indexed neighborhoods, local features, filtering, SMRF, height normalization, ICP and surface change</a>.</td>
</tr>
<tr>
<td>Terrain</td>
<td><a href="/luma-contributors/docs/api/gpu-terrain/terrain-processing">Breaching, branching flow consumers, sink hierarchy, catchment summaries, DEM alignment and tiled routing</a>.</td>
</tr>
</tbody>
</table>
<p>Numerical tests use independent pinned references where available, together with conservation,
covariance, topology, capacity and repeated-execution invariants. A composed reference is labeled
as such. See the domain pages for tested models and remaining modes; measured large-data throughput
and full upstream-package parity are separate claims.</p>`};export{e as default};