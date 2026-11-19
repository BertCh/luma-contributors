var e={slug:`api/gpu-network/flow-models-mob10`,title:`Flow models on origin-destination aggregation`,html:`<h1 id="flow-models-on-origin-destination-aggregation">Flow models on origin-destination aggregation</h1>
<p>Two contributors that work on zone-level flows: a parameter-free model that predicts flows from
populations and distances, and a kernel smoother for observed flows. Both are reference-tier: work
grows with the cube of the zone count, and construction rejects more than 1024 zones.</p>
<p>Both take zone centroids, planar or great-circle (<code>spatialContext</code>). Their inputs match what
<code>GPUFlowAggregation</code> writes: <code>zoneOutWeights</code> is the <code>outflows</code> of the radiation model when the
aggregation runs with <code>excludeSelfFlows: true</code> (<code>T_i</code> is the trips that leave zone <code>i</code>; without the
option the zone totals also count trips that stay inside it), and the
smoother reads <code>flowOriginZoneIds</code>, <code>flowDestinationZoneIds</code>, <code>flowWeights</code> and <code>status.count</code> (as <code>validPairCount</code>)
directly.</p>
<h2 id="gpuradiationflowmodel"><code>GPURadiationFlowModel</code></h2>
<p>The radiation model (Simini, Gonzalez, Maritan and Barabasi, Nature 2012). A trip leaving zone <code>i</code>
is absorbed by zone <code>j</code> with probability <code>m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))</code>, where <code>m_i</code>
and <code>n_j</code> are the populations of the origin and destination and <code>s_ij</code> is the population of the
zones strictly nearer to <code>i</code> than <code>j</code>, excluding both. The output is <code>T_i</code> times that probability,
the expected flow, laid out <code>flows[i * zoneCount + j]</code> with a zero diagonal. It draws no random
integers, unlike scikit-mobility's <code>Radiation.generate</code>.</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>positions</code>, <code>populations</code>, <code>outflows</code></td>
<td>One row per zone. <code>outflows</code> is <code>T_i</code>, the trips leaving each zone, not counting trips that stay inside it</td>
</tr>
<tr>
<td><code>normalization</code></td>
<td><code>'none'</code> (default), or <code>'finite-size'</code>, which divides by <code>1 - m_i / M</code> so every row sums to <code>T_i</code></td>
</tr>
<tr>
<td><code>output.flows</code></td>
<td><code>zoneCount^2</code> rows</td>
</tr>
</tbody>
</table>
<p>Ties in distance: zones at exactly the distance of the destination do not count in <code>s_ij</code>.
Implementations that sort by distance count one of two tied zones and not the other, so they differ
there.</p>
<p>Verified against a hand-computed three-zone case, against a binary64 reference that sorts the
zones per origin and takes a prefix sum of population (a different route from the kernel), and
against the closed-form row sum <code>T_i (1 - m_i / M)</code>. scikit-mobility was not run.</p>
<p>{/* lens:GPURadiationFlowModel */}</p>
<h2 id="gpukernelflowsmoothing"><code>GPUKernelFlowSmoothing</code></h2>
<p>Smoothing of OD flows: each flow is re-estimated as the kernel-weighted mean of the flows between
neighbouring origins and neighbouring destinations. Smoothing over neighbourhoods of both ends is the idea
of Guo and Zhu (IEEE TVCG 2014); this contributor does not reproduce their estimator. The definition used:</p>
<ol>
<li>The neighbourhood of zone <code>i</code> is every zone within the smallest distance <code>h_i</code> at which the
<code>locationWeights</code> of the zones within <code>h_i</code> (zone <code>i</code> included) sum to at least
<code>neighborhoodSize</code>. Ties at <code>h_i</code> are included.</li>
<li>A neighbour <code>k</code> has the Gaussian weight <code>exp(-d_ik^2 / (2 sigma_i^2))</code> with
<code>sigma_i = bandwidthRatio * h_i</code>; <code>K_ii = 1</code>.</li>
<li><code>S_ij = sum_a sum_b K_ia K_jb M_ab / (sum_a K_ia * sum_b K_jb)</code>. Pairs without a flow count as
zero.</li>
</ol>
<p>This is a normalised kernel mean, so it does not conserve the total flow. Each row of weights is
divided by its own sum; the columns are not. A zone that is a neighbour of many zones spreads its flows
into all of them and the total grows, and a zone that is nobody's neighbour loses part of its flows. With
zones at x = 0, 1 and 10, unit weights, <code>neighborhoodSize</code> 2 and <code>bandwidthRatio</code> 0.5, one flow of 100
inside the middle zone sums to 125.3 after smoothing and one flow of 100 inside the far zone sums to 77.6.
Rescale the output when totals must match the input.</p>
<p>The neighbourhood size, the kernel and the bandwidth ratio are this library's own choices; the paper
was not available to check them and no packaged implementation exists. The tests use a binary64
reference written from the three points above, the three-zone case worked by hand, and closed-form
limits (a zero-size neighbourhood returns the input; a very wide kernel returns the mean). Flow
selection, which removes duplicate smoothed flows, is not provided.</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>flows</code></td>
<td><code>{kind: 'matrix', matrix}</code> with <code>zoneCount^2</code> rows, or <code>{kind: 'pairs', originZoneIds, destinationZoneIds, weights, validPairCount}</code> as written by <code>GPUFlowAggregation</code> (distinct pairs)</td>
</tr>
<tr>
<td><code>locationWeights</code></td>
<td>Weight of each zone in the neighbourhood size</td>
</tr>
<tr>
<td><code>parameters</code></td>
<td>Per-frame, <code>getGPUKernelFlowSmoothingParameterValues</code>: neighbourhood size, bandwidth ratio</td>
</tr>
<tr>
<td><code>output.smoothed</code></td>
<td><code>zoneCount^2</code> rows</td>
</tr>
<tr>
<td><code>output.radii</code></td>
<td>Optional neighbourhood radius of each zone</td>
</tr>
</tbody>
</table>
<p>{/* lens:GPUKernelFlowSmoothing */}</p>
<p>{/* parameters: GPUKernelFlowSmoothing generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpukernelflowsmoothing-parameters"><code>GPUKernelFlowSmoothing</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUKernelFlowSmoothingParameterValues(settings)</code>, writes <code>GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUKernelFlowSmoothingParameterValues</code>:</p>
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
<td><code>neighborhoodSize</code></td>
<td><code>number</code></td>
<td>Weighted size <code>p</code> of a zone's neighbourhood, in the unit of <code>locationWeights</code>: the nearest zones are taken until their weights sum to at least <code>p</code>. A value of 0 or less keeps only the zone itself (and zones at distance 0), which reproduces the input.</td>
</tr>
<tr>
<td><code>bandwidthRatio</code> (optional)</td>
<td><code>number</code></td>
<td>Gaussian standard deviation as a fraction of the neighbourhood radius: <code>sigma = ratio * radius</code>. Defaults to 0.5.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};