var e={slug:`api/gpu-spatial-analysis/space-time-clustering`,title:`Space-time clustering`,html:`<h1 id="space-time-clustering">Space-time clustering</h1>
<p><code>GPUSpatialClustering</code> (DBSCAN) and <code>GPUOPTICS</code> take an optional <code>times</code> column that turns them into
ST-DBSCAN and ST-OPTICS. The time test is one more condition on the neighbor relation, so cores,
clusters, borders, noise, reachability and extraction follow from the same kernels as the purely
spatial versions. Without <code>times</code>, nothing changes: the generated kernels contain no time code.</p>
<p><code>GPUHierarchicalDensityClustering</code> also supports ST-HDBSCAN through a continuous scaled Euclidean metric. See <a href="#st-hdbscan">ST-HDBSCAN</a>.</p>
<p>Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>.</p>
<h2 id="time-input">Time input</h2>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>times</code></td>
<td>One <code>float32</code> per row: ticks after the origin of <code>temporalContext</code>. Typically the output of <code>GPUTimeWordsRelative</code>, which takes the difference to the origin exactly on Int64 words.</td>
</tr>
<tr>
<td><code>temporalContext</code></td>
<td>A <a href="/luma-contributors/docs/api/gpu-spatial-analysis/temporal-context"><code>GPUTemporalContext</code></a> describing <code>times</code>. Required with <code>times</code> and rejected without. Only <code>encoding: 'float32-relative'</code> is read (a bigint origin as <code>epoch</code>); any unit, time zone and axis. Nothing is converted.</td>
</tr>
</tbody>
</table>
<p>The window is given in the unit of <code>temporalContext</code>:</p>
<ul>
<li><code>GPUSpatialClustering</code>: <code>timeEpsilon</code> in the packed parameters (<code>getGPUSpatialClusteringParameterValues({bounds, epsilon, minimumPoints, timeEpsilon})</code>, stored at index 6 of the existing 8-element layout, 0 when omitted). It is per-frame: rewrite it without recompiling. <code>times</code> is compile-time presence, per-frame contents.</li>
<li><code>GPUOPTICS</code>: <code>maximumTimeDistance</code>, compile-time, positive and finite.</li>
</ul>
<p>Times are float32. Differences are exact for <code>|dt| &#x3C; 2^24</code> ticks, so choose an origin near the data and a unit coarse enough that the span stays inside that range when exact thresholds matter. The window is stored as float32 too.</p>
<h2 id="definitions">Definitions</h2>
<p>Two rows are related when <code>distance &#x3C;= epsilon</code> (the existing closed spatial test) and
<code>|t_a - t_b| &#x3C;= timeEpsilon</code> (closed as well, so a difference equal to the window is related). A row
relates to itself.</p>
<ul>
<li>ST-DBSCAN: a point is core when its related neighborhood holds at least <code>minimumPoints</code> points, itself included. Clusters are the connected components of core points under the relation, the root of a cluster is its smallest core row, a border point joins the adjacent cluster with the smallest root, and compact labels rank the roots in row order. This is the canonical labeling of <code>GPUSpatialClustering</code>, applied to the time-filtered relation.</li>
<li>ST-OPTICS: core distances, the frontier, reachability, predecessors and extraction (<code>dbscan</code> or <code>xi</code>) use only related pairs, so a pair outside the window behaves as if it were farther than <code>maximumDistance</code>. A hard filter on the neighbor graph.</li>
<li>A row with a non-finite time has no neighbors: in ST-DBSCAN it is noise (not core even with <code>minimumPoints: 1</code>), and in ST-OPTICS it is invalid like a non-finite position and counted in <code>status[2]</code>.</li>
<li><code>denseBoxShortcut</code> is rejected with <code>times</code>, because a dense box proves spatial closeness only.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUTimeWordsRelative} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/dataframe-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUSpatialClustering, getGPUSpatialClusteringParameterValues} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> converter</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTimeWordsRelative</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timestamps,                       </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// uint32x2 Int64 milliseconds</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  temporalContext: {unit: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'millisecond'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, epoch: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'unix'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, encoding: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'int64-words'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, axis: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'irregular'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  origin: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1_700_000_000_000</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">n</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: relativeTimes             </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(converter);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialClustering</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  times: relativeTimes,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  temporalContext: converter.outputTemporalContext,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  gridSize: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {labels}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUSpatialClusteringParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  bounds, epsilon: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, minimumPoints: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  timeEpsilon: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">6</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 3600</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> *</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 1000</span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">      // six hours, in the context unit (milliseconds)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h2 id="cost">Cost</h2>
<p>ST-DBSCAN adds one gather of <code>times</code> into grid-cell order and one float compare per scanned
candidate. The grid is spatial only, so the scan visits every spatial candidate and a short window
does not prune work; a time-sorted or three-dimensional index would, and is a separate item.
ST-OPTICS keeps the frontier of <code>GPUOPTICS</code> and its limits: at most 4096 rows with the exhaustive reachability step, at most 65,536 (including xi extraction) with <code>neighborIndex: 'grid'</code>, whose radius list also applies the time window. <code>orderingMethod: 'parallel'</code> distributes the indexed frontier across workgroups with the same strict time filter and exact row-index tie rule; the default remains <code>'sequential'</code>.</p>
<h2 id="reference-and-agreement">Reference and agreement</h2>
<ul>
<li>ST-DBSCAN: <code>st-dbscan</code> 0.2.3 (MIT), <code>ST_DBSCAN(eps1, eps2, min_samples).fit(X)</code> with <code>X = [time, x, y]</code>. Below 20000 rows it builds <code>where(|dt| &#x3C;= eps2, euclidean, 2 * eps1)</code> and runs scikit-learn <code>DBSCAN(eps=eps1, metric='precomputed')</code>, the same relation as above. The fixture <code>test/oracles/fixtures/space-time-clustering.json</code> (generator <code>test/oracles/space-time-clustering.py</code>) holds its labels for five scenes on integer lattices, which exercise the closed boundaries. The GPU labels match them exactly, and an independent double-precision CPU implementation in the spec reproduces the labels, core flags, roots and cluster count.</li>
<li>From 20000 rows <code>st-dbscan</code> 0.2.3 switches to a sparse path. On one 20000-row lattice scene it labeled 3733 points as clustered, where the dense definition above gives 4983 (and 2311 if a point did not count itself), so the sparse path is a different computation. The GPU follows the dense definition and is not compared against the sparse path.</li>
<li>ST-OPTICS: the <code>st-clustering</code> 0.1.0 <code>ST_OPTICS</code> matrix (distance <code>2 * eps1</code> outside the window) run on scikit-learn <code>OPTICS(metric='precomputed')</code>. That matrix equals the hard filter whenever <code>maximumDistance &#x3C; 2 * eps1</code>; with a larger <code>max_eps</code> the library keeps out-of-window pairs at distance <code>2 * eps1</code> instead of removing them, which <code>times</code> does not reproduce. <code>st-clustering</code> itself does not import with current joblib (<code>Memory(cachedir=...)</code>), so the fixture reproduces its matrix with scikit-learn. Ordering, predecessors and DBSCAN-extraction labels match exactly on two scenes, reachability and core distances to 1e-5 relative, and the same CPU oracle agrees.</li>
</ul>
<p>The pinned environment (Python 3.15.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1) differs from the 3.14 pins of <code>requirements.txt</code>. <code>st-dbscan</code> 0.2.3 is published as a source distribution only and its pyscaffold build fails under pip on Python 3.15, so the generator imports it from the unpacked sdist (SHA-256 <code>ca1810025630571aa9ffe204f81d9c0a52dc29daa2d05bb2cd7941f56f32fd8d</code>, recorded in the fixture).</p>
<h2 id="st-hdbscan">ST-HDBSCAN</h2>
<p><code>GPUHierarchicalDensityClustering</code> accepts optional <code>times</code>, <code>temporalContext</code> and <code>parameters</code>.
Its distance is <code>sqrt(dx² + dy² + (timeScale * dt)²)</code>, equivalent to Euclidean HDBSCAN on
<code>[x, y, timeScale * t]</code>. <code>timeScale</code> is spatial position units per temporal-context tick.
For example, with positions in meters and times in seconds, <code>timeScale: 2</code> assigns a one-second
separation the same distance as two meters. Zero gives spatial distances while still excluding
non-finite times. There is no hard time window.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUHierarchicalDensityClustering,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUHierarchicalDensityClusteringParameterValues</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUParameterBuffer} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUParameterBuffer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(device, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'hdbscan-parameters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, length: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  values: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUHierarchicalDensityClusteringParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({timeScale: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUHierarchicalDensityClustering</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions,                        </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// planar float32x2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  times: relativeTimes,              </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// float32, one per position</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  temporalContext: converter.outputTemporalContext,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: parameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  minimumSamples: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,                 </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// includes the point itself</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  minimumClusterSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">10</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {labels, coreDistances, status: {count: clusterCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>getGPUHierarchicalDensityClusteringParameterValues</code> packs <code>[timeScale, 0, 0, 0]</code> and rejects
negative or non-finite float32 scales. Rewrite the parameter buffer between encodings to change
the scale without recompiling. Invalid raw scale values make every row invalid. Row count,
<code>minimumSamples</code>, <code>minimumClusterSize</code>, contexts and optional-view presence are compile-time;
positions, times and scale values are per-frame. Only planar positions and relative float32 times
are supported; project geographic input and choose a time origin near the observations.
The shader subtracts times before scaling. Float32 subtraction, multiplication and squared-distance
overflow remain numerical limits; center inputs and rescale units to keep distances representable.</p>
<p>The exact indexed core-distance walk and Boruvka search reuse the spatial point BVH. Spatial box
distance is a conservative lower bound for the combined metric. The BVH does not prune by time,
so many observations at the same spatial site can require many candidate visits. The existing
GPU spanning tree, dendrogram, condensation and excess-of-mass selection are shared. Limits stay
at 1,048,576 rows and 32 minimum samples; construction validates the command-node estimate and
graph declaration checks scratch against the device storage binding limit.</p>
<p>Outputs retain the spatial HDBSCAN contract: one label per input row (<code>0xffffffff</code> for noise),
optional self-inclusive core distances (<code>Infinity</code> for invalid rows or insufficient neighbors),
and optional <code>status.count</code> containing the number of selected clusters. Finite rows with invalid
times are excluded from both cores and hierarchy. No candidate list truncation or approximation
is used, so no overflow/approximation port is needed. Boruvka and tree rounds have sufficient
fixed logarithmic budgets and GPU convergence gates, with no caller-set iteration cutoff.</p>
<p>Selection matches scikit-learn <code>HDBSCAN</code> defaults: Euclidean distance, EOM, no single cluster,
zero selection epsilon. Equal-weight MST edges are ordered by input rows and can produce a
partition different from scikit-learn on ties; duplicate densities use the largest float32
instead of infinity, as in spatial HDBSCAN. There is no HDBSCAN CPU operation twin.</p>
<p>The independent fixture <code>modules/analysis/test/oracles/fixtures/space-time-hdbscan.json</code> is generated
by <code>modules/analysis/test/oracles/space-time-hdbscan.py</code> using scikit-learn on scaled coordinates;
it records the NumPy, SciPy and scikit-learn versions. It covers overlapping events, zero and
small time scales, maximum samples, duplicate ties and a 257-row padded BVH. Headless specs compare
partitions (adjusted Rand index at least 0.97, allowing tied merges), cluster counts and every core
distance (absolute/relative tolerance <code>2e-5 * max(1, reference)</code>). Separate cases exercise invalid
rows, an empty finite set, one row and too few samples.</p>`};export{e as default};