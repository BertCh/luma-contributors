var e={slug:`api/gpu-spatial-analysis/track-similarity-measures`,title:`Track similarity: EDR, ERP, SSPD and bounded search`,html:`<h1 id="track-similarity-edr-erp-sspd-and-bounded-search">Track similarity: EDR, ERP, SSPD and bounded search</h1>
<p><code>GPUTrackSimilarity</code> computes, besides Hausdorff, discrete Frechet, DTW and LCSS, three more
measures on the same pair list, and <code>GPUTrackSimilaritySearch</code> can prune on cheap lower bounds
before it evaluates any of them. Tracks are planar or great-circle vertex lines without time.</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Measure</th>
<th>Source</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>editDistanceOnRealSequence</code></td>
<td>EDR</td>
<td>Chen, Ozsu and Oria, SIGMOD 2005</td>
</tr>
<tr>
<td><code>editDistanceWithRealPenalty</code></td>
<td>ERP</td>
<td>Chen and Ng, VLDB 2004</td>
</tr>
<tr>
<td><code>symmetricSegmentPathDistance</code></td>
<td>SSPD</td>
<td>Besse, Guillouet, Loubes and Royer, 2016</td>
</tr>
<tr>
<td><code>hausdorff</code> (existing)</td>
<td>symmetric Hausdorff</td>
<td>Shapely/GEOS, the one <code>movingpandas.Trajectory.hausdorff_distance</code> calls</td>
</tr>
</tbody>
</table>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrackSimilarity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  linesA, linesB, pairA, pairB,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  editDistanceTolerance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,          </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// EDR match distance, context length unit</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  editDistanceGapPoint: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],       </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// ERP gap point</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    editDistanceOnRealSequence: edr,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    editDistanceWithRealPenalty: erp,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    symmetricSegmentPathDistance: sspd</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<h2 id="definitions">Definitions</h2>
<ul>
<li><strong>EDR.</strong> The least number of insertions, deletions and substitutions that turns one (densified)
vertex sequence into the other. Two vertices match, at cost 0, when their distance is at most
<code>editDistanceTolerance</code>; a substitution of non-matching vertices, an insertion and a deletion
cost 1. The result is a whole number and is not normalized; divide by the longer vertex count for
the normalized form. The match test is a metric ball like this package's <code>lcssTolerance</code>; Chen's
original uses a per-axis box.</li>
<li><strong>ERP.</strong> A substitution costs the distance between the two vertices; an insertion or a deletion
costs the distance of the vertex to the fixed <code>editDistanceGapPoint</code>. ERP is a metric. The gap
point is <code>[x, y]</code> in planar contexts and <code>[longitude, latitude]</code> degrees on the sphere.</li>
<li><strong>SSPD.</strong> The mean over the original vertices of one track of the distance to the nearest point
of the other track's polyline (the segments, not only the vertices), averaged over both
directions. It is not densified and has no vertex cap.</li>
<li><strong>Hausdorff</strong> is the existing output, unchanged.</li>
</ul>
<p>For geographic Hausdorff and SSPD, projection and interpolation use one wrapped start-to-end
longitude arc. The nearest-segment candidate is compared with both endpoints; its distance never
exceeds either endpoint distance. This holds for queries on the opposite side of the globe as well
as for queries near the antimeridian. For example, the point <code>(0°, 0°)</code> and segment
<code>(100°, 0°) → (-100°, 0°)</code> have Hausdorff and SSPD distances of <code>100° × sphereRadius × π / 180</code>
with default densification. The local projection remains an approximation for long edges.</p>
<p>EDR and ERP run the anti-diagonal dynamic program of DTW with one extra table row and column for
the empty prefix, so the boundary prefix costs come from the same recurrence; they share the
shorter-track cap <code>maximumFrechetVertexCount</code> (<code>frechetCapExceeded</code> in <code>pairStatus</code>, NaN result)
and <code>densify</code>. SSPD uses 64 lanes per pair with fixed-order lane sums and a fixed binary tree (no
atomics). EDR is exact in f32; ERP sums in f32, so its error grows with <code>n + m</code> like DTW.</p>
<p>The CPU twin creates densified points only for requested measures that use them. Requests limited
to <code>maximumDistance</code> and/or <code>symmetricSegmentPathDistance</code> use the original vertices, even with a
small positive <code>densify</code>. For measures that densify, each expanded track must have at most
<code>2^32 - 1</code> vertices (the JavaScript array length limit); the CPU checks that count before
allocating the dense points. An explicit <code>maximumFrechetVertexCount</code> also binds CPU dynamic-program
measures: beyond the shorter densified track cap, status is <code>frechetCapExceeded</code> and their distances
are NaN (decisions are zero). Hausdorff, SSPD and maximum distance remain available. Omitting that
prop leaves the CPU shorter-track count uncapped.
The GPU also checks expanded counts only when a requested measure uses densification, and checks
the shorter-track cap only when a dynamic-program measure is requested. A maximum-distance-only
or SSPD-only request therefore remains valid when an unused expanded count would exceed uint32.</p>
<p>Hausdorff, EDR, ERP and SSPD are not in the MovingPandas 0.23 distance list except Hausdorff, DTW,
LCSS and Frechet, so EDR, ERP and SSPD are verified against double-precision definitions written
from the papers (see Verification).</p>
<h2 id="lower-bounds-and-pruned-search">Lower bounds and pruned search</h2>
<p><code>GPUTrackSimilaritySearch</code> takes <code>metric</code> <code>'hausdorff'</code>, <code>'edit-distance-on-real-sequence'</code>,
<code>'edit-distance-with-real-penalty'</code> or <code>'symmetric-segment-path'</code> in addition to the previous
metrics. For each pair it first computes an <code>O(n + m)</code> lower bound from the original vertices and
writes it to <code>output.lowerBound</code>. Pairs whose bound exceeds the <code>threshold</code> (with a relative f32
margin of <code>1e-6</code>) get status 1 and skip the measure, and their distance is NaN. The survivors are
compacted and evaluated, and <code>topK</code> selects the exact stable top-k among the pairs within the
threshold, grouped by <code>pairA</code>.</p>
<table>
<thead>
<tr>
<th>Measure</th>
<th>Lower bound</th>
<th>Why it never exceeds the measure</th>
</tr>
</thead>
<tbody>
<tr>
<td>Hausdorff</td>
<td>largest distance of a vertex of one track to the bounding box of the other</td>
<td>the other track lies in its box, so the distance to the box is at most the distance to the track</td>
</tr>
<tr>
<td>SSPD</td>
<td>the same with means</td>
<td>the mean of box distances is at most the mean of polyline distances</td>
</tr>
<tr>
<td>EDR</td>
<td><code>max(n, m)</code> minus the fewer vertices that lie within the tolerance-widened box of the other track</td>
<td>each edit script costs at least <code>max(n, m)</code> minus its matches, and a vertex outside the widened box cannot match</td>
</tr>
<tr>
<td>ERP</td>
<td>absolute difference of the two sums of vertex distances to the gap point</td>
<td>ERP is a metric and the ERP of a track to the empty sequence is that sum (reverse triangle inequality)</td>
</tr>
<tr>
<td>Frechet, DTW</td>
<td>larger endpoint distance (existing)</td>
<td>the first and last vertices are always coupled</td>
</tr>
</tbody>
</table>
<p>The EDR and SSPD bounds subtract an <code>f32</code> slack proportional to <code>n + m</code> so that summed rounding
cannot push a bound above the evaluated distance; the ERP bound subtracts <code>(n + m) * 1.2e-7 * (sumA + sumB)</code>.</p>
<p>Pruning is exact: it only skips pairs that cannot be within the threshold. The threshold is the
caller's. To get the k nearest tracks without a known radius, run a first search with a generous
threshold or take the k-th distance of an earlier frame, widen it by about <code>1e-4</code> for f32 rounding,
and search again; a threshold-free adaptive top-k that tightens the radius while it runs is not
implemented.</p>
<p>Why this matters: a 2024 study (Ni et al., PVLDB 2024, arXiv 2311.00960) measured batched GPU DTW
over 512 pairs at only about 4.5 times a CPU (28.11 s against 6.21 s), so the pairs that never
reach the dynamic program are the larger saving than a faster program.</p>
<h2 id="verification">Verification</h2>
<ul>
<li><code>test/oracles/track-similarity-edit-measures.py</code> (MovingPandas 0.23.0, Shapely 2.2.0, Python
3.15) writes <code>fixtures/track-similarity-edit-measures.json</code>. <code>hausdorff</code> and
<code>dynamicTimeWarping</code> come from <code>Trajectory.hausdorff_distance</code> and <code>Trajectory.dtw_distance</code> and
are asserted equal to Shapely and to an independent DTW program; EDR, ERP and SSPD are written
from the paper definitions in recursive head/rest form, which differs from the GPU's table.</li>
<li>Specs in <code>test/gpu-spatial-analysis/track-similarity/</code>: the GPU against the fixture (planar and
great circle, tracks across the 256-lane strip boundary, cap status), fixed-order SSPD
repeatability, lower bounds at most the true distance for 404 random and degenerate pairs in
double precision and on the GPU, and pruned top-k equal to the unpruned top-k and to a CPU
reference.</li>
</ul>`};export{e as default};