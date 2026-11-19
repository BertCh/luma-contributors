var e={slug:`api/gpu-spatial-analysis/operators-lines`,title:`GPU Spatial Analysis: Lines & trajectories`,html:`<h1 id="lines--trajectories">Lines &#x26; trajectories</h1>
<h3 id="gpulinesegmentize-gpugreatcirclearcs-gpulinesmooth-and-gpulinechunk"><code>GPULineSegmentize</code>, <code>GPUGreatCircleArcs</code>, <code>GPULineSmooth</code>, and <code>GPULineChunk</code></h3>
<p><code>GPULineSmooth</code>: Optional source-row and source-chainage metadata follows the emitted vertices.
Metadata rows are at most 1,073,741,823 and position addresses at most 2,147,483,647.</p>
<p><strong><code>GPULineSegmentize</code></strong> — Densifies each polyline so no segment exceeds a maximum length, writing new vertices and source-line indices.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-segmentize rung=result caption="Grey: 223 Wicker Park streets · blue: densified to segments of at most 25 m"}</p>
<p><strong><code>GPUGreatCircleArcs</code></strong> — Tessellates a great-circle arc between each origin and destination pair into equal-angle vertices, writing one line per pair.</p>
<p>::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=result caption="Grey: 40 destinations · blue: great-circle arcs from ORD, 32 segments each"}</p>
<p><strong><code>GPULineSmooth</code></strong> — Smooths each polyline with Chaikin corner cutting, writing the refined vertices and source-line indices.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-smooth rung=result caption="Grey: 223 Wicker Park streets · blue: two Chaikin rounds with ratio 0.25"}</p>
<p><strong><code>GPULineChunk</code></strong> — Splits each polyline into pieces of a fixed length, writing the pieces with their source-line indices.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-chunk rung=result caption="Grey: 223 Wicker Park streets · blue: 100 m chunks"}</p>
<p>Geometry-producing line contributors. Each writes a <code>GPULineGeometryOutput</code>: flat <code>positions</code>, clamped
<code>lineOffsets</code> (deck.gl <code>PathLayer</code> <code>startIndices</code>), <code>status</code> (<code>count</code>, <code>overflow</code>, and optional <code>requiredCount</code>,
<code>approximation</code>, <code>vertexCount</code> and <code>requiredVertexCount</code>), optional <code>sourceLineIndices</code> (output line to input line), <code>sourceRows</code> (output vertex to input
segment row) and <code>measures</code> (distance along the source line). As in the generated-geometry layout
(contributor contract rule 8), <code>count</code> and <code>requiredCount</code> count output lines and <code>vertexCount</code> and
<code>requiredVertexCount</code> count vertices; <code>vertexCount</code> is the indirect draw count of <code>positions</code>.
<strong>Breaking (2026-11-10):</strong> <code>count</code> and <code>requiredCount</code> counted vertices before, and <code>status.lineCount</code>
is removed (it is <code>count</code>). A line truncated by the vertex capacity still counts as written. The vertex capacity is
<code>positions.length</code>; offsets are clamped to it, so a truncated result is still a valid layout.
Inputs are packed <code>positions</code> sorted by line with <code>lineOffsets</code> (<code>lineCount + 1</code> rows).
<code>GPULineSegmentize</code>, <code>GPULineChunk</code> and <code>GPUGreatCircleArcs</code> declare <code>measure: 'distance'</code>;
Lambert equal-area local frames throw (use an azimuthal-equidistant frame for planar lines).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineSegmentize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, lineOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  spatialContext: {coordinateSpace: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'longitude-latitude'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, metric: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'great-circle'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// omit for planar</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: segmentizeParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {positions: densified, lineOffsets: densifiedOffsets, status: {count, overflow, vertexCount}, measures}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">segmentizeParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPULineSegmentizeParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({maximumSegmentLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">50000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGreatCircleArcs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  sources, targets, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// longitude/latitude pairs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: arcParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {positions: arcPositions, lineOffsets: arcOffsets, status: {count, overflow, vertexCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">arcParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUGreatCircleArcsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({maximumSegmentLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, minimumSegments: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineSmooth</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({positions, lineOffsets, iterationCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, closed: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: smoothParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), output}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineChunk</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({positions, lineOffsets, mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'chunk'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'substring'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: chunkParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {positions: pieces, lineOffsets: pieceOffsets, status: {count, overflow, vertexCount}, sourceLineIndices}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">chunkParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPULineChunkParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({chunkLength: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<ul>
<li><code>GPULineSegmentize</code> (PostGIS <code>ST_Segmentize</code>, densify): every segment of length <code>L</code> becomes
<code>clamp(ceil(L / maximumSegmentLength), 1, maximumPieceCount)</code> equal pieces; input vertices
are kept. With compile-time <code>closedLines: true</code> (default false), each line is a ring that closes
implicitly: the closing edge is densified and no vertex is repeated. A last row that repeats
the first row bit-exactly is dropped; other repeated input vertices are preserved. Output keeps
one line per input line, including empty and single-vertex lines, following the implicit polygon
ring convention used by <code>GPURectangleClip</code>. The spherical closing edge follows the short way
across the antimeridian using the last row's longitude shift.
With a great-circle <code>spatialContext</code>, positions are longitude/latitude degrees: lengths
are great-circle distances times the context's <code>sphereRadius</code> (default mean Earth radius,
<code>GPU_EARTH_MEAN_RADIUS_METERS</code>; meters, or kilometers with <code>units: 'kilometers'</code>), interior points
are slerped, and longitudes are unwrapped along each line so it stays continuous across the
antimeridian. <strong>Breaking:</strong> the <code>coordinateSystem</code> and <code>radius</code> props of <code>GPULineSegmentize</code> and
<code>GPULineChunk</code> (and <code>radius</code> of <code>GPUGreatCircleArcs</code>) are replaced by <code>spatialContext</code>; planar is
the default. <code>GPUGreatCircleArcs</code> defaults to a great-circle longitude/latitude context and
throws for planar ones; <code>GPULineSmooth</code> is planar only and throws for longitude/latitude.</li>
<li><code>status.approximation</code> (optional, <code>GPULineSegmentize</code> and <code>GPUGreatCircleArcs</code> only; other contributors that
write this layout reject a view for it) is set to 1 when an edge or arc needed more than <code>maximumPieceCount</code>
(<code>maximumSegmentCount</code>) pieces for the requested maximum length, so some output segments are longer than
requested; otherwise 0, rewritten on every encoding.</li>
<li><code>GPUGreatCircleArcs</code> (turf <code>greatCircle</code>): one line of <code>segments + 1</code> vertices per
origin/destination pair, <code>segments = clamp(max(ceil(distance / maximumSegmentLength), minimumSegments), 1, maximumSegmentCount)</code>. Endpoints are exact; longitudes are unwrapped from the
source (Tokyo to San Francisco ends at 237.6), never split. Draw with <code>PathLayer</code>
<code>wrapLongitude: true</code> or in a globe view. Near-antipodal interpolation can be numerically unstable
in float32; split such arcs in the caller. There is no whole-arc rejection: counts and endpoints
are emitted even when the shared interpolator returns NaN for interior vertices within 1e-6
radians of an antipode. Exactly antipodal endpoints do not determine a unique great circle.</li>
<li><code>GPULineSmooth</code> (Chaikin corner cutting, turf <code>polygonSmooth</code>): <code>iterationCount</code> (compile time,
1 to 10 passes) each replace edge <code>(a, b)</code> with <code>mix(a, b, r)</code> and <code>mix(a, b, 1 - r)</code> for a per-frame
ratio <code>r</code> (default 0.25). Open lines keep their endpoints (<code>n</code> vertices become <code>n * 2^k</code>); with
<code>closed: true</code> a repeated closing vertex is ignored and each ring repeats its first vertex.
Output sizes depend only on the layout, so every level's offsets come from two scans in closed
form and each level is one kernel; levels ping-pong through two capacity-sized transients.</li>
<li><code>GPULineChunk</code>: <code>'chunk'</code> splits each line into pieces of <code>chunkLength</code> (turf <code>lineChunk</code>), the
last piece ending at the line end; consecutive pieces share their boundary point. The piece
count is data dependent: the line capacity is <code>output.lineOffsets.length - 1</code>, and
<code>status.count</code>/<code>sourceLineIndices</code> report the pieces; <code>requiredCount</code> is the unclamped piece count,
computed independently of output capacity and saturated at <code>0xffffffff</code> when it reaches the uint32
limit. Per-line requirements use float32 division of the measured length by <code>chunkLength</code>.
Requesting <code>requiredCount</code> in chunk mode adds a per-line count scan; layout offsets remain bounded
even when the full requirement exceeds uint32. <code>'substring'</code> extracts <code>[startMeasure, endMeasure]</code> from every line (turf <code>lineSliceAlong</code>, PostGIS <code>ST_LineSubstring</code> by distance);
measures are clamped to the line and a start past the end gives an empty line. Zero-length
lines are copied as one piece. Both modes support a great-circle <code>spatialContext</code>.</li>
<li>Everything per frame lives in the parameter buffer, so resolution, ratio, chunk length and
ranges change with no recompile. All kernels are element-wise or sequential per line:
deterministic, no float atomics.</li>
<li>Precision (f32, measured against f64 oracles on Apple Metal): planar densify and chunks within
2e-5 of the coordinate scale; spherical densify within 2.9 m, arcs up to 16,500 km within 5.7 m.</li>
</ul>
<p><code>getGPULineSegmentizeParameterValues</code>, <code>getGPUGreatCircleArcsParameterValues</code>, and
<code>getGPULineChunkParameterValues</code> accept optional <code>settings.spatialContext</code>, the context of the
lengths. A lon/lat context warns when a positive finite <code>maximumSegmentLength</code> or <code>chunkLength</code>
is below ten times the float32 position resolution (about 16.97 meters on the default sphere):
<code>&#x3C;ClassName> distance is too short for float32 lon/lat; use createGPULocalFrame</code>.
Pass the same context used by the contributor. This setting does not change the four-float packed
layout; planar contexts and omitted contexts do not warn.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPULineSegmentize */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-segmentize rung=datum datum=0 caption="Road 0: its vertices before and after 25 m densification"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-segmentize rung=formula caption="Pieces per segment from its length and the 25 m limit"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-segmentize rung=kernel caption="The compute shader that writes the densified vertices"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-segmentize rung=graph caption="Command graph nodes and buffers that produce the densified vertices"}</p>
<p>{/* lens:GPUGreatCircleArcs */}</p>
<p>::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=datum datum=11 caption="Route 11: ORD to JFK, 33 vertices along the great circle"}</p>
<p>::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=formula caption="Arc vertices by spherical interpolation at equal angle steps"}</p>
<p>::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=kernel caption="The compute shaders that count and write the arc vertices"}</p>
<p>::lens{scenario=gpu-spatial-analysis.great-circle-arcs rung=graph caption="Command graph nodes and buffers that produce the arcs"}</p>
<p>{/* lens:GPULineSmooth */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-smooth rung=datum datum=0 caption="Road 0: corners cut twice at ratio 0.25"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-smooth rung=formula caption="Each edge is replaced by two points at the cut ratio"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-smooth rung=kernel caption="The compute shader that writes the smoothed vertices"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-smooth rung=graph caption="Command graph nodes and buffers that produce the smoothed vertices"}</p>
<p>{/* lens:GPULineChunk */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-chunk rung=datum datum=0 caption="Piece 0: the first 100 m chunk of road 0"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-chunk rung=kernel caption="The compute shader that writes the chunk vertices"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-chunk rung=graph caption="Command graph nodes and buffers that produce the chunk vertices"}</p>
<p>{/* parameters: GPUGreatCircleArcs generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpugreatcirclearcs-parameters"><code>GPUGreatCircleArcs</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUGreatCircleArcsParameterValues(settings)</code>, writes <code>GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUGreatCircleArcsParameterValues</code>:</p>
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
<td>Context of the lengths; a lon/lat context warns when the length is below float32 resolution.</td>
</tr>
<tr>
<td><code>maximumSegmentLength</code></td>
<td><code>number</code></td>
<td>Largest arc segment length in sphere-radius units (meters by default). <code>0</code> or <code>Infinity</code> lets <code>minimumSegments</code> alone decide the resolution.</td>
</tr>
<tr>
<td><code>minimumSegments</code> (optional)</td>
<td><code>number</code></td>
<td>Smallest number of segments per arc. Set <code>maximumSegmentLength</code> to <code>0</code> and this to <code>n</code> for a fixed <code>n</code> segments per arc. Default 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPULineChunk generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulinechunk-parameters"><code>GPULineChunk</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULineChunkParameterValues(settings)</code>, writes <code>GPU_LINE_CHUNK_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULineChunkParameterValues</code>:</p>
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
<td>Context of the lengths; a lon/lat context warns when the length is below float32 resolution.</td>
</tr>
<tr>
<td><code>chunkLength</code> (optional)</td>
<td><code>number</code></td>
<td><code>'chunk'</code> mode: piece length in position (or sphere-radius) units.</td>
</tr>
<tr>
<td><code>startMeasure</code> (optional)</td>
<td><code>number</code></td>
<td><code>'substring'</code> mode: start distance from each line's first vertex, clamped to the line.</td>
</tr>
<tr>
<td><code>endMeasure</code> (optional)</td>
<td><code>number</code></td>
<td><code>'substring'</code> mode: end distance, clamped to the line. A start past the end gives an empty line.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPULineSegmentize generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulinesegmentize-parameters"><code>GPULineSegmentize</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULineSegmentizeParameterValues(settings)</code>, writes <code>GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULineSegmentizeParameterValues</code>:</p>
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
<td>Context of the lengths; a lon/lat context warns when the length is below float32 resolution.</td>
</tr>
<tr>
<td><code>maximumSegmentLength</code></td>
<td><code>number</code></td>
<td>Largest output segment length, in position units (<code>'planar'</code>) or sphere-radius units (<code>'spherical'</code>, meters by default). <code>0</code> or <code>Infinity</code> disables densification.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPULineSmooth generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulinesmooth-parameters"><code>GPULineSmooth</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULineSmoothParameterValues(settings?)</code>, writes <code>GPU_LINE_SMOOTH_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULineSmoothParameterValues</code>:</p>
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
<td><code>ratio</code> (optional)</td>
<td><code>number</code></td>
<td>Chaikin cut ratio in <code>(0, 0.5]</code>. Each edge <code>(a, b)</code> is replaced by <code>mix(a, b, ratio)</code> and <code>mix(a, b, 1 - ratio)</code>. Default 0.25 (classic Chaikin).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugeometrymeasures-gpugeodesicpairs-and-gpugeodesicdestination"><code>GPUGeometryMeasures</code>, <code>GPUGeodesicPairs</code>, and <code>GPUGeodesicDestination</code></h3>
<p>The <a href="/luma-contributors/docs/api/gpu-spatial-analysis/geodesic-precision">geodesic precision guide</a> defines per-row status, conditioning
exclusions, measured error envelopes and differences from the float64 CPU operation twins.</p>
<p><strong><code>GPUGeometryMeasures</code></strong> — Measures length, area, centroid, bounding box and vertex count of line and polygon features, planar or on the ellipsoid.</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-measures rung=result caption="Streets coloured by WGS84 geodesic length in metres, beside the 223 Wicker Park streets"}</p>
<p><strong><code>GPUGeodesicPairs</code></strong> — Computes distance, initial and final bearing and midpoint for each origin and target pair of longitude/latitude points.</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=result caption="Grey: 40 ORD destinations · blue: route midpoints; distances are WGS84 Vincenty"}</p>
<p><strong><code>GPUGeodesicDestination</code></strong> — Projects each origin along a bearing and distance, writing the destination point and final bearing.</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=result caption="Grey: ORD · blue: destinations at 8 bearings and 500, 1,000 and 2,000 km (WGS84)"}</p>
<p>Measurement columns for line and polygon features and for point pairs, in planar coordinates or
on the Earth (turf <code>area</code>, <code>length</code>, <code>centroid</code>, <code>bbox</code>, <code>distance</code>, <code>bearing</code>, <code>destination</code>,
<code>midpoint</code>; PostGIS <code>ST_Area</code>, <code>ST_Length</code>, <code>ST_Perimeter</code>, <code>ST_Centroid</code>, <code>ST_Envelope</code>,
<code>ST_NPoints</code>, <code>ST_Distance(geography)</code>, <code>ST_Azimuth</code>, <code>ST_Project</code>).</p>
<p><code>GPUGeometryMeasures.spatialContext</code> is optional (native planar by default). <code>GPUGeodesicPairs</code> and
<code>GPUGeodesicDestination</code> require a longitude/latitude context and declare <code>measure: 'distance'</code>;
geometry measures declare no measure because they output both lengths and areas. All three use
<code>resolveGPUSpatialContext</code> and its shared validation errors.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGeometryMeasures</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  geometryType: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">  // GeoArrow four-level layout: feature -> polygons -> rings -> vertices</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, featureOffsets, polygonOffsets, ringOffsets,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">  // planar (native units), great-circle (spherical), or ellipsoidal (geodesic edges by default)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  spatialContext: {coordinateSpace: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'longitude-latitude'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, metric: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'ellipsoidal'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, units: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'meters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    areas, lengths, centroids, bounds, vertexCounts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {invalidCount} </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// geographic polygons with a pole or half-world edge, see below</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  },</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  groupIds, groupCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">12</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  groupOutput: {areas: groupAreas, centroids: groupCentroids, featureCounts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> wgs84</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {coordinateSpace: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'longitude-latitude'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, metric: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'ellipsoidal'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, units: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'meters'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">as</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGeodesicPairs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({origins, targets, spatialContext: wgs84,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {distances, initialBearings, finalBearings, midpoints, status: {converged}}}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUGeodesicDestination</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({origins, bearings, distances, spatialContext: wgs84,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {destinations, finalBearings}}));</span></span></code></pre>
<ul>
<li>Features: one invocation per feature walks its rings in row order with Neumaier-compensated f32
sums, relative to the feature's first vertex, so projected coordinates far from the origin keep
their precision (a 1 m square at x = 1e6 measures exactly 1). Rings close implicitly; a repeated
closing vertex adds a zero edge. Outputs: length (lines) or perimeter (polygons), signed and
absolute area, area-weighted (polygons) or length-weighted (lines) centroid falling back to the
vertex mean, bounds, vertex count. Empty features give zeros and NaN centroid and bounds.</li>
<li>Input layout: <code>geometryType</code> selects the layout. <code>'polygons'</code> takes the four-level layout of
<code>GPUPolygonGeometryFields</code> (<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>; rows are
features). <code>'lines'</code> takes <code>positions</code> and <code>lineOffsets</code>, plus optional <code>featureOffsets</code>
(feature -> lines) for multi-lines. <code>'points'</code> takes <code>positions</code>, plus optional <code>featureOffsets</code>
(feature -> points) for multi-points; without it every point is a feature.</li>
<li>Ring roles are positional: the first ring of every polygon is a shell and later rings are holes
whatever the winding, so a multi-polygon with mixed winding and a hole wound like its shell measure
correctly, and <code>areas</code> is non-negative. <code>GPUShapeDescriptors</code>, <code>GPUPolygonTriangulation</code> and
<code>GPUAffineTransform</code> (centroid origins) use the same layout and rule; <code>GPULabelPoint</code> uses even-odd,
which agrees with it for valid (non-overlapping) polygons.</li>
<li>With <code>metric: 'great-circle'</code> and linear edges (the default), area uses Chamberlain-Duquette
(turf parity): an approximation of degree-straight edges with second-order error in latitude
span, exact for longitude/latitude boxes. With <code>edgeModel: 'great-circle'</code>, area uses signed
spherical excess on the context sphere (<code>areaModel: 'sphere-polygon'</code>), summed per edge as the
Lambert chord term from the feature origin less the bulge of the great-circle arc over that
chord, so a ring a hundred meters across keeps the same f32 relative precision (about <code>5e-7</code>)
as a continental one. Perimeter uses
great-circle chords for both edge models; centroids retain the straight Lambert equal-area
construction. For linear-edge area, pass
<code>sphereRadius: GPU_WGS84_SEMI_MAJOR_AXIS_METERS</code> for turf <code>area</code> parity. Ellipsoidal with
<code>ellipsoidalEdgeModel: 'coordinate-linear'</code> maps latitude to authalic latitude (exact equal-area
ellipsoid mapping) and measures edges with an f32 Vincenty inverse.</li>
<li><strong>Measure model.</strong> The instance's <code>measures</code> field (<code>GPUMeasureDescription</code>) says what the length
and area outputs mean: <code>lengthModel</code> (<code>'planar'</code>, <code>'great-circle'</code>, <code>'ellipsoidal-geodesic'</code>),
<code>areaModel</code> (<code>'planar'</code>, <code>'sphere-equal-area'</code>, <code>'sphere-polygon'</code>, <code>'wgs84-authalic'</code>, <code>'geodesic'</code>) and the length
and area units. Units follow <code>getGPULengthUnits</code>: native planar uses position units; a meters local
frame reports meters and square meters. Geographic lengths default to meters and areas to square
meters; <code>units: 'kilometers'</code> gives kilometers and square kilometers, including ellipsoidal kernels
(previously rejected). Feature and group areas use the context's length unit squared.</li>
<li>Linear-edge spherical areas use <code>getGPUAuthalicSphereRadius</code> (<code>areaRadius</code>); great-circle-edge
areas use <code>getGPUSphereRadius</code>. Lengths keep the
mean radius (<code>radius</code>). With the default sphere, area results change by about <code>5e-7</code> relative.
Ellipsoidal kernels honor <code>spatialContext.ellipsoid</code> (default WGS84); the geodesic classes expose the
resolved <code>ellipsoid</code> in the context's length unit.</li>
<li><strong>Antimeridian and poles (geographic polygons).</strong> Non-pole longitude steps are wrapped, so an
edge with a longitude difference above 180 degrees (Fiji's 179 to -179) takes the short way and
the ring is measured correctly. Same-pole edges retain their raw step, so explicitly capped
Antarctica rings have zero winding and finite area. Unprepared rings with winding +-360,
rings wider than 360 degrees, and ambiguous edges report NaN <code>areas</code> and
<code>centroids</code> (lengths, bounds and vertex counts stay valid) and are counted in
<code>output.status.invalidCount</code>, a one-row <code>uint32</code> view cleared every encoding. Linear edges with
a wrapped step of 180 degrees are ambiguous; great-circle edges are ambiguous when antipodal
or passing through a pole. Prepare global rings with <code>GPULonLatPolygonPreparation</code>.
Ellipsoidal contexts take <code>ellipsoidalEdgeModel</code>; declaring <code>edgeModel</code> throws.
The check covers the serial and cooperative
kernels. Line features are unaffected.</li>
<li>Groups: features are stably sorted by group ID and one workgroup per group reduces them with a
fixed tree, so group sums are bitwise reproducible. Group centroids are area- (or length-)
weighted means of feature centroids. <strong>Behavior change:</strong> in geographic contexts the mean is
taken on 3D unit vectors (a spherical centroid), not on longitude and latitude numbers, so a
group straddling the antimeridian lands on the antimeridian (for example 179 and -179 give 180)
instead of at longitude 0. An antipodal group has no mean direction and reports NaN; a group
containing an invalid feature reports NaN. Planar groups are unchanged.</li>
<li><code>'geodesic'</code> (<code>pyproj.Geod.geometry_area_perimeter</code>, geo <code>GeodesicArea</code>): lengths and perimeters are
Vincenty geodesic lengths, as <code>'wgs84'</code>, and polygon areas follow geodesic edges on the context's
ellipsoid (WGS84 by default). Edges of 20 km or more are sampled along the geodesic (spans of at most 250 km, at most 16)
and summed as a chord plus the exact parabolic-segment bulge in local equal-area coordinates, so f32
never subtracts two huge areas. Within 1e-6 relative of pyproj on regional and continental polygons
(perimeter 8e-6), where <code>'wgs84'</code> straight edges are off by up to 30% on very long edges. Centroids
still use straight equal-area edges, edges where Vincenty does not converge stay straight, and edges
under 20 km ignore the bulge.</li>
<li><code>output.extremeVertices</code> (<code>uint32x4</code>): per feature <code>[argMinX, argMinY, argMaxX, argMaxY]</code> as global
<code>positions</code> rows, lowest row on ties, <code>0xffffffff</code> for empty features. Geographic x is the unwrapped
longitude of <code>bounds</code>. Per feature only (rejected in <code>groupOutput</code>).</li>
<li><code>geometryType: 'points'</code>: every feature is one point or a multi-point. Outputs are lengths (0), centroids
(vertex mean, as Shapely <code>MultiPoint.centroid</code>), bounds, vertex counts and <code>extremeVertices</code>;
<code>areas</code> is rejected. Group centroids weight by vertex count.</li>
<li><code>GPUGeodesicPairs</code>: distance, initial and final bearing (degrees clockwise from north in
<code>(-180, 180]</code>), and midpoint per pair. <code>'sphere'</code> uses a Vincenty-form central angle that is well
conditioned from millimeters to antipodal; <code>'wgs84'</code> runs bounded Vincenty inverse iteration.
See the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/geodesic-precision#gpugeodesicpairs">precision guide</a> for the iteration limits,
conditioning exclusions, fallback status and CPU twin. luma.gl's
<code>GPUHaversineDistance</code> computes the same sphere distance with a
default radius of 6371 (kilometers); this class takes radius and unit from the spatial context
(6371008.8 m by default) and takes longitude differences on the seam side, so short pairs across
the antimeridian keep their precision in every model.
Pair outputs, including status views, must not overlap one another or either input view;
disjoint slices of one buffer are allowed and checked at construction.</li>
<li><code>GPUGeodesicDestination</code>: destination and final bearing from origin, bearing and distance
columns, on the sphere or by Vincenty's direct solution.
No warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</li>
<li><code>model: 'rhumb'</code> on both classes (turf <code>rhumbDistance</code>, <code>rhumbBearing</code>, <code>rhumbDestination</code>; geo <code>Rhumb</code>):
a line of constant bearing on a sphere of <code>radius</code> (default 6371008.8 m). Pairs give <code>distances</code> (in
<code>radius</code> units), <code>initialBearings</code> equal to <code>finalBearings</code> (constant, in <code>(-180, 180]</code>) and the rhumb
<code>midpoints</code> (the midpoint by distance: the mean latitude, and the longitude of the loxodrome at
that latitude, which is linear in the Mercator latitude and differs from the mean longitude
unless the line runs east-west). Coincident pairs have zero distance and zero initial and final
bearings. The destination longitude is continuous with the origin, and <code>finalBearings</code> flips to the
opposite north-south sense when a move overshoots a pole and is reflected, as turf does (the reflected
case is approximate). Longitude differences take the short way across the antimeridian, latitudes clamp
at 89.9999 degrees, equal-latitude edges use the east-west limit <code>cos(lat)</code>, and the Mercator-stretched
latitude difference is an <code>atanh</code> of a cancellation-free ratio. Sphere only (turf and geo are spherical
too), <code>status.converged</code> is <code>'wgs84'</code>-only, and the model switch is compile-time. Distance error against an f64
closed form was 0.74 m (6e-8 relative) over 12,108 km and bearings were within 1e-4 degrees. Rhumb length
in <code>GPUGeometryMeasures</code> and rhumb densify in <code>GPULineSegmentize</code> are not provided.</li>
<li>Geometry-measure precision (f32 vs f64 oracles, Apple Metal): WGS84 and spherical box areas
within 3e-7 relative from 100 m blocks to 2,000 km boxes. Geodesic-pair and destination bounds
and the distinction between nominal arithmetic measurements and GPU certification are in the
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/geodesic-precision#measurements-and-acceptance-bounds">precision guide</a>.
See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a> for input quantization.</li>
</ul>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUGeometryMeasures */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-measures rung=datum datum=0 caption="Road 0: WGS84 geodesic length in metres"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-measures rung=formula caption="Road length as the sum of geodesic edge distances"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-measures rung=kernel caption="The compute shader that writes the road lengths"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geometry-measures rung=graph caption="Command graph nodes and buffers that produce the road lengths"}</p>
<p>{/* lens:GPUGeodesicPairs */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=datum datum=15 caption="Route 15: ORD to LHR distance and initial bearing"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=kernel caption="The compute shader that runs the capped Vincenty inverse per pair"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-pairs rung=graph caption="Command graph nodes and buffers that produce the pair columns"}</p>
<p>{/* lens:GPUGeodesicDestination */}</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=datum datum=0 caption="Row 0: the destination 500 km due north of ORD"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=kernel caption="The compute shader that runs the capped Vincenty direct per row"}</p>
<p>::lens{scenario=gpu-spatial-analysis.geodesic-destination rung=graph caption="Command graph nodes and buffers that produce the destinations"}</p>
<h3 id="geodesic_wgsl-creategpulocalframe-and-gpulocalframeprojection"><code>GEODESIC_WGSL</code>, <code>createGPULocalFrame</code>, and <code>GPULocalFrameProjection</code></h3>
<p><strong><code>GPULocalFrameProjection</code></strong> — Projects longitude/latitude positions on the GPU into a local azimuthal-equidistant frame in meters.</p>
<p>::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=result caption="Blue: 223 Wicker Park streets projected from longitude/latitude to local metres"}</p>
<p>Two ways to work on longitude/latitude data, used consistently across the package:</p>
<ul>
<li><strong>Measure on the sphere.</strong> <code>GEODESIC_WGSL</code> is the public WGSL source of the spherical functions
every great-circle path in this package uses; include it once in your own compute shader.
Positions are <code>vec2&#x3C;f32>(longitude, latitude)</code> degrees and distances are in the units of the
<code>radius</code> argument (<code>GPU_EARTH_MEAN_RADIUS_METERS</code> for meters, the package default).
<code>geodesicHaversineDistance(a, b, radius)</code> is the great-circle distance;
<code>geodesicSegmentFraction(point, start, end)</code> and
<code>geodesicSegmentDistance(point, start, end, fraction, radius)</code> give the point-to-segment distance
for segments that are straight in longitude/latitude (the edge model of the containment tests and
network edges; exact to first order for kilometer-length segments, densify longer ones). Projection
and interpolation follow the same wrapped start-to-end longitude arc. The fraction helper compares
its projected candidate with both endpoints, so its distance cannot exceed either endpoint distance.
An interior projection survives a rounded distance tie; a tie between endpoint candidates selects
the start. CPU line simplification uses the same wrapped arc and endpoint comparison as the GPU.
Local GPU candidates are ordered by their common haversine terms before inverse trigonometry,
so different <code>asin</code>/<code>atan2</code> rounding cannot displace a local interior projection. Distant candidates
keep central-angle comparisons to preserve resolution near antipodes.
The distance helper evaluates the supplied fraction; it does not select a new fraction. Also
available: <code>geodesicHaversineTerm</code>, <code>geodesicCentralAngle</code>, <code>geodesicInitialBearingDegrees</code>,
<code>geodesicDestination</code>, <code>geodesicInterpolate</code> and <code>geodesicWrapLongitudeDelta</code>.</li>
<li><strong>Project first.</strong> Planar operators throw on longitude/latitude contexts. <code>createGPULocalFrame({origin, kind})</code> returns a binary64 CPU frame in meters: <code>'azimuthal-equidistant'</code> (spherical, mean radius;
distances and bearings from the origin exact) or <code>'lambert-equal-area'</code> (WGS84 ellipsoidal, areas
exact), matching PROJ <code>aeqd +R=6371008.8</code> and <code>laea +ellps=WGS84</code> to a millimeter. It offers
<code>project(lng, lat)</code>, <code>unproject(x, y)</code> and <code>projectArray(lngLats, stride)</code> for uploads, and a
<code>spatialContext</code> that carries the frame identity: contexts of frames with different origins are
incompatible, so data projected around different origins is not mixed. Float32 frame coordinates
keep about 1e-7 of the frame extent (millimeters at 10 km).
See <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a> for float32 degrees.
<code>GPULocalFrameProjection</code> projects GPU-resident <code>float32x2</code> or binary64
(<code>uint32x4</code>) longitude/latitude rows into the frame without a readback (compiled adaptive patches
from <code>gpu-project</code>, default tolerance 1 cm, output in absolute frame meters). Add it with
<code>graph.add(new GPULocalFrameProjection({frame, positions, output, bounds}))</code>.
The frame's <code>spatialContext</code> is <code>{coordinateSpace: 'planar', metric: 'native', units: 'meters', frame}</code>.
Choose <code>'azimuthal-equidistant'</code> for distance operators and <code>'lambert-equal-area'</code> for area/density
operators; the wrong frame kind throws at operator construction. <code>GPULocalFrameProjection.spatialContext</code>
is the frame's context.</li>
</ul>
<h4 id="how-it-computes-2">How it computes</h4>
<p>{/* lens:GPULocalFrameProjection */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=datum datum=0 caption="Vertex 0: its longitude/latitude and projected frame metres"}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=formula caption="Local polynomial patch maps normalised lon/lat to frame metres"}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=kernel caption="The compute shader that writes the projected positions"}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-frame-projection rung=graph caption="Command graph nodes and buffers that produce the projected positions"}</p>
<h3 id="gpulinearreferencing-and-gpulinelocate"><code>GPULinearReferencing</code> and <code>GPULineLocate</code></h3>
<p><strong><code>GPULinearReferencing</code></strong> — Snaps points to their nearest polyline within a radius, reporting the road, distance along it and foot point.</p>
<p>::lens{scenario=gpu-spatial-analysis.linear-referencing rung=result caption="Grey: 223 Wicker Park streets · blue: foot points of 40 Overture places within 500 m"}</p>
<p><strong><code>GPULineLocate</code></strong> — Places events on polylines by measure, writing the located point and a status code per event. A measure outside the line is clamped to the nearest end (status <code>clamped</code>); a NaN measure, an empty line or a line index out of range is <code>invalid</code> with NaN outputs.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-locate rung=result caption="Grey: 223 Wicker Park streets · blue: 40 Overture places located by their measure"}</p>
<p>For <code>GPULinearReferencing.radius</code>, no warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Maxima (functions of the device, default-device values for 128 MiB bindings in parentheses): <code>GPULinearReferencing</code> takes at most <code>getGPULinearReferencingMaximumQueryCount(device)</code> (3,728,270) <code>queryPositions</code> and <code>positions</code> of at most <code>getGPULinearReferencingMaximumPlanarPositionCount(device)</code> (2^23, planar BVH) or <code>getGPULinearReferencingMaximumSphericalPositionCount(device)</code> (2^21, arc table). <code>GPULineLocate</code> takes at most <code>getGPULineLocateMaximumEventCount(device)</code> (5,592,405) events.</p>
<p>Linear referencing on many polylines (<code>positions</code> sorted by path, <code>lineOffsets</code>): snap points to
the nearest path and report where they are along it, or place events on paths by measure.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULinearReferencing</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  queryPositions, positions, lineOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  radius: radiusParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph), </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// one float32 row, per frame</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, spatialSort: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {lineIndices, segmentIndices, fractions, footPoints, distances, measures, sides, signedOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">           status: {candidateOverflow, requiredCandidateCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineLocate</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, lineOffsets, eventLineIndices, eventMeasures, eventOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  measureMode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'distance'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'fraction'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: locateParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {positions: eventPositions, angles, statuses}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">locateParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPULineLocateParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({measureOffset: elapsedMeters})); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// animate</span></span></code></pre>
<ul>
<li><code>GPULinearReferencing</code> (turf <code>nearestPointOnLine</code>, PostGIS <code>ST_ClosestPoint</code> and
<code>ST_LineLocatePoint</code>): every vertex row becomes a segment to the next row of its line and
<code>GPUNearestFeatureJoin</code> finds each point's nearest segment within the radius through its BVH.
Ties go to the smallest segment row (earliest along the line, lowest line). Outputs per point:
line, segment within the line, fraction along the segment, foot point, distance, measure (distance
along the line to the foot point), side (<code>1</code> left of the line direction, <code>-1</code> right, <code>0</code> on it)
and <code>side * distance</code>. Points with nothing in range get <code>0xffffffff</code> indices, distance <code>-1</code>,
side <code>0</code> and NaN elsewhere. Both capacities (BVH leaves and, planar, the bounding-box candidates)
are intermediate: the required <code>output.status.candidateOverflow</code> is 1 when either was too small,
so some points may have missed their nearest line, and the optional
<code>output.status.requiredCandidateCount</code> is the unclamped candidate count (planar; 0 for a
great-circle context), so <code>candidateCapacity</code> of at least that value removes the candidate cause.
There is no <code>count</code> and no <code>overflow</code>: every row is one query point. <strong>Breaking (2026-11-10):</strong>
<code>status.overflow</code> (the same flag) and <code>status.candidateCount</code> are removed, and the unclamped
candidate count moved from <code>status.requiredCount</code> to <code>status.requiredCandidateCount</code>. <code>queryPositions</code> are the points to snap; <code>positions</code> and
<code>lineOffsets</code> are the lines.</li>
<li><code>GPULineLocate</code> (turf <code>along</code>, PostGIS <code>ST_LineInterpolatePoint</code> and <code>ST_LocateAlong</code>): each
event's measure (distance or fraction, optionally <code>measure * scale + offset</code> from the parameter
buffer) is clamped to the line (<code>statuses</code> 1) and located by binary search over cumulative vertex
measures, then moved by an optional lateral offset along the left normal. Outputs: position,
segment, unit tangent, angle (degrees counter-clockwise from +x, the deck.gl <code>getAngle</code>
convention) and status (<code>2</code> for empty or out-of-range lines, NaN position).</li>
<li>Cumulative measures are a per-line Neumaier-compensated f32 prefix (optionally output as
<code>output.vertexMeasures</code>). Planar by default: project geographic data first, or use <code>'spherical'</code> below.</li>
<li><code>output.normalizedMeasures</code> (both coordinate systems) is <code>measures / lineLength</code>, 0 for zero-length
lines and NaN when unmatched (Shapely <code>line_locate_point(normalized=True)</code>).
<code>GPULineLocate({measureMode: 'fraction'})</code> is <code>line_interpolate_point(normalized=True)</code>.</li>
<li><code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> on both classes
(planar is the default): positions are longitude and latitude in degrees and segments are
great-circle arcs. <code>distances</code>, <code>measures</code>, <code>signedOffsets</code> and the search <code>radius</code> use the context's
length unit (meters by default, kilometers with <code>units: 'kilometers'</code>) on its <code>sphereRadius</code>
(default mean Earth radius, 6371008.8 meters; <code>radius</code> already names the search radius).
Both declare <code>measure: 'distance'</code>: a Lambert equal-area local frame throws; use an azimuthal-equidistant frame.
<strong>Breaking:</strong> the <code>coordinateSystem</code> and <code>sphereRadius</code> props are removed in favor of
<code>spatialContext</code>; other contexts throw.
The foot point is the closest point on the arc (geo <code>HaversineClosestPoint</code>), with a longitude that is
continuous with the segment start, so it can exceed 180 across the antimeridian. <code>distances</code> is the
cross-track distance and <code>measures</code> the along-track measure. <code>GPULineLocate</code> slerps along the arc
(<code>Geod.fwd</code>, <code>npts</code>): <code>tangents</code> are <code>(east, north)</code> compass components, offsets move along the left
compass bearing in the context's length unit and <code>angles</code> are counter-clockwise from east. The nearest-arc search walks
a <code>GPUBVH</code> over conservative longitude/latitude boxes of the arcs (the great circle's latitude peak
included; arcs across the antimeridian or near a pole span every longitude), Hilbert sorted from 256
positions (<code>spatialSort</code>), branch-and-bound with a great-circle box lower bound and exact arc
distances at the leaves, ties to the lowest segment row. It needs no <code>candidateCapacity</code>;
<code>output.status.overflow</code> is 1 only when <code>leafCapacity</code> is below the position count. Arcs over about
179 degrees are unsupported. Against <code>pyproj.Geod(a=6371008.8, f=0)</code> the worst error was 1.5 m for distance, 0.8 m for measure and 1.7 m for the foot point.</li>
</ul>
<h4 id="how-it-computes-3">How it computes</h4>
<p>{/* lens:GPULinearReferencing */}</p>
<p>::lens{scenario=gpu-spatial-analysis.linear-referencing rung=datum datum=0 caption="Place 0: nearest road, distance along it and distance to it"}</p>
<p>::lens{scenario=gpu-spatial-analysis.linear-referencing rung=formula caption="Measure along the road: prefix length plus fraction of the nearest segment"}</p>
<p>::lens{scenario=gpu-spatial-analysis.linear-referencing rung=kernel caption="The compute shader that writes the foot points and measures"}</p>
<p>::lens{scenario=gpu-spatial-analysis.linear-referencing rung=graph caption="Command graph nodes and buffers that produce the foot points and measures"}</p>
<p>{/* lens:GPULineLocate */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-locate rung=datum datum=0 caption="Place 0: nearest road, measure along it and status"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-locate rung=formula caption="Event measure becomes a segment fraction, then a point on that segment"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-locate rung=kernel caption="The compute shader that writes the located points"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-locate rung=graph caption="Command graph nodes and buffers that produce the located points"}</p>
<p>{/* parameters: GPULineLocate generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulinelocate-parameters"><code>GPULineLocate</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULineLocateParameterValues(settings)</code>, writes <code>GPU_LINE_LOCATE_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULineLocateParameterValues</code>:</p>
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
<td><code>measureScale</code> (optional)</td>
<td><code>number</code></td>
<td>Multiplies every event measure before locating. Default 1.</td>
</tr>
<tr>
<td><code>measureOffset</code> (optional)</td>
<td><code>number</code></td>
<td>Added to every scaled event measure, for example elapsed distance in an animation. Default 0.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpulinesplit-and-gpulinemerge"><code>GPULineSplit</code> and <code>GPULineMerge</code></h3>
<p><strong><code>GPULineSplit</code></strong> — Splits polylines at their mutual intersections, writing the pieces with their source-line indices.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-split rung=result caption="Grey: 223 Wicker Park streets · blue: pieces split at street intersections"}</p>
<p><strong><code>GPULineMerge</code></strong> — Merges polylines that meet end to end at degree-two nodes into longer chains.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-merge rung=result caption="Grey: 223 Wicker Park streets · blue: chains merged through degree-two endpoints"}</p>
<p>Maximum (<code>GPULineSplit</code>): <code>lines.positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows, and the line count plus four times <code>intersectionCapacity</code> is at most <code>getGPULineSplitMaximumPieceBound(device)</code> (4194304 on a default device).</p>
<p><code>GPULineSplit</code> is planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first. <code>GPULineMerge</code> also accepts a longitude-latitude <code>spatialContext</code>: lines join only where end points have identical coordinates (bit patterns), nothing is measured, and longitude 180 and -180 are different coordinates.</p>
<p><code>GPULineSplit</code> splits linestrings at every crossing, touch, shared vertex and overlap end (turf <code>lineSplit</code>
against the union of the lines; verified against Shapely <code>unary_union</code>). It reuses <code>GPUSegmentIntersection</code>
in self mode, groups the split events by segment with a stable <code>GPUSort</code>, removes duplicates and writes
GeoArrow-layout pieces. Props: <code>lines</code> (<code>{kind: 'lines', positions, lineOffsets}</code>), <code>intersectionCapacity</code>,
<code>output</code>, optional <code>spatialSort</code> and <code>leafCapacity</code>. <code>output</code> is <code>{pieces, status?: {uncertainCount?}}</code>; <code>pieces</code> is a
<code>GPUGeneratedGeometryPort</code>: <code>geometry</code> holds <code>{kind: 'lines', positions, lineOffsets}</code>, <code>sourceIds</code>
identifies the source line, and <code>status</code> holds <code>count</code> (pieces, not vertices), <code>overflow</code>, optional <code>candidateOverflow</code>,
optional <code>requiredCount</code> and optional <code>vertexCount</code> and <code>requiredVertexCount</code>, which report the flattened vertex cardinality. Pieces are ordered by
source line, then traversal order. Splits at line ends are ignored, and a line with fewer than two
vertices yields no piece. Piece or vertex truncation sets <code>status.overflow</code>; an insufficient
<code>intersectionCapacity</code> independently sets <code>status.candidateOverflow</code>.</p>
<p><code>GPULineMerge</code> is the inverse (Shapely <code>linemerge</code>, turf <code>lineMerge</code>): it joins linestrings at endpoints
shared by exactly two line ends (an exact f32 match) into maximal chains. Junctions of three or more ends
stop a chain, a closed line never joins itself, and lines with fewer than two vertices are ignored. With
<code>directed: true</code> (Shapely <code>line_merge(directed=True)</code>) only a line end meeting a line start joins, chains
follow the input direction and <code>lineReversed</code> is all 0.
Props: <code>positions</code>, <code>lineOffsets</code> and <code>output: {chainOffsets, positions, status: {count, requiredCount?, overflow?, vertexCount?, requiredVertexCount?}, sourceIds?, lineChains?, lineOrders?, lineReversed?}</code>;
<code>status.count</code> counts chains. The capacities always suffice, so <code>requiredCount</code> equals <code>count</code> and
<code>overflow</code> is 0. <code>sourceIds</code> is the source row of each chain, its first line; <code>lineChains</code> and
<code>lineOrders</code> list every member. The list-ranking rounds are the shared helper in <code>line-merge/list-ranking.ts</code>, which <code>GPUSharedPaths</code> also uses. The result is deterministic: chains are ordered by their lowest line, paths start at the
lower endpoint ID and cycles start at their lowest line and are emitted closed. Chains are ranked by
pointer jumping (list ranking), so a chain of <code>n</code> lines takes <code>O(log n)</code> dispatches rather than <code>n</code>
serial steps. Merging with a tolerance is not provided.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineSplit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  lines: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, lineOffsets}, intersectionCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    pieces: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      geometry: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: piecePositions, lineOffsets: offsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      sourceIds: lineIds,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      status: {count, overflow, vertexCount}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineMerge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({positions, lineOffsets, output: {chainOffsets, positions: merged, status: {count}}}));</span></span></code></pre>
<p><code>GPUNetworkNoding</code> in <a href="/luma-contributors/docs/api/gpu-network">GPU Network</a> builds a routable
network from the pieces.</p>
<h4 id="how-it-computes-4">How it computes</h4>
<p>{/* lens:GPULineSplit */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-split rung=datum datum=0 caption="Piece 0: the first sub-line between intersections"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-split rung=kernel caption="The compute shader that writes the split pieces"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-split rung=graph caption="Command graph nodes and buffers that produce the split pieces"}</p>
<p>{/* lens:GPULineMerge */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-merge rung=datum datum=0 caption="Road 0: the merged chain it belongs to"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-merge rung=kernel caption="The compute shader that writes the merged chains"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-merge rung=graph caption="Command graph nodes and buffers that produce the merged chains"}</p>
<h3 id="gpulineclipbypolygon-and-gpusharedpaths"><code>GPULineClipByPolygon</code> and <code>GPUSharedPaths</code></h3>
<p><strong><code>GPULineClipByPolygon</code></strong> — Clips polylines against a polygon ring, keeping the pieces inside it.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=result caption="Grey: 223 Wicker Park streets · blue: pieces inside the West Town boundary"}</p>
<p><strong><code>GPUSharedPaths</code></strong> — Finds the stretches that two sets of polylines share, writing them as runs with their source-line indices.</p>
<p>::lens{scenario=gpu-spatial-analysis.shared-paths rung=result caption="Grey: Chicago community-area boundaries · blue: stretches West Town shares with its neighbours"}</p>
<p>Maximum: each geometry has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) <code>positions</code> rows. <code>GPULineClipByPolygon</code> also needs <code>lines.positions</code> rows plus twice <code>intersectionCapacity</code> at most <code>getGPULineClipByPolygonMaximumSubPieceBound(device)</code> (4194304 on a default device); <code>GPUSharedPaths</code> needs <code>intersectionCapacity</code> at most <code>getGPUSharedPathsMaximumIntersectionCapacity(device)</code> (2796202 on a default device).</p>
<p>Both classes are planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first.</p>
<p><code>GPULineClipByPolygon</code> clips linestrings to polygons (GeoPandas <code>clip(lines, polygons)</code>, Shapely
<code>intersection(line, polygon)</code>); <code>mode: 'outside'</code> is <code>difference</code>. It is bounded because it never computes
a new intersection between two polygons: every output vertex is an input vertex or a crossing of one line
segment with one polygon edge. <code>GPUSegmentIntersection</code> lists the crossings, each segment's crossings are
ranked in place (pairs arrive sorted by line segment) and deduplicated into sub-pieces. Sub-pieces inside a
collinear overlap are boundary and count as inside; the rest are classified by <code>GPUPointInPolygonJoin</code> at
their midpoint, and kept sub-pieces that continue each other merge. Several polygon features act as their
union. Props: <code>lines</code>, <code>polygons</code>, <code>mode</code>, <code>intersectionCapacity</code>, <code>candidateCapacity</code> and
<code>output: {pieces, status?: {uncertainCount}}</code> (<code>pieces</code> has the <code>GPULineSplitPieces</code> layout). Overflow of pairs,
candidates, pieces or vertices sets <code>output.pieces.status.overflow</code>; the optional <code>output.pieces.status.candidateOverflow</code> is set only when the <code>intersectionCapacity</code> pair list was exceeded, as <code>GPULineSplit</code> reports it. <code>status.count</code> counts pieces. Limits: a line is not noded at its own crossings, a line running over itself keeps both
traversals (Shapely dedupes), a point-only touch yields no piece (<code>keep_geom_type=True</code>), and a sub-piece a
few f32 ulps long beside an edge can be misclassified. Against Shapely 2.1.2, 52 lines in both modes
matched on total length (under 2e-3) and piece end points.</p>
<p><code>GPUSharedPaths</code> is Shapely <code>shared_paths(a, b)</code> for many line pairs. Collinear overlaps from
<code>GPUSegmentIntersection</code> are oriented along the left line, tagged forward or backward by the exact sign of
the segment directions, and chained into one polyline per (left line, right line, direction). The output
<code>output: {runs, status}</code>: <code>runs</code> (<code>GPUSharedPathsRuns</code>) has <code>leftSourceIds</code> and <code>rightSourceIds</code> (the <code>left</code> and
<code>right</code> line rows of each run, contributor contract rule 8; <strong>breaking (2026-11-10):</strong> renamed from <code>leftLineIds</code> and
<code>rightLineIds</code>), <code>forward</code>, <code>offsets</code> and <code>positions</code>; <code>status</code> has <code>count</code>, <code>vertexCount?</code>, <code>overflow?</code>, <code>requiredCount?</code>, <code>requiredVertexCount?</code>,
<code>uncertainCount?</code> and <code>candidateOverflow?</code> (<code>count</code> counts runs, not vertices; <code>candidateOverflow</code> is 1 when the
<code>intersectionCapacity</code> pair list overflowed, which also sets <code>overflow</code>). Closed left lines are cut at their first vertex and two
coincident right segments emit duplicate runs. Runs are sized and written by list ranking (Wyllie pointer jumping, <code>ceil(log2(intersectionCapacity))</code>
rounds per pass) rather than one invocation walking each run, so a long shared road costs the same dispatches as a short one; the work is
<code>O(intersectionCapacity * log(intersectionCapacity))</code>.</p>
<h4 id="how-it-computes-5">How it computes</h4>
<p>{/* lens:GPULineClipByPolygon */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=datum datum=0 caption="Piece 0: the first street piece inside West Town"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=kernel caption="The compute shader that writes the clipped pieces"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-clip-by-polygon rung=graph caption="Command graph nodes and buffers that produce the clipped pieces"}</p>
<p>{/* lens:GPUSharedPaths */}</p>
<p>::lens{scenario=gpu-spatial-analysis.shared-paths rung=datum datum=0 caption="Run 0: the first stretch West Town shares with a neighbour"}</p>
<p>::lens{scenario=gpu-spatial-analysis.shared-paths rung=kernel caption="The compute shader that writes the shared runs"}</p>
<p>::lens{scenario=gpu-spatial-analysis.shared-paths rung=graph caption="Command graph nodes and buffers that produce the shared runs"}</p>
<h3 id="gpulinesimplification"><code>GPULineSimplification</code></h3>
<p>Simplifies polylines with Visvalingam or Douglas-Peucker importance, keeping vertices above a metre tolerance.</p>
<p>::lens{scenario=gpu-spatial-analysis.line-simplification rung=result caption="Grey: 223 Wicker Park streets · blue: simplified with a 5 m tolerance"}</p>
<p>Pass optional <code>spatialContext</code> to <code>getGPULineSimplificationParameterValues(settings)</code> to warn on short Douglas-Peucker <code>tolerance</code> values; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p><code>spatialContext</code> defaults to native planar coordinates, with the tolerance in position units. <code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> (optional <code>sphereRadius</code>, default mean Earth radius) runs Douglas-Peucker (both metrics) with great-circle distances: <code>tolerance</code> and <code>importance</code> use the context's length unit (meters by default, now also kilometers with <code>units: 'kilometers'</code>), chords are straight in longitude/latitude, and the distance from a vertex to its chord is the haversine distance to the closest point of the chord (found in the local equirectangular frame of the vertex, exact to first order for chords of kilometers). The <code>'time-ratio'</code> expected position is interpolated linearly in longitude/latitude with the longitude difference wrapped. Great-circle distances use f32 transcendental functions, so the bit-exactness below holds for planar contexts only. <code>method: 'visvalingam'</code> is planar only and throws in a great-circle context (its tolerance is an area, its rounds mirror a bit-exact planar triangle area, and no spherical-excess variant exists); other metrics throw. For Visvalingam, or to keep planar bit-exactness, project first (<code>createGPULocalFrame</code>, <code>GPULocalFrameProjection</code>). Douglas-Peucker declares <code>measure: 'distance'</code> (a Lambert equal-area frame throws; use azimuthal-equidistant); Visvalingam declares <code>measure: 'area'</code> (an azimuthal-equidistant frame throws; use Lambert equal-area).</p>
<p>Douglas-Peucker (and approximate Visvalingam-Whyatt) simplification of many polylines or tracks, stored as rows sorted by line and
delimited by <code>lineOffsets</code>. It runs in two parts: a
per-vertex <code>importance</code> column computed once, and a per-frame selection at a tolerance from a
parameter buffer. Changing the tolerance re-runs only the mask, compaction, and publish nodes.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// One-shot graph: importance.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">importanceGraph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineSimplification</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, lineOffsets, importance,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  metric: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'segment'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// or 'time-ratio' (TD-TR) with float32 \`timestamps\`</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {status: {converged, iterationCount}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Per-frame graph: selection only.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">frameGraph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPULineSimplification</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, lineOffsets, importance, computeImportance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: toleranceParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(frameGraph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {ids, status: {count, overflow}, outputMask, lineCounts, lineStarts}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">toleranceParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPULineSimplificationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({tolerance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">})); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// no recompile</span></span></code></pre>
<ul>
<li>Importance: every line starts as one open interval between its endpoints (importance
<code>+Infinity</code>). Each round, every open interval picks its farthest row (segmented argmax with ties
to the smallest row) and splits there. The row's importance is
<code>min(distance, min(importance[a], importance[b]))</code>, so importance never increases from a split
to its children. An interval whose largest distance is 0 is resolved at once with importance 0.</li>
<li>Rounds are GPU gated. They stop when no row is left (<code>status.converged = 1</code>) or after <code>maximumIterations</code>. The rounds are unrolled, so the command node count grows with <code>maximumIterations</code>: <code>estimatedCommandNodeCount</code> (exact for the importance rounds, excluding the <code>GPUCompaction</code> nodes of a row selection) is checked at construction against the shared 32,768-node budget (<code>validateCommandNodeBudget</code>); the largest accepted <code>maximumIterations</code> (1024) needs about five thousand nodes, so the budget is never the limit.
The CPU twin also honors an explicitly supplied iteration limit and reports <code>converged = 0</code>
when work remains, retaining the current importance. Unresolved Visvalingam rows have infinite
importance; unresolved Douglas-Peucker rows retain their parent split bound. Omitting the CPU
limit leaves its sequential computation uncapped; CPU Visvalingam counts sequential removals.
A balanced split tree takes about <code>log2(n)</code> rounds and a spiral can take up to <code>n - 2</code>. If the
cap is reached, the remaining rows get their interval's split importance, so the kept set is a
superset of Douglas-Peucker's.</li>
<li>Selection: a row is kept when <code>importance > tolerance</code>. <code>output.ids</code> holds the kept rows in
ascending order, and <code>status.count</code> is clamped so it can drive an indirect draw. <code>outputMask</code> is the
per-row 0/1 mask. <code>lineCounts</code> (not
clamped) and <code>lineStarts</code> give each line's range inside <code>output.ids</code>.</li>
<li>Exactness: with <code>status.converged = 1</code>, the kept set at any tolerance <code>e</code> equals classic recursive
Douglas-Peucker at <code>e</code> (split when <code>dmax > e</code>, first maximum on ties) under the same metric.
The f32 distances are bit-identical to a <code>Math.fround</code> CPU oracle: products are prevented from
fusing into FMAs, and division and square root are corrected to defined values.</li>
<li>Metrics: <code>'segment'</code> measures the distance to the chord segment, which is the perpendicular
distance inside the segment and the distance to the nearer anchor outside it (GEOS and
simplify-js behave the same way). <code>'time-ratio'</code> is TD-TR's synchronized Euclidean distance,
which needs float32 <code>timestamps</code>.</li>
<li>Use tile- or view-local coordinates. Simplified lines may self-intersect, because topology is
not preserved (Douglas-Peucker and Visvalingam alike).</li>
</ul>
<p><strong>Visvalingam-Whyatt.</strong> <code>method: 'visvalingam'</code> writes each vertex's effective triangle area (the context's
length unit squared; square meters in a default local frame) into <code>importance</code>, and the per-frame <code>tolerance</code> is then an area: a vertex is kept when
<code>importance > tolerance</code>, and endpoints are always kept. The selection outputs are the same as for
Douglas-Peucker. Each round removes every surviving vertex whose area is the smallest within
<code>neighborhoodRadius</code> steps both ways (1 to 8, default 3, compile-time), then unlinks them; the effective
area is the maximum of its own area and those of its removed neighbors, which is the sequential running
maximum. It is approximate: the sequential heap order cannot be parallelised exactly. Against
<code>simplification.simplify_coords_vw</code> on 3000-vertex random walks the kept set differed by 3.1, 0.65 and 0.13
percent of vertices at radius 1, 2 and 3, and lines up to 300 vertices matched exactly at radius 3. It needs
about 31 rounds for 300 vertices and 55 for 3000; <code>maximumIterations</code> (default 256) caps them, and undecided
rows get <code>+Infinity</code> (a superset). Not supported: <code>metric: 'time-ratio'</code> and topology-preserving VW
(<code>simplify_coords_vwp</code>).</p>
<h4 id="how-it-computes-6">How it computes</h4>
<p>{/* lens:GPULineSimplification */}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-simplification rung=datum datum=0 caption="Vertex 0: Douglas-Peucker importance in metres"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-simplification rung=kernel caption="The compute shader that writes the importance and retained vertices"}</p>
<p>::lens{scenario=gpu-spatial-analysis.line-simplification rung=graph caption="Command graph nodes and buffers that produce the importance and retained vertices"}</p>
<p>{/* parameters: GPULineSimplification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpulinesimplification-parameters"><code>GPULineSimplification</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULineSimplificationParameterValues(settings)</code>, writes <code>GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULineSimplificationParameterValues</code>:</p>
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
<td><code>tolerance</code></td>
<td><code>number</code></td>
<td>Douglas-Peucker tolerance in position units (the context length unit in a great-circle context). A vertex is kept when its importance is strictly greater than the tolerance, which is the classic <code>dmax > epsilon</code> split rule. Line endpoints are always kept. Visvalingam uses an effective-area tolerance in squared position units. Must be finite and non-negative.</td>
</tr>
<tr>
<td><code>method</code> (optional)</td>
<td><code>GPULineSimplificationMethod</code></td>
<td>Algorithm used by the operator. Defaults to <code>'douglas-peucker'</code>; <code>'visvalingam'</code> uses an area tolerance and skips the distance warning. Does not change the packed values.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpucoveragesimplification"><code>GPUCoverageSimplification</code></h3>
<p>Simplifies a polygon coverage with Douglas-Peucker, simplifying each shared border once so neighbours stay gap-free, and writes the kept vertices.</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=result caption="Grey: 17 community areas, 3,201 vertices · blue: 397 after 50 m simplification, borders still shared"}</p>
<p>Gap-free simplification of a polygon coverage (Shapely <code>simplify_coverage</code>, PostGIS <code>ST_CoverageSimplify</code>).
Every arc shared by two polygons is simplified once, so both neighbors keep the same vertices along it.
Arcs are runs of ring edges with the same partner polygon, broken at each ring start; the lower polygon ID
owns a shared arc. The kept vertices are the Douglas-Peucker survivors of the owned arcs plus the junctions,
shared through point IDs, and the output rings are the input rings filtered by the mask, in the same order
and direction.</p>
<p>Props: the four-level polygon layout (<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>; rows
are features and a ring's coverage identity is its feature),
<code>snapTolerance</code> (default 0; vertices snap to cells <code>floor((coordinate - origin) / snapTolerance + 0.5)</code> with
the first vertex as origin, so every vertex must be within <code>2^24 * snapTolerance</code> of the first vertex on both
axes; vertices outside that range are counted in the optional <code>status.uncertainCount</code>, not in <code>overflow</code>), <code>parameters</code> (packed with <code>getGPULineSimplificationParameterValues</code>; the
tolerance changes per frame without recompiling), <code>maximumIterations</code>, and <code>output</code> with
<code>positions</code> (the kept vertices, capacity), <code>ringOffsets</code> (<code>ringCount + 1</code>), optional <code>outputMask</code> (per input
vertex), and <code>status: {overflow, requiredCount?, converged?, uncertainCount?}</code>. Feature and polygon offsets are
unchanged.</p>
<p>Maximum: <code>positions</code> has at most <code>getGPUSegmentIntersectionMaximumVertexCount(device)</code> (4194304 on a default device: the 32-byte segment-table row per vertex must fit one storage binding) rows.</p>
<p>Planar only: <code>spatialContext</code> defaults to native planar coordinates and a longitude-latitude context throws; project to a planar frame first (<code>createGPULocalFrame</code>, <code>GPULocalFrameProjection</code>). The tolerance is in position units. It measures distance, so <code>local-lambert-equal-area</code> frames throw.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCoverageSimplification</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, featureOffsets, polygonOffsets, ringOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: toleranceBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {positions: outPositions, ringOffsets: outRingOffsets, status: {overflow}}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">toleranceBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPULineSimplificationParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({tolerance: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.05</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Topology preservation: arc endpoints (junctions and ring starts) are always kept and a ring never keeps
fewer than three vertices (the farthest original vertices are restored, even with <code>topologyIterationCount: 0</code>).
Then <code>topologyIterationCount</code> (compile-time, default 4, 0 turns the repair off) iterations compact the
simplified rings, run <code>GPUSegmentIntersection</code> with exact predicates and, for every pair of segments that
cross, touch or overlap without sharing a coverage point, restore the original vertex farthest from each
segment (one Douglas-Peucker split; decisions go through point IDs, so neighbors still agree). Each
iteration also tests every kept vertex against the area between each span's original chain and its shortcut
(exact point-in-ring signs, candidates from one BVH over the input segments): a shortcut that passes over an
island crosses nothing but would move the island into the neighboring polygon, so such a span is repaired
the same way. <code>output.topologyStatistics</code> (4 <code>uint32</code>, length
<code>GPU_COVERAGE_SIMPLIFICATION_TOPOLOGY_STATISTICS_LENGTH</code>) reports the crossings and swept spans before repair,
those remaining, the vertices restored and a candidate-pair overflow flag; <code>topologyPairCapacity</code> (default <code>max(256, 4 * vertices)</code>) sizes the candidate list. Residual crossings
are reported, never hidden, and input that already crosses itself cannot be repaired (it counts in both
numbers). On a wavy-strip scene the repaired output passes Shapely <code>coverage_is_valid</code>, where plain
Douglas-Peucker gave three invalid polygons. The criterion is distance, not area (Shapely's
<code>simplify_coverage</code> is area based and keeps a different vertex set), and there is no guarantee of zero
residual within the iteration count.</p>
<p><code>simplifyBoundary: false</code> (Shapely <code>coverage_simplify(simplify_boundary=False)</code>) keeps every vertex that
touches an unshared edge and simplifies only the shared arcs.</p>
<p>One thread walks each arc, so one huge arc is a serial walk (and a long dropped span costs a serial walk
per kept vertex), and non-finite vertices are not supported. A shared arc that crosses a ring start keeps
one extra vertex.</p>
<h4 id="how-it-computes-7">How it computes</h4>
<p>{/* lens:GPUCoverageSimplification */}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=datum datum=0 caption="Vertex 0: a ring start, always kept as an arc endpoint"}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=kernel caption="The compute shaders that mark arcs, simplify them and repair crossings"}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-simplification rung=graph caption="Command graph nodes and buffers that produce the kept vertices"}</p>
<p>{/* parameters: GPUCoverageSimplification generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpucoveragesimplification-parameters"><code>GPUCoverageSimplification</code> parameters</h4>
<ul>
<li>Packer: <code>getGPULineSimplificationParameterValues(settings)</code>, writes <code>GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPULineSimplificationParameterValues</code>:</p>
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
<td><code>tolerance</code></td>
<td><code>number</code></td>
<td>Douglas-Peucker tolerance in position units (the context length unit in a great-circle context). A vertex is kept when its importance is strictly greater than the tolerance, which is the classic <code>dmax > epsilon</code> split rule. Line endpoints are always kept. Visvalingam uses an effective-area tolerance in squared position units. Must be finite and non-negative.</td>
</tr>
<tr>
<td><code>method</code> (optional)</td>
<td><code>GPULineSimplificationMethod</code></td>
<td>Algorithm used by the operator. Defaults to <code>'douglas-peucker'</code>; <code>'visvalingam'</code> uses an area tolerance and skips the distance warning. Does not change the packed values.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpucoveragedissolve"><code>GPUCoverageDissolve</code></h3>
<p>Merges the features of a polygon coverage that share a label, writing the boundary rings of each merged region.</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=result caption="Grey: 17 community areas · blue: 4 rings after dissolving three west-to-east bands (labels from easting)"}</p>
<p>Dissolves a polygon coverage by a per-feature <code>uint32</code> label (the four-level polygon layout; GeoPandas <code>dissolve(by, method='coverage')</code>,
Shapely <code>coverage_union_all</code> per label). An edge is dropped when another polygon with the same label has
the same edge (snapped point IDs, <code>(min, max)</code> keys). Kept edges are oriented interior-left and chained by
<code>GPUSegmentRingAssembly</code> with the label as the group, so the output has shells, holes and
<code>polygons.sourceIds</code>. Vertices at nodes are kept, as GEOS does. This is not the excluded vector
overlay: a coverage has no crossings, so the result is a subset of the input edges and is bounded by the
input vertex count plus one closing vertex per ring, with ring and vertex capacities and an overflow flag.
The input must be a valid coverage (check it with <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-validity#gpucoveragevalidity"><code>GPUCoverageValidity</code></a>).
Planar only: <code>spatialContext</code> defaults to native planar coordinates and <code>longitude-latitude</code> throws.
<code>snapTolerance</code> snaps vertices (cells relative to the first vertex of <code>positions</code>; every vertex must be
within <code>2^24 * snapTolerance</code> of it on both axes; vertices outside that range are counted in the optional
<code>status.uncertainCount</code>, and <code>overflow</code> stays for dropped rows) and
the output uses the coordinates of the lowest vertex index of each point. Ring winding is the exact turn at
each ring's lexicographically smallest vertex. Dissolve edge sets,
area, parts and holes matched <code>gdf.dissolve(method='coverage')</code> exactly on 5 scenes and 18 labels. The
per-ring area walk is serial per ring. The directory is <code>polygon-coverage-topology/</code> because <code>.gitignore</code>
ignores <code>coverage-*/</code>.</p>
<h4 id="how-it-computes-8">How it computes</h4>
<p>{/* lens:GPUCoverageDissolve */}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=datum datum=0 caption="Area 0 (community area 3): its band label"}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=kernel caption="The compute shaders that match edges and orient the surviving boundary"}</p>
<p>::lens{scenario=gpu-spatial-analysis.coverage-dissolve rung=graph caption="Command graph nodes, sorts and ring assembly that produce the dissolved rings"}</p>
<h3 id="gputrajectorymetrics"><code>GPUTrajectoryMetrics</code></h3>
<p>Computes per-track length, duration, average and maximum speed, and a bounded stop list from time-ordered positions.</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=result caption="Grey: 12 AIS tracks · right: the same tracks coloured by average speed in m/s"}</p>
<p>Per-track metrics for rows sorted by track and time and delimited by <code>trackOffsets</code>: path length,
duration, average speed, and maximum speed (via <code>GPUSegmentedReduction</code>). Stop detection finds
maximal runs of slow steps (speed below a per-frame threshold) that last at least a per-frame
minimum duration. It publishes a bounded stop list in row order (track, start row, end row,
duration, centroid) and unclamped per-track stop counts.</p>
<p><code>timestamps</code> may be <code>float32</code>, <code>float32</code> plus <code>timestampsLow</code> (double-single), or exact <code>Int64</code>
words (<code>uint32x2</code>). Every time difference (steps, track durations, dwells) is taken exactly or in
double-single before conversion to float32, so multi-year epoch-millisecond tracks keep
millisecond steps. Durations and <code>stopMinimumDuration</code> use the timestamp unit; speeds and
<code>stopSpeedThreshold</code> use the context's length unit per timestamp unit.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrajectoryMetrics</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, timestamps, trackOffsets, parameters: stopParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    trackLengths, averageSpeeds, maximumSpeeds, trackStopCounts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    stops: {ids: stopTracks, status: {count, overflow}, startRows, endRows, centroids, durations}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">stopParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTrajectoryMetricsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({stopSpeedThreshold: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, stopMinimumDuration: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">300</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Optional per-vertex columns (<code>positions.length</code> float32 rows) describe the step that ends at each row:
<code>stepSpeeds</code> (distance over delta time, 0 if the delta time is not positive), <code>stepHeadings</code> (<code>atan2(dy, dx)</code> in radians, 0 for a zero-distance step) and <code>stepAccelerations</code> (<code>(speed_i - speed_{i-1}) / deltaTime_i</code>, 0 for the first two rows of a track). The first row of a track and rows outside tracks are 0,
so a drawn segment takes the value of its end row. The track inputs are the trajectory layout
(<code>positions</code>, <code>timestamps</code>, <code>trackOffsets</code>; <code>GPUTrajectoryGeometryFields</code>, timestamps <code>float32</code> or Int64
<code>uint32x2</code> words), every result view goes in the
<code>output</code> bag, and <code>output.stops</code> is a <code>GPUCompactOutput</code> extended with the per-stop columns. The packer
takes a <code>GPUTrajectoryMetricsSettings</code>. <code>output.stops.drawInstanceCount</code> (one <code>uint32</code> row) receives
the clamped stop count, for example an indirect draw record's instance count.</p>
<p><code>spatialContext</code> is optional. Omitted (or planar) means Euclidean distances in position units and headings
<code>atan2(dy, dx)</code>. With <code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> (optional
<code>sphereRadius</code>, default the mean Earth radius in the context's length unit) the positions are <code>(longitude, latitude)</code> degrees,
step distances and <code>trackLengths</code> are in the context's length unit (meters by default; kilometers with
<code>units: 'kilometers'</code>); speeds and <code>stopSpeedThreshold</code> use that unit per timestamp unit.
<code>stepHeadings</code> are initial bearings in the planar heading convention, radians counterclockwise from
east in <code>(-pi, pi]</code> (<code>pi / 2 - bearing</code>, where <code>GPUGeodesicPairs</code> reports the bearing in degrees clockwise
from north), so headings mean the same in every context. Longitude/latitude without the <code>great-circle</code> metric throws. It measures distance, so <code>local-lambert-equal-area</code> frames throw.
Stop <code>centroids</code> are arithmetic means of coordinates, so a dwell straddling the antimeridian or a pole has a
meaningless centroid. For stop detection, the affected distance is <code>stopSpeedThreshold</code> × sampling
interval: float32 position noise is about ±0.8 m/s at 1 Hz in Chicago, so use a local frame.
No warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.
The <code>spatialContext</code> prop used to be required and
planar-only.</p>
<h4 id="how-it-computes-9">How it computes</h4>
<p>{/* lens:GPUTrajectoryMetrics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=datum datum=0 caption="Track 0 (MMSI 366993970): length, duration, average speed and stops"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=formula caption="Track length as summed step distances; average speed is length over duration"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=kernel caption="The compute shaders that write the track metrics and stops"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-metrics rung=graph caption="Command graph nodes and buffers that produce the track metrics"}</p>
<p>{/* parameters: GPUTrajectoryMetrics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectorymetrics-parameters"><code>GPUTrajectoryMetrics</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryMetricsParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryMetricsParameterValues</code>:</p>
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
<td><code>stopSpeedThreshold</code></td>
<td><code>number</code></td>
<td>Speed threshold in the context's length unit per timestamp unit. A step is slow below this value. Must be finite and non-negative. A threshold of 0 only treats zero-distance steps as slow.</td>
</tr>
<tr>
<td><code>stopMinimumDuration</code></td>
<td><code>number</code></td>
<td>A dwell (a maximal run of slow steps) is reported as a stop when its duration is at least this value. Must be finite and non-negative.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectoryplayhead-and-gputrajectoryresample"><code>GPUTrajectoryPlayhead</code> and <code>GPUTrajectoryResample</code></h3>
<p><strong><code>GPUTrajectoryPlayhead</code></strong> — Interpolates every track at a playhead time, writing position, heading, speed and status per track.</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=result caption="Grey: 12 AIS tracks · dots: each track at 15:00Z by status, 9 active"}</p>
<p><strong><code>GPUTrajectoryResample</code></strong> — Resamples each track to a fixed number of points, spaced evenly in time, path length or on a shared clock.</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=result caption="Grey: 12 AIS tracks · blue: 16 samples per track, evenly spaced along path length"}</p>
<p><code>GPUTrajectoryPlayhead</code> interpolates every track at a per-frame playhead: one invocation per track
binary-searches the track's sorted timestamps and interpolates the bracketing segment. It writes
per-track columns (position, optional elevation, heading in radians, speed, status, segment row and
fraction) and an optional compact, ascending list of active tracks with a clamped count, so a
layer can draw one instance per moving object with an indirect instance count. Moving the playhead
only rewrites the parameter buffer.</p>
<p>Maximum (<code>GPUTrajectoryPlayhead</code>): the track count is at most <code>getGPUTrajectoryPlayheadMaximumTrackCount(device)</code> (6710886 on a default device).</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrajectoryPlayhead</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, timestamps, trackOffsets, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// same layout as GPUTrajectoryMetrics</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  elevations, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// optional float32 z per row</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  parameters: playheadParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    currentPositions, currentElevations, headings, speeds, trackStatus, segmentRows, segmentFractions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    activeTracks: {ids, status: {count, overflow}}, drawInstanceCount</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">playheadParameters.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUTrajectoryPlayheadParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({playhead: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">125.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, maximumGap: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">60</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Int64 epoch-ms words: a uint32 parameter buffer and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// getGPUTrajectoryPlayheadWordParameterValues({playhead: Date.now() + 0.25, maximumGap: 60_000})</span></span></code></pre>
<ul>
<li>Status (<code>output.trackStatus</code>, <code>GPU_TRAJECTORY_PLAYHEAD_STATUS</code>): <code>empty</code> (no rows), <code>active</code>, <code>beforeStart</code>,
<code>afterEnd</code>, and <code>gap</code> (the playhead lies strictly inside an interval longer than <code>maximumGap</code>; 0
disables the rule). Outside the range the position clamps to the first or last sample and the
heading is that of the first or last segment; in a gap the position holds at the last fix.
Speed is non-zero only for active tracks.</li>
<li>The segment is <code>[j - 1, j]</code> with <code>t[j - 1] &#x3C;= p &#x3C; t[j]</code>, or the last segment when <code>p</code> equals the
last time. A playhead exactly on a sample time returns that sample's position bit-exactly.</li>
<li>Duplicate timestamps: the last row of a run of equal times wins ("latest row wins"), so the
interpolation interval always has a positive duration and nothing divides by zero.</li>
<li>Heading is <code>atan2(dy, dx)</code> (0 along +x, counterclockwise); compass bearing is <code>π/2 - heading</code>.
Zero-length segments give 0. Positions are planar unless <code>spatialContext</code> is great-circle.</li>
<li>Great-circle (<code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code>, optional
<code>sphereRadius</code>, default the mean Earth radius in the context's length unit): positions are <code>(longitude, latitude)</code> degrees and
each segment is interpolated by spherical linear interpolation instead of a linear blend of degrees. The
heading is the segment's initial bearing in the planar convention (radians counterclockwise from east,
<code>pi / 2 - bearing</code>, in <code>(-pi, pi]</code>), and <code>speeds</code> use the context's length unit per timestamp unit
(meters by default; kilometers with <code>units: 'kilometers'</code>). Interpolated longitudes are continuous with the segment's start
row, so a segment that crosses the antimeridian yields longitudes beyond ±180 (the end row is <code>b</code> shifted
by a multiple of 360 degrees) rather than jumping; wrap downstream if needed. Fractions 0 and 1 return
the start and end row exactly (up to that shift), including antipodal endpoints. Interior
interpolation within 1e-6 radians of an antipode returns NaN because that path is not certified. Other contexts throw. Ellipsoidal interpolation is a
non-goal. Both playhead and resample measure distance, so <code>local-lambert-equal-area</code> frames throw.</li>
<li>Time: comparisons are exact in both modes (Int64 words compare integer words, then the f32
playhead fraction); differences are subtracted exactly and then rounded to f32.</li>
</ul>
<p><code>GPUTrajectoryResample</code> resamples every track to <code>sampleCount</code> samples, uniformly in time
(<code>spacing: 'time'</code>, default) or along the path (<code>'arc-length'</code>: length in the context's unit; a
great-circle <code>spatialContext</code> also slerps the samples as the playhead does), into a dense
<code>[trackCount × sampleCount]</code> <code>float32x2</code> buffer (row <code>track * sampleCount + k</code>), with optional
elevations and sample times relative to the track start. It is the fixed-length input for
trajectory similarity, k-means, and fixed-vertex trails.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrajectoryResample</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, timestamps, trackOffsets, sampleCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, spacing: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'arc-length'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {samples, sampleElevations, sampleTimes}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Both take the trajectory layout (<code>positions</code>, <code>timestamps</code>, <code>trackOffsets</code>) as flat props, and all result
views in <code>output</code>. The playhead packer takes a <code>GPUTrajectoryPlayheadSettings</code> and the clock packers a
<code>GPUTrajectoryClockSettings</code>.</p>
<p><code>spacing: 'clock'</code> samples every track at the instants of one shared clock (a per-frame <code>clock</code> view
<code>[start, step]</code>, packed with <code>getGPUTrajectoryClockParameterValues</code> or, for Int64 timestamps,
<code>getGPUTrajectoryClockWordParameterValues</code>; <code>GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH</code> words). Column <code>k</code> is the
same instant for every track and the sample is NaN outside a track's own time span, which is the layout
<code>GPUTrajectoryEncounters</code> expects. With <code>spacing: 'clock'</code> or <code>'time'</code>, a sample is NaN too when it lies
strictly inside a bracketing interval longer than the optional per-frame <code>maximumGap</code> view (one <code>float32</code>
row, the timestamps' unit); see the gap rule below. Omitted or 0 means no limit, the previous behaviour.</p>
<ul>
<li>Sample <code>k</code> targets <code>total * k / (sampleCount - 1)</code>; the last sample is the last row exactly. It
uses the playhead's search and duplicate rule. Single-row tracks repeat that row; empty tracks
write zeros.</li>
<li>Arc length sums step lengths sequentially per track (deterministic, O(track length) per
invocation).</li>
<li>Inputs must be single packed views; chunked vectors are not supported.</li>
</ul>
<h4 id="how-it-computes-10">How it computes</h4>
<p>{/* lens:GPUTrajectoryPlayhead */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=datum datum=1 caption="Track 1 (MMSI 366999411): bracketing segment, position and status at 3600 s"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=formula caption="Interpolated position at the playhead; speed is segment length over duration"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=kernel caption="The compute shader that searches the bracketing segment and interpolates"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-playhead rung=graph caption="Command graph nodes and buffers that produce the playhead columns"}</p>
<p>{/* lens:GPUTrajectoryResample */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=datum datum=0 caption="Sample 0: the first point of track 0 and its time since the track start"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=formula caption="Target progress per sample, then interpolation inside the bracketing segment"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=kernel caption="The compute shader that writes the resampled points"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-resample rung=graph caption="Command graph nodes and buffers that produce the resampled points"}</p>
<h4 id="the-maximumgap-rule">The <code>maximumGap</code> rule</h4>
<p><code>GPUTrajectoryPlayhead</code>, <code>GPUTrajectoryResample</code> with clock or time spacing, <code>addClockEncounters</code>,
<code>GPUTrajectoryClosestApproach</code>, <code>GPUZoneEvents</code>, <code>GPUTrajectoryKalman</code> and <code>GPUTrajectoryRestriction</code>
use one rule. <code>maximumGap</code> is in the unit of the timestamps (word timestamps: the integer unit, and
<code>context.unit</code> when a <code>temporalContext</code> is declared). A step between two consecutive samples is a gap only
when it is strictly longer than the limit; a step exactly equal to the limit is still interpolated.
The samples at both ends of a gap are real fixes and stay present, so only times strictly inside the
gap are affected. Omitted (views) or 0 (everywhere) means no limit. Arc-length resampling takes no
limit: its samples are spaced along the path, and a straight line across a gap is still path length.</p>
<h4 id="trajectory-port-and-sort-check">Trajectory port and sort check</h4>
<p>These contributors share the <code>GPUTrajectoryGeometryPort</code> layout, including Kalman and restriction,
which also keep their <code>times</code> and <code>offsets</code> names. The optional <code>temporalContext</code> must match the rows:
<code>float32</code> is <code>float32-relative</code> (<code>double-single</code> with <code>timestampsLow</code>), <code>uint32x2</code> is <code>int64-words</code>,
and the axis is <code>{kind: 'irregular'}</code>; a mismatch throws at construction.</p>
<p><code>addTrajectorySortCheck(graph, {positions, timestamps, trackOffsets, output: {status: {invalidCount}}})</code>
is an optional check of the sorted-by-track-then-time precondition, which nothing else validates. It
writes one <code>uint32</code> row, the number of rows out of order, and costs nothing unless called.
Each non-finite float timestamp counts once, including a track's first or only row. For
double-single timestamps, either non-finite component makes the row invalid. A finite row after
an invalid row starts a new ordered run and is not counted again for that predecessor. Int64
timestamps retain exact word comparisons. A reversed offset range counts the rows in that range.
<code>status.invalidCount</code> must contain at least one row; an empty view throws before any node is added.</p>
<p>{/* parameters: GPUTrajectoryPlayhead generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectoryplayhead-parameters"><code>GPUTrajectoryPlayhead</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryPlayheadParameterValues(playhead)</code>, writes <code>GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
<li>Packer: <code>getGPUTrajectoryPlayheadWordParameterValues(playhead)</code>, writes <code>GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryPlayheadParameterValues</code>:</p>
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
<td><code>playhead</code></td>
<td><code>number | bigint</code></td>
<td>Playhead instant. Float32 mode: a number in the timestamps' relative epoch. Word mode: a <code>bigint</code> (exact integer) or a <code>number</code> whose fractional part becomes the f32 sub-unit fraction.</td>
</tr>
<tr>
<td><code>maximumGap</code> (optional)</td>
<td><code>number</code></td>
<td>Largest bracketing interval that is still interpolated, in time units. When the playhead lies strictly inside a longer interval the track status is <code>gap</code>. 0 (default) disables the rule.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPUTrajectoryResample generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectoryresample-parameters"><code>GPUTrajectoryResample</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryClockParameterValues(clock)</code>, writes <code>GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>clock</code>.</li>
</ul>
</li>
<li>Packer: <code>getGPUTrajectoryClockWordParameterValues(clock)</code>, writes <code>GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH</code> (4) <code>uint32</code> values.
<ul>
<li>Input port: <code>clock</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUTrajectoryClockParameterValues</code>:</p>
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
<td><code>start</code></td>
<td><code>number | bigint</code></td>
<td>First clock instant. Float32 mode: a number in the timestamps' relative epoch. Word mode: a <code>bigint</code> (exact integer) or a <code>number</code> whose fractional part becomes the f32 sub-unit fraction.</td>
</tr>
<tr>
<td><code>step</code></td>
<td><code>number</code></td>
<td>Distance between consecutive clock samples, finite and positive.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpuzoneevents"><code>GPUZoneEvents</code></h3>
<p>Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually
impose much lower limits.</p>
<p><strong><code>GPUZoneEvents</code></strong> — Finds when each track enters and exits polygon zones, with crossing times and dwell time and visits per track and zone.</p>
<p>::lens{scenario=gpu-spatial-analysis.zone-events rung=result caption="Right: 12 AIS tracks coloured by zone entries and exits, 111 events in all · grey: zones"}</p>
<p><strong><code>addFleetDwellZoneEventsRecipe</code></strong> — Chains zone events and a per-zone roll-up into total dwell time and visiting-track counts per zone.</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=result caption="Right: 13 harbour zones coloured by total vessel dwell in seconds · grey: tracks"}</p>
<p>With the optional per-frame <code>maximumGap</code> view (one <code>float32</code> row, the timestamps' unit), a segment
longer than the limit is a data gap. Crossings on it are reported at the first fix after the gap
(its time, position and row) and the gap adds no dwell, even when the track is inside the zone on
both sides of it. Omitted or 0 keeps the previous straight-line behaviour.</p>
<p>Entry and exit events of trajectories against polygon zones, with interpolated times, bounded events per
track, and dwell time and visit count per <code>(track, zone)</code>.</p>
<p><code>spatialContext</code> is planar only (the default). Segments and zone edges are straight lines in the coordinate
plane, which is not the geometry of degrees (a great-circle segment is not straight in longitude/latitude, and
edges cross the antimeridian), so any longitude-latitude context throws; project positions and zones first
with <code>createGPULocalFrame</code>.</p>
<ul>
<li>Inputs: <code>tracks</code>, a tagged trajectory geometry (<code>kind: 'trajectories'</code>, <code>positions</code> <code>float32x2</code>,
<code>timestamps</code> <code>float32</code> relative or <code>uint32x2</code> Int64 words, <code>trackOffsets</code> with <code>trackCount + 1</code> rows), and
<code>zones</code>, a tagged polygon geometry (<code>kind: 'polygons'</code>) in the four-level layout (<code>positions</code>,
<code>featureOffsets</code>, <code>polygonOffsets</code>, <code>ringOffsets</code>). Zone <code>z</code> is feature <code>z</code>, so the zone count is
<code>zones.featureOffsets.length - 1</code>; rings of one feature (holes, extra parts) share the zone index and the
first ring of a polygon is its shell. Rings close implicitly, a repeated closing vertex is tolerated, and
rings with fewer than three vertices or non-finite vertices are ignored. A first kernel derives one edge per
zone vertex from the layout. Plus <code>candidateCapacity</code> and <code>trackEventCapacity</code> (events kept per track).</li>
<li>Outputs, all in <code>output</code>: <code>events</code> (a <code>GPUCompactOutput</code> extended with the columns: <code>ids</code> is the track index),
optional <code>eventZones</code>, <code>eventTypes</code> (<code>GPU_ZONE_EVENT_TYPE.enter</code> 0, <code>.exit</code> 1), <code>eventTimes</code> (relative to
the track's first timestamp, f32) and <code>eventRows</code> (the segment end row); optional dense
<code>[trackCount * zoneCount]</code> <code>dwellTimes</code> and <code>visitCounts</code>; and <code>trackEventCounts</code> (unclamped).</li>
<li>Events are sorted by <code>(track, time, edge row)</code>. Each segment queries a <code>GPUBVH</code> of edge bounds and
runs an exact crossing test (<code>orientSign</code>, never f32 parameters); the time is
<code>t[r - 1] + s * (t[r] - t[r - 1])</code>. Enter versus exit comes from parity, not ring orientation: the first
sample of every track probes with a +x ray (even-odd), so a track that starts inside a zone works, and
that first interval counts as a visit with dwell from <code>t = 0</code>. Open visits close at the track's last
sample. Dwell and visits use every event, even when the event list is truncated. No kernel walks a
track's events one by one: a stable sort by <code>(track, zone)</code> makes the events of each cell a
time-ordered run, so event types, ranks, visit counts and spans are O(1) per event or cell and dwell
is a load-balanced segmented sum. A track with thousands of crossings costs no more than many short
tracks with the same event total.</li>
<li>Boundary rule (the library's shared Simulation-of-Simplicity rule, <code>utils/exact-geometry-wgsl.ts</code>): every track sample <code>(x, y)</code> is read as <code>(x - e^2, y - e)</code> for an infinitesimal <code>e > 0</code>
(the track moves, the polygon does not), for the initial ray and the segment crossings alike, so the
parity of a track's crossings always equals the change of its inside flag. A track that only touches a
polygon apex or an edge from outside has no event; one that runs through a vertex into the interior has
one; a sample on an edge is just below it (inside when the interior lies below) and on a vertical edge
just west of it. Event times along a collinear overlap use the f32 projection.</li>
<li><code>output.events.status.overflow</code> is 1 when more events were kept than <code>ids.length</code>. Optional
<code>output.events.status.candidateOverflow</code> reports candidate scratch or per-track event truncation.
<code>requiredCount</code> counts retained events before final truncation, so it is a lower bound when
candidate scratch or <code>trackEventCapacity</code> dropped events.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUZoneEvents</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  tracks: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'trajectories'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, timestamps, trackOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zones: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'polygons'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions: zonePositions, featureOffsets, polygonOffsets, ringOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  candidateCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">65536</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, trackEventCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    events: {ids, status: {count, overflow, candidateOverflow: eventCandidateOverflow}, eventZones, eventTypes, eventTimes},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    dwellTimes, visitCounts,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {candidateOverflow: candidateScratchOverflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>output.events.eventPositions</code> (<code>float32x2</code>) holds the interpolated crossing position of each event.
<code>output.visitTable</code> is a sparse <code>(track, zone)</code> table that replaces the dense matrices for large
<code>trackCount * zoneCount</code>: rows are ordered by track, then zone, as a <code>GPUCompactOutput</code> (<code>ids</code> is the track,
plus <code>status.count</code>, <code>status.overflow</code> and <code>status.requiredCount</code>) extended with the columns <code>zones</code>, <code>visits</code>, <code>dwellTimes</code>,
<code>firstEnterTimes</code> and <code>lastExitTimes</code> (see <code>GPUZoneVisitTableOutput</code>). <code>addFleetDwellZoneEventsRecipe</code> forwards both outputs. Times are
relative to the track start, a track that starts inside has first enter 0, and an open visit has last exit
equal to the track duration. The table is built from the dense per-cell state, so it bounds the output, not
the working memory.</p>
<p>Optional <code>visitTable.status.candidateOverflow</code> reports candidate-scratch loss affecting the dense
visits, dwell and table. It does not include <code>trackEventCapacity</code> truncation: that cap only bounds
the displayed event list and leaves visits/dwell intact. Encounter and closest-approach compact
pair ports likewise propagate their hit/grid/entry loss through <code>pairs.status.candidateOverflow</code>.
The fleet-dwell recipe preserves caller event status views; its <code>zone-events</code> stage uses the
compact event candidate flag (scratch or track loss), while its separate scratch diagnostic
continues to identify scratch loss alone.</p>
<p>Limits: the scratch of <code>candidateCapacity</code> rows is sorted (three stable <code>GPUSort</code> passes) every encoding;
the walk and dwell kernels use one invocation per track; the dense matrices cost <code>trackCount * zoneCount</code>
rows; samples exactly on an edge follow the half-open rule and can disagree with other predicates by one
event; zones with overlapping rings are not supported (even-odd); at equal timestamps the crossing position is not unique.</p>
<p>Status: the optional <code>output.status</code> takes four one-row <code>uint32</code> views, <code>candidateCount</code>,
<code>candidateOverflow</code>, <code>trackOverflow</code> and <code>eventOverflow</code>.</p>
<ul>
<li><code>candidateCount</code> is the unclamped number of segment-edge bounding-box candidates, including one ray query
per track. Set <code>candidateCapacity</code> to at least this for exact results.</li>
<li><code>candidateOverflow</code> means the scratch was too small, so events may be missing.</li>
<li><code>trackOverflow</code> means a track has more than <code>trackEventCapacity</code> events. Its list is truncated, but dwell and
visits stay exact.</li>
<li><code>eventOverflow</code> means there are more events than <code>output.events.ids.length</code>. <code>output.events.status.requiredCount</code>
holds the size needed.</li>
</ul>
<p><code>output.events.status.overflow</code> reports <code>eventOverflow</code> only; optional
<code>output.events.status.candidateOverflow</code> reports the OR of candidate and track overflow.</p>
<p>An event-list readout is complete only when both event status flags are zero. Dwell totals,
visits and unclamped per-track counts instead require the scratch-only
<code>output.status.candidateOverflow</code> to be zero: an event-list cap alone does not invalidate them.
The standalone and fleet-dwell lenses capture these as <code>eventOverflow</code>,
<code>eventCandidateOverflow</code> and <code>candidateScratchOverflow</code>, with the two loss domains kept separate.
The website qualifies retained crossings as incomplete on either event-list flag, and qualifies
dwell only on scratch loss; pending status readbacks never certify completeness. The standalone
lens example accepts <code>candidateCapacity</code> (default 131072 scratch rows) independently of
<code>trackEventCapacity</code> (the maximum retained events per track).</p>
<h4 id="how-it-computes-11">How it computes</h4>
<p>{/* lens:GPUZoneEvents */}</p>
<p>::lens{scenario=gpu-spatial-analysis.zone-events rung=datum datum=0 caption="Track 0 (MMSI 366993970): its zone events, dwell and visit counts"}</p>
<p>::lens{scenario=gpu-spatial-analysis.zone-events rung=kernel caption="The compute shaders that find crossings and sum dwell per track and zone"}</p>
<p>::lens{scenario=gpu-spatial-analysis.zone-events rung=graph caption="Command graph nodes and buffers that produce the events and dwell"}</p>
<p>{/* lens:addFleetDwellZoneEventsRecipe */}</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=datum datum=0 caption="Anchorage 0 (Upper Bay Anchorage Number 20B): visiting tracks and total dwell"}</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=kernel caption="The compute shaders that key dwell by zone for the roll-up"}</p>
<p>::lens{scenario=gpu-spatial-analysis.fleet-dwell-zone-events-recipe rung=graph caption="Command graph of the recipe: zone events, zone keys and zone statistics"}</p>
<h3 id="gputrajectoryencounters"><code>GPUTrajectoryEncounters</code></h3>
<p>Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually
impose much lower limits.</p>
<p>Finds pairs of tracks within a distance of each other in the same time bucket of resampled tracks.</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=result caption="GPU pair table: 8 track pairs within 200 m at the same clock, smallest distance in metres"}</p>
<p>Pairs of tracks within <code>distance</code> in the same time bucket (MobilityDB <code>tdwithin</code> on resampled tracks).</p>
<p>Construction warns when the host-known <code>maximumDistance</code> derived from <code>cellSize</code> is too short for float32 lon/lat; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a> for this check and the unchecked per-frame <code>distance</code> GPU view.</p>
<p>Maximum: <code>columns * rows * bucketCount</code> of the lattice is at most <code>getGPUGridIndexMaximumCellCount(device)</code> (2^25 - 1 on a default device); raise <code>cellSize</code> or lower <code>bucketCount</code>.</p>
<ul>
<li>Inputs: dense <code>samples</code> (<code>[trackCount * bucketCount]</code> <code>float32x2</code>, row <code>track * bucketCount + bucket</code>,
the <code>GPUTrajectoryResample.samples</code> layout), <code>trackCount</code>, <code>bucketCount</code>, optional <code>mask</code>,
per-frame <code>distance</code> (one float row, clamped to <code>cellSize</code>), compile-time <code>cellSize</code> and
<code>bounds [minX, minY, maxX, maxY]</code>, <code>hitCapacity</code> and optional <code>bucketTimes</code>.</li>
<li>Outputs, all in <code>output</code>: <code>output.pairs</code> (a <code>GPUCompactOutput</code>, <code>ids</code> is the lower track, extended with the
optional columns <code>partners</code>, <code>firstBuckets</code>, <code>minimumDistances</code>, <code>bucketCounts</code> and <code>firstTimes</code> (needs
<code>bucketTimes</code>)) and the required <code>output.status.candidateOverflow</code>. Pairs have <code>track &#x3C; partner</code>,
are deduplicated across buckets and sorted by <code>(track, partner)</code>; the sentinel is <code>0xffffffff</code> (IDs,
buckets) and 0 after <code>status.count</code>. <code>output.pairs.status.overflow</code> reports only pair-capacity truncation;
<code>output.status.candidateOverflow</code> reports hit-scratch or grid incompleteness.</li>
<li>It uses a 3D <code>GPUGridIndex</code> (x, y and the bucket as an exact third axis) with a 3x3 scan per sample,
three stable sorts of the hits and <code>GPUCompaction</code> for the runs.</li>
<li>Great-circle: with <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code>
(optional <code>sphereRadius</code>) the samples are <code>(longitude, latitude)</code> degrees, <code>distance</code> and the reported
<code>minimumDistances</code> are in the context's length unit (meters by default; kilometers with
<code>units: 'kilometers'</code>), and <code>cellSize</code> and <code>bounds</code> stay in degrees. The default sphere radius is the
mean Earth radius in that unit. It measures distance, so <code>local-lambert-equal-area</code> frames throw. The test is the
great-circle distance of the two samples of one bucket, so there is no tangent-plane approximation. Pruning
is conservative: two points within angle <code>theta</code> differ by at most <code>theta</code> in latitude and by at most
<code>asin(sin(theta) / cos(L))</code> in longitude, where <code>L</code> is the largest <code>|latitude|</code> of <code>bounds</code>. The lattice
therefore searches up to <code>maximumDistance = radius * asin(sin(cellSize) * cos(L))</code> (times <code>1 - 1e-6</code>),
exposed as <code>encounters.maximumDistance</code>, and <code>distance</code> is clamped to it. Verified against pyproj: walking
every azimuth at that distance from latitude ±L stays within one cell. Pick latitude bounds that exclude the
polar caps (the reach shrinks to 0 as <code>L</code> approaches 90). Bounds with longitudes exactly <code>[-180, 180]</code> wrap
the lattice across the antimeridian (sample longitudes are folded into that range first and <code>cellSize</code> must
be below 120 degrees); other longitude bounds do not wrap, and <code>cellSize</code> must be below 90. Latitudes must
lie in <code>(-90, 90)</code>. Planar contexts keep <code>maximumDistance = cellSize</code>. <code>addClockEncounters</code> forwards
<code>spatialContext</code> to the resample and the encounters.</li>
<li>It is discrete in time: the samples must share a clock (column <code>k</code> is one instant for all tracks), and
encounters between buckets are missed. With <code>GPUTrajectoryResample</code> that holds only when the tracks cover
the same window.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrajectoryEncounters</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  samples, trackCount, bucketCount, distance, cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, bounds: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  hitCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 16</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    pairs: {ids, status: {count, overflow}, partners, firstBuckets, minimumDistances},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {candidateOverflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>addClockEncounters(graph, props)</code> builds the shared-clock chain in one call: it resamples all tracks
onto one clock (<code>GPUTrajectoryResample</code> with <code>spacing: 'clock'</code>, a per-frame <code>[start, step]</code> buffer) and
feeds <code>GPUTrajectoryEncounters</code>, so tracks need not cover the same window. It takes the trajectory layout
(<code>positions</code>, <code>timestamps</code>, <code>trackOffsets</code>) and the encounter options as flat props. Optional
<code>output.bucketTimes</code> and <code>output.samples</code> expose the clock and the resampled tracks
(<code>AddClockEncountersProps</code>, <code>ClockEncounters</code>). The optional per-frame <code>maximumGap</code> view is forwarded
to the clock resample, so a bucket inside an over-long track interval reads that track as absent and
produces no encounter.</p>
<p>Clock mode compares positions only at the shared bucket instants, so two tracks that pass each other
between buckets are missed. Treat it as a fast preview and use <code>GPUTrajectoryClosestApproach</code> for the
exact answer.</p>
<h4 id="how-it-computes-12">How it computes</h4>
<p>{/* lens:GPUTrajectoryEncounters */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=datum datum=0 caption="Pair 0: lower track, partner, buckets within 200 m and smallest distance"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=formula caption="Bucket count and minimum distance over the clock buckets within range"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=kernel caption="The compute shaders that scan the grid and reduce each pair"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-encounters rung=graph caption="Command graph nodes and buffers that produce the pairs"}</p>
<h3 id="gputrajectoryclosestapproach"><code>GPUTrajectoryClosestApproach</code></h3>
<p>Candidate capacities reserve a strictly higher saturated-count sentinel; device bindings usually
impose much lower limits.</p>
<p>Finds pairs of tracks within a distance of each other at the same instant, with closest-approach distance, time and duration.</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=result caption="GPU pair table: 10 track pairs passing within 200 m, closest approach and time inside"}</p>
<p>Exact encounters between trajectories on their raw samples, the MobilityDB <code>tdwithin</code> and
<code>nearestApproachDistance</code> question: tracks move linearly between samples, and two tracks are an encounter
while their positions at the same instant are at most <code>distance</code> apart. Each sample row becomes a segment to
the next sample of its track with a space-time box (<code>x</code> and <code>y</code> padded by half the distance), boxes are
binned into a hashed grid of <code>cellSize</code> by <code>cellSize</code> by <code>timeCellSize</code> cells and sorted, and every pair of
segments of different tracks that shares a cell is tested once (in the cell holding the minimum corner of
the two boxes' intersection). The test solves the closest point of approach over the common time interval:
the separation is <code>D + V tau</code>, its minimum is at <code>tau = -(D . V) / |V|^2</code>, and the time within the distance
lies between the roots of <code>|D + V tau|^2 = distance^2</code>. With the optional per-frame <code>maximumGap</code> view
(one <code>float32</code> row, the timestamps' unit) a segment longer than the limit is not motion: it adds no hits
and no cell entries, so a long gap can no longer raise <code>entryOverflow</code>.</p>
<ul>
<li>Inputs: the trajectory layout (<code>positions</code>, <code>timestamps</code> as <code>float32</code> values or <code>uint32x2</code> Int64 words in the declared timestamp unit,
<code>trackOffsets</code>), optional <code>mask</code>, a per-frame one-row <code>distance</code>, and the compile-time <code>cellSize</code>,
<code>timeCellSize</code>, <code>entryCapacity</code> (cell entries) and <code>hitCapacity</code> (segment-pair hits). Planar only; project
longitude/latitude first with <code>createGPULocalFrame</code>. <code>timeCellSize</code> uses the declared timestamp unit,
so rescale it with timestamps when changing units.</li>
<li>Outputs, in <code>output</code>: <code>pairs</code>, a <code>GPUCompactOutput</code> (<code>ids</code> is the lower track) in <code>(track, partner)</code>
order with optional <code>partners</code>, <code>minimumDistances</code>, <code>closestTimes</code>, <code>firstTimes</code> and <code>durations</code>, all
times in the declared timestamp unit relative to the first sample of the lower track, with durations in that same unit; and <code>status.entryOverflow</code> and
<code>status.hitOverflow</code> for the two scratch capacities (a segment covering more than 4096 cells also sets
<code>entryOverflow</code>).</li>
<li>Word timestamps stay exact: times are measured from a segment's own start row before the f32
conversion. Verified against <code>test/oracles/trajectory-closest-approach.py</code>, a float64 brute force over
every segment pair, self-checked by dense sampling and against shapely path distances.</li>
</ul>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrajectoryClosestApproach</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  positions, timestamps, trackOffsets, distance,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  cellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, timeCellSize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">60</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, entryCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, hitCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> &#x3C;&#x3C;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> 18</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    pairs: {ids, status: {count, overflow}, partners, minimumDistances, closestTimes, firstTimes, durations},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status: {entryOverflow, hitOverflow}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-13">How it computes</h4>
<p>{/* lens:GPUTrajectoryClosestApproach */}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=datum datum=0 caption="Pair 0: tracks, closest-approach distance, its time and the time within 200 m"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=formula caption="Closest time on a common segment interval and the distance at that time"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=kernel caption="The compute shaders that test segment pairs and reduce each track pair"}</p>
<p>::lens{scenario=gpu-spatial-analysis.trajectory-closest-approach rung=graph caption="Command graph nodes and buffers that produce the pairs"}</p>
<h3 id="gputracksimilarity"><code>GPUTrackSimilarity</code></h3>
<p>invalidPair also reports invalid track ranges and densification overflow; an invalid pair must not
be interpreted as a valid similarity.</p>
<p>Measures Hausdorff and discrete Frechet distance between pairs of tracks or rings, with optional maximum distance and dynamic time warping cost.</p>
<p>::lens{scenario=gpu-spatial-analysis.track-similarity rung=result caption="All 66 track pairs: Hausdorff and Fréchet distance in metres"}</p>
<p>Hausdorff (symmetric, vertex-to-segment as GEOS computes it, unbounded length) and discrete Frechet
(vertex-based) distances for given pairs of tracks or rings (<code>hausdorff_distance(a, b, densify)</code>,
<code>frechet_distance(a, b, densify)</code>). Behavior change: Hausdorff used to be vertex-to-vertex, which agreed
with Shapely only by coincidence; it now measures from the (densified) vertices to the other polyline's
segments, so values can be slightly smaller than before. Inputs are <code>linesA</code> (a tagged line geometry:
<code>kind: 'lines'</code>, <code>positions</code> and <code>lineOffsets</code>; optionally <code>linesB</code>, by
default the same set), <code>pairA</code>, <code>pairB</code> (<code>uint32</code> views) and an optional GPU-written <code>activePairCount</code>.
Outputs, all in <code>output</code>, are <code>hausdorff</code>, <code>frechet</code>, optional <code>maximumDistance</code> (largest vertex-to-vertex distance per pair, Sedona
<code>ST_MaxDistance</code>, original vertices) and <code>pairStatus</code> (bits of <code>GPU_TRACK_SIMILARITY_STATUS</code>: <code>emptyTrack</code> 1,
<code>frechetCapExceeded</code> 2, <code>invalidPair</code> 4). Skipped pairs get NaN. <code>densify</code> (0 or in <code>[1/4096, 1]</code>, compile-time) splits each segment into <code>round(1/densify)</code> parts, as
Shapely does; the virtual vertices are computed on the fly. Frechet sweeps the longer track in strips of
256 lanes, so only the shorter (densified) track is capped: <code>maximumFrechetVertexCount</code> is 1 to 2048 (default
256). A pair over the cap reports <code>frechetCapExceeded</code> with a NaN Frechet while its Hausdorff is still
computed. Polygon boundaries are passed as closed vertex lists. Against Shapely 2.1.2 (GEOS 3.13.1) all
three measures were within 2e-4 relative on line and polygon-boundary pairs, including a 700 by 300 Frechet
over three strips. Rings are open vertex
sequences, so Frechet is sensitive to the start and the orientation. Pairs can come straight from
<code>GPUTrajectoryEncounters</code> (<code>output.pairs.ids</code>, <code>output.pairs.partners</code> and <code>output.pairs.status.count</code>).</p>
<p>For <code>frechetEpsilon</code>, no warning is possible because the distance is a GPU view; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#precision-of-longitudelatitude-positions">Precision of longitude/latitude positions</a>.</p>
<p>Maximum: the pair capacity <code>pairA.length</code> is at most <code>GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT</code> (2^24 - 1).</p>
<p><code>spatialContext</code> defaults to native planar (Euclidean distances in the f32 input coordinates; the planar kernels
compile no geodesic code). <code>{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}</code> makes Hausdorff,
Frechet and <code>maximumDistance</code> great-circle distances in the context's length unit (meters by default;
kilometers with <code>units: 'kilometers'</code>) on a sphere of <code>sphereRadius</code> (default the mean Earth radius in
that unit). Other geographic metrics throw. It measures distance, so <code>local-lambert-equal-area</code> frames throw. Vertices are <code>[longitude, latitude]</code> degrees
and segments are straight in longitude/latitude, as everywhere in this package. The vertex-to-segment distance is
the haversine distance to the closest point of the segment, with that point found in the local equirectangular
frame of the vertex (<code>geodesicSegmentFraction</code>, <code>geodesicSegmentDistance</code>): exact to first order for segments of
kilometers, so <code>densify</code> long segments. Densified points interpolate longitude and latitude linearly with the
longitude difference wrapped to +-180, so tracks may cross the antimeridian. Against numpy haversine with a
20000-part segment densification (cross-checked with <code>pyproj.Geod(a=6371008.8, f=0)</code>) the f32 results are
expected within 2e-3 relative; the spec covers a mid-latitude, a 66 degree and an antimeridian pair.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrackSimilarity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  linesA: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, lineOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pairA: pairs.ids, pairB: pairs.partners, activePairCount: pairs.status.count,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximumFrechetVertexCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">256</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, densify: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {hausdorff, frechet, maximumDistance, pairStatus}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p>Two more measures share the Frechet layout and its shorter-track cap. <code>output.frechetWithin</code> (<code>uint32</code>,
with the per-frame one-row <code>frechetEpsilon</code> in the context's length unit) is the Frechet decision, 1 when the discrete Frechet distance
is at most epsilon: endpoint distances reject pairs before scanning their vertices; a fixed monotone coupling
accepts pairs in <code>O(n + m)</code>. Undecided planar pairs also use coordinate-extrema bounding-box lower bounds.
The rest run the boolean free-space program, evaluating distances only for cells with reachable predecessors; it stops once no
monotone path can survive a row or two consecutive anti-diagonals. Use it to filter candidate pairs at a
threshold without requesting the full distance or DTW, which still compute when requested alongside the decision.
<code>track-similarity-dtw-frechet-ab.spec.ts</code> times decision-only and distance-only graphs separately.
<code>output.dynamicTimeWarping</code> is the dynamic time warping cost, the
smallest sum of point distances in the context's length unit over monotone warping paths (<code>similaritymeasures.dtw</code>; not the square root
of summed squares of tslearn and dtaidistance), with an optional compile-time Sakoe-Chiba band
<code>dynamicTimeWarpingBandRadius</code> equal to <code>tslearn.metrics.sakoe_chiba_mask(n, m, radius)</code>; cells outside
the band are never evaluated. Both are verified against <code>test/oracles/track-similarity-dtw-frechet.py</code>
(similaritymeasures, shapely, tslearn), planar and great-circle.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrackSimilarity</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  linesA: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'lines'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, positions, lineOffsets},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pairA: pairs.ids, pairB: pairs.partners, activePairCount: pairs.status.count,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  frechetEpsilon, dynamicTimeWarpingBandRadius: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {frechetWithin, dynamicTimeWarping}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h4 id="how-it-computes-14">How it computes</h4>
<p>{/* lens:GPUTrackSimilarity */}</p>
<p>::lens{scenario=gpu-spatial-analysis.track-similarity rung=datum datum=0 caption="Pair 0 (tracks 0 and 1): Hausdorff, Frechet and maximum vertex distance"}</p>
<p>::lens{scenario=gpu-spatial-analysis.track-similarity rung=kernel caption="The compute shaders that measure each pair"}</p>
<p>::lens{scenario=gpu-spatial-analysis.track-similarity rung=graph caption="Command graph nodes and buffers that produce the pair distances"}</p>`};export{e as default};