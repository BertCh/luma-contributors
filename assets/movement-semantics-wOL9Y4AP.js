var e={slug:`api/gpu-spatial-analysis/movement-semantics`,title:`Movement semantics`,html:`<h1 id="movement-semantics">Movement semantics</h1>
<p>Staypoints, locations, triplegs and trips, mobility indicators on staypoints, and trajectory
cleaning. Everything here reads the shared <a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts"><code>GPUTrajectoryGeometryPort</code></a> layout:
<code>positions</code> (<code>float32x2</code>), <code>trackOffsets</code> and <code>timestamps</code> (<code>float32</code> relative or <code>uint32x2</code> Int64
words, with an optional <a href="/luma-contributors/docs/api/gpu-spatial-analysis/temporal-context"><code>temporalContext</code></a>). Durations and the shared gap
rule, <code>maximumGap</code>, are in the timestamps' unit: a step is a gap only if it is strictly longer than
the limit, and 0 or absent means no limit.</p>
<p>The families follow, in order, trackintel (MIT), scikit-mobility 1.3.1 (BSD) and MovingPandas 0.23
(BSD-3); fixtures from those packages check them. Parameters are per-frame packed float32
buffers unless a section says otherwise.</p>
<p>Out of scope (as the 2026-11 spatiotemporal review concluded): exact
flocks and convoys, TRACLUS, T-LoCoH, and continuous-time movement model fitting (CTMM, OU/OUF).</p>
<h2 id="staypoints-locations-and-triplegs">Staypoints, locations and triplegs</h2>
<p>The contributors compose, each reading the columns the previous one writes:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">write</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">  getGPUTrajectoryStaypointsParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    distanceThreshold: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">100</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timeThreshold: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">300_000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    maximumGap: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">900_000</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">  new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTrajectoryStaypoints</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    spatialContext: {coordinateSpace: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'longitude-latitude'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, metric: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'great-circle'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    positions,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    timestamps,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    trackOffsets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    parameters: parameterBuffer.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">importToGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output: {ids, status: {count, overflow}, startRows, finishRows, centroids, durations, staypointIds}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  })</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span></code></pre>
<h3 id="gputrajectorystaypoints"><code>GPUTrajectoryStaypoints</code></h3>
<p>Detects staypoints with trackintel's <code>sliding</code> rule (Li et al. 2008) in a bounded number of passes:
every row searches forward independently for its first break (a gap, or a fix at or beyond
<code>distanceThreshold</code>), at most <code>maximumSearchRows</code> rows ahead; pointer doubling in
<code>ceil(log2(rowCount))</code> fixed passes then keeps the rows reachable from each track's first row. That
reproduces the sequential chain exactly as long as no search hits the cap;
<code>status.candidateOverflow</code> reports when one did.</p>
<p><code>output.status.overflow</code> reports final staypoint-table capacity only. A capped search can set
<code>candidateOverflow</code> while all retained staypoints fit and final overflow stays zero. Both words
are rewritten each submission, so a frame whose searches fit the bound clears candidate loss.</p>
<p>Compact candidate-status words on trajectory metrics stops and trajectory segments are outputs,
too. They must be separate from input <code>trackOffsets</code> and the other output words; publication
must never alter the trajectory used by the next submission.</p>
<p>Unlike a per-step speed test, one parked stop stays one stop under GPS jitter: the window restarts
only when a fix leaves <code>distanceThreshold</code> of its anchor. The cost is <code>N * min(maximumSearchRows, L)</code> distance tests for stays of <code>L</code> rows, so 1 Hz data parked for a long time needs a cap of at least
that many rows, or thinning first.</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>parameters</code></td>
<td>Packed <code>getGPUTrajectoryStaypointsParameterValues</code>: distance, time, gap, include-last</td>
</tr>
<tr>
<td><code>maximumSearchRows</code></td>
<td>Forward search cap per row, compile-time, default 4096</td>
</tr>
<tr>
<td><code>output.ids</code></td>
<td>Track of each staypoint, ordered by track then time</td>
</tr>
<tr>
<td><code>output.startRows</code>, <code>output.endRows</code>, <code>output.finishRows</code></td>
<td>First row, exclusive end row and the row whose time is <code>finished_at</code></td>
</tr>
<tr>
<td><code>output.centroids</code></td>
<td>Mean of the staypoint's distinct coordinates (trackintel <code>union_all</code>)</td>
</tr>
<tr>
<td><code>output.durations</code></td>
<td><code>finished_at - started_at</code></td>
</tr>
<tr>
<td><code>output.staypointIds</code></td>
<td>Per-row <code>staypoint_id</code>, <code>0xffffffff</code> outside staypoints</td>
</tr>
</tbody>
</table>
<p>{/* lens:GPUTrajectoryStaypoints */}</p>
<p>{/* parameters: GPUTrajectoryStaypoints generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectorystaypoints-parameters"><code>GPUTrajectoryStaypoints</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryStaypointsParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryStaypointsParameterValues</code>:</p>
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
<td>Spatial context of the positions, for the short-distance warning only.</td>
</tr>
<tr>
<td><code>distanceThreshold</code></td>
<td><code>number</code></td>
<td>Distance at or beyond which a fix leaves the current staypoint (trackintel <code>dist_threshold</code>), in position units, or the context length unit for great-circle. Finite and positive.</td>
</tr>
<tr>
<td><code>timeThreshold</code></td>
<td><code>number</code></td>
<td>Smallest staypoint duration (<code>finished - started</code>, trackintel <code>time_threshold</code>) in the timestamps' unit. Finite and non-negative.</td>
</tr>
<tr>
<td><code>maximumGap</code> (optional)</td>
<td><code>number</code></td>
<td>Shared gap rule (<code>GPUTrajectoryGeometryPort</code>): a step is a gap only when strictly longer than this limit, in the timestamps' unit; 0 or absent means no limit. A gap restarts the search at the fix after it without emitting a staypoint (trackintel <code>gap_threshold</code>).</td>
</tr>
<tr>
<td><code>includeLast</code> (optional)</td>
<td><code>boolean</code></td>
<td>Emit a staypoint for the window still open at the end of a track (trackintel <code>include_last</code>).</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectorylocations"><code>GPUTrajectoryLocations</code></h3>
<p>Staypoint locations per track with trackintel's <code>generate_locations</code> (<code>dbscan</code>, <code>num_samples=1</code>,
<code>agg_level='user'</code>): the connected components of staypoints of one track within <code>epsilon</code>
(inclusive), labelled by bounded min-label hooking with path compression (<code>maximumIterations</code>
rounds, <code>status.converged</code>). Locations are numbered by first visit within a track, tracks in order.
Dataset-level locations and <code>num_samples > 1</code> are <code>GPUSpatialClustering</code> jobs; the convex-hull
<code>extent</code> is not produced.</p>
<p>{/* lens:GPUTrajectoryLocations */}</p>
<p>{/* parameters: GPUTrajectoryLocations generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectorylocations-parameters"><code>GPUTrajectoryLocations</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryLocationsParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryLocationsParameterValues</code>:</p>
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
<td>Spatial context of the positions, for the short-distance warning only.</td>
</tr>
<tr>
<td><code>epsilon</code></td>
<td><code>number</code></td>
<td>Clustering radius (trackintel <code>epsilon</code>, inclusive), in position or context length units.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectorysegments"><code>GPUTrajectorySegments</code></h3>
<p>Triplegs (<code>between_staypoints</code>) and trips (<code>generate_trips</code>) from the per-row staypoint IDs and the
staypoint table. A tripleg is a maximal run of rows outside every staypoint, broken at a gap, with
at least two rows that are not all the same coordinate. A trip aggregates triplegs and non-activity
staypoints between activities (staypoints strictly longer than <code>activityMinimumDuration</code>), breaks at
a gap between events, and starts at the last activity of an activity run. Origin and destination
staypoints are the activities directly around a trip, unknown across a gap or at a track end, as
<code>origin_staypoint_id</code> and <code>destination_staypoint_id</code> are in trackintel. <code>triplegTripIds</code> is the
<code>trip_id</code> of each tripleg. The <code>overlap_staypoints</code> method is not implemented.</p>
<p>Supply available staypoint <code>status.overflow</code> and <code>status.candidateOverflow</code> beside <code>status.count</code>.
Their loss is propagated into <code>triplegs.status.candidateOverflow</code> and <code>trips.status.candidateOverflow</code>.
Trips consume only the stored tripleg prefix, so tripleg capacity loss also sets trip candidate
overflow. Each table's <code>status.overflow</code> still means only that table's final output capacity was
exceeded. A trip count/required count derived from an incomplete input describes the partial result;
the candidate flag must be checked before treating it as a complete trip table.
Trip construction requires a caller-visible <code>trips.status.candidateOverflow</code> destination. Supplying
either staypoint loss input also requires <code>triplegs.status.candidateOverflow</code>. Both operation forms
and the direct contributor reject omitted destinations at construction; private scratch never
substitutes for a public loss diagnostic.</p>
<p>Pass both tripleg <code>overflow</code> and <code>candidateOverflow</code> to modal split and tracking quality. If either
supplied flag is nonzero, modal split suppresses every aggregate to NaN and every predicted mode to
<code>0xffffffff</code>; tracking quality suppresses every track's value to NaN. The flags describe whole-table
loss, so these consumers cannot identify unaffected tracks. Omitting both flags asserts that the
input table is complete. Complete empty tracks still produce zero. Website views hide incomplete
modal bars/classes and label tracking quality unavailable; Kepler exports unavailable numeric values
as null. Never interpret NaN or the missing-mode sentinel as a count or a speed class.</p>
<p>The thinning and staypoint CPU twins honor an explicit <code>maximumSearchRows</code>, including the GPU's
anchor jumps/window splits and candidate-overflow flag. Omitted CPU search budgets remain uncapped.
Thinning count and candidate-overflow words have separate writers, so adjacent non-overlapping
scalar subviews of one status allocation are supported.</p>
<p>{/* lens:GPUTrajectorySegments */}</p>
<p>{/* parameters: GPUTrajectorySegments generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectorysegments-parameters"><code>GPUTrajectorySegments</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectorySegmentsParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectorySegmentsParameterValues</code>:</p>
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
<td><code>maximumGap</code> (optional)</td>
<td><code>number</code></td>
<td>Shared gap rule (<code>GPUTrajectoryGeometryPort</code>): a step strictly longer than this limit breaks a tripleg and a trip (trackintel <code>gap_threshold</code>); 0 or absent means no limit.</td>
</tr>
<tr>
<td><code>activityMinimumDuration</code></td>
<td><code>number</code></td>
<td>A staypoint is an activity when its duration is strictly longer than this, in the timestamps' unit (trackintel <code>create_activity_flag</code>). Finite and non-negative.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="mobility-indicators">Mobility indicators</h2>
<h3 id="gpumobilityindicators"><code>GPUMobilityIndicators</code></h3>
<p>Per-track indicators on the staypoint table (and, for tracking quality, the tripleg table),
following scikit-mobility 1.3.1 <code>individual</code> measures and trackintel:</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Definition</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>radiusOfGyration</code></td>
<td>RMS haversine distance of the staypoints from their arithmetic mean position</td>
</tr>
<tr>
<td><code>jumpLengths</code></td>
<td>Distance between consecutive staypoints (0 for a track's first)</td>
</tr>
<tr>
<td><code>waitingTimes</code></td>
<td>Difference between consecutive staypoint start times</td>
</tr>
<tr>
<td><code>randomEntropy</code></td>
<td><code>log2(distinct locations)</code></td>
</tr>
<tr>
<td><code>uncorrelatedEntropy</code></td>
<td>Shannon entropy in bits of location shares</td>
</tr>
<tr>
<td><code>realEntropy</code></td>
<td>Lempel-Ziv estimator of the location sequence, <code>n * log2(n) / sum(lambda)</code> as in scikit-mobility</td>
</tr>
<tr>
<td><code>homeLocations</code>, <code>workLocations</code></td>
<td>Locations with the longest and second longest summed duration (trackintel FREQ, no pre-filter)</td>
</tr>
<tr>
<td><code>trackingQuality</code></td>
<td>Sum of positive durations over staypoints and triplegs, divided by the extent from first start to last finish (trackintel <code>temporal_tracking_quality</code>, granularity <code>all</code>)</td>
</tr>
</tbody>
</table>
<p>Location statistics are <code>O(m^2)</code> per track and real entropy <code>O(m^3)</code> in the worst case for <code>m</code>
staypoints; <code>maximumEntropySequenceRows</code> bounds the last and reports <code>status.candidateOverflow</code>.
Pass available staypoint <code>status.overflow</code> and <code>status.candidateOverflow</code> alongside <code>status.count</code>,
including when the table comes directly from <code>GPUTrajectoryStaypoints</code>. Either nonzero input
suppresses every numeric indicator to NaN and both location outputs to <code>0xffffffff</code>, across all
tracks and all capacity rows. The optional output <code>status.candidateOverflow</code> is the OR of supplied
staypoint loss and entropy sequence loss; it resets on each frame, including when real entropy is
not requested. Suppression remains observable when that optional status destination is omitted.
The CPU function takes the same loss arrays beside <code>staypoints.count</code>. Omitting both loss inputs
asserts completeness. Website views hide these indicators and label them unavailable for incomplete
staypoints; Kepler exports missing numeric and location values as null.</p>
<p>{/* lens:GPUMobilityIndicators */}</p>
<h3 id="gpumodalsplit"><code>GPUModalSplit</code></h3>
<p>Per-(track, mode) accumulation of tripleg count, distance or duration (trackintel
<code>calculate_modal_split</code>). Modes come from <code>modes</code>, or are predicted from average tripleg speed when
<code>modeSpeedBreaks</code> is given (trackintel <code>simple-coarse</code>, left-closed classes, for example
<code>[15 / 3.6, 100 / 3.6, Infinity]</code> in metres per second). The last class has no upper bound, so a
speed at or above a finite last break stays in the last class. A tripleg with no speed (no time
between its first and last row, or a non-finite length) gets the missing key <code>0xffffffff</code> in
<code>output.predictedModes</code> and is left out of the split; the same key in a caller's <code>modes</code> is left
out too.</p>
<p>{/* lens:GPUModalSplit */}</p>
<h2 id="cleaning">Cleaning</h2>
<h3 id="gputrajectoryoutlierflags"><code>GPUTrajectoryOutlierFlags</code></h3>
<p>Point-centred speed and acceleration flags. The speed bit flags a fix when both adjacent steps fail,
so the valid fix after a spike is not. The acceleration bit flags a fix when the step speed rises by
more than the limit into it and falls by more than the limit after the step out of it (the two
accelerations on either side of the two fast steps; the one between them is near zero), so a
sustained jump flags no row. At a track end the missing side is waived. A single pass; two
consecutive displaced fixes are not caught (the step between them is slow), unlike
<code>movingpandas.OutlierCleaner</code>, which compares each fix to the last kept one. <code>addTrackQualityRecipe</code> applies the same rule with <code>speedOutlierMode: 'point-centred'</code>;
its default remains <code>'step'</code>.</p>
<p>{/* lens:GPUTrajectoryOutlierFlags */}</p>
<p>{/* parameters: GPUTrajectoryOutlierFlags generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectoryoutlierflags-parameters"><code>GPUTrajectoryOutlierFlags</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryOutlierFlagsParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_OUTLIER_FLAGS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryOutlierFlagsParameterValues</code>:</p>
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
<td><code>maximumSpeed</code></td>
<td><code>number</code></td>
<td>Speed limit, length unit per timestamp unit. Finite and non-negative; 0 disables the test.</td>
</tr>
<tr>
<td><code>maximumAcceleration</code> (optional)</td>
<td><code>number</code></td>
<td>Acceleration limit, length per timestamp unit squared. 0 disables the test.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectoryhampel"><code>GPUTrajectoryHampel</code></h3>
<p>Hampel filter per track (the <code>hampel</code> package used by PTRAIL): a value is an outlier when
<code>|value - median| > numberOfSigmas * 1.4826 * MAD</code> over a window centred on it; outliers become the
window median, and rows within half a window of a track end pass through unchanged. Compile-time
<code>windowSize</code>, at most 129.</p>
<p>{/* lens:GPUTrajectoryHampel */}</p>
<p>{/* parameters: GPUTrajectoryHampel generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectoryhampel-parameters"><code>GPUTrajectoryHampel</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryHampelParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_HAMPEL_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryHampelParameterValues</code>:</p>
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
<td><code>numberOfSigmas</code></td>
<td><code>number</code></td>
<td>Number of scaled median absolute deviations above which a value is an outlier.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectorysplit"><code>GPUTrajectorySplit</code></h3>
<p>Splits tracks and rewrites offsets: <code>'gap'</code> (MovingPandas <code>ObservationGapSplitter</code>), <code>'stop'</code>
(maximal runs of rows outside every staypoint, the trackintel tripleg rule: the rows of a staypoint
are in no segment, unlike the MovingPandas <code>StopSplitter</code>, which keeps the boundary fixes of each
stop), and <code>'value-change'</code> (MovingPandas <code>ValueChangeSplitter</code>, where each segment but the last
repeats the first row of the next segment, so <code>rowIds</code> can reference a row twice). Segments need at
least two rows. The output is <code>segmentOffsets</code>, <code>segmentTracks</code> and <code>rowIds</code>
with optional copied <code>positions</code> and <code>timestamps</code>, so a split feeds the next contributor without
readback.</p>
<p>{/* lens:GPUTrajectorySplit */}</p>
<p>{/* parameters: GPUTrajectorySplit generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectorysplit-parameters"><code>GPUTrajectorySplit</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectorySplitParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectorySplitParameterValues</code>:</p>
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
<td><code>maximumGap</code></td>
<td><code>number</code></td>
<td>Mode <code>'gap'</code>: the shared gap rule; a step strictly longer than this splits. 0 means no limit.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectorythinning"><code>GPUTrajectoryThinning</code></h3>
<p>Minimum-distance and minimum-time generalization (<code>MinDistanceGeneralizer</code>,
<code>MinTimeDeltaGeneralizer</code>): the first and last row of a track are kept, and each kept row is followed
by the first later row at or beyond the tolerance. As with staypoints, the sequential rule is a
bounded forward search (<code>maximumSearchRows</code>, <code>status.candidateOverflow</code>) plus pointer doubling.</p>
<p>{/* lens:GPUTrajectoryThinning */}</p>
<p>{/* parameters: GPUTrajectoryThinning generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gputrajectorythinning-parameters"><code>GPUTrajectoryThinning</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUTrajectoryThinningParameterValues(settings)</code>, writes <code>GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUTrajectoryThinningParameterValues</code>:</p>
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
<td><code>tolerance</code></td>
<td><code>number</code></td>
<td>Minimum distance (context length unit) or time (timestamps' unit) between consecutive kept rows.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gputrajectorytrackrelativetimes"><code>GPUTrajectoryTrackRelativeTimes</code></h3>
<p>Int64 word timestamps to float32 ticks after each track's first row, exact while the track spans
fewer than 2^24 ticks. Use its output as the float32 time column of the time-aware simplification
(<code>GPULineSimplification</code>, <code>metric: 'time-ratio'</code>) to run TD-TR on word timestamps, and of any other
float32 trajectory path.</p>
<p>{/* lens:GPUTrajectoryTrackRelativeTimes */}</p>`};export{e as default};