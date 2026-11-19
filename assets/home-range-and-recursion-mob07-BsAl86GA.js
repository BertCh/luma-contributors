var e={slug:`api/gpu-spatial-analysis/home-range-and-recursion-mob07`,title:`Home ranges and recursion`,html:`<h1 id="home-ranges-and-recursion">Home ranges and recursion</h1>
<p>Brownian bridge utilization distributions with volume isopleths, and recursion analysis (revisits,
residence time, first passage time) of trajectories. Planar only. Out of scope: continuous-time
movement models and autocorrelated kernel density estimation (ctmm).</p>
<p>None of this was checked against the R packages <code>move</code> or <code>recurse</code>: R is not installed here, the
licence of <code>move</code> is not verified, and both are used only as definitions. Tests use binary64 CPU
references written from the formulas below, and cases with hand-computable answers.</p>
<h2 id="gpubrownianbridgeutilization"><code>GPUBrownianBridgeUtilization</code></h2>
<p>The utilization distribution of Horne et al. (Ecology 2007). Between fixes <code>a</code> and <code>b</code>, <code>T</code> apart
in time, the position <code>alpha T</code> after <code>a</code> is bivariate normal with mean <code>a + alpha (b - a)</code> and
variance <code>T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta_a^2 + alpha^2 delta_b^2</code>, with <code>sigma^2</code>
the Brownian motion variance (length squared per time unit) and <code>delta</code> the location errors. The
density of a cell is the duration-weighted average, over all segments, of the time average of that
normal density at the cell centre. The time average over a segment is the midpoint rule with
<code>subStepCount</code> steps (default 15).</p>
<p>The dynamic Brownian bridge (Kranstauber et al. 2012) differs only in giving a variance per
segment: <code>motionVariances</code> has one row per fix, the variance of the segment that starts there. The
CPU helpers <code>estimateBrownianMotionVariance</code> and <code>estimateWindowedBrownianMotionVariance</code> produce
it (binary64, a leave-one-out maximum likelihood in the manner of Horne et al.). The likelihood
predicts every second fix from its two neighbours with variance
<code>T alpha (1 - alpha) sigma^2 + (1 - alpha)^2 delta_a^2 + alpha^2 delta_b^2 + delta_i^2</code>: the last
term is the location error of the predicted fix itself, which the three-term bridge variance of
<code>BBMM</code> and <code>move</code> leaves out. With a location error above 0 the estimate is therefore lower than
theirs (15.6 against 28.5 on a simulated walk of true variance 15 and location error 20); with no
location error the two are the same. The windowed estimate is a sliding-window variance; the
behavioural change point step of the dynamic Brownian bridge is not implemented.</p>
<p>Both estimators include the exact lower bound of zero motion variance. For diagnostics,
<code>estimateBrownianMotionVarianceWithStatus</code> returns <code>{motionVariance, status: {lowerBoundReached}}</code>;
the flag is true when the estimate is zero. <code>estimateWindowedBrownianMotionVarianceWithStatus</code>
returns <code>{motionVariances, segmentStatus}</code>, with the same diagnostic for each segment's window.
Windows with fewer than three fixes, unusable terms (including non-finite coordinates, times or
location errors), or no finite likelihood candidate return NaN and <code>lowerBoundReached: false</code>.
Boundary comparisons use stable likelihood differences, including when coordinates and errors
are rescaled or the optimum is positive but very near zero. The likelihood difference
keeps a small-change series near zero and uses log/reciprocal-variance differences for large ratios,
so heterogeneous location errors do not erase the boundary likelihood. The scan includes the
individual terms' error-transition and positive-optimum scales and refines each detected local
peak; it does not assume the mean residual scale brackets the best peak. Zero-residual terms skip
the reciprocal penalty, including for subnormal positive error variances. The estimator uses the
closed-form mean residual scale when all location errors are zero. For nonzero errors,
overlapping logarithmic scan intervals are merged before scoring; the interval from zero to the
first positive sample is refined even when that sample has a lower likelihood than zero. Thus a
positive optimum below the scan spacing is not labelled as a boundary optimum. The numeric-only helper
does not retain a per-segment diagnostic array. The original
helpers still return a number or <code>Float64Array</code>. The CPU and GPU utilization operations both
accept zero motion variance; positive location error still supplies the bridge's uncertainty.</p>
<p>The windowed helpers return one variance per segment (<code>fixCount - 1</code>), while utilization requires
one row per fix. Allocate the per-fix column and copy each track's segment variances at its first
fix row, leaving that track's unused final row zero. For one three-fix track:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {estimateWindowedBrownianMotionVariance} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> track</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {x: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], y: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], times: [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]};</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> segmentVariances</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> estimateWindowedBrownianMotionVariance</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(track, {locationError: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> motionVariances</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Float32Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(track.times.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">motionVariances.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">set</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(segmentVariances); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// [0, 0, 0], including the unused final row</span></span></code></pre>
<p>For packed tracks, use <code>motionVariances.set(segmentVariances, trackOffsets[trackRow])</code> for each
separately estimated track. Do not concatenate the segment columns without the final rows.</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>positions</code>, <code>timestamps</code>, <code>trackOffsets</code></td>
<td>The trajectory port</td>
</tr>
<tr>
<td><code>motionVariances</code>, <code>locationErrors</code></td>
<td>One float32 per fix</td>
</tr>
<tr>
<td><code>gridSize</code>, <code>parameters</code></td>
<td>Grid size (compile-time) and extent from <code>getGPUBrownianBridgeUtilizationParameterValues</code></td>
</tr>
<tr>
<td><code>output.density</code></td>
<td>Density per unit area at each cell centre, row 0 at the smallest y</td>
</tr>
</tbody>
</table>
<p>Segments farther than 7 standard deviations from a cell are skipped for that cell. The first and
last segments are used (<code>move</code> omits them by default). The tier is reference: every cell loops over
all fixes, so the fix count is limited to 16384.</p>
<p>{/* lens:GPUBrownianBridgeUtilization */}</p>
<p>{/* parameters: GPUBrownianBridgeUtilization generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpubrownianbridgeutilization-parameters"><code>GPUBrownianBridgeUtilization</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUBrownianBridgeUtilizationParameterValues(settings)</code>, writes <code>GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUBrownianBridgeUtilizationParameterValues</code>:</p>
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
<td><code>extent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Grid extent <code>[minX, minY, maxX, maxY]</code>; the cells are <code>gridSize</code> equal rectangles, row 0 at the smallest y.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpuutilizationvolumeisopleths"><code>GPUUtilizationVolumeIsopleths</code></h2>
<p>Sorts the cells by density and takes a cumulative sum, giving each cell the share of total mass
held by the cells at least as dense as it, its own mass included (<code>output.volume</code>), and a mask
<code>output.inside</code> of the smallest set of densest cells whose mass reaches <code>level</code>. A cell of zero
density has volume 1, so <code>volume &#x3C;= level</code> never selects an empty cell. Cells of equal density are
ordered by cell index. Sums are float32 in a fixed order. The volume raster of <code>move</code>
(<code>getVolumeUD</code>) serves the same purpose; whether it counts a cell's own mass was not checked, so
values can differ from it by one cell's share.</p>
<p>{/* lens:GPUUtilizationVolumeIsopleths */}</p>
<p>{/* parameters: GPUUtilizationVolumeIsopleths generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuutilizationvolumeisopleths-parameters"><code>GPUUtilizationVolumeIsopleths</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUUtilizationVolumeIsoplethsParameterValues(settings)</code>, writes <code>GPU_UTILIZATION_VOLUME_ISOPLETHS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUUtilizationVolumeIsoplethsParameterValues</code>:</p>
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
<td><code>level</code></td>
<td><code>number</code></td>
<td>Isopleth level in <code>(0, 1]</code>, for example 0.95 for the 95% home range.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gputrajectoryrecursions"><code>GPUTrajectoryRecursions</code></h2>
<p>For a circle of radius <code>r</code> around each fix (several radii in one pass), after Bracis et al.
(<code>recurse::getRecursions</code>, Ecography 2018). The path between fixes is the straight segment at
constant speed, so circle crossings are exact and times are linearly interpolated.</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Definition</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>revisits</code></td>
<td>Entries into the closed disc, not counting the visit that contains the fix</td>
</tr>
<tr>
<td><code>residenceTimes</code></td>
<td>Total time inside the disc over the whole track</td>
</tr>
<tr>
<td><code>firstPassageTimes</code></td>
<td>Time from the fix until the track first leaves the disc; if the track ends inside, the time to the end</td>
</tr>
<tr>
<td><code>firstPassageCensored</code></td>
<td>1 where the previous value is the time to the end</td>
</tr>
</tbody>
</table>
<p><code>firstPassageTimes</code> is the forward leg only: from the fix at the centre to the first exit. The
first passage time of Fauchald and Tveraa (2003) and of <code>adehabitatLT::fpt</code> is the time to cross the
whole circle, backward leg plus forward leg; on a straight track at constant speed <code>v</code> that is
<code>2 r / v</code> against <code>r / v</code> here. Add the value of the reversed track to get the two-sided time.</p>
<p>Only the fix's own track is examined, and each fix examines at most <code>maximumTrackSegments</code> segments
(default 4096); a longer track is examined in a window around the fix and sets
<code>status.candidateOverflow</code>. Not provided: polygon locations, a minimum time outside to end a visit,
and interval residence time.</p>
<p>{/* lens:GPUTrajectoryRecursions */}</p>`};export{e as default};