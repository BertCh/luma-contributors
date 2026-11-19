var e={slug:`api/gpu-spatial-analysis/temporal-context`,title:`Temporal context`,html:`<h1 id="temporal-context">Temporal context</h1>
<p>A <code>GPUTemporalContext</code> declares what a time column means, the way
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts"><code>GPUSpatialContext</code></a> declares coordinates. It is a description only: nothing here
converts data. Import it from <code>@luma-contributors/analysis/analysis-core</code> or
<code>@luma-contributors/analysis/gpu-spatial-analysis</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTemporalContext</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  unit</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'second'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'millisecond'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'microsecond'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'nanosecond'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// ArrowAnalyticsTimeUnit</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  epoch</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'unix'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> bigint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// bigint: origin as signed Int64 ticks of \`unit\` after the Unix epoch</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  encoding</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'float32-relative'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'double-single'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'int64-words'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  timeZone</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// as declared by the source; never parsed</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  axis</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">    |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">kind</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'regular'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">step</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> number</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> bigint</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">calendarUnit</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUTemporalCalendarUnit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">interval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}}</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">    |</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">kind</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'irregular'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span></code></pre>
<p><code>unit</code> is also the unit of durations given with the times, such as <code>maximumGap</code>.</p>
<h2 id="encodings">Encodings</h2>
<table>
<thead>
<tr>
<th>Encoding</th>
<th>Rows</th>
<th>Reads</th>
<th>Epoch</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>int64-words</code></td>
<td>One <code>uint32x2</code> Int64 <code>(low, high)</code> row, two's complement</td>
<td>Exact. Take differences on the words with <code>timeWordsDifferenceToF32</code>, then use f32</td>
<td><code>'unix'</code> or an origin</td>
</tr>
<tr>
<td><code>double-single</code></td>
<td>A <code>float32</code> high and a <code>float32</code> low part; the time is their sum</td>
<td>About 48 bits</td>
<td><code>'unix'</code> or an origin</td>
</tr>
<tr>
<td><code>float32-relative</code></td>
<td>One <code>float32</code>, ticks after the origin</td>
<td>Exact to 2^24 ticks from the origin</td>
<td>An origin; <code>'unix'</code> is rejected because float32 cannot hold absolute time</td>
</tr>
</tbody>
</table>
<p><code>0n</code> is the same instant as <code>'unix'</code>; declare it for times that really are near 1970 or have no
calendar meaning.</p>
<h2 id="resolving-and-checking">Resolving and checking</h2>
<p><code>resolveGPUTemporalContext(id, context, support?)</code> validates a context and checks it against what an
operation accepts (<code>encodings</code>, <code>units</code>, <code>axisKinds</code>, <code>unixEpoch</code>). There is no default context,
because a unit or epoch must be declared. It rejects an unknown unit or encoding, an origin outside
Int64, <code>float32-relative</code> with <code>'unix'</code>, an empty <code>timeZone</code>, a non-positive or non-finite step, a
fractional step on <code>int64-words</code>, and a bad calendar step.</p>
<p><code>getGPUTemporalContextCompatibility(source, target)</code> and
<code>assertCompatibleGPUTemporalContexts(id, source, target)</code> compare two columns on unit, epoch and
encoding. Time zone and axis are not compared: stored instants stay comparable across both. A
contributor that needs matching axes checks them itself.</p>
<h2 id="arrow-uploads">Arrow uploads</h2>
<p><code>makeGPUTimeVectorFromPreparedArrowColumn</code> returns the upload with a <code>temporalContext</code>, and
<code>getGPUTemporalContextFromPreparedArrowColumn(column)</code> gives the same context without a device:
<code>int64-words</code>, Unix epoch, the column's own unit and declared time zone, and an irregular axis.
A signed Int64 column prepared with a declared unit is taken to count from the Unix epoch; spread
another <code>epoch</code> over the result if it does not.</p>
<h2 id="what-each-family-accepts-today">What each family accepts today</h2>
<p>These contributors take a context. The rest still document their conventions per prop, and the table
states them in terms of this contract. A context is always optional where the time view alone fixes the
encoding, and it never converts anything: it is checked against the bound views and the operation's
support at construction.</p>
<table>
<thead>
<tr>
<th>Family</th>
<th>Contributors</th>
<th>Encoding</th>
<th>Unit and epoch</th>
</tr>
</thead>
<tbody>
<tr>
<td>Calendar</td>
<td><code>GPUCalendarBuckets</code></td>
<td><code>int64-words</code></td>
<td>Any unit (<code>unit</code> or <code>temporalContext</code>), Unix epoch required</td>
</tr>
<tr>
<td>Period index and relative words</td>
<td><code>GPUPeriodIndex</code>, <code>GPUTimeWordsRelative</code></td>
<td><code>int64-words</code> in; <code>GPUTimeWordsRelative</code> writes <code>float32-relative</code> and exposes the context of its output as <code>outputTemporalContext</code></td>
<td>Any unit; the context is required</td>
</tr>
<tr>
<td>Time filters and flows</td>
<td><code>GPUTimeWindowFilter</code>, <code>GPUFlowAggregation</code></td>
<td><code>int64-words</code>, <code>float32-relative</code>, or <code>double-single</code> through <code>timestampsLow</code></td>
<td>Any unit, caller's epoch</td>
</tr>
<tr>
<td>Temporal reduction</td>
<td><code>GPUTemporalReduction</code></td>
<td><code>int64-words</code> or <code>float32-relative</code></td>
<td>Any unit, caller's epoch</td>
</tr>
<tr>
<td>Trajectory metrics, resample, playhead, zone events, encounters, closest approach</td>
<td><code>GPUTrajectoryGeometryPort.temporalContext</code></td>
<td><code>int64-words</code> for <code>uint32x2</code> rows, <code>float32-relative</code> for <code>float32</code> rows; metrics also take <code>double-single</code> with <code>timestampsLow</code>. The context must match the bound view</td>
<td>Any unit; irregular axis only; <code>maximumGap</code> in the context unit</td>
</tr>
<tr>
<td>Kalman and restriction</td>
<td><code>GPUTrajectoryKalman</code>, trajectory restriction (<code>temporalContext</code> beside the <code>times</code> or <code>timestamps</code> fields)</td>
<td><code>int64-words</code> or <code>float32-relative</code></td>
<td>Kalman derives <code>timeUnitsPerSecond</code> from the context unit (1, 1e3, 1e6 or 1e9) for either encoding, and a <code>timeUnitsPerSecond</code> given with it must agree; without a context it is for words only, and float32 times are seconds</td>
</tr>
<tr>
<td>Pair tests</td>
<td><code>GPUKnoxTest</code>, <code>GPUMantelTest</code></td>
<td><code>float32-relative</code>, origin subtracted on the CPU</td>
<td>Caller's unit; no context prop</td>
</tr>
<tr>
<td>Near-repeat, local Knox and Jacquez</td>
<td><code>GPUNearRepeatKnox</code>, <code>GPULocalKnox</code>, <code>GPUJacquezTest</code> (<code>temporalContext</code>)</td>
<td><code>float32-relative</code> only; <code>int64-words</code> and <code>double-single</code> are refused</td>
<td>Any unit, any axis, and the epoch must be a declared bigint origin (<code>'unix'</code> is refused for float32). <code>getGPUSpaceTimeGuardedRelativeTimes</code> builds the times and returns the context; <code>GPUTimeWordsRelative.outputTemporalContext</code> is the context of times produced on the GPU</td>
</tr>
<tr>
<td>Space-time kriging</td>
<td><code>GPUSpaceTimeKriging</code> (<code>temporalContext</code>, optional <code>queryTemporalContext</code>)</td>
<td><code>float32-relative</code> only; the query context must agree with the sample context on unit, epoch and encoding</td>
<td>Any unit and axis. <code>GPUTimeWordsRelative.outputTemporalContext</code> provides the context</td>
</tr>
<tr>
<td>Space-time clustering</td>
<td><code>GPUSpatialClustering</code>, <code>GPUOPTICS</code> (<code>temporalContext</code> with <code>times</code>; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/space-time-clustering">Space-time clustering</a>)</td>
<td><code>float32-relative</code> only; the context is required with <code>times</code> and rejected without</td>
<td>Any unit, time zone and axis; <code>timeEpsilon</code> and <code>maximumTimeDistance</code> are in the context unit</td>
</tr>
<tr>
<td>Neighbor search</td>
<td><code>GPUNeighborSearch</code> (<code>temporalContext</code> with <code>times</code>; see <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-weights"><code>GPUNeighborSearch</code></a>)</td>
<td><code>float32-relative</code> only; the context is required with <code>times</code> and rejected without</td>
<td>Any unit, time zone and axis; <code>timeRadius</code> and <code>timeBounds</code> are in the context unit</td>
</tr>
<tr>
<td>Variogram</td>
<td><code>GPUSpaceTimeVariogram</code></td>
<td><code>float32-relative</code></td>
<td>Caller's unit and origin</td>
</tr>
<tr>
<td>Cube statistics</td>
<td><code>GPUEmergingHotSpots</code>, <code>addSpaceTimeHotSpotsRecipe</code>, <code>GPUValueCube</code>, <code>GPUCellPeriodKeys</code></td>
<td>Integer slice index, cell-major</td>
<td>The recipe's event words are Int64 milliseconds from the Unix epoch</td>
</tr>
<tr>
<td>Dynamics and panels</td>
<td>Markov chains, spatial panels</td>
<td>Integer period index, period-major</td>
<td>Not a time column</td>
</tr>
<tr>
<td>Raster stacks</td>
<td><code>GPUChangeDetection</code>, <code>GPURasterTemporal</code>, <code>GPURasterSeasonalBreak</code> (<code>dateAxis</code>)</td>
<td>Integer slice index on regular slots, or <code>int64-words</code> with <code>{kind: 'irregular'}</code> through <code>dateAxis</code></td>
<td>Any unit and epoch for the irregular axis, read as centred fractional years of 365.2425 days</td>
</tr>
<tr>
<td>Raster time series</td>
<td><code>GPURasterWhittakerSmoother</code>, <code>GPURasterHANTS</code>, <code>GPURasterTemporalResample</code> (<code>dateAxis</code>); <code>GPURasterStreamingAccumulator</code> and the RTS-06 climate contributors take chunks with slot indices</td>
<td><code>int64-words</code> with <code>{kind: 'irregular'}</code> for dates; slot indices otherwise</td>
<td>See <a href="/luma-contributors/docs/api/gpu-raster/raster-time-series">Raster time series</a></td>
</tr>
</tbody>
</table>`};export{e as default};