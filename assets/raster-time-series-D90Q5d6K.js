var e={slug:`api/gpu-raster/raster-time-series`,title:`Raster time series`,html:`<h1 id="raster-time-series">Raster time series</h1>
<p><code>GPURasterHarmonicRegression</code>, <code>GPURasterClimateRunAccumulator</code>, <code>GPURasterGeometricMedianComposite</code>,
<code>GPURasterBestAvailablePixel</code>, <code>GPURasterBFASTMonitor</code>, <code>GPURasterClimateAnomaly</code>, <code>GPURasterHANTS</code>,
<code>GPURasterEOFProjection</code> and <code>GPURasterCOLDMonitor</code> accept aligned chunked cell
records. Keep each complete time series, mask and output record within a chunk; slice limits still
apply per cell. Harmonic regression defaults to storage-backed QR; the other reference slice caps remain.</p>
<p>Both time-series execution routes use the same cell topology: packed fields can accompany a
single-chunk vector when each field holds exactly <code>cellCount * entriesPerCell</code> entries. Once a
field has multiple chunks, every cell field must have the same number of chunks and the same
cell counts per chunk, with no split cell records. Shared date and parameter tables stay packed.
Calls using only packed fields retain their documented spare-capacity support.
HANTS and temporal resampling type their output views independently of the source view: a packed
source can feed matching one-chunk outputs, and a one-chunk source can feed packed outputs.</p>
<p>Writable logical ranges must be disjoint across fields and chunks. Co-bound writable ranges
must also have disjoint storage-aligned binding ranges. HANTS and temporal resampling can use
one pooled output buffer under these rules. A dispatch's read-only inputs and writable outputs
must use separate physical buffers, even for disjoint ranges. A physical buffer participating in
writes must use one imported handle; a buffer used only by read-only inputs may have multiple
imports. Pooling output fields requires an explicit per-dispatch access declaration in the
contributor; callers without one retain separate output buffers because a later dispatch may read
one output while writing another. HANTS and temporal resampling declare their accesses explicitly.</p>
<p>Contributors for stacks of raster time slices beyond regular composites: an irregular acquisition-date
axis, streaming accumulators without a slice limit, penalised and harmonic smoothers, gap filling onto a
regular grid, and climate indices. Import from <code>@luma-contributors/analysis/raster-analysis</code>. All are
<code>GPUCommandNodeProducer</code> classes. Every contributor is cell-major (<code>cell * sliceCount + slice</code>) and takes
<code>mask</code> per observation with zero rejecting a sample; non-finite values are missing. Each class
documents its per-frame and compile-time props in its source.</p>
<h2 id="irregular-date-axis">Irregular date axis</h2>
<p><code>GPURasterIrregularDateAxis</code> declares a stack's acquisition dates: one <code>uint32x2</code> Int64 <code>(low, high)</code> row
per slice, with a <a href="/luma-contributors/docs/api/gpu-spatial-analysis/temporal-context"><code>GPUTemporalContext</code></a> of encoding
<code>int64-words</code> and <code>{kind: 'irregular'}</code>, any unit and epoch. Dates must be strictly increasing. They become
<strong>centred fractional years</strong> in float32: years of 365.2425 days after the midpoint of the first and last
acquisition, so the trend column is O(1) and harmonics stay in <code>[-1, 1]</code>. Elapsed years come from the Int64
words directly, so nearby dates keep full float32 precision. <code>getGPURasterCentredFractionalYears</code> gives the
same axis on the CPU, for example to choose a regular output grid.</p>
<p><code>GPURasterTemporal</code>, <code>GPUChangeDetection</code> and <code>GPURasterSeasonalBreak</code> take this axis as <code>dateAxis</code>.
Omitted, slot behaviour and results are unchanged.</p>
<ul>
<li><code>GPURasterTemporal</code>: the Savitzky-Golay polynomial is fitted on real elapsed time over the same window of
neighbouring acquisitions. Composites do not depend on time.</li>
<li><code>GPUChangeDetection</code>: <code>senSlope</code> is per year of real elapsed time. Other outputs depend on order only.</li>
<li><code>GPURasterSeasonalBreak</code>: <code>period</code> is in fractional years, the slope is per year, and <code>missingPolicy: 'interpolate'</code> uses elapsed time. Dummy seasons are defined on regular slots and are rejected.</li>
</ul>
<h2 id="gpurasterstreamingaccumulator"><code>GPURasterStreamingAccumulator</code></h2>
<p>Accumulates counts, sums, squared deviations and extrema chunk by chunk, so a stack has no slice limit.
Each dispatch consumes at most 256 slices; memory is O(cells x calendar slots). Slot indices assign a
calendar slot (for example a month) to each chunk slice.</p>
<p>Set <code>reset</code> on the first chunk of a stream. <code>activeSliceCount</code> allows a short last chunk. Declare
<code>maximumValueMagnitude</code>: a sample above it is treated as missing. Sums and squared deviations are <strong>96-bit
fixed-point integers</strong> (three <code>uint32</code> words per slot, decoded with <code>getGPURasterFixedPointValues</code>), because
float32 compensated summation was optimized away by the shader compiler on the tested GPU (the low word stayed
zero). Fixed-point sums are exact to the declared unit, about <code>maximumValueMagnitude * 2^-70</code> at the default
stream length.</p>
<p>{/* parameters: GPURasterStreamingAccumulator generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterstreamingaccumulator-parameters"><code>GPURasterStreamingAccumulator</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)</code>, writes <code>GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH</code> (5) <code>float32</code> values.
<ul>
<li><code>chunkSliceCount</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterStreamingAccumulatorParameterValues</code>:</p>
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
<td><code>reset</code></td>
<td><code>boolean</code></td>
<td>True on the first chunk of a stream, which clears the state first.</td>
</tr>
<tr>
<td><code>activeSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Slices of this chunk that hold data; defaults to the whole chunk.</td>
</tr>
<tr>
<td><code>maximumValueMagnitude</code></td>
<td><code>number</code></td>
<td>Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare it tightly, because a sample above it is treated as missing, and the unit is about <code>maximumValueMagnitude * 2^-70</code> at the default stream length.</td>
</tr>
<tr>
<td><code>maximumStreamSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Longest stream in slices, for the fixed-point headroom. Defaults to 2^24.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterrollingmean"><code>GPURasterRollingMean</code></h2>
<p>Trailing rolling mean over a chunked stream, the aggregation SPI and SPEI apply before fitting. A missing
sample in the window makes that output missing, as xclim's <code>rolling(...).mean(skipna=False)</code>. Window state
lives in caller-owned buffers between chunks.</p>
<p>{/* parameters: GPURasterRollingMean generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterrollingmean-parameters"><code>GPURasterRollingMean</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)</code>, writes <code>GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH</code> (5) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
<li><code>chunkSliceCount</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterStreamingAccumulatorParameterValues</code>:</p>
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
<td><code>reset</code></td>
<td><code>boolean</code></td>
<td>True on the first chunk of a stream, which clears the state first.</td>
</tr>
<tr>
<td><code>activeSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Slices of this chunk that hold data; defaults to the whole chunk.</td>
</tr>
<tr>
<td><code>maximumValueMagnitude</code></td>
<td><code>number</code></td>
<td>Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare it tightly, because a sample above it is treated as missing, and the unit is about <code>maximumValueMagnitude * 2^-70</code> at the default stream length.</td>
</tr>
<tr>
<td><code>maximumStreamSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Longest stream in slices, for the fixed-point headroom. Defaults to 2^24.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterwhittakersmoother"><code>GPURasterWhittakerSmoother</code></h2>
<p>Weighted Whittaker smoother per cell: minimises <code>sum w (y - z)^2 + lambda |D z|^2</code> by a banded Givens QR of
<code>[sqrt(W); sqrt(lambda) D]</code> on the values minus their per-cell mean (added back afterwards; the smoother is
exact on constants), with zero weight on missing observations so gaps are filled. <code>D</code> is the order <code>d</code> difference (1 to 3). With
<code>dateAxis</code> it is the divided difference on acquisition dates, scaled by the mean spacing, so a regular axis
reduces to ordinary differences and <code>lambda</code> keeps its usual meaning. <code>lambda</code> is limited to <code>1e7</code>.</p>
<p>Fixed robust reweighting passes: Tukey <code>bisquare</code> on the residual over 6 MAD, or <code>upper-envelope</code> for
cloud-depressed series. Reference: the phenofit <code>smooth_wWHIT</code> definition (Eilers 2003, divided differences);
R is not installed, so the fixtures use a binary64 SciPy <code>solve_banded</code> implementation of the same algorithm
(<code>test/oracles/rts02-whittaker-hants-scipy.py</code>). phenofit itself remains a pending oracle.</p>
<p>float32 limits: the result agrees with a binary64 solve to about one float32 step of the published value
for every difference order and every <code>lambda</code> up to the cap of <code>1e7</code> (256 slices, series at 0.5, 300 and
3000 with one gap in five), and to within <code>2e-5</code> on the irregular fixture series. The error does not grow
with the offset of the data (kelvin, scaled reflectance). Status 2 means the triangular factor has a zero
or non-finite diagonal or the solution is not finite, for example <code>lambda = 0</code> with a gap; it is not raised
for a system that is only badly conditioned. Very sparse support is solved, with status 0: 2, 3 or 5 valid
observations out of 256 slices, at either end, in the middle or spread out, agree with an exact rational
solve (<code>test/oracles/rts02-whittaker-sparse-exact.py</code>) for <code>lambda</code> from 1 to <code>1e7</code>. There, orders 2 and 3
continue the curve as a line or a parabola across the gap and the float32 error grows with the gap: over
250 slices beyond 3 or 5 adjacent observations it stays within <code>6e-4</code> (order 2) and <code>1.3e-2</code> (order 3) of
the spread of the curve, and within one float32 step of the values for order 1 or for observations spread
over the series (order 3 spread: <code>1e-3</code> of the spread).</p>
<p>{/* parameters: GPURasterWhittakerSmoother generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterwhittakersmoother-parameters"><code>GPURasterWhittakerSmoother</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterWhittakerSmootherParameterValues(settings)</code>, writes <code>GPU_RASTER_WHITTAKER_SMOOTHER_PARAMETER_LENGTH</code> (2) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterWhittakerSmootherParameterValues</code>:</p>
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
<td><code>smoothingParameter</code></td>
<td><code>number</code></td>
<td>Penalty weight lambda of the Whittaker smoother; larger is smoother.</td>
</tr>
<tr>
<td><code>minimumWeight</code> (optional)</td>
<td><code>number</code></td>
<td>Robust-weight floor, so a point is never removed outright. Defaults to <code>0.01</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterhants"><code>GPURasterHANTS</code></h2>
<p>Harmonic ANalysis of Time Series (Roerink, Menenti and Verhoef 2000; GRASS <code>r.hants</code>): a least-squares
harmonic fit (<code>getRasterLeastSquaresWGSL</code>) with a bounded reject-and-refit loop that drops the worst
observation beyond the fit-error tolerance on the chosen side. A cell out of rejection passes publishes its
latest fit and reports status 8 (<code>converged = 0</code>). <code>2 * harmonicCount + 1 + degreeOfOverdetermination</code>
observations are always kept. Coefficients are centred on the axis midpoint. The GRASS module is a definition
reference only; the fixture is a binary64 NumPy implementation, and GRASS itself remains a pending oracle.</p>
<p>Working memory is checked at construction: 256 slices allow two harmonics; 128 slices allow six.</p>
<p>{/* parameters: GPURasterHANTS generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterhants-parameters"><code>GPURasterHANTS</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterHANTSParameterValues(settings)</code>, writes <code>GPU_RASTER_HANTS_PARAMETER_LENGTH</code> (3) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterHANTSParameterValues</code>:</p>
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
<td><code>period</code></td>
<td><code>number</code></td>
<td>Base period of the harmonics, in fractional years with a date axis, otherwise in slices.</td>
</tr>
<tr>
<td><code>fitErrorTolerance</code></td>
<td><code>number</code></td>
<td>Fit-error tolerance: observations further than this from the curve are rejected.</td>
</tr>
<tr>
<td><code>degreeOfOverdetermination</code> (optional)</td>
<td><code>number</code></td>
<td>Observations kept above the <code>2 * harmonicCount + 1</code> minimum; defaults to 1.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurastertemporalresample"><code>GPURasterTemporalResample</code></h2>
<p>Fills a regular time grid (<code>gridStart</code>, <code>gridStep</code> in centred fractional years, up to 4096 points) by linear
interpolation between bracketing valid observations. Grid points outside the observed range get status 1;
points inside a gap larger than <code>maximumGap</code> get status 2. Nothing is extrapolated. A Whittaker or HANTS output
is finite in its gaps, so it feeds this node directly, which then feeds regular-time operations.</p>
<p>{/* parameters: GPURasterTemporalResample generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurastertemporalresample-parameters"><code>GPURasterTemporalResample</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterTemporalResampleParameterValues(settings)</code>, writes <code>GPU_RASTER_TEMPORAL_RESAMPLE_PARAMETER_LENGTH</code> (3) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterTemporalResampleParameterValues</code>:</p>
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
<td><code>gridStart</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>gridStep</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>maximumGap</code> (optional)</td>
<td><code>number</code></td>
<td>Largest allowed span between the bracketing observations; defaults to no limit.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="compositing-and-phenology">Compositing and phenology</h2>
<p>Per-cell compositing across the slices of a stack, and phenology metrics on a smoothed curve. The multivariate
contributors use the layout <code>(cell * sliceCount + slice) * bandCount + band</code> and treat an observation with any
non-finite band as missing as a whole; <code>mask</code> is per observation (<code>cell * sliceCount + slice</code>) and zero
rejects it. A cell without a valid observation publishes NaN and <code>0xffffffff</code> as its slice.</p>
<p>Reference: hdmedians (Apache-2.0) defines <code>nanmedoid</code> and the geometric median. hdmedians is not installed
here, so the fixture <code>test/oracles/fixtures/rts05-compositing-numpy.json</code> comes from binary64 NumPy
implementations of those definitions (<code>test/oracles/rts05-compositing-numpy.py</code>); the geometric median is
also checked against the true minimiser found by SciPy. TIMESAT parity and double-logistic fits are out of
scope: the latter are ill-conditioned in float32.</p>
<h3 id="gpurastermedoidcomposite"><code>GPURasterMedoidComposite</code></h3>
<p>The valid observation with the smallest sum of Euclidean distances (over all bands) to the other valid
observations. The result is a real acquisition, so <code>output.sliceIndex</code> is the provenance of every band. Ties go
to the earliest slice. Work is <code>O(slices^2 * bands)</code> per cell on one invocation, so <code>sliceCount</code> is limited to 64
and <code>bandCount</code> to 16. Float32 distance sums that differ by less than rounding can pick a different slice than
binary64.</p>
<h3 id="gpurastergeometricmediancomposite"><code>GPURasterGeometricMedianComposite</code></h3>
<p>The point minimising the sum of Euclidean distances to the valid observations over all bands together (Roberts,
Mueller and McIntyre 2017): band relationships are preserved, so an index computed after compositing equals the
index computed before. The result is synthetic. It runs Weiszfeld's recursion from the mean for exactly
<code>iterationCount</code> iterations (at most 64, no data-dependent exit), with each distance floored at
<code>distanceEpsilon</code>. <code>output.lastStep</code> is the length of the last step, the convergence indicator to read when the
observations are few or skewed. The fixture cells agree with binary64 to within <code>2e-5</code> and sit within <code>1e-5</code>
relative of the SciPy minimum after 32 iterations.</p>
<p>Call <code>getGPURasterGeometricMedianCompositeOutputLayout({cellCount, bandCount, lastStep})</code> before
allocating outputs. It returns scalar formats and element counts without allocating storage.
<code>lastStep</code> defaults to false: its descriptor is absent unless requested. The contributor's
validation, <code>GPURasterGeometricMedianCompositeOperation</code>, the CPU operation's graph outputs and
the pure CPU result's allocation all use this layout. The pure CPU function always returns
<code>lastStep</code>; the operation publishes it only when <code>output.lastStep</code> names a graph value.</p>
<p>{/* geometric-median-output-layout:start */}</p>
<table>
<thead>
<tr>
<th>Output field</th>
<th>Scalar format</th>
<th>Element count</th>
<th>Condition and meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>composite</code></td>
<td><code>float32</code></td>
<td><code>cellCount * bandCount</code></td>
<td>Always: <code>cell * bandCount + band</code>; NaN when the cell has no valid observation.</td>
</tr>
<tr>
<td><code>status</code></td>
<td><code>uint32</code></td>
<td><code>cellCount</code></td>
<td>Always: per-cell code 0 for a composite, 1 for no valid observation.</td>
</tr>
<tr>
<td><code>lastStep</code></td>
<td><code>float32</code></td>
<td><code>cellCount</code></td>
<td>When <code>lastStep: true</code>: final step length per cell; NaN when there is no valid observation.</td>
</tr>
</tbody>
</table>
<p>{/* geometric-median-output-layout:end */}</p>
<p>The table is checked against the public layout in the node tests. Counts and the
<code>cellCount * bandCount</code> product must be positive integers no greater than 4,294,967,295.
The layout covers both execution forms; GPU execution additionally limits each cell to 256 slices,
16 bands and 64 iterations. The CPU twin retains the iteration budget and uint32 indexing limits,
and allows larger slice, band and iteration counts. The GPU operation uses exact-sized packed
graph values; a direct contributor also accepts packed output views with spare capacity and
nonzero byte offsets, writing only the used prefix. Chunked cell fields must follow the shared
time-series topology rules above. Output formats must match the table, including <code>uint32</code> status.</p>
<p>For example, two cells with three bands need six <code>float32</code> composite entries and two <code>uint32</code>
status entries. Requesting <code>lastStep: true</code> adds two <code>float32</code> entries. Input masks and output
buffers remain caller-owned.</p>
<p>{/* parameters: GPURasterGeometricMedianComposite generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurastergeometricmediancomposite-parameters"><code>GPURasterGeometricMedianComposite</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterGeometricMedianCompositeParameterValues(settings?)</code>, writes <code>GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterGeometricMedianCompositeParameterValues</code>:</p>
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
<td><code>distanceEpsilon</code> (optional)</td>
<td><code>number</code></td>
<td>Smallest distance used as a Weiszfeld weight denominator, in value units: an estimate that lands on an observation stays there instead of dividing by zero. Defaults to <code>1e-5</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpurasterbestavailablepixel"><code>GPURasterBestAvailablePixel</code></h3>
<p>Best-available-pixel scoring (Griffiths et al. 2013): <code>score = wDay * exp(-0.5 (dDay / sigma)^2) + wYear * max(0, 1 - |year - targetYear| / maximumYearDistance) + wCloud * clamp((cloudDistance - minimum) / (maximum - minimum), 0, 1)</code>, with <code>dDay</code> the circular distance in days to the target day of year. The cell takes
the highest score, the earliest slice on ties, and optionally its band values. Per-slice day of year and year come
in <code>sliceAttributes</code>. <code>cloudDistance</code> is per observation, in pixels to the nearest cloud or shadow of its slice:
run <code>GPUDistanceField</code> on each slice's cloud mask and pass the result (positive infinity means no cloud in
reach; without the view the cloud score is zero).</p>
<p>{/* parameters: GPURasterBestAvailablePixel generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurasterbestavailablepixel-parameters"><code>GPURasterBestAvailablePixel</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterBestAvailablePixelParameterValues(settings)</code>, writes <code>GPU_RASTER_BEST_AVAILABLE_PIXEL_PARAMETER_LENGTH</code> (9) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterBestAvailablePixelParameterValues</code>:</p>
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
<td><code>targetDayOfYear</code></td>
<td><code>number</code></td>
<td>Target day of year in <code>[0, 366]</code>.</td>
</tr>
<tr>
<td><code>dayOfYearSigma</code></td>
<td><code>number</code></td>
<td>Standard deviation of the Gaussian day-of-year score, in days.</td>
</tr>
<tr>
<td><code>dayOfYearWeight</code></td>
<td><code>number</code></td>
<td>Weight of the day-of-year score; Griffiths et al. 2013 example <code>0.5</code>.</td>
</tr>
<tr>
<td><code>targetYear</code></td>
<td><code>number</code></td>
<td>Target year in the unit of the <code>sliceAttributes</code> year column.</td>
</tr>
<tr>
<td><code>maximumYearDistance</code></td>
<td><code>number</code></td>
<td>Year distance at which the linear year score reaches zero.</td>
</tr>
<tr>
<td><code>yearWeight</code></td>
<td><code>number</code></td>
<td>Weight of the year score; example <code>0.2</code>.</td>
</tr>
<tr>
<td><code>minimumCloudDistance</code></td>
<td><code>number</code></td>
<td>Cloud distance at or below which the cloud score is zero (pixels), example <code>10</code>.</td>
</tr>
<tr>
<td><code>maximumCloudDistance</code></td>
<td><code>number</code></td>
<td>Cloud distance at or above which the cloud score is one (pixels), example <code>100</code>.</td>
</tr>
<tr>
<td><code>cloudWeight</code></td>
<td><code>number</code></td>
<td>Weight of the cloud-distance score; example <code>0.3</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpurasterphenologymetrics"><code>GPURasterPhenologyMetrics</code></h3>
<p>Up to two seasons per cell from amplitude thresholds on the smoothed curve, for example <code>output.smoothed</code> of
<code>GPURasterWhittakerSmoother</code> (same <code>cell * sliceCount + slice</code> layout; pass the same <code>dateAxis</code> and times are
centred fractional years, otherwise slice indices). Base is the curve minimum and amplitude the maximum minus the
minimum. A season starts where the curve last falls below <code>startFraction</code> of the amplitude before its peak and ends
where it first falls below <code>endFraction</code> after it, both by linear interpolation (the Copernicus MR-VPP
definitions use 0.25 and 0.15). The first season is the highest sample; the second is the highest sample outside
the first season's span that reaches <code>secondSeasonMinimumFraction</code>. Seasons are ordered by peak time. A crossing
that is not found cuts the season at the series end or at the other season and sets a bit of
<code>GPU_RASTER_PHENOLOGY_FLAG</code>. <code>output.metrics</code> holds the ten fields of <code>GPU_RASTER_PHENOLOGY_METRIC</code> per season:
start, end, peak time, length, peak value, base value, amplitude, left and right slope, and the integral between
the curve and the base from start to end. Cell status is 1 for a non-finite curve and 2 for an amplitude below
<code>minimumAmplitude</code>.</p>
<p>{/* parameters: GPURasterPhenologyMetrics generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpurasterphenologymetrics-parameters"><code>GPURasterPhenologyMetrics</code> parameters</h4>
<ul>
<li>Packer: <code>getGPURasterPhenologyMetricsParameterValues(settings)</code>, writes <code>GPU_RASTER_PHENOLOGY_METRICS_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterPhenologyMetricsParameterValues</code>:</p>
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
<td><code>startFraction</code></td>
<td><code>number</code></td>
<td>Fraction of the curve amplitude above the base where a season starts, in <code>(0, 1)</code>; MR-VPP uses <code>0.25</code>.</td>
</tr>
<tr>
<td><code>endFraction</code></td>
<td><code>number</code></td>
<td>Fraction of the curve amplitude above the base where a season ends, in <code>(0, 1)</code>; MR-VPP uses <code>0.15</code>.</td>
</tr>
<tr>
<td><code>minimumAmplitude</code></td>
<td><code>number</code></td>
<td>Cells whose amplitude (maximum minus minimum) is below this get status 2 and no season.</td>
</tr>
<tr>
<td><code>secondSeasonMinimumFraction</code> (optional)</td>
<td><code>number</code></td>
<td>A second season needs a peak at least this fraction of the amplitude above the base. It must be at least both start and end fractions, so the shoulders of the first season are not a second one. Defaults to the larger of the two.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="spi-and-spei">SPI and SPEI</h2>
<p><code>GPURasterStandardizedIndex</code> evaluates up to 16,777,216 slices per chunk, subject to device
bindings. Calibration keeps its separate per-dispatch limit. Cell records and aligned outputs
accept chunked input without changing the fitted calendar-slot model.</p>
<p><code>GPURasterStandardizedIndexCalibration</code> streams calibration chunks (for example 30 years of months) into
per-slot statistics, with zeros and non-positive values excluded from the fit and counted, and writes a fit
per calendar slot. <code>GPURasterStandardizedIndex</code> then evaluates any chunk against that fit. Pass the rolling
mean output for aggregations such as SPI-3. Both match xclim's <code>method='APP'</code> (Thom's gamma estimator for
<code>gamma</code> with <code>floc</code>, the method-of-moments estimator for <code>fisk</code> with <code>floc</code>) and xclim's defaults for zero
inflation (<code>prob_zero_interpolation='upper'</code>, <code>plotting_position_zero='ecdf'</code>). <code>location</code> is the fixed
<code>floc</code> used for water-balance series that may be negative.</p>
<p>Reference: xclim 1.0.0, Apache-2.0, run for <code>test/oracles/fixtures/rts06-climate-indices-xclim.json</code> (numba and
dask imports stubbed; see the fixture). SPI-1, SPI-3 and SPEI agree with xclim to within 0.01 over 420 months in
float32.</p>
<p>float32 limits and what the code does about them:</p>
<ul>
<li><strong>APP, not iterative ML.</strong> Newton iterations on the shape are noisy in float32; APP is closed form. Only
sums over the slot enter it.</li>
<li><strong>Fit sums are fixed-point, not float32.</strong> The gamma estimator is driven by <code>ln(mean) - mean(ln x)</code>, a
difference of nearly equal numbers. The sums of <code>x</code>, <code>ln x</code> and squared deviations are exact 96-bit integers,
so the only float32 rounding is the final conversion to a mean.</li>
<li><strong>Tails are computed directly.</strong> The gamma CDF and quantile near 1 cannot be represented as <code>1 - p</code> in
float32, so whichever tail is small is computed directly (series below, continued fraction above) and the
quantile is taken of that tail. Results stay accurate on both sides.</li>
<li><strong>Clipped to <code>+-8.21</code></strong>, xclim's bound. Tails flush to zero near <code>1e-38</code> in float32, well beyond that.</li>
<li><strong>The fit needs at least two positive values per slot</strong>; a slot without a usable fit marks every output NaN
with status 2.</li>
</ul>
<p>{/* parameters: GPURasterStandardizedIndex generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterstandardizedindex-parameters"><code>GPURasterStandardizedIndex</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterStandardizedIndexParameterValues(settings, chunkSliceCount)</code>, writes <code>GPU_RASTER_STANDARDIZED_INDEX_PARAMETER_LENGTH</code> (2) <code>float32</code> values.
<ul>
<li><code>chunkSliceCount</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterStandardizedIndexParameterValues</code>:</p>
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
<td><code>activeSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>location</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<p>{/* parameters: GPURasterStandardizedIndexCalibration generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterstandardizedindexcalibration-parameters"><code>GPURasterStandardizedIndexCalibration</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterStandardizedIndexCalibrationParameterValues(settings, chunkSliceCount)</code>, writes <code>GPU_RASTER_STANDARDIZED_CALIBRATION_PARAMETER_LENGTH</code> (7) <code>float32</code> values.
<ul>
<li><code>chunkSliceCount</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterStandardizedIndexCalibrationParameterValues</code>:</p>
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
<td><code>reset</code></td>
<td><code>boolean</code></td>
<td>-</td>
</tr>
<tr>
<td><code>activeSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>location</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>maximumValueMagnitude</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>maximumStreamSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterclimaterunaccumulator"><code>GPURasterClimateRunAccumulator</code></h2>
<p>Streaming spell lengths, exceedance counts and degree days per cell against a threshold. The condition
(<code>above</code>, <code>at-or-above</code>, <code>below</code>, <code>at-or-below</code>) is fixed at construction and matches xclim's comparison
operators; spell length, wet/dry spell and degree-day definitions match xclim 1.0.0 for reset-per-year chunks.
A missing sample breaks the run and is counted. <code>minimumSpellLength</code> counts spells when they reach the length.</p>
<p>{/* parameters: GPURasterClimateRunAccumulator generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterclimaterunaccumulator-parameters"><code>GPURasterClimateRunAccumulator</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterClimateRunAccumulatorParameterValues(settings, chunkSliceCount)</code>, writes <code>GPU_RASTER_CLIMATE_RUN_ACCUMULATOR_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li><code>chunkSliceCount</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterClimateRunAccumulatorParameterValues</code>:</p>
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
<td><code>reset</code></td>
<td><code>boolean</code></td>
<td>-</td>
</tr>
<tr>
<td><code>activeSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>maximumIncrement</code></td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>maximumStreamSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterclimateanomaly"><code>GPURasterClimateAnomaly</code></h2>
<p>Reads a <code>GPURasterStreamingAccumulator</code>'s state: per-slot mean and sample standard deviation, and the anomaly
(or z-score) of a chunk against them.</p>
<p>{/* parameters: GPURasterClimateAnomaly generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterclimateanomaly-parameters"><code>GPURasterClimateAnomaly</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterStreamingAccumulatorParameterValues(settings, chunkSliceCount)</code>, writes <code>GPU_RASTER_STREAMING_ACCUMULATOR_PARAMETER_LENGTH</code> (5) <code>float32</code> values.
<ul>
<li>Input port: <code>accumulatorParameters</code>.</li>
<li><code>chunkSliceCount</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPURasterStreamingAccumulatorParameterValues</code>:</p>
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
<td><code>reset</code></td>
<td><code>boolean</code></td>
<td>True on the first chunk of a stream, which clears the state first.</td>
</tr>
<tr>
<td><code>activeSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Slices of this chunk that hold data; defaults to the whole chunk.</td>
</tr>
<tr>
<td><code>maximumValueMagnitude</code></td>
<td><code>number</code></td>
<td>Largest absolute sample the stream can hold. It sets the fixed-point unit of the sums: declare it tightly, because a sample above it is treated as missing, and the unit is about <code>maximumValueMagnitude * 2^-70</code> at the default stream length.</td>
</tr>
<tr>
<td><code>maximumStreamSliceCount</code> (optional)</td>
<td><code>number</code></td>
<td>Longest stream in slices, for the fixed-point headroom. Defaults to 2^24.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};