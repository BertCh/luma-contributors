var e={slug:`api/gpu-raster/fractional-zonal-distribution`,title:`GPURasterFractionalZonalDistribution`,html:`<h1 id="gpurasterfractionalzonaldistribution">GPURasterFractionalZonalDistribution</h1>
<p>Computes coverage-weighted quantiles, population and reliability sample variance, and effective
sample support from the <code>(zoneId, cellId, fraction)</code> output of <code>GPURasterFractionalCoverage</code>.
The value band contains intensive cell observations. Statistical weights are coverage fractions times optional independent per-cell observation weights. Optional per-cell areas remain separate from those weights; there is no implicit conversion of cell totals into densities.</p>
<p>This is RS01's bounded planar distribution contract. All computation, sorting and grouping stays
inside the command graph. Construction does not submit GPU work or read data back to the CPU.
The input buffers and all outputs belong to the caller.</p>
<h2 id="inputs-and-shapes">Inputs and shapes</h2>
<table>
<thead>
<tr>
<th>Property</th>
<th>Contract</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>grid</code></td>
<td>Raster dimensions and an invertible planar affine transform; area is the absolute affine determinant. The determinant and the maximum covered area must be representable in float32.</td>
</tr>
<tr>
<td><code>input</code></td>
<td>A buffer raster band. Existing band canonicalization applies calibration, masks, nodata and finite-value filtering.</td>
</tr>
<tr>
<td><code>zoneCount</code></td>
<td>Dense source-zone rows, from 0 through 65,536. Empty zones retain their row.</td>
</tr>
<tr>
<td><code>coverage</code></td>
<td>The complete existing fractional-coverage output object: <code>zoneIds</code>, <code>cellIds</code>, <code>fractions</code>, <code>zoneValidity</code>, and <code>status.count</code>, <code>status.requiredCount</code>, <code>status.overflow</code>. Tuple capacity is 1 through 65,536; the active count may be zero.</td>
</tr>
<tr>
<td><code>probabilities</code></td>
<td>A float32 view of 1 through 32 probabilities. Values may change between encodings. Each must be finite and in <code>[0,1]</code>; invalid probabilities yield NaN and zero quantile validity.</td>
</tr>
<tr>
<td><code>observationWeights</code></td>
<td>Optional packed float32 cell column. Finite nonnegative reliability weights; zero excludes a cell from distribution statistics but retains its valid area. Negative/nonfinite weights invalidate the zone. Default one.</td>
</tr>
<tr>
<td><code>cellAreas</code></td>
<td>Optional packed float32 cell column. Positive finite areas supplied by the caller, including geographic/geodesic areas. Default absolute planar affine determinant, or ground area with <code>cellSizeMode</code>. Areas never silently become statistical weights.</td>
</tr>
<tr>
<td><code>cellSizeMode</code></td>
<td><code>'uniform'</code> (default, planar coordinate units), <code>'geographic'</code> (degrees) or <code>'web-mercator'</code> (EPSG:3857 meters). The latter two multiply the default cell area by the WGS84 ground area per world area of the cell's row (row edges evaluated at the row center), so <code>coveredArea</code> and <code>validArea</code> are ground square meters. The row edges are read from <code>parameters</code> (slots 1 and 5) on every encoding when it is given, else baked from <code>grid.affine</code>. Needs an axis-aligned grid; rejected together with <code>cellAreas</code>. The mode is compile-time.</td>
</tr>
<tr>
<td><code>parameters</code></td>
<td>Optional. The per-frame georeference view given to <code>GPURasterFractionalCoverage</code> (<code>getGPURasterFractionalCoverageParameterValues</code>). Without <code>cellAreas</code>, the cell area is then read from it on every encoding as the absolute determinant, so the area outputs follow a georeference repacked for another cell size. With neither <code>cellAreas</code> nor <code>parameters</code> the cell area is the determinant of <code>grid.affine</code> fixed at compile time.</td>
</tr>
<tr>
<td><code>quantileMethod</code></td>
<td><code>'exactextract'</code> (default) distinct-value interpolation, or <code>'inverse-cdf'</code> without interpolation. Both sort the exact observed float32 values; neither bins a continuous range.</td>
</tr>
<tr>
<td><code>output</code></td>
<td>Caller-owned, nonaliasing buffers described below.</td>
</tr>
</tbody>
</table>
<p>Tuple buffers have matching capacities and only their active prefix is read. Fractions must be
finite and in <code>[0,1]</code>; zero contributes no observation. A <code>(zoneId, cellId)</code> pair must occur once.
Duplicates are rejected, including a duplicate carrying zero weight. Negative/nonfinite observation weights, nonpositive/nonfinite cell areas, or overflowing combined weights also invalidate the zone. Different zones may refer to
the same cell. Invalid source-cell IDs invalidate their zone; an out-of-range zone ID invalidates
all zones because its intended destination cannot be recovered.</p>
<p>The input band, fractions, IDs, probabilities, counts, validity and overflow words are read again on
every encoding. There is no prepared cache to invalidate. Sizes, calibration and the grid affine
are construction-time settings, except the cell area when <code>parameters</code> is supplied (see above):
pass it whenever the coverage georeference is rewritten between encodings.
Prepared coverage may also supply <code>status.candidateOverflow</code>; both operation forms preserve this
optional input and suppress dependent results when it is nonzero. <code>requiredCount</code> must equal
<code>count</code>, which must fit the tuple capacity. Coverage status inputs (<code>count</code>, <code>requiredCount</code>,
<code>overflow</code> and optional <code>candidateOverflow</code>) must each contain at least one word. Only the first
logical word is read, including for offset views; spare words do not affect completeness.
Ground modes check both packed cross-axis words on every run, suppressing all affected results
when nonzero; a later aligned frame restores validity.</p>
<h2 id="outputs-and-definitions">Outputs and definitions</h2>
<p>For valid cells of one zone, let <code>w</code> be coverage fraction times observation weight, <code>W = sum(w)</code>, <code>W2 = sum(w*w)</code>,
<code>mean = sum(w*x)/W</code>, and <code>M2 = sum(w*(x-mean)^2)</code>.</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Shape</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>means</code></td>
<td><code>zoneCount</code> float32</td>
<td>Coverage-weighted mean.</td>
</tr>
<tr>
<td><code>populationVariances</code></td>
<td><code>zoneCount</code> float32</td>
<td><code>M2/W</code>; zero for singleton or constant observed values.</td>
</tr>
<tr>
<td><code>sampleVariances</code></td>
<td><code>zoneCount</code> float32</td>
<td>Reliability-weighted correction <code>M2/(W-W2/W)</code>. NaN with fewer than two observed source cells or a nonpositive denominator. Fractions are not frequency counts.</td>
</tr>
<tr>
<td><code>effectiveSampleSizes</code></td>
<td><code>zoneCount</code> float32</td>
<td>Kish effective support <code>W*W/W2</code>; zero without observations. This measures weight concentration, not independent spatial information or inferential degrees of freedom.</td>
</tr>
<tr>
<td><code>weightSums</code>, <code>squaredWeightSums</code></td>
<td><code>zoneCount</code> float32 each</td>
<td><code>W</code> and <code>W2</code>, computed from individual source cells before equal values are grouped.</td>
</tr>
<tr>
<td><code>coveredArea</code>, <code>validArea</code></td>
<td><code>zoneCount</code> float32 each</td>
<td>Sum of fraction times the corresponding cell area, over all accepted cells or only finite unmasked values. Zero observation weight does not remove valid area. Nodata remains visible as missing support.</td>
</tr>
<tr>
<td><code>quantiles</code></td>
<td><code>zoneCount * probabilities.length</code> float32</td>
<td>Zone-major quantiles: <code>zone * probabilities.length + probabilityIndex</code>.</td>
</tr>
<tr>
<td><code>quantileValidity</code></td>
<td>Same shape, uint32</td>
<td>One for a valid finite quantile, otherwise zero.</td>
</tr>
<tr>
<td><code>validity</code></td>
<td><code>zoneCount</code> uint32</td>
<td>One for a complete zone with supported finite mean/population results. Sample variance additionally needs sufficient sample support.</td>
</tr>
<tr>
<td><code>status</code></td>
<td><code>zoneCount</code> uint32</td>
<td>Bit mask below; overwritten on every encoding.</td>
</tr>
</tbody>
</table>
<p>A complete empty/all-nodata zone has zero support and area sums as appropriate, NaN distribution
statistics and zero validity. Incomplete or malformed coverage yields NaN numeric outputs and zero
validity, so a truncated set of tuples cannot appear to be a complete distribution.</p>
<h3 id="quantile-convention">Quantile convention</h3>
<p>By default quantiles match the <strong>exactextract 0.3.0 distinct-value histogram convention</strong>. Equal finite values
are grouped and their combined coverage/observation weights added. For <code>m</code> sorted distinct values, group weight <code>g[j]</code>,
and cumulative weight strictly before group <code>j</code>, plotting positions are
<code>p[j] = j*g[j] + (m-1)*cumulativeWeightBefore[j]</code>. The requested position is <code>q*(m-1)*W</code>.
Linear interpolation occurs between adjacent plotting positions. A singleton distinct value is
returned for every valid probability; endpoints return the observed minimum and maximum.</p>
<p>This convention differs from an inverse empirical CDF and from applying unweighted type-7
quantiles to replicated cells. Grouping equal values affects quantiles, but does <strong>not</strong> collapse
source cells when calculating <code>W2</code>, sample variance or effective support. The reference generator
calls the external library directly rather than reimplementing its quantile algorithm.</p>
<p><code>quantileMethod: 'inverse-cdf'</code> selects the first distinct value whose cumulative positive weight is at least <code>q*W</code>; <code>q=0</code> selects the minimum. This matches NumPy 2.2.6 <code>quantile(method='inverted_cdf', weights=...)</code>. Zero-weight observations are absent from either distribution. Direct exactextract <code>weighted_mean</code> / <code>weighted_variance</code>, NumPy weighted quantiles and <code>cov(aweights=...)</code> are executed by <code>raster-round-two.py</code>.</p>
<h3 id="status-bits">Status bits</h3>
<p><code>GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_STATUS</code> exports these flags:</p>
<table>
<thead>
<tr>
<th>Bit</th>
<th>Name</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><code>INCOMPLETE_COVERAGE</code></td>
<td>Upstream tuple or candidate overflow, inconsistent active/required count, failed source-zone validity, or a non-axis-aligned effective ground-area frame.</td>
</tr>
<tr>
<td>2</td>
<td><code>INVALID_TUPLE</code></td>
<td>Invalid fraction, cell or zone ID, or duplicate cell within a zone.</td>
</tr>
<tr>
<td>4</td>
<td><code>NO_VALID_SUPPORT</code></td>
<td>No positive-weight finite observations.</td>
</tr>
<tr>
<td>8</td>
<td><code>INSUFFICIENT_SAMPLE_SUPPORT</code></td>
<td>Fewer than two observed source cells or a nonpositive sample-variance denominator. A valid singleton still has valid population results.</td>
</tr>
<tr>
<td>16</td>
<td><code>NON_FINITE_RESULT</code></td>
<td>Float32 arithmetic exceeded finite moment/support range.</td>
</tr>
</tbody>
</table>
<p>Invalid probabilities are represented by <code>quantileValidity</code>, independently of valid moment outputs.</p>
<h2 id="example">Example</h2>
<p>The output buffers are allocated by the application. <code>coverage</code> is the output of an earlier
<code>GPURasterFractionalCoverage</code> producer in the same graph.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPURasterFractionalZonalDistribution} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// probabilities contains [0.25, 0.5, 0.75]. Quantile outputs need zoneCount * 3 rows.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterFractionalZonalDistribution</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  input: observationBand,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  zoneCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  coverage,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  probabilities,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    means,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    populationVariances,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    sampleVariances,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    effectiveSampleSizes,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    weightSums,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    squaredWeightSums,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    coveredArea,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    validArea,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    quantiles,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    quantileValidity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    validity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    status</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h2 id="partition-and-numerical-limits">Partition and numerical limits</h2>
<p>For tiled inputs, prepare an explicit tuple union with disjoint cell cores and cell IDs rebased to
the explicitly assembled value band. Exclude halos. This operation does not merge source chunks
or tile summaries. Never average tile quantiles: the combined weighted histogram is required.
Repeated source cells are rejected to protect support and area accounting.</p>
<p>GPU radix sorting produces deterministic <code>(zone,value,cell)</code> order; moments use a centered,
two-pass calculation to avoid subtracting two large raw second moments. A separate GPU pass
materializes centered observations before reduction, preventing backend arithmetic reassociation
from restoring a large baseline inside the residual calculation. Reductions are float32,
so extreme dynamic ranges can lose small contributions. A nonpositive rounded sample denominator
is reported as insufficient support, and non-finite moment results set bit 16.</p>
<p>The tuple/zone/probability bounds are implementation limits, not a measured throughput promise.
Per-zone moment scans stripe sums across 256 local partials and reduce a fixed tree to limit float32 accumulation drift; the scan itself is serial. Each quantile scans its zone's exact distinct-value histogram. A large skewed
zone with 32 quantiles can dominate execution. Every pass respects the default eight-storage-buffer
limit; sort scratch is graph-owned and proportional to tuple capacity. There is no claim of a
browser speed advantage or production-scale envelope until dedicated GPU measurements are recorded.</p>
<h2 id="reference-and-verification">Reference and verification</h2>
<p>The pinned generator is
<code>modules/analysis/test/oracles/fractional-zonal-distribution.py</code>, using Python 3.12,
exactextract 0.3.0, NumPy 2.3.4 and Shapely 2.1.2. Direct <code>exact_extract</code> calls provide quantiles,
mean, population variance and coverage count. <code>numpy.cov(aweights=coverage, ddof=1)</code> independently
provides sample variance; sum/dot composition provides effective support. Geometry fixtures use
exactly representable planar strips and a partial polygon with a hole.</p>
<p>Tests cover repeated values, unequal weights, singleton/empty/all-nodata zones, tiny positive
support, large offsets with small variance, malformed/duplicate tuples, capacity exhaustion,
source triangulation failure, per-frame updates, overlapping zones and explicit disjoint tile
composition. Node kernel audits and numerical GPU parity are separate checks.</p>
<p>References: <a href="https://isciences.github.io/exactextract/operations.html">exactextract operations</a>,
<a href="https://github.com/isciences/exactextract/blob/v0.3.0/src/raster_stats.h">version-pinned histogram integration</a>,
<a href="https://github.com/isciences/exactextract/blob/v0.3.0/src/weighted_quantiles.cpp">version-pinned quantile implementation</a>,
<a href="https://pypi.org/project/exactextract/0.3.0/">exactextract 0.3.0</a>.</p>
<h2 id="executable-examples">Executable examples</h2>
<p>These examples use explicit synthetic inputs and expose real GPU graph captures.</p>
<h3 id="gpurasterfractionalzonaldistribution-1">GPURasterFractionalZonalDistribution</h3>
<p>{/* lens:GPURasterFractionalZonalDistribution */}</p>
<p>::lens{scenario=gpu-raster.raster-fractional-zonal-distribution rung=graph}</p>`};export{e as default};