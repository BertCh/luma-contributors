var e={slug:`api/gpu-raster/focal-pairs`,title:`Focal cross-band statistics`,html:`<h1 id="focal-cross-band-statistics">Focal cross-band statistics</h1>
<p><code>GPURasterFocalPairs</code> computes unweighted covariance, Pearson correlation and simple ordinary
least-squares regression in square raster neighborhoods. The regression is
<code>response = intercept + slope * predictor</code>; its fit statistics are descriptive. There are no
p-values, uncertainty estimates, weights, multiple predictors or geographic wraparound.</p>
<h2 id="inputs-and-outputs">Inputs and outputs</h2>
<p>Provide <code>grid</code>, aligned <code>predictor</code> and <code>response</code> raster bands, <code>radius</code>, and caller-owned
<code>output</code> views. Both bands must already occupy the same grid. Buffer and texture inputs use
existing raster canonicalization: native nodata and validity are checked before scale/offset,
and nonfinite calibrated values are invalid. Only observations valid in <strong>both</strong> bands enter
any statistic. Values and masks can change between graph executions without recompilation;
window options and calibration metadata are graph topology.</p>
<p><code>radius</code> is an integer from 0 through 8, bounding the footprint to 17 × 17 samples. Radius zero
is permitted but cannot meet the minimum of two pairs. <code>minimumCount</code> defaults to 2 and may
range from 2 through 289. <code>covarianceNormalization</code> defaults to <code>'sample'</code> (divide by <code>n - 1</code>);
<code>'population'</code> divides by <code>n</code>. The count threshold applies to all float metrics.</p>
<p>Every output view contains at least <code>grid.width * grid.height</code> packed rows. <code>pairedCount</code> and
<code>status</code> are required uint32 outputs. Request at least one float32 metric:</p>
<table>
<thead>
<tr>
<th>Output</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>covariance</code></td>
<td>Centered cross-product sum divided by the selected count denominator</td>
</tr>
<tr>
<td><code>correlation</code></td>
<td>Pearson correlation, undefined when either band is constant</td>
</tr>
<tr>
<td><code>slope</code>, <code>intercept</code></td>
<td>Simple OLS coefficients, undefined when the predictor is constant</td>
</tr>
<tr>
<td><code>rSquared</code></td>
<td><code>1 - residualSumSquares / responseCenteredSumSquares</code>, undefined for a constant response</td>
</tr>
<tr>
<td><code>residualSumSquares</code></td>
<td>Sum of squared centered fitted residuals</td>
</tr>
</tbody>
</table>
<p>An undefined float result is NaN. No output buffer may alias an input or another output.
Destroying the compiled graph releases its scratch buffers and leaves borrowed inputs and
outputs owned by the caller.</p>
<h2 id="missing-pairs-and-borders">Missing pairs and borders</h2>
<p><code>noDataPolicy: 'ignore'</code> (default) drops a pair when either band is invalid.
<code>'propagate'</code> makes every float result undefined if any footprint position is invalid.
<code>propagateCenterNoData: true</code> additionally invalidates results when either center observation is
invalid; the default is false. <code>pairedCount</code> is always the number of participating valid pairs,
even when propagation or insufficient support makes all metrics undefined.</p>
<p><code>borderMode</code> defaults to <code>'nodata'</code>: positions outside the grid are invalid, so <code>'ignore'</code>
clips the window and <code>'propagate'</code> invalidates edge windows. <code>'clamp'</code> repeats the nearest edge
sample. <code>'reflect'</code> reflects without duplicating the edge pixel, including repeated reflection
for tiny rasters; a one-cell axis maps every offset to that cell. Repeated clamp/reflect samples
count as repeated observations.</p>
<p>The contributor advertises <code>requiredHalo = radius</code>. For tiles, load both bands and masks over
the full halo-expanded footprint, execute there, and retain only each tile's core outputs.
Apply border conventions at the true domain boundary. Applying them at an interior tile edge
changes the neighborhood and its observation count.</p>
<h2 id="diagnostic-bits-and-degeneracy">Diagnostic bits and degeneracy</h2>
<p><code>GPU_RASTER_FOCAL_PAIRS_STATUS</code> defines combinable per-cell bits (documented on <code>GPURasterFocalPairsOutput.status</code>; the radius limit is <code>MAXIMUM_FOCAL_PAIRS_RADIUS</code>, 8, and <code>minimumCount</code> is at most <code>MAXIMUM_FOCAL_PAIRS_WINDOW_CELLS</code>, 289):</p>
<table>
<thead>
<tr>
<th>Bit</th>
<th>Name</th>
<th>Consequence</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><code>insufficientPairs</code></td>
<td>Fewer than <code>minimumCount</code> pairs; all metrics NaN</td>
</tr>
<tr>
<td>2</td>
<td><code>missingPair</code></td>
<td>Required neighborhood or center invalidity; all metrics NaN</td>
</tr>
<tr>
<td>4</td>
<td><code>constantPredictor</code></td>
<td>Zero predictor centered sum of squares; covariance remains zero, regression/correlation/R² are NaN</td>
</tr>
<tr>
<td>8</td>
<td><code>constantResponse</code></td>
<td>Zero response centered sum of squares; covariance remains zero and correlation/R² are NaN</td>
</tr>
<tr>
<td>16</td>
<td><code>nonFiniteResult</code></td>
<td>Moment/fit overflow or nonfinite arithmetic; all metrics NaN</td>
</tr>
</tbody>
</table>
<p>With a varying predictor and constant response, the fit is valid: slope zero, intercept equal
to the response, and residual sum of squares zero. Status 8 records the undefined correlation
and R². When both bands are constant, bits 4 and 8 are set. Constant flags are evaluated only
once count and missing-pair checks permit a fit. Zero status means every metric is defined.</p>
<h2 id="numerical-method-and-limits">Numerical method and limits</h2>
<p>The GPU scans each footprint in fixed row-major order. A first pass accumulates differences
from the first valid pair, a second accumulates centered moments, and a third evaluates fitted
residuals directly. Each reduction corrects the remaining residual mean using only the small
centered values. This preserves centering when a shader backend reassociates subtraction into
a rounded large centroid. It avoids subtracting raw squared moments and avoids deriving
residual squares by subtracting explained variance from total variance. Pearson correlation and R² are clamped to their
mathematical ranges to absorb float32 rounding.</p>
<p>Arithmetic and calibration are float32. Input differences lost to float32 rounding cannot be
recovered; squared sums can underflow to zero or overflow. Zero centered predictor variance is
treated as singular, with no implicit ridge or pseudoinverse. No near-singularity threshold or
cross-device error guarantee beyond the verified fixtures is claimed.</p>
<p>Canonicalization schedules one pass per band; moments and output projection add two passes.
Moment scratch uses six float32 values per pixel (24 bytes), plus canonical values and validity
for each band (16 bytes per pixel). The moments pass uses five storage bindings and maximal
output projection uses seven, within the default eight-binding limit. Work is bounded by three
17 × 17 scans per cell at the maximum radius. Large-domain performance is not yet benchmarked.</p>
<h2 id="reference-and-verification">Reference and verification</h2>
<p>The generator at <code>modules/analysis/test/oracles/raster-focal-pairs.py</code> pins <strong>NumPy 2.5.4</strong>
and records its SHA-256 in <code>fixtures/raster-focal-pairs.json</code>. It uses independent <code>numpy.cov</code>,
<code>numpy.corrcoef</code> and centered <code>numpy.linalg.lstsq</code> calls after explicit window extraction and
paired mask/calibration adaptation. Reference inputs are float32-rounded and the library
computations use float64. This original fixture is a composed NumPy reference.</p>
<p>The additional <code>modules/analysis/test/oracles/raster-terra.R</code> generator directly executes <strong>terra 1.9.50</strong> with <strong>R 4.4.3</strong> and jsonlite 2.0.0. Its <code>raster-terra.json</code> fixture compares GPU covariance, Pearson correlation, intercept, slope and paired counts for six nodata/clamp/reflect × ignore/propagate cases. Clamp/reflect use explicit padded input rasters before calling terra, then crop the core; nodata uses native terra fillvalue=NA. The generator also checks temporal stack composites. All six focal cases passed native GPU comparison.</p>
<p>The supported unweighted subset corresponds to
<a href="https://rspatial.github.io/terra/reference/focalPairs.html">terra focalPairs</a> with Pearson
correlation and paired NA removal, and
<a href="https://rspatial.github.io/terra/reference/focalReg.html">terra focalReg</a> with an intercept and
one predictor. In terra's layer order, response precedes predictor. Clamp/reflect and the
explicit singular-fit flags are this API's documented conventions.</p>
<p>Fixtures cover positive/negative and noisy fits, population/sample covariance, separate band
masks, calibration, native nodata, insufficient pairs, constant bands, large offsets, tiny
rasters, large-offset fractional means and maximum radius. GPU tests also cover changing inputs/masks, nonzero buffer offsets,
overflow, integer inputs, ownership and whole-domain versus halo-expanded tile results.</p>
<h2 id="executable-examples">Executable examples</h2>
<p>These examples use explicit synthetic inputs and expose real GPU graph captures.</p>
<h3 id="gpurasterfocalpairs">GPURasterFocalPairs</h3>
<p>{/* lens:GPURasterFocalPairs */}</p>
<p>::lens{scenario=gpu-raster.raster-focal-pairs rung=graph}</p>`};export{e as default};