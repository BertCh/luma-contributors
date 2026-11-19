var e={slug:`api/gpu-spatial-analysis/dasymetric-redistribution`,title:`Dasymetric redistribution`,html:`<h1 id="dasymetric-redistribution">Dasymetric redistribution</h1>
<p><code>addDasymetricRedistributionRecipe</code> redistributes nonnegative source totals using fractional
source-polygon coverage and an ancillary raster. Targets own whole raster cells through a
<code>uint32</code> target-zone raster. The recipe runs coverage, calibration, eligibility, weighted areal
interpolation, transfer and mass accounting on the GPU. It performs no readback or CPU fallback.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {addDasymetricRedistributionRecipe} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> result</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addDasymetricRedistributionRecipe</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(graph, {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  grid,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  source: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'coverage'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, coverage},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  sourceTotals,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  targetZones,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  targetZoneCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  ancillary: {id: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'density'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, storage: {kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'buffer'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, values: density}},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mask: eligibleCells,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  denominator: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'overlap'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pairCapacity: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">32</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p><code>source: {kind: 'polygons', zones, tupleCapacity, parameters}</code> also constructs fractional coverage from
packed planar polygons; <code>parameters</code> is the <code>getGPURasterFractionalCoverageParameterValues(grid)</code> view. <code>sourceTotals.length</code> is the source-feature count. Feature rows and
target IDs remain stable, including empty sources and targets. Prepared coverage uses the
<code>GPURasterFractionalCoverage</code> output layout and retains its original status views.
This includes optional <code>candidateOverflow</code>, independently of final tuple <code>overflow</code>. Prepared
coverage is usable only with matching in-capacity count/requiredCount, both supplied overflow
flags zero, and valid source zones. Candidate-only loss suppresses the mass ledger and sets
<code>status.complete = 0</code>; increasing final pair or tuple capacity alone cannot repair candidate loss.
Each prepared-coverage status view must contain at least one word. The recipe reads its first
logical word, including for offset views, and ignores any spare words.</p>
<h2 id="measure-and-conservation">Measure and conservation</h2>
<p>A valid source/cell tuple contributes
<code>fraction * calibratedAncillaryValue * abs(affineDeterminant)</code> to its source's eligible measure.
The target-zone raster assigns that measure to at most one target. An ID at or above
<code>targetZoneCount</code> means uncovered support. The recipe does <strong>not</strong> multiply independent source
and target fractions or claim arbitrary fractional target-polygon intersection.</p>
<p>The optional mask is an eligibility decision: zero excludes the cell. Ancillary zero also
excludes land. Native nodata and band validity are resolved before calibration. Missing,
nonfinite or negative calibrated ancillary values contribute no eligible measure; their
<code>intermediates.cellWeights</code> are NaN, while explicitly excluded cells have zero weight.
Ancillary weights describe relative density per square affine-coordinate unit; a binary
eligibility raster is a special case. Geographic/spherical cell-area weighting is not provided.</p>
<p>For a source with total <code>M</code>, eligible measure <code>A</code> and target-covered measure <code>C</code>:</p>
<table>
<thead>
<tr>
<th>Denominator</th>
<th>Allocated source total</th>
<th>Unallocated source total</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>overlap</code> (default), <code>C > 0</code></td>
<td><code>M</code></td>
<td><code>0</code></td>
</tr>
<tr>
<td><code>zone</code>, <code>A > 0</code></td>
<td><code>M * C / A</code></td>
<td><code>M * (1 - C / A)</code></td>
</tr>
<tr>
<td>No eligible target support</td>
<td><code>0</code></td>
<td><code>M</code></td>
</tr>
</tbody>
</table>
<p>Zero eligible measure is valid accounting, with <code>sourceValidity = 0</code> and the whole original
mass in <code>sourceUnallocatedTotals</code>. Its transferred fraction is NaN because the denominator
is undefined. Targets with no allocated source support have zero totals and support weights.
Source features can overlap: their masses remain independent. <code>targetSupportWeights</code> adds
source measures; it is not the geometric union area or a fraction bounded by one.</p>
<p>The ledger preserves the existing areal-interpolation float32 results. It does not clamp
fractions or negative residuals caused by roundoff. Verification uses absolute <code>1e-4</code> plus
relative <code>1e-5</code> tolerance against independently computed values, and the same tolerance scaled
by total source mass for conservation. A small negative residual within this tolerance is
numerical roundoff, not physical negative mass. This tested tolerance is not a universal error
bound for arbitrarily ill-conditioned workloads; callers should inspect their mass residuals.</p>
<h2 id="outputs-and-evidence">Outputs and evidence</h2>
<ul>
<li><code>targetTotals</code>, <code>targetSupportWeights</code>: one row per target.</li>
<li><code>sourceEligibleWeights</code>, <code>sourceTransferredFractions</code>, <code>sourceAllocatedTotals</code>,
<code>sourceUnallocatedTotals</code>, <code>sourceValidity</code>: one row per original source.</li>
<li><code>status.complete</code>: execution evidence is complete and published arithmetic is finite.</li>
<li><code>status.zeroEligibleCount</code>: sources with zero eligible measure; these retain their totals.</li>
<li><code>status.invalidTupleCount</code>, <code>status.invalidSourceCount</code>, <code>status.arithmeticInvalidCount</code>:
malformed row values, invalid extensive totals and arithmetic failures, respectively.</li>
<li><code>status.stages</code>: original coverage count/required-count/overflow, coverage validation,
areal pair overflow/required count, source-value, arithmetic and conservation evidence.</li>
</ul>
<p>Coverage failure, invalid source totals or tuple values, truncated pair output, and detected
float32 arithmetic overflow set <code>complete = 0</code> and invalidate published numeric outputs with
NaN. This includes a finite ancillary value times finite cell area overflowing float32, source
support sums overflowing, and target support or mass sums overflowing. <code>sourceValidity</code> becomes
zero. Coverage status and the caller's input buffers remain available for diagnosis.</p>
<p>The intermediate weights port shares the areal stage's pair-overflow and required-count views.
The producer rewrites them on every submission before transfer, so complete support restores
usable outputs after an earlier overflow.</p>
<p><code>intermediates</code> retains the original grid, coverage, ancillary band, mask, source-total and
target-zone views, plus the derived cell weights, expanded tuple identities and weights, and
interpolation CSR. All output overrides must be distinct from one another and from inputs.</p>
<h2 id="prepared-partitions-and-bounds">Prepared partitions and bounds</h2>
<p>Prepared coverage must contain <strong>unique <code>(source, cell)</code> tuples</strong>, with fractions in <code>[0, 1]</code>
and valid IDs. Uniqueness and source geometry topology are caller preconditions;
<code>status.complete</code> does not certify them and the recipe does not deduplicate malformed input.
Generated coverage supplies one row per source/cell. Halo cells must not be repeated.</p>
<p>Concatenate disjoint partition cores, remapping cell IDs to the common grid, <strong>before</strong>
normalization. A source can span several cores; independently normalized tile allocations
cannot be added. The tests reorder two disjoint core partitions and compare the result with
the complete-domain oracle. This is prepared-partition composition, not automatic tile IO or
cross-tile geometry stitching.</p>
<p><code>targetZoneCount * tupleCapacity</code> is bounded by 4,194,304 for the explicit support summary.
Polygon coverage also retains its own 4,194,304 source/cell candidate limit. Weighted areal
interpolation's device storage and pair-count limits still apply. The grid determinant must
be finite, positive in magnitude and representable as nonzero float32. These are shape limits,
not performance claims.</p>
<h2 id="reference-evidence">Reference evidence</h2>
<p>The generator <code>modules/analysis/test/oracles/dasymetric-redistribution.py</code> records package
versions, units, conventions and tolerances in its fixture. Fractional-boundary and continuous
weight cases use a <strong>composed oracle</strong>: Shapely/GEOS source/cell intersections followed by an
independent weighted extensive-allocation calculation. Binary, aligned-cell cases additionally
call <a href="https://pysal.org/tobler/stable/generated/tobler.dasymetric.masked_area_interpolate.html">Tobler's <code>masked_area_interpolate</code></a>
on a temporary GeoTIFF, with <code>allocate_total</code> selected explicitly. They do not imply full Tobler
option parity. Tobler's raster extraction has different partial-boundary behavior, so partial
source-cell cases are labeled composed rather than direct Tobler parity.</p>
<p>Pinned direct dependencies are Tobler 0.14.0, Shapely 2.1.2, GeoPandas 1.1.1, Rasterio 1.4.3 and
NumPy 2.2.6; additional runtime versions are recorded in the generated JSON. GPU tests cover
both denominator modes, empty support, excluded and missing land, source polygon composition,
prepared partitions, changes to masks/values/target ownership, failure recovery, truncation,
skewed measures and arithmetic overflow. Node construction tests cover ownership and default
storage-binding limits.</p>
<h2 id="executable-example">Executable example</h2>
<p>{/* lens:addDasymetricRedistributionRecipe */}</p>
<p>The example retains the source mass ledger and the actual eligibility and completion captures.</p>
<p>::lens{scenario=gpu-spatial-analysis.dasymetric-redistribution-recipe rung=graph}</p>`};export{e as default};