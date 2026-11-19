var e={slug:`api/gpu-spatial-analysis/geographical-detector-redcap-entropy`,title:`Geographical detector, REDCAP, spatial entropy and the contiguity-order correlogram`,html:`<h1 id="geographical-detector-redcap-spatial-entropy-and-the-contiguity-order-correlogram">Geographical detector, REDCAP, spatial entropy and the contiguity-order correlogram</h1>
<p>STX-08, first group. Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. Each method names its reference and what was
run against it. Where a reference package could not be installed, a binary64 CPU evaluation of the published definition stands in, and the section says so.</p>
<h2 id="geographical-detector-gpugeographicaldetector">Geographical detector: <code>GPUGeographicalDetector</code></h2>
<p>The factor detector of Wang et al. (2010): <code>q = 1 - SSW / SST</code>, the share of the variance of a response <code>values</code> explained by a
stratification <code>labels</code> (a classification, an administrative code, a zoning). It reads the sums of squares, strata sizes and strata means
from <code>GPURegionPartitionEvaluation</code> (which gained an optional <code>output.regionMeans</code> for this), then reduces the strata means in one tiled pass.
No float atomics; results are bitwise reproducible.</p>
<p><code>output.summary</code> is laid out by <code>GPU_GEOGRAPHICAL_DETECTOR_LAYOUT</code>: <code>q</code>, <code>fStatistic</code>
<code>F = ((N - L) / (L - 1)) q / (1 - q)</code>, <code>nonCentrality</code>, <code>strataCount</code> <code>L</code>, <code>observationCount</code> <code>N</code>, the two degrees of freedom and
<code>totalSumOfSquares</code>. Rows labelled <code>0xffffffff</code> or at or above <code>labelCapacity</code>, and rows whose value is NaN or infinite, are left out. A constant response gives NaN.</p>
<p>The p-value is computed on the host, from the read-back summary: <code>getGeographicalDetectorPValue(summary)</code> is the upper tail of the
non-central F distribution, summed as a Poisson mixture of incomplete beta functions in binary64 (<code>getNonCentralFUpperTail</code> is the
distribution tail itself). Both beta arguments are formed directly, retaining the smaller argument
even when the other rounds to one. Stable log-beta normalization handles disparate degrees of
freedom; weighted survival terms are accumulated in log space and rounded only after summation,
including subnormal tails. For finite positive <code>F</code>, supported non-centrality is at most 10000;
larger or infinite non-centrality throws an error. With valid positive degrees of freedom and finite
non-centrality, <code>F = Infinity</code> returns zero and <code>F &#x3C;= 0</code> returns one before applying that mixture
limit. This includes a perfect detector (<code>q = 1</code>) with non-centrality above 10000.
NaN statistics return NaN. Each mixture direction
and beta continued fraction is bounded by 100000 iterations; a convergence failure throws an error.
It is not computed on the GPU.</p>
<p>For consumers that can display a summary without inference, <code>readGeographicalDetectorPValue(summary)</code>
returns <code>{pValue, unavailableReason}</code>. A supported result has a numeric <code>pValue</code> and null reason;
an undefined statistic, unsupported range or numerical convergence failure has null <code>pValue</code> and
an explanatory reason. Unrelated read errors still throw. The strict helper above keeps its
throwing contract. Kepler preserves all summary rows, publishes null in the <code>pValue</code> row,
sets <code>summary.pValueStatus = 'unavailable'</code> and reports a
<code>geographical-detector-p-value-unavailable</code> warning with the reason. Supported probabilities have
<code>pValueStatus = 'available'</code>. The website readout likewise retains q and F and labels an
unavailable probability with its reason. For example, strata <code>[A,B,B,B]</code> with responses
<code>[1000,1001,1002,1003]</code> retain <code>q = 0.6</code> and <code>F = 3</code> even though their non-centrality exceeds 10000.</p>
<p>Reference: <code>geodetector</code> 0.2.0 (MIT) <code>q_statistic</code> and <code>q_significance_test</code>, whose formulas match the R <code>GD</code> package; its <code>_stats.py</code> ran in
<code>stx08-spatial-methods.py</code>. Differences from the reference, stated as they are:</p>
<ul>
<li>The package drops strata with a single row. This contributor keeps them (they add nothing to SSW but count in <code>L</code> and <code>N</code>). Compare on data whose strata have at least two rows.</li>
<li><code>nonCentrality</code> is the published formula <code>(sum_h mean_h^2 - (sum_h sqrt(N_h) mean_h)^2 / N) / sigma^2</code> with the population variance. It is not invariant to shifting <code>values</code>, and it differs from the textbook <code>sum_h N_h (mean_h - mean)^2 / sigma^2</code>. It was implemented as the reference computes it, not corrected.</li>
<li>float32 accuracy: <code>q</code> agrees to about <code>2e-5</code> in the fixture. <code>nonCentrality</code> subtracts nearly equal sums when the strata means dwarf their spread (the fixture's offset case, means near 50), and keeps one to two digits there; centre and scale <code>values</code> first when that matters, bearing in mind the formula is not shift invariant.</li>
</ul>
<p>{/* lens:GPUGeographicalDetector */}</p>
<h2 id="redcap-single-linkage-gpuredcapsinglelinkage">REDCAP single linkage: <code>GPURedcapSingleLinkage</code></h2>
<p>Contiguity-constrained single-linkage clustering, cut at a region count that is a per-frame parameter. It is the tree of first-order
single-linkage REDCAP (Guo 2008) with a different partition step, so it does not return the REDCAP regions: REDCAP partitions the tree by removing,
at each step, the edge whose removal lowers the within-region sum of squared deviations the most, which for this tree is what
<code>GPUSpatialWeightsMinimumSpanningTree</code> followed by <code>GPUSkaterRegions</code> computes; this class removes the heaviest edges. On a path with values
<code>[0, 1, 2.1, 3.3, 4.6, 6.0]</code> and two regions it returns <code>{0..4}, {5}</code> (within-region sum of squares 13.26) where the SSD partition is
<code>{0, 1, 2}, {3, 4, 5}</code> (5.85). Single linkage chains, so expect single outlying rows to be split off first. Use SKATER when REDCAP regions are
wanted. The minimum spanning tree of the contiguity graph under squared Euclidean dissimilarity of the standardised attributes holds every
single-linkage merge, so removing its <code>k - c</code> heaviest edges (<code>c</code> trees in the forest) leaves <code>k</code> contiguous regions. It composes
<code>GPUSpatialWeightsMinimumSpanningTree</code> (edges and dissimilarities), a stable <code>GPUSort</code> on <code>(cost, edge slot)</code>, a masked copy of the weights and the weak
components of what remains (<code>createMaskedWeakComponentNodes</code>). Region <code>k</code> refines region <code>k - 1</code>.</p>
<p><code>parameters</code> holds the region count (<code>getGPURedcapSingleLinkageParameterValues</code>). Outputs: <code>labels</code> (the smallest row index of each region),
and optionally <code>regionCount</code> and <code>status</code> with <code>converged</code>, <code>iterationLimitReached</code> and <code>invalidCount</code> (the tree's non-finite entry count; an edge that touches a non-finite row costs <code>+Infinity</code>
and is cut first). The independent optional <code>status.iterationCount</code> receives the number of executed
label-propagation rounds: always <code>componentIterations</code>, because all scheduled rounds run even after
labels stabilize. It excludes the spanning-tree stage and is written even for a partition with no
retained edges; requesting it does not require either convergence output.
<code>status.invalidCount</code> requires a packed <code>uint32</code> view with at least one word, validated at
construction. Region count, convergence/iteration-limit flags and invalid count write only their
first word; spare scalar capacity remains untouched on ordinary, suppressed and recovery frames.</p>
<p>A supplied nonzero <code>weights.status.overflow</code> or <code>candidateOverflow</code> suppresses the partition:
every label is <code>0xffffffff</code>, <code>regionCount</code> is 0, and both convergence/iteration-limit flags are 0.
The executed iteration count and the attribute invalid count retain their own meanings. When the
producer restores complete weights and clears the loss flag, the next submission publishes the
ordinary partition again; a partial forest is never presented as a valid region assignment.
REDCAP passes the original CSR and both supplied loss words directly to its minimum-spanning-tree
stage. That stage suppresses its tree flags and edge count on the GPU; REDCAP also suppresses the
dependent partition outputs. Clean status words produce the same partition as weights without
status, and changing either word requires no graph recompilation.
REDCAP snapshots its declared weights, loss views and output maps before validation. Replacing
the caller's maps after construction cannot disconnect a validated loss port; change buffer
contents to resubmit with the same bindings.</p>
<p>{/* lens:GPURedcapSingleLinkage */}</p>
<p>Limits and non-goals:</p>
<ul>
<li>The label propagation runs <code>componentIterations</code> iterations (default 32, at most 256) and <code>output.status.converged</code> reports whether that sufficed. A 300-row path with regions of 100 rows converged in 32; a region that is a much longer thin path may need more, and an unconverged labelling is not valid.</li>
<li>No minimum region size or bound (the GeoDa and <code>spopt</code> floor), and no full-order, average or complete linkage variants (they need the tree rebuilt per region). Weights must be symmetric.</li>
<li>Ties among edge costs are broken by the lower edge slot; other libraries tie-break differently, so compare on distinct costs.</li>
<li>float32: costs are float32 squared distances, so two edges whose true costs differ by about one part in <code>1e7</code> can swap order.</li>
</ul>
<p>Reference: no REDCAP package was pinned or run (<code>spopt</code> 0.8.0 has none). The reference is the constrained single-linkage definition, evaluated in
binary64 with SciPy (<code>minimum_spanning_tree</code>, the <code>k - c</code> heaviest edges removed, <code>connected_components</code>); it is the same partition as
<code>sklearn.cluster.AgglomerativeClustering(linkage='single', connectivity=W)</code>, which was not run. The fixture has a 7 by 7 and a 9 by 5 lattice and a
two-component graph, at region counts from 1 to above the row count.</p>
<p>{/* parameters: GPURedcapSingleLinkage generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuredcapsinglelinkage-parameters"><code>GPURedcapSingleLinkage</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURedcapSingleLinkageParameterValues(settings)</code>, writes <code>GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH</code> (1) <code>uint32</code> values.</li>
</ul>
<p>Settings of <code>getGPURedcapSingleLinkageParameterValues</code>:</p>
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
<td><code>regionCount</code></td>
<td><code>number</code></td>
<td>Target region count, at least 1. A value at or below the number of trees of the spanning forest (the connected components of <code>weights</code>) cuts nothing; a value above the row count cuts every edge.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="spatial-entropy-gpuspatialentropy-and-gpucategorypaircounts">Spatial entropy: <code>GPUSpatialEntropy</code> and <code>GPUCategoryPairCounts</code></h2>
<p><code>GPUCategoryPairCounts</code> counts, over the links of a weights graph, the ordered category pairs and the rows per category, with exact integer atomics
(at most 32 categories). <code>GPUSpatialEntropy</code> builds on it: the Shannon entropy of the categories, the entropy of the origin and destination category of a link, the
co-occurrence entropy of the ordered pair (Leibovici 2009), and the spatial mutual information
<code>H(origin) + H(destination) - H(pair)</code> (Altieri, Cocchi and Roli 2018), with its share of the destination entropy. All in nats. Every active slot (positive weight, another row: the shared neighbor rule) is a link;
self slots and zero-weight slots are not, and the size of a positive weight is not used: use contiguity weights, or distance-band weights per distance class for the distance-class form. <code>output.summary</code> is laid out by <code>GPU_SPATIAL_ENTROPY_LAYOUT</code>.</p>
<p>Entropies are float32 from exact counts and agree with binary64 to about <code>2e-6</code>. Batty's area-weighted entropy and the Altieri conditional terms are not implemented.
Reference: the R package <code>spatentropy</code> (GPL) was not installed, so the reference is a binary64 NumPy evaluation of the definitions above (<code>stx08-spatial-methods.py</code>).</p>
<p>{/* lens:GPUCategoryPairCounts */}</p>
<p>{/* lens:GPUSpatialEntropy */}</p>
<h2 id="correlogram-over-contiguity-orders-addcontiguityordercorrelogramrecipe">Correlogram over contiguity orders: <code>addContiguityOrderCorrelogramRecipe</code></h2>
<p>Global Moran's I with analytic inference for the neighbors at exactly 1 to <code>maximumOrder</code> steps: per order <code>GPUSpatialWeightsAlgebra</code> <code>higherOrder</code>,
<code>GPUSpatialWeightsTransform</code> <code>'row'</code> and <code>GPUGlobalSpatialStatistics</code>. <code>outputs.correlogram</code> holds <code>GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS.length</code> floats per order (I, expectation,
both variances, z and p under normality and randomization, <code>n</code>, island count). <code>outputs.incompleteOrders</code> is one flag per order; it includes upstream final/candidate loss and either loss introduced by that order. A flagged order publishes NaN Moran statistics and inference. Clearing input loss and resubmitting the same graph restores defined results when the order fits.
Capacities grow with the order: the defaults (<code>slotCapacity</code> and <code>workCapacity</code>) hold a square lattice, and memory is <code>maximumOrder</code> times the order capacity.</p>
<p>Reference: a binary64 breadth-first ring construction and Moran's I with the <code>esda.Moran</code> expectation and variances, in <code>stx08-spatial-methods.py</code>; libpysal and esda
were not installed. The fixture has an 8 by 8 and a 10 by 6 lattice for orders 1 to 5. The reference excludes rows with no neighbor at an order, as <code>islands: 'exclude'</code> does; no fixture exercises an island order.</p>
<p>{/* lens:addContiguityOrderCorrelogramRecipe */}</p>`};export{e as default};