var e={slug:`api/gpu-spatial-analysis/spatial-validation-advanced`,title:`Spatial fold designs and nested interpolation validation`,html:`<h1 id="spatial-fold-designs-and-nested-interpolation-validation">Spatial fold designs and nested interpolation validation</h1>
<p><code>GPUSpatialCrossValidation</code> deals blocks to folds in a seeded random order (rank of a per-block Philox draw, modulo the fold count) when <code>design</code> is omitted. The optional design consumes resident covariates and prediction locations; outcomes are never an input to fold construction. All coordinates and exclusion distances use native planar units; an optional <code>spatialContext</code> accepts only the default planar context. Reference tier: fold designs are one serial invocation (256 rows; 64 for NNDM) and buffered masks cost O(folds * rows^2) at up to 65,536 rows and 64 folds. Fold-major training/testing masks remain the estimator boundary.</p>
<table>
<thead>
<tr>
<th>Design</th>
<th>Executed contract</th>
<th>Bounds and distinctions</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>groups</code></td>
<td>Sort distinct valid numeric group IDs, assign rank modulo fold count, preserve every group</td>
<td>At most 256 rows. Matches blockCV 4.0.1 <code>cv_group(balance=FALSE)</code> including leave-group-out. Seed does not change fixed group ranks.</td>
</tr>
<tr>
<td><code>environmental</code></td>
<td>Complete-row feature scaling, deterministic farthest-first seeds, Lloyd assignment/centroid iterations</td>
<td>At most 256 rows, eight feature columns and 16 clusters; 1–1024 iterations. Standardization uses sample mean/SD of supplied complete rows, not raster-wide moments. Constant columns contribute zero; insufficient distinct centers/empty clusters fail explicitly. blockCV partition comparison uses <code>cv_cluster(scale=FALSE)</code> on extracted covariates and compares memberships up to label permutation. R's Hartigan-Wong/random initialization is not replicated.</td>
</tr>
<tr>
<td><code>nndm</code></td>
<td>Leave-one-out nearest-neighbor distance-distribution matching with the prediction domain, bounded maximum distance and minimum retained training proportion</td>
<td>At most 64 complete sample rows and 256 prediction locations; fold count equals sample count. Direct blockCV <code>cv_nndm</code> radii and masks agree, including strict training distance <strong>greater than</strong> the final exclusion radius. The upstream minimum-proportion rule operates during neighbor deletion; final strict-radius exclusion can retain fewer rows when distances tie.</td>
</tr>
<tr>
<td><code>knndm</code></td>
<td>Score every valid resident candidate partition by empirical 1-Wasserstein distance between test-to-training and prediction-to-training nearest distances; choose the lowest score subject to nonempty folds and maximum fold fraction</td>
<td>At most 256 sample/prediction rows and 64 candidates. Omitted candidates generate deterministic seeded farthest-first/Lloyd spatial partitions (up to 16 folds). Caller-supplied partitions can instead come from any resident clustering graph. Direct blockCV hierarchical candidate partitions and <code>cv_knndm</code> selected partition/objective are verified. Automatic generation is a bounded Lloyd restart search, not blockCV's hierarchical/overcluster/merge ladder or its KS-test random fallback.</td>
</tr>
</tbody>
</table>
<p><code>design.diagnostics</code>, when supplied, contains <code>[status, iterations, objective, selectedCandidate]</code>; NNDM appends one exclusion radius per source row. Status is 0 success, 1 invalid/incomplete required inputs or nonfinite distance/objective arithmetic, 2 infeasible clustering/partition, or 3 exhausted Lloyd iterations. Unused objective/candidate fields are NaN. Failed designs publish <code>0xffffffff</code> folds and empty masks, so existing per-fold status reports empty training/testing. NNDM and kNNDM require all sample and prediction positions to be finite and unmasked; environmental/group designs explicitly omit masked/incomplete rows. Deterministic initialization, seed, candidate order and returned folds provide replay provenance. Candidate ties choose the lowest index. No CPU clustering, distance matching, readback or fitted-outcome reuse occurs.</p>
<p><code>GPUSpatialCrossValidatedOLS</code> evaluates each held-out fold with training-only centers, slopes and
the correction for rounded means. Its fold-major predictions use the same centered primitive as
OLS fitted values, preserving response-level detail with large coordinate-like predictors. The
published fold coefficients retain the raw intercept representation.
Its metric status reports bit1 for no scored rows, bit2 for missing outcomes/predictions,
bit4 for undefined R² and bit8 for nonfinite metric arithmetic. Arithmetic failure preserves
scored/missing counts and returns NaN for all four aggregate metrics. Every fold-design
input and diagnostic is included in the recipe's input/output alias checks.</p>
<h2 id="nested-variogram-and-bandwidth-selection">Nested variogram and bandwidth selection</h2>
<p><code>GPUNestedInterpolationValidation</code>: Default mode tiled supports 4,096 rows; mode reference retains
256 rows and 32 neighbors. Both retain fold, candidate and device limits.</p>
<p><code>GPUNestedInterpolationValidation</code> takes explicit outer <code>training</code>/<code>testing</code> masks, inner block IDs and a finite candidate ladder. Each candidate is either an ordinary-kriging covariance/variogram specification (<code>model</code>, effective <code>range</code>, <code>sill</code>, <code>nugget</code>) or an RBF kernel with inverse bandwidth <code>epsilon</code> and smoothing. This is discrete predictive hyperparameter selection: it does not run an unbounded continuous optimizer or silently fit an empirical variogram on the complete dataset.</p>
<p>For every outer fold the graph:</p>
<ol>
<li>Rejects intersecting outer training/testing masks and excludes nonfinite training responses.</li>
<li>Constructs inner block folds and optional buffers <strong>inside the outer training mask</strong>.</li>
<li>Fits every candidate separately on each inner training mask. Only inner held-out responses enter its score.</li>
<li>Selects the lowest complete inner MSE. Any missing inner prediction invalidates that candidate, so missing coverage cannot improve its score.</li>
<li>Fits candidates using only outer training observations, publishes the selected model's outer test predictions, and computes held-out coverage, MSE, MAE, signed prediction-minus-observation bias and R².</li>
</ol>
<p>Changing outer held-out responses changes the outer metrics but leaves that fold's candidate scores, selected index and predictions unchanged. Changed-buffer replay verifies this invariant. Kriging observations are packed with per-fit validity; RBF excluded responses are NaN. Existing local solvers retain rank, capacity and missing-neighborhood failure semantics. An empty inner test fold contributes no observations; all-empty inner validation cannot select a model. No complete candidate yields selected index <code>0xffffffff</code> and missing predictions.</p>
<p>Bounds are 4,096 observations by default (256 with <code>mode: 'reference'</code>), 2–16 outer folds, 2–8 inner folds, 1–16 candidates and 2–32 local neighbors. Work scales with the product of these counts; a supported bound is not a latency guarantee. Outputs are caller-owned. Prediction rows outside the outer test mask are NaN. Status bit1 marks overlapping outer masks, bit2 no selected candidate, bit4 missing held-out predictions bit8 an empty held-out set and bit16 nonfinite held-out metric arithmetic (the four aggregate metrics become NaN). Undefined R² is NaN when the held-out outcome variance is zero.</p>
<p>The independent reference generator is <code>modules/analysis/test/oracles/spatial-validation-advanced.R</code>, pinned to blockCV 4.0.1. <code>spatial-validation-advanced.spec.ts</code> compares real native GPU results against that fixture, tests complete test coverage, disjoint masks, invalid/empty/exhausted designs, automatic kNNDM generation and nested changed-input leakage invariance. Independent NumPy 2.2.6 bordered covariance solves and SciPy 1.15.3 RBFInterpolator verify every nested candidate score, chosen model, held-out prediction and metric. Maximal shader options have their own kernel-audit cases.</p>
<p>{/* lens:GPUNestedInterpolationValidation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=result caption="Two outer folds select between supplied kriging and RBF models using training-only inner folds."}</p>
<p>::lens{scenario=gpu-spatial-analysis.nested-interpolation-validation rung=graph caption="Inspect the separate inner masks, fits, candidate scores, outer refits and held-out metrics."}</p>
<h2 id="conformal-prediction-intervals">Conformal prediction intervals</h2>
<p><code>GPUConformalPredictionIntervals</code> and <code>GPUGeoConformalPredictionIntervals</code> turn held-out predictions into prediction intervals with a coverage guarantee, without a distributional assumption. The calibration scores are <code>R_i = |y_i - prediction_i| / scale_i</code>, where <code>scale_i</code> is optional (for example the kriging standard deviation). The held-out predictions come from the fold-major <code>predictions</code> of <code>GPUSpatialCrossValidatedOLS</code>, or from one holdout model. Pass the fold of each row (<code>GPUSpatialCrossValidation</code> <code>output.folds</code>) as <code>calibration.folds</code>; a row with an unassigned fold, a non-finite value or a false mask is excluded.</p>
<table>
<thead>
<tr>
<th>Contributor</th>
<th>Interval</th>
<th>Guarantee under exchangeability</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUConformalPredictionIntervals</code>, <code>foldCount</code> 1</td>
<td>Split conformal: <code>p -+ s R_(r)</code> with <code>r = ceil((1 - alpha)(n + 1))</code></td>
<td>Coverage at least <code>1 - alpha</code>; at most <code>1 - alpha + 1/(n + 1)</code> when calibration and test scores are almost surely distinct</td>
</tr>
<tr>
<td><code>GPUConformalPredictionIntervals</code>, <code>foldCount</code> above 1</td>
<td>CV+: lower is the <code>floor(alpha (n + 1))</code>-th smallest of <code>p_k(i) - s R_i</code>, upper the <code>ceil((1 - alpha)(n + 1))</code>-th smallest of <code>p_k(i) + s R_i</code> (Barber et al. 2021)</td>
<td>Coverage at least <code>1 - 2 alpha</code>; in practice close to <code>1 - alpha</code></td>
</tr>
<tr>
<td><code>GPUGeoConformalPredictionIntervals</code></td>
<td>Weighted quantile of the neighbouring scores with the query's own weight as a point at infinity (Lou, Luo and Meng 2024, on Tibshirani et al. 2019)</td>
<td>Approximate; local when the weights reflect the exchangeability structure</td>
</tr>
</tbody>
</table>
<h3 id="gpuconformalpredictionintervals"><code>GPUConformalPredictionIntervals</code></h3>
<p>Ties can increase split-conformal coverage above the stated distinct-score upper bound. For example, seven calibration scores equal to one, prediction zero and <code>alpha = 0.25</code> give <code>[-1, 1]</code> with status 0. An exchangeable population always equal to one has coverage 1, exceeding <code>1 - alpha + 1/(n + 1) = 0.875</code>; the lower coverage guarantee still holds.</p>
<p><code>alpha</code> is a per-frame parameter: write <code>getGPUConformalPredictionParameterValues({miscoverage, selfWeight})</code> into a <code>GPUParameterBuffer</code> between encodings to change it without recompiling. Calibration scores are sorted once by (fold, score) with the shared stable radix sort. Each query then finds its exact order statistic by a 32-step bisection over the float32 keys of the candidate values, with one binary search per fold, so no loop runs over the calibration rows and a query costs <code>32 x folds x log n</code> comparisons. A rank outside <code>[1, n]</code> gives an unbounded side (<code>-Infinity</code> or <code>Infinity</code>) with a status bit instead of a clamped value.</p>
<p>The ranks are exact integers for the packed float32 <code>alpha</code>: the shader multiplies its significand by the integer <code>n + 1</code> using 64-bit integer words, then shifts to obtain <code>lowerRank = floor(alpha (n + 1))</code> and uses <code>upperRank = n + 1 - lowerRank</code>. There is no rank slack or snapping to a nearby decimal. For example, <code>alpha = 0.1</code>, <code>n = 99</code> gives ranks 10 / 90; <code>alpha = 0.7</code>, <code>n = 9</code> gives 6 / 4 because float32 0.7 is slightly below 7/10. The coverage statements refer to the packed alpha. At <code>n = 8,388,607</code>, <code>alpha = 0.25</code>, the exact ranks are 2,097,152 / 6,291,456. Use <code>getConformalPredictionRanks(n, alpha)</code> to inspect ranks on the CPU without allocating calibration rows. The global contributor accepts at most <code>GPU_CONFORMAL_MAXIMUM_CALIBRATION_COUNT</code> (4,294,967,294) calibration rows so that <code>n + 1</code> fits uint32; storage and sort limits can impose smaller bounds. Localized intervals retain a 2^-20 relative tolerance on their approximate float32 weighted-quantile threshold. Interval arithmetic is float32 (<code>p + s R</code>), so with a non-power-of-two <code>scale</code> the bound can differ from a binary64 evaluation by one ulp.</p>
<p><code>status</code> per query has bit1 for an unbounded lower side, bit2 for an unbounded upper side, bit4 for a query prediction or scale that is not a finite positive number (the interval is NaN), bit8 for no usable calibration row, and bit16 (localized only) when a query has more than 256 neighbour entries and the tail was dropped. <code>output.calibrationCount</code> holds the number of usable calibration rows.</p>
<p>{/* lens:GPUConformalPredictionIntervals */}</p>
<p>{/* parameters: GPUConformalPredictionIntervals generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuconformalpredictionintervals-parameters"><code>GPUConformalPredictionIntervals</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUConformalPredictionParameterValues(settings)</code>, writes <code>GPU_CONFORMAL_PREDICTION_PARAMETER_LENGTH</code> (4) <code>float32</code> values.
<ul>
<li>Input port: <code>parameters</code>.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUConformalPredictionParameterValues</code>:</p>
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
<td><code>miscoverage</code></td>
<td><code>number</code></td>
<td>Target miscoverage <code>alpha</code> in the open interval (0, 1): intervals cover at least <code>1 - alpha</code>.</td>
</tr>
<tr>
<td><code>selfWeight</code> (optional)</td>
<td><code>number</code></td>
<td>Localized intervals only: weight of the query location itself in the weighted quantile, the kernel value at distance zero. Default 1, right for kernels scaled so that <code>K(0) = 1</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h3 id="gpugeoconformalpredictionintervals"><code>GPUGeoConformalPredictionIntervals</code></h3>
<p>The localized contributor takes a compressed-sparse-row neighbour list per query (<code>offsets</code>, <code>indices</code>, <code>weights</code>; at most 256 entries are used per query) with kernel weights from any neighbour search, and uses one model (<code>foldCount</code> 1). An entry that is out of range, excluded or has a zero, negative or non-finite weight is skipped. When every usable weight equals the positive <code>selfWeight</code>, the quantile uses the global contributor's exact packed-alpha rank over those same usable entries. Both bounds and the unbounded status bits therefore agree, including at decimal boundaries: 49 scores <code>1..49</code>, prediction zero and alpha 0.02 produce <code>[-Infinity, Infinity]</code> with status 3 for both contributors. This equivalence is independent of the common weight's magnitude; a truncated neighbour list still sets its truncation bit. Weights are summed in float32 in list order, so otherwise the weighted quantile retains its approximate float32 threshold.</p>
<p>{/* lens:GPUGeoConformalPredictionIntervals */}</p>
<h3 id="conformal-verification">Conformal verification</h3>
<p><code>conformal-prediction-intervals.spec.ts</code> checks real GPU results against a brute-force reference that sorts every candidate value and derives ranks independently by multiplying packed float32 alpha by the small fixture count exactly in binary64 (split, CV+ with 64 folds, scales, masks, non-finite inputs, unbounded sides, the exact-integer rank cases, tied scores, and the localized quantile with dyadic weights). Coverage is verified by simulation with seeded replicates of one compiled graph. For split conformal with continuous scores, calibration coverage is Beta(r, n + 1 - r), so a replicate covering <code>m</code> test points has mean <code>r/(n + 1)</code> and variance <code>p(1 - p)/m + (1 - 1/m) Var(C)</code>; the spec requires the mean over 1000 replicates to lie within 4.5 standard errors of that mean for <code>n</code> 99 with <code>alpha</code> 0.1 and <code>n</code> 19 with <code>alpha</code> 0.05. A heteroscedastic case (noise standard deviation 0.2 on the left half and 2 on the right) shows global intervals covering the right half only about 80 percent while the localized intervals reach 90 percent in both halves.</p>
<p>Reference: the external oracle (the authors' GeoConformal code) is pending; the reference used is the binary64 order-statistic implementation above. Lou, Luo and Meng (2024) is cited for the definition only.</p>
<h2 id="area-of-applicability">Area of applicability</h2>
<p><code>GPUAreaOfApplicability</code> computes the dissimilarity index (DI) and the area of applicability (AOA) of Meyer and Pebesma (2021), following the definition of the R package CAST 1.2.0 (GPL: used as a definition and test oracle only, no code is copied). Predictors are standardized with the sample mean and standard deviation (<code>n - 1</code>) of the usable training rows and multiplied by optional variable-importance weights. The index of a point is its Euclidean distance to the nearest training point divided by <code>meanDistance</code>, the mean over training rows of their mean distance to all other training rows. With <code>crossValidation</code> (the <code>folds</code> and fold-major <code>training</code> masks of <code>GPUSpatialCrossValidation</code>) a training point's nearest neighbour is searched among the training rows of the fold in which it is held out, so a spatial buffer is honored; without it every other row is a candidate. The threshold is <code>min(Q3 + 1.5 IQR, max)</code> of the training indices, with type 7 quartiles as in R <code>quantile</code>, and a query row is inside the area when its index is at most the threshold. <code>summary</code> holds <code>[meanDistance, threshold, usableTrainingRows, scoredTrainingRows, Q1, Q3]</code>; <code>status</code> has bit1 for fewer than two usable rows, bit2 for a predictor without variance and bit4 for no scored training row.</p>
<p>Differences from CAST: Euclidean distance only; categorical predictors must be supplied as dummy columns; a training row held out in no fold or without a neighbour is left out of the threshold (CAST would give it an infinite index); rows with non-finite predictors are unusable. The distances are exhaustive and in float32: the training pass costs <code>O(rows^2 x predictors)</code> (at most 65,536 rows and 16 predictors) and each query row <code>O(rows x predictors)</code>. <code>area-of-applicability.spec.ts</code> compares real GPU results against a binary64 implementation of the CAST definition written independently of the kernels (with and without folds, buffered masks, weights, a row mask and non-finite entries), and checks two hand-computed examples, one of which exercises the maximum cap of the threshold. The R oracle is pending: R and CAST are not installed in the test environment.</p>
<p>{/* lens:GPUAreaOfApplicability */}</p>
<h2 id="measured-execution-envelopes">Measured execution envelopes</h2>
<p>The native Chromium WebGPU run recorded ten correctness-checked workloads on the Apple Metal-3 adapter. Timings include CPU encode, submit and queue completion. Warm medians use five repetitions after the first execution; source hashes, upload, compile, first-run, readback, three changed-input repetitions and caller/transient bytes are recorded in <code>dev-docs/research/evidence/spatial-validation-2026-10-23.json</code>. Shared device caches were not cleared; these observations are not latency guarantees or CPU speedup claims. GPU timestamps were unavailable for these runs.</p>
<table>
<thead>
<tr>
<th>Workload</th>
<th align="right">Warm median ms</th>
<th align="right">First execution ms</th>
<th align="right">Physical transient bytes</th>
</tr>
</thead>
<tbody>
<tr>
<td>validation-groups-64</td>
<td align="right">2.600</td>
<td align="right">53.900</td>
<td align="right">0</td>
</tr>
<tr>
<td>validation-groups-256</td>
<td align="right">5.200</td>
<td align="right">46.400</td>
<td align="right">0</td>
</tr>
<tr>
<td>validation-environmental-64</td>
<td align="right">0.400</td>
<td align="right">3.800</td>
<td align="right">0</td>
</tr>
<tr>
<td>validation-environmental-256</td>
<td align="right">0.700</td>
<td align="right">4.100</td>
<td align="right">0</td>
</tr>
<tr>
<td>validation-nndm-16</td>
<td align="right">0.500</td>
<td align="right">6.300</td>
<td align="right">1024</td>
</tr>
<tr>
<td>validation-nndm-64</td>
<td align="right">9.200</td>
<td align="right">23.600</td>
<td align="right">16384</td>
</tr>
<tr>
<td>validation-knndm-64</td>
<td align="right">3.400</td>
<td align="right">50.600</td>
<td align="right">1040</td>
</tr>
<tr>
<td>validation-knndm-256</td>
<td align="right">42.400</td>
<td align="right">148.600</td>
<td align="right">4112</td>
</tr>
<tr>
<td>nested-interpolation-12</td>
<td align="right">0.600</td>
<td align="right">28.100</td>
<td align="right">1584</td>
</tr>
<tr>
<td>nested-interpolation-256</td>
<td align="right">2.100</td>
<td align="right">270.900</td>
<td align="right">33792</td>
</tr>
</tbody>
</table>
<p>The 256-row nested measurement uses two outer folds, two inner folds, two candidates and four neighbors. Increasing these axes multiplies the fit count; maximal per-kernel capacities are separately audited, and the combined maximum graph is not a measured interactive workload. Automatic kNNDM measurements use four deterministic clustering candidates. Every workload validates disjoint masks or constant-function prediction and repeats after changing resident input values.</p>`};export{e as default};