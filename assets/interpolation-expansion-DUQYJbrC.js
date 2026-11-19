var e={slug:`api/gpu-spatial-analysis/interpolation-expansion`,title:`Bounded GPU interpolation and simulation`,html:`<h1 id="bounded-gpu-interpolation-and-simulation">Bounded GPU interpolation and simulation</h1>
<p>These contributors keep observations, fitting, prediction and simulation on the GPU. All graph views are packed, caller-owned, and belong to the target graph. Capacities specialize a graph. <code>GPULocalKriging</code> covariance models and parameters always come from a runtime buffer; input buffer contents can change between encodings. Outputs must have distinct buffers and must not alias inputs. Coordinates are native planar coordinates; project longitude/latitude before using these APIs.</p>
<table>
<thead>
<tr>
<th>Contributor</th>
<th>Contract</th>
<th>Explicit capacity</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPUVariogramModelFit</code></td>
<td>Spherical, exponential, Gaussian and Matérn 3/2 and 5/2 empirical-bin fitting, pair/unweighted/Cressie weights</td>
<td>5 model candidates, 3–1024 ladder points, 0–64 refinement iterations</td>
</tr>
<tr>
<td><code>GPULocalKriging</code></td>
<td>Simple, ordinary, universal/custom drift, external drift and multivariate LMC co-kriging; spherical, exponential, Gaussian and Matérn (smoothness 1/2, 3/2, 5/2) covariances in a runtime parameter buffer</td>
<td>32 nearest observations, 8 drift columns, 64 weighted quadrature points; 4 variables and 4 PSD structures</td>
</tr>
<tr>
<td><code>GPUSpaceTimeKriging</code></td>
<td>Local simple or ordinary space-time kriging, five covariance families, metric/octant/staged neighbor search, per-frame parameters</td>
<td>32 retained neighbors, 8 per octant, 64 staged candidates; O(queries · samples) scan</td>
</tr>
<tr>
<td><code>GPULocalRBF</code></td>
<td>Eight SciPy kernels, scalar/per-sample smoothing, resident fixed-width or CSR neighborhoods</td>
<td>32 nearest observations, polynomial degree −1 through 2 subject to kernel constraints</td>
</tr>
<tr>
<td><code>GPUConditionalGaussianSimulation</code></td>
<td>Exact joint conditional Gaussian realizations with known mean</td>
<td>4096 observation and query rows combined; dense factor and realization batch must fit device storage bindings</td>
</tr>
<tr>
<td><code>GPUNaturalNeighbor</code></td>
<td>Sibson stolen-Voronoi-area interpolation</td>
<td>64 local sites and 64 polygon vertices; indexed Delaunay cavities over up to 2²⁴ sites</td>
</tr>
</tbody>
</table>
<p><code>GPUVariogramModelFit</code> is exported from <code>@luma-contributors/analysis/dataframe-analysis</code>; the prediction contributors are exported from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. Capacities are explicit and audited against core WebGPU binding and shader-memory limits. Indexed natural-neighbor interpolation consumes resident Delaunay topology and containing-triangle seeds; it does not silently construct a CPU index.</p>
<h2 id="resident-variogram-fitting">Resident variogram fitting</h2>
<p>Provide equal-length <code>distances: float32</code>, <code>semivariances: float32</code>, and <code>pairCounts: uint32</code> graph views. Outputs contain one row per requested model, in the same order:</p>
<ul>
<li><code>parameters: float32x4</code>: nugget, partial sill, effective range, weighted squared residual.</li>
<li><code>diagnostics: float32x4</code>: usable bin count, unweighted squared residual, Gaussian residual AIC with three parameters, final log-range bracket width.</li>
<li><code>status: uint32</code>: 0 interior converged fit, 1 insufficient usable bins, 2 nonfinite fit, 3 boundary optimum or exhausted refinement.</li>
</ul>
<p>The range search uses <code>[minimumLag / 10, maximumLag * 3]</code>. Every fixed range solves nonnegative nugget/sill least squares, comparing both boundary fits. Cressie weighting starts from pair weights and refits three times with <code>pairCount / fittedSemivariance²</code>; the f32 denominator floor is <code>1e-20</code>. Invalid/nonpositive lag or empty bins are skipped. At least two distinct positive lags are needed. AIC compares the same empirical bins and is an unweighted residual diagnostic, not a spatial likelihood. Status 3 returns a candidate without claiming an identified interior range. The existing <code>fitVariogramModel</code> remains an explicitly CPU-side helper.</p>
<p>Explicit <code>rangeSteps</code> (3–1024) and <code>maximumIterations</code> (0–64) bind the GPU contributor, both
operation forms and the CPU helpers for every weighting pass. The ladder defaults to 200 points;
omitted refinement budgets retain GPU 32 and CPU 60 steps. Zero performs no refinement steps,
while still evaluating the ladder and the initial two interior candidates. Both operation forms
publish the actual final log-range bracket and status 3 for an endpoint ladder optimum or a bracket
wider than <code>1e-5</code>; finite candidates remain available. GPU can stop once the bracket is at most
<code>1e-5</code>, while CPU uses float64 refinements up to its budget.</p>
<p>Range candidates execute in parallel. Refinement uses one 64-lane workgroup per model to reduce bin statistics and residuals, with ordered golden-section steps and Cressie rounds. Floating accumulation order differs from the CPU helper and the earlier serial refinement; compare numerical tolerances and statuses rather than bit identity. The refinement adds fixed workgroup storage without increasing the candidate-search buffer or combining input batches.</p>
<h2 id="kriging-modes-and-support">Kriging modes and support</h2>
<p>Supply sample <code>positions: float32x2</code> and <code>observations: float32x4</code> rows <code>[value, measurementErrorVariance, variableIndex, validity]</code>. Validity zero omits an observation. Nonfinite positions/value/error, negative errors, or missing sample drift are omitted. A query with no usable neighbors is invalid. <code>supports</code> stores query-major quadrature points. <code>float32x2</code> uses equal weights; <code>float32x4</code> stores <code>[x, y, weight, reserved]</code>. Nonnegative weights are normalized per query; negative, nonfinite, or zero-total weights invalidate that query. <code>supportCount</code> defaults to one and is at most 64. Neighborhoods use distance from the weighted support centroid, breaking equal-distance ties by original row order.</p>
<p>Candidates come from caller <code>neighborhoods</code> (fixed rows or CSR) when supplied. Otherwise <code>sampleIndex</code> (compile-time) chooses: <code>'brute-force'</code> scans every sample from each query (rejected above <code>GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK</code>, 2³² sample-query tests); <code>'grid'</code> writes the support centroids and a mask of the usable samples, then a <code>GPUNeighborSearch</code> kNN grid lists the <code>neighborCount + max(neighborCount, 8)</code> nearest usable samples of each centroid by float32 <code>dx² + dy²</code>. The kernel ranks the candidates exactly as the scan does, by <code>distance()</code> with ties to the lower row; when a query's candidate list is full and its farthest candidate is within 1e-6 (relative) of the retained boundary, a tie may continue past the list and that query rescans every sample, so the grid keeps exactly the scan's neighbors even on float32 near-ties. Queries run in chunks whose candidate rows, together with a copy of the sample positions, fit one storage binding; the grid kernel therefore binds as many storage buffers as the scan. <code>'auto'</code> (default) scans only when there are at most <code>GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT</code> (4,096) samples and at most 2³² sample-query tests, and uses the grid otherwise, so the default is never an uncapped scan; it throws only when neither fits (more than 2²⁴ samples plus queries, beyond the grid's 128 MiB search-target binding, and more than 2³² tests). An explicit <code>'grid'</code> checks its search-target binding against the device limit when its nodes are built. Every choice retains the same neighbors in the same order, so results do not depend on it. Optional <code>spatialContext</code> declares the coordinate space; only native planar coordinates are accepted.</p>
<table>
<thead>
<tr>
<th>Mode</th>
<th>Mean/design contract</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>simple</code></td>
<td>Known scalar <code>means[0]</code>, default zero; solves the covariance system without an unbiasedness constraint.</td>
</tr>
<tr>
<td><code>ordinary</code></td>
<td>Unknown constant mean with one unbiasedness constraint.</td>
</tr>
<tr>
<td><code>universal</code></td>
<td>Defaults to an unknown affine trend <code>[1, x, y]</code> with locally centered/scaled coordinates. Supplying <code>sampleDrift</code> and <code>queryDrift</code> selects an arbitrary caller-defined design with 1–8 columns; rank-deficient designs fail explicitly.</td>
</tr>
<tr>
<td><code>external-drift</code></td>
<td>Caller supplies 1–8 columns in <code>sampleDrift</code> and support-averaged <code>queryDrift</code>. Flat <code>float32</code> stores row-major <code>driftCount</code> columns; <code>float32x4</code> remains available for up to 4 columns. Include an intercept explicitly. The caller computes support-averaged drift using the same quadrature weights; missing query drift invalidates prediction.</td>
</tr>
<tr>
<td><code>co-kriging</code></td>
<td>Set <code>variableCount</code> to 2–4 and select <code>targetVariable</code>. <code>coKrigingMean: "known"</code> uses supplied means; <code>"ordinary"</code> imposes a separate unknown intercept per variable. <code>coregionalizationModels</code> supplies 1–4 structures, each with its own covariance and square row-major loading factors, so <code>C_ab(h)=Σ_s (L_s L_sᵀ)_ab C_s(h)</code> is PSD. The original 2×2 <code>coregionalization</code> shortcut remains supported. With <code>"ordinary"</code>, a secondary variable that has no sample in a neighbourhood drops out of that neighbourhood's system (its constraint is vacuous), so with every secondary variable absent the prediction is ordinary kriging of the target. The system needs at least as many retained samples as kept constraints (one for the target plus one per secondary variable present), so a neighbourhood with fewer samples than <code>variableCount</code> predicts as long as that holds: one target sample alone predicts its value. A neighbourhood without a sample of the target variable has no unbiased predictor: it reports status 1 when it retains fewer samples than kept constraints and status 2 otherwise.</td>
</tr>
</tbody>
</table>
<p>Covariance values always use the layout from <code>getGPULocalKrigingParameterValues</code>. Pass a caller-owned <code>parameters</code> view and write its <code>GPUParameterBuffer</code> between frames, or keep the convenience <code>covariance</code>, <code>coregionalization</code> and <code>coregionalizationModels</code> props as initial values and call <code>setCovariance(settings)</code> to change them. Model family, nugget, sill, range and loadings change without recompiling; variable and structure counts stay fixed. Submit preceding encodings before updating parameters. Each compiled graph owns its convenience parameter upload buffer and releases it on destruction; the contributor needs no separate cleanup. On devices where adding the runtime buffer would exceed the storage-binding limit, the convenience path appends parameters to a transient copy of the observation table, preserving the existing drift/weights configurations. That copy must fit one storage binding. Initialized zero-loading co-kriging models use normalization scale 1, keeping zero process variance distinct from an unwritten parameter buffer. With known means and positive observation error, such a model predicts the target mean with latent variance zero; an unwritten all-zero buffer still reports status 1.</p>
<p>The local kriging, RBF, space-time kriging and Vecchia kernels use the analysis-core LU factorization and substitution. The prediction adapter retains four sequential symmetric Ruiz passes and the existing pivot thresholds, so conditioning remains comparable to earlier results within float32 tolerance.</p>
<p>Each covariance structure has a supplied spherical/exponential/Gaussian/Matérn effective range and partial sill. <code>model: 'matern'</code> needs <code>smoothness</code> 1/2, 3/2 or 5/2 and uses the closed form with the argument <code>a = factor * h / range</code>, where <code>factor</code> is 3 at 1/2 (the exponential model), 4.749031386012701 at 3/2 and 5.924462618951797 at 5/2, so the correlation at <code>h = range</code> is <code>exp(-3)</code>, like the exponential and Gaussian models. For <code>a &#x3C; 0.1</code> the correlation is evaluated by a Taylor series, because the f32 product <code>(1 + a) exp(-a)</code> can round above 1 and blurs the small <code>1 - correlation</code> that separates near-duplicate rows. To use GSTools <code>Matern</code>, set <code>len_scale = sqrt(nu) * range / factor</code>. Process nugget applies at coincident positions. Measurement error is added only to observation covariance diagonals; it is never added to latent prediction variance. Positive errors smooth sample hits. Duplicate observations with zero error can make the conditioning matrix singular and are reported; this API never silently adds jitter or merges them.</p>
<p>For block prediction, sample-to-block covariance sums weighted support covariances, and block variance uses the double weighted sum over every support pair. This includes the process-nugget contribution of repeated quadrature sites. The quadrature count is explicit provenance; it is a discrete weighted approximation to the intended support, not an exact polygon integral.</p>
<p><code>output.values: float32x4</code> is <code>[prediction, latent variance, pivot ratio, retained neighbor count]</code>. <code>status</code> is 0 on success, 1 for missing/invalid data and 2 for deficient rank or unreliable variance. A query with no usable sample in its neighbourhood is status 1 with NaN by default (<code>emptyNeighborhood: 'insufficient'</code>). With <code>emptyNeighborhood: 'prior'</code> (compile-time, mode <code>simple</code> only, since only a known mean has a prior prediction) it reports status 0 with the known mean, the prior point or block variance, pivot ratio 1 and retained count 0, the simple kriging predictor of an empty neighbourhood; <code>GPUVecchiaLogLikelihood</code> uses it for sites whose conditioning rows are all excluded. Optional <code>weights</code> contains query-major neighbor weights in <code>(distance, row)</code> order, padded with zeros to <code>neighborCount</code>. Partial-pivot elimination of the equilibrated system rejects pivots below <code>1e-6</code> of the initial matrix scale and pivot ratios below <code>KRIGING_PIVOT_RATIO_TOLERANCE</code> (<code>1e-4</code>, the limit <code>GPUKriging</code> uses, measured at 5% weight error for a ratio of 3e-5; <code>GPULocalKriging</code> and <code>GPULocalRBF</code> used <code>1e-5</code> before and accepted such systems with status 0, now status 2); normalized negative variance within <code>1e-4</code> of the prior variance is clamped to zero. Outputs at failed queries contain NaN predictions. Weights are only valid with successful status. These f32 conditioning guards deliberately reject some solvable f64 systems.</p>
<h3 id="conditioning-of-smooth-covariances">Conditioning of smooth covariances</h3>
<p>The pivot-ratio limit above is the only conditioning rule. No nugget floor is applied; a Matérn 3/2, 5/2 or Gaussian system with a small or zero nugget either passes the limit or reports status 2 with its pivot ratio in <code>values.z</code>. A floor was not added because it would silently change the requested model at every query, while this API never adds jitter, and it would change existing Gaussian results; the 1e-3-of-sill floor that the float32 Vecchia experiment recommends concerns the likelihood, not the prediction solve. Measured once on headless Chromium WebGPU over random 16- and 32-point neighborhoods (sill 1, range 0.6 of the scene), every system accepted at nugget 0 predicted within 1.6e-3 of the double-precision solve and within 2.5e-5 in variance; Matérn 3/2 and 5/2 systems had prediction errors of at most 1.4e-5 and 1.5e-4, and status 2 occurred for Gaussian and for one Matérn 5/2 query. Treat status 2 as a failed site in a likelihood or cross-validation sweep; do not skip it.</p>
<h2 id="space-time-kriging">Space-time kriging</h2>
<p><code>GPUSpaceTimeKriging</code> takes sample <code>positions</code> with relative float32 <code>times</code>, the observation rows of <code>GPULocalKriging</code>, query <code>queries</code> with <code>queryTimes</code>, and a per-frame <code>parameters</code> buffer from <code>getGPUSpaceTimeKrigingParameterValues</code>. <code>temporalContext</code> declares the time columns and is resolved with <code>float32-relative</code> (the output context of <code>GPUTimeWordsRelative</code>); <code>queryTemporalContext</code> defaults to it and must have the same unit and epoch, checked by <code>assertCompatibleGPUTemporalContexts</code>. Compile-time: the sample and query counts, the mode (<code>simple</code> or <code>ordinary</code>), <code>neighborCount</code>, and <code>neighborSearch</code> with its capacities. Everything in the buffer, including the model, changes without a recompile.</p>
<p>Covariance models (<code>SpaceTimeVariogramModel</code>) use the gstat 2.1-6 <code>vgmST</code> definitions, with <code>h</code> the spatial lag and <code>u</code> the time lag:</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Covariance</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>separable</code></td>
<td><code>sill · cs(h) · ct(u)</code>, on structures standardized to nugget + sill = 1</td>
</tr>
<tr>
<td><code>product-sum</code></td>
<td><code>k Cs(h) Ct(u) + Cs(h) + Ct(u)</code></td>
</tr>
<tr>
<td><code>metric</code></td>
<td><code>Cj(sqrt(h² + (anisotropy · u)²))</code></td>
</tr>
<tr>
<td><code>sum-metric</code></td>
<td><code>Cs(h) + Ct(u) + Cj(sqrt(h² + (anisotropy · u)²))</code></td>
</tr>
<tr>
<td><code>gneiting</code></td>
<td><code>sill ψ^-τ exp(-c h^(2γ) ψ^(-βγ)) + nugget at (0,0)</code>, <code>ψ = a u^(2α) + 1</code>, <code>c = 3 / spatialRange^(2γ)</code>, <code>a = (exp(3/τ) - 1) / temporalRange^(2α)</code>, with <code>β ∈ [0,1]</code>, <code>τ ≥ β</code></td>
</tr>
</tbody>
</table>
<p>Each structure has the families and effective-range convention of the spatial model, and a nugget that applies at its own zero lag: <code>Cs</code> at <code>h = 0</code>, <code>Ct</code> at <code>u = 0</code>, <code>Cj</code> at zero joint lag. <code>anisotropy</code> is spatial units per unit of time (gstat <code>stAni</code>). Gneiting is not a gstat model; the definition is Gneiting (2002).</p>
<p>Neighbor search semantics (gstat documents staged and octant search as not yet available, so these are this library's):</p>
<table>
<thead>
<tr>
<th><code>neighborSearch</code></th>
<th>Semantics</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'metric'</code></td>
<td>The <code>neighborCount</code> samples nearest in the metric <code>sqrt(h² + (searchAnisotropy · u)²)</code>.</td>
</tr>
<tr>
<td><code>'octant'</code></td>
<td>Per quadrant of the sample offset from the query and per side in time (sample time before the query is the past, otherwise the future), the <code>neighborsPerOctant</code> nearest in the metric; then the <code>neighborCount</code> nearest of those. A sample exactly at the query goes to the positive-offset quadrant.</td>
</tr>
<tr>
<td><code>'staged'</code></td>
<td>The <code>stagedCandidateCount</code> samples nearest in the metric, then the <code>neighborCount</code> of those with the largest model covariance to the query, ties going to the nearer candidate.</td>
</tr>
</tbody>
</table>
<p>Exact metric ties resolve to the lower sample row. <code>searchRadius</code> and <code>maximumTimeLag</code> exclude samples before the search. <code>searchAnisotropy</code> defaults to the anisotropy of a metric or sum-metric model and is required otherwise. The kernel keeps the conditioning rule of <code>GPULocalKriging</code>: status 2 below <code>KRIGING_PIVOT_RATIO_TOLERANCE</code>, no nugget floor. <code>output.values</code> and <code>status</code> match <code>GPULocalKriging</code>. A query scans every sample, so the cost is queries × samples; there is no resident neighbor index input yet.</p>
<p><code>fitSpaceTimeVariogramModel</code> fits a model to the bins of <code>GPUSpaceTimeVariogram</code> on the CPU, reading back at most 409 bins, like <code>fitVariogramModel</code>. Use <code>getGPUSpaceTimeVariogramBinLags</code> for the representative lags (zero for class 0 of an axis, class midpoints otherwise). Nonlinear parameters (ranges, anisotropy, nugget fractions of the separable and product-sum structures, Gneiting <code>β</code>) use the ladder and golden-section search of <code>fitVariogramModel</code> in coordinate cycles, and the remaining sills and nuggets are exact non-negative weighted least squares. The spatial and temporal margins start the ranges. <code>atBoundary</code> reports a parameter at the edge of its ladder. Gneiting <code>α</code>, <code>γ</code> and <code>τ</code> are fixed options, defaulting to 0.5, 0.5 and 1. gstat <code>variogramST</code> and <code>krigeST</code> fixtures are pending because R is not installed in this environment; covariances and predictions are verified against independent binary64 solves, and GSTools (metric models only, its one space-time family) covers the metric family.</p>
<p>{/* lens:GPUSpaceTimeKriging */}</p>
<h2 id="local-radial-basis-interpolation">Local radial-basis interpolation</h2>
<p><code>GPULocalRBF</code> accepts scalar values, sample positions and query positions. Nonfinite samples are omitted. <code>epsilon</code> multiplies radial distance; <code>smoothing</code> adds a nonnegative kernel diagonal; optional <code>sampleSmoothing</code> supplies one diagonal per sample and replaces the scalar. Kernels are thin-plate-spline, cubic, Gaussian, linear, quintic, multiquadric, inverse-multiquadric and inverse-quadratic. Polynomial degree is −1, 0, 1, 2: thin-plate/cubic require at least 1; quintic requires 2; linear/multiquadric require at least 0. Gaussian and inverse kernels default to −1; set degree 0 to match SciPy's default constant polynomial.</p>
<p>Optional <code>neighborhoods: uint32</code> restricts candidate indices. Fixed rows use <code>neighborhoodCapacity</code> slots and <code>0xffffffff</code> padding; <code>neighborhoodOffsets</code> instead accepts resident CSR directly from <code>GPUNeighborSearch</code>. Each row contains distinct sample indices. The kernel reorders candidates by distance and original row index and retains at most 32. Without candidate inputs it scans all samples. The augmented indefinite system uses partial-pivot elimination, with centered/scaled polynomial coordinates. Zero-smoothing duplicate data and rank-deficient polynomial designs fail explicitly. Outside-hull extrapolation is allowed; <code>searchRadius</code> and <code>neighborCount</code> delimit the local model. <code>output.values</code> is <code>[prediction, pivot ratio, retained count, 0]</code>; statuses are 0 success, 1 insufficient neighborhood, 2 singular/numerically unreliable solve. Per-query scratch is bounded and there is no global quadratic matrix.</p>
<h2 id="conditional-gaussian-simulation">Conditional Gaussian simulation</h2>
<p><code>GPUConditionalGaussianSimulation</code> uses the same covariance and observation row format, ignoring the variable-index lane, and rejects <code>model: 'matern'</code>: its dense Cholesky has no pivot ratio check, and Matérn is validated only for local kriging. It conditions a joint Gaussian field with a supplied known <code>mean</code>. Up to 128 observations and 128 query sites retain the four-node cooperative path. Larger domains use 16-row blocked Cholesky: a small diagonal factor, parallel panel solves, and 16×16 tiled trailing updates synchronize through GPU dispatch boundaries. Blocked forward solves and parallel query moments prepare a shared conditional factor, then tiled multiplication generates all realizations. There is no CPU solve or readback. By default each encoding repeats preparation. The supported capacity is 4096 observation and query rows combined, counting masked input slots. The factor requires <code>4 * (2 + N² + 5N)</code> bytes for combined capacity <code>N</code> (about 64 MiB at 4096), and blocked draws add <code>4 * queryCount * replicateCount</code> bytes of temporary normal samples. Both must fit an individual device storage binding. Use smaller explicit replicate batches to reduce draw storage; the joint site domain must remain intact. This is still a dense cubic-work, quadratic-storage method, not a large-raster approximation. Query covariance may be semidefinite, so exact observed query sites can have zero variance. Singular observation covariance fails; masks may omit all observations, producing an unconditional joint field.</p>
<p><code>values</code> is replicate-major. <code>moments: float32x2</code> holds each query's conditional mean and variance. <code>status</code> has one row per replicate: 0 success, 1 invalid input, 2 numerical failure. <code>seed</code> and <code>replicateOffset</code> identify independent Philox streams, so explicit batches preserve replicate identities. Box–Muller uses f32 transcendental functions: compare reproducibility on the same backend and distributional moments across implementations. Blocking preserves the covariance model and observation-error handling, without silently adding jitter; changed floating-point evaluation order need not reproduce older draw bits. Exact-hit and repeated query sites retain semidefinite support. Splitting query sites into separate calls creates separate joint fields and does not preserve cross-call covariance. Conditioning, loading factors and random normals remain GPU-resident.</p>
<h3 id="reusing-preparation">Reusing preparation</h3>
<p>Set <code>reusePreparation: true</code> to retain the factor and conditional moments across encodings. The constructor's <code>replicateCount</code> becomes the maximum mutable batch size. The producer owns persistent factor, moment, control and validity buffers: 17,980 bytes at 32+32 or 268,348 bytes at 128+128, excluding caller inputs and outputs. The blocked path uses GPU-written indirect dispatch gates to skip factor and moment workgroups when their retained generations are current. It can be added to one graph once. Destroy compiled graphs before calling the producer's <code>destroy()</code>; input and output buffers remain caller-owned.</p>
<table>
<thead>
<tr>
<th>Update</th>
<th>Required operation</th>
</tr>
</thead>
<tbody>
<tr>
<td>Seed or replicate range</td>
<td><code>setDrawParameters({seed?, replicateOffset?, replicateCount?})</code>; factor and moments are reused</td>
</tr>
<tr>
<td>Observation values</td>
<td>Write the observation buffer, then call <code>invalidateObservations()</code>; validation and conditional moments refresh</td>
</tr>
<tr>
<td>Known mean</td>
<td><code>setMean(mean)</code>; the mean must be finite in f32 and conditional moments refresh</td>
</tr>
<tr>
<td>Positions, queries, validity flags or measurement errors</td>
<td>Write inputs, then call <code>invalidatePreparation()</code></td>
</tr>
<tr>
<td>Compile-time covariance (<code>covariance</code>) or view shapes</td>
<td>Create a new producer and graph specialization</td>
</tr>
<tr>
<td>Covariance model and parameters through <code>parameters</code> (<code>getGPULocalKrigingParameterValues</code>)</td>
<td>Write the parameter buffer; the graph does not recompile</td>
</tr>
</tbody>
</table>
<p>Input writes do not trigger automatic invalidation. Observation invalidation also checks compacted geometry and error variances and rebuilds if they differ, but callers should use full invalidation for geometry changes. Failed validation never draws from stale moments; invalid inputs and recovery retain the default status contract. Reuse validity is committed by GPU execution, so discarding an encoded command buffer does not mark unbuilt state valid.</p>
<p>The setters enqueue buffer writes. Submit preceding encodings before changing their controls, just as with other mutable input buffers; encoding alone does not snapshot parameter values. Application code still owns command encoding, submission and completion. <code>replicateCount</code> must be at least one and at most the constructor capacity, and the last replicate identity must fit uint32. Rows beyond the active batch prefix are cleared to NaN with status 1. Only the active prefix belongs to the requested batch; moments remain one row per query.</p>
<p>After <code>graph.add(producer)</code>, <code>preparationStatistics</code> exposes a <code>uint32x2</code> view containing actual GPU factorization and moment-solve attempt counts. It is undefined before declaration and after destruction. Counts are diagnostic, accumulate across executions and use uint32 arithmetic; they are not timing measurements or cache-validity tokens.</p>
<h2 id="natural-neighbors">Natural neighbors</h2>
<p><code>GPUNaturalNeighbor</code> computes actual Sibson coordinates: it builds the inserted query's Voronoi cell, intersects it with each original sample Voronoi cell, and normalizes the stolen areas. A nearest-triangle method does not provide the same interpolation. Without a topology input, the direct path scans a bounded local site set. With <code>triangles: uint32</code>, each six-word triangle stores <code>[a,b,c,neighborAB,neighborBC,neighborCA]</code>, using <code>0xffffffff</code> at hull edges. <code>queryTriangles</code> supplies a containing triangle per query, or <code>0xffffffff</code> for outside-hull invalidity. The GPU traverses intersected circumcircles and gathers a bounded cavity (<code>maximumNeighborCount</code>, default 32, maximum 64; at most 4× that many inspected triangles), then computes the same stolen areas. Input triangles must be a valid Delaunay triangulation with unique sites and reciprocal adjacency. Site count may reach 2²⁴, subject to GPU buffer limits; no global pair scan occurs in indexed mode. Topology construction and seed location remain explicit caller operations.</p>
<p>Successful <code>output.values</code> is <code>[prediction, inserted cell area, weight sum, contributing count]</code>. Optional query-major <code>weights</code> is scalar in original sample order for the direct path. Indexed mode uses <code>float32x2</code> rows <code>[sampleIndex, weight]</code> in cavity discovery order, with <code>maximumNeighborCount</code> slots per query; unused rows are <code>[-1,0]</code>. The 2²⁴ site cap preserves exact integer indices in these f32 diagnostic rows. Weights are nonnegative, sum to one and reproduce affine fields within f32 geometry tolerances. Sample hits reproduce the sample; multiple samples exactly at the query average. The direct path reports duplicate sites elsewhere or globally collinear samples as status 2; indexed mode requires duplicate-free Delaunay topology. Hull-edge queries use the limiting interpolation between the closest bracketing edge samples; outside-hull queries return status 1. Exhausted cavity, traversal, or polygon capacity returns status 3 without a successful partial estimate. Geometry uses query-centered coordinates scaled by the maximum sample distance, with explicit f32 tolerances; extremely thin configurations may be rejected rather than assigned unreliable weights.</p>
<h2 id="verification-and-limits">Verification and limits</h2>
<p>The original numerical suite <code>interpolation-coverage.spec.ts</code> uses pinned NumPy covariance-system fixtures, SciPy <code>RBFInterpolator</code>, Shapely/GEOS Voronoi intersection areas, analytic conditional moments and a 4096-replicate ensemble covariance check. <code>test/oracles/interpolation-coverage.py</code> regenerates independent fixtures with NumPy 2.2.6, SciPy 1.15.3 and Shapely 2.1.2. Tests also cover rank failure, hull edges, exact hits, outside-hull invalidity, forced geometric capacity, and all existing variogram weighting conventions. Node tests parse all mode kernels and enforce the default eight-storage-binding budget.</p>
<p>A native Apple WebGPU/Metal 3 run on October 23, 2026 validated ten small/bounded larger benchmark cases with unchanged and changed input buffers. At the larger shapes, warm encode/submit/queue-completion medians were 0.9 ms for 16-neighbor kriging over 128 queries with 16 support points, 0.7 ms for 16-neighbor RBF over 128 queries, 2.9 ms for 32-sample Sibson over 128 queries, and 3.9 ms for 128 conditional realizations of 16 sites given 16 observations. The original serial three-model variogram fit used 52.1 ms at 256 bins (51.8–117.4 ms over five samples). These historical results predate cooperative fitting and simulation factorization.</p>
<p>The <a href="https://github.com/visgl/luma.gl/blob/master/modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md">benchmark procedure and measured phases</a> and raw measurement artifact separate host compile, upload, first execution, warm samples, result readback, updates, caller storage and graph transient bytes. These are device-specific observations, not latency guarantees or GPU-only timestamps. The device exposed expanded limits; a separate static audit passed the default eight-binding and shader-memory budgets at maximum supported options. Original run source identifiers were omitted; extraction-time hashes carry an explicit provenance limitation.</p>
<p>These earlier measurements cover the first bounded implementation. The expanded indexed and shared-factor paths are measured below. No general throughput advantage is asserted. Applications should measure their own graph preparation, upload, execution and optional readback before choosing these contributors for a workload.</p>
<h2 id="direct-reference-parity-and-expanded-scaling">Direct reference parity and expanded scaling</h2>
<p><code>interpolation-advanced.spec.ts</code> adds pinned GSTools 1.7.0 and PyKrige 1.7.3 mode-by-mode prediction/variance fixtures, SciPy 1.15.3 eight-kernel quadratic RBF fixtures, CGAL 6.0.1 natural-neighbor coordinates and linear interpolation, and gstat 2.1-6 weighted block and three-variable/two-structure LMC predictions. The committed <code>test/oracles/interpolation-advanced.py</code>, <code>interpolation-cgal.cpp</code>, and <code>interpolation-gstat.R</code> regenerate these references. CGAL and R dependencies are offline reference tools, not bundled runtime dependencies. GSTools effective ranges are converted explicitly (<code>len_scale = range/3</code> for exponential); gstat Gaussian uses <code>range/√3</code>. Arbitrary eight-column designs also use independent NumPy constrained solves.</p>
<p>The simulation comparison checks GSTools conditional means/variances and marginal conditioned-field moments, and gstat sequential conditional simulation for full joint covariance. GSTools 1.7 <code>CondSRF</code> pointwise rescales an unconditional random field by kriging variance, which generally gives a different off-diagonal covariance from exact joint Gaussian conditioning; this implementation does not claim joint distribution equality with that approximation. Seeded sequences are checked for this backend, while independent-library ensembles are compared statistically.</p>
<p>Expanded benchmarks include resident grid/CSR RBF at 1,024 and 16,384 samples, indexed Sibson at 1,024/16,384/65,536 sites, and shared-factor simulation at 32+32 and 128+128 observations/sites. They measure compile, upload, first/warm/updated execution, readback and transient storage separately. A repeated-fit comparison measures resident fitting against actual empirical-bin readback, CPU fitting, parameter upload and queue completion at 20/256 bins. These are observed workload comparisons, not a general GPU speedup claim.</p>
<p>On the same native Apple WebGPU/Metal device, the October 23 expanded run recorded these historical warm encode/submit/queue-completion medians (five samples, output readback excluded):</p>
<table>
<thead>
<tr>
<th>Workload</th>
<th>Shape</th>
<th align="right">Median</th>
<th align="right">Graph transient bytes</th>
</tr>
</thead>
<tbody>
<tr>
<td>Indexed Sibson</td>
<td>1,024 / 16,384 / 65,536 sites, 128 queries</td>
<td align="right">0.3 / 0.3 / 0.4 ms</td>
<td align="right">0</td>
</tr>
<tr>
<td>Resident grid + CSR RBF</td>
<td>1,024 / 16,384 sites, 128 queries, 32 neighbors</td>
<td align="right">5.6 / 6.9 ms</td>
<td align="right">76,356 / 420,424</td>
</tr>
<tr>
<td>Joint simulation</td>
<td>32+32 / 128+128 observations+queries, 128 realizations</td>
<td align="right">5.1 / 146.2 ms</td>
<td align="right">17,672 / 267,272</td>
</tr>
<tr>
<td>Three-model resident fit</td>
<td>20 / 256 bins</td>
<td align="right">4.3 / 51.3 ms</td>
<td align="right">0</td>
</tr>
<tr>
<td>Actual CPU fit round trip</td>
<td>Same bins, including readback and parameter reupload</td>
<td align="right">1.3 / 3.2 ms</td>
<td align="right">Caller buffers only</td>
</tr>
</tbody>
</table>
<p>At that checkpoint, the CPU fit round trip was faster than the serial GPU fit. Later cooperative refinement records 0.7 ms warm at 256 bins in three sessions and 0.7 ms for changed-input GPU execution versus 2.3–2.5 ms CPU round trips; small fits have no consistent comparator winner. Cooperative simulation factorization records 8.1 ms at 128+128 across three sessions. See the refinement report (<code>dev-docs/research/gpu-geospatial-performance-2026-10-23/integration-notes.md</code> in the repository) and simulation report (<code>dev-docs/research/gpu-geospatial-performance-2026-10-23/integration-notes.md</code> in the repository) for source hashes, first-use variability and validation. These results precede preparation reuse. A later reuse comparison (<code>dev-docs/research/gpu-geospatial-performance-2026-10-23/integration-notes.md</code> in the repository) measures complete sequences: 100 large draw batches take 92.55 ms retained versus 816.70 ms forced, including first preparation and per-batch completion. Each batch has 128 replicates and overwrites the same output buffer; per-batch readback and output retention are excluded. Value updates improve 2.66 times, geometry updates show no gain, and single-batch reuse has no demonstrated benefit. Indexed Sibson timings exclude topology construction and seed location; RBF timings include GPU grid construction and neighbor search. The full expanded record includes first execution, phase timings, every warm sample, changed-input validation, device metadata and capture-time source hashes. GPU timestamps were not enabled.</p>
<p>The blocked dense path adds full conditional-covariance checks at 129+129 and 128+384 rows, including repeated query sites, noiseless observation hits, masked counts crossing block boundaries, and an unconditional field. Retained lifecycle tests cover both algorithms, including discarded encodings and smaller active replicate batches. The optional <code>conditional-simulation-reuse-benchmark.spec.ts</code> also includes 256+256 and 512+512 domains; enable it with <code>VITE_SIMULATION_REUSE_BENCHMARK=true</code> and select workloads using the shared benchmark filter. These added sizes have no published timing claim; the historical measurements above cover the cooperative path.</p>`};export{e as default};