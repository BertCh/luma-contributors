var e={slug:`api/dataframe-analysis/coregionalization-fit`,title:`GPUCoregionalizationFit`,html:`<h1 id="gpucoregionalizationfit">GPUCoregionalizationFit</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-dataframe-analysis#gpucoregionalizationfit">dataframe API</a>.</p>
<p>Fit a linear model of coregionalization (LMC) to experimental direct and signed cross
variograms with fixed basic structures. Exported by
<code>@luma-contributors/analysis/dataframe-analysis</code>, together with
<code>GPUCoregionalizationFitOperation</code>, <code>CPUCoregionalizationFitOperation</code>,
<code>computeCoregionalizationFitOnCPU</code> and <code>fitLinearModelOfCoregionalization</code>.</p>
<p>The reference is <a href="https://doi.org/10.1007/BF00893750">Goulard and Voltz (1992)</a>.
The block update and PSD projection also appear in
<a href="https://www.jackwbaker.com/Publications/Loth_Baker_(2013)_Spatial_Cross_Correlation,_EESD.pdf">Loth and Baker, section 4.1, equations 36–38</a>.</p>
<h2 id="input-and-algorithm">Input and algorithm</h2>
<p>For <code>variableCount = p</code>, <code>semivariances</code> contains one packed upper triangle per lag:
<code>(0,0), (0,1), …, (0,p-1), (1,1), …, (p-1,p-1)</code>. Distances and pair counts have one
row per lag. All direct and cross variograms must use the same lag supports and
complete-case sample pairs. Assemble the triangles from <code>GPUVariogram</code> and
<code>GPUCrossVariogram</code> outputs using those common supports and masks. Negative cross
semivariances are retained. A nonfinite matrix entry, nonpositive/nonfinite lag,
or zero pair count excludes the whole bin. No missing cross entries are imputed.</p>
<p>The objective is <code>sum_h N_h ||Gamma_h - sum_s B_s g_s(h)||_F^2 / sum_h N_h</code>.
Off-diagonal errors count twice. Weights are shared across a lag matrix; pair-specific
weights, Cressie reweighting, and automatic model/range selection are outside this fit.</p>
<p>Start every <code>B_s</code> at zero. For each structure, fit the residual after removing the
other structures, then project the conditional symmetric matrix onto the PSD cone
by clipping negative eigenvalues to zero. Updates are sequential within a sweep.
Sufficient statistics keep each sweep independent of the number of lag bins. Dependent structures or too few distinct lag bins
can produce nonunique coefficients; only the summed covariance is identified.</p>
<h2 id="parameters-and-outputs">Parameters and outputs</h2>
<p><code>getGPUCoregionalizationFitParameterValues(structures, tolerance = 1e-5)</code> packs
<code>[tolerance, modelCode, nugget, sill, range, …]</code>. Families are <code>spherical</code>, <code>exponential</code>, <code>gaussian</code>, <code>matern-3/2</code> and
<code>matern-5/2</code>, with the existing kriging effective-range convention. Each basis is
<code>g(h) = nugget + sill * shape(h/range)</code> at positive lags. Unit-sill structures use
<code>sill: 1, nugget: 0</code>; a pure nugget uses <code>sill: 0, nugget: 1</code> and any positive range.
Model codes, nuggets, sills, ranges and tolerance are per-frame input values.
Counts, shapes and <code>maximumIterations</code> (default 128) are compile-time.
<code>GPUCoregionalizationFit</code> captures these options and all input/output/status views at
construction, including getter-backed fields. Its props and port maps are frozen snapshots;
later edits to the caller's objects do not change generated kernels or bindings. Views and
buffers retain their identities and their contents remain per-frame inputs. Construct a
new contributor to change the shape, budget or bindings.
No coordinates or spatial context are read: lags and ranges must use the same units.</p>
<p>Outputs <code>matrices</code> and <code>loadings</code> contain <code>structureCount * p * p</code> float32 elements,
structure-major and row-major. <code>B = L L^T</code>; <code>L</code> is a square spectral factor, not necessarily
triangular. Eigenvector signs and bases at repeated eigenvalues are arbitrary; compare
<code>B</code> or <code>L L^T</code> across implementations. Diagnostics contain
<code>[weighted objective, relative coefficient change, usable bins, total pair weight]</code>.
Relative change is the largest absolute coefficient change divided by the largest
absolute old/new coefficient over the whole sweep (zero when both are zero).</p>
<p>The required named uint32 scalar status ports are <code>validity</code>, <code>iterationCount</code>,
<code>converged</code>, <code>iterationLimitReached</code> and <code>invalidCount</code>. A budget exit publishes the
current PSD fit with validity 1, converged 0 and iterationLimitReached 1. Zero budget
publishes zero matrices and factors. Empty usable input, invalid parameters, a zero-energy
structure, or nonfinite arithmetic gives validity 0, convergence/limit flags 0 and NaN
coefficients/objective/change. Usable-bin and weight diagnostics remain available.
<code>invalidCount</code> counts excluded lag matrices. Output views may have trailing capacity;
only the documented result prefix is written. Views must be packed and output buffers distinct.</p>
<h2 id="cpucoregionalizationfitoperation-and-supplied-kriging-model">CPUCoregionalizationFitOperation and supplied kriging model</h2>
<p>Prefer the float64 CPU twin for small fits. It has the same graph ports and computes in
float64 with a single float32 rounding at output. The shared CPU exponential and Gaussian
shape evaluation uses <code>expm1</code> to preserve small lag/range values. NumPy fixtures include
Gaussian and both Matérn lag/range ratios of <code>1e-8</code> and <code>1e-9</code> with approximately identity
fitted matrices. The CPU Matérn shapes use a bounded positive exponential-remainder series
below argument <code>0.5</code>; scalar accuracy and fitted models are checked against independent
80-digit Python Decimal evaluation of the closed forms, including the series transition.
It drops GPU-only bin and sweep caps.
Float32 accumulation can change convergence counts and eigenvector bases; headless
specs compare matrices and reconstructed factors with 0.3% relative/absolute tolerance,
and long-fit iteration counts within three sweeps. Ordinary cases agree more closely.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {fitLinearModelOfCoregionalization} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/dataframe-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> fit</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> fitLinearModelOfCoregionalization</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  {distances, semivariances, pairCounts},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    variableCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    structures: [</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      {model: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'spherical'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, range: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, sill: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, nugget: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">},</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">      {model: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'exponential'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, range: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">200</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, sill: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    ],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    maximumIterations: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">512</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Inspect fit.validity[0], fit.converged[0] and fit.iterationLimitReached[0].</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> (</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">fit.validity[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">throw</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> Error</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'Coregionalization fit is invalid'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Pass to GPULocalKriging (with positions, observations, supports and outputs):</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> suppliedModel</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mode: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'co-kriging'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  variableCount: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  covariance: fit.coregionalizationModels[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">].covariance,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  coregionalizationModels: fit.coregionalizationModels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span></code></pre>
<p><code>GPULocalKriging</code> also requires its <code>covariance</code> prop when no parameter buffer is supplied;
use the first fitted structure as above. The co-kriging model uses every fitted structure.
Invalid fits return an empty <code>coregionalizationModels</code> list. The GPU operation returns
factors in the same layout: read back each <code>p * p</code> slice to supply <code>loadings</code>.</p>
<h2 id="gpucoregionalizationfitoperation">GPUCoregionalizationFitOperation</h2>
<p>The graph operation uses <code>GPU_CONTRIBUTOR_OPERATION_LOWERING</code>; the CPU operation can
replace it using the same properties, graph names, formats and lengths.
Both operations capture their shape, iteration budget, input names and nested output/status
names at construction, including getter-backed properties. Later edits to the caller's objects
cannot change the declared contract or a subsequent execution/lowering. Construct a new
operation to change these options or ports; graph input values remain dynamic on every run.</p>
<h2 id="reference-tier-bounds-and-verification">Reference-tier bounds and verification</h2>
<p>GPU limits are 2–4 variables, 1–4 structures, 0–256 lag matrices and 0–1024 sweeps.
The existing co-kriging bounds are unchanged. Construction validates shapes and limits
with named helpers. Each sweep uses one gated fit kernel and one gate-update kernel;
finished sweeps dispatch no fit work. The exact node estimate is <code>2 * maximumIterations + 3</code>.
The maximum configuration has 2051 command nodes and at most eight storage bindings.</p>
<p>The independent NumPy fixture generator forms bin residuals directly and uses
<code>numpy.linalg.eigh</code>; its Matérn shapes are evaluated with 80-digit Python Decimal arithmetic
to avoid cancellation in the oracle itself. Node specs cover signed crosses, indefinite matrices, multiple
structures including 4-by-4 bounds, padding, empty bins and budget exits. Headless specs
compare every GPU/CPU output (factor Gram matrices for loadings), then feed a fitted LMC
to local co-kriging and check a hand-solved two-observation predictor and variance.
The maximal configuration is registered in the kernel audit.</p>`};export{e as default};