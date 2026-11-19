var e={slug:`api/gpu-spatial-analysis/instrumental-variables-covariance`,title:`Instrumental-variable robust covariance`,html:`<h1 id="instrumental-variable-robust-covariance">Instrumental-variable robust covariance</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/operators-regression#gpuspatialtwostageleastsquares">spatial regression API</a>.</p>
<p><code>GPUInstrumentalVariablesCovariance</code> and <code>GPUSpatialTwoStageLeastSquares.robust</code> support
<code>HC0</code>, <code>HC1</code>, <code>HC2</code>, <code>HC3</code>, <code>HAC</code> and one-way <code>CLUSTER</code>. The graph operation and
<code>CPUSpatialTwoStageLeastSquaresOperation</code> accept the same method and cluster options.
No new estimator or public contributor class is introduced.</p>
<h2 id="formula-and-corrections">Formula and corrections</h2>
<p>Let <code>Z</code> be the structural design, <code>H</code> the instruments, both including an intercept,
<code>P = H (H'H)^-1 H'</code>, <code>B = (Z'PZ)^-1</code>, and <code>A = B Z'H (H'H)^-1</code>.
The structural residual is <code>u = y - Z delta</code>; influence score <code>s_i = A H_i u_i</code>.
This is the IV sandwich used by <a href="https://pysal.org/spreg/_modules/spreg/robust.html">PySAL spreg's robust_vm</a>,
rather than the OLS sandwich on endogenous <code>Z</code>.</p>
<ul>
<li>HC0: <code>sum_i s_i s_i'</code>.</li>
<li>HC1: HC0 times <code>n / (n-p)</code>, where <code>p</code> includes the intercept, consistent with
spreg's <code>sig2n_k=True</code> White correction. HAC has no finite-sample correction.</li>
<li>HC2: replace <code>u_i</code> by <code>u_i / sqrt(1-h_i)</code>.</li>
<li>HC3: replace <code>u_i</code> by <code>u_i / (1-h_i)</code>.</li>
<li>CLUSTER: <code>G/(G-1) * (n-1)/(n-p) * sum_g S_g S_g'</code>, <code>S_g = sum_{i in g} s_i</code>.
This is one-way CR1. <code>G</code> counts occupied groups, not the supplied key capacity.
No extra HC1 factor is applied. With singleton groups it equals HC1.</li>
</ul>
<p>HC2/HC3 are extensions of spreg's sandwich, with the explicitly selected <strong>projected-design</strong>
leverage <code>h_i = (PZ)_i B (PZ)_i'</code>, including the intercept. This orthogonal second-stage hat
matrix is distinct from the oblique structural IV hat matrix. No leverage convention is inferred
from the endogenous raw design. <code>h_i &#x3C; -1e-6</code>, nonfinite leverage, or <code>1-h_i &#x3C;= 1e-6</code> fails
inference, including when <code>u_i = 0</code>; values are never clipped.</p>
<h2 id="cluster-input-and-limits">Cluster input and limits</h2>
<p><code>clusters</code> is one packed uint32 key per estimation row. <code>clusterCount</code> is a compile-time dense
key capacity in <code>[1,n]</code>, with active keys in <code>[0,clusterCount)</code>. Unused slots and unsorted keys
are supported. At least two groups must be occupied. <code>0xffffffff</code> and all other out-of-range
keys fail inference; no observations are silently deleted. Stable sort and shared segmented
reductions sum scores by group. Empty slots contribute zero outer products and never increase <code>G</code>.
There is no CPU scan over GPU inputs and no all-pairs cluster kernel.</p>
<p>The GPU factors at most 15 structural columns excluding the intercept and 16 instrument columns
excluding the intercept. Rows are at most <code>2^24</code>, further limited by device storage bindings.
All shape/method checks run at construction; device binding checks run before graph nodes are made.
The contributor exposes and validates <code>estimatedCommandNodeCount</code>, including a conservative
allowance for published sort/gather internals. CPU twins retain float64 QR and drop GPU matrix caps.</p>
<h2 id="psd-certificate-and-status">PSD certificate and status</h2>
<p>Every method symmetrizes its computed covariance <code>V</code>. For nonzero finite <code>V</code>, it factors
<code>V / max(abs(V)) + 1e-5 I</code> using the shared Cholesky solver. A successful factorization certifies
PSD <strong>within this float32 relative tolerance</strong>; it does not assert exact arithmetic PSD.
Zero and rank-deficient matrices are accepted. The published covariance is not shifted or
projected onto the PSD cone. A failed certificate returns NaNs, including robust table inference,
while the point estimates, structural residuals and classical diagnostics remain available.
This certifies the resulting small covariance, not every eigenvalue of a potentially huge HAC kernel.
HAC callers must still supply a symmetric unit-diagonal nonnegative PSD kernel.</p>
<p>Use <code>output.status: {code: statusView}</code> for the named solver port. The legacy scalar
<code>output.status: statusView</code> remains supported. <code>GPU_INSTRUMENTAL_VARIABLES_COVARIANCE_STATUS</code>
exports these names:</p>
<table>
<thead>
<tr>
<th>Code</th>
<th>Name</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>SUCCESS</td>
<td>Covariance passed the numerical certificate</td>
</tr>
<tr>
<td>1</td>
<td>SINGULAR_PROJECTION</td>
<td>Projection system could not be solved</td>
</tr>
<tr>
<td>2</td>
<td>INSUFFICIENT_ROWS</td>
<td><code>n &#x3C;= p</code></td>
</tr>
<tr>
<td>3</td>
<td>NONFINITE_INPUT</td>
<td>Input or computed covariance is nonfinite</td>
</tr>
<tr>
<td>4</td>
<td>INVALID_KERNEL</td>
<td>Invalid HAC structure, symmetry, diagonal or overflow</td>
</tr>
<tr>
<td>5</td>
<td>INVALID_LEVERAGE</td>
<td>HC2/HC3 denominator is unresolved or invalid</td>
</tr>
<tr>
<td>6</td>
<td>INVALID_CLUSTERS</td>
<td>Invalid key or fewer than two occupied groups</td>
</tr>
<tr>
<td>7</td>
<td>NOT_POSITIVE_SEMIDEFINITE</td>
<td>Covariance failed the shifted Cholesky certificate</td>
</tr>
</tbody>
</table>
<p>The existing <code>robust.output.status</code> graph-value name maps to this covariance code. The point-fit
solver code remains separate. GPU projection failures take precedence over leverage/cluster
failures; invalid HAC kernels take precedence over projection failures, preserving existing behavior.
The CPU twin uses the same numeric tolerances, with float64 arithmetic before its single output
rounding. It can differ near the thresholds and retains its documented QR/rank differences.</p>
<h2 id="independent-verification">Independent verification</h2>
<p><code>spatial-two-stage-least-squares-operation-fixtures.py</code> constructs a NumPy SVD projector,
dense sandwich matrices and occupied-group CR1 correction without calling first-party solvers.
It covers overidentification, second-order instruments, gapped keys and indefinite HAC covariance.
Node specs compare CPU matrices, tables and diagnostics against those fixtures. Headless specs
compare the direct contributor against NumPy and a hand-derived rank-one sandwich, and compare
GPU/CPU operations on every output, including covariance failures. Zero covariance, unit leverage,
invalid/missing keys, empty cluster slots and residual scales from <code>1e-12</code> to <code>1e12</code> are covered.</p>`};export{e as default};