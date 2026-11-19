var e={slug:`api/gpu-spatial-analysis/gaussian-random-field`,title:`Gaussian random fields`,html:`<h1 id="gaussian-random-fields">Gaussian random fields</h1>
<h2 id="gpugaussianrandomfield"><code>GPUGaussianRandomField</code></h2>
<p>{/* lens:GPUGaussianRandomField */}</p>
<p>::lens{scenario=gpu-spatial-analysis.gaussian-random-field rung=formula}</p>
<p>The synthetic lens records a seeded finite-mode field on a 4 by 4 planar lattice, including <code>output.status.approximation</code> and clipped eigenvalue mass.</p>
<p>Generates one stationary isotropic Gaussian realization on a planar grid or a packed set of
points. Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. No datasets or weights
are included. A new seed or realization counter changes the draw without recompiling.</p>
<p>Provide exactly one of <code>grid</code> (published <code>GPURasterMetadata</code>) or <code>positions</code> (<code>float32x2</code>), a
<code>parameters</code> view of 16 float32 words, and caller-owned output views: <code>values</code> (float32),
<code>validity</code> (uint32), <code>status.approximation</code> (one uint32), and <code>clippedEigenvalueMass</code> (one float32).
<code>values</code> and <code>validity</code> have the same length, at least the domain row count. Extra rows are NaN
and invalid. All views are single packed views, on the target graph, with disjoint output buffers.
Host the packed parameters in a <code>GPUParameterBuffer</code>, or upload the packer's result to a
caller-owned storage buffer. Pass the same grid dimensions to the contributor and packer.
The grid's affine is read from the parameter buffer, so it can change each frame. Grid samples
are cell centers, respecting area/point interpretation and signed axis steps. Only native planar
Euclidean distances and axis-aligned grids are supported; project geographic coordinates first.
Generated grid coordinates stay relative to the first cell center for both spectral evaluation
and residual kriging. World-coordinate origins are never added to float32 grid positions, so
translating a grid preserves its spacing and realization, including when conditioned.
For points, center coordinates before float32 upload and set the phase <code>origin</code> near the data.</p>
<h3 id="methods-and-limits">Methods and limits</h3>
<p><code>method: 'auto'</code> uses the published <code>GPUFFT2D</code> for grids with both sides at most 1024, and
spectral randomization for larger grids or arbitrary points. <code>fft</code> and <code>spectral</code> can be selected
explicitly. Counts, method, mode count, coordinate context and conditioning topology are
compile-time; every view's contents, variogram, mean, seed, realization and affine are per-frame.</p>
<p>FFT embeds the covariance in a doubled power-of-two torus, capped at 2048 per side. With the
published unnormalized forward and normalized inverse transforms, the spectrum is multiplied
by <code>sqrt(N * max(lambda, 0))</code> and independent standard normal real/imaginary components.
The real inverse field has the requested covariance when the embedding is positive semidefinite.
This follows <a href="https://smai-jcm.centre-mersenne.org/articles/10.5802/smai-jcm.89/">circulant embedding</a>.
Large ranges can produce negative eigenvalues. They are clipped; <code>clippedEigenvalueMass</code> reports
<code>sum(max(-lambda, 0))/N</code>, the added variance. Any clipping sets <code>status.approximation = 1</code>; there is
no automatic embedding growth or silent claim of exactness. FFT supports spherical,
exponential, Gaussian and Matérn 3/2 or 5/2 variograms (including <code>matern</code> smoothness aliases).</p>
<p>The finite-mode method uses
<code>sqrt(sill/M) * sum(A*cos(k dot x) + B*sin(k dot x))</code> plus independent row nugget noise,
with independent standard normal amplitudes. Frequencies are drawn from the 2D covariance
spectrum, as in <a href="https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.field.generator.RandMeth.html">GSTools RandMeth</a>.
For <code>U</code> uniform on (0,1), the radial inverse CDF is <code>sqrt(-12*log(U))/range</code> for Gaussian,
and <code>a*sqrt(U^(-1/nu)-1)/range</code> for exponential (<code>nu=1/2</code>, <code>a=3</code>) and Matérn
(<code>nu=3/2</code> or <code>5/2</code>, using the library's effective-range factors). The polar angle is uniform.
These inversions follow the <a href="https://geostat-framework.readthedocs.io/projects/gstools/en/stable/_modules/gstools/covmodel/models.html">GSTools radial distributions</a>,
rescaled to this library's effective ranges. Finite modes always set <code>status.approximation = 1</code>.
Spherical spectra are unsupported here and produce NaN/invalid rows; select FFT instead.
Nugget is independent per domain row, so coincident point rows are separate microscale samples.
When the partial sill is zero, spectral frequency generation and summation are skipped: a pure
nugget draw remains independent of range, including ranges whose unused frequencies would
overflow float32. For a positive partial sill, an unrepresentable sampled frequency invalidates
the realization (NaN values and zero validity); a nonfinite phase invalidates that point.
These checks run on the GPU each frame before trigonometric evaluation. Modes are never
silently dropped or clipped. Increase the range or rescale the coordinate frame to restore
representable frequencies and phases; the compiled graph can be reused after updating parameters.</p>
<p>Named construction limits: 2^24 domain rows, 1–4096 modes (default 256), and at most 2^30
mode-point products per run. All bindings and row-proportional scratch are checked against the
device storage binding limit before nodes are declared. The largest FFT complex binding is
32 MiB. The declared command count includes published FFT passes and an allowance for the
hierarchical negative-mass reduction; tests check the estimate. There is no iterative fitting,
loss history, or CPU twin: field evaluation is large data-parallel work.</p>
<h3 id="hard-conditioning">Hard conditioning</h3>
<p><code>conditioning</code> supplies observations on rows of the output domain: <code>indices</code> (uint32), <code>values</code>
(float32), optional <code>mask</code> (uint32), fixed query-major <code>neighborhoods</code> (uint32), and
<code>neighborCount</code> (1–32). Candidates are observation-row indices, distinct within a query;
<code>0xffffffff</code> pads unused slots. Obtain candidates from a resident spatial index. Every observed
query must include itself. Observed domain indices must be distinct, and conditioning requires
distinct coordinates across the whole domain (row-independent nugget noise otherwise differs
from the coordinate-based nugget covariance used by local kriging).
Invalid indices, masked observations and nonfinite values are skipped. Empty candidate lists
retain the unconditional prior. Failed local solves produce NaN and zero validity.</p>
<p>The contributor gathers <code>data - unconditionalField</code> at observed rows, runs the existing
<code>GPULocalKriging</code> in zero-mean simple mode with zero measurement error, then adds the kriged
residual to the unconditional field. Successful hard observations are finally restored from their input float32 values exactly.
The observation count is independent of the number of query rows and has no dense joint
4096-row cap. Conditioning always sets <code>status.approximation = 1</code> because the supplied neighborhood
can be local. For an exact conditional covariance, include all observations in every neighborhood
(up to 32) and use a non-clipped FFT embedding; spectral covariance remains approximate.
Conditioning outside the output domain requires adding those points to the point domain.
The <a href="../../../modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/gaussian-random-field-oracle.py">NumPy oracle</a>
uses a separate binary64 dense covariance solve, not the local-kriging implementation. Its
NumPy statistical checks also use 8,192 independent realizations per model and method and verify
the conditional covariance against a Schur complement.</p>
<h3 id="validation">Validation</h3>
<p>Checked-in NumPy fixtures cover every FFT model, spectral Gaussian/exponential/Matérn,
negative-eigenvalue mass, north-up non-square grids, residual conditioning, masked data and
empty padded neighbor slots. Headless tests also check replay, changing the realization in one
compiled graph, output padding, empty domains, unwritten parameters, and mean/variogram
confidence bounds over 256 realizations. The statistical bounds come from independent normal
and chi-square distributions at a two-sided error probability of 10^-6 (approximate for finite
spectral mixtures). No GPU execution is required to regenerate the small JSON fixture.</p>
<p>{/* parameters: GPUGaussianRandomField generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpugaussianrandomfield-parameters"><code>GPUGaussianRandomField</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUGaussianRandomFieldParameterValues(settings)</code>, writes <code>GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH</code> (16) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUGaussianRandomFieldParameterValues</code>:</p>
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
<td><code>variogram</code></td>
<td><code>GPUPredictionCovariance</code></td>
<td>Isotropic variogram, with the effective-range convention of fitVariogramModel.</td>
</tr>
<tr>
<td><code>seed</code></td>
<td><code>number</code></td>
<td>Full uint32 Philox seed. Packed as bits, never rounded to float32.</td>
</tr>
<tr>
<td><code>realization</code> (optional)</td>
<td><code>number</code></td>
<td>Full uint32 realization counter; defaults to zero.</td>
</tr>
<tr>
<td><code>mean</code> (optional)</td>
<td><code>number</code></td>
<td>Constant field mean; defaults to zero.</td>
</tr>
<tr>
<td><code>grid</code> (optional)</td>
<td><code>GPURasterMetadata</code></td>
<td>Axis-aligned planar grid. Must match the contributor dimensions; omit for points.</td>
</tr>
<tr>
<td><code>origin</code> (optional)</td>
<td><code>readonly [number, number]</code></td>
<td>Point-mode phase origin, subtracted before evaluating modes; defaults to [0, 0].</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};