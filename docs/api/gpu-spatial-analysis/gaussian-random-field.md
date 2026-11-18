# Gaussian random fields

## `GPUGaussianRandomField`

{/* lens:GPUGaussianRandomField */}

::lens{scenario=gpu-spatial-analysis.gaussian-random-field rung=formula}

The synthetic lens records a seeded finite-mode field on a 4 by 4 planar lattice, including `output.status.approximation` and clipped eigenvalue mass.

Generates one stationary isotropic Gaussian realization on a planar grid or a packed set of
points. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. No datasets or weights
are included. A new seed or realization counter changes the draw without recompiling.

Provide exactly one of `grid` (published `GPURasterMetadata`) or `positions` (`float32x2`), a
`parameters` view of 16 float32 words, and caller-owned output views: `values` (float32),
`validity` (uint32), `status.approximation` (one uint32), and `clippedEigenvalueMass` (one float32).
`values` and `validity` have the same length, at least the domain row count. Extra rows are NaN
and invalid. All views are single packed views, on the target graph, with disjoint output buffers.
Host the packed parameters in a `GPUParameterBuffer`, or upload the packer's result to a
caller-owned storage buffer. Pass the same grid dimensions to the contributor and packer.
The grid's affine is read from the parameter buffer, so it can change each frame. Grid samples
are cell centers, respecting area/point interpretation and signed axis steps. Only native planar
Euclidean distances and axis-aligned grids are supported; project geographic coordinates first.
Generated grid coordinates stay relative to the first cell center for both spectral evaluation
and residual kriging. World-coordinate origins are never added to float32 grid positions, so
translating a grid preserves its spacing and realization, including when conditioned.
For points, center coordinates before float32 upload and set the phase `origin` near the data.

### Methods and limits

`method: 'auto'` uses the published `GPUFFT2D` for grids with both sides at most 1024, and
spectral randomization for larger grids or arbitrary points. `fft` and `spectral` can be selected
explicitly. Counts, method, mode count, coordinate context and conditioning topology are
compile-time; every view's contents, variogram, mean, seed, realization and affine are per-frame.

FFT embeds the covariance in a doubled power-of-two torus, capped at 2048 per side. With the
published unnormalized forward and normalized inverse transforms, the spectrum is multiplied
by `sqrt(N * max(lambda, 0))` and independent standard normal real/imaginary components.
The real inverse field has the requested covariance when the embedding is positive semidefinite.
This follows [circulant embedding](https://smai-jcm.centre-mersenne.org/articles/10.5802/smai-jcm.89/).
Large ranges can produce negative eigenvalues. They are clipped; `clippedEigenvalueMass` reports
`sum(max(-lambda, 0))/N`, the added variance. Any clipping sets `status.approximation = 1`; there is
no automatic embedding growth or silent claim of exactness. FFT supports spherical,
exponential, Gaussian and Matérn 3/2 or 5/2 variograms (including `matern` smoothness aliases).

The finite-mode method uses
`sqrt(sill/M) * sum(A*cos(k dot x) + B*sin(k dot x))` plus independent row nugget noise,
with independent standard normal amplitudes. Frequencies are drawn from the 2D covariance
spectrum, as in [GSTools RandMeth](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.field.generator.RandMeth.html).
For `U` uniform on (0,1), the radial inverse CDF is `sqrt(-12*log(U))/range` for Gaussian,
and `a*sqrt(U^(-1/nu)-1)/range` for exponential (`nu=1/2`, `a=3`) and Matérn
(`nu=3/2` or `5/2`, using the library's effective-range factors). The polar angle is uniform.
These inversions follow the [GSTools radial distributions](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/_modules/gstools/covmodel/models.html),
rescaled to this library's effective ranges. Finite modes always set `status.approximation = 1`.
Spherical spectra are unsupported here and produce NaN/invalid rows; select FFT instead.
Nugget is independent per domain row, so coincident point rows are separate microscale samples.
When the partial sill is zero, spectral frequency generation and summation are skipped: a pure
nugget draw remains independent of range, including ranges whose unused frequencies would
overflow float32. For a positive partial sill, an unrepresentable sampled frequency invalidates
the realization (NaN values and zero validity); a nonfinite phase invalidates that point.
These checks run on the GPU each frame before trigonometric evaluation. Modes are never
silently dropped or clipped. Increase the range or rescale the coordinate frame to restore
representable frequencies and phases; the compiled graph can be reused after updating parameters.

Named construction limits: 2^24 domain rows, 1–4096 modes (default 256), and at most 2^30
mode-point products per run. All bindings and row-proportional scratch are checked against the
device storage binding limit before nodes are declared. The largest FFT complex binding is
32 MiB. The declared command count includes published FFT passes and an allowance for the
hierarchical negative-mass reduction; tests check the estimate. There is no iterative fitting,
loss history, or CPU twin: field evaluation is large data-parallel work.

### Hard conditioning

`conditioning` supplies observations on rows of the output domain: `indices` (uint32), `values`
(float32), optional `mask` (uint32), fixed query-major `neighborhoods` (uint32), and
`neighborCount` (1–32). Candidates are observation-row indices, distinct within a query;
`0xffffffff` pads unused slots. Obtain candidates from a resident spatial index. Every observed
query must include itself. Observed domain indices must be distinct, and conditioning requires
distinct coordinates across the whole domain (row-independent nugget noise otherwise differs
from the coordinate-based nugget covariance used by local kriging).
Invalid indices, masked observations and nonfinite values are skipped. Empty candidate lists
retain the unconditional prior. Failed local solves produce NaN and zero validity.

The contributor gathers `data - unconditionalField` at observed rows, runs the existing
`GPULocalKriging` in zero-mean simple mode with zero measurement error, then adds the kriged
residual to the unconditional field. Successful hard observations are finally restored from their input float32 values exactly.
The observation count is independent of the number of query rows and has no dense joint
4096-row cap. Conditioning always sets `status.approximation = 1` because the supplied neighborhood
can be local. For an exact conditional covariance, include all observations in every neighborhood
(up to 32) and use a non-clipped FFT embedding; spectral covariance remains approximate.
Conditioning outside the output domain requires adding those points to the point domain.
The [NumPy oracle](../../../modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/gaussian-random-field-oracle.py)
uses a separate binary64 dense covariance solve, not the local-kriging implementation. Its
NumPy statistical checks also use 8,192 independent realizations per model and method and verify
the conditional covariance against a Schur complement.

### Validation

Checked-in NumPy fixtures cover every FFT model, spectral Gaussian/exponential/Matérn,
negative-eigenvalue mass, north-up non-square grids, residual conditioning, masked data and
empty padded neighbor slots. Headless tests also check replay, changing the realization in one
compiled graph, output padding, empty domains, unwritten parameters, and mean/variogram
confidence bounds over 256 realizations. The statistical bounds come from independent normal
and chi-square distributions at a two-sided error probability of 10^-6 (approximate for finite
spectral mixtures). No GPU execution is required to regenerate the small JSON fixture.

{/* parameters: GPUGaussianRandomField generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUGaussianRandomField` parameters

- Packer: `getGPUGaussianRandomFieldParameterValues(settings)`, writes `GPU_GAUSSIAN_RANDOM_FIELD_PARAMETER_LENGTH` (16) `float32` values.

Settings of `getGPUGaussianRandomFieldParameterValues`:

| Field | Type | Description |
|---|---|---|
| `variogram` | `GPUPredictionCovariance` | Isotropic variogram, with the effective-range convention of fitVariogramModel. |
| `seed` | `number` | Full uint32 Philox seed. Packed as bits, never rounded to float32. |
| `realization` (optional) | `number` | Full uint32 realization counter; defaults to zero. |
| `mean` (optional) | `number` | Constant field mean; defaults to zero. |
| `grid` (optional) | `GPURasterMetadata` | Axis-aligned planar grid. Must match the contributor dimensions; omit for points. |
| `origin` (optional) | `readonly [number, number]` | Point-mode phase origin, subtracted before evaluating modes; defaults to [0, 0]. |

{/* /parameters */}
