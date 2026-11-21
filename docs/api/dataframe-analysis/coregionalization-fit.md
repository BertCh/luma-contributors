# GPUCoregionalizationFit

See also the [dataframe API](../gpu-dataframe-analysis.md#gpucoregionalizationfit).

Fit a linear model of coregionalization (LMC) to experimental direct and signed cross
variograms with fixed basic structures. Exported by
`@luma-contributors/analysis/dataframe-analysis`, together with
`GPUCoregionalizationFitOperation`, `CPUCoregionalizationFitOperation`,
`computeCoregionalizationFitOnCPU` and `fitLinearModelOfCoregionalization`.

The reference is [Goulard and Voltz (1992)](https://doi.org/10.1007/BF00893750).
The block update and PSD projection also appear in
[Loth and Baker, section 4.1, equations 36–38](https://www.jackwbaker.com/Publications/Loth_Baker_(2013)_Spatial_Cross_Correlation,_EESD.pdf).

## Input and algorithm

For `variableCount = p`, `semivariances` contains one packed upper triangle per lag:
`(0,0), (0,1), …, (0,p-1), (1,1), …, (p-1,p-1)`. Distances and pair counts have one
row per lag. All direct and cross variograms must use the same lag supports and
complete-case sample pairs. Assemble the triangles from `GPUVariogram` and
`GPUCrossVariogram` outputs using those common supports and masks. Negative cross
semivariances are retained. A nonfinite matrix entry, nonpositive/nonfinite lag,
or zero pair count excludes the whole bin. No missing cross entries are imputed.

The objective is `sum_h N_h ||Gamma_h - sum_s B_s g_s(h)||_F^2 / sum_h N_h`.
Off-diagonal errors count twice. Weights are shared across a lag matrix; pair-specific
weights, Cressie reweighting, and automatic model/range selection are outside this fit.

Start every `B_s` at zero. For each structure, fit the residual after removing the
other structures, then project the conditional symmetric matrix onto the PSD cone
by clipping negative eigenvalues to zero. Updates are sequential within a sweep.
Sufficient statistics keep each sweep independent of the number of lag bins. Dependent structures or too few distinct lag bins
can produce nonunique coefficients; only the summed covariance is identified.

## Parameters and outputs

`getGPUCoregionalizationFitParameterValues(structures, tolerance = 1e-5)` packs
`[tolerance, modelCode, nugget, sill, range, …]`. Families are `spherical`, `exponential`, `gaussian`, `matern-3/2` and
`matern-5/2`, with the existing kriging effective-range convention. Each basis is
`g(h) = nugget + sill * shape(h/range)` at positive lags. Unit-sill structures use
`sill: 1, nugget: 0`; a pure nugget uses `sill: 0, nugget: 1` and any positive range.
Model codes, nuggets, sills, ranges and tolerance are per-frame input values.
Counts, shapes and `maximumIterations` (default 128) are compile-time.
`GPUCoregionalizationFit` captures these options and all input/output/status views at
construction, including getter-backed fields. Its props and port maps are frozen snapshots;
later edits to the caller's objects do not change generated kernels or bindings. Views and
buffers retain their identities and their contents remain per-frame inputs. Construct a
new contributor to change the shape, budget or bindings.
No coordinates or spatial context are read: lags and ranges must use the same units.

Outputs `matrices` and `loadings` contain `structureCount * p * p` float32 elements,
structure-major and row-major. `B = L L^T`; `L` is a square spectral factor, not necessarily
triangular. Eigenvector signs and bases at repeated eigenvalues are arbitrary; compare
`B` or `L L^T` across implementations. Diagnostics contain
`[weighted objective, relative coefficient change, usable bins, total pair weight]`.
Relative change is the largest absolute coefficient change divided by the largest
absolute old/new coefficient over the whole sweep (zero when both are zero).

The required named uint32 scalar status ports are `validity`, `iterationCount`,
`converged`, `iterationLimitReached` and `invalidCount`. A budget exit publishes the
current PSD fit with validity 1, converged 0 and iterationLimitReached 1. Zero budget
publishes zero matrices and factors. Empty usable input, invalid parameters, a zero-energy
structure, or nonfinite arithmetic gives validity 0, convergence/limit flags 0 and NaN
coefficients/objective/change. Usable-bin and weight diagnostics remain available.
`invalidCount` counts excluded lag matrices. Output views may have trailing capacity;
only the documented result prefix is written. Views must be packed and output buffers distinct.

Validity does not imply convergence. For example, at the default tolerance `1e-5`,
at unit lag a unit spherical structure
fits the positive-definite matrix `[[2, -1], [-1, 3]]` on its first sweep. That sweep still
measures a nonzero coefficient change, so a one-sweep budget reports `iterationCount: 1`,
`converged: 0`, `iterationLimitReached: 1`. A second sweep measures zero change and reports
`iterationCount: 2`, `converged: 1`, `iterationLimitReached: 0`. A zero budget reports zero
iterations, no convergence and a reached limit while the valid input retains validity 1.
These status meanings apply to the contributor and both operation forms. Every execution
rewrites the result prefix, including clearing earlier nonzero flags when inputs change.
The stopping comparison is inclusive: with tolerance `1`, the first sweep's relative change
`3 / 3 = 1` already satisfies it, giving one iteration, convergence 1 and limit 0.

## CPUCoregionalizationFitOperation and supplied kriging model

Prefer the float64 CPU twin for small fits. It has the same graph ports and computes in
float64 with a single float32 rounding at output. The shared CPU exponential and Gaussian
shape evaluation uses `expm1` to preserve small lag/range values. NumPy fixtures include
Gaussian and both Matérn lag/range ratios of `1e-8` and `1e-9` with approximately identity
fitted matrices. The CPU Matérn shapes use a bounded positive exponential-remainder series
below argument `0.5`; scalar accuracy and fitted models are checked against independent
80-digit Python Decimal evaluation of the closed forms, including the series transition.
It drops GPU-only bin and sweep caps.
Float32 accumulation can change convergence counts and eigenvector bases; headless
specs compare matrices and reconstructed factors with 0.3% relative/absolute tolerance,
and long-fit iteration counts within three sweeps. Ordinary cases agree more closely.

```typescript
import {
  fitLinearModelOfCoregionalization,
  type CPUCoregionalizationFitInput
} from '@luma-contributors/analysis/dataframe-analysis';
import type {GPULocalKrigingProps} from '@luma-contributors/analysis/gpu-spatial-analysis';

// Three lags, each with the packed upper triangle (0,0), (0,1), (1,1).
const input: Omit<CPUCoregionalizationFitInput, 'parameters'> = {
  distances: Float32Array.of(50, 100, 200),
  semivariances: Float32Array.of(1, -0.2, 2, 2, -0.5, 3, 3, -0.8, 4),
  pairCounts: Uint32Array.of(20, 30, 40)
};

const fit = fitLinearModelOfCoregionalization(
  input,
  {
    variableCount: 2,
    structures: [
      {model: 'spherical', range: 1, sill: 0, nugget: 1},
      {model: 'exponential', range: 200, sill: 1}
    ],
    maximumIterations: 512
  }
);
// Inspect fit.validity[0], fit.converged[0] and fit.iterationLimitReached[0].
if (!fit.validity[0]) throw new Error('Coregionalization fit is invalid');
// Pass to GPULocalKriging (with positions, observations, supports and outputs):
const suppliedModel: Pick<
  GPULocalKrigingProps, 'mode' | 'variableCount' | 'covariance' | 'coregionalizationModels'
> = {
  mode: 'co-kriging',
  variableCount: 2,
  covariance: fit.coregionalizationModels[0].covariance,
  coregionalizationModels: fit.coregionalizationModels
};
```

`GPULocalKriging` also requires its `covariance` prop when no parameter buffer is supplied;
use the first fitted structure as above. The co-kriging model uses every fitted structure.
Invalid fits return an empty `coregionalizationModels` list. The GPU operation returns
factors in the same layout: read back each `p * p` slice to supply `loadings`.

## GPUCoregionalizationFitOperation

The graph operation uses `GPU_CONTRIBUTOR_OPERATION_LOWERING`; the CPU operation can
replace it using the same properties, graph names, formats and lengths.
Both operations reject empty graph names, duplicate output destinations and outputs that overwrite
an input. Errors use the operation ID, including a custom `id`.
Both operations capture their shape, iteration budget, input names and nested output/status
names at construction, including getter-backed properties. Later edits to the caller's objects
cannot change the declared contract or a subsequent execution/lowering. Construct a new
operation to change these options or ports; graph input values remain dynamic on every run.

## Reference-tier bounds and verification

GPU limits are 2–4 variables, 1–4 structures, 0–256 lag matrices and 0–1024 sweeps.
The existing co-kriging bounds are unchanged. Construction validates shapes and limits
with named helpers. Each sweep uses one gated fit kernel and one gate-update kernel;
finished sweeps dispatch no fit work. The exact node estimate is `2 * maximumIterations + 3`.
The maximum configuration has 2051 command nodes and at most eight storage bindings.

The independent NumPy fixture generator forms bin residuals directly and uses
`numpy.linalg.eigh`; its Matérn shapes are evaluated with 80-digit Python Decimal arithmetic
to avoid cancellation in the oracle itself. Node specs cover signed crosses, indefinite matrices, multiple
structures including 4-by-4 bounds, padding, empty bins and budget exits. Headless specs
compare every GPU/CPU output (factor Gram matrices for loadings), then feed a fitted LMC
to local co-kriging and check a hand-solved two-observation predictor and variance.
The maximal configuration is registered in the kernel audit.
