# GPUCoregionalizationFit

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
import {fitLinearModelOfCoregionalization} from '@luma-contributors/analysis/dataframe-analysis';

const fit = fitLinearModelOfCoregionalization(
  {distances, semivariances, pairCounts},
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
// Pass to GPULocalKriging (with positions, observations, supports and outputs):
const suppliedModel = {
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
