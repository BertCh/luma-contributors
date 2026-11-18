# Space-time Hawkes process (STM-08)

`GPUSpaceTimeHawkes` fits a self-exciting point process to a catalogue of events with a position and a time: how much of the activity is background, how many events each event triggers, how fast and how far the triggering reaches. It is the fixed-kernel space-time Hawkes model fitted by expectation-maximization (Veen and Schoenberg 2008, the "branching process" EM), with the parent search bounded in time and distance. ETAS (earthquake magnitudes, Omori-Utsu decay, parametric MLE steps) is later work and is not in this item.

```ts
import {
  getGPUSpaceTimeHawkesParameterValues,
  GPUSpaceTimeHawkes
} from '@luma-contributors/analysis/gpu-spatial-analysis';
```

The classes are in `src/gpu-spatial-analysis/space-time-hawkes/`.

## Model

For an event at place `s` and time `t` the conditional intensity is

```
lambda(s, t) = mu + sum over earlier events i of
               omega * beta * exp(-beta (t - t_i)) / (2 pi sigma^2) * exp(-|s - s_i|^2 / (2 sigma^2))
```

| Parameter | Meaning | Unit |
| --- | --- | --- |
| `mu` (`background`) | Background rate, constant in space and time | events per unit area and time unit |
| `omega` (`branchingRatio`) | Expected offspring per event; below one the process is stable | none |
| `beta` (`decayRate`) | Rate of the exponential time kernel; mean lag `1 / beta` | per time unit |
| `sigma` (`spatialStandardDeviation`) | Standard deviation of the isotropic Gaussian offspring displacement | position units |

Each iteration is one E-step and one M-step. The E-step gives every event `j` the probability `p_jj = mu / lambda_j` that it is background and `p_ij` that earlier event `i` triggered it. The M-step sets `mu = sum_j p_jj / (A T)`, `omega = sum_j (1 - p_jj) / sum_i (1 - exp(-beta (T - t_i)))`, `beta = sum p_ij / sum p_ij lag_ij` and `sigma^2 = sum p_ij d_ij^2 / (2 sum p_ij)`, with `A` the window area and `T` the duration. The sum over `i` in `omega` accounts for offspring that would fall after the end of the window.

What is not modelled, and what that costs:

- The spatial window edge: offspring that would fall outside the window are not accounted for, as the compensator uses the whole plane. Keep `sigma` small against the window; in the recovery spec (`sigma` 0.02 on a unit square) this biases `omega` low by about 3 percent.
- A background that varies in space or time (the research plan names a kernel-smoothed `mu(s)`); `mu` is one number.
- Pairs beyond `maximumLag` or `maximumDistance` are dropped and the kernel is not renormalized: choose them at several `1 / beta` and several `sigma`.
- Simultaneous events (equal times) do not trigger each other.
- Standard errors and any inference on the parameters. EM gives point estimates.

## `GPUSpaceTimeHawkes`

On GPU, squared initialSpatialStandardDeviation must be finite normal float32 and at least 2^-126.
The CPU reference retains its wider variance domain.

```ts
const hawkes = new GPUSpaceTimeHawkes({
  positions,        // GraphDataView<'float32x2'>
  times,            // GraphDataView<'float32'>, relative to the start of the window
  parameters,       // GraphDataView<'float32'> from getGPUSpaceTimeHawkesParameterValues
  maximumIterations: 100,
  maximumParents: 256,
  output: {
    estimates,      // 4 rows: background, branchingRatio, decayRate, spatialStandardDeviation
    logLikelihood,  // optional, 1 row at the parameters entering the last update
    fittedLogLikelihood, // optional, 1 row at the published estimates
    backgroundProbability, // optional, one row per event, in input row order
    status: {converged, iterationLimitReached, iterationCount, candidateOverflow, invalidCount, approximation}
  }
});
graph.add(hawkes);

getGPUSpaceTimeHawkesParameterValues({
  windowArea: 1,
  windowDuration: 400,
  maximumLag: 5,
  maximumDistance: 0.12,
  initialBackground: 1,
  initialBranchingRatio: 0.3,
  initialDecayRate: 1,
  initialSpatialStandardDeviation: 0.05
});
```

Per-frame (no recompile): the contents of `parameters` (window area and duration, `maximumLag`, `maximumDistance`, `tolerance`, the four starting values) and of `positions` and `times`. Compile-time (a new graph): the event count, `maximumIterations`, `maximumParents`, and which optional outputs exist. Extra props: `id`, `spatialContext` (planar/native only) and `temporalContext` (`float32-relative`, checked and exposed, nothing converted).

Inputs:

- `times` are float32 in `[0, windowDuration]`, relative to the start of the observation period, in the unit of the time kernel. A row with a non-finite position or a time outside the window is excluded and counted in `status.invalidCount`.
- Coordinates are float32. Subtract a nearby origin in float64 before upload when they are large (UTM metres); differences of large float32 values lose their low bits, and the origin cannot be removed on the GPU after rounding.
- Starting values must be positive. If they or the window are not, the estimates are NaN, `iterationCount` is 0 and neither `converged` nor `iterationLimitReached` is set.

Outputs and status:

- `estimates` hold the iterate after the last executed iteration. `fittedLogLikelihood` is evaluated at those estimates and is the diagnostic to use when comparing fitted models. Legacy `logLikelihood` remains the likelihood at the parameters entering the last update; even a converged fit can have a different value within its tolerance. Both compensators use the whole-plane spatial mass.
- `backgroundProbability` is `p_jj` at the published estimates; NaN for excluded rows or an invalid fit, including a catalogue consisting only of events at the observation-window endpoint. `1 - p_jj` is the probability that the event was triggered.
- `converged` is one when the largest relative change of any estimate in an iteration is at most `tolerance`; `iterationLimitReached` is one when `maximumIterations` ran out first. In that case the latest estimates are still published. `iterationCount` is the number of executed iterations.
- `candidateOverflow` is the number of events whose candidate parents inside the lag bound exceeded `maximumParents`: only the most recent `maximumParents` were examined for them, so their intensity is understated. A nonzero value means the bound should be raised (or the lag bound lowered).
- `approximation` is always one: the spatial edge and the parent bounds are intentional approximations.

The background compensator multiplies bounded mantissas and combines exponents before rescaling,
so a finite three-factor result survives intermediate overflow from changing area or time units.
The Kepler fit summary pairs estimates with `fittedLogLikelihood`; the optional entering-iterate
`logLikelihood` remains a separate contributor diagnostic.

Limits (named constants, checked with named messages): `maximumIterations` from 1 to `GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS` (1000); `maximumParents` from 1 to `GPU_SPACE_TIME_HAWKES_MAXIMUM_PARENTS` (4096; defaults `GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS` 100 and `GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS` 256); events from 2 to 2^24, the exact float32 integer range of the counts, and at most 8,388,608 for the 16-byte event table in a default 128 MiB binding. `estimatedCommandNodeCount` is an upper bound of the command nodes: six fixed nodes, an allowance of 64 for the published sort, and three nodes per iteration.

{/* lens:GPUSpaceTimeHawkes */}

### Parameter packer

`getGPUSpaceTimeHawkesParameterValues` packs `[windowArea, windowDuration, maximumLag, maximumDistance, tolerance, initialBackground, initialBranchingRatio, initialDecayRate, initialSpatialStandardDeviation]` (`GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH` is 9; `tolerance` defaults to `GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE`, 1e-4) and throws if a value is not positive and finite. `GPU_SPACE_TIME_HAWKES_ESTIMATE_LENGTH` is 4. `GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR` and `GPU_SPACE_TIME_HAWKES_BRANCHING_RATIO_FLOOR` are the floors described below.

{/* parameters: GPUSpaceTimeHawkes generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUSpaceTimeHawkes` parameters

- Packer: `getGPUSpaceTimeHawkesParameterValues(settings)`, writes `GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH` (9) `float32` values.

Settings of `getGPUSpaceTimeHawkesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `windowArea` | `number` | Area of the observation window, in squared position units. Positive and finite. |
| `windowDuration` | `number` | Length of the observation period in the unit of `times`. Event times lie in `[0, duration]`; an event outside it is excluded. Positive and finite. |
| `maximumLag` | `number` | Parents more than this many time units before an event are not considered. Positive. |
| `maximumDistance` | `number` | Parents farther than this many position units from an event are not considered. Positive. |
| `tolerance` (optional) | `number` | Largest relative change of any estimate between two iterations that counts as converged. Defaults to `GPU_SPACE_TIME_HAWKES_DEFAULT_TOLERANCE`. |
| `initialBackground` | `number` | Starting background rate, events per unit area and time unit. Positive. |
| `initialBranchingRatio` | `number` | Starting branching ratio, expected offspring per event. Positive. |
| `initialDecayRate` | `number` | Starting rate of the exponential time kernel, per time unit. Positive. |
| `initialSpatialStandardDeviation` | `number` | Starting standard deviation of the Gaussian offspring displacement, in position units. Positive. |

{/* /parameters */}

## How it runs

1. Events are stably sorted by time (a radix sort of the order-preserving float key) into a table of position and time; excluded rows get a NaN time and sort last.
2. For each event one thread binary-searches the first earlier event inside the lag bound and walks the candidates up to `maximumParents` back, the ordered pair enumeration of a time-sorted list used by the Knox tests. Pairs outside the distance bound or with a non-positive lag are skipped.
3. The intensity is accumulated in the log domain in one pass with a running maximum (log-sum-exp), together with the lag- and distance-weighted sums, so no division by a tiny `lambda` occurs.
4. Event statistics are summed in tiles of consecutive events in float32 and the tile partials by a fixed pairwise (cascade) sum, so results are bitwise repeatable on a device.
5. `maximumIterations` iterations are unrolled. A one-thread node tests convergence and disarms a GPU dispatch gate, after which the remaining iterations launch nothing.
6. When `fittedLogLikelihood` is requested, two final reduction nodes evaluate the log intensities and temporal mass at the published estimates. This same event evaluation writes `backgroundProbability` when requested, replacing its separate evaluation. The command-node estimate adds an allowance of two nodes. The operation form and CPU twin accept the same optional fitted-likelihood output.

Cost per iteration is `events * min(maximumParents, candidates in the lag bound)` exp evaluations. No benchmark has been run on a GPU for this item.

## Single precision: what was measured

The single-precision techniques come from Ko, Suchard and Holbrook (2024), who report a float32 kernel that runs reliably at one million events with log-sum-exp and a floor on tiny sums. They were measured here with [stm08-hawkes-float32.mjs](../../../dev-docs/research/spatiotemporal-review-2026-11-09/experiments/stm08-hawkes-float32.mjs), a CPU emulation of binary32 (`Math.fround` after every operation, denormals flushed) on a simulated catalogue of 987,262 events (unit square, `mu` 2, `omega` 0.5, `beta` 2, `sigma` 0.01, six EM iterations), against binary64 arithmetic on the same float32-rounded inputs:

| Technique | Measured effect |
| --- | --- |
| Log-sum-exp (running maximum) against a plain float32 sum of the terms | No difference at ordinary scales: both agree with binary64 to about 1e-8 per EM step. They differ only when the rates approach the float32 range. Rescaling positions so that `mu` and the offspring terms fall to about 1e-34 (position scale 1e17) leaves log-sum-exp at 1e-7 and the plain sum at 4e-6; at 1e18 it is 1e-7 against 3e-4 (the background responsibility off by 0.02); at 1e19 the plain sum gives NaN and log-sum-exp still fails because the rate itself is no longer representable. These scales are not physical, they locate where the plain sum fails. |
| Floor on the background rate before the logarithm | A floor of 1e-30 (the first choice here) ruined the fit at position scale 1e17 and beyond (relative error about 1), because the true rate was below it. The floor is therefore the smallest normal float32, `GPU_SPACE_TIME_HAWKES_INTENSITY_FLOOR`. With a start of `mu = 0` and no floor the logarithm is minus infinity; with the floor it is finite. |
| Tiled summation of the log-likelihood terms (about 244 events per tile, plain float32 inside) | One float32 accumulator over 987,262 events: relative error 5.5e-3. Tiles with the tile partials summed in float32: 1.5e-6. Neumaier compensation across tiles, on the CPU: 1.3e-6. The tiling does nearly all the work; the compensation adds little at this size, and on a GPU whose shader compiler reassociates float sums a float Neumaier term is removed altogether, so the tile partials are now merged by a fixed pairwise sum (about `log2(tileCount)` roundings of the total). |
| Float32 arithmetic of the E-step, all of the above in place | About 1e-8 relative error per EM step at one million events. |

The dominant error at one million events is not the arithmetic but the float32 time axis: rounding the inputs to float32 alone moved one EM step by 0.9 percent (relative, largest estimate) for a window of 250,000 time units with mean lag 0.5, because float32 spacing at 250,000 is 0.016. Choose the time unit and origin so that `windowDuration * 6e-8` is far below the shortest lag scale; a window longer than about 1e4 to 1e5 lag scales needs to be split in time. Position rounding is the same issue in space (subtract an origin in float64).

## Validation

- **Reference.** A binary64 EM written from the definition above (`stm08-hawkes-reference.ts`), checked against a hand calculation of a three-event catalogue. The GPU matches it to 7e-7 relative in the estimates after 1 to 3 iterations and 4e-7 in the log-likelihood; at convergence the stopping iteration can differ by one and the estimates differ by up to 4e-5 (specs allow 1e-5 and 3e-4).
- **External oracle: pending.** The ETAS R package (0.7.2, GPL) is the oracle named for this item. R is not installed where this was built, ETAS fits a magnitude-dependent model this operator does not have, and no ETAS fixture has been generated. Nothing here claims agreement with ETAS.
- **Recovery on simulated data with known parameters.** Catalogues of about 1,500 events are simulated from `mu` 2, `omega` 0.5, `beta` 2, `sigma` 0.02 on a unit square over 400 time units. The sampling distribution of the estimator is measured by fitting 24 independent simulations in binary64: standard deviations of 0.059 (`mu`), 0.016 (`omega`), 0.106 (`beta`) and 0.00037 (`sigma`), and means 2.018, 0.486, 2.028 and 0.0197, the 3 percent shortfall of `omega` being the unmodelled spatial edge. A GPU fit must lie within the absolute bias plus four standard deviations of the truth, a two-sided normal probability of about 6e-5 per parameter; three fits pass. Negative controls: a catalogue simulated with `omega` 0 returns `omega` below 0.1, and the generating background and triggered labels are separated by the background probability (mean 0.85 or more for background events, 0.25 or less for triggered events).
