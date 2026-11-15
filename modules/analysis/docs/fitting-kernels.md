# Shared fitting kernels

## Overview

Import these building blocks from `@luma-contributors/analysis/analysis-core`. They fit models supplied
by the caller; no model, pretrained weights or dataset is included. These are GPU building blocks,
without a CPU twin: their work is parallel and a CPU stage would force gradient/state readbacks.

Compile-time shapes and loop bounds are validated at construction. Mutable values live in graph
views. Each packed scalar binding is capped at 128 MiB and checked against the device at the start
of `getCommandNodes`. Fitted models have at most 100,000 parameters. Use minibatches to bound
row-gradient and encoded-feature storage. Encoder work is capped at 5 billion bounded operations.
The limits are planning bounds, not throughput measurements on additional devices.

## `GPUFittingGradientSums`

Takes `rowCount`, `parameterCount`, row-major scalar `gradients`, optional uint32 `mask`, and
`output.sums` (one float per parameter). A zero mask skips the row. Empty input writes zero sums.
Finite retained gradients are required; non-finite terms propagate. For a mean gradient, scale
contributions by the retained batch size in the derivative kernel. A loss can be reduced as a term.

One invocation sums one parameter over at most 256 rows. Separate dispatches merge adjacent tile
partials in a fixed binary tree, including odd final tiles. The node count is
`1 + ceil(log2(max(1, ceil(rowCount / 256))))` (at most 17). Padding invocations never access rows.
Results are repeatable on a device/compiler; floating-point reassociation and trig precision mean
there is no cross-device bit identity guarantee. No atomics or compensated-sum identity is used.

## `GPUAdamStep`

Implements [Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980):

`first = beta1 * first + (1 - beta1) * gradient`,
`second = beta2 * second + (1 - beta2) * gradient²`,
`weight -= learningRate * (first / (1 - beta1^t)) / (sqrt(second / (1 - beta2^t)) + epsilon)`.

The second-moment contribution uses a significand/exponent decomposition for gradients above
2^63, so a compiler cannot square the unscaled gradient before its small multiplier. The corrected
square root likewise divides bounded significands, takes a square root in a bounded range, and
restores half the exponent difference. Integer bit operations separate these calculations from
the original magnitudes and prevent floating-point reassociation across the decomposition.
For a first gradient of 1e20, the stored moments and the roughly 1e20 denominator are representable,
while both an unscaled square and `second / secondCorrection` would require roughly 1e40.
The final update also multiplies/divides bounded significands, combines the learning rate,
first-moment correction and denominator exponents, and restores only the final update's scale.
This preserves representable updates when `learningRate * correctedFirst` would overflow or
underflow. Scale restoration uses integer float32 encoding, with nearest-even rounding at
subnormal boundaries; device flushing of subnormal arithmetic results can still apply.

Pass finite current gradients, a scalar current loss and `parameters` packed by
`getGPUAdamStepParameterValues` (a float32 `GPUParameterBuffer` can host it). The packer accepts
`learningRate`, `beta1`, `beta2`, `epsilon`, `tolerance` and `skipZeroGradient`. Defaults are
0.001, 0.9, 0.999, 1e-8, zero (convergence disabled) and false. Values can change without recompiling.
Changing betas during a fit uses the new beta raised to the global step count; keep them fixed for
standard Adam. Zero betas are supported. Epsilon must be at least 1e-30.

Bias correction keeps the gap `C(t) = 1 - beta^t` directly, starting from `C(1) = 1 - beta`.
It composes blocks using `C(2n) = C(n) * (2 - C(n))` and
`C(n+m) = C(n) + (1 - C(n)) * C(m)`, by the bits of t (at most 14 rounds for the budget).
The sum has nonnegative terms and never subtracts an approximate power from one;
[WGSL pow accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy) inherits the error of
`exp2(t * log2(beta))`, which that subtraction would amplify near one. Headless regressions
compare both the corrections and single-step weights to NumPy at the first step, binary block
boundaries and the 10,000-step budget, including the largest float32 beta below one.

The output owns persistent `weights`, `firstMoment`, `secondMoment`, `lossHistory`, and required
uint32 status scalars `iterationCount`, `converged`, `iterationLimitReached`, `invalidCount`.
Use separate buffers. Initialize moments and all status scalars to zero and supply initial weights.
The history has `maximumIterations + 1` floats (budget 1–10,000). Entries through the latest evaluated
iterate are valid; later entries have unspecified contents. Reset state explicitly to start a new fit.

Each call emits two nodes: record the loss and decide whether to step, then an indirect-gated update.
The caller evaluates the model and its derivatives between steps. Call once more after the final
update with its loss to record `lossHistory[iterationCount]`; no additional update is performed.
A complete fit composes evaluation and step nodes for the budget, followed by this final evaluation
and step. Return the current weights even at the limit. The last update sets `converged=0` and
`iterationLimitReached=1`; a plateau detected at the final evaluation sets converged instead.
The helper itself never unrolls a fit or reads status back to the CPU. A composing contributor must
budget its whole graph and gate its evaluation nodes too, keeping each run to tens of dispatches.

A positive tolerance stops before an update when
`abs(loss - previousLoss) <= tolerance * max(1, abs(previousLoss))`. The implementation compares
a normalized relative change to tolerance, using a magnitude ratio and the losses' signs when
`abs(previousLoss) >= 1`. The ratio uses exponent/significand decomposition, so large opposite
losses cannot overflow the difference or tolerance product and falsely indicate convergence.
Same-sign losses use their finite difference before normalization, preserving small changes
between adjacent float32 values. For smaller previous losses the denominator is one and the
direct difference stays in range.
The current loss is recorded.
Finished calls preserve optimizer state. Invalid loss or packed parameters sets `invalidCount=1`
and skips the update. Gradients, weights and moments must be finite and second moments nonnegative;
they are not scanned by the control node. `skipZeroGradient` preserves weights and moments at zero
gradient, using the global bias-correction clock (not a sparse per-entry clock).

## `GPUFittingMinibatchOrder`

Takes `rowCount`, three uint32 `parameters` packed by
`getGPUFittingMinibatchOrderParameterValues([seedLow, seedHigh], epoch)`, and `output.order`.
A registered Philox 4x32-10 purpose generates one 32-bit key per original row; published stable
`GPUSort` sorts keys and original row IDs. Equal keys retain source order. Every row occurs exactly
once, even on collisions; finite-key ties introduce a small bias relative to an exactly uniform
permutation. Empty input is a no-op. Change the epoch for a different reproducible order. Slice
consecutive ranges of this permutation into minibatches in the model's GPU derivative kernel;
the final batch may be short. There is no row-sized CPU shuffling. The published sort has a
128-node conservative allowance, verified at the maximum row count by the node audit.

## `GPUFittingCoordinateEncoder`

Takes scalar-packed `coordinates`, `rowCount`, optional `mask`, `spatialContext`, and
`output.features`/`output.validity`. Invalid/masked rows are zero-filled with validity zero. The
encoding, row count, widths and spatial context are compile-time; coordinates, masks and caller
frequency vectors are per-frame. Output is row-major, one scalar float per feature.

For `encoding: 'fourier'`, specify `dimensions` (2 or 3), `frequencyCount` (1–256) and caller-supplied
`frequencies` shaped `[frequencyCount, dimensions]`, in cycles per normalized coordinate unit.
The output interleaves `[cos(2π b·x), sin(2π b·x)]` per frequency, unit amplitude, as in
[Tancik et al., page 2](https://arxiv.org/pdf/2006.10739). Use a planar context and normalized local
coordinates (with an optional normalized third time/height component). No CRS transformation,
normalization or frequency sampling is implicit. A non-finite coordinate or phase, or absolute
phase above 8192 radians, invalidates the whole row.

For `encoding: 'spherical-harmonics'`, specify `maximumDegree` (0–39) and explicitly supply a
longitude/latitude context with metric `none` or `great-circle`. Coordinates are longitude then
latitude in **degrees**, in [-180,180] × [-90,90]. There are `(maximumDegree + 1)²` features.
The degree-39 limit supports 1600 basis values but does not imply compatibility with a pretrained
model's basis convention. No pretrained model is supplied.

The complex basis is [DLMF 14.30.1](https://dlmf.nist.gov/14.30.E1), including the Condon-Shortley
phase. For each degree l, write orders m=-l..l at index `l*l+l+m`:

| Order | Real feature |
| --- | --- |
| m < 0 | sqrt(2) Im(Y_l^abs(m)) |
| m = 0 | Y_l^0 |
| m > 0 | sqrt(2) Re(Y_l^m) |

The associated Legendre recurrence is [DLMF 14.10.3](https://dlmf.nist.gov/14.10.E3), with
normalization carried through each step to avoid factorial overflow. The diagonal follows
[Rodrigues' formula](https://dlmf.nist.gov/14.7.E8). Latitude uses `sin(latitude)` and `cos(latitude)`
to preserve polar offsets. Both poles and both representations of the antimeridian are covered.
The NumPy/SciPy fixture generator under `test/analysis-core/fitting` compares this convention to
[`scipy.special.sph_harm`](https://docs.scipy.org/doc/scipy-1.15.1/reference/generated/scipy.special.sph_harm.html),
whose arguments are azimuth then colatitude in radians. Converting from another basis convention
is the caller's responsibility. GPU tolerances are stated in the headless specs.
