# Conditional LOSH residual inference

`GPULOSHResidualPermutationTest` combines descriptive local spatial heteroscedasticity
(LOSH) with a conditional residual-label randomization test. This is a distinct null
model. It does not implement esda LOSH inference and does not permute the original
observations and refit their local means on every draw.

## Statistic and null model

For finite, unmasked rows, compute a local positive-weight neighbor mean. Self links,
nonfinite or nonpositive weights and excluded neighbors do not contribute. A row with
no eligible neighbor has no residual. Set `r[i] = abs(y[i] - localMean[i]) ** a`, where
`a = residualPower` is 1 or 2 (default 2), and divide every finite residual by their global
mean. The optional `output.normalizedResiduals` exposes that field. A zero or nonfinite
global residual mean yields NaN residual outputs and no inferential tests.

The implementation accumulates weighted neighbor-minus-focus differences before division to
avoid forming a rounded local mean near the focus. Harmonic fields with exactly cancelling
differences retain zero residuals, including when the neighbors are unequal. Compensation retains
rounding errors from subtraction, multiplication and addition, and sums those corrections with
compensation too. For example, focus `d = 2^-26` with unit-weight neighbors `[2d, 1, 2d, -1]` has
exactly zero residual for either power; changing the focus to zero gives the nonzero residual
`d ** residualPower`. With `d = 2^-25` and `u = 1 - 2^-24`, focus `0.25 + d` with neighbors
`[1, 0.25, -0.5 + 2d, 0.25 + 2d]` and weights `[1, u, 1, u]` also has exactly zero residual;
weights `[1, 1, 1, u]` give the residual `2^-49 / (4 - 2^-24)` before the power.

The descriptive statistic is the weighted mean of eligible normalized neighbor
residuals. In a conditional draw for focus `i`, retain its residual and assign distinct
other finite residual labels without replacement to its active neighbor slots, in CSR
slot order. The graph, residual field, denominator and slot weights stay fixed. Finite
residuals remain eligible donors even when they have no eligible residual neighbor of
their own. Each row has its own conditional experiment; these draws do not constitute
one shared global permutation across all rows.

The implementation delegates randomization to the existing `GPULocalPermutationTest`
`localG` kernel. Its unnormalized weighted residual lag is a positive row-specific
multiple of LOSH, so ordering, ties and tail counts agree. The `localG` observed output
is deliberately not exposed as LOSH. The public `output.statistic` comes from the
descriptive LOSH contributor.

## Parameters and outputs

Supply `values`, `weights`, optional `mask`, optional `residualPower`, `parameters`,
`maximumPermutations`, optional `maximumNeighbors` (default 32, maximum 64), and an
explicit `alternative`: `greater`, `lesser` or `two-sided`. Construct `parameters` with
`getGPUPermutationParameterValues({seed, permutationCount, significanceLevel})`.

Seed is a nonnegative safe integer. Draws use the existing Philox counter stream and
partial Fisher–Yates sampler. Repeated inputs, seed and count reproduce the same draw
sequence; seed/count buffers can change between frames without rebuilding the graph.
The active count must be in `[1, maximumPermutations]`. Counts are never clamped.
Construction validates the composed inference contract, including parameter view format,
`maximumPermutations` in `[1, 1048576]` and the neighbor bound. Settings and port maps are captured
once; write resident buffer contents to update inputs between submissions.

For `P` draws, count `g = count(simulated >= observed)` and
`l = count(simulated <= observed)`. Ties enter both tails:

| Tail | Unadjusted pseudo p-value |
| --- | --- |
| `greater` | `(g + 1) / (P + 1)` |
| `lesser` | `(l + 1) / (P + 1)` |
| `two-sided` | `min(1, 2 * (min(g, l) + 1) / (P + 1))` |

Caller-owned `output.inference` contains `exceedances`, `pseudoPValues`, optional
`adjustedPValues`, optional `significant`, and `status.overflow`. The required
`output.status` is the separate per-row descriptive `GPU_LOCAL_VARIABILITY_STATUS`.
Optional `output.residualCount` reports the finite residual normalization population exactly.
Descriptive success does not imply inferential completion: inspect both statuses.
Untested rows have NaN p-values, exceedances `GPU_LOCAL_PERMUTATION_NOT_TESTED`, and
zero significance. Invalid permutation counts or incomplete upstream weights invalidate
all inference; exceeding the neighbor bound invalidates affected rows and sets overflow.
All output views must be mutually disjoint and disjoint from input views.

`multipleTesting` uses the shared adjustment contributor; its family is every finite
tested p-value. Excluded, undefined and untested rows do not enter the family. `none`,
Bonferroni, Holm, Benjamini–Hochberg and Benjamini–Yekutieli follow the existing API.
Bonferroni/Holm address familywise error; BY is the conservative FDR choice under
arbitrary dependence. BH requires the usual dependence assumptions. Any error-control
interpretation also requires the residual-label exchangeability null to be appropriate;
residuals obtained from overlapping fitted neighborhoods are not automatically
exchangeable under an arbitrary raw-observation null.

## Evidence and limits

`test/oracles/losh-backlog-inference.py` uses only Python standard-library exact rational
arithmetic and enumerates every ordered distinct donor assignment for a five-row graph.
Its fixture records descriptive residuals/statistics and exact inclusive null-tail
probabilities. GPU tests compare seeded Monte Carlo estimates with those exhaustive
probabilities, verify Bonferroni composition, and exercise undefined residuals,
upstream incompleteness, invalid counts and neighbor-capacity failure. Node tests parse
maximal-option kernels and enforce the default eight storage-binding limit and output
alias rejection. These tests establish this conditional composition, not parity with
an external LOSH inference library or multivariate joint randomization.

`permutation-handoff-audit.oracle.py` independently constructs exact rational residuals and an
arbitrary-integer Philox stream. The adjacent headless audit supplies literal seeded exceedances,
p-values and significance for powers 1/2 and all three tails, plus every descriptive/status output,
affine copies and candidate-loss recovery. WebGPU execution of these new cases remains a lead gate.

All numerical stages remain on the GPU, with no hidden readback or CPU fallback.
The descriptive LOSH denominator uses parallel maximum, count and sum reductions; its
scaled sum avoids overflow of the unscaled residual total. A resident `residualPower` parameter
view can change the descriptive power between encodings. Multivariate joint inference and
raw-observation LOSH refitting are outside this API.
