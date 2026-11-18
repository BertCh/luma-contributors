# Rate mapping recipes

`addExcessRiskRecipe(graph, props)` and `addSpatialRateRecipe(graph, props)` append GPU nodes
without submitting commands or reading results. Import them from `@luma-contributors/analysis/gpu-spatial-analysis`.
Both return ordinary recipe results: `outputs.excessRisk` or `outputs.spatialRates`, contributors,
intermediates and stage status. Excess risk reports `invalidCount` in an `excess-risk` stage;
spatial rates preserve the supplied
weights status in a `weights` stage. Pass `output` for a caller-owned readback-capable view;
otherwise the terminal view is graph-owned. Imported buffers remain caller-owned. Inputs are packed
`float32` or `uint32` graph views, with matching row counts; there is no implicit vector chunk packing.
For uint32 counts, GPU conversion to float32 loses integer precision above 2^24.

Both recipes prepare their complete declarations on a separate graph before changing the caller's
graph. Invalid shapes, aliases, foreign views and contributor expansion failures leave the caller's
nodes and resources unchanged, so a corrected configuration can reuse the same recipe ID. Preparation
uses the same device limits and preserves buffer aliases and view offsets; it never compiles or
allocates GPU storage. It repeats host declaration work once. This guarantee covers configuration
validation; existing caller node-ID collisions, a graph already frozen by compilation, and allocation
failures during final declaration are outside it. Supply unused recipe/resource IDs on a mutable graph.

```ts
const risk = addExcessRiskRecipe(graph, {events, populations, mask});
const spatial = addSpatialRateRecipe(graph, {
  events, populations, weights, mask, scale: 100000
});
```

Excess risk is dimensionless observed/expected:

```text
pooled = sum(events) / sum(populations)
risk[i] = events[i] / (populations[i] * pooled)
```

The sums include exactly rows with finite events, finite positive population and nonzero optional
mask. This uses a pooled ratio, never the arithmetic mean of row rates. A nonpositive expected count
produces NaN, including the all-zero event case. Excluded rows produce NaN. Finite negative events
are accepted consistently with the existing empirical-Bayes input contract; callers needing count
validation must provide it before upload. `GPUEmpiricalBayesRates` now exposes the same optional
`output.excessRisk`, reusing its fixed-order GPU reductions; its existing outputs and defaults are
unchanged. The recipe computes the existing moments even when only risk is requested.

Spatial rates use the focus row and its included nonself neighbors:

```text
spatial[i] = scale * (events[i] + sum(neighbor events)) /
                     (populations[i] + sum(neighbor populations))
```

`scale` is a finite nonnegative compile-time multiplier, default 1. Excess risk has no scale option.
`weights` is a square CSR structure: numeric weights are intentionally ignored (the GeoDa binary
neighborhood convention), including nonunit or zero stored weights. Self edges are skipped to count
the focus row once. Duplicate nonself slots count repeatedly; supply unique slots for a simple
adjacency graph. Out-of-range neighbors are skipped. Masked, nonfinite or nonpositive-population
neighbors are excluded. A focus row excluded by those same rules produces NaN.

The default `islandPolicy: 'undefined'` produces NaN when there is no included nonself neighbor,
including islands created by the mask. Set `'own-rate'` to return the focus rate in that case.
The existing `GPUSpatialEmpiricalBayesRates.output.spatialRates` retains its own-rate island
behavior. This recipe reuses that contributor and applies the explicit island/scale policy in a
final GPU pass. It does not average rates or reinterpret CSR numeric weights as exposure weights.
At most seven storage buffers are bound by the final pass, within default WebGPU limits. The
existing spatial contributor can do extra moment calculations; there is no duplicate rate kernel.

## Reference audit and limits

The reference is GeoDa commit
[`37510860bd9f7b8ae145c15759f7abc5ab217cf3`](https://github.com/GeoDaCenter/geoda/blob/37510860bd9f7b8ae145c15759f7abc5ab217cf3/ShapeOperations/RateSmoothing.cpp),
`RateSmoother_ExcessRisk`, `RateSmoother_SRS`, `RateSmoother_EBS` and `RateSmoother_SEBS`.
The audit confirmed pooled excess risk, binary neighborhood sums plus self, and undefined SRS
islands. Its undefined flag and zero placeholder are represented here by NaN. Its positive finite
input calculations agree with these formulas. This is an independent implementation using existing
luma kernels, with no upstream source copied.

For invalid populations, GeoDa can accumulate values before marking them undefined; these recipes
instead exclude invalid rows from both global totals and neighborhoods, preserving the existing
luma rate contract. Nonfinite inputs are likewise explicitly excluded. These are documented
extensions, not full GeoDa behavior parity for malformed input. GeoDa's EB standardized output
also differs from the existing Assuncao-Reis convention; that output is unchanged. Raw, global EB
and spatial EB remain available through the existing contributors; this addition supplies the two
missing named compositions only.

GPU tests use independent hand arithmetic for pooled rates, a deliberately unequal-population
example distinguishing pooling from average-of-rates, nonunit weights, self edges, mask-induced
islands, invalid rows, zero events and mutated-input replay on the same compiled graph. Existing
rate-smoothing suites retain their esda fixtures for regression coverage. All arithmetic is float32;
finite inputs whose totals or intermediate arithmetic overflow are outside the numerical parity
contract. There is no CPU fallback. Global recipes must receive the whole intended population:
executing independently per source batch changes the pooled estimator.

{/* lens:addExcessRiskRecipe */}

::lens{scenario=gpu-spatial-analysis.excess-risk-recipe rung=graph}

{/* lens:addSpatialRateRecipe */}

::lens{scenario=gpu-spatial-analysis.spatial-rate-recipe rung=graph}
