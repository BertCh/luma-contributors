# Flow models on origin-destination aggregation

This is the canonical task guide. The [legacy URL](./flow-models-mob10.md) remains available.

Import from `@luma-contributors/analysis/gpu-network`.

Two contributors that work on zone-level flows: a parameter-free model that predicts flows from
populations and distances, and a kernel smoother for observed flows. Both are reference-tier: work
grows with the cube of the zone count, and construction rejects more than 1024 zones.

Both take zone centroids, planar or great-circle (`spatialContext`). Their inputs match what
`GPUFlowAggregation` writes: `zoneOutWeights` is the `outflows` of the radiation model when the
aggregation runs with `excludeSelfFlows: true` (`T_i` is the trips that leave zone `i`; without the
option the zone totals also count trips that stay inside it), and the
smoother reads `flowOriginZoneIds`, `flowDestinationZoneIds` and `flowWeights` directly, with
producer status `{count, overflow, pairOverflow}` (and `candidateOverflow` when available).
`status.count` bounds the readable compact pair prefix; forward both candidate-loss words when present.

## `GPURadiationFlowModel`

The radiation model (Simini, Gonzalez, Maritan and Barabasi, Nature 2012). A trip leaving zone `i`
is absorbed by zone `j` with probability `m_i n_j / ((m_i + s_ij)(m_i + n_j + s_ij))`, where `m_i`
and `n_j` are the populations of the origin and destination and `s_ij` is the population of the
zones strictly nearer to `i` than `j`, excluding both. The output is `T_i` times that probability,
the expected flow, laid out `flows[i * zoneCount + j]` with a zero diagonal. It draws no random
integers, unlike scikit-mobility's `Radiation.generate`.

| Prop | Meaning |
| --- | --- |
| `positions`, `populations`, `outflows` | One row per zone. `outflows` is `T_i`, the trips leaving each zone, not counting trips that stay inside it |
| `normalization` | `'none'` (default), or `'finite-size'`, which divides by `1 - m_i / M` so each row sums to `T_i` when no two destinations are at the same distance from the origin (with a tie the row sum can differ: zones at x = 0, -1, +1 with unit populations and `T_0` = 10 give 7.5 to each of the other two); evaluated as `M / (M - m_i)` with `M - m_i` summed over the other zones, so an origin holding nearly all of the population keeps its outflow |
| `output.flows` | `zoneCount^2` rows |

Positions must be valid in the chosen spatial context. CPU radiation rejects invalid zone
coordinates with the zone index. Resident invalid coordinates invalidate all origin rows:
`output.status.invalidInput` has one uint32 word per origin, off-diagonal flows become NaN and the
diagonal remains zero. The GPU operation exposes this as the optional named destination
`output.invalidInput` (`zoneCount` uint32 rows), lowered to the resident `output.status.invalidInput`.
Valid CPU operation calls requesting `output.invalidInput` receive a zero word per origin; invalid
CPU centroids still throw with the zone index. Operation names, options and spatial context are
captured at construction. In JavaScript, an explicit `normalization: null` resolves to `'none'`,
as does omission, in both graph operations and the resident/pure CPU forms. Other values outside
`'none'` and `'finite-size'` are rejected. Finite-size normalization independently sums the population outside the
origin, avoiding subtraction cancellation when the origin holds nearly all population.
Positions are planar context units or longitude/latitude degrees; populations and outflows are
nonnegative population and trip weights, and flows have the outflow unit.

Ties in distance: zones at exactly the distance of the destination do not count in `s_ij`.
Implementations that sort by distance count one of two tied zones and not the other, so they differ
there.

Verified against a hand-computed three-zone case, against a binary64 reference that sorts the
zones per origin and takes a prefix sum of population (a different route from the kernel), and
against the closed-form row sum `T_i (1 - m_i / M)`. scikit-mobility was not run.

{/* lens:GPURadiationFlowModel */}

## `GPUKernelFlowSmoothing`

Smoothing of OD flows: each flow is re-estimated as the kernel-weighted mean of the flows between
neighbouring origins and neighbouring destinations. Smoothing over neighbourhoods of both ends is the idea
of Guo and Zhu (IEEE TVCG 2014); this contributor does not reproduce their estimator. The definition used:

1. The neighbourhood of zone `i` is every zone within the smallest distance `h_i` at which the
   `locationWeights` of the zones within `h_i` (zone `i` included) sum to at least
   `neighborhoodSize`. Ties at `h_i` are included.
2. A neighbour `k` has the Gaussian weight `exp(-d_ik^2 / (2 sigma_i^2))` with
   `sigma_i = bandwidthRatio * h_i`; `K_ii = 1`. The kernel evaluates `exp(-(d_ik / sigma_i)^2 / 2)`
   and gives a zone at distance 0 weight one before any division, so a bandwidth too small for
   float32 keeps each zone's own flow instead of producing NaN.
3. `S_ij = sum_a sum_b K_ia K_jb M_ab / (sum_a K_ia * sum_b K_jb)`. Pairs without a flow count as
   zero.

This is a normalised kernel mean, so it does not conserve the total flow. Each row of weights is
divided by its own sum; the columns are not. A zone that is a neighbour of many zones spreads its flows
into all of them and the total grows, and a zone that is nobody's neighbour loses part of its flows. With
zones at x = 0, 1 and 10, unit weights, `neighborhoodSize` 2 and `bandwidthRatio` 0.5, one flow of 100
inside the middle zone sums to 125.3 after smoothing and one flow of 100 inside the far zone sums to 77.6.
Rescale the output when totals must match the input.

The neighbourhood size, the kernel and the bandwidth ratio are this library's own choices; the paper
was not available to check them and no packaged implementation exists. The tests use a binary64
reference written from the three points above, the three-zone case worked by hand, and closed-form
limits (a zero-size neighbourhood returns the input; a very wide kernel returns the mean). Flow
selection, which removes duplicate smoothed flows, is not provided.

| Prop | Meaning |
| --- | --- |
| `flows` | `{kind: 'matrix', matrix}` with `zoneCount^2` rows, or `{kind: 'pairs', originZoneIds, destinationZoneIds, weights, status: {count, overflow, pairOverflow}}` as written by `GPUFlowAggregation` (distinct pairs) |
| `locationWeights` | Weight of each zone in the neighbourhood size |
| `parameters` | Per-frame, `getGPUKernelFlowSmoothingParameterValues`: neighbourhood size, bandwidth ratio |
| `output.smoothed` | `zoneCount^2` rows |
| `output.radii` | Optional neighbourhood radius of each zone |

Pair imports with no producer use `{count, completeness: 'known-complete'}` only when those rows
are known complete. Producer status needs final `overflow` and available
`pairOverflow`/`candidateOverflow`; absent candidate evidence means unknown completeness unless
`status.completeness: 'no-candidate-loss'` explicitly guarantees it. Do not mark a bounded producer
as a known-complete import to bypass its loss flags.

Optional `output.status.complete` certifies the supplied pair population. With that port requested,
incomplete smoothing retains incumbent values and reports zero. Without it, incomplete input
suppresses the smoothed matrix with NaN. Dense matrix input is assumed complete. `radii` uses the
centroid distance unit; `smoothed` uses the input flow-weight unit and `locationWeights` determines
the unit of `neighborhoodSize`. `bandwidthRatio` is a fraction.

{/* lens:GPUKernelFlowSmoothing */}

{/* parameters: GPUKernelFlowSmoothing generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPUKernelFlowSmoothing` parameters

Import from `@luma-contributors/analysis/gpu-network`: `GPUKernelFlowSmoothing`.

- Packer: `getGPUKernelFlowSmoothingParameterValues(settings)`, writes `GPU_KERNEL_FLOW_SMOOTHING_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUKernelFlowSmoothingParameterValues`:

| Field | Type | Description |
|---|---|---|
| `neighborhoodSize` | `number` | Weighted size `p` of a zone's neighbourhood, in the unit of `locationWeights`: the nearest zones are taken until their weights sum to at least `p`. A value of 0 or less keeps only the zone itself (and zones at distance 0), which reproduces the input. |
| `bandwidthRatio` (optional) | `number` | Gaussian standard deviation as a fraction of the neighbourhood radius: `sigma = ratio * radius`. Defaults to 0.5. |

{/* /parameters */}
