# Track similarity: EDR, ERP, SSPD and bounded search

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

`GPUTrackSimilarity` computes, besides Hausdorff, discrete Frechet, DTW and LCSS, three more
measures on the same pair list, and `GPUTrackSimilaritySearch` can prune on cheap lower bounds
before it evaluates any of them. Tracks are planar or great-circle vertex lines without time.
The contributor and `computeTrackSimilarityOnCPU` require `pairB.length` to equal `pairA.length`.
A mismatch reports the expected and received lengths.

Multipart `featureOffsets` is accepted by the contributor, GPU/CPU operations and direct CPU
function for Hausdorff, maximum distance and SSPD. Pair indices select logical features. Distances
use the union of the parts, with no connecting segments; densification stays within each part.
SSPD averages over all original vertices of each feature, including repeated part endpoints.
Empty features produce NaN distances and `emptyTrack`; invalid/inactive pair rows produce NaN
and `invalidPair`. The operations accept an optional graph-value `featureOffsets` on either line
input. Their descriptors are captured at construction.
Entire empty coordinate columns with `lineOffsets=[0]` and `featureOffsets=[0,0]` denote one
empty feature: track outputs publish NaNs/status 1; search publishes NaNs/status 2 and `matched=0`.
Terminal-only `lineOffsets=[0]`, `featureOffsets=[0]` denote zero features; a requested pair row 0
is absent (base status 4), while an empty pair list has no output rows. Generated
geometry can have spare part capacity. CPU forms use the terminal feature offset as its active
part count and require remaining line offsets to repeat the terminal vertex offset.

Frechet, DTW, EDR, ERP and LCSS require ordered, continuous traversals. A multipart geometry does
not specify the order, direction or connection between disconnected parts. Those measures refuse
`linesA.featureOffsets` / `linesB.featureOffsets`; supply explicit single-part tracks with the
chosen traversal, or select a set metric. This decision applies to the direct CPU function too;
its default request for every measure therefore requires explicit `measures` for grouped inputs.

`GPUTrackSimilaritySearch` accepts the same grouping for `metric: 'hausdorff'` and
`'symmetric-segment-path'`. Bounds use each feature's union box; exact evaluation uses its
disconnected parts. Top-k indices and summaries are grouped by logical feature rows, including
empty features. Sequence-metric search has the same explicit-traversal refusal. Search has no
separate operation form or CPU twin.

| Output | Measure | Source |
| --- | --- | --- |
| `editDistanceOnRealSequence` | EDR | Chen, Ozsu and Oria, SIGMOD 2005 |
| `editDistanceWithRealPenalty` | ERP | Chen and Ng, VLDB 2004 |
| `symmetricSegmentPathDistance` | SSPD | Besse, Guillouet, Loubes and Royer, 2016 |
| `hausdorff` (existing) | symmetric Hausdorff | Shapely/GEOS, the one `movingpandas.Trajectory.hausdorff_distance` calls |

```ts
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUTrackSimilarity} from '@luma-contributors/analysis/gpu-spatial-analysis';

export function addTrackSimilarityMeasuresExample0(graph: GPUCommandGraph, inputs: {
  linesA: NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesA']>;
  linesB: NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesB']>;
  pairA: NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['pairA']>;
  pairB: NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['pairB']>;
  edr: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['editDistanceOnRealSequence']>;
  erp: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['editDistanceWithRealPenalty']>;
  sspd: NonNullable<NonNullable<ConstructorParameters<typeof GPUTrackSimilarity>[0]['output']>['symmetricSegmentPathDistance']>;
}) {
  const {linesA, linesB, pairA, pairB, edr, erp, sspd} = inputs;
  graph.add(new GPUTrackSimilarity({
    linesA, linesB, pairA, pairB,
    editDistanceTolerance: 25,          // EDR match distance, context length unit
    editDistanceGapPoint: [0, 0],       // ERP gap point
    output: {
      editDistanceOnRealSequence: edr,
      editDistanceWithRealPenalty: erp,
      symmetricSegmentPathDistance: sspd
    }
  }));
}
```

With gap `(0,0)`, `[(1,0),(2,0)]` and `[(1,0),(0,0),(2,0)]` have ERP zero:
inserting the gap vertex costs zero. Distinct tracks can compare equal; `erp === 0` is not a
safe deduplication test. The reverse-triangle lower bound remains valid.

## Definitions

- **EDR.** The least number of insertions, deletions and substitutions that turns one (densified)
  vertex sequence into the other. Two vertices match, at cost 0, when their distance is at most
  `editDistanceTolerance`; a substitution of non-matching vertices, an insertion and a deletion
  cost 1. The result is a whole number and is not normalized; divide by the longer vertex count for
  the normalized form. The match test is a metric ball like this package's `lcssTolerance`; Chen's
  original uses a per-axis box.
- **ERP.** A substitution costs the distance between the two vertices; an insertion or a deletion
  costs the distance of the vertex to the fixed `editDistanceGapPoint`. ERP is a pseudometric on raw sequences when the gap point is a permitted vertex. The gap
  point is `[x, y]` in planar contexts and `[longitude, latitude]` degrees on the sphere.
- **SSPD.** The mean over the original vertices of one track of the distance to the nearest point
  of the other track's polyline (the segments, not only the vertices), averaged over both
  directions. It is not densified and has no vertex cap.
- **Hausdorff** is the existing output, unchanged.

Planar Fréchet decisions, EDR matches and LCSS matches use inclusive Euclidean membership on the
stored coordinates, before rounding a displayed norm. Clear comparisons take the fast path;
ambiguous GPU comparisons expand the original endpoint products into exact integer sums, so
rounding the coordinate subtraction cannot erase a separation. At radius 1, both
`(-2^-25,0) → (1,0)` (exact distance `1+2^-25`) and `(0,0) → (1,2^-13)` (exact squared distance
`1+2^-26`) are outside. Each singleton pair can display Fréchet `1` while `frechetWithin=0`,
EDR is `1` and normalized LCSS is `0`. `(0,0) → (1,0)` is exact equality and is included:
`frechetWithin=1`, EDR `0`, normalized LCSS `1`. Direct and named forms use the same rule;
search uses the producer's Fréchet decision and the same LCSS membership in its possible-match bound.
This agreement applies when the predicates receive the same vertex coordinates.
Virtual densified vertices can differ between CPU and GPU: the CPU interpolates in float64,
while GPU virtual vertices use float32 arithmetic. Near a threshold, these different vertices
can change Fréchet, EDR and LCSS decisions even though both predicates compare their inputs
correctly. For example, parallel segments `[(1048576,0),(1048576.125,4)]` and
`[(1048576.125,0),(1048576.25,4)]`, with `densify=0.5` and tolerance `0.125`, have CPU
midpoints separated by `0.125`, but GPU midpoint x coordinates round to `1048576` and `1048576.25`.
The CPU returns within 1, EDR 0, LCSS 1; GPU interpolation gives within 0, EDR 1, normalized
LCSS `2/3` (rounded to float32). Original-vertex planar comparisons share the stored-coordinate rule;
geographic comparisons also retain backend differences in distance arithmetic.

For geographic Hausdorff and SSPD, projection and interpolation use one wrapped start-to-end
longitude arc. The nearest-segment candidate is compared with both endpoints; its distance never
exceeds either endpoint distance. This holds for queries on the opposite side of the globe as well
as for queries near the antimeridian. For example, the point `(0°, 0°)` and segment
`(100°, 0°) → (-100°, 0°)` have Hausdorff and SSPD distances of `100° × sphereRadius × π / 180`
with default densification. The local projection remains an approximation for long edges.

EDR and ERP run the anti-diagonal dynamic program of DTW with one extra table row and column for
the empty prefix, so the boundary prefix costs come from the same recurrence; they share the
shorter-track cap `maximumFrechetVertexCount` (`frechetCapExceeded` in `pairStatus`, NaN result)
and `densify`. SSPD uses 64 lanes per pair with fixed-order lane sums and a fixed binary tree (no
atomics). EDR is exact in f32; ERP sums in f32, so its error grows with `n + m` like DTW.

The CPU twin creates densified points only for requested measures that use them. Requests limited
to `maximumDistance` and/or `symmetricSegmentPathDistance` use the original vertices, even with a
small positive `densify`. For measures that densify, each expanded track must have at most
`2^32 - 1` vertices (the JavaScript array length limit); the CPU checks that count before
allocating the dense points. An explicit `maximumFrechetVertexCount` also binds CPU dynamic-program
measures: beyond the shorter densified track cap, status is `frechetCapExceeded` and their distances
are NaN (decisions are zero). Hausdorff, SSPD and maximum distance remain available. Omitting that
prop leaves the CPU shorter-track count uncapped.
The GPU also checks expanded counts only when a requested measure uses densification, and checks
the shorter-track cap only when a dynamic-program measure is requested. A maximum-distance-only
or SSPD-only request therefore remains valid when an unused expanded count would exceed uint32.

Hausdorff, EDR, ERP and SSPD are not in the MovingPandas 0.23 distance list except Hausdorff, DTW,
LCSS and Frechet, so EDR, ERP and SSPD are verified against double-precision definitions written
from the papers (see Verification).

## Lower bounds and pruned search

Empty positions with `lineOffsets=[0]` and `featureOffsets=[0]` contain zero tracks;
`featureOffsets=[0,0]` contains one empty track.

`GPUTrackSimilaritySearch` takes `metric` `'hausdorff'`, `'edit-distance-on-real-sequence'`,
`'edit-distance-with-real-penalty'` or `'symmetric-segment-path'` in addition to the previous
metrics. For each pair it first computes an `O(n + m)` lower bound from the original vertices and
writes it to `output.lowerBound`. Pairs whose bound exceeds the `threshold` (with a relative f32
margin of `1e-6`) get status 1 and skip the measure, and their distance is NaN. The survivors are
compacted and evaluated, and `topK` selects the exact stable top-k among the pairs within the
threshold, grouped by `pairA`.

| Measure | Planar lower bound | Great-circle lower bound | Why it never exceeds the measure |
| --- | --- | --- | --- |
| Hausdorff | largest distance of a vertex of one track to the bounding box of the other | `0` | planar tracks lie in their boxes; geographic zero is conservative and does not prune |
| SSPD | the same with means | `0` | planar mean box distances cannot exceed mean polyline distances; geographic zero does not prune |
| EDR | `max(n, m)` minus the fewer vertices that lie within the tolerance-widened box of the other track | `abs(n - m)` | planar vertices outside the widened box cannot match; every extra geographic vertex costs an edit |
| ERP | absolute difference of the two sums of vertex distances to the gap point | the same with great-circle gap distances | ERP obeys the reverse triangle inequality even as a pseudometric; its distance to the empty sequence is that sum |
| Frechet, DTW | larger endpoint distance | larger great-circle endpoint distance | the first and last vertices are always coupled |

Here `n` and `m` are original vertex counts. Separated geographic singleton tracks have EDR
lower bound `abs(1 - 1) = 0`, even if the evaluated EDR is 1.
Planar EDR widens its box by `tolerance * 1.000001` and publishes an integer count difference;
it does not subtract summation slack. The SSPD and ERP bounds account for summed-distance
rounding: planar SSPD multiplies by `max(0, 1 - (n + m) * 1.2e-7)`, and ERP subtracts
`(n + m) * 1.2e-7 * (sumA + sumB)` in either context.

Pruning is exact: it only skips pairs that cannot be within the threshold. The threshold is the
caller's. To get the k nearest tracks without a known radius, run a first search with a generous
threshold or take the k-th distance of an earlier frame, widen it by about `1e-4` for f32 rounding,
and search again; a threshold-free adaptive top-k that tightens the radius while it runs is not
implemented.

Why this matters: a 2024 study (Ni et al., PVLDB 2024, arXiv 2311.00960) measured batched GPU DTW
over 512 pairs at only about 4.5 times a CPU (28.11 s against 6.21 s), so the pairs that never
reach the dynamic program are the larger saving than a faster program.

## Verification

- `test/oracles/track-similarity-edit-measures.py` (MovingPandas 0.23.0, Shapely 2.2.0, Python
  3.15) writes `fixtures/track-similarity-edit-measures.json`. `hausdorff` and
  `dynamicTimeWarping` come from `Trajectory.hausdorff_distance` and `Trajectory.dtw_distance` and
  are asserted equal to Shapely and to an independent DTW program; EDR, ERP and SSPD are written
  from the paper definitions in recursive head/rest form, which differs from the GPU's table.
- Specs in `test/gpu-spatial-analysis/track-similarity/`: the GPU against the fixture (planar and
  great circle, tracks across the 256-lane strip boundary, cap status), fixed-order SSPD
  repeatability, lower bounds at most the true distance for 404 random and degenerate pairs in
  double precision and on the GPU, and pruned top-k equal to the unpruned top-k and to a CPU
  reference.

### Operation forms

`GPUTrackSimilarityOperation` and `CPUTrackSimilarityOperation` accept logical planar or
longitude/latitude great-circle line
positions/offsets for one or two line sets, pair-index columns, explicit `pairCount`, optional
active-pair count, and optional Fréchet epsilon. Select distance/relationship outputs by name;
`pairStatus` is the per-pair bit set that distinguishes invalid/empty/capped results. Inactive
pairs share the invalid bit with invalid indices: use `activePairCount` to identify the
inactive suffix. The CPU honors an explicit `maximumFrechetVertexCount` for dynamic-program
measures; omitting it removes that CPU cap. CPU/GPU rounding may differ. The GPU operation compiles with
`GPU_CONTRIBUTOR_OPERATION_LOWERING`; it is the base metric form of `GPUTrackSimilarity`.

For decision-only Fréchet requests, inspect `pairStatus` before using the decision: a zero
relationship can mean invalid, empty or capped input rather than a measured rejection.
`activePairCount` belongs to base similarity and its operations; it identifies that base pair
list's inactive suffix. `GPUTrackSimilaritySearch` has no `activePairCount` prop. For LCSS search,
request `output.similarity` and `output.upperBound`; for distance-metric searches request
`output.distance` and `output.lowerBound`. Ports inactive for the selected metric are rejected
at construction (for example `output.distance` with `'longest-common-subsequence'`). All search
pairs are supplied explicitly; top-k summarizes evaluated matches and its own capacity/status.

Read `decodeGPUTrackSimilarityPairStatus` on base pair-status readback. Bit 8 is `invalidGeometry`;
bit 16 is `invalidDecisionThreshold`. An identical pair with negative epsilon can have distance 0,
`frechetWithin` 0 and status 16: the decision is unavailable, not a measured rejection. Use
`decisionValid` before interpreting the decision and the decoder's measure-validity fields for distances.
