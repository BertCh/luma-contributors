# Track similarity: EDR, ERP, SSPD and bounded search

`GPUTrackSimilarity` computes, besides Hausdorff, discrete Frechet, DTW and LCSS, three more
measures on the same pair list, and `GPUTrackSimilaritySearch` can prune on cheap lower bounds
before it evaluates any of them. Tracks are planar or great-circle vertex lines without time.

| Output | Measure | Source |
| --- | --- | --- |
| `editDistanceOnRealSequence` | EDR | Chen, Ozsu and Oria, SIGMOD 2005 |
| `editDistanceWithRealPenalty` | ERP | Chen and Ng, VLDB 2004 |
| `symmetricSegmentPathDistance` | SSPD | Besse, Guillouet, Loubes and Royer, 2016 |
| `hausdorff` (existing) | symmetric Hausdorff | Shapely/GEOS, the one `movingpandas.Trajectory.hausdorff_distance` calls |

```ts
new GPUTrackSimilarity({
  linesA, linesB, pairA, pairB,
  editDistanceTolerance: 25,          // EDR match distance, context length unit
  editDistanceGapPoint: [0, 0],       // ERP gap point
  output: {
    editDistanceOnRealSequence: edr,
    editDistanceWithRealPenalty: erp,
    symmetricSegmentPathDistance: sspd
  }
});
```

## Definitions

- **EDR.** The least number of insertions, deletions and substitutions that turns one (densified)
  vertex sequence into the other. Two vertices match, at cost 0, when their distance is at most
  `editDistanceTolerance`; a substitution of non-matching vertices, an insertion and a deletion
  cost 1. The result is a whole number and is not normalized; divide by the longer vertex count for
  the normalized form. The match test is a metric ball like this package's `lcssTolerance`; Chen's
  original uses a per-axis box.
- **ERP.** A substitution costs the distance between the two vertices; an insertion or a deletion
  costs the distance of the vertex to the fixed `editDistanceGapPoint`. ERP is a metric. The gap
  point is `[x, y]` in planar contexts and `[longitude, latitude]` degrees on the sphere.
- **SSPD.** The mean over the original vertices of one track of the distance to the nearest point
  of the other track's polyline (the segments, not only the vertices), averaged over both
  directions. It is not densified and has no vertex cap.
- **Hausdorff** is the existing output, unchanged.

EDR and ERP run the anti-diagonal dynamic program of DTW with one extra table row and column for
the empty prefix, so the boundary prefix costs come from the same recurrence; they share the
shorter-track cap `maximumFrechetVertexCount` (`frechetCapExceeded` in `pairStatus`, NaN result)
and `densify`. SSPD uses 64 lanes per pair with fixed-order lane sums and a fixed binary tree (no
atomics). EDR is exact in f32; ERP sums in f32, so its error grows with `n + m` like DTW.

Hausdorff, EDR, ERP and SSPD are not in the MovingPandas 0.23 distance list except Hausdorff, DTW,
LCSS and Frechet, so EDR, ERP and SSPD are verified against double-precision definitions written
from the papers (see Verification).

## Lower bounds and pruned search

`GPUTrackSimilaritySearch` takes `metric` `'hausdorff'`, `'edit-distance-on-real-sequence'`,
`'edit-distance-with-real-penalty'` or `'symmetric-segment-path'` in addition to the previous
metrics. For each pair it first computes an `O(n + m)` lower bound from the original vertices and
writes it to `output.lowerBound`. Pairs whose bound exceeds the `threshold` (with a relative f32
margin of `1e-6`) get status 1 and skip the measure, and their distance is NaN. The survivors are
compacted and evaluated, and `topK` selects the exact stable top-k among the pairs within the
threshold, grouped by `pairA`.

| Measure | Lower bound | Why it never exceeds the measure |
| --- | --- | --- |
| Hausdorff | largest distance of a vertex of one track to the bounding box of the other | the other track lies in its box, so the distance to the box is at most the distance to the track |
| SSPD | the same with means | the mean of box distances is at most the mean of polyline distances |
| EDR | `max(n, m)` minus the fewer vertices that lie within the tolerance-widened box of the other track | each edit script costs at least `max(n, m)` minus its matches, and a vertex outside the widened box cannot match |
| ERP | absolute difference of the two sums of vertex distances to the gap point | ERP is a metric and the ERP of a track to the empty sequence is that sum (reverse triangle inequality) |
| Frechet, DTW | larger endpoint distance (existing) | the first and last vertices are always coupled |

The EDR and SSPD bounds subtract an `f32` slack proportional to `n + m` so that summed rounding
cannot push a bound above the evaluated distance; the ERP bound subtracts `(n + m) * 1.2e-7 *
(sumA + sumB)`.

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
