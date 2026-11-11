# Space-time clustering

`GPUSpatialClustering` (DBSCAN) and `GPUOPTICS` take an optional `times` column that turns them into
ST-DBSCAN and ST-OPTICS. The time test is one more condition on the neighbor relation, so cores,
clusters, borders, noise, reachability and extraction follow from the same kernels as the purely
spatial versions. Without `times`, nothing changes: the generated kernels contain no time code.

ST-HDBSCAN is not available. See [Not available: ST-HDBSCAN](#not-available-st-hdbscan).

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

## Time input

| Prop | Meaning |
| --- | --- |
| `times` | One `float32` per row: ticks after the origin of `temporalContext`. Typically the output of `GPUTimeWordsRelative`, which takes the difference to the origin exactly on Int64 words. |
| `temporalContext` | A [`GPUTemporalContext`](./temporal-context.md) describing `times`. Required with `times` and rejected without. Only `encoding: 'float32-relative'` is read (a bigint origin as `epoch`); any unit, time zone and axis. Nothing is converted. |

The window is given in the unit of `temporalContext`:

- `GPUSpatialClustering`: `timeEpsilon` in the packed parameters (`getGPUSpatialClusteringParameterValues({bounds, epsilon, minimumPoints, timeEpsilon})`, stored at index 6 of the existing 8-element layout, 0 when omitted). It is per-frame: rewrite it without recompiling. `times` is compile-time presence, per-frame contents.
- `GPUOPTICS`: `maximumTimeDistance`, compile-time, positive and finite.

Times are float32. Differences are exact for `|dt| < 2^24` ticks, so choose an origin near the data and a unit coarse enough that the span stays inside that range when exact thresholds matter. The window is stored as float32 too.

## Definitions

Two rows are related when `distance <= epsilon` (the existing closed spatial test) and
`|t_a - t_b| <= timeEpsilon` (closed as well, so a difference equal to the window is related). A row
relates to itself.

- ST-DBSCAN: a point is core when its related neighborhood holds at least `minimumPoints` points, itself included. Clusters are the connected components of core points under the relation, the root of a cluster is its smallest core row, a border point joins the adjacent cluster with the smallest root, and compact labels rank the roots in row order. This is the canonical labeling of `GPUSpatialClustering`, applied to the time-filtered relation.
- ST-OPTICS: core distances, the frontier, reachability, predecessors and extraction (`dbscan` or `xi`) use only related pairs, so a pair outside the window behaves as if it were farther than `maximumDistance`. A hard filter on the neighbor graph.
- A row with a non-finite time has no neighbors: in ST-DBSCAN it is noise (not core even with `minimumPoints: 1`), and in ST-OPTICS it is invalid like a non-finite position and counted in `status[2]`.
- `denseBoxShortcut` is rejected with `times`, because a dense box proves spatial closeness only.

```ts
import {GPUTimeWordsRelative} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUSpatialClustering, getGPUSpatialClusteringParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

const converter = new GPUTimeWordsRelative({
  timestamps,                       // uint32x2 Int64 milliseconds
  temporalContext: {unit: 'millisecond', epoch: 'unix', encoding: 'int64-words', axis: {kind: 'irregular'}},
  origin: 1_700_000_000_000n,
  output: relativeTimes             // float32
});
graph.add(converter);
graph.add(new GPUSpatialClustering({
  positions,
  times: relativeTimes,
  temporalContext: converter.outputTemporalContext,
  parameters,
  gridSize: [256, 256],
  output: {labels}
}));
parameters.write(getGPUSpatialClusteringParameterValues({
  bounds, epsilon: 50, minimumPoints: 5,
  timeEpsilon: 6 * 3600 * 1000      // six hours, in the context unit (milliseconds)
}));
```

## Cost

ST-DBSCAN adds one gather of `times` into grid-cell order and one float compare per scanned
candidate. The grid is spatial only, so the scan visits every spatial candidate and a short window
does not prune work; a time-sorted or three-dimensional index would, and is a separate item.
ST-OPTICS keeps the frontier of `GPUOPTICS` and its limits: at most 4096 rows with the exhaustive reachability step or xi extraction, at most 65,536 with `neighborIndex: 'grid'`, whose radius list also applies the time window.

## Reference and agreement

- ST-DBSCAN: `st-dbscan` 0.2.3 (MIT), `ST_DBSCAN(eps1, eps2, min_samples).fit(X)` with `X = [time, x, y]`. Below 20000 rows it builds `where(|dt| <= eps2, euclidean, 2 * eps1)` and runs scikit-learn `DBSCAN(eps=eps1, metric='precomputed')`, the same relation as above. The fixture `test/oracles/fixtures/space-time-clustering.json` (generator `test/oracles/space-time-clustering.py`) holds its labels for five scenes on integer lattices, which exercise the closed boundaries. The GPU labels match them exactly, and an independent double-precision CPU implementation in the spec reproduces the labels, core flags, roots and cluster count.
- From 20000 rows `st-dbscan` 0.2.3 switches to a sparse path. On one 20000-row lattice scene it labeled 3733 points as clustered, where the dense definition above gives 4983 (and 2311 if a point did not count itself), so the sparse path is a different computation. The GPU follows the dense definition and is not compared against the sparse path.
- ST-OPTICS: the `st-clustering` 0.1.0 `ST_OPTICS` matrix (distance `2 * eps1` outside the window) run on scikit-learn `OPTICS(metric='precomputed')`. That matrix equals the hard filter whenever `maximumDistance < 2 * eps1`; with a larger `max_eps` the library keeps out-of-window pairs at distance `2 * eps1` instead of removing them, which `times` does not reproduce. `st-clustering` itself does not import with current joblib (`Memory(cachedir=...)`), so the fixture reproduces its matrix with scikit-learn. Ordering, predecessors and DBSCAN-extraction labels match exactly on two scenes, reachability and core distances to 1e-5 relative, and the same CPU oracle agrees.

The pinned environment (Python 3.15.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1) differs from the 3.14 pins of `requirements.txt`. `st-dbscan` 0.2.3 is published as a source distribution only and its pyscaffold build fails under pip on Python 3.15, so the generator imports it from the unpacked sdist (SHA-256 `ca1810025630571aa9ffe204f81d9c0a52dc29daa2d05bb2cd7941f56f32fd8d`, recorded in the fixture).

## Not available: ST-HDBSCAN

`GPUHierarchicalDensityClustering` cannot take the time filter by adding a predicate:

- Its core distances come from `GPUNearestFeatureJoin` neighbors mode, which has no per-pair predicate.
- Its Boruvka minimum spanning tree walks a point BVH with spatial boxes and prunes a node when `max(core, box distance)` exceeds the best weight. With a hard time filter, pruning would also need a time extent in every node; with the `st-clustering` penalty (`2 * eps1` outside the window) a pair can be nearer than its box distance, which makes the spatial lower bound invalid.
- A hard filter makes the graph a forest, and the dendrogram, condensed tree and selection assume one spanning tree.

Doing this needs changes in `spatial-join` and the hierarchical-density kernels and a decision on the forest semantics, not a filter pass.
