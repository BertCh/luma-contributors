# Space-time clustering

`GPUSpatialClustering` (DBSCAN) and `GPUOPTICS` take an optional `times` column that turns them into
ST-DBSCAN and ST-OPTICS. The time test is one more condition on the neighbor relation, so cores,
clusters, borders, noise, reachability and extraction follow from the same kernels as the purely
spatial versions. Without `times`, nothing changes: the generated kernels contain no time code.

`GPUHierarchicalDensityClustering` also supports ST-HDBSCAN through a continuous scaled Euclidean metric. See [ST-HDBSCAN](#st-hdbscan).

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
ST-OPTICS keeps the frontier of `GPUOPTICS` and its limits: at most 4096 rows with the exhaustive reachability step, at most 65,536 (including xi extraction) with `neighborIndex: 'grid'`, whose radius list also applies the time window. `orderingMethod: 'parallel'` distributes the indexed frontier across workgroups with the same strict time filter and exact row-index tie rule; the default remains `'sequential'`.

## Reference and agreement

- ST-DBSCAN: `st-dbscan` 0.2.3 (MIT), `ST_DBSCAN(eps1, eps2, min_samples).fit(X)` with `X = [time, x, y]`. Below 20000 rows it builds `where(|dt| <= eps2, euclidean, 2 * eps1)` and runs scikit-learn `DBSCAN(eps=eps1, metric='precomputed')`, the same relation as above. The fixture `test/oracles/fixtures/space-time-clustering.json` (generator `test/oracles/space-time-clustering.py`) holds its labels for five scenes on integer lattices, which exercise the closed boundaries. The GPU labels match them exactly, and an independent double-precision CPU implementation in the spec reproduces the labels, core flags, roots and cluster count.
- From 20000 rows `st-dbscan` 0.2.3 switches to a sparse path. On one 20000-row lattice scene it labeled 3733 points as clustered, where the dense definition above gives 4983 (and 2311 if a point did not count itself), so the sparse path is a different computation. The GPU follows the dense definition and is not compared against the sparse path.
- ST-OPTICS: the `st-clustering` 0.1.0 `ST_OPTICS` matrix (distance `2 * eps1` outside the window) run on scikit-learn `OPTICS(metric='precomputed')`. That matrix equals the hard filter whenever `maximumDistance < 2 * eps1`; with a larger `max_eps` the library keeps out-of-window pairs at distance `2 * eps1` instead of removing them, which `times` does not reproduce. `st-clustering` itself does not import with current joblib (`Memory(cachedir=...)`), so the fixture reproduces its matrix with scikit-learn. Ordering, predecessors and DBSCAN-extraction labels match exactly on two scenes, reachability and core distances to 1e-5 relative, and the same CPU oracle agrees.

The pinned environment (Python 3.15.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1) differs from the 3.14 pins of `requirements.txt`. `st-dbscan` 0.2.3 is published as a source distribution only and its pyscaffold build fails under pip on Python 3.15, so the generator imports it from the unpacked sdist (SHA-256 `ca1810025630571aa9ffe204f81d9c0a52dc29daa2d05bb2cd7941f56f32fd8d`, recorded in the fixture).

## ST-HDBSCAN

`GPUHierarchicalDensityClustering` accepts optional `times`, `temporalContext` and `parameters`.
Its distance is `sqrt(dx² + dy² + (timeScale * dt)²)`, equivalent to Euclidean HDBSCAN on
`[x, y, timeScale * t]`. `timeScale` is spatial position units per temporal-context tick.
For example, with positions in meters and times in seconds, `timeScale: 2` assigns a one-second
separation the same distance as two meters. Zero gives spatial distances while still excluding
non-finite times. There is no hard time window.

```ts
import {
  GPUHierarchicalDensityClustering,
  getGPUHierarchicalDensityClusteringParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';

import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

const parameters = new GPUParameterBuffer(device, {
  id: 'hdbscan-parameters', format: 'float32', length: 4,
  values: getGPUHierarchicalDensityClusteringParameterValues({timeScale: 2})
});
graph.add(new GPUHierarchicalDensityClustering({
  positions,                        // planar float32x2
  times: relativeTimes,              // float32, one per position
  temporalContext: converter.outputTemporalContext,
  parameters: parameters.importToGraph(graph),
  minimumSamples: 5,                 // includes the point itself
  minimumClusterSize: 10,
  output: {labels, coreDistances, status: {count: clusterCount}}
}));
```

`getGPUHierarchicalDensityClusteringParameterValues` packs `[timeScale, 0, 0, 0]` and rejects
negative or non-finite float32 scales. Rewrite the parameter buffer between encodings to change
the scale without recompiling. Invalid raw scale values make every row invalid. Row count,
`minimumSamples`, `minimumClusterSize`, contexts and optional-view presence are compile-time;
positions, times and scale values are per-frame. Only planar positions and relative float32 times
are supported; project geographic input and choose a time origin near the observations.
The shader subtracts times before scaling. Float32 subtraction, multiplication and squared-distance
overflow remain numerical limits; center inputs and rescale units to keep distances representable.

The exact indexed core-distance walk and Boruvka search reuse the spatial point BVH. Spatial box
distance is a conservative lower bound for the combined metric. The BVH does not prune by time,
so many observations at the same spatial site can require many candidate visits. The existing
GPU spanning tree, dendrogram, condensation and excess-of-mass selection are shared. Limits stay
at 1,048,576 rows and 32 minimum samples; construction validates the command-node estimate and
graph declaration checks scratch against the device storage binding limit.

Outputs retain the spatial HDBSCAN contract: one label per input row (`0xffffffff` for noise),
optional self-inclusive core distances (`Infinity` for invalid rows or insufficient neighbors),
and optional `status.count` containing the number of selected clusters. Finite rows with invalid
times are excluded from both cores and hierarchy. No candidate list truncation or approximation
is used, so no overflow/approximation port is needed. Boruvka and tree rounds have sufficient
fixed logarithmic budgets and GPU convergence gates, with no caller-set iteration cutoff.

Selection matches scikit-learn `HDBSCAN` defaults: Euclidean distance, EOM, no single cluster,
zero selection epsilon. Equal-weight MST edges are ordered by input rows and can produce a
partition different from scikit-learn on ties; duplicate densities use the largest float32
instead of infinity, as in spatial HDBSCAN. There is no HDBSCAN CPU operation twin.

The independent fixture `modules/analysis/test/oracles/fixtures/space-time-hdbscan.json` is generated
by `modules/analysis/test/oracles/space-time-hdbscan.py` using scikit-learn on scaled coordinates;
it records the NumPy, SciPy and scikit-learn versions. It covers overlapping events, zero and
small time scales, maximum samples, duplicate ties and a 257-row padded BVH. Headless specs compare
partitions (adjusted Rand index at least 0.97, allowing tied merges), cluster counts and every core
distance (absolute/relative tolerance `2e-5 * max(1, reference)`). Separate cases exercise invalid
rows, an empty finite set, one row and too few samples.
