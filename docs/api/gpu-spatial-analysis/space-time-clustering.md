# Space-time clustering

`GPUDBSCAN` (DBSCAN) and `GPUOPTICS` take an optional `times` column that turns them into
ST-DBSCAN and ST-OPTICS. The time test is one more condition on the neighbor relation, so cores,
clusters, borders, noise, reachability and extraction follow from the same kernels as the purely
spatial versions. Without `times`, nothing changes: the generated kernels contain no time code.

`GPUHDBSCAN` also supports ST-HDBSCAN through a continuous scaled Euclidean metric. See [ST-HDBSCAN](#st-hdbscan).

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

## Time input

| Prop | Meaning |
| --- | --- |
| `times` | One `float32` per row: ticks after the origin of `temporalContext`. Typically the output of `GPUTimeWordsRelative`, which takes the difference to the origin exactly on Int64 words. |
| `temporalContext` | A [`GPUTemporalContext`](./temporal-context.md) describing `times`. Required with `times` and rejected without. Only `encoding: 'float32-relative'` is read (a bigint origin as `epoch`); any unit, time zone and axis. Nothing is converted. |

The window is given in the unit of `temporalContext`:

- `GPUDBSCAN`: `timeEpsilon` in the packed parameters (`getGPUDBSCANParameterValues({bounds, epsilon, minimumSamples, timeEpsilon})`, stored at index 6 of the existing 8-element layout, 0 when omitted). It is per-frame: rewrite it without recompiling. `times` is compile-time presence, per-frame contents.

For planar DBSCAN and ST-DBSCAN, the encoded float32 `epsilon` must be in
`[2^-63, 2^63]` (approximately `[1.0842e-19, 9.2234e18]` position units). This keeps
squared-distance comparisons in range. The parameter packer rejects radii outside that range;
writing such raw parameters directly makes the planar lattice invalid and all labels noise.
The geographic route instead requires bounds inside `[-180, 180] x [-90, 90]` degrees and
`0.5 * epsilon / sphereRadius < 1.5707963` radians, with finite lattice steps. Typed-array bounds supplied by
JavaScript callers remain supported.
- `GPUOPTICS`: `maximumTimeDistance`, compile-time, positive and finite.

Times are float32. Differences are exact for `|dt| < 2^24` ticks, so choose an origin near the data and a unit coarse enough that the span stays inside that range when exact thresholds matter. The window is stored as float32 too.

## Definitions

Two rows are related when `distance <= epsilon` (the existing closed spatial test) and
`|t_a - t_b| <= timeEpsilon` (closed as well, so a difference equal to the window is related). A row
relates to itself.

- ST-DBSCAN: a point is core when its related neighborhood holds at least `minimumSamples` points, itself included. Clusters are the connected components of core points under the relation, the root of a cluster is its smallest core row, a border point joins the adjacent cluster with the smallest root, and compact labels rank the roots in row order. This is the canonical labeling of `GPUDBSCAN`, applied to the time-filtered relation.
- ST-OPTICS: core distances, the frontier, reachability, predecessors and extraction (`dbscan` or `xi`) use only related pairs, so a pair outside the window behaves as if it were farther than `maximumDistance`. A hard filter on the neighbor graph.
- A row with a non-finite time has no neighbors. ST-DBSCAN treats it as noise even with `minimumSamples: 1`. ST-OPTICS excludes the row and includes it in the optional `output.ports.status.invalidCount`, as it does for non-finite positions.
- `denseBoxShortcut` is rejected with `times`, because a dense box proves spatial closeness only.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {GPUTimeWordsRelative} from '@luma-contributors/analysis/dataframe-analysis';
import {GPUDBSCAN, getGPUDBSCANParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';

declare const graph: GPUCommandGraph;
declare const timestamps: GraphDataView<'uint32x2'>;
declare const relativeTimes: GraphDataView<'float32'>;
declare const positions: GraphDataView<'float32x2'>;
declare const labels: GraphDataView<'uint32'>;
declare const parameterBuffer: GPUParameterBuffer<'float32'>;
declare const bounds: readonly [number, number, number, number];
const parameters = parameterBuffer.importToGraph(graph);

const converter = new GPUTimeWordsRelative({
  timestamps,                       // uint32x2 Int64 milliseconds
  temporalContext: {unit: 'millisecond', epoch: 'unix', encoding: 'int64-words', axis: {kind: 'irregular'}},
  origin: 1_700_000_000_000n,
  output: relativeTimes             // float32
});
graph.add(converter);
const clustering = new GPUDBSCAN({
  positions,
  times: relativeTimes,
  temporalContext: converter.outputTemporalContext,
  parameters,
  gridSize: [256, 256],
  output: {labels}
});
graph.add(clustering);
parameterBuffer.write(clustering.getParameterValues({
  bounds, epsilon: 50, minimumSamples: 5,
  timeEpsilon: 6 * 3600 * 1000      // six hours, in the context unit (milliseconds)
}));
```

## Cost

ST-DBSCAN adds one gather of `times` into grid-cell order and one float compare per scanned
candidate. The grid is spatial only, so the scan visits every spatial candidate and a short window
does not prune work; a time-sorted or three-dimensional index would, and is a separate item.
ST-OPTICS keeps the frontier of `GPUOPTICS` and its limits: at most 4096 rows with the exhaustive reachability step, at most 1,048,576 with `neighborIndex: 'grid'` (xi extraction at most 65,536), whose radius list also applies the time window. `orderingMethod: 'parallel'` distributes the indexed frontier across workgroups with the same strict time filter and exact row-index tie rule; the default remains `'sequential'`.
The grid row ceiling does not qualify a full parallel traversal: its command-node budget also
requires `estimatedCommandNodeCount <= GPU_OPTICS_MAXIMUM_COMMAND_NODE_COUNT` (592,128 nodes). Reductions cost more
nodes per visit at larger row counts, so construction can reject a full traversal within the grid
row ceiling; reduce the explicit visit budget or use sequential ordering.

`GPUOPTICS` and `GPUOPTICSOperation` optionally publish `output.ports.status.reason`
(a one-row `uint32` view in the contributor, a graph value name in the operation).
`GPU_OPTICS_REASON` names its values: `complete = 0`, `candidateOverflow = 1`, and
`iterationLimitReached = 2`. Candidate loss is retained internally even when the optional
`candidateOverflow` port is omitted. Overflow suppresses extracted labels and takes precedence
over the visit limit. The optional one-row `output.ports.status.iterationCount`, `converged`
and `invalidCount` report visited rows, completion and excluded rows, respectively.
Resubmitting after inputs fit the candidate capacity clears the loss reason.
`CPUOPTICSOperation` also exposes the reason. Its exhaustive search returns completion
or the visit limit.

## Reference and agreement

- ST-DBSCAN: `st-dbscan` 0.2.3 (MIT), `ST_DBSCAN(eps1, eps2, min_samples).fit(X)` with `X = [time, x, y]`. Below 20000 rows it builds `where(|dt| <= eps2, euclidean, 2 * eps1)` and runs scikit-learn `DBSCAN(eps=eps1, metric='precomputed')`, the same relation as above. The fixture `test/oracles/fixtures/space-time-clustering.json` (generator `test/oracles/space-time-clustering.py`) holds its labels for five scenes on integer lattices, which exercise the closed boundaries. The GPU labels match them exactly, and an independent double-precision CPU implementation in the spec reproduces the labels, core flags, roots and cluster count.
- From 20000 rows `st-dbscan` 0.2.3 switches to a sparse path. On one 20000-row lattice scene it labeled 3733 points as clustered, where the dense definition above gives 4983 (and 2311 if a point did not count itself), so the sparse path is a different computation. The GPU follows the dense definition and is not compared against the sparse path.
- ST-OPTICS: the `st-clustering` 0.1.0 `ST_OPTICS` matrix (distance `2 * eps1` outside the window) run on scikit-learn `OPTICS(metric='precomputed')`. That matrix equals the hard filter whenever `maximumDistance < 2 * eps1`; with a larger `max_eps` the library keeps out-of-window pairs at distance `2 * eps1` instead of removing them, which `times` does not reproduce. `st-clustering` itself does not import with current joblib (`Memory(cachedir=...)`), so the fixture reproduces its matrix with scikit-learn. Ordering, predecessors and DBSCAN-extraction labels match exactly on two scenes, reachability and core distances to 1e-5 relative, and the same CPU oracle agrees.

The pinned environment (Python 3.15.0, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1) differs from the 3.14 pins of `requirements.txt`. `st-dbscan` 0.2.3 is published as a source distribution only and its pyscaffold build fails under pip on Python 3.15, so the generator imports it from the unpacked sdist (SHA-256 `ca1810025630571aa9ffe204f81d9c0a52dc29daa2d05bb2cd7941f56f32fd8d`, recorded in the fixture).

## ST-HDBSCAN

`GPUHDBSCAN` accepts optional `times`, `temporalContext` and `parameters`.
Its distance is `sqrt(dx² + dy² + (timeScale * dt)²)`, equivalent to Euclidean HDBSCAN on
`[x, y, timeScale * t]`. `timeScale` is spatial position units per temporal-context tick.
For example, with positions in meters and times in seconds, `timeScale: 2` assigns a one-second
separation the same distance as two meters. Zero gives spatial distances while still excluding
non-finite times. There is no hard time window.

```ts
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import type {GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import {
  GPUHDBSCAN,
  getGPUHDBSCANParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

declare const graph: GPUCommandGraph;
declare const device: Device;
declare const temporalContext: GPUTemporalContext;
declare const positions: GraphDataView<'float32x2'>;
declare const relativeTimes: GraphDataView<'float32'>;
declare const labels: GraphDataView<'uint32'>;
declare const coreDistances: GraphDataView<'float32'>;
declare const clusterCount: GraphDataView<'uint32'>;

const parameters = new GPUParameterBuffer(device, {
  id: 'hdbscan-parameters', format: 'float32', length: 4,
  values: getGPUHDBSCANParameterValues({timeScale: 2})
});
graph.add(new GPUHDBSCAN({
  positions,                        // planar float32x2
  times: relativeTimes,              // float32, one per position
  temporalContext,
  parameters: parameters.importToGraph(graph),
  minimumSamples: 5,                 // includes the point itself
  minimumClusterSize: 10,
  output: {labels, coreDistances, status: {count: clusterCount}}
}));
```

`getGPUHDBSCANParameterValues` packs `[timeScale, 0, 0, 0]` and rejects
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
at 4,194,304 rows (`GPU_HDBSCAN_MAXIMUM_ROW_COUNT`) and 32 minimum samples; construction validates the command-node estimate and
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

For typed uploads, `clustering.getParameterValues(settings)` uses the constructed spatial context
and requires a positive `timeEpsilon` when `times` is present. The free packer still permits omitted
time windows for planar-only layouts. Bounds always contain exactly four coordinates; JavaScript
callers may supply a four-element typed array.

For geographic centroids near the antimeridian, choose a central meridian `c` and transform each
longitude with `wrap(longitude - c)` into `[-180, 180]`. Rotate the bounds into that same frame,
cluster, then transform the centroid back with `wrap(centroidLongitude + c)`. For equatorial events
at `[179.9, -179.9]`, `c = 180` gives approximately `[-0.1, 0.1]`: at 30,000 meters and
`minimumSamples: 2` they form one cluster, and the inverse centroid lies at the seam. The original
coordinates also form one cluster but their arithmetic centroid is near zero. Naively unwrapping
to `[179.9, 180.1]` requires unsupported geographic bounds and invalidates the lattice.
