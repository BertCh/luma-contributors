---
title: "GPU Spatial Analysis: Spatial weights"
sidebar_label: "Spatial weights"
description: "Spatial weights contributors in @luma-contributors/analysis/gpu-spatial-analysis."
---


# Spatial weights


A `GPUSpatialWeights` is the sparse `W` that spatial statistics are built on, in CSR form: row `i`
lists the neighbors `j` of focus row `i` and the weight `w_ij` of each. Producers write it on the
GPU, transforms rewrite it, and the statistics below read it, so one statistic works with any
neighborhood definition. Applications can also upload weights built elsewhere.

See [helper conventions and audit coverage](./spatial-weights-audit.md) for grouping, loss,
dispatch, construction snapshots and row-local/space-time algebra semantics.

Every producer guarantees: `offsets` holds `rows + 1` non-decreasing entries starting at 0; row `i`
occupies slots `[offsets[i], offsets[i + 1])`; neighbor IDs are strictly ascending within a row; a
ordinary self join excludes its own row (`w_ii = 0`), while explicit self-weight algebra and
uploaded matrices may declare a different diagonal policy; weights are finite and non-negative.
Capacity overflow clamps offsets to `neighbors.length` and sets an `overflow` flag on the GPU.

A `GPUSpatialWeights` is a first-class object, not only buffers. Every producer in this section
(`GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsTransform`,
`GPUSpatialWeightsAlgebra`, `GPUSpatialWeightsTranspose`) exposes `outputWeights`: the caller-owned
views with `information` (`rowCount`, `transform` state `'raw' | 'binary' | 'row' | 'double' | 'variance' |
'kernel' | 'symmetric' | 'unknown'`, `symmetric` as `true | false | 'unknown'` only where
guaranteed, `hasSelfWeights`, `provenance` with source, criterion, `k`, `radius` and `order`) and
`status` (`overflow` and, where the producer computes it exactly, `requiredCount`) attached. Pass
`outputWeights` to consumers: statistics read `information.transform` (a `'row'` W is
re-standardized over the neighbors that remain after masking) and gate on both loss flags.
Uploaded weights may omit `information`, which reads as `'unknown'`. `validateGPUSpatialWeights`
checks on the CPU what it can: view formats and lengths, `information.rowCount`, the declared
transform and symmetry, and the status formats.

The optional `spatialContext` describes the coordinates and metric used to build the weights;
`distances` and `information.provenance.radius` use its length unit. `GPUNeighborSearch` sets it;
`GPUNearestFeatureWeights` sets it from the producing join's context passed to the adapter.
Transforms, transpose and algebra propagate it, and they carry a positive `status.complete`
certificate unchanged (chunked transpose folds `complete == 0` into aggregate
`candidateOverflow`); algebra checks declared input contexts with
`assertCompatibleGPUSpatialContexts`. Topology-only producers and uploaded weights may omit it.

For weighted statistics, a slot is an active neighbor when its weight is
positive, its neighbor ID is below the row count and it is not the row itself. A zero-weight slot is
listed but is not a neighbor. An island is a row with no active slot, whatever slots it lists.

`GPUSpatialWeightsMinimumSpanningTree` uses topology instead: every listed slot is a link,
including reciprocal zero-weight slots. It ignores the weight value and uses attribute
dissimilarity as edge cost. SKATER and REDCAP forest compositions keep that topology-only rule;
zero-weight links must be removed from the CSR if they should not connect forest components.

Uploaded CSR must satisfy the invariants above; consumers do not re-check them (a binary search
over unsorted neighbors silently gives wrong `S1`, `S2` and symmetry counts). Run
`GPUSpatialWeightsValidation` once after upload, read its actual `invalidCount` before submitting
consumers, and repair violations on the CPU. Sort and resolve duplicate directed pairs according
to their meaning; repair self slots according to the declared diagonal policy, rather than
unconditionally deleting them. See the [ordered PySAL upload](#ordered-pysal-upload).

Capacity grow loop: `GPUContiguityWeights`, `GPULatticeWeights` and `GPUSpatialWeightsAlgebra`
write into one caller-owned `output: GPUSpatialWeightsOutput`, the CSR views plus
`status: {overflow, requiredCount?, candidateOverflow?}` (`overflow` required). Read `overflow` and
`candidateOverflow` after submission first. Recover the stage that lost coverage before sizing
final CSR storage: contiguity grows `pairCapacity` with its `getCapacityPlan`; higher-order
algebra grows `workCapacity` when its intermediate frontier overflowed. Algebra has no `getCapacityPlan`.
When algebra inherits loss from an input, repair and resubmit that upstream operand first;
larger union or other algebra output storage cannot recover omitted operand edges. Both causes
can coexist, so inspect the operands and the intermediate stage. Lattice has no bounded
candidate stage and clears its requested candidate-loss word. Only after the relevant causes
clear may an exact `requiredCount` size final storage; if final `overflow` is
nonzero, allocate `getGPUSpatialWeightsCapacityPlan({rowCount, capacity,
requiredCount: {kind: 'exact', value: requiredCount}}).capacity` slots (the required count plus
25 percent headroom, `GPU_SPATIAL_WEIGHTS_CAPACITY_HEADROOM`) and rebuild. The plan's
`estimatedOutputBytes` is the bytes of the CSR views at that capacity
(`GPU_SPATIAL_WEIGHTS_SLOT_BYTE_LENGTH` per slot, plus `GPU_SPATIAL_WEIGHTS_DISTANCE_BYTE_LENGTH`
with distances, plus the offsets). The contiguity and lattice producers also expose
`getCapacityPlan(observation)`, which adds their own transient bytes (from the same constants their
`getCommandNodes` allocates with; the device-dependent gpgpu sort and scan scratch is not included)
and, for contiguity, the intermediate pair stage. Rows past the point of truncation are empty, so
never use weights whose `overflow` or `candidateOverflow` is nonzero.

For two triangles sharing an edge, contiguity with `pairCapacity: 1` and final capacity 2 can
report `requiredCount=1, overflow=0, candidateOverflow=1`. The complete graph needs two directed
slots. The observed one is only a lower bound: retry the pair stage before accepting or sizing
from it. Submit → inspect candidate loss → retry candidate stage → size from the exact final
requirement → rerun on final overflow.

The generic planner accepts `{kind: 'unknown'}` or `{kind: 'exact' | 'lower-bound' | 'saturated',
value}` as `requiredCount`; a numeric value is treated as exact. Its `retryStage` is `'candidates'`
for unknown, lower-bound or saturated evidence (final capacity stays unchanged), `'final'` for
exact evidence exceeding final capacity, and `'none'` otherwise. For contiguity, pass the readback
to `contiguity.getCapacityPlan({requiredCount, candidateOverflow: candidateOverflow !== 0})`,
rebuild with `pairCapacity: plan.pairs.capacity`, and use `plan.slots.retryStage` to choose the
stage. A missing candidate-loss observation is not a completeness certificate.

Capacity-plan `headroom` must be finite. Growth headroom saturates at the CSR uint32 limit
`0xffffffff`; if the exact requirement itself exceeds that limit, the plan returns `limited: true`
and retains the full `estimatedRequiredCount`. Split such a matrix into chunks. Nonfinite
fractions are refused; negative finite headroom retains the zero-headroom rule.
With unknown, lower-bound or saturated evidence, `limited` remains false and capacity stays
unchanged even if the observed lower bound exceeds uint32. Resolve `retryStage: 'candidates'`
before final sizing. The generic planner takes no device limits: `limited: false` does not
certify that a GPU allocation or aligned binding fits, candidate coverage, CSR validity or a
complete result. With exact evidence it means only that the final count fits the planner's
uint32 capacity bound. Inspect a producer's candidate-plan `limited` separately when deciding
whether that stage can grow, then check the actual device binding/allocation limits.

<a id="ordered-pysal-upload"></a>

#### Ordered PySAL upload

Keep feature identifiers separate from CSR row indices. Given a PySAL weights object `w` and
a data frame `frame` with a `value` column, align attributes by `w.id_order` before uploading:

```python
import numpy as np

row_ids = list(w.id_order)
assert len(row_ids) == len(set(row_ids))
assert frame.index.is_unique
assert len(frame) == len(row_ids) and set(frame.index) == set(row_ids)
w.transform = 'R'  # Explicitly choose row-standardized statistical influence.
values = frame.loc[row_ids, 'value'].to_numpy(dtype=np.float32)
csr = w.sparse.tocsr(copy=True)
csr.sum_duplicates()  # Sum only because these entries represent additive influence.
csr.sort_indices()
csr.eliminate_zeros()
row_count = len(row_ids)
assert csr.ndim == 2 and csr.shape == (row_count, row_count)
assert 0 < row_count < 0xffffffff and csr.nnz <= 0xffffffff
assert csr.indptr.size == row_count + 1 and csr.indptr[0] == 0
assert csr.indptr[-1] == csr.nnz and np.all(np.diff(csr.indptr) >= 0)
assert np.all(csr.indptr >= 0) and np.all(csr.indptr <= 0xffffffff)
assert np.all(csr.indices >= 0) and np.all(csr.indices < row_count)
assert np.all(np.isfinite(csr.data)) and np.all(csr.data >= 0)
source_rows = np.repeat(np.arange(row_count), np.diff(csr.indptr))
assert np.all(csr.indices != source_rows)  # This upload declares diagonal='forbid'.
assert values.shape == (row_count,) and np.all(np.isfinite(values))
offsets = csr.indptr.astype(np.uint32)
slot_capacity = max(1, csr.nnz)  # GPU CSR views require nonempty physical storage, even for islands.
neighbors = np.zeros(slot_capacity, dtype=np.uint32)
weights = np.zeros(slot_capacity, dtype=np.float32)
neighbors[:csr.nnz] = csr.indices.astype(np.uint32)
weights[:csr.nnz] = csr.data.astype(np.float32)
assert np.all(np.isfinite(weights))  # Catch overflow during the float32 cast.
```

Upload `offsets` as `row_count + 1` packed uint32 words, `neighbors` as packed uint32 CSR row
indices, and `weights` and aligned `values` as packed float32 words. Declare
`information: {rowCount: row_count, transform: 'row', symmetric: 'unknown'}` to match `'R'`;
row normalization need not preserve symmetry. Retain `row_ids` for joining outputs back to
features. On the four-row chain with row-standardized weights, values `[1,2,4,8]` give global
Moran I `67/230`; swapping the middle attributes to `[1,4,2,8]` gives `-101/230` despite identical
dimensions and valid CSR. Alignment changes the statistic.
For three islands, offsets remain `[0,0,0,0]`, with one unused physical neighbor and weight
slot. The terminal offset is still zero; padding adds no edge. All three `row_ids` are retained
and recoverable as islands. Slot capacity and the live count `offsets[row_count]` are distinct.

`w.islands` contains feature IDs. For GPU-side discovery, request
`GPUSpatialWeightsSummary.output.cardinality`, first check all producer loss flags for
completeness, then read cardinality and recover IDs with
`[row_ids[row] for row, count in enumerate(cardinality) if count == 0]`. An isolate count alone
does not identify the islands.

Duplicate reduction is a domain choice. Offsets `[0,2,2]`, neighbors `[1,1]` and road costs `[2,3]`
give validation flags `[4,0]` and `invalidCount=1`. For routing, choose minimum directed cost 2;
for adjacency statistics, use canonical binary influence 1. The statistical CSR is then
offsets `[0,1,1]`, neighbors `[1]`, weights `[1]`, with flags `[0,0]` and `invalidCount=0`.
Inverse-cost influence is another explicit policy; travel-cost magnitudes are not implicitly
statistical influence. Resolve duplicates before choosing and applying a statistical transform.

Keep the complete producer port (for example, `contiguity.outputWeights`) when wiring weights
into a minimum spanning forest, SKATER, REDCAP or Ward. These consumers accept supplied loss
words even when both are zero, read their contents on the GPU on every submission, and suppress
partitions on either loss cause. Forest component labels and region labels become `0xffffffff`;
forest edge counts and SKATER/REDCAP region counts become zero/unavailable. Forest compact edge status reports
either upstream cause as `candidateOverflow`, while its `overflow` describes final edge capacity
only. SKATER cut gains become NaN. Clearing the flags restores ordinary results without rebuilding
the graph. See the [regionalization contract](operators-statistics.md#gpuspatialweightsminimumspanningtree-gpuskaterregions-and-gpuregionpartitionevaluation)
for the operation and CPU-twin ports.
Forest status is available without compact edges: its packed `candidateOverflow` word reports
either upstream cause, separately from the attribute `invalidCount` word. Forest and SKATER lens
captures retain both producer loss flags and refuse a record if either is nonzero. Forest figures
size their edge key spaces and segment captures from the published compact count, including
disconnected graphs, rather than assuming `rowCount - 1` edges.

| Producer | Neighborhood | PySAL equivalent |
| --- | --- | --- |
| `GPUNeighborSearch` | k nearest neighbors or distance band between points | `KNN`, `DistanceBand`, `Kernel` |
| `GPUContiguityWeights` | polygons sharing a vertex (queen) or an edge (rook) | `Queen`, `Rook` |
| `GPULatticeWeights` | regular grid cells | `lat2W` |
| `GPUSpatialPredicateJoin` with `weights` | any predicate join (intersects, within, dwithin, ...) | - |
| `GPUSpatialWeightsAlgebra` | union, intersection, difference, higher order, self weights, subgraph, block | `w_union`, `higher_order`, `w_subset`, `block_weights` |

### `GPUContiguityWeights`

Join Chicago tract polygons that share a vertex (queen) or an edge (rook) into binary neighbour weights.

::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=result caption="Queen links join 791 Chicago tracts that share a vertex; tracts with no link are islands."}

Polygon contiguity as binary weights. `criterion: 'queen'` joins features that share at least one
vertex; `'rook'` joins features that share a ring edge (both endpoints, either direction). Input is
`polygons: GPUPolygonGeometryPort` (the four-level layout `positions`, `ringOffsets`,
`polygonOffsets`, `featureOffsets`), with one output row per FEATURE: every ring of every part
counts, holes included, so a multipart feature is the neighbor of anything any of its parts touches,
and parts of one feature are never self neighbors. This is the row space of
`GPUZonalStatistics` and the points-in-polygons recipe, so points -> counts -> W -> Moran chain on
one feature table without a cast.

- `snapTolerance` (default 0, exact f32 equality) snaps vertices with
  `floor((x - origin) / tolerance + 0.5)` per axis (from 2^23 steps the float32 quotient is an
  integer and is used as it is), where the origin is the first vertex of the data, so the grid
  depends on the extent of the data and not on its distance from the coordinate origin. It is the
  snap key of `GPUCoverageDissolve` and `GPUCoverageSimplification`. This is a grid snap, not a distance test: two vertices closer than the tolerance can
  fall in different cells. The extent divided by the tolerance must stay at or below
  `GPU_CONTIGUITY_WEIGHTS_MAXIMUM_SNAP_CELLS` (2^24); pass `coordinateExtent` to have that checked
  at construction.
- Exact 64-bit vertex keys, stable radix sorts and run detection; no hashing, so no collisions, and
  the output is deterministic and symmetric.
- A point shared by `m` features emits `m^2` candidate pairs, bounded by `pairCapacity` (default
  `GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR` = four times the neighbor capacity); exceeding it
  sets `candidateOverflow` only. Overflow reports final neighbor-slot loss only. The producer's
  outputWeights always exposes both flags, allocating candidate status in the graph if omitted.
- `output.status.requiredCount` is exact unless the pair list overflowed (then a lower bound).
  `getCapacityPlan({requiredCount, candidateOverflow})` returns the next `slots` and `pairs`
  capacities (pairs keep the default ratio and double after a candidate overflow). Pair growth
  saturates at `0xffffffff`; `pairs.estimatedRequiredCount` retains the uncapped estimate and
  `pairs.limited` reports when that estimate cannot fit. Split the work when limited. The default
  pair capacity uses the same cap. `outputWeights.information` is binary, symmetric, with provenance
  `{source: 'GPUContiguityWeights', criterion}`.

Ordinary contiguity writes `offsets`, `neighbors`, binary `weights`, required `status.overflow`
and optional `status.count`, `status.requiredCount` and `status.candidateOverflow`. It rejects `output.distances`
because topology supplies no measured distances; attach separately measured distances to the
returned weights afterward when needed. `status.count` reports retained slots from the terminal
CSR offset, bounded by capacity. `requiredCount` is the complete requirement only after candidate
loss clears; after candidate loss it can remain a lower bound. Neither count certifies candidate
coverage. Only the retained prefix is readable; tails beyond it are unspecified.

`computeContiguityWeightsOnCPU` and `computePolygonContiguityOnCPU` return the CSR arrays and
one-word uint32 arrays under `status.overflow`, `status.requiredCount` and
`status.candidateOverflow`. Pass the result directly to a CPU weights consumer to preserve
those diagnostics.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUContiguityWeights} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUContiguityWeights>[0];
const {polygons} = inputs;
const {offsets, neighbors, weights} = inputs.output;
const {positions, ringOffsets, polygonOffsets, featureOffsets} = polygons;
const {overflow, requiredCount} = inputs.output.status;

graph.add(new GPUContiguityWeights({
  criterion: 'rook',
  polygons: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},
  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}
}));
```

#### How it computes

{/* lens:GPUContiguityWeights */}

::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen neighbours, and how many of the 791 tracts are islands."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=formula caption="Follow the rule: tracts are neighbours when their snapped vertex keys intersect."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=kernel caption="Inspect the key sort, pair emission and CSR compaction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.contiguity-weights rung=graph caption="Trace the command graph from tract vertices to the CSR neighbour lists."}

### `GPULatticeWeights`

Connect the cells of a width by height grid to their rook or queen neighbours as binary CSR weights.

::lens{scenario=gpu-spatial-analysis.lattice-weights rung=result caption="Queen neighbours on a 32 by 32 terrain lattice: 8 inside, 5 along edges, 3 in corners."}

Weights over a `width x height` grid with row-major IDs `y * width + x`. `criterion: 'rook'` uses
Manhattan distance `<= radius`, `'queen'` Chebyshev distance `<= radius` (`radius` defaults to 1, at
most 32). A zero in the optional `mask` removes the cell as a row and as a neighbor. With `cellSize`
it also writes Euclidean `distances`, so kernel transforms apply. `output.status.requiredCount` is
the unclamped neighbor total, saturated at `0xffffffff`; `getCapacityPlan({requiredCount})` plans
the next slots. Individual prefixes also saturate before capacity clamping, so offsets remain
nondecreasing across a uint32 carry.
`outputWeights.information` is binary and symmetric, with provenance `{criterion, order: radius}`.

`cellSize` accepts either a numeric `[width, height]` default or a packed float32 view from
`getGPULatticeWeightsParameterValues`. Host that view in a `GPUParameterBuffer` and write new
values between encodings to update distances on the same compiled graph. Width, height,
criterion, radius, capacities and optional views remain compile-time; changing the distance
scale keeps the same neighbors and capacity status.

The capacity plan includes 8 bytes for uploaded numeric cell sizes when `output.distances` is
present. Borrowed parameter views and configurations without distance output add no parameter
scratch; the per-cell count and offset buffers use 8 bytes per cell. Scan scratch is excluded
from this estimate.

#### How it computes

{/* lens:GPULatticeWeights */}

::lens{scenario=gpu-spatial-analysis.lattice-weights rung=datum bind=page datum=528 view=neighbourhood-view caption="Inspect cell 528: its 8 queen neighbours, with the degree histogram of all 1,024 cells."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=formula caption="Follow the rule: two cells are neighbours within the rook or queen radius."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=kernel caption="Inspect the neighbour count and fill kernels over 1,024 cells."}
::lens{scenario=gpu-spatial-analysis.lattice-weights rung=graph caption="Trace the command graph from lattice size to CSR neighbours."}

{/* parameters: GPULatticeWeights generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULatticeWeights` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPULatticeWeights`.

- Packer: `getGPULatticeWeightsParameterValues(settings?)`, writes `GPU_LATTICE_WEIGHTS_PARAMETER_LENGTH` (2) `float32` values.

Settings of `getGPULatticeWeightsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `cellSize` (optional) | `readonly [number, number]` | Positive cell width and height. Defaults to `[1, 1]`. |

{/* /parameters */}

### `GPUSpatialWeightsTransform`

Rewrite CSR weights by row, binary, kernel, symmetric, double or variance standardisation into a new weights graph.

::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=result caption="Row standardisation keeps the queen links of 791 tracts and rescales each row to sum to one."}

Rewrites weights in place or into a separate `output`. Chain several instances, for example a kernel
followed by row standardization. An output in the source buffer must cover the exact same byte
range for transforms that read old weights; distinct subviews are refused for those operations.
A kernel transform reads distances and may write a distinct subview of the old weights buffer.
Use a separate buffer to preserve the source.
Numeric `bandwidth` uses the length unit of
`weights.spatialContext`, matching `weights.distances`. The optional `spatialContext` declares the
bandwidth's context and is checked against `weights.spatialContext` with
`assertCompatibleGPUSpatialContexts` (`purpose: 'metric-results'` for kernels, so implicit meters
and explicit kilometers conflict) when both are supplied.

- `'row'`: `w_ij / sum_j w_ij`. A row whose weights sum to zero (or to a non-finite value) stays all
  zero: it has no active neighbor, so it is an island for every statistic (no uniform fallback).
  To restore row standardization after masking or `'subgraph'` removed neighbors, apply `'row'` again
  to the result.
- `'binary'`: `w_ij > 0 ? 1 : 0`, keeping every slot.
- `'kernel'`: `K(d_ij / h_i)` from `weights.distances`, with `kernel` one of `gaussian`
  (`exp(-z^2 / 2) / sqrt(2 pi)`), `triangular` (`max(1 - z, 0)`), `epanechnikov` (`3/4 max(1 - z^2, 0)`,
  PySAL quadratic), `bisquare` and `quartic` (equal shape spellings for
  `15/16 max(1 - z^2, 0)^2`) or `uniform` (`1/2` for
  `z <= 1`). `bandwidth` is required (no default) and is either a positive number, the same `h` for
  every row (PySAL fixed bandwidth; for distance-band weights pass the band radius), or `'adaptive'`:
  PySAL's adaptive bandwidth, `h_i` = the distance to the k-th nearest neighbor times
  `eps = 1.0000001` (`GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON`). On the CSR that is the row's
  largest distance, but only for kNN rows, so `'adaptive'` requires
  `weights.information.provenance.criterion === 'knn'` and throws otherwise. Because of `eps` the k-th
  neighbor keeps a tiny positive weight, as in libpysal (in float32 `eps` rounds to `1 + 2^-23`, so
  the tail weight is about 1e-7, never exactly 0). Only the stored neighbors take part, so the
  Gaussian is truncated at the neighbor set. Ordinary search weights exclude self slots, so
  `K(0)` is not written there: compute it with `getGPUSpatialWeightsKernelValue(kernel, 0)` and pass it as `selfWeight`
  to `GPUHotSpotAnalysis` for a kernel Gi*. `getGPUSpatialWeightsKernelValue(kernel, z)` is the CPU
  value of every profile. Negative or non-finite `z` produces zero in both the GPU transform and
  CPU helper, including for the Gaussian and uniform profiles.
- `'double'`: `w_ij / S0`, with `S0` the sum of all weights, so the weights sum to 1 (PySAL `'D'`).
  `doubleSum: 'rows'` scales by `n / S0` instead, so the weights sum to `n`.
  Before computing a scale that would overflow for positive finite `S0`, the apply pass
  combines integer exponents and bounded mantissas for `w_ij / S0`, then multiplies by `n`.
  This retains representable results even when shader fast math regroups float expressions.
- `'variance'`: `s_ij = w_ij / sqrt(sum_j w_ij^2)`, then `w'_ij = s_ij n / sum s` over the whole
  matrix, with `n` the row count (PySAL `'V'`). When the row's squared norm overflows or enters
  float32 underflow range, it is computed from weights divided by their row maximum so a
  positive rescaling of finite weights preserves the result within float32 rounding.
- `'average-reciprocal'`: `(w_ij + w_ji) / 2`, treating a missing reverse slot as 0. It keeps the sparsity
  pattern, so it is only symmetric for symmetric patterns (contiguity, lattice, distance band), and
  needs a separate `output`.
  For operands at least `2^127`, the average halves normal values by decrementing their
  integer exponent fields before addition. It never forms an overflowing sum or depends on
  an overflowed intermediate to choose the safe path.

PySAL's `W.symmetrize` adds the missing reverse links. To reproduce it, compute
`GPUSpatialWeightsTranspose`, then take the support union with `GPUSpatialWeightsAlgebra` using
`operation: 'union'`, the original weights as `left` and `weightRule: 'left'`. This preserves
existing weights and copies the original weight to each new reverse link. A true support-union
transform is not provided. [PySAL weights reference](https://pysal.org/libpysal/stable/generated/libpysal.weights.W.html#libpysal.weights.W.symmetrize)

`'double'` and `'variance'` compute a per-row partial, reduce it with `GPUReduction`, and apply it in
a second pass, in place or into `output`. A zero or non-finite total gives zero weights. A
non-finite row sum contributes to the `'double'` total, so its overflow suppresses all rows. The
operation, kernel profile and adaptive-versus-fixed choice are compile-time options. A fixed
`bandwidth` also accepts a packed float32 view from `getGPUSpatialWeightsTransformParameterValues`;
write new values through its `GPUParameterBuffer` between encodings. Numeric bandwidths remain
supported as uploaded defaults. Updating bandwidth changes only existing slots; it does not
rebuild the neighbor set or update a previously supplied numeric provenance radius. `output`, when given, must have exactly
`weights.neighbors.length` values.

`outputWeights.information.transform` is the operation (`'average-reciprocal'` gives `'symmetric'`). `symmetric`
is kept only where it is guaranteed: `'binary'`, `'double'` and `'average-reciprocal'` keep a `true`
input flag; `'row'`, `'variance'` and every distance kernel give `'unknown'`
(row standardization makes a symmetric pattern asymmetric unless all row sums are equal).
Numerically symmetric zero-weight input does not guarantee reciprocal stored slots or equal
distances, so a fixed-bandwidth kernel can create asymmetric positive weights.
Transform destinations must not overlap preserved CSR or status words. Ports actually bound
read-only with the destination require separate physical buffers; an unbound port may share a
buffer through disjoint logical views. For example, binary transformation can write at byte 256
while preserving two distance words at byte zero. Exact in-place weight updates remain supported.
Only caller views actually bound by the selected route must fit the device's aligned storage
binding range: offsets and destination on every route, old weights except on kernel, neighbors
on `'average-reciprocal'`, and distances plus any resident bandwidth on kernel. Every supplied view still
must belong to the graph. Whole `double` and `variance` additionally require `rowCount` float32
partials in one binding; their reduction also uses scalar scratch. Whole row, binary and kernel
do not allocate that whole-row partial array. Check the actual device limits before lowering.

#### How it computes

{/* lens:GPUSpatialWeightsTransform */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen weights after row standardisation, and the islands among 791 tracts."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=formula caption="Follow the row rule: each weight is divided by its row sum; empty rows stay zero."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=kernel caption="Inspect the row-sum and rescale kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transform rung=graph caption="Trace queen contiguity into row standardisation on the command graph."}

{/* parameters: GPUSpatialWeightsTransform generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialWeightsTransform` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpatialWeightsTransform`.

- Packer: `getGPUSpatialWeightsTransformParameterValues(settings)`, writes `GPU_SPATIAL_WEIGHTS_TRANSFORM_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUSpatialWeightsTransformParameterValues`:

| Field | Type | Description |
|---|---|---|
| `bandwidth` | `number` | Positive fixed bandwidth, in the units of the resident neighbor distances. |

{/* /parameters */}

### `GPUSpatialLag`

If either supplied `weights.status.overflow` or `weights.status.candidateOverflow` is nonzero,
every lag component is NaN; a retained prefix is never presented as a complete weighted sum.
For chunked weights, loss on the outer status or any chunk suppresses the whole result.
Clearing the loss words and submitting complete CSR data restores the lag on the same graph.
Optional caller-owned `status: {overflow, candidateOverflow}` contains one uint32 word per
requested cause. Each submission overwrites each word with the OR of that cause over the outer
weights and every chunk. `overflow` describes final-edge loss; `candidateOverflow` describes
intermediate loss. Either suppresses the entire lag. These words do not certify CSR validity or
diagnose deliberately excluded coordinates and values.
`GPUSpatialLag` rejects supplied `weights.status.complete` certificates at outer or chunk scope,
even when failure-polarity flags are also supplied. Its current loss gate is not completeness-aware.
Do not strip a certificate to make construction succeed. Establish complete input with a
completeness-aware upstream stage or on the host before constructing an appropriate lag input;
retain the original diagnostics alongside the result.

Average neighbouring values through row-standardised weights, turning a tract variable into its spatial lag.

::lens{scenario=gpu-spatial-analysis.spatial-lag rung=result caption="Poverty percentage beside its neighbour-weighted lag across 791 tracts; the lag smooths local extremes."}

`lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`). An optional `mask` skips masked neighbors and writes
0 for masked rows (square weights only); `normalize: true` divides by the included weight sum, which
gives the row-standardized lag without a transformed copy of the weights. Rows are summed by a
load-balanced (merge-path) CSR loop, so a hub row with 1e5 neighbors spreads over many workgroups
instead of one thread. The association depends only on the offsets, so the result is bitwise
reproducible; it matches a sequential sum within f32 rounding. A weights matrix with a self term (`GPUSpatialWeightsAlgebra`
`'selfWeight'`) includes it in the sum.

Weights need not be square. The value row count is `values.length / columnCount`: the rows of
`weights` are targets and the neighbor IDs index those value rows; neighbors at or above it are
skipped. `columnCount` (default 1) lags several columns in one pass: `values` is row-major,
`output` is `rows * columnCount`, and `normalize` shares one weight sum per row. This is how the area
shares of `GPUArealInterpolation` transfer values between zone systems. `values` may be `float32` or
`uint32` (counts, exact below 2^24; one conversion node is added).

Empty rows and rows whose included weights sum to zero produce zero in every normalized
column. Out-of-range neighbors contribute to neither the weighted sum nor its normalization
denominator. With chunked weights, masks and neighbor IDs remain global; a chunk boundary
does not renumber either. Slots beyond each CSR's terminal offset are unused.

For an affine change of values `y' = a*y + b`, the raw lag becomes
`a*lag + b*sum(included weights)`. The normalized lag becomes `a*lag + b` on rows with a
positive included weight sum; zero-sum rows stay zero, subject to float32 rounding.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUContiguityWeights, GPUSpatialWeightsTransform, GPUSpatialLag} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUContiguityWeights>[0];
const {polygons, output: weights} = inputs;
declare const values: ConstructorParameters<typeof GPUSpatialLag>[0]['values'];
declare const lag: ConstructorParameters<typeof GPUSpatialLag>[0]['output'];

const contiguity = new GPUContiguityWeights({criterion: 'queen', polygons, output: weights});
graph.add(contiguity);
const transform = new GPUSpatialWeightsTransform({operation: 'row', weights: contiguity.outputWeights});
graph.add(transform);
graph.add(new GPUSpatialLag({weights: transform.outputWeights, values, output: lag}));
```

An in-place transform shares storage and publishes a new description of those bytes. Pass
`transform.outputWeights` onward; the original handle still describes binary weights. Mask-aware
statistics use `information.transform: 'row'` to re-standardize surviving neighbors. For the
undirected chain `0—1—2—3`, values `[1,2,4,8]` and mask `[1,1,1,0]`, the old binary description
over row-standardized bytes gives global Moran I `+1/35`; the transformed description gives
`−1/28`. Included deviations are `[-4,-1,5]/3`; their squares sum to `14/3`. The old description
has `S0=5/2`, weighted cross-product `1/9`; re-standardizing gives `S0=3`, cross-product `−1/6`.
`GPUSpatialLag` itself uses its explicit `normalize` option to normalize included weights.

#### How it computes

{/* lens:GPUSpatialLag */}

::lens{scenario=gpu-spatial-analysis.spatial-lag rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its neighbours and weights, and the islands that receive a zero lag."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=formula caption="Follow the lag: the sum of each neighbour weight times its value."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=kernel caption="Inspect the load-balanced segment sum over the CSR slots."}
::lens{scenario=gpu-spatial-analysis.spatial-lag rung=graph caption="Trace queen weights, row standardisation and the lag on the command graph."}

### `GPUNeighborSearch`

Find the k nearest tract centroids by great-circle distance and write them as CSR neighbour weights.

::lens{scenario=gpu-spatial-analysis.neighbor-search rung=result caption="Each of 791 tract centroids links to its 4 nearest others, giving 3,164 directed links."}

Maximum: `gridSize[0] * gridSize[1]` is at most `getGPUGridIndexMaximumCellCount(device)` (2^25 - 1 on a default device), checked when the contributor is added to a graph.

`GPUNeighborSearch` finds exact k nearest neighbors (`mode: 'knn'`, `neighborCount` up to `GPU_NEIGHBOR_SEARCH_MAXIMUM_K`, 256) or every point
within a distance band (`mode: 'radius'`) between planar points, or between longitude/latitude points
with great-circle distances (see below). A self join never lists a row as its
own neighbor; pass `queryPositions` for a cross join. It writes a `GPUSpatialWeights` CSR:
`offsets` (`queryRows + 1`), `neighbors` (the slot capacity), `weights`, and optional `distances`.

- Rows list neighbor IDs in strictly ascending order with distances and weights aligned, so
  consumers can binary-search `w_ji`.
- kNN keeps the `k` smallest `(d^2, id)` pairs, so ties go to the lowest ID. An optional
  `maximumDistance` bounds the search. Fewer valid targets give a shorter row.
- Weights: `binary`, `inverse-distance` (`max(d, distanceFloor)^-power`, default floor `1e-6`,
  the same default as `GPUNearestFeatureWeights`; two rows at the same position then weigh
  `1e6^power` to each other and, with `rowStandardize`, take nearly all of each other's row, so pass
  a floor on the scale of the data when positions repeat), or `kernel` (gaussian,
  triangular, epanechnikov, bisquare, uniform). The bandwidth is `searchRadius` in radius mode and the
  row's k-th neighbor distance times 1.0000001 in kNN mode, which is libpysal `Kernel(fixed=False)`
  with the self slot removed (a row with fewer than `k` neighbors uses its farthest listed
  neighbor). `rowStandardize` divides each row by its sum. Non-finite weights are written as 0.
  Rows never hold a self slot: the kernel's self weight `K(0)` is the consumer's `selfWeight`
  (`getGPUSpatialWeightsKernelValue(kernel, 0)`). A zero-weight slot (for example a triangular
  kernel at `d = radius`, or coincident points under inverse distance with `distanceFloor: 0`) is not a
  neighbor: consumers skip slots whose weight is not positive.
- Capacity: offsets are clamped to `neighbors.length`, so readers always see consistent, possibly
  truncated rows. `output.status.overflow` is set on the GPU, and `output.status.requiredCount` and
  `output.neighborCounts` stay unclamped. The total `requiredCount` saturates at `0xffffffff`;
  offsets remain monotonic after a uint32 carry, preserving the output prefix and setting overflow.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import type {Device} from '@luma.gl/core';
declare const device: Device;
import {GPUNeighborSearch, getGPUNeighborSearchParameterValues, getGPUNeighborSearchWeightTransform, GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUNeighborSearch>[0];
const {positions} = inputs;
const {offsets, neighbors, weights, distances} = inputs.output;
const {overflow} = inputs.output.status;
declare const bounds: readonly [number, number, number, number];

const searchParameters = {mode: 'knn' as const, bounds, rowStandardize: true};
const parameters = new GPUParameterBuffer(device, {
  id: 'knn', format: 'float32', length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  values: getGPUNeighborSearchParameterValues(searchParameters)
});
graph.add(new GPUNeighborSearch({
  mode: 'knn', neighborCount: 8, gridSize: [256, 256], positions,
  parameters: parameters.importToGraph(graph),
  output: {offsets, neighbors, weights, distances, status: {overflow}},
  weightTransform: getGPUNeighborSearchWeightTransform(searchParameters)
}));
```

Use `getGPUNeighborSearchParameterValues({mode: 'radius', bounds, searchRadius: 1.1})`
for a distance band: `searchRadius` must remain positive and finite when packed as float32.
For planar searches, a finite positive distance limit must also have a finite float32 square;
larger limits are rejected rather than producing an empty radius result or an unbounded kNN query.
Rescale the planar coordinates and distance together for such extreme magnitudes.
For kNN, `{mode: 'knn', bounds, maximumDistance: 1.1}` applies an optional cutoff; omit it for
an unbounded search. Both distance props use the spatial context's length unit.

Neighbor weighting uses `weightType: 'binary' | 'inverse-distance' | 'kernel'` (default `'binary'`).
`getGPUNeighborSearchWeightTransform` describes the weights as `'raw'` for `'inverse-distance'`,
or `'row'` when row standardization is enabled.
`GPUNearestFeatureWeights` supports only `'binary'` and `'inverse-distance'`; it rejects `'kernel'`.
Kernel weights accept both `'quartic'` and `'bisquare'` for code 3 and the same
`15/16 * (1 - z²)²` weighting, without changing bandwidth or normalization.

**Spatial context.** `spatialContext` (compile-time) defaults to native planar coordinates. Planar
contexts require metric `'native'`; local frames with `units: 'meters'` are accepted. This operator
measures distance, so `local-lambert-equal-area` frames throw. The resolved context is attached to
`outputWeights.spatialContext`. For
longitude/latitude pass `{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units?:
'meters' | 'kilometers', sphereRadius?}` (`sphereRadius` in the context's length unit, default the mean Earth radius).
Positions are `[longitude, latitude]` degrees, `bounds` are `[minLongitude, minLatitude,
maxLongitude, maxLatitude]` degrees (longitude span at most 360, for example the world), and
`searchRadius`, `maximumDistance`, `distanceFloor` and the output `distances` are great-circle distances in `units`
(meters by default). Any other space or metric throws at construction (project to a planar frame
first). Distances are f32 central angles on a sphere. The lattice stays in degrees: radius mode
visits every cell that can hold a target in the cap (latitude span `radius / R`, longitude span the
exact cap extent `asin(sin(a) / cos(latitude))`, the whole circle when the cap reaches a pole, and
bands wrap across +-180), and kNN stops only when a conservative spherical lower bound of the
distance to every unvisited cell exceeds the k-th distance, so results never depend on `gridSize`.
The bound is deliberately loose near the poles, which only costs speed. Points with latitude beyond
+-90 are invalid. Whole longitude turns do not change query distances or neighbor ordering:
distance evaluation reduces each stored float32 longitude exactly modulo 360 using integer
significand arithmetic, including very large finite values. Candidate lookup shifts that canonical
angle by whole turns toward the lattice midpoint; the shift does not affect distance evaluation.

**Planar precision.** Planar f32 coordinates keep about seven digits: UTM northings (~4e6 m) and
Web Mercator (~1e7) quantize to 0.5 to 1 m and make band edges and ties noisy. Project to a local
frame with `createGPULocalFrame` (or subtract an origin) before upload; the node cannot recover
digits lost in its inputs.

Pass optional `spatialContext` to `getGPUNeighborSearchParameterValues(settings)` to warn on short `searchRadius` or `maximumDistance` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

**Row standardization.** The optional compile-time `rowStandardize` prop is the single source of truth
when given: the weights kernel uses it as a constant and ignores the per-frame `rowStandardize`
parameter. Consumers re-standardize rows over the neighbors left after masking only when
`info.transform` is `'row'`, so set it for any row-standardized band. `rowStandardize: true` with a
`weightTransform` other than `'row'`, or `false` with `'row'`, throws. When it is omitted, the per-frame
field and the `weightTransform` declaration apply as before.

**Output.** `outputWeights` is `output` with `information` and `status` attached, for consumers:
`rowCount`, `transform` (`'row'` when the compile-time `rowStandardize` prop is true, otherwise the
optional compile-time `weightTransform`, else `'unknown'`, because weight kind and row standardization
are per-frame parameters; derive it from the same parameters with
`getGPUNeighborSearchWeightTransform`), `symmetric` (true only for a planar radius self join
without `queryMask` and a declared non-row transform; otherwise `'unknown'`), `provenance`
(`criterion` `'knn'` or `'distance-band'`, and `k`) and `status` (`overflow`, `requiredCount`,
`invalidCount`, and optional `candidateOverflow`). A requested candidateOverflow is overwritten
with 0 on every submission, since neighbor search has no bounded intermediate candidate list.
The inherited `count`, `uncertainCount`, `iterationCount`, `converged`,
`iterationLimitReached` and `approximation` status views are refused: this exact,
non-iterative search does not publish them. Use `offsets[queryRows]` for the clamped
written count and `requiredCount` for the unclamped total.
kNN output has criterion `'knn'`, which `bandwidth: 'adaptive'`
kernel transforms require. PySAL's `min_threshold_distance` (the smallest band that leaves no
island) is the maximum of the returned `distances` of a `k = 1` kNN run.

**Dropped rows.** Targets and self-join rows count only when their mask is nonzero, their coordinates are finite, and
they lie inside the per-frame `bounds`; the others silently become isolates and never a neighbor. A
cross-join query (`queryPositions`) needs only a nonzero `queryMask` and finite coordinates: a query outside `bounds`
returns exactly the brute-force neighbors, so `bounds` need only cover the targets. Great-circle queries also require latitude within ±90 degrees;
longitude wrapping includes targets across the seam, even with regional bounds. A planar query far outside `bounds` (beyond a side or corner by many lattice
widths) cannot stop its kNN ring walk early and visits every lattice cell in one invocation: with 200,000 targets in
a 512 x 512 lattice, k = 8 and 2,000 queries the dispatch took about 7.5 ms with the queries inside and about 25 ms
with them 1e6 beyond a side or corner, same results. When every squared distance of a query overflows float32
(offsets beyond about 1.8e19), all targets tie at infinity and kNN returns the lowest target IDs, as the scan does,
on the grid and on the BVH; the written distances of those rows are infinite.
Pass the optional `output.status.invalidCount` (one uint32) to learn how many masked rows were dropped for a
non-finite coordinate or for lying outside `bounds` (target rows, plus the query rows of a cross
join). A self join applies `mask` to the query rows too. Bounds,
radius, the weight function and row standardization are per-frame parameters, so changing them
never rebuilds the graph. Targets are bucketed with `GPUGridIndex`, whose `boundsBuffer` carries the
per-frame lattice (cells at least `radius` wide) without recompiling. kNN
searches expanding cell rings with a private sorted top-k list and stops once the k-th distance is
strictly inside the visited box, with a slack for f32 rounding. Radius mode counts, scans, emits, and
insertion-sorts each row by ID. No float atomics are used, so outputs are bitwise reproducible.

**Search structure.** The optional compile-time `searchStructure` (`'auto'` by default, `'grid'` or
`'bvh'`) only changes speed: all three give the same IDs, offsets, distances and weights (bitwise),
except which neighbors the one row cut by an exceeded capacity keeps. A grid cell that holds most of
the points (clustered data) makes every query in it scan that cell, so `'bvh'` instead walks a
Hilbert-packed point BVH in near-first depth-first order with the same `(d^2, id)` top-k
list and radius limit, then shares the grid path's count, scan, emit and sort-by-ID stages. Invalid
targets (mask, non-finite, outside the per-frame bounds) get empty boxes, so per-frame inputs still
never recompile. `'auto'` now uses exactly the grid graph, without allocating or building a BVH or
computing an occupancy gate. All four measured 200k-point cases (clustered/uniform, kNN/radius)
favored the grid; occupancy alone did not predict traversal cost. `'bvh'` remains opt-in.
`GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD` is retained for compatibility but no longer used.
`GPUNeighborSearch.getExecutionPlan(shape, device.limits)` exposes this actual route and numeric
grid/list admission without allocating buffers. `GPUNeighborSearch.limits` and
`GPU_NEIGHBOR_SEARCH_ROUTES` use the catalog route descriptor, including planner identity and
byte expressions. Grid offsets require `(columns * rows * timeBucketCount + 1) * 4` bytes,
even for forced BVH under the retained grid admission rule; each kNN ID/distance list needs
`queryRowCount * neighborCount * 4` bytes. Each must fit `maxStorageBufferBindingSize` (128 MiB by default).
`neighborCount` is required in kNN mode and at most 256; temporal buckets are at most 65,536.
The plan also checks the selected route's point records and scratch before declaration:
grid target/query records use three uint32 words per row (four with times); BVH leaves round up
to at least two and a power of two, with `2 * leafCapacity - 1` nodes and five uint32 words per
packed node. It includes node bounds/children/leaf IDs, query positions/counts, optional invalid
flags and internal distance scratch when the shape supplies `distanceCapacity`. Scratch must fit
both `maxStorageBufferBindingSize` and `maxBufferSize`. `timeBucketCount` without `hasTimes`
is rejected just as it is at construction. Set `hasQueryPositions`, `hasQueryMask` and
`includeInvalidCount` to describe the graph's optional routes and diagnostics.
The plan reports these numeric constraints; graph ownership, actual view binding ranges,
resource usage and output aliases still require declaration checks.
The BVH path is planar
only: `'bvh'` with a longitude-latitude `spatialContext` throws, and great-circle search always uses
the grid. The new prop is compile-time and not part of the per-frame parameter
packer, so the Parameters section below is unchanged.

#### Space-time search

Pass `times` (one relative `float32` per target, with a `float32-relative` `temporalContext`) to
search a space-time cylinder: a target is a neighbor only when it also satisfies
`|t_target - t_query| <= timeRadius`, and kNN means the `k` nearest targets inside the window. A
cross join also needs `queryTimes`. The parameter view grows to
`GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH` (15) words holding `timeRadius` and `timeBounds`,
packed by the same `getGPUNeighborSearchParameterValues`. A target or self-join row whose time is non-finite or
outside `timeBounds` is invalid like a row outside `bounds`, and `invalidCount` counts it; a cross-join query
needs only a finite time.

`timeBucketCount` (compile-time, default 1, at most `GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS`) also
indexes the targets by time, so a window much shorter than `timeBounds` skips most of them. It
never changes a result. On 100,000 clustered points with a window of 5% of the time span, 8 buckets
took 29.7 ms against 40.4 ms for one (radius) and 13.1 ms against 25.2 ms (kNN, `k = 16`); 64
buckets gave no further gain. A kNN query whose window holds fewer than `k` targets finishes every
ring, so set `maximumDistance` when windows may be sparse. `searchStructure: 'bvh'` does not take times.

#### How it computes

{/* lens:GPUNeighborSearch */}

::lens{scenario=gpu-spatial-analysis.neighbor-search rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its four nearest centroids, and the degree histogram of all 791 tracts."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=formula caption="Follow the rule: keep the k smallest distance and ID pairs, excluding the tract itself."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=kernel caption="Inspect the grid search kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.neighbor-search rung=graph caption="Trace centroids into the grid and k-nearest search on the command graph."}

{/* parameters: GPUNeighborSearch generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUNeighborSearch` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUNeighborSearch`.

- Packer: `getGPUNeighborSearchParameterValues(settings)`, writes `GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH` (12; without `timeRadius`) or `GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH` (15; with `timeRadius` and `timeBounds`) `float32` values.
- Packer: `getGPUNeighborSearchParameterValues(settings)`, writes `GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH` (12; without `timeRadius`) or `GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH` (15; with `timeRadius` and `timeBounds`) `float32` values.
- Packer: `getGPUNeighborSearchParameterValues(settings)`, writes `GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH` (12; without `timeRadius`) or `GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH` (15; with `timeRadius` and `timeBounds`) `float32` values.

Settings of `getGPUNeighborSearchParameterValues(settings)` for `GPUNeighborSearchSettings & { mode: "radius"; searchRadius: number; maximumDistance?: undefined; }`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `bounds` | `readonly [number, number, number, number]` | `[minX, minY, maxX, maxY]` extent of the search lattice. Targets and self-join rows outside it (or with non-finite coordinates) are excluded: they become isolates and never a neighbor, so it must cover every target of interest (count the dropped rows with `invalidCount`). The queries of a cross join are not clipped by it: a query outside the bounds still finds its exact neighbors (great-circle latitude must be within +-90). For a great-circle search these are degrees `[minLongitude, minLatitude, maxLongitude, maxLatitude]` with a longitude span of at most 360 (`[-180, -90, 180, 90]` for the world). |
| `searchRadius` | `number` | Distance band in spatial-context length units; finite positive float32 in radius mode. |
| `maximumDistance` (optional) | `never` | Forbidden for this settings branch. |
| `weightType` (optional) | `'binary' \| 'inverse-distance' \| 'kernel'` | Weight rule, defaults to `'binary'`. |
| `power` (optional) | `number` | Inverse-distance exponent. Defaults to 1. |
| `distanceFloor` (optional) | `number` | Inverse distance uses `max(d, distanceFloor)`, in the length unit of the distances. Defaults to `1e-6`, the default of `GPUNearestFeatureWeights` too, so coincident points (`d = 0`) get the finite weight `distanceFloor^-power` (`1e6` at power 1). An explicit 0 removes the floor: coincident points then get weight 0, because their weight is not finite, and a zero-weight slot is not a neighbor. Under the default, a coincident neighbor outweighs every ordinary neighbor by many orders of magnitude and takes nearly all of a row-standardized row; pass a floor on the scale of the data when positions repeat (see the class documentation). |
| `kernel` (optional) | `'gaussian' \| 'bisquare' \| 'triangular' \| 'epanechnikov' \| 'quartic' \| 'uniform'` | Kernel profile for `weightType: 'kernel'`. Defaults to `'triangular'`. |
| `rowStandardize` (optional) | `boolean` | Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false. |
| `timeRadius` (optional) | `number` | Time window of a space-time search, in the `temporalContext` unit: a neighbor satisfies `\|t_target - t_query\| <= timeRadius`. Non-positive or non-finite makes every row empty. Required with `times`, and selects the `GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH`-word layout. |
| `timeBounds` (optional) | `readonly [number, number]` | `[minimumTime, maximumTime]` covered by the index. Rows outside it are invalid, like rows outside `bounds`. Required with `timeRadius`. |
| `mode` | `'radius'` | - |

Settings of `getGPUNeighborSearchParameterValues(settings)` for `GPUNeighborSearchSettings & { mode: "knn"; searchRadius?: undefined; maximumDistance?: number | undefined; }`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `bounds` | `readonly [number, number, number, number]` | `[minX, minY, maxX, maxY]` extent of the search lattice. Targets and self-join rows outside it (or with non-finite coordinates) are excluded: they become isolates and never a neighbor, so it must cover every target of interest (count the dropped rows with `invalidCount`). The queries of a cross join are not clipped by it: a query outside the bounds still finds its exact neighbors (great-circle latitude must be within +-90). For a great-circle search these are degrees `[minLongitude, minLatitude, maxLongitude, maxLatitude]` with a longitude span of at most 360 (`[-180, -90, 180, 90]` for the world). |
| `searchRadius` (optional) | `never` | Forbidden for this settings branch. |
| `maximumDistance` (optional) | `number` | Optional kNN distance cutoff in spatial-context length units; omitted means unbounded. |
| `weightType` (optional) | `'binary' \| 'inverse-distance' \| 'kernel'` | Weight rule, defaults to `'binary'`. |
| `power` (optional) | `number` | Inverse-distance exponent. Defaults to 1. |
| `distanceFloor` (optional) | `number` | Inverse distance uses `max(d, distanceFloor)`, in the length unit of the distances. Defaults to `1e-6`, the default of `GPUNearestFeatureWeights` too, so coincident points (`d = 0`) get the finite weight `distanceFloor^-power` (`1e6` at power 1). An explicit 0 removes the floor: coincident points then get weight 0, because their weight is not finite, and a zero-weight slot is not a neighbor. Under the default, a coincident neighbor outweighs every ordinary neighbor by many orders of magnitude and takes nearly all of a row-standardized row; pass a floor on the scale of the data when positions repeat (see the class documentation). |
| `kernel` (optional) | `'gaussian' \| 'bisquare' \| 'triangular' \| 'epanechnikov' \| 'quartic' \| 'uniform'` | Kernel profile for `weightType: 'kernel'`. Defaults to `'triangular'`. |
| `rowStandardize` (optional) | `boolean` | Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false. |
| `timeRadius` (optional) | `number` | Time window of a space-time search, in the `temporalContext` unit: a neighbor satisfies `\|t_target - t_query\| <= timeRadius`. Non-positive or non-finite makes every row empty. Required with `times`, and selects the `GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH`-word layout. |
| `timeBounds` (optional) | `readonly [number, number]` | `[minimumTime, maximumTime]` covered by the index. Rows outside it are invalid, like rows outside `bounds`. Required with `timeRadius`. |
| `mode` | `'knn'` | - |

Settings of `getGPUNeighborSearchParameterValues(settings)` for `GPUNeighborSearchSettings & { mode?: undefined; }`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `bounds` | `readonly [number, number, number, number]` | `[minX, minY, maxX, maxY]` extent of the search lattice. Targets and self-join rows outside it (or with non-finite coordinates) are excluded: they become isolates and never a neighbor, so it must cover every target of interest (count the dropped rows with `invalidCount`). The queries of a cross join are not clipped by it: a query outside the bounds still finds its exact neighbors (great-circle latitude must be within +-90). For a great-circle search these are degrees `[minLongitude, minLatitude, maxLongitude, maxLatitude]` with a longitude span of at most 360 (`[-180, -90, 180, 90]` for the world). |
| `searchRadius` (optional) | `number` | Distance band in spatial-context length units; finite positive float32 in radius mode. |
| `maximumDistance` (optional) | `number` | Optional kNN distance cutoff in spatial-context length units; omitted means unbounded. |
| `weightType` (optional) | `'binary' \| 'inverse-distance' \| 'kernel'` | Weight rule, defaults to `'binary'`. |
| `power` (optional) | `number` | Inverse-distance exponent. Defaults to 1. |
| `distanceFloor` (optional) | `number` | Inverse distance uses `max(d, distanceFloor)`, in the length unit of the distances. Defaults to `1e-6`, the default of `GPUNearestFeatureWeights` too, so coincident points (`d = 0`) get the finite weight `distanceFloor^-power` (`1e6` at power 1). An explicit 0 removes the floor: coincident points then get weight 0, because their weight is not finite, and a zero-weight slot is not a neighbor. Under the default, a coincident neighbor outweighs every ordinary neighbor by many orders of magnitude and takes nearly all of a row-standardized row; pass a floor on the scale of the data when positions repeat (see the class documentation). |
| `kernel` (optional) | `'gaussian' \| 'bisquare' \| 'triangular' \| 'epanechnikov' \| 'quartic' \| 'uniform'` | Kernel profile for `weightType: 'kernel'`. Defaults to `'triangular'`. |
| `rowStandardize` (optional) | `boolean` | Divides each row's weights by their sum (zero-sum rows stay zero). Defaults to false. |
| `timeRadius` (optional) | `number` | Time window of a space-time search, in the `temporalContext` unit: a neighbor satisfies `\|t_target - t_query\| <= timeRadius`. Non-positive or non-finite makes every row empty. Required with `times`, and selects the `GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH`-word layout. |
| `timeBounds` (optional) | `readonly [number, number]` | `[minimumTime, maximumTime]` covered by the index. Rows outside it are invalid, like rows outside `bounds`. Required with `timeRadius`. |
| `mode` (optional) | `never` | Forbidden for this settings branch. |

{/* /parameters */}

### `GPUSpatialWeightsAlgebra`

Combine or expand CSR weights by union, intersection, higher order, self weights, subgraph or block into a new graph.

::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=result caption="Second-order queen neighbours of 791 tracts: each tract links to tracts exactly two steps away."}

Builds a new `GPUSpatialWeights` from existing ones with a capacity-bounded `output` (rows + 1
offsets; it must not alias an input). `operation` is one of:

- `'union'`, `'intersection'`, `'difference'`, `'symmetric-difference'` with `left` and `right` (same
  row count) and an optional `weightRule`: `'left'` (default), `'right'`, `'sum'`, `'min'`, `'max'`,
  `'product'` or `'binary'`. The rule applies where both operands list a link, and one-sided links keep
  their weight. `'binary'` gives 1 everywhere, which reproduces libpysal 4.15 `Graph` set operations.
  Distances come from `left` where it lists the link, else from `right`.
- `'higherOrder'` with `weights`, `order` (1 to 32), `cumulative` (default false) and `workCapacity`
  (intermediate capacity, default the output capacity). Writes the neighbors exactly `order` steps away
  by shortest path, excluding the diagonal, with weight 1; `cumulative` matches `lower_order=True`.
  Cost grows with the squared frontier degree per row, so it suits sparse contiguity and kNN weights.
- `'selfWeight'` with `weights` and `selfWeight` (a number at least 0, or a per-row `float32` view).
  Inserts or replaces `w_ii` with distance 0. It relaxes the `w_ii = 0` invariant. `GPUSpatialLag`,
  `GPUSpatialWeightsTransform`, `GPUSpatialWeightsAlgebra` and `GPUSpatialWeightsSummary` accept it;
  `GPUGlobalSpatialStatistics`, the permutation tests and the local autocorrelation kernels ignore the
  self slot.
- `'subgraph'` with `weights` and a per-row `mask`. Rows keep their IDs: masked rows become empty and
  slots that point at masked rows are dropped.
- `'block'` with `groupIds` and `groupCount`. Row `i` lists every other row of the same group with
  weight 1; IDs at or above `groupCount` have no group. The output holds the sum of squared group
  sizes minus the row count.

`output.status.overflow` (one `uint32`, required) is 1 when the capacity was too small, and the
optional `output.status.requiredCount` holds the slots needed (published as
`outputWeights.status.requiredCount`; omitted there for `'higherOrder'` with
`order > 1`, where intermediate loss makes the complete requirement unknown, so raise
`workCapacity` instead). On overflow the output is still a valid CSR: the leading rows are complete,
one row is a prefix and the rest are empty. Output rows are sorted and unique with finite
non-negative weights. Prefixes and the total saturate at `0xffffffff` before capacity clamping,
so uint32 carries cannot make offsets run backward. `outputWeights.information` carries the transform state (`'binary'` for
`weightRule: 'binary'`, `'higherOrder'` and `'block'`; an unscaled `'raw'` or `'binary'` input
stays so; a row-standardized input becomes `'unknown'` because slots changed) and `symmetric: true`
only when it follows from symmetric operands.

Whole and chunked binary set operations preserve possible diagonal slots in
`information.hasSelfWeights`: union and symmetric difference retain possible diagonal slots from
either operand. Intersection publishes `false` whenever either operand explicitly excludes its
diagonal, including when the other operand is unknown. Difference uses the left operand, since
every retained slot comes from the left.
Chunked algebra applies these rules to each matching row chunk using local declarations with
matrix defaults. The matrix excludes self slots only when every output chunk excludes them;
a single excluded chunk does not establish a whole-matrix guarantee.

Algebra carries either operand loss cause into candidateOverflow, independently of whether
the retained input fits the new destination. Higher-order work scratch loss also sets
candidateOverflow; overflow is only destination slot loss. Increasing destination capacity
cannot repair inherited loss. The named operation and CPU twin require a candidate output name
when operand loss ports or higher-order intermediate work are supplied. Named contiguity operations
always require both output loss names; their candidate scratch is bounded. Correlogram incompleteOrders includes either upstream cause
as well as loss introduced by that order, and incomplete orders suppress global Moran inference.
The CPU twin honors an explicit higher-order workCapacity at every intermediate frontier and
visited union. When omitted, its frontiers are unbounded. `computeSpatialWeightsAlgebraOnCPU`
returns the CSR arrays with its one-word uint32 diagnostics under `status.overflow`,
`status.requiredCount` and `status.candidateOverflow`. After candidate loss, `requiredCount`
in the raw algebra destination counts retained work. Correlogram `intermediates.orderWeights`
uses the algebra's described output and omits `requiredCount` for orders above one, preserving
`candidateOverflow` through row standardization.
This retained-work count is not a bound on the complete result. An
exact-order search can forget a direct neighbor and rediscover it at a later order, so the
retained count can exceed the exact answer. Clear the loss by increasing `workCapacity` before
using the count to describe the complete graph.

Algebra, contiguity and neighbor-search constructors snapshot their declared props and nested
port/status maps once, including inherited getters. Their GPU buffer contents remain per-frame
inputs. Rebuild the contributor to change a view, status destination, operation or capacity.
Both named contiguity backends validate scalar options before output-port requirements and report
the same scalar error even when a candidate destination is missing.
Both forms also require distinct named output values: reusing a name for final and candidate loss
is rejected before either diagnostic can be overwritten.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: Extract<ConstructorParameters<typeof GPUSpatialWeightsAlgebra>[0], {left: unknown; right: unknown}>;
const {left: contiguity, right: knn} = inputs;
const {offsets, neighbors, weights} = inputs.output;
const {overflow, requiredCount} = inputs.output.status;

graph.add(new GPUSpatialWeightsAlgebra({
  operation: 'union', left: contiguity, right: knn, weightRule: 'binary',
  output: {offsets, neighbors, weights, status: {overflow, requiredCount}}
}));
```

#### How it computes

{/* lens:GPUSpatialWeightsAlgebra */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its second-order neighbours, and the degree histogram of all 791 tracts."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=kernel caption="Inspect the frontier expansion, sort and compaction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-algebra rung=graph caption="Trace first-order queen weights into second-order expansion on the command graph."}

### `GPUSpatialWeightsSummary`

Summarise CSR weights into S0, S1, S2, slot counts, asymmetry, isolates and degree range.

::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=result caption="Read S0, S1, S2 and the isolate count of the queen weights of 791 tracts."}

Summary statistics of a weights matrix, written into `output: {statistics, counts, cardinality?}`.
`statistics` (`float32`, at least 3) receives `S0`,
`S1 = 1/2 sum (w_ij + w_ji)^2` and `S2 = sum_i (row_i + col_i)^2`, as in libpysal. `counts` (`uint32`,
at least 6, see `GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT`) receives the used slots, the asymmetric slots,
the isolates, the minimum and maximum cardinality and the empty rows. The optional `cardinality` is the
per-row active neighbor count. `symmetryTolerance` accepts a numeric default (0 for exact comparison)
or a packed one-float view from `getGPUSpatialWeightsSummaryParameterValues` hosted in a
`GPUParameterBuffer`. Updating that buffer changes the reverse-weight comparison on the same compiled
graph, for both whole and chunked weights. View shapes and chunk layout stay compile-time. Sums
run in a fixed order. Isolates are rows with no active slot, the one island definition of every
statistic (libpysal `islands` for a W built without zero-weight entries); `emptyRows` counts rows with
no slot at all. Cardinality counts active slots, so a zero-weight slot is not a neighbor: this deviates
from libpysal `cardinalities`, which counts every listed neighbor, and agrees with it for a W built
without zero-weight entries (pinned against libpysal in the specs). The used slots count stays the
listed total. Connected components are not computed. A pair of unequal weights counts as two
asymmetric slots, where libpysal `asymmetry()` returns pairs.

Any nonzero `weights.status.overflow` or `candidateOverflow` (at matrix or chunk level) suppresses
the incomplete summary: the three statistics become NaN, the six counts and optional per-row
cardinalities become zero. Entries beyond the published statistics/counts layouts stay untouched.
Clearing all loss flags restores the results on the next encoding of the same compiled graph.

#### How it computes

{/* lens:GPUSpatialWeightsSummary */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=datum bind=page datum=420 caption="Inspect tract 420: its degree among the 791 tracts, beside the global moments."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=formula caption="Follow three moments: S0, S1 and S2 from row sums, column sums and reverse weights."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=kernel caption="Inspect the row, column and blocked reduction kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-summary rung=graph caption="Trace the row, column and global reduction passes on the command graph."}

{/* parameters: GPUSpatialWeightsSummary generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialWeightsSummary` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpatialWeightsSummary`.

- Packer: `getGPUSpatialWeightsSummaryParameterValues(settings?)`, writes `GPU_SPATIAL_WEIGHTS_SUMMARY_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUSpatialWeightsSummaryParameterValues`:

| Field | Type | Description |
|---|---|---|
| `symmetryTolerance` (optional) | `number` | Non-negative absolute tolerance for counting unequal reverse weights. Defaults to 0. |

{/* /parameters */}

### `GPUSpatialWeightsTranspose`

Reverse every link of CSR weights, turning outgoing weights into incoming ones.

::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=result caption="Transposed row-standardised queen weights: rows no longer sum to one, since they hold incoming weights."}

The CSR transpose `W'` of any `GPUSpatialWeights`, square or rectangular (libpysal `Graph` with focal and
neighbor swapped). `columnCount` (default: the row count) sets the number of rows of the result for
cross weights whose neighbor IDs index target rows; slots with a neighbor ID at or above it are dropped,
and columns that nothing reaches give empty rows. `output` is a caller-owned CSR with `columnCount + 1`
offsets and at least the input capacity in `neighbors` and `weights` (and `distances` when the input has
them); slots past `offsets[columnCount]` are zeroed. Slots are keyed by column, stable-sorted, counted, scanned and gathered, so every output
row is ascending and the result is bitwise reproducible with no atomics. The optional
`output.asymmetricSlots`
(one `uint32`, square weights only, with `symmetryTolerance`, default 0) counts the positions `(i, j)` of
the union of both patterns whose weights differ or that are present in one only, so a one-way link counts
twice and 0 means `W = W'`. Reference: libpysal 4.15 (no deviations; the check is exact by default).

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUSpatialWeightsTranspose>[0];

graph.add(new GPUSpatialWeightsTranspose(inputs));
```

`symmetryTolerance` accepts a numeric default or a packed one-float view from
`getGPUSpatialWeightsTransposeParameterValues` hosted in a `GPUParameterBuffer`. Updating that
buffer changes the symmetry comparison on the same compiled graph. Equality at the tolerance is
symmetric; a missing reverse slot always differs, including when the present slot's weight is zero.
CSR values, distances and tolerance are per-frame; view shapes, column count and optional outputs
stay compile-time. The tolerance is unused when `asymmetricSlots` is omitted.

`outputWeights` carries the input loss flags (a truncated `W` gives a truncated `W'`) and the
input `symmetric` flag for square output. Packed transpose omits input `count` and `requiredCount`:
dropping columns can change those cardinalities. Read the terminal output offset for the retained
count. Destination `output.status` ports are refused because packed transpose does not write them;
use chunked output for destination capacity diagnostics. Requested distances require source
distances on every input chunk. Writable output fields must have disjoint logical bytes, with
disjoint aligned ranges for fields bound together. A `'row'` or `'variance'` state becomes `'unknown'`;
`'double'` also becomes `'unknown'` unless the output is square and the input declares
`symmetric: true`: the handle declares no source column bound, so out-of-range neighbors can
remove normalized mass even when `columnCount` equals the row count.

#### How it computes

{/* lens:GPUSpatialWeightsTranspose */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its incoming weights, which were outgoing weights of its neighbours."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=kernel caption="Inspect the column sort and row boundary search kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-transpose rung=graph caption="Trace row standardisation into transposition on the command graph."}

{/* parameters: GPUSpatialWeightsTranspose generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialWeightsTranspose` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUSpatialWeightsTranspose`.

- Packer: `getGPUSpatialWeightsTransposeParameterValues(settings?)`, writes `GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUSpatialWeightsTransposeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `symmetryTolerance` (optional) | `number` | Non-negative absolute reverse-weight tolerance. Missing reverse slots always differ. Defaults to 0. |

{/* /parameters */}

### `GPUSpatialWeightsValidation`

Flag rows of uploaded CSR weights whose offsets, neighbour order, range, diagonal or weights break the invariants.

::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=result caption="Validation flags per tract for queen weights; expect zero of 791 rows flagged."}

Checks the `GPUSpatialWeights` invariants of weights that were built elsewhere and uploaded, on the GPU:
`offsets[0] = 0`, `targetRowCount + 1` offsets in non-decreasing order with
`offsets[targetRowCount] <= neighbors.length`, every neighbor ID below `columnCount`, strictly
ascending neighbor IDs within each row (sorted and unique), the declared diagonal policy, and
finite non-negative weights. `columnCount` is the source-row count and defaults to the target-row
count for square matrices. For cross weights pass the source bound explicitly: it is
`values.length / lagColumnCount` for a lag with `lagColumnCount` value columns; the validator's
`columnCount` counts source rows, not value columns.

`diagonal: 'forbid'` (default) rejects every self slot; `'allow'` accepts self slots;
`'require-unit'` requires one exactly unit-weight diagonal per target row and requires
`columnCount >= targetRowCount`. Missing or nonunit required diagonals set bit 8. Do not remove
unit diagonals required by a HAC consumer. Two-row identity CSR `[0,1,2]`, `[0,1]`, `[1,1]` with
`diagonal: 'require-unit'` gives flags `[0,0]`. A rectangular two-target/three-source CSR
`[0,1,2]`, `[2,0]`, `[1,1]` with `columnCount: 3` and default forbidden diagonal also gives
`[0,0]`; under the default square bound its neighbor 2 is invalid.

`output.status.invalidCount` (one `uint32`) receives the number of rows with at least one
violation; the optional `output.flags` (one `uint32` per row) receives the bitmask of
`GPU_SPATIAL_WEIGHTS_VIOLATION` (`offsets` 1, `neighborRange` 2, `order` 4, `self` 8, `weight` 16). Run it
once after upload and before using the weights; there is no GPU sort or dedupe, so fix a failing CSR on the
CPU. Submit validation and read the actual output buffer before submitting consumers; an attached
`invalidCount` is host-readback metadata, not an automatic GPU loss gate. A zero count certifies
the declared CSR policy, independently of producer capacity-loss flags.
`outputWeights` is the checked weights with `status.invalidCount` attached, preserving supplied
subtype fields such as `neighborCounts` by identity through both public `props.weights` and
`outputWeights`, including fields on chunk weights and inherited getters. A separate fixed-field
snapshot supplies the CSR bindings. Output maps are captured at construction, including a single
read of a status getter. `invalidCount` must be a one-word view;
use a one-word slice of a larger allocation rather than a multiword destination.

The GPU and CPU operation forms require a positive `rowCount` equal to `offsets.length - 1`,
including when the optional flags output is omitted. Invalid CSR contents are reported as
violation bits and counts; a disagreement in this declared shape is refused.
Both named backends capture weights, output and status maps once, so all input names and output
destinations belong to the same construction-time configuration.
Later changes to the caller's props object do not change the operation; input CSR contents
remain per-frame data.
Graph names must be distinct across all ports; in particular `flags` and `invalidCount` cannot
share a destination name. Both named constructors reject such collisions before building outputs.
Declared ports may be exposed by inherited or non-enumerable getters. Named validation,
chunk transforms and transpose status inheritance capture these fields by name; enumeration
alone does not decide whether a declared port or loss flag exists.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUSpatialWeightsValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUSpatialWeightsValidation>[0];

graph.add(new GPUSpatialWeightsValidation(inputs));
```

Keep a separate consumer graph so its submission can follow the host check. The application
provides `submitValidation`, which submits and completes the graph above, and retains the
backing `Buffer` and byte offset of that graph's `invalidCount` view:

```ts
import type {Buffer} from '@luma.gl/core';
declare const invalidCountBuffer: Buffer;
declare const invalidCountByteOffset: number;
declare const submitValidation: () => Promise<void>;
declare const submitConsumers: () => Promise<void>;

async function requireValidCSR(buffer: Buffer, byteOffset = 0): Promise<void> {
  const bytes = await buffer.readAsync(byteOffset, 4);
  const invalidCount = new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
  if (invalidCount !== 0) throw new Error(`CSR validation found ${invalidCount} invalid rows`);
}

await submitValidation();
await requireValidCSR(invalidCountBuffer, invalidCountByteOffset);
await submitConsumers();
```

#### How it computes

{/* lens:GPUSpatialWeightsValidation */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=datum bind=page datum=420 caption="Inspect tract 420: its violation flags, and the total number of invalid rows."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=kernel caption="Inspect the per-row invariant check kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-validation rung=graph caption="Trace queen contiguity into validation on the command graph."}

### `GPUChunkedSpatialWeightsValidation`

Validate chunked CSR weights with one rows dispatch per chunk.

Runs `GPUSpatialWeightsValidation` over a `GPUChunkedSpatialWeights` whose chunks are consecutive
target-row ranges, each with its own CSR buffers. Chunk `c` checks its own offsets and capacity,
neighbor IDs against `columnCount` (the total target-row count by default), and the declared
`diagonal` policy using global target rows. `output.flags` and `output.status.invalidCount` follow
the whole-CSR contract in global row order. Dispatch count is linear in the chunk count and
results do not depend on the chunking. Read the aggregate invalid count before consumers.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUChunkedSpatialWeightsValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUChunkedSpatialWeightsValidation>[0];

graph.add(new GPUChunkedSpatialWeightsValidation(inputs));
```

#### How it computes

{/* lens:GPUChunkedSpatialWeightsValidation */}

### `GPUChunkedSpatialWeightsTranspose`

Reverse every link of chunked CSR weights into packed or chunked CSR.

Keep the input's child and outer loss metadata: a larger transpose destination cannot recover
omitted source neighbors. When child loss metadata exists, packed `candidateOverflow` combines
root and child nonzero `overflow`/`candidateOverflow` flags and zero `complete` certificates.
Removing a child port can conceal incomplete input. Packed output forwards matrix-level input status except `count` and `requiredCount`.
An input `requiredCount` needs its producer's final overflow flag, either at the same scope or
at the root when it covers that chunk. Verify completeness before omitting `converged`,
`iterationLimitReached` or `unresolvedCount` and passing the CSR.
Read `offsets[columnCount]` for the retained slot count. Packed output writes CSR offsets,
neighbors, weights, optional measured distances and optional `asymmetricSlots`; it refuses
destination `output.status` because it has no writer for those ports. Use independently chunked
output when destination capacity diagnostics are needed. Requested distances require measured
distances on every input chunk. Writable destinations must have disjoint logical bytes and,
when bound together, disjoint aligned binding ranges. The packed neighbor/weight/distance
bindings and `columnCount + 1` offsets must each fit the device limit; chunked input alone does
not remove that packed-output limit.

The [chunked transpose guide](chunked-spatial-weights-transpose.md)
details output layouts, capacity status and binding limits.

The packed-output form below remains supported. For independently partitioned output chunks,
use the caller-owned `output.chunks` layout in that guide, including a status port per chunk.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUChunkedSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUChunkedSpatialWeightsTranspose>[0];

graph.add(new GPUChunkedSpatialWeightsTranspose(inputs));
```

The per-frame `symmetryTolerance` view uses `getGPUSpatialWeightsTransposeParameterValues` in
both output forms. See the guide for compilation-time shapes and output-metadata semantics.

#### How it computes

{/* lens:GPUChunkedSpatialWeightsTranspose */}

### `GPUChunkedSpatialWeightsTransform`

Standardise, kernel-weight or average reciprocal weights in chunked CSR matrices.

Takes the operations of `GPUSpatialWeightsTransform`. `row`, `binary` and `kernel` are row-local and run once per chunk. `double` and `variance` write per-row partials into one global array, take one fixed-order sum and apply per chunk. `average-reciprocal` transposes the chunks into one whole scratch CSR and looks the reverse weight up there, so the summed chunk capacity must fit one binding. Results equal the whole transform at any chunking, up to float32 association of the partial sums. Optional `output` entries give a destination per chunk. Reciprocal averaging requires a separate destination per chunk. Other operations write in place when a destination is omitted.
Kernel profiles include both `bisquare` and `quartic`, with the same
`15/16 max(1-z²,0)²` normalization as the whole transform.
Row, binary and kernel require fitting chunk bindings and chunk-local scratch; they allocate
no whole-matrix row partial array. Double and variance allocate `rowCount` float32 partials in
one binding, plus scalar reduction scratch. `average-reciprocal` allocates rowCount + 1 uint32 scratch
offsets plus summed chunk-capacity uint32 neighbors and float32 weights, each in one binding.
These route-specific scratch limits apply in addition to every actually bound caller view's
aligned range. An unused neighbor binding does not impose a device range limit on row, binary,
kernel, double or variance, though its graph identity is still checked.
A fixed `bandwidth` parameter view is shared by every chunk and uses
`getGPUSpatialWeightsTransformParameterValues`, exactly as for the whole-CSR transform.
Matrix-level provenance and spatial context are inherited by plain CSR chunks; child row counts
remain local. Adaptive kernels therefore accept outer-only kNN provenance.
Published chunks retain their resolved local provenance and the actual transform state, so
chunk-only kNN declarations also survive a binary transform before an adaptive kernel.

A chunk information or provenance field that is present but undefined does not erase the matrix
value; a defined chunk field takes precedence.
Chunk layout and destination-array entries are captured once at construction. Replacing a caller
array entry later does not redirect row-local, global-normalization or symmetrization writes;
`outputWeights` names the views actually written. Their GPU contents remain per-frame data.
Symmetrization accepts local loss-status ports on complete chunks. Its private transpose reads
only CSR views; the transformed public description retains all original loss flags so downstream
statistics still suppress incomplete neighborhoods.
Chunked variance normalization rescales a row when its squared norm overflows or underflows,
matching the whole transform. Double normalization retains nonfinite partial sums through the
global reduction so a nonfinite total produces zero weights in every chunk.
Each destination uses a separate buffer or the exact source byte range, as in the whole transform;
distinct source/destination subviews are refused for operations that read old weights. Kernels
may share the unused old-weights buffer. Chunks dispatched separately may occupy disjoint logical
ranges of one buffer, including in-place row normalization; overlapping source/destination ranges
across chunks are refused when they would clobber a later read.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUChunkedSpatialWeightsTransform} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: ConstructorParameters<typeof GPUChunkedSpatialWeightsTransform>[0];
const {weights: chunked} = inputs;

graph.add(new GPUChunkedSpatialWeightsTransform({operation: 'row', weights: chunked}));
```

#### How it computes

{/* lens:GPUChunkedSpatialWeightsTransform */}

### `GPUChunkedSpatialWeightsAlgebra`

Combine chunked CSR weights with row-local set algebra.

Supports `union`, `intersection`, `difference`, `symmetric-difference`, `selfWeight` and `subgraph` of `GPUSpatialWeightsAlgebra`. One count, scan and emit pipeline runs per chunk and writes the matching chunk of a `GPUChunkedSpatialWeightsOutput`, so output rows do not depend on the chunking, there is no repacking, and overflow is reported per chunk. Binary operands and the output must be chunked identically. `higherOrder` and `block` are not chunked; use `GPUSpatialWeightsAlgebra`.

Requested distance outputs require distances on each corresponding input chunk. Binary operations
always require left distances; union and symmetric difference also require right distances.
Packed and chunked transpose outputs require distances on every input chunk when distances are
requested. Construction refuses a missing measurement; omit that output when distances are unavailable.
Declared spatial contexts on either operand or its chunks must have compatible metric results (metric, effective length unit, radius and ellipsoid), and the resolved context is validated. The resolved
context is carried on the matrix and output chunks, including when only the right operand declares it.
Resolution accumulates explicit declarations: an omitted outer unit cannot hide a conflict between
two chunk units, and compatible chunk units remain explicit on the returned matrix.

The producer's outputWeights adds aggregate matrix overflow for destination truncation across output
chunks and candidateOverflow for either operand loss cause or intermediate loss. Writable output
ranges, including diagnostics, must be disjoint across chunks. Its chunks omit local loss fields so it can feed statistics directly; callers
retain per-chunk diagnostics in their original output allocation. Aggregate words are overwritten
on every submission, including recovery. Lattice and fuzzy-contiguity producers, which have no
bounded intermediate stage, overwrite a requested candidateOverflow with zero each submission.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import {GPUChunkedSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';
declare const inputs: Extract<ConstructorParameters<typeof GPUChunkedSpatialWeightsAlgebra>[0], {left: unknown; right: unknown}>;
const {left, right, output} = inputs;

graph.add(new GPUChunkedSpatialWeightsAlgebra({operation: 'union', left, right, output}));
```

#### How it computes

{/* lens:GPUChunkedSpatialWeightsAlgebra */}

Lens coverage is pending for a supported row-local algebra scenario. The named whole-CSR
`spatial-weights-algebra` scenario demonstrates `higherOrder` with order 2, which this class cannot
run, so that scenario does not establish coverage for this chunked variant.

### `GPUMapColoring`

Assign each tract a colour index so that no two neighbouring tracts share one.

::lens{scenario=gpu-spatial-analysis.map-coloring rung=result caption="At most five colours separate 791 rook-adjacent tracts; neighbouring tracts never match."}

Consumes an adjacency pattern only, so it takes no `spatialContext`. Weights may carry their own
context; no compatibility check is needed because coloring uses only adjacency.

Greedy parallel graph coloring (Jones-Plassmann, largest degree first by default) over a
symmetric `GPUSpatialWeights` such as contiguity, for choropleth maps where neighbors need distinct
colors (QGIS topological coloring). `output.colors` holds one `uint32` per row,
`GPU_MAP_COLORING_UNCOLORED` for rows left unfinished. Optional outputs are `colorCount`, `conflictCount`
(pairs sharing a color; zero certifies a proper coloring only with symmetric adjacency and
completed coloring) and `status: {converged, iterationCount}`. The per-frame
`parameters` view (`getGPUMapColoringParameterValues({seed})`, `GPU_MAP_COLORING_PARAMETER_LENGTH` `uint32`
words) selects the priorities without recompiling, and `maximumIterations` (default 64, at most 1024)
bounds the rounds. The compile-time `priority` prop selects the row order:
`'largest-degree-first'` (default; Welsh-Powell order, degree is the CSR row length, ties broken by the seeded
hash and then the lowest ID) or `'random'` (seeded hash, then lowest ID). The result equals sequential greedy
coloring in that order, a proper and seed-deterministic coloring that uses at most the maximum degree plus one
colors, not the minimum number. Largest-degree-first typically saves a color on planar maps (6 versus 6 to 7
on Delaunay graphs, matching networkx `largest_first`) at about the same number of rounds. Different seeds
give different valid colorings.

The CPU operation honors an explicit `maximumIterations` with ascending-row sweeps: a row waits
while any higher-priority neighbor remains uncolored. It stops when every row is colored or the
budget is exhausted; `converged`, `iterationCount`, color count and uncolored sentinels describe
that partial result. GPU scheduling can complete different rows within a round, so a budgeted
CPU result need not match a GPU partial coloring or round count. Without an explicit budget the
CPU keeps its complete, single priority-ordered greedy sweep. Uncolored sentinel pairs never count
as conflicts.
Read `converged` together with `conflictCount`: a partial coloring can have zero conflicts while
some rows remain `GPU_MAP_COLORING_UNCOLORED`. An isolated row completes in one round with color 0,
`colorCount = 1`, `conflictCount = 0` and `converged = 1`; all supplied status words are rewritten
on each submission. On asymmetric inputs the conflict diagnostic counts an equal-color CSR entry
only when `neighbor > row`; a descending entry such as `1 -> 0` is not counted. Zero conflicts
therefore do not certify an asymmetric coloring as proper. Proper coloring requires a symmetric
pattern. Self slots affect degree priority and never count as conflicts.
If whole-matrix or any chunk's `overflow` or `candidateOverflow` is nonzero, every color becomes
`GPU_MAP_COLORING_UNCOLORED`; `colorCount`, `conflictCount` and `converged` become zero.
`iterationCount` retains the rounds actually run. These outputs are restored by their ordinary
writers on the next complete frame. Independently written scalar outputs may occupy disjoint
logical words of one buffer, including when a weights loss port is supplied.
The contributor captures nested view descriptors, and both operation forms capture scalar
configuration and nested input/output names once at construction. Later mutation of the caller's
budget or status-name map cannot change CPU execution
or GPU lowering; per-frame parameter-buffer contents remain mutable. GPU operation budgets are
validated against the contributor's 1024-round limit; the CPU operation permits larger budgets.
Both operation forms accept optional `weights.status: {overflow, candidateOverflow}` names.
They retain those scalar inputs and apply the same per-frame suppression and recovery contract,
including preserving `iterationCount` on incomplete inputs.

```ts
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
declare const graph: GPUCommandGraph<void>;
import type {Device} from '@luma.gl/core';
declare const device: Device;
import {GPUContiguityWeights, GPUMapColoring, getGPUMapColoringParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
declare const inputs: ConstructorParameters<typeof GPUMapColoring>[0];
declare const adjacency: ConstructorParameters<typeof GPUContiguityWeights>[0];
const {colors, conflictCount} = inputs.output;
const {converged} = inputs.output.status ?? {};
const parameters = new GPUParameterBuffer(device, {id: 'coloring', format: 'uint32', length: 1});

const contiguity = new GPUContiguityWeights({...adjacency, criterion: 'rook'});
graph.add(contiguity);
graph.add(new GPUMapColoring({weights: contiguity.outputWeights, parameters: parameters.importToGraph(graph), output: {colors, conflictCount, status: {converged}}}));
parameters.write(getGPUMapColoringParameterValues({seed: 3}));
```

#### How it computes

{/* lens:GPUMapColoring */}

::lens{scenario=gpu-spatial-analysis.map-coloring rung=datum bind=page datum=420 caption="Inspect tract 420: its colour and its neighbours, which all differ."}
::lens{scenario=gpu-spatial-analysis.map-coloring rung=kernel caption="Inspect the priority round kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.map-coloring rung=graph caption="Trace rook contiguity into rounds of greedy colouring on the command graph."}

{/* parameters: GPUMapColoring generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMapColoring` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUMapColoring`.

- Packer: `getGPUMapColoringParameterValues(settings?)`, writes `GPU_MAP_COLORING_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUMapColoringParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Seed of the hashed priorities, a uint32. Default 0. Different seeds give different valid colorings (with `'largest-degree-first'` the seed only orders rows of equal degree). |

{/* /parameters */}
