# GPUChunkedSpatialWeightsTranspose

See also the [spatial weights API](operators-weights.md#gpuchunkedspatialweightstranspose).

`GPUChunkedSpatialWeightsTranspose` accepts `GPUChunkedSpatialWeights` and writes either a packed
CSR or independently partitioned output chunks. Import it and
`GPUChunkedSpatialWeightsTransposeOutput` from
`@luma-contributors/analysis/gpu-spatial-analysis`.

```ts
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUChunkedSpatialWeightsTranspose,
  type GPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeightsTransposeOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// input views must belong to graph; columnCount >= 2 for this two-chunk example.
function addTranspose(
  graph: GPUCommandGraph,
  input: GPUChunkedSpatialWeights,
  columnCount: number,
  outputCapacities: readonly [number, number]
) {
  const firstRowCount = Math.floor(columnCount / 2);
  const makeChunk = (name: string, rowStart: number, rowCount: number, capacity: number) => ({
    rowStart,
    rowCount,
    weights: {
      offsets: createTransientView(graph, `${name}-offsets`, 'uint32', rowCount + 1),
      neighbors: createTransientView(graph, `${name}-neighbors`, 'uint32', capacity),
      weights: createTransientView(graph, `${name}-weights`, 'float32', capacity),
      status: {
        count: createTransientView(graph, `${name}-count`, 'uint32', 1),
        requiredCount: createTransientView(graph, `${name}-required`, 'uint32', 1),
        overflow: createTransientView(graph, `${name}-overflow`, 'uint32', 1)
      }
    }
  });
  const output: GPUChunkedSpatialWeightsTransposeOutput = {
    chunks: [
      makeChunk('first', 0, firstRowCount, outputCapacities[0]),
      makeChunk('second', firstRowCount, columnCount - firstRowCount, outputCapacities[1])
    ]
  };
  const transpose = new GPUChunkedSpatialWeightsTranspose({weights: input, columnCount, output});
  graph.add(transpose);
  // Feed transpose.outputWeights directly to a consumer accepting GPUChunkedSpatialWeights.
  return {outputWeights: transpose.outputWeights, output};
}
```

`outputCapacities` supplies independent slot capacities; each chunk's offsets and slot views must
fit the device binding limit. For example, two eight-slot input chunks can use `[8, 8]` output
capacities with a 32-byte binding limit. Summing all input capacities into each output chunk would
request a 64-byte binding and defeat this route's chunking.

After submission, first read `outputWeights.status.candidateOverflow`: any inherited input loss
must be repaired at its producer and retried before the transpose's requirements certify sizing.
Then read each `output.chunks[index].weights.status.overflow` and `requiredCount`. Grow only an
overflowing chunk to its certified requirement, or repartition its rows if that capacity cannot
fit one binding. `count` is its readable retained prefix, not the unclamped requirement. Rebuild
the transpose and its consumer graph after changing capacities; same-shaped view correction does
not change capacities. The outer `outputWeights.status.overflow` reports whether any output chunk
truncated. A complete retry has both outer loss flags zero.

Each output chunk is an ordinary CSR with offsets starting at zero. Chunks cover
`[0, columnCount)` consecutively, with at least one row and one slot of physical capacity each.
Their boundaries can differ from the input chunk boundaries. Neighbor IDs remain global input
row IDs, ascending within each output row. Empty rows and entirely empty chunks are supported;
input padding and neighbor IDs outside `[0, columnCount)` are discarded.

Each chunk requires separate one-element `uint32` status views. `count` is the number of retained
slots, `requiredCount` is the exact requirement before truncation, and `overflow` is one when this
chunk dropped slots. An overflowing chunk retains the prefix of its complete CSR, possibly ending
inside a row; all its offsets are clamped to capacity. Extra capacity in a different chunk is not
borrowed. Unused slots are zeroed, including distances. These values reset on every encoding.
A supplied chunk `candidateOverflow` is also overwritten with zero: this transpose has no bounded
candidate stage. Matrix-level input loss remains attached to the returned description.
Matrix-level input `count` and `requiredCount` are omitted because output chunks can truncate;
read each output chunk's own count and requirement.
Other destination status fields are refused because transpose does not compute them; omit those
ports and run the corresponding diagnostic contributor afterward.

`asymmetricSlots` counts mismatched positions in the union of the patterns of the input and its
transpose, using `symmetryTolerance` (default zero). It is computed from the input and remains
exact even when output chunks overflow. A structurally absent reverse slot counts as a mismatch
even when the present weight is zero. The input must satisfy the usual CSR invariants, including
sorted unique neighbor IDs in each row.

The described `outputWeights` retains the output layout and spatial context. For chunked output it
aggregates loss at matrix level, as chunked algebra does: `status.overflow` is one when any output
chunk dropped slots, and `status.candidateOverflow` is one when the input reported `overflow` or
`candidateOverflow`, or any root/child positive `complete` certificate is zero (a larger output
cannot recover that loss). Certificates are read with their positive polarity on every submission,
so repairing a producer from `complete = 0` to `complete = 1` clears inherited loss. Its chunk
descriptions keep `count`
and `requiredCount` but no loss flags, so local Moran, hot spot, the additional and global
statistics and another chunked transpose accept it; before this they refused it at construction.
Packed output forwards matrix-level input status except `count` and `requiredCount`. When child
loss flags or completeness certificates are present, its `candidateOverflow` aggregates all
inherited loss, including `complete == 0`.

An input `requiredCount` must travel with the producer's final overflow flag for those slots.
For a child chunk, the flag can be on that chunk or aggregated at the matrix root by the producer.
Verify completeness before omitting `converged`, `iterationLimitReached` or `unresolvedCount` and
passing the CSR.

Read `offsets[columnCount]` for the retained slot count. The caller's output chunks still receive their
own capacity status. Output views must not share buffers with
matrix-level or chunk-level input status views. Row, variance and double transform metadata becomes
`unknown`: chunk capacity may truncate the result and break global normalization. Packed output
also reports `unknown` for double normalization: no source column bound is declared by the input
handle, so out-of-range neighbors can remove mass even when `columnCount` equals the focus row
count (two rows with neighbors `[2, 0]`, weights `[0.5, 0.5]` and `columnCount: 2` retain only
`0.5`). The exception is square packed output of input declared `symmetric: true`, whose neighbors
index `[0, rowCount)`: every slot is kept and `double` is carried over. Chunked output
symmetry metadata is also `unknown` because capacity can truncate it.

The packed output API remains supported: pass `offsets`, `neighbors`, `weights`, optional
`distances` and optional `asymmetricSlots` directly in `output`. Its capacity must still be at least
the sum of input capacities.
Packed output does not accept destination `status` ports: it preserves input loss and cannot
truncate at that capacity. It omits input `count` and `requiredCount`, which can change when
out-of-range columns are dropped; read the terminal output offset for the retained count.
Omit `output.status`, or use chunked output for destination diagnostics.

## Limits and work

No chunked output path packs the whole CSR or allocates scratch for all output rows. Every input
sort and every output scan fits its own storage binding; device-dependent binding checks include
alignment prefixes and run before scratch allocation or node creation. A rejected binding preflight
leaves the graph available for retry with corrected views. The `outputWeights` description and its
chunk wrappers stay stable, and their CSR view properties and status ports follow same-shaped
corrections. Before scratch allocation, corrected inputs and outputs are checked again for the
original port presence, shapes, row ranges, required distances and aliases. Removing a port or
changing its shape requires rebuilding the contributor. A rejected correction can be repaired
with distinct same-shaped views and retried on the same graph, including descriptions saved
before a failed preflight.
Replace views before generating successful nodes or wiring downstream consumers; shapes, formats,
chunk ranges and optional-view presence remain fixed at construction. Packed input chunk capacities
are limited to `2^31` slots by published `GPUSort`.
The summed input capacity is at most `2^32 - 1`, or `2^31 - 1` with symmetry counting, so counts
cannot wrap. `columnCount` is in `[1, 2^32 - 2]`.

The shared transpose pipeline runs once per output range and repeats stable input sorts for that
range. Work grows with `inputChunks * outputChunks`; symmetry checking additionally visits input
chunk pairs. The construction-time `estimatedCommandNodeCount` includes conservative sort, scan
and reduction allowances and must fit the shared 32,768-node budget. Choose fewer, larger row
chunks when the command budget is exceeded, while respecting device binding sizes.

View shapes, chunk ranges, `columnCount` and optional output presence are compile-time, matching
the packed transpose. CSR buffer contents and a supplied `symmetryTolerance` view may change
between encodings; a numeric tolerance sets the construction-time default. There is no CPU operation twin;
the GPU specs compare against an independent column-first CPU transpose.

{/* parameters: GPUChunkedSpatialWeightsTranspose generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

## `GPUChunkedSpatialWeightsTranspose` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPUChunkedSpatialWeightsTranspose`.

- Packer: `getGPUSpatialWeightsTransposeParameterValues(settings?)`, writes `GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH` (1) `float32` values.
  - Input port: `symmetryTolerance`.

Settings of `getGPUSpatialWeightsTransposeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `symmetryTolerance` (optional) | `number` | Non-negative absolute reverse-weight tolerance. Missing reverse slots always differ. Defaults to 0. |

{/* /parameters */}
