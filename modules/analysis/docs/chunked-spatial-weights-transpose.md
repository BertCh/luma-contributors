# GPUChunkedSpatialWeightsTranspose

`GPUChunkedSpatialWeightsTranspose` accepts `GPUChunkedSpatialWeights` and writes either a packed
CSR or independently partitioned output chunks. Import it and
`GPUChunkedSpatialWeightsTransposeOutput` from
`@luma-contributors/analysis/gpu-spatial-analysis`.

```ts
const output: GPUChunkedSpatialWeightsTransposeOutput = {
  chunks: [
    {
      rowStart: 0,
      rowCount: firstRowCount,
      weights: {
        offsets: firstOffsets, // firstRowCount + 1 uint32 entries
        neighbors: firstNeighbors, // global input row IDs
        weights: firstWeights,
        distances: firstDistances, // optional; requires distances on every input chunk
        status: {
          count: firstCount,
          requiredCount: firstRequiredCount,
          overflow: firstOverflow
        }
      }
    },
    {
      rowStart: firstRowCount,
      rowCount: columnCount - firstRowCount,
      weights: {
        offsets: secondOffsets,
        neighbors: secondNeighbors,
        weights: secondWeights,
        status: {
          count: secondCount,
          requiredCount: secondRequiredCount,
          overflow: secondOverflow
        }
      }
    }
  ],
  asymmetricSlots // optional, square matrices only
};
const transpose = new GPUChunkedSpatialWeightsTranspose({weights: input, columnCount, output});
graph.add(transpose);
// Feed transpose.outputWeights directly to a consumer accepting GPUChunkedSpatialWeights.
```

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

`asymmetricSlots` counts mismatched positions in the union of the patterns of the input and its
transpose, using `symmetryTolerance` (default zero). It is computed from the input and remains
exact even when output chunks overflow. A structurally absent reverse slot counts as a mismatch
even when the present weight is zero. The input must satisfy the usual CSR invariants, including
sorted unique neighbor IDs in each row.

The described `outputWeights` retains the output layout, spatial context and matrix-level input
status. Output chunks retain their own capacity status. Output views must not share buffers with
matrix-level or chunk-level input status views. Row, variance and double transform metadata becomes
`unknown`: chunk capacity may truncate the result and break global normalization. Chunked output
symmetry metadata is also `unknown` because capacity can truncate it.

The packed output API remains supported: pass `offsets`, `neighbors`, `weights`, optional
`distances` and optional `asymmetricSlots` directly in `output`. Its capacity must still be at least
the sum of input capacities.

## Limits and work

No chunked output path packs the whole CSR or allocates scratch for all output rows. Every input
sort and every output scan fits its own storage binding; device-dependent binding checks include
alignment prefixes and run before scratch allocation or node creation. A rejected binding preflight
leaves the graph available for retry with corrected views. The `outputWeights` description and its
chunk wrappers stay stable, and their CSR view properties and status ports follow same-shaped
corrections made to the caller's output before node generation, including descriptions saved before
a failed preflight.
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

View shapes, chunk ranges, `columnCount` and `symmetryTolerance` are compile-time, matching the
packed transpose. CSR buffer contents may change between encodings. There is no CPU operation twin;
the GPU specs compare against an independent column-first CPU transpose.
