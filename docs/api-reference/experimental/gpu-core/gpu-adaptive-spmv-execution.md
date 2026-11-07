import {GPUOperationContract} from '@site/src/components/docs/gpu-operation-contract';

# Adaptive CSR SpMV execution

<GPUOperationContract operation="gpu-adaptive-spmv-execution" />

## Overview

The public `GPUGraphSpMV` operation keeps sparse matrix-vector multiplication expressed as one operation. Its internal WebGPU lowering, `GPUAdaptiveSpMV`, selects an execution family appropriate to the matrix shape and device.

```text
CSR matrix + vector
        │
        ▼
strategy selection
   ┌────┼──────────────┬─────────────┬────────────┐
   ▼    ▼              ▼             ▼            ▼
scalar subgroup     workgroup      long-row    hybrid-row
 rows    rows           rows          split     (explicit)
   │      │              │             │            │
   └──────┴──────────────┴─────────────┴────────────┘
                    │
                    ▼
               output vector
```

## Scalar rows

For matrices with very short rows, synchronization and cooperative reduction can cost more than the arithmetic. One invocation therefore owns one complete row. A workgroup processes many independent rows.

## Subgroup rows

Medium rows can map naturally to one subgroup. Lanes walk the row cooperatively and `subgroupAdd` produces the row result without a full-workgroup reduction.

This path is selected only when the device exposes WebGPU subgroup support.

## Workgroup rows

Longer rows use one workgroup per row. Threads accumulate strided nonzeros and combine their partial sums through workgroup memory. This is the portable cooperative baseline.

## Very long rows

A single workgroup can underutilize the GPU when a row contains thousands of nonzeros. The long-row strategy splits each row across several workgroups:

```text
row
 ├── workgroup 0 ── partial 0
 ├── workgroup 1 ── partial 1
 ├── workgroup 2 ── partial 2
 └── workgroup 3 ── partial 3
                       │
                       ▼
                  finalize row
```

The partial buffer is a graph transient and participates in lifetime-based allocation reuse.

## Explicit hybrid rows

Set `strategy: 'hybrid-row'` on `GPUGraphSpMV` to use merge and carry stages for long rows and sequential accumulation for rows of at most 256 entries. `GPUJacobiPCG` forwards the same choice through `spmvStrategy: 'hybrid-row'`. The choice works without subgroup support and is never selected automatically; omitting it preserves the existing strategy policy.

This option requires each of the five CSR and vector views (row offsets, column indices, values, input vector, and output) to have exactly one physical chunk. The sum of output rows and nonzero slots must fit in `uint32`. Unsupported explicit requests throw instead of silently choosing another strategy. Nonzero row-offset and buffer byte offsets are supported, including unused leading and trailing slots. Empty rows and empty flat matrices are supported. As with other parallel reductions, the addition order for long rows can differ from sequential accumulation, so float32 rounding can differ.

The internal `GPUAdaptiveSpMV` lowering exposes `getStrategy(graph).id` and `explain(graph).strategy` for inspection. For hybrid execution, `explain(graph).merge` describes its merge stage; the existing row fields describe the sequential stage. `GPUJacobiPCG.getResult().spmvStrategy` reports the selected solver strategy through the public API.

## No tuning readback

Strategy selection does not read CSR row offsets back to JavaScript. Row count and total nonzero count are already graph construction metadata and provide average row length. Producers may optionally supply row-shape statistics when they already know them.

A future GPU classifier can bucket rows and generate indirect queues entirely on the GPU for mixed matrices.

## Why this matters beyond SpMV

Adaptive SpMV is the first irregular workload to exercise the common GPU strategy layer. Reduction chooses hierarchy shapes; SpMV chooses fundamentally different mappings of work to GPU lanes. If both fit the same strategy contract, that contract is a credible foundation for later FFT, MatMul, sort and graph-operation planning.


## Chunked storage

CSR row offsets, column indices, values, input vector, and output accept independently partitioned graph vectors. Row boundaries, nonzero indices, and column IDs are global. Adjacent offsets can straddle chunks, and one row may consume nonzeros and vector entries from many chunks. The four automatic strategies remain available for chunked storage; explicit `hybrid-row` requires flat storage as described above. Subgroup execution uses one subgroup per row without assuming a fixed subgroup width. Long-row partial scratch is bounded by a row block and the storage binding limit. Each encoding overwrites the previous result before accumulating contributions. `GPUGraphSpMV` uses the same lowering and preserves external `GPUData[]` bindings.

### Fragmentation-aware routing

For `R` row blocks, `N` aligned nonzero spans and `V` nonempty vector chunks, direct routing
requires `R × N × V` row dispatches. When `R × V > R + V`, the lowering first gathers
`vector[columnIndices]` for each nonzero span and reuses those values across row blocks.
Scalar, subgroup and workgroup paths then require `N × (V + R)` dispatches; long-row execution
also finalizes each row block. A single vector chunk or a small row set keeps direct routing.

This is indexed algorithm scratch, not concatenation of the source vector. Source buffers and
chunk boundaries remain unchanged. Gathered spans are sequenced so their storage can be reused;
physical gather scratch is bounded by the largest aligned nonzero span (four bytes per entry).
Long-row partials have their own allocation lifetimes. The gather executes again on every encoding,
including when the operation is GPU-gated, so changes to vector or index contents are observed.
Invalid column indices contribute nothing, even when the corresponding matrix value is non-finite.

The dispatch-count heuristic does not guarantee a speedup for every matrix. It trades an indexed
scratch write/read for fewer kernels and repeated CSR scans. Floating-point addition order may
change. Routing still grows with `N × (V + R)`; it is not constant-cost for arbitrarily fragmented
inputs. No CPU readback is used to select the path.
