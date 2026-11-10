# Proposal for luma.gl: GPU-resident iteration by batched indirect dispatch (PLAT-04)

**Status: a draft kept in this repository. Nothing has been sent upstream: no issue, no pull request, no message to the luma.gl maintainers.** Under [AGENTS.md](../AGENTS.md) the feature belongs in luma.gl, so it is not implemented here.

## Problem

`GPUCommandGraph` compiles a fixed list of passes. Algorithms that run until a condition holds (shortest paths, connected components, label propagation, k-means, centrality, DBSCAN expansion) must either unroll a worst-case number of rounds at compile time or read a convergence flag back after each round.

- Unrolling is the current workaround. `GPUNetworkCentrality` unrolls reachability, sigma and delta rounds (`maximumIterations` 1 to 1,024, default 64) and rejects more than 32,768 command nodes (roadmap NW-CENT, [upstream-workarounds.md](./upstream-workarounds.md)). Exact betweenness above roughly 15,000 to 20,000 nodes needs sampled pivots.
- Readback per round costs latency. WGLog (arXiv 2607.17571, 2026-07-20) measured about 2.5 ms for a 4-byte `mapAsync` readback, roughly 285 times a CUDA 4-byte copy, and 0.18 ms for an empty fence.

## Evidence

WGLog, a WebGPU engine for recursive queries, removes both costs:

1. A small kernel writes the next stage's `dispatchWorkgroupsIndirect` arguments from device-side counts. A converged round dispatches zero workgroups.
2. K rounds are recorded into one command buffer (K = 30, halved adaptively), double buffered, and the host reads the fixpoint flag once per batch.
3. Measured: indirect dispatch alone gave 4.7 times; with batching the total was 15.4 times over a readback-per-iteration port. It ran unchanged in Chrome on an M5 Pro. See [research-gpu-landscape.md](./research/spatiotemporal-review-2026-11-09/research-gpu-landscape.md) section 3.

These are the paper's numbers on its workloads (recursive joins). They are not measured on this library.

## Proposed luma.gl surface (for discussion)

- A graph node whose dispatch size is read from a caller buffer (`dispatchWorkgroupsIndirect`), with the arguments buffer produced by an earlier node in the same graph.
- A `GPUIterationBatch` (name open) that records K copies of a sub-graph with a device-side "active" flag, a convergence word written by the sub-graph, and one readback per batch. The host loop decides the next K.
- Zero-size dispatch after convergence must be a no-op on every backend; the document should state the guarantee the spec gives.

## What this repository would do after it ships

Replace the unrolled rounds in `GPUNetworkCentrality`, then the other iterative operators, and remove the NW-CENT guard and the register row. Until then nothing changes here.

## Open questions

- Whether the existing indirect-dispatch support in published `@luma.gl/graph` already covers the first bullet. Check the alpha before drafting the issue.
- The WebGPU rule against read-write aliasing of one buffer in a pass forces double buffering of frontier state.
- Validation: a test with a known iteration count and a deliberately early convergence.
