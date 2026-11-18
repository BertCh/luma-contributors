# Network and movement measurements

One native Chromium WebGPU run on Apple Metal3. Times below are median wall-clock milliseconds for three warm encode/submit/queue-completion runs. Upload/preparation, compilation, cold execution and readback are recorded separately in the [JSON fixture](../../../modules/analysis/test/oracles/fixtures/network-movement-benchmark.json).

| Operation | Workload size | Warm ms |
| --- | ---: | ---: |
| flow | 8 | 0.80 |
| flow | 32 | 0.70 |
| flow | 128 | 1.30 |
| louvain | 4 | 0.60 |
| louvain | 16 | 5.00 |
| louvain | 32 | 28.90 |
| time-routing | 8 | 0.30 |
| time-routing | 64 | 0.70 |
| time-routing | 256 | 0.40 |
| point-routing | 8 | 0.30 |
| point-routing | 64 | 0.60 |
| point-routing | 128 | 0.70 |
| facility | 4 | 0.70 |
| facility | 8 | 0.70 |
| facility | 16 | 3.20 |
| raam | 4 | 1.20 |
| raam | 16 | 0.80 |
| raam | 64 | 2.50 |
| kalman | 32 | 0.60 |
| kalman | 512 | 0.30 |
| kalman | 4096 | 1.00 |
| restriction | 32 | 0.70 |
| restriction | 512 | 0.70 |
| restriction | 4096 | 0.70 |
| similarity | 64 | 0.80 |
| similarity (95% pruned) | 64 | 0.80 |
| similarity | 1024 | 2.00 |
| similarity (95% pruned) | 1024 | 0.80 |
| similarity | 4096 | 5.20 |
| similarity (95% pruned) | 4096 | 1.10 |

Run `yarn test-benchmarks modules/analysis/test/gpu-network/network-optimization/network-movement-benchmark.spec.ts` after selecting the repository Node version. The launcher enables benchmark flags and forwards the selected file.

These fixtures use chain flow/routes, disconnected dense community blocks, p-dispersion over a distance grid, balanced accessibility, and 16-observation tracks. They establish a measured bounded envelope, not worst-case complexity, end-to-end production scale or CPU acceleration. Louvain objective reevaluation and exhaustive facility enumeration remain expensive. The similarity comparison changes survivor selectivity; it does not compare equivalent workloads to a CPU baseline.

The source records uploaded/output bytes; transient graph allocation is not included in those counts. Similarity dynamic-program allocation is bounded explicitly by `survivorCapacity`. Device/driver breadth, longer single-track latency and large sparse graph solvers remain separate performance work.
