# Raster phase measurements

This opt-in benchmark records RS02–RS04 and RS06–RS11 at two sizes, including both
RS06 mean and Savitzky–Golay operations. RS01/RS05 phase records remain in
[`geospatial-coverage-tranche-one-2026-10-23.json`](../../../../../dev-docs/research/evidence/geospatial-coverage-tranche-one-2026-10-23.json).
The [recorded native run](../../../../../dev-docs/research/evidence/raster-coverage-2026-10-24.json) retains all 20 workload records, per-phase samples, GPU timestamps and source hashes.
That earlier artifact explicitly did **not** measure categorical or spectral-index throughput;
the new harness includes RS02 categorical statistics and RS09 EVI to close those gaps.

Run alone on the GPU, from the repository root:

```sh
nvm use
VITE_BENCHMARK_REVISION="$(git rev-parse HEAD)" \
VITE_BENCHMARK_SOURCE_STATE=describe-retained-source-state \
yarn test-benchmarks modules/analysis/test/gpu-raster/remote-sensing/raster-performance-benchmark.spec.ts
```

The shared `run-gpu.py` lock wrapper was used during the recorded multi-agent run. The benchmark
launcher selects the benchmark inventory and enables its workload flags automatically; ordinary
correctness runs exclude these files. Keep every `RASTER_PERFORMANCE_METADATA` and
`RASTER_PERFORMANCE_MEASUREMENT` console record and the final passing result. A failed or
interrupted run supplies partial observations only. Source hashes in the retained JSON identify
benchmark and raster implementation files; the revision plus dirty-source description prevents
mistaking the benchmark additions for code already present at that revision.

| Case | Small | Larger | Update and correctness reference |
| --- | --- | --- | --- |
| RS02 categories | 16² cells | 128² cells | Four initial classes become three; direct CPU coverage/class sums, stable winners and areas |
| RS03 Cartesian remap | 16² → 8² | 128² → 64² | Double values; independent CPU 2×2 area means, covered area and status |
| RS04 GLCM | 16² cells | 128² cells | Invert checkerboard; CPU directional pair histograms and seven analytic metrics, radius 2/two levels |
| RS06 mean | 16 × 9 dates | 512 × 31 dates | Add 0.25; direct CPU means, counts, provenance and statuses |
| RS06 SG | 16 × 9 dates | 512 × 31 dates | Add 0.25; independent double-precision local polynomial solves, seven-sample windows/order 2 |
| RS07 periodic registration | 16² cells | 64² cells | Change exact periodic translation; known shift, status and normalized-peak bounds; upsample 4 |
| RS08 region graph | 16² / 4 regions | 128² / 32 regions | Double stripe-region means; CPU boundary counts and fixed-linkage mappings. Threshold 1.5 separates initial distance 1 from updated distance 2 |
| RS09 EVI | 16² cells | 128² cells | Increase NIR by 0.1; direct calibrated formula and validity for three resident bands |
| RS10 simplex unmixing | 16 pixels / 8 bands | 256 pixels / 32 bands | Change known three-member abundances; analytic orthogonal-endmember solution, residual/status and iteration bounds |
| RS11 harmonic break | 4 × 24 dates | 32 × 64 dates | Positive jump becomes negative; independent CPU regression/BIC search validates coefficients, components, break date, improvement and status |

Preparation includes deterministic input construction, buffer allocation and graph declarations.
Upload measures explicit writes plus queue completion, separately from synchronous graph compilation.
The harness records first execution, one extra warmup, five resident repetitions, one changed-input
upload and three updated repetitions. Each execution includes encode, submit and queue completion;
readback and assertions are outside that phase. The initial, repeated and updated complete numerical
outputs are validated. Registration peak and unmixing iteration counts use explicit diagnostic bounds
because their exact values are not the independent reference's outputs. Integer statuses, IDs, counts
and provenance compare exactly; float outputs use `3e-4 * max(1, abs(reference))` tolerance.

When timestamp queries are supported, one separate profiled initial-state execution and one profiled
updated-state execution record the sum of GPU node durations. These enable per-node passes and query
capture, so they are **not** the same scheduling path as the normal resident samples. Timestamp mapping
is outside the profiled wall-time sample. Unsupported timestamps remain null. Normal execution samples
remain unprofiled even when timestamp queries are available.

CPU reference durations are measured for all three validation phases. Direct categorical, remapping,
texture, temporal, EVI and seasonal implementations are relevant simple CPU baselines; they are not
optimized native-library comparisons. Registration and unmixing use known analytic solutions, so their
reference durations are only validation cost and must not be presented as CPU algorithm timings.
No GPU speedup ratio or latency requirement is asserted.

Memory records separate uploaded/input, update and output bytes from compiled graph logical and actual
physical transient allocation bytes. Zero graph scratch for a single-dispatch algorithm is legitimate:
shader private arrays and driver/pipeline memory are not graph allocations. Driver peak memory is
unavailable. Profiling query/snapshot allocations are also outside graph transient accounting.
Device metadata records the supplied **max-feature** test-device limits; this is not a portable-default
limit benchmark. Browser timer quantization may produce zero durations for short phases.
Pipeline caches are not cleared, and first execution can include deferred driver compilation.

These bounded, structured workloads demonstrate measured phases and resident updates for every
remaining raster row. They do not establish worst-case scaling, all option combinations, noisy-scene
registration accuracy, ill-conditioned unmixing throughput, full BFAST, or cross-device performance.
Correctness against upstream fixtures and larger boundary tests remains separate evidence in the
[remote-sensing contract](../../../../../docs/api/gpu-raster/remote-sensing.md).
