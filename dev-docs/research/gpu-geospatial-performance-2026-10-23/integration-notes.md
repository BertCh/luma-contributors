# Analysis integration notes

Condensed findings from the analysis-specific integration reports in the luma.gl `map-contributors` branch (`dev-docs/research/gpu-geospatial-performance-2026-10-23/`, 2026-10-24). Each entry names the original file; the full reports (tables, raw sample counts, test logs, `../../perf/...` artifact links) remain in luma.gl. All timings are warm medians from native Apple Metal WebGPU sessions with blocked before/after ordering and unfrozen driver caches: they are not cross-device, cold-start or tail-latency claims.

The PCG, shared-SpMV, allocator, Jacobi and residual reports are luma.gl graph solver infrastructure and were not migrated.

## s2-integration-2026-10-24.md (S2 polygon covering)
- Cell vertices, edge normals and normalized normals are computed once per visited cell instead of inside repeated containment tests; polygon-vertex containment moved out of the four-edge loop.
- Level-6 octant workload: 68.8-69.0 ms to 17.4-17.7 ms (3.9x); level 4: 14.4 to 4.4 ms (3.3x).
- Depth-first traversal and ordered, truncated output are unchanged on purpose: sorting a different traversal would change the budget-limited prefix. A parallel frontier is not implemented.
- Parity fixture (93 configurations) was captured from the original shader before the change. Complex rings and deeper covers still need profiling.

## sur-integration-2026-10-24.md (ordinary seemingly-unrelated regression)
- Replaced the single-invocation path with nine graph nodes: 256-row tile reductions for means, OLS statistics, residual covariance and GLS statistics, plus small Cholesky completion nodes.
- 16,384 rows: 1.00-1.25 ms across three sessions, versus a 64.3 ms serial diagnostic baseline that also failed accuracy; 1,024 rows: about 1 ms versus 4.25 ms. No small-input (64-row) win is established.
- Numerics: compensated tile merging, centered GLS responses, per-row division kept at accumulation, and joint error covariance normalized by its largest diagonal (no ridge, no changed divisor, no widened tolerance).
- Independent whitened-QR f64 reference agrees with the benchmark reference at 1e-8; device gate is 1e-4. First-use latency of the staged kernels is higher, so warm gains are not a first-use win.

## watershed-integration-2026-10-24.md (raster watershed)
- Indexed binary heap with inverse slots replaces a full scan per extraction: O(N log N) work, O(N) scratch, region limit still 16,384 cells.
- 24 by 24 warm median: 35.5-35.8 ms to 2.5-2.6 ms (14x); 8 by 8 only 1.45x. Scratch grows by 8N bytes (4,608 bytes at 24 by 24); main kernel uses seven bindings instead of eight.
- Seed order, neighbor order, tie-breaking (maximum path elevation then FIFO age) and exhaustion semantics are unchanged; labels match independent references including at the 128 by 128 maximum.
- Not a tiled large-image segmentation method.

## variogram-refinement-integration-2026-10-24.md (variogram model fit)
- One 64-lane workgroup per model stripes bins across lanes; lane zero keeps the strict-tie ladder selection, then all lanes run an ordered golden-section search. Cressie reweighting keeps four ordered rounds.
- 256-bin, three-model, pair-weighted fit: 8.0 ms to 0.7 ms (11.4x over the candidate-parallel version); changed-input GPU execution beats a 2.3-2.5 ms CPU round trip. Small fits vary; no universal CPU/GPU crossover is claimed.
- Search domain, constraints, weighting and status meanings unchanged; correctness is judged against tolerances and references, not serial floating bit equality (reduction order changes parameter bits).
- Reduction results are made uniform before barrier-bearing control flow; no cross-workgroup spin wait or subgroup feature.

## simulation-integration-2026-10-24.md (conditional Gaussian simulation)
- Joint factorization moved to one cooperative 64-lane workgroup (pivot per lane, rows distributed), giving four nodes: validation/compaction, factorization, conditional moments, realizations.
- 128 observations + 128 queries + 128 replicates: about 149 ms to 8.1 ms (18.4x); 32+32 about 4x. Memory unchanged at 267,272 bytes maximum.
- Factor sharing across replicates already existed and is not part of this gain. All lanes reach every barrier including failed inputs; no inter-workgroup synchronization.

## simulation-reuse-integration-2026-10-24.md (prepared simulation reuse)
- Opt-in preparation reuse retains geometry-dependent factors and conditional moments across draw batches.
- 128+128, 100 batches of 128 replicates: 816.7 ms forced preparation versus 92.55 ms retained (8.8x); updating observations every batch 2.7x; geometry updates rebuild and gain nothing; single-batch reuse shows no benefit.
- Preparation counts are verified on the GPU in all 24 workloads. Ratios compare forced versus retained preparation only, not against CPU or the earlier serial factorization.

## resistance-components-integration-2026-10-24.md (resistance connectivity components)
- Serial component discovery is kept when vertices and edges are both at most 256; larger inputs use bounded atomic parallel unions with verified completion and an exact serial fallback (exhaustion triggers the fallback, never reports incomplete connectivity). Component IDs stay minimum vertex IDs.
- Main workload 160.3 ms to 75.8 ms. The 256/256 cutoff is a bounded small-work policy, not a proven optimum for every device.
- COO construction waits for component completion so sort scratch can be reused; physical transient growth was cut from 2,097,168 bytes to a 16-byte control payload.
- Earlier smoke runs had invalid timestamp query sets; the repaired harness checks them. Next targets (Jacobi setup, true residual, physical certificate) stay separate experiments.

## pointcloud-grid-integration-2026-10-24.md and pointcloud-index-research-2026-10-24.md (point cloud grid index)
- Research: the grid builder sorted 32-bit keys with eight radix passes; sorting only `ceil(log2(cells + 1))` significant bits preserves the invalid-point sentinel ordering (using `ceil(log2(cells))` is unsafe at powers of two) and cuts passes to 1-5.
- Integration, accepted by a preregistered threshold (the larger of 5% or 0.2 ms, same direction in every session): 1M-point lattice neighborhoods 2.20 to 1.60 ms, SMRF ground classification 4.50 to 3.90 ms.
- The coincident-query workload (429 ms) is dominated by candidate traversal and is not improved; top-k insertion guarding remains a separate experiment. Results are synthetic and single-device.

## sparse-louvain-design-2026-10-24.md (design only)
- Sparse replacement must keep the directed/undirected modularity objective; move delta and quotient construction were checked in exact arithmetic (quotient conserves total weight and strengths; lifted partition scores equal).
- Graph coloring alone cannot make parallel moves safe (recorded six-vertex counterexample); conservative acceptance uses batches whose source and destination communities are pairwise disjoint.
- In f32, recompute the actual objective rather than trusting summed gains, handle harmful batches explicitly, and keep budget exhaustion distinct from no improving move. Does not make Louvain into Leiden. GPU implementation was pending.

## portable-resistance-research-2026-10-24.md (resistance planning checkpoint)
- Records that residual traversal helps at core maximum features but its frozen gate failed (one of 15 control pairs still above the bound: 4.65 ms versus 4.62 ms limit) and does not authorize adoption; certificate reduction is a separate numerical problem. Focus is SpMV inside PCG (luma solver infrastructure, so the underlying reports were not migrated).
