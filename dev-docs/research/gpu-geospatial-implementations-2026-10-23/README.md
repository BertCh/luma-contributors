# GPU implementation research for geospatial contributors

Research date: October 23, 2026. Six GPT-6.1 Sol subagents surveyed the current library and
[78-item geospatial backlog](../../roadmaps/geospatial-coverage-backlog.md), then investigated
primary implementation sources. This report prioritizes selected opportunities; it is not an
exhaustive implementation match for all 78 items.

The baseline is the working tree on commit `3dd6bdb150e4bad8480cd14c1d2d53188b6f5e26`, including
uncommitted geospatial work. The commit alone does not reproduce that state. Implementation presence
is not a new correctness certification. External implementations were inspected, not installed or
benchmarked. Suggested priorities, designs and acceptance experiments are our engineering judgments.

## Recommendation

Invest first in shared execution contracts and a few new domains with concrete GPU precedents:
live-count worklists, batched numerical kernels, 3D neighborhoods, and raster registration/texture.
Much of the conventional GIS kernel coverage already exists. The most useful external material
often supplies a scheduling, data-layout or numerical pattern rather than a complete replacement
contributor.

Keep two work streams distinct: strengthen the scalability and contracts of existing contributors,
and prototype missing backlog capabilities. A new implementation should earn its place through
same-input comparison with the current path, including compilation and data movement.

## Research map

| Report | Scope | Most useful references |
| --- | --- | --- |
| [Shared GPU patterns](./shared-patterns.md) | Live cardinality, scans/sorts, sparse work, small solves and reproducibility | GPUPrefixSums, FidelityFX Parallel Sort, ModernGPU, MAGMA |
| [Geometry, point clouds and coordinates](./geometry.md) | GE, CR, PC; selected IN/UR dependencies | cuSpatial, PCL GPU, Cupoch, gCDT, S2, PROJ, GeographicLib |
| [Raster and remote sensing](./raster.md) | RS; existing segmentation scaling | cuCIM, CuPy, HaraliCU, BFAST GPU implementations |
| [Terrain and hydrology](./terrain.md) | TE and resistance-connectivity dependencies | FastFlow, MFD GPU research, Barnes tiled algorithms, depression hierarchies, xDEM |
| [Statistics and interpolation](./statistics.md) | IN, ST, PP and batched numerical dependencies | GWmodel CUDA, CuPy RBF, ParallelVecchiaGP, RAFT, Random123 |
| [Networks, movement and urban recipes](./network-movement.md) | NW, MV, UR | Gunrock, ECL-MaxFlow, cuGraph, cuOpt, CUDA-DTW, parallel Kalman scans |

Each report identifies local integration points, separates GPU code from papers and CPU semantic
references, and describes WebGPU adaptations and correctness gates. Licenses are reported only
where inspected; linked branches are discovery references, not implementation pins.

## Current-code reconciliation

Historical roadmap proposals must be checked against source before becoming new tasks.

| Already present | Consequence for the research |
| --- | --- |
| [Stable radix sort](../../../modules/graph/src/gpu-core/gpu-sort.ts), [chunk-aware scan](../../../modules/graph/src/gpu-core/gpu-scan.ts), optional subgroup paths | Study live counts, gating and workload crossover; do not add another basic scan/sort API. |
| [Load-balanced iteration](../../../modules/graph/src/utils/load-balanced-for-each.ts), [fixed-point accumulation](../../../modules/graph/src/utils/fixed-point-accumulator.ts), exact lattice predicates | Extend tested contracts and reuse the shared implementations. |
| [Graph-native FFT2D](../../../modules/graph/src/gpu-core/gpu-fft2d.ts) | RS07 phase correlation can reuse this implementation; a foundational FFT migration is unnecessary. |
| Snap-rounded noding, Delaunay, constrained triangulation, polygon repair and prepared joins | GE work adds particular geometry policies or scaling behavior; broad geometry functionality is not absent. |
| D8 tree contraction, branching flow, spill-graph filling and stream extraction | TE02/TE06 extend weighted consumers and cross-tile semantics rather than inventing flow routing. |
| Gaussian GWR bandwidth search and condition diagnostics; generalized GWR and panel subsets | ST06 must identify the estimator and missing diagnostic precisely. Generalized search/inference and full MGWR uncertainty remain separate work. |
| DTW/LCSS bands, Fréchet decision bounds, map-matching tables, sparse facility costs and fractional coverage | MV03, NW and UR work should compose or extend these capabilities. |

See domain reports for exact files and limits. In particular, exact integer predicates still have
declared precision domains; the presence of exact arithmetic does not establish unlimited robust
geometry semantics.

## Prioritized experiments

These are research-to-implementation candidates, not additions to the available API catalogue.
The order weighs reuse and tractability; it does not replace the backlog's product priorities.

| Order | Candidate and backlog consumers | Experiment and decision gate |
| --- | --- | --- |
| 1 | Live-count and gated scan/sort/compaction; GE02, PC02, NW02/NW04, MV03 | Keep fixed capacity but carry GPU count and overflow explicitly. Sweep sparse through dense activity, count changes and chunk boundaries. Measure avoided work and graph/encode overhead against current primitives. |
| 2 | Shared batched small solves and diagnostics; PC01, IN02/IN05, ST01/ST06, RS05/RS10, MV01 | Prototype centered covariance and selected matrix factorizations with per-system validity, rank and residuals. Select algorithms by matrix structure; do not apply Cholesky to indefinite kriging systems. |
| 3 | Resident variogram fitting and robust 2SLS covariance; IN01/ST01 | Extend the existing fitter and estimator rather than replace them. Compare complete fit objectives/covariance matrices against pinned independent references; measure repeated fits and sparse HAC work. |
| 4 | True 3D neighborhoods and point features; PC01 then PC02 | Use PCL/Cupoch as GPU dataflow references. Benchmark indexing plus features on uneven density and translated coordinates. Return support/truncation and orientation policy. |
| 5 | Phase correlation and bounded local texture; RS07 and RS04 | Compose existing FFTs for registration; prototype quantized local histograms for texture. Validate masks, shift sign/subpixel refinement, halo behavior and levels-squared memory. |
| 6 | Fractional distribution/remote-sensing recipes; RS01/RS02/RS06, UR04 | Extend tuple grouping and weighted reductions. Test conservation and global weighted quantiles across tiles; never merge quantiles by averaging them. |
| 7 | Weighted flow and explicit boundary summaries; TE02/TE06 | Compare weighted D8 and branching paths before adding global tile orchestration. Test long chains, split/merge flow and identical whole-domain/tiled results without duplicated runoff. |
| 8 | Sparse graph frontier applications; NW02/NW04 and existing centrality | Start with integral-capacity push/relabel or Louvain as distinct prototypes. Require flow/cut certificates or objective/partition invariants; measure frontier skew and command-graph size. |
| 9 | Geometry policy closure and subdivision; GE03/GE02 | Compose current exact noding/cleanup/repair with explicit collapse policies, or subdivision with provenance. Test reconstruction/idempotence and candidate reduction on repeated queries. |
| 10 | Trajectory batches; MV01/MV03 | Compare per-track filtering with associative-scan smoothing; add admissible search bounds around current distance kernels. Require covariance checks and zero false pruning. |

Do not automatically parallelize away semantic ordering. The current
[marker-controlled watershed](../../../modules/graph/src/gpu-raster/gpu-raster-watershed-segmentation.ts)
is explicitly a bounded, single-invocation quadratic reference path with FIFO discovery ties.
A parallel watershed with different plateau labels would need a separate method contract.
Likewise, [network centrality](../../../modules/graph/src/gpu-network/network-centrality/gpu-network-centrality.ts)
still declares repeated sigma/delta rounds: gating execution alone does not eliminate the cost of
building and compiling those nodes. Reuse the existing loop infrastructure where its ordering
contract fits, and measure graph construction, compilation and encoding as well as GPU duration.

## Portability and source-selection rules

The [WGSL memory model](https://www.w3.org/TR/WGSL/#memory-model) specifies relaxed atomic operations
and workgroup-scoped synchronization. Treat CUDA warp assumptions, global producer/consumer
protocols, wide arithmetic and floating atomic reductions as redesign points. Keep dispatch
boundaries for global phases unless a particular protocol has a valid WebGPU correctness argument.

GPUPrefixSums is useful executable WGSL evidence, but its packed publication state is not a
drop-in replacement for luma's full-width scan arithmetic. The shared report documents the
30-bit payload issue. Preserve full-width arithmetic, chunk carry behavior and deterministic
ordering before considering a faster path.

[cuSpatial](https://github.com/rapidsai/cuspatial) was archived on July 28, 2025. It remains valuable
historical GPU implementation evidence, not a maintained dependency recommendation. CPU references
such as S2, PROJ, PySAL and GeographicLib establish useful semantics without proving an efficient
GPU implementation. Some relevant GPU sources have GPL/LGPL or noncommercial terms, or no verified
license; retain them as clearly labeled algorithm references rather than treating all public code
as copyable MIT source.

Keep primitive GPU execution in `@luma.gl/gpgpu`, shared analytical contracts below domain
contributors, and Arrow conversion in `@luma.gl/arrow`. Preserve caller-owned outputs, graph-owned
scratch, source chunk boundaries and explicit packing. Reuse the existing
[status vocabulary](../../../modules/graph/src/analysis-core/contracts/status.ts) for
cardinality, overflow, convergence, invalidity and approximation evidence.

## Evidence required before promotion

For each selected prototype, pin the upstream revision and exact oracle options, then retain:

1. Independent numerical fixtures, degeneracy and forced-budget tests, including repeated graph
   execution with changed inputs. Constructor and shader-validation tests are complementary evidence.
2. Identical workload comparisons against current luma paths and an appropriate optimized CPU
   reference. Separate upload, declaration/compile, first execution, warm execution and readback.
3. Device/browser/limit metadata, source-state identity, memory/candidate counts, timing distributions
   and correctness status. Test baseline storage limits and optional-feature fallbacks explicitly.
4. Whole-domain versus tiled/chunked equality or a documented numerical tolerance. Preserve source
   provenance and distinguish halos from owned contributions.

The existing [performance harness](../../../modules/graph/test/geospatial-expansion/backlog-performance-README.md)
is a useful recording model. Its [October 23 samples](../../perf/geospatial-backlog-2026-10-23.json)
are bounded smoke observations on raised device limits with concurrent CPU work. They do not
establish production envelopes or optimized CPU/GPU speedups. In particular, the watershed scaling
observation motivates an experiment; it is not evidence that a proposed replacement is faster.

## Research boundaries

No compelling portable GPU implementation was established for every backlog area. Robust vector
straight skeletons, full spherical overlay, complete time-dependent routing, general co-kriging
and full statistical estimator parity remain scoped research projects. Existing CUDA projects can
inform components without resolving these complete contracts.

## Verification of this research change

This change adds this index and six domain reports; it makes no runtime implementation changes.
The domain notes describe their individual read-only research passes.

- `nvm use`: passed with Node 22.22.1.
- `yarn lint fix`: passed; 7,150 files checked, no fixes applied. The repository configuration
  does not format `dev-docs` Markdown, so this is not a Markdown-formatting certification.
- All relative Markdown file links in the research pack resolve locally.
- Dependency installation, runtime builds/tests and website builds were not run for this
  documentation-only research. No upstream GPU implementation was executed or benchmarked.
