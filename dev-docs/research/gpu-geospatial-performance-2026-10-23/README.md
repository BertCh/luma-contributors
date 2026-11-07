> **Note (2026-11-07).** Copied from the luma.gl `map-contributors` branch (`dev-docs/research/gpu-geospatial-performance-2026-10-23/`); paths and `../../perf/...` links refer to that branch. Only the analysis-level documents are kept here. The PCG, shared-SpMV, allocator, Jacobi and residual experiment reports (about 80 files) concern luma.gl graph solver infrastructure and stay in luma.gl; links to them in this README are intentionally unresolved. Analysis-specific integration reports are condensed in [integration-notes.md](./integration-notes.md).

# Geospatial contributor performance findings and priorities

The [isolated retained-scalar follow-up](pcg-retention-isolated-validation-2026-10-24.md) now passes
all three allocator captures: 18 tests, 81 supported configurations, 972 observations and 2,997
graph invocations. Exact prior/fresh-control outputs, traces, retained scalar fields and ownership
checks pass. Predicate isolation adds four bytes in 42 of 81 distinct configuration/arm comparisons;
the capacity limit saves up to 88 bytes and adds up to 32 bytes in others. These observer costs
are separate from the [nine accepted primary captures](pcg-policy-boundary-validation-2026-10-24.md).
The earlier provenance-stopped observer attempt remains preserved. The
[next latency design](../../perf/pcg-policy-latency-design-2026-10-24/README.md) and concrete consumer
adapter specification are CPU-verified planning artifacts; all 72 fresh configuration checks,
including full consumers and exact trace-free sources, remain required. No new runtime is adopted
and no candidate speedup has been measured.

The [semantic phase capture](pcg-semantic-phase-audit-2026-10-24.md) now passes 90 tests:
399 metadata graphs match 399 controls under all three allocator policies, with unchanged
prior counters. Actual scalar slots, gate dispatch tuples, phase invalidation and declared
resource lifetimes pass the construction audit. This is no-solve structural evidence;
combined primary boundaries and targeted retained-scalar checks now pass separately; broader
semantic/observer coverage and latency remain open.

The [generic allocator validation](generic-allocator-validation-2026-10-24.md) now passes
180 tests and 360 GPU readback replays across the original allocator and both candidates,
synchronous/asynchronous compilation, and whole/one-node submissions. The targeted retention,
condition-offset, lifetime, usage and ownership checks pass. Small adverse cases remain explicit:
predicate isolation adds 24 bytes and one buffer; the capacity limit adds up to 252 bytes and
one buffer. No generic fixture saves capacity. This supports continued research alongside the
larger PCG wins. Semantic metadata and combined primary boundaries now pass separately;
targeted retained-scalar coverage now passes; full consumers and latency remain pending.

The [submission-boundary validation](pcg-phase-boundary-validation-2026-10-24.md) now passes
5,328 graph invocations across 144 supported configurations using the original allocator.
Whole-graph and one-node submissions preserve exact outputs; budgets 64/128 exercise 16 active
updates and inactive warm replays; paired solvers and all 1,296 compiled-graph ownership checks
pass. The independent audit decodes 2,052 traces. Allocator-policy primary boundary coverage now passes separately; broader
semantic phase coverage, full consumers and latency remain separate gates. No new runtime
is adopted.

The [allocation-policy validation](pcg-allocation-policy-validation-2026-10-24.md) now removes
all five phase-reuse capacity regressions in isolated research captures. Both policies preserve
prior outputs/traces across 3,378 executions and use fewer transient bytes than the original
baseline in all 133 supported matrix cases. On the fragmented scalar example, predicate isolation
reaches 4.51 MB and a four-times capacity limit reaches 3.01 MB, versus 6.20 MB for phase reuse
alone. Both add small amounts of capacity on 70 cases relative to phase reuse alone (at most
32/24 bytes). Keep the capacity-limit policy as the stronger memory research candidate and
isolation as a narrower comparison. The bounded generic corpus and combined primary boundaries
now pass, as does targeted retained-scalar coverage; full-consumer validation and setup/latency remain. No new runtime is adopted.

The [phase-reuse command model](pcg-phase-cost-2026-10-24.md) now makes the scheduling candidate
concrete: it identifies 27.6–40.3% fewer solver commands across 133 supported budget-8 configurations,
including 695→456 commands for the large hybrid solve. These are source-derived counts, not
measured speedups. Active-write boundaries, indirect-buffer lifetimes and prototype gates are
specified; the new runtime candidate remains unapplied.

Latest cost evidence: the [configured-budget study](pcg-budget-cost-2026-10-24.md) completes
14,400 timed requests across three sessions. An already-converged core solve rises from 4.80 ms
at budget 8 to 34.40 ms at budget 64; the large two-update hybrid solve rises from 11.20 to
152.45 ms with identical outputs. All 12 primary and 12 secondary A/A controls pass. This
prioritizes scheduling research while preserving convergence budgets; the recoverable fraction
and full-consumer benefit remain unmeasured. New runtime proposals remain unapplied.

Latest iteration evidence: the [diagnostic trace study](pcg-iteration-attribution-2026-10-24.md)
validates 563 traces and exact paired outputs across 232 supported configurations. Every supported
strategy uses three updates on the four padded mixed-row shapes and two on the other eleven.
Forward/reverse/shuffle counts match, so iteration variation does not explain their timing gaps.
Five or six inactive budget slots remain. The new budget study measures their combined scheduling
and resource effect on selected fixed-work inputs. Both scalar accuracy failures remain unchanged. Full-consumer traces and predicate-reuse research are specified in the
[shared plan](shared-spmv-integration-plan-2026-10-24.md#proposed-scheduling-experiment-reuse-predicates-within-stable-phases).

Start with the [performance decision brief](performance-decision-brief-2026-10-24.md) for
the strongest contributors, remaining slow paths and prioritized research decisions.

Latest strategy timing: the [eligible-strategy comparison](shared-spmv-strategy-performance-2026-10-24.md)
completes 97,920 requests across three sessions, with 2,176 output validations including preflight.
Identical-source controls pass 12/12 primary and 12/12 secondary checks. Hybrid has a useful
gain over the best observed existing strategy in all sessions for 26/56 eligible cohorts in the
primary window and 26/56 in the secondary window. The complete matrix retains existing alternatives,
metadata-aware defaults, resource costs and exclusions. This is resident synthetic evidence on
one Apple adapter; it does not justify a universal default or include metadata/setup cost.
The existing long-row strategy is outside the competitive band in all 180 primary observations.
Exact metadata selects it instead of subgroup on 20 maximum-feature cohorts, and all 20 regress
in every session. Selector quality therefore needs separate evaluation alongside hybrid traversal.

The [numerical survey](shared-spmv-strategy-numerics-2026-10-24.md) preserves two forced-scalar PCG
failures and 34 unsupported requests among 420 configurations. Exact metadata changes 48 of 60
default choices. The forced-scalar residual gap persists in physical row-code and tighter-tolerance
confirmations; those failed configurations receive no timing claim.

The prior [complete resistance comparison](shared-spmv-performance-2026-10-24.md) measures 209.30
to 43.50 ms at core limits and 58.30 to 43.50 ms at maximum features on the large route workflow.
All 78 output comparisons pass, but only 37/78 primary and 39/78 secondary performance comparisons
pass. Its failed broad gate and small/short-row regressions remain. The [shared numerical validation](shared-spmv-validation-2026-10-24.md)
retains no baseline-valid regression, two reviewed status improvements and 162 rejected observations.
New runtime proposals remain unapplied. Full-consumer iteration attribution, metadata/preparation cost,
[real networks](real-network-validation-plan-2026-10-24.md), other backends and full adoption gates remain.

Prior performance findings: [mixed-row traversal evaluation](pcg-mixed-row-selection-2026-10-24.md) completes
186 numerically valid comparisons and 37,200 timed requests. Eight larger-hub configurations improve
complete PCG by at least 15% in all three sessions. All nine primary performance failures are the
255/256/257-entry boundary-row PCG cases; a row above 256 does not guarantee a speedup.
The shuffled seven-hub solve improves 245.90 to 11.20 ms, with ordering and iteration qualifications.

The preceding [short-row controls](pcg-short-row-selection-2026-10-24.md) already rejected an
entry-count-only default. Keep selection explicit while developing shared SpMV. The
[implementation plan](shared-spmv-integration-plan-2026-10-24.md#explicit-selection-contract) now
requires actual forced-strategy validation before exposing a public preference. The focused, domain and complete-workflow studies cover actual forced execution; broader
real-data and portability checks remain; the eligible mixed-row strategy timing is now complete.
Earlier opportunistic fallback alone did not establish hybrid support.

The [original-residual resistance evaluation](pcg-original-residual-evaluation-2026-10-24.md) remains
the complete geospatial-workflow evidence: 194.70 to 43.00 ms (77.9%) and 72 passing primary pairs.
Its 420 numerical observations preserve baseline-valid cases with two reviewed status improvements.
High-contrast precision limits and earlier secondary failures remain; merge residual stays unadopted.

Earlier residual findings: the [portable resistance study](portable-resistance-research-2026-10-24.md)
records the completed control follow-up (14 of 15 pairs pass) and diagnostic core SpMV costs of
150–168 ms. The [72-pair residual evaluation](residual-overlay-evaluation-2026-10-24.md) improves
primary wall time by 32.5% at max features and 8.7% at core limits, but fails its frozen acceptance
gate. Residual and PCG drafts remain unapplied.

Start with the [current assessment and ordered research plan](current-assessment-and-research-plan.md)
for what is performing well, remaining slow paths and the next experiments. The historical survey
and detailed evidence follow below.

The [Jacobi integration](jacobi-integration-2026-10-24.md) fixes the empty-input defect and reduces
the complete resistance primary from 75.15 to 58.00 ms (22.8%). The revised design passes all
39 paired primitive controls; two noisy consumer controls pass a fixed additional follow-up.
119 focused device tests, strict types and the full build pass. The full repository suite passes: 12,486 Node tests and 12,414 headless tests,
with all 7,749 frozen sources unchanged. The [research plan](current-assessment-and-research-plan.md)
reconciles this checkpoint with the remaining priorities.

The preceding [resistance component integration](resistance-components-integration-2026-10-24.md) reduces the
65,538-vertex route workload from 160.30 to 75.80 ms across three session medians. Small graphs
retain the original serial path. One noisy small control required a fixed 100-sample follow-up,
which passes all three pairs with unchanged thresholds. Focused checks and the final full
repository gate pass: 12,486 Node tests and 12,386 headless tests, with all sources unchanged. The report retains the initial failed comparisons.

Research date: October 23, 2026. Audience: maintainers of luma's analytical contributors.

Design and integration follow-up: October 24, 2026. The research plans and subsequent measured
implementations are retained below.

The earlier [evidence audit](research-decisions-and-evidence-audit-2026-10-24.md) records invalid
timestamp diagnostics in the original resistance smoke runs and the incomplete first comparison.
The subsequent [component integration](resistance-components-integration-2026-10-24.md) repairs the
measurement harness and supplies the accepted hybrid comparisons and full repository gate.
Those earlier failures remain part of the evidence history.

The latest [research decisions](next-research-decisions.md) distinguish measured improvements from
remaining slow paths. The new [resistance experiment plan](resistance-research-plan-2026-10-24.md)
specifies connectivity completion and fallback, long-row balancing, physical certificates,
memory budgets and adversarial workloads. This follow-up continues research and planning;
it does not adopt additional runtime changes.

The [cooperative neighborhood integration](neighborhood-cooperative-integration-2026-10-24.md)
reduces the million-point coincident workload with 64 capacity-four queries from 428.15 to
6.3 ms across three session medians. A dense planar neighborhood-to-feature graph improves from
77.55 to 0.7 ms. Selection is explicit because sparse queries can favor the original serial path.
Default serial-guard experiments were rejected after adverse-order regressions. Focused checks,
the final full build and the dedicated full repository suite pass: 12,486 Node tests and
12,348 headless tests, with all captured sources unchanged.

The [dense neighborhood and resistance plan](irregular-optimization-plan-2026-10-24.md) adds
the latest source review, algorithm alternatives, bounded-memory design and proposed acceptance
criteria. Its original serial dense-query baseline remained about 428 ms despite faster grid construction.
Resistance component discovery is the largest measured diagnostic phase. No further runtime
change was adopted in that planning update; the cooperative integration above follows it.

The current [research decisions](next-research-decisions.md) reconcile the original survey with
later measured improvements. The [remaining experiment specifications](remaining-experiment-specifications.md)
provide concrete plans for dense neighborhoods, resistance, sparse Louvain and prepared simulation,
including correctness contracts, memory estimates and a parallel-move counterexample. Integration
to date is tracked in [integration status](integration-status.md). The
[preparation-reuse report](simulation-reuse-integration-2026-10-24.md) records the latest measured
batch gains, default-path checks and passing full-suite gate.

The [point-cloud indexing study](pointcloud-index-research-2026-10-24.md) adds a verified
integer-order argument for narrower radix keys and separates that proposal from dense-query
selection work. The [sparse Louvain design](sparse-louvain-design-2026-10-24.md) records exact
objective, coarsening and conservative parallel-move checks. The subsequent
[grid integration](pointcloud-grid-integration-2026-10-24.md) improves million-point lattice
neighborhoods from 2.2 to 1.6 ms and SMRF from 4.5 to 3.9 ms across median session times.
The [irregular-workload profile](irregular-workload-attribution-2026-10-24.md) identifies the
remaining dense-query and resistance bottlenecks. Sparse Louvain remains pending GPU evaluation.

The [October 24 research update](research-update-2026-10-24.md) retains the earlier SUR numerical
analysis and sparse Louvain derivation. The [cooperative refinement report](variogram-refinement-integration-2026-10-24.md)
records final variogram sessions, profiling and the passing full repository checkpoint.

luma has an effective foundation for GPU geospatial analysis: spatially ordered indexes, sparse
traversal, scans, reductions, graph scheduling and reusable GPU buffers. Its performance is uneven
because some newer domain operations still put dataset-sized loops into one invocation. The first
priority is to remove those serial cores while preserving their numerical and completeness contracts.
The second is to measure complete workflows, including preparation, index construction, transfers,
and first use. A fast resident query alone does not establish a fast application.

The original survey below describes the October 23 baseline. Subsequent retained variogram captures
show the candidate-parallel implementation reducing 256-bin warm execution from a median of
50.7 ms across sessions to 8.0 ms (6.3 times faster). At that checkpoint, CPU round trips were
faster at 2.8–2.9 ms versus 8.4 ms in the changed-input comparator. The later cooperative refinement
has three final 0.7 ms warm medians at 256 bins, versus 8.0 ms in three fresh candidate-parallel
baselines. Its changed-input GPU median is 0.7 ms versus 2.3–2.5 ms CPU round trips. The
[refinement report](variogram-refinement-integration-2026-10-24.md) retains startup observations,
sample drift, small-fit qualifications and verification. The tiled SUR implementation runs the
16,384-row case in 1.00–1.25 ms across three sessions and passes the previously failing accuracy
check. Its serial diagnostic baseline took 64.30 ms but failed that check. See the integration
status for qualifications; historical rows below are not current rerun results. Cooperative
conditional simulation subsequently reduced the 128+128, 128-replicate case from 146.85–149.35 ms
to 8.1 ms across three sessions, with full covariance checks passing.
Opt-in preparation reuse subsequently reduces 100 batches at that shape from 816.70 to 92.55 ms
across median session times, including first preparation and per-batch completion. Value updates
improve 2.66 times; geometry updates show no gain. These sequence timings exclude per-batch readback
and retention of earlier output batches; see the reuse report for the complete measurement boundary.
The indexed watershed queue then reduced the 24-by-24 case from 35.5–35.8 ms to 2.5–2.6 ms,
with exact label/status parity and additional bounded scratch documented in its integration report.
The [S2 classifier integration](s2-integration-2026-10-24.md) reduces the isolated level-6 octant
workload from 68.8–69.0 ms to 17.4–17.7 ms while retaining ordered output and traversal-budget behavior.

This research covers spatial queries and geometry, interpolation and statistics, raster and terrain,
networks, point clouds, movement, and their shared WebGPU infrastructure. The proposed designs are
engineering conclusions from current source and primary references; they are not measured speedups.

- [Algorithm practices](algorithm-practices.md) maps the literature to specific implementations,
  alternatives, semantic constraints and experiments.
- [Benchmark and acceptance plan](benchmark-plan.md) defines comparable workloads, timing, memory,
  correctness and promotion gates.
- [Implementation plan](implementation-plan.md) scopes the first benchmark, variogram and SUR
  changes, with data dependencies, scratch budgets and existing test entry points.
- [Simulation covering and watershed design review](semantic-design-review.md) resolves factor
  synchronization, cover-prefix selection and bounded heap requirements before implementation.
- [Primary sources](sources.md) identifies the evidence and its applicability to WebGPU.
- [Evidence manifest](evidence.json) preserves extracted local measurements and hashes of 28 reviewed
  source/document files and the measurement inputs. It is not an execution-time source snapshot.

The earlier [implementation survey](../gpu-geospatial-implementations-2026-10-23/README.md) covers
additional source implementations and licensing. This pack adds performance decisions and reconciles
new implementations that landed after parts of that survey, including resident variogram fitting,
3D point features and shared-factor conditional simulation.

## Evidence and interpretation

The local measurements are mostly Chromium WebGPU on an Apple Metal adapter with three or five warm
samples. The adapter often exposes approximately 4 GiB storage bindings and ten storage buffers per
stage. These are raised-limit observations, not a portable baseline certification. Exact GPU model
and driver are generally unavailable.

In the recent captures, warm wall time generally includes CPU encode, submit and queue completion;
allocation, upload and readback are separate. Historical reports use other timing boundaries and
must not be ranked as if they came from one run. The original research collected no new GPU
benchmarks; subsequent October 24 captures are recorded separately in the integration status.
The reviewed checkout is based on `e6375c5d683ee8a6289789f07ea8e24e7b87c21e` with substantial
uncommitted work; a commit alone does not reproduce it.

Three kinds of evidence are distinguished below:

1. **Recorded measurements:** values extracted from retained fixtures, with workload qualifications.
2. **Source findings:** observable implementation structure, such as one invocation accumulating all
   rows. This identifies a candidate bottleneck but is not a per-kernel profile.
3. **Proposed experiments:** replacements to test under equal semantics; external CUDA performance
   does not predict their WebGPU speed.

## Strong measured paths

| Operation | Recorded warm wall time | What the result establishes |
| --- | --- | --- |
| Point-cloud neighborhood and feature graph | 19.1 ms at 1M points | Index rebuild plus neighborhoods and features scales well on a unit lattice, capacity 8. Not a dense-cluster guarantee. |
| SMRF ground classification | 12.8 ms at 1M points | Effective processing on the synthetic ground/object scene and 256 by 256 raster. |
| ICP registration | 27.6 ms at 1M points, four iterations | Good repeated execution for a translated lattice and static target grid. Does not establish time to convergence for difficult alignments. |
| Indexed natural-neighbor interpolation | 0.4 ms for 128 queries against 65,536 sites | Fast cavity queries with supplied triangulation and containing-triangle seeds. Both preparation steps are outside this query timing. |
| Kalman movement filtering | 1.0 ms for 4,096 observations | Good throughput over short tracks; the fixture uses 16 observations per track. |
| S2 stable cell measures | 0.4 ms for 16,384 cells | Regular per-cell calculations are much cheaper than adaptive covering in the same capture. |

Sources: [point-cloud features](../../perf/pointcloud-coverage-2026-10-23.json),
[SMRF and ICP](../../perf/pointcloud-advanced-2026-10-23.json),
[interpolation](../../perf/interpolation-advanced-2026-10-23.json),
[movement](../../../docs/api-reference/experimental/gpu-network/network-movement-benchmarks.md),
[S2](../../benchmarks/data/global-coordinates-expanded-2026-10-23.json).

Earlier measurements also support the architecture of spatially ordered joins and cooperative
reductions. The [contributor inventory](../../roadmaps/gpugraph-roadmap.md) records point-in-polygon
over 250K points and 14.4K cells improving from 218 to 6.7 ms, and nearest-feature queries over 100K
points and 50K segments from 309 to 4.1 ms after feature sorting. The
[October 14 implementation record](../../roadmaps/gpu-analysis-algorithms-roadmap.md) reports a
hub-heavy spatial lag improving from 18.1 to 4.6 ms and network statistics from 21.5 to 0.9 ms.
These are historical A/B observations, not newly reproduced results.

## Historical slow paths and research priorities

| Operation | Recorded result | Source finding at the original survey | Original implication |
| --- | --- | --- | --- |
| Variogram model fit | 256 bins: 51.3 ms resident GPU versus 3.2 ms CPU readback, fit and reupload | One invocation per candidate model scans all bins and all range candidates | Parallelize candidate evaluation and bin reductions first. Residency alone does not rescue this implementation. |
| SUR regression | 63.8 ms at 16,384 rows | One fitting invocation accumulates dataset-wide statistics and solves | Separate parallel sufficient-statistic reductions from the small solve. |
| Conditional Gaussian simulation | 146.2 ms for 128 observations, 128 queries and 128 replicates | Joint factorization is shared across replicates, but remains a serial storage-memory Cholesky | Investigate cooperative factorization and explicit factor reuse; do not repeat the already completed sharing change. |
| S2 adaptive polygon covering | 73.3 ms at level 4; 309.1 ms at level 6 for a three-vertex polygon | One invocation traverses a stack and repeatedly classifies cell boundaries | Parallelize independent frontier cells and reuse geometric bounds. |
| Louvain | 28.9 ms at 32 vertices | One invocation repeatedly reevaluates a dense objective; hard cap 32 vertices | This is a bounded reference path, not a scalable sparse community detector. |
| Marker-controlled raster watershed | 36.1 ms at 24 by 24 cells | One invocation scans the pending cells at each extraction, giving quadratic work | Replace queue selection before increasing the supported region size. Preserve FIFO ties. |
| Coincident point neighborhoods | 439.4 ms for 64 queries against 1M coincident points | Dense candidate work defeats the favorable grid distribution | Benchmark exact support and stable-ID selection separately; improve dense-cell work distribution. |
| Sparse resistance recipe | 149.1 ms at 65,538 vertices, 696 nodes | Recipe builds a sparse grounded Laplacian and uses Jacobi-PCG | Attribute preparation, CSR construction, encoding and solve cost before choosing a new preconditioner. |

Sources: [variogram and simulation](../../perf/interpolation-advanced-2026-10-23.json),
[SUR](../../../modules/graph/test/oracles/fixtures/statistics-performance-baseline.json),
[S2](../../benchmarks/data/global-coordinates-expanded-2026-10-23.json),
[Louvain](../../../docs/api-reference/experimental/gpu-network/network-movement-benchmarks.md),
[watershed](../../perf/geospatial-backlog-2026-10-23.json),
[point clouds](../../perf/pointcloud-advanced-2026-10-23.json),
[resistance](../../perf/urban-ecology-coverage-2026-10-23.json).

The variogram comparison uses the same bins, models and warmed device, but the CPU performs 60
refinement iterations while GPU convergence can stop earlier. Its recorded correctness checks
passed; it is not an instruction-for-instruction comparison. The 3.2 ms CPU figure measures the
whole round trip directly, rather than adding independently measured phase medians.

S2 timings include possible shared-device contention. The watershed capture ran with concurrent
CPU build/test work. Repeat them in isolation before setting a latency budget. Their source-level
serial structure remains an independent reason to investigate scaling. The resistance fixture is
hub-heavy with at most eight solver iterations; it is not a general raster-circuit benchmark.

## Improved algorithms with material latency

The October 14 record reports Delaunay at 517 ms for 100K sites, robust intersection at 651 ms and
dissolve at 181 ms for 100K vertices, HDBSCAN at 125 ms for 100K points, and depression filling at
126 ms for a 2048 by 2048 pits fixture. Large-polygon triangulation improved to about 115–118 ms
for 50K vertices, including a former 34-second pathological case.

These improvements matter, but these workloads still belong in preprocessing or scheduled analysis
unless a consumer tolerates that latency. Reuse topology when inputs permit it. A contributor can
be work-efficient yet too expensive to rebuild on every pointer movement.

Two other results resist a simple fast/slow classification:

- **OPTICS:** the [complete later capture](../../perf/point-process-complete-2026-10-23.json) records
  24.6 ms for a full 4,096-point ordering, versus 347 ms for its independent JavaScript reference.
  At 256 points, GPU execution is 4.0 ms versus 3.6 ms for that reference. The API's supported
  frame is bounded at 4,096. The earlier capture processed only 64 frontier steps at 4,096 rows
  and cannot establish a complete-ordering speedup.
- **Projection:** the [table report](../../benchmarks/gpu-project-table-performance.md) shows useful
  UTM acceleration but near ties for some Albers workflows. At about 65K rows, 17 physical batches
  cost 22–26 ms for a UTM round trip versus about 9.5–9.7 ms with two batches. This is a reason to
  profile submission and readback structure, not permission to silently combine source batches.

## Practices to adopt across contributors

**Expose enough useful parallel work.** A single invocation is appropriate for advancing a cursor
or solving a tiny reduced system. It is suspect when it also scans every data row, graph edge,
candidate cell or range parameter. Decompose independent candidates and row reductions before
changing workgroup size. MAGMA's tiny-matrix work illustrates why independent systems and matrix
size need different mappings; it does not make a single 256 by 256 factorization a tiny batched
problem. [MAGMA research](sources.md#magma-factorizations)

**Optimize total work and memory traffic.** Spatial pruning, reuse and sufficient statistics often
remove more work than instruction-level tuning. Avoid a broad replacement of grids with BVHs:
the local October 14 comparison found the BVH slower on every tested neighborhood workload.
Choose an index from distribution, radius, update rate and candidate inflation, then measure
build plus query. [Spatial indexing references](sources.md#spatial-indexing-and-geometry)

**Balance irregular work.** Split long CSR rows, dense grid cells and huge polygons across useful
work units while keeping cheap paths for short rows. Existing luma merge-path traversal and
adaptive SpMV are the foundation. [Merge-based SpMV](https://mgarland.org/papers/2016/spmv/) shows
why average row length alone can conceal large performance variation.

**Carry live counts and explicit status.** A compacted output should let downstream work process its
live prefix rather than full capacity. Preserve required count, stored count, overflow, convergence
and incompleteness separately. Extending the existing sort/scan/compaction contracts is preferable
to bespoke per-contributor equivalents. Do not optimize by truncating work silently.

**Keep repeated work resident and prepared.** Reuse compiled graphs, parameter buffers, static
indexes and factorizations through explicit ownership and invalidation. Do not encode loop counters
or changing parameter values into WGSL source. Measure first use separately: the 1M-point ICP
capture is 470.2 ms on first execution and 27.6 ms warm. Asynchronous pipeline preparation improves
responsiveness; it does not eliminate compilation cost. [WebGPU lifecycle](sources.md#webgpu-and-measurement)

**Preserve numerical and semantic contracts.** Exact topology, tolerance-bounded geometry, local
interpolation, approximate density and full joint simulation are different products. A faster method
must match the existing contract or have an explicit method option. Deterministic ordering does not
by itself guarantee cross-device floating-point identity. [Numerical references](sources.md#numerics-and-reproducibility)

## Proposed implementation sequence

The ordering below weighs measured severity, reuse and semantic risk. It is a proposed work queue,
not a statement that the experiments have succeeded.

| Order | Experiment | Existing foundation | Required decision evidence |
| --- | --- | --- | --- |
| 1 | Isolated rebaseline and phase instrumentation | Retained benchmark harnesses | Matching source hashes, correctness, warm distributions, first-use latency and memory for each target |
| 2 | Parallel variogram candidates and SUR sufficient statistics | Reductions, centered regression helpers | Same fit objectives, boundary/rank statuses and a clear full-workflow improvement |
| 3 | Cooperative conditional factorization and explicit reusable preparation | Existing shared factor and Philox identity | Conditional moments and joint covariance, exact-hit/singular cases, changed-input invalidation |
| 4 | Parallel S2 covering frontier | Scan, compaction, indirect dispatch, S2 predicates | Cover containment, interior safety, ordering, seam/pole cases and explicit budget exhaustion |
| 5 | Watershed queue improvement and separate scalable segmentation study | Existing bounded oracle and terrain spill graph | FIFO plateau parity for replacement; separate method contract if labels differ |
| 6 | Sparse Louvain prototype | CSR, segmented reductions, coarsening | Independent modularity/objective checks, loop/self-edge conventions, convergence and reproducibility policy |
| 7 | Dense point neighborhoods and sparse resistance profiling | 3D grid, adaptive SpMV, Jacobi-PCG | Distribution sweeps, true residual, preprocessing amortization and exact support evidence |
| 8 | Shared live-count and graph-overhead work | Existing primitives and GPU program machinery | Benefits in at least two complete consumers, with zero-live and all-live regression controls |

The common primitives must stay within the [repository package boundaries](../../../AGENTS.md).
Primitive GPU data belongs in `@luma.gl/gpgpu/gpu-data`; graph scheduling, tables and analytical
algorithms belong in their existing `@luma.gl/graph/*` subpaths. Arrow conversion remains an adapter
concern. Packing is explicit; borrowed storage is never destroyed by aggregate owners.

## Decisions that should remain closed unless new measurements disagree

- Keep the grid as the current automatic neighborhood choice; BVH availability is not evidence of
  a win on a particular distribution.
- Keep viewshed traversal selectable. The recorded pyramid proposal lost on a smaller workload;
  more hierarchy is not universally faster.
- Keep binned KDE explicitly approximate and retain its error budget.
- Preserve the robust geometry path instead of restoring the faster legacy path with known
  precision failures.
- Treat the October 6 performance report as theory, not measured speedups. The October 13 review
  is historical; later work implemented scan selection, triangulation, load balancing, iteration
  gating and asynchronous app preparation.
- Do not introduce automatic CPU fallbacks. CPU implementations are performance and correctness
  baselines under the repository's GPU-resident execution policy. A CPU win identifies a GPU
  engineering target rather than authorizing a backend change.
