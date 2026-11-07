# Geospatial contributor performance: decision brief

October 25 follow-up: [resumed integration](resumed-integration-2026-10-25.md) promotes explicit
hybrid selection while preserving automatic defaults. The historical measurements and failed
broad performance gates below remain unchanged; scheduling and allocator candidates are still
research overlays. The following text records the October 24 research checkpoint.

Research checkpoint: October 24, 2026. This brief consolidates the survey and its follow-up studies.
The current instruction is continued research and planning. Existing retained improvements remain
in place; new shared SpMV and residual proposals remain research overlays.

Luma performs well when it prunes spatial candidates, distributes reductions across GPU workers,
or reuses expensive preparation. The major weaknesses are irregular work concentrated in a few
rows, serial dataset-wide loops, bounded reference algorithms, and setup or scheduling costs that
dominate small requests. There is no single GPU strategy that wins across these workloads.

## What is working well

These are different workloads and measurement boundaries, not comparable leaderboard entries.
Measurements come from the existing local Apple adapter studies; cross-device generalization is
not established. Each linked report retains its controls and qualifications.

| Contributor / workload | Evidence | Practical conclusion |
| --- | --- | --- |
| Indexed natural-neighbor interpolation | 0.4 ms for 128 queries against 65,536 sites, with triangulation and seeds already supplied | Good resident query path; account for index and seed preparation for one-shot use |
| Regular geometry and movement | S2 cell measures: 0.4 ms for 16,384 cells; Kalman: 1.0 ms for 4,096 observations in short tracks | Regular independent work maps well; complex covering and long tracks need separate evidence |
| Variogram fitting | Final cooperative refinement: 0.7 ms for 256 bins versus fresh 8.0 ms candidate-parallel baselines | Cooperative reduction removed a major remaining bottleneck |
| SUR | 1.00–1.25 ms for 16,384 rows with stronger accuracy checks passing | Tiled sufficient statistics are effective; the historical 64.30 ms serial result failed accuracy and is not an unqualified speedup baseline |
| Conditional simulation | 128 observations, 128 queries, 128 replicates: 8.1 ms; explicit preparation reuse reduces 100 batches from 816.70 to 92.55 ms | Reuse pays when geometry is stable; the sequence excludes per-batch readback and retention of earlier outputs |
| Point-cloud neighborhoods | Million-point lattice: 2.2 to 1.6 ms after grid-key improvement; dense coincident case: 428.15 to 6.3 ms with explicit cooperative queries | Index construction and query distribution are separate decisions; sparse controls can favor serial traversal |
| Sparse resistance preparation | Component changes: 160.30 to 75.80 ms; later matched Jacobi comparison: 75.15 to 58.00 ms | Removing serial preparation helped the complete workflow; solver and certificate work remain |

Sources: [original indexed/regular measurements](README.md#strong-measured-paths),
[variogram](variogram-refinement-integration-2026-10-24.md), [SUR](sur-integration-2026-10-24.md),
[simulation](simulation-integration-2026-10-24.md), [reuse](simulation-reuse-integration-2026-10-24.md),
[grid](pointcloud-grid-integration-2026-10-24.md),
[dense neighborhoods](neighborhood-cooperative-integration-2026-10-24.md),
[components](resistance-components-integration-2026-10-24.md),
and [Jacobi](jacobi-integration-2026-10-24.md).

## What is still slow, or lacks scaling evidence

| Priority | Area | Finding | Research decision |
| --- | --- | --- | --- |
| 1 | Sparse resistance and PCG | Shared hybrid draft reduces the large route workflow from 209.30 to 43.50 ms at core limits, and 58.30 to 43.50 ms with maximum features. Only 37/78 primary comparisons pass the broad performance gate; small/short-row workflows regress | Keep selection explicit. Use the completed strategy and iteration studies to prioritize scheduling and selector research; validate full consumers and real networks before proposing a default |
| 2 | Louvain | Historical 28.9 ms at the current 32-vertex supported maximum; exact sparse move/coarsening checks exist, but a validated GPU replacement does not | Resolve f32 ties, parallel move conflicts, convergence and quality before scaling sparse implementation |
| 3 | Complex S2 and watershed | Simple level-6 octant covering improves to 17.4–17.7 ms; watershed 24 by 24 improves to 2.5–2.6 ms | Expand input complexity and complete consumers; preserve covering budgets, label order and queue ties |
| 4 | Expensive geometry / terrain | Historical Delaunay 517 ms at 100K sites, robust intersection 651 ms, dissolve 181 ms at 100K vertices, HDBSCAN 125 ms at 100K points, depression filling 126 ms at 2048 by 2048 | Capture fresh full workflows before redesign. Existing measurements favor preparation/reuse over frequent rebuilds |
| 5 | First use and small requests | Historical ICP first execution 470.2 ms versus 27.6 ms warm; small solver graphs pay extra dispatch costs | Measure construction, compilation, upload, submission, completion and readback separately; include all requested outputs in application comparisons |

Sources: [shared resistance performance](shared-spmv-performance-2026-10-24.md),
[Louvain design](sparse-louvain-design-2026-10-24.md), [S2](s2-integration-2026-10-24.md),
[watershed](watershed-integration-2026-10-24.md), and
[historical expensive paths](README.md#improved-algorithms-with-material-latency).

## Best practices supported by the research

The latest [strategy comparison](shared-spmv-strategy-performance-2026-10-24.md) completes 97,920
timed requests across three sessions, with all 2,176 output validations and all 12 identical-source
controls passing. Hybrid consistently delivers at least 15% and 0.2 ms gain against the best
existing alternative on 26 of 56 eligible workload/configuration combinations. On the large
one-hub PCG solve, workgroup to hybrid improves 15.60 to 10.60 ms at core limits; subgroup to
hybrid improves 14.80 to 10.70 ms at maximum features. Small solves still regress.

The comparison also exposes a selector issue: exact metadata sends 20 maximum-feature cohorts
from subgroup to long-row, and all 20 become slower beyond the allowance in every session.
Long-row is outside the competitive band in all 180 primary cohort observations. Improving the
existing selection heuristic deserves separate evaluation; more metadata is not automatically
a faster application, particularly when obtaining it also has a cost.

1. **Reduce work before tuning kernels.** Spatial pruning, bounded candidate lists and reusable
   preparation have direct consumer value. Treat approximate/local methods as separate semantics
   when they change the result. See the [algorithm practices](algorithm-practices.md) and
   [primary-source register](sources.md).
2. **Measure skew and competent alternatives.** Row count and mean degree hide a few expensive
   hubs. Compare scalar, subgroup, workgroup and load-balanced traversal, including known row
   metadata and its acquisition cost. Merge-based CSR research supports balancing entries and row
   boundaries; it does not predict a WebGPU crossover.
   [Merrill and Garland](https://mgarland.org/papers/2016/spmv/)
3. **Gate performance on independent correctness.** The local forced-scalar PCG study has a true
   maximum residual of 0.0003662109375 despite a tiny recursive squared residual, on a matrix with
   condition number 129/127. Increasing the iteration budget or tightening the recursive threshold
   did not repair that result. Preserve physical acceptance checks and retain failed configurations.
   [Numerical study](shared-spmv-strategy-numerics-2026-10-24.md),
   [Netlib stopping-criteria guidance](https://www.netlib.org/templates/templates.pdf)
4. **Separate kernel throughput from application latency.** Warm resident execution omits costs
   that matter to one-shot users. Report setup, first execution, updates, transfers and complete
   consumers. Record timer resolution and profiling flags; GPU timestamps and wall completion
   measure different boundaries.
   [Chrome measurement guidance](https://developer.chrome.com/docs/web-platform/webgpu/developer-features)
5. **Preserve memory and completion contracts.** Keep source chunk boundaries and borrowed-buffer
   ownership; packing is explicit. Count logical scratch, graph physical allocation and total
   resident state separately. Fewer commands can extend a small buffer's lifetime and increase
   physical capacity; evaluate scheduling and allocation policy independently. Resource reuse
   requires nonoverlapping lifetimes and sufficient capacity, while backend synchronization rules
   still apply. The [AMD allocation guidance](https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/resource_aliasing.html)
   motivates these checks; its native D3D12 aliasing mechanism is not a WebGPU API. Exhaustion,
   overflow and convergence flags are part of the result. [Benchmark contract](benchmark-plan.md)
6. **Use representative controls before changing defaults.** Interleave candidates, qualify with
   identical-source controls, retain adverse results and compare multiple sessions. Add another
   backend and real data before portability claims. Near-tied observations support a competitive
   set rather than a unique winner. [Measurement plan](benchmark-plan.md)

## Next research deliverables

The [iteration trace study](pcg-iteration-attribution-2026-10-24.md) now validates 563 traces:
all strategies use the same two or three active updates per mixed-row shape. Ordering-related
latency differences therefore need another explanation. The new [budget-cost study](pcg-budget-cost-2026-10-24.md)
measures 14,400 requests with unchanged useful work and outputs: an already-converged core solve
rises from **4.80 to 34.40 ms** between budgets 8 and 64; the large two-update hybrid solve rises
from **11.20 to 152.45 ms**. All 12 primary and 12 secondary A/A controls pass. Both host submission
and subsequent completion wait grow. These measurements prioritize scheduling research; they do
not establish a safe lower production budget or predict the gain from predicate reuse.

The scheduling candidate reuses predicates only within phases where the active state and dispatch
geometry are unchanged. It retains the configured budget; phase, accuracy and ownership checks
precede timing. The [shared plan](shared-spmv-integration-plan-2026-10-24.md#proposed-scheduling-experiment-reuse-predicates-within-stable-phases)
sets out that candidate and its acceptance gates. The [concrete command model](pcg-phase-cost-2026-10-24.md)
now identifies 27.6–40.3% fewer solver commands across 133 supported budget-8 configurations.
The large hybrid solve changes from 695 to 456 commands in the model. This is a structural
opportunity. The [first isolated prototype](pcg-phase-reuse-validation-2026-10-24.md) now preserves
exact outputs and diagnostics across 1,689 executions, but increases physical transient memory
from 4.51 MB to 6.20 MB on five fragmented configurations. A 12-byte shared predicate occupies a
large reusable gather allocation while another gather needs it. Twenty-two allocation maps are
reproduced independently. The subsequent [allocator validation](pcg-allocation-policy-validation-2026-10-24.md)
now executes predicate isolation and a four-times capacity limit: both preserve exact prior results
across 3,378 executions and remove all five regressions. The fragmented scalar case reaches
4.51 MB with isolation and 3.01 MB with the capacity limit, versus 6.20 MB for phase reuse alone.
All 133 combined candidates use less capacity than the original baseline; relative to phase reuse
alone, 70 cases add at most 32 bytes with isolation or 24 bytes with the capacity limit. The main
capacity gain on the large case comes from allocation policy itself. Generic validation is now
complete within the scope below; combined primary boundaries and targeted retained-scalar checks
now pass, while broader semantic/full-consumer coverage remains required before timing. No new runtime is adopted.

The [boundary follow-up](pcg-phase-boundary-validation-2026-10-24.md) now validates 5,328 graph
invocations under the original allocator: one-node completed/queued submissions, budgets 64/128,
paired solvers and borrowed-buffer survival. All 2,052 decoded traces and exact output comparisons
pass. The higher-budget path takes 16 active updates, zero on warm replay, and has a maximum true
residual of 1.67e-6. A metadata-only audit also identifies missing scalar-slot and gate-geometry
fields in older graph snapshots; the new host-only capture below now supplies them for the fixed
matrix. This narrows correctness work without establishing latency or runtime adoption.

The [generic allocator study](generic-allocator-validation-2026-10-24.md) now passes 180 tests
and 360 readback replays across three policies and synchronous/asynchronous, whole/stepped controls.
It covers retained values, mixed storage/indirect use, nonzero condition offsets, usage unions and
ownership. It also exposes costs: isolation changes one fixture from 24 bytes/one buffer to
48 bytes/two; the four-times limit changes another from 1,024 bytes/one buffer to 1,276 bytes/two.
No generic fixture saves capacity. These are graph allocation counters, not driver residency or
timing, and four is an experimental threshold. Keep the large PCG benefits and these adverse cases
in the same decision. The two failed harness attempts remain archived with their corrections.

The [untraced semantic capture](pcg-semantic-phase-audit-2026-10-24.md) now passes 90 tests,
399 metadata graphs and 399 matching controls across all three policies. Scalar-slot identity,
active-write invalidation, actual gate geometry and declared lifetimes pass without diagnostic
GPU reads. Prior counters are unchanged. It performs no solves; construction metadata cannot
prove arbitrary shader writes or replace numerical/latency checks.

| Next gate | Current evidence | Required follow-up |
| --- | --- | --- |
| Semantic phase metadata | Fixed budget-eight matrix passes under all three policies; 399 control descriptions match | Extend to combined boundary/observer scenarios; preserve distinction from actual shader-write observations |
| Combined scheduling/allocation | All nine primary captures and the three isolated retained-scalar captures pass | Preserve the stopped attempt; broader scratch/predicate observers remain outside scope |
| Numerical validity | Exact parity passes; two known scalar accuracy failures remain | Independent original-system and full-consumer acceptance; preserve exclusions |
| Performance/default choice | Prior scheduling cost and memory opportunity measured; new candidate latency unknown | Matched setup, warm and full-consumer timing; representative networks and another backend |

The [combined boundary study](pcg-policy-boundary-validation-2026-10-24.md) now passes all nine
primary captures: **96 tests, 432 supported configurations, 15,984 graph invocations and 6,156
decoded traces**. Exact prior/fresh-control outputs and schedules match under both policies.
Predicate isolation leaves primary physical bytes unchanged; the capacity limit changes only
paired high-budget cases, with savings up to 88 bytes and increases up to 32 bytes. The original
allocator's retained-scalar extension also passes. A concurrent reporting-script addition causes
the exact source-inventory gate to reject the predicate observer after its browser tests pass;
the capacity-limit observer was not run in that attempt. The separate
[isolated follow-up](pcg-retention-isolated-validation-2026-10-24.md) now passes all three fresh
retained-scalar captures: 18 tests, 972 observations and 1,944 decoded traces, with exact prior
and fresh-control parity. Predicate isolation adds four bytes in 42 of 81 distinct retained
configuration/arm comparisons; the capacity limit saves up to 88 bytes and adds up to 32 bytes.
The old stopped attempt remains preserved. Arbitrary scratch/predicate retention remains outside scope.

The [next latency design](../../perf/pcg-policy-latency-design-2026-10-24/README.md) now bounds the
experiment to twelve case/budget cohorts: six phase/allocation combinations across six balanced
blocks, with 18 fresh browser processes and 31,680 planned timed requests. It includes three full
resistance consumers, policy-local identical-source controls and a shared process-drift control.
Targeted retention now passes; fresh numerical checks of all 72 selected configurations and a
trace-free consumer adapter are still required before timing. This is a design, not an executed performance result; it cannot justify
a global allocator default.

The [full-consumer adapter specification](../../perf/pcg-policy-latency-design-2026-10-24/full-consumer-adapter-plan.md)
now identifies the exact recipe, scalar residual helper, output checks, replay/reset behavior and
borrowed-buffer ownership needed for those consumers. A CPU source-identity check verifies that
removing the marked diagnostic blocks preserves the proposed scheduling logic. No adapter has
been implemented and no trace-free GPU result is implied by that identity check.

The numerically eligible strategy comparison is complete. The next solver decisions require
full-consumer iteration attribution, a costed metadata/preparation model and selector evaluation. The
[real-network plan](real-network-validation-plan-2026-10-24.md) specifies acquisition provenance,
graph transformations, independent reference solves and a convergence-budget ladder. Dataset
catalog entries alone are not executed real-data evidence.

Then broaden the already measured improvements to a second backend and complete consumers;
continue the sparse Louvain arithmetic/scheduling design and complex S2/watershed sweeps. Runtime
adoption requires its own final build and full test gate. Research-overlay checks do not certify
an applied implementation. The [current assessment](current-assessment-and-research-plan.md),
[integration status](integration-status.md) and [detailed survey](README.md) retain the full history.
