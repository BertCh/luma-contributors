# Geospatial performance research decisions

The [semantic phase capture](pcg-semantic-phase-audit-2026-10-24.md) now passes 90 tests:
399 metadata graphs match 399 controls under all three allocator policies, with unchanged
prior counters. Actual scalar slots, gate dispatch tuples, phase invalidation and declared
resource lifetimes pass the construction audit. This is no-solve structural evidence;
combined solver/policy boundaries, retained solver observers and latency remain next.

The [generic allocator validation](generic-allocator-validation-2026-10-24.md) now passes
180 tests and 360 GPU readback replays across the original allocator and both candidates,
synchronous/asynchronous compilation, and whole/one-node submissions. The targeted retention,
condition-offset, lifetime, usage and ownership checks pass. Small adverse cases remain explicit:
predicate isolation adds 24 bytes and one buffer; the capacity limit adds up to 252 bytes and
one buffer. No generic fixture saves capacity. This supports continued research alongside the
larger PCG wins. Semantic metadata is now captured separately; combined solver/policy boundaries
and latency remain pending.

The [submission-boundary validation](pcg-phase-boundary-validation-2026-10-24.md) now passes
5,328 graph invocations across 144 supported configurations using the original allocator.
Whole-graph and one-node submissions preserve exact outputs; budgets 64/128 exercise 16 active
updates and inactive warm replays; paired solvers and all 1,296 compiled-graph ownership checks
pass. The independent audit decodes 2,052 traces. Allocator-policy boundary coverage, broader
semantic phase coverage, full consumers and latency remain separate gates. No new runtime
is adopted.

The [allocation-policy validation](pcg-allocation-policy-validation-2026-10-24.md) now removes
all five phase-reuse capacity regressions in isolated research captures. Both policies preserve
prior outputs/traces across 3,378 executions and use fewer transient bytes than the original
baseline in all 133 supported matrix cases. On the fragmented scalar example, predicate isolation
reaches 4.51 MB and a four-times capacity limit reaches 3.01 MB, versus 6.20 MB for phase reuse
alone. Both add small amounts of capacity on 70 cases relative to phase reuse alone (at most
32/24 bytes). Keep the capacity-limit policy as the stronger memory research candidate and
isolation as a narrower comparison. The bounded generic corpus is now complete; setup/latency
and combined solver/policy submission-boundary validation remain. No new runtime is adopted.

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

Reviewed October 24, 2026. This planning record now incorporates the subsequent preparation-reuse
captures. Cooperative variogram refinement is measured and validated. Prepared simulation reuse
has measured repeated-batch gains and a passing final full suite. Dense neighborhoods and
resistance now have [phase attribution](irregular-workload-attribution-2026-10-24.md); sparse Louvain
needs GPU validation of its move policy before scaling.

The [survey](README.md) covers all contributor families, [algorithm practices](algorithm-practices.md)
map primary references to implementations, and [remaining experiment specifications](remaining-experiment-specifications.md)
define the next workloads, dependencies, memory budgets and adoption decisions. The
[integration record](integration-status.md) tracks existing code and verification separately.
No new runtime change is part of this planning update.
The subsequent [cooperative neighborhood integration](neighborhood-cooperative-integration-2026-10-24.md)
now supplies three comparisons matching the retained runtime: 428.15 to 6.3 ms for the dense million-point query
fixture and 77.55 to 0.7 ms for its separate dense planar feature consumer. The serial default is
preserved; default-guard experiments failed controls. Resistance component discovery now has an
accepted hybrid comparison and passing full repository gate. After the subsequent Jacobi improvement, its remaining targets are
true residual, certificate reductions and core-limit PCG SpMV. The earlier
[evidence audit](research-decisions-and-evidence-audit-2026-10-24.md) retains the failed original
profiling attempts; the [component report](resistance-components-integration-2026-10-24.md) records
the repaired captures. The revised Jacobi performance comparison is accepted with its documented follow-up qualification; its final full-suite verification now passes.
The [detailed irregular-workload plan](irregular-optimization-plan-2026-10-24.md) now compares
guard-only, cooperative and tiled neighborhood selection, and examines the adjacency and
degree-skew costs of reusing existing connected components for resistance. The follow-up
[resistance experiment plan](resistance-research-plan-2026-10-24.md) adds completion invariants,
verified fallback, numerical reduction constraints and explicit memory estimates. Further work
in this checkpoint is research and planning.

## Performance assessment

| Area | Assessment from retained evidence | Research decision |
| --- | --- | --- |
| Spatially ordered joins and indexed interpolation | Strong resident queries; natural neighbors took 0.4 ms for 128 queries against 65,536 sites, excluding triangulation and seed location | Preserve the architecture; measure preparation amortization and updates |
| Point-cloud features, SMRF and ICP | Million-point synthetic workloads took 19.1, 12.8 and 27.6 ms respectively; ICP used four iterations | Retain favorable paths; add clustered distributions and time-to-convergence measurements |
| Regular cell calculations and short-track filtering | S2 measures took 0.4 ms for 16,384 cells; Kalman took 1.0 ms for 4,096 observations in short tracks | No replacement justified; test long tracks separately |
| Variogram fitting | Three final cooperative-refinement sessions take 0.7 ms at 256 bins versus fresh 8.0 ms candidate-parallel baselines; changed-input GPU is 0.7 ms versus 2.3–2.5 ms CPU round trips | Study first-use cost, small fits and consumer batching; broaden devices and weighting modes |
| Ordinary SUR | 16,384-row final sessions take 1.00–1.25 ms and pass stronger accuracy checks; the 64.30 ms serial diagnostic baseline fails accuracy | Broaden conditioning and device coverage; no unqualified speedup ratio against the failed baseline |
| Conditional simulation | Cooperative execution takes 8.1 ms at 128+128 sites and 128 replicates; later reuse reduces 100 draw batches from 816.70 to 92.55 ms and improves value updates 2.66 times | Study output handling, pipelined throughput, broader devices and real consumers |
| S2 polygon covering | Fresh level-6 octant sessions improve from 68.8–69.0 to 17.4–17.7 ms after classification reuse | Profile validation, classification and traversal on complex rings before frontier changes |
| Marker watershed | 24-by-24 sessions improve from 35.5–35.8 to 2.5–2.6 ms with an indexed heap | Study larger supported shapes and consumer cost; preserve FIFO and budget semantics |
| Louvain | Historical 28.9 ms at its supported maximum of 32 vertices | Treat as a bounded reference; develop sparse gains and coarsening with an explicit move policy |
| Dense neighborhoods | Three paired sessions: serial 428.15 ms versus cooperative 6.3 ms for 1M coincident sources, 64 queries, k=4; dense planar feature consumer 77.55 versus 0.7 ms | Keep the strategy explicit: the 100K-point lattice control regresses from 0.7 to 1.2 ms; broader device/data coverage remains open |
| Sparse resistance | Hybrid components improve 160.30 to 75.80 ms; the subsequent matched Jacobi series improves 75.15 to 58.00 ms | Research residual traversal and certificate arithmetic separately before changing the preconditioner |

Historical and current captures use different timing boundaries and sampling. They are priorities,
not a common leaderboard. Historical evidence is linked from the [survey](README.md); current results
are in the [variogram](../../perf/variogram-integration-2026-10-24.json),
[SUR](../../perf/sur-integration-2026-10-24.json),
[simulation](../../perf/simulation-integration-2026-10-24.json),
[watershed](watershed-integration-2026-10-24.md) and [S2](s2-integration-2026-10-24.md) records.
Current integration captures are from one Apple Metal environment, generally with blocked
before/after ordering. They do not establish other devices' latency or a cold-driver result.

## Variogram refinement now has measured attribution

The [final refinement report](variogram-refinement-integration-2026-10-24.md) supersedes the
single exploratory capture. Three final sessions report 0.7 ms warm at 256 bins, versus 8.0 ms
in all three fresh candidate-parallel baselines. Changed-input GPU medians are 0.7 ms versus
2.5, 2.5 and 2.3 ms for CPU readback, fit and reupload. Small changed-input fits have no consistent
winner. The full repository checkpoint passes against the final source snapshot.

| Diagnostic stage at 256 bins | Before median | After median |
| --- | --- | --- |
| Prepare | 0.059498 ms | 0.055604 ms |
| Candidates | 0.258743 ms | 0.258417 ms |
| Refinement | 7.079776 ms | 0.121042 ms |

This isolates refinement as the successful target. Diagnostic per-node timestamp passes can
change scheduling; these medians must not be summed into an application latency. Final graph
scratch remains 9,664 bytes; the cooperative kernel uses 2,080 workgroup bytes and seven storage
bindings. Bin reductions change floating accumulation order, while golden-section steps remain
ordered. WGSL workgroup synchronization does not synchronize separate workgroups.
[WGSL specification](https://www.w3.org/TR/2026/CRD-WGSL-20261021/#sync-builtin-functions)

First-use and sample drift remain material: the first final 256-bin execution takes 136.2 ms,
then 8.6 and 9.1 ms in later sessions; caches were not cleared. The first session's first-ten warm
median is 1.85 ms versus 0.7 ms for its last ten. Preserve these observations rather than deriving
a cold-start regression from uncontrolled cache state. Next research should test complete consumers,
non-pair weighting timings and batching near the submission floor, plus another backend.

## Prepared simulation needs explicit state and invalidation

The [current implementation](../../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts)
still prepares and factors once per encoding by default, sharing the factor across replicates.
The opt-in retained path now has three final repeated-batch captures; the
[integration report](simulation-reuse-integration-2026-10-24.md) records its gains and passing full
suite. The conditional mean and covariance equations
support separating geometry-dependent factors from observation-dependent means and random draws.
The ownership and invalidation policies below are luma engineering decisions.
[Gaussian Processes for Machine Learning, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf)

| Change | Reusable state if valid | Required refresh |
| --- | --- | --- |
| Seed or replicate offset/count | Factor and conditional moments | Draw parameters, capacity, output initialization and status |
| Finite observation values or known mean | Factor when inclusion and error variances are unchanged | Validation, observation solve and conditional means |
| Error variance or observation validity flag | No general full-factor reuse | Compaction, covariance and factorization |
| Observation/query positions or covariance parameters | No general full-factor reuse | Geometry-dependent state and factorization |
| Output buffers | Prepared state only within compatible graph/device ownership | Bindings, capacity and output initialization |

An included nonfinite observation currently fails validation; it is not silently omitted. The
validity flag controls omission. Prepared reuse must preserve that distinction and reject invalid
updates before drawing from stale state. A finite value update can also overflow after mean
subtraction, which has a different failure status. Recovery must recreate valid state explicitly.

The default path specializes seed, mean, offset and output shape in the graph. The existing reuse
prototype instead exposes explicit geometry/value invalidation and mutable mean/draw controls,
with constructor replicate count as capacity. It retains factors and moments in owned imported
buffers; borrowed inputs remain borrowed. Covariance and view-shape changes still require a new
graph specialization. Buffer identity alone cannot certify unchanged contents.

Retained factors cannot use ordinary graph-transient storage: transient allocations may be reused
after their compiled last reader. The prototype uses GPU-visible generations and validity so an
encoded-but-discarded rebuild cannot make later commands trust an uninitialized factor. Control
setters enqueue buffer writes, so preceding encodings must be submitted before their parameters
change. This lifetime and submission contract belongs in any adoption decision.

The retained validation logs record 59 focused device tests, seven Node tests and zero strict-type
diagnostics. They include changed/recovered inputs, discarded encoding, ownership and full
conditional covariance. The dedicated reuse full suite now certifies that checkpoint. Its
[experiment specification](remaining-experiment-specifications.md#prepared-simulation-evaluation)
requires sequence amortization, default-path checks and final repository gates. The measured
100-batch large draw sequence now improves 8.82 times and the value-update sequence 2.66 times;
geometry updates show no gain. Final lint/build/types and all nine final captures pass, while the
new full suite passes with 12,486 Node and 12,187 headless tests and unchanged captured sources. These timings exclude per-batch readback and retained outputs.

For evaluation, define `B` as the cost per batch with the current path, `P` as one-time preparation
and `D` as the prepared draw/update cost. The proposed path wins on latency only when
`P + R * D < R * B`, or `R > P / (B - D)` when `B > D`. These are planning variables, not measured
phase times: the existing 8.1 ms total cannot be treated as factor cost. Include setup, validation,
parameter uploads, any recompilation and destruction consistently in both workflows.

Measure sequences of 1, 2, 10 and 100 batches for fixed geometry, value updates and full geometry
updates. Preserve full joint covariance, semidefinite query behavior, invalid/recovered state and
Philox replicate identity. A marginal-only simulator or neighborhood approximation is a separately
named method. Retained-factor memory must be reported alongside reduced latency.

## S2 and watershed have completed their first bounded experiments

The [S2 integration](s2-integration-2026-10-24.md) caches cell geometry and removes redundant
polygon-vertex containment tests. The [watershed integration](watershed-integration-2026-10-24.md)
replaces repeated full scans with an indexed `(pathCost, entryAge)` heap. Both preserve their
ordered semantics and have three before/after sessions; repeating their original proposals would
ignore work already present.

For S2, vary hierarchy depth, ring/vertex count, holes, poles, complements and output/visit budgets
independently. Separate input validation from per-cell classification and traversal. Complex-ring
validation can have quadratic edge comparisons even when only a few cells are visited. A frontier
prototype should follow attribution, with either exact preservation of the depth-first truncated
prefix or an explicitly different API contract. Sorting final cells cannot restore an unexplored
prefix. Google's coverer informs pruning, not luma's ordered-budget oracle.
[S2RegionCoverer source](https://raw.githubusercontent.com/google/s2geometry/master/src/s2/s2region_coverer.cc)

For watershed, measure the current supported range through 128 by 128, distinguishing correctness
coverage from timed performance. Preserve four/eight connectivity, row-major discovery, FIFO ties,
reserved-marker barriers, masks, calibrated elevations and completion-versus-exhaustion behavior.
The heap adds 8N scratch bytes; include this in consumer memory, including canonicalization.
A parallel plateau or minimum-spanning-forest method requires its own label contract. Related
parallel Priority-Flood results concern depression filling and do not establish marker-label parity.
[Parallel Priority-Flood](https://arxiv.org/abs/1606.06204)

## Dense neighborhoods and resistance now have measured targets

The [neighborhood kernel](../../../modules/graph/src/gpu-pointcloud/gpu-pointcloud-neighborhoods.ts)
counts every radius match and retains the nearest 1–64 IDs with stable original-ID ties.
Coincident points defeat spatial pruning. Filling top-k cannot terminate the full support count.

The [grid-index study](pointcloud-index-research-2026-10-24.md) identified a separate shared
candidate: restrict radix sorting to `ceil(log2(cells + 1))` bits while preserving the full invalid
sentinel and stable cell ordering. Its subsequent [integration](pointcloud-grid-integration-2026-10-24.md)
passes focused device checks and three comparisons in each direction, improving million-point
lattice queries from 2.2 to 1.6 ms and SMRF from 4.5 to 3.9 ms. Dense coincident query latency
remained essentially unchanged at that checkpoint. The later
[cooperative query study](neighborhood-cooperative-integration-2026-10-24.md) measures a large
traversal/selection gain on dense inputs. Its original serial default is preserved after guard
experiments failed controls. The table below now defines further distribution and tiling research,
not an unimplemented first cooperative experiment.

| Experiment | Workloads and measurements | Decision rule |
| --- | --- | --- |
| Candidate distribution | Uniform, clustered, coincident and elongated clouds; vary radius, queries and k independently; record candidate and accepted counts | Choose a dense path from candidate work, not total point count alone |
| Selection cost | Profile construction, traversal/counting, top-k maintenance and output writes separately | Consider tiled local top-k only if selection or serial candidate work dominates |
| Tiled selection design | Bound local-list bytes and merge traffic under `(distance, originalId)` order | Require exact counts, IDs, ties, inclusive radius and self-exclusion; retain the sparse path when tiling overhead loses |

For the total distance/ID order, an item outside its tile's top-k cannot enter the global top-k:
at least k items in that tile precede it. This justifies merging tile top-k lists for selection,
but every tile must still contribute exact support counts. It does not establish a device speedup.

The [resistance recipe](../../../modules/graph/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance.ts)
used single-invocation component discovery in the retained baseline before sort/CSR construction,
Jacobi-PCG, explicit true residual evaluation and current extraction. The accepted hybrid
component path now uses bounded unions and verified fallback for larger graphs, retaining the
original small path. Final residual/status aggregation remains serial.
The historical 696-node count alone did not establish the dominant phase. The earlier
[diagnostic attribution](irregular-workload-attribution-2026-10-24.md) motivated component work.
The subsequent [accepted component checkpoint](resistance-components-integration-2026-10-24.md)
puts component discovery at 0.1794 ms, Jacobi at 17.9182 ms, true residual at 18.8413 ms and
certificate plus capture at 12.0073 ms. These instrumented medians are not additive application
latency. The subsequent revised Jacobi comparison reduces its diagnostic stage to 0.9508 ms and fixes the
empty-entry failure. True residual remains 19.0023 ms and certificate plus capture 12.0471 ms.
The [current plan](current-assessment-and-research-plan.md) and
[numerical memo](residual-certificate-research-2026-10-24.md) define the remaining decisions.

Profile component discovery, sort/CSR, initialization, SpMV, reductions, convergence checks,
current extraction and host encoding. Test lattice, chain, hub and disconnected graphs, strong
conductance contrasts, and repeated right-hand sides. Record `norm(Ax-b)/norm(b)` alongside the existing absolute maximum-residual and reconstructed
sink-balance certificate; do not replace the current acceptance rule. Use an absolute criterion
for zero right-hand sides, and preserve current conservation and failure status. The
[resistance specification](remaining-experiment-specifications.md#resistance-workflows) describes
the exact current certificate and the distinction between configured and executed iterations. Changing grounding
changes the operator; positive-to-zero conductance updates can change connected components.

Reuse topology/preconditioners only under explicit dependencies. Consider stronger preconditioning
when iteration savings repay setup and retained memory; consider graph/dispatch changes when host
cost dominates. Circuitscape's solver guidance supports this setup/solve distinction and explicit
true-residual acceptance. Its CPU solver results do not select a WebGPU preconditioner for luma.
[Circuitscape solver guidance](https://docs.circuitscape.org/Circuitscape.jl/stable/compute/)

## Sparse Louvain is an algorithm design task

The [existing derivation](research-update-2026-10-24.md#louvain-sparse-work-without-changing-the-objective-silently)
starts from luma's directed/self-loop conventions and modularity objective. First verify sparse
move deltas against exhaustive objective differences on small graphs, then verify coarsening and
weight conservation. Only then evaluate a conflict policy and parallel moves. Individually
positive gains need not produce a positive combined move; recompute the actual objective after
accepted batches. Keep convergence, budget exhaustion and objective traces explicit.

Sparse adjacency removes dense reevaluation but does not make runtime independent of degree skew,
rounds or levels. Test chains, grids, stars, heavy hubs, disconnected components, self-loops and
planted communities. Compare objective, partition invariants, determinism and elapsed time; do not
require identical labels from distinct heuristic schedules unless that is the advertised contract.

Louvain can produce poorly connected communities. Leiden introduces a refinement phase to address
that problem; it is a separate algorithm choice, not a drop-in speed optimization. Add connectivity
quality to the research comparison if community structure matters to a consumer.
[Original Leiden paper](https://arxiv.org/abs/1810.08473)

## Complete workflows and promotion decisions

The table below retains the broader consumer workstreams. The [current ordered experiments](current-assessment-and-research-plan.md#ordered-research-deliverables)
put the Jacobi correctness gap and remaining resistance stages first.

| Priority | Next research deliverable | Adoption evidence |
| --- | --- | --- |
| 1 | Extend prepared simulation consumer evaluation | Broader devices, output handling and pipelined throughput; measured resident sequences and the repository gate already pass |
| 2 | Variogram consumer and device coverage | First-use latency, small-fit crossover, non-pair timing and batching without accuracy regressions |
| 3 | Resistance completion design and dense-query distribution coverage | Preserve physical certificates and exact support; retain explicit cooperative selection while sparse controls favor serial |
| 4 | S2 geometry-complexity and watershed supported-size sweeps | Preserved ordered/budget semantics and complete consumer benefit |
| 5 | Sparse Louvain objective and scheduling prototype specification | Exact delta/coarsening checks before scaling; objective and connectivity quality after scaling |
| 6 | Shared live-count and graph-overhead experiment | Benefit in at least two complete consumers, including allocation/compaction, without chunk or ownership regressions |

For each consumer record preparation/allocation, upload, host graph compilation, first execution,
warm encoding/submission/completion, updates and readback. Compare resident GPU work with resident
CPU/GPU work at the appropriate boundary; compare one-shot requests using the complete request.
Preparation reuse and query latency should be shown together. Preserve source batch/chunk boundaries;
packing must remain explicit table behavior rather than an incidental optimization.

Use the [benchmark plan](benchmark-plan.md): freeze correctness criteria and practical effect size
before final comparisons, retain five warmups and thirty samples across at least three sessions,
prefer alternating A/B, and retain failed captures. Examine sample-order drift and session variation
rather than pooling samples into an apparently larger independent population. Collect additional
samples when required by instability; do not delete slow samples without a documented cause.

Keep diagnostic timestamps separate from uninstrumented latency, unsupported timestamps distinct
from zero, and declared nodes distinct from executed work. Record adapter limits and browser flags.
Verify at baseline WebGPU limits and another backend before claiming portability; do not equate
an Apple Metal capture with fleet coverage. Independent CPU/reference checks use the same quantized
inputs and unchanged accuracy/failure requirements.

This research update edits documentation and derived evidence only. New runtime work remains outside
this checkpoint's scope. Final implementation acceptance still requires focused parity/replay checks,
`yarn lint fix`, a final full build and full repository tests. The completed refinement checkpoint covers watershed, S2 and cooperative variogram refinement
(12,486 Node and 12,182 headless tests passed). The later retained-simulation checkpoint passes
12,486 Node and 12,187 headless tests against its own final source snapshot.
The subsequent grid checkpoint passes 12,486 Node and 12,198 headless tests with all 7,523
captured sources unchanged. Cooperative neighborhoods are now present in the runtime and have
211 focused device checks and 11 focused Node checks. Their later dedicated full suite passes
12,486 Node and 12,348 headless tests with all 7,525 captured sources unchanged. Prior
checkpoints do not certify later source changes. The component checkpoint subsequently passes
12,486 Node and 12,386 headless tests; the revised Jacobi candidate subsequently passes its own expanded gate: 12,486 Node and 12,414 headless tests.
This research follow-up changes documentation only; verification is tracked in the integration record.
