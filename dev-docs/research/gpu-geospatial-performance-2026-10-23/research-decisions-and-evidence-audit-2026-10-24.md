# Geospatial performance priorities and evidence audit

October 24, 2026. This follow-up continues research and planning. The highest-value next
experiment is a valid comparison of the resistance connectivity candidate already present in
the checkout. Its smoke results are promising, but neither those captures nor the interrupted
long comparison meet the acceptance plan. No runtime or benchmark implementation is changed
by this follow-up.

The [portfolio survey](README.md) and [algorithm practices](algorithm-practices.md) cover the
contributor families. This update reconciles their latest implementation status, records the
measurement failure, and specifies the decisions needed before further optimization.

## Which paths are doing well and which need attention

These are workload-specific findings, not a common hardware-independent leaderboard. Values
come from the linked retained reports; no new GPU timing was collected for this audit.

| Area | Current evidence | Decision |
| --- | --- | --- |
| Indexed spatial queries | Natural-neighbor queries take 0.4 ms for 128 queries against 65,536 sites, with triangulation and seeds supplied | Preserve pruning; measure preparation plus updates before recommending one-shot use |
| Variogram fitting | Cooperative refinement takes 0.7 ms at 256 bins across three final sessions | Strong warm path; first use and small-fit CPU crossover remain separate questions |
| SUR | 16,384 rows take 1.00–1.25 ms with accuracy checks passing | Strong tested path; the slower serial comparator fails accuracy, so it cannot support an unqualified speedup ratio |
| Conditional simulation | 8.1 ms for 128 observations, 128 queries and 128 replicates; retained preparation reduces 100 draw batches from 816.70 to 92.55 ms | Favor explicit reuse for stable geometry; evaluate output retention/readback and real consumers |
| Point-cloud grid | Million-point lattice neighborhoods improve from 2.2 to 1.6 ms; SMRF from 4.5 to 3.9 ms | Shared construction improvement is measured; distribution remains decisive |
| Dense neighborhoods | 1M coincident points, 64 queries, k=4: 428.15 to 6.3 ms; planar feature consumer: 77.55 to 0.7 ms | Cooperative strategy works well here; retain explicit choice because a sparse control goes from 0.7 to 1.2 ms |
| S2 covering | Level-6 octant improves to 17.4–17.7 ms from 68.8–69.0 ms | Still material latency; profile complex-ring validation before choosing parallel traversal |
| Watershed | 24-by-24 fixture improves to 2.5–2.6 ms from 35.5–35.8 ms | Successful bounded queue improvement; larger supported sizes remain a performance question |
| Resistance | Historical matched attribution: 150.05 ms for 65,538 vertices; latest candidate smoke: 74.2 ms versus 151.8 ms fresh baseline | Highest-priority unresolved comparison; smoke observations are not an accepted improvement |
| Louvain | Historical 28.9 ms at the supported maximum of 32 vertices | Algorithm redesign needed; sparse objective/coarsening checks precede GPU scheduling and scaling |

Sources: [original query evidence](README.md#strong-measured-paths),
[integration status](integration-status.md), [current neighborhood results](neighborhood-cooperative-integration-2026-10-24.md),
[resistance phase attribution](irregular-workload-attribution-2026-10-24.md), and
[resistance capture audit](../../perf/resistance-components-validation-2026-10-24/research-audit.json).
The older Delaunay, geometry overlay, HDBSCAN and depression-filling observations remain material
preprocessing costs; the [survey](README.md#improved-algorithms-with-material-latency) retains their
dates and input sizes. They have not been rerun as part of this update.

## Resistance evidence needs repair before promotion

The checkout now contains bounded edge unions, flattening, completion verification and an exact
serial fallback. The earlier description of an unapplied draft is superseded. The scheduled
candidate has [75 passing focused device tests](../../perf/resistance-components-validation-2026-10-24/device-schedule.log)
and [zero strict-type diagnostics](../../perf/resistance-components-validation-2026-10-24/types-schedule.log).
The completed neighborhood full suite predates it and does not certify it.

The [audit](../../perf/resistance-components-validation-2026-10-24/research-audit.json) parses the
three retained smoke logs and the first long baseline log, records their SHA-256 hashes, and
extracts wall medians, sample counts, allocation sizes and diagnostic capacity. Each smoke has
12 readable workload records and three wall samples per workload. The long baseline has 11
readable records and one malformed record. Its comparison driver never completed the six-run
series. A process exit of zero and a stable source snapshot establish neither valid diagnostics
nor a complete comparison.

The harness requests `2 * (declaredNodes + 1)` timestamp queries. WebGPU validates query-set count
against a fixed maximum of 4,096; this is not a luma tuning limit or a requestable device-limit
increase. Consequently this harness can cover at most 2,047 declared nodes in one such set.
[WebGPU query creation specification](https://www.w3.org/TR/webgpu/#dom-gpudevice-createqueryset),
[reviewed specification source](https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs).

| Workload | Baseline requested queries | Candidate requested queries | Diagnostic status |
| --- | --- | --- | --- |
| 65,538-vertex routes | 1,400 | 1,408 | Fits the query-count bound; individual timing validity still requires checking |
| 65-vertex chain | 8,914 | 8,922 | Exceeds the bound in every retained variant |
| 81-vertex grid | 11,090 | 11,098 | Exceeds the bound in every retained variant |

The chain and grid declare more solver work despite having fewer vertices. Their invalid
query-set warnings appear in all three smoke logs, not just the long run. In the long baseline,
stderr interrupts a large JSON measurement record. The preserved logs remain useful failure
evidence; oversized diagnostics must not be interpreted as zero-cost work. Wall observations
remain exploratory because the runs are not clean acceptance captures.

The scheduled smoke also has adverse controls: the four-vertex route increases from 4.7 to
6.5 ms, and the 1,026-vertex route from 7.1 to 7.7 ms. Three samples cannot distinguish a stable
regression from run-order drift, but these observations prevent accepting the hub gain alone.
Do not select an automatic strategy threshold from these smoke medians.

### Proposed measurement repair

1. Check diagnostic capacity before creating the query set. For over-capacity graphs, initially
   omit per-node GPU profiling and record `null` with reason `query-capacity-exceeded`; keep all
   wall samples and correctness checks. Record unavailable, invalid and valid timings separately.
2. Retain full diagnostics for the route fixture that fits. Aggregate phases within each sample
   before summarizing across samples. Do not add independently calculated phase medians or use
   their sum as complete wall time. Record expected, available and missing timestamps per phase.
3. Write structured measurements separately from human stdout/stderr. Store node metadata once,
   compact sample arrays separately, and validate all 12 records and requested sample counts
   before treating a capture as complete. Keep raw logs for warnings and failures.
4. Amend the diagnostic policy before fresh comparisons, preserving the original acceptance
   plan and failed directories. Run a short validation capture, then restart all six runs with
   matching harness hashes: five warmups, thirty wall samples, three sessions per variant in the
   preregistered alternating order. Diagnostic omission must be consistent across variants.
5. Preserve the primary threshold of at least 20% lower complete warm latency and improvement
   in every session; every other workload must remain within the larger of 5% or 0.2 ms
   regression. A failed control calls for investigation, a revised explicit strategy, or rejection.

Multi-query-set profiling is a later alternative requiring verified query-index ownership,
resolution and timestamp lifetimes. Splitting or replaying graph segments solely for profiling
could change buffer state or scheduling, so it is not a substitute for uninstrumented wall time.
Record Chrome developer flags: its documented default timestamp quantization is 100 microseconds.
[Chrome timestamp guidance](https://developer.chrome.com/docs/web-platform/webgpu/developer-features).

## Memory should be evaluated from the compiled schedule

On the primary route smoke, baseline physical transient allocation is 13,651,260 bytes. The
initial candidate allocates 15,748,428 bytes, although its additional logical control payload is
only 16 bytes. The scheduled candidate returns to 13,651,276 bytes. Logical allocation is
22,220,588 bytes for baseline and 22,220,604 bytes for both candidates. These are graph allocator
statistics, not whole-process or device peak memory.

The existing candidate orders COO construction after component completion. The retained
snapshots show why dependency and lifetime review belongs beside kernel design: changing pass
order can alter scratch reuse by much more than a new buffer's payload. Future proposals must
report input/output bytes, persistent prepared state, logical scratch and physical graph
allocation separately. Add representative repeated-consumer peak measurements before claiming
an application memory improvement. [Captured allocation evidence](../../perf/resistance-components-validation-2026-10-24/research-audit.json).

## Broaden resistance research around the requested result

Circuitscape distinguishes scalar resistance requests from current/voltage maps and documents
a fixed-network shortcut reducing all-pairs scalar work to one solve per focal point relative
to an anchor. Its direct and iterative solvers also have different setup, reuse and memory
tradeoffs. These are reasons to research separate consumers; their CPU performance does not
predict luma's WebGPU performance. [Circuitscape solver guidance](https://docs.circuitscape.org/Circuitscape.jl/stable/compute/).

For a connected undirected network with fixed nonnegative conductances, choose anchor `a` and
let `G` be its reduced grounded Laplacian. For each focal vertex `j != a`, solve `G x_j = e_j`.
Linearity gives the pair solution `x_i - x_j`; hence
`R(i,j) = x_i[i] + x_j[j] - x_i[j] - x_j[i]`, with anchor entries defined as zero. In exact
arithmetic symmetry reduces the two cross terms to twice either one. Keep both terms in an
approximate-solve oracle to expose asymmetry. This is our proposed application of the standard
effective-resistance quadratic form, not an implemented luma API.
[Spielman, Effective Resistance and Schur Complements](https://www.cs.yale.edu/homes/spielman/561/lect13-18.pdf).

For F focal points in one connected component, this offers F−1 basis solves rather than
F(F−1)/2 pair solves. Extracting only focal potentials can retain an O(F²) response table while
streaming one O(V) solution, rather than storing O(VF) full solutions. This reduces solve count
only when enough pairs are requested; a few selected pairs may favor direct pair solves.
Producing every full current map still incurs output and edge-processing cost.

There are two correctness gates before a prototype:

- The current recipe grounds the requested sink. Reusing a fixed anchor requires explicit
  operator and source/sink semantics, disconnected-component handling, and validation of all
  invalid input edges. It cannot silently reuse a matrix assembled for another sink.
- Approximate basis residuals subtract as well: `r_pair = r_i - r_j`. If each basis has an
  absolute residual bound t, a pair can reach 2t. Passing the current tolerance on each basis
  does not guarantee passing it on a derived pair. Preserve the actual pair residual, sink
  balance and energy contract, or prove a sufficient tighter basis contract including f32
  subtraction error. Output pruning alone must not remove physical acceptance checks.

## Next experiments and stop conditions

| Priority | Research deliverable | Proceed only when |
| --- | --- | --- |
| 1 | Repair the resistance evidence specification and establish a valid component comparison | All workload/control, completion/fallback, memory and full-suite gates pass |
| 2 | Profile the accepted resistance graph again; compare adaptive SpMV plus subtraction with bounded long-row Jacobi/residual reductions | Duplicate diagonals, chunk boundaries, invalidity and unchanged numerical thresholds pass; short rows do not regress |
| 3 | Define scalar-pair, full-map and repeated-fixed-topology resistance consumers | Grounding, invalidation, pair certificates and output costs are explicit; each has an independent reference |
| 4 | Evaluate neighborhood strategy selection across density, radius, k and query count | Exact support counts and ordered IDs hold, and a crossover predicts held-out distributions |
| 5 | Extend S2 complex-ring and watershed supported-size sweeps | Ordered output, FIFO ties and budget-exhaustion semantics remain unchanged |
| 6 | Validate sparse Louvain f32 gains, conflict policy and coarsening on-device | Objective traces, convergence/budget state and community quality pass; no claim based only on exact-arithmetic checks |
| 7 | Measure complete consumers and another GPU backend | Preparation, first use, transfers, updates, output retention and baseline device limits are included |

Long-row balancing has a published foundation in merge-based CSR traversal, but the best mapping
depends on row skew and the cost on short rows. Prefer a local experiment over a blanket kernel
replacement. [Merrill and Garland](https://mgarland.org/papers/2016/spmv/).

The general practices remain: eliminate dataset-sized serial loops; reduce candidate work before
tuning workgroups; reuse prepared state with explicit invalidation; preserve exact versus
approximate contracts; and judge complete workflows with source-stable, correctness-gated
measurements. The next resistance change should be measurement repair, not a simultaneous new
preconditioner, connectivity algorithm and reduction scheme.
