---
title: "Network status and count checks"
description: "Independent expectations for network diagnostics and bounded results."
---

# Network status and count checks

Counts and diagnostics have independent expectations in the network regression tests. A routing
reference supplies paths or distances; output capacity, pending work and input validity determine
the accompanying status words. The `*-status-oracles` specs use small enumerable fixtures and
poison GPU outputs before execution, so an unwritten zero word cannot satisfy a zero expectation.
Headless assertions require execution on a WebGPU device; a CPU result or successful typecheck
does not qualify the GPU implementation.

| Operation | Fixture and expected diagnostics |
| --- | --- |
| Alternative routes | A two-edge chain needs two edge examinations. After one, the second edge remains unexamined and `complete=0`. Construction reads the expansion getter once and retains that allowance through execution and repeated lowering. Two pending diamond prefixes require frontier capacity two. |
| K shortest paths | Three simple paths cost 2, 3 and 4. Round two must retain two spur proposals; candidate capacity one sets `candidateOverflow=1`. Accepting all three takes three Yen rounds. |
| Map matching | Forward fixes on one directed edge need one traversal entry. Reversing them forces a restart, `[edge, separator, edge]`, requiring three entries. `traversedEdges.status.overflow` depends on that requirement; final matched-row `status.overflow` remains zero because each fix has an output slot. |
| Matched edge volumes | Lists `[0,1,separator,2]` and `[0,1,2]` contain movements `0→1` twice and `1→2` once. Counts and incoming/outgoing totals are independent of scoring or routing. Input incompleteness propagates to both volume and transition diagnostics. |
| Analytics columns | An edge joining differently labelled nodes must change a label in the first component sweep. That sweep cannot certify unchanged labels. An isolated-node replay is stable. |
| Centrality | One local hop on an eight-edge chain leaves the last node unresolved. A zero-weight cycle is a separate reason from exhausting search or dependency rounds. `approximation` follows the requested sampling mode. |
| Cost tables | The directed chain `0→1→2` has six reachable ordered pairs, including the diagonal. One local hop finds five and leaves `(0,2)` unresolved. Output truncation and search exhaustion have separate flags. |
| Flow | Two parallel unit edges require two augmentations and two decomposed paths. Path capacity one retains one but reports two required paths. This truncation does not invalidate a completed solver. Invalid endpoints and an exhausted solver budget have different diagnostics. |
| Point routing | A required turn with edge costs 1 and 2 gives cost 3 and status 0 when complete. An invalid source gives status 1. Budget zero leaves the required turn unexamined and gives status 2, in both reference and scalable modes. |
| Network K | Two endpoint events and two directed copies of one road create four snapping candidates. Capacities one and three overflow; four fits. A six-node chain independently forces `[converged, limit]=[0,1]` at one local-hop round and `[1,0]` at eight. |
| Inhomogeneous network K | A valid length-4 edge and radii `[4,3]` give full diagnostic words `[0,16]`: only the second radius has invalid-order bit 16. Repairing to `[3,4]` clears both words. This fixture exercises the radius bit, not every nonzero diagnostic reason. |
| Network noding | An integer crossing needs four pieces/eight vertices. Capacity three retains three/six while requirements stay four/eight and overflow becomes one. One uncertifiable pair forces uncertainty one; replaying the exact crossing clears it. |
| Path extraction | Targets 2 and 0 in `0→1→2` need four node rows and two edge rows. Capacity truncates those counts; a walk bound of one suppresses the long path and raises candidate loss instead. |
| Reachability | A three-node zero-weight chain can have an unresolved predecessor. A two-node zero-weight cycle instead has zero unresolved predecessors but pending tie work after one hop: `[unresolved,converged,limit]=[0,0,1]`; two hops give `[0,1,0]`. |
| Service areas | Zero active rounds reach only the seed of `0→1→2`, with count one and status `[0,1]`, even after caller mutation; four one-hop rounds complete. On `0→1→2→3`, settled unit costs and one label hop give assignments `[0,0,NONE,NONE]`, parents `[NONE,0,NONE,NONE]`, count two and status `[0,1]`. On the zero-weight chain, one tie hop leaves two parents unresolved and status `[0,1]`; disabling predecessors skips that phase, returns all `NONE` on CPU and reports `[1,0]` after cost/label completion. CPU round budgets 268435455 and 268435456 both exhaust the three-node chain with costs `[0,1,2]`, count three and status `[1,0]`: only three cost hops are needed. |
| Structure | Exact SCC traversal reports `[converged=1, iterationCount=1, iterationLimitReached=0]`. |
| Time routing | For reverse-listed transitions `1→2`, then `0→1`, sweep one reaches node 1, sweep two reaches node 2, and sweep three certifies unchanged labels. A one-sweep budget therefore reports an incomplete result. |
| Travel-time percentiles | One-edge departure fixtures independently count reachable departures and check both words of every `[valid, converged]` pair, including `[1,0]` at zero budget. |

## Traversal and continuation budgets

See [traversal phases, limits and usable results](./concepts.md#traversal-phases-limits-and-usable-results).
Alternative routes allows 1–10,000,000 examined CSR slots per query per encoding, including rejected
slots, and 1–65,536 simultaneous pending prefixes. `counts[query]` bounds its usable
cost/lexicographic-edge-ordered prefix; `complete` certifies the requested top-k result per query.
K-shortest retains its separate batch interruption status and cost-ordered certified prefix.

Centrality continuation has independent window and total budgets: `roundsPerStep` is 1–1,023 rounds
per submitted window (default 64), while `maximumIterations` is 1–4,294,967,294 rounds per phase per
pivot batch across all windows (default vertex count). Search, path counting and dependency phases
retain GPU state across windows. A boundary queue readback advances scheduling; it is not a new
whole-search allowance. `reset: false` (or 0) adds totals and merges diagnostics; aggregate
`iterationLimitReached` stays set through later successful batches until `reset: true` (or 1).
Per-pivot outputs are overwritten by each batch; do not change topology or pivots mid-batch.

The headless correctness fixtures are queued for device execution; this page does not claim a
fresh qualification run. [Hardware measurements](./network-movement-benchmarks.md) record separate
host/device conditions and timings. Benchmarks establish neither correctness nor completion for
another graph, budget or device.

These checks preserve each operation's documented CPU/GPU numerical and scheduling differences.
Unused storage is asserted only where the output contract defines it. Compact counts remain the
authority for which rows a caller may consume.
