# Geospatial backlog performance harness

This opt-in harness records bounded workload observations for the dense marked-point, priority
watershed and strongly connected component implementations. It does not establish a production
performance envelope until actual device results have been collected. It asserts numerical
correctness, never a timing threshold or a GPU speed advantage.

Run from the repository root, with no other GPU suites or source edits running:

```sh
nvm use
VITE_BENCHMARK_REVISION="$(git rev-parse HEAD)" \
VITE_BENCHMARK_SOURCE_STATE=clean \
yarn test-benchmarks modules/analysis/test/geospatial-expansion/backlog-performance-benchmark.spec.ts
```

Use `clean` only after checking `git status --short`. For a dirty checkout, preserve the patch and
new files alongside the log and set `VITE_BENCHMARK_SOURCE_STATE` to an identifying artifact name.
The commit alone cannot reproduce uncommitted changes. Keep the complete console log, including
metadata, all six measurement records, and the final test outcome. Partial runs retain completed
records but are not a complete envelope. The launcher selects the benchmark inventory and enables
its workload flags automatically; ordinary correctness runs exclude these files.

| Workload | Small | Bounded larger | Correctness reference |
| --- | --- | --- | --- |
| Translation-corrected marked pairs | 64 points, 4 radii | 512 points, 4 radii | Direct double-precision ordered-pair sums |
| Four-neighbor priority watershed | 8 × 8 cells, 3 markers | 24 × 24 cells, 3 markers | CPU binary heap with FIFO discovery tie ordering |
| Directed SCC | 16 vertices | 128 vertices | Tarjan SCC; one-way linked 8-vertex components |

The seed and generator version are recorded. Inputs use quantized coordinates/elevations, with
noninteger pair-radius boundaries. All initial outputs, outputs after repeated execution, and
updated outputs are checked. Pair statistics use a relative float32 tolerance of `5e-5` with a
unit absolute floor; counts, statuses, watershed labels and SCC labels must match exactly.

The same compiled graph is reused for a maximum-radius update, an elevation inversion, or a CSR
destination update that closes all components into one cycle. SCC updates upload the full neighbor
buffer even though only one destination changes. Upload bytes are reported. These cases measure
repeated execution and one input/parameter transition; they do not model streaming or recompilation.

Each record separates allocation/declaration, upload plus queue completion, synchronous compile,
first execution, one further warmup, five repeated executions, output readback, update upload and
three updated executions. Execution wall time includes CPU encode, submission and queue completion.
Optional GPU timestamp totals are reported only when the device has profiling enabled; otherwise
they are `null`. GPU timestamp readback is outside the wall-clock execution sample. The harness
does not request larger device limits, but records the limits supplied by the test device; do not
describe a run as a default-limit run unless that metadata confirms it. Software adapters are
included and explicitly identified by device metadata.

Graph compilation timing can omit deferred driver work. First execution can include deferred
compilation, and shared pipeline caches are not cleared. Graph transient allocation bytes are
reported separately from imported input/output bytes; driver peak memory is unavailable and is
`null`. CPU timings are single-run correctness-oracle costs, not optimized CPU library benchmarks.
No CPU/GPU speedup ratio is computed. Browser, adapter, limits and source identity must match when
comparing runs; retain the sample arrays rather than only the median.

The harness intentionally does not cover all expansion families, maximum supported shape bounds,
alternative-route enumeration, CPU library comparisons, or cross-device performance claims. Extend
it only with independently checked workloads and measured device results.

CPU fixture/reference checks run without a GPU:

```sh
yarn test-node --maxWorkers=2 modules/analysis/test/geospatial-expansion/backlog-performance.node.spec.ts
```
