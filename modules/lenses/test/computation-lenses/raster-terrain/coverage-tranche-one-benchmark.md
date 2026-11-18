# Coverage tranche one benchmark

Run only after the coordinator grants the browser lease, with no competing GPU work:

```sh
nvm use
VITE_BENCHMARK_REVISION="$(git rev-parse HEAD)" \
VITE_BENCHMARK_SOURCE_STATE=clean \
yarn test-benchmarks modules/lenses/test/computation-lenses/raster-terrain/coverage-tranche-one-benchmark.spec.ts
```

Use `clean` only when `git status --short` confirms it. Otherwise preserve the patch and new files
with the log, and set the source state to that artifact's name. The launcher selects the benchmark
inventory and enables workload flags automatically. Retain every metadata/measurement record and
the final test outcome.

Six cases exercise the actual public example wrappers: focal paired regression, fractional zonal
distribution, and prepared-coverage dasymetric redistribution, each on 8 × 8 and 32 × 32 grids.
The seed is `0x5eed2026`; generator/schema version is 1. Distribution and redistribution use eight
source zones and exact binary fractions. Focal windows have radius two. Redistribution has four
target zones. These are bounded observations, not established production workload envelopes.

Measurements separate setup/allocation, initial upload, synchronous compile, first execution,
five warmed executions, readback, and three alternating input updates. Focal response intercept,
distribution values, or source population totals change in place; each transition is numerically
checked. Independent analytic fits or CPU weighted sums check regression coefficients, moments,
source mass and target totals. The worker's independent-reference suites cover the full APIs.

Execution wall time includes CPU encoding, submission, and queue completion. Assertions and
readback are outside that clock. A first execution may include deferred compilation; pipeline
caches are not cleared. Graph memory includes retained captures for observability and is not
process/driver peak memory. GPU timestamp duration and driver peak are unavailable (`null`).
The metadata records actual adapter/browser/device limits without requesting larger limits.
No CPU library throughput comparison, speed ratio, or timing threshold is asserted. Categorical
and spectral throughput, polygon-source redistribution, maximum supported bounds, and other
input distributions remain unmeasured. A partial log is partial evidence, not a completed run.
