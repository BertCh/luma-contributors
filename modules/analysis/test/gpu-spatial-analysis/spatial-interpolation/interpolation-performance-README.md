# Interpolation performance evidence

This opt-in suite follows the [backlog benchmark accounting](../../geospatial-expansion/backlog-performance-README.md). It measures ten deterministic bounded workloads and asserts correctness, with no speedup ratio or latency threshold. Run it alone on the GPU:

```sh
nvm use
node scripts/benchmark-geospatial.mjs --suite interpolation --workload variogram \
  --warmups 5 --samples 30 --update-samples 5 \
  --output dev-docs/research/evidence/variogram-baseline-session-one
```

The output directory must be new and its parent must exist. The wrapper preserves source hashes
before execution, the full log, and completion metadata. It rejects a comparison capture if relevant
sources change during execution. Run it without concurrent GPU suites or builds; it does not acquire
a machine-wide GPU lock. Repeat with fresh output directories for independent sessions. Omit
`--workload` to capture every interpolation workload; use `--suite statistics --workload sur` for SUR.

Direct `yarn test-benchmarks modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-benchmark.spec.ts` use remains supported with `VITE_BENCHMARK_WARMUPS`,
`VITE_BENCHMARK_SAMPLES`, `VITE_BENCHMARK_UPDATE_SAMPLES` and `VITE_BENCHMARK_WORKLOAD`.
Provenance is null without a valid `VITE_BENCHMARK_SOURCE_SNAPSHOT`; no historical snapshot is
silently substituted. The graph uses the test device and records its actual adapter/limits.
Software adapters are not hardware throughput evidence. The retained tables below describe their
original sample counts, not the configurable settings of a new capture.

| Workload | Small | Bounded larger | Correctness evidence |
| --- | --- | --- | --- |
| External-drift kriging | 8 observations, 4 queries, 1 support point | Maximum 16 neighbors, 4 drift columns, 16 support points; 128 queries; all 8 bindings | Bilinear drift reproduction, finite nonnegative variance, weight partition, successful status |
| Thin-plate RBF | 8 observations, 4 queries | Maximum 16 neighbors and degree 1; 128 queries | Affine reproduction with smoothing, retained counts and rank status |
| Sibson natural neighbors | 8 observations, 4 queries | Maximum 32 observations and 64-vertex capacity; 128 queries | Affine precision, nonnegative weights and partition of unity |
| Conditional Gaussian simulation | 8 observations, 4 joint sites, 16 replicates | Maximum 16 observations and 16 joint sites; 128 replicates | Independent f64 elimination moments, finite draws, reproducible stream replay and expected mean shift |
| Resident variogram fit | 20 bins, three model candidates | 256 bins, three model candidates | Generating exponential parameters, all candidate admissibility and explicit convergence/boundary status |

All five families replay the same compiled graph. Observation/value updates add three; empirical semivariances double. Initial, warmed, and updated results are validated. Simulation uses seed `0x494e5445`; other coordinates are deterministic rational grids and modular query sequences. Fixture generator version is recorded. The variogram workload uses 200 range candidates and 32 refinement iterations; 256 bins is a measured workload, not a documented bin-count maximum.

Each measurement separates declaration/allocation, deferred upload plus queue completion, synchronous compile, first execution, one warmup, five warm executions, output readback, update upload and three updated executions. Execution wall time includes CPU encoding/submission and queue completion. GPU timestamps are reported when enabled, otherwise null. GPU timestamp mapping and result reads are outside execution timing. CPU fixture generation occurs before measured case phases; validation costs are reported separately.

Pipeline caches remain shared and deferred driver compilation can appear in first execution. First execution is not a guaranteed cold-driver result. Logical/physical graph transient bytes are separate from caller-owned input/output bytes; driver peak memory is unknown. These contributors use invocation-local scratch and may report zero graph transient allocation; that does not imply zero shader storage or register pressure.

The node suite constructs audited combinations, including eight-bound external drift, 32-sample/64-vertex Sibson, 16+16 simulation and 128+128 prepared simulation (both with 128 replicates), and 1024-candidate/64-iteration variogram fitting. These simulation fixtures audit cooperative-path storage; they do not construct the 4096-joint-row blocked dense maximum. It audits WGSL and records exact static private/function storage, workgroup memory and bindings as `GEOSPATIAL_INTERPOLATION_SHADER_BUDGETS`. These are static source budgets rather than measured register occupancy or driver allocation:

```sh
yarn test-node --silent=false --maxWorkers=2 modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance.node.spec.ts
```

The benchmark establishes only the recorded device/workload observations. It does not compare optimized CPU libraries, global large-data prediction, different adapters, indexed topology reuse, or end-to-end variogram readback avoidance. The independent numerical suite separately checks noisy SciPy variogram fits, kriging covariance equations, Shapely stolen areas and simulation ensemble covariance.

## Recorded run: October 23, 2026

[Full measurements](../../../../../dev-docs/research/evidence/interpolation-coverage-2026-10-23.json) preserve all ten workload records, phase samples, input/output bytes, correctness results and maximal shader budgets. All initial, repeated and changed-input checks passed on native Apple WebGPU/Metal 3, HeadlessChrome 151, with a shared test device. The device exposed 10 storage bindings, 32 KiB workgroup memory and 1024 invocations per workgroup; this was not a native default-limit-device run. The independent static audit passed the stricter core budgets, including eight storage bindings.

The table reports milliseconds. Warm execution is the median of five encode/submit/queue-completion samples. Readback is the single output read after those five samples; GPU timestamps were unavailable in the recordings despite device support. Browser-clock granularity and the small sample count limit interpretation of sub-millisecond differences.

| Workload | Upload | Host compile | First execution | Warm median | Readback |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kriging: 8 samples, 4 queries | 1.1 | 0.4 | 87.9 | 0.4 | 1.8 |
| Kriging: 16 samples, 128 queries, 16 support points | 0.3 | 0.1 | 102.4 | 0.9 | 1.9 |
| RBF: 8 samples, 4 queries | 0.2 | 0.2 | 70.8 | 0.4 | 1.7 |
| RBF: 16 samples, 128 queries | 0.2 | 0.2 | 84.8 | 0.7 | 0.6 |
| Sibson: 8 samples, 4 queries | 0.2 | 0.2 | 125.2 | 0.6 | 0.9 |
| Sibson: 32 samples, 128 queries | 0.1 | 0.2 | 120.5 | 2.9 | 1.0 |
| Simulation: 8 observations, 4 sites, 16 replicates | 0.2 | 0.2 | 66.1 | 0.3 | 0.7 |
| Simulation: 16 observations, 16 sites, 128 replicates | 0.1 | 0.2 | 107.6 | 3.9 | 1.5 |
| Variogram: 20 bins, 3 models | 0.3 | 0.1 | 127.9 | 5.0 | 1.0 |
| Variogram: 256 bins, 3 models | 0.8 | 0.1 | 173.8 | 52.1 | 2.7 |

The 256-bin fitting warm samples ranged from 51.8 to 117.4 ms. Its serial per-model candidate/bin loops are a material current cost; these measurements do not establish a benefit over CPU fitting or over reading empirical bins back. First execution was much slower than warm execution throughout, consistent with deferred pipeline/driver work being included. No causal partition of that first-execution cost was measured.

All ten graphs allocated zero transient graph bytes because their scratch arrays are invocation-local. At maximal supported shader options, private/function bytes were 1764/308 for kriging, 1600/176 for RBF, 1028/1028 for Sibson, 0/4872 for simulation, and 24/88 for fitting. All use zero workgroup storage. Driver/process memory and register occupancy were not measured.

The run omitted the source revision/state environment variables; the original metadata therefore retains nulls. The artifact adds extraction-time revision and source hashes with an explicit provenance caveat. Preserve the final dirty source snapshot alongside it; these observations should not be treated as a clean-commit, precisely replayable historical baseline.


## Expanded indexed and shared-factor run

The same benchmark entry now also runs `interpolation-scaling-fixtures.ts` and the actual CPU
round-trip comparator in `interpolation-fit-comparison.ts`. Run with `--maxWorkers=1 --silent=false`
after enabling `VITE_GEOSPATIAL_BACKLOG_BENCHMARK=true`. Schedule it separately from other native GPU
tests; the capture wrapper does not acquire a machine-wide lock. `interpolation-round-two-source.json` belongs to the retained historical
run and is no longer used as provenance for new runs.

The expanded artifact is `dev-docs/research/evidence/interpolation-advanced-2026-10-23.json`: 17 workloads and
two repeated-fit comparisons, all initial/repeated/changed-input checks passed. Indexed Sibson
warm medians were 0.3/0.3/0.4 ms at 1,024/16,384/65,536 sites for 128 queries; fixture topology and
seed construction are outside these timings. Grid-index construction plus CSR RBF used 5.6/6.9 ms
at 1,024/16,384 sites. Shared-factor simulation used 5.1 ms at 32+32 observations/sites and
146.2 ms at 128+128, with 17,672/267,272 graph transient bytes.

Resident three-model fitting cost 4.3/51.3 ms at 20/256 bins, while actual empirical-bin readback,
CPU fitting and fitted-parameter reupload together cost 1.3/3.2 ms. This workload shows no benefit
from avoiding readback. Both branches start from updated resident empirical buffers, use a
200-step range ladder, and compare fitted parameters; CPU always does 60 refinements whereas GPU
can stop at its documented convergence tolerance. Output validation readback is outside both
branch timings. No GPU speedup is claimed.
