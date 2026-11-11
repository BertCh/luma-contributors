# interpolation-clustering (wave 1)

## 1. Done

### Item 1, GPU-05 for `GPULocalKriging` (`spatial-interpolation/gpu-local-kriging.ts`)
- New optional prop `sampleIndex?: 'auto' | 'brute-force' | 'grid'` (type `GPULocalKrigingSampleIndex`), resolved as `readonly sampleIndex: 'supplied' | 'brute-force' | 'grid'`. Caller `neighborhoods` still win (`'supplied'`), and `'grid'` together with `neighborhoods` throws.
- `'grid'` path. A prepare kernel writes the support centroids (same arithmetic as the kriging kernel) and a mask of the usable samples. That mask uses one shared WGSL `isUsableSample`, now also called by the scan, which is semantically unchanged. A `GPUNeighborSearch` kNN cross join (`createNeighborSearchDerivedParameterNodes`, no radius) then lists `neighborCount + 4` candidates per centroid. Its targets are the samples followed by the masked-out centroids, so the derived bounds also cover queries outside the sample extent. A pad kernel turns that CSR into fixed-width rows padded with `0xffffffff`, which costs one binding instead of two, and the existing candidate loop runs on them. The +4 margin absorbs the rounding difference between the search's `dx²+dy²` and the kernel's `distance()`. `searchRadius` stays in the kernel: taking the nearest rows and then filtering by radius keeps the same rows as a scan.
- Named limits:
  - `GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT = 4096`: the `'auto'` cut-over.
  - `GPU_LOCAL_KRIGING_MAXIMUM_BRUTE_FORCE_WORK = 2^32`: the limit on samples × queries for an explicit `'brute-force'`; above it construction throws through `validateMaximumCount`.
- Defaults: `'auto'` stays on the scan up to 4096 samples, so results there are unchanged. Above 4096 samples it uses the grid, which gives the same neighbours in the same order. It still keeps the scan when the grid would break a configuration that works today on a default device:
  - external drift, `parameters` and `output.weights` already use 8 storage bindings, or
  - `queries × (neighborCount + 4) × 4` bytes is above 128 MiB.
- Also: optional `spatialContext` (STD-06, planar native only), and the class doc now lists which props are per-frame and which are compile-time.
- Operator catalog: new `GPULocalKriging` entry in `ontology/all-entry-points/operator-annotations.ts` with three limits: neighborCount 32, the default scan at 4096, and the explicit brute-force work cap.
- Kernel audit: three `grid-sample-index-*` cases in `test/analysis-core/kernel-audit/cases-coverage-interpolation.ts`.
- Docs: `docs/api/gpu-spatial-analysis/interpolation-expansion.md` ("Kriging modes and support").
- Spec `test/gpu-spatial-analysis/spatial-interpolation/local-kriging-sample-index.spec.ts`. Grid and brute force run on the same inputs and are compared bitwise (status, values, weights) on three scenes:
  - a 9×9 lattice with two- and four-way ties, queries outside the extent, and samples excluded four ways (validity 0, NaN value, negative error, NaN position);
  - universal kriging with weighted two-point supports and `searchRadius` set on a tie;
  - only 5 usable samples with `neighborCount` 8.

  The spec also covers `auto` resolution, including the 8-binding fallback, and rejection messages for `sampleIndex`, `'grid'` together with `neighborhoods`, and a longitude/latitude context.

### Item 2, PP04 / GPU-05 for `GPUOPTICS` (`spatial-clustering/gpu-optics.ts`)
- New optional props `neighborIndex?: 'auto' | 'brute-force' | 'grid'` and `neighborCapacity?`, and a new optional output `output.candidateOverflow`.
- The `'grid'` reachability step is exact. A widened `GPUNeighborSearch` radius self join lists the candidates once, with radius and time window × (1 + 1e-4), so the list is a superset. Each visit relaxes only those candidates, using the same distance and time tests as the scan. The next row is chosen by (reachability, row) from a seed list of unvisited rows with finite reachability, or, when that list is empty, as the lowest unvisited row past a cursor. That choice equals the exhaustive argmin.
- Storage bindings: each frontier batch binds 7 (a packed `[x, y, t, core]` copy, the CSR, three outputs, one scratch view). A final kernel writes the status.
- Named limits:
  - `GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS = 4096`: brute force, and xi extraction on either path, since xi walks the ordering in a single invocation.
  - `GPU_OPTICS_GRID_MAXIMUM_ROWS = 65536`.
  - `GPU_OPTICS_DEFAULT_NEIGHBORS_PER_ROW = 64`: sets the default `neighborCapacity`.
  - `GPU_OPTICS_FIXED_COMMAND_NODE_ALLOWANCE = 256`.
- `'auto'` uses brute force up to 4096 rows, so results there are unchanged. Above 4096 it uses the grid; those inputs were rejected before.
- New `estimatedCommandNodeCount` and `validateCommandNodeBudget` (rule 4). This newly rejects nothing that was accepted before: at 4096 rows and batch size 1 the count is 4352.
- If the radius list overflows `neighborCapacity`, `candidateOverflow` is 1 and the ordering is not computed (`status` shows 0 visited, complete 0).
- Operator catalog: the `GPUOPTICS` entry stays `tier: 'reference'` and now has two row limits, 4096 and 65,536.
- Kernel audit: a 65,536-row grid case in `cases-coverage-point-process.ts`, and planar plus geographic space-time grid cases in `cases-space-time-clustering.ts`.
- Docs: `point-process.md` (table and the OPTICS section) and `space-time-clustering.md`.
- Spec `test/gpu-spatial-analysis/spatial-clustering/optics-grid-index.spec.ts`. Grid equals brute force on every output (ordering, core, reachability, predecessors, labels, status, xi intervals and summary) for six scenes:
  - 600 rows with lattice duplicates, noise and NaN rows;
  - a radius covering whole clusters;
  - a run stopped at `maximumIterations`;
  - xi extraction;
  - ST-OPTICS;
  - great-circle positions across the antimeridian.

  It also checks:
  - candidate overflow reporting;
  - a 20,000-row `auto` (grid) run: complete, a permutation, and predecessors consistent;
  - `auto` resolution, the row and xi caps, and the command-node budget, both the estimate as an upper bound and the rejection.

### Item 3, STD-06
Optional `spatialContext` (planar, native metric, `measure: 'distance'`) is resolved first thing in each constructor of `GPULocalKriging`, `GPULocalRBF`, `GPUNaturalNeighbor`, `GPUConditionalGaussianSimulation`, `GPUVecchiaFisherScoring` and `GPUVecchiaLogLikelihood`; the last also forwards it to its inner kriging. Omitting it behaves exactly as before. The four clustering classes already had a context; `GPUFocalStatistics` reads no point coordinates. Spec: `test/gpu-spatial-analysis/spatial-interpolation/interpolation-spatial-context.node.spec.ts`.

### Item 4, STD-03
All 10 bare `assert` calls in `gpu-conditional-gaussian-simulation.ts` are replaced with `validateCapacity`, `validateMaximumCount`, `validateFiniteNumber` and `validateRequirement`. The `assert` import from `@luma.gl/core` is removed. Four existing `.toThrow()` checks in `conditional-simulation-prepared.spec.ts` were tightened to the named messages. `yarn check-contributor-contract --all` now lists no bare asserts in my directories.

## 2. Verified
- `npx vitest run --project headless`:
  - the new local kriging and OPTICS specs, interpolation-advanced, covariance-parameters, kriging-conditioning, prediction-scale-robustness and vecchia-prediction (latest batch: 3 files, 39 tests pass);
  - interpolation-advanced, conditional-simulation-blocked, conditional-simulation-prepared, vecchia-log-likelihood, vecchia-fisher-scoring and interpolation-coverage (65 pass);
  - point-process, point-process-advanced, optics-selection and space-time-clustering (39 pass);
  - conditional-simulation-prepared and conditional-simulation-cooperative (10 pass).
- `npx vitest run --project node`:
  - kernel-audit-coverage-interpolation (40 pass);
  - kernel-audit-coverage-point-process, kernel-audit-space-time-clustering, point-process.node, space-time-clustering.node and analysis-operator-catalog (64 pass);
  - spatial-interpolation.node, interpolation-coverage.node and conditional-simulation-blocked.node (pass);
  - interpolation-spatial-context.node (6 pass).
- The operator-catalog spec failed once (operator count 262 instead of 261, and `GPULocationAllocation` had no summary), but only because another stream's export was mid-edit. It passed 11 of 11 when rerun.
- `yarn check-contributor-contract`: no new findings. `node scripts/check-upstream-first.mjs`: no violations.
- `node scripts/typecheck-status.mjs` with a regex over my files: 0 errors, but the last watcher cycle finished at 6:50:54 PM, which may be before my last `gpu-optics.ts` edit (the command-node budget) and the final spec tweaks. Recheck once the watcher cycles.
- `npx biome check --write` was run on every file I touched.
- Not run: the full test suites, build, or the website (forbidden or out of scope). The kriging grid path was never run above 4096 samples on a GPU; only the `'auto'` resolution was checked by construction.

## 3. Not done
- Item 5, STD-05: no change. `GPULocalKriging` already reads the model from a `parameters` view (`getGPULocalKrigingParameterValues`) as an opt-in, and changing sill, range or nugget there does not recompile. Making that the default (a parameter buffer created internally from `covariance`) changes the generated graph and cannot be shown in this wave to leave results unchanged.
- OPTICS:
  - Xi extraction stays at 4096 rows, in a single invocation.
  - The ordering itself stays sequential in one workgroup. Exact OPTICS order has no parallel formulation here, so the tier stays `reference`.
  - Above 65,536 rows a different algorithm is needed, for example a parallel MST-based OPTICS or HDBSCAN ordering.
  - The initialize kernel still strides over rows in one workgroup (as before).

## 4. Wave 2 requests
- No renames.
- Contract checker: `single-invocation-data-loop` is still reported for `gpu-optics-xi.ts:29` and `conditional-simulation-blocked.ts:323`; both are reference-tier code this pass did not touch.
- If the website or lenses want large OPTICS or kriging runs, they need nothing new: `'auto'` picks the grid.

## 5. Roadmap
- GPU-05: remove "local kriging, OPTICS" from the brute-force list. Remaining for these two: OPTICS xi extraction stays at 4096 rows, and the ordering is sequential (reference tier, 65,536 rows with the grid). The local-kriging grid falls back to the scan under `'auto'` when 8 bindings or 128 MiB of candidate rows would be exceeded.
- PP04: "Indexed core distances and an indexed seed-list frontier landed 2026-11-10 (`neighborIndex`, exact, at most 65,536 rows); xi extraction stays at 4096. Remaining: a parallel ordering for larger clouds." Effort S.
- STD-06: remove local kriging, local RBF, natural neighbour and conditional simulation from the remaining list (Vecchia likelihood and Fisher scoring also gained a context). Kalman, restriction and similarity search remain.
- STD-03: `gpu-conditional-gaussian-simulation.ts` is clean (10 asserts fewer).
- STD-05: change "Baked: the covariance model in `GPULocalKriging`" to "`GPULocalKriging` bakes the model unless `parameters` is supplied (per-frame path exists); making it the default remains".

## 6. Noticed
- `GPUSpaceTimeKriging` docs say "A query scans every sample ... there is no resident neighbor index input yet". It is still brute force and is not in my scope.
- `GPUNeighborSearch` drops cross-join query rows outside the per-frame bounds. Any caller that derives bounds from the targets alone (the `createNeighborSearchDerivedParameterNodes` helper) silently loses queries outside the target extent. I worked around it by appending the queries as masked targets. A `queryPositions` option on the helper would be cleaner (neighbor-search owner).
- The kernel audit enforces 8 storage bindings. `GPULocalKriging` with external drift + `parameters` + `output.weights` + caller CSR `neighborhoods` already needs 10, and this is undocumented outside the prop comment.
