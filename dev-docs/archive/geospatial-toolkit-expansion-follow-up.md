> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/geospatial-toolkit-expansion-follow-up.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Open gates are carried into the roadmap; this copy is the original record.

# Geospatial toolkit expansion follow up

Status recorded October 15, 2026. All 30 candidate areas and 12 recipes from the
[expansion proposal](./geospatial-toolkit-expansion.md#implementation-map) now have public
implementation paths, reference fixtures, documentation and executable computation examples.
The next work is validation closure and measured scaling. The implementation is experimental;
coverage of the proposal does not imply every reference-library option or production-scale
performance is supported.

October 23 research update: the [GPU geospatial coverage backlog](./geospatial-coverage-backlog.md)
adds 78 scoped extensions, new operations and recipes, with an expanded inspiration-library survey.
The [function coverage matrix](./geospatial-function-coverage.md) maps all 78 IDs to reviewed
function/option scopes and separates implementation presence from validation evidence.
Its current-tree reconciliation supersedes the historical semantic-gap list below where later work
has landed, including adaptive generalized GWR, unit-effects panels, LOSH residual permutation,
four-quadrant nodata filling and tile-routed reprojection. The October 15 verification record remains
historical evidence; this research does not certify a new full-suite result.

## Backlog implementation round — October 23, 2026

This round preserves the original expansion and adds six bounded extensions:

| Area | New implementation | Remaining boundary |
| --- | --- | --- |
| LOSH inference | `GPULOSHResidualPermutationTest` uses shared seeded conditional permutations and multiple-testing correction on the fixed normalized residual field | Not esda LOSH inference or raw-value permutation with refitting; joint multivariate inference remains open |
| Generalized GWR | Explicit adaptive complete-case neighbor bandwidth for Poisson/logistic fits, including deterministic ties and degenerate-response status | No automatic bandwidth selection or general separation certificate |
| Spatial panels | `effects: 'unit'` performs GPU within transformation with corrected absorbed-effect residual degrees and homoskedastic covariance | Balanced complete panels and time-invariant weights; time/random/dynamic effects and clustered covariance remain open |
| Nodata filling | Separate `gdal-four-quadrant` method with direct Rasterio/GDAL fixtures and original donors | Integer bounded radius and zero smoothing; existing all-donor IDW remains a different method |
| Tile reprojection | `GPURasterTileReprojection` explicitly assembles a bounded GPU mosaic, then samples across tile seams | Shared source CRS/lattice/resolution, dense scratch and float32 precision; no implicit tile fetching |
| Alternative routes | Bounded GPU min-heap uniform-cost simple-path search replaces exhaustive DFS and certifies a returned prefix | Explicit frontier/step/storage budgets; no production-scale throughput claim |

The domain contracts are linked from the raster, network and spatial analysis references.
New independent reference generators, executable examples, numerical GPU tests and maximal-option
kernel audits accompany these APIs. Spatial expansion captures now use semantic output names and
avoid duplicate captures of the same view.

Fresh focused GPU verification closed the historical join, visibility and geomorphon failures:
all five affected test files passed in one serialized run. Final full-suite and build outcomes on
the frozen implementation tree are recorded below. Two physical-renderer comparisons remain open.

### Current-tree verification

- `nvm use`, `yarn install`, `yarn lint fix`, `git diff --check` and the final `yarn build` passed.
- `yarn website:build` and `(cd website && yarn build)` passed, including 1,026 raw documentation pages.
- Five public package smoke checks passed for ESM, CommonJS and declarations.
- Full node phase of `CI=1 VITEST_MAX_WORKERS=2 yarn test`: 12,003 passed and 347 skipped across
  1,125 files (1,123 passed, two skipped). The subsequent software-GPU phase was interrupted because
  `CI` selects SwiftShader; the complete hardware GPU phase is recorded separately below.
- Native hardware GPU coverage through `yarn test-headless`: all 971 files accounted for, with
  10,954 tests passed, 654 skipped and two failed. The monolithic attempt lost its browser connection;
  three of four one-worker shards also disconnected. Seven fresh-browser recovery batches completed
  all 99 unfinished files successfully. These totals deduplicate completed file results across the
  four shards and recovery batches. No numerical assertion or tolerance was relaxed. This completes
  file coverage, but the two renderer failures and browser disconnects mean the full-suite gate is
  not green. Logs and JSON reports are in `/tmp/luma-contributors-backlog-20261023`.
- New extension GPU checks, existing pooled statistics compatibility, centrality and adversarial
  recipe composition passed. Central backlog and expansion audits passed all 112 cases; inventory
  checks passed eight tests. Focused discovery/docs/scenario integration passed 62 tests.
- The previous join, visibility and geomorphon failures passed in a fresh hardware run. Two existing
  physical-renderer WebGPU comparisons still differed at sparse highlights (maximum channel error
  five and ten against a tolerance of two). Both WebGL comparisons passed. Source, fixtures and
  tolerances were preserved after two diagnostic reviews found no demonstrated correction.

### Bounded performance observations

The opt-in harness now verifies dense marked pairs, priority watershed and strongly connected
components before and after GPU input updates. Six workloads passed. The
recorded samples (local research log, not checked in) preserve device limits, compile,
upload, execution, readback and graph allocation measurements. On this Apple Metal adapter,
repeated encode/submit/completion medians were 1.9/2.5 ms for 64/512 marked points,
2.8/36.1 ms for 8×8/24×24 watersheds, and 0.2/0.3 ms for 16/128-vertex SCC inputs.

These are bounded smoke observations: CPU build and node verification were concurrent, the test
device used raised limits, and there is no optimized CPU-library or cross-device comparison.
The watershed increase warrants scaling work; it does not establish a supported production envelope.
The fixture README describes the isolated rerun protocol and retains every timing sample.
Centrality also limits declared rounds to the graph's node count when that is below the requested
budget; its maximum-size graph still needs the broader continuation redesign.

## What was completed

| Area | Delivered | Contract reference |
| --- | --- | --- |
| Statistics and regression | LOSH, multivariate local Geary, marked/cross-type point statistics, cross and space-time variograms, generalized GWR, multiscale backfitting, pooled spatial panels and regime OLS | [Statistics](../../docs/api-reference/experimental/gpu-spatial-analysis/statistical-expansion.md) |
| Raster and terrain | Fractional coverage and weighted zonal summaries, nodata filling, rank filtering, D8 distance and connected stream reaches, walking costs, reconstruction, hole filling, reprojection, watershed segmentation, skeletons and maximal disks | [Raster and terrain](../../docs/api-reference/experimental/gpu-raster/expansion.md) |
| Geometry | Alpha-complex hulls and grouped outlines, constrained Delaunay refinement, shared triangulation below raster and spatial APIs | [Geometry contracts](../../docs/api-reference/experimental/gpu-raster/expansion.md#group-outlines-and-recipes) |
| Networks, cells and tracks | Edge betweenness, SCCs, bridges, articulation points and biconnected blocks, bounded alternative routes, H3 edge/vertex identities, LCSS and bounded facility allocation | [Network contracts](../../docs/api-reference/experimental/gpu-network/expansion.md) |
| Workflows | Zonal distributions, network disruption, interpolation diagnostics, facility scenarios, population exposure, regional change, cluster footprints, drainage networks, track quality, local variability, cell flows and surface preparation | [All twelve recipes](../../docs/api-reference/experimental/gpu-spatial-analysis/expansion-recipes.md) |

Integration includes public barrels, discovery ontology entries, computation-pack descriptors and
formulas, 41 executable lens examples, documentation navigation, canonical status layouts,
ownership checks and mandatory maximal-option kernel audits. The examples construct real graphs;
null-device construction checks are separate from numerical GPU execution.

The final dependency layout matters when continuing this work:

- Import network disruption, facility scenarios and track quality from
  `@luma.gl/graph/gpu-network`. Track quality moved there because optional map matching
  depends on network code. The other expansion recipes are exposed by spatial analysis.
- Shared polygon triangulation lives in analysis-core, with spatial public re-exports preserved.
  Raster fractional coverage must not import higher-level spatial contributors.
- Retain explicit directory `/index` imports, caller-owned outputs, per-stage failure evidence and
  the default eight-storage-buffer limit. Arrow adapters remain outside these runtime packages.

## Verification at handoff

| Check | Recorded result |
| --- | --- |
| Setup | `nvm use` and `yarn install` completed |
| Formatting | Final `yarn lint fix` and `git diff --check` passed |
| Runtime build | Final `yarn build` passed; log checked for TypeScript and bundler errors |
| Focused GPU verification | 100 passed: 49 expansion cases, 17 existing triangulation regressions and 34 fresh-browser import regressions |
| Integration node verification | 198 passed across 11 files, including 93 maximal-option audit variants, layering, vocabulary, imports, inventories, centrality, docs and examples |
| Lens construction | 41 passed; included as construction evidence, not additional GPU parity evidence |
| Package verification | Five checks passed: geospatial, spatial analysis, raster, terrain and network; ESM, CommonJS and declaration imports |
| Website | Both `yarn website:build` and `(cd website && yarn build)` passed, including `llms.txt` and 1,020 raw documentation pages |
| Full node attempt | 11,836 passed, 345 skipped, 10 failed across 1,113 files. Expansion integration failures were subsequently fixed and covered by focused reruns |
| Full headless attempt | 5,918 passed and 120 failed before browser disconnection; the suite was incomplete. Three import-failed suites later passed all 34 tests on a fresh browser |

The full suite has not been demonstrated green on the final frozen tree. The full-run totals
precede the final integration corrections and must not be presented as a final failure count.
The [implementation verification record](./geospatial-toolkit-expansion.md#implementation-verification)
provides the accompanying context. No commit, merge or release is part of this handoff.

## Remaining work in priority order

### 1 Close full suite verification

Run the complete node and headless suites on a frozen tree, without concurrent source edits.
Keep focused tests as supporting evidence rather than a substitute for the required full suite.
Use bounded workers or shards when resource contention prevents a reliable run, and retain each
shard's outcome. Reproduce failures against a clean baseline before attributing them to this expansion.

The historical failure counts in the handoff record are superseded where focused reruns passed.
They must not be presented as a current failure inventory.

| Area | Latest recorded disposition and remaining evidence |
| --- | --- |
| Join parity, terrain visibility and geomorphons | Five affected files passed a fresh serialized hardware run, as recorded above. Preserve those fixes/conventions and include them in the frozen-tree full suite; the earlier 64/42/12 counts are historical |
| Physical materials | Two WebGPU comparisons still differed at sparse highlights; both WebGL comparisons passed. Reproduce against a clean baseline in `modules/three/test/physical-renderer.spec.ts` before changing implementation or tolerance |
| Long audits and browser disconnection | The maximal-option backlog/expansion audits passed their focused run. Complete full-suite scheduling/browser-lifetime evidence; an interrupted run remains incomplete |
| Centrality compile/graph size | Declared rounds are capped by node count, but larger cases still unroll sigma/delta phases. Measure bounded-continuation changes against the existing numerical contract; see the algorithm follow-up |

Do not weaken numerical assertions or skip contributors to obtain a green run. Once relevant
failures are resolved or independently established, record a fresh full-suite result and repeat
the required merge checks after any final source and formatting changes.

### 2 Establish practical performance envelopes

Use the GPU implementation survey (local research log, not checked in)
for candidate algorithms and the [current algorithm follow-up](./gpu-analysis-algorithms-roadmap.md#current-follow-up-2026-10-23)
for shared work. Existing FFTs, scans/sorts, load balancing and basic flow routing are reuse points.
The bounded samples above motivate scaling experiments without establishing speed advantages.

Correctness fixtures and shape bounds are present; representative throughput and speed advantages
remain unproven. Measure compile time, upload, GPU execution, optional readback, peak scratch memory
and repeated parameter updates against established CPU libraries. Include small inputs and default
WebGPU device limits as well as realistic large workloads.

Prioritize dense pair statistics, GLM/MGWR neighborhoods and bandwidth search, grouped alpha hulls,
fractional zone/cell clipping, SCC traversals, alternative-route enumeration, watershed queues and
medial-disk searches. Publish supported workload/device envelopes before increasing bounds or
promoting discovery evidence beyond its current partial status. Replace expensive traversals only
with equivalent contracts and independent reference checks.

### 3 Broaden reference and composition evidence

- Add independently generated larger and adversarial datasets, preserving package versions,
  rounding, units, masks, topology conventions and tolerances. Label direct library calls separately
  from compositions; reference-family availability is not full parity coverage.
- Expand complete-domain versus halo/tile tests where an operation supports tiling. Nodata fill
  must use original donors, rank filters need their footprint halo, and summaries must not count
  halo cells twice. Quantiles cannot be merged by averaging tile quantiles.
- Extend topology, conservation, failure propagation and changing-input tests across the recipes.
  Preserve source identities and status evidence when reusing baselines or prepared results.
- Preserve the semantic output labels added in the October 23 round; keep new example formulas
  and captured fields aligned and retain checked public API examples. Persisted captures, if
  required, must come from actual execution.

### 4 Select additional semantics only for concrete use cases

The table below records the October 15 boundaries. The October 23 round above supersedes the
implemented portions; the remaining extensions are still separate work. These are deliberate
boundaries, not undisclosed completed features. Treat extensions
as separate scoped work with an independent oracle and explicit API contract.

| Current implementation | Possible subsequent work |
| --- | --- |
| Descriptive LOSH and multivariate Geary; separate Moran inference | Validated LOSH/joint multivariate randomization, inference and multiple-testing conventions |
| Rectangular translation correction, supplied intensities and annular correlation | Intensity estimation, additional study-window corrections and kernel-smoothed point correlation |
| Fixed-bandwidth Poisson/logistic GWR | Adaptive or selected bandwidths, separation handling and supported inference |
| Gaussian MGWR finite-grid CV and conditional submodel errors | Broader bandwidth search and full backfit/post-selection uncertainty; current errors hold other surfaces and bandwidths fixed |
| Pooled lag 2SLS and fully varying regime OLS | Fixed/random effects, dynamic panels, clustered covariance, regime restrictions, covariance and Chow tests |
| Exact bounded p-center; demand-order capacitated greedy allocation | Certified capacitated optimization or documented approximation guarantees; the current heuristic is not an infeasibility certificate |
| Circumradius alpha geometry and maximal-inscribed-disk centers | Ratio-based concave hull or a distinct one-pixel medial-axis contract, if needed; do not relabel current outputs as those algorithms |
| Original-donor nearest/IDW fill | GDAL four-quadrant filling as a separate method; current all-donor IDW is different |
| D8 drainage over one complete domain | Cross-tile network stitching and explicit D-infinity/MFD branching semantics |
| Caller-selected single source tile and float32 inverse reprojection | Explicit source-tile routing and additional precision/filtering contracts |
| Bounded local routes and dense graph traversals | Scalable k-shortest/simple-route search and generic graph adapters while preserving completeness flags |

## Resume points

Start with the [implementation map](./geospatial-toolkit-expansion.md#implementation-map) and the
domain contract links above. Numerical fixtures and generators live under
[`modules/graph/test/oracles`](../../modules/graph/test/oracles); expansion GPU tests
are in [`test/geospatial-expansion`](../../modules/graph/test/geospatial-expansion),
[`recipes/expansion-recipes.spec.ts`](../../modules/graph/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts)
and [`regression/geospatial-expansion.spec.ts`](../../modules/graph/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts).
Mandatory maximal-option cases are registered in
[`kernel-audit/cases-expansion.ts`](../../modules/graph/test/analysis-core/kernel-audit/cases-expansion.ts).

The next maintainer should first close the full-suite evidence gap, then select a measured scaling
or semantic extension from the priorities above. The original proposal's delivery order is historical;
it is not a list of unimplemented candidate IDs.
