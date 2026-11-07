# County statistics implementation lane

## Delivered files

- `examples/deck/spatial-analysis-showcase/src/scenes/regression/county-panel.compute.ts` and `.scene.ts`: historical unemployment panel, five distinct panel specifications, and separate 1990 lag/error cross-sections.
- `examples/deck/spatial-analysis-showcase/src/scenes/statistics/county-regionalization.compute.ts` and `.scene.ts`: seeded AZP, sequential and mutual-nearest Ward, and population-constrained Max-P.
- `examples/deck/spatial-analysis-showcase/src/data/datasets/ncovr-new-jersey.dataset.ts`, `scripts/data/ncovr-new-jersey/{build.py,README.md}`, and generated `public/data/ncovr-new-jersey/**` assets.

No shared scene registry, shell, index, or existing scene was edited.

## Source and data

The source is [GeoDa's NCOVR Homicides + Socio-Economics archive](https://geodacenter.github.io/data-and-lab/ncovr/), which GeoDa provides as-is with no warranties; this lane asserts no additional license. The archive is described as containing 3,085 county units, 69 variables and four census decades. The supplied codebook identifies `UE` as civilian labor-force unemployment percent, `FP` as families below the period poverty threshold percent, `DNL` as log population density, and `PO` as decennial population. The selected source fields are the observed 1970/1980/1990 `UE`, census-vintage 1969/1979/1989 `FP`, 1970/1980/1990 `DNL`, actual `PO` counts, and source `POL` fields. The distinct 1960 poverty definition is excluded; nominal income is not used.

The builder selects all 21 New Jersey source rows and joins geometry by exact source `FIPSNO`, sorted in source order. It pins archive and supplied codebook SHA-256 values and fails if either drifts; `--allow-unverified-source` is an explicit override recorded in the manifest. Every generated asset has its own SHA-256. The county units and polygons are the archive's archive historical common-support template; the template label alone does not imply changed present county identities. Queen and rook are built from exact shared source vertices/edges, without manufactured edges. Both graphs have 82 directed CSR slots. Population remains an extensive count. Source GeoJSON stays EPSG:4326; analysis uses a spherical azimuthal-equidistant local meter frame centered at `[-74.55, 40.05]`.

Pinned source hashes: archive `e8cb04e6da634c6cd21808bd8cfe4dad6e295b22e8d40cc628e666887719cfe9`; codebook `229aac156c4005e70c2b43d15c69c73c0aacf4884a6af55f57487dbffe6b3de7`.

Rebuild command:

```sh
python3 examples/deck/spatial-analysis-showcase/scripts/data/ncovr-new-jersey/build.py \
  --archive /path/to/ncovr.zip --codebook /path/to/codebook.pdf
```

## Wave A: unemployment model comparison

The balanced input is 21 units × 3 observed snapshots (1970, 1980, 1990), period-major, with unemployment as response and family-poverty percentage plus source log density as time-varying predictors. Predictors are standardized once for the pooled panel. Queen weights are row-normalized and declared `transform: 'row', symmetric: false`; weights do not generally remain symmetric after row normalization. Cross-section choices force the response, map title/legend and tooltip to 1990 even when the disabled period control retains an earlier panel choice. Observed fills use the same cividis ramp as their legend; residual fills use a diverging ramp.

- `GPUSpatialPanelRegression`: pooled and unit-effects spatial 2SLS, `instrumentOrder: 1`, with coefficient table, status and contributor-defined residuals. It uses a balanced period-major panel with common entity weights and slopes; it has no time effects, random effects or dynamic lags. Unit effects absorb stable entity intercepts. Its inference is homoskedastic, without entity clustering or serial-correlation correction.
- `GPUSpatialPanelLagMaximumLikelihood`: random- and unit-effects Gaussian spatial-lag ML; fixed at 48 outer iterations, 40 search iterations and tolerance `1e-5`. **Required numerical correction:** the golden-section bracket starts at width 1.998; after 24 reductions by 0.61803398875 it remains about `1.927e-5`, wider than the requested `1e-5` tolerance. The previous 24-iteration budget therefore guaranteed nonconvergence status 3 regardless of fit quality. Both instances now use 40 iterations; tolerance and runtime status gates remain intact. Both expose fitted values, residuals, coefficients, summary and solver status. Random-effects mode receives `hausman` as the documented top-level optional prop. The readout only interprets tests at status 0; status 1 means a failed fit, status 2 an invalid covariance difference. Test degrees of freedom follow `spreg.k = predictorCount + 2` (4 here), rather than the contrast rank. This reference-oriented contributor supports at most 32 entities, 32 periods and 7 predictors; this scene uses 21, 3 and 2.
- `GPUSpatialPanelErrorMaximumLikelihood`: unit-effects Gaussian spatial-error ML with its fit/convergence status and residual output. Optional LM error, robust LM lag and robust LM error statistics are pooled, untransformed screens; the scene labels them as such, not as diagnostics of fitted unit-error residuals. It does not provide random/time effects, Hausman comparison or cluster-robust inference.
- `GPUSpatialLagMaximumLikelihood` and `GPUSpatialErrorMaximumLikelihood`: distinct 1990-only cross-sectional fits with exact traces and `maximumPower: 64`. Their fit/convergence statuses gate coefficient and residual interpretation. The lag output feeds `GPUSpatialLagImpacts`; direct, indirect and total impacts are read only after valid fit and convergence statuses.

The display map has one observed-value selection pass with three storage-buffer bindings and a conditional residual pass per model with five bindings, under the showcase cap of 16. Each physical buffer is imported once per command graph. Map readback supplies the actual selected CPU tooltip value and finite-unit count, guarded by an option generation. Invalid fits render as nodata. The tooltip describes its county-label match as nearest, because this lane does not claim polygon containment picking.

## Wave B: contiguous regionalization

The selected 1990 attributes are unemployment, 1989 family-poverty percent and source 1990 log density. Raw columns are passed to `GPUAZPRegions` and `GPUWardRegions` with `standardize: true`, so the contributor standardizes once; no population-weighted averaging of rates is used. `GPUMaxPRegions` receives actual 1990 population counts as extensive values. The graph uses binary, symmetric queen adjacency.

AZP starts from six explicit source-FIPS groups. Before graph publication the code checks all 21 rows are assigned exactly once, every group has at least two units, and every group is connected in the supplied queen CSR graph. AZP then uses six-region capacity, minimum size 2, a 48-iteration budget and exposed seed. Its status words are iteration count, converged, iteration-limit flag and invalid count. A converged local result is not a global optimum certificate.

Ward sequential and mutual-nearest outputs both target six regions with a 15-merge budget and exposed seed. Their four base status words are merge/iteration count, converged, iteration-limit flag and invalid count; mutual-nearest additionally reports merge rounds in the fifth word. They remain distinct schedules and may yield different partitions.

Max-P uses 8 starts, 128 growth steps, 24 improvement moves and the exposed seed; the population threshold is adjustable from 250,000 through 1,000,000. Its six words are region count, feasibility, infeasible-region count, invalid count, selected start and improvement moves. The readout checks actual label populations and reports min/max. Rows without valid labels and regions below the threshold render as nodata; an infeasible Max-P result is not colored as a feasible population partition.

Readback carries every algorithm's labels, objective and status. A generation guard rejects stale results after seed, threshold or algorithm changes. Counts omit `0xffffffff`; the check reports assignment count, connected components and actual population range. Ward/AZP convergence is explicitly not described as proof of global optimality. Labels remain exploratory algorithm output, not administrative recommendations.

## Wave C: population Dorling cartogram

`statistics/county-dorling.scene.ts` and `.compute.ts` use exported `GPUDorlingCartogram` from `@luma.gl/graph/gpu-spatial-analysis`; the public contract is documented in `modules/graph/src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram.ts`. Inputs are planar `float32x2` positions, nonnegative values and optional `uint32` source IDs. Options are `areaScale`, `maximumIterations` (0–256), `overlapTolerance` and `attraction`; capacity is at most 1,024 rows. Caller-owned outputs are displaced `float32x2` centers, `float32` radii, copied `uint32` IDs, `float32x2` maximum penetration/displacement and four `uint32` status words: iterations, converged, budget exhausted and excluded rows. It uses bounded O(n²) relaxation. Invalid coordinates, values or radii are excluded; zero values remain valid zero-radius points. The contributor does not read results back.

The scene feeds all 21 authored polygon-vertex-mean anchors projected into an azimuthal-equidistant local metre frame, actual `population1990`, and NCOVR source FIPS. The anchor is the arithmetic mean of every authored vertex, including ring-closing vertices and holes; it is explicitly a display anchor, not a centroid or guaranteed interior point. `areaScale = π × (20,000 m)² / 1,000,000 people`; origin circles use the same radius equation. Displaced circles read their radii from the contributor's returned GPU radius buffer and use those metre radii against the metre coordinate frame without pixel clamps. The size key converts these true radii to pixels at the current camera scale.

The display option switches among anchors only, population circles at their origins, and returned displaced circles. GPU-written leader segments join each source anchor to its displaced center when displaced circles are shown. Attraction `0`, `0.01` and `0.05`, plus budgets `0`, `25`, `100` and `256`, are exposed compile-time sensitivity options. Rebuilds use `GraphSwap`; each variant owns its compiled graph, output buffers and summary reader, and stale generations are ignored. A zero-budget stage reports the actual initial overlap from returned GPU output. Status readouts include terminal status, exclusions, returned circle count and source-FIPS ID mapping; they do not claim global optimality.

The summary readback copies GPU centers, radii, IDs, metrics and status. Tooltip hit tests use displayed circle centers/radii in planar coordinates and exact source FIPS mapping. In anchors-only view, the tooltip labels its result as the nearest anchor. The readback now verifies that all 21 returned IDs match source-FIPS order and checks the maximum relative radius error against actual population before showing returned circles; invalid mappings or radii are reported and those circles are hidden. GPU compilation/render/readback has not run, so numerical outputs remain pending runtime verification.

## Not covered

- The scene data ships rook adjacency, but both delivered stories choose queen adjacency; no live rook/queen control is implemented.
- Other circle-placement solvers, polygon cartograms, modern county boundaries, and a county-vintage crosswalk are deferred. Dorling convergence and overlap still need actual GPU verification.
- No modern county boundaries, annual interpolation, causal interpretation, unsupported panel effects, or serial/cluster inference are claimed.

## Verification and status

Completed scoped checks:

- Rebuilt the 21-unit dataset from `/tmp/statistics-ncovr.zip` and `/tmp/statistics-ncovr-codebook.pdf`; pinned hashes matched. Manifest records 20 columns including FIPS and queen/rook CSR arrays, plus per-file hashes. A separate CSR check certified all 21 source FIPS exactly once across six connected seed groups of sizes 5, 4, 3, 3, 4 and 2.
- `./node_modules/.bin/biome check --write` on both scene/compute pairs and the dataset definition: passed (5 TypeScript files checked). The Python builder also passed `python3 -m py_compile`.
- `node examples/deck/spatial-analysis-showcase/scripts/check-story-steps.mjs regression statistics`: passed.
- `node examples/deck/spatial-analysis-showcase/scripts/check-cartography.mjs --strict regression statistics`: one unrelated pre-existing finding in `regression/meuse-metal-model-form.scene.ts` (diverging legend has no midpoint); county scenes passed.
- An isolated strict TypeScript check of both county scene/compute pairs and the dataset definition passed with the showcase's external ambient declarations supplied from `/tmp`.
- The panel and regionalization scene/compute files remain frozen after their scoped checks. This continuation changed only `statistics/county-dorling.compute.ts`, `.scene.ts`, the uniquely named `statistics/county-dorling-anchor.ts` helper, and this lane note.
- `node examples/deck/spatial-analysis-showcase/scripts/check-wgsl-identifiers.mjs`: passed; 32,726 templates scanned, with no reserved WGSL identifiers or constant non-finite bitcasts. The new custom kernel binds only anchors, centers and line output. GPU/browser validation remains pending.

For this continuation, `yarn workspace luma.gl-examples-deck-spatial-analysis-showcase typecheck` reported no county Dorling diagnostics, but failed on existing errors outside this lane in `points/earthquake-pair-structure.compute.ts` and `points/earthquake-pair-structure.scene.ts`. These neighboring files were outside this lane and were left untouched.

Continuation checks: scoped `biome check --write` and a clean scoped `biome check` passed for the compute, scene and anchor helper; `node examples/deck/spatial-analysis-showcase/scripts/check-story-steps.mjs statistics` passed; `node examples/deck/spatial-analysis-showcase/scripts/check-cartography.mjs --strict statistics` passed. The showcase workspace `typecheck` ran and is blocked only by the unrelated file diagnostics listed above; it produced no county Dorling diagnostics. No GPU execution or acceptance, browser run, or independent numeric overlap certificate was performed. The scene is source-ready and GPU-pending.

Not run in this lane: `nvm use`, `yarn install`, `yarn build`, `yarn test`, `yarn lint fix`, `yarn website:build`, `(cd website && yarn build)`, browser/GPU acceptance, and numerical-reference checks. The supervising integration lane owns repository-wide validation and GPU acceptance. This is a scoped implementation status, not an all-green build or GPU claim.

## Files, verification, status

This continuation owns only `statistics/county-dorling.compute.ts`, `statistics/county-dorling.scene.ts`, `statistics/county-dorling-anchor.ts`, and this lane note. The panel and regionalization sources are frozen. Repository compilation, global tests and actual GPU rendering/readback remain pending integration acceptance.

## Wave D: county cross-variogram story

`statistics/county-cross-variogram.scene.ts` and `.compute.ts` add a five-step NCOVR New Jersey pair-bin story using `GPUCrossVariogram` and `GPUSpaceTimeVariogram` from `@luma.gl/graph/gpu-dataframe`. The source remains the existing 21-unit GeoDa archive dataset; no data was downloaded or rebuilt for this scene.

- One-period variants use the exact `unemployment1970`, `unemployment1980`, or `unemployment1990` field for the chosen period. Cross mode adds the matching `familyPoverty1969`, `familyPoverty1979`, or `familyPoverty1989` field; the source poverty measure is explicitly identified as preceding-year. Univariate runs omit `secondValues`.
- Space-time variants use all 63 actual rows in period-major order, with each county at times 0, 10 and 20 years, `temporalLagCount: 3`, `maxTime: 20`, and `lagCount: 6`. The resulting time intervals are `[0, 6.667)`, `[6.667, 13.333)`, and `[13.333, 20]`; actual observed lags are 0, 10 and 20. No annual interpolation is introduced. Unordered `i < j` pairs are counted once, including same-county positive-time-lag pairs at zero spatial distance.
- Distance cutoffs are 100,000, 200,000 and 300,000 meters via a parameter-buffer slider in 100,000-meter increments. Six equal-width bins use the contributor’s left-closed/right-open assignment, with the last cutoff endpoint included. Both contributors are compiled for all relevant field modes and switched without compilation in `encode`; cutoff updates write the shared `GPUParameterBuffer`.
- GPU mean, pair-count and status buffers feed the chart and per-bin readout. Space-time renders all 18 bins as three temporal series over six spatial lags. Empty bins remain `NaN` chart gaps; the readout prints the actual bin means, counts and status codes. Cross-semivariance retains signed values and uses a symmetric zero-centered chart domain in percentage-points squared; it is not labeled correlation.
- The map colors source unemployment or poverty by fixed scales and draws actual county polygons. Directionless links show pair support in a selected spatial band and, in space-time mode, temporal band. Pair locations are the same authored polygon-vertex-mean anchors as the Dorling story, projected into AEQD `[-74.55, 40.05]`; the card and furniture label them as areal-support proxies rather than stations or true centroids. Same-county links can be zero length; GPU pair counts retain them.
- The story describes these as descriptive pair-bin summaries, not fitted covariance, range/sill, prediction, causal inference or a dynamic model. Source and method references are [GeoDa’s NCOVR archive](https://geodacenter.github.io/data-and-lab/ncovr/), [gstat `variogram`](https://r-spatial.github.io/gstat/reference/variogram.html) and [gstat `variogramST`](https://r-spatial.github.io/gstat/reference/variogramST.html). The gstat references define distance cutoffs/bin widths, pair-count summaries, spatio-temporal lag boundaries and the actual temporal-lag interpretation used to explain this descriptive grouping.
- All 63 space-time input values are concatenated explicitly with `Float32Array.set` in period-major order; no `Array.flat()` conversion is used for typed arrays. A scoped CPU check against the shipped binary fields confirms 63 numeric observations, county alignment within all three periods, and correct first/last record alignment. The map tooltip reports the selected period, source value, units and source FIPS at the nearest authored anchor.
- Readbacks include a copied per-run generation token and compare it to the currently active generation. Pending readers retain their stale request and every reader is flushed while active, so a cutoff change or A→B→A selection cannot publish an older result or strand a pending update. Summary chart domains use finite, nonempty GPU-returned bin means for the selected support and mode; the axis may rescale after a cutoff change, while bin means remain the reported values. GPU acceptance is still pending.
- Independent source: `dev-docs/research/showcase-overhaul-2026-10-25/county-pairs-reference.json` (24 pair configurations and 12 Dorling physics configurations). CPU input alignment was checked against shipped dataset binaries; no GPU comparison with the pair-bin reference has been performed.
- The Dorling map title now names New Jersey and its initial camera uses zoom 7.1. No solver physics or accepted panel/regionalization computations changed.

## County-lane visual corrections and verification

- `county-dorling.compute.ts` now calls its existing thresholded `updateLegend()` from every `encode`, so camera-only zoom updates the pixel-size key without rebuilding or rerunning the GPU graph. Existing buffer, status and source-ID handling is preserved.
- County panel and regionalization overview cameras now start at zoom `7.1`; the panel’s explicit opening-story camera matches. Panel title: “Does the model change when counties are followed over time?” Regionalization title: “Which neighboring counties form similar groups?” Its map furniture names New Jersey.
- Scoped Biome formatting/checks passed for the two new scene files and the three visual-fix files. `check-story-steps.mjs statistics regression` and strict `check-cartography.mjs statistics regression` passed.
- An isolated strict TypeScript check of the variogram scene, compute module and anchor helper passed using the workspace ambient declarations. The showcase-wide `typecheck` could not complete because another concurrently edited file, `src/scenes/geometry/noding-and-coverage.scene.ts`, currently has TypeScript parse errors around line 531. That file is outside this lane and was not changed. An earlier workspace typecheck after initial implementation also showed only unrelated external diagnostics plus variogram type errors that were fixed before the isolated pass.
- No browser, GPU execution, production numeric acceptance, independent pair-bin reference comparison, root build/lint/install or repository-wide test was run. The graph path is source-checked; GPU means/counts/status and visual rendering remain unverified for the independent planner/acceptance slot.

This continuation owns only `statistics/county-cross-variogram.compute.ts`, `statistics/county-dorling.compute.ts`, `statistics/county-dorling.scene.ts`, and this lane note. County panel and regionalization computations are preserved. Scoped Biome, story-step, strict cartography and CPU input-alignment checks passed; source is ready for GPU review, while GPU acceptance and comparison against the independent pair-bin reference remain pending. No root build, lint, install, GPU check or commit was run.
