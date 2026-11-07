# Point-density and point-pattern lane

## Status

- **KDE implementation:** source fixes and requested controls are implemented. It is **not numerically accepted or production-ready**: GPU execution, independent CPU comparison, and visual review remain pending.
- **Direct pointwise CSR:** bounded implementation is integrated into the existing point-pattern story and is selected on demand. It is **not GPU-validated or production-accepted**.
- **Existing shared-replicate global tests:** their implementation and public behavior were preserved. The prescribed acceptance script could not launch its browser in this environment, so their continued runtime pass is not verified in this lane.
- **No browser/GPU, root build, or root lint gate was run successfully.** No release or merge readiness is claimed.

## KDE implementation delivered

- Added the independent `nature-kernel-density` companion using the actual 2023 Chicago iNaturalist observation rows and `GPUKernelDensity`; the existing camera-following binned `GPUPointDensity` scene remains distinct.
- It retains every source row projected into the disclosed central Chicago 16 km square, with a fixed 64 × 64 raster of 250 m cells, the Lambert equal-area analysis frame, actual source indices, research-grade weights, outside-window counts, and a visible window outline. No arbitrary subsampling is used.
- The scene exposes Gaussian, quartic, and Epanechnikov kernels; none, uniform, and Diggle edge correction; intensity and probability normalization; fixed bandwidth plus 2D Scott/Silverman reference rules; observation and research-grade weights; and exact, binned, or auto Gaussian evaluation. Gaussian-only controls are disabled for the compact kernels. A 2,500 m fixed bandwidth exercises auto's binned range; very large bandwidths fall back to exact. No padding guarantee is claimed beyond the contributor's padding diagnostic.
- The raster and the cell, peak, and integral summaries come from GPU output. Observation records are described as observer activity, not organism abundance. Raster-integral quadrature and each method's limitations are explained in the story.

## KDE source-review fixes delivered

- Rebuild requests now snapshot options and use a latest-desired-key plus generation guard with `GraphSwap`. A→B→A commits only the final A request; bandwidth and weight changes during a pending compile invalidate and refresh that request.
- The initial compiled graph and every replacement graph are tracked by scene resources. Replaced graphs are released explicitly. Summary readers release their readback rings when superseded or destroyed; callbacks are guarded by the reader's captured generation and immutable option snapshot.
- Rebuild and parameter changes clear the old raster and analysis readouts while new output is pending. The callback no longer reads the live method/rule for old results.
- A valid surface requires all 4,096 cells finite, zero invalid input rows, sufficient padding when binned evaluation is actually resolved, and a finite positive GPU-reported weight sum. Empty or zero-weight input cannot publish a valid probability surface. Zero-weight rows remain valid inputs under the GPU API, but contribute no weight; `zeroWeight` counts those rows. Auto and non-Gaussian method labels reflect the resolved contributor path.
- Removed the `ctx.options` casts and confirmed only one `validCells` clear remains.

## Direct `GPURipleyEnvelope` pointwise implementation delivered

- Added an `Envelope source` choice to the existing `point-patterns` story. The default remains the existing shared patterns and accepted global-test route. Direct CSR is a separate graph created and run only when selected; it does not feed or replace the global test.
- Direct mode runs `GPURipleyEnvelope` with 39 GPU-generated CSR patterns, rank 1, `l-minus-r`, the same selected mask, rectangle window, Ripley parameters, radii, and edge correction as the observed pattern. The attainable two-sided pointwise level is 2/40 = 0.05.
- GPU-produced lower, upper, theoretical, and summary-statistics values feed a separate chart/readout. The run captures the requested radius, snaps to the nearest actual computed grid radius, and indexes the observed and GPU bounds at that grid index without interpolation. If they differ, both requested and tested radii are shown. The chart marker stays on the tested grid radius.
- The pointwise interpretation is for one predetermined radius. The continuous band is only a visual guide between grid radii, and exploring additional radii afterward is not familywise controlled.
- Direct mode clears its chart/readouts and withholds inference unless the observed and null supports are both the rectangle, jitter is zero, the selected count is at least two, edge correction is none or isotropic, every observed radius and curve value is finite, all GPU bounds/theory are finite, `validSimulationCount` is 39, and the GPU reports zero unplaced points. `GPURipleyEnvelope.statistics` supplies n and validSimulationCount without observed K; no permissive global p-value is exposed. That count is curve-level (at least one finite radius), and the envelope node may ignore an invalid replicate at an individual radius, so it alone does not prove 39 values at every radius. The restricted rectangle, n ≥ 2, finite-radius, none/isotropic case makes each simulated radius mathematically defined; floating-point or other numeric failures remain possible, which is why finite outputs and the summary count are still checked. Polygon windows, border correction, and unsupported states have explicit status messages; no fallback to another window occurs.
- Direct CSR uses a fixed disclosed seed (`2026102`), exactly representable in its float32 parameter, so revisiting the same selected data/options reproduces the same independent null ensemble. The graph generation guards async results but does not change the PRNG seed. It is not forced to share the existing shared ensemble.
- The direct band is pointwise and is not a confidence band for the true statistic. It is an independently generated null ensemble and is not asserted to equal the existing 39 shared patterns. No direct-GPU global p-value is surfaced.
- Existing shared global maximum-deviation, integrated-square, and extreme-rank tests; their 39 matching-window curves, p-grid/tie interval, alpha handling, and invalidation behavior were left intact in code. They remain unverified at runtime here.

## Scientific references and limitations

- `spatstat.explore::density.ppp` documents point-process intensity per area and location-based uniform versus point-based Jones–Diggle edge correction: <https://search.r-project.org/CRAN/refmans/spatstat.explore/html/density.ppp.html>.
- `pointpats` documents that projected CRS units determine KDE bandwidth units: <https://pysal.org/pointpats/_modules/pointpats/kde.html>.
- `spatstat.explore::envelope` distinguishes pointwise from global envelopes: <https://search.r-project.org/CRAN/refmans/spatstat.explore/help/envelope.html>; the reference implementation notes are at <https://github.com/spatstat/spatstat.explore/blob/master/man/envelope.Rd>.
- Gaussian exact evaluation truncates support at five bandwidths. Binned Gaussian is a grid-binning/filter approximation; auto selects exact or binned evaluation using bandwidth and padding. Scott and Silverman coincide in two dimensions. Compact-kernel parameters are not rescaled for exact spatstat sigma parity.
- A 39-simulation rank-1 two-sided pointwise band has 0.05 level at one preselected radius and does not control the full curve. It is not a confidence band for the true intensity/statistic. The no-correction KDE loses mass beyond the window; uniform correction does not force an exact raster integral; Diggle better preserves point mass but cell-center quadrature remains approximate.

## Verification and remaining barriers

- `npx biome check --write` passed on the four scoped scene TypeScript files. `node examples/deck/spatial-analysis-showcase/scripts/check-cartography.mjs --strict points` and `node examples/deck/spatial-analysis-showcase/scripts/check-story-steps.mjs` both passed.
- The showcase package typecheck is currently blocked by unrelated errors in `src/scenes/geometry/block-relationships.compute.ts`. A project limited to the four scoped scene files reported no diagnostics in them, but exited on an imported-source `globalThis` typing issue in `src/engine/summary-reader.ts`.
- The accepted `scripts/check-point-pattern-global-test.mjs` was attempted but Chromium aborted during startup (`SIGABRT` / `EPERM` while Playwright cleaned up). It did not reach the test assertions. No additional browser/GPU attempt is authorized in this lane.
- Pending: serialized GPU execution and readback validation for KDE and direct CSR; independent CPU numerical comparison for KDE and direct-envelope bounds; visual review; and runtime verification that the existing shared global tests still pass. Until those barriers are completed, neither implementation is numerically accepted or production-ready.
